"""The AdamW ``P`` against the update ``torch.optim.AdamW`` actually applies.

``test_preconditioner.py``'s T-adamw-state compares the built ``P`` to a *hand
computation from the optimizer's internals* — the same formula this repo typed
into ``Preconditioner.adamw``, evaluated twice. That catches transcription slips
but not a wrong *identification*: if the convention itself named the wrong
object, both sides would agree and both would be wrong.

Here the reference is the optimizer's behaviour. From

    AdamW:  dtheta = -lr * ( m_hat / (sqrt(v_hat) + eps) + wd * theta )

the geometry factor is recoverable elementwise from what the step did:

    P_measured = -(dtheta + lr * wd * theta) / (lr * m_hat),   m_hat = exp_avg/(1-beta1^t)

with ``dtheta``, ``exp_avg``, ``exp_avg_sq`` and ``step`` all read out of a real
optimizer after a real step. That pins the ``eps`` placement, the
bias-correction exponent, and the *exclusion* of momentum and decoupled weight
decay — against torch, not against the prose.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from vatis.core.preconditioner import Preconditioner

FP64 = torch.float64


def _tiny(seed: int = 0) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.5)
    return model.double()


def _flat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).to(FP64) for t in tensors])


def _run_adamw(
    *,
    steps: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    foreach: bool | None,
    seed: int = 0,
) -> dict:
    """Run real AdamW and record everything needed to *measure* ``P``."""
    model = _tiny(seed)
    params = list(model.parameters())
    opt = torch.optim.AdamW(
        params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd, foreach=foreach
    )
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(8, 5, generator=g, dtype=FP64)
    y = torch.randn(8, 3, generator=g, dtype=FP64)
    record: dict = {}
    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        # a drifting loss scale so v is not a fixed point of the EMA
        (((model(x) - y) ** 2).mean() * (1.0 + 0.1 * t)).backward()
        theta_before = _flat(params).clone()
        grads = _flat([p.grad for p in params if p.grad is not None]).clone()
        pre_v = (
            _flat([opt.state[p]["exp_avg_sq"] for p in params]).clone()
            if opt.state
            else torch.zeros_like(theta_before)
        )
        pre_step = int(opt.state[params[0]]["step"].item()) if opt.state else 0
        opt.step()
        record = {
            "theta_before": theta_before,
            "theta_after": _flat(params).clone(),
            "grads": grads,
            "pre_v": pre_v,
            "pre_step": pre_step,
            "post_v": _flat([opt.state[p]["exp_avg_sq"] for p in params]).clone(),
            "post_m": _flat([opt.state[p]["exp_avg"] for p in params]).clone(),
            "post_step": int(opt.state[params[0]]["step"].item()),
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "wd": wd,
        }
    return record


def _measured_p(rec: dict) -> torch.Tensor:
    """``P`` read off the update torch applied."""
    dtheta = rec["theta_after"] - rec["theta_before"]
    m_hat = rec["post_m"] / (1.0 - rec["beta1"] ** rec["post_step"])
    return -(dtheta + rec["lr"] * rec["wd"] * rec["theta_before"]) / (rec["lr"] * m_hat)


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize(
    "beta1,beta2,eps,wd",
    [
        (0.0, 0.999, 1e-8, 0.0),
        (0.9, 0.999, 1e-8, 0.0),
        (0.9, 0.95, 1e-6, 0.1),
        (0.9, 0.999, 0.0, 0.0),
    ],
)
def test_adamw_P_equals_the_update_torch_applied(
    foreach: bool, beta1: float, beta2: float, eps: float, wd: float
) -> None:
    """``Preconditioner.adamw`` == the ``P`` that generated the real step."""
    rec = _run_adamw(steps=20, lr=3e-3, beta1=beta1, beta2=beta2, eps=eps, wd=wd, foreach=foreach)
    want = _measured_p(rec)
    got = Preconditioner.adamw(rec["post_v"], step=rec["post_step"], beta2=beta2, eps=eps).diag
    assert float(((got - want).abs() / want.abs()).max()) < 1e-11, (foreach, beta1, beta2, eps, wd)


def test_adamw_P_matches_at_every_step_not_only_the_last() -> None:
    beta1, beta2, eps, lr = 0.9, 0.98, 1e-8, 2e-3
    model = _tiny(3)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=0.0)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(6, 5, generator=g, dtype=FP64)
    y = torch.randn(6, 3, generator=g, dtype=FP64)
    worst = 0.0
    for t in range(1, 26):
        opt.zero_grad()
        (((model(x) - y) ** 2).mean() * (1.0 + 0.05 * t)).backward()
        before = _flat(params).clone()
        opt.step()
        after = _flat(params).clone()
        step = int(opt.state[params[0]]["step"].item())
        m_hat = _flat([opt.state[p]["exp_avg"] for p in params]) / (1.0 - beta1**step)
        want = -(after - before) / (lr * m_hat)
        got = Preconditioner.adamw(
            _flat([opt.state[p]["exp_avg_sq"] for p in params]), step=step, beta2=beta2, eps=eps
        ).diag
        worst = max(worst, float(((got - want).abs() / want.abs()).max()))
    assert worst < 1e-10, worst


@pytest.mark.parametrize("beta2", [0.9, 0.99, 0.999])
def test_bias_correction_exponent_is_pinned_against_torch(beta2: float) -> None:
    """Only exponent ``t`` reproduces the real step; ``t-1`` and ``t+1`` do not."""
    rec = _run_adamw(steps=7, lr=1e-3, beta1=0.0, beta2=beta2, eps=1e-8, wd=0.0, foreach=False)
    want = _measured_p(rec)
    t = rec["post_step"]
    good = Preconditioner.adamw(rec["post_v"], step=t, beta2=beta2, eps=1e-8).diag
    assert float(((good - want).abs() / want.abs()).max()) < 1e-11
    for wrong in (t - 1, t + 1):
        if wrong < 1:
            continue
        bad = Preconditioner.adamw(rec["post_v"], step=wrong, beta2=beta2, eps=1e-8).diag
        assert float(((bad - want).abs() / want.abs()).max()) > 1e-4, (beta2, wrong)


def test_eps_placement_matches_torch_and_the_alternatives_do_not() -> None:
    """``1/(sqrt(v_hat) + eps)`` — not inside the root, and ``eps`` is not corrected."""
    eps = 1e-3  # large enough that the placements are numerically distinguishable
    rec = _run_adamw(steps=5, lr=1e-3, beta1=0.0, beta2=0.9, eps=eps, wd=0.0, foreach=False)
    want = _measured_p(rec)
    t, v = rec["post_step"], rec["post_v"]
    bias_correction = 1.0 - 0.9**t
    good = Preconditioner.adamw(v, step=t, beta2=0.9, eps=eps).diag
    assert float(((good - want).abs() / want.abs()).max()) < 1e-11
    alternatives = {
        "1/sqrt(v_hat + eps)": 1.0 / (v / bias_correction + eps).sqrt(),
        "no bias correction": 1.0 / (v.sqrt() + eps),
        "eps inside the correction": 1.0 / ((v.sqrt() + eps) / math.sqrt(bias_correction)),
    }
    for name, alt in alternatives.items():
        assert float(((alt - want).abs() / want.abs()).max()) > 1e-6, name


@pytest.mark.parametrize("beta2", [0.9, 0.99, 0.999])
@pytest.mark.parametrize("steps", [10, 40])
def test_prestep_state_is_exactly_one_ema_update_behind(beta2: float, steps: int) -> None:
    """The frozen pre-step convention, made exact — and its error bound corrected.

    ``v_hat_t`` is a **convex combination** of ``v_hat_{t-1}`` and the fresh
    ``g_t^2``:

        v_hat_t = (1 - w) v_hat_{t-1} + w g_t^2,   w = (1 - beta2)/(1 - beta2^t)

    (from ``1 - beta2^t = beta2 (1 - beta2^{t-1}) + (1 - beta2)``.) The mixing
    weight is ``w``, **not** ``1 - beta2``: at ``t << 1/(1 - beta2)`` it is
    ``~1/t``, so ``00_context.md``'s "one EMA update, O(1-beta2) difference"
    understates the early-checkpoint error by ``1/(1 - beta2^t)`` — a factor 67 at
    ``beta2 = 0.999, t = 15``, which is exactly the regime a few-hundred-step
    bundle lives in. The convention is frozen and vatis implements it faithfully;
    this test exists so the *size* of what it drops is on record.
    """
    rec = _run_adamw(steps=steps, lr=1e-3, beta1=0.0, beta2=beta2, eps=1e-8, wd=0.0, foreach=False)
    t = rec["post_step"]
    assert rec["pre_step"] == t - 1 >= 1
    v_hat_pre = rec["pre_v"] / (1.0 - beta2 ** rec["pre_step"])
    v_hat_post = rec["post_v"] / (1.0 - beta2**t)
    w = (1.0 - beta2) / (1.0 - beta2**t)
    predicted = (1.0 - w) * v_hat_pre + w * rec["grads"] ** 2
    assert float(((predicted - v_hat_post).abs() / v_hat_post.abs()).max()) < 1e-12

    eps = 1e-8
    rel = (1.0 / (v_hat_pre.sqrt() + eps) - 1.0 / (v_hat_post.sqrt() + eps)).abs() / (
        1.0 / (v_hat_post.sqrt() + eps)
    )
    # governed by w, and unbounded above only in the max norm
    assert float(rel.max()) <= 1.0
    assert float(rel.median()) <= 4.0 * max(1.0 / math.sqrt(1.0 - w) - 1.0, 1e-6)
    assert float(rel.max()) > 0.3 * w, (
        "the pre-step offset is O((1-beta2)/(1-beta2^t)), not O(1-beta2)",
        beta2,
        float(rel.max()),
        w,
    )
