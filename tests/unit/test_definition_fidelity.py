"""Does ``chi_pos`` equal ``sum_k gamma_k^2 p_k`` from an explicit eigendecomposition?

``test_chi_net.py`` already checks the *raw* self-pair triple against an explicit
Jacobian. This file extends the same ground truth to the two cases WP0 added and
that were never covered: the **preconditioned** ruler (``eigh(Theta_P)``, not
``eigh(Theta)``) and the **cross** pair, both on a batch with *asymmetric*
padding and unequal batch sizes — and it goes through the whole ``Analyzer``
rather than the estimator alone, so the normalisation and the combinator are in
scope too.

Two ground-truth layers, deliberately:

1. the reference reports ``chi_pos`` **twice** — ``sum_k gamma_k^2 p_k`` from
   ``eigh``, and the ratio ``delta_loss/(chi_loss*chi_net)`` — and every test
   asserts the two agree. That equality *is* the LNP factorisation; without it
   nothing below means anything.
2. because ``chi_net`` is a stochastic estimate for a dense kernel, each
   assertion is made twice: once on the published number with the derived
   Rademacher bound, and once with the *exact* trace substituted for the
   estimate, which removes the estimator noise and turns the statement about the
   **definition** into a near-exact one.

``test_the_spectral_reference_is_discriminating`` is the vacuity guard: mixing
the eigenbases (``gamma`` from ``Theta``, ``p`` from ``Theta_P``) — the exact
mistake this file exists to rule out — must give a *different* number on this
fixture.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from tests.unit.test_chi_net import _build_full_jacobian_and_u
from vatis.analyzer import Analyzer
from vatis.core.preconditioner import Preconditioner
from vatis.models.hf import ModelBundle

FP64 = torch.float64
VOCAB = 16
N_HUTCH = 512
K_SIGMA = 5.0
# The analyzer builds delta_loss from an fp32 flat gradient and chi_loss from an
# fp32 softmax, so ~1e-5 relative is the floor for an fp32 toy. Same tolerance
# philosophy as test_chi_net.py's exact-NTK tests.
EXACT_REL = 1e-5


def _fixture() -> dict:
    torch.manual_seed(42)
    model = TinyTransformer(
        vocab_size=VOCAB, seq_len=8, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters())
    batches = {
        # unequal batch sizes AND unequal padding -> N_A != N_B, which is what
        # makes the sqrt(N_A N_B) factors observable rather than cancelling.
        "A": make_tiny_lm_batch(
            batch_size=3, seq_len=8, vocab_size=VOCAB, pad_fraction=0.25, seed=0
        ),
        "B": make_tiny_lm_batch(
            batch_size=2, seq_len=8, vocab_size=VOCAB, pad_fraction=0.5, seed=1
        ),
    }
    n_params = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(5)
    p_diag = torch.exp(torch.randn(n_params, generator=gen, dtype=FP64) * 3.0)
    gt = {}
    for name, batch in batches.items():
        jac, u, n_valid = _build_full_jacobian_and_u(model, batch)
        groups, off = [], 0
        for n_pos in (batch["labels"] != -100).sum(dim=1).tolist():
            groups.append((off, off + int(n_pos) * VOCAB))
            off += int(n_pos) * VOCAB
        assert off == jac.shape[0]
        gt[name] = {"jac": jac, "u": u, "n_valid": n_valid, "groups": groups}
    return {
        "model": model,
        "params": params,
        "batches": batches,
        "p_diag": p_diag,
        "gt": gt,
        "bundle": ModelBundle(
            model=model,
            params=params,
            forward_fn=lambda m, b: m(b["input_ids"]),
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            identifier="toy@step0",
        ),
    }


def _lnp_self(jac: torch.Tensor, u: torch.Tensor, p_diag: torch.Tensor | None) -> dict[str, float]:
    """Self-pair components from an explicit ``eigh(Theta_P)``."""
    weight = None if p_diag is None else p_diag.to(FP64)
    grad = jac.T @ u
    if weight is None:
        theta = jac @ jac.T
        chi_net = float((jac * jac).sum())
        delta = float((grad * grad).sum())
    else:
        theta = (jac * weight.unsqueeze(0)) @ jac.T
        chi_net = float(((jac * jac) * weight.unsqueeze(0)).sum())
        delta = float((grad * weight * grad).sum())
    chi_loss = float((u * u).sum())
    evals, evecs = torch.linalg.eigh(theta)
    trace = float(evals.sum())
    gamma_sq = (evecs.T @ u) ** 2 / chi_loss
    return {
        "chi_loss": chi_loss,
        "chi_net": chi_net,
        "chi_net_trace": trace,
        "delta_loss": delta,
        "chi_pos_spectral": float((gamma_sq * (evals / evals.sum())).sum()),
        "chi_pos_ratio": delta / (chi_loss * trace),
        "gamma_sq_sum": float(gamma_sq.sum()),
    }


def _lnp_cross(
    ja: torch.Tensor,
    ua: torch.Tensor,
    jb: torch.Tensor,
    ub: torch.Tensor,
    p_diag: torch.Tensor | None,
) -> dict[str, float]:
    """Cross-pair components per ``00_context.md`` §"Two-batch form"."""
    ga, gb = ja.T @ ua, jb.T @ ub
    weight = None if p_diag is None else p_diag.to(FP64)
    if weight is None:
        delta = float((gb * ga).sum())
        net_a, net_b = float((ja * ja).sum()), float((jb * jb).sum())
    else:
        delta = float((gb * weight * ga).sum())
        net_a = float(((ja * ja) * weight.unsqueeze(0)).sum())
        net_b = float(((jb * jb) * weight.unsqueeze(0)).sum())
    loss_a, loss_b = float((ua * ua).sum()), float((ub * ub).sum())
    chi_loss_cross = (loss_a * loss_b) ** 0.5
    chi_net_cross = (net_a * net_b) ** 0.5
    return {
        "delta_loss": delta,
        "chi_loss_cross": chi_loss_cross,
        "chi_net_cross": chi_net_cross,
        "chi_pos_cross": delta / (chi_loss_cross * chi_net_cross),
        "chi_net_a": net_a,
        "chi_net_b": net_b,
    }


def _projector(u: torch.Tensor, groups: list[tuple[int, int]]) -> torch.Tensor:
    pi = torch.zeros(u.numel(), u.numel(), dtype=FP64)
    for lo, hi in groups:
        block = u[lo:hi]
        nsq = float((block * block).sum())
        if nsq > 0:
            pi[lo:hi, lo:hi] = torch.outer(block, block) / nsq
    return pi


def _trace_std(
    entry: dict, p_diag: torch.Tensor | None, *, n_h: int, micro: int, method: str
) -> float:
    """Exact std of the pipeline's ``chi_net`` estimate for this configuration.

    ``n_h`` probes are drawn **per micro-batch**, so the per-micro variances add.
    ``per_sequence_cv``'s residual kernel is ``(I - Pi) J P J^T (I - Pi)``
    restricted to the micro's rows; ``hutchinson``'s is the whole block.
    """
    jac, u, groups = entry["jac"], entry["u"], entry["groups"]
    weight = None if p_diag is None else p_diag.to(FP64)
    var = 0.0
    for start in range(0, len(groups), micro):
        stop = min(start + micro, len(groups))
        lo, hi = groups[start][0], groups[stop - 1][1]
        jm = jac[lo:hi]
        theta = jm @ jm.T if weight is None else (jm * weight.unsqueeze(0)) @ jm.T
        if method == "per_sequence_cv":
            pi = _projector(u[lo:hi], [(a - lo, b - lo) for a, b in groups[start:stop]])
            eye = torch.eye(theta.shape[0], dtype=FP64)
            theta = (eye - pi) @ theta @ (eye - pi)
        off = theta - torch.diag(torch.diagonal(theta))
        var += 2.0 * float((off * off).sum()) / float(n_h)
    return math.sqrt(var)


def _run(case: dict, p_diag: torch.Tensor | None, *, method: str, micro: int, n_h: int) -> dict:
    precond = (
        None
        if p_diag is None
        else Preconditioner.from_diagonal(p_diag.to(torch.float32), descriptor="explicit")
    )
    analyzer = Analyzer(
        eval_batches=dict(case["batches"]),
        preconditioner=precond,
        chi_net_method=method,
        n_hutchinson=n_h,
        micro_batch_size=micro,
        cross_pairs=[("A", "B")],
        sink=None,
        seed=1234,
    )
    rows = analyzer.run([(case["bundle"], "step0")])[0].rows
    out: dict = {(r.batch_a, r.batch_b, r.observable): r.value for r in rows}
    out["_n"] = {(r.batch_a, r.batch_b): (r.n_valid_a, r.n_valid_b) for r in rows}
    return out


# --------------------------------------------------------------- the reference


def test_the_reference_is_self_consistent() -> None:
    """``sum_k gamma_k^2 p_k`` == the ratio form, and ``sum_k gamma_k^2 == 1``."""
    case = _fixture()
    for name in ("A", "B"):
        for p_diag in (None, torch.ones_like(case["p_diag"]), case["p_diag"]):
            ref = _lnp_self(case["gt"][name]["jac"], case["gt"][name]["u"], p_diag)
            assert abs(ref["gamma_sq_sum"] - 1.0) < 1e-10, name
            assert ref["chi_net_trace"] == pytest.approx(ref["chi_net"], rel=1e-9)
            assert ref["chi_pos_spectral"] == pytest.approx(ref["chi_pos_ratio"], rel=1e-10)


def test_the_spectral_reference_is_discriminating() -> None:
    """Vacuity guard: mixing the eigenbases must give a different answer."""
    case = _fixture()
    for name in ("A", "B"):
        jac, u = case["gt"][name]["jac"], case["gt"][name]["u"]
        theta = jac @ jac.T
        theta_p = (jac * case["p_diag"].unsqueeze(0)) @ jac.T
        ev, q = torch.linalg.eigh(theta)
        ev_p, q_p = torch.linalg.eigh(theta_p)
        chi_loss = float((u * u).sum())
        gamma = (q.T @ u) ** 2 / chi_loss
        gamma_p = (q_p.T @ u) ** 2 / chi_loss
        raw = float((gamma * (ev / ev.sum())).sum())
        right = float((gamma_p * (ev_p / ev_p.sum())).sum())
        mixed = float((gamma * (ev_p / ev_p.sum())).sum())
        # (a) the mistake this file rules out: gamma from Theta, p from Theta_P.
        assert abs(right - mixed) > 0.2 * abs(right), (name, right, mixed, "mixed basis")
        # (b) the P ruler is a different number from the raw one, so a P assertion
        # cannot pass by accidentally measuring raw. The margin is smaller than (a)
        # because `right` and `raw` are both genuine rulers on the same kernel
        # family; 5% is comfortably above the ~1% Hutchinson noise these tests run at.
        assert abs(right - raw) > 0.05 * abs(raw), (name, right, raw, "P vs raw")


# ------------------------------------------------------------------ the checks


def test_chi_loss_is_the_sigma_over_sqrt_n_convention() -> None:
    """``chi_loss = sigma^2/N`` and ``chi_loss_normalized = sigma^2``.

    ``00_context.md`` fixes ``||grad_f L^B|| = sigma_loss^B / sqrt(|B|)``. The
    only self-consistent reading is ``sigma^2 = (1/N) sum_i ||grad_{f_i} l_i||^2``
    over the **per-token** loss gradients, computed here by one autograd call per
    valid token on the *unreduced* CE — i.e. without reusing vatis's closed form.

    The *direction* of the factor is pinned by invariance rather than algebra:
    for iid data ``sigma^2`` must not depend on the token count, and only
    ``sqrt(N_A N_B) * chi_loss`` has that property.
    """
    case = _fixture()
    rows = _run(case, None, method="per_sequence_cv", micro=1, n_h=2)
    for name, batch in case["batches"].items():
        logits = case["model"](batch["input_ids"]).detach().requires_grad_(True)
        flat_logits = logits.reshape(-1, VOCAB)
        flat_labels = batch["labels"].reshape(-1)
        sq_norms = []
        for pos in range(flat_logits.shape[0]):
            if int(flat_labels[pos]) == -100:
                continue
            ell = F.cross_entropy(
                flat_logits[pos : pos + 1], flat_labels[pos : pos + 1], reduction="sum"
            )
            (g,) = torch.autograd.grad(ell, logits, retain_graph=True)
            sq_norms.append(float((g.to(FP64) ** 2).sum()))
        n_valid = len(sq_norms)
        assert rows["_n"][(name, name)] == (n_valid, n_valid)
        sigma_sq = sum(sq_norms) / n_valid
        assert rows[(name, name, "chi_loss")] == pytest.approx(sigma_sq / n_valid, rel=EXACT_REL)
        assert rows[(name, name, "chi_loss_normalized")] == pytest.approx(sigma_sq, rel=EXACT_REL)


@pytest.mark.parametrize("method", ["per_sequence_cv", "hutchinson"])
@pytest.mark.parametrize("p_kind", ["none", "identity", "spread"])
def test_self_pair_chi_pos_equals_the_spectral_sum(method: str, p_kind: str) -> None:
    """``chi_pos`` / ``chi_pos_P`` == ``sum_k gamma_k^2 p_k`` from ``eigh``."""
    case = _fixture()
    p_diag = {
        "none": None,
        "identity": torch.ones_like(case["p_diag"]),
        "spread": case["p_diag"],
    }[p_kind]
    micro = 1
    rows = _run(case, p_diag, method=method, micro=micro, n_h=N_HUTCH)

    for name in ("A", "B"):
        entry = case["gt"][name]
        ref = _lnp_self(entry["jac"], entry["u"], None)
        assert rows[(name, name, "chi_loss")] == pytest.approx(ref["chi_loss"], rel=EXACT_REL)
        assert rows[(name, name, "delta_loss")] == pytest.approx(ref["delta_loss"], rel=EXACT_REL)
        std = _trace_std(entry, None, n_h=N_HUTCH, micro=micro, method=method)
        assert abs(rows[(name, name, "chi_net")] - ref["chi_net"]) <= K_SIGMA * std + 1e-9

        # With the estimator noise removed the definitional identity is exact.
        substituted = rows[(name, name, "delta_loss")] / (
            rows[(name, name, "chi_loss")] * ref["chi_net"]
        )
        assert substituted == pytest.approx(ref["chi_pos_spectral"], rel=EXACT_REL)
        tol = abs(ref["chi_pos_spectral"]) * (K_SIGMA * std / ref["chi_net"] + EXACT_REL) + 1e-12
        assert abs(rows[(name, name, "chi_pos")] - ref["chi_pos_spectral"]) <= tol

        if p_diag is None:
            assert (name, name, "chi_pos_P") not in rows
            continue

        ref_p = _lnp_self(entry["jac"], entry["u"], p_diag)
        assert rows[(name, name, "delta_loss_P")] == pytest.approx(
            ref_p["delta_loss"], rel=EXACT_REL
        )
        std_p = _trace_std(entry, p_diag, n_h=N_HUTCH, micro=micro, method=method)
        assert abs(rows[(name, name, "chi_net_P")] - ref_p["chi_net"]) <= K_SIGMA * std_p + 1e-9
        substituted_p = rows[(name, name, "delta_loss_P")] / (
            rows[(name, name, "chi_loss")] * ref_p["chi_net"]
        )
        assert substituted_p == pytest.approx(ref_p["chi_pos_spectral"], rel=EXACT_REL)
        tol_p = (
            abs(ref_p["chi_pos_spectral"]) * (K_SIGMA * std_p / ref_p["chi_net"] + EXACT_REL)
            + 1e-12
        )
        assert abs(rows[(name, name, "chi_pos_P")] - ref_p["chi_pos_spectral"]) <= tol_p


@pytest.mark.parametrize("p_kind", ["none", "spread"])
def test_cross_pair_matches_the_two_batch_definition(p_kind: str) -> None:
    """``chi_pos(A,B) = dL(A,B) / (||u_A|| ||u_B|| ||J_A||_F ||J_B||_F)``.

    The geometric-mean convention of ``00_context.md`` §"Two-batch form", against
    explicit ``J^A``, ``J^B``, ``u^A``, ``u^B`` — including the ``P`` version,
    where the means are ``||P^{1/2} J^X||_F``.
    """
    case = _fixture()
    p_diag = None if p_kind == "none" else case["p_diag"]
    micro, method = 1, "per_sequence_cv"
    rows = _run(case, p_diag, method=method, micro=micro, n_h=N_HUTCH)
    a, b = case["gt"]["A"], case["gt"]["B"]

    ref = _lnp_cross(a["jac"], a["u"], b["jac"], b["u"], None)
    assert rows[("A", "B", "delta_loss")] == pytest.approx(ref["delta_loss"], rel=EXACT_REL)
    assert rows[("A", "B", "chi_loss")] == pytest.approx(ref["chi_loss_cross"], rel=EXACT_REL)
    std_a = _trace_std(a, None, n_h=N_HUTCH, micro=micro, method=method)
    std_b = _trace_std(b, None, n_h=N_HUTCH, micro=micro, method=method)
    rel_net = 0.5 * (K_SIGMA * std_a / ref["chi_net_a"] + K_SIGMA * std_b / ref["chi_net_b"])
    assert abs(rows[("A", "B", "chi_net")] - ref["chi_net_cross"]) <= (
        rel_net * ref["chi_net_cross"] + 1e-9
    )
    substituted = rows[("A", "B", "delta_loss")] / (ref["chi_loss_cross"] * ref["chi_net_cross"])
    assert substituted == pytest.approx(ref["chi_pos_cross"], rel=EXACT_REL)
    assert abs(ref["chi_pos_cross"]) <= 1.0 + 1e-12, "Cauchy-Schwarz bounds the cross ruler"

    if p_diag is None:
        return
    ref_p = _lnp_cross(a["jac"], a["u"], b["jac"], b["u"], p_diag)
    assert rows[("A", "B", "delta_loss_P")] == pytest.approx(ref_p["delta_loss"], rel=EXACT_REL)
    std_pa = _trace_std(a, p_diag, n_h=N_HUTCH, micro=micro, method=method)
    std_pb = _trace_std(b, p_diag, n_h=N_HUTCH, micro=micro, method=method)
    rel_net_p = 0.5 * (
        K_SIGMA * std_pa / ref_p["chi_net_a"] + K_SIGMA * std_pb / ref_p["chi_net_b"]
    )
    assert abs(rows[("A", "B", "chi_net_P")] - ref_p["chi_net_cross"]) <= (
        rel_net_p * ref_p["chi_net_cross"] + 1e-9
    )
    substituted_p = rows[("A", "B", "delta_loss_P")] / (
        ref["chi_loss_cross"] * ref_p["chi_net_cross"]
    )
    assert substituted_p == pytest.approx(ref_p["chi_pos_cross"], rel=EXACT_REL)
    assert abs(ref_p["chi_pos_cross"]) <= 1.0 + 1e-12
