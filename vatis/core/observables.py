"""Closed-form computation of chi_loss, delta_loss, and chi_pos.

The math here matches CLAUDE.md:

    chi_loss = ||grad_f L||^2 = sum_{b,s,v in valid} (dL/dz_{b,s,v})^2
    delta_loss(A, B) = <grad_theta L^A, grad_theta L^B>
    chi_pos = delta_loss / (chi_loss * chi_net)

For cross-entropy on logits ``z`` with mean reduction over N valid tokens,
``dL/dz_{b,s,v} = (1/N) * (softmax(z_{b,s})_v - onehot(y_{b,s})_v)`` at valid
positions, and zero elsewhere. We exploit this closed form so chi_loss costs
zero backwards through theta. v1 only supports the cross-entropy path;
non-CE losses are deferred to v1.2 (see SESSION_SUMMARY.md).

This module knows nothing about HF, DDP, sinks, or even nn.Module — it takes
plain tensors and parameter iterables.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from vatis.core.normalization import (
    DEFAULT_IGNORE_INDEX,
    valid_token_mask,
)

# chi_pos is degenerate only when its denominator is exactly zero — chi_loss and
# chi_net are both non-negative, so ``chi_loss * chi_net == 0`` means one of them
# vanished (no valid tokens, or a Jacobian that is identically zero) and the
# ratio has no value. There used to be an *absolute* 1e-30 floor here, which had
# two failure modes of its own: a legitimately tiny-over-tiny ratio was reported
# as 0, and (with the fp32 arithmetic this function used to do) an out-of-range
# product underflowed to 0 or overflowed to inf and was also reported as 0. Both
# looked exactly like a true zero.
_CHI_POS_DEGENERATE = 0.0


def chi_loss_cross_entropy_unnormalized(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form ``Sigma_{valid} (softmax - onehot)^2`` — without the 1/N^2 factor.

    This is the raw building block of chi_loss. The analyzer uses it inside
    the micro-batch loop and divides by ``N_total^2`` once at the end, which
    ensures correctness when ``N_total`` spans multiple micro-batches and/or
    DDP ranks.

    Returns:
        A 0-dim fp32 tensor holding the unnormalized squared sum of
        ``(softmax(z) - onehot(y))`` at valid positions.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"shape mismatch: logits.shape[:-1]={tuple(logits.shape[:-1])} != "
            f"targets.shape={tuple(targets.shape)}"
        )

    mask = valid_token_mask(targets, ignore_index=ignore_index, attention_mask=attention_mask)
    if int(mask.sum().item()) <= 0:
        return torch.zeros((), dtype=torch.float32, device=logits.device)

    z = logits.detach().to(dtype=torch.float32)
    probs = F.softmax(z, dim=-1)
    safe_targets = targets.clone()
    safe_targets[~mask] = 0
    onehot = F.one_hot(safe_targets, num_classes=z.shape[-1]).to(dtype=torch.float32)
    diff = probs - onehot  # (..., V)
    mask_f = mask.to(dtype=torch.float32).unsqueeze(-1)
    masked_diff = diff * mask_f
    return (masked_diff * masked_diff).sum().to(dtype=torch.float32)


def chi_loss_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form chi_loss for mean-reduced cross entropy on a single batch.

    This computes ``sum_{valid} (dL/dz)^2`` where ``L = (1/N) * sum_{valid}
    CE(z, y)`` and ``N`` is the number of valid tokens **in the input batch**.
    No backward through theta is performed.

    For multi-micro-batch or DDP usage, prefer the unnormalized variant
    :func:`chi_loss_cross_entropy_unnormalized` and divide by ``N_total^2``
    at the end so the normalization uses the global token count.

    Supports two shapes:

    - LM-style logits ``(B, S, V)`` with targets ``(B, S)``.
    - Classifier-style logits ``(B, V)`` with targets ``(B,)``.

    Returns:
        A 0-dim fp32 tensor holding ``chi_loss``. Always non-negative.
    """
    mask = valid_token_mask(targets, ignore_index=ignore_index, attention_mask=attention_mask)
    n_valid = float(mask.sum().item())
    if n_valid <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=logits.device)
    raw = chi_loss_cross_entropy_unnormalized(
        logits, targets, ignore_index=ignore_index, attention_mask=attention_mask
    )
    return (raw.to(dtype=torch.float64) / (n_valid * n_valid)).to(dtype=torch.float32)


def parameter_grad_vector(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten ``p.grad`` for each parameter into a single 1-D fp32 tensor.

    Parameters with ``grad is None`` contribute zeros of the right size.
    """
    flats: list[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            flats.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        else:
            flats.append(p.grad.detach().to(dtype=torch.float32).reshape(-1))
    if not flats:
        raise ValueError("no parameters supplied to parameter_grad_vector")
    return torch.cat(flats)


def delta_loss_self(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> torch.Tensor:
    """Compute ``delta_loss(A, A) = ||grad_theta L||^2`` for a single batch.

    Performs one backward through theta and returns ``sum_p ||grad_p||^2``.
    Leaves the parameter grads populated (so the caller can re-use them
    immediately, e.g. for cross-pair work) but does NOT zero them; the
    caller is responsible for ``model.zero_grad()`` afterwards.

    Args:
        loss: scalar loss tensor with a graph back to ``params``.
        params: list of parameters; the order matters only if the caller
            wants to consume them via :func:`parameter_grad_vector`.

    Returns:
        A 0-dim fp32 tensor holding ``||grad_theta L||^2``.
    """
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    total = torch.zeros((), dtype=torch.float32, device=loss.device)
    for p, g in zip(params, grads, strict=False):
        if g is None:
            continue
        # Stash the grad on the parameter as well so the caller can reuse it.
        p.grad = g.detach()
        total = total + (g.to(dtype=torch.float32) ** 2).sum()
    return total


def delta_loss_cross(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> torch.Tensor:
    """Compute ``delta_loss(A, B) = <grad_theta L^A, grad_theta L^B>``.

    Performs two ordinary backwards (one per loss) plus a single dot product.
    Does NOT use the parameters' ``.grad`` attribute, so the caller's training
    state is untouched.

    Args:
        loss_a, loss_b: scalar loss tensors, each with their own graph back
            to ``params``.
        params: parameter list. Order is irrelevant for the dot product result.

    Returns:
        A 0-dim fp32 tensor holding the inner product.
    """
    grads_a = torch.autograd.grad(loss_a, params, retain_graph=False, allow_unused=True)
    grads_b = torch.autograd.grad(loss_b, params, retain_graph=False, allow_unused=True)
    total = torch.zeros((), dtype=torch.float32, device=loss_a.device)
    for ga, gb in zip(grads_a, grads_b, strict=False):
        if ga is None or gb is None:
            continue
        total = total + (ga.to(dtype=torch.float32) * gb.to(dtype=torch.float32)).sum()
    return total


def chi_pos(
    delta_loss_value: torch.Tensor | float,
    chi_loss_value: torch.Tensor | float,
    chi_net_value: torch.Tensor | float,
) -> torch.Tensor:
    """Combine the three observables into chi_pos.

    chi_pos = delta_loss / (chi_loss * chi_net)

    The math is invariant under the normalization choice: the
    ``sqrt(N_A * N_B)`` factors that appear in the normalized chi_loss and
    normalized chi_net cancel exactly. So we can pass either the raw
    (unnormalized) values or the normalized values; the result is the same.

    **The division runs in fp64.** The callers already hold fp64 Python floats
    (the analyzer accumulates in fp64), and this function used to cast them down
    to fp32 before dividing — which cost ~6e-08 relative for nothing and, worse,
    put the fp32 *range* in the way: ``chi_pos(1.0, 1e30, 1e30)`` returned
    ``0.0`` instead of ``1e-60`` because the product overflowed fp32 to ``inf``.
    Not reachable with realistic magnitudes (``chi_loss * chi_net`` sits around
    1e1..1e8 even with an AdamW ``P`` at the ``1/eps`` ceiling), but the failure
    *shape* — a zero indistinguishable from a true zero — is the one this repo
    keeps getting bitten by, so it is removed rather than documented.

    Args:
        delta_loss_value: scalar (tensor or float).
        chi_loss_value: scalar (tensor or float).
        chi_net_value: scalar (tensor or float).

    Returns:
        A 0-dim fp64 tensor on CPU holding chi_pos. ``0.0`` when the denominator
        is exactly zero (see :data:`_CHI_POS_DEGENERATE`); non-finite inputs are
        allowed to propagate rather than being masked, because a NaN that reaches
        the sink is a bug worth seeing.
    """
    delta = _to_scalar_fp64(delta_loss_value)
    cl = _to_scalar_fp64(chi_loss_value)
    cn = _to_scalar_fp64(chi_net_value)
    denom = cl * cn
    if float(denom) == 0.0:
        return torch.tensor(_CHI_POS_DEGENERATE, dtype=torch.float64)
    return torch.tensor(float(delta / denom), dtype=torch.float64)


def chi_pos_exceeds_cauchy_schwarz(value: float, *, slack: float = 0.5) -> bool:
    """Whether ``chi_pos`` has left the ``[-1, 1]`` range its definition guarantees.

    ``|chi_pos| <= 1`` is a Cauchy-Schwarz consequence, for the self pair
    (``u^T Theta u <= ||u||^2 lambda_max <= ||u||^2 Tr Theta``) and for the cross
    pair (``|u_B^T J_B J_A^T u_A| <= ||J_B^T u_B|| ||J_A^T u_A|| <=
    chi_loss^{AB} chi_net^{AB}``). It therefore cannot be violated by a correct
    measurement, only by an *inconsistency* between the three components — most
    plausibly a ``loss_fn`` that is not mean-reduced over the same valid tokens
    ``chi_loss`` normalizes by (a sum-reduced CE inflates ``chi_pos`` by the square
    of the valid-token count *per micro-batch* — ``N^2`` only when the whole batch
    is one micro-batch), or an ``ignore_index`` mismatch between the analyzer and
    the loss.

    ``slack`` exists because ``chi_net`` is a *stochastic* estimate: at small
    ``n_hutchinson`` a legitimate ``chi_pos`` near 1 can overshoot. The default
    0.5 matches the bound the golden gate already asserts.
    """
    return abs(value) > 1.0 + slack


def _to_scalar_fp64(value: torch.Tensor | float) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float64).cpu()
    return torch.tensor(float(value), dtype=torch.float64)
