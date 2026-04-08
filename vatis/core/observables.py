"""Closed-form computation of chi_loss, delta_loss, and chi_pos.

The math here matches CLAUDE.md:

    chi_loss = ||grad_f L||^2 = sum_{b,s,v in valid} (dL/dz_{b,s,v})^2
    delta_loss(A, B) = <grad_theta L^A, grad_theta L^B>
    chi_pos = delta_loss / (chi_loss * chi_net)

For cross-entropy on logits ``z`` with mean reduction over N valid tokens,
``dL/dz_{b,s,v} = (1/N) * (softmax(z_{b,s})_v - onehot(y_{b,s})_v)`` at valid
positions, and zero elsewhere. We exploit this closed form so chi_loss costs
zero backwards through theta.

For other loss functions we fall back to a single ``torch.autograd.grad(L,
logits)`` call, which is one cheap backward through the loss head only.

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

# A small floor on chi_loss * chi_net to avoid 0/0 in chi_pos when both
# components are degenerate (e.g. uniform predictions on labels with all-zero
# loss gradients).
_CHI_POS_DENOM_EPS = 1e-30


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


def chi_loss_from_autograd(
    loss: torch.Tensor,
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generic chi_loss via one autograd backward through the loss head only.

    For non-CE loss functions, we cannot exploit the closed form, but we can
    still get ``grad_f L`` cheaply with a single ``torch.autograd.grad`` call
    that backprops only through the loss function (not through theta). This
    is the fallback path for custom ``loss_fn``s.

    Args:
        loss: scalar loss tensor with a graph back to ``logits``.
        logits: the logit tensor that ``loss`` was computed from. Must have
            ``requires_grad=True`` (set by the caller before the forward).
        valid_mask: optional boolean mask broadcastable to ``logits[..., 0]``;
            invalid positions are zeroed before squaring (so they do not
            contribute even if the loss function happens to leak gradient
            into them).

    Returns:
        A 0-dim tensor holding ``chi_loss``.
    """
    if not logits.requires_grad:
        raise RuntimeError(
            "chi_loss_from_autograd requires logits.requires_grad=True. Set "
            "this *before* the forward pass."
        )
    (grad_f,) = torch.autograd.grad(
        loss,
        logits,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    grad_f = grad_f.to(dtype=torch.float32)
    if valid_mask is not None:
        m = valid_mask.to(dtype=torch.bool)
        while m.ndim < grad_f.ndim:
            m = m.unsqueeze(-1)
        grad_f = grad_f * m.to(dtype=torch.float32)
    return (grad_f * grad_f).sum()


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

    Args:
        delta_loss_value: scalar (tensor or float).
        chi_loss_value: scalar (tensor or float).
        chi_net_value: scalar (tensor or float).

    Returns:
        A 0-dim fp32 tensor on CPU holding chi_pos.
    """
    delta = _to_scalar_fp32(delta_loss_value)
    cl = _to_scalar_fp32(chi_loss_value)
    cn = _to_scalar_fp32(chi_net_value)
    denom = cl * cn
    if abs(float(denom)) < _CHI_POS_DENOM_EPS:
        # NaN here would mask real bugs; return 0 instead, the analyzer will
        # log this row with chi_loss == 0 too so the user can see what happened.
        return torch.zeros((), dtype=torch.float32)
    return torch.tensor(float(delta / denom), dtype=torch.float32)


def _to_scalar_fp32(value: torch.Tensor | float) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).cpu()
    return torch.tensor(float(value), dtype=torch.float32)
