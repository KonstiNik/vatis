"""The analytic one-hot / diagonal toy — the code-vs-theory bridge fixture.

This is the smallest setting in which **every** LNP component has a closed form,
so a mismatch means vatis and the theory name different objects rather than that
an estimator is noisy.

Setup
-----
``M`` modes (tokens) ``k = 0 .. M-1`` with multiplicities ``n_k``, batch size
``b = sum_k n_k`` and frequencies ``pi_k = n_k / b`` that are **exact** because
the batch is constructed to realise them. The model is

    f_j(theta) = theta[i_j]                      (one output per sample)

so ``J`` has rows ``e_{i_j}`` and ``Theta = J J^T`` is the block-indicator
matrix. With the mean-reduced squared loss

    L = (1/(2b)) sum_j (f_j - y_{i_j})^2,        r_k := theta_k - y_k,  s_k := |r_k|

    u_j          = r_{i_j} / b
    chi_loss     = ||u||^2       = (1/b^2) sum_k n_k r_k^2
    chi_net      = Tr(Theta)     = b
    grad_theta L = (n_k / b) r_k
    delta_loss   = sum_k (n_k/b)^2 r_k^2
    chi_pos      = sum_k n_k^2 r_k^2 / (b sum_k n_k r_k^2)

and under a preconditioner:

* **Newton / Gauss-Newton**, ``P = (J^T J)^+ = diag(1/n_k)``: ``Theta_P`` has
  eigenvalue 1 on each present mode and 0 elsewhere, and ``u`` lies entirely in
  that span, so ``chi_pos^P = 1/M`` **for any residual**;
* **full-batch Adam**, ``P = 1/(|g| + eps)`` at ``eps = 0``, i.e.
  ``P_k = 1/(pi_k s_k)``: ``lambda_k^P = b / s_k`` and
  ``gamma_k^2 = pi_k s_k^2 / sum_j pi_j s_j^2``, hence

      chi_pos^P = (sum_k pi_k s_k) / ((sum_k pi_k s_k^2) (sum_k 1/s_k))

  which is ``<= 1/M`` with equality iff every residual is equal (Cauchy-Schwarz
  twice).

Why this toy is *exact* rather than statistical
----------------------------------------------
With one output per sample (``m = 1``) the per-sequence-CV control variate's
output-space projector ``Pi = block_diag(u_hat_b u_hat_b^T)`` is the **identity**,
so its Hutchinson residual is identically zero and
``PerSequenceControlVariateEstimator`` returns ``Tr(Theta_P)`` exactly. Plain
``hutchinson`` is exact only when every ``n_k == 1`` (then ``Theta_P`` is
diagonal and the Rademacher variance ``2 sum_{i != j} A_ij^2`` vanishes); the
non-uniform case needs the derived bound.

Caveat worth knowing: with repeated tokens ``J`` is **rank-deficient**, so the
general Gauss-Newton statement is ``chi_pos^P = ||Pi u||^2 / (||u||^2 rank)``
and it collapses to ``1/rank`` only because this toy's ``u`` lies in
``range(J)``. ``00_context.md``'s "``1/M`` for any loss gradient" needs the
full-row-rank hypothesis; `tests/fixtures/preconditioned_toys.py`'s T-GN
construction supplies it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

FP64 = torch.float64


class OneHotModel(nn.Module):
    """``f(theta)[j, 0] = theta[token_j]`` — a prescribed one-hot Jacobian."""

    def __init__(self, n_modes: int, *, init: torch.Tensor | None = None) -> None:
        super().__init__()
        self.n_modes = int(n_modes)
        theta = torch.zeros(n_modes, dtype=FP64) if init is None else init.detach().to(FP64).clone()
        self.theta = nn.Parameter(theta)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.theta[tokens].unsqueeze(-1)  # (b, 1)


def onehot_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    """``forward_fn`` for the ``{"tokens": ...}`` batches this toy uses."""
    return model(batch["tokens"])


def onehot_all_valid(batch: Any, logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    """``valid_mask_fn`` — every output position is valid in this toy."""
    return torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)


def make_onehot_mse_loss(targets: torch.Tensor) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """``L = (1/(2b)) sum_j (f_j - y_{i_j})^2``, so ``u_j = r_{i_j} / b``.

    The ``1/2`` is what makes ``grad_theta L`` come out as ``(n_k/b) r_k``
    exactly, matching the closed forms in the module docstring.
    """

    def loss_fn(logits: torch.Tensor, batch: Any) -> torch.Tensor:
        tok = batch["tokens"]
        resid = logits.reshape(-1) - targets.to(logits.dtype)[tok]
        return (resid * resid).sum() / (2.0 * float(tok.numel()))

    return loss_fn


def build_onehot_case(
    *,
    multiplicities: list[int],
    residuals: list[float],
    seed: int,
) -> dict[str, Any]:
    """Model, batch, the two reference ``P``s and every closed-form answer.

    ``theta`` is initialised to ``r`` and the targets to zero, so the residual is
    exactly the ``r`` the caller asked for. The token order is shuffled with
    ``seed`` so no test depends on modes arriving in blocks.
    """
    n_k = torch.tensor(multiplicities, dtype=FP64)
    r_k = torch.tensor(residuals, dtype=FP64)
    if int(n_k.numel()) != int(r_k.numel()):
        raise ValueError("one multiplicity per mode")
    m_modes = int(n_k.numel())
    b = int(n_k.sum().item())
    pi_k = n_k / float(b)
    s_k = r_k.abs()
    if bool((s_k <= 0).any()):
        raise ValueError("residuals must be non-zero: the Adam P reference divides by s_k")

    tokens = torch.cat(
        [torch.full((int(n),), k, dtype=torch.long) for k, n in enumerate(multiplicities)]
    )
    tokens = tokens[torch.randperm(b, generator=torch.Generator().manual_seed(seed))]

    model = OneHotModel(m_modes, init=r_k.clone())
    loss_fn = make_onehot_mse_loss(torch.zeros(m_modes, dtype=FP64))

    jac = torch.zeros(b, m_modes, dtype=FP64)
    jac[torch.arange(b), tokens] = 1.0
    u = r_k[tokens] / float(b)

    return {
        "model": model,
        "params": [model.theta],
        "batch": {"tokens": tokens},
        "loss_fn": loss_fn,
        "jac": jac,
        "u": u,
        "n_k": n_k,
        "pi_k": pi_k,
        "s_k": s_k,
        "b": b,
        "m_modes": m_modes,
        # P = (J^T J)^+ : diagonal because J^T J = diag(n_k) here
        "p_newton": 1.0 / n_k,
        # P = 1/(|g| + 0) with g_k = (n_k/b) r_k = pi_k r_k, exact by construction
        "p_adam": 1.0 / (pi_k * s_k),
        "chi_loss": float((u * u).sum()),
        "chi_net": float(b),
        "closed_form_raw": float((n_k**2 * s_k**2).sum() / (float(b) * (n_k * s_k**2).sum())),
        "closed_form_newton": 1.0 / m_modes,
        "closed_form_adam": float(
            (pi_k * s_k).sum() / ((pi_k * s_k * s_k).sum() * (1.0 / s_k).sum())
        ),
    }


def lnp_from_explicit_kernel(
    jac: torch.Tensor,
    u: torch.Tensor,
    p_diag: torch.Tensor | None = None,
) -> dict[str, float]:
    """``chi_pos`` from an explicit ``eigh(Theta_P)``, plus the ratio form.

    Reports both so a test can assert they agree — that equality *is* the LNP
    factorisation, and it is what makes the cheap ratio vatis computes a
    statement about the spectrum.
    """
    jac = jac.to(FP64)
    u = u.to(FP64)
    weight = None if p_diag is None else p_diag.to(FP64)
    if weight is None:
        theta = jac @ jac.T
        chi_net = float((jac * jac).sum())
        grad = jac.T @ u
        delta = float((grad * grad).sum())
    else:
        theta = (jac * weight.unsqueeze(0)) @ jac.T
        chi_net = float(((jac * jac) * weight.unsqueeze(0)).sum())
        grad = jac.T @ u
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
