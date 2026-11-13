"""Toys and ground-truth helpers for the chi_pos^P (preconditioned ruler) tests.

Everything here is fp64 and seeded. Three families:

- :class:`LinearJacobianModel` — a linear model with an *explicitly chosen*
  Jacobian ``J``. Because ``J`` is an input, the exact answer (``Theta_P``, its
  spectrum, ``chi_pos^P``) is known in closed form before the estimator runs.
  Used for T-GN and T-flattening-direction.
- :class:`Fp64MLP` — a genuinely non-linear tiny MLP whose Jacobian has to be
  materialized by autograd (:func:`materialize_jacobian`). Used for T-bruteforce
  and T-identity.
- Ground-truth / tolerance helpers: :func:`materialize_jacobian`,
  :func:`ce_loss_direction`, :func:`preconditioned_reference`,
  :func:`rademacher_trace_std`, :data:`FP32_EPS`.

Note on the loss used by the linear toys: ``L(f) = sum_{a,k} w[a,k] f[a,k]`` is
linear in the outputs, so ``u = grad_f L = w`` *exactly* and the residual
direction is an input to the test rather than something to be inferred. It is a
legitimate loss for every quantity under test (``delta_loss``, ``chi_net`` and
the control-variate estimator only ever need ``u``); ``chi_loss = ||u||^2`` is
then also known exactly, which is what lets the T-GN / T-flattening assertions
be equalities rather than tolerances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relative round-off of one fp32 operation (2^-24). The estimators and
# ``chi_pos`` return fp32 scalars by the repo's pre-existing convention, so any
# "exact" assertion on their *outputs* is bounded below by a few multiples of
# this — see ``fp32_output_slack``.
FP32_EPS = 2.0**-24


def fp32_output_slack(value: float, *, n_ops: int = 4) -> float:
    """Absolute slack from fp32 round-off in the public scalar API.

    ``ChiNetResult.chi_net`` / ``chi_net_p`` are cast to fp32 (matching the raw
    path, which T-identity requires bit-for-bit), and
    :func:`vatis.core.observables.chi_pos` casts its three inputs to fp32 and
    divides in fp32. So a quantity whose true value is ``value`` can be off by
    ``n_ops`` roundings of ``2^-24`` relative — with ``n_ops = 4`` covering the
    cast of each of the three inputs plus the fp32 division. This is a
    floating-point bound, not a fudge factor: it does not grow with the amount
    of work done, and it vanishes if the same comparison is made on the fp64
    components instead.
    """
    return abs(value) * n_ops * FP32_EPS


# --------------------------------------------------------------- linear toys


class LinearJacobianModel(nn.Module):
    """``f(theta)[a, k] = sum_n J[a, k, n] * theta[n]`` — a linear model with a
    prescribed Jacobian.

    The "batch" is a 1-D int tensor of sample indices, so the analyzer's and the
    estimators' micro-batch slicing works unchanged (slicing the index tensor
    selects the corresponding rows of ``J``) and each micro-batch really only
    sees its own outputs.

    ``theta`` starts at zero, which is irrelevant: ``J`` is constant for a linear
    model, so every observable is independent of ``theta``.

    Args:
        jac: ``(n_samples, m, n_params)`` fp64 Jacobian. Flattened over
            ``(sample, output)`` in row-major order it is the ``M x N`` matrix
            ``J`` with ``M = n_samples * m``.
    """

    def __init__(self, jac: torch.Tensor) -> None:
        super().__init__()
        if jac.ndim != 3:
            raise ValueError(f"jac must be (n_samples, m, n_params), got {tuple(jac.shape)}")
        self.register_buffer("jac", jac.detach().clone())
        self.n_samples, self.m, self.n_params = (int(d) for d in jac.shape)
        self.theta = nn.Parameter(torch.zeros(self.n_params, dtype=jac.dtype))

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        return torch.einsum("amn,n->am", self.jac[index], self.theta)

    @property
    def flat_jacobian(self) -> torch.Tensor:
        """``J`` as an ``(M, N)`` matrix, row-major over ``(sample, output)``."""
        return self.jac.reshape(self.n_samples * self.m, self.n_params)


def index_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    """``forward_fn`` for the index-batched linear toys."""
    return model(batch)


def make_linear_loss(w: torch.Tensor) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """``L(f) = sum_{a,k} w[a, k] f[a, k]`` restricted to the batch's samples.

    Linear in ``f``, so ``grad_f L = w[index]`` exactly — the residual direction
    is dictated by the test rather than inferred.
    """

    def loss_fn(logits: torch.Tensor, batch: Any) -> torch.Tensor:
        return (logits * w[batch]).sum()

    return loss_fn


def all_valid_mask(batch: Any, logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    """``valid_mask_fn`` marking every output position valid."""
    return torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)


def gauss_newton_jacobian(
    *,
    n_samples: int,
    m: int,
    n_params: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random ``M x N`` Jacobian (``M = m*n_samples < N``) whose Gauss-Newton
    preconditioner ``(J^T J)^+`` is **diagonal**.

    Construction: ``J[:, live] = Q D`` with ``Q`` a random orthogonal ``M x M``
    matrix (from the QR of a Gaussian), ``D`` a random positive diagonal, and
    ``live`` a random size-``M`` subset of the ``N`` parameter coordinates (the
    remaining coordinates do not affect the outputs — a model with unused
    parameters). Then

        J^T J        = 0 except on live x live, where it is D Q^T Q D = D^2
        (J^T J)^+    = diag(1/d_i^2 on live, 0 elsewhere)
        Theta_P      = J (J^T J)^+ J^T = Q D D^-2 D Q^T = Q Q^T = I_M

    so ``chi_pos^P = u^T Theta_P u / (||u||^2 Tr Theta_P) = 1/M`` for *any* loss
    gradient ``u``, which is what T-GN asserts.

    Why constrain the randomness this way: vatis's ``P`` is diagonal by design
    (non-diagonal ``P`` is explicitly out of scope for WP0), so a genuinely
    dense ``(J^T J)^+`` could not be fed through the production estimator at
    all. Orthogonalizing the live columns keeps ``J`` random, keeps ``N > M``
    and ``rank(J) = M``, and makes the Gauss-Newton ``P`` land in the class the
    estimator supports — so the 1/M assertion tests the *shipping* code path.
    ``test_gn_dense_reference_also_gives_one_over_m`` covers the fully dense
    case as a pure-math reference.

    Returns:
        ``(jac, live, d)`` with ``jac`` of shape ``(M, N)``, ``live`` the
        parameter indices that carry the outputs, and ``d`` the singular values.
    """
    if m * n_samples >= n_params:
        raise ValueError(f"T-GN needs N > M; got N={n_params}, M={m * n_samples}")
    g = torch.Generator().manual_seed(seed)
    big_m = m * n_samples
    q, _ = torch.linalg.qr(torch.randn(big_m, big_m, generator=g, dtype=torch.float64))
    d = torch.exp(torch.randn(big_m, generator=g, dtype=torch.float64) * 0.7)
    live = torch.randperm(n_params, generator=g)[:big_m]
    jac = torch.zeros(big_m, n_params, dtype=torch.float64)
    jac[:, live] = q * d.unsqueeze(0)
    return jac, live, d


def diagonal_spectrum_jacobian(spectrum: torch.Tensor) -> torch.Tensor:
    """``J = diag(sqrt(pi))`` for the diagonal linear toy of T-flattening.

    ``Theta = J J^T = diag(pi)`` exactly, so the eNTK spectrum is the input
    ``spectrum``; ``P = diag(pi)^-1`` flattens it to ``Theta_P = I``.
    """
    big_m = int(spectrum.numel())
    jac = torch.zeros(big_m, big_m, dtype=spectrum.dtype)
    jac[torch.arange(big_m), torch.arange(big_m)] = spectrum.sqrt()
    return jac


# ------------------------------------------------------------------ MLP toys


class Fp64MLP(nn.Module):
    """A tiny fp64 MLP classifier / regressor: ``(B, in_dim) -> (B, n_out)``.

    Deliberately non-linear (tanh) so its Jacobian genuinely has to be
    materialized rather than read off the weights, and small enough
    (a few hundred parameters) that materializing it is instant.
    """

    def __init__(
        self,
        *,
        in_dim: int = 6,
        hidden: int = 8,
        n_out: int = 4,
        seed: int = 0,
        scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_out),
        )
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64) * scale)
        self.double()
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def tuple_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    """``forward_fn`` for ``(inputs, targets)`` tuple batches."""
    return model(batch[0])


def ce_loss(logits: torch.Tensor, batch: Any) -> torch.Tensor:
    """Mean-reduced cross entropy over a ``(x, y)`` tuple batch."""
    return F.cross_entropy(logits, batch[1], reduction="mean")


def mse_loss(logits: torch.Tensor, batch: Any) -> torch.Tensor:
    """Mean-reduced squared error over a ``(x, y)`` tuple batch."""
    return ((logits - batch[1]) ** 2).mean()


def make_gaussian_class_batch(
    *,
    n: int,
    in_dim: int,
    teacher: torch.Tensor,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iid Gaussian inputs with labels ``argmax(x @ teacher)``.

    iid inputs are what T-batchsize needs; the deterministic teacher makes the
    *population* mean gradient substantial, which is what keeps the cross-pair
    ``delta_loss(A, B) -> ||mu||^2`` well above its own sampling noise.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g, dtype=dtype)
    y = (x @ teacher.to(dtype=dtype)).argmax(dim=-1)
    return x, y


# ------------------------------------------------------------- ground truth


def materialize_jacobian(
    model: nn.Module,
    batch: Any,
    params: list[torch.nn.Parameter],
    *,
    forward_fn: Callable[[nn.Module, Any], torch.Tensor] = tuple_forward,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The explicit ``(R, N)`` output-Jacobian, one autograd backward per row.

    Row ``r = idx(sample, output)`` is ``grad_theta f[sample, output]`` flattened
    over ``params`` in list order — i.e. the same flat parameter order the
    preconditioner uses. Only rows at valid positions are included, in row-major
    order.

    Cost is ``R`` backwards, so this is for the smallest toys only. It is the
    same technique as ``tests/unit/test_chi_net.py::_build_full_jacobian_and_u``
    (the repo's existing exact-NTK reference), generalized over ``forward_fn``.
    """
    logits = forward_fn(model, batch)
    rows: list[torch.Tensor] = []
    flat_valid = None if valid_mask is None else valid_mask.reshape(-1)
    n_positions = int(torch.tensor(logits.shape[:-1]).prod().item())
    n_out = int(logits.shape[-1])
    flat_logits = logits.reshape(n_positions, n_out)
    for pos in range(n_positions):
        if flat_valid is not None and not bool(flat_valid[pos]):
            continue
        for out in range(n_out):
            grads = torch.autograd.grad(
                flat_logits[pos, out], params, retain_graph=True, allow_unused=True
            )
            parts = [
                torch.zeros(p.numel(), dtype=torch.float64)
                if g is None
                else g.detach().to(dtype=torch.float64).reshape(-1)
                for p, g in zip(params, grads, strict=True)
            ]
            rows.append(torch.cat(parts))
    return torch.stack(rows, dim=0)


def ce_loss_direction(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``u = grad_f L`` in closed form for mean-reduced CE, flattened row-major.

    ``u_{i,v} = (softmax(z_i)_v - onehot(y_i)_v) / N``.
    """
    n_valid = int(targets.numel())
    probs = F.softmax(logits.detach().to(dtype=torch.float64), dim=-1)
    onehot = F.one_hot(targets, num_classes=int(logits.shape[-1])).to(dtype=torch.float64)
    return ((probs - onehot) / float(n_valid)).reshape(-1)


def preconditioned_reference(
    jac: torch.Tensor,
    u: torch.Tensor,
    p_diag: torch.Tensor,
) -> dict[str, float]:
    """Ground truth for the preconditioned ruler from an explicit ``J``.

    Forms ``Theta_P = J P J^T`` and takes its **eigendecomposition**, then
    reports both the LNP-spectral form and the ratio form of ``chi_pos^P``:

        chi_loss    = ||u||^2
        chi_net^P   = Tr(Theta_P) = sum_k lambda_k
        delta_loss^P= u^T Theta_P u
        chi_pos^P   = sum_k gamma_k^2 p_k,
                      gamma_k^2 = (u^T q_k)^2 / ||u||^2,  p_k = lambda_k / sum lambda
                    = delta_loss^P / (chi_loss * chi_net^P)

    The two ``chi_pos^P`` values are reported separately
    (``chi_pos_p_spectral`` / ``chi_pos_p_ratio``) so a test can assert they
    agree — that equality *is* the LNP factorization, and it is what makes the
    cheap ratio the estimator computes a statement about the spectrum.
    """
    theta_p = (jac * p_diag.unsqueeze(0)) @ jac.T
    evals, evecs = torch.linalg.eigh(theta_p)
    chi_loss = float((u * u).sum())
    trace = float(evals.sum())
    delta_loss_p = float(u @ theta_p @ u)
    gamma_sq = (evecs.T @ u) ** 2 / chi_loss
    p_k = evals / evals.sum()
    return {
        "chi_loss": chi_loss,
        "chi_net_p": trace,
        "delta_loss_p": delta_loss_p,
        "chi_pos_p_spectral": float((gamma_sq * p_k).sum()),
        "chi_pos_p_ratio": delta_loss_p / (chi_loss * trace),
        "chi_net": float((jac * jac).sum()),
        "delta_loss": float(((jac.T @ u) ** 2).sum()),
    }


def rademacher_trace_std(a: torch.Tensor, n_probes: int) -> float:
    """Standard deviation of the ``n``-probe Rademacher estimate of ``Tr(A)``.

    For a Rademacher probe ``z`` and symmetric ``A``,

        E[z^T A z] = Tr(A)
        Var[z^T A z] = 2 * sum_{i != j} A_ij^2        (= 2(||A||_F^2 - sum_i A_ii^2))

    because ``E[z_i z_j z_k z_l]`` vanishes unless the indices pair up and
    ``z_i^2 = 1`` kills the diagonal's contribution. Averaging ``n`` independent
    probes divides the variance by ``n``, hence

        std = sqrt( 2 * sum_{i != j} A_ij^2 / n ).

    This is the *whole* derivation behind the stochastic tolerances used by the
    brute-force tests — with ``A = Theta_P`` for ``hutchinson`` and
    ``A = (I - Pi) Theta_P (I - Pi)`` for ``per_sequence_cv`` (whose exact
    control-variate term contributes no variance at all). Note it predicts
    **zero** variance whenever ``A`` is diagonal, which is why the T-GN and
    T-flattening constructions (``Theta_P = I``) are exact rather than
    statistical.
    """
    off_diagonal = a - torch.diag(torch.diagonal(a))
    variance = 2.0 * float((off_diagonal * off_diagonal).sum())
    return (variance / float(n_probes)) ** 0.5


def loss_direction_projector(u: torch.Tensor, n_samples: int, m: int) -> torch.Tensor:
    """``Pi = block_diag(u_hat_b u_hat_b^T)`` — per_sequence_cv's control variate.

    The output-space projector onto the per-sample loss directions. Needed to
    derive that estimator's residual variance (see
    :func:`rademacher_trace_std`).
    """
    big_m = n_samples * m
    blocks = u.reshape(n_samples, m)
    pi = torch.zeros(big_m, big_m, dtype=u.dtype)
    for b in range(n_samples):
        ub = blocks[b]
        norm_sq = float((ub * ub).sum())
        if norm_sq > 0.0:
            pi[b * m : (b + 1) * m, b * m : (b + 1) * m] = torch.outer(ub, ub) / norm_sq
    return pi
