"""The diagonal preconditioner ``P`` and the elementwise weighting it induces.

An optimizer with update ``dtheta = -eta * P grad L`` does not train on the raw
eNTK ``Theta = J J^T`` but on the **effective kernel** ``Theta_P = J P J^T``
(``00_context.md`` §"Preconditioned ruler", ``background.md`` §4.2). vatis
measures the same LNP triple on that kernel:

    delta_loss^P(A, B) = grad L^B . P grad L^A     (elementwise-weighted dot)
    chi_net^P          = ||P^{1/2} J||_F^2 = Tr(Theta_P)
    chi_loss           = unchanged (kernel-independent)
    chi_pos^P          = delta_loss^P / (chi_loss * chi_net^P)

``P`` is **diagonal** throughout vatis: a 1-D vector in the canonical parameter
order (:class:`~vatis.core.param_layout.ParameterLayout`), segment-for-segment
aligned with the flat gradient vector the analyzer builds and with
``OptimizerStateView.flat_state``. Non-diagonal ``P`` (Muon's orthogonalization)
is explicitly out of scope for WP0 and this module is deliberately not designed
for it — every consumer here assumes an elementwise weight.

This module owns three things and nothing else:

- :class:`Preconditioner` — a validated diagonal ``P`` plus the ``descriptor``
  string that ends up in the parquet ``preconditioner`` column;
- the builders for the P sources of ``03_estimator.md`` §1 that are pure math
  (``identity``, ``explicit``, ``adamw``). The bundle plumbing for ``adamw``
  (which optimizer state, which hparams, what to do at step 0) lives in
  :mod:`vatis.models.local_bundle`, because that is where the saved state is;
- the elementwise-weighted reductions the estimators and the analyzer need
  (:func:`weighted_sum_sq_fp64`, :func:`weighted_dot_fp64`,
  :func:`weighted_sum_sq_grads`).

It knows nothing about HF, DDP, sinks or ``nn.Module`` beyond ``nn.Parameter``
shapes, and is device-agnostic: every reduction runs on the device its inputs
are on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

# The parquet ``preconditioner`` value of a raw (un-preconditioned) row. Not a
# valid ``Preconditioner.descriptor`` — a raw row is the *absence* of a P, not
# an identity P, and the two must stay distinguishable downstream.
RAW_DESCRIPTOR = "none"

# Chunk size for the fp64 reductions below, matching the analyzer's own chunked
# helpers: it caps the fp64 temporaries at ~32M elements instead of one
# full-length copy (a P-sized fp64 temporary is ~11 GiB for a 1.4B model).
_CHUNK = 1 << 25


@dataclass(frozen=True, eq=False)
class Preconditioner:
    """A diagonal preconditioner in the canonical parameter order.

    Attributes:
        diag: 1-D tensor of length ``n_params`` holding the diagonal of ``P``.
            Must be finite and non-negative — ``chi_net^P`` is
            ``E_z ||P^{1/2} (J^T z)||^2``, so a negative entry has no square
            root and would silently turn the trace estimate into a difference.
        descriptor: what the parquet ``preconditioner`` column records for rows
            computed with this ``P`` (e.g. ``"adamw"``, ``"identity"``,
            ``"explicit"``, ``"gn"``). Must not be :data:`RAW_DESCRIPTOR`.

    Instances are immutable; :meth:`to` returns a new one.
    """

    diag: torch.Tensor
    descriptor: str

    def __post_init__(self) -> None:
        if not isinstance(self.diag, torch.Tensor):
            raise TypeError(
                f"Preconditioner.diag must be a torch.Tensor, got {type(self.diag).__name__}"
            )
        if self.diag.ndim != 1:
            raise ValueError(
                f"Preconditioner.diag must be 1-D (the canonical flat parameter order), "
                f"got shape {tuple(self.diag.shape)}"
            )
        if self.diag.numel() == 0:
            raise ValueError("Preconditioner.diag is empty")
        if not self.diag.is_floating_point():
            raise ValueError(f"Preconditioner.diag must be a float tensor, got {self.diag.dtype}")
        if not bool(torch.isfinite(self.diag).all()):
            raise ValueError(
                "Preconditioner.diag contains non-finite entries; P must be finite "
                "(an inf/NaN weight would poison delta_loss^P and chi_net^P silently)"
            )
        if bool((self.diag < 0).any()):
            raise ValueError(
                "Preconditioner.diag contains negative entries; P must be positive "
                "semi-definite so that P^{1/2} exists (chi_net^P is estimated as "
                "E_z ||P^{1/2}(J^T z)||^2)"
            )
        if not isinstance(self.descriptor, str) or not self.descriptor:
            raise ValueError(
                f"Preconditioner.descriptor must be a non-empty str, got {self.descriptor!r}"
            )
        if self.descriptor == RAW_DESCRIPTOR:
            raise ValueError(
                f"{RAW_DESCRIPTOR!r} is reserved for raw (un-preconditioned) rows; "
                "pick a descriptor that names the preconditioner"
            )

    # -------------------------------------------------------------- accessors

    @property
    def n_params(self) -> int:
        """Length of the flat parameter vector this ``P`` weights."""
        return int(self.diag.numel())

    @property
    def device(self) -> torch.device:
        """Device the diagonal lives on."""
        return self.diag.device

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the diagonal."""
        return self.diag.dtype

    def __len__(self) -> int:
        return self.n_params

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Preconditioner(descriptor={self.descriptor!r}, n_params={self.n_params}, "
            f"dtype={self.diag.dtype}, device={self.diag.device})"
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Preconditioner:
        """Return this ``P`` on another device / in another dtype.

        Returns ``self`` when nothing would change, so the common
        already-on-device case costs no copy.
        """
        diag = self.diag
        if device is not None:
            diag = diag.to(device=device)
        if dtype is not None:
            diag = diag.to(dtype=dtype)
        if diag is self.diag:
            return self
        return Preconditioner(diag=diag, descriptor=self.descriptor)

    # ------------------------------------------------------------ validation

    def validate_for(
        self,
        params: Sequence[torch.nn.Parameter] | Sequence[torch.Tensor],
        *,
        context: str = "parameter list",
    ) -> None:
        """Raise unless ``P`` has one entry per parameter element of ``params``.

        This is the alignment guard: the flat gradient vector vatis builds is
        ``cat([p.grad.reshape(-1) for p in params])``, so ``P`` must have
        exactly ``sum(p.numel())`` entries — anything else means the
        preconditioner and the gradient are not describing the same parameter
        space, and every ``chi_pos^P`` computed from them would be a plausible
        looking wrong number. Length agreement is necessary, not sufficient;
        the *order* is guaranteed by both sides going through the same
        :class:`~vatis.core.param_layout.ParameterLayout`.
        """
        expected = sum(int(p.numel()) for p in params)
        if expected != self.n_params:
            raise ValueError(
                f"preconditioner {self.descriptor!r} has {self.n_params} entries but "
                f"{context} flattens to {expected} elements; P must be aligned with the "
                "canonical parameter order (vatis.core.param_layout.ParameterLayout)"
            )

    def parameter_views(
        self,
        params: Sequence[torch.nn.Parameter] | Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Split ``diag`` into per-parameter views shaped like ``params``.

        The estimators consume ``torch.autograd.grad`` output, which is a tuple
        of per-parameter tensors rather than one flat vector. These views let
        them apply the elementwise weight without ever flattening a gradient
        (keeping the O(P)-memory property of the raw paths).

        The result aliases :attr:`diag`; do not mutate it.
        """
        self.validate_for(params, context="parameter list")
        views: list[torch.Tensor] = []
        offset = 0
        for p in params:
            n = int(p.numel())
            views.append(self.diag[offset : offset + n].view(tuple(p.shape)))
            offset += n
        return views

    # -------------------------------------------------------------- builders

    @classmethod
    def identity(
        cls,
        n_params: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        descriptor: str = "identity",
    ) -> Preconditioner:
        """``P = I`` — the SGD ruler, materialized.

        The ones are materialized on purpose rather than short-circuited: the
        point of the ``P = I`` runs (``03_estimator.md`` T-identity) is to drive
        the *preconditioned* code path — its alignment, normalization and
        combinator — and check that it reproduces the raw columns. A
        short-circuit would make that test vacuous. Multiplying by exactly
        ``1.0`` is exact in IEEE-754 and the reductions keep the raw
        accumulation order, so the agreement is bit-for-bit, not merely within
        tolerance.

        Costs one extra fp32 parameter-sized vector; use ``preconditioner=None``
        (not ``identity``) when you simply do not want the P ruler.
        """
        return cls(
            diag=torch.ones(int(n_params), dtype=dtype, device=device),
            descriptor=descriptor,
        )

    @classmethod
    def from_diagonal(
        cls,
        diag: torch.Tensor,
        *,
        descriptor: str = "explicit",
        dtype: torch.dtype | None = None,
    ) -> Preconditioner:
        """The ``explicit`` P source: a user-supplied 1-D diagonal.

        The tensor is detached (a ``P`` carrying an autograd graph would make
        every backward in the analyzer differentiate through the ruler).
        """
        d = diag.detach()
        if dtype is not None:
            d = d.to(dtype=dtype)
        return cls(diag=d, descriptor=descriptor)

    @classmethod
    def adamw(
        cls,
        exp_avg_sq: torch.Tensor,
        *,
        step: int,
        beta2: float,
        eps: float,
        descriptor: str = "adamw",
    ) -> Preconditioner:
        """The ``adamw`` P source: ``P = diag(1 / (sqrt(v_hat) + eps))``.

        With ``v_hat = exp_avg_sq / (1 - beta2**step)`` — the bias-corrected
        second moment (``00_context.md`` §"P conventions").

        ``step`` is the **saved** step counter. vatis reads bundles with
        *pre-step* semantics: the checkpoint at step ``t`` stores the optimizer
        state as it was *before* the step-``t`` update, so the number of updates
        already folded into ``exp_avg_sq`` is exactly that counter, and it is
        therefore the bias-correction exponent that matches the saved tensor.
        (The exact step-``t`` Adam ``P`` would additionally fold in the step-``t``
        gradient — one EMA update, an ``O(1 - beta2)`` difference, documented as
        negligible in ``00_context.md`` and not implemented.)

        **Deliberately excluded from P: the first moment ``m_hat`` and decoupled
        weight decay.** The full AdamW update is
        ``dtheta = -lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)``. ``P`` is
        the *geometry* factor only — the linear map that turns the gradient into
        the update direction — because ``Theta_P = J P J^T`` is defined for a
        preconditioner, and momentum is a (gradient-history) filter, not a
        metric. Weight decay does not act on the gradient at all. Including
        either would not correspond to a kernel, and the spec fixes the
        convention: geometry factor only.

        **Coordinates with ``exp_avg_sq == 0`` get ``P = 1/eps``, and that is
        intended.** Real runs have such coordinates: any parameter element the
        gradient never touched keeps ``v = 0`` exactly — unused vocabulary rows
        of an embedding are the obvious population (measured on a real bundle:
        ``exp_avg_sq.min() == 0`` at every checkpoint). With ``eps = 1e-8`` those
        entries get ``P = 1e8``, of order ``1e6`` times a live coordinate's
        weight. Consequences to be aware of when reading a number:

        - ``chi_net^P = Tr(J P J^T) = sum_i P_i ||J_{:,i}||^2``, so a dead
          coordinate is harmless *only if* its Jacobian column also vanishes on
          the measurement batch. That holds when the measurement batch uses only
          tokens the training batches used — and **fails in general**: a token
          present in a held-out measurement batch but absent from the batches
          that built ``v`` has a non-zero Jacobian column *and* ``P = 1/eps``, so
          it enters ``chi_net^P`` with ~``1e6`` times the weight of a live
          direction. On held-out data that is the normal case, not a corner case.
        - No clipping, flooring or dead-coordinate mask is applied: the
          convention ``P = 1/(sqrt(v_hat) + eps)`` is a frozen project decision
          (``00_context.md`` §"P conventions", "do not revisit"), and quietly
          altering it would make the reported ruler not the one the paper
          defines. The right response is to *see* the effect, not to hide it:
          :func:`epsilon_cap_mask` identifies the affected coordinates so a
          caller can report what share of ``chi_net^P`` they carry (the golden
          test does exactly that).

        Args:
            exp_avg_sq: 1-D flat second moment in the canonical order, e.g.
                ``optimizer_state.flat_state("exp_avg_sq")``.
            step: number of updates already applied (``>= 1``).
            beta2: Adam's second-moment decay.
            eps: Adam's epsilon (inside the square root's *sum*, matching
                ``torch.optim.AdamW``'s default ``eps`` placement).
            descriptor: parquet descriptor for the resulting rows.

        Raises:
            ValueError: ``step < 1`` (bias correction undefined — a step-0
                checkpoint has no Adam preconditioner at all), ``beta2`` outside
                ``(0, 1)``, ``eps < 0``, or a bias correction that underflows.
        """
        if exp_avg_sq.ndim != 1:
            raise ValueError(
                f"adamw preconditioner expects a flat exp_avg_sq, got shape "
                f"{tuple(exp_avg_sq.shape)}"
            )
        if int(step) < 1:
            raise ValueError(
                f"adamw bias correction needs step >= 1, got {step}. With pre-step "
                "semantics a step-0 checkpoint has an empty optimizer state and no "
                "Adam preconditioner exists for it; P = 1/(sqrt(0) + eps) = 1/eps is "
                "a garbage preconditioner, not a neutral one."
            )
        if not 0.0 < float(beta2) < 1.0:
            raise ValueError(f"adamw beta2 must be in (0, 1), got {beta2}")
        if float(eps) < 0.0:
            raise ValueError(f"adamw eps must be >= 0, got {eps}")
        bias_correction = 1.0 - float(beta2) ** int(step)
        if not bias_correction > 0.0:
            raise ValueError(
                f"adamw bias correction 1 - beta2**step underflowed to {bias_correction} "
                f"(beta2={beta2}, step={step})"
            )
        v_hat = exp_avg_sq.detach() / bias_correction
        diag = 1.0 / (v_hat.sqrt() + float(eps))
        return cls(diag=diag, descriptor=descriptor)


def epsilon_cap_mask(
    diag: torch.Tensor,
    eps: float,
    *,
    rtol: float = 1e-6,
) -> torch.Tensor:
    """Which entries of an AdamW ``P`` sit at the ``1/eps`` ceiling.

    ``P_i = 1/(sqrt(v_hat_i) + eps)`` is maximal, and equal to ``1/eps``, exactly
    where ``v_hat_i == 0`` — a parameter coordinate no gradient ever touched (see
    :meth:`Preconditioner.adamw` for why those exist in real runs and why they
    are *not* clipped). Such coordinates carry ~``1/eps`` times the weight of a
    converged one, so a ``chi_net^P`` that happens to be dominated by them is
    saying something about dead directions rather than about learning.

    This helper exists so that fact can be **surfaced** — count the mask, or use
    it to split ``chi_net^P`` into a live and a capped part (run the estimator
    twice with the same probe seed, once with ``P`` and once with
    ``P * mask``; the trace estimate is linear in ``P`` at fixed probes, so the
    two parts add up exactly).

    Args:
        diag: the ``P`` diagonal (``Preconditioner.diag``).
        eps: the ``eps`` the ``P`` was built with.
        rtol: relative slack, so fp32 round-off near the ceiling still counts.

    Returns:
        A boolean tensor, ``True`` where ``P`` is at the ceiling.
    """
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0 to have a 1/eps ceiling, got {eps}")
    return diag >= (1.0 / float(eps)) * (1.0 - float(rtol))


# ------------------------------------------------------- weighted reductions


def weighted_sum_sq_fp64(
    flat: torch.Tensor,
    diag: torch.Tensor,
    *,
    chunk: int = _CHUNK,
) -> torch.Tensor:
    """``sum_i diag_i * flat_i^2`` accumulated in fp64, chunked.

    The P-weighted counterpart of the analyzer's ``_sum_sq_fp64``: with
    ``flat = grad_theta L`` this is ``delta_loss^P(A, A) = grad L^T P grad L``.
    Chunking keeps the fp64 temporaries bounded (see :data:`_CHUNK`).

    ``diag`` may live on another device than ``flat`` (e.g. a host-offloaded
    cross-pair gradient): each chunk is moved as needed.
    """
    if flat.numel() != diag.numel():
        raise ValueError(
            f"weighted_sum_sq_fp64: flat has {flat.numel()} elements, diag has {diag.numel()}"
        )
    total = torch.zeros((), dtype=torch.float64, device=flat.device)
    for i in range(0, flat.numel(), chunk):
        c = flat[i : i + chunk].to(dtype=torch.float64)
        w = diag[i : i + chunk].to(dtype=torch.float64, device=flat.device)
        total = total + (w * c * c).sum()
    return total


def weighted_dot_fp64(
    a: torch.Tensor,
    b: torch.Tensor,
    diag: torch.Tensor,
    *,
    chunk: int = _CHUNK,
) -> torch.Tensor:
    """``sum_i diag_i * a_i * b_i`` accumulated in fp64, chunked.

    With ``a = grad_theta L^A`` and ``b = grad_theta L^B`` this is the cross
    ``delta_loss^P(A, B) = grad L^B . P grad L^A``.
    """
    if a.numel() != b.numel() or a.numel() != diag.numel():
        raise ValueError(
            f"weighted_dot_fp64: mismatched lengths a={a.numel()}, b={b.numel()}, "
            f"diag={diag.numel()}"
        )
    total = torch.zeros((), dtype=torch.float64, device=a.device)
    for i in range(0, a.numel(), chunk):
        ca = a[i : i + chunk].to(dtype=torch.float64)
        cb = b[i : i + chunk].to(dtype=torch.float64, device=a.device)
        w = diag[i : i + chunk].to(dtype=torch.float64, device=a.device)
        total = total + (w * ca * cb).sum()
    return total


def weighted_sum_sq_grads(
    grads: Iterable[torch.Tensor | None],
    views: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    """``sum_p sum_i P_{p,i} * grad_{p,i}^2`` from an ``autograd.grad`` tuple.

    The P-weighted counterpart of ``per_seq_cv._sum_sq`` / the Hutchinson
    accumulator: with ``grad = J^T z`` this is ``||P^{1/2} (J^T z)||^2``, the
    P-weighted VJP norm whose expectation over probes is ``Tr(J P J^T)``.

    ``views`` comes from :meth:`Preconditioner.parameter_views` and must be
    positionally aligned with ``grads`` (same order as ``params``) and on the
    same device — a device mismatch raises rather than transferring silently.
    Missing gradients (``None``, for unused parameters) contribute zero, exactly
    as in the raw path.
    """
    total = torch.zeros((), dtype=torch.float64, device=device)
    for g, w in zip(grads, views, strict=False):
        if g is None:
            continue
        g64 = g.to(dtype=torch.float64)
        total = total + (w.to(dtype=torch.float64) * g64 * g64).sum()
    return total


def resolve_adamw_hparams(
    hparams: Any,
    *,
    source: str = "optimizer param_groups",
) -> tuple[float, float]:
    """Extract ``(beta2, eps)`` from a saved optimizer's hyper-parameters.

    ``hparams`` is a mapping — in practice
    :attr:`vatis.models.local_bundle.OptimizerStateView.hparams`, i.e. the
    ``param_groups`` of the saved ``optim.pt``. Those are authoritative: they
    are what the optimizer actually ran with, whereas ``metadata.json`` records
    what the launcher *intended*.

    ``amsgrad=True`` is rejected rather than approximated: AMSGrad preconditions
    with ``max_exp_avg_sq``, not ``exp_avg_sq``, so the documented
    ``v_hat = exp_avg_sq / (1 - beta2**t)`` convention would be the wrong P.
    """
    if not hasattr(hparams, "get"):
        raise TypeError(f"{source}: expected a mapping of hyper-parameters, got {hparams!r}")
    betas = hparams.get("betas")
    if betas is None:
        beta2 = hparams.get("beta2")
        if beta2 is None:
            raise ValueError(
                f"{source}: no 'betas' (or 'beta2') recorded; cannot bias-correct "
                "exp_avg_sq for the adamw preconditioner"
            )
    else:
        if isinstance(betas, (str, bytes)) or not isinstance(betas, Sequence) or len(betas) < 2:
            raise ValueError(f"{source}: 'betas' is not a (beta1, beta2) pair: {betas!r}")
        beta2 = betas[1]
    eps = hparams.get("eps")
    if eps is None:
        raise ValueError(f"{source}: no 'eps' recorded; cannot build the adamw preconditioner")
    if bool(hparams.get("amsgrad", False)):
        raise ValueError(
            f"{source}: amsgrad=True. AMSGrad preconditions with max_exp_avg_sq, so the "
            "documented v_hat = exp_avg_sq / (1 - beta2**t) convention would build the "
            "wrong P. Refusing rather than reporting a plausible wrong number."
        )
    return float(beta2), float(eps)
