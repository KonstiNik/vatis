"""ChiNetEstimator ABC and result dataclass.

All estimators take the same input — a model, a (sub-)batch, a forward
function, and a loss function — and return ``ChiNetResult`` containing at
least an unnormalized ``chi_net`` scalar plus optional bonus observables
(e.g. the per-sample alignment matrix from PerSequenceControlVariateEstimator).

The math core knows nothing about HF or DDP; the caller is responsible for
making the model.forward(batch) interface work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from vatis.core.preconditioner import Preconditioner

# Forward function type:
#   forward_fn(model, batch) -> logits tensor
ForwardFn = Callable[[nn.Module, Any], torch.Tensor]

# Loss function type:
#   loss_fn(logits, batch) -> 0-dim scalar tensor
# The estimator uses this only when it needs ``u = grad_f L`` (per_seq_cv).
LossFn = Callable[[torch.Tensor, Any], torch.Tensor]

# Valid mask function type:
#   valid_mask_fn(batch, logits) -> Optional[boolean tensor matching logits[..., 0]]
ValidMaskFn = Callable[[Any, torch.Tensor], torch.Tensor | None]


@dataclass
class ChiNetResult:
    """Output of one ``estimator.compute(...)`` call.

    Attributes:
        chi_net: 0-dim fp32 tensor holding the unnormalized estimate of
            ``Tr(Theta) = sum_b ||grad_theta f(x_b)||^2``. Sum is over all
            valid output positions across the (micro-) batches.
        n_backwards: total number of model backward passes used (for cost
            accounting and tests).
        n_valid_tokens: number of valid output positions covered. The caller
            uses this to apply the ``sqrt(N_A * N_B)`` normalization.
        extras: estimator-specific bonus observables. Currently used by
            PerSequenceControlVariateEstimator to ship the cross-sample
            alignment matrix.
        chi_net_p: the preconditioned counterpart ``Tr(Theta_P) =
            ||P^{1/2} J||_F^2``, or ``None`` when ``compute`` was called without
            a preconditioner. Computed in the **same pass** as ``chi_net`` (the
            two share every backward and differ only in the elementwise weight
            applied to the resulting VJP), so requesting it costs no extra
            gradient evaluation.
        chi_net_p_capped: the part of ``chi_net_p`` contributed by the entries
            of ``P`` that sit at its ceiling (``Preconditioner.ceiling``, i.e.
            ``1/eps`` for AdamW), or ``None`` when the preconditioner has no
            ceiling. The estimate is linear in ``P`` at fixed probes, so this
            and the remaining ("live") part add up to ``chi_net_p``.
        chi_net_var: 0-dim fp64 tensor, the estimated **variance of**
            ``chi_net`` **under the probe draw**, or ``None`` when it cannot be
            estimated (one probe per micro-batch, or an estimator that does not
            report it). ``chi_net`` is a sum over micro-batches of a mean of
            ``n_hutchinson`` independent per-probe values, so this is the sum
            over micro-batches of ``s^2 / n_hutchinson``, ``s^2`` the unbiased
            sample variance of that micro-batch's per-probe values
            (:class:`ProbeSpread`). It is unbiased for the variance of the
            returned number at the micro-batch cut it was computed with. The
            batch is held fixed: this says nothing about data sampling.
        chi_net_p_var: the same for ``chi_net_p`` (``None`` without a
            preconditioner, or when ``chi_net_var`` is ``None``).
        chi_net_p_capped_var: the same for ``chi_net_p_capped``.
    """

    chi_net: torch.Tensor
    n_backwards: int
    n_valid_tokens: int
    extras: dict[str, Any] = field(default_factory=dict)
    chi_net_p: torch.Tensor | None = None
    chi_net_p_capped: torch.Tensor | None = None
    chi_net_var: torch.Tensor | None = None
    chi_net_p_var: torch.Tensor | None = None
    chi_net_p_capped_var: torch.Tensor | None = None


class ProbeSpread:
    """Running spread of the per-probe values of **one** micro-batch.

    Both estimators report, per micro-batch, the mean of ``n`` independent and
    identically distributed per-probe values ``q_1 .. q_n`` (plus, for
    ``per_sequence_cv``, a term that does not depend on the probes). The
    variance of that mean is estimated without bias by ``s^2 / n`` with
    ``s^2 = sum_i (q_i - mean)^2 / (n - 1)``; :meth:`variance_of_mean` returns
    it, or ``None`` for ``n < 2``, where the spread of one value is undefined.

    The sums are kept relative to the first value (``d_i = q_i - q_1``), which
    removes the cancellation ``sum q^2 - (sum q)^2 / n`` suffers when the
    spread is small against the mean. Everything stays a 0-dim fp64 tensor on
    the device, so adding a value costs no host synchronisation.
    """

    def __init__(self, device: torch.device) -> None:
        self._n = 0
        self._shift: torch.Tensor | None = None
        self._sum = torch.zeros((), dtype=torch.float64, device=device)
        self._sum_sq = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, value: torch.Tensor) -> None:
        """Record one per-probe value (a 0-dim tensor)."""
        v = value.detach().to(dtype=torch.float64)
        if self._shift is None:
            self._shift = v
        d = v - self._shift
        self._sum = self._sum + d
        self._sum_sq = self._sum_sq + d * d
        self._n += 1

    def variance_of_mean(self) -> torch.Tensor | None:
        """``s^2 / n`` for the values recorded so far; ``None`` below two."""
        n = self._n
        if n < 2:
            return None
        sample_var = (self._sum_sq - self._sum * self._sum / n) / (n - 1)
        # Round-off can leave a tiny negative number where every value is equal.
        return torch.clamp(sample_var, min=0.0) / n

    def add_variance_of_mean_to(self, total: torch.Tensor) -> torch.Tensor:
        """``total`` plus :meth:`variance_of_mean`; ``total`` itself below two values."""
        micro_var = self.variance_of_mean()
        return total if micro_var is None else total + micro_var


class ChiNetEstimator(ABC):
    """Abstract base for chi_net estimators.

    Subclasses must implement :meth:`compute` and :attr:`name`.
    """

    name: str = "abstract"

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        batch: Any,
        forward_fn: ForwardFn,
        *,
        loss_fn: LossFn,
        valid_mask_fn: ValidMaskFn,
        params: list[torch.nn.Parameter],
        micro_batch_size: int,
        generator: torch.Generator | None = None,
        preconditioner: Preconditioner | None = None,
    ) -> ChiNetResult:
        """Estimate chi_net for the given (sub-)batch.

        Args:
            model: the network. Should be in eval mode and have ``params``
                with ``requires_grad=True``.
            batch: opaque batch object that ``forward_fn(model, batch)``
                accepts. The estimator may slice along the batch dim using
                ``vatis.data.collate.iter_micro_batches``.
            forward_fn: callable returning the model output logits for a batch.
            loss_fn: callable returning the scalar batch loss given
                ``(logits, batch)``. Used by control-variate estimators to
                derive ``u = grad_f L``.
            valid_mask_fn: ``valid_mask_fn(batch, logits) -> mask`` returning a
                boolean tensor that broadcasts over ``logits[..., 0]`` and
                identifies valid output positions. May return ``None`` for
                "all positions valid".
            params: list of parameters to backprop into. Order matters only
                for callers that flatten grads themselves.
            micro_batch_size: how many sequences per backward. The estimator
                handles the actual splitting.
            generator: torch.Generator for reproducible probes; the caller
                seeds this once per ``(checkpoint, eval_batch)`` call, with a
                different stream on every DDP rank (ranks probing their own
                shards with the same stream have correlated errors).
            preconditioner: optional diagonal ``P``
                (:class:`~vatis.core.preconditioner.Preconditioner`), aligned
                with ``params``. When given, the estimator additionally fills
                :attr:`ChiNetResult.chi_net_p` with ``Tr(J P J^T)``, estimated
                by the *same* mechanism and the *same* probes as ``chi_net``:
                every place the raw path takes ``||g||^2`` of a VJP or a
                per-sample gradient, the P path takes ``||P^{1/2} g||^2``. Raw
                numerics are untouched — with or without a preconditioner,
                ``chi_net`` is bit-identical for a given seed.

        Returns:
            A ChiNetResult.
        """
