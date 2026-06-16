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
    """

    chi_net: torch.Tensor
    n_backwards: int
    n_valid_tokens: int
    extras: dict[str, Any] = field(default_factory=dict)


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
                seeds this once per ``(checkpoint, eval_batch)`` call so that
                ranks share probes in DDP.

        Returns:
            A ChiNetResult.
        """
