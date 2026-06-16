"""Stub for the opacus-based chi_net estimator. Not yet implemented.

The opacus path uses opacus's ``GradSampleModule`` + ghost clipping to compute
per-sample gradient norms in a single backward, packing ``M`` samples per
forward. With Hutchinson over the ``(S, V)`` axes only (and not the batch
axis), it's the fastest path per backward count — but it has tight model
compatibility requirements (no tied embeddings, no in-place ops, only
opacus-supported layers).

In v1 we ship only the stub. Calling ``compute`` raises NotImplementedError
with a clear pointer to the supported alternatives.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vatis.core.chi_net.base import (
    ChiNetEstimator,
    ChiNetResult,
    ForwardFn,
    LossFn,
    ValidMaskFn,
)


class OpacusEstimator(ChiNetEstimator):
    """Stub for the opacus-based estimator. Not implemented in v1."""

    name = "opacus"

    def __init__(self, **_kwargs: object) -> None:
        # Accept kwargs silently so callers don't blow up at construction time;
        # the error is deferred to compute() where it's actionable.
        pass

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
        raise NotImplementedError(
            "the opacus chi_net path is v1.1 work; "
            "use chi_net_method='hutchinson' or 'per_sequence_cv' instead. "
            "see CLAUDE.md §'chi_net estimation: three methods' for the trade-offs."
        )
