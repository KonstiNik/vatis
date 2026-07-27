"""Plain Hutchinson chi_net estimator.

For each micro-batch, draw ``n_hutchinson`` Rademacher (or Gaussian) probe
vectors of the same shape as the model output, then accumulate
``||grad_theta (out * v).sum()||^2`` and divide by ``n_hutchinson``. The result
is an unbiased estimate of ``Tr(M_b) = ||J_b||_F^2`` per micro-batch, summed
over the (sub-)batch.

This is the always-works fallback. It is the default for large batches
(``B >= 32``) where the per-sequence path becomes too expensive.

Preconditioned ruler
====================

``Tr(J P J^T) = E_z ||P^{1/2} (J^T z)||^2`` for the same probes ``z``
(``00_context.md`` §"Preconditioned ruler"), so the P estimate needs **no extra
backward**: the accumulator simply reduces the very same gradient tuple with the
elementwise weight ``P`` applied. Raw and P are therefore computed in one pass
and share every probe, and the raw value is bit-identical to a run without a
preconditioner.
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
from vatis.core.preconditioner import Preconditioner, weighted_sum_sq_grads
from vatis.core.probes import ProbeDistribution, make_probe, mask_probe
from vatis.data.collate import iter_micro_batches


class HutchinsonEstimator(ChiNetEstimator):
    """Plain Hutchinson trace estimator. The always-works fallback.

    Args:
        n_hutchinson: number of probe vectors per micro-batch.
        distribution: ``"rademacher"`` (default) or ``"gaussian"``.
    """

    name = "hutchinson"

    def __init__(
        self,
        *,
        n_hutchinson: int = 32,
        distribution: ProbeDistribution = "rademacher",
    ) -> None:
        if n_hutchinson < 1:
            raise ValueError(f"n_hutchinson must be >= 1, got {n_hutchinson}")
        self.n_hutchinson = n_hutchinson
        self.distribution: ProbeDistribution = distribution

    def compute(
        self,
        model: nn.Module,
        batch: Any,
        forward_fn: ForwardFn,
        *,
        loss_fn: LossFn,  # noqa: ARG002 — accepted for ABC compatibility
        valid_mask_fn: ValidMaskFn,
        params: list[torch.nn.Parameter],
        micro_batch_size: int,
        generator: torch.Generator | None = None,
        preconditioner: Preconditioner | None = None,
    ) -> ChiNetResult:
        device = next(model.parameters()).device
        chi_net_acc = torch.zeros((), dtype=torch.float64, device=device)
        # Preconditioned accumulator: only allocated when a P was supplied, so
        # the raw path's arithmetic is untouched when it is not.
        p_views: list[torch.Tensor] | None = None
        chi_net_p_acc: torch.Tensor | None = None
        if preconditioner is not None:
            p_views = preconditioner.to(device=device).parameter_views(params)
            chi_net_p_acc = torch.zeros((), dtype=torch.float64, device=device)
        n_valid_total = 0
        n_backwards = 0

        # We loop over micro-batches; for each, draw n_hutchinson independent
        # probes. Each probe costs one backward through theta.
        for _start, _stop, micro in iter_micro_batches(batch, micro_batch_size):
            # Forward once per probe (cannot reuse the graph because we want
            # independent random projections — and PyTorch builds a fresh
            # graph each forward anyway).
            for _ in range(self.n_hutchinson):
                logits = forward_fn(model, micro)
                vmask = valid_mask_fn(micro, logits)

                probe = make_probe(
                    tuple(logits.shape),
                    distribution=self.distribution,
                    device=logits.device,
                    dtype=torch.float32,
                    generator=generator,
                )
                probe = mask_probe(probe, vmask)
                # Cast probe to forward dtype before multiplying so the
                # backward graph stays in forward dtype.
                projected = (logits * probe.to(dtype=logits.dtype)).sum()

                grads = torch.autograd.grad(
                    projected,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                n_backwards += 1
                grad_sq = torch.zeros((), dtype=torch.float64, device=device)
                for g in grads:
                    if g is None:
                        continue
                    grad_sq = grad_sq + (g.to(dtype=torch.float64) ** 2).sum()
                chi_net_acc = chi_net_acc + grad_sq
                if chi_net_p_acc is not None and p_views is not None:
                    # Same gradient tuple, P-weighted: ||P^{1/2} (J^T v)||^2.
                    chi_net_p_acc = chi_net_p_acc + weighted_sum_sq_grads(
                        grads, p_views, device=device
                    )

                # Tally valid tokens once per probe? No — once per
                # micro-batch. Use the first probe of the micro-batch.
                if _ == 0:
                    if vmask is None:
                        # All output positions count: numel of logits / V.
                        n_valid_total += int(torch.tensor(logits.shape[:-1]).prod().item())
                    else:
                        n_valid_total += int(vmask.sum().item())

        # The Hutchinson estimator is unbiased for E[||J^T v||^2] = Tr(JJ^T)
        # only after dividing by the number of probes per micro-batch.
        chi_net = (chi_net_acc / float(self.n_hutchinson)).to(dtype=torch.float32)
        chi_net_p = (
            None
            if chi_net_p_acc is None
            else (chi_net_p_acc / float(self.n_hutchinson)).to(dtype=torch.float32)
        )

        return ChiNetResult(
            chi_net=chi_net,
            n_backwards=n_backwards,
            n_valid_tokens=n_valid_total,
            extras={
                "n_hutchinson": self.n_hutchinson,
                "distribution": self.distribution,
            },
            chi_net_p=chi_net_p,
        )
