"""Per-sequence-control-variate chi_net estimator.

Math
====

We want to estimate ``Tr(M_full) = sum_b Tr(M_b) = chi_net``, where
``M_full = JJ^T`` is the eNTK over all valid output positions in the
(micro-) batch and ``M_b`` is its per-sample diagonal block.

Standard Hutchinson with a Rademacher probe ``v`` of the same shape as the
model output gives ``E[||J^T v||^2] = Tr(M_full)``.

This estimator improves variance by exploiting that we can compute, for free
along with one extra backward per sample, the per-sample loss-direction
contribution exactly:

    Per-sample backward of ``L_total`` restricted to sample b yields
        g_b = J_b^T u_b   where u_b is the b-th block of grad_f L_total

    Then ||g_b||^2 / ||u_b||^2 = u_hat_b^T M_b u_hat_b is exact.

Define the block-diagonal projection P = block_diag(u_hat_b u_hat_b^T)_b.
Then:

    Tr((I - P) M_full) = Tr(M_full) - sum_b u_hat_b^T M_b u_hat_b
                       = chi_net - sum_b ||g_b||^2 / ||u_b||^2

and Hutchinson with the projected probe gives an unbiased estimate:

    E[||J^T (I - P) v||^2] = Tr((I - P) M_full)

So the control-variate estimator is:

    chi_net_cv =  (1/n_h) sum_i ||J^T v_i - sum_b alpha_{i,b} g_b/||u_b||||^2
                + sum_b ||g_b||^2 / ||u_b||^2

where ``alpha_{i,b} = <v_i[b], u_hat_b>`` is computed from the probe and the
known per-sample loss directions, both at zero extra backward cost.

Cost
====

Per (ckpt, batch): ``B + n_h`` model backwards (B for the per-sample passes,
n_h for the Hutchinson passes). Compare to plain Hutchinson with effective
``~ B * n_h`` to reach the same per-sample variance — this is why per-seq-CV
wins for small B.

Bonus observable
================

Because we already store the parameter-space gradients ``g_b``, we get the
cross-sample alignment matrix ``C_{bb'} = <g_b, g_{b'}>`` essentially for
free (one matmul over the per-sample grad matrix). It is exactly the eNTK
projected onto the per-sample loss directions, sample-resolved. We surface
it via ``ChiNetResult.extras["alignment_matrix"]``.

Memory caveat
=============

This estimator stores ``B`` per-sample parameter-space gradient vectors at
once. For an 8B model with B=32 that is ~32 * 16 GB = 512 GB — not feasible.
The auto-selection rule in the analyzer only picks per_seq_cv for B <= 32,
but the user is still responsible for noticing that for very large models
the memory cost becomes prohibitive. For 1B-class models and below this is
totally fine.
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
from vatis.core.probes import ProbeDistribution, make_probe, mask_probe
from vatis.data.collate import iter_micro_batches


class PerSequenceControlVariateEstimator(ChiNetEstimator):
    """Per-sequence backward + Hutchinson control variate.

    The default for small batches (``B <= 32``).

    Args:
        n_hutchinson: number of Hutchinson probes per micro-batch (after the
            B per-sample backwards).
        distribution: ``"rademacher"`` (default) or ``"gaussian"``.
        compute_alignment_matrix: if True (default), surface the cross-sample
            alignment matrix in ``result.extras["alignment_matrix"]``.
    """

    name = "per_sequence_cv"

    def __init__(
        self,
        *,
        n_hutchinson: int = 32,
        distribution: ProbeDistribution = "rademacher",
        compute_alignment_matrix: bool = True,
    ) -> None:
        if n_hutchinson < 1:
            raise ValueError(f"n_hutchinson must be >= 1, got {n_hutchinson}")
        self.n_hutchinson = n_hutchinson
        self.distribution: ProbeDistribution = distribution
        self.compute_alignment_matrix = compute_alignment_matrix

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
        device = next(model.parameters()).device
        chi_net_acc = torch.zeros((), dtype=torch.float64, device=device)
        n_valid_total = 0
        n_backwards = 0

        # Per-sample grads collected across micro-batches, indexed by global
        # sample id (in batch order). Each entry is a flat fp32 vector on
        # ``device``. Used for the alignment matrix bonus.
        per_sample_grads: list[torch.Tensor] = []

        for _start, _stop, micro in iter_micro_batches(batch, micro_batch_size):
            m = _micro_batch_size(micro)

            logits = forward_fn(model, micro)
            vmask = valid_mask_fn(micro, logits)

            # Step 1: get u_full = grad_f L_total via one autograd through
            # the loss head. retain_graph so subsequent backwards still work.
            loss_total = loss_fn(logits, micro)
            (u_full_with_grad,) = torch.autograd.grad(
                loss_total,
                logits,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            u_full = u_full_with_grad.detach()

            # Step 2: B per-sample backwards. For sample b, build a
            # grad_outputs tensor that is u_full only at sample b's slice.
            g_b_flat_list: list[torch.Tensor] = []
            u_b_norm_sq_list: list[float] = []
            exact_loss_direction = torch.zeros((), dtype=torch.float64, device=device)

            for b in range(m):
                # Construct grad_outputs that is non-zero only at sample b.
                grad_outputs = torch.zeros_like(u_full)
                grad_outputs[b] = u_full[b]

                grads_b = torch.autograd.grad(
                    outputs=logits,
                    inputs=params,
                    grad_outputs=grad_outputs,
                    retain_graph=True,  # need it alive for next b and the Hutchinson loop
                    create_graph=False,
                    allow_unused=True,
                )
                n_backwards += 1
                g_b_flat = _flatten_grads(grads_b, params, device=device)
                g_b_flat_list.append(g_b_flat)

                u_b_sq = float((u_full[b].to(dtype=torch.float64) ** 2).sum().item())
                u_b_norm_sq_list.append(u_b_sq)

                if u_b_sq > 0.0:
                    g_b_sq = (g_b_flat.to(dtype=torch.float64) ** 2).sum()
                    exact_loss_direction = exact_loss_direction + g_b_sq / u_b_sq

            # Step 3: Hutchinson loop with control variate.
            hutch_acc = torch.zeros((), dtype=torch.float64, device=device)
            for h in range(self.n_hutchinson):
                v = make_probe(
                    tuple(logits.shape),
                    distribution=self.distribution,
                    device=logits.device,
                    dtype=torch.float32,
                    generator=generator,
                )
                v = mask_probe(v, vmask)

                projected = (logits * v.to(dtype=logits.dtype)).sum()
                # Use retain_graph until the very last backward of this
                # micro-batch, so we can free the forward graph then.
                last_backward = h == self.n_hutchinson - 1
                grads_v = torch.autograd.grad(
                    projected,
                    params,
                    retain_graph=not last_backward,
                    create_graph=False,
                    allow_unused=True,
                )
                n_backwards += 1
                grad_v_flat = _flatten_grads(grads_v, params, device=device)

                # Build the per-sample correction in parameter space:
                # correction = sum_b alpha_b * g_b / ||u_b||
                # where alpha_b = <v[b], u_hat_b> = <v[b], u_b> / ||u_b||
                # so alpha_b / ||u_b|| = <v[b], u_b> / ||u_b||^2.
                correction = torch.zeros_like(grad_v_flat)
                for b in range(m):
                    u_b_sq = u_b_norm_sq_list[b]
                    if u_b_sq <= 0.0:
                        continue
                    inner = (v[b].to(dtype=torch.float64) * u_full[b].to(dtype=torch.float64)).sum()
                    coef = float(inner.item()) / u_b_sq
                    correction = correction + coef * g_b_flat_list[b]

                grad_v_perp = grad_v_flat - correction
                hutch_acc = hutch_acc + (grad_v_perp.to(dtype=torch.float64) ** 2).sum()

            chi_net_micro = (hutch_acc / float(self.n_hutchinson)) + exact_loss_direction
            chi_net_acc = chi_net_acc + chi_net_micro

            # Tally valid output positions for normalization.
            if vmask is None:
                n_valid_total += int(torch.tensor(logits.shape[:-1]).prod().item())
            else:
                n_valid_total += int(vmask.sum().item())

            if self.compute_alignment_matrix:
                # Save per-sample grads for the global alignment matrix; live
                # on device for now (caller may move to CPU).
                per_sample_grads.extend(g_b_flat_list)

            # Free the forward graph and large intermediates.
            del logits, loss_total, u_full, u_full_with_grad
            del g_b_flat_list

        chi_net = chi_net_acc.to(dtype=torch.float32)

        extras: dict[str, Any] = {
            "n_hutchinson": self.n_hutchinson,
            "distribution": self.distribution,
        }
        if self.compute_alignment_matrix and per_sample_grads:
            grad_matrix = torch.stack(per_sample_grads, dim=0)  # (B_total, P)
            alignment = grad_matrix @ grad_matrix.T  # (B_total, B_total)
            extras["alignment_matrix"] = alignment.detach().to(dtype=torch.float32, device="cpu")

        return ChiNetResult(
            chi_net=chi_net,
            n_backwards=n_backwards,
            n_valid_tokens=n_valid_total,
            extras=extras,
        )


def _micro_batch_size(micro: Any) -> int:
    """Return the leading-dim size of a (potentially dict/sequence) micro-batch."""
    from vatis.data.collate import batch_size

    return batch_size(micro)


def _flatten_grads(
    grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
    params: list[torch.nn.Parameter],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Flatten an autograd.grad output into a single fp32 vector on ``device``.

    Missing grads (``None``, e.g. for unused params) are replaced with zeros
    of the matching size.
    """
    flats: list[torch.Tensor] = []
    for p, g in zip(params, grads, strict=False):
        if g is None:
            flats.append(torch.zeros(p.numel(), dtype=torch.float32, device=device))
        else:
            flats.append(g.detach().to(dtype=torch.float32, device=device).reshape(-1))
    if not flats:
        raise ValueError("no parameters supplied to _flatten_grads")
    return torch.cat(flats)
