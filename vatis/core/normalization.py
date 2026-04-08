"""Valid-token masking and normalization factors.

All masking happens here. Every accumulator that touches loss-gradient or
network-Jacobian space multiplies by this mask.
"""

from __future__ import annotations

import torch

# HuggingFace's standard ignore index for causal-LM cross-entropy.
DEFAULT_IGNORE_INDEX: int = -100


def valid_token_mask(
    targets: torch.Tensor,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a boolean mask of which target positions are "valid".

    A token is valid if its target is not equal to ``ignore_index`` AND its
    position in ``attention_mask`` (if provided) is non-zero.

    Args:
        targets: integer-typed tensor of target token ids. For an LM with
            HF-style shifted labels this is shape ``(B, S)``; for an MLP-style
            classifier it is shape ``(B,)``. The shape is preserved.
        ignore_index: target value used to mark masked positions.
        attention_mask: optional 0/1 mask, must broadcast to ``targets.shape``.

    Returns:
        Boolean tensor with the same shape as ``targets``. ``True`` means the
        position contributes to the observables.
    """
    mask = targets != ignore_index
    if attention_mask is not None:
        mask = mask & (attention_mask.to(dtype=torch.bool))
    return mask


def count_valid(mask: torch.Tensor) -> int:
    """Return the number of valid (True) entries in a boolean mask."""
    return int(mask.sum().item())


def normalization_factor(n_valid_a: int, n_valid_b: int) -> float:
    """Return ``sqrt(N_A * N_B)`` — the batch-size factor used in the
    normalized LNA components.

    Per the derivation in ``background_info.tex`` §A.2:

        chi_loss_normalized = sqrt(N_A * N_B) * chi_loss
        chi_net_normalized  = chi_net / sqrt(N_A * N_B)

    For self pairs (A = B) this is just ``N``.
    """
    if n_valid_a <= 0 or n_valid_b <= 0:
        raise ValueError(
            f"valid token counts must be positive; got n_valid_a={n_valid_a}, n_valid_b={n_valid_b}"
        )
    # Convert to float first to avoid integer overflow on huge batches.
    return float(float(n_valid_a) ** 0.5 * float(n_valid_b) ** 0.5)
