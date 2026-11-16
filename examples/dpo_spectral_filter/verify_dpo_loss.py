"""Verify this example's DPO loss matches open-instruct (Olmo-3's DPO recipe).

The R_dir / R_spec diagnostics in :mod:`diagnostics` build the DPO loss by hand
(vatis ships only the closed-form CE path). This script pins that hand-written
loss to the reference implementation Olmo-3 actually trained with —
``allenai/open-instruct`` — so a future edit to the loss can't silently drift.

It re-implements open-instruct's ``dpo_loss`` and ``_get_batch_logps`` *verbatim*
(from ``open_instruct/dpo_utils.py``) rather than importing open-instruct, so it
stays a dependency-free CPU check that runs in CI. Two things are checked, for
both loss variants:

- ``--no-dpo-norm`` (vanilla ``dpo``, summed sequence log-prob), and
- ``--dpo-norm``    (length-normalized ``dpo_norm``, mean per-token log-prob —
  open-instruct's default and Olmo-3's recipe).

1. ``_logp_from_logits`` (our pre-shifted-label convention) == open-instruct's
   ``_get_batch_logps`` (which shifts internally) on the same logits.
2. The full ``-logsigmoid(margin)`` loss == open-instruct's ``dpo_loss``.

Run standalone::

    .venv/bin/python examples/dpo_spectral_filter/verify_dpo_loss.py

or under pytest::

    .venv/bin/python -m pytest examples/dpo_spectral_filter/verify_dpo_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from examples.dpo_spectral_filter.diagnostics import _logp_from_logits  # noqa: E402


# --------------------------------------------------------------------------
# open-instruct, verbatim (allenai/open-instruct: open_instruct/dpo_utils.py)
# --------------------------------------------------------------------------
def oi_get_batch_logps(
    per_token_logps: torch.Tensor, labels: torch.Tensor, average_log_prob: bool = False
) -> torch.Tensor:
    per_token_logps = per_token_logps[:, :-1]
    loss_mask = labels[:, 1:] != -100
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
    return (per_token_logps * loss_mask).sum(-1)


def oi_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    return (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )


def _oi_per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids[i+1] | x_i) at position i (last col padded; dropped by [:, :-1])."""
    lp = F.log_softmax(logits.float(), dim=-1)
    gathered = lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return F.pad(gathered, (0, 1), value=0.0)


# --------------------------------------------------------------------------
# tiny shared fixture
# --------------------------------------------------------------------------
def _build_scenario(seed: int = 0):
    """Random logits + the two label conventions (ours pre-shifted, oi unshifted)."""
    torch.manual_seed(seed)
    B, T, V = 3, 12, 50
    ids = torch.randint(0, V, (B, T))
    prompt_len = torch.tensor([3, 5, 2])  # per-sequence prompt boundary

    # ours: labels[i] = ids[i+1] for completion positions, else -100.
    our_labels = torch.full((B, T), -100)
    for b in range(B):
        our_labels[b, :-1] = ids[b, 1:]
        pl = int(prompt_len[b])
        if pl >= 1:
            our_labels[b, : pl - 1] = -100

    # open-instruct: unshifted labels[i] = ids[i], -100 on prompt tokens.
    oi_labels = ids.clone()
    for b in range(B):
        oi_labels[b, : int(prompt_len[b])] = -100

    logits = torch.randn(B, T, V)
    return logits, ids, our_labels, oi_labels


def _our_logps(logits: torch.Tensor, labels: torch.Tensor, *, average: bool) -> torch.Tensor:
    return torch.stack(
        [
            _logp_from_logits(logits[b : b + 1], labels[b : b + 1], average=average)
            for b in range(logits.shape[0])
        ]
    )


# --------------------------------------------------------------------------
# checks (test_* so pytest collects them)
# --------------------------------------------------------------------------
def test_logp_matches_openinstruct() -> None:
    logits, ids, our_labels, oi_labels = _build_scenario()
    oi_ptl = _oi_per_token_logps(logits, ids)
    for average in (False, True):
        ours = _our_logps(logits, our_labels, average=average)
        oi = oi_get_batch_logps(oi_ptl, oi_labels, average_log_prob=average)
        diff = (ours - oi).abs().max().item()
        assert diff < 1e-5, f"logp mismatch (average={average}): {diff}"


def test_dpo_loss_matches_openinstruct() -> None:
    logits, _ids, our_labels, _oi = _build_scenario()
    ref_logits = logits + 0.3 * torch.randn_like(logits)
    beta = 0.1
    rej_order = [1, 2, 0]  # arbitrary chosen/rejected pairing
    for average in (False, True):
        for b, r in enumerate(rej_order):
            lp_w = _logp_from_logits(logits[b : b + 1], our_labels[b : b + 1], average=average)
            lp_l = _logp_from_logits(logits[r : r + 1], our_labels[r : r + 1], average=average)
            lp_w_ref = _logp_from_logits(
                ref_logits[b : b + 1], our_labels[b : b + 1], average=average
            )
            lp_l_ref = _logp_from_logits(
                ref_logits[r : r + 1], our_labels[r : r + 1], average=average
            )
            margin = beta * ((lp_w - lp_w_ref) - (lp_l - lp_l_ref))
            ours = -F.logsigmoid(margin)
            oi = oi_dpo_loss(lp_w, lp_l, lp_w_ref, lp_l_ref, beta)
            d = (ours - oi).abs().item()
            assert d < 1e-6, f"loss mismatch (average={average}, pair {b}): {d}"


def main() -> None:
    logits, ids, our_labels, oi_labels = _build_scenario()
    oi_ptl = _oi_per_token_logps(logits, ids)
    for average in (False, True):
        ours = _our_logps(logits, our_labels, average=average)
        oi = oi_get_batch_logps(oi_ptl, oi_labels, average_log_prob=average)
        tag = "MEAN (dpo_norm)" if average else "SUM (dpo)"
        print(f"logp {tag:18s}: max|ours-oi| = {(ours - oi).abs().max().item():.2e}")
    test_logp_matches_openinstruct()
    test_dpo_loss_matches_openinstruct()
    print("\nALL CHECKS PASSED — DPO path == open-instruct (dpo and dpo_norm).")


if __name__ == "__main__":
    main()
