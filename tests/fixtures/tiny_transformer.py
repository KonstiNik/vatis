"""A tiny ~1M-param transformer fixture for unit and cross-validation tests.

Sized so that ``S * V <= 1024`` — the convergence test for chi_net needs to
iterate over the full output dimension to compute the exact trace, so we keep
``S * V`` small.

Defaults: ``S = 16``, ``V = 64``, hidden dim 64, 2 layers, 4 heads. ~80k
parameters with embeddings, easily backwarded by hand even on CPU.

Provides:
    - :class:`TinyTransformer` — bare nn.Module returning logits ``(B, S, V)``.
    - :class:`TinyMLP` — even smaller MLP classifier ``(B, V)``, used for the
      cross-validation tests against perspic (perspic prefers per-sample
      classification setups).
    - :func:`make_tiny_lm_batch` and :func:`make_tiny_mlp_batch` —
      reproducible batches for testing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformer(nn.Module):
    """A minimal causal-LM transformer.

    Layout:
        token_embed -> pos_embed -> N x (LayerNorm -> SelfAttn -> LayerNorm -> MLP) -> LayerNorm -> lm_head

    Untied embeddings (so we don't trip the opacus pathway later).
    """

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        seq_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        # Untied head — different parameter from token_embed.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Reproducible-but-non-trivial init: small Gaussian + zero biases.
        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        positions = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.final_ln(h)
        return self.lm_head(h)  # (B, S, V)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[1]
        causal_mask = torch.full((s, s), float("-inf"), device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class TinyMLP(nn.Module):
    """A tiny per-sample classifier ``(B, in_dim) -> (B, n_classes)``.

    Used for cross-validation against perspic — perspic operates on
    per-sample batches and a small MLP is the cleanest way to match its
    normalization conventions exactly.
    """

    def __init__(self, in_dim: int = 8, hidden: int = 16, n_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_classes),
        )
        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, mean=0.0, std=0.1)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_tiny_lm_batch(
    *,
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 64,
    pad_fraction: float = 0.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a reproducible (input_ids, attention_mask, labels) dict."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    attention_mask = torch.ones_like(input_ids)
    if pad_fraction > 0:
        # Pad some trailing positions in some sequences.
        n_pad = max(1, int(round(seq_len * pad_fraction)))
        for b in range(batch_size):
            attention_mask[b, -n_pad:] = 0
    # HF causal-LM convention: labels = next-token; -100 at positions that
    # should be ignored. We mark padded tokens AND the very last position.
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, :-1] = input_ids[:, 1:]
    labels[attention_mask == 0] = -100
    # Also mask label position whose target token is padded.
    shifted_mask = torch.ones_like(input_ids)
    shifted_mask[:, :-1] = attention_mask[:, 1:]
    shifted_mask[:, -1] = 0
    labels[shifted_mask == 0] = -100
    return {
        "input_ids": input_ids.to(device=device),
        "attention_mask": attention_mask.to(device=device),
        "labels": labels.to(device=device),
    }


def make_tiny_mlp_batch(
    *,
    batch_size: int = 8,
    in_dim: int = 8,
    n_classes: int = 4,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(batch_size, in_dim, generator=g)
    y = torch.randint(0, n_classes, (batch_size,), generator=g)
    return x.to(device=device), y.to(device=device)


def causal_lm_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean-reduced cross entropy over valid label positions, HF convention.

    Args:
        logits: ``(B, S, V)``.
        batch: dict containing ``"labels"`` of shape ``(B, S)`` with -100 at
            invalid positions.
    """
    labels = batch["labels"]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


def mlp_loss(logits: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Mean-reduced cross entropy for ``(x, y)`` tuple batches."""
    _x, y = batch
    return F.cross_entropy(logits, y, reduction="mean")


def causal_lm_valid_mask(
    batch: dict[str, torch.Tensor],
    logits: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    """Boolean mask of shape ``(B, S)`` matching the labels' valid positions."""
    return batch["labels"] != -100


def mlp_valid_mask(
    batch: tuple[torch.Tensor, torch.Tensor],
    logits: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    """All positions valid for an MLP classifier — return all-ones mask."""
    _x, y = batch
    return torch.ones_like(y, dtype=torch.bool)


# Sanity check on the parameter count, run at import time.
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_sizes_within_budget() -> None:
    """Test helper that asserts the toy fixtures stay tiny."""
    tx = TinyTransformer()
    n = _count_params(tx)
    assert n < 200_000, f"TinyTransformer parameter count {n} > 200k"
    s = tx.seq_len * tx.vocab_size
    assert s <= 1024, f"S*V = {s} exceeds 1024 (would make exact ground truth slow)"

    mlp = TinyMLP()
    n = _count_params(mlp)
    assert n < 10_000, f"TinyMLP parameter count {n} > 10k"


_BUDGETS_OK = False
try:
    assert_sizes_within_budget()
    _BUDGETS_OK = True
except AssertionError as exc:  # pragma: no cover - debugging aid
    raise RuntimeError(f"toy fixture size budget violated: {exc}") from exc


# Useful as ``import math`` is unused otherwise; keep the import explicit.
_ = math
