"""Unit tests for vatis.core.observables.

Covers:
    - closed-form chi_loss == autograd chi_loss for cross entropy
    - delta_loss(A, A) symmetry: self path == cross path with two backwards
    - chi_pos combinator (incl. degenerate denom guard)
"""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.tiny_transformer import (
    TinyMLP,
    TinyTransformer,
    causal_lm_loss,
    make_tiny_lm_batch,
    make_tiny_mlp_batch,
    mlp_loss,
)
from vatis.core.observables import (
    chi_loss_cross_entropy,
    chi_loss_from_autograd,
    chi_pos,
    delta_loss_cross,
    delta_loss_self,
)


def _build_lm() -> tuple[TinyTransformer, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = TinyTransformer().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    batch = make_tiny_lm_batch(batch_size=4, seed=0)
    return model, batch


def _build_lm_with_padding() -> tuple[TinyTransformer, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = TinyTransformer().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    batch = make_tiny_lm_batch(batch_size=4, seed=0, pad_fraction=0.25)
    return model, batch


def _build_mlp() -> tuple[TinyMLP, tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    batch = make_tiny_mlp_batch(batch_size=8, seed=0)
    return model, batch


def test_chi_loss_closed_form_matches_autograd_lm() -> None:
    model, batch = _build_lm()
    logits = model(batch["input_ids"])

    closed = chi_loss_cross_entropy(logits.detach(), batch["labels"])
    loss = causal_lm_loss(logits, batch)
    auto = chi_loss_from_autograd(loss, logits, valid_mask=(batch["labels"] != -100))

    assert torch.allclose(closed, auto, atol=1e-7, rtol=1e-5)


def test_chi_loss_closed_form_matches_autograd_with_padding() -> None:
    model, batch = _build_lm_with_padding()
    logits = model(batch["input_ids"])

    closed = chi_loss_cross_entropy(logits.detach(), batch["labels"])
    loss = causal_lm_loss(logits, batch)
    auto = chi_loss_from_autograd(loss, logits, valid_mask=(batch["labels"] != -100))

    assert torch.allclose(closed, auto, atol=1e-7, rtol=1e-5)


def test_chi_loss_closed_form_matches_autograd_mlp() -> None:
    model, batch = _build_mlp()
    x, _y = batch
    logits = model(x)

    targets = batch[1]
    closed = chi_loss_cross_entropy(logits.detach(), targets)
    loss = mlp_loss(logits, batch)
    auto = chi_loss_from_autograd(loss, logits, valid_mask=None)

    assert torch.allclose(closed, auto, atol=1e-7, rtol=1e-5)


def test_chi_loss_returns_zero_when_no_valid_tokens() -> None:
    logits = torch.randn(2, 4, 8, requires_grad=False)
    targets = torch.full((2, 4), -100, dtype=torch.long)
    out = chi_loss_cross_entropy(logits, targets)
    assert float(out) == 0.0


def test_chi_loss_shape_mismatch_raises() -> None:
    logits = torch.randn(2, 4, 8)
    targets = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="shape mismatch"):
        chi_loss_cross_entropy(logits, targets)


def test_delta_loss_self_matches_cross_self_pair() -> None:
    """The self pair via delta_loss_cross should match delta_loss_self.

    The two computation paths use different code (one backward + .grad squared
    sum vs two backwards + dot product). They should agree numerically.
    """
    model, batch = _build_lm()
    params = list(model.parameters())

    loss_for_self = causal_lm_loss(model(batch["input_ids"]), batch)
    dl_self = float(delta_loss_self(loss_for_self, params))

    model.zero_grad()
    loss_a = causal_lm_loss(model(batch["input_ids"]), batch)
    loss_b = causal_lm_loss(model(batch["input_ids"]), batch)
    dl_cross = float(delta_loss_cross(loss_a, loss_b, params))

    assert torch.allclose(
        torch.tensor(dl_self),
        torch.tensor(dl_cross),
        atol=1e-7,
        rtol=1e-5,
    )


def test_delta_loss_cross_is_symmetric() -> None:
    model, batch = _build_lm()
    params = list(model.parameters())
    other = make_tiny_lm_batch(batch_size=4, seed=1)

    model.zero_grad()
    la = causal_lm_loss(model(batch["input_ids"]), batch)
    lb = causal_lm_loss(model(other["input_ids"]), other)
    dl_ab = float(delta_loss_cross(la, lb, params))

    model.zero_grad()
    la2 = causal_lm_loss(model(batch["input_ids"]), batch)
    lb2 = causal_lm_loss(model(other["input_ids"]), other)
    dl_ba = float(delta_loss_cross(lb2, la2, params))

    assert torch.allclose(torch.tensor(dl_ab), torch.tensor(dl_ba), atol=1e-7, rtol=1e-5)


def test_chi_pos_basic_combinator() -> None:
    out = chi_pos(0.4, 2.0, 0.5)  # 0.4 / (2.0 * 0.5) = 0.4
    assert abs(float(out) - 0.4) < 1e-7


def test_chi_pos_handles_zero_denom_without_nan() -> None:
    out = chi_pos(0.0, 0.0, 0.0)
    assert float(out) == 0.0
    assert not torch.isnan(out)
