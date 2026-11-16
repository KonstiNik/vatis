"""Guards on the toy model fixtures themselves.

Every other test measures observables *of* these models at their seeded
initialisation, so a fixture whose Jacobian is (nearly) empty silently turns the
suite into a test of a degenerate special case. That has happened:
``TinyTransformer`` used to zero every 1-D parameter, the LayerNorm gains
included, which made its logits identically zero and left the final LayerNorm
as the only module with a non-zero Jacobian or loss gradient.

These tests pin the properties the rest of the suite relies on:

- the logits are not trivial,
- every parameter tensor is live in the Jacobian and in the loss gradient,
- ``TinyTransformer`` is causal and its positions actually interact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.fixtures.tiny_transformer import (
    TinyMLP,
    TinyTransformer,
    causal_lm_loss,
    make_tiny_lm_batch,
    make_tiny_mlp_batch,
    mlp_loss,
)

# The small configuration the exact-Jacobian tests build (tests/unit/test_chi_net.py).
_SMALL_LM = dict(vocab_size=16, seq_len=8, d_model=16, n_layers=1, n_heads=2, d_ff=32)

# A tensor counts as live when its share of the gradient norm is above this.
# The smallest share measured on these fixtures is ~3e-3, the defect this file
# guards against is an exact zero.
_LIVE_REL_FLOOR = 1e-6
# Logits whose largest entry is below this are indistinguishable from the
# all-zero output of the degenerate fixture (measured: >= 0.17 on all three).
_LOGIT_FLOOR = 1e-3

Case = tuple[nn.Module, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


def _default_lm() -> Case:
    torch.manual_seed(0)
    model = TinyTransformer().eval()
    batch = make_tiny_lm_batch(batch_size=4, seed=0)
    return model, model(batch["input_ids"]), lambda logits: causal_lm_loss(logits, batch)


def _small_lm() -> Case:
    torch.manual_seed(42)
    model = TinyTransformer(**_SMALL_LM).eval()
    batch = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=0)
    return model, model(batch["input_ids"]), lambda logits: causal_lm_loss(logits, batch)


def _mlp() -> Case:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    batch = make_tiny_mlp_batch(batch_size=8, seed=0)
    return model, model(batch[0]), lambda logits: mlp_loss(logits, batch)


_CASES: dict[str, Callable[[], Case]] = {
    "tiny-transformer-default": _default_lm,
    "tiny-transformer-small": _small_lm,
    "tiny-mlp": _mlp,
}


def _dead_tensors(model: nn.Module, scalar: torch.Tensor) -> list[str]:
    """Names of the parameter tensors whose gradient of ``scalar`` is (relatively) zero."""
    names = [name for name, _ in model.named_parameters()]
    grads = torch.autograd.grad(scalar, list(model.parameters()), allow_unused=True)
    norms = [0.0 if g is None else float(g.norm()) for g in grads]
    floor = _LIVE_REL_FLOOR * max(norms)
    return [name for name, norm in zip(names, norms, strict=True) if not norm > floor]


@pytest.mark.parametrize("case", list(_CASES))
def test_fixture_logits_are_not_trivial(case: str) -> None:
    _model, logits, _loss = _CASES[case]()
    assert float(logits.detach().abs().max()) > _LOGIT_FLOOR


@pytest.mark.parametrize("case", list(_CASES))
def test_every_parameter_tensor_is_live_in_the_jacobian(case: str) -> None:
    """``J^T v`` for a seeded Rademacher ``v`` reaches every parameter tensor."""
    model, logits, _loss = _CASES[case]()
    generator = torch.Generator().manual_seed(1234)
    probe = torch.randint(0, 2, logits.shape, generator=generator).to(logits.dtype) * 2.0 - 1.0
    dead = _dead_tensors(model, (logits * probe).sum())
    assert not dead, f"parameter tensors with an empty Jacobian: {dead}"


@pytest.mark.parametrize("case", list(_CASES))
def test_every_parameter_tensor_has_a_loss_gradient(case: str) -> None:
    model, logits, loss_fn = _CASES[case]()
    dead = _dead_tensors(model, loss_fn(logits))
    assert not dead, f"parameter tensors with a zero loss gradient: {dead}"


@pytest.mark.parametrize("config", [{}, _SMALL_LM], ids=["default", "small"])
def test_tiny_transformer_is_causal_and_positions_interact(config: dict[str, Any]) -> None:
    """Changing a token moves later logits (attention is live) and no earlier one."""
    torch.manual_seed(0)
    model = TinyTransformer(**config).eval()
    ids = make_tiny_lm_batch(
        batch_size=2, seq_len=model.seq_len, vocab_size=model.vocab_size, seed=0
    )["input_ids"]

    def logit_shift_per_position(position: int) -> torch.Tensor:
        changed = ids.clone()
        changed[:, position] = (changed[:, position] + 1) % model.vocab_size
        with torch.no_grad():
            return (model(changed) - model(ids)).abs().amax(dim=(0, 2))

    first = logit_shift_per_position(0)
    assert float(first[1:].min()) > 1e-4, "a token does not reach every later position"
    last = logit_shift_per_position(model.seq_len - 1)
    assert float(last[:-1].max()) <= 1e-6, "a token leaks into earlier positions"
