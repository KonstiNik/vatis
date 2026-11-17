"""The golden gate's own toy model must be a real model at initialisation.

``tests/utils/bundle_fixture.py::GoldenTinyTransformer`` used to zero every 1-D
parameter, LayerNorm gains included. Every block then output zero, the logits
were identically zero (``loss == ln(vocab)`` exactly) and only ``final_ln`` had a
non-zero Jacobian, so the step-0 checkpoint of every bundle the golden gate
writes was degenerate and its training run started with all blocks switched off.
These tests pin the repaired state: non-trivial logits, and every parameter
tensor taking part in the Jacobian and in the loss gradient.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.utils.bundle_fixture import (
    DEFAULT_SEQ_LEN,
    DEFAULT_VOCAB_SIZE,
    GoldenTinyTransformer,
    batch_from_tokens,
    lm_loss,
    token_stream,
)


def _model_and_batch(tie_head: bool) -> tuple[GoldenTinyTransformer, dict[str, torch.Tensor]]:
    model = GoldenTinyTransformer(tie_head=tie_head, seed=0)
    tokens = token_stream(n_rows=4, seq_len=DEFAULT_SEQ_LEN, vocab_size=DEFAULT_VOCAB_SIZE, seed=1)
    return model, batch_from_tokens(tokens)


def _dead(model: torch.nn.Module, scalar: torch.Tensor) -> list[str]:
    named = list(model.named_parameters())
    grads = torch.autograd.grad(scalar, [p for _, p in named], allow_unused=True)
    return [
        name
        for (name, _), g in zip(named, grads, strict=True)
        if g is None or float(g.abs().max()) == 0.0
    ]


@pytest.mark.parametrize("tie_head", [False, True])
def test_the_untrained_golden_model_has_non_trivial_logits(tie_head: bool) -> None:
    model, batch = _model_and_batch(tie_head)
    logits = model(batch["input_ids"])
    assert float(logits.abs().max()) > 1e-3
    # identically-zero logits predict uniformly: the loss would be exactly ln(V)
    assert abs(float(lm_loss(logits, batch)) - math.log(DEFAULT_VOCAB_SIZE)) > 1e-6


@pytest.mark.parametrize("tie_head", [False, True])
def test_every_parameter_tensor_of_the_golden_model_is_live(tie_head: bool) -> None:
    model, batch = _model_and_batch(tie_head)
    logits = model(batch["input_ids"])
    probe = torch.randn(logits.shape, generator=torch.Generator().manual_seed(7))
    assert _dead(model, (logits * probe).sum()) == []
    assert _dead(model, lm_loss(model(batch["input_ids"]), batch)) == []


def test_the_layernorm_gains_are_one_and_the_weight_matrices_are_seeded() -> None:
    model = GoldenTinyTransformer(seed=0)
    gains = [m.weight for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert gains and all(bool((g == 1.0).all()) for g in gains)
    twin = GoldenTinyTransformer(seed=0)
    assert all(
        torch.equal(p, q) for p, q in zip(model.parameters(), twin.parameters(), strict=True)
    )
