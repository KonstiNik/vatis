"""Guards on the toy model fixtures themselves.

Every other test measures observables *of* these models at their seeded
initialisation, so a part of the model that contributes nothing a tolerance can
resolve is a part of the package the suite is not testing. That has happened
twice with ``TinyTransformer``:

- every 1-D parameter was zero-initialised, LayerNorm gains included: the logits
  were identically zero and only the final LayerNorm had a non-zero Jacobian;
- with N(0, 0.02) weights and gains at one every tensor was "live", but the
  attention queries and keys carried ~4e-7 of the NTK trace and attention was
  uniform to within 1%, far below the 1e-5 relative tolerance the exact-Jacobian
  tests assert — a bug in their gradients could not fail a test.

So "live" is tied here to what the suite can detect — for the quantities it
checks exactly (``delta_loss`` and the exact-Jacobian comparisons, at 1e-5). The
``chi_net`` estimators are only checked statistically, at 2-5%, and the output
head carries about 80% of the trace on these models, so a bug confined to a
low-share group can still pass those tests. The guards run on the very
models the suite builds (``test_observables._build_lm`` is the default
configuration of the analyzer and DDP tests, ``test_chi_net._build_small_lm``
the one of the exact-Jacobian tests):

- the logits are not trivial;
- every parameter group carries at least ``SHARE_FLOOR`` of the squared norm of
  a seeded-probe VJP (an unbiased estimate of its share of the NTK trace) and of
  the loss gradient. A group is one sub-module's parameters, with the attention
  input projection split into its query/key rows and its value rows;
- attention matters: making it uniform moves the logits;
- ``TinyTransformer`` is causal and its positions interact.

Not covered, on purpose: single tensors inside a group. The key bias has exactly
zero gradient in any softmax attention, and on the default configuration the
query bias alone carries ~2e-5 of either quantity.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from tests.fixtures.tiny_transformer import (
    TinyMLP,
    causal_lm_loss,
    make_tiny_mlp_batch,
    mlp_loss,
)
from tests.unit.test_chi_net import _build_small_lm
from tests.unit.test_observables import _build_lm

# Five times the 1e-5 relative tolerance of the exact-reference tests. Smallest
# shares measured on the fixtures: 1.2e-4 (a LayerNorm of the default
# configuration, loss gradient), 5.6e-4 on the small configuration, 0.13 on
# TinyMLP. On the two defective initialisations described above the query/key
# groups are at 0 and at most 3.0e-5.
SHARE_FLOOR = 5e-5
# Relative change of the logits when attention is made uniform. Measured: 0.071
# (default) and 0.078 (small); at most 0.010 with the N(0, 0.02) initialisation.
ATTENTION_FLOOR = 0.02
# Logits below this are indistinguishable from the all-zero output of the
# zero-gain fixture (measured: >= 0.17 on all three fixtures).
LOGIT_FLOOR = 1e-3

LM_CASES: dict[str, Callable[[], tuple[nn.Module, dict[str, torch.Tensor]]]] = {
    "tiny-transformer-default": _build_lm,
    "tiny-transformer-small": _build_small_lm,
}


def _lm(case: str) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    model, batch = LM_CASES[case]()
    logits = model(batch["input_ids"])
    return model, logits, causal_lm_loss(logits, batch)


def _mlp() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    batch = make_tiny_mlp_batch(batch_size=8, seed=0)
    logits = model(batch[0])
    return model, logits, mlp_loss(logits, batch)


CASES: dict[str, Callable[[], tuple[nn.Module, torch.Tensor, torch.Tensor]]] = {
    **{case: (lambda case=case: _lm(case)) for case in LM_CASES},
    "tiny-mlp": _mlp,
}


def _query_key_rows(param: torch.Tensor) -> slice:
    """Rows of ``in_proj_weight`` / ``in_proj_bias`` that build queries and keys."""
    return slice(0, 2 * (param.shape[0] // 3))


def _group_shares(model: nn.Module, scalar: torch.Tensor) -> dict[str, float]:
    """Share of ``||grad_theta scalar||^2`` carried by each parameter group."""
    names = [name for name, _ in model.named_parameters()]
    grads = torch.autograd.grad(scalar, list(model.parameters()), retain_graph=True)
    sq: dict[str, float] = {}
    for name, grad in zip(names, grads, strict=True):
        module, _, leaf = name.rpartition(".")
        if leaf.startswith("in_proj_"):
            rows = _query_key_rows(grad)
            parts = {f"{module}.in_proj[qk]": grad[rows], f"{module}.in_proj[v]": grad[rows.stop :]}
        else:
            parts = {module: grad}
        for group, part in parts.items():
            sq[group] = sq.get(group, 0.0) + float((part.double() ** 2).sum())
    total = sum(sq.values())
    return {group: value / total for group, value in sq.items()}


def _weak_groups(model: nn.Module, scalar: torch.Tensor) -> dict[str, float]:
    return {g: s for g, s in _group_shares(model, scalar).items() if not s >= SHARE_FLOOR}


def _probe_projection(logits: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    probe = torch.randint(0, 2, logits.shape, generator=generator).to(logits.dtype) * 2.0 - 1.0
    return (logits * probe).sum()


def _uniform_attention_shift(model: nn.Module, input_ids: torch.Tensor) -> float:
    """``||f - f_uniform|| / ||f||``, ``f_uniform`` = the logits with all queries and keys zeroed."""
    saved = {name: p.detach().clone() for name, p in model.named_parameters() if "in_proj" in name}
    with torch.no_grad():
        logits = model(input_ids)
        for name, param in model.named_parameters():
            if name in saved:
                param[_query_key_rows(param)] = 0.0
        uniform = model(input_ids)
        for name, param in model.named_parameters():
            if name in saved:
                param.copy_(saved[name])
    return float((logits - uniform).norm() / logits.norm())


@pytest.mark.parametrize("case", list(CASES))
def test_fixture_logits_are_not_trivial(case: str) -> None:
    _model, logits, _loss = CASES[case]()
    assert float(logits.detach().abs().max()) > LOGIT_FLOOR


@pytest.mark.parametrize("case", list(CASES))
def test_every_parameter_group_carries_a_visible_share_of_the_jacobian(case: str) -> None:
    model, logits, _loss = CASES[case]()
    weak = _weak_groups(model, _probe_projection(logits))
    assert not weak, f"groups below {SHARE_FLOOR} of ||J^T v||^2: {weak}"


@pytest.mark.parametrize("case", list(CASES))
def test_every_parameter_group_carries_a_visible_share_of_the_loss_gradient(case: str) -> None:
    model, _logits, loss = CASES[case]()
    weak = _weak_groups(model, loss)
    assert not weak, f"groups below {SHARE_FLOOR} of ||grad L||^2: {weak}"


@pytest.mark.parametrize("case", list(LM_CASES))
def test_attention_is_not_uniform(case: str) -> None:
    model, batch = LM_CASES[case]()
    assert _uniform_attention_shift(model, batch["input_ids"]) > ATTENTION_FLOOR


@pytest.mark.parametrize("case", list(LM_CASES))
def test_tiny_transformer_is_causal_and_positions_interact(case: str) -> None:
    """Changing a token moves later logits and no earlier one."""
    model, batch = LM_CASES[case]()
    ids = batch["input_ids"]

    def logit_shift_per_position(position: int) -> torch.Tensor:
        changed = ids.clone()
        changed[:, position] = (changed[:, position] + 1) % model.vocab_size
        with torch.no_grad():
            return (model(changed) - model(ids)).abs().amax(dim=(0, 2))

    first = logit_shift_per_position(0)
    assert float(first[1:].min()) > 1e-4, "a token does not reach every later position"
    last = logit_shift_per_position(ids.shape[1] - 1)
    assert float(last[:-1].max()) <= 1e-6, "a token leaks into earlier positions"


def test_the_guards_see_a_model_whose_queries_and_keys_are_dead() -> None:
    """Vacuity check: zeroed query/key rows — every tensor still has a gradient,
    attention is exactly uniform — must trip the share and the attention guard."""
    model, batch = _build_small_lm()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("in_proj_weight"):
                param[_query_key_rows(param)] = 0.0
    logits = model(batch["input_ids"])
    assert set(_weak_groups(model, _probe_projection(logits))) == {"layers.0.attn.in_proj[qk]"}
    assert set(_weak_groups(model, causal_lm_loss(logits, batch))) == {"layers.0.attn.in_proj[qk]"}
    assert _uniform_attention_shift(model, batch["input_ids"]) == 0.0
