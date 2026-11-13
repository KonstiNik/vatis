"""Unit tests for :mod:`vatis.core.preconditioner` and the AdamW P source.

Covers the ``Preconditioner`` type's invariants (what it refuses, and why each
refusal matters), the elementwise-weighted reductions, and — the headline —
**T-adamw-state**: after 20 real ``torch.optim.AdamW`` steps, the ``P`` vatis
builds must be *elementwise equal* to ``1/(sqrt(exp_avg_sq/(1 - beta2**t)) +
eps)`` recomputed by hand from the optimizer's own internals.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.fixtures.synthetic_bundle import (
    BundleTinyLM,
    bundle_metadata,
    handwritten_optim_state,
    make_batch_a,
    tiny_lm_factory,
    write_checkpoint,
    write_run_files,
)
from vatis.core.param_layout import ParameterLayout
from vatis.core.preconditioner import (
    RAW_DESCRIPTOR,
    Preconditioner,
    epsilon_cap_mask,
    resolve_adamw_hparams,
    weighted_dot_fp64,
    weighted_sum_sq_fp64,
    weighted_sum_sq_grads,
)
from vatis.models.local_bundle import (
    BundleSchemaError,
    OptimizerStateView,
    adamw_preconditioner_from_state,
    load_local_bundle,
)

BETA1, BETA2, EPS, WEIGHT_DECAY, LR = 0.9, 0.95, 1e-8, 0.1, 1e-2
N_ADAMW_STEPS = 20


# =========================================================== the type's invariants


def test_diag_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        Preconditioner(diag=torch.ones(2, 3), descriptor="explicit")


def test_diag_must_be_finite() -> None:
    bad = torch.ones(4)
    bad[2] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        Preconditioner(diag=bad, descriptor="explicit")


def test_diag_must_be_non_negative() -> None:
    """A negative entry has no square root, so chi_net^P would silently become a
    difference of norms rather than a trace."""
    bad = torch.ones(4)
    bad[1] = -0.5
    with pytest.raises(ValueError, match="positive semi-definite"):
        Preconditioner(diag=bad, descriptor="explicit")


def test_descriptor_none_is_reserved_for_raw_rows() -> None:
    with pytest.raises(ValueError, match="reserved for raw"):
        Preconditioner(diag=torch.ones(4), descriptor=RAW_DESCRIPTOR)


def test_validate_for_rejects_length_mismatch() -> None:
    """The alignment guard: P and the flat gradient must span the same space."""
    params = [nn.Parameter(torch.zeros(3, 2)), nn.Parameter(torch.zeros(4))]
    Preconditioner.identity(10).validate_for(params)
    with pytest.raises(ValueError, match="must be aligned with the canonical"):
        Preconditioner.identity(9).validate_for(params)


def test_parameter_views_tile_the_diagonal_in_order() -> None:
    params = [nn.Parameter(torch.zeros(2, 2)), nn.Parameter(torch.zeros(3))]
    diag = torch.arange(7, dtype=torch.float32)
    views = Preconditioner.from_diagonal(diag).parameter_views(params)
    assert [tuple(v.shape) for v in views] == [(2, 2), (3,)]
    assert torch.equal(views[0].reshape(-1), diag[:4])
    assert torch.equal(views[1], diag[4:])


def test_identity_is_ones_not_a_short_circuit() -> None:
    """T-identity depends on the identity P driving the real P code path."""
    identity = Preconditioner.identity(5)
    assert torch.equal(identity.diag, torch.ones(5))
    assert identity.descriptor == "identity"


def test_to_is_a_no_op_when_nothing_changes() -> None:
    p = Preconditioner.identity(4)
    assert p.to() is p
    assert p.to(device="cpu", dtype=torch.float32) is p


# ============================================================ weighted reductions


def test_weighted_reductions_match_explicit_sums() -> None:
    g = torch.Generator().manual_seed(0)
    a = torch.randn(1000, generator=g, dtype=torch.float64)
    b = torch.randn(1000, generator=g, dtype=torch.float64)
    w = torch.rand(1000, generator=g, dtype=torch.float64) + 0.1
    assert abs(float(weighted_sum_sq_fp64(a, w)) - float((w * a * a).sum())) < 1e-12
    assert abs(float(weighted_dot_fp64(a, b, w)) - float((w * a * b).sum())) < 1e-12


def test_weighted_reductions_are_chunk_invariant() -> None:
    """Chunking must not change the answer beyond fp64 reassociation."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(5000, generator=g, dtype=torch.float64)
    w = torch.rand(5000, generator=g, dtype=torch.float64) + 0.1
    full = float(weighted_sum_sq_fp64(a, w, chunk=1 << 25))
    chunked = float(weighted_sum_sq_fp64(a, w, chunk=97))
    assert abs(full - chunked) <= 1e-12 * abs(full)


def test_weighted_sum_sq_grads_skips_missing_gradients() -> None:
    params = [nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(3))]
    views = Preconditioner.from_diagonal(torch.arange(5, dtype=torch.float64)).parameter_views(
        params
    )
    grads = [torch.tensor([1.0, 2.0], dtype=torch.float64), None]
    total = float(weighted_sum_sq_grads(grads, views, device=torch.device("cpu")))
    assert total == pytest.approx(0.0 * 1.0 + 1.0 * 4.0)


def test_weighted_reductions_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="flat has"):
        weighted_sum_sq_fp64(torch.ones(3), torch.ones(4))
    with pytest.raises(ValueError, match="mismatched lengths"):
        weighted_dot_fp64(torch.ones(3), torch.ones(3), torch.ones(4))


# ================================================================ the adamw math


def test_adamw_rejects_step_zero() -> None:
    """Bias correction is undefined at step 0 and P = 1/eps is garbage, not neutral."""
    with pytest.raises(ValueError, match="step >= 1"):
        Preconditioner.adamw(torch.ones(4), step=0, beta2=BETA2, eps=EPS)


def test_adamw_rejects_bad_hyperparameters() -> None:
    with pytest.raises(ValueError, match="beta2 must be in"):
        Preconditioner.adamw(torch.ones(4), step=3, beta2=1.0, eps=EPS)
    with pytest.raises(ValueError, match="eps must be"):
        Preconditioner.adamw(torch.ones(4), step=3, beta2=BETA2, eps=-1.0)


def test_adamw_bias_correction_is_applied() -> None:
    """The bias correction must actually be there — a missing 1/(1-beta2^t) is
    the classic silent factor-of-2 bug at small t."""
    v = torch.full((3,), 4.0, dtype=torch.float64)
    step, beta2 = 3, 0.9
    p = Preconditioner.adamw(v, step=step, beta2=beta2, eps=0.0)
    expected = 1.0 / (4.0 / (1.0 - beta2**step)) ** 0.5
    assert torch.allclose(p.diag, torch.full((3,), expected, dtype=torch.float64))
    uncorrected = 1.0 / 2.0
    assert abs(float(p.diag[0]) - uncorrected) > 0.1, "bias correction had no effect"


def test_epsilon_cap_mask_flags_untouched_coordinates() -> None:
    """v = 0 coordinates land at the 1/eps ceiling; the mask must find them."""
    v = torch.tensor([0.0, 1.0, 0.0, 4.0], dtype=torch.float64)
    p = Preconditioner.adamw(v, step=5, beta2=BETA2, eps=EPS)
    mask = epsilon_cap_mask(p.diag, EPS)
    assert mask.tolist() == [True, False, True, False]
    assert float(p.diag[0]) == pytest.approx(1.0 / EPS)


def test_resolve_adamw_hparams_rejects_amsgrad() -> None:
    """AMSGrad preconditions with max_exp_avg_sq, so the frozen v_hat convention
    would build the wrong P — refuse instead of reporting a plausible number."""
    with pytest.raises(ValueError, match="amsgrad"):
        resolve_adamw_hparams({"betas": (0.9, 0.95), "eps": 1e-8, "amsgrad": True})


def test_resolve_adamw_hparams_needs_betas_and_eps() -> None:
    with pytest.raises(ValueError, match="no 'betas'"):
        resolve_adamw_hparams({"eps": 1e-8})
    with pytest.raises(ValueError, match="no 'eps'"):
        resolve_adamw_hparams({"betas": (0.9, 0.95)})


# ============================================================== T-adamw-state


class _AdamWToy(nn.Module):
    """A toy with an embedding (so some rows stay untouched by the gradient), a
    dense layer and an untied head — a few hundred parameters."""

    def __init__(self, vocab: int = 16, dim: int = 6) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.fc(self.embed(ids))))


def _run_real_adamw(n_steps: int = N_ADAMW_STEPS) -> tuple[_AdamWToy, torch.optim.AdamW]:
    torch.manual_seed(0)
    toy = _AdamWToy()
    optimizer = torch.optim.AdamW(
        list(toy.parameters()), lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY
    )
    gen = torch.Generator().manual_seed(3)
    # Only the first 8 vocabulary ids appear, so the remaining embedding rows keep
    # exp_avg_sq == 0 exactly — the real-run condition documented on
    # Preconditioner.adamw.
    ids = torch.randint(0, 8, (4, 5), generator=gen)
    targets = torch.roll(ids, shifts=-1, dims=1)
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = toy(ids)
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return toy, optimizer


def test_adamw_state_matches_hand_computation_elementwise() -> None:
    """T-adamw-state: P from the estimator == 1/(sqrt(exp_avg_sq/(1-beta2^t)) + eps)
    recomputed by hand from ``torch.optim.AdamW``'s own internals.

    Elementwise **equality**, asserted with ``torch.equal`` (bit-for-bit, not a
    tolerance): the hand computation performs the same operations in the same
    order on the same fp32 tensors, so anything other than exact agreement means
    vatis is doing different arithmetic than the documented convention.

    Also pins the two things a wrong-answer bug would most likely get wrong:
    ``t`` (must be the saved ``step``, i.e. 20 after 20 updates) and the
    flattening order (the hand version concatenates by canonical name).
    """
    toy, optimizer = _run_real_adamw()
    layout = ParameterLayout.from_named_parameters(toy)
    state = OptimizerStateView(optimizer.state_dict(), layout, source="<live optimizer>")

    assert state.step_count == N_ADAMW_STEPS, (
        f"pre-step semantics: after {N_ADAMW_STEPS} updates the saved step counter must "
        f"be {N_ADAMW_STEPS}, got {state.step_count}"
    )
    beta2, eps = resolve_adamw_hparams(state.hparams)
    assert (beta2, eps) == (BETA2, EPS)

    built = adamw_preconditioner_from_state(state, context="<live optimizer>")

    named = dict(toy.named_parameters())
    hand_parts: list[torch.Tensor] = []
    for name in layout.names:
        entry = optimizer.state[named[name]]
        v = entry["exp_avg_sq"].detach()
        t = int(entry["step"].item())
        assert t == N_ADAMW_STEPS
        hand_parts.append((1.0 / ((v / (1.0 - BETA2**t)).sqrt() + EPS)).reshape(-1))
    hand = torch.cat(hand_parts)

    assert built.diag.shape == hand.shape
    assert torch.equal(built.diag, hand), (
        "P is not elementwise equal to the hand computation: max |diff| = "
        f"{float((built.diag - hand).abs().max()):.3e}"
    )
    assert built.descriptor == "adamw"

    # The 1/eps ceiling really is populated on a real run (untouched vocab rows).
    capped = epsilon_cap_mask(built.diag, EPS)
    assert bool(capped.any()), (
        "expected some coordinates to keep exp_avg_sq == 0 on this toy; without them "
        "the test would not exercise the documented 1/eps behaviour"
    )
    assert float(built.diag.max()) == pytest.approx(1.0 / EPS, rel=1e-6)


def test_adamw_state_wrong_step_would_change_p() -> None:
    """Guard the T-adamw-state assertion against being vacuous: using t-1 or t+1
    must produce a *different* P, so the equality above really pins ``t``."""
    toy, optimizer = _run_real_adamw()
    layout = ParameterLayout.from_named_parameters(toy)
    state = OptimizerStateView(optimizer.state_dict(), layout, source="<live optimizer>")
    correct = adamw_preconditioner_from_state(state)
    v = state.flat_state("exp_avg_sq")
    for wrong_step in (N_ADAMW_STEPS - 1, N_ADAMW_STEPS + 1):
        other = Preconditioner.adamw(v, step=wrong_step, beta2=BETA2, eps=EPS)
        assert not torch.equal(correct.diag, other.diag), (
            f"P at step={wrong_step} is identical to P at step={N_ADAMW_STEPS}; the "
            "bias correction is not being applied and T-adamw-state is vacuous"
        )


def test_adamw_from_state_rejects_missing_second_moment() -> None:
    """An SGD-style state has no exp_avg_sq: refuse loudly (its ruler is raw)."""
    toy = _AdamWToy()
    layout = ParameterLayout.from_named_parameters(toy)
    sgd_state = {
        "state": {
            i: {"momentum_buffer": torch.zeros(shape)} for i, shape in enumerate(layout.shapes)
        },
        "param_groups": [{"lr": 0.1, "momentum": 0.9, "params": list(range(len(layout)))}],
    }
    state = OptimizerStateView(sgd_state, layout, source="<sgd>")
    with pytest.raises(BundleSchemaError, match="no 'exp_avg_sq'"):
        adamw_preconditioner_from_state(state, context="<sgd>")


def test_adamw_from_state_rejects_empty_state() -> None:
    toy = _AdamWToy()
    layout = ParameterLayout.from_named_parameters(toy)
    state = OptimizerStateView(
        {"state": {}, "param_groups": [{"lr": 0.1, "params": list(range(len(layout)))}]},
        layout,
        source="<empty>",
    )
    with pytest.raises(BundleSchemaError, match="optimizer state is empty"):
        adamw_preconditioner_from_state(state, context="<empty>")


# ================================================ the bundle-level adamw source


def _write_bundle(
    run_dir: Path,
    *,
    steps: tuple[int, ...] = (0, 10),
    metadata_overrides: dict[str, object] | None = None,
) -> tuple[BundleTinyLM, list[str]]:
    """A bundle whose step-0 checkpoint has an EMPTY optimizer state (as a real
    pre-step step-0 checkpoint does) and whose later checkpoints do not."""
    model = BundleTinyLM(seed=0)
    metadata = bundle_metadata(model, betas=(BETA1, BETA2), eps=EPS)
    if metadata_overrides:
        metadata = {**metadata, **metadata_overrides}
    order = write_run_files(run_dir, model, metadata=metadata)
    for step in steps:
        if step == 0:
            optim_state: dict[str, object] = {
                "state": {},
                "param_groups": [
                    {
                        "lr": LR,
                        "betas": (BETA1, BETA2),
                        "eps": EPS,
                        "weight_decay": WEIGHT_DECAY,
                        "amsgrad": False,
                        "params": list(range(len(order))),
                    }
                ],
            }
        else:
            optim_state = handwritten_optim_state(
                model,
                order,
                step=step,
                betas=(BETA1, BETA2),
                eps=EPS,
                exp_avg_sq_constants={name: 0.25 * (i + 1) for i, name in enumerate(order)},
            )
        write_checkpoint(
            run_dir, step, model=model, optim_state=optim_state, batch_a=make_batch_a(seed=step)
        )
    return model, order


def test_source_adamw_preconditioner_builds_from_saved_state(tmp_path: Path) -> None:
    model, order = _write_bundle(tmp_path / "run")
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    factory = source.adamw_preconditioner()
    checkpoint = source.checkpoint_for(10)
    precond = factory(checkpoint.identifier)
    assert precond is not None
    assert precond.descriptor == "adamw"
    assert precond.n_params == source.layout.total_numel

    v = checkpoint.optimizer_state.flat_state("exp_avg_sq")
    expected = 1.0 / ((v / (1.0 - BETA2**10)).sqrt() + EPS)
    assert torch.equal(precond.diag, expected)
    del model


def test_source_adamw_preconditioner_skips_step_zero(tmp_path: Path) -> None:
    """The step-0 policy, as chosen and documented: skip the P rows, warn loudly.

    Not P = I (which would publish chi_pos_P == chi_pos under the 'adamw'
    descriptor) and not 1/eps (which would scale every observable by ~1e16).
    """
    _write_bundle(tmp_path / "run")
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    factory = source.adamw_preconditioner()
    checkpoint = source.checkpoint_for(0)
    assert checkpoint.optimizer_state.is_empty
    with pytest.warns(UserWarning, match="SKIPPED"):
        assert factory(checkpoint.identifier) is None


def test_source_adamw_preconditioner_warns_on_metadata_mismatch(tmp_path: Path) -> None:
    """metadata.json is a cross-check; the saved param_groups win, but loudly."""
    _write_bundle(
        tmp_path / "run",
        steps=(10,),
        metadata_overrides={"optimizer_hparams": {"lr": LR, "betas": [BETA1, 0.999], "eps": 1e-6}},
    )
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    checkpoint = source.checkpoint_for(10)
    with pytest.warns(UserWarning, match="hyper-parameters disagree"):
        precond = source.adamw_preconditioner()(checkpoint.identifier)
    assert precond is not None
    # The saved param_groups value (BETA2 / EPS) is the one that was used.
    v = checkpoint.optimizer_state.flat_state("exp_avg_sq")
    assert torch.equal(precond.diag, 1.0 / ((v / (1.0 - BETA2**10)).sqrt() + EPS))


def test_source_adamw_preconditioner_honours_overrides(tmp_path: Path) -> None:
    _write_bundle(tmp_path / "run", steps=(10,))
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    checkpoint = source.checkpoint_for(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precond = source.adamw_preconditioner(beta2=0.5, eps=1e-3)(checkpoint.identifier)
    assert precond is not None
    v = checkpoint.optimizer_state.flat_state("exp_avg_sq")
    assert torch.equal(precond.diag, 1.0 / ((v / (1.0 - 0.5**10)).sqrt() + 1e-3))


def test_bundle_param_names_json_is_a_plain_list(tmp_path: Path) -> None:
    """Guard the writer/reader contract the golden fixture also has to honour."""
    _write_bundle(tmp_path / "run", steps=(10,))
    records = json.loads((tmp_path / "run" / "param_names.json").read_text())
    assert isinstance(records, list)
    assert all(
        isinstance(r, list) and len(r) == 2 and isinstance(r[0], str) and isinstance(r[1], list)
        for r in records
    )
