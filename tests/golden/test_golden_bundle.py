"""The WP0 golden end-to-end gate (``03_estimator.md`` §Golden).

Run it with ``pytest -m golden`` (the ``tests/conftest.py`` hook skips it
otherwise, so it never slows the unit tier down).

What it does, per parametrized architecture:

1. Writes its own checkpoint bundle by running ~100 CPU AdamW steps
   (:mod:`tests.utils.bundle_fixture`), in the frozen schema, with pre-step
   semantics and a full-sequence ``batch_a.npy``. The suite therefore stays
   self-contained after merge — no bundle from outside this repository is needed.
2. Proves the bundle reproduces its own update batch: what
   :attr:`LocalBundleCheckpoint.batch_a` reconstructs from ``batch_a.npy`` must be
   token-for-token the batch the loop trained on. A bundle that cannot do that
   silently corrupts every downstream number.
3. Runs vatis over it: raw + ``adamw`` rulers, ``k = 2`` fixed measurement
   batches ``A1``/``A2`` and one eval batch ``B``, all from held-out data, with
   ``delta_loss(A_i, B)`` cross pairs.
4. Asserts: the pipeline completes; every value is finite; a ``P = I`` re-run
   reproduces the raw columns (atol 1e-6); ``|chi_pos| <= 1.5`` and
   ``|chi_pos_P| <= 1.5``.
5. **Prints, with no hard assert**: (a) the linearization sum
   ``sum_t eta_t * delta_loss(A_t, B)`` against the realized change in
   ``loss(B)`` across the run, for both rulers; (b) how much of ``chi_net^P`` is
   carried by coordinates sitting at AdamW's ``1/eps`` ceiling. Both are
   diagnostics whose *value* is the number, not a pass/fail.

Parametrizations: the default tiny transformer (always) and the project Llama
(tied weights, rotary) when ``/workspace/antonio_code/llama_model`` is present —
skipped cleanly otherwise.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from tests.utils import llama_factory
from tests.utils.bundle_fixture import (
    DEFAULT_MODEL_FACTORY,
    batch_from_tokens,
    lm_loss,
    token_stream,
    write_training_bundle,
)
from vatis.analyzer import ALL_OBSERVABLES, P_OBSERVABLES, Analyzer
from vatis.core.chi_net import build_estimator
from vatis.core.preconditioner import (
    RAW_DESCRIPTOR,
    Preconditioner,
    epsilon_cap_mask,
)
from vatis.models.local_bundle import load_local_bundle

LLAMA_FACTORY = "tests.utils.llama_factory:build_llama_model"

# Gate settings. Small on purpose: the point is end-to-end correctness, not scale.
N_STEPS = 100
BUNDLE_EVERY = 10
LR = 3e-3
EPS = 1e-8
VOCAB_SIZE = 48
SEQ_LEN = 16
TRAIN_BATCH = 4
MEASURE_BATCH = 4
N_HUTCHINSON = 8
CHI_POS_LIMIT = 1.5


def _held_out(seed: int, n_rows: int = MEASURE_BATCH) -> dict[str, torch.Tensor]:
    """A measurement batch drawn from the held-out end of the token process.

    Seeds >= 900 are never used by the training stream (which uses ``seed + 1``
    with ``seed = 0``), so ``A1``, ``A2`` and ``B`` are genuinely held out.
    """
    return batch_from_tokens(
        token_stream(n_rows=n_rows, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=seed)
    )


def _rows_by(rows: list, descriptor: str) -> dict[tuple[str, str, str], float]:
    return {
        (r.batch_a, r.batch_b, r.observable): r.value
        for r in rows
        if r.preconditioner == descriptor
    }


def _analyzer(source: object, eval_batches: dict, preconditioner: object, seed: int) -> Analyzer:
    return Analyzer(
        eval_batches=eval_batches,
        preconditioner=preconditioner,  # type: ignore[arg-type]
        chi_net_method="per_sequence_cv",
        n_hutchinson=N_HUTCHINSON,
        micro_batch_size=2,
        cross_pairs=[(name, "B") for name in eval_batches if name != "B"],
        sink=None,
        seed=seed,
        device="cpu",
        dtype="fp32",
    )


@pytest.mark.golden
@pytest.mark.parametrize(
    "architecture",
    [
        pytest.param("default", id="default-tiny-transformer"),
        pytest.param("llama", id="project-llama"),
    ],
)
def test_golden_bundle_end_to_end(tmp_path: Path, architecture: str) -> None:
    if architecture == "llama":
        if not llama_factory.llama_available():
            pytest.skip(
                "project Llama not available (set VATIS_LLAMA_MODEL_PATH to the directory "
                "containing llama_model/); the gate stays self-contained without it"
            )
        factory = LLAMA_FACTORY
        model_config_extra = llama_factory.llama_model_config(
            vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, tie_head=True
        )
    else:
        factory = DEFAULT_MODEL_FACTORY
        model_config_extra = None

    run_dir = tmp_path / "run"
    fixture = write_training_bundle(
        run_dir,
        model_factory=factory,
        n_steps=N_STEPS,
        bundle_every=BUNDLE_EVERY,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        batch_size=TRAIN_BATCH,
        lr=LR,
        eps=EPS,
        tie_head=True,
        model_config_extra=model_config_extra,
    )
    print(
        f"\n[golden/{architecture}] bundle: {len(fixture.steps)} checkpoints "
        f"{fixture.steps}, {fixture.n_parameters} parameters in "
        f"{len(fixture.param_names)} tensors, lr={LR}"
    )

    # ---- (2) the bundle must reproduce its own update batch ------------------
    source = load_local_bundle(run_dir, factory, dtype="fp32")
    assert source.steps == fixture.steps
    for step in fixture.steps:
        checkpoint = source.checkpoint_for(step)
        expected = fixture.expected_batch(step)
        reconstructed = checkpoint.batch_a
        assert torch.equal(reconstructed["input_ids"], expected["input_ids"]), (
            f"step {step}: batch_a.npy does not round-trip to the batch the fixture trained on"
        )
        assert torch.equal(reconstructed["labels"], expected["labels"]), (
            f"step {step}: reconstructed labels differ from the fixture's — the stored "
            "sequence and the training objective disagree"
        )
        # x = ids[:, :-1] -> y = ids[:, 1:] is the pair set the labels encode.
        ids = reconstructed["input_ids"]
        assert torch.equal(reconstructed["labels"][:, :-1], ids[:, 1:])
        assert int((reconstructed["labels"][:, -1] == -100).sum()) == ids.shape[0]
        checkpoint.release()

    # ---- (3) the measured run: raw + adamw ----------------------------------
    eval_batches = {"A1": _held_out(901), "A2": _held_out(902), "B": _held_out(903)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = _analyzer(source, eval_batches, source.adamw_preconditioner(), seed=17).run(
            source.model_specs()
        )
    skip_warnings = [w for w in caught if "SKIPPED" in str(w.message)]
    print(
        f"[golden/{architecture}] step-0 policy: {len(skip_warnings)} checkpoint(s) had an "
        "empty optimizer state and had their P rows skipped"
    )

    assert len(results) == len(fixture.steps)
    n_p_checkpoints = 0
    for result in results:
        observables = {r.observable for r in result.rows}
        assert set(ALL_OBSERVABLES) <= observables, (
            f"{result.checkpoint_id}: missing raw observables {set(ALL_OBSERVABLES) - observables}"
        )
        has_p = bool(observables & set(P_OBSERVABLES))
        if has_p:
            n_p_checkpoints += 1
            assert set(P_OBSERVABLES) <= observables
        for row in result.rows:
            assert row.value == row.value, f"{result.checkpoint_id}/{row.observable} is NaN"
            assert abs(row.value) < 1e30, (
                f"{result.checkpoint_id}/{row.observable} is not finite: {row.value!r}"
            )
            if row.observable in ("chi_pos", "chi_pos_P"):
                assert abs(row.value) <= CHI_POS_LIMIT, (
                    f"{result.checkpoint_id} {row.batch_a}/{row.batch_b} "
                    f"{row.observable} = {row.value!r} exceeds |chi_pos| <= {CHI_POS_LIMIT}"
                )
        assert {r.preconditioner for r in result.rows} <= {RAW_DESCRIPTOR, "adamw"}
    # The step-0 checkpoint legitimately has no P; everything else must have one.
    assert n_p_checkpoints == len(fixture.steps) - 1, (
        f"expected P rows at every checkpoint but step 0, got {n_p_checkpoints} of "
        f"{len(fixture.steps)}"
    )

    # ---- (4) the P = I re-run must reproduce the raw columns ----------------
    identity_results = _analyzer(source, eval_batches, "identity", seed=17).run(
        source.model_specs()
    )
    worst_identity = 0.0
    for raw_result, identity_result in zip(results, identity_results, strict=True):
        assert raw_result.checkpoint_id == identity_result.checkpoint_id
        raw = _rows_by(raw_result.rows, RAW_DESCRIPTOR)
        identity = _rows_by(identity_result.rows, "identity")
        for (batch_a, batch_b, name), value in identity.items():
            raw_name = name.removesuffix("_P")
            reference = raw[(batch_a, batch_b, raw_name)]
            deviation = abs(value - reference)
            worst_identity = max(worst_identity, deviation)
            assert deviation <= 1e-6, (
                f"{identity_result.checkpoint_id} {batch_a}/{batch_b}: P=I {name}={value!r} "
                f"!= raw {raw_name}={reference!r} (abs {deviation:.3e})"
            )
    print(
        f"[golden/{architecture}] P=I re-run vs raw columns: worst |deviation| = "
        f"{worst_identity:.3e} (limit 1e-6)"
    )

    # ---- (5a) the 1/eps ceiling diagnostic ---------------------------------
    _report_epsilon_cap_share(source, eval_batches["B"], architecture)

    # ---- (5b) the linearization sum ----------------------------------------
    _report_linearization_sum(source, eval_batches["B"], fixture, architecture)


def _report_epsilon_cap_share(
    source: object,
    batch_b: dict[str, torch.Tensor],
    architecture: str,
) -> None:
    """Print what share of ``chi_net^P`` comes from AdamW's ``1/eps`` ceiling.

    Coordinates the gradient never touched keep ``exp_avg_sq == 0`` exactly, so
    ``P = 1/eps`` there — ~``1e6`` times a live coordinate's weight at
    ``eps = 1e-8``. The convention is frozen (no clipping), so the honest thing
    is to make its contribution visible: ``Tr(J P J^T)`` is *linear* in ``P`` at
    fixed probes, so running the estimator twice with the same probe seed — once
    with ``P`` and once with ``P`` masked to the capped coordinates — splits the
    trace exactly into a capped and a live part.
    """
    checkpoint = source.checkpoints[-1]  # type: ignore[attr-defined]
    bundle = checkpoint.model_bundle
    state = checkpoint.optimizer_state
    precond = Preconditioner.adamw(
        state.flat_state("exp_avg_sq"),
        step=int(state.step_count or 0),
        beta2=float(state.hparams["betas"][1]),
        eps=float(state.hparams["eps"]),
    )
    mask = epsilon_cap_mask(precond.diag, float(state.hparams["eps"]))
    capped_only = Preconditioner.from_diagonal(
        precond.diag * mask.to(dtype=precond.diag.dtype), descriptor="adamw-capped-only"
    )

    def trace(p: Preconditioner) -> float:
        generator = torch.Generator().manual_seed(2024)
        estimator = build_estimator("per_sequence_cv", n_hutchinson=N_HUTCHINSON)
        result = estimator.compute(
            bundle.model,
            batch_b,
            bundle.forward_fn,
            loss_fn=bundle.loss_fn,
            valid_mask_fn=bundle.valid_mask_fn,
            params=bundle.params,
            micro_batch_size=2,
            generator=generator,
            preconditioner=p,
        )
        assert result.chi_net_p is not None
        return float(result.chi_net_p)

    total = trace(precond)
    capped = trace(capped_only)
    fraction = float(mask.to(dtype=torch.float64).mean())
    share = capped / total if total != 0.0 else float("nan")
    # Which parameters the capped coordinates belong to — far more actionable
    # than a bare percentage.
    layout = source.layout  # type: ignore[attr-defined]
    mask_f = mask.to(dtype=torch.float32)
    per_parameter = {
        name: f"{int(layout.segment(mask_f, name).sum())}/{layout.numel_of(name)}"
        for name in layout.names
        if float(layout.segment(mask_f, name).sum()) > 0.0
    }
    print(
        f"[golden/{architecture}] AdamW 1/eps ceiling at {checkpoint.revision}: "
        f"{fraction:.2%} of P entries are at 1/eps ({int(mask.sum())}/{mask.numel()}), "
        f"in {per_parameter or '{}'}; they carry {share:.2%} of chi_net^P on the "
        f"held-out batch B (chi_net^P total {total:.6g}, capped part {capped:.6g}). "
        "The convention P = 1/(sqrt(v_hat) + eps) is frozen (00_context.md), so this is "
        "reported, not clipped. A 0% share is a property of THIS fixture (its capped "
        "coordinates are also masked out on B), not a general guarantee: a coordinate "
        "that is dead in training but live on a held-out batch enters chi_net^P with "
        "~1/eps weight."
    )
    checkpoint.release()


def _report_linearization_sum(
    source: object,
    batch_b: dict[str, torch.Tensor],
    fixture: object,
    architecture: str,
) -> None:
    """Print ``sum_t eta_t delta_loss(A_t, B)`` against the realized ``delta loss(B)``.

    No hard assert — this is the validation-sum *report* of ``03_estimator.md``
    §Golden step 4. Two things it compares:

    - the **raw** ruler's prediction ``-sum_t eta_t delta_loss(A_t, B)``, i.e. what
      a plain-SGD update of size ``eta`` would have achieved, and
    - the **preconditioned** ruler's prediction ``-sum_t eta_t delta_loss^P(A_t, B)``,
      which is the one that matches the optimizer that actually ran: AdamW's
      update is ``-eta P grad L^A`` (up to the momentum filter and decoupled weight
      decay, both excluded from ``P`` by project decision), so the first-order
      change in ``loss(B)`` is ``grad L^B . dtheta = -eta delta_loss^P(A_t, B)``.

    Because the bundle samples every ``BUNDLE_EVERY``-th step, the sum over
    bundled checkpoints is scaled by that stride to cover the whole run; both the
    unscaled and the scaled numbers are printed so the assumption (``delta_loss``
    varies slowly between bundled steps) is visible rather than buried.
    """
    steps = list(source.steps)  # type: ignore[attr-defined]
    stride = int(fixture.bundle_every)  # type: ignore[attr-defined]

    # delta_loss(A_t, B) with A_t = the stored update batch of checkpoint t.
    analyzer = Analyzer(
        eval_batches={"update_batch": source.batch_a_spec(), "B": batch_b},  # type: ignore[attr-defined]
        preconditioner=source.adamw_preconditioner(),  # type: ignore[attr-defined]
        chi_net_method="per_sequence_cv",
        n_hutchinson=N_HUTCHINSON,
        micro_batch_size=2,
        cross_pairs=[("update_batch", "B")],
        sink=None,
        seed=5,
        device="cpu",
        dtype="fp32",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = analyzer.run(source.model_specs())  # type: ignore[attr-defined]

    raw_terms: dict[int, float] = {}
    p_terms: dict[int, float] = {}
    for step, result in zip(steps, results, strict=True):
        for row in result.rows:
            if (row.batch_a, row.batch_b) != ("update_batch", "B"):
                continue
            if row.observable == "delta_loss" and row.preconditioner == RAW_DESCRIPTOR:
                raw_terms[step] = row.value
            if row.observable == "delta_loss_P" and row.preconditioner == "adamw":
                p_terms[step] = row.value

    # loss(B) at every checkpoint, from the bundle's own weights.
    losses: dict[int, float] = {}
    learning_rates: dict[int, float] = {}
    for checkpoint in source.checkpoints:  # type: ignore[attr-defined]
        bundle = checkpoint.model_bundle
        with torch.no_grad():
            losses[checkpoint.step] = float(
                lm_loss(bundle.forward_fn(bundle.model, batch_b), batch_b)
            )
        learning_rates[checkpoint.step] = float(checkpoint.meta.lr or LR)
        checkpoint.release()

    contributing = list(steps[:-1])  # the last checkpoint applies no update
    raw_steps = [s for s in contributing if s in raw_terms]
    p_steps = [s for s in contributing if s in p_terms]
    raw_sum = sum(learning_rates[s] * raw_terms[s] for s in raw_steps)
    p_sum = sum(learning_rates[s] * p_terms[s] for s in p_steps)
    realized = losses[steps[-1]] - losses[steps[0]]

    def ratio(predicted: float) -> str:
        if realized == 0.0:
            return "n/a"
        return f"{predicted / realized:+.3f}x"

    print(
        f"\n[golden/{architecture}] LINEARIZATION SUM (report only, no assert)\n"
        f"  loss(B) at first checkpoint (step {steps[0]}): {losses[steps[0]]:.6f}\n"
        f"  loss(B) at last  checkpoint (step {steps[-1]}): {losses[steps[-1]]:.6f}\n"
        f"  realized change in loss(B):                    {realized:+.6f}\n"
        f"  sum_t eta_t*delta_loss(A_t,B)   over {len(raw_steps)} bundled steps "
        f"{raw_steps}:\n"
        f"      {raw_sum:.6f}  -> predicted change {-raw_sum:+.6f}\n"
        f"      x stride {stride} (whole run): {stride * raw_sum:.6f}  -> predicted change "
        f"{-stride * raw_sum:+.6f} ({ratio(-stride * raw_sum)} of realized)\n"
        f"  sum_t eta_t*delta_loss^P(A_t,B) over {len(p_steps)} bundled steps "
        f"{p_steps}:\n"
        f"      {p_sum:.6f}  -> predicted change {-p_sum:+.6f}\n"
        f"      x stride {stride} (whole run): {stride * p_sum:.6f}  -> predicted change "
        f"{-stride * p_sum:+.6f} ({ratio(-stride * p_sum)} of realized)\n"
        f"  per-step terms (raw): "
        f"{ {s: round(raw_terms[s], 5) for s in raw_steps} }\n"
        f"  per-step terms (P):   "
        f"{ {s: round(p_terms[s], 5) for s in p_steps} }\n"
        f"  CAVEATS, stated rather than buried: (a) only every {stride}-th step is bundled, "
        f"so the sum is scaled by the stride, which assumes delta_loss varies slowly between "
        f"bundled steps; (b) step {steps[0]} contributes to the raw sum but NOT to the P sum "
        f"(its optimizer state is empty, so no P exists there — see the step-0 policy); "
        f"(c) the optimizer that ran is AdamW, so the P row is the ruler that matches its "
        f"update (-eta P grad L, momentum and decoupled weight decay excluded from P by "
        f"project decision); the raw row is the plain-SGD prediction and is expected to be "
        f"far too small, because eta was tuned for the preconditioned step."
    )
