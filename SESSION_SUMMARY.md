# vatis — session summary

End-to-end build of `vatis` per `CLAUDE.md`, plus a v1.1 hardening pass per
the original `TASKS_NEXT.md`, plus three post-task follow-ups on the
deployment example, plus a v1.1→v1.2 spec reconciliation that applied the
`CLAUDE.md.proposed` updates and triaged `FOLLOWUPS.md` into a fresh
`TASKS_NEXT.md` for v1.2. **67 tests green at the time of the
reconciliation.** Read `TASKS_NEXT.md` next to see what's queued for v1.2.

## Read first

If you're returning to this project: start by skimming this file, then run

```bash
.venv/bin/python -m pytest tests -q
```

You should see `56 passed` in ~30 s. After that, the code in
`vatis/core/observables.py` and `vatis/core/chi_net/per_seq_cv.py` are the
two files worth reading carefully — they encode all the math; everything
else is plumbing on top of them.

## What was built

The full v1 surface from `CLAUDE.md`, in the order specified by the
"Implementation order" section:

```
vatis/
├── __init__.py            lazy-import public surface (analyze, Analyzer, EvalBatchSpec)
├── _version.py
├── _worker.py             torchrun worker entry — calls back into cli.run_from_args
├── __main__.py            `python -m vatis` shim
├── analyzer.py            Analyzer class + analyze() one-shot wrapper
├── cli.py                 argparse-based `vatis run ...` CLI
├── core/
│   ├── normalization.py   valid_token_mask, count_valid, sqrt(N_A·N_B)
│   ├── probes.py          Rademacher / Gaussian probe gen + position masking
│   ├── observables.py     closed-form chi_loss (CE), autograd fallback,
│   │                      delta_loss self/cross, chi_pos combinator
│   └── chi_net/
│       ├── base.py        ChiNetEstimator ABC + ChiNetResult
│       ├── hutchinson.py  plain Hutchinson (always-works fallback)
│       ├── per_seq_cv.py  per-sample backward + control-variate Hutchinson;
│       │                  also surfaces the per-sample alignment matrix
│       └── opacus.py      stub raising NotImplementedError (v1.1)
├── models/
│   └── hf.py              load_hf_model, ModelBundle, causal_lm_{forward,loss,valid_mask}
├── data/
│   ├── batches.py         EvalBatchSpec (fixed / resample / callable / auto)
│   └── collate.py         batch_size, slice_batch, iter_micro_batches, shifted_lm_targets
├── distributed/
│   ├── ddp.py             init/shutdown/all-reduce wrappers, single-process safe
│   ├── sharding.py        shard_batch with even-remainder distribution
│   └── launcher.py        torchrun re-exec wrapper
└── sinks/
    ├── base.py            ResultSink ABC + ResultRow long-format dataclass
    ├── parquet.py         canonical ParquetSink (streaming + buffered modes)
    ├── wandb.py           optional W&B sink (lazy import)
    └── tensorboard.py     optional TB sink (lazy import)
```

Total: ~2.9k lines of Python in `vatis/`, ~1.6k lines of tests.

### Critical math notes

Two correctness-relevant implementation details that aren't obvious from
`CLAUDE.md`:

1. **chi_loss across micro-batches.** The naive approach of computing
   `chi_loss_cross_entropy` per micro-batch and summing is **wrong** because
   each micro divides by its own `N_micro`, not the global `N_total`. The fix
   (in `vatis/core/observables.py`): expose
   `chi_loss_cross_entropy_unnormalized` which returns `Σ_valid (softmax -
   onehot)²` *without* the `1/N²` factor. The analyzer accumulates this raw
   sum across all micros + DDP ranks, then divides by `N_total²` exactly
   once. This is verified by `test_analyzer_chi_loss_invariant_to_micro_batch_size`
   which asserts bit-identical chi_loss for micro=1, 2, 4 on the same batch.

2. **delta_loss across micro-batches and DDP.** Each rank computes its
   local-mean-loss gradient `∇_θ ((1/n_local) Σ_local ℓ)`, weighted by
   `(n_local/n_global)`, and the all-reduce-sum gives the global mean loss
   gradient `∇_θ ((1/n_global) Σ_all ℓ)`. Squaring locally on each rank
   would give `Σ_r ‖g_r‖²`, which is **not** the same as `‖Σ_r g_r‖²`.
   So we all-reduce the *flat gradient vector* and square after.
   Verified by `test_ddp_loopback_matches_single_process` which compares
   single-process vs 2-rank torchrun on CPU.

3. **Per-sequence control variate construction.** The math is documented
   inside `vatis/core/chi_net/per_seq_cv.py`. Briefly: for each micro-batch
   we compute `u_full = ∇_f L_total` (one autograd through the loss head),
   then `B` per-sample backwards `g_b = J_b^T u_full[b]` (one each, with
   `retain_graph=True`). The exact loss-direction term is
   `Σ_b ‖g_b‖²/‖u_b‖²`. The Hutchinson part uses probes `v` and computes
   `‖J^T v − Σ_b α_b g_b/‖u_b‖‖²` (the parameter-space projection of `v`
   minus its loss-direction component, all done as cheap dot products in
   parameter space — no extra backwards). Sum gives the unbiased total.

   The bonus `alignment_matrix = G G^T` (where `G` is the per-sample flat
   grad matrix) is built once at the end of the estimator call.

## What works

| Feature | Status | Test |
|---|---|---|
| Closed-form `chi_loss` for CE | ✓ | `tests/unit/test_observables.py::test_chi_loss_closed_form_matches_autograd_*` |
| Autograd `chi_loss` fallback | ✓ | same |
| `delta_loss` self & cross | ✓ | `test_delta_loss_*` |
| `chi_pos` combinator | ✓ | `test_chi_pos_*` |
| Hutchinson `chi_net` | ✓ | `test_hutchinson_unbiased_converges_to_exact` |
| Per-sequence-CV `chi_net` | ✓ | `test_per_seq_cv_unbiased_converges_to_exact` |
| Per-sample alignment matrix bonus | ✓ | `test_per_seq_cv_records_alignment_matrix` |
| Auto-method selection (B≤32 → cv, else hutch) | ✓ | `test_select_chi_net_method_auto` |
| Opacus estimator stub raises | ✓ | `test_opacus_estimator_raises_not_implemented` |
| Padding via `attention_mask`/`labels=-100` | ✓ | `test_chi_loss_closed_form_matches_autograd_with_padding`, `test_analyzer_with_padding_uses_attention_mask` |
| Single-GPU end-to-end (TinyTransformer + ParquetSink) | ✓ | `test_analyzer_*` |
| Single-GPU end-to-end (TinyMLP) | ✓ | `test_analyzer_works_for_mlp` |
| chi_loss / delta_loss invariant under micro-batching | ✓ | `test_analyzer_chi_loss_invariant_to_micro_batch_size` |
| ParquetSink streaming flush | ✓ | `test_parquet_sink_streaming_flush` |
| Cross-validation against perspic (functorch + opacus, both vatis methods, self + cross pairs) | ✓ | `tests/cross_validation/test_vs_perspic.py` (31 tests) |
| DDP 2-rank loopback (CPU torchrun) | ✓ | `tests/integration/test_ddp_loopback.py` |
| HF model loading + analysis on real Llama-1.25M | ✓ | `tests/integration/test_hf_load.py` |
| `python -m vatis run --help` | ✓ | manual smoke |

### Test counts (after v1.1 hardening pass)

```
tests/unit/                49 tests   (test_normalization 8, test_probes 7,
                                       test_observables 6, test_chi_net 15,
                                       test_distributed 4, test_analyzer 9)
tests/cross_validation/    14 tests   (test_vs_perspic, parametrized over
                                       engine × method)
tests/integration/          4 tests   (test_ddp_loopback + test_hf_load)
                          ----
                           67 tests
```

Run with `.venv/bin/python -m pytest tests` (~65 s on the reference single
RTX 3090 Ti, ~30 s for unit tests only).

Test count delta from v1 (56 → 67):
- `test_observables`: −3 (`chi_loss_from_autograd` tests deleted in task 6)
- `test_chi_net`: +8 (5 memory pre-check, 2 exact-NTK ground truth, 1 LM padding)
- `test_analyzer`: +2 (cross-pair contract, revision threading)
- `test_vs_perspic`: +4 (2 tight-tolerance parametrized, 2 heavy-padding parametrized)

### Cross-validation key map (verified against perspic)

| perspic | vatis | tolerance |
|---|---|---|
| `chi_loss` | `chi_loss_normalized` | exact (`rel ≤ 1e-4`, `abs ≤ 1e-6`) |
| `chi_net` | `chi_net_normalized` | < 2% (Hutchinson with n=2048) |
| `grad_norm_squared` | `delta_loss` | exact (`rel ≤ 1e-4`, `abs ≤ 1e-6`) |
| `chi_coup` (≡ `chi_align`) | `chi_pos` | < 2% (inherits chi_net noise) |

The mapping is verified for both perspic backends (`functorch` + `opacus`)
and both vatis chi_net methods (`hutchinson` + `per_sequence_cv`).

The math behind the mapping:
- vatis stores `chi_loss = Σ(softmax-onehot)²/N²` (the "raw" form);
  multiplied by `N` it becomes `chi_loss_normalized = (1/N) Σ(softmax-onehot)²`
  which is exactly what perspic emits with `normalize=True`.
- vatis stores `chi_net = Σ_b Tr(M_b)` (raw); divided by `N` it becomes
  `chi_net_normalized = (1/N) Σ_b Tr(M_b)`, matching perspic.
- `delta_loss = ‖∇_θ L‖²` raw on both sides.
- `chi_pos = δL / (chi_loss × chi_net)` is invariant under the
  normalization choice — the `√(N·N)` factors cancel — so it equals
  perspic's `chi_coup` directly.

## What is NOT in v1 (deferred to v1.1)

- **`OpacusEstimator` full implementation.** The stub is in
  `vatis/core/chi_net/opacus.py` and raises `NotImplementedError` with a
  clear pointer. Per CLAUDE.md the v1.1 work is to port the layer-compat
  check + in-place neutralization + tied-embedding detection from
  `perspic/calculator/samplewise_opacus.py`.
- **FSDP / tensor parallel.** v1 only does plain DDP with the model
  replicated on each rank. CLAUDE.md says this is intentional.
- **Custom `loss_fn` (non-CE) is out of scope until v1.2.** The analyzer
  always uses the closed-form CE path for `chi_loss`. The previously-shipped
  `chi_loss_from_autograd` helper was deleted in the v1.1 hardening pass
  because it was unused outside its own tests; if a future user needs a
  non-CE path, the smaller building blocks (`torch.autograd.grad(loss,
  logits)` plus a masked squared sum) make it easy to re-add behind a
  bundle flag.
- **Per-sample variance observables beyond the cross-sample alignment
  matrix.** Per CLAUDE.md scope.
- **OOM auto-tuning of `micro_batch_size`.** Per CLAUDE.md the user is
  responsible for picking a size that fits.

## What is blocked or risky

Nothing is blocked. There is no `BLOCKED.md` — no permission denial or
external obstacle was hit during the build or the v1.1 hardening pass.
A few notes on things that could trip a future maintainer:

1. **`hash()` randomization.** Originally `_per_call_seed` used Python's
   built-in `hash((seed, ckpt_id, batch_name))` which is randomized per
   process via `PYTHONHASHSEED`. This caused intermittent cross-validation
   failures depending on test ordering. The fix
   (`vatis/analyzer.py::_per_call_seed`) uses `hashlib.sha256` for a
   deterministic mix. **Don't undo this** — the deterministic seed is what
   makes Hutchinson reproducible across runs.
2. **Per-sequence-CV memory cost.** ~~Latent footgun~~ **Now guarded.** Task 7
   added a startup check (`PerSequenceControlVariateEstimator.check_memory_feasible`)
   that raises if `B * n_params * 4 > 0.5 * available_device_memory`. The
   auto-selection rule still picks `per_sequence_cv` for `B ≤ 32` only, but
   the check fires regardless of how the estimator was constructed. For
   8B-class models the check kicks in early and recommends switching to
   `chi_net_method="hutchinson"`.
3. **`ParquetSink` streaming uses `pyarrow.parquet.ParquetWriter`** which
   keeps the file handle open across flushes. Don't try to read the parquet
   file from a separate process while a streaming writer is active. Also:
   each new `analyze()` call constructs a fresh `ParquetSink` for the same
   path, **truncating** the file. To accumulate across multiple `analyze()`
   calls, either pass a single `ParquetSink` instance or use the
   `revisions=[...]` parameter so a single `analyze()` call drives the
   whole sweep. The deployment example in `examples/pythia_sweep.py`
   demonstrates the latter. (See FOLLOWUPS.md for the systematic fix.)
4. **The analyzer's cross-pair path** (`_compute_cross_pair`):
   ~~Latent footgun~~ **Now guarded.** Task 5 added a precondition assert
   that raises a clear `RuntimeError` if `_compute_cross_pair` is called
   before the corresponding `_compute_self_pair`. The cache dicts are
   eagerly initialized in `__init__`. The public `Analyzer.run` always
   runs the self loop first, so the contract is invisible to normal users.
5. **`Bundle.valid_mask_fn` and `chi_loss` token counting can disagree.**
   The analyzer's `chi_loss` accumulator uses `valid_token_mask` (which
   honors `labels=-100`) to compute the raw squared sum, but the
   `n_valid` count (used to divide by `N²`) comes from
   `bundle.valid_mask_fn`. If a user supplies a `valid_mask_fn` that
   doesn't agree with the labels-vs-ignore_index convention, the two
   paths will use different masks and chi_loss will be subtly wrong.
   Task 4 sidesteps this by using a consistent `valid_mask_fn`. The
   systematic fix is one line in the analyzer; see FOLLOWUPS.md.
6. **The opacus warning during cross-validation tests** about
   `register_full_backward_hook` is from perspic's internal implementation,
   not ours. Ignore.

## What changed vs. CLAUDE.md spec

- The implementation order in CLAUDE.md says to add `vatis/data/batches.py`
  in step 7, but the chi_net estimators (steps 3-4) need
  `vatis.data.collate.iter_micro_batches`, so I wrote `data/collate.py` in
  step 4 alongside the estimators. `data/batches.py` came in step 6 as
  scheduled.
- I added `vatis/data/collate.py` (not explicitly listed in the
  architecture diagram in CLAUDE.md) with `iter_micro_batches`,
  `slice_batch`, `move_batch`, `shifted_lm_targets`. The spec says micro-
  batch splitting belongs to `data/collate.py`, so this is consistent.
- I added a `chi_loss_cross_entropy_unnormalized` helper in
  `core/observables.py` to make the micro-batch chi_loss computation
  correct (see the "Critical math notes" above). The CLAUDE.md formula
  for `chi_loss` is unambiguous; this is an internal refactor.

## How to use it

```python
from vatis import analyze

# Option A: HF model by name + revisions
results = analyze(
    model="EleutherAI/pythia-160m",
    revisions=["step1000", "step2000", "step10000"],
    eval_batches={
        "val_fixed": frozen_batch_dict,
        "val_resample": val_dataloader,
        "train_at_step": lambda ckpt: ...,
    },
    chi_net_method=None,           # auto: per_seq_cv if B≤32 else hutchinson
    n_hutchinson=32,
    micro_batch_size=4,
    sink="results.parquet",
    num_gpus=4,                    # only honored when launched via CLI
    dtype="bf16",
)

# Option B: pre-built ModelBundle for non-HF models
from vatis.models.hf import ModelBundle
bundle = ModelBundle(model=my_module, params=list(my_module.parameters()),
                     forward_fn=my_fwd, loss_fn=my_loss,
                     valid_mask_fn=my_mask, identifier="custom@v1")
results = analyze(model=bundle, eval_batches={"val": batch}, sink="out.parquet")

# CLI form (under torchrun for num_gpus > 1)
$ python -m vatis run \
    --model EleutherAI/pythia-160m \
    --revisions step1000 step2000 \
    --eval-batch val=path/to/batch.pt \
    --sink results.parquet \
    --num-gpus 4
```

## Repro the test suite

```bash
cd /tikhome/knikolaou/PycharmProjects/vatis

# Lint + types + tests, all from the venv:
.venv/bin/ruff check vatis/ tests/      # 0 errors
.venv/bin/ruff format vatis/ tests/ --check
.venv/bin/mypy vatis/                    # strict, 0 errors
.venv/bin/python -m pytest tests        # 67 passed in ~65s

# Or run individual marker subsets:
.venv/bin/python -m pytest tests/unit -q                                   # 49 tests
.venv/bin/python -m pytest tests/cross_validation -m cross_validation -q   # 14 tests
.venv/bin/python -m pytest tests/integration -m integration -q             # 4 tests
```

## Files most worth reading

In rough order of "look at this first":

1. **`vatis/core/observables.py`** — closed-form chi_loss, delta_loss
   self/cross, the chi_pos combinator. The math is short and
   self-contained. (The autograd fallback was removed in v1.1 task 6;
   non-CE losses are deferred to v1.2.)
2. **`vatis/core/chi_net/per_seq_cv.py`** — the control-variate estimator
   with the full math derivation in the module docstring. This is the most
   subtle file.
3. **`vatis/core/chi_net/hutchinson.py`** — the always-works fallback,
   ~50 LOC.
4. **`vatis/analyzer.py`** — the orchestrator. Look at `_compute_self_pair`
   to see how chi_loss / chi_net / delta_loss are stitched together with
   the all-reduce semantics.
5. **`tests/cross_validation/test_vs_perspic.py`** — the perspic mapping,
   the actual numerical agreement, and the tolerance choices.
6. **`tests/unit/test_chi_net.py::_exact_chi_net`** — the exact ground truth
   computation that the Hutchinson convergence test runs against. It
   iterates over `(B, S, V)` and backprops each scalar — only feasible
   because the toy fixture is sized so `S * V ≤ 1024`.

If you want to understand DDP, read:

7. **`vatis/distributed/sharding.py`** + the DDP block in
   `vatis/analyzer.py::_compute_self_pair` (the `if is_distributed():`
   branch), which shows the rank-weighting trick for the loss gradient.

## v1.1 hardening pass

A second autonomous session driven by `TASKS_NEXT.md`. Goal was *trust*
and *demonstrability*, not new features. Eight tasks plus the baseline
commit. Worked through them in the recommended order
(`0 → 5 → 6 → 7 → 2 → 3 → 4 → 1 → 8` plus the final SESSION_SUMMARY
update). All tests still green at the end of every task.

### What got done

| task | one-line summary | files touched | tests added |
|---|---|---|---|
| 0 | baseline commit (`git init`, snapshot of v1 build state) | `.gitignore` | — |
| 5 | assert self-pair precedence in `_compute_cross_pair`; eagerly init caches | `vatis/analyzer.py`, `tests/unit/test_analyzer.py` | +1 |
| 6 | delete unused `chi_loss_from_autograd`; non-CE losses deferred to v1.2 | `vatis/core/observables.py`, `tests/unit/test_observables.py`, `SESSION_SUMMARY.md`, `vatis/analyzer.py` (docstring) | −3 |
| 7 | startup memory pre-check in `PerSequenceControlVariateEstimator` (refuses configs that would need >50% of free GPU/host RAM in per-sample grad cache) | `vatis/core/chi_net/per_seq_cv.py`, `tests/unit/test_chi_net.py` | +5 |
| 2 | exact-NTK ground-truth tests for `chi_pos` and `delta_loss(A, B)` via an explicit per-token Jacobian builder; tight 1e-5 tolerance | `tests/unit/test_chi_net.py` | +2 |
| 3 | tight-tolerance perspic cross-check at `n_hutchinson=16384` (rel 0.3%, abs 1e-6) for both vatis methods | `tests/cross_validation/test_vs_perspic.py` | +2 |
| 4 | heavy-padding cross-validation: TinyMLP with 5/8 samples ignored vs perspic on the unpadded subset, plus an LM-padding regression against the exact-NTK Jacobian builder | `tests/cross_validation/test_vs_perspic.py`, `tests/unit/test_chi_net.py` | +3 |
| 1 | deployment example + benchmark on `pythia-14m`: `examples/pythia_sweep.py` (~20 s warm / ~50 s cold end-to-end on RTX 3090 Ti), `examples/BENCHMARK.md` (18-row benchmark sweep + budget arithmetic), four plots, parquet output. Surfaced and fixed a latent bug: self-pair rows were emitting `revision=""` regardless of input. **Three post-task follow-ups** (separate commits): (a) replaced the random-integer eval batches with two real-text batches (Pride and Prejudice prose + a Python module) tokenized at runtime with the model's own tokenizer, and added the prose×code cross pair via `cross_pairs=[("prose","code")]`. (b) extended the checkpoint set with four log-spaced early checkpoints (step1, step8, step64, step512) so the early restructuring phase is visible — uncovered a chi_net U-shape, a chi_loss "warmup cliff" at step64, and non-monotonic cross delta_loss between step1 and step1000. (c) added `examples/analyze_results.py`, a small post-processing script that reads the parquet and computes the normalized cross-batch gradient correlation `cos(g_A,g_B) = δL(A,B)/√(δL(A,A)·δL(B,B))`, showing that most prose↔code decorrelation happens by step512 (early-phase ~26× drop) and the post-step1000 rebound is much smaller in correlation terms than the absolute delta_loss plot makes it look. The analysis script intentionally has zero vatis imports — it's the canonical "downstream consumer reading the parquet" demo. | `examples/`, `vatis/analyzer.py`, `tests/unit/test_analyzer.py`, `pyproject.toml`, `uv.lock` | +1 |
| 8 | `CLAUDE.md.proposed` with allowed-section updates; `FOLLOWUPS.md` for items in the forbidden sections | `CLAUDE.md.proposed`, `FOLLOWUPS.md` | — |

Total test delta: 56 → 67 (+11). All 67 tests pass; lint, format, and
strict mypy are clean.

### What got skipped (and why)

Nothing was skipped. There is no `BLOCKED.md` from this session — every
task ran to completion. `FOLLOWUPS.md` lists three categories of
items that were noticed but deliberately not addressed:
1. Stale paragraphs in CLAUDE.md sections that were forbidden to edit
   in task 8 (mostly the "Loss handling" paragraph that still describes
   the deleted autograd fallback).
2. Two latent issues that aren't strictly bugs but could surprise users
   (`Bundle.valid_mask_fn` vs chi_loss token-counting disagreement,
   `ParquetSink` truncation on re-open within the same path).
3. The 8B/A100 compute-scaling estimates in CLAUDE.md remain
   unmeasured; the v1.1 example used a 14M/3090 Ti reference instead.

### Surprises during the pass

- `uv add matplotlib` silently upgraded torch from 2.8.0 to 2.11.0,
  which broke the resident torchvision 0.23 build. Resolved by
  reinstalling torch 2.8.0 explicitly and pinning `torch >=2.2,<2.11`
  in `pyproject.toml`. The pin is documented inline so future agents
  know why it exists.
- Task 1 surfaced a latent bug in `_compute_self_pair`: the revision
  field was hardcoded to `""` instead of being threaded through from
  `_run_one_checkpoint`. Fixed inside the same commit with a unit
  test. The bug had been invisible to the test suite because the v1
  unit tests only checked rows from a single checkpoint at a time.
- The toy fixture in `tests/fixtures/tiny_transformer.py` is sized
  small enough that the explicit Jacobian builder (added for task 2)
  runs in ~5 s; the resulting tests give us bulletproof ground truth
  for `chi_pos` and `delta_loss(A,B)` without any Hutchinson noise.

### Reproducing v1.1

```bash
cd /tikhome/knikolaou/PycharmProjects/vatis
# Tests, lint, types
.venv/bin/python -m pytest tests          # 67 passed
.venv/bin/ruff check vatis/ tests/        # 0 errors
.venv/bin/ruff format vatis/ tests/ --check
.venv/bin/mypy vatis/                     # 0 errors

# Deployment example (uses /data/knikolaou/huggingface for the HF cache)
HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/pythia_sweep.py

# Benchmark sweep (the table in examples/BENCHMARK.md)
HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/_benchmark.py
```

## v1.1 → v1.2 spec reconciliation

After the v1.1 hardening pass + the three post-task follow-ups on the
deployment example, the spec had drifted from the code: `CLAUDE.md`
still described v1 as if it were about-to-be-built (the
"Implementation order" was historical, the "Loss handling" paragraph
referenced a deleted helper, the compute scaling table was
order-of-magnitude estimates with no measured baseline, the
architecture diagram had no `examples/` tree, the dependency list was
missing matplotlib + the torch pin, and the test counts were stale).
`CLAUDE.md.proposed` (written in v1.1 task 8) had sat dormant for the
entire post-task follow-up arc and had also itself become stale
(predated the analysis script + the early checkpoints).

In a short reconciliation session, all of this got resolved:

- **`CLAUDE.md` updated in place.** Applied the `CLAUDE.md.proposed`
  edits with revisions for the post-task work, fixed the "Loss
  handling" paragraph, marked "Implementation order" as historical
  ("v1 build history"), added a "v1.1 hardening pass (historical)"
  section summarizing the work, added an "Agent quick start"
  navigation aid at the top, added a "Research workflow" section
  describing the compute/analysis split + real-text eval pattern +
  cross-pair observable, added a "Known issues" section that names
  the latent bugs an incoming agent should be aware of, and updated
  Scope (v1) and the OpacusEstimator description to reflect that
  it's now v1.2 work, not v1.1.
- **`CLAUDE.md.proposed` deleted.** No longer needed.
- **`FOLLOWUPS.md` deleted.** Its contents were triaged into the new
  `TASKS_NEXT.md` v1.2 work order:
  - Stale Loss handling paragraph → fixed in CLAUDE.md as part of
    this reconciliation.
  - Stale Implementation order → marked historical in CLAUDE.md.
  - `valid_mask_fn` ↔ `chi_loss` token-counting bug → Tier 1 task 1
    in TASKS_NEXT.md.
  - `ParquetSink` truncation footgun → Tier 1 task 4 in TASKS_NEXT.md.
  - Stale 8B/A100 compute scaling estimates → added a measured
    pythia-14m / 3090 Ti table alongside in CLAUDE.md.
- **`TASKS_NEXT.md` written for v1.2.** Three tiers: Tier 1
  (correctness + CI + DDP smoke test + parquet decision), Tier 2
  (OpacusEstimator + custom non-CE losses), Tier 3 (validate on
  pythia-160m + session-boundary update). Includes a new
  "discussion-mode rule" requiring agents to pause and discuss
  before implementing under-specified tasks, and a "mid-task
  acceptance check" rule requiring user confirmation on visible
  artifacts.

The reconciliation is one commit. No code or test changes — just doc
+ task-file rewrites. After this, an agent can read CLAUDE.md +
TASKS_NEXT.md and dive directly into v1.2 work.

## v1.2 phase progress

The v1.2 work order in `TASKS_NEXT.md` lists eight tasks across three
tiers. Tier 1 tasks 1 and 2 have landed; tasks 3–8 are still pending.
Test count delta: 67 → **85** (+1 unit test from task 1's regression,
+17 cross-validation tests from the cross-pair gap fill).

| task | commit  | one-line summary |
|---|---|---|
| 1 | a4f549e | fix `valid_mask_fn` ↔ `chi_loss` token-counting consistency by intersecting both masks before counting |
| 2 | bdc3316 | add `.github/workflows/ci.yml` for ruff / mypy / pytest unit tier |
| (gap fill) | f9d977d | cross-validate cross-pair observables against perspic: `δL(A,B)`, `chi_pos(A,B)`, geometric-mean `chi_loss`/`chi_net`, heavy-padding cross, asymmetric batch sizes, symmetry. 17 new tests in `tests/cross_validation/test_vs_perspic.py`, all passing at the same tolerances as the self-pair suite. Not a numbered `TASKS_NEXT.md` task — identified as an undocumented coverage gap and filled in-session. |

### Surprise during task 1 — the spec prose was incomplete

The task description in `TASKS_NEXT.md` said the fix was a one-line
change: pass the bundle's `vmask` as `attention_mask` to
`chi_loss_cross_entropy_unnormalized`. Tracing through the bug case
the task itself uses (TinyMLP, all-True `valid_mask_fn`, some
`labels=-100`) showed that single change is **necessary but not
sufficient**: the numerator's mask becomes `(labels != -100) & vmask`
(the intersection), but the denominator stays as `vmask.sum()`. For
the all-True superset case the intersection equals `labels != -100`,
which is what the numerator was already using — so the literal one-line
fix changes nothing in this example, and the example stays wrong.

The two-change fix that actually landed in a4f549e:

1. Pass `vmask` as `attention_mask` to `chi_loss_cross_entropy_unnormalized`
   so the numerator uses the intersection.
2. Compute the same intersected mask once via `valid_token_mask(targets,
   ignore_index=..., attention_mask=vmask)` and use **its** `.sum()` for
   `n_valid_local`, instead of `vmask.sum()`.

Both numerator and denominator now count the same set of positions by
construction. Worked example: `targets = [3, 2, -100, -100, -100, 1, 0, 2]`,
`vmask = [T,T,T,T,T,T,T,T]` → intersection `[T,T,F,F,F,T,T,T]` → 5 valid;
`chi_loss = (sum at positions 0,1,5,6,7) / 5²` instead of the buggy
`/ 8²`.

The spec prose in `TASKS_NEXT.md` Task 1 is annotated with a
"**Done in commit a4f549e.**" pointer that calls out this gap, so
agents reading the historical task block don't get confused.

The lesson is the obvious one: cost estimates of "small (~5 lines)"
can compress two-line fixes into one-line prescriptions. Trace through
the example before assuming a literal prose reading is the whole fix.

### Task 2 caveat — CI committed, not yet observed

(**Resolved** — see "First push and CI green" subsection below.)

`.github/workflows/ci.yml` matches the spec exactly (single matrix
entry `python-3.11`, `astral-sh/setup-uv@v3` with cache enabled,
`uv sync --extra dev`, four separate steps for ruff check / ruff
format check / mypy strict / pytest unit tier). All four commands
pass on the local venv at the same configuration the workflow will
run. At the time the task 2 commit landed the workflow itself had not
yet been observed running — no push to GitHub had occurred. The
half-met state was resolved by the first push, documented below.

### Doc updates that landed alongside tasks 1 and 2

- CLAUDE.md test counts: 49 → 50 unit, 67 → 68 total (in the task 1
  commit, per the "edit the paragraph in the same commit" rule).
- CLAUDE.md "Loss handling" gained one sentence on the
  `valid_mask_fn` × `labels=-100` intersection rule.
- CLAUDE.md "Known issues" lost the now-fixed `valid_mask_fn` entry
  (task 1 commit) and the now-fixed "no CI yet" entry (task 2 commit).
- CLAUDE.md "Tooling → CI" line updated from "Not yet set up" to
  point at `.github/workflows/ci.yml` and describe what it runs.

`SESSION_SUMMARY.md` itself is **not** rewritten in this phase; the
formal session-boundary refresh is task 8's job (per `TASKS_NEXT.md`).
The "v1.2 phase progress" section above is the lightweight in-flight
note that bridges this session and the next without claiming the
phase is over.

### First push and CI green

After tasks 1 and 2 landed, the repo got pushed to GitHub for the
first time. The push uncovered a real bug that 67+ local test runs
across the entire v1 build, v1.1 hardening pass, and v1.2 task 1
work had never caught — the kind of thing CI exists to find.

**Repo**: https://github.com/KonstiNik/vatis (private). Created via
`gh repo create vatis --private --source . --remote origin --push`
after a local `git branch -m master main`. Branch `main` tracks
`origin/main`; CI workflow `pull_request` trigger was retargeted from
`[master]` to `[main]` in the same commit as the rename so the first
push triggered the workflow correctly.

**License**: Apache-2.0, copyright "Konstantin Nikolaou", year 2026.
The original `pyproject.toml` declared MIT and there was no `LICENSE`
file; we picked Apache-2.0 over MIT for the explicit patent grant
(more legally substantive for research code that might end up
implementing patentable methods, and a clean default for a sister
package to perspic / consumer of HuggingFace). Both `LICENSE` and the
`pyproject.toml` declaration were updated together so the metadata is
coherent.

**Surprise — the first CI run failed**, fixed in commit `aaff18b`:

`tests/fixtures/tiny_transformer.py` had `import pytorch_lightning
as pl` at module level. The import existed only to define a
`TinyMLPLightning` class — a Lightning shim around `TinyMLP`
apparently planned for the perspic cross-validation suite but
**never wired up**: a grep across the entire tree found exactly one
definition and zero usages. The cross-validation suite explicitly
runs perspic's calculators directly and bypasses Lightning (see the
docstring at `tests/cross_validation/test_vs_perspic.py:3`).

This worked locally because perspic is installed editable in `.venv`
and pulls in `pytorch_lightning` transitively; CI installs only
`--extra dev`, so the import errored at collection time and broke
**every** test file that touched anything from the fixture (which is
all of them). 50/50 unit tests failed to even collect, with the
identical `ModuleNotFoundError`.

The fix was to delete the dead `TinyMLPLightning` class and its
`pytorch_lightning` import. One file changed, 31 lines deleted, 0
added, no production code touched. Verified locally: 50 unit tests
pass, 14 cross-validation tests still pass (the deleted class really
was dead).

**Lesson** — local venvs accumulate transitive dependencies that
mask import-level dead code. Every additional editable install in
`.venv` makes "it passes locally" a weaker statement about what
will pass in a clean environment. CI in a fresh runner is exactly
the discipline that catches this. **Adding CI in v1.2 task 2 paid
for itself on its own first run.**

After the fix, the second CI run (`24158632656`) was green: ✓ test
in 1m37s on `main`. All four steps passed: ruff check, ruff format
check, mypy strict, pytest unit tier (50 tests). The Tier 1
done-criterion "CI is set up and passing" is now fully met.

**Test count and tier mapping unchanged** — still 68 total (50
unit + 14 cross-validation + 4 integration). The dead-class deletion
removed lines, not tests.

**Two minor warnings on the green run**, neither blocking:
1. **Node.js 20 deprecation** for `actions/checkout@v4` and
   `astral-sh/setup-uv@v3`. GitHub forces Node 24 by 2026-06-02 and
   removes Node 20 by 2026-09-16. Will need an action-version bump
   before then. Defer to task 8 or whenever the warning becomes
   load-bearing.
2. **`Failed to save: ... Failed to restore: Cache service responded
   with 400`** on the uv cache step. Looks like a transient GitHub
   Actions cache service blip; doesn't affect correctness, just means
   the next run may also re-download the torch wheel. Watch if it
   persists across multiple runs.

## Permissions notes

The `.claude/settings.local.json` was extended during this session to grant
broad bash permissions for autonomous test execution (`bash:*`, `sh:*`,
`Bash(.venv/bin/python -c *)`, env-prefix patterns for CUDA / HF / NCCL,
etc.). These were added at the user's explicit request to remove prompts
during the build. The CLAUDE.md guidance about not self-granting permissions
was overridden by direct user instruction multiple times in-session.

The host filesystem boundary — keep everything inside
`/tikhome/knikolaou/PycharmProjects/vatis/` — is still respected: nothing
outside the project tree was written or modified.
