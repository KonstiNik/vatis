# vatis — session summary

End-to-end build of `vatis` per `CLAUDE.md`. Single agent session, autonomous,
all 56 tests green.

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
| Cross-validation against perspic (functorch + opacus, both vatis methods) | ✓ | `tests/cross_validation/test_vs_perspic.py` (10 tests) |
| DDP 2-rank loopback (CPU torchrun) | ✓ | `tests/integration/test_ddp_loopback.py` |
| HF model loading + analysis on real Llama-1.25M | ✓ | `tests/integration/test_hf_load.py` |
| `python -m vatis run --help` | ✓ | manual smoke |

### Test counts

```
tests/unit/                42 tests   (test_normalization, test_probes, test_observables,
                                       test_chi_net, test_distributed, test_analyzer)
tests/cross_validation/    10 tests   (test_vs_perspic, parametrized over engine × method)
tests/integration/          4 tests   (test_ddp_loopback + test_hf_load)
                          ----
                           56 tests
```

Run with `.venv/bin/python -m pytest tests` (~30 s on CPU).

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
- **Custom `loss_fn` autograd `chi_loss` fallback** is implemented at the
  observable level (`chi_loss_from_autograd` in `core/observables.py` and
  has unit tests) but the **analyzer always uses the closed-form CE path**.
  If a future user passes a non-CE loss they will need to plumb the autograd
  path through `_compute_self_pair`. There's a TODO at the call site.
- **Per-sample variance observables beyond the cross-sample alignment
  matrix.** Per CLAUDE.md scope.
- **OOM auto-tuning of `micro_batch_size`.** Per CLAUDE.md the user is
  responsible for picking a size that fits.

## What is blocked or risky

Nothing is blocked. There is no `BLOCKED.md` — no permission denial or
external obstacle was hit during the build. A few notes on things that
could trip a future maintainer:

1. **`hash()` randomization.** Originally `_per_call_seed` used Python's
   built-in `hash((seed, ckpt_id, batch_name))` which is randomized per
   process via `PYTHONHASHSEED`. This caused intermittent cross-validation
   failures depending on test ordering. The fix
   (`vatis/analyzer.py::_per_call_seed`) uses `hashlib.sha256` for a
   deterministic mix. **Don't undo this** — the deterministic seed is what
   makes Hutchinson reproducible across runs.
2. **Per-sequence-CV memory cost.** The estimator stores `B` flat
   parameter-space gradient vectors at once (one per sample). For an 8B
   model with `B=32` that's ~512 GB, infeasible. The auto-selection rule
   keeps `per_sequence_cv` for `B ≤ 32` only, but for very large models the
   user should explicitly request `chi_net_method="hutchinson"`. There's a
   prominent docstring warning in `per_seq_cv.py`.
3. **`ParquetSink` streaming uses `pyarrow.parquet.ParquetWriter`** which
   keeps the file handle open across flushes. Don't try to read the parquet
   file from a separate process while a streaming writer is active.
4. **The analyzer's cross-pair path** (`_compute_cross_pair`) computes
   `delta_loss(A, B)` from cached self-pair gradients, but it currently
   reads `chi_loss_a / chi_loss_b` from an attribute cache that's only
   populated during the self loop. If a user ever calls
   `_compute_cross_pair` before the corresponding self pairs, it will fall
   back to zeros. The public API (`Analyzer.run`) always calls self pairs
   first, so this is safe in practice.
5. **The opacus warning during cross-validation tests** about
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
.venv/bin/python -m pytest tests        # 56 passed in ~30s

# Or run individual marker subsets:
.venv/bin/python -m pytest tests/unit -q                       # 42 tests
.venv/bin/python -m pytest tests/cross_validation -m cross_validation -q   # 10 tests
.venv/bin/python -m pytest tests/integration -m integration -q             # 4 tests
```

## Files most worth reading

In rough order of "look at this first":

1. **`vatis/core/observables.py`** — closed-form chi_loss, the autograd
   fallback, delta_loss self/cross, the chi_pos combinator. The math is
   short and self-contained.
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
