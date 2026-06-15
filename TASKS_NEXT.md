# vatis — v1.2 work order

## Context

Read `SESSION_SUMMARY.md` first — it is the authoritative state of the
repo at the start of this phase. Then read `CLAUDE.md` (especially the
"Agent quick start" and "Known issues" sections). This file
(`TASKS_NEXT.md`) is the work order for the v1.2 phase.

The v1 build is complete and the v1.1 hardening pass is complete. 67
tests pass. The deployment example runs end-to-end on real Pythia
checkpoints in ~20 s (warm cache) and produces meaningful trajectories.

**This phase is about closing the correctness gap, getting CI in
place, and then implementing the OpacusEstimator (the v1 spec's
"v1.1 deferred" feature work).** Three tiers, in priority order. Do
**not** skip Tier 1 to get to Tier 2; Tier 1 contains a real
correctness bug.

## Hard constraints

- **Venv only.** Use `.venv/bin/python`, `.venv/bin/pytest`, `uv add`
  for new deps. Never `pip install` globally. Verify with `which
  python` before installs.
- **Single GPU.** Whatever GPU is visible to the session is all you
  get. Do not assume multi-GPU. Do not write code paths you can't
  actually run. (One Tier 1 task is "test the DDP path on real GPUs"
  — that one is gated on GPU availability and may have to be skipped
  with a `BLOCKED.md` note if the session runs single-GPU.)
- **HuggingFace cache goes to `/data/knikolaou/huggingface`.** Models
  and datasets are large. Set `HF_HOME=/data/knikolaou/huggingface`
  (and `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` to the same root) at
  the top of every script and example, and export it in your shell
  before running anything that calls `from_pretrained` or
  `load_dataset`. **Do not** let HF default to `~/.cache/huggingface`
  — the home filesystem will fill up.
- **No edits to `settings.local.json`.** The allow list is the user's
  contract. If something is blocked, log it to `BLOCKED.md` and skip.
- **Filesystem boundary.** Everything stays inside
  `/tikhome/knikolaou/PycharmProjects/vatis/`, with the single
  explicit exception of `/data/knikolaou/huggingface` for the HF
  cache. Do not write anywhere else, even via Python `open()`.
- **All tests must stay green at the end of every task.** Run the
  full suite (`.venv/bin/python -m pytest tests`) before each commit.
  If a task breaks tests, fix it before committing or revert.
- **The perspic cross-validation suite is the safety net.** It must
  stay passing. **If a change you make requires loosening a
  cross-validation tolerance, that is a red flag — investigate the
  underlying disagreement before loosening.** Do not loosen tolerances
  to make a test pass.

## Discussion-mode rule (new in v1.2)

The v1.1 phase was prescriptive ("do task N, commit, move on") and
that worked because each task was well-defined. v1.2 has tasks where
the design isn't fully nailed down (custom-loss plumbing, OpacusEstimator
internals, the parquet append story). For those tasks, **pause and
write a discussion message before implementing**. Do not autonomously
make architectural decisions on under-specified tasks. The user would
rather have a 30-second pause to confirm than a 2-hour rewrite.

Specific triggers for "stop and discuss":
- The task says "decide between option A and option B."
- The CLAUDE.md spec doesn't fully specify the API for the thing you're
  building.
- You're about to refactor a file outside the immediate task scope.
- You're about to add a new top-level module or directory.
- You discover an issue while implementing that wasn't in the task
  description.

For tightly-specified tasks (e.g. "fix the `valid_mask_fn` consistency
bug — the change is one line plus a regression test"), implement
directly.

## Mid-task acceptance checks

For any task that produces a user-visible artifact (a plot, a parquet,
a printed table), **show it to the user before declaring the task
done**. The v1.1 task 1 example would have benefited from a "look at
this plot, is it what you wanted" check before commit; instead it
needed three follow-up commits to get right. Future tasks should bake
in the show-and-confirm step explicitly.

## Git workflow (prescriptive — follow exactly)

Same as v1.1:

- **After each numbered task below:** stage only the files that task
  touched (`git add <specific paths>`, never `git add -A`). Commit
  with message `task N: <one-line summary>`. One commit per task. No
  squashing, no amending, no rebasing.
- **Generated artifacts** (parquet files, plots, BENCHMARK.md numbers)
  belong in the same commit as the code that produced them.
- **Never** run `git reset --hard`, `git rebase`, `git push`,
  `git checkout <branch>`, `git branch -D`, `git clean -f`, or
  `--no-verify`. If you think you need any of these, stop and write
  to `BLOCKED.md` instead.
- **If a task fails partway:** commit what works as
  `task N (partial): <summary>`, note the gap in `SESSION_SUMMARY.md`,
  and move on to the next independent task. Do not block the whole
  session on one failure.
- **Sanity-check after every commit:** `git status` should be clean,
  `git log --oneline` should show your commit at the top.

## Recommended order

`Tier 1 → Tier 2 → Tier 3`. Within each tier the tasks are roughly
independent; do them in the order listed unless one blocks.

## Tier 1 — correctness and infrastructure (do first, do not skip)

### Task 1 — Fix `valid_mask_fn` ↔ `chi_loss` token-counting disagreement

**Done in commit a4f549e.** The literal prose below describes a one-line
fix (pass `vmask` as `attention_mask`); the as-implemented fix is two
lines (also update `n_valid_local` to count the same intersected mask).
The single-line version is necessary but not sufficient — it leaves the
denominator using `vmask.sum()`, so the example this task itself uses
(MLP, all-True vmask, some `labels=-100`) is still wrong after it. See
`SESSION_SUMMARY.md` "v1.2 phase progress" for the gap analysis.

**The bug.** The analyzer's `chi_loss` accumulator uses
`_extract_targets(micro)` and `chi_loss_cross_entropy_unnormalized`,
which check `valid_token_mask(targets, ignore_index=...)` internally
— i.e. they honor `labels=-100` but **not** the bundle's
`valid_mask_fn`. The `n_valid` count for normalization, on the other
hand, comes from `bundle.valid_mask_fn(micro, logits).sum()`. If a
user supplies a `valid_mask_fn` that disagrees with the
labels-vs-ignore_index convention (e.g. all-True for an MLP with
`-100` labels), `chi_loss` and the `n_valid` count will use different
masks and the normalization will be silently wrong.

**The fix.** Make the analyzer pass the `valid_mask_fn`'s output as
the `attention_mask` argument to
`chi_loss_cross_entropy_unnormalized`, so the two paths agree by
construction. Specifically:

1. In `vatis/core/observables.py::chi_loss_cross_entropy_unnormalized`,
   the `attention_mask` parameter already exists — verify it's wired
   into `valid_token_mask` correctly.
2. In `vatis/analyzer.py::_compute_self_pair`, where the chi_loss
   accumulator loop runs, compute `vmask = bundle.valid_mask_fn(...)`
   once and pass it as the explicit `attention_mask` to
   `chi_loss_cross_entropy_unnormalized` (in addition to the
   labels-derived mask that's already in there).
3. Add a regression test to `tests/unit/test_analyzer.py` that builds
   a bundle whose `valid_mask_fn` disagrees with `labels != -100`
   (e.g. an MLP bundle with all-True valid mask but a few `-100`
   labels) and asserts that vatis's chi_loss matches a manual
   computation that respects the user's `valid_mask_fn` choice.
4. Also re-run `tests/cross_validation/test_vs_perspic.py` to make
   sure the existing tests still pass — the heavy-padding test
   already exercises this path with a custom `valid_mask_fn`, but it
   was sidestepping the bug rather than testing the fixed path.

**Cost estimate:** ~5–15 lines of code, ~30 lines of test, ~30 min.

**Acceptance:** the new regression test passes; all existing tests
still pass; a quick test with an MLP + `-100` labels + all-True
valid_mask_fn produces the same chi_loss as the same MLP with the
3 valid samples extracted.

Commit: `task 1: fix valid_mask_fn / chi_loss token-counting consistency`.

### Task 2 — Add CI

**Done in commit bdc3316.** `.github/workflows/ci.yml` matches the
spec below; the workflow has not been observed running yet because no
push to GitHub has occurred since the commit landed.

There is no CI yet. Add `.github/workflows/ci.yml` that runs:

- `ruff check vatis/ tests/`
- `ruff format vatis/ tests/ --check`
- `mypy vatis/`
- `pytest tests -m "not integration and not cross_validation"`

Single matrix entry: `python-3.11`. Use `uv` to install dependencies
(matches the local development workflow). The cross-validation tier
is skipped because perspic isn't easily installable in CI; the
integration tier is skipped because it needs HF cache and a real GPU.

The workflow file should:
- Trigger on `push` to any branch and `pull_request` to `master`.
- Use `actions/checkout@v4` and `astral-sh/setup-uv@v3` (or current
  pinned versions — check the official docs).
- Install vatis in dev mode with `uv sync --extra dev`.
- Run each of the 4 commands above as a separate step so failures are
  attributable.

**Discussion-mode trigger:** if you find that the CI environment is
significantly different from what the local workflow expects (e.g. a
torch wheel that doesn't match the local pin), pause and write a
discussion message. Do not silently widen the version pin to make CI
pass.

Commit: `task 2: add GitHub Actions CI for ruff, mypy, and unit tests`.

### Task 3 — Multi-GPU DDP smoke test on real GPUs

**Substantially DONE 2026-06-15 (as a benchmark, not yet a gated pytest test).**
Verified on 4× A100-SXM4-40GB via `examples/benchmark/ddp_scaling/` on
pythia-1.4b: multi-GPU == single-GPU (chi_loss/delta_loss exact, chi_net/chi_pos
within Hutchinson noise — the "Correctness" section of `DDP_SCALING_RESULTS.md`),
strong scaling 3.86× on 4 GPUs. The DDP all-reduce algebra is now validated on
real GPUs, closing this gap. **Remaining (optional):** promote it into a gated
`tests/integration/test_ddp_real_gpus.py` (`@pytest.mark.integration` +
`requires_multi_gpu`) as the task originally specified, so it lives in the suite
rather than only as a benchmark. Cluster notes for whoever does it: partition
`alpha`, `--gres=gpu:N` (not `gpu:A100:N`), `--nodes=1`, ≤6 cpus/GPU, and
`unset CUDA_VISIBLE_DEVICES` in the job so torchrun assigns one GPU per rank.

The DDP path is tested on a 2-rank CPU loopback only
(`tests/integration/test_ddp_loopback.py`). It has never run on
actual GPUs. **This is gated on GPU availability** — if the session
runs on a single-GPU box, log to `BLOCKED.md` and skip.

If multiple GPUs are available:

1. Add `tests/integration/test_ddp_real_gpus.py` that uses `torchrun`
   to launch a 2-process DDP run on two CUDA devices, runs vatis on
   `pythia-14m@step3000` with `B=4`, asserts the result matches a
   single-GPU run on the same model + batch + seed (within
   Hutchinson noise).
2. Mark it `@pytest.mark.integration` and `@pytest.mark.requires_multi_gpu`
   (define the marker in `pyproject.toml`'s `markers` list).
3. Document in the test docstring how to run it (`CUDA_VISIBLE_DEVICES=0,1
   .venv/bin/python -m pytest tests/integration/test_ddp_real_gpus.py
   -m integration`).

If only one GPU is available, write a `BLOCKED.md` entry like:

```
## Task 3 (Tier 1) — Multi-GPU DDP smoke test

Blocked: only one GPU available in this session
(`CUDA_VISIBLE_DEVICES=0`, single 24 GB RTX 3090 Ti per `nvidia-smi`).

The DDP path needs at least 2 GPUs to exercise. The CPU loopback test
(`tests/integration/test_ddp_loopback.py`) is the closest thing we
have, and it does pass.

What the user needs to do to unblock: run this session on a
multi-GPU box, or skip the task. The blocker doesn't affect any
other Tier 1 or Tier 2 task.
```

Commit (only if not blocked): `task 3: real-GPU DDP smoke test on pythia-14m`.

### Task 4 — Decide and document the `ParquetSink` re-open story

**Discussion-mode trigger.** This task has two valid resolutions and
the user should pick. Write a discussion message before implementing,
laying out:

- **Option A: fix.** Make `ParquetSink(path)` open in append mode if
  the file already exists and the schema matches. This is a real
  feature add — it changes the semantics of the constructor and
  needs a test for the schema-mismatch failure mode. ~50 lines of
  code + ~30 lines of test.
- **Option B: document and assert.** Leave the truncation behavior
  as-is, but raise a clear `RuntimeError` in `ParquetSink.__init__`
  if the file already exists, telling the user to either delete the
  file first or use a single `analyze()` call with multiple
  `revisions`. The error message should point at the deployment
  example as the canonical pattern. ~10 lines of code + ~10 lines
  of test.

Option B is cheaper and matches the existing example pattern. Option
A is more robust but introduces a constructor flag (`mode="append"`
or similar) that the example doesn't need.

**My recommendation if you have to pick without discussion:** Option
B. Don't expand the API surface for a single-script convenience that
the example already sidesteps.

After the user picks, implement, test, and document the choice in
`CLAUDE.md` under "Result format".

Commit: `task 4: parquet sink re-open behavior (option A|B)`.

### Task 9 — `per_sequence_cv` O(P)-memory reformulation (added 2026-06-15)

**DONE 2026-06-15.** The default path is now O(P) (output-space projection,
no per-sample grad storage); the O(B*P) storing path + the cross-sample
alignment matrix + the memory pre-check all live behind
`compute_alignment_matrix=True`. Offloading was deliberately NOT implemented
(see "Related" below — still its own future task). Verified: O(P)↔storing
exact-equivalence and hutchinson↔per_seq_cv convergence unit tests, plus the
perspic cross-validation tier; ruff + mypy-strict clean.

**Tier 1 priority. Claim verified numerically this session** — see
`examples/benchmark/_verify_per_seq_cv_reformulation.py` (subagent check on the
TinyTransformer fixture: chi_net rel diff **1.34e-8**, per-probe `grad_v_perp`
vector rel diff **2.6e-7** — an *exact* algebraic identity, not an
approximation).

**The problem.** `PerSequenceControlVariateEstimator` caches **B per-sample
parameter-gradient vectors** `g_b` (each size `P = n_params`, fp32) → peak
memory **O(B·P)**. This is the binding constraint at scale: the memory
pre-check refuses it for 1B+ models (e.g. pythia-1.4b, B=16 → ~90 GB of cache),
and it's what blocks the 9B/DPO target on the 40 GB A100s. The single-GPU
benchmark shows it directly: hutchinson peak is flat (~4.6 GB on 160m) while
per_seq_cv climbs linearly in B (10.8 → 15.8 → 26.4 GB at B=4/8/16). See
`examples/benchmark/results_pythia-160m_plot.png`.

**The fix.** The control-variate correction is
`Σ_b coef_b·g_b = Σ_b coef_b·Jᵀu_b = Jᵀ(Σ_b coef_b·u_b) = Jᵀ(P v)`, where `P`
projects the probe `v` onto the per-sample loss directions **in output space**.
Therefore `grad_v_perp = Jᵀv − Jᵀ(Pv) = Jᵀ((I−P)v)`. So:
1. Per probe, form `w = (I−P)v` in **output space** (per-sample subtract the
   component along `û_b`; cost O(S·V) per sample, *not* O(P)) and take **one**
   backward of `(logits·w).sum()` → `grad_v_perp` directly. No stored `g_b`.
2. The exact term `Σ_b ‖g_b‖²/‖u_b‖²` still needs the B per-sample backwards,
   but only their scalar squared norms — compute `‖g_b‖²`, accumulate, `del g_b`.
   Backward *count* is unchanged (`B + n_h`); memory drops **O(B·P) → O(P)**,
   matching hutchinson.

**Caveat — the one thing it gives up.** The cross-sample alignment matrix bonus
(`extras["alignment_matrix"]`, `C_{bb'} = ⟨g_b, g_{b'}⟩`) genuinely needs the
stored `g_b`. Keep it behind the existing `compute_alignment_matrix` flag:
default to the O(P) path; only materialize/store `g_b` when the alignment matrix
is explicitly requested (and even then, consider streaming them to CPU — see the
related item below).

**Related (broader theme):** the same "stop holding full-`P` vectors on the GPU"
issue affects `delta_loss`. **Partly addressed 2026-06-15 (commit `9c8cbff`):**
the analyzer used to square/dot those full-`P` vectors via `.to(float64)`,
materializing a P·8-byte fp64 *copy* (~10.5 GiB at 1.4B; ~21 GiB for the
cross-pair dot) — that's fixed with chunked-fp64 reductions, and 1.4b now runs
(peak ~16 GiB on a 40 GB card). **Still open (own task — the path to 9B):** the
fp32 flat gradient itself (`P·4` = 5.3 GiB at 1.4B, 36 GiB at 9B; built in
`_compute_self_pair` and cached per eval batch for cross-pairs, all on-device)
is the remaining wall. Two options: (a) compute `delta_loss` *per parameter*
(no full-`P` flat vector at all, like the chi_net estimators) — gives up
cross-pair gradient-cache reuse; or (b) CPU-offload the cached flat grads (host
has ~1 TB). Either, plus the DPO `loss_fn` work (Tier 2 Task 6), is what
unblocks 9B + DPO on the 40 GB A100s.

**Cost estimate:** ~40–60 lines in `per_seq_cv.py` + a regression test (promote
the verification script to a unit test asserting the O(P) path equals the
current path within 1e-5 on the toy fixture). ~half a day.

**Acceptance:**
- New O(P) path returns chi_net equal to the current implementation within
  Hutchinson-free fp tolerance (rel < 1e-5) on the toy fixture, probe-for-probe.
- Peak number of P-sized vectors alive is 1 (assert, not B).
- Alignment matrix still correct when `compute_alignment_matrix=True`.
- `tests/cross_validation/test_vs_perspic.py` still passes (per_seq_cv path).

Commit: `task 9: O(P)-memory per_sequence_cv via output-space projection`.

## Tier 2 — feature work (do after Tier 1 is green)

### Task 5 — `OpacusEstimator` full implementation

This is the original "v1.1 deferred" work, now a v1.2 task. The stub
in `vatis/core/chi_net/opacus.py` raises `NotImplementedError`; the
goal is to replace it with a working estimator that mirrors
`perspic/calculator/samplewise_opacus.py`.

**Scope (per CLAUDE.md `## Method 3: OpacusEstimator`):**

- Layer-compatibility check at startup: enumerate the model's
  modules, check each against opacus's supported layer set, raise a
  clear error if anything is unsupported.
- Tied-embedding detection: opacus can't handle parameter tying
  (Pythia tied embeddings break it). Detect via `param.data_ptr()`
  comparison and raise a clear error pointing the user at
  `chi_net_method="hutchinson"`.
- In-place op neutralization: opacus can't handle in-place activations
  (e.g. `inplace=True` ReLU). Either neutralize them (set
  `inplace=False`) or detect and raise.
- Per-sample gradient computation via opacus's `GradSampleModule` +
  ghost clipping. Hutchinson over `(S, V)` only.
- Cross-validation against perspic's `samplewise_opacus.py` backend
  in `tests/cross_validation/test_vs_perspic.py`. The pattern from
  the existing parametrized tests should drop in.

**Discussion-mode trigger.** This is a substantial, design-heavy
task. Before starting, write a discussion message that:

1. Lists the perspic functions you plan to mirror (file + function
   name) and confirms you've actually read them.
2. Lays out the proposed API: what does the estimator's `compute()`
   look like, what does the layer-compat check return, how does the
   tied-embedding error message look.
3. Lists the test plan: which cross-validation tests will be added,
   which existing tests need updating, what the expected wallclock
   on `pythia-14m` is.

**Acceptance:**
- All existing tests still pass.
- New cross-validation tests against perspic's opacus backend pass at
  the same tolerances as the existing functorch tests.
- The estimator runs end-to-end on `pythia-14m` (which has *untied*
  embeddings on this small variant — verified during v1.1) and
  produces a chi_net value within Hutchinson noise of the
  `hutchinson` and `per_seq_cv` results on the same batch.
- The estimator raises a clear, actionable error on a tied-embedding
  model (test on a manually-constructed toy model with shared
  embedding/lm_head weights).

Commit: `task 5: implement OpacusEstimator with layer-compat check`.

### Task 6 — Custom (non-CE) `loss_fn` for chi_loss

`vatis/core/observables.py::chi_loss_cross_entropy_unnormalized` is
the only chi_loss path. The bundle's `loss_fn` is used for
`delta_loss` but ignored by chi_loss. Add a path that uses the user's
`loss_fn` to compute chi_loss via one extra `torch.autograd.grad(loss,
logits)` call (one cheap backward through the loss head only).

**Discussion-mode trigger.** Before implementing, decide:

- Is this a separate flag on the bundle (`bundle.loss_kind="cross_entropy"`
  vs `"custom"`)? Or auto-detected from the loss_fn?
- Where does the autograd path live — back in `core/observables.py`
  (resurrecting the v1.1-deleted `chi_loss_from_autograd`) or as a
  new method on the bundle?
- How does the micro-batch invariant work? The closed-form path
  accumulates an unnormalized sum and divides by `N_total²` once;
  the autograd path needs the same trick.

Write a discussion message proposing an answer to each before
coding.

**Acceptance:**
- A new end-to-end test using a non-CE loss (label-smoothed CE on
  the toy MLP, or MSE on logits) that runs `analyze()` and produces
  a finite, reproducible chi_loss / chi_net / chi_pos.
- The closed-form CE path remains unchanged and bit-identical.
- The "Loss handling" paragraph in `CLAUDE.md` is updated to remove
  the v1.1 caveat.

Commit: `task 6: custom non-CE loss_fn support for chi_loss`.

## Tier 3 — polish (do if there's time)

### Task 7 — Validate the deployment example on a bigger model

The current example uses `pythia-14m` (~14M params). It would be
useful to validate that vatis works on a 10×-larger model and to
add a measured row to the compute scaling table for it.

Suggested model: `pythia-160m`. ~10× more parameters, still loads in
seconds, fits comfortably in 24 GB GPU memory at `B=8`. Per-call
wallclock should be ~5–10× the 14m numbers (per CLAUDE.md compute
scaling estimate). Total sweep wallclock: probably ~3–5 min.

Discussion-mode trigger. Before running:

- Verify `pythia-160m` has the same checkpoint family
  (`step1, step8, ..., step143000`) as `pythia-14m`. Use `huggingface_hub`
  to list refs.
- Check whether `pythia-160m` has *tied* embeddings (the bigger Pythia
  models do, AFAIK). If yes, this validates that the
  `hutchinson` and `per_sequence_cv` paths work with tied
  embeddings (they should — only opacus cares).

How to do this without regenerating `examples/pythia_sweep.py`'s
artifacts:

1. Add a constant at the top of the script that picks the model name,
   and a CLI argument to override it.
2. Run the script with `--model EleutherAI/pythia-160m`, write the
   output to `examples/results_160m.parquet` and a separate set of
   plots.
3. Add a section to `examples/BENCHMARK.md` with the measured 160m
   numbers. **Do not delete the 14m numbers** — keep both as
   reference points.

**Acceptance:**
- The example runs end-to-end on pythia-160m within 15 min.
- The chi_net U-shape and chi_pos peak are checked on the bigger
  model. (If they reproduce, that's interesting; if they don't,
  even more so.)

Commit: `task 7: deployment example also runs on pythia-160m`.

### Task 8 — Update SESSION_SUMMARY.md and prep TASKS_NEXT.md for v1.3

Same "session boundary" pattern as v1.1:

- Update `SESSION_SUMMARY.md` with the v1.2 work. Bump test counts.
  Add a "v1.2 phase" section. Update the "Known issues" list (remove
  fixed items).
- Write `TASKS_NEXT.md` for v1.3. Even if there's no agreed scope
  yet, draft a triage of remaining `## Known issues` items and any
  new items the v1.2 work surfaced.
- Verify `git log --oneline` shows the expected commit history.
- Verify `git status` is clean.

Commit: `task 8: session-boundary update for end of v1.2 phase`.

## Done criteria

When you finish (or run out of independent tasks):

1. All tests still green: `.venv/bin/python -m pytest tests` shows
   the new total, all passing.
2. CI is set up and passing on the master branch (Tier 1 task 2).
3. The `valid_mask_fn` ↔ `chi_loss` consistency bug is fixed and has
   a regression test (Tier 1 task 1).
4. `git log --oneline` shows one commit per task, in order.
5. `git status` is clean.
6. `SESSION_SUMMARY.md` is updated and `TASKS_NEXT.md` for v1.3 is
   drafted.
7. If anything was skipped or blocked, `BLOCKED.md` exists at the
   project root with the exact obstacle.

That's it. Stop when done. Do not start new work beyond this list
without writing it to `TASKS_NEXT.md` for v1.3 first.
