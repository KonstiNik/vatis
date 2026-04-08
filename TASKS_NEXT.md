# vatis — v1.1 hardening pass

## Context

Read `SESSION_SUMMARY.md` first — it is the authoritative state of the
repo as of the end of the build session. This file (`TASKS_NEXT.md`) is
the work order for the next session.

The v1 build is complete and 56 tests pass. This pass is about *trust*
and *demonstrability*, not new features. Do not re-litigate v1 scope.
Do not add features that aren't in the task list below. If you think
something else needs doing, write it to a `FOLLOWUPS.md` file and keep
moving.

## Hard constraints

- **Venv only.** Use `.venv/bin/python`, `.venv/bin/pytest`, `uv add` for
  new deps. Never `pip install` globally. Verify with `which python`
  before installs.
- **Single GPU.** Whatever GPU is visible to the session is all you get.
  Do not assume multi-GPU. Do not write code paths you can't actually run.
- **HuggingFace cache goes to `/data/knikolaou/huggingface`.** Models and
  datasets are large. Set `HF_HOME=/data/knikolaou/huggingface` (and
  `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` to the same root) at the top
  of every script and example, and export it in your shell before running
  anything that calls `from_pretrained` or `load_dataset`. **Do not** let
  HF default to `~/.cache/huggingface` — the home filesystem will fill up.
  If `/data/knikolaou/huggingface` does not exist, create it once with
  `mkdir -p` and proceed.
- **No edits to `settings.local.json`.** The allow list is the user's
  contract. If something is blocked, log it to `BLOCKED.md` and skip.
- **Filesystem boundary.** Everything stays inside
  `/tikhome/knikolaou/PycharmProjects/vatis/`, with the single explicit
  exception of `/data/knikolaou/huggingface` for the HF cache. Do not
  write anywhere else, even via Python `open()`.
- **All tests must stay green at the end of every task.** Run the full
  suite (`.venv/bin/python -m pytest tests`) before each commit. If a
  task breaks tests, fix it before committing or revert.

## Git workflow (prescriptive — follow exactly)

- **Task 0, before anything else:** if the repo is not yet a git repo,
  `git init`. Then `git add -A && git commit -m "snapshot: end of build
  session"` to baseline the entire current tree. This baseline is what
  makes every subsequent task individually revertable. Do this *first*,
  before reading any other file or running any test.
- **After each numbered task below:** stage only the files that task
  touched (`git add <specific paths>`, never `git add -A` after the
  baseline). Commit with message `task N: <one-line summary>`. One commit
  per task. No squashing, no amending, no rebasing.
- **Generated artifacts** (parquet files, plots, BENCHMARK.md numbers)
  belong in the same commit as the code that produced them.
- **Never** run `git reset --hard`, `git rebase`, `git push`,
  `git checkout <branch>`, `git branch -D`, `git clean -f`, or
  `--no-verify`. If you think you need any of these, stop and write to
  `BLOCKED.md` instead.
- **If a task fails partway:** commit what works as
  `task N (partial): <summary>`, note the gap in `SESSION_SUMMARY.md`,
  and move on to the next independent task. Do not block the whole
  session on one failure.
- **Sanity-check after every commit:** `git status` should be clean,
  `git log --oneline` should show your commit at the top. If not, stop
  and diagnose before continuing.

## Independence map

Tasks 1–7 are independent of each other. If any of them blocks, skip it
and move on. Task 8 (CLAUDE.md proposal) must run last because it
references the work done in 1–7.

Recommended order: 0 → 5 → 6 → 7 → 2 → 3 → 4 → 1 → 8. Rationale: cheap
footguns first (5–7), then math hardening (2–4), then the expensive
example (1), then the writeup (8). Reorder freely if you hit a block.

## Tasks

### Task 0 — Baseline commit

`git init` if needed, then baseline-commit the entire current tree.
Verify with `git log --oneline` that exactly one commit exists. No code
changes in this task.

### Task 1 — Deployment example with benchmark character

**Goal:** produce a runnable example on the single available GPU that a
human (or future agent) can look at and say "yes, χ_pos behaves the way
the theory predicts on a real checkpoint sweep." This example also
serves as the project's reference benchmark.

**Hard runtime budget:** the final example must complete end-to-end in
**under 15 minutes** on the single available GPU, *including model
loading from the HF cache* (assume cache is cold the first time, warm
after). The benchmark exploration phase has its own separate budget
(see below).

**Procedure — do this in order, do not skip steps:**

1. **One-shot probe.** Pick a small published causal LM that can plausibly
   show a non-trivial χ_pos trajectory across revisions. `pythia-14m` is
   the safe default; `pythia-70m` or `pythia-160m` are acceptable if the
   probe shows headroom. Ensure `HF_HOME=/data/knikolaou/huggingface` is
   set. Load one revision, run a single `analyze()` call with `B=8`,
   `n_hutchinson=16`, both `chi_net_method` values. Record:
   - wallclock for model load
   - wallclock per `analyze()` call (per method)
   - peak GPU memory (`torch.cuda.max_memory_allocated()`)
   - the observable values themselves (sanity check they're finite and
     non-zero)

   Write these numbers to `examples/BENCHMARK.md` under a section called
   "One-shot probe". **If the per-call wallclock alone exceeds 2 minutes,
   drop to a smaller model.** Do not try to rescue the budget by shrinking
   the sweep — the example needs a non-trivial trajectory.

2. **Compute the budget arithmetic explicitly.** In `BENCHMARK.md`, write
   out:
   ```
   budget = 15 min - model_load - 30% headroom
   per_call = max(hutch_wallclock, cv_wallclock)
   max_calls = budget / per_call
   max_calls = revisions × eval_batches × methods
   ```
   Show the actual numbers. Pick `revisions`, `eval_batches`, `methods`
   from this arithmetic. **Show your work** — if a future maintainer
   wants to scale this up they need to see how you sized it.

3. **Benchmark sweep.** Before writing the example, run a benchmark
   sweep covering at least:
   - both `chi_net_method` values (`hutchinson`, `per_sequence_cv`)
   - at least 3 `n_hutchinson` values spanning ~8× (e.g. 8, 32, 128) to
     show the variance vs cost tradeoff
   - at least 2 batch sizes

   Extend further if time allows. **Hard ceiling: 20 minutes total
   wallclock** for the benchmark phase (separate from the 15-minute
   example budget). Record results as a table in `BENCHMARK.md` with
   columns: method, n_hutchinson, batch_size, wallclock_s, peak_mem_mb,
   chi_loss, chi_net, chi_pos, delta_loss. The goal is that a future
   user can read this table and pick sensible parameters for their own
   setup without re-running anything.

4. **Write `examples/pythia_sweep.py`.** Single file, runnable as
   `.venv/bin/python examples/pythia_sweep.py`. It should:
   - Set `HF_HOME` at the top.
   - Load the chosen model across the chosen revisions.
   - Run `analyze()` with the parameters picked in step 2.
   - Write `examples/results.parquet`.
   - Produce 3 plots (`examples/chi_loss.png`, `examples/chi_net.png`,
     `examples/chi_pos.png`) showing each observable vs training step,
     one line per eval batch. Use matplotlib; add it via `uv add
     matplotlib` if not already present.
   - Print a one-paragraph summary at the end (wallclock, what was run,
     where outputs went).

5. **Commit everything in one commit:** `examples/pythia_sweep.py`,
   `examples/BENCHMARK.md`, `examples/results.parquet`, the three PNGs.
   Commit message: `task 1: deployment example + benchmark on
   <model-name>`.

**If the probe step (1) blocks on download or OOM:** drop to a smaller
model first. If even `pythia-14m` doesn't work, write the obstacle to
`BLOCKED.md` and skip this task — don't fake it with a synthetic model,
the whole point is real weights.

### Task 2 — Exact-NTK ground-truth tests

The unit suite already has `_exact_chi_net` in `tests/unit/test_chi_net.py`
that iterates over output dims to compute `Tr(M_b)` exactly on the toy
transformer. Extend the same primitive to validate `chi_pos` and
`delta_loss(A, B)`.

- Add `test_chi_pos_matches_exact_ntk` in `tests/unit/test_observables.py`
  (or `test_chi_net.py`, wherever the exact NTK helper lives). Build the
  full per-sample Jacobian `J_b ∈ R^{(S·V)×P}` on the toy fixture, form
  `Θ = J Jᵀ`, compute the loss-direction Rayleigh quotient
  `(uᵀ Θ u) / (uᵀ u × Tr Θ)` where `u = ∇_f L`, and assert vatis's
  `chi_pos` matches to relative tolerance `1e-5`. Use the smallest toy
  fixture that makes this tractable (`S * V * B` should stay under
  ~2048 — shrink the fixture if needed).
- Add `test_delta_loss_cross_matches_exact_ntk_jacobian_product` that
  computes `δL(A, B)` two ways: (a) the gradient dot product vatis
  uses, (b) `uᴬᵀ Jᴬ Jᴮᵀ uᴮ` from the explicit Jacobians. Assert
  agreement to relative tolerance `1e-5`.
- Both tests live under `tests/unit/` so they run in CI.

Commit: `task 2: exact-NTK ground-truth tests for chi_pos and
delta_loss`.

### Task 3 — Tight-tolerance perspic cross-check

The current cross-validation uses `~2%` tolerance, which hides any
constant-factor bias smaller than the Hutchinson noise floor. Add **one**
new test in `tests/cross_validation/test_vs_perspic.py`:

- Toy model only (don't blow up CI time).
- `n_hutchinson=16384`.
- Tolerance: relative `0.3%`, absolute `1e-6`.
- Both vatis methods (`hutchinson`, `per_sequence_cv`), one perspic
  backend is enough (functorch — it's the cleaner reference).
- Single fixed seed, single fixed batch.

If this test fails at the tight tolerance, that is a real bug — do not
loosen the tolerance to make it pass. Investigate, fix in vatis (perspic
is the reference), and *then* commit.

Commit: `task 3: tight-tolerance perspic cross-check at n=16384`.

### Task 4 — Heavy-padding cross-validation test

Most normalization bugs hide in padding-heavy regimes. Add a test (in
`tests/cross_validation/test_vs_perspic.py` or
`tests/unit/test_observables.py`, your call) where:

- The eval batch has >50% of tokens masked (mix of `attention_mask=0`
  and `labels=-100`, both should be exercised).
- All four observables (`chi_loss`, `chi_net`, `delta_loss`, `chi_pos`)
  agree with perspic to the same tolerances as the existing
  cross-validation tests.
- Bonus: also assert agreement with the exact-NTK ground truth from
  task 2 if feasible on the toy fixture.

Commit: `task 4: heavy-padding cross-validation test`.

### Task 5 — Fix `_compute_cross_pair` ordering footgun

`SESSION_SUMMARY.md` item 4 under "What is blocked or risky": the
analyzer's cross-pair path reads `chi_loss_a / chi_loss_b` from an
attribute cache populated only during the self-pair loop, and falls back
to zeros if called out of order. The public `Analyzer.run` always calls
self pairs first so this is safe in practice — but it's a latent
footgun.

Fix it the cheap way: add an assertion at the top of `_compute_cross_pair`
that the relevant self-pair entries exist in the cache, with a clear
error message pointing the caller at the contract. Add a unit test that
calling `_compute_cross_pair` before the self pair raises. Do not
restructure the analyzer.

Commit: `task 5: assert self-pair precedence in _compute_cross_pair`.

### Task 6 — Custom `loss_fn`: wire or delete

`chi_loss_from_autograd` exists in `core/observables.py` and has unit
tests, but the analyzer always uses the closed-form CE path. There's
a TODO at the call site. Pick one:

- **Wire it through.** Plumb a user-provided `loss_fn` through the
  analyzer's self-pair path so non-CE losses actually work. Add an
  end-to-end test using a non-CE loss (e.g. label-smoothed CE or MSE on
  logits) that exercises the analyzer with the autograd path. Remove
  the TODO.
- **Delete it.** Remove `chi_loss_from_autograd` and its unit tests,
  remove the TODO, and add a one-line note in `SESSION_SUMMARY.md` that
  custom losses are out of scope until v1.2.

Pick whichever is smaller. Do not leave the dead code in place with the
TODO — that's the worst of both worlds.

Commit: `task 6: wire custom loss_fn through analyzer` *or*
`task 6: remove unused chi_loss_from_autograd and TODO`.

### Task 7 — Per-seq-CV memory pre-check

`SESSION_SUMMARY.md` item 2 under "What is blocked or risky":
`per_sequence_cv` stores `B` flat parameter-space gradient vectors,
~512 GB for an 8B model with `B=32`. Currently the analyzer accepts the
configuration and OOMs at the end of a 30-minute checkpoint load.

Add a startup check in the analyzer (or in
`PerSequenceControlVariateEstimator.__init__`):

```
estimated_bytes = B * n_params * 4   # fp32 per-sample grad vectors
if estimated_bytes > 0.5 * available_memory:
    raise ValueError(
        f"per_sequence_cv would need ~{estimated_bytes/1e9:.1f} GB for "
        f"B={B}, n_params={n_params}. Use chi_net_method='hutchinson' "
        f"for large models. See CLAUDE.md auto-selection rule."
    )
```

`available_memory` should be GPU memory if CUDA is available, host RAM
otherwise. Pick a sensible API (`torch.cuda.mem_get_info` works). Add a
unit test that the check fires for a deliberately oversized config and
does not fire for a sensible one.

Commit: `task 7: per-seq-CV memory pre-check at estimator init`.

### Task 8 — CLAUDE.md proposal (run last)

**Do not edit `CLAUDE.md` directly.** Write proposed changes to
`CLAUDE.md.proposed` at the project root. Allowed sections to propose
changes to:

- The architecture diagram (if tasks above added/moved files).
- The dependency list (if you added matplotlib or anything else).
- The test layout / counts.
- The compute scaling table — replace the order-of-magnitude guesses
  with the **real numbers from `examples/BENCHMARK.md`**, clearly noted
  as measured on the specific GPU you ran on.

**Forbidden sections** — do not propose changes to these, even if you
think they're wrong:

- "Scope (v1)" / "Out (v1)"
- "Core idea (math)" and all subsections
- "Working conventions" / "Permission model" / "Unattended-mode behavior"
- The glossary
- The implementation order

If you think one of the forbidden sections needs updating, write the
suggestion to `FOLLOWUPS.md` instead. The user will decide.

Commit: `task 8: propose CLAUDE.md updates from v1.1 work`.

## Done criteria

When you finish (or run out of independent tasks):

1. All tests still green: `.venv/bin/python -m pytest tests` shows the
   new total, all passing.
2. `git log --oneline` shows the baseline commit plus one commit per
   completed task, in order.
3. `git status` is clean.
4. Update `SESSION_SUMMARY.md`:
   - Bump test count to the new total.
   - Add a "v1.1 hardening pass" section listing what got done, what got
     skipped and why.
   - Update the "What is blocked or risky" list — remove items that
     were fixed (5, 6, 7), add anything new.
5. If anything was skipped or blocked, `BLOCKED.md` exists at the project
   root with: the exact obstacle, what you tried, what the user needs to
   do to unblock it, which tasks are affected.
6. If you found things worth doing that weren't in this file, they live
   in `FOLLOWUPS.md`, not done, not committed as code.

That's it. Stop when done. Do not start new work beyond this list.
