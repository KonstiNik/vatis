# vatis deployment examples — findings & sizing

Two end-to-end examples live under `examples/`, and this document covers both:

1. **`pythia_sweep.py` (pythia-14m, HF checkpoints)** — the canonical deployment
   example. Everything up to "Reproducing" is about it.
2. **`two_rulers_llama.py` (project Llama, native checkpoint bundle)** — the WP0
   GPU example: both rulers, `chi_pos` *and* `chi_pos^P`, over a bundle written by
   our own training run. See [WP0 GPU example](#wp0-gpu-example--two-rulers-on-a-project-llama-bundle)
   at the end.

What each example evaluates, why it's sized the way it is, and the non-trivial
things it surfaces on real checkpoints.

> **Performance benchmarks** (wallclock, memory, DDP scaling, estimator
> variance) live in [`examples/benchmark/`](benchmark/README.md), measured on
> A100s — the representative hardware. That document is about method- and
> hardware-independent *observable-value findings*; the one exception here is the
> WP0 example's timing table, which is a sizing deliverable in its own right
> (it calibrates the analysis budget for the WP1 grid) and is therefore reported
> with its hardware and its contention caveat attached.

## What the deployment example evaluates on

`examples/pythia_sweep.py` uses **two real-text eval batches**:

- **prose**: the opening of Jane Austen's *Pride and Prejudice* (public
  domain), tokenized with the model's own tokenizer and sliced into
  `B × S = 8 × 128 = 1024` non-overlapping tokens.
- **code**: a small self-contained Python module (linear-algebra
  primitives, sorting, primality), same `B × S = 1024` tokens after
  tokenization.

Real text matters here. The observable values must be theory-relevant: the
chi_net term is the squared Frobenius norm of the parameter Jacobian
**evaluated at the input**; feeding random tokens puts the evaluation at an
off-distribution point in input-space and the resulting trajectory across
checkpoints reflects nothing about Pythia's actual training dynamics. Tokenized
prose puts the evaluation back on the data manifold the model was trained on.
(Shape-only *performance* benchmarks are different — measuring wallclock/memory
is fine on synthetic random integers, which is what `examples/benchmark/` does.)

The example also computes the **cross-pair** observable
`δL(prose, code) = ⟨∇_θ L^prose, ∇_θ L^code⟩` and the corresponding
`chi_pos(prose, code)`. This is essentially free (one extra dot product per
checkpoint, zero extra backward passes — the gradients are already cached from
the self-pair work) and is the most LNP-relevant quantity in the example: it
measures whether a gradient step on prose helps or hurts the model on code.

## Sizing

The example is deliberately sized to run end-to-end in **well under 15 minutes**
on modest hardware, so a maintainer on a slow GPU or a cold HF cache still gets a
quick turnaround. As committed (`per_sequence_cv`, `n_h=32`, `B=8`, `S=128`, 13
revisions × 2 self pairs + 1 cross pair = 39 measurements) the measured wallclock
on an RTX 3090 Ti, with the code of 17 November 2026 (the run that wrote the
committed outputs), is:

- **~76 s on the first run** (cold HF cache, all 13 checkpoints downloaded),
- **~52–55 s on subsequent runs** (warm cache; also with the hub offline).

**This is slower than the April 2026 state, and the bottleneck has moved.** The
April files recorded ~0.54 s of compute per (checkpoint, pair) measurement and
~20 s / ~50 s warm / cold, so loading dominated. The November run records a
median ~1.5 s per measurement (38.5 s of the warm total), so the analysis is now
the larger share. The cause was not measured here; one candidate is the change to
drawing the Hutchinson probes on the CPU (`CHANGELOG.md`), since a pythia probe
has 8 × 128 × 50,304 entries, about three times the Llama example's. The run is
still far inside the 15-minute budget, and bumping `n_hutchinson`, adding eval
batches, or running both methods would stay within it (see
`examples/benchmark/single_gpu/` for the A100 wallclock/memory sweep).

## Revisions and early-phase findings

The example uses **13 revisions** spanning the published Pythia training
schedule, with four log-spaced early checkpoints:

```
step1, step8, step64, step512,                          # log-2 phase
step1000, step2000, step4000, step8000, step16000,      # main phase,
step32000, step64000, step128000, step143000            # roughly log-2
```

The early phase is essential. Without it the trajectories look deceptively
smooth and monotonic; with it several non-trivial features are visible:

1. **chi_loss is flat at 1.0 through step64** on both batches. The model
   literally doesn't improve on next-token prediction in the first 64 SGD
   steps — those steps are warmup, not learning.
2. **chi_net has a U-shape in the first ~100 steps.** It starts at ~1e8 at
   step1 (the random-init Frobenius norm), drops by ~3× to ~3e7 at step64, then
   climbs back up and grows monotonically to ~5e10 by step143000. The dip is not
   noise — it shows up identically on both prose and code, and in self
   delta_loss too. Plausible reading: the very first SGD steps "smooth out"
   high-magnitude initialization noise before the model starts building
   structured representations.
3. **The cross delta_loss has non-monotonic structure.** It starts at **+3.25
   at step1** (prose and code gradients well-aligned on a random model), drops to
   **+0.95 at step64** (essentially orthogonal), climbs back to **+2.62 at
   step1000**, then declines and **crosses zero around step4000**, ending at
   **−18 by step143000**. Without the early checkpoints the trajectory looks
   like a clean monotonic decline from positive to negative — a simpler, and
   partially wrong, story.

The negative end of the cross delta_loss trajectory is the negative-`chi_pos`
interference regime discussed in the paper (arXiv:2605.31244 §A.2): a gradient
step that decreases prose loss *increases* code loss, and vice versa. None of
this would be visible on random-integer inputs, where every batch is
statistically equivalent to every other.

## Reproducing

```bash
.venv/bin/python examples/pythia_sweep.py      # compute → results.parquet + plots
.venv/bin/python examples/analyze_results.py   # analyze → cos_similarity.png (zero vatis imports)
```

(Needs the `examples` extra for matplotlib: `uv pip install ".[examples]"`. Set
`HF_HOME` via your shell or the repo-root `.env` to relocate the model cache.)

`analyze_results.py` reads `results.parquet` and produces `cos_similarity.png` —
the normalized cross-batch gradient correlation
`cos(g_A, g_B) = δL(A,B) / sqrt(δL(A,A) · δL(B,B))` — plus a printed summary.
The cosine view exists because the raw cross `delta_loss` plot is visually
misleading: it shows a step1 → step512 decline followed by a "rebound" at
step1000, but the rebound is amplified by gradient magnitudes growing
simultaneously. In normalized terms the rebound is real but ~10× smaller than
the absolute number suggests, and the dominant feature is the early-phase
decline (`+0.31 → +0.012`, ~26×) — i.e. **most of the prose-vs-code gradient
decorrelation is over by step512**, well before the absolute cross delta_loss
even starts looking interesting.

The analysis script is intentionally separate from `pythia_sweep.py`: compute is
expensive (~1 min on an RTX 3090 Ti), analysis is cheap (sub-second), and the analysis script
imports only `pyarrow` (not vatis) — making the parquet schema the contract, the
right pattern for any downstream consumer.

Outputs (under `examples/pythia_sweep/`):

- `results.parquet` — canonical long-format table (234 rows: 13 ckpts × 6
  observables × (2 self pairs + 1 cross pair)).
- `chi_loss.png`, `chi_net.png` — self-pair trajectories (one line per eval
  batch).
- `delta_loss.png`, `chi_pos.png` — self pairs + the prose×code cross pair
  (symlog, since the cross values cross zero).
- `cos_similarity.png` (from `analyze_results.py`) — the normalized cross-batch
  correlation, with gradient magnitudes divided out.

These files were regenerated on 17 November 2026 with the code that adds the
eight run-setting columns (commit `9aa0751`), on an RTX 3090 Ti, replacing
files from April 2026; the parquet now has all 20 columns of the current schema.
Compared row by row with the April file: `chi_loss` agrees to 1.2e-7 relative
and `delta_loss` to 1.1e-5 on the self pairs (6.7e-4 on the smallest cross-pair
value, an absolute difference of 1.5e-3 next to self values of ~580), which is
float32 rounding; `chi_net` and `chi_pos` differ by up to 13% (median 1%),
because the Hutchinson probes are now a different random draw for the same seed.
Those differences are what the probe noise predicts: divided by √2 times the
new `value_probe_stderr`, they have a root-mean-square of 1.1. Every number
quoted in the sections above is unchanged at the precision quoted.

## WP0 GPU example — two rulers on a project Llama bundle

`examples/two_rulers_llama.py` (compute) + `examples/two_rulers_llama_evaluate.py`
(figure and tables) are the end-to-end example for the **native checkpoint
bundle** path and the **preconditioned ruler**: a real training run writes a
bundle, vatis measures `chi_pos` and `chi_pos^P` over every checkpoint on one
GPU, and the result is one figure plus a timing table. Same two-phase split as
the pythia example — the compute phase is the expensive one and writes the
parquet; the evaluate phase imports no vatis code.

**It needs a bundle, so it is not runnable from a clean checkout.** The training
half lives in the training repo and hands over on disk:

```bash
# training stack's venv (torch 2.9 + lightning + perspic), in that repo:
bash runs/example_bundle_run.sh 300 32
#   -> $WP0_EXAMPLE_ROOT/bundle          the checkpoint bundle
#   -> $WP0_EXAMPLE_ROOT/eval_batches    3 held-out measurement batches (.npy)

# vatis venv, separate process:
.venv/bin/python examples/two_rulers_llama.py \
    --bundle "$WP0_EXAMPLE_ROOT/bundle" \
    --eval-batches "$WP0_EXAMPLE_ROOT/eval_batches" --device cuda
.venv/bin/python examples/two_rulers_llama_evaluate.py --dir examples/output
```

`--device cuda` is the default and falls back to CPU with a printed note, so the
example runs (slowly) anywhere. If the `eval_batch_*.npy` files are missing the
script falls back to **random tokens with a loud warning**, stamps
"scientifically empty" on the figure and into `run_summary.json`, and keeps
going — a pipeline demo, never a measurement.

### What it measures on

`k = 2` fixed measurement batches (`A1`, `A2`) plus one eval batch (`B`), each
`8 × 512` tokens taken from the **tail of the held-out split**, frozen across
checkpoints, with `cross_pairs = [(A1, B), (A2, B)]`. The batches carry the
training stack's own padding (~41–55% of positions are `[EOS]` padding at
`seq_len = 512`).

**Padding is scored the way the run trained on it.** The script applies the pad
convention the bundle records (`mask_pad_in_loss` / `loss_ignore_index` in
`metadata.json`, see `docs/SPEC.md` → Native checkpoint bundles) to the held-out
batches, exactly as the loader applies it to the update batch: if the training
loss ignored the pad targets, so do these rows; if it did not, every position
counts. Measuring a different objective than the one that was trained would be
the larger error in either direction. **For the run committed here nothing is
masked** — its training criterion masks no padding, and every number below was
computed with all positions counted.

### Sizing, as measured (RTX 3090 Ti, **shared card**)

Llama `dim=320, n_layers=6, n_heads=4`, vocab 4096 → **10,981,440 parameters in
57 tensors**; 300 AdamW steps at `B=32, S=512`; `--bundle_steps log:20` → 19
checkpoints. Analysis: `per_sequence_cv`, `n_hutchinson=32`,
`micro_batch_size=8`, fp32, 3 eval batches + 2 cross pairs, 1020 result rows.
These are the numbers of the committed run in `examples/output/` (17 November
2026, see [Outputs](#outputs-under-examplesoutput)):

| stage | per checkpoint | total |
|---|---|---|
| bundle load (weights → GPU) | 0.07–2.71 s (mean 0.61) | 11.5 s |
| analysis (3 self pairs + 2 cross pairs, both rulers) | 3.61–4.35 s (mean 3.95) | 75.1 s |
| **sweep** (load + both rulers + sink) | — | **87.1 s** |
| script total (+ the `P = I` re-check and the epsilon split) | — | 100.3 s |
| peak GPU memory (torch allocator, this process) | — | 1683 MiB |

Budget a grid run with the *sweep* row; the extra 13 s are this example's own
diagnostics, which a production sweep does not pay.

**Caveat attached to every number above: the card is shared.** Another job's
work runs on it at times; `nvidia-smi` showed it idle (18 MiB in use, 0%
utilization) immediately before and after this run, but nothing watched it
during the run, so contention cannot be ruled out. (The `gpu_contended` entry in
`run_summary.json` is a fixed sentence the script writes on every GPU run, not a
measurement.) Nothing was tuned or best-of-N'd. The run these files replace (July
2026, same card) recorded 3.22–6.51 s per checkpoint (mean 3.90 s) of analysis,
and the run this table showed before that one a mean of 4.13 s, which gives an
idea of the run-to-run spread. Both rulers cost the same as one: the P observables reuse the raw pass's
probes, so `chi_net^P` adds no backward.

Rule of thumb for budgeting the WP1 grid from this: **~4 s per checkpoint per
(3 batches + 2 cross pairs) at 11M parameters and 4k measurement tokens**, plus
the time to read a checkpoint (~105 MB of weights and optimizer state) from disk:
0.07–0.14 s on 11 of the 19 checkpoints of this run and 0.86–2.71 s on the
other 8, with the bundle on a network file system (the July run read its bundle
from `/tmp` at 0.37–0.47 s per checkpoint). Checkpoint loading is *not* the
bottleneck here (unlike the HF path, where downloads dominate).

### The step-0 hole, and why the figure has one

Under pre-step semantics a step-0 checkpoint has an **empty optimizer state** —
torch creates `exp_avg_sq` lazily inside the first `step()` — so no AdamW `P`
exists there and `adamw_preconditioner()` skips the P rows with a warning
(18 of 19 checkpoints carry P rows here). The example prints that as a result,
and the figure simply starts the `chi_pos^P` curves at step 1 with an annotation.
Plotting a zero there would invent a measurement.

### The epsilon ceiling, measured for this run

`P = 1/(sqrt(v_hat) + eps)` puts every coordinate the optimizer never touched at
`1/eps = 1e8`, versus ~1e0–1e2 for a live coordinate. Nothing is clipped,
floored or masked — the convention is a frozen project decision — so the honest
move is to **measure the share and publish it**. `epsilon_cap_mask` gives the
mask; running the estimator twice on the same checkpoint/batch/seed (once with
`P`, once with the ceiling entries zeroed) splits `chi_net^P` exactly, because
the trace estimate is linear in `P` at fixed probes.

For this run (300 steps, ~4.9M training tokens over a 4096-token vocabulary):

| step | P diagonal at `1/eps` | dead vocab rows | of those, present in batch B | `chi_net^P` from the ceiling |
|---|---|---|---|---|
| 1 | 835,840 / 10,981,440 (7.61%) | 2612 / 4096 | 139 | **10.23%** |
| 20 | 189,440 / 10,981,440 (1.73%) | 592 / 4096 | 5 | **0.37%** |
| 299 | 78,080 / 10,981,440 (0.71%) | 244 / 4096 | 0 | **0.00%** |

Three things worth knowing, all visible in that table:

- **Every ceiling coordinate is in `model.tok_embeddings.weight`** — zero
  elsewhere, including the untied `model.output.weight`, whose every row gets
  gradient from the softmax on every token. An embedding row only gets gradient
  when its token appears in a training batch, which is exactly why unused
  vocabulary rows are the population.
- **A dead coordinate only hurts if the measurement batch uses it.** `chi_net^P`
  weights `P_i` by `‖J_{:,i}‖²`, and the Jacobian column of an embedding row for
  a token absent from batch B is identically zero. That is why the step-299
  share is exactly `0.00%` rather than merely small: by then no dead row is used
  by B at all.
- **Run length is what shrinks it.** The 50-step gate run measured 28–39% of the
  diagonal at the ceiling and 71–86% of `chi_net^P`; six times the steps (and
  ~30× the tokens) brings that to 7.6%/10.2% at step 1 and 0.7%/0.0% by step 299.
  A grid run long enough to cover its vocabulary should not see `chi_pos^P`
  materially suppressed by dead directions — but **check the number, don't
  assume it**, and note that the diagonal share alone overstates the effect
  (0.71% of coordinates carried 0% of the trace here).

The result file now carries this split on every P row, not only on the three
checkpoints above: `chi_net_P_capped` and `delta_loss_P_capped` are the parts of
`chi_net_P` and `delta_loss_P` contributed by the ceiling coordinates, and
`preconditioner_capped_fraction` is the diagonal share. For batch B's self pair
in the committed file, `chi_net_P_capped / chi_net_P` is 10.23% at step 1, 0.37%
at step 20 and exactly 0 at step 299 — the last column of the table.

### Outputs (under `examples/output/`)

**These files were regenerated on 17 November 2026** with the code that adds
the eight run-setting columns (the `feat/result-schema` branch, commit
`9aa0751`), on an RTX 3090 Ti as before, with the command and defaults above
(`--device cuda`, seed 0). They replace files from 28–29 July 2026, which
predated the `*_capped` rows and the eight columns.

**The bundle is not the one the July files were measured on.** That bundle sat
in `/tmp` and was lost; the training run was repeated on 31 July 2026 with, per
the data directory's notes, the same command and seeds. The two training runs agree bit for bit up to step 27
(identical recorded train loss at every checkpoint) and then drift apart, most
likely because GPU training is not bit-for-bit deterministic: from step 37 on the recorded train
loss differs, by 3e-6 relative at step 37 and up to about 0.5% later (1.7778 vs
1.7816 at step 299). Every row of the July file has a counterpart in the new one, and the
comparison follows that split exactly:

- steps 0–8: all rows agree to better than 1e-6 relative (largest 7.8e-7);
- steps 11–27: they agree to better than 2.5e-5 relative, the difference growing
  step by step as the two training runs' weights separate;
- steps 37–299: they differ by up to ~1% in `chi_loss`, up to ~5% in `chi_net`
  and up to ~3% in `chi_net_P`, and by much more in the small, signed cross-pair
  values. In particular both AdamW cross pairs (`A1 × B`, `A2 × B`) of
  `delta_loss_P` / `chi_pos_P` were positive at step 299 in the July file and
  are negative in this one, as the raw cross pairs already were in both.

A CPU run of the current code over steps 0 and 1 agrees with the July file to
5.2e-7 relative and with this one to 1.7e-6 (largest on a `delta_loss_P_capped`
row; 5.9e-7 on every other row), so the device does not matter at this level
either. So the rows that measure the same weights reproduce, and the rest describe a
different (equally valid) training run; the July values for steps 37–299 cannot
be reproduced, because their bundle no longer exists.

- `results.parquet` — 1020 rows: 19 checkpoints × (3 self + 2 cross pairs) ×
  the raw observables, plus the P observables and their `_capped` parts on the
  18 checkpoints that have an optimizer state, in the canonical long format
  with all 20 columns of the current schema.
- `run_summary.json` — everything the parquet has no column for: per-checkpoint
  train loss / lr / tokens seen (from the bundle's `meta.json`), the performance
  table, the `P = I` re-check, the epsilon-ceiling records, the data-fallback
  flag.
- `performance.csv` — the timing table on its own.
- `two_rulers_llama.png` — the figure: `chi_pos` and `chi_pos^P` versus training
  step (left) and versus train loss (right, time running left to right), for the
  eval batch's self pair and the two cross pairs. Symlog y, because a cross pair
  is signed and this run does cross zero late in training.

The figure is **descriptive**. A 300-step demo on a 11M-parameter model says
nothing about training dynamics at scale, and no claim is made here beyond
"the pipeline produces these curves, and this is what they look like".
