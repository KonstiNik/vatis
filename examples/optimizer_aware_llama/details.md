# Llama bundle example — details

The longer notes behind [README.md](README.md). `run.py` (compute) and
`evaluate.py` (figure and tables) are the end-to-end example for the **native
checkpoint bundle** path and the **preconditioned observables**: a real
training run writes a bundle, vatis measures `chi_pos` and `chi_pos^P` over
every checkpoint on one GPU, and the result is one figure plus a timing table.
It uses the same two-phase split as the [Pythia example](../pythia_sweep/README.md):
the compute phase is the expensive one and writes the parquet; the evaluate
phase imports no vatis code.

`--device cuda` is the default and falls back to the CPU with a printed note,
so the example runs (slowly) anywhere. If the `eval_batch_*.npy` files are
missing, the script falls back to **random tokens with a loud warning**, stamps
"scientifically empty" on the figure and into `run_summary.json`, and keeps
going — a pipeline demo, never a measurement.

## What it measures on

`k = 2` fixed measurement batches (`A1`, `A2`) plus one eval batch (`B`), each
`8 × 512` tokens taken from the **tail of the held-out split**, frozen across
checkpoints, with `cross_pairs = [(A1, B), (A2, B)]`. The batches carry the
training stack's own padding (~41–55% of positions are `[EOS]` padding at
`seq_len = 512`).

**Padding is scored the way the run trained on it.** The script applies the pad
convention the bundle records (`mask_pad_in_loss` / `loss_ignore_index` in
`metadata.json`, see [`docs/SPEC.md` → Native checkpoint bundles](../../docs/SPEC.md))
to the held-out batches, exactly as the loader applies it to the update batch:
if the training loss ignored the pad targets, so do these rows; if it did not,
every position counts. Measuring a different objective than the one that was
trained would be the larger error in either direction. **For the run committed
here nothing is masked** — its training criterion masks no padding, and every
number below was computed with all positions counted.

## Sizing (RTX 3090 Ti, shared card)

Model: Llama `dim=320, n_layers=6, n_heads=4`, vocabulary 4096, **10,981,440
parameters**, trained for 300 AdamW steps at `B=32, S=512`, with 19 saved
checkpoints. Analysis: `per_sequence_cv`, `n_hutchinson=32`,
`micro_batch_size=8`, fp32. The committed run took:

| stage | per checkpoint | total |
|---|---|---|
| bundle load (weights → GPU) | 0.07–2.71 s (mean 0.61) | 11.5 s |
| analysis (3 self pairs + 2 cross pairs, raw and preconditioned) | 3.61–4.35 s (mean 3.95) | 75.1 s |
| **sweep** (load + raw and preconditioned observables + writing the rows) | — | **87.1 s** |
| script total (+ the `P = I` re-check and the epsilon split) | — | 100.3 s |
| peak GPU memory (torch allocator, this process) | 1641–1687 MiB | 1683 MiB as `run_summary.json` records it; the largest per-checkpoint value in `performance.csv` is 1687 MiB |

To plan a grid of runs, use the *sweep* row: about **4 s of analysis per
checkpoint** for 3 batches and 2 cross pairs at this size, plus reading the
checkpoint (~105 MB): about 0.1 s when cached, up to 2.7 s here, where the
bundle is on a network file system. The other 13 s are this example's own
checks. The preconditioned observables need no extra backward pass: the P
values reuse the raw pass's probes. This run did not time the raw observables
alone, so their extra cost is not measured here.

The timings are as measured, not tuned. The card is shared with other jobs; it
was idle just before and just after this run, but nothing watched it during
the run. (The `gpu_contended` line in `run_summary.json` is fixed text the
script writes for every GPU run, not a measurement. `run.py` now writes "not
checked -- the card may be shared"; the committed file still carries the
earlier fixed text, "yes -- another process shares this card".)

This timing table is the one hardware-specific number set kept with an example
rather than in [`../benchmark/`](../benchmark/README.md): it calibrates the
analysis budget for a planned grid of training runs, so it is reported with its
hardware and its contention caveat attached.

## The step-0 gap, and why the figure has one

Under pre-step semantics a step-0 checkpoint has an **empty optimizer state** —
torch creates `exp_avg_sq` lazily inside the first `step()` — so no AdamW `P`
exists there, and `adamw_preconditioner()` skips the P rows with a warning
(18 of 19 checkpoints carry P rows here). The example prints that as a result,
and the figure starts the `chi_pos^P` curves at step 1 with an annotation.
Plotting a zero there would invent a measurement.

## The epsilon ceiling, measured for this run

`P = 1/(sqrt(v_hat) + eps)` puts every coordinate the optimizer never touched
at `1/eps = 1e8`. A live coordinate is far below that: the smallest entry of
`P` (`p_min` in `run_summary.json`) is 0.66 at step 1 and grows with training
to 183 at step 299. Nothing is clipped, floored
or masked — the convention is a frozen project decision — so the honest move is
to **measure the share and publish it**. `epsilon_cap_mask` gives the mask;
running the estimator twice on the same checkpoint, batch and seed (once with
`P`, once with the ceiling entries zeroed) splits `chi_net^P` exactly, because
the trace estimate is linear in `P` at fixed probes.

For this run (300 steps, ~4.9M training tokens over a 4096-token vocabulary):

| step | P diagonal at `1/eps` | dead vocab rows | of those, present in batch B | `chi_net^P` from the ceiling |
|---|---|---|---|---|
| 1 | 835,840 / 10,981,440 (7.61%) | 2612 / 4096 | 139 | **10.23%** |
| 20 | 189,440 / 10,981,440 (1.73%) | 592 / 4096 | 5 | **0.37%** |
| 299 | 78,080 / 10,981,440 (0.71%) | 244 / 4096 | 0 | **0.00%** |

Three things worth knowing, all visible in that table:

- **Every ceiling coordinate is in `model.tok_embeddings.weight`** — none
  elsewhere, including the untied `model.output.weight`, whose every row gets
  gradient from the softmax on every token. An embedding row only gets gradient
  when its token appears in a training batch, which is exactly why unused
  vocabulary rows are the population.
- **A dead coordinate only matters if the measurement batch uses it.**
  `chi_net^P` weights `P_i` by `‖J_{:,i}‖²`, and the Jacobian column of an
  embedding row for a token absent from batch B is identically zero. That is
  why the step-299 share is exactly `0.00%` rather than merely small: by then
  no dead row is used by B at all.
- **Run length is what shrinks it.** The 50-step gate run measured 28–39% of
  the diagonal at the ceiling and 71–86% of `chi_net^P`; six times the steps
  (and ~30× the tokens) brings that to 7.6% / 10.2% at step 1 and 0.7% / 0.0%
  by step 299. A grid run long enough to cover its vocabulary should not see
  `chi_pos^P` materially suppressed by dead directions — but **check the
  number, don't assume it**, and note that the diagonal share alone overstates
  the effect (0.71% of coordinates carried 0% of the trace here).

The share is not monotonic in the step. From the `_capped` rows of
`results.parquet` (batch B, `chi_net_P_capped / chi_net_P`): 10.2% at step 1,
a peak of 13.9% at step 2, 13.4% at step 3, 11.1% at step 4, then falling to
0.02% at step 15, back to 0.37% at step 20, at most 0.28% from step 27 to step
90, and 0.00% from step 122 on. The table above shows three steps of this
series.

The result file carries this split on every P row: `chi_net_P_capped` and
`delta_loss_P_capped` are the parts that come from ceiling coordinates, and
`preconditioner_capped_fraction` is the diagonal share. For batch B,
`chi_net_P_capped / chi_net_P` gives the last column of the table above.

## Outputs and the re-run of the training

All under `examples/optimizer_aware_llama/`, regenerated on 17 November 2026
with the then-current code, using the command in the README (`--device cuda`,
seed 0):

- `results.parquet` — 1020 rows in the long format: the raw observables for 19
  checkpoints × (3 self + 2 cross pairs), plus the P observables and their
  `_capped` parts for the 18 checkpoints that have an optimizer state.
- `run_summary.json` — everything the parquet has no column for: per-checkpoint
  train loss, learning rate and tokens seen (from the bundle's `meta.json`), the
  performance table, the `P = I` re-check, the epsilon-ceiling records, the
  data-fallback flag.
- `performance.csv` — the timing table on its own.

A fresh run of the current `run.py` writes a `run_summary.json` that differs
from the committed one in more than the timings: the scripts were later
restructured (support code moved into `llama_bundle.py` and `llama_checks.py`),
and the text of the performance table, which the summary stores under
`performance.table`, was reworded at the same time ("raw and preconditioned"
for "both rulers"; the peak-memory line and the peak column of the "mean" row
now give the largest and the mean of the per-checkpoint peaks instead of the
last checkpoint's peak, 1683 MiB, which is still what `performance.peak_gpu_mb`
records). `gpu_contended` changed as described above. The computed values do
not change: a re-run with the restructured script reproduced the committed
`results.parquet` to within 1.2·10⁻⁷ relative.
- `chi_pos_raw_vs_adamw.png` — the figure: `chi_pos` and `chi_pos^P` versus
  training step (left) and versus train loss (right, time running left to
  right), for the eval batch's self pair and the two cross pairs. Symlog y,
  because a cross pair is signed and this run does cross zero late in training.

**These outputs come from a re-run of the training.** The bundle behind the
earlier outputs (July 2026) was lost, and the same training was run again. The
two runs record the same train loss up to step 27, and the measurements agree
(to within 1e-6 relative through step 8 and 2.5e-5 through step 27). From step
37 on, the runs drift apart (train loss up to 0.5% different), and so do the
measurements. Largest relative difference from step 37 on, comparing the July
file (`examples/output/results.parquet` before `e1aa6f4`) with the committed
one: about 1% in `chi_loss`, 5% in `chi_net` and 3% in `chi_net_P`; on the self
pairs about 40% in `delta_loss` and `chi_pos` and about 20% in `delta_loss_P`
and `chi_pos_P`; on the cross pairs, whose values are small and close to zero,
a difference of up to about 20 times the July value for the raw ones and about 6
times for the P ones. So `delta_loss` and `chi_pos` differ between the runs far
more than the training loss does. One sign changes: at step 299 the two AdamW cross
pairs (`chi_pos^P` for `A1 × B` and `A2 × B`) are now slightly negative, where
they were slightly positive before.

The figure is **descriptive**. A 300-step demo on an 11M-parameter model says
nothing about training dynamics at scale, and no claim is made here beyond
"the pipeline produces these curves, and this is what they look like".
