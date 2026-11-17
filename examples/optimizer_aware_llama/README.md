# What do `chi_pos` and its AdamW-preconditioned version look like over a training run?

A small Llama (11M parameters) is trained for 300 AdamW steps; 19 checkpoints
are saved with the optimizer's state. At each one vatis measures `chi_pos`
twice: plainly ("raw"), and weighted the way AdamW weights each parameter
("preconditioned", `chi_pos^P`). The weights are AdamW's per-parameter factors
`P = 1/(√v̂ + ε)`, read from the saved optimizer state. The example shows that
the pipeline works on a real run; it says nothing about training at scale.
Longer notes — the measurement batches, the padding convention, the full
sizing, the step-0 gap, the `1/ε` ceiling and the re-run of the training — are
in [details.md](details.md).

## What you need

- **A checkpoint bundle and its held-out batches, which are not part of this
  repository.** They are written by the training repository's
  `runs/example_bundle_run.sh`, in that repository's own environment (torch 2.9,
  lightning, perspic). The two environments never share a process; the bundle
  on disk is the hand-over.
- The training repository's model code, which `run.py` imports to rebuild the
  Llama (point the environment variable `WP0_LLAMA_SRC` at that repository).
- A GPU. Without one the script falls back to the CPU with a printed note and
  runs slowly.
- About 100 s on an RTX 3090 Ti for the committed run (87 s of it the sweep).

## How to run

```bash
# in the training repository, with its own environment:
bash runs/example_bundle_run.sh 300 32
#   -> $WP0_EXAMPLE_ROOT/bundle          the checkpoint bundle
#   -> $WP0_EXAMPLE_ROOT/eval_batches    3 held-out measurement batches (.npy)

# in vatis, separate process:
python examples/optimizer_aware_llama/run.py \
    --bundle "$WP0_EXAMPLE_ROOT/bundle" \
    --eval-batches "$WP0_EXAMPLE_ROOT/eval_batches" --device cuda
python examples/optimizer_aware_llama/evaluate.py --dir examples/optimizer_aware_llama
```

Without the `eval_batch_*.npy` files `run.py` falls back to random tokens, which
is a pipeline demo, never a measurement ([details.md](details.md)).

## What it shows

![chi_pos and chi_pos^P versus training step and versus train loss](chi_pos_raw_vs_adamw.png)

Left: `chi_pos` (blue) and `chi_pos^P` (orange) versus training step; right:
the same versus train loss, with time running left to right. One line each for
the eval batch `B` on itself and for the two cross pairs `A1 × B` and `A2 × B`
(a cross pair compares the gradients of two different batches).
The y axis is symlog because cross pairs are signed.

- **There is no `chi_pos^P` at step 0.** The optimizer state of a step-0
  checkpoint is empty, so no AdamW `P` exists there; the P rows are skipped
  (18 of 19 checkpoints have them) and the orange curves start at step 1.
- **Parameters AdamW has never updated distort `chi_pos^P` early, then stop
  mattering.** AdamW gives them the weight `1/ε = 10⁸`; they are all
  embedding rows of tokens that have not yet appeared in a training batch. At
  step 1 they are 7.6% of the parameters and carry 10% of `chi_net^P` on batch
  B; the share peaks at 14% at step 2, is below 0.4% from step 11 on, and by
  step 299 they are 0.7% of the parameters and carry nothing, because B uses
  none of those tokens.
- **The cross pairs turn negative at the last checkpoint:** at step 299 both
  `chi_pos` and `chi_pos^P` are slightly negative for `A1 × B` and `A2 × B`.
- **The preconditioned observables need no extra backward pass:** they reuse
  the raw pass's probes. Raw and preconditioned together took about 4 s of
  analysis per checkpoint for 3 batches and 2 cross pairs; this run did not time
  the raw observables alone. Largest per-checkpoint peak GPU memory 1687 MiB
  (`performance.csv`).

The loss the run recorded and every observable, raw and preconditioned, over
training — the diagnostic figure every example draws:
[observables_over_training.png](observables_over_training.png).

## What it does not show

- Anything about training dynamics at scale. A 300-step run of an 11M-parameter
  model shows that the pipeline produces these curves and what they look like;
  no claim is made beyond that.
- Anything about padding masking: this run's training loss masks no padding, so
  every position is counted ([details.md](details.md#what-it-measures-on)).
- Timings for an idle card: the card is shared, and nothing watched it during
  the run (it was idle just before and after). The timings are as measured, not
  tuned.
- The training run behind the earlier (July 2026) outputs: that bundle was
  lost, and the committed files come from a re-run of the same training, which
  drifts from the first one after step 27
  ([details.md](details.md#outputs-and-the-re-run-of-the-training)).

## About the committed outputs

- Files: `results.parquet` (1020 rows), `run_summary.json` (settings,
  per-checkpoint train loss and learning rate, the timing table, the `P = I`
  re-check, the ceiling records), `performance.csv` (the timing table),
  `chi_pos_raw_vs_adamw.png` and `observables_over_training.png` (both drawn
  by `evaluate.py` from `results.parquet` and `run_summary.json`).
- Scripts: `run.py` shows the vatis calls; the model factory and the
  measurement batches are in `llama_bundle.py`, the timing, the two
  self-checks and the summary in `llama_checks.py`. Code outside this
  repository that needs the model factory imports it with this folder on
  `sys.path`: `from llama_bundle import llama_factory`.
- Settings: `chi_net_method="per_sequence_cv"`, `n_hutchinson=32`,
  `micro_batch_size=8`, fp32, seed 0, `--device cuda` (recorded in
  `run_summary.json`).
- Hardware: NVIDIA GeForce RTX 3090 Ti, torch 2.10.0+cu128 (recorded in
  `run_summary.json`).
- Regenerated on 17 November 2026 with the then-current code and committed in
  `e1aa6f4` (#15); moved into this folder, unchanged, in `00944c4` (#17).
  The two figures were redrawn later from the same files (a new title, and the
  new overview figure); the data are unchanged.
