# What the rows of the step that was taken cost — 11M-parameter Llama bundles

Measured with `update_response_cost.py` (this directory). Two native bundles of
the same 11M-parameter Llama (10,981,440 parameters; one fp32 parameter vector
is 42 MB), one trained with AdamW and one with SGD with momentum, both with
fp32 weight files, a pad-masked loss and gradient-norm clipping at 1.0 (stated
with `--clip-grad-norm 1.0`). Three modes:

| mode | what runs |
|---|---|
| `baseline` | the existing raw rows only — no update rule |
| `formula` | + `update_response=source.update_response(realized=False)`: the `loss_change_first_order*` rows and the two step rows |
| `realized` | + the realized rows (`loss_change_first_order_weights`, `loss_change`, `loss_change_residual`) |

Every measurement is one process that analyzes four checkpoints (steps 1, 2,
661, 6611) with the checkpoint's own update batch (32 × 512 tokens) and held-out
batches of 8 × 512 tokens as eval batches. No preconditioner, no cross pairs.

**What is reported.** Wall-clock is `time.perf_counter()` around
`Analyzer.run`, divided by the four checkpoints; it includes loading each
checkpoint from disk. Peak RSS is the peak resident set size of the whole
process (`ru_maxrss`: interpreter, torch and the model included); "peak − start"
subtracts the resident size just before the bundle was opened. Peak GPU memory
is `torch.cuda.max_memory_allocated()` of the process after the run, reset
before it: this process's own tensors, not other processes on the card and not
the allocator's cache. Each cell is the median over the runs with the smallest
and the largest value in brackets.

**How far to trust it.** The machine and the card are shared. On the CPU the
wall-clock follows the machine's load (the 1-minute load average is recorded
with every run and given below), so a difference of a few percent between two
modes is inside the spread and is not a result; the memory columns are much
more stable. The countable costs behind these numbers — forward and backward
passes, buffers held — are pinned exactly, on toy bundles, by
`tests/unit/test_update_response_cost.py`.

**Not measured:** models larger than 11M parameters, more than one GPU, DDP,
and bf16 forward passes. At this size the fixed costs of a checkpoint (loading
it, Python overhead) are a large part of the wall-clock, so the *ratios* below
should not be carried over to a large model; the *counts* (no extra backward
pass; two extra forward passes per micro-batch for the realized rows; one more
parameter-sized vector per evaluation batch) do carry over.

## CPU

- model: 10,981,440 parameters (one fp32 parameter vector = 42 MB)
- checkpoints per run: steps [1, 2, 661, 6611]; eval batches: the update batch [32, 512] plus held-out batches of [8, 512] tokens (number of eval batches per label: {'adamw': 2, 'sgd': 2, 'adamw-4-eval-batches': 4, 'adamw-nh8': 2})
- `chi_net_method=hutchinson`, `n_hutchinson` per label: {'adamw': 2, 'sgd': 2, 'adamw-4-eval-batches': 2, 'adamw-nh8': 8}, `micro_batch_size=8`, fp32, device cpu, 8 torch threads, torch 2.10.0+cu128
- 1-minute load average of the machine during the runs: 7.4 – 12.2

| bundle | mode | runs | wall-clock per checkpoint, s: median (min – max) | vs baseline | peak RSS, MB: median (min – max) | vs baseline, MB | peak − start RSS, MB (median) |
|---|---|---|---|---|---|---|---|
| adamw (adamw) | baseline | 5 | 7.4 (7.4 – 11.5) | — | 2551 (2541 – 2568) | — | 2021 |
| adamw (adamw) | formula | 5 | 7.6 (7.5 – 12.1) | ×1.02 | 2656 (2627 – 2682) | +105 | 2127 |
| adamw (adamw) | realized | 5 | 9.6 (8.9 – 14.2) | ×1.30 | 2645 (2625 – 2665) | +94 | 2115 |
| sgd (sgd) | baseline | 5 | 7.6 (7.4 – 11.9) | — | 2535 (2519 – 2598) | — | 2006 |
| sgd (sgd) | formula | 5 | 7.6 (7.4 – 11.9) | ×1.00 | 2626 (2599 – 2632) | +90 | 2096 |
| sgd (sgd) | realized | 5 | 9.0 (8.8 – 14.2) | ×1.18 | 2634 (2609 – 2672) | +99 | 2104 |
| adamw-4-eval-batches (adamw) | baseline | 3 | 14.7 (12.7 – 17.6) | — | 2572 (2542 – 2584) | — | 2042 |
| adamw-4-eval-batches (adamw) | formula | 3 | 17.1 (12.2 – 17.9) | ×1.16 | 2760 (2738 – 2778) | +188 | 2230 |
| adamw-4-eval-batches (adamw) | realized | 3 | 20.8 (13.0 – 21.1) | ×1.41 | 2756 (2734 – 2758) | +184 | 2226 |
| adamw-nh8 (adamw) | baseline | 2 | 24.5 (22.5 – 26.4) | — | 2714 (2695 – 2733) | — | 2184 |
| adamw-nh8 (adamw) | formula | 2 | 22.5 (20.7 – 24.2) | ×0.92 | 2838 (2837 – 2840) | +124 | 2309 |
| adamw-nh8 (adamw) | realized | 2 | 23.2 (23.1 – 23.3) | ×0.95 | 2856 (2845 – 2868) | +143 | 2327 |


Reading the CPU table (8 torch threads on a shared 16-core machine):

- **The wall-clock follows the machine's load, and the spread is larger than
  most differences.** The `adamw` and `sgd` runs were interleaved repeat by
  repeat. The first two repeats ran at a 1-minute load average of 8 to 10.6 and
  took 10 to 14 s per checkpoint; the last three ran at about 8 (this
  benchmark's own threads) and took 7.4 to 9.6 s. The two side sweeps
  (`adamw-4-eval-batches`, `adamw-nh8`) were run before that, not interleaved,
  at loads of 8.3 to 12.2.
- **Formula rows.** In the three quiet repeats they add 0.1 to 0.2 s per
  checkpoint to 7.4 s (about 2%), for both optimizers. Over all runs that is
  inside the spread. In the `adamw-nh8` sweep (two runs per mode) the mode with
  the rule even came out faster than the one without: that is the spread, not
  an effect.
- **Realized rows.** In the quiet repeats they add 1.4 to 2.2 s per checkpoint
  (×1.2 to ×1.3): two more forward passes per micro-batch, and the second
  checkpoint read from disk.
- **AdamW rule against SGD rule.** No difference in wall-clock that this
  measurement resolves (formula rows, quiet repeats: 7.5 to 7.6 s against 7.5
  to 7.6 s; without the rule both bundles take 7.4 s).
- **Memory.** Peak RSS rises by 105 MB (AdamW) and 90 MB (SGD) with the formula
  rows. The counts say three and two buffers of 42 MB (126 MB and 84 MB): one
  kept gradient plus two or one state vectors. RSS is not a fine enough
  instrument to separate the two optimizers (the ranges overlap). The realized
  rows add nothing further to the peak. **With four evaluation batches instead
  of two the rule adds 188 MB instead of 105 MB**: memory grows with the number
  of evaluation batches (the counts say five buffers, 210 MB).

## GPU

- model: 10,981,440 parameters (one fp32 parameter vector = 42 MB)
- checkpoints per run: steps [1, 2, 661, 6611]; eval batches: the update batch [32, 512] plus held-out batches of [8, 512] tokens (number of eval batches per label: {'adamw': 2, 'sgd': 2, 'adamw-4-eval-batches': 4, 'adamw-nh8': 2})
- `chi_net_method=hutchinson`, `n_hutchinson` per label: {'adamw': 2, 'sgd': 2, 'adamw-4-eval-batches': 2, 'adamw-nh8': 8}, `micro_batch_size=8`, fp32, device NVIDIA GeForce RTX 3090 Ti, 8 torch threads, torch 2.10.0+cu128
- 1-minute load average of the machine during the runs: 2.4 – 8.9

| bundle | mode | runs | wall-clock per checkpoint, s: median (min – max) | vs baseline | peak RSS, MB: median (min – max) | vs baseline, MB | peak − start RSS, MB (median) | peak GPU memory, MB: median (min – max) | vs baseline, MB |
|---|---|---|---|---|---|---|---|---|---|
| adamw (adamw) | baseline | 5 | 0.8 (0.7 – 1.0) | — | 1198 (1191 – 1212) | — | 668 | 1173 (1173 – 1173) | — |
| adamw (adamw) | formula | 5 | 1.0 (0.7 – 4.2) | ×1.21 | 1313 (1297 – 1315) | +115 | 783 | 1213 (1213 – 1213) | +40 |
| adamw (adamw) | realized | 5 | 1.2 (0.9 – 1.2) | ×1.50 | 1336 (1332 – 1351) | +138 | 806 | 1214 (1214 – 1214) | +40 |
| sgd (sgd) | baseline | 5 | 0.6 (0.6 – 0.6) | — | 1205 (1192 – 1221) | — | 675 | 1173 (1173 – 1173) | — |
| sgd (sgd) | formula | 5 | 0.7 (0.7 – 0.8) | ×1.12 | 1241 (1231 – 1259) | +36 | 712 | 1213 (1213 – 1213) | +40 |
| sgd (sgd) | realized | 5 | 0.8 (0.8 – 0.9) | ×1.40 | 1261 (1251 – 1278) | +55 | 731 | 1214 (1214 – 1214) | +40 |
| adamw-4-eval-batches (adamw) | baseline | 3 | 0.8 (0.8 – 0.8) | — | 1202 (1201 – 1206) | — | 672 | 1174 (1174 – 1174) | — |
| adamw-4-eval-batches (adamw) | formula | 3 | 0.9 (0.9 – 1.0) | ×1.16 | 1322 (1297 – 1352) | +120 | 792 | 1300 (1300 – 1300) | +127 |
| adamw-4-eval-batches (adamw) | realized | 3 | 1.2 (1.1 – 1.2) | ×1.44 | 1339 (1338 – 1340) | +138 | 809 | 1300 (1300 – 1300) | +127 |
| adamw-nh8 (adamw) | baseline | 2 | 1.5 (1.5 – 1.5) | — | 1207 (1207 – 1207) | — | 677 | 1174 (1174 – 1174) | — |
| adamw-nh8 (adamw) | formula | 2 | 1.6 (1.6 – 1.6) | ×1.08 | 1349 (1343 – 1355) | +142 | 819 | 1216 (1216 – 1216) | +42 |
| adamw-nh8 (adamw) | realized | 2 | 1.8 (1.8 – 1.8) | ×1.18 | 1341 (1337 – 1346) | +135 | 812 | 1216 (1216 – 1216) | +42 |


Reading the GPU table (one RTX 3090 Ti, shared; no run failed with an
allocation error, none was repeated):

- **Wall-clock.** The formula rows add 0.1 to 0.2 s per checkpoint (AdamW
  bundle 0.8 → 1.0 s, SGD bundle 0.6 → 0.7 s), the realized rows 0.2 to 0.4 s
  (1.2 s and 0.8 s). With `n_hutchinson = 8` the analysis without the rule takes
  1.5 s per checkpoint and the additions are about the same number of seconds
  (1.6 s and 1.8 s), so the ratio falls as the `χ_net` estimate gets more
  expensive. One run is an outlier (4.2 s per checkpoint: the first `formula`
  run on the AdamW bundle); the medians are not affected.
- **AdamW rule against SGD rule.** On the card both cost the same memory. In
  wall-clock the AdamW bundle is slower in every mode including the one without
  the rule (0.8 s against 0.6 s), so that difference is not the rule's; the
  rule's own addition is 0.2 s (AdamW) against 0.1 s (SGD) per checkpoint for
  the formula rows, which is at the resolution of this measurement. On the host
  the AdamW rule holds two state vectors and the SGD rule one (peak RSS +115 MB
  against +36 MB).
- **GPU memory grows with the number of evaluation batches.** With two eval
  batches the rule adds 40 MB at the peak, with four it adds 127 MB: one kept
  gradient (42 MB) per evaluation batch beyond the first. The optimizer state
  does not show here because it stays on the CPU.
- **The realized rows do not raise the peak GPU memory** beyond the formula
  rows at this size. The second model (42 MB of weights) is on the card only
  during the realized pass, which is forward-only; the peak is reached earlier,
  during a backward pass.
