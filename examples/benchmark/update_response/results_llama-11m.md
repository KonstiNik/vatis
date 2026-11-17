# What the rows of the step that was taken cost — 11M-parameter Llama

Measured with `update_response_cost.py` in this directory, on two bundles of the
same Llama (10,981,440 parameters; one fp32 parameter vector is 42 MB): one
trained with AdamW, one with SGD with momentum. Both have fp32 weight files, a
pad-masked loss and gradient-norm clipping at 1.0 (stated with
`--clip-grad-norm 1.0`).

| mode | what runs |
|---|---|
| `baseline` | the analysis without an update rule |
| `formula` | + the `loss_change_first_order*` rows and the step rows (`update_response(realized=False)`) |
| `realized` | + `loss_change_first_order_weights`, `loss_change`, `loss_change_residual` |

Each measurement is one process that analyzes four checkpoints (steps 1, 2, 661,
6611) with the update batch (32 × 512 tokens) and held-out batches of 8 × 512
tokens. No preconditioner, no cross pairs.

**Columns.** Wall-clock is `time.perf_counter()` around `Analyzer.run`, per
checkpoint, loading included. Peak RSS is the peak resident memory of the whole
process (`ru_maxrss`). Peak GPU memory is `torch.cuda.max_memory_allocated()`,
reset before the run: this process's tensors only. A cell is the median over
the runs, with the smallest and largest value in brackets.

**Limits.**

- The machine and the card are shared. A wall-clock difference of a few percent
  is not a result; the memory columns are much more stable.
- Not measured: larger models, more than one GPU, DDP, bf16. At this size
  loading and Python overhead are a large part of the wall-clock, so the ratios
  do not carry over to a large model. The counts do — no extra backward pass,
  two forward passes per micro-batch for the realized rows, one parameter-sized
  vector per evaluation batch — and `tests/unit/test_update_response_cost.py`
  pins them.
- The tables were measured before the result-file columns and the `hutchinson`
  default were merged into this branch. The merge does not change what the
  script measures: it names the estimator, and the passes are the same. A check
  after the merge is in `results_gpu_after_merge.jsonl` (AdamW bundle, GPU,
  three runs per mode): 0.6 / 0.7 / 0.9 s per checkpoint (×1.23 and ×1.52) and
  1173 / 1213 / 1214 MB of GPU memory, i.e. the memory of the table below and
  ratios within its spread.
- The `adamw` and `sgd` CPU runs were interleaved by calling the script once
  per repeat, so the `repeat` field of those records in `results_cpu.jsonl` is
  0 throughout; which repeat a record belongs to follows only from the order of
  the lines. The script now takes `--first-repeat` for such runs.

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

- **Wall-clock.** Of the five `adamw` and `sgd` repeats the first two were
  slower (10 to 14 s per checkpoint) than the last three (7.4 to 9.6 s). The
  recorded load average does not separate them clearly (7.7 to 10.6 against 7.4
  to 8.7), so the cause is not established. In the last three repeats the
  formula rows add 0 to 0.2 s per checkpoint to 7.4 s (AdamW +0.12 to +0.18 s,
  SGD −0.02 to +0.09 s) and the realized rows 1.4 to 2.2 s (×1.2 to ×1.3). Over
  all runs the formula rows are inside the spread: in `adamw-nh8` the mode with
  the rule came out faster than the one without.
- **AdamW rule against SGD rule.** No difference in wall-clock that this
  measurement resolves.
- **Memory.** With the formula rows peak RSS rises by 105 MB (AdamW) and 90 MB
  (SGD); the counts say 126 MB and 84 MB (one kept gradient plus two or one
  state vectors of 42 MB). The realized rows add nothing further to the peak.
  With four evaluation batches instead of two the rule adds 188 MB (counts:
  210 MB): memory grows with the number of evaluation batches.

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


Reading the GPU table (one shared RTX 3090 Ti; no run failed with an allocation
error and none was repeated):

- **Wall-clock.** The formula rows add 0.1 to 0.2 s per checkpoint and the
  realized rows 0.2 to 0.4 s. With `n_hutchinson = 8` the baseline takes 1.5 s
  and the additions stay about the same in seconds, so the ratio falls as the
  `χ_net` estimate gets more expensive. One `formula` run on the AdamW bundle
  took 4.2 s per checkpoint; the medians are not affected.
- **AdamW rule against SGD rule.** The same GPU memory. The AdamW sweep is
  slower in every mode, the baseline included (0.8 s against 0.6 s); it ran
  first, at a higher machine load, and in the check after the merge its
  baseline is 0.6 s as well. No difference between the two rules is resolved.
  On the host the AdamW rule holds two state vectors and the SGD rule one (peak
  RSS +115 MB against +36 MB).
- **Memory.** With two evaluation batches the rule adds 40 MB of GPU memory at
  the peak, with four it adds 127 MB: one kept gradient (42 MB) per further
  evaluation batch. The optimizer state stays on the CPU. The realized rows do
  not raise the peak: the second model is on the card only during the
  forward-only realized pass, and the peak is reached during a backward pass.
