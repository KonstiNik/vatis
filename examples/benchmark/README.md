# vatis benchmarks (A100)

Performance and accuracy benchmarks for vatis, measured on NVIDIA A100 GPUs —
the production-class hardware vatis is meant to run on. Three independent
benchmarks, each in its own subdirectory with its committed results:

| benchmark | question it answers | model | hardware |
|---|---|---|---|
| [`single_gpu/`](single_gpu/results_pythia-160m.md) | wallclock + peak memory vs `(method, n_hutchinson, batch_size)` | pythia-160m | 1× A100 (MIG 3g.20gb slice) |
| [`ddp_scaling/`](ddp_scaling/DDP_SCALING_RESULTS.md) | strong / weak DDP scaling efficiency across ranks | pythia-1.4b | 1–4× A100 (40 GB) |
| [`accuracy/`](accuracy/accuracy_pythia-14m.png) | χ_net estimator variance vs `n_hutchinson` (both methods) | pythia-14m | 1× A100 (June 2026); re-run on 1× RTX 3090 Ti (November 2026) |

Each subdirectory holds its sweep script, an sbatch job, the committed result
(`*.md` / `*.json`), and a plot (`*.png`).

## Running them

Machine-specific settings live in the repo-root `.env` (see `run_config.py` and
`.env.example`) — `HF_HOME` for the model cache and `SBATCH_PARTITION` /
`SBATCH_ACCOUNT` for SLURM. Set those once; the scripts read them.

**1. Build the shared eval batch** (needed by `ddp_scaling/` and `accuracy/`;
`single_gpu/` uses synthetic batches and skips this). Run on a login node so the
tokenizer can be fetched:

```bash
.venv/bin/python examples/benchmark/_build_eval_batch.py
```

**2a. Submit the SLURM jobs** via the wrapper, which pulls `SBATCH_*` from `.env`:

```bash
examples/benchmark/submit.sh examples/benchmark/single_gpu/single_gpu_sweep.sbatch
examples/benchmark/submit.sh examples/benchmark/accuracy/accuracy_sweep.sbatch
examples/benchmark/submit.sh examples/benchmark/ddp_scaling/ddp_scaling.sbatch   # needs a 4-GPU node
```

**2b. Or run the single-GPU sweep interactively** (e.g. on a SLURM interactive
partition):

```bash
srun -p <interactive-partition> -A <account> -N 1 --gres=gpu:1 -c 2 --mem=16G -t 00:30:00 \
  .venv/bin/python examples/benchmark/single_gpu/single_gpu_sweep.py
```

Plots are regenerated from the result tables by the `plot_*.py` script in each
subdirectory (CPU-only; needs `vatis[examples]` for matplotlib).

## What each benchmark shows

- **single_gpu** — `hutchinson` and `per_sequence_cv` produce identical
  observables at the same `n_hutchinson` (the closed-form terms are
  deterministic and the control variate is mean-zero on the seeded probe).
  Wallclock is ~linear in `n_hutchinson`. `per_sequence_cv` uses somewhat more
  peak memory than `hutchinson` (the extra per-sequence backwards), but after
  the O(P)-memory reformulation the gap is modest, and `per_sequence_cv` is
  competitive or faster at large `n_hutchinson`.
- **ddp_scaling** — vatis replicates the model on every rank and shards the eval
  batch along its first dim (data-parallel over eval data, not model size).
  Strong scaling fixes the total batch across ranks; weak scaling fixes the
  per-rank batch. The results table reports speedup, efficiency, throughput, and
  a correctness check that observables are rank-count-invariant.
- **accuracy** — the χ_net estimators are unbiased, so accuracy is governed by
  variance, which falls like ~1/√n_hutchinson. `per_sequence_cv`'s control
  variate gives lower variance than plain `hutchinson` at the same
  `n_hutchinson`, most visibly in the high-`χ_pos` regime. (In the committed
  sweep — pythia-14m at step3000, `χ_pos` ≈ 7e-8 — the two methods' relative
  spreads agree to three significant digits at every `n_hutchinson`, in the
  June file as in the November one, so that sweep does not show the reduction.)

> The `accuracy/` sweep was re-run on 17 November 2026 with the current code on
> an RTX 3090 Ti, with the settings recorded in its JSON (100 seeds, `n_h` 1–64,
> reference at `n_h = 4096`); its values are a new probe draw, and the
> conclusions (unbiased means, spread falling roughly like 1/√n) are unchanged.
> Its sbatch file says `--ref-n-h 2048`, but both the June and the November JSON
> use 4096.
>
> The `single_gpu/` and `ddp_scaling/` results were measured in June 2026 with
> the code of that time and have not been re-run since (no A100s here). Today
> `chi_net` and `chi_pos` for the same seed are a different Hutchinson draw (the
> probes are drawn on the CPU now, and every DDP rank beyond the first has its
> own probe stream; see `CHANGELOG.md`), while `chi_loss` and `delta_loss` are
> unchanged; the timings may differ as well, since drawing the probes on the
> CPU may cost time that the June runs did not spend.
>
> Numbers are hardware-specific and meant as orientation, not guarantees. For a
> consumer-GPU smoke check and the end-to-end deployment-example findings, see
> [`../BENCHMARK.md`](../BENCHMARK.md).
