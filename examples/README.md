# vatis examples

Each example is one folder: a `run.py` (or job script) that computes, an
evaluate script that draws the figures without importing vatis, the committed
outputs, and a README that says what question it answers, what it needs, how
to run it, what it shows and what it does not.

**Start with [`pythia_sweep/`](pythia_sweep/README.md).** It runs from a clean
checkout, downloads a small public model, and takes about a minute.

The Llama example and the reconstruction example need a **training bundle that
is not part of this repository**: a checkpoint bundle written by a separate
training run. Without one they cannot be run, but their committed outputs and
READMEs can still be read.

## Examples

| example | question it answers | what it needs | time (as recorded) |
|---|---|---|---|
| [`pythia_sweep/`](pythia_sweep/README.md) | How do the observables of Pythia-14m change over training, and do prose and code gradients help or hurt each other? | the public `EleutherAI/pythia-14m` checkpoints (downloaded); GPU optional | 52–55 s cached, 76 s first run (RTX 3090 Ti) |
| [`optimizer_aware_llama/`](optimizer_aware_llama/README.md) | What do `chi_pos` and its AdamW-preconditioned version look like over a real training run? | a checkpoint bundle of an 11M-parameter Llama (not in this repository) and the training repository's model code; a GPU (falls back to the CPU, slowly) | about 100 s (RTX 3090 Ti) |
| [`adamw_vs_sgd_reconstruction/`](adamw_vs_sgd_reconstruction/README.md) | Which update rule reproduces the step an AdamW run took? | a checkpoint bundle with consecutive fp32 checkpoints (not in this repository) and the training repository's model code (or your own model factory); GPU optional | 11 s (RTX 3090 Ti), 115 s (CPU) |
| [`olmo_sft_analysis/`](olmo_sft_analysis/README.md) | What do the observables look like on a 7B SFT checkpoint and real chat data, and what does that cost? | `Olmo-3-7B-Think-SFT` and its data (downloaded); one GPU with about 93 GiB; SLURM | about 210 s (1× H100) |

## Benchmarks

What vatis costs and how accurate its estimates are, measured with synthetic or
fixed batches. Index and how to run them: [`benchmark/`](benchmark/README.md).

| benchmark | question it answers | model | hardware |
|---|---|---|---|
| [`benchmark/single_gpu/`](benchmark/single_gpu/results_pythia-160m.md) | wall-clock and peak memory versus estimator, `n_hutchinson` and batch size | pythia-160m | 1× A100 (MIG slice) |
| [`benchmark/ddp_scaling/`](benchmark/ddp_scaling/DDP_SCALING_RESULTS.md) | strong and weak scaling across GPUs | pythia-1.4b | 1–4× A100 |
| [`benchmark/accuracy/`](benchmark/accuracy/accuracy_pythia-14m.png) | spread of the `chi_net` estimators versus `n_hutchinson` | pythia-14m | 1× A100 (June 2026), re-run on 1× RTX 3090 Ti (November 2026) |
| [`benchmark/update_response/`](benchmark/update_response/results_llama-11m.md) | what the rows of the step that was taken cost in wall-clock and memory | 11M-parameter Llama bundles | CPU and 1× RTX 3090 Ti |

## Research notes — `dev` branch only

Investigations that use vatis but are not examples of how to use it. They live
on the `dev` branch only; this section is removed when the examples are
promoted to `main` (see `MAINTAINING.md`).

| folder | question it answers | what it needs |
|---|---|---|
| [`spectral_tail/`](spectral_tail/README.md) | Does `chi_pos` show a transient bump when training reaches what two texts share? Negative at 14m–410m. | five Pythia sizes at 13 checkpoints each (about 34 GB of downloads) |
| [`dpo_spectral_filter/`](dpo_spectral_filter/README.md) | When is a model ready to switch from SFT to DPO, and does DPO act in the spectral tail of supervised learning? Work in progress, maintained separately. | see its README |
