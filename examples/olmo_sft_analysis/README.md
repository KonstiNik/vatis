# Realistic SFT analysis: `Olmo-3-7B-Think-SFT`

Load Ai2's `Olmo-3-7B-Think-SFT` and compute the LNP decomposition —
`chi_loss`, `chi_net`, `delta_loss`, and the headline **`chi_pos`** (spectral
position) — on real `Dolci-Think-SFT` data: 10 code + 10 English samples,
tokenized with the model's chat template and **completion-only loss masking**
(cross-entropy on the assistant's response only — the actual SFT objective).
Wall time, peak GPU memory, and GPU engine activity are tracked throughout.

It doubles as a production-scale template: point it at your own causal-LM
checkpoint + chat data and the same config should run (see
[Adapting to your model](#adapting-to-your-model)).

## What it computes

Two eval batches (`code`, `english`) as self-pairs, plus the `(english, code)`
cross pair:

| observable | meaning |
|---|---|
| `chi_pos` (self) | spectral position of the batch's SFT loss gradient, in `[0, 1]`: →1 bulk, →0 tail. |
| `chi_pos` (cross) | parameter-space cosine between the two batches' SFT gradients, in `[-1, 1]`. |
| `delta_loss` | `‖∇θ L‖²` (self) / `⟨∇θ L_A, ∇θ L_B⟩` (cross). |
| `chi_loss`, `chi_net` | loss-curvature and Jacobian-norm factors (and normalized forms). |

On-distribution data matters: `chi_net` is the Jacobian norm *evaluated at the
input*, so evaluate on data the model actually trained on (here, its own SFT
mixture) — off-distribution tokens make the value meaningless.

## What you need

- A GPU with about 93 GiB (the committed run used one H100) and about 210 s
  for the shipped config — see [Resources](#resources-measured-1-h100-93-gib).
- The model `Olmo-3-7B-Think-SFT` and its `Dolci-Think-SFT` data, fetched once
  by `prefetch.py` on a machine with internet access.
- A SLURM cluster for the provided job script (`run.sbatch`), the `examples`
  extra for matplotlib, and `datasets` for the prefetch only (below).

## How to run

The compute nodes here are offline (`HF_HUB_OFFLINE=1` in the repo `.env`), so
it runs in two stages: an online prefetch, then offline compute.

**Prerequisites:** `HF_HOME` set (shell or `.env`, see
[`../../run_config.py`](../../run_config.py)); `uv sync --extra examples` for
matplotlib; and `datasets` for the prefetch *only* — deliberately not a vatis
dependency, so install it into the venv:

```bash
uv pip install --python .venv/bin/python datasets
```

**1. Prefetch — login node, online.** Selects 10 code + 10 English samples into
`samples.json` and warms the model into the HF cache:

```bash
.venv/bin/python examples/olmo_sft_analysis/prefetch.py
```

**2. Analyze — GPU node, offline.** Runs `analyze()` with resource tracking →
`results.parquet` + `resources.json`:

```bash
examples/benchmark/submit.sh examples/olmo_sft_analysis/run.sbatch
```

**3. Plots & tables — anywhere, CPU.** Reads the parquet + resources (zero vatis
imports) → `spectral_position.png`, `lnp_components.png`, `gpu_activity.png`:

```bash
.venv/bin/python examples/olmo_sft_analysis/analyze_results.py
```

> Optional pre-flight: `mem_profile.sbatch` measures peak GPU memory for your
> model/config before a long run — useful when adapting to a new checkpoint.

## What it shows

The three committed figures are the result of this example; the run's result
file (`results.parquet`) and resource log (`resources.json`) are not committed.
`analyze_results.py` draws the figures from those two files:

- [`spectral_position.png`](spectral_position.png) — `chi_pos` for the code and
  English self-pairs plus the cross pair `chi_pos(english, code)`.
- [`lnp_components.png`](lnp_components.png) — `chi_loss_normalized`,
  `chi_net_normalized`, `delta_loss` and `chi_pos` as small multiples.
- [`gpu_activity.png`](gpu_activity.png) — the GPU activity timeline, with the
  load and analyze phases shaded.

The measured cost of the run is in [Resources](#resources-measured-1-h100-93-gib).

## What it does not show

- Anything about how `chi_pos` changes over training: it is one checkpoint.
- More than a small sample: 10 code and 10 English samples, cut to
  `seq_len=1024` (the full code traces are about 16k tokens).
- Values from the current code: the figures were not re-run since June 2026
  (see [Resources](#resources-measured-1-h100-93-gib)).

## About the committed outputs

The three figures are from June 2026, from a run on 1× H100 with the shipped
config, and were committed in `a09c62b` (16 June 2026). The shipped config, as
`run.sbatch` and `olmo_sft_analyze.py` set it today: `chi_net_method="hutchinson"`,
`n_hutchinson=32`, `micro_batch_size=1`, bf16, `seq_len=1024`,
`cross_grad_storage="auto"`, seed 0. The run's `results.parquet` and
`resources.json` were committed with the figures and stopped being tracked in
`153bd3b`.

## The analyzer config, and why

`run.sbatch` calls `analyze()` with the settings below. They are what let a 7B
model fit a single ~93 GiB GPU; the rationale is what you adapt for your own model:

- **`chi_net_method="hutchinson"`** — one backward per probe. `per_sequence_cv`
  needs more peak memory and overflows the card at 7B; hutchinson (the vatis
  default) fits (~73 GB). Raise `--n-hutchinson` to tighten `chi_net`.
- **`micro_batch_size=1`** — one sequence per backward, the safe default at 7B.
  Peak is essentially sequence-length-independent here (it is param-bound).
- **`cross_grad_storage="auto"`** — the cross-pair dot needs *both* batches'
  parameter-space gradients; at 7B they won't co-reside on the GPU (~104 GB), so
  `auto` offloads the cached gradients to host RAM and runs the dot on the CPU
  (a one-time ~29 GB copy, negligible vs the analyze phase). Small models stay
  on-GPU automatically. Force with `--cross-grad-storage gpu|cpu`.
- **`dtype="bf16"`** — matches OLMo's training precision; gradients still
  accumulate in fp32 (needed for `delta_loss` correctness).
- **`seq_len=1024`** — this is the binding limit at 7B. Attention activations
  grow ~**O(S²)** (per-layer scores retained across the 32 layers), so peak
  climbs steeply: ~78 GB at S=2048, ~98 GB at S=4096, OOM by S=8192. Treat
  **~3k as the ceiling** on a 93 GiB card. The attention backend does *not*
  help (sdpa, the transformers default, still materializes the scores for this
  model); covering the full ~16k-token code traces needs **activation/gradient
  checkpointing** (recompute instead of retain) — a future addition.

## Resources (measured: 1× H100, ~93 GiB)

The figures and numbers below are from June 2026 and were not re-run (a 7B
model on an H100 is out of reach here). With the current code, `chi_net` and
`chi_pos` would be a different random probe draw for the same seed, and the run
may take longer, because the probes are now drawn on the CPU.

From `resources.json` / `gpu_activity.png` for the shipped config (code +
English self-pairs **plus** the cross pair, `seq_len=1024`):

- **wall:** ~210 s (model load ~18 s + analyze ~193 s).
- **peak GPU memory:** ~84 GB allocated / ~95 GB reserved — the cross pair adds
  no GPU peak because its gradients are offloaded to host RAM.
- **GPU engine activity (DCGM):** GR-engine ~82%, SM-active ~73%, DRAM
  (memory-bandwidth) ~51%, tensor-core ~14%, ~430 W (peak ~580 W) — a
  memory-bandwidth-heavy profile (autograd over the 100k-vocab head), not
  compute/tensor-core-bound. (DCGM's `FB_USED` under-reports the process; the
  `torch.cuda` peak is the authoritative memory figure.)

Tracking lives in [`resource_tracker.py`](resource_tracker.py): wall time +
per-phase peak `torch.cuda` memory + GPU engine activity sampled via
`dcgmi dmon`, with an automatic `nvidia-smi` fallback.

## Adapting to your model

- Point `prefetch.py --model / --dataset` at your causal-LM checkpoint + chat
  dataset (keep them on-distribution). `--n-per-batch`, `--seq-len`,
  `--max-prompt-frac` tune sample selection.
- **Larger model:** keep `cross_grad_storage=auto` (offload) and `hutchinson`.
  **Smaller model:** `auto` keeps everything on-GPU.
- `--no-cross-pairs` skips the cross observables; `--cross-grad-storage gpu|cpu`
  overrides the auto memory decision.
- The example uses the standard cross-entropy SFT loss — vatis v1 supports the
  closed-form CE path for `chi_loss`; custom losses are future work.
