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

## The analyzer config, and why

`run.sbatch` calls `analyze()` with the settings below. They are what let a 7B
model fit a single ~93 GiB GPU; the rationale is what you adapt for your own model:

- **`chi_net_method="hutchinson"`** — one backward per probe. `per_sequence_cv`
  (the small-batch default) keeps extra gradients resident and overflows the
  card at 7B; hutchinson holds one gradient at a time (~73 GB). Trade-off:
  higher variance per probe — raise `--n-hutchinson` to tighten `chi_net`.
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
  **Smaller model:** `auto` keeps everything on-GPU, and `per_sequence_cv`
  becomes affordable (lower-variance `chi_net`).
- `--no-cross-pairs` skips the cross observables; `--cross-grad-storage gpu|cpu`
  overrides the auto memory decision.
- A custom DPO loss (toward the [`../dpo_spectral_filter`](../dpo_spectral_filter/README.md)
  diagnostics) is not wired in — vatis v1 only supports the closed-form CE path
  for `chi_loss`. This example stays within that path.
