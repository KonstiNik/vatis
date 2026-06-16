# Realistic SFT analysis: spectral position of `Olmo-3-7B-Think-SFT`

A production-scale, real-data vatis example. It loads Ai2's **`Olmo-3-7B-Think-SFT`**
checkpoint and computes the LNP decomposition — `chi_loss`, `chi_net`,
`delta_loss`, and the headline **`chi_pos` (spectral position)** — on real
supervised-fine-tuning data: 10 *code* and 10 *English* samples drawn from
**`Dolci-Think-SFT`**, the exact mixture this checkpoint was trained on. Wall
time, peak GPU memory, and GPU engine activity are tracked throughout.

## What gets computed

Each sample is tokenized with the model's **chat template** and **completion-only
loss masking** — cross-entropy on the assistant's response tokens only, the
prompt masked with `-100`. That is the actual SFT objective, so the gradient we
probe is the real SFT gradient.

The two batches (`code`, `english`) are evaluated as self-pairs, plus the
`(english, code)` cross-pair (free — the gradients are already cached):

| observable | meaning here |
|---|---|
| `chi_pos` (self) | **spectral position** of that batch's SFT loss gradient, in `[0, 1]`. →1 = gradient rides the bulk (dominant, well-resolved eigenmodes); →0 = it lives in the tail (weak, fine-grained directions). |
| `chi_pos` (cross) | parameter-space cosine between the code and English SFT gradients, in `[-1, 1]`. |
| `delta_loss` (cross) | `⟨∇θ L_english, ∇θ L_code⟩` — does an SFT step on English help (`>0`) or fight (`<0`) code? |
| `chi_loss_*`, `chi_net_*` | the loss-curvature and Jacobian-norm factors of the decomposition. |

Real, on-distribution text matters: `chi_net` is the squared Frobenius norm of
the parameter Jacobian *evaluated at the input*, so off-distribution tokens make
the observable meaningless. Using the model's own training mixture
(`Dolci-Think-SFT`) puts the evaluation on the manifold the weights actually saw.

## The two-stage offline workflow

The compute nodes on this cluster have **no internet** (`HF_HUB_OFFLINE=1` in the
repo `.env`). So everything is split into an online prefetch and an offline
compute step.

### Prerequisites

- `HF_HOME` set (shell or repo-root `.env`) — see [`../../run_config.py`](../../run_config.py).
- vatis with the examples extra (for matplotlib): `uv sync --extra examples`.
- **`datasets`** — needed *only* by `prefetch.py`. It is deliberately **not** a
  vatis dependency (vatis is dataset-agnostic); install it into the venv just
  for the prefetch step:

  ```bash
  uv pip install --python .venv/bin/python datasets
  ```

- **DCGM** for GPU-activity tracking — uses the system `dcgmi` CLI, **no Python
  package**. If `dcgmi` (or its profiling fields, e.g. on a MIG slice) is
  unavailable, the tracker automatically falls back to sampling `nvidia-smi`.

### Stage 1 — prefetch (login node, **online**)

Downloads the model into the HF cache and selects the 20 samples into
`samples.json` (so the offline step needs no dataset dependency at all):

```bash
.venv/bin/python examples/olmo_sft_analysis/prefetch.py
```

`prefetch.py` forces HF online (overriding `.env`'s `offline=1`, which is meant
for compute nodes) while keeping `HF_HOME` from `.env`. Use
`--skip-model-download` to fetch only the tokenizer + data (the ~14 GB weights
can be pulled later). Knobs: `--n-per-batch`, `--seq-len`, `--max-prompt-frac`,
`--scan-limit`.

### Stage 2 — analyze (GPU node, **offline**)

```bash
examples/benchmark/submit.sh examples/olmo_sft_analysis/run.sbatch
# or interactively on a GPU:
.venv/bin/python examples/olmo_sft_analysis/olmo_sft_analyze.py
```

Loads the cached model in bf16, builds the two completion-only batches, runs
`analyze()` wrapped in resource tracking. Writes `results.parquet` and
`resources.json`. `run.sbatch` pins `--chi-net-method hutchinson`,
`--seq-len 1024`, `--no-cross-pairs` for memory reasons — see Hardware below.

### Stage 3 — plots & tables (CPU)

```bash
.venv/bin/python examples/olmo_sft_analysis/analyze_results.py
```

Zero vatis imports — only `pyarrow` + `matplotlib`. The parquet schema and
`resources.json` are the entire contract with the compute step (the same
compute/analysis split as [`../analyze_results.py`](../analyze_results.py)).

## Resource tracking

[`resource_tracker.py`](resource_tracker.py) records three things, sliced per
phase (`load_model`, `analyze`):

- **wall time** — `perf_counter`.
- **peak GPU memory** — `torch.cuda.max_memory_allocated` / `max_memory_reserved`.
- **GPU activity** — `dcgmi dmon` in a background thread: `SM active`,
  `tensor-pipe active`, `DRAM (memory-bandwidth) active`, `GR engine active`
  (fractions), plus power / framebuffer / temperature.

Why DCGM and not just `nvidia-smi`? The coarse `utilization.gpu` is the percent
of wall time *at least one* kernel was running — on a single-stream,
backward-heavy job it pins near 100% and says nothing about how hard the GPU
works. DCGM's SM-active / tensor-active / DRAM-active counters do (they show, for
example, whether the bf16 matmuls actually light up the tensor cores and whether
the job is compute- or memory-bound).

## Hardware & runtime

Target: a single ~93 GiB GPU (measured on a capella H100). It took five runs and
a dedicated profiler ([`mem_profile.py`](mem_profile.py)) to nail the 7B memory
model. The **measured** per-batch breakdown (`mem_profile.md`):

- **delta_loss peak ≈ 75 GB, flat in `seq_len` and batch size.** It is
  *param-bound*: bf16 weights (14.6) + the fp32 flat gradient accumulator for
  `delta_loss` (29.2) + a bf16 grad tuple (14.6) + ~16 GB backward workspace.
  Activations are negligible at these sizes, so **`seq_len` is essentially free**
  (delta_loss peak: 74.7 GB at S=64, 74.8 GB at S=512).
- `chi_net` step: `hutchinson` ~44 GB, `per_sequence_cv` ~64 GB (both in
  isolation); with the live 29 GB flat-grad that becomes ~73 GB / ~94 GB.

The four earlier OOMs were **a vatis bug, not a sizing limit** (fixed in
`analyzer.py`): the analyzer cached *every* batch's full gradient
unconditionally — only cross pairs ever consume it — so the second batch ran its
own ~75 GB `delta_loss` while the first batch's ~29 GB gradient was still
resident (~104 GB → OOM). The fix caches a gradient only if a cross pair needs
it. With cross pairs off, nothing is retained and each batch peaks at ~75 GB.

So `run.sbatch` uses **`--chi-net-method hutchinson`** (keeps the chi_net step
~73 GB, under the delta_loss peak; `per_sequence_cv` + the live flat-grad would
be ~94 GB, too tight) at **`--seq-len 1024`** (free, since seq_len barely moves
peak), plus `PYTORCH_ALLOC_CONF=expandable_segments:True` and `--no-cross-pairs`.
Trade-off: hutchinson re-runs the forward per probe (slower) and is
higher-variance than the control-variate method.

Knobs: raise `--seq-len` to capture more of the long `<think>` traces — it costs
almost no memory, only wallclock (raise `--time` accordingly). `--n-hutchinson`
does **not** change peak.

**Still out of reach on one GPU:** the **cross pair** needs *two* full gradients
resident simultaneously (~104 GB) — the natural fix is to CPU-offload the cached
gradient (future work). `per_sequence_cv` is borderline (~94 GB) for the same
flat-grad-coexistence reason.

**GPU activity sampler:** `dcgmi` requires a running host engine and is not
accessible on capella, so the tracker falls back to `nvidia-smi` (coarse
util% / memory / power). Peak memory still comes from `torch.cuda`, which is the
number that matters most here.

## Outputs (next to the scripts)

- `samples.json` — the 20 selected chat samples + selection metadata (from prefetch).
- `results.parquet` — canonical long-format LNP table.
- `resources.json` — time / memory / GPU-activity summary + per-sample timeline.
- `spectral_position.png`, `lnp_components.png`, `gpu_activity.png`.

## Findings

First successful run: `Olmo-3-7B-Think-SFT` on 10 code + 10 English
`Dolci-Think-SFT` samples, `S=1024`, `hutchinson` `n=32`, self-pairs, 1× H100.

| observable | code | english |
|---|---|---|
| **`chi_pos` (spectral position)** | 1.11e-9 | 1.65e-9 |
| `delta_loss` (‖∇θ L‖²) | 2.95 | 4.38 |
| `chi_loss_normalized` | 0.322 | 0.323 |
| `chi_net_normalized` | 8.25e9 | 8.22e9 |
| `chi_loss` | 4.03e-5 | 4.19e-5 |
| `chi_net` (Tr eNTK) | 6.58e13 | 6.34e13 |

- **The SFT loss gradient sits deep in the spectral tail.** `chi_pos ≈ 1e-9` for
  both batches — essentially the bottom of `[0, 1]`. For a *fully* SFT-trained
  checkpoint this is exactly what the [`dpo_spectral_filter`](../dpo_spectral_filter/README.md)
  hypothesis predicts: late SFT has exhausted the bulk, so the CE gradient lives
  in its own tail (`chi_pos(CE, CE)` small). The smallness is dominated by the
  enormous `chi_net` (Tr eNTK ≈ 6.6e13); with `hutchinson n=32` carrying variance
  on `chi_net`, treat the absolute `chi_pos` as order-of-magnitude, not 3-sig-fig.
- **Code and English are near-identical in `chi_loss`/`chi_net`** and differ
  mainly in `delta_loss` (English's gradient is larger). The more interpretable
  *cross*-distribution alignment (`chi_pos(english, code)`, the parameter-space
  gradient cosine) is the natural next measurement — it needs the cross pair,
  hence the memory work below.
- **Resources (1× H100):** wall 195 s (load 18 s + analyze 177 s); **torch peak
  84 GB allocated / 94 GB reserved** (fits the 93 GiB card). DCGM shows the run is
  **autograd/memory-bound, not tensor-core-bound**: GR-engine ~86%, SM-active
  ~54%, tensor-core ~0%, DRAM ~29%, ~276 W. (DCGM's `FB_USED` under-reports here
  (~1.7 GB) — trust the `torch.cuda` peak for memory.) The real-run peak (84 GB)
  is a bit above the per-step profiler figure (~75 GB) because `chi_loss`,
  `delta_loss`, and `chi_net` overlap within one `_compute_self_pair` at S=1024.

**Next steps:** (1) the cross pair + `per_sequence_cv` need two gradients
resident — CPU-offload the cached gradient to fit; (2) longer sequences are cheap
on memory but cost wallclock; (3) higher `n_hutchinson` to tighten `chi_net`.

## Notes & knobs

- **Sample selection** is deterministic: the first `--n-per-batch` code samples
  (sources matching `Code|Python|Algorithms|Nemotron`) and English samples
  (curated natural-language chat sources, e.g. `Persona Precise IF`) whose prompt
  fits in `--max-prompt-frac × seq_len` so real completion tokens survive
  truncation. The selected `id`s and lengths are recorded in `samples.json`.
- **The model is the SFT-stage Think checkpoint** (`Olmo-3-7B-Think-SFT`),
  evaluated on its own training mixture. Point `--model` / `prefetch.py --dataset`
  elsewhere (e.g. `Olmo-3-7B-Instruct-SFT` + `Dolci-Instruct-SFT`) to compare
  branches; keep model and data on-distribution for the observables to mean
  anything.
- A custom **DPO** loss (the next step toward the `dpo_spectral_filter`
  diagnostics) is not wired in here — vatis v1 only supports the closed-form CE
  path for `chi_loss`. This example deliberately stays within that supported path.
