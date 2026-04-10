# vatis

Compute LNA observables (`chi_loss`, `chi_net`, `chi_pos`) on **pretrained model checkpoints**, scaled across multiple GPUs via DDP. Sister package to `perspic`, which does the same thing during Lightning training. vatis is for the regime where you can't pretrain yourself and instead probe published checkpoints (Pythia, OLMo, …).

## Agent quick start

If you're an agent walking into this repo, read in this order:

1. **`SESSION_SUMMARY.md`** — authoritative state of the repo right now: what's built, what works, what's known broken, what was last touched. This is the file that bridges sessions.
2. **`TASKS_NEXT.md`** — prioritized work order for the current development phase. Contains hard constraints, the git workflow, and per-task instructions. **Do not start work without reading this file** — it's the contract for what the current session is about.
3. **`## Working conventions`, `## Conventions`, and `## Known issues` below** — read before running anything that touches files outside the project tree, modifies git history, or installs packages. The "Known issues" section names the latent bugs you should not be surprised by.
4. **The rest of this file** — the durable reference for math, architecture, scope. The "v1 build history" and "v1.1 hardening pass" sections at the bottom describe past phases; read them for context but don't try to re-execute the implementation order — that work is done.

If you contradict a paragraph in this file with the code you write, **edit the paragraph in the same commit**. Don't leave stale text. The "v1.1 hardening pass" learned this the hard way and ended up with a `CLAUDE.md.proposed` sibling file that sat dormant for an entire session.

## Core idea (math)

For a loss `L = (1/N) Σᵢ ℓᵢ` over `N` "samples" (= valid tokens for an LM), the LNA decomposition factors the linearized loss change:

```
δL(A, B) = χ_loss · χ_net · χ_pos
```

with

- `χ_loss = ‖∇_f L‖²`  — closed form from logits, **no backward through θ**.
- `χ_net  = ‖∇_θ f‖_F² = Tr(Θ)` where `Θ = ∇_θ f (∇_θ f)^T` is the eNTK.
- `χ_pos = δL / (χ_loss · χ_net)` — recovered indirectly, no eNTK eigendecomposition.
- `δL(A, B) = ⟨∇_θ L^A, ∇_θ L^B⟩` — two ordinary backwards plus a dot product.

The full derivation lives in `background_info.tex` (LNA-decomposition for mini-batch training). Read it before touching the math.

### How `χ_loss` and `δL` are computed (cheap, fixed cost)

These two are the same regardless of which `χ_net` estimator we use.

- `χ_loss` for cross-entropy is closed form from logits:
  ```
  ∂ℓ_{b,s} / ∂z_{b,s,v} = (softmax(z_{b,s})_v − onehot(y_{b,s})_v) / N_valid
  χ_loss = Σ_{b,s,v ∈ valid} (∂L/∂z_{b,s,v})²
  ```
  Zero backward passes through θ — pure tensor op on logits. For other losses we call `torch.autograd.grad(L, logits)`, which is one cheap backward through the loss head only.

- `δL(A, B)`: one full backward of `L^A` w.r.t. θ → flatten gradient `g_A`. Same for `g_B`. Then `δL = g_A · g_B`. **Two backwards total**, regardless of granularity. For the self case `δL(A, A) = ‖g_A‖²`, only one backward is needed.

Total fixed cost per `(checkpoint, eval-batch-pair)`: **2 backwards** (or 1 for self). Everything else goes into `χ_net`.

### `χ_net` estimation: three methods

`χ_net = Σ_b Tr(M_b)` where `M_b = J_b J_bᵀ` and `J_b = ∇_θ f(x_b)` is the per-sequence Jacobian. This is **the only expensive observable**, and the reason vatis exists. We support three methods, selected by the user (or auto-picked based on batch size and model compatibility). They share the same outer loop and produce the same observable; they differ only in *how* `Tr(M_b)` is estimated.

The math reason there's more than one method: a single VJP is a projection (`g = Jᵀ v` collapses information), so trace estimation needs either (a) deterministic basis vectors over the full output dim — exact but `O(S·V)` backwards, infeasible for LMs — or (b) random probes (Hutchinson) — `O(n)` backwards with controllable variance. The methods below differ in *how the random probes are arranged across the batch*.

#### Method 1: `HutchinsonEstimator` (default for large `B`)

Pure plain-torch Hutchinson at micro-batch level. For each micro-batch (sequences `[b₁, ..., b_M]`), draw `n_h` independent Rademacher vectors `v ∈ R^(M·S·V)`, each with the same shape as the micro-batch output. Per draw:

```python
out = model(x_micro)              # (M, S, V)
projected = (out * v).sum()       # scalar
g, = torch.autograd.grad(projected, model.parameters())
chi_net_accumulator += g.pow(2).sum()  # accumulates Σ_b ‖J_bᵀ v‖²
```

Cost: `n_h` backwards per micro-batch. No opacus, no vmap, no per-sample machinery. Variance is `~B× higher per draw` than the opacus path because the same `v` doesn't get reused across samples for free.

**When to use:** large `B` (≥ 32), or when other methods don't apply. This is the always-works fallback.

#### Method 2: `PerSequenceControlVariateEstimator` (default for small `B`)

For each sequence `b`, do **one extra backward** of `L_b` (per-sequence loss) w.r.t. `θ`. This gives `g_b = J_bᵀ u_b` where `u_b = ∇_f L_b` is known in closed form. From `(g_b, u_b)` we get exactly:

- `‖g_b‖²` — a Rayleigh quotient of `M_b` along the loss direction (free, deterministic).
- `⟨g_b, g_{b'}⟩` for all `b, b'` — the cross-sample alignment matrix, a bonus observable.

These are used as a **control variate** for the Hutchinson estimate of `Tr(M_b)`: the loss-direction projection is correlated with the trace, known exactly, has zero variance, and subtracting its noisy Hutchinson counterpart and adding back the exact value reduces total variance — substantially when the loss direction overlaps the top eigenmodes of `M_b`, which is exactly the high-`χ_pos` regime we care about.

Cost per (ckpt, batch): `n_h + B` backwards. Linear in `B`, so this method becomes unattractive for `B > ~64`.

**Bonus output:** the cross-sample alignment matrix `C_{bb'} = ⟨g_b, g_{b'}⟩` is written to the sink as an additional observable. It's exactly the eNTK projected onto the loss direction, sample-resolved. Useful for sample-level variance / interference analysis (the negative-`χ_pos` regime per `background_info.tex §A.2`).

**When to use:** moderate `B` (≤ 32), or when the user wants the per-sample alignment matrix.

#### Method 3: `OpacusEstimator` (opt-in fast path, deferred to v1.2)

When the model contains only opacus-supported layers and has no parameter tying, opacus + ghost clipping computes per-sample gradient norms in one backward, packing `B` samples per pass. With Hutchinson over `(S, V)` only, it's the fastest method **per backward count** — but the wall-time win only materializes when you can fit `M > 1` samples per micro-batch, which for an 8B model on an 80GB GPU is borderline. For smaller models (1B and below) it's a clear ~`B`× speedup.

Cost: `~n_h` backwards, where `n_h` is `~10–32`. Requires layer-compatibility check at startup; falls back with a clear error message if the model has tied embeddings (Pythia!), unsupported layers, or in-place ops that can't be neutralized.

**Status:** stub only in v1 and v1.1 — `vatis/core/chi_net/opacus.py` raises `NotImplementedError`. Originally listed as "v1.1 work" but the v1.1 hardening pass focused on trust and demonstrability instead of features. Full implementation is now Tier 2 in `TASKS_NEXT.md` (v1.2). When implemented, follow the pattern from `perspic/calculator/samplewise_opacus.py`.

#### Auto-selection rule

```python
def select_chi_net_method(B_total: int, requested: str | None) -> str:
    if requested is not None:
        return requested  # user override always wins
    if B_total <= 32:
        return "per_sequence_cv"
    return "hutchinson"
```

`opacus` is never auto-selected in v1.

### Compute scaling

Two reference points: order-of-magnitude **estimates** at the 8B / A100 production target, and a **measured** single-GPU smoke test on `pythia-14m`. Keep both. The 8B numbers are still the right production target; the 14m numbers are what an agent can actually verify on the available hardware.

#### 8B / A100 80GB / 4 GPUs (estimated, never measured here)

With `micro_batch_size=4` (activation checkpointing on, near memory ceiling), per-rank backward ≈ 1.8 s, ~25k tok/s/GPU. Per `(ckpt, eval-batch)` cost for `B=32`:

| method | backwards | wallclock | notes |
|---|---|---|---|
| `hutchinson` (n≈320, B-corrected) | ~320 | ~10 min | always applicable |
| `per_sequence_cv` (n≈32, +B=32) | ~64 | ~2 min | also gives cross-sample matrix |
| `opacus` (n≈10) | ~12 | ~22 s | not applicable to Pythia (tied embeddings) |

For a 50-checkpoint × 3-eval-batch sweep (150 pairs):

| method | total wallclock |
|---|---|
| `hutchinson` | ~25 h |
| `per_sequence_cv` | ~5 h |
| `opacus` | ~55 min |

These numbers are order-of-magnitude estimates — actual perf depends heavily on the exact model, achievable micro-batch, and disk/network speed for checkpoint loading. For 1B-class models everything is 5–10× faster and `hutchinson` becomes interactive.

The real bottleneck for long sweeps is often **checkpoint loading**, not compute (~30 s/checkpoint cached, more from network). Cache aggressively; if multiple ranks share a node, share the HF cache directory.

#### pythia-14m / RTX 3090 Ti / 1 GPU (measured, v1.1 hardening pass)

Reference point from the v1.1 deployment example. All numbers use fp32 forward, `S=64` for the benchmark sweep / `S=128` for the deployment example, single GPU, no DDP, same fixed seed. The full 18-row table is in `examples/BENCHMARK.md`; the rows below are the most informative slice.

| method            | n_h | B  | wall_s | peak_mb | notes                                |
|-------------------|----:|---:|------:|--------:|---------------------------------------|
| hutchinson        |   8 |  4 |  0.07 |     567 | toy upper bound for variance-vs-cost  |
| hutchinson        |  32 |  4 |  0.24 |     567 | sensible default at small B           |
| hutchinson        | 128 |  4 |  0.96 |     567 | n_h plateau on this fixture           |
| per_sequence_cv   |  32 |  4 |  0.22 |    1194 | matches hutchinson at this B          |
| hutchinson        |  32 |  8 |  0.32 |     885 |                                       |
| per_sequence_cv   |  32 |  8 |  0.35 |    1875 | the example's auto-pick (B≤32)        |
| hutchinson        |  32 | 16 |  0.38 |    1529 |                                       |
| per_sequence_cv   |  32 | 16 |  0.64 |    3280 | per-sample grad cache visible here    |

For the deployment example as committed (`per_sequence_cv`, n=32, B=8, S=128, 13 revisions × 2 self pairs + 1 cross pair = 39 measurements), end-to-end wallclock including model loading is **~20 s with a warm HF cache** and **~50 s on first run** (the 4 log-spaced early checkpoints add ~30 s of cold downloads). The dominant cost is checkpoint loading, not analysis.

**Reading guidance:** use the 3090 Ti table only as a smoke check. If your toy or 1B-class model runs at ~10× the wallclock of the rows above, your micro-batch is probably wrong. Use the 8B/A100 estimates above as the production target.

### Sample unit and normalization

There is **one** observable, computed at token granularity, normalized by the number of valid (non-pad, non-ignored) tokens. The per-sequence vs per-token dichotomy is a normalization choice, not a different quantity:

- `χ_loss` and `χ_net` are sums over `(b, s, v)` regardless. Frobenius norms of Jacobians decompose additively across output dims.
- The `√|A|·|B|` factors from `background_info.tex §A.2` are batch-size normalization. We use **valid token counts**, not sequence counts. Track `n_valid_A`, `n_valid_B` per call.
- Normalized observables (default output):
  ```
  χ̃_loss = √(N_A · N_B) · χ_loss
  χ̃_net  = χ_net / √(N_A · N_B)
  ```
  where `N_A`, `N_B` are valid token counts. Both unnormalized and normalized values are written to the result sink so downstream analysis can pick.

## Scope (v1 / v1.1, current)

- **In:** HuggingFace causal-LM checkpoints (Pythia, OLMo, GPT-NeoX class). Tokenized data already in model-ready form. DDP across user-specified `num_gpus`. Two `χ_net` estimators (`hutchinson`, `per_sequence_cv`) with auto-selection. Parquet result sink (canonical) plus optional W&B and TensorBoard. CLI via `python -m vatis run …` that wraps `torchrun`. End-to-end deployment example on `pythia-14m` with cross-pair observables (added in v1.1).
- **Out (current):** Tokenization (the example tokenizes inline; vatis core does not provide a tokenizer). Checkpoint discovery / sweep scheduling. FSDP. Tensor / pipeline parallelism. `OpacusEstimator` (stub only — full implementation deferred to v1.2). Custom non-CE losses for `chi_loss` (deferred to v1.2). Per-sample variance observables beyond the cross-sample alignment matrix. Models that don't fit on one GPU.

Adding any of these later should be straightforward because the core (`vatis/core/`) is decoupled from model loading, data plumbing, and parallelism. The v1.2 work order in `TASKS_NEXT.md` lists the specific items that have been triaged for the next phase.

## Architecture

```
vatis/
├── __init__.py            # public API: analyze(), Analyzer
├── core/
│   ├── observables.py     # chi_loss, delta_loss, chi_pos (combine χ_net from estimator)
│   ├── chi_net/
│   │   ├── base.py        # ChiNetEstimator ABC
│   │   ├── hutchinson.py  # HutchinsonEstimator (always-works fallback)
│   │   ├── per_seq_cv.py  # PerSequenceControlVariateEstimator (default for small B)
│   │   └── opacus.py      # OpacusEstimator (stub in v1; full impl deferred to v1.2)
│   ├── probes.py          # Rademacher / Gaussian probe vector generation, masking
│   └── normalization.py   # valid-token counting, √(N_A·N_B) factors
├── models/
│   ├── __init__.py        # load_model(spec) dispatch (registry-ready, single impl for now)
│   └── hf.py              # load_hf_model(name, revision) → nn.Module + loss_fn
├── data/
│   ├── batches.py         # EvalBatchSpec: "fixed" | "resample" | "callable"
│   └── collate.py         # micro-batch splitting, attention-mask aware
├── distributed/
│   ├── launcher.py        # `python -m vatis run` → torchrun wrapper
│   ├── ddp.py             # init_process_group, all_reduce of accumulators
│   └── sharding.py        # split eval batch across ranks; pad-and-mask if uneven
├── sinks/
│   ├── base.py            # ResultSink ABC
│   ├── parquet.py         # canonical: long-format rows
│   ├── wandb.py           # optional
│   └── tensorboard.py     # optional
├── analyzer.py            # Analyzer class: orchestrates checkpoint × eval_batch loop
├── cli.py                 # argparse / typer entrypoint
└── _version.py
```

**Deployment example tree** (added in v1.1, lives next to the package):

```
examples/
├── pythia_sweep.py        # canonical example: 13 pythia-14m revisions × prose + code
├── analyze_results.py     # post-processing of results.parquet — derived observables
├── BENCHMARK.md           # benchmark sweep table + budget arithmetic
├── _probe.py              # one-shot probe used to size the example
├── _benchmark.py          # 18-row benchmark sweep used to fill BENCHMARK.md
├── results.parquet        # canonical long-format output of pythia_sweep.py
├── chi_loss.png           # generated by pythia_sweep.py
├── chi_net.png
├── delta_loss.png
├── chi_pos.png
└── cos_similarity.png     # generated by analyze_results.py
```

The deployment example is the canonical "does vatis work end-to-end on real weights" check. After any significant change to the analyzer or an estimator, **re-run `examples/pythia_sweep.py` and check the plots haven't moved unexpectedly**. The plots are an integration test the unit suite can't replicate.

### Public API

```python
from vatis import analyze

results = analyze(
    model="EleutherAI/pythia-160m",
    revisions=["step1000", "step2000", "step10000"],
    eval_batches={
        "val_fixed":   fixed_batch,             # tensor / dict, frozen
        "val_resample": val_dataloader,         # DataLoader, redrawn per checkpoint
        "train_at_step": lambda ckpt: ...,      # callable(checkpoint_id) -> batch
    },
    observables=["chi_loss", "chi_net", "chi_pos", "delta_loss"],
    chi_net_method=None,                        # None = auto: per_seq_cv if B≤32 else hutchinson
    n_hutchinson=32,                            # interpretation depends on method
    hutchinson_distribution="rademacher",       # or "gaussian"
    micro_batch_size=1,                         # safe default; user can grow it
    num_gpus=4,
    distributed="ddp",                          # only option for v1
    sink="results.parquet",                     # str → ParquetSink; or list of sinks
    dtype="bf16",                               # forward dtype; grads in fp32
)
```

`analyze()` is the one-shot convenience. The underlying `Analyzer` class is exposed for users who want to drive the loop themselves (e.g. interleave with their own logging).

### EvalBatchSpec

```python
from vatis.data import EvalBatchSpec

EvalBatchSpec.fixed(batch)            # frozen tensor or dict, reused at every checkpoint
EvalBatchSpec.resample(dataloader)    # next(iter(loader)) at each checkpoint
EvalBatchSpec.callable(fn)            # fn(checkpoint_id) -> batch
```

The `eval_batches` argument to `analyze()` is just `dict[str, EvalBatchSpec | Tensor | DataLoader | Callable]` — bare values are auto-wrapped. Multiple named eval batches are computed at every checkpoint and emitted as separate rows in the sink. This is the central feature: comparing observables across eval distributions at the same checkpoint is the primary use case.

For `δL(A, B)`, the user can specify `pairs=[("val_fixed", "train_at_step"), …]` to compute cross-batch alignments. By default `δL` is computed only for the "self" pair `(b, b)` for each named batch.

### Result format (parquet, long)

One row per `(checkpoint_id, eval_batch_name_A, eval_batch_name_B, observable, value)`. Long format because:
- It handles arbitrary numbers of eval batches and pair combinations without schema churn.
- Trivial to load into pandas and pivot.
- Append-friendly across DDP ranks (rank 0 writes after all-reduce).

Schema:
```
checkpoint_id      str
revision           str
batch_a            str
batch_b            str           # equals batch_a for self-observables
n_valid_a          int64
n_valid_b          int64
observable         str           # "chi_loss" | "chi_loss_normalized" | "chi_net" | ...
value              float64
n_hutchinson       int32         # null if not applicable
hutchinson_seed    int64         # null if not applicable
wallclock_s        float64
```

### Distributed semantics

- User passes `num_gpus=N`. CLI launches `torchrun --nproc_per_node=N -m vatis._worker …`.
- Each rank loads the same model checkpoint (v1 assumes the model fits on one GPU).
- Each eval batch is sharded across ranks along the batch dim. If `B % N != 0`, the last rank gets a smaller shard and contributes fewer valid tokens — the all-reduce handles this naturally because we accumulate token-weighted sums, not means.
- Per micro-batch on each rank: forward, compute logits → `χ_loss` accumulator. For `χ_net` Hutchinson: draw `n` probe vectors **with the same seed across ranks** so each rank computes a partial estimate of the same trace; all-reduce sums them. For `δL`: each rank computes its partial gradient `g_A^rank`, all-reduce-sum to get the full `g_A`, dot with `g_B` (same procedure).
- All-reduce happens **once per checkpoint × eval-batch** combination, not per micro-batch. Accumulators live on each rank as plain tensors.
- `micro_batch_size=1` is the always-safe fallback. The launcher does **not** attempt OOM auto-tuning in v1 — it's the user's responsibility to pick a size that fits.

### Probe vector details

- Default `n_hutchinson=32`, Rademacher distribution. Variance scales as `1/n`. For `HutchinsonEstimator` the effective variance is `~B×` higher per draw, so the auto-selection uses `per_sequence_cv` for small `B`.
- Probe seed is **shared across ranks** within one `(checkpoint, batch)` call so all ranks compute partial estimates of the same probe; different seed across calls.
- The probe vector shape matches the local micro-batch output: for HF causal LMs, `(M, S, V)`. **Padding positions are zeroed** in the probe vector so masked tokens contribute to neither `χ_net` nor `χ_loss`.
- No exact-vocab fallback in v1. If users want a deterministic check on a tiny test model, they crank `n_hutchinson` up.

### Loss handling

vatis ships with the standard HF causal-LM CE loss. **v1 only supports the closed-form CE path for `chi_loss`.** The bundle exposes a `loss_fn(logits, batch) -> scalar` slot, and the analyzer uses it for `delta_loss` (autograd through θ), but `chi_loss` always goes through the closed-form softmax-minus-onehot shortcut regardless of what `loss_fn` is set to. If you pass a non-CE `loss_fn`, `delta_loss` will be correct but `chi_loss` will be wrong.

The bundle's `valid_mask_fn` and the `labels=-100` (`ignore_index`) sentinel are **intersected** inside the `chi_loss` accumulator: positions excluded by either are excluded from both the numerator and the `n_valid` denominator, so the two paths are guaranteed to count the same set. (Fixed in v1.2 task 1; before that, the numerator used only `labels != -100` while the denominator used `vmask.sum()`, so chi_loss was silently wrong whenever the two masks disagreed.)

The v1.1 hardening pass deleted the partially-implemented autograd fallback (`chi_loss_from_autograd`) precisely to avoid silent wrong-answer modes — better to fail loudly on a documented restriction than to ship a half-working path. **Custom loss support is deferred to v1.2** (see `TASKS_NEXT.md`); the building blocks are simple (one extra `torch.autograd.grad(loss, logits)` plus a masked squared sum) but plumbing them through the analyzer cleanly is its own task.

### Precision

- Forward in `bf16` by default (matches Pythia/OLMo training precision).
- Gradients accumulated in `fp32`. Hutchinson probes drawn in `fp32`, cast to forward dtype for the backward, gradient norm computed in `fp32`.
- `dtype="fp32"` is available for tiny test models.

## Research workflow

The deployment example (`examples/pythia_sweep.py` + `examples/analyze_results.py`) demonstrates the canonical vatis usage pattern, which has two distinct phases that should stay separated:

**Phase 1: compute** (`pythia_sweep.py`). Loads model checkpoints, runs `analyze()`, writes `results.parquet`. This is the expensive step: model loading dominates wallclock, and re-running it for every analysis tweak is wasteful. The script does this once per checkpoint sweep.

**Phase 2: analysis** (`analyze_results.py`). Reads `results.parquet`, computes derived observables (e.g. normalized cross-batch gradient correlation `cos(g_A, g_B) = δL(A, B) / sqrt(δL(A, A) · δL(B, B))`), produces additional plots and printed tables. This script has **zero vatis imports** — only `pyarrow` and `matplotlib`. The parquet schema is the contract between vatis and any downstream consumer; making the analysis script consume only the parquet enforces that contract by example.

The split matters because compute is expensive (~20 s warm / ~50 s cold for the example sweep) and analysis is cheap (sub-second). Iterating on derived quantities, plot styles, or normalizations should not require re-running the analyzer. Any user code that touches a vatis output should look like `analyze_results.py`: `pyarrow` in, plot/table out.

A few patterns the example also demonstrates:

- **Real text, not random integers, for any observable that gets discussed.** The chi_net term is the squared Frobenius norm of the parameter Jacobian *evaluated at the input*; off-distribution random tokens put the evaluation at a meaningless point in input-space. Tokenized real text (Pride and Prejudice prose + a Python module in the example) puts the evaluation back on the model's training manifold. Synthetic data is fine for shape-only benchmarks (`examples/_benchmark.py` is correctly using random integers because it's measuring wallclock and memory only) but anything that reports an observable value needs real data.
- **Cross-pair observables for the LNA-relevant story.** `cross_pairs=[(A, B)]` adds `δL(A, B)` and `chi_pos(A, B)` rows to the parquet at zero extra backward cost (the gradients are already cached from the self-pair work). The cross observables are the most informative LNA quantities — they tell you whether learning on A helps or hurts B (positive `δL(A, B)` → transfer, negative → interference). The deployment example uses `(prose, code)` and shows the cross `δL` going from `+3.25` at step1 → `−18` at step143000 — i.e. prose and code gradients become anti-aligned during training, the negative-`chi_pos` regime from `background_info.tex §A.2`.
- **Log-spaced early checkpoints.** Pythia ships log-spaced revisions (`step0`, `step1`, `step2`, ..., `step512`) before the every-1000-steps main phase. The early phase reveals dynamics that are invisible from `step1000` onward — for `pythia-14m` the example surfaces a `chi_net` U-shape at step64 and a `chi_loss` "warmup cliff" where the model literally doesn't improve in the first 64 SGD steps. **Always include early checkpoints when sweeping a Pythia-like model**; the cost is ~4 extra checkpoint loads.

## Dependencies

Hard:
- `torch >= 2.2, < 2.11` — upper bound exists because `uv add matplotlib` in the v1.1 hardening pass silently upgraded torch to 2.11 and broke the resident `torchvision 0.23` build. The bound is loose; if a future maintainer upgrades `torchvision` manually they can drop it.
- `transformers >= 4.40`
- `pyarrow` (parquet sink)
- `numpy`
- `matplotlib >= 3.10` — only used by the deployment example (`examples/pythia_sweep.py`, `examples/analyze_results.py`). vatis core does not import matplotlib.

Soft (extras):
- `wandb` → `vatis[wandb]`
- `tensorboard` → `vatis[tensorboard]`
- `pytest`, `ruff`, `mypy` → `vatis[dev]`

**No** `perspic`, `opacus`, `functorch`, or `pytorch-lightning`. The math is reimplemented in `vatis/core/observables.py` (~50 lines). This keeps vatis lightweight and lets it evolve independently of perspic. (`perspic` is installed in the project's `.venv/` for cross-validation testing only — the cross-validation suite is gated behind `pytest -m cross_validation` and is skipped in CI.)

## Tooling

- **Package manager:** `uv`. `pyproject.toml` is the single source of truth.
- **Python:** `>= 3.11`.
- **Lint + format:** `ruff` (for both — `ruff check` and `ruff format`). Black-compatible style. Drop-in for black, much faster, one tool.
- **Types:** `mypy --strict` on `vatis/`. Tests are unchecked.
- **Tests:** 85 total (50 unit + 31 cross-validation + 4 integration). Run via `.venv/bin/python -m pytest tests`. CI runs the unit tier only.
  - `tests/unit/` (50 tests) — runs in CI. Uses a hand-built ~1M-param toy transformer (defined in `tests/fixtures/tiny_transformer.py`). Tests math correctness: Hutchinson convergence to the exact trace as `n → ∞`, `χ_loss` closed-form correctness on the supported CE shapes, `δL` symmetry, normalization invariants, the chi_pos combinator, the per-seq-CV memory pre-check, and the analyzer's contract assertions. v1.1 added exact-NTK ground-truth tests for `chi_pos` and `δL(A, B)` via an explicit per-token Jacobian builder (`_build_full_jacobian_and_u` in `tests/unit/test_chi_net.py`) — this is the most rigorous correctness reference because it bypasses Hutchinson entirely. v1.2 task 1 added a regression test for the `valid_mask_fn`/`chi_loss` intersection fix.
  - `tests/integration/` (4 tests) — gated behind `pytest -m integration`, **skipped in CI**, runnable locally. Uses `EleutherAI/pythia-14m` (smallest published Pythia) to verify the full HF loading + DDP path on real checkpoints. The DDP test is currently a 2-rank CPU loopback only — **the multi-GPU DDP path has never been validated on real GPUs**. Adding a real-GPU DDP test is on the v1.2 work order.
  - `tests/cross_validation/` (31 tests) — gated behind `pytest -m cross_validation`, **skipped in CI** (requires perspic in the environment), runnable locally. **Cross-validates every observable against perspic** as the ground-truth reference implementation — both self-pair and cross-pair. The protocol:
    1. Build a small model that perspic can wrap (a `pl.LightningModule` with `criterion`) — e.g. a tiny MLP or the toy transformer with a Lightning shim.
    2. Run `perspic.analyzer(...)` on a single training step with a fixed batch and seed, capturing the logged `chi_loss`, `chi_net`, `chi_align`, `chi_coup`, and `grad_norm_squared` (= `δL`).
    3. Run `vatis.analyze(...)` on the **same model weights, same batch, same seed**, with the requested `chi_net_method` and an `n_hutchinson` large enough to drive the Hutchinson noise below the asserted tolerance.
    4. Assert that `vatis.chi_loss ≈ perspic.chi_loss`, `vatis.chi_net ≈ perspic.chi_net` (within Hutchinson std), `vatis.delta_loss ≈ perspic.grad_norm_squared`, and **`vatis.chi_pos ≈ perspic.chi_align ≈ perspic.chi_coup`**.
    5. Repeated with `chi_net_method="per_sequence_cv"` to verify the control-variate path agrees.
    6. Repeated with both `opacus` and `functorch` perspic backends to make sure normalization conventions match across the board.

    v1.1 added a tight-tolerance test (`test_all_observables_tight_tolerance_at_n16384`) at relative 0.3% / absolute 1e-6 with `n_hutchinson = 16384`, plus a heavy-padding test (`test_heavy_padding_matches_perspic_on_valid_subset`) that drives 5/8 samples through `labels=-100` and asserts the padded vatis run agrees with perspic on the unpadded subset.

    v1.2 added **cross-pair cross-validation** (17 tests), closing the gap where only self-pair observables (`δL(A,A)`, `chi_pos(A,A)`) were validated against perspic. The cross-pair tests use perspic's `Linearizer.compute(x1, y1, x2, y2)` for cross `δL(A,B)` and `SamplewiseCalculator.compute_cross_metrics()` for the geometric-mean `chi_loss_cross` / `chi_net_cross`, then `CouplingCalculator` for `chi_coup_cross`. Coverage:

    - **Basic cross `δL` and `chi_pos`** (`test_cross_delta_loss_matches_perspic`, `test_cross_chi_pos_matches_perspic`): two batches from different seeds, both perspic engines × both vatis methods. `δL` is deterministic (rel 1e-4); `chi_pos` within 2% (Hutchinson noise via geometric-mean `chi_net`).
    - **Tight tolerance at n=16384** (`test_cross_observables_tight_tolerance_at_n16384`): catches constant-factor normalization bugs hidden by the 2% bound.
    - **Heavy-padding cross** (`test_cross_heavy_padding_matches_perspic`): asymmetric masking (3/8 valid in A, 5/8 valid in B) verifies that the gradient cache doesn't leak masked positions into the cross dot product.
    - **Asymmetric batch sizes** (`test_cross_asymmetric_batch_sizes`): batch A has 6 samples, B has 4. Catches bugs in `normalization_factor(N_A, N_B)` that equal-sized batches would miss.
    - **Symmetry** (`test_cross_symmetry_delta_loss_and_chi_pos`): asserts `δL(A,B) == δL(B,A)` and `chi_pos(A,B) == chi_pos(B,A)` at bit-level precision. Vatis-internal (no perspic needed) — catches ordering bugs the perspic comparison can't.
    - **Explicit geometric-mean check** (`test_cross_geometric_mean_chi_loss_chi_net_match_perspic`): directly compares `chi_loss_normalized` and `chi_net_normalized` from the cross rows against perspic's `compute_cross_metrics` output. Catches reciprocal-factor bugs that would cancel in the `chi_pos` ratio.

    **Naming map for the cross-validation tests:**

    | perspic | vatis |
    | --- | --- |
    | `chi_loss` | `chi_loss_normalized` |
    | `chi_net` | `chi_net_normalized` |
    | `chi_align` / `chi_coup` | **`chi_pos`** |
    | `grad_norm_squared` | `delta_loss` |

    **Any disagreement larger than the Hutchinson noise floor is a bug in vatis** (perspic is the reference). If a change you make requires loosening one of these tolerances to pass, that's a red flag — investigate before loosening.
- **CI:** GitHub Actions workflow at `.github/workflows/ci.yml`. Single matrix entry `python-3.11`, runs `ruff check`, `ruff format --check`, `mypy`, `pytest -m "not integration and not cross_validation"` against `uv sync --extra dev`. Triggers on push to any branch and PRs to `master`. Added in v1.2 task 2.

## Working conventions (for the agent building this)

- **Use subagents where it makes sense.** Spawn `Explore` subagents for codebase searches that span more than 2–3 files (e.g. "find all places perspic computes χ_loss" before writing the cross-validation tests). Spawn `Plan` subagents for non-trivial implementation steps where the design isn't obvious from CLAUDE.md alone. Spawn parallel `general-purpose` subagents when several independent investigations can run at once (e.g. checking HF API for `from_pretrained(revision=...)` while simultaneously checking how perspic's logger formats its outputs). Don't spawn subagents for trivial edits or single-file reads — that's overhead for no gain.
- **All installs go into the existing venv only. Nothing global, ever.** The venv lives at `.venv/` in the project root and already has perspic installed editable. Use `uv pip install …` (which respects the active venv) or `uv add …` for pyproject-tracked deps. **Never** run `pip install` without the venv activated, **never** use `--user`, **never** touch the system Python. If a tool seems to want a global install, stop and ask. Verify with `which python` → should resolve to `.venv/bin/python` before any install.
- **Run tests through the venv too:** `.venv/bin/pytest …` or `uv run pytest …`. Same for `ruff`, `mypy`, `python -m vatis`.

### Permission model (what's enforced vs convention)

The `settings.local.json` allow list bounds *some* of what you can do, but **most safety is behavioral, not technical.** Read this section before making any change that touches things outside the vatis project tree.

**Technically enforced (the system will block you):**
- `Write` and `Edit` tool calls are restricted to `/tikhome/knikolaou/PycharmProjects/vatis/**`. You cannot use these tools to modify files outside the project.
- Recursive `rm -r` / `rm -rf` is **not** in the allow list. Only flat `rm -- <files>` is allowed. Recursive deletes will prompt and stall in unattended mode.
- `git push`, `git reset --hard`, `git rebase`, `git checkout` of branches, `git branch -D`, `git clean`, anything that writes to remotes or rewrites history — **not allowed**. Only read commands and safe writes (`add`, `commit`, `restore`, `stash`) are.
- `sed`, `awk`, `find`, `source` — **removed from the allow list** because they each provided escape hatches around the file-write restriction. Use `Edit` for edits, `Glob`/`Grep` for searching, and never `source` arbitrary scripts.
- No network write tools (no `curl`, no `gh`, no `ssh`). `WebFetch`/`WebSearch` are read-only.

**Convention only — the system will NOT stop you, but you must not do these things:**
- **Arbitrary Python execution is the fundamental escape hatch.** `Bash(python:*)`, `Bash(uv:*)`, `Bash(.venv/bin/python:*)`, and `Bash(pytest:*)` all let you run Python code, which can do *anything Python can do* — including `open(path, 'w')` on any path, `os.remove`, `subprocess.run`, network calls. This bypasses every `Write`/`Edit` path restriction. **You must not use Python's standard library to write, delete, or move files outside the project tree.** If you find yourself writing `open("/some/path/outside/vatis", "w")` or `subprocess.run(["rm", "-rf", ...])` or `os.system(...)`, stop. The allow list doesn't catch this; the user trusts you not to do it.
- **Don't `pip install` random packages.** Stick to dependencies declared in `pyproject.toml`. If you genuinely need a new dep, add it via `uv add <pkg>` (which records it in `pyproject.toml` so the user can review later) — never via `pip install <pkg>` ad hoc.
- **Don't `mv` or `cp` files outside the project.** `Bash(mv:*)` and `Bash(cp:*)` are wide for convenience; the user accepted the residual risk on the understanding that you only operate inside the vatis tree.
- **Don't modify `settings.local.json` to grant yourself permissions you weren't given.** The allow list is the user's contract. If you need a permission you don't have, log it to `BLOCKED.md` and stop.
- **Don't modify files in `~/.ssh`, `~/.gitconfig`, `~/.aws`, `~/.config`, or any dotfile in `$HOME`.** Even though you technically *could* via Python, you must not.

**The real safety boundary is the host filesystem.** Treat anything outside `/tikhome/knikolaou/PycharmProjects/vatis/` as off-limits, regardless of which tool would let you reach it. The only exceptions are read-only operations (which are explicitly allowed everywhere via `Read(/**)` etc.) and the HuggingFace cache (which transformers will populate at `~/.cache/huggingface/` as a side effect of `from_pretrained` — that's expected and fine).

**When in doubt, log to `BLOCKED.md` and skip.** It's always better to leave a step undone with a clear note than to take a risky workaround.
- **Unattended-mode behavior (graceful fallback when blocked).** This project is built to run autonomously when the user is not around. If a tool call gets denied by the permission system:
  1. **Do not retry the same call.** Retrying just blocks again. There is no human to approve.
  2. **Try an allowed alternative.** Examples: if `Bash(curl …)` is denied, try `WebFetch`. If a `Bash` install command is denied, try `uv add` (which is allowed). If `Write` to a path outside the project is denied, write inside the project instead.
  3. **If no alternative exists, skip the blocked step and keep going on independent work.** Make a note in a `BLOCKED.md` file at the project root with: the exact tool call that was denied, what you were trying to do, what the user needs to do to unblock it, and which downstream tasks depend on it. Continue with everything that doesn't depend on the blocked step.
  4. **Never use destructive workarounds** to "make the obstacle go away" — no `rm -rf` to dodge a permission check, no `--no-verify` on git, no editing settings.local.json to grant yourself permissions you weren't given. The allow list is the user's contract; expanding it is the user's job, not yours.
  5. **At the end of an unattended session, summarize in `BLOCKED.md`** what got built, what got skipped and why, and what the user needs to approve to resume. This is the file the user reads first when they come back.

## Conventions

- One concept per module. Don't bundle Hutchinson with normalization with sharding "because they're all numerical".
- The math core (`vatis/core/`) takes plain tensors and `nn.Module`. It knows nothing about HF, DDP, or sinks. This is the only part that needs to be bulletproof — everything else is plumbing.
- All-reduce only at the boundaries of `(checkpoint, batch)` computations, never in inner loops.
- Never materialize per-sample gradients. If a future feature needs them, it goes behind a separate `vatis.advanced` module and is opt-in.
- Padding is handled at one place: `vatis/core/normalization.py::valid_token_mask`. Every accumulator multiplies by this mask. No ad-hoc masking elsewhere.

## Glossary

| symbol | code name | meaning |
| --- | --- | --- |
| `χ_loss` | `chi_loss` | `‖∇_f L‖²` |
| `χ_net` | `chi_net` | `‖∇_θ f‖_F² = Tr(Θ)` |
| `χ_pos` | `chi_pos` | `δL / (χ_loss · χ_net)` — "spectral position". **Naming note:** in vatis this is `chi_pos`. The same quantity is called `chi_align` (and sometimes `chi_coup`) in perspic and in `background_info.tex`. When cross-validating against perspic (see "Testing" below), map `perspic.chi_align ↔ vatis.chi_pos` and `perspic.chi_coup ↔ vatis.chi_pos`. |
| `δL` | `delta_loss` | `⟨∇_θ L^A, ∇_θ L^B⟩` |
| `Θ` | (not stored) | empirical NTK; never materialized |
| `N_A`, `N_B` | `n_valid_a`, `n_valid_b` | valid (non-pad) token counts in batches A and B |
| `n` | `n_hutchinson` | number of Hutchinson probe vectors |

## Known issues

Latent bugs and quirks that an agent walking into the repo should be aware of. The detailed write-ups and the work to fix them live in `TASKS_NEXT.md`; this section is just an index.

- **`ParquetSink` truncates on re-open within the same path** — calling `analyze()` twice with `sink="path.parquet"` overwrites the file each time. The deployment example sidesteps this via the single-`analyze()`-with-multiple-revisions pattern, but it's an obvious footgun for any user who loops manually. Either fix or document explicitly. Tier 1 in `TASKS_NEXT.md`.
- **Multi-GPU DDP path is unverified on real GPUs** — only the 2-rank CPU loopback test exercises it. No real-GPU integration test exists. Tier 1 in `TASKS_NEXT.md`.
- **Custom (non-CE) `loss_fn` is not really supported** — `delta_loss` will use whatever `loss_fn` you pass, but `chi_loss` always uses the closed-form CE shortcut. Mixing the two produces wrong observables. The autograd fallback was deleted in v1.1 task 6 to avoid silent wrong-answer modes. Real support is the v1.2 feature work. Tier 2 in `TASKS_NEXT.md`.
- **`OpacusEstimator` is still a stub** — the original v1 spec called this v1.1 work; v1.1 was hardening instead. It's now Tier 2 in `TASKS_NEXT.md`.

## v1 build history (historical)

The v1 build was executed in this order. **Do not re-execute** — the work is done. Documented here so future agents understand why files exist where they do.

1. `vatis/core/normalization.py`, `vatis/core/probes.py` — valid-token masking, Rademacher/Gaussian probe generation.
2. `vatis/core/observables.py` — closed-form `χ_loss`, `δL`, `χ_pos` combinator.
3. `vatis/core/chi_net/base.py` + `hutchinson.py` — `ChiNetEstimator` ABC and the always-works fallback. Convergence test against an exact-trace ground truth on the toy fixture.
4. `vatis/core/chi_net/per_seq_cv.py` — per-sequence backward + control variate. Brought `vatis/data/collate.py` (`iter_micro_batches`) along with it because the estimators need micro-batch iteration; this is a divergence from the original spec which had `data/` later.
5. `vatis/core/chi_net/opacus.py` — stub only, raises `NotImplementedError`. Full implementation deferred.
6. `vatis/models/hf.py` — `load_hf_model(name, revision)`.
7. `vatis/data/batches.py` — `EvalBatchSpec` and auto-wrapping.
8. `vatis/sinks/parquet.py` — long-format writer, then base ABC, then optional sinks.
9. `vatis/analyzer.py` — single-GPU loop, then DDP sharding + all-reduce.
10. `vatis/distributed/` — DDP sharding + all-reduce. Tested on 2-rank CPU loopback only.
11. `vatis/cli.py` + `python -m vatis` — torchrun wrapper.
12. Integration test on `pythia-14m` (gated, local).

End of v1 build: 56 tests, all green.

## v1.1 hardening pass (historical)

The v1.1 pass was about *trust* and *demonstrability*, not new features. Eight numbered tasks plus a baseline commit and three post-task follow-ups on the deployment example. End state: 67 tests, all green; the deployment example produces meaningful trajectories on real Pythia checkpoints.

Tasks (in commit order):

- **task 0** — git baseline commit of the v1 build state.
- **task 5** — `_compute_cross_pair` ordering footgun: added a contract assertion + unit test so cross-pair calls before the matching self-pair fail loudly instead of silently returning zeros.
- **task 6** — deleted the unused `chi_loss_from_autograd` helper (3 tests removed). Documented in `## Loss handling` above.
- **task 7** — `PerSequenceControlVariateEstimator.check_memory_feasible` startup check that raises if `B × n_params × 4` exceeds 50% of free GPU/host RAM. Refuses oversized configs at startup instead of OOMing partway through.
- **task 2** — exact-NTK ground-truth tests for `chi_pos` and `δL(A, B)` via `_build_full_jacobian_and_u`. Tight 1e-5 tolerance, the most rigorous correctness check we have because it bypasses Hutchinson entirely.
- **task 3** — tight-tolerance perspic cross-check at `n_hutchinson = 16384` (rel 0.3%, abs 1e-6) for both vatis methods. Caught nothing — vatis agrees with perspic at the noise floor.
- **task 4** — heavy-padding cross-validation test using TinyMLP with `labels=-100` on 5/8 samples, plus an LM-padding regression against the exact-NTK Jacobian builder.
- **task 1** — deployment example + benchmark on `pythia-14m`: `examples/pythia_sweep.py`, `examples/BENCHMARK.md`, four plots, parquet output. Surfaced and fixed a latent bug where self-pair rows were emitting `revision=""` regardless of input.
- **task 8** — wrote `CLAUDE.md.proposed` listing recommended edits to the architecture diagram, dependency list, test counts, and compute scaling table. The proposal was applied to `CLAUDE.md` directly during the v1.1→v1.2 spec reconciliation (the version of this document you are reading).

Three **post-task follow-ups** on the deployment example, prompted by user discussion after task 8:

1. **Real-text eval batches** — replaced random-integer batches with tokenized prose (Pride and Prejudice) and tokenized code (a Python module). The chi_net term is Jacobian-of-the-model-evaluated-at-the-input; off-distribution random tokens make the trajectory uninterpretable. Real text puts the evaluation back on Pythia's training manifold. Also added `cross_pairs=[("prose", "code")]` for the LNA-relevant cross-distribution gradient correlation.
2. **Log-spaced early checkpoints** — added `step1, step8, step64, step512` to the sweep alongside the existing `step1000` → `step143000` main phase. Revealed three findings invisible from `step1000` onward: `chi_net` U-shape at step64 (parameter Jacobian *shrinks* in the first ~100 steps before growing), `chi_loss` warmup cliff (no measurable learning in the first 64 steps), non-monotonic cross `δL` (drops, rebounds, then declines through zero).
3. **Analysis script** — `examples/analyze_results.py`, a small post-processing script that reads `results.parquet` and computes the normalized cross-batch gradient correlation `cos(g_A, g_B)`. Demonstrates the compute/analysis split (compute is expensive, analysis is cheap) and that the parquet schema is the contract — the script has zero vatis imports.

End of v1.1: 67 tests, all green; 13 commits over the hardening + follow-up arc; CLAUDE.md and SESSION_SUMMARY.md fully reconciled with the code.
