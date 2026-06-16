# vatis — specification

Compute LNP observables (`chi_loss`, `chi_net`, `chi_pos`) on **pretrained
model checkpoints**, scaled across multiple GPUs via DDP. Sister package to
`perspic`, which computes the same quantities *during* Lightning training;
vatis is for the regime where you can't pretrain yourself and instead probe
published checkpoints (Pythia, OLMo, …).

This document is the durable reference for the math, architecture, public API,
result schema, and scope. The full derivation of the LNP decomposition is in
the paper, [arXiv:2605.31244](https://arxiv.org/abs/2605.31244).

## Core idea (math)

For a loss `L = (1/N) Σᵢ ℓᵢ` over `N` "samples" (= valid tokens for an LM), the
LNP decomposition factors the linearized loss change:

```
δL(A, B) = χ_loss · χ_net · χ_pos
```

with

- `χ_loss = ‖∇_f L‖²`  — closed form from logits, **no backward through θ**.
- `χ_net  = ‖∇_θ f‖_F² = Tr(Θ)` where `Θ = ∇_θ f (∇_θ f)^T` is the eNTK.
- `χ_pos = δL / (χ_loss · χ_net)` — recovered indirectly, no eNTK eigendecomposition.
- `δL(A, B) = ⟨∇_θ L^A, ∇_θ L^B⟩` — two ordinary backwards plus a dot product.

### How `χ_loss` and `δL` are computed (cheap, fixed cost)

These two are the same regardless of which `χ_net` estimator is used.

- `χ_loss` for cross-entropy is closed form from logits:
  ```
  ∂ℓ_{b,s} / ∂z_{b,s,v} = (softmax(z_{b,s})_v − onehot(y_{b,s})_v) / N_valid
  χ_loss = Σ_{b,s,v ∈ valid} (∂L/∂z_{b,s,v})²
  ```
  Zero backward passes through θ — a pure tensor op on logits.

- `δL(A, B)`: one full backward of `L^A` w.r.t. θ → flatten gradient `g_A`. Same
  for `g_B`. Then `δL = g_A · g_B`. **Two backwards total**, regardless of
  granularity. For the self case `δL(A, A) = ‖g_A‖²`, only one backward is needed.

Total fixed cost per `(checkpoint, eval-batch-pair)`: **2 backwards** (or 1 for
self). Everything else goes into `χ_net`.

### `χ_net` estimation: three methods

`χ_net = Σ_b Tr(M_b)` where `M_b = J_b J_bᵀ` and `J_b = ∇_θ f(x_b)` is the
per-sequence Jacobian. This is **the only expensive observable**, and the reason
vatis exists. The three methods share the same outer loop and produce the same
observable; they differ only in *how* `Tr(M_b)` is estimated.

A single VJP is a projection (`g = Jᵀ v` collapses information), so trace
estimation needs either (a) deterministic basis vectors over the full output
dim — exact but `O(S·V)` backwards, infeasible for LMs — or (b) random probes
(Hutchinson) — `O(n)` backwards with controllable variance.

#### Method 1: `HutchinsonEstimator` (default for large `B`)

Plain-torch Hutchinson at micro-batch level. For each micro-batch (sequences
`[b₁, ..., b_M]`), draw `n_h` independent Rademacher vectors `v ∈ R^(M·S·V)`,
each with the same shape as the micro-batch output. Per draw:

```python
out = model(x_micro)              # (M, S, V)
projected = (out * v).sum()       # scalar
g, = torch.autograd.grad(projected, model.parameters())
chi_net_accumulator += g.pow(2).sum()  # accumulates Σ_b ‖J_bᵀ v‖²
```

Cost: `n_h` backwards per micro-batch. Variance is `~B×` higher per draw than the
per-sequence path. **When to use:** large `B` (≥ 32), or when other methods don't
apply. This is the always-works fallback.

#### Method 2: `PerSequenceControlVariateEstimator` (default for small `B`)

For each sequence `b`, do **one extra backward** of `L_b` (per-sequence loss)
w.r.t. `θ`. This gives `g_b = J_bᵀ u_b` where `u_b = ∇_f L_b` is known in closed
form. From `(g_b, u_b)`:

- `‖g_b‖²` — a Rayleigh quotient of `M_b` along the loss direction (free,
  deterministic).
- `⟨g_b, g_{b'}⟩` for all `b, b'` — the cross-sample alignment matrix, a bonus
  quantity (opt-in via `compute_alignment_matrix=True`; returned in
  `ChiNetResult.extras`, not written to the sink by default).

These are used as a **control variate** for the Hutchinson estimate of
`Tr(M_b)`: the loss-direction projection is correlated with the trace, known
exactly, has zero variance; subtracting its noisy Hutchinson counterpart and
adding back the exact value reduces total variance — substantially in the
high-`χ_pos` regime.

Cost per `(ckpt, batch)`: `n_h + B` backwards. Linear in `B`, so unattractive
for `B > ~64`. **When to use:** moderate `B` (≤ 32), or when the per-sample
alignment matrix is wanted.

#### Method 3: `OpacusEstimator` (opt-in fast path)

When the model contains only opacus-supported layers and has no parameter tying,
opacus + ghost clipping computes per-sample gradient norms in one backward,
packing `B` samples per pass. Falls back with a clear error on tied embeddings
(Pythia!), unsupported layers, or in-place ops.

**Status:** stub only — `vatis/core/chi_net/opacus.py` raises
`NotImplementedError`. Not yet implemented; never auto-selected.

#### Auto-selection rule

```python
def select_chi_net_method(B_total: int, requested: str | None) -> str:
    if requested is not None:
        return requested  # user override always wins
    if B_total <= 32:
        return "per_sequence_cv"
    return "hutchinson"
```

### Sample unit and normalization

There is **one** observable, computed at token granularity, normalized by the
number of valid (non-pad, non-ignored) tokens.

- `χ_loss` and `χ_net` are sums over `(b, s, v)` regardless. Frobenius norms of
  Jacobians decompose additively across output dims.
- Batch-size normalization uses **valid token counts**, not sequence counts.
  Track `n_valid_A`, `n_valid_B` per call.
- Normalized observables (default output):
  ```
  χ̃_loss = √(N_A · N_B) · χ_loss
  χ̃_net  = χ_net / √(N_A · N_B)
  ```
  Both unnormalized and normalized values are written to the result sink.

## Scope

- **In:** HuggingFace causal-LM checkpoints (Pythia, OLMo, GPT-NeoX class).
  Tokenized data already in model-ready form. DDP across user-specified
  `num_gpus`. Two `χ_net` estimators (`hutchinson`, `per_sequence_cv`) with
  auto-selection. Parquet result sink (canonical) plus optional W&B and
  TensorBoard. CLI via `python -m vatis run …` that wraps `torchrun`.
- **Out:** Tokenization (examples tokenize inline; vatis core provides no
  tokenizer). Checkpoint discovery / sweep scheduling. FSDP. Tensor / pipeline
  parallelism. `OpacusEstimator` (stub only). Custom non-CE losses for
  `chi_loss`. Per-sample variance observables beyond the cross-sample alignment
  matrix. Models that don't fit on one GPU.

The core (`vatis/core/`) is decoupled from model loading, data plumbing, and
parallelism, so adding any of these later is straightforward.

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
│   │   └── opacus.py      # OpacusEstimator (stub)
│   ├── probes.py          # Rademacher / Gaussian probe vector generation, masking
│   └── normalization.py   # valid-token counting, √(N_A·N_B) factors
├── models/
│   ├── __init__.py        # exports load_hf_model
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
├── cli.py                 # argparse entrypoint
└── _version.py
```

**Examples tree:**

```
examples/
├── _helpers.py                    # shared: tokenize_into_lm_batch, step_from_revision
├── pythia_sweep.py                # canonical example: 13 pythia-14m revisions × prose + code
├── analyze_results.py             # post-processing of results.parquet — derived observables
├── BENCHMARK.md                   # deployment-example findings & sizing
├── pythia_sweep/                  # outputs from pythia_sweep.py + analyze_results.py
└── benchmark/                     # A100 performance suite (see benchmark/README.md)
    ├── README.md                  # index for the three benchmarks below
    ├── submit.sh                  # sbatch wrapper that reads .env (HF_HOME, SBATCH_*)
    ├── _build_eval_batch.py       # builds the fixed eval batch fixture
    ├── single_gpu/                # single-GPU (method, n_h, B) sweep + plots
    ├── ddp_scaling/               # multi-GPU strong/weak scaling worker + results
    └── accuracy/                  # estimator-variance vs n_hutchinson sweep + plots
```

The deployment example (`examples/pythia_sweep.py`) is the canonical "does vatis
work end-to-end on real weights" check. Running the example/benchmark scripts
respects `HF_HOME` (set in your shell or the repo-root `.env`; see
`run_config.py`).

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
    distributed="ddp",
    sink="results.parquet",                     # str → ParquetSink; or list of sinks
    dtype="bf16",                               # forward dtype; grads in fp32
)
```

`analyze()` is the one-shot convenience. The underlying `Analyzer` class is
exposed for users who want to drive the loop themselves.

### EvalBatchSpec

```python
from vatis.data import EvalBatchSpec

EvalBatchSpec.fixed(batch)            # frozen tensor or dict, reused at every checkpoint
EvalBatchSpec.resample(dataloader)    # next(iter(loader)) at each checkpoint
EvalBatchSpec.from_callable(fn)       # fn(checkpoint_id) -> batch
```

The `eval_batches` argument is just `dict[str, EvalBatchSpec | Tensor |
DataLoader | Callable]` — bare values are auto-wrapped. Multiple named eval
batches are computed at every checkpoint and emitted as separate rows. For
`δL(A, B)`, pass `cross_pairs=[("val_fixed", "train_at_step"), …]`. By default
`δL` is computed only for the self pair `(b, b)`.

### Result format (parquet, long)

One row per `(checkpoint_id, eval_batch_name_A, eval_batch_name_B, observable,
value)`. Long format handles arbitrary eval batches / pair combinations without
schema churn, loads trivially into pandas, and is append-friendly across DDP
ranks (rank 0 writes after all-reduce).

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

- User passes `num_gpus=N`. CLI launches `torchrun --nproc_per_node=N -m
  vatis._worker …`.
- Each rank loads the same checkpoint (vatis assumes the model fits on one GPU).
- Each eval batch is sharded across ranks along the batch dim. If `B % N != 0`,
  the last rank gets a smaller shard — the all-reduce handles this because we
  accumulate token-weighted sums, not means.
- Per micro-batch: forward → `χ_loss` accumulator. For `χ_net` Hutchinson, draw
  `n` probe vectors **with the same seed across ranks** so each rank computes a
  partial estimate of the same trace; all-reduce sums them. For `δL`, each rank
  computes its partial gradient and all-reduce-sums.
- All-reduce happens **once per checkpoint × eval-batch**, never in inner loops.
- `micro_batch_size=1` is the always-safe fallback. The launcher does not
  attempt OOM auto-tuning — the user picks a size that fits.

### Probe vector details

- Default `n_hutchinson=32`, Rademacher distribution. Variance scales as `1/n`.
- Probe seed is **shared across ranks** within one `(checkpoint, batch)` call;
  different across calls.
- Probe vector shape matches the local micro-batch output `(M, S, V)`. **Padding
  positions are zeroed** so masked tokens contribute to neither `χ_net` nor
  `χ_loss`.

### Loss handling

vatis ships with the standard HF causal-LM CE loss. **Only the closed-form CE
path is supported for `chi_loss`.** The bundle exposes a `loss_fn(logits, batch)
-> scalar` slot used by the analyzer for `delta_loss` (autograd through θ), but
`chi_loss` always uses the closed-form softmax-minus-onehot shortcut. If you pass
a non-CE `loss_fn`, `delta_loss` is correct but `chi_loss` is wrong.

The bundle's `valid_mask_fn` and the `labels=-100` (`ignore_index`) sentinel are
**intersected** inside the `chi_loss` accumulator: positions excluded by either
are excluded from both the numerator and the `n_valid` denominator, so the two
paths count the same set.

### Precision

- Forward in `bf16` by default (matches Pythia/OLMo training precision).
- Gradients accumulated in `fp32`. Hutchinson probes drawn in `fp32`, cast to
  forward dtype for the backward, gradient norm computed in `fp32`.
- `dtype="fp32"` is available for tiny test models.

## Research workflow

The deployment example demonstrates the canonical two-phase pattern:

**Phase 1: compute** (`pythia_sweep.py`). Loads checkpoints, runs `analyze()`,
writes `results.parquet`. The expensive step — model loading dominates wallclock.

**Phase 2: analysis** (`analyze_results.py`). Reads `results.parquet`, computes
derived observables (e.g. `cos(g_A, g_B) = δL(A, B) / sqrt(δL(A,A)·δL(B,B))`),
produces plots and tables. **Zero vatis imports** — only `pyarrow` and
`matplotlib`. The parquet schema is the contract; making analysis consume only
the parquet enforces that contract by example.

Two patterns the example also demonstrates:

- **Real text, not random integers, for any observable that gets discussed.**
  `chi_net` is the squared Frobenius norm of the parameter Jacobian *evaluated at
  the input*; off-distribution random tokens evaluate at a meaningless point.
  Synthetic data is fine for shape-only benchmarks (the sweeps under
  `examples/benchmark/` correctly use random integers — they measure
  wallclock/memory only).
- **Cross-pair observables.** `cross_pairs=[(A, B)]` adds `δL(A, B)` and
  `chi_pos(A, B)` rows at zero extra backward cost — they tell you whether
  learning on A helps or hurts B (positive `δL` → transfer, negative →
  interference).

## Dependencies

Hard: `torch >= 2.2, < 2.11`, `transformers >= 4.40`, `pyarrow`, `numpy`.

Soft (extras): `matplotlib` + `tqdm` → `vatis[examples]` (needed only to run the
example/benchmark scripts; the package never imports them); `wandb` →
`vatis[wandb]`; `tensorboard` → `vatis[tensorboard]`; `pytest`, `ruff`, `mypy` →
`vatis[dev]`.

**No** `perspic`, `opacus`, `functorch`, or `pytorch-lightning` — the math is
reimplemented in `vatis/core/observables.py`. `perspic` is needed only for the
cross-validation test tier (gated, skipped in CI).

## Conventions

- One concept per module.
- The math core (`vatis/core/`) takes plain tensors and `nn.Module` — it knows
  nothing about HF, DDP, or sinks. This is the only part that must be
  bulletproof; everything else is plumbing.
- All-reduce only at the boundaries of `(checkpoint, batch)` computations.
- Never materialize per-sample gradients outside an opt-in path.
- Padding is handled in one place: `vatis/core/normalization.py::valid_token_mask`.

## Tooling and tests

- **Package manager:** `uv`; `pyproject.toml` is the single source of truth.
- **Lint + format:** `ruff` (`ruff check`, `ruff format`).
- **Types:** `mypy --strict` on `vatis/`.
- **Tests:**
  - `tests/unit/` — runs in CI. Math correctness against a hand-built toy
    transformer (Hutchinson convergence to the exact trace, `χ_loss` closed-form
    correctness, `δL` symmetry, normalization invariants, the chi_pos
    combinator, analyzer contract assertions, exact-NTK ground-truth tests).
  - `tests/integration/` — gated behind `pytest -m integration`, skipped in CI.
    Uses `EleutherAI/pythia-14m` to verify HF loading + DDP on real checkpoints.
    The DDP test is currently a 2-rank CPU loopback.
  - `tests/cross_validation/` — gated behind `pytest -m cross_validation`,
    skipped in CI (requires `perspic`). Cross-validates every observable
    (self-pair and cross-pair) against perspic as the reference implementation.
  - GPU-only tests are marked `@pytest.mark.gpu` and auto-skip when no CUDA
    device is present (see `tests/conftest.py`).
- **CI:** `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`,
  `mypy`, and the unit tier against `uv sync --extra dev`, on push to any branch
  and PRs to `main`.

## Glossary

| symbol | code name | meaning |
| --- | --- | --- |
| `χ_loss` | `chi_loss` | `‖∇_f L‖²` |
| `χ_net` | `chi_net` | `‖∇_θ f‖_F² = Tr(Θ)` |
| `χ_pos` | `chi_pos` | `δL / (χ_loss · χ_net)` — "spectral position". Called `chi_align` / `chi_coup` in perspic. |
| `δL` | `delta_loss` | `⟨∇_θ L^A, ∇_θ L^B⟩` |
| `Θ` | (not stored) | empirical NTK; never materialized |
| `N_A`, `N_B` | `n_valid_a`, `n_valid_b` | valid (non-pad) token counts |
| `n` | `n_hutchinson` | number of Hutchinson probe vectors |

## Known limitations

- **`ParquetSink` truncates on re-open within the same path** — calling
  `analyze()` twice with the same `sink="path.parquet"` overwrites it. Use the
  single-`analyze()`-with-multiple-revisions pattern.
- **Multi-GPU DDP is exercised only by a 2-rank CPU loopback test** — the
  real-GPU DDP path is validated by the `examples/benchmark/ddp_scaling/` job but
  has no automated test.
- **Custom (non-CE) `loss_fn` is not fully supported** — `chi_loss` always uses
  the closed-form CE shortcut (see Loss handling).
- **`OpacusEstimator` is a stub.**
