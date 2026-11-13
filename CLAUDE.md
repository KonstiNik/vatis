# vatis

Compute LNP observables (`chi_loss`, `chi_net`, `chi_pos`) on **pretrained model checkpoints**, scaled across multiple GPUs via DDP. Sister package to `perspic`, which does the same thing during Lightning training. vatis is for the regime where you can't pretrain yourself and instead probe published checkpoints (Pythia, OLMo, …).

## Agent quick start

If you're an agent walking into this repo, read in this order:

1. **`MAINTAINING.md`** — the branch model (`dev` = working branch + full history + research; `main` = curated public release with an unrelated/orphan history) and the release/promotion workflow. **Read this before any git work** — the `dev`/`main` invariants are easy to break.
2. **`## Working conventions`, `## Conventions`, and `## Known issues` below** — read before running anything that touches files outside the project tree, modifies git history, or installs packages. The "Known issues" section names the latent bugs you should not be surprised by.
3. **The rest of this file** — the durable reference for math, architecture, scope. The "v1 build history" and "v1.1 hardening pass" sections at the bottom describe past phases; read them for context but don't try to re-execute the implementation order — that work is done.

If you contradict a paragraph in this file with the code you write, **edit the paragraph in the same commit**. Don't leave stale text. The "v1.1 hardening pass" learned this the hard way and ended up with a `CLAUDE.md.proposed` sibling file that sat dormant for an entire session.

## Core idea (math)

For a loss `L = (1/N) Σᵢ ℓᵢ` over `N` "samples" (= valid tokens for an LM), the LNP decomposition factors the linearized loss change:

```
δL(A, B) = χ_loss · χ_net · χ_pos
```

with

- `χ_loss = ‖∇_f L‖²`  — closed form from logits, **no backward through θ**.
- `χ_net  = ‖∇_θ f‖_F² = Tr(Θ)` where `Θ = ∇_θ f (∇_θ f)^T` is the eNTK.
- `χ_pos = δL / (χ_loss · χ_net)` — recovered indirectly, no eNTK eigendecomposition.
- `δL(A, B) = ⟨∇_θ L^A, ∇_θ L^B⟩` — two ordinary backwards plus a dot product.

The full derivation is in the paper, [arXiv:2605.31244](https://arxiv.org/abs/2605.31244). Read it before touching the math.

### The preconditioned ruler (`chi_pos_P`)

An optimizer with update `Δθ = −η P ∇L` does not train on the raw eNTK but on the **effective kernel** `Θ_P = J P Jᵀ`. vatis measures the same LNP triple on that kernel — the "optimizer's-eye view" — alongside the raw one:

```
delta_loss_P(A, B) = ∇L^B · P ∇L^A        (diagonal P: elementwise-weighted dot)
chi_net_P          = ‖P^{1/2} J‖_F² = Tr(Θ_P)
chi_loss           = unchanged (kernel-independent)
chi_pos_P          = delta_loss_P / (chi_loss · chi_net_P)
```

with the same σ-based batch-size normalization as the raw components, and the cross pair taking the geometric mean `‖P^{1/2}J^A‖_F · ‖P^{1/2}J^B‖_F` exactly as raw does. `P = I` reproduces the raw triple identically — that is a unit test (`tests/unit/test_chi_pos_p.py`), not a hope.

**`P` is diagonal throughout** — a 1-D vector in the canonical parameter order (`vatis/core/preconditioner.py::Preconditioner`), segment-for-segment aligned with the flat gradient and with `OptimizerStateView.flat_state`. Non-diagonal `P` (Muon's orthogonalization) is out of scope and the code is deliberately not designed for it.

P sources:

| source | `P` | where |
| --- | --- | --- |
| none | — (raw ruler only) | `preconditioner=None`, the default |
| `identity` | `I` | `preconditioner="identity"` |
| `explicit` | user-supplied 1-D tensor | `Preconditioner.from_diagonal(...)` |
| `adamw` | `diag(1/(√v̂ + ε))`, `v̂ = exp_avg_sq/(1−β2^t)` | `LocalBundleSource.adamw_preconditioner()` |
| `gn` (tests only) | `(JᵀJ)⁺` | built in the test from a materialized `J` |

**AdamW conventions (project decisions — do not revisit).** `t` is the **saved** `step_count`: bundles are pre-step, so that is the number of updates already folded into the saved `exp_avg_sq` and hence the matching bias-correction exponent. β2 and ε come from the saved `param_groups` (authoritative — what the optimizer ran with), with `metadata.json` as a cross-check that warns on disagreement. **Momentum (`m̂`) and decoupled weight decay are excluded**: `P` is the geometry factor only, because `Θ_P = J P Jᵀ` is defined for a preconditioner and momentum is a gradient-history filter, not a metric. `amsgrad=True` is refused (it preconditions with `max_exp_avg_sq`).

**`delta_loss_P` is therefore not the first-order loss change of the AdamW step that was taken** — it is the response to the hypothetical update `−η P_t ∇L^A`. The real step applies `P_{t+1}` (one more EMA update, weight `(1−β2)/(1−β2^{t+1}) ≈ 1/(t+1)` early on — *not* negligible there) to the first moment `m̂`, plus decoupled weight decay. The exact three-term split and what follows from it are in `docs/SPEC.md` §"What `delta_loss_P` is not". Measuring the actual-update response (history and weight-decay terms, and its residual against realized loss changes) is the planned follow-up; nothing in this package does it yet, and the golden gate only *prints* the linearization sum.

**Step-0 / empty optimizer state.** A step-0 checkpoint legitimately has no optimizer state. `adamw_preconditioner()` then **skips the P rows for that checkpoint** and emits a `UserWarning`; the raw rows are unaffected. Neither fallback is honest: `P = 1/(√0 + ε) = 1/ε` scales every observable by ~`1/ε²`, and `P = I` would publish `chi_pos_P == chi_pos` under the `adamw` descriptor — a plausible-looking "the preconditioner does nothing yet" reading that is an artifact of the fallback. An absent row cannot be mistaken for a measured one.

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

Both live methods also produce `chi_net_P` when `compute(..., preconditioner=P)` is given, in the **same pass and from the same probes** (`ChiNetResult.chi_net_p`) — the P estimate needs no extra backward, because everywhere the raw path takes `‖g‖²` of a VJP or a per-sample gradient the P path takes `‖P^{1/2} g‖²`. Raw values are bit-identical with and without a `P`. `opacus` stays a stub and has no P path.

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

Preconditioned: `Tr(J P Jᵀ) = E_z ‖P^{1/2}(Jᵀz)‖²`, so the accumulator simply reduces the same `g` with the elementwise weight.

**When to use:** large `B` (≥ 32), or when other methods don't apply. This is the always-works fallback.

#### Method 2: `PerSequenceControlVariateEstimator` (default for small `B`)

For each sequence `b`, do **one extra backward** of `L_b` (per-sequence loss) w.r.t. `θ`. This gives `g_b = J_bᵀ u_b` where `u_b = ∇_f L_b` is known in closed form. From `(g_b, u_b)` we get exactly:

- `‖g_b‖²` — a Rayleigh quotient of `M_b` along the loss direction (free, deterministic).
- `⟨g_b, g_{b'}⟩` for all `b, b'` — the cross-sample alignment matrix, a bonus observable.

These are used as a **control variate** for the Hutchinson estimate of `Tr(M_b)`: the loss-direction projection is correlated with the trace, known exactly, has zero variance, and subtracting its noisy Hutchinson counterpart and adding back the exact value reduces total variance — substantially when the loss direction overlaps the top eigenmodes of `M_b`, which is exactly the high-`χ_pos` regime we care about.

Cost per (ckpt, batch): `n_h + B` backwards. Linear in `B`, so this method becomes unattractive for `B > ~64`.

Preconditioned: the decomposition carries over term by term — `Tr(Π J P Jᵀ) = Σ_b ‖P^{1/2}g_b‖²/‖u_b‖²` stays exact, and the Hutchinson residual estimates `Tr((I−Π) J P Jᵀ)` with the same output-space projection `w = (I−Π)v` (`Π` lives in output space and is set by the loss direction, not by the parameter metric). Derivation in the module docstring.

**Bonus output:** the cross-sample alignment matrix `C_{bb'} = ⟨g_b, g_{b'}⟩` is available opt-in via `compute_alignment_matrix=True` (returned in `ChiNetResult.extras`; the analyzer does **not** enable it or write it to the sink by default). It's exactly the eNTK projected onto the loss direction, sample-resolved. Useful for sample-level variance / interference analysis (the negative-`χ_pos` regime; see the paper, arXiv:2605.31244 §A.2).

**When to use:** moderate `B` (≤ 32), or when the user wants the per-sample alignment matrix.

#### Method 3: `OpacusEstimator` (opt-in fast path, deferred to v1.2)

When the model contains only opacus-supported layers and has no parameter tying, opacus + ghost clipping computes per-sample gradient norms in one backward, packing `B` samples per pass. With Hutchinson over `(S, V)` only, it's the fastest method **per backward count** — but the wall-time win only materializes when you can fit `M > 1` samples per micro-batch, which for an 8B model on an 80GB GPU is borderline. For smaller models (1B and below) it's a clear ~`B`× speedup.

Cost: `~n_h` backwards, where `n_h` is `~10–32`. Requires layer-compatibility check at startup; falls back with a clear error message if the model has tied embeddings (Pythia!), unsupported layers, or in-place ops that can't be neutralized.

**Status:** stub only in v1 and v1.1 — `vatis/core/chi_net/opacus.py` raises `NotImplementedError`. Originally listed as "v1.1 work" but the v1.1 hardening pass focused on trust and demonstrability instead of features. Full implementation is not yet done. When implemented, follow the pattern from `perspic/calculator/samplewise_opacus.py`.

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
- The `√|A|·|B|` factors from the paper (arXiv:2605.31244 §A.2) are batch-size normalization. We use **valid token counts**, not sequence counts. Track `n_valid_A`, `n_valid_B` per call.
- Normalized observables (default output):
  ```
  χ̃_loss = √(N_A · N_B) · χ_loss
  χ̃_net  = χ_net / √(N_A · N_B)
  ```
  where `N_A`, `N_B` are valid token counts. Both unnormalized and normalized values are written to the result sink so downstream analysis can pick.

## Scope (v1 / v1.1, current)

- **In:** HuggingFace causal-LM checkpoints (Pythia, OLMo, GPT-NeoX class). Native **checkpoint bundles** written by our own training runs (`load_local_bundle`, see "Native checkpoint bundles" below). Tokenized data already in model-ready form. DDP across user-specified `num_gpus`. Two `χ_net` estimators (`hutchinson`, `per_sequence_cv`) with auto-selection. The **preconditioned ruler** `chi_pos_P` (diagonal `P`; `none`/`identity`/`explicit`/`adamw` sources) alongside the raw one. Parquet result sink (canonical) plus optional W&B and TensorBoard. CLI via `python -m vatis run …` that wraps `torchrun`. End-to-end deployment example on `pythia-14m` with cross-pair observables (added in v1.1).
- **Out (current):** Tokenization (the example tokenizes inline; vatis core does not provide a tokenizer). Checkpoint discovery / sweep scheduling. FSDP. Tensor / pipeline parallelism. `OpacusEstimator` (stub only — full implementation deferred to v1.2). Custom non-CE losses for `chi_loss` (deferred to v1.2). Per-sample variance observables beyond the cross-sample alignment matrix. Models that don't fit on one GPU. **Non-diagonal `P`** (Muon's orthogonalization) — out of scope for WP0 and deliberately not designed for; momentum inclusion in `P`.

Adding any of these later should be straightforward because the core (`vatis/core/`) is decoupled from model loading, data plumbing, and parallelism. Those items are tracked as future work.

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
│   ├── param_layout.py    # ParameterLayout: THE canonical parameter flattening order
│   ├── preconditioner.py  # Preconditioner (diagonal P) + elementwise-weighted reductions
│   └── normalization.py   # valid-token counting, √(N_A·N_B) factors
├── models/
│   ├── __init__.py        # exports load_hf_model + load_local_bundle (registry-ready)
│   ├── hf.py              # load_hf_model(name, revision) → nn.Module + loss_fn
│   └── local_bundle.py    # load_local_bundle(run_dir, model_factory) → bundle source
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
├── _helpers.py                    # shared: tokenize_into_lm_batch, step_from_revision
├── pythia_sweep.py                # canonical example: 13 pythia-14m revisions × prose + code
├── analyze_results.py             # post-processing of results.parquet — derived observables
├── BENCHMARK.md                   # deployment-example findings & sizing
├── pythia_sweep/                  # outputs from pythia_sweep.py + analyze_results.py
│   ├── results.parquet
│   ├── chi_loss.png, chi_net.png, delta_loss.png, chi_pos.png
│   └── cos_similarity.png
├── benchmark/                     # A100 perf suite: single_gpu / ddp_scaling / accuracy (see benchmark/README.md)
├── spectral_tail_experiment.py    # research: spectral tail overlap (4 probes, 5 model scales)
├── spectral_tail_evaluate.py      # evaluation plots (--compute flag for FLOPs x-axis)
├── spectral_tail/                 # outputs + README with hypothesis and findings
│   ├── README.md
│   ├── results_pythia-{14m,31m,70m,160m,410m}.parquet
│   └── figures/                   # all plots (step-based and _compute variants)
├── two_rulers_llama.py            # WP0 GPU example, phase 1: both rulers over a bundle
├── two_rulers_llama_evaluate.py   # WP0 GPU example, phase 2: the figure + tables
└── output/                        # outputs of the WP0 example
    ├── results.parquet, run_summary.json, performance.csv
    └── two_rulers_llama.png       # chi_pos and chi_pos^P vs step and vs train loss
```

The deployment example is the canonical "does vatis work end-to-end on real weights" check. After any significant change to the analyzer or an estimator, **re-run `examples/pythia_sweep.py` and check the plots haven't moved unexpectedly**. The plots are an integration test the unit suite can't replicate.

The **WP0 GPU example** (`examples/two_rulers_llama.py`, findings and sizing in `examples/BENCHMARK.md`) is the same check for the *bundle* path: a real checkpoint bundle written by the project's own training run, both rulers on the GPU, and the one figure a human looks at to see `chi_pos^P` on a transformer for the first time. It needs a bundle to exist, so it is not runnable from a clean checkout — the training half lives in the training repo (`runs/example_bundle_run.sh`) and hands over on disk.

The spectral tail experiment (`examples/spectral_tail/README.md`) is a research investigation using vatis to probe whether `chi_pos(A, B)` can detect when training resolves a shared semantic feature (matrix multiplication) between texts in different programming languages. It runs across five Pythia model scales (14m–410m) with four probes (Python matmul, C++ matmul, English prose matmul, C++ string-processing as negative control).

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
    observables=None,                           # None = all raw (+ all P if a preconditioner is set)
    preconditioner=None,                        # None | "identity" | Preconditioner | fn(ckpt_id)->P|None
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
EvalBatchSpec.from_callable(fn)       # fn(checkpoint_id) -> batch
```

The `eval_batches` argument to `analyze()` is just `dict[str, EvalBatchSpec | Tensor | DataLoader | Callable]` — bare values are auto-wrapped. Multiple named eval batches are computed at every checkpoint and emitted as separate rows in the sink. This is the central feature: comparing observables across eval distributions at the same checkpoint is the primary use case.

For `δL(A, B)`, the user can specify `pairs=[("val_fixed", "train_at_step"), …]` to compute cross-batch alignments. By default `δL` is computed only for the "self" pair `(b, b)` for each named batch.

### Native checkpoint bundles (the local source)

Besides the hub, vatis reads **checkpoint bundles** written by our own training runs. A bundle is a run directory in a frozen schema (specified in the WP0 project repo, `wp0_specs/00_context.md`):

```
<run_dir>/
  metadata.json                # model config, optimizer type + hparams, lr schedule,
                               # seed, B_train, seq_len, dataset id, git hashes
  param_names.json             # ordered [ [name, shape], ... ] — THE canonical flattening
                               # order; index-aligned with the optimizer state_dict indices
  checkpoints/step_00000123/
    model.safetensors          # weights, bf16
    optim.pt                   # torch optimizer.state_dict(), fp32, saved PRE-step
    batch_a.npy                # int32 [B_train, seq_len] token ids of the update batch A_t
    meta.json                  # step, tokens_seen, lr, train_loss, wall_time
```

```python
from vatis import load_local_bundle

source = load_local_bundle(run_dir, "my_project.models:build_model")  # or a callable
analyzer.run(source.model_specs())                    # drop-in for the HF (name, revision) loop

for ckpt in source:                                   # or drive it yourself
    v = ckpt.optimizer_state.flat_state("exp_avg_sq")  # 1-D, canonical order
    t = ckpt.optimizer_state.step_count                # pre-step update count
    batch = ckpt.batch_a                               # {"input_ids", "labels"}

# The preconditioned ruler: P built per checkpoint from its saved optimizer state.
Analyzer(eval_batches=..., preconditioner=source.adamw_preconditioner(), ...).run(source.model_specs())
```

Four things to know:

- **vatis stays project-agnostic.** The architecture comes from a user-supplied `model_factory(metadata) -> nn.Module` (a callable, or a `"module:function"` import string). vatis package code never imports a training stack; the factory for a specific project lives in that project or in a test fixture.
- **`param_names.json` is the canonical flattening order, and there is one code path for it.** It is parsed into a `ParameterLayout` (`vatis/core/param_layout.py`), and *that same object* builds `ModelBundle.params` (by looking parameters up **by name in file order**, not by iterating `named_parameters()`) and flattens the optimizer state. So `flat_state("exp_avg_sq")` is segment-for-segment aligned with the gradient vector `parameter_grad_vector` / the analyzer's `_add_grads_into_flat` produce. A permutation here would silently corrupt every preconditioned observable, so the loader refuses anything it can prove is misaligned: optimizer `param_groups` index lists that aren't `[0 … n−1]` in order, state tensors whose shape contradicts the recorded shape, model parameters the order doesn't list, canonical names the model or the weight file doesn't have. `tests/unit/test_local_bundle.py` writes bundles whose `param_names.json` is a *permutation* of `named_parameters()` order, with a distinct `exp_avg_sq` constant per parameter and two pairs of identically-shaped parameters — so a same-shape swap (invisible to shape checks) fails the test.
- **Pre-step semantics.** The checkpoint at step `t` holds `θ_t` *and the optimizer state from before the step-t update*; `batch_a.npy` is the batch step `t` is about to consume. `step_count` is therefore the number of updates already applied — the bias-correction exponent that matches the *saved* `exp_avg_sq`. A step-0 checkpoint legitimately has an empty optimizer state (`is_empty`); `flat_state` raises rather than inventing zeros.
- **Lazy per checkpoint.** Bundles run ~0.5 GB × 100. Opening a bundle only reads the two JSON files and stats the checkpoint dirs; tensors load on first attribute access, weights are upcast bf16 → fp32 on load, and iterating the source `release()`s checkpoint `i−1` before yielding `i`. `Module.to(dtype)` is deliberately *not* used for the dtype cast — it would also hit complex buffers (precomputed rotary `freqs_cis`), silently discarding the imaginary part.

### Result format (parquet, long)

One row per `(checkpoint_id, eval_batch_name_A, eval_batch_name_B, preconditioner, observable, value)`. Long format because:
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
preconditioner     str           # "none" on raw rows; the P descriptor on P rows
```

The preconditioned ruler extends this format **minimally**: its results are new `observable` *values* (`delta_loss_P`, `chi_net_P`, `chi_pos_P`), not new columns, plus the single `preconditioner` descriptor column. Raw rows are untouched and carry `preconditioner = "none"`; a run without a preconditioner emits exactly the rows it always did. To select one ruler, filter on `preconditioner`.

There is deliberately no `chi_net_P_normalized` row: it is `chi_net_P / sqrt(n_valid_a * n_valid_b)` from the same row's own columns, and `chi_pos_P` — like `chi_pos` — is invariant under the normalization anyway.

### Distributed semantics

- User passes `num_gpus=N`. CLI launches `torchrun --nproc_per_node=N -m vatis._worker …`.
- Each rank loads the same model checkpoint (v1 assumes the model fits on one GPU).
- Each eval batch is sharded across ranks along the batch dim. If `B % N != 0`, the last rank gets a smaller shard and contributes fewer valid tokens — the all-reduce handles this naturally because we accumulate token-weighted sums, not means.
- Per micro-batch on each rank: forward, compute logits → `χ_loss` accumulator. For `χ_net` Hutchinson: each rank draws `n` probe vectors **for its own shard** from a generator seeded identically across ranks (reproducibility — *not* a partition of one global probe, see `## Probe vector details`) and estimates the trace over its shard; all-reduce sums them, which is correct because `Tr(Θ)` is a sum over rows and the shards partition the rows. For `δL`: each rank computes its partial gradient `g_A^rank`, all-reduce-sum to get the full `g_A`, dot with `g_B` (same procedure).
- All-reduce happens **once per checkpoint × eval-batch** combination, not per micro-batch. Accumulators live on each rank as plain tensors.
- `micro_batch_size=1` is the always-safe fallback. The launcher does **not** attempt OOM auto-tuning in v1 — it's the user's responsibility to pick a size that fits.

### Probe vector details

- Default `n_hutchinson=32`, Rademacher distribution. Variance scales as `1/n`. For `HutchinsonEstimator` the effective variance is `~B×` higher per draw, so the auto-selection uses `per_sequence_cv` for small `B`.
- **Probes are drawn on the CPU and copied to the target device**, so a seed identifies one probe set on every device. This matters: `torch.Generator("cpu")` and `torch.Generator("cuda")` yield *different* sequences from the same seed, and while probes were drawn on-device a CPU reference run and a GPU production run of the same `analyze()` call agreed on `chi_net` only to within the Hutchinson noise floor (~4e-2 relative at `n_h = 400`, growing as `1/√n_h`) even though every deterministic observable agreed to ~5e-08 — and the `hutchinson_seed` column could not tell the two apart. `make_probe` therefore **refuses a CUDA generator** rather than ignoring it.
- **The Rademacher draw is bit-packed, and it has to be.** A Rademacher entry needs one random bit; a uniform `uint8` supplies eight. `make_probe` draws `ceil(N/8)` bytes on the CPU, ships 1/8 as many bytes, and unpacks on the device. Measured on the WP0 example's probe shape `(8, 512, 4096)` (16.8M entries, RTX 3090 Ti): on-device draw 0.69 ms/probe, naive CPU `int64`→fp32→H2D **125 ms**, CPU `uint8`→H2D→expand 44 ms, **packed 11.85 ms**. The naive CPU draw cost 3.5× end-to-end on the example (4.13 s → 14.55 s mean analysis per checkpoint) and would have made the 8B production target infeasible; packed it is ~26% (≈1.1 s per checkpoint against ≈4.1 s of backwards). Bit `k` of a uniform byte is an independent fair coin, so the distribution is exactly Rademacher — `tests/unit/test_probes.py` asserts no per-bit-position bias, which is the failure a packing/shift bug would produce while leaving the overall mean looking fine. `gaussian` probes cannot be packed and take the plain CPU draw; they are not the default.
- Probe seed is **shared across ranks** within one `(checkpoint, batch)` call, and different across calls. Note what this does and does not buy: each rank draws a *full* probe for **its own shard**, so ranks do not partition one global probe — the same random numbers land on different samples. It makes a run reproducible; it does not make the DDP total equal the single-process total (the two are different unbiased estimates, differing by a few percent), and sharing the seed mildly *correlates* the shards' estimates. `tests/integration/test_ddp_loopback.py` therefore pins DDP against single-process with the probe sequence pinned to a shape-keyed constant, which makes the comparison exact.
- The probe vector shape matches the local micro-batch output: for HF causal LMs, `(M, S, V)`. **Padding positions are zeroed** in the probe vector so masked tokens contribute to neither `χ_net` nor `χ_loss`. The mask used is the *intersection* the analyzer builds (`valid_mask_fn` ∧ `labels != ignore_index`), not `valid_mask_fn` alone — see `## Loss handling`.
- No exact-vocab fallback in v1. If users want a deterministic check on a tiny test model, they crank `n_hutchinson` up.

### Loss handling

vatis ships with the standard HF causal-LM CE loss. **v1 only supports the closed-form CE path for `chi_loss`.** The bundle exposes a `loss_fn(logits, batch) -> scalar` slot, and the analyzer uses it for `delta_loss` (autograd through θ), but `chi_loss` always goes through the closed-form softmax-minus-onehot shortcut regardless of what `loss_fn` is set to. If you pass a non-CE `loss_fn`, `delta_loss` will be correct but `chi_loss` will be wrong.

The bundle's `valid_mask_fn` and the `labels=-100` (`ignore_index`) sentinel are **intersected once**, in `analyzer.py::_intersected_valid_mask_fn`, and *that one closure* is threaded through **every** accumulator: the `chi_loss` sum and its `n_valid` denominator, the per-micro-batch weighting that reassembles `∇_θ L` for `delta_loss`, and the probe mask the `χ_net` / `χ_net^P` estimator applies. The physics is the same in all three places: a position whose label is `ignore_index` contributes nothing to the loss, so its Jacobian rows have no business in `χ_net = ‖∇_θ f‖²_F` either. As a permanent tripwire the analyzer asserts the estimator's own `ChiNetResult.n_valid_tokens` equals the `chi_loss` count — post-fix they agree by construction, and a drift between them would silently rescale `chi_pos` by their ratio.

**History, because the fix came in two halves.** v1.2 task 1 intersected the masks in the `chi_loss` accumulator only (before that the numerator used `labels != -100` while the denominator used `vmask.sum()`). `delta_loss` and `χ_net` were left on the un-intersected `valid_mask_fn`, so with a disagreeing mask — the "all-True for an MLP with `-100` labels" case that fix itself cites — `delta_loss` came out rescaled by ~`(n_vmask/n_labels)²` **and dependent on `micro_batch_size`** (measured 2.34× / 2.27× / 2.12× at micro = 1 / 2 / 4 on a 4×8 padded toy), and `χ_net` summed the Jacobian rows of positions the loss ignores (+37% on the same fixture). Since `χ_pos = δL/(χ_loss·χ_net)` the two composed into an `O(1)`-wrong `χ_pos` with no non-finite value and no warning. Both are fixed and pinned by `tests/unit/test_masking_intersection.py`. **No published number moved**: the shipped loaders make the intersection a no-op (`causal_lm_valid_mask` returns `labels != ignore_index`), which is why the defects survived, and a bitwise re-run of the WP0 example bundle on CPU with the probe sequence pinned confirmed value-for-value identity across the fix.

The v1.1 hardening pass deleted the partially-implemented autograd fallback (`chi_loss_from_autograd`) precisely to avoid silent wrong-answer modes — better to fail loudly on a documented restriction than to ship a half-working path. **Custom loss support is deferred to v1.2**; the building blocks are simple (one extra `torch.autograd.grad(loss, logits)` plus a masked squared sum) but plumbing them through the analyzer cleanly is its own task.

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

- **Real text, not random integers, for any observable that gets discussed.** The chi_net term is the squared Frobenius norm of the parameter Jacobian *evaluated at the input*; off-distribution random tokens put the evaluation at a meaningless point in input-space. Tokenized real text (Pride and Prejudice prose + a Python module in the example) puts the evaluation back on the model's training manifold. Synthetic data is fine for shape-only benchmarks (the sweeps under `examples/benchmark/` correctly use random integers because they measure wallclock and memory only) but anything that reports an observable value needs real data.
- **Cross-pair observables for the LNP-relevant story.** `cross_pairs=[(A, B)]` adds `δL(A, B)` and `chi_pos(A, B)` rows to the parquet at zero extra backward cost (the gradients are already cached from the self-pair work). The cross observables are the most informative LNP quantities — they tell you whether learning on A helps or hurts B (positive `δL(A, B)` → transfer, negative → interference). The deployment example uses `(prose, code)` and shows the cross `δL` going from `+3.25` at step1 → `−18` at step143000 — i.e. prose and code gradients become anti-aligned during training, the negative-`chi_pos` regime from the paper (arXiv:2605.31244 §A.2).
- **Log-spaced early checkpoints.** Pythia ships log-spaced revisions (`step0`, `step1`, `step2`, ..., `step512`) before the every-1000-steps main phase. The early phase reveals dynamics that are invisible from `step1000` onward — for `pythia-14m` the example surfaces a `chi_net` U-shape at step64 and a `chi_loss` "warmup cliff" where the model literally doesn't improve in the first 64 SGD steps. **Always include early checkpoints when sweeping a Pythia-like model**; the cost is ~4 extra checkpoint loads.

## Dependencies

Hard:
- `torch >= 2.2, < 2.11` — upper bound exists because `uv add matplotlib` in the v1.1 hardening pass silently upgraded torch to 2.11 and broke the resident `torchvision 0.23` build. The bound is loose; if a future maintainer upgrades `torchvision` manually they can drop it.
- `transformers >= 4.40`
- `pyarrow` (parquet sink)
- `numpy`
- `safetensors >= 0.4` — reading `model.safetensors` of a native checkpoint bundle. Already a transitive dependency of `transformers`; declared because `vatis/models/local_bundle.py` imports it directly.
- `matplotlib >= 3.10` and `tqdm` — only used by the scripts under `examples/`; vatis core imports neither. Shipped as the `examples` extra (`vatis[examples]`), not a hard dependency.

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
- **Tests:** 307 collected (266 unit + 31 cross-validation cases + 8 integration + 2 golden). Run via `.venv/bin/python -m pytest tests`. CI runs the unit tier only. Measured tiers: `-m "not integration and not cross_validation"` → 266 passed, 2 skipped (the golden gate opts itself out — see below); `-m golden` → 2 passed; `-m integration` → 8 passed; `-m cross_validation` → 30 failed, 1 passed **when `perspic` is not installed** (every failure is `ModuleNotFoundError: No module named 'perspic'`; that tier needs perspic in the venv). **With perspic available** (measured 2026-11-13 from a perspic environment, torch 2.9.1): 31 passed, after the two tight tests' probe count was raised — see "Known issues".
  - `tests/unit/` (266 tests) — runs in CI. Uses a hand-built ~1M-param toy transformer (defined in `tests/fixtures/tiny_transformer.py`). Tests math correctness: Hutchinson convergence to the exact trace as `n → ∞`, `χ_loss` closed-form correctness on the supported CE shapes, `δL` symmetry, normalization invariants, the chi_pos combinator, the per-seq-CV memory pre-check, and the analyzer's contract assertions. v1.1 added exact-NTK ground-truth tests for `chi_pos` and `δL(A, B)` via an explicit per-token Jacobian builder (`_build_full_jacobian_and_u` in `tests/unit/test_chi_net.py`) — this is the most rigorous correctness reference because it bypasses Hutchinson entirely. v1.2 task 1 added a regression test for the `valid_mask_fn`/`chi_loss` intersection fix. `tests/unit/test_local_bundle.py` (37 tests) covers the native bundle source: round-trip exactness (bf16 → fp32 θ, fp32 optimizer tensors, metadata/meta fields), the canonical-order alignment tests described under "Native checkpoint bundles", laziness (≤1 checkpoint's tensors live, checked by gc tensor-byte accounting + weakrefs), the `module:function` factory path, and every mis-alignment the loader must reject. Its bundles are hand-written by `tests/fixtures/synthetic_bundle.py`, which imports **none** of the loader's code — otherwise an ordering bug would cancel out between writer and reader.
  - **The preconditioned ruler (WP0 component 03)** adds three files. `tests/unit/test_preconditioner.py` (27) covers the `Preconditioner` invariants, the weighted reductions, and **T-adamw-state** (20 real `torch.optim.AdamW` steps; the built `P` is `torch.equal` to a hand computation from the optimizer's own internals, with a companion test proving the bias-correction exponent is really pinned). `tests/unit/test_chi_pos_p.py` (15) is the exact-answer suite: **T-identity** (`P = I` reproduces the raw triple at 1e-10 on fp64), **T-GN** (`P = (JᵀJ)⁺` ⇒ `chi_pos_P = 1/M`), **T-bruteforce** (explicit `Θ_P` by eigendecomposition — an exact branch plus a stochastic branch whose Rademacher tolerance `√(2Σ_{i≠j}A_ij²/n)` is derived in the test, run over **both** `per_sequence_cv` memory paths — the opt-in `O(B·P)` one forms both P terms from stored per-sample gradients rather than an output-space projection, so it needs its own reference; a companion test also pins the two paths to each other probe-for-probe), **T-flattening-direction** (`P = diag(π)⁻¹` ⇒ exactly `1/M`, and raw `chi_pos` below it on a tail residual), **T-batchsize** (cross-pair components independent of `|A| ∈ {8,32,128}` over 12 seeded repeats, by a 4σ standard-error test plus a 20% backstop). `tests/unit/test_analyzer_preconditioned.py` (11) covers the plumbing: raw rows unchanged bit-for-bit, the parquet descriptor column, the step-0 skip, cross-pair P symmetry, and the pipeline half of T-identity through `analyze()` → parquet at 1e-6. Shared toys and ground-truth helpers live in `tests/fixtures/preconditioned_toys.py`. The later audit added eight more unit files: `test_preconditioner_vs_torch.py` (19 — `P` against the update `torch.optim.AdamW` really applies, incl. the pre-step offset), `test_definition_fidelity.py` (11), `test_onehot_bridge.py` (18), `test_layout_alignment.py` (5 — incl. the same-shape permutation the loader *cannot* detect), `test_masking_intersection.py` (13), `test_chi_net_probe_exactness.py` (7), `test_guards.py` (10) and `test_parquet_sink.py` (5). An independent review then injected 40 behaviour-changing bugs; 11 survived the unit + golden tiers, and `test_bundle_analyzer_binding.py` (21) closes the ones that mattered: per-step binding of weights / optimizer state / update batch through the analyzer (the older bundle fixtures write the same θ at every step), the cross-pair `chi_pos_P` row, the `chi_pos_P` branch of the range warning, the multi-chunk weighted reductions, and the loader inputs it mishandled (0-dim parameter, never-updated parameter, per-group `eps`, non-Adam optimizer, step mismatch, persistent buffers, descriptor under an override).
  - **Two tolerance facts that recur in those tests** and are worth knowing before tightening anything: (1) the estimators return **fp32** scalars by the repo's pre-existing convention (the `chi_pos` combinator itself has divided in fp64 and returned fp64 since the S3 guard) (and T-identity *requires* it — an fp64 `chi_net_p` could never match the fp32 `chi_net` at 1e-10), so an assertion on a public scalar is bounded below by a few multiples of `2^-24` relative; where the spec asks for 1e-8 the tests assert both the fp32 value against `max(1e-8, fp32 slack)` and the same quantity recomputed from the fp64 components against 1e-8. (2) Rademacher Hutchinson has variance `2Σ_{i≠j}A_ij²/n`, which is **exactly zero for diagonal `A`** — that is why the `Θ_P = I` constructions (T-GN, T-flattening) are equalities rather than statistical statements, and the tests assert that precondition instead of assuming it.
  - `tests/golden/` (2 cases) — the **WP0 acceptance gate**, marked `@pytest.mark.golden`. It writes its own checkpoint bundle (`tests/utils/bundle_fixture.py`: ~100 CPU AdamW steps, frozen schema, pre-step semantics, full-sequence `batch_a.npy`), asserts the bundle round-trips to the batches its own loop trained on, runs both rulers over it, checks a `P = I` re-run against the raw columns and `|χ_pos| ≤ 1.5`, then **prints without asserting** the linearization sum `Σ_t η_t δL(A_t,B)` against the realized change in `loss(B)` and the share of `χ_net^P` carried by `1/ε`-capped coordinates. Parametrized over the self-contained tiny transformer (always) and the project Llama from `/workspace/antonio_code` (tied weights, rotary — skipped cleanly when absent, so the suite stays self-contained). Unlike `integration`/`cross_validation`, which callers deselect with `-m "not ..."`, `golden` postdates those call sites and therefore **opts itself out** via the `tests/conftest.py` hook: it runs only when the `-m` expression names `golden`, and shows as skipped-with-reason otherwise.
  - `tests/integration/` (8 tests) — gated behind `pytest -m integration`, **skipped in CI**, runnable locally. `test_hf_load.py` (3) uses `SimpleStories/SimpleStories-1.25M` to verify the HF loading path on a real hub checkpoint; `test_ddp_loopback.py` (5) pins DDP against single-process, including the preconditioned triple. The DDP test is currently a 2-rank CPU loopback only — **the multi-GPU DDP path has never been validated on real GPUs**. Adding a real-GPU DDP test is on the v1.2 work order.
  - `tests/cross_validation/` (31 tests) — gated behind `pytest -m cross_validation`, **skipped in CI** (requires perspic in the environment), runnable locally. **Cross-validates every observable against perspic** as the ground-truth reference implementation — both self-pair and cross-pair. The protocol:
    1. Build a small model that perspic can wrap (a `pl.LightningModule` with `criterion`) — e.g. a tiny MLP or the toy transformer with a Lightning shim.
    2. Run `perspic.analyzer(...)` on a single training step with a fixed batch and seed, capturing the logged `chi_loss`, `chi_net`, `chi_align`, `chi_coup`, and `grad_norm_squared` (= `δL`).
    3. Run `vatis.analyze(...)` on the **same model weights, same batch, same seed**, with the requested `chi_net_method` and an `n_hutchinson` large enough to drive the Hutchinson noise below the asserted tolerance.
    4. Assert that `vatis.chi_loss ≈ perspic.chi_loss`, `vatis.chi_net ≈ perspic.chi_net` (within Hutchinson std), `vatis.delta_loss ≈ perspic.grad_norm_squared`, and **`vatis.chi_pos ≈ perspic.chi_align ≈ perspic.chi_coup`**.
    5. Repeated with `chi_net_method="per_sequence_cv"` to verify the control-variate path agrees.
    6. Repeated with both `opacus` and `functorch` perspic backends to make sure normalization conventions match across the board.

    v1.1 added a tight-tolerance test (now `test_all_observables_tight_tolerance_at_n262144`; see "Known issues" for why the probe count was raised) at relative 0.3% / absolute 1e-6, originally with `n_hutchinson = 16384`, plus a heavy-padding test (`test_heavy_padding_matches_perspic_on_valid_subset`) that drives 5/8 samples through `labels=-100` and asserts the padded vatis run agrees with perspic on the unpadded subset.

    v1.2 added **cross-pair cross-validation** (17 tests), closing the gap where only self-pair observables (`δL(A,A)`, `chi_pos(A,A)`) were validated against perspic. The cross-pair tests use perspic's `Linearizer.compute(x1, y1, x2, y2)` for cross `δL(A,B)` and `SamplewiseCalculator.compute_cross_metrics()` for the geometric-mean `chi_loss_cross` / `chi_net_cross`, then `CouplingCalculator` for `chi_coup_cross`. Coverage:

    - **Basic cross `δL` and `chi_pos`** (`test_cross_delta_loss_matches_perspic`, `test_cross_chi_pos_matches_perspic`): two batches from different seeds, both perspic engines × both vatis methods. `δL` is deterministic (rel 1e-4); `chi_pos` within 2% (Hutchinson noise via geometric-mean `chi_net`).
    - **Tight tolerance at n=262144** (`test_cross_observables_tight_tolerance_at_n262144`, originally n=16384): catches constant-factor normalization bugs hidden by the 2% bound.
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
- **CI:** GitHub Actions workflow at `.github/workflows/ci.yml`. Single matrix entry `python-3.11`, runs `ruff check`, `ruff format --check`, `mypy`, `pytest -m "not integration and not cross_validation"` against `uv sync --extra dev`. Triggers on push to any branch and PRs to `main`.

## Working conventions (for the agent building this)

- **Use subagents where it makes sense.** Spawn `Explore` subagents for codebase searches that span more than 2–3 files (e.g. "find all places perspic computes χ_loss" before writing the cross-validation tests). Spawn `Plan` subagents for non-trivial implementation steps where the design isn't obvious from CLAUDE.md alone. Spawn parallel `general-purpose` subagents when several independent investigations can run at once (e.g. checking HF API for `from_pretrained(revision=...)` while simultaneously checking how perspic's logger formats its outputs). Don't spawn subagents for trivial edits or single-file reads — that's overhead for no gain.
- **All installs go into the existing venv only. Nothing global, ever.** The venv lives at `.venv/` in the project root and already has perspic installed editable. Use `uv pip install …` (which respects the active venv) or `uv add …` for pyproject-tracked deps. **Never** run `pip install` without the venv activated, **never** use `--user`, **never** touch the system Python. If a tool seems to want a global install, stop and ask. Verify with `which python` → should resolve to `.venv/bin/python` before any install.
- **Run tests through the venv too:** `.venv/bin/pytest …` or `uv run pytest …`. Same for `ruff`, `mypy`, `python -m vatis`.

### Permission model (what's enforced vs convention)

The `settings.local.json` allow list bounds *some* of what you can do, but **most safety is behavioral, not technical.** Read this section before making any change that touches things outside the vatis project tree.

**Technically enforced (the system will block you):**
- `Write` and `Edit` tool calls are restricted to `<project-root>/**`. You cannot use these tools to modify files outside the project.
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

**The real safety boundary is the host filesystem.** Treat anything outside `<project-root>/` as off-limits, regardless of which tool would let you reach it. The only exceptions are read-only operations (which are explicitly allowed everywhere via `Read(/**)` etc.) and the HuggingFace cache (which transformers will populate at `~/.cache/huggingface/` as a side effect of `from_pretrained` — that's expected and fine).

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
- Padding is handled at one place: `vatis/core/normalization.py::valid_token_mask`. Every accumulator multiplies by this mask. No ad-hoc masking elsewhere. Concretely: the analyzer builds *one* closure (`analyzer.py::_intersected_valid_mask_fn`) that composes the bundle's `valid_mask_fn` with `valid_token_mask`, and passes that same closure to the `chi_loss` sum, the `delta_loss` micro-batch weighting and the `χ_net` estimator. If you add an accumulator, it takes that closure — not `bundle.valid_mask_fn`. Two accumulators drifting apart here is this repo's most expensive past bug class (twice).

## Glossary

| symbol | code name | meaning |
| --- | --- | --- |
| `χ_loss` | `chi_loss` | `‖∇_f L‖²` |
| `χ_net` | `chi_net` | `‖∇_θ f‖_F² = Tr(Θ)` |
| `χ_pos` | `chi_pos` | `δL / (χ_loss · χ_net)` — "spectral position". **Naming note:** in vatis this is `chi_pos`. The same quantity is called `chi_align` (and sometimes `chi_coup`) in perspic and in the paper. When cross-validating against perspic (see "Testing" below), map `perspic.chi_align ↔ vatis.chi_pos` and `perspic.chi_coup ↔ vatis.chi_pos`. |
| `δL` | `delta_loss` | `⟨∇_θ L^A, ∇_θ L^B⟩` |
| `P` | `Preconditioner.diag` | diagonal preconditioner, 1-D in the canonical parameter order |
| `Θ_P` | (not stored) | effective kernel `J P Jᵀ` — what the optimizer trains on |
| `χ_net^P` | `chi_net_P` | `‖P^{1/2}∇_θ f‖²_F = Tr(Θ_P)` |
| `δL^P` | `delta_loss_P` | `∇_θL^B · P ∇_θL^A` |
| `χ_pos^P` | `chi_pos_P` | `δL^P / (χ_loss · χ_net^P)` — spectral position on the optimizer's ruler |
| `Θ` | (not stored) | empirical NTK; never materialized |
| `N_A`, `N_B` | `n_valid_a`, `n_valid_b` | valid (non-pad) token counts in batches A and B |
| `n` | `n_hutchinson` | number of Hutchinson probe vectors |

## Known issues

Latent bugs and quirks that an agent walking into the repo should be aware of.

- **`ParquetSink` truncates on re-open within the same path** — calling `analyze()` twice with `sink="path.parquet"` overwrites the file each time. The deployment example sidesteps this via the single-`analyze()`-with-multiple-revisions pattern, but it's an obvious footgun for any user who loops manually. Either fix or document explicitly.
- **Multi-GPU DDP path is unverified on real GPUs** — only the 2-rank CPU loopback test exercises it. No real-GPU integration test exists.
- **Custom (non-CE) `loss_fn` is not really supported** — `delta_loss` will use whatever `loss_fn` you pass, but `chi_loss` always uses the closed-form CE shortcut. Mixing the two produces wrong observables: a *sum*-reduced CE inflates `chi_pos` by the square of the valid-token count **per micro-batch** (`N_valid²` only when the whole batch is one micro-batch; 49 instead of 3136 at `micro_batch_size=1` on the 56-token test batch) while leaving `chi_loss` and `chi_net` bit-identical. The autograd fallback was deleted in v1.1 task 6 to avoid silent wrong-answer modes. Real support is the v1.2 feature work. **There is now a backstop**: `|χ_pos| ≤ 1` is a Cauchy–Schwarz consequence, so the analyzer warns when a published `chi_pos`/`chi_pos_P` leaves `[−1, 1]` by more than 0.5 (the slack covers a legitimate overshoot at small `n_hutchinson`). Read it as a backstop, not a detector: it only fires once `N²·χ_pos` actually exceeds the bound, which is true for essentially any realistic `N` but *not* on a tiny toy — measured at `N = 14`, a fully wrong sum-reduced run lands at `0.85` and stays silent (`tests/unit/test_guards.py::test_the_range_warning_is_a_backstop_not_a_detector`).
- **The two tight cross-validation tests used to be decided by their seed (fixed 2026-11-13).** `test_all_observables_tight_tolerance_at_n16384` and its cross-pair sibling asserted `chi_net` / `chi_pos` within 0.3% of perspic at `n_hutchinson = 16384`, on the premise that the noise floor is "far below 0.3%" there. It was not: the analytic Rademacher standard deviation of that estimate (`√(2Σ_{i≠j}A_ij²/n)/Tr A` for the test's MLP) is 0.33%, and over 32 seeds it measured 0.28–0.34% for both estimators on both `dev` and this code, with 11–13 of 32 seeds outside the tolerance. The pinned seed 0 passed with the old on-device probe draw (−0.23%) and failed with the bit-packed CPU draw (−0.33%) — the draw, not a bias (32-seed means within 2σ of perspic on both trees; `make_probe`'s second moments imply a trace bias of +0.002% ± 0.007% over 4e7 probes). Both tests now run at `n_hutchinson = 262144` (`TIGHT_N_HUTCHINSON`, tests renamed `..._at_n262144`), where the standard deviation is 0.083% and the **unchanged** 0.3% tolerance is 3.6σ. They pass on this code and on `dev`'s package code. The price is runtime: the tier takes about 11 minutes single-threaded instead of about 2.5.
- **`OpacusEstimator` is still a stub** — the original v1 spec called this v1.1 work; v1.1 was hardening instead. It remains open. It also has no `chi_net_P` path, so `preconditioner=...` with `chi_net_method="opacus"` fails at the same `NotImplementedError`.
- **AdamW's `P` puts untouched coordinates at `1/ε`, by design.** Any parameter element the gradient never touched keeps `exp_avg_sq == 0` exactly (unused vocabulary rows are the obvious population; on a real bundle `exp_avg_sq.min() == 0` at every checkpoint), so `P = 1/(√0 + ε) = 1/ε` — ~`1e6`× a live coordinate's weight at `ε = 1e-8`. Since `χ_net^P = Σ_i P_i ‖J_{:,i}‖²`, such a coordinate is harmless only if its Jacobian column also vanishes on the *measurement* batch; a token present in a held-out batch but absent from the batches that built `v` is **not** harmless, and on held-out data that is the normal case. The convention is a frozen project decision, so vatis applies no clipping, flooring or dead-coordinate mask — instead `vatis.core.preconditioner.epsilon_cap_mask` identifies those coordinates so a caller can report their share of `χ_net^P` (the golden test prints exactly that, with a per-parameter breakdown). Read a `χ_net^P` without checking that share at your peril.
- **A resident `P` costs one extra parameter-sized fp32 vector; building it from a bundle peaks at about five** (the loaded `exp_avg` + `exp_avg_sq`, the flattened copy that becomes `P` in place, and the validation temporaries — measured 4.77 at 60M elements; it was 7 before the in-place build). The optimizer state is released as soon as `P` exists. `_should_offload_cross_grads` (the cross-pair GPU/host heuristic) does not account for it. Irrelevant at toy scale; for a multi-billion-parameter model with `cross_pairs` on a borderline card, pass `cross_grad_storage="cpu"` explicitly.

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

1. **Real-text eval batches** — replaced random-integer batches with tokenized prose (Pride and Prejudice) and tokenized code (a Python module). The chi_net term is Jacobian-of-the-model-evaluated-at-the-input; off-distribution random tokens make the trajectory uninterpretable. Real text puts the evaluation back on Pythia's training manifold. Also added `cross_pairs=[("prose", "code")]` for the LNP-relevant cross-distribution gradient correlation.
2. **Log-spaced early checkpoints** — added `step1, step8, step64, step512` to the sweep alongside the existing `step1000` → `step143000` main phase. Revealed three findings invisible from `step1000` onward: `chi_net` U-shape at step64 (parameter Jacobian *shrinks* in the first ~100 steps before growing), `chi_loss` warmup cliff (no measurable learning in the first 64 steps), non-monotonic cross `δL` (drops, rebounds, then declines through zero).
3. **Analysis script** — `examples/analyze_results.py`, a small post-processing script that reads `results.parquet` and computes the normalized cross-batch gradient correlation `cos(g_A, g_B)`. Demonstrates the compute/analysis split (compute is expensive, analysis is cheap) and that the parquet schema is the contract — the script has zero vatis imports.

End of v1.1: 67 tests, all green; 13 commits over the hardening + follow-up arc; CLAUDE.md and SESSION_SUMMARY.md fully reconciled with the code.
