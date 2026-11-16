# vatis — specification

Compute LNP observables (`chi_loss`, `chi_net`, `chi_pos`) on **pretrained
model checkpoints**, scaled across multiple GPUs via DDP. Sister package to
`perspic`, which computes the same quantities *during* Lightning training;
vatis is for the regime where you can't pretrain yourself and instead probe
published checkpoints (Pythia, OLMo, …). It also reads native checkpoint bundles
that carry the optimizer state, and can then measure the same decomposition on
the optimizer's preconditioned kernel.

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

### Optimizer-aware measurement (`chi_pos_P`)

The observables above describe plain gradient descent. An optimizer-aware
measurement asks the same questions about the update an optimizer actually
forms. What vatis implements so far is the **preconditioner** part of that: the
three *preconditioned observables* below, which account for how the optimizer
rescales the gradient coordinate by coordinate (its geometry). Momentum and
weight decay are not included yet — see "What `delta_loss_P` is not".

An optimizer with update `Δθ = −η P ∇L` does not train on the raw eNTK but on
the **effective kernel** `Θ_P = J P Jᵀ`, with `J = ∇_θ f` the parameter Jacobian
of the outputs (so `Θ = J Jᵀ`). vatis measures the same LNP triple on
that kernel, alongside the raw one:

```
delta_loss_P(A, B) = ∇L^B · P ∇L^A        (diagonal P: elementwise-weighted dot)
chi_net_P          = ‖P^{1/2} J‖_F² = Tr(Θ_P)
chi_loss           = unchanged (kernel-independent)
chi_pos_P          = delta_loss_P / (chi_loss · chi_net_P)
```

The cross pair takes the geometric mean `‖P^{1/2}J^A‖_F · ‖P^{1/2}J^B‖_F`,
exactly as the raw cross pair does. `P = I` reproduces the raw triple
identically (asserted in `tests/unit/test_chi_pos_p.py`).

It costs **no extra backward**: `delta_loss_P` re-reduces the gradient vector the
raw `δL` already built, and `chi_net_P` comes out of the same estimator call and
the same probes as `chi_net`, via `Tr(J P Jᵀ) = E_z ‖P^{1/2} Jᵀ z‖²`. The
additional memory while a checkpoint is analyzed is one parameter-sized fp32
vector for `P` (building it from a bundle is costlier, see "Native checkpoint
bundles").

**`P` is diagonal throughout** — a 1-D vector in the canonical parameter order
(`vatis/core/preconditioner.py::Preconditioner`), segment-for-segment aligned
with the flat gradient. Non-diagonal `P` (e.g. Muon's orthogonalization) is out
of scope and the code is not designed for it.

P sources:

| source | `P` | how |
| --- | --- | --- |
| none | — (raw observables only) | `preconditioner=None`, the default |
| `identity` | `I` | `preconditioner="identity"` |
| `explicit` | user-supplied 1-D tensor | `Preconditioner.from_diagonal(...)` |
| `adamw` | `diag(1/(√v̂ + ε))`, `v̂ = exp_avg_sq/(1−β2^t)` | `source.adamw_preconditioner()` on a checkpoint bundle |
| `sgd` | `I` | `source.preconditioner()` on a bundle written by `torch.optim.SGD` |

`source.preconditioner()` picks between the last two from the bundle's saved
optimizer `param_groups`, so an optimizer comparison reads the same three
observables for both and the `preconditioner` column says which geometry a row
used. For SGD (with or without momentum) the geometry factor is the identity:
its `sgd` rows repeat the raw values exactly and exist at every checkpoint,
step 0 included. Momentum and weight decay are excluded for SGD for the same
reason as for AdamW (see "What `delta_loss_P` is not"). Any other optimizer is
refused.

**`P` leaves the learning rate out, so the bundle sources require one rate.**
For a single rate `η` it is a common factor of the update and `chi_pos_P` does
not depend on it. With per-group rates the update is `−η_g P ∇L` on group `g`:
the rates are then part of the optimizer's geometry, and neither the AdamW `P`
nor SGD's `P = I` describes it. A saved state whose param groups disagree on
`lr` is therefore refused by both bundle sources, at every checkpoint and
regardless of a `beta2` / `eps` override. The saved `lr` is the scheduled value
at that step, so groups that share one schedule and one base rate agree at
every checkpoint. For a run with per-group rates build the `P` you mean with
`Preconditioner.from_diagonal(...)`, e.g. the AdamW `P` scaled per group by its
rate relative to a reference rate.

**AdamW conventions.**

- `t` is the **saved** step counter. Bundles are pre-step (see "Native
  checkpoint bundles"), so that is the number of updates already folded into the
  saved `exp_avg_sq`, and hence the bias-correction exponent that matches it.
- β2 and ε come from the saved optimizer `param_groups` (what the optimizer ran
  with); `metadata.json` is a cross-check that warns on disagreement.
- `amsgrad=True` is refused (it preconditions with `max_exp_avg_sq`).
- **Step 0 / empty optimizer state:** the P rows for that checkpoint are
  **skipped** with a `UserWarning`; the raw rows are unaffected. Neither
  fallback would be honest — `P = 1/ε` scales every observable by ~`1/ε²`, and
  `P = I` would publish `chi_pos_P == chi_pos` under the `adamw` descriptor.
- **Untouched coordinates sit at `P = 1/ε`, by design.** Any parameter element
  the gradient never touched keeps `exp_avg_sq == 0` (unused vocabulary rows are
  the obvious population); relative to a coordinate with history that is a
  weight ratio of `√v̂/ε` (`1e6` for `√v̂ = 1e-2` at `ε = 1e-8`). Since
  `chi_net_P = Σ_i P_i ‖J_{:,i}‖²`, such a coordinate is harmless only if its
  Jacobian column also vanishes on the *measurement* batch; a token present in a
  held-out batch but absent from the batches that built `v` is not harmless.
  **The update batch is exposed in the same way:** `P` is the pre-step one, so a
  token the run sees for the first time at step `t` enters that step's
  `batch_a` rows at `1/ε`, while the step itself divides by a `v̂` that already
  contains its gradient. vatis applies no clipping or mask; it reports what
  those coordinates contribute instead — see "Coordinates at the ceiling".

#### Coordinates at the ceiling (`*_capped` rows)

AdamW's `P_i = 1/(√v̂_i + ε)` has a largest possible value, `1/ε`, and takes it
exactly where the second moment is zero — on every coordinate that has not yet
received a non-zero gradient. With `ε = 1e-8` such a coordinate weighs `√v̂/ε`
times as much as one with history, typically five to seven orders of magnitude.
Two situations make that matter:

- **Held-out data.** A token that occurs in the measurement batch but never
  occurred in training has a non-zero Jacobian column on its embedding row and
  `P = 1/ε` there.
- **The update batch itself.** `P` is built from the state *before* the step. A
  token the run meets for the first time at step `t` still has `v = 0` when
  step `t`'s own batch is measured, although its gradient on that batch is not
  zero. (The step itself is not affected in this way: it divides by a `v̂` that
  already includes that gradient.)

In either case a handful of coordinates can carry nearly all of `delta_loss_P`
and `chi_net_P`. In a small test model with a single first-seen token in the
update batch, 12 of 276 coordinates carried more than 99% of both. `chi_pos_P`
is then a statement about those few coordinates, not about the model.

vatis does not clip, floor or mask `P` — that would silently replace the stated
definition with another one. Instead, whenever the preconditioner has a ceiling
(the `adamw` source; `Preconditioner.ceiling`), each P observable is accompanied
by the same quantity computed with `P` set to zero everywhere except at the
coordinates sitting at the ceiling:

| row | meaning |
| --- | --- |
| `delta_loss_P_capped` | `Σ_{i at ceiling} P_i ∇_iL^A ∇_iL^B` — the part of `delta_loss_P` from those coordinates |
| `chi_net_P_capped` | `Tr(J P_c Jᵀ)` with `P_c` = `P` restricted to those coordinates — for a self pair, the part of `chi_net_P` from them |

How to read them:

- `delta_loss_P_capped / delta_loss_P` and (self pairs) `chi_net_P_capped /
  chi_net_P` are the shares. Near 0: the P observables describe the trained
  coordinates. Near 1: they describe coordinates the optimizer has not touched.
- `delta_loss_P` (self and cross pairs) and the self-pair `chi_net_P` are linear
  in `P`, so for them the remainder is the value restricted to coordinates with
  history. `chi_pos_P` on those coordinates alone is therefore available
  without re-running, but the formula differs between the two kinds of pair:
  - **Self pair** `(A, A)`, from that pair's own rows:

    ```
    (delta_loss_P − delta_loss_P_capped) / (chi_loss · (chi_net_P − chi_net_P_capped))
    ```

  - **Cross pair** `(A, B)`. Here `chi_net_P` is the geometric mean `√(a·b)` of
    the two batches' traces, which is not linear in `P`, and its capped row is
    `√(a_c·b_c)`, the geometric mean of the two capped parts. With `a`, `a_c`
    the `chi_net_P`, `chi_net_P_capped` rows of the self pair `(A, A)` and `b`,
    `b_c` those of `(B, B)`, and the numerator and `chi_loss` taken from the
    cross pair's rows:

    ```
    (delta_loss_P − delta_loss_P_capped) / (chi_loss · √((a − a_c)·(b − b_c)))
    ```

    The cross rows' own difference `chi_net_P − chi_net_P_capped` is
    `√(a·b) − √(a_c·b_c)`, which is **not** `√((a − a_c)·(b − b_c))`; do not use
    it as the restricted denominator, and use the two self-pair rows to form
    `chi_net_P` shares.
- A value of exactly `0` means no coordinate is at the ceiling (or none of them
  is touched by the batch).

**What "at the ceiling" means.** A coordinate counts when
`P_i ≥ (1 − 1e-6)/ε` (`vatis.core.preconditioner.epsilon_cap_mask` gives the
same mask), i.e. when `√v̂_i` is below about `1e-6 · ε`: a second moment that is
zero to rounding. A coordinate whose `√v̂_i` is merely comparable to `ε` is
**not** counted, although `ε` still sets most of its weight — at `√v̂_i = ε` it
has half the ceiling weight. The `*_capped` rows are therefore a lower bound on
the share of an observable that is dominated by `ε`.

The rows cost no backward and no additional parameter-sized vector: each
gradient the estimator already has is reduced once more with the restricted
weight. A `P` without a ceiling (`identity`, `sgd`, an `explicit` diagonal
unless you pass `ceiling=`) has no such rows.

#### What `delta_loss_P` is not

`delta_loss_P` is the first-order loss response to the *hypothetical* update
`Δθ = −η P_t ∇L^A`. **It is not the first-order loss change of the AdamW step
that was actually taken**, and should not be read as one. Three things separate
the two:

1. **Momentum is excluded.** AdamW applies `P` to the first moment `m̂`, not to
   the current gradient. `P` here is the geometry factor only: `Θ_P = J P Jᵀ` is
   defined for a preconditioner, and momentum is a gradient-history filter, not
   a metric.
2. **Decoupled weight decay is excluded.** It does not act on the gradient.
3. **`P` is built from the pre-step second moment.** The step applies
   `v̂_{t+1} = (1 − w) v̂_t + w g_t²` with `w = (1 − β2)/(1 − β2^{t+1})`. That
   weight is `O(1 − β2)` only once `t ≫ 1/(1 − β2)`; early in a run it is
   `≈ 1/(t+1)`, and on the update batch itself the missing term is that batch's
   own `g_t²`.

The exact first-order response of one AdamW step on batch `B` is

```
∇L^B · Δθ = − η (1−β1)/(1−β1^{t+1}) · ∇L^B · P_{t+1} ∇L^A      (current gradient)
            − η  β1   /(1−β1^{t+1}) · ∇L^B · P_{t+1} m_t        (momentum history)
            − η λ · ∇L^B · θ_t                                   (weight decay)
```

of which vatis currently measures the first dot product, with `P_t` in place of
`P_{t+1}` and without its prefactor. The other two terms — and a comparison of
the sum against realized loss changes — are not implemented yet. Until they are,
treat `chi_pos_P` as a statement about the *geometry* the optimizer imposes.
`−η · delta_loss_P` is the first-order prediction for the *hypothetical* update
`−η P_t ∇L^A`. **Do not use it as the loss change of the step that was taken.**

How far apart the two are was measured on a real bundle (an 11M-parameter
Llama trained with AdamW, `β = (0.9, 0.95)`, `ε = 1e-8`; `B` = the step's own
update batch, fp32 on the CPU). "Step taken" is the sum of the three terms
above, built from the optimizer states saved before and after the step and the
loss gradient at the stored weights:

| step | step taken, first order | current-gradient term | momentum term | `−η · delta_loss_P` | `delta_loss_P` with `P_t` instead of `P_{t+1}` |
| --- | --- | --- | --- | --- | --- |
| 1 | −1.098 | −0.577 | −0.522 | −1.272 (16% larger in magnitude) | 16% larger |
| 3 | −1.393 | −0.429 | −0.964 | −1.845 (32% larger in magnitude) | 25% larger |

The weight-decay term was below `1e-4` in magnitude at both steps. The same
change computed as `∇L^B · (θ_{t+1} − θ_t)` from the two stored weight files was
−1.097 and −1.392. Momentum already carries nearly half of the step at step 1
and more than two thirds at step 3, so the gap is not a small correction. This
is one bundle: do not read the percentages as constants.

**Step 0 is missing from the P rows, and it is the least SGD-like step.** No
pre-step `P` exists at step 0 (the optimizer state is empty), so the P rows
start at the first checkpoint after it. Yet the first update is the most
non-SGD one: `v̂_1 = g_0²`, so it is essentially sign descent,
`Δθ ≈ −η · sign(g_0)`.

#### Reading the P rows

- **`|chi_pos_P| ≤ 1` for any non-negative `P`**: `delta_loss_P` is an inner
  product of `P^{1/2}`-weighted gradients, and Cauchy–Schwarz bounds it by
  `chi_loss · chi_net_P`. The published value divides by an *estimate* of
  `chi_net_P` and can overshoot, hence a warning rather than an error (see
  "Loss handling").
- **`chi_pos_P` does not change when `P` is multiplied by a constant;
  `delta_loss_P` and `chi_net_P` do.** They carry the units of `P`, so their
  values are not comparable between `adamw` rows and `sgd` rows.
- **To isolate the effect of the geometry, compare `chi_pos_P` with `chi_pos`
  on the same checkpoint.** An `adamw` run and an `sgd` run are different
  trajectories; a difference between their rows mixes the geometry with where
  each run went.
- **`sgd` rows equal the raw rows by construction** (`P = I`); they carry no
  information beyond them.
- **No standard error is reported.** `chi_net_P` comes from the same probes as
  `chi_net` and is typically the noisier of the two, because `P` is very
  uneven and a few heavily weighted coordinates carry much of the estimate.

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
  Native **checkpoint bundles** with optimizer state (`load_local_bundle`). The
  **preconditioned observables** (`chi_pos_P`) for diagonal `P` (`identity`, `explicit`,
  `adamw` sources) alongside the raw one.
  Tokenized data already in model-ready form. DDP across user-specified
  `num_gpus`. Two `χ_net` estimators (`hutchinson`, `per_sequence_cv`) with
  auto-selection. Parquet result sink (canonical) plus optional W&B and
  TensorBoard. CLI via `python -m vatis run …` that wraps `torchrun`.
- **Out:** Tokenization (examples tokenize inline; vatis core provides no
  tokenizer). Checkpoint discovery / sweep scheduling. FSDP. Tensor / pipeline
  parallelism. `OpacusEstimator` (stub only). Custom non-CE losses for
  `chi_loss`. Per-sample variance observables beyond the cross-sample alignment
  matrix. Models that don't fit on one GPU. Non-diagonal `P`. Momentum and
  weight decay in the optimizer-aware response (see "What `delta_loss_P` is
  not"). **Optimizer states in third-party formats**: only the native bundle
  schema is read — there is no loader for the optimizer states published with
  Pythia, OLMo or other hub checkpoints, so `adamw` is available for bundles
  only (an `explicit` `P` works with any model if you build and align the
  vector yourself). The CLI does not expose bundles or the preconditioner; use
  the Python API.

The core (`vatis/core/`) is decoupled from model loading, data plumbing, and
parallelism, so adding any of these later is straightforward.

## Architecture

```
vatis/
├── __init__.py            # public API: analyze(), Analyzer, Preconditioner, load_local_bundle
├── core/
│   ├── observables.py     # chi_loss, delta_loss, chi_pos (combine χ_net from estimator)
│   ├── chi_net/
│   │   ├── base.py        # ChiNetEstimator ABC
│   │   ├── hutchinson.py  # HutchinsonEstimator (always-works fallback)
│   │   ├── per_seq_cv.py  # PerSequenceControlVariateEstimator (default for small B)
│   │   └── opacus.py      # OpacusEstimator (stub)
│   ├── probes.py          # Rademacher / Gaussian probe vector generation, masking
│   ├── param_layout.py    # ParameterLayout: the canonical parameter flattening order
│   ├── preconditioner.py  # Preconditioner (diagonal P) + elementwise-weighted reductions
│   └── normalization.py   # valid-token counting, √(N_A·N_B) factors
├── models/
│   ├── __init__.py        # exports load_hf_model + load_local_bundle
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
    observables=None,                           # None = all raw (+ all P if a preconditioner is set)
    preconditioner=None,                        # None | "identity" | Preconditioner | fn(ckpt_id) -> P | None
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

### Native checkpoint bundles

Besides the hub, vatis reads **checkpoint bundles** written by a training run:
a run directory in a fixed schema that carries the optimizer state and the
update batch next to the weights.

```
<run_dir>/
  metadata.json                # model config, optimizer type + hparams, lr schedule,
                               # seed, B_train, seq_len, dataset id, git hashes
  param_names.json             # ordered [ [name, shape], ... ] — the canonical flattening
                               # order; index-aligned with the optimizer state_dict indices
  checkpoints/step_00000123/
    model.safetensors          # weights, bf16
    optim.pt                   # torch optimizer.state_dict(), fp32, saved PRE-step
    batch_a.npy                # int32 [B_train, seq_len] token ids of the update batch A_t
    meta.json                  # step, tokens_seen, lr, train_loss, wall_time
```

```python
from vatis import Analyzer, load_local_bundle

source = load_local_bundle(run_dir, "my_project.models:build_model")  # or a callable

analyzer = Analyzer(
    eval_batches={"train": source.batch_a_spec(), "val": val_batch},
    preconditioner=source.preconditioner(),            # P per checkpoint, from its saved state
    cross_pairs=[("train", "val")],
    sink="results.parquet",
)
analyzer.run(source.model_specs())                     # drop-in for the HF (name, revision) loop

for ckpt in source:                                    # or drive the loop yourself
    v = ckpt.optimizer_state.flat_state("exp_avg_sq")  # 1-D, canonical order
    t = ckpt.optimizer_state.step_count                # pre-step update count
    batch = ckpt.batch_a                               # {"input_ids", "labels"}
```

- **This is the only optimizer-state format vatis reads.** The schema is vatis's
  own; a training run has to write it. Optimizer states shipped with hub
  checkpoints are not supported (see Scope).
- **vatis stays project-agnostic.** The architecture comes from a user-supplied
  `model_factory(metadata) -> nn.Module` (a callable, or a `"module:function"`
  import string). Package code never imports a training stack.
- **`param_names.json` is the canonical flattening order, with one code path.**
  It is parsed into a `ParameterLayout` (`vatis/core/param_layout.py`), and that
  same object builds `ModelBundle.params` (looked up **by name in file order**)
  and flattens the optimizer state, so `flat_state("exp_avg_sq")` is
  segment-for-segment aligned with the flat gradient. The loader refuses what it
  can prove is misaligned: optimizer `param_groups` index lists that aren't
  `[0 … n−1]` in order, state tensors whose shape contradicts the recorded
  shape, model parameters the order doesn't list, names the model or weight file
  doesn't have.
- **What the loader cannot prove.** A torch optimizer `state_dict` stores state
  by position, not by name. If a writer records `param_names.json` in an order
  that differs from the order the optimizer was built with, and the swapped
  parameters have the same shape, nothing in the bundle reveals it
  (`tests/unit/test_layout_alignment.py` pins this). **The writer must build the
  optimizer from, and record names in, the same `named_parameters()` order.**
- **Pre-step semantics.** The checkpoint at step `t` holds `θ_t` and the
  optimizer state from *before* the step-`t` update; `batch_a.npy` is the batch
  step `t` is about to consume. The directory step is the number of optimizer
  updates already applied, so it equals the optimizer state's own step counter.
  When the two disagree, the optimizer state and the weights may not belong
  together, and `adamw_preconditioner()` / `preconditioner()` **refuse the
  checkpoint** (`BundleSchemaError` naming both numbers). For a bundle whose
  step was merely stored wrongly, `on_step_mismatch="warn"` on either method
  downgrades that to a `UserWarning` and builds `P` from the saved state's own
  counter; `"raise"` is the default and no other value is accepted. (An SGD
  state has no step counter, so there is nothing to compare for it.) A step-0
  checkpoint legitimately has an empty optimizer state; `flat_state` raises
  rather than inventing zeros.
- **`batch_a.npy` holds whole token sequences.** The loader turns a row `ids`
  into inputs `ids` and next-token labels (`labels[:, :-1] = ids[:, 1:]`, last
  position ignored), i.e. the pairs `ids[:, :-1] → ids[:, 1:]`. The file carries
  no padding information, so every position counts as a target.
- **Runs whose training loss ignored some targets (pad tokens) are not
  supported for the update batch on this branch.** The loader does not apply
  the pad convention a bundle records (`mask_pad_in_loss` / `loss_ignore_index`
  in `metadata.json`), so for such a run the update-batch rows describe a
  different loss than the one that was descended. A `valid_mask_fn` that
  rejects the pad positions does **not** repair this: it restricts the token
  counts and the probes but not the loss, which leaves `delta_loss` wrong and
  dependent on `micro_batch_size` (see "Loss handling"). The correct handling —
  the labels as the single source of truth for which positions count, with the
  loader applying the run's recorded pad convention to them — is a follow-up
  change.
- **Weights are stored as parameters only.** A persistent buffer the model
  factory creates keeps the factory's value, and the loader warns about it:
  correct for constants, wrong for anything a training run changes.
- **A bf16 weight file holds a rounded `θ_t`.** The loader casts whatever
  floating dtype the file holds (bf16 by default) to the dtype requested from
  `load_local_bundle(dtype=...)`; the cast adds no error, but it does not undo
  the rounding. Measured on a real bundle (an 11M-parameter Llama trained in
  fp32, weights stored as bf16): the loss gradient at the stored weights agreed
  with the gradient the optimizer itself used (reconstructed from consecutive
  first-moment states) to 0.24% relative error, cosine 0.999997, at steps 1 and
  3 — for a gradient at one checkpoint the rounding is a sub-percent
  perturbation. The *difference* between consecutive weight files is another
  matter: early in training it is dominated by the rounding. Against the
  update predicted from the optimizer state, `θ_{t+1} − θ_t` from the files had
  a relative error of 0.81 at step 1 and 0.66 at step 3. **Comparisons that
  need a single step's weight change element by element need fp32 weight
  files.**
- **`load_local_bundle(device=..., dtype=...)` decide placement and precision.**
  A bundle source hands the analyzer ready-built models, so `Analyzer(device=,
  dtype=)` do not move or recast them. The optimizer state is read and kept on
  the CPU whatever `device` is — only the finished `P` is moved — so
  `flat_state(...)` returns a CPU vector unless called with `device=`.
- **Only Adam / AdamW states are accepted** by `adamw_preconditioner()`. The
  optimizer is identified from the saved `param_groups`, not from a name:
  RAdam, NAdam and Adamax groups also carry `betas`, but only Adam / AdamW
  groups have the `amsgrad` key, and a state without it is refused whether or
  not `metadata.json` names an optimizer. A bundle whose `metadata.json` names
  another optimizer (under `optimizer_name`, `optimizer_type`, `optimizer` or
  `optimizer_class`) is refused as well rather than labelled `"adamw"`; so are
  an SGD state (use `source.preconditioner()`),
  `amsgrad=True` in any parameter group, parameter groups that disagree on
  `lr` (also under an override, and for SGD bundles in `preconditioner()` — see
  "Optimizer-aware measurement"), parameter groups that disagree on
  `betas` / `eps` (unless both are passed explicitly) and a parameter that never
  received a gradient. An `eps` / `beta2` override is written into the descriptor, e.g.
  `adamw(eps=0.001)`.
- **Lazy per checkpoint.** Opening a bundle reads the two JSON files and stats
  the checkpoint directories; tensors load on first access, and iterating the
  source releases checkpoint `i−1` before yielding `i`. Building `P` needs the
  loaded optimizer state (two parameter-sized vectors) plus about three more
  transiently, all in host memory: the state is read and `P` is built on the
  CPU, and only the finished `P` is moved to the device. The state is released
  as soon as `P` exists. Under DDP every rank does this.

### Result format (parquet, long)

One row per `(checkpoint_id, eval_batch_name_A, eval_batch_name_B,
preconditioner, observable, value)`. Long format handles arbitrary eval batches / pair combinations without
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
preconditioner     str           # "none" on raw rows; the P descriptor on P rows
```

The preconditioned observables extend the format minimally: they are new
`observable` *values* (`delta_loss_P`, `chi_net_P`, `chi_pos_P`, and for a `P`
with a ceiling `delta_loss_P_capped`, `chi_net_P_capped`), not new columns, plus
the single `preconditioner` descriptor column (appended last). That column says
which `P` a row's value was computed with: `"none"` on the raw rows — including
`chi_loss`, which does not depend on the kernel and is written once — and the
descriptor (`"adamw"`, `"sgd"`, ...) on the P rows. A run without a
preconditioner emits exactly the rows it always did.

The observable names already tell the raw and the preconditioned values apart,
so the usual way to read a file is one pivot, which puts everything measured for
a checkpoint and batch pair on one line:

```python
import pandas as pd

df = pd.read_parquet("results.parquet")
key = ["checkpoint_id", "batch_a", "batch_b"]
wide = df.pivot(index=key, columns="observable", values="value")
wide["preconditioner"] = df[df.preconditioner != "none"].groupby(key).preconditioner.first()
# columns: chi_loss, chi_net, delta_loss, chi_pos, ..., chi_net_P, delta_loss_P, chi_pos_P, ...
```

Do not select the preconditioned values by filtering on `preconditioner`: that
drops `chi_loss`, which they share with the raw ones. A checkpoint without P
rows (an AdamW step-0 checkpoint) shows up as empty cells in the `_P` columns.
There is no `chi_net_P_normalized` row: it is `chi_net_P / √(n_valid_a ·
n_valid_b)` from the same row's own columns, and `chi_pos_P` — like `chi_pos` —
is invariant under the normalization.

### Distributed semantics

- User passes `num_gpus=N`. CLI launches `torchrun --nproc_per_node=N -m
  vatis._worker …`.
- Each rank loads the same checkpoint (vatis assumes the model fits on one GPU).
- Each eval batch is sharded across ranks along the batch dim. If `B % N != 0`,
  the last rank gets a smaller shard — the all-reduce handles this because we
  accumulate token-weighted sums, not means.
- Per micro-batch: forward → `χ_loss` accumulator. For `χ_net` Hutchinson, each
  rank draws `n` probe vectors **for its own shard** from a generator seeded
  identically across ranks and estimates the trace over its shard; all-reduce
  sums them, which is correct because `Tr(Θ)` is a sum over rows and the shards
  partition the rows. For `δL`, each rank computes its partial gradient and
  all-reduce-sums.
- All-reduce happens **once per checkpoint × eval-batch**, never in inner loops.
- `micro_batch_size=1` is the always-safe fallback. The launcher does not
  attempt OOM auto-tuning — the user picks a size that fits.

### Probe vector details

- Default `n_hutchinson=32`, Rademacher distribution. Variance scales as `1/n`.
- **Probes are drawn on the CPU and copied to the target device**, so a seed
  identifies one probe set on every device (`torch.Generator("cpu")` and
  `torch.Generator("cuda")` yield different sequences from the same seed).
  `make_probe` refuses a CUDA generator. The Rademacher draw is bit-packed
  (one random bit per entry) to keep the CPU draw affordable.
- Probe seed is **shared across ranks** within one `(checkpoint, batch)` call;
  different across calls. Each rank draws a full probe for its own shard, so
  ranks do not partition one global probe: a DDP run is reproducible, but its
  `chi_net` is a different unbiased estimate from the single-process one — and
  a noisier one. Every rank draws the *same* probe stream, so the ranks'
  estimation errors are correlated instead of independent, and the variance of
  the multi-rank `chi_net` / `chi_net_P` is larger than that of a
  single-process run with the same `n_hutchinson` (see "Known limitations").
- Probe vector shape matches the local micro-batch output `(M, S, V)`. **Padding
  positions are zeroed** so masked tokens contribute to neither `χ_net` nor
  `χ_loss`. The mask is the intersection described under "Loss handling".

### Loss handling

vatis ships with the standard HF causal-LM CE loss. **Only the closed-form CE
path is supported for `chi_loss`.** The bundle exposes a `loss_fn(logits, batch)
-> scalar` slot used by the analyzer for `delta_loss` (autograd through θ), but
`chi_loss` always uses the closed-form softmax-minus-onehot shortcut. If you pass
a non-CE `loss_fn`, `delta_loss` is correct but `chi_loss` is wrong.

The bundle's `valid_mask_fn` and the `labels=-100` (`ignore_index`) sentinel are
**intersected once**, and that one mask is used for the **counts, the gradient
weights and the probes**: the `chi_loss` sum and its `n_valid` denominator, the
per-micro-batch weighting that reassembles `∇_θ L` for `delta_loss`, and the
probe mask of the `chi_net` / `chi_net_P` estimator. **It does not reach the
loss itself**: `loss_fn` decides from the labels alone which positions it
averages over. The two ways the mask and the labels can disagree therefore
behave differently:

- **`valid_mask_fn` wider than the labels** (it accepts positions whose label
  is `ignore_index`) — handled; this is the direction that was fixed. Such a
  position contributes nothing to the loss, and the intersection now also
  removes it from the counts, the weights and the probes, so its Jacobian rows
  are excluded from `chi_net` as well.
- **`valid_mask_fn` stricter than the labels** (it rejects positions the labels
  keep) — **not handled.** The loss still averages over the rejected positions
  while the counts, weights and probes leave them out, so `delta_loss` is wrong
  and depends on `micro_batch_size`, and the `chi_net` of `per_sequence_cv`
  (whose control variate is built from that loss) is biased. A position that
  must not count has to carry `ignore_index` in the labels. See "Known
  limitations".

The shipped loaders' `valid_mask_fn` already equals `labels != ignore_index`,
so for them the intersection is a no-op.

Labels are **pre-shifted**: `labels[b, s]` is the target of the logits at
position `s` (the next token), with `ignore_index` where there is none. A batch
whose `labels` equal its `input_ids` — the HuggingFace / TRL convention, where
the model shifts internally — is not detected and is scored one position off
(see "Known limitations").

As a backstop against an inconsistent `loss_fn` (e.g. a sum-reduced CE), the
analyzer warns when a published `chi_pos` / `chi_pos_P` leaves `[−1, 1]` by more
than 0.5 — `|χ_pos| ≤ 1` is a Cauchy–Schwarz consequence of the definitions.

### Precision

- Forward in `bf16` by default (matches Pythia/OLMo training precision).
- Gradients accumulated in `fp32`. Hutchinson probes drawn in `fp32`, cast to
  forward dtype for the backward, gradient norm computed in `fp32`.
- `dtype="fp32"` is available for tiny test models.
- Scalar reductions over the flat gradient (`delta_loss`, `delta_loss_P`) and
  the `chi_pos` / `chi_pos_P` division run in `fp64`.

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
    The preconditioned observables are checked against references that cannot share a
    bug with the code: the update `torch.optim.AdamW` actually applies, an
    explicit eigendecomposition of `Θ_P`, and closed forms (`P = I`,
    `P = (JᵀJ)⁺`). The bundle loader is checked against bundles written by a
    fixture that imports none of the loader's code.
  - `tests/golden/` — marked `@pytest.mark.golden`; runs only when the `-m`
    expression names `golden`. Trains a tiny transformer for ~100 AdamW steps,
    writes a bundle, and runs the raw and preconditioned observables over it end to end.
  - `tests/integration/` — gated behind `pytest -m integration`, skipped in CI.
    Verifies HF loading on a small hub checkpoint and the DDP path — including
    the preconditioned observables — with a 2-rank CPU loopback.
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
| `P` | `Preconditioner.diag` | diagonal preconditioner, 1-D in the canonical parameter order |
| `Θ_P` | (not stored) | effective kernel `J P Jᵀ` |
| `χ_net^P` | `chi_net_P` | `‖P^{1/2}∇_θ f‖²_F = Tr(Θ_P)` |
| `δL^P` | `delta_loss_P` | `∇_θL^B · P ∇_θL^A` |
| `χ_pos^P` | `chi_pos_P` | `δL^P / (χ_loss · χ_net^P)` — spectral position on the preconditioned kernel |
| — | `delta_loss_P_capped`, `chi_net_P_capped` | the same two quantities with `P` restricted to the coordinates at its ceiling (`1/ε`) |
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
- **Under DDP every rank draws the same probe stream for its own shard.** A run
  is reproducible that way, but the ranks' estimation errors are correlated
  instead of independent, so a multi-rank `chi_net` / `chi_net_P` (and the
  `chi_pos` / `chi_pos_P` built on it) has a larger variance than a
  single-process run with the same `n_hutchinson`. The estimate stays unbiased;
  the size of the inflation depends on the model, the data and the number of
  ranks. On two small test transformers (ranks emulated in one process, 8
  sequences, `n_hutchinson = 4`, 400 seeds) the variance of `chi_net` was about
  1.5× the single-process one at 2 ranks and 2.5–2.8× at 4 ranks, with both
  estimators. `chi_loss` and `delta_loss` do not use probes and are not
  affected. Fixed in a follow-up pull request.
- **A `valid_mask_fn` stricter than the labels does not reach the loss.** The
  mask restricts the token counts, the gradient weights and the probes, but
  `loss_fn` still averages over every position the labels keep. For such a mask
  `delta_loss` is wrong and in general depends on `micro_batch_size`, and the
  `chi_net` of `per_sequence_cv` is biased; `chi_pos` inherits both. Nothing
  warns. Until the follow-up pull request that makes the labels the single
  source of truth, exclude a position by giving it the label `ignore_index`,
  not through `valid_mask_fn` alone (see "Loss handling").
- **Unshifted labels are scored one position off, without an error.** vatis
  expects pre-shifted labels (`labels[:, :-1] = input_ids[:, 1:]`, last position
  `ignore_index`). A batch in the HuggingFace / TRL convention — `labels` equal
  to `input_ids`, shifted inside the model — is taken at face value: each
  position's logits are scored against its own input token instead of the next
  one, and every observable describes that loss. Fixed in a follow-up pull
  request; until then shift the labels yourself.
- **A bundle's recorded pad convention is not applied to the update batch.**
  `mask_pad_in_loss` / `loss_ignore_index` in `metadata.json` are not read, so
  `batch_a` counts every position as a target. For a run whose training loss
  ignored pad targets, the update-batch rows describe a different loss than the
  one that was descended (see "Native checkpoint bundles"). Fixed in a follow-up
  pull request.
- **Custom (non-CE) `loss_fn` is not fully supported** — `chi_loss` always uses
  the closed-form CE shortcut (see Loss handling).
- **`OpacusEstimator` is a stub**, and has no `chi_net_P` path.
- **`delta_loss_P` is not AdamW's first-order loss change** — momentum, weight
  decay and the step's own second-moment update are not included (see "What
  `delta_loss_P` is not").
- **Optimizer state is read from native checkpoint bundles only**, and the
  loader cannot detect a same-shape parameter permutation between
  `param_names.json` and the optimizer state (see "Native checkpoint bundles").
- **AdamW's `P` puts coordinates that never received a non-zero gradient at
  `1/ε`.** Read the `*_capped` rows before interpreting a P observable on
  held-out data or on an update batch with first-seen tokens (see "Coordinates
  at the ceiling").
- **A resident `P` costs one extra parameter-sized fp32 vector** during a
  checkpoint's analysis, and building it peaks at about five in host memory
  (see "Native checkpoint bundles"). The cross-pair GPU/host offload heuristic
  does not account for either; on a borderline card pass
  `cross_grad_storage="cpu"` explicitly.
- **The `chi_pos` range warning is a weak backstop.** A sum-reduced `loss_fn`
  inflates `chi_pos` by the square of the valid-token count *per micro-batch*,
  and the warning fires only if that pushes `|chi_pos|` past 1.5. With small
  micro-batches and the small `chi_pos` values of real models it stays silent.
- **Bundle `checkpoint_id`s are `<run directory name>@step<N>`.** Two runs whose
  directories share a name produce identical ids (and identical probe seeds);
  keep run directory names distinct before concatenating result files.
