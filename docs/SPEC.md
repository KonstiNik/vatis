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

of which `delta_loss_P` is the first dot product with `P_t` in place of
`P_{t+1}` and without its prefactor. Treat `chi_pos_P` as a statement about the
*geometry* the optimizer imposes. `−η · delta_loss_P` is the first-order
prediction for the *hypothetical* update `−η P_t ∇L^A`. **Do not use it as the
loss change of the step that was taken** — that is what the `loss_change_first_order*`
rows measure, term by term (next section).

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
`Δθ ≈ −η · sign(g_0)`. The `loss_change_first_order*` rows do cover it.

#### The response to the update that was taken (`loss_change_first_order*` rows)

`Analyzer(update_response=...)` measures what `delta_loss_P` is not: the
first-order loss change under the update the optimizer applied at the
checkpoint's step, for the update batch and for every other eval batch `B`,

```
loss_change_first_order(A, B) = ∇L^B · Δθ          Δθ = θ_{t+1} − θ_t,   A = the batch step t consumed
```

`Δθ` is rebuilt from what the checkpoint saved **before** the step — the
weights `θ_t`, the optimizer state, the hyper-parameters of the saved
`param_groups` — and from `g = ∇L^A`, which the analyzer computes anyway. It is
a sum of three parts, and each part is a row. For `torch.optim.AdamW`, with `t`
the number of updates already applied:

```
v_{t+1} = β2 v_t + (1−β2) g²                    the second moment after it absorbed g
P_{t+1} = 1 / (√(v_{t+1} / (1−β2^{t+1})) + ε)   the P the step used (not the pre-step P_t)

loss_change_first_order_grad = − η (1−β1)/(1−β1^{t+1}) · ∇L^B · P_{t+1} g      the step's own gradient
loss_change_first_order_hist = − η  β1   /(1−β1^{t+1}) · ∇L^B · P_{t+1} m_t    the gradient history
loss_change_first_order_wd   = − η λ · ∇L^B · θ_t                              decoupled weight decay
loss_change_first_order      = the sum of the three
```

| observable | value | measured |
| --- | --- | --- |
| `loss_change_first_order_grad` | part of `∇L^B · Δθ` carried by the step's own gradient | every checkpoint |
| `loss_change_first_order_hist` | part carried by the first moment / momentum buffer saved before the step | every checkpoint |
| `loss_change_first_order_wd` | part carried by weight decay | every checkpoint |
| `loss_change_first_order` | their sum: the first-order loss change of the step, from the saved state | every checkpoint |
| `loss_change_first_order_weights` | `∇L^B · (θ_{t+1} − θ_t)` from the two stored weight files | step `t+1` stored, fp32 weight files |
| `loss_change` | `L_B(θ_{t+1}) − L_B(θ_t)`, the realized change | same |
| `loss_change_residual` | `loss_change − loss_change_first_order_weights`: what first order leaves over | same |

```python
source = load_local_bundle(run_dir, "my_project.models:build_model")
analyzer = Analyzer(
    eval_batches={"train": source.batch_a_spec(), "val": val_batch},
    update_response=source.update_response(),          # the update each step applied
    preconditioner=source.preconditioner(),            # optional, independent
    sink="results.parquet",
)
analyzer.run(source.model_specs())
```

- **Units and sign.** These are loss changes: they contain the learning rate
  and are negative when the step lowers the loss on `B`. `delta_loss` and
  `delta_loss_P` are gradient products without the learning rate and with the
  opposite sign convention; the comparable quantity is `−η · delta_loss_P`,
  which equals `loss_change_first_order_grad` only where `P_t = P_{t+1}`, the
  prefactor `(1−β1)/(1−β1^{t+1})` is one and the gradient was not clipped. It
  is a different quantity, not an estimate of this one: where the two are set
  side by side below, the ratio says how far apart the hypothetical update and
  the step taken are, not how wrong either is.
- **Where the rows are.** On the pair `(batch_a, batch_b)` = (update batch,
  `B`) for every eval batch `B`, the update batch itself included; the response
  is not symmetric in the two. The update batch must be one of the eval batches
  (`source.batch_a_spec()`): the analyzer recognises it by comparing the eval
  batches' tensors with the checkpoint's stored batch, whatever its name, and
  refuses when none matches — before the checkpoint's first forward pass, not
  after its `χ_net` estimates. (To make that possible the eval batches of a
  checkpoint are fetched once, up front, when an update rule is configured.)
  The four formula rows carry the update
  rule's descriptor in the `preconditioner` column (`"adamw_update"`,
  `"adam_update"`, `"sgd_update"`); the three realized-change rows involve no
  optimizer formula and carry `"none"`.
- **They exist at step 0.** An empty optimizer state is a well-defined starting
  point for an update (`m_0 = v_0 = 0`, so the first step is
  `−η g/(|g| + ε)`), unlike for a pre-step `P`. The first update is in the
  rows even though it has no P rows.
- **Step numbering.** The saved step counter `t` counts the updates already
  applied; the step being described is update number `t + 1`, and both bias
  corrections take that exponent. A counter that differs from the checkpoint
  directory's step is refused as for the P source (`on_step_mismatch="warn"`
  uses the saved state as it is); so is an empty state at a step other than 0.
- **The learning rate is the one saved in the optimizer state**, i.e. the
  scheduled value the step applied (a scheduler that is stepped after every
  update has already written the next step's rate into the param group when
  the pre-step checkpoint is saved). When every group has the same rate it is
  compared with the checkpoint's `meta.json` `lr`; a difference beyond `1e-6`
  relative is refused, because every row is proportional to it.
- **Hyper-parameters are taken per parameter**, from the group each parameter
  belongs to. Weight decay, learning rate, `betas` and `eps` that differ
  between groups (no decay on biases and norms is common) are therefore exact.
  This is where the rows differ from the P source, which refuses per-group
  learning rates: one diagonal `P` without the rate cannot represent them, an
  update vector can. With several groups, `param_names.json` has to list the
  parameters in the optimizer's own order — group after group — because that is
  how a torch `state_dict` numbers them.
- **Other update rules.** `torch.optim.Adam` (weight decay added to the
  gradient): `g` is replaced by `g + λθ_t` in `v_{t+1}`, and the decay part is
  `−η (1−β1)/(1−β1^{t+1}) · ∇L^B · P_{t+1} λθ_t`. `torch.optim.SGD` with
  momentum `μ`, dampening `τ`, weight decay `λ` (coupled) and buffer `b_t`:
  `Δθ = −η (c_g (g + λθ_t) + c_b b_t)` with `(c_g, c_b) = (1−τ, μ)`, `(1, 0)` on
  the first step or without momentum, and `(1 + μ c_g, μ c_b)` under Nesterov;
  the three parts are the `g`, `b_t` and `λθ_t` terms. Whether an Adam-family
  group decays decoupled is read from the group's `decoupled_weight_decay`
  entry; a group without it and with non-zero decay needs the optimizer named
  in `metadata.json`, and is refused otherwise. Refused outright: every other
  optimizer (RAdam, NAdam, Adamax, RMSprop, ...), `amsgrad=True`,
  `maximize=True`, and a parameter that has no state while others have.

**What the saved state cannot show.** The rows describe the step that was
taken only if `g`, recomputed from `batch_a.npy` at the stored weights, is the
gradient the optimizer saw. vatis cannot verify that from one checkpoint:

- *Gradient clipping* rescales the gradient before the optimizer sees it and
  leaves no trace in the state. It is therefore never assumed, in either
  direction. `source.update_response()` reads `metadata.json`:
  `gradient_clip_val` null or `0` means "not clipped"; a positive value together
  with `gradient_clip_algorithm: "norm"` means the global gradient norm was
  clipped to it, exactly as `torch.nn.utils.clip_grad_norm_` over all
  parameters does (`g ← g · min(1, c / (‖g‖ + 1e-6))`), and that is applied. A
  bundle that does not record `gradient_clip_val` at all, or records a value
  without the algorithm, is **refused** until you state it:
  `update_response(clip_grad_norm=None)` or `update_response(clip_grad_norm=c)`.
  A statement that contradicts the recorded value is refused too, and so is
  clipping by value. The clipping factor of a row can be recovered from the
  update batch's own `delta_loss` row, which is `‖g‖²`.
- *Rounded weights.* A bf16 weight file holds a rounded `θ_t`, so `g` is the
  gradient at a slightly different point (see "Native checkpoint bundles" for
  how different). The same holds for unrounded files opened with
  `load_local_bundle(dtype="bf16")` or `"fp16"`, which rounds them in memory.
  The formula rows are still reported; the realized rows are not (below).
- *A parameter that receives no gradient at the step* — it is not in the graph
  of the update batch (an unused branch, an unused parameter). torch treats
  the two ways this can look differently: a parameter whose `.grad` is `None`
  is skipped altogether, with no momentum and no weight decay (the default
  `zero_grad()` leaves `None`); one whose `.grad` holds zeros
  (`zero_grad(set_to_none=False)` after an earlier gradient) gets both. The
  rebuilt update sees a zero gradient and applies both. Which of the two the
  run did is not in a bundle, but the analyzer does know which parameters got
  no gradient on the update batch, so it settles the cases it can: if the
  saved state would not move such a parameter anyway (no weight decay on it
  and no momentum history), both readings agree and the rows are exact;
  otherwise the rows are **refused**, naming the parameter. For Adam and AdamW
  a parameter that was skipped at an earlier step also shows in the state —
  its step counter lags, or it has no state at all — and that was already
  refused.
- *Parameters outside the canonical list* (a source opened with
  `allow_unlisted_params` whose model has parameters `param_names.json` does
  not name). They are in no gradient vector and in no saved optimizer state.
  If the run trained them, their part of the step is missing from every
  `loss_change_first_order*` row while `loss_change` still contains its effect; if
  they merely had gradients when the norm was clipped, the clipping factor is
  wrong. A bundle does not say, so `update_response()` **refuses** such a
  bundle unless the caller confirms with `unlisted_params_frozen=True` that
  those parameters received no gradient and no update during the run.
- *Gradient accumulation.* `batch_a.npy` must be the whole effective batch and
  the run must have descended its token-weighted mean loss (see the pad
  convention under "Native checkpoint bundles"); then accumulation changes
  nothing here.
- *Everything else the loop did to the gradient or the weights*: dropout or
  other train-mode randomness (the analyzer evaluates in `eval()` mode), mixed
  precision (the gradient is recomputed in the loader's dtype, and the formula
  rows take `θ_t` at that precision too: with `dtype="bf16"` both are those of
  the rounded model, whatever the files hold), weight
  averaging, a manual edit of the weights between steps. None of it is
  detectable from a bundle. The contract a bundle has to meet is: *reloading
  step `t` and applying one optimizer step on `batch_a.npy` reproduces the
  run's `θ_{t+1}`*.

Where the bundle holds the next step in fp32, that contract is checkable, and
the rows check it: `loss_change_first_order` (rebuilt from the saved state) and
`loss_change_first_order_weights` (the stored weight difference, no formula) are two
routes to the same number. **If they disagree by more than rounding, one of the
assumptions above does not hold for that bundle.** "Rounding" includes the
optimizer's own: the rule is evaluated in fp64, the run stepped in fp32, and a
step far below the resolution of the weights is partly lost in the run itself.
On the real bundles below the two rows agree to `1e-8 … 2e-5` relative, except
at the very first SGD step with a warm-up rate of `1e-5`, where the applied
fp32 step is 1.6% smaller than the exact one (and `loss_change` follows the
applied one).

**The realized change** needs `θ_{t+1}`. Where the bundle also holds step
`t + 1` — whether or not it is among the selected `steps` — both weight
files are stored as fp32 or wider and the source loads them as fp32
(`load_local_bundle(dtype="fp32")`, the default), the three realized rows are
measured:
`loss_change_first_order_weights`, `loss_change` and `loss_change_residual`. The
residual is taken against the *stored* weight difference, not against the
rebuilt update, so it contains the loss surface's higher orders and nothing
about the optimizer formula. With bf16 (or fp16) weight files, or with a source
that loads the weights as bf16 or fp16, these rows are
**skipped with a warning**: the difference of two rounded sets of weights is
dominated by the rounding early in training (measured below), and a residual
computed from it would be an artefact. A checkpoint with no stored successor
has no such rows, silently. `update_response(realized=False)` switches them off.
Both losses are evaluated forward-only, as token-weighted means over
micro-batches and DDP ranks, with the logits upcast to fp64 before the loss, so
the cross entropy itself adds no fp32 round-off to the difference of two nearly
equal losses (`test_loss_change_is_the_fp64_loss_of_the_fp32_logits`). The
forward pass stays in the model's own precision, and that sets a noise floor:
an fp32 forward perturbs each loss by an amount that does not cancel in the
difference. `loss_change`, and with it `loss_change_residual`, is therefore
only resolved down to that floor — measured at about `1e-8` to `1e-7` on the
real model below — and a residual of that size is noise, whatever its sign.
The alternative, evaluating both losses on an fp64 copy of the model, was not
taken: the analyzer does not know the model (casting one wholesale also hits
complex buffers and fp32 tables, which is why the loader casts parameters
only), and an fp64 copy of both models doubles the memory of the realized
rows, which a model near the limit of its device does not have.

**Cost.** No extra backward. The gradients of all eval batches are kept for
the checkpoint (one parameter-sized fp32 vector each, placed as
`cross_grad_storage` says), the saved optimizer state stays on the CPU for the
checkpoint (two parameter-sized vectors for AdamW), and the reductions run in
fp64 chunks. The realized rows hold a second model and cost two forward passes
per eval batch — only if they are asked for: `θ_{t+1}` is loaded only when one
of the three is among `observables`, and the forward passes run only for
`loss_change` / `loss_change_residual`. Under DDP every rank rebuilds the same
update from the all-reduced gradients; only the two loss evaluations (and the
record of which parameters received no gradient) are all-reduced.

**Precision.** The rule itself is exact: on an fp64 model the rebuilt update
equals the one `torch.optim` applied element by element to `1e-10` of its
largest entry, for AdamW, Adam and SGD (`tests/unit/test_update_response_vs_torch.py`).
Through a bundle everything is fp32 — weights, optimizer state, and the
analyzer's gradient vector — and the rows agree with the run to about `1e-7`
relative on the test bundles (`tests/unit/test_update_response_bundle.py`,
asserted at `1e-5`).

**Measured on real bundles** (an 11M-parameter Llama, CPU, fp32 analysis,
`micro_batch_size=8`; `B` = the step's own update batch of 32 × 512 tokens
unless stated). These are single runs of single bundles: read them as what the
rows look like, not as constants.

*A bundle with bf16 weight files* (AdamW, `β = (0.9, 0.95)`, `ε = 1e-8`,
`λ = 0.01`, warm-up, no clipping, unmasked loss). There is no usable
`θ_{t+1}`, so the reference is the update the optimizer applied, rebuilt from
the state saved *after* the step (`m_{t+1}`, `v_{t+1}`), which involves no
recomputed gradient:

| step | `loss_change_first_order` | `_grad` | `_hist` | `_wd` | applied update (post-step state) | two bf16 weight files | `−η · delta_loss_P` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | −0.07932 | −0.07932 | 0 | −9.6e−8 | −0.07932 (differs by 5e−6) | −0.07922 (0.13%) | no `P` at step 0 |
| 1 | −1.09819 | −0.57660 | −0.52158 | −8.2e−7 | −1.09818 (4e−6) | −1.09736 (0.07%) | −1.27180 (16% larger) |
| 2 | −1.91379 | −0.75004 | −1.16376 | +1.3e−5 | −1.91377 (1e−5) | −1.91177 (0.11%) | −2.69596 (41% larger) |
| 3 | −1.39350 | −0.42945 | −0.96408 | +3.0e−5 | −1.39346 (3e−5) | −1.39230 (0.09%) | −1.84469 (32% larger) |

The gradient recomputed at the bf16-rounded weights differed from the one the
optimizer saw (reconstructed from consecutive first moments) by 0.24–0.25%
relative, cosine 0.999997; the three terms nevertheless reproduce the applied
first-order change to a few parts in `1e5`. Element by element, the difference
of the two bf16 weight files was off from the applied update by 5.7, 1.4, 0.97
and 0.88 times its norm at steps 0–3 — which is why no realized row is
computed from such files. On a held-out batch of 8 × 512 tokens the same four
steps gave the same agreement (4e−6 … 5e−5), with `−η · delta_loss_P` 2%, 37%
and 25% larger at steps 1–3. Later in the same run `−η · delta_loss_P` on the
update batch was 38%, 92% and 189% larger at steps 4, 6 and 8, and about 65,
15, 11.5 and 9.5 times `loss_change_first_order` at steps 20, 49, 122 and 299 (how
much of that the coordinates at the `1/ε` ceiling carry was not examined).

*Bundles with fp32 weight files* (the same model; pad-masked loss; gradient
norm clipped at 1.0, stated with `clip_grad_norm=1.0` because the bundles record
the value but not the algorithm; AdamW with peak rate 2e-3, SGD with momentum
0.9 and peak rate 1.0, both with a warm-up from 1e-5):

| run | step | `η` | clip factor | `loss_change_first_order` | `_grad` | `_hist` | `loss_change_first_order_weights` (rel. diff.) | `loss_change` | `loss_change_residual` | `−η · delta_loss_P` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW | 0 | 1.0e−5 | 1 | −0.010192 | −0.010192 | 0 | −0.010192 (1.4e−5) | −0.010194 | −2.1e−6 | no `P` at step 0 |
| AdamW | 1 | 3.0e−5 | 1 | −0.026263 | −0.016011 | −0.010252 | −0.026263 (3e−7) | −0.026279 | −1.7e−5 | −0.156016 (5.9×) |
| AdamW | 9 | 1.9e−4 | 0.837 | −0.166286 | −0.038766 | −0.127525 | −0.166286 (2e−7) | −0.164681 | +1.6e−3 | −0.358982 (2.2×) |
| AdamW | 661 | 2.0e−3 | 1 | −0.138667 | −0.146740 | +0.008074 | −0.138667 (2e−7) | −0.120782 | +1.8e−2 | −1.583337 (11.4×) |
| AdamW | 66105 | 2.0e−3 | 1 | −0.059246 | −0.061295 | +0.002052 | −0.059246 (4e−7) | −0.057270 | +2.0e−3 | −0.713140 (12.0×) |
| SGD | 0 | 1.0e−5 | 1 | −6.556e−6 | −6.556e−6 | 0 | −6.454e−6 (1.6e−2) | −6.454e−6 | below the noise floor (reported: +2.9e−10) | −6.556e−6 (equal) |
| SGD | 1 | 1.0e−2 | 1 | −0.010984 | −0.006045 | −0.004940 | −0.010984 (5e−7) | −0.010991 | −6.4e−6 | −0.006045 (0.55×) |
| SGD | 661 | 1.0 | 1 | −0.134733 | −0.153860 | +0.019127 | −0.134733 (2e−7) | −0.110690 | +2.4e−2 | −0.153860 (1.14×) |

Here the recomputed gradient matched the one the optimizer saw to `4e-7 … 3e-5`
relative, also at the step where the clip engaged, and also on a run that
accumulated gradients over 4 micro-batches with token weighting (`4e-7`; rows
within `2e-7` of the stored weight difference at step 1). Late in the AdamW run
`−η · delta_loss_P` is about `1/(1−β1) = 10` times the gradient part and more
than that against the whole step: the current gradient enters the update with
weight `1−β1`, and on its own batch the history is nearly orthogonal to it. For
SGD `delta_loss_P` is `delta_loss`, so `−η · delta_loss_P` is the gradient part
— exactly so only for a step that was not clipped, has no dampening and is not
Nesterov, which holds for the three SGD rows above (the gradient part carries
the factors `clip`, `1−τ` and, under Nesterov, `1 + μ(1−τ)`) — and the rest is
momentum. On the held-out batch the first-order change
late in training is small (−0.008 at AdamW step 661) against a residual of
+0.012: there the realized change (+0.003) has the opposite sign.
**The noise floor of `loss_change`**, measured on this model: computed with the
whole model in fp64, `loss_change` differed from the fp32-forward value by at
most `1.3e-8` on the update batch (32 × 512 tokens) and by at most `8.3e-8` on
the held-out batch (8 × 512 tokens), over steps 0, 9 and 661 of the AdamW run
and steps 0, 1 and 661 of the SGD run (twelve comparisons; the same with 2 and
with 8 threads). Residuals of `1e-6` and larger in the table are therefore not
round-off. The residual of the first SGD step is: the analyzer reported
`+2.9e-10` on the update batch where the fp64 model gives `+2.1e-10`, and
`−9.3e-9` on the held-out batch where the fp64 model gives `+1.2e-9` — numbers
of that size are not reproducible and should be read as "below `1e-7`".

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

#### `micro_batch_size` is a noise setting, not only a memory setting

Both estimators draw `n_hutchinson` probes **per micro-batch**, and one probe
spans the whole micro-batch (shape `(M, S, V)`). Its error therefore contains
not only each sequence's own off-diagonal Gram entries but also every
cross-sequence block `J_b J_{b'}ᵀ` of the micro-batch. `χ_net` stays unbiased,
but at fixed `n_hutchinson` its variance grows about linearly with the
micro-batch size, `≈ 1 + c·(M − 1)` times the variance at `micro_batch_size=1`,
with a model-dependent slope `c` (for plain Hutchinson with Rademacher probes
`c = mean_{b≠b'} ‖J_b J_{b'}ᵀ‖²_F / mean_b ‖offdiag J_b J_bᵀ‖²_F`, the mean over
the pairs that share a micro-batch). `χ_pos` and the `_P` observables built on
`χ_net` inherit it.

Measured in a single process (`B = 8`, `n_hutchinson = 4`, Rademacher, variance
relative to `micro_batch_size=1`; 400 seeds, 200 for the hub model; the same for
`hutchinson` and `per_sequence_cv` to within 0.07):

| model | micro-batch 2 | micro-batch 4 | micro-batch 8 |
|---|---|---|---|
| toy transformer trained 3000 AdamW steps | ×1.7 | ×3.4 | ×6.5 |
| `SimpleStories-1.25M` (pretrained hub model), real text | ×1.1 | ×1.6 | ×1.9 |
| toy transformer, untrained | ×2.3 | ×5.2 | ×10.6 |

These are three CPU test models of at most 1.25M parameters; the slope differs
several-fold between them (about 0.8, 0.1–0.2 and 1.4 in the order of the
table), and nothing larger has been measured — do not read a number for a
production model off this table.

What follows for use:

- `micro_batch_size=1` (the default) is the least noisy setting as well as the
  one that needs the least memory.
- The variance is proportional to `1/n_hutchinson` (the estimate is a mean of
  independent probes per micro-batch). Raising `n_hutchinson` by the same factor
  as `micro_batch_size` keeps the number of probe backwards unchanged and, in
  the measurements above, at least restored the noise level on the trained toy
  and on the hub model (×6.5 and ×1.9 at micro-batch 8, both below 8) but not
  quite on the untrained toy (×10.6).
- A multi-rank run shards the batch before micro-batching, so at
  `micro_batch_size > 1` its noise differs from a single process's — see
  "Probe vector details".

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
  weight decay in `P` (they are in the `loss_change_first_order*` rows instead, see
  "The response to the update that was taken"); update rules other than
  AdamW, Adam and SGD there. **Optimizer states in third-party formats**: only the native bundle
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
│   ├── update_response.py # OptimizerUpdate: the update a step applied, in three parts
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
    update_response=None,                       # None | fn(ckpt_id) -> OptimizerUpdate | None
    chi_net_method=None,                        # None = auto: per_seq_cv if B≤32 else hutchinson
    n_hutchinson=32,                            # interpretation depends on method
    hutchinson_distribution="rademacher",       # or "gaussian"
    micro_batch_size=1,                         # least memory AND least noise; raising it raises χ_net variance
    num_gpus=4,
    distributed="ddp",
    sink="results.parquet",                     # str → ParquetSink; or list of sinks
    dtype="bf16",                               # forward dtype; grads in fp32
    ignore_index=-100,                          # the label value loss_fn ignores (see Loss handling)
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

### Batch format and the label convention

A causal-LM batch is a dict with `input_ids` (`(B, S)`), optionally
`attention_mask`, and `labels`. **The labels are the single source of truth for
what enters the loss**: a position counts iff its label is not `-100`
(`ignore_index`). Padding, prompt tokens of a completion-only objective and any
other exclusion are expressed by writing `-100` into the labels, as HuggingFace
`Trainer` / TRL do.

**vatis labels are pre-shifted**: `labels[:, i]` is the target of
`logits[:, i]`, i.e. `input_ids[:, i + 1]`, and the last position is `-100`.
`causal_lm_loss` and the closed-form `chi_loss` compare `logits[:, i]` with
`labels[:, i]` and shift nothing. This is *not* the HuggingFace / TRL collation,
where `labels` is aligned with `input_ids` (a copy of it with `-100` at ignored
positions) and the model shifts inside its loss. Convert such a batch before
handing it to vatis:

```python
from vatis.data import shift_hf_labels

batch["labels"] = shift_hf_labels(batch["labels"])   # out[:, i] = labels[:, i + 1]; out[:, -1] = -100
```

`shift_hf_labels` applies the shift `transformers`' causal-LM loss applies (pad
one `ignore_index` on the right, drop the first label), so the converted batch
counts the same targets with the same shift as the HuggingFace model does on
the original. The loss *values* agree to float rounding with fp32 logits (they
are identical on the test model). HuggingFace upcasts the logits to fp32 before
its loss and `causal_lm_loss` does not, so with bf16 logits the two values
differ at bf16 precision (measured 0.4% and 0.7% on two small bf16 GPT-2s).

Labels that are **already shifted** need no conversion: what `transformers`
calls `shift_labels` (some trainers hand it to the model instead of `labels`)
is exactly vatis's form. Converting it again would shift twice.

The shipped `causal_lm_loss` **refuses** labels that are provably unshifted
instead of scoring every target one position off. A sequence is refused iff,
over its non-ignored labels, every label equals the input token at the *same*
position and at least one differs from the input token at the *next* position
(the last position is not compared — its target is not in `input_ids`). A
correctly pre-shifted sequence can never meet that condition. The batch is
refused as soon as one of its sequences is.

**What the guard cannot catch.** A sequence that satisfies both readings is
ambiguous and is scored as given:

- a counted label on the **last position only** — an unshifted single-token
  completion with nothing after it. With right padding this is the share of
  sequences that fill their row (4% of the rows in a simulation with lengths
  uniform in 8…32 padded to 32, and no batch of 8 escaped, because the other
  rows gave it away); with left padding, or a batch of one unpadded sequence,
  it is every sequence;
- a counted label that equals the token after it — a run of one repeated token,
  and in particular a completion consisting only of the end-of-text token when
  the pad token is that same token (every such sequence, in the same
  simulation).

So the guard catches the ordinary mistake, not every conceivable one. It also
does not apply to a custom `loss_fn`, or to a batch without `input_ids`.

A dict without `labels` gets next-token targets derived from `input_ids` (and
`attention_mask`, if present). `(input, target)` tuples and lists are accepted
for classifier-style models; a bare tensor has no targets and is refused.

### Native checkpoint bundles

Besides the hub, vatis reads **checkpoint bundles** written by a training run:
a run directory in a fixed schema that carries the optimizer state and the
update batch next to the weights.

```
<run_dir>/
  metadata.json                # model config, optimizer type + hparams, lr schedule,
                               # seed, B_train, seq_len, dataset id, git hashes;
                               # optional: mask_pad_in_loss, loss_ignore_index,
                               # gradient_clip_val, gradient_clip_algorithm
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
  position ignored), i.e. the pairs `ids[:, :-1] → ids[:, 1:]`.
- **The update batch is scored with the pad convention the run recorded.** Two
  optional run-level fields of `metadata.json` say whether the training loss
  ignored a target id:

  | field | type | meaning |
  | --- | --- | --- |
  | `mask_pad_in_loss` | bool | the training cross entropy was given an `ignore_index` |
  | `loss_ignore_index` | token id (`0 ≤ id < vocab_size`), or `null` when unmasked | that index (the tokenizer's pad id) |

  With an index recorded, the loader takes the training loss to be the mean
  over the targets of the stored batch that differ from it
  (`F.cross_entropy(logits, targets, ignore_index=<index>)`), and writes `-100`
  into the `batch_a` labels wherever the next-token target equals that index.
  Every update-batch row (`n_valid`, `chi_loss`, `delta_loss`, `chi_net`, and
  the `*_P` rows) is then computed over exactly the targets that loss counts. (A
  run that accumulates gradients over micro-batches has to weight them by their
  counted-target numbers for that mean over the concatenated batch to be the
  objective it descended.) Only the labels change: `input_ids` keep their pad
  tokens and no `valid_mask_fn` is involved. With both fields absent, or
  `mask_pad_in_loss` false/absent and
  `loss_ignore_index` null/absent, every target counts, as for a bundle written
  before the fields existed. The resolved value is
  `source.loss_ignore_index` (`None` when unmasked).

  The index is matched **by value**, as the training cross entropy matches it.
  When the pad id is also the end-of-text id, masking the pad targets therefore
  drops the real end-of-text targets as well — which is what such a training
  run did, and so what its update-batch rows have to describe.

  **Contradictory fields are refused** when the bundle is opened
  (`BundleSchemaError`): `mask_pad_in_loss: true` without an index, an index
  while `mask_pad_in_loss` is false or absent, or a value of the wrong JSON type.
  Either reading of such metadata could put a number on a loss the run did not
  descend, and the two fields alone do not say which. An index no target can
  equal is refused the same way — a negative one, or one not below the
  `vocab_size` of the metadata's model config when that is stated — since it
  would be accepted and then mask nothing. The refusal happens at
  `load_local_bundle`, so it also stops uses that never touch the update batch
  (held-out batches only, reading the optimizer state). The fields describe the
  *update* batch only; an eval batch you pass yourself carries its own labels
  (to score a held-out batch under the run's objective, write `-100` where its
  target equals `source.loss_ignore_index`, as `examples/two_rulers_llama.py`
  does).

  **Repairing contradictory pad fields.** The refusal is about two entries of
  `metadata.json`; the weights, optimizer states and batches are untouched and
  remain usable. The remedy is to correct the two fields to what the run
  actually did: remove both (or `false` / `null`) if its loss counted every
  target, `true` plus the pad id if it ignored pad targets. When it is not known
  which entry is wrong, the bundle itself can decide, because each checkpoint's
  `meta.json` records `train_loss` — the loss the run computed on its update
  batch at `θ_t`:

  1. Make the bundle loadable under either reading (for instance, remove both
     fields for the duration of the check). If the bundle is read-only, or you
     would rather not edit it, do this on a stand-in: a new directory holding
     your own `metadata.json` next to symbolic links to the original
     `param_names.json` and `checkpoints/` (or a full copy). Give that directory
     the **same name** as the original run directory: checkpoint ids are
     `<run directory name>@step<N>` and the probe seeds derive from them, so
     another name gives different `χ_net` draws. The same stand-in is how a
     corrected `metadata.json` is used permanently when the original cannot be
     changed.
  2. For a checkpoint, recompute the loss of `batch_a.npy` at that checkpoint's
     weights twice — every target counted, and the candidate pad id ignored:

     ```python
     ids = ckpt.batch_a_tokens                      # raw ids; not ckpt.batch_a, whose labels follow the fields
     logits = ckpt.model_bundle.model(ids[:, :-1])
     flat, targets = logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
     all_counted = F.cross_entropy(flat, targets)
     pad_ignored = F.cross_entropy(flat, targets, ignore_index=pad_id)
     ```
  3. The one that matches `ckpt.meta.train_loss` is the convention the run used;
     write the fields accordingly.

  What "matches" can and cannot mean:

  - **Closer by a wide margin, not equal.** A bf16 weight file holds a rounded
    `θ_t`, so the matching value is only approximately the recorded one. On
    bundles of an 11M-parameter transformer the matching candidate was within
    `2e-5` to `3e-3` of `train_loss` with bf16 weight files and within `7e-7`
    with fp32 weight files (also under gradient accumulation), while the other
    candidate was off by `0.04` to `9` on update batches with 40–50% pad targets.
    (`tests/unit/test_bundle_pad_convention.py` shows the same on a toy bundle
    with bf16 weights: `≤ 2e-4` against `≥ 0.16` once it has trained.)
  - **The margin shrinks with the share of pad targets.** On a run whose update
    batches held about 1% pad targets the other candidate was off by only
    `2e-3` to `8e-3` in its first steps — decisive with fp32 weight files, not
    with bf16 ones.
  - **Use trained checkpoints, and more than one.** The two candidates are the
    same number when the batch contains no target equal to the candidate id, and
    close to each other while little has been learned (on the toy bundle the
    wrong candidate is off by about `9e-5` at step 0, no more than the bf16
    rounding). All checkpoints
    examined must give the same verdict; if neither candidate is close, the run
    used some other convention and the fields should not be guessed.
  - **`train_loss` must be the loss of the stored batch at the stored weights.**
    Under gradient accumulation that is the token-weighted loss of the whole
    window (sum of the per-token losses over the number of counted targets),
    which is what makes it the loss of the concatenated batch `batch_a.npy`
    holds. A writer that records something else there — a mean of micro-batch
    means under masking, a smoothed or post-step value — defeats the recipe.
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
- **`source.update_response()` rebuilds the step each checkpoint took** from
  the same saved state, for the `loss_change_first_order*` rows ("The response to the
  update that was taken"). It reads two more optional run-level fields of
  `metadata.json`:

  | field | type | meaning |
  | --- | --- | --- |
  | `gradient_clip_val` | number or `null` | the gradient was clipped to this threshold before the optimizer step; `null` or `0`: not clipped |
  | `gradient_clip_algorithm` | `"norm"` or `"value"` | how: the global L2 norm over all parameters (`torch.nn.utils.clip_grad_norm_`), or each entry by value |

  Unlike the pad fields, these have no default: a bundle that does not record
  `gradient_clip_val`, or records a value without the algorithm, is refused by
  `update_response()` until the caller states the clipping. Only `"norm"` is
  implemented. Nothing else in the loader reads the two fields.
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

The update-response rows extend it the same way and add no column: seven more
`observable` values on the pair `(batch_a, batch_b)` = (update batch, evaluated
batch), with `n_hutchinson`, `hutchinson_seed` and `wallclock_s` null. The four
rows rebuilt from the optimizer state carry the update rule's descriptor in the
`preconditioner` column (`"adamw_update"`, `"adam_update"`, `"sgd_update"`);
`loss_change_first_order_weights`, `loss_change` and `loss_change_residual` carry
`"none"`. A pair that has only these rows (the update batch against an eval
batch that is not in `cross_pairs`) pivots to a line with empty `chi_*` cells.

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
- Each eval batch is sharded across ranks along the batch dim, in contiguous
  slices. If `B % N != 0`, the first `B % N` ranks get one sequence more than the
  others (`shard_batch`) — the all-reduce handles this because we accumulate
  token-weighted sums, not means.
- Per micro-batch: forward → `χ_loss` accumulator. For `χ_net` Hutchinson, each
  rank draws `n` probe vectors **for its own shard** from **its own stream**
  (see "Probe vector details") and estimates the trace over its shard;
  all-reduce sums them, which is correct because `Tr(Θ)` is a sum over rows and
  the shards partition the rows. For `δL`, each rank computes its partial
  gradient and all-reduce-sums.
- All-reduce happens **once per checkpoint × eval-batch**, never in inner loops.
- `micro_batch_size=1` is the always-safe fallback, and the least noisy
  setting (see "`micro_batch_size` is a noise setting, not only a memory
  setting"). The launcher does not attempt OOM auto-tuning — the user picks a
  size that fits.

### Probe vector details

- Default `n_hutchinson=32`, Rademacher distribution. Variance scales as `1/n`.
- **Probes are drawn on the CPU and copied to the target device**, so a seed
  identifies one probe set on every device (`torch.Generator("cpu")` and
  `torch.Generator("cuda")` yield different sequences from the same seed).
  `make_probe` refuses a CUDA generator. The Rademacher draw is bit-packed
  (one random bit per entry) to keep the CPU draw affordable.
- Probe streams are **per rank**, and different across `(checkpoint, batch)`
  calls. The per-call seed — SHA-256 of `seed`, the checkpoint id and the batch
  name, reduced modulo `2³¹ − 1`; this is the `hutchinson_seed` column, on every
  rank — seeds rank 0, and therefore every single-process run, directly. Rank
  `r > 0` seeds its generator with `sha256("{hutchinson_seed}|rank={r}")`,
  reduced the same way. So the probes are determined by `seed`, the checkpoint
  id, the batch name, the rank and the world size (which fixes the shards), and
  a DDP run is reproducible. Each rank draws full probes for its own shard —
  ranks do not partition one global probe — so the multi-rank `χ_net` (and with
  it `χ_pos`, `χ_net^P`, `χ_pos^P`) is a different unbiased estimate from the
  single-process one and **depends on the number of ranks** as well as on the
  seed.
- **The variance depends on how the batch is cut into micro-batches** (see
  "`micro_batch_size` is a noise setting, not only a memory setting"), so it is
  not the same for a multi-rank and a single-process run in general. Sharding
  cuts the batch *before* micro-batching, so a multi-rank run has the
  single-process variance only when every micro-batch holds one sequence
  (measured: 0.8–1.2×, consistent with 1); with larger micro-batches its
  micro-batches are no larger than a shard and it is typically *less* noisy
  (measured at 4 ranks, micro-batch 4, `B = 8`: 0.47× on the untrained toy
  transformer, 0.49× on the trained one, 0.62× on the hub model).
- Ranks must **not** share one stream. A Rademacher probe `v` applied to sample
  `a` errs by `Σ_{i≠j} v_i v_j M_a[i,j]` (`M_a = J_a J_aᵀ`), so the same `v` on
  two samples gives errors with covariance `2⟨offdiag M_a, offdiag M_b⟩_F`
  (Gaussian probes: `2⟨M_a, M_b⟩_F`, which also counts the diagonals and is
  never negative), and the Gram matrices of different inputs are not
  orthogonal. With a shared seed (the behaviour through 0.2.1) the per-rank
  errors are positively correlated, and the variance of the all-reduced `χ_net`
  is between 1× and world-size× that of **independent rank streams on the same
  shards** — that is the reference, not a single process; the two coincide only
  at `micro_batch_size = 1`. Where in that range depends on the model and on the
  probe distribution. It has been measured only on CPU test models of at most
  1.25M parameters (nothing larger): 1.3× at 2 ranks and 2.1–2.6× at 4 ranks
  with Rademacher probes, 1.4–1.9× and 2.6–3.3× with Gaussian probes.
- Probe vector shape matches the local micro-batch output `(M, S, V)`. **Padding
  positions are zeroed** so masked tokens contribute to neither `χ_net` nor
  `χ_loss`. The mask is the intersection described under "Loss handling".

### Loss handling

vatis ships with the standard causal-LM CE loss (a mean over the non-ignored
targets; labels pre-shifted, see below). **Only the closed-form CE
path is supported for `chi_loss`.** The bundle exposes a `loss_fn(logits, batch)
-> scalar` slot used by the analyzer for `delta_loss` (autograd through θ), but
`chi_loss` always uses the closed-form softmax-minus-onehot shortcut. If you pass
a non-CE `loss_fn`, `delta_loss` is correct but `chi_loss` is wrong.

**The labels are the single source of truth for what enters the loss** (see
"Batch format and the label convention"). The bundle's `valid_mask_fn` and the
`labels=-100` (`ignore_index`) sentinel are **intersected once**. That one mask
is used for the counts, the gradient weights and the probes — the `chi_loss`
sum and its `n_valid` denominator, the per-micro-batch weighting that
reassembles `∇_θ L` for `delta_loss`, and the probe mask of the `chi_net` /
`chi_net_P` estimator — and it reaches the loss itself through the labels (the
"stricter" case below). The
shipped loaders' `valid_mask_fn` already equals `labels != ignore_index`, so for
them none of the following changes anything. A custom `valid_mask_fn` can
disagree with the labels in two directions, and both are handled:

- **Wider** — it accepts a position whose label is `ignore_index`. The
  intersection removes the position: it contributes nothing to the loss, so its
  Jacobian rows are excluded from `χ_net` as well.
- **Stricter** — it rejects a position whose label is valid. The mask alone
  cannot restrict `loss_fn`, which only sees the batch. So wherever a loss is
  evaluated (the `delta_loss` backward and the loss direction of
  `per_sequence_cv`), the analyzer evaluates `loss_fn` on a batch whose labels
  carry the mask: `ignore_index` is written at every rejected position (into
  `labels`, else `targets`, of a dict batch; into element 1 of a tuple/list
  batch; a dict with only `input_ids` gains its masked next-token `labels`).
  This is the HuggingFace `Trainer` / TRL procedure, and a stricter mask gives
  exactly what the same restriction written into the labels gives, at every
  `micro_batch_size`. The caller's batch is never modified.

**The analyzer's `ignore_index` must be the value the `loss_fn` ignores.** It
is `-100` by default and is an argument of both `Analyzer(...)` and
`analyze(...)`. The analyzer uses it to decide which positions count and writes
it into the labels wherever a mask rejects a position; it never guesses a loss's
own ignore value. A dataset and loss that mark ignored positions with another
value (say `-1`) therefore need `ignore_index=-1`. Left at the default, such a
setup is refused: when the labels a mask rejects carry a value that is neither a
class index nor `ignore_index`, the analyzer raises a `ValueError` that names
the value to pass — before the loss is called, because on CUDA a loss handed a
target it does not ignore dies in a device-side assert that cannot be caught.
If a loss raises on rewritten labels for a reason that cannot be seen in advance
(its ignore value differs but no label gives that away), the error is re-raised
as a `RuntimeError` with this explanation and the original exception chained; on
CUDA that residual case can still surface as the device-side assert itself.

A `loss_fn` therefore has to read its targets from the batch and honour
`ignore_index`. When labels were rewritten, the analyzer checks that the
gradient of the loss with respect to the logits vanishes at the rejected
positions and raises a `ValueError` otherwise — a loss that ignores the
rewritten labels would be silently unmasked. It raises as well when no gradient
reaches the logits at all (a `loss_fn` that ignores its `logits` argument and
runs its own forward): the check then has nothing to inspect, and a loss that
cannot be verified is not accepted as verified. Called directly (outside the
analyzer), `per_sequence_cv` raises the same way if `loss_fn` counts positions
its `valid_mask_fn` rejects: its control variate is built from the loss
direction and is unbiased only when that direction is confined to the masked
positions. `hutchinson` evaluates no loss.

Labels are **pre-shifted**: `labels[b, s]` is the target of the logits at
position `s` (the next token), with `ignore_index` where there is none. A batch
whose `labels` equal its `input_ids` — the HuggingFace / TRL convention, where
the model shifts internally — is refused by the shipped `causal_lm_loss` when it
is provably unshifted, and `vatis.data.shift_hf_labels` converts it (see "Batch
format and the label convention" for the conversion and for what the guard
cannot catch).

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
  The labels are the single source of truth for what enters the loss; the
  analyzer derives one mask and one loss from them (see "Loss handling").

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
    fixture that imports none of the loader's code. The update-response rule is
    checked against real `torch.optim` steps in fp64 — the whole update element
    by element, and each of its three parts against an ablation of the real
    optimizer — and through bundles written by real training runs.
  - `tests/golden/` — marked `@pytest.mark.golden`; runs only when the `-m`
    expression names `golden`. Trains a tiny transformer for ~100 AdamW steps,
    writes a bundle, and runs the raw and preconditioned observables over it end to end.
    A second gate does the same for the `loss_change_first_order*` rows on a bundle
    with consecutive fp32 checkpoints, against the weight files themselves.
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
| `∇L^B · Δθ` | `loss_change_first_order`, `loss_change_first_order_grad` / `_hist` / `_wd` | first-order loss change on `B` under the step that was taken, rebuilt from the pre-step optimizer state, and its three parts |
| `∇L^B · (θ_{t+1} − θ_t)` | `loss_change_first_order_weights` | the same from two stored weight files |
| — | `loss_change`, `loss_change_residual` | `L_B(θ_{t+1}) − L_B(θ_t)`, and that minus `loss_change_first_order_weights`: the realized change, and what the first-order term from the stored weight difference leaves over |
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
- **The unshifted-label guard is not exhaustive.** The shipped `causal_lm_loss`
  refuses HuggingFace / TRL-style labels only when a sequence proves them
  unshifted; a single labelled token at the very end of a sequence, or a label
  equal to the token after it, looks the same in both conventions and is scored
  as given. A custom `loss_fn` is not guarded at all (see "Batch format and the
  label convention").
- **Custom (non-CE) `loss_fn` is not fully supported** — `chi_loss` always uses
  the closed-form CE shortcut (see Loss handling).
- **`OpacusEstimator` is a stub**, and has no `chi_net_P` path.
- **`delta_loss_P` is not AdamW's first-order loss change** — momentum, weight
  decay and the step's own second-moment update are not included (see "What
  `delta_loss_P` is not"). The `loss_change_first_order*` rows measure that change
  ("The response to the update that was taken").
- **The `loss_change_first_order*` rows rest on a recomputed gradient.** They describe
  the step that was taken only if one optimizer step on the stored update batch
  at the stored weights is what the run did. Gradient clipping has to be
  stated; dropout, mixed precision, a gradient accumulation that is not
  token-weighted, or bf16-rounded weights make the recomputed gradient differ
  from the one the optimizer saw, and only the comparison with
  `loss_change_first_order_weights` — available for consecutive fp32 checkpoints —
  shows it. The realized-change rows need fp32 weight files.
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
