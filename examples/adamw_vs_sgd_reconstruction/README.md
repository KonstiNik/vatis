# Which update rule reproduces the step an AdamW run took?

At a checkpoint of an AdamW run the optimizer moved the weights from θ_t to
θ_{t+1}. To first order this changed the loss on a batch B by
`truth = ∇L_B · (θ_{t+1} − θ_t)`, read from the two stored weight files (row
`loss_change_first_order_weights`). This example compares three predictions of
that number that use only what was known *before* the step. That the AdamW rule
rebuilt by vatis matches the stored weights is a **consistency check of vatis
and the bundle, not a finding**; the comparison with the other two rules is the
point. It is a small check, not a study: one run, one seed, an 11M-parameter
model, 13 steps, 2 batches.

| prediction | what it is | from the rows |
|---|---|---|
| AdamW rule ("optimizer's rule" in the figure and in `table.md`) | the AdamW update rebuilt from the saved optimizer state: a gradient part, a momentum part and a weight-decay part | `loss_change_first_order` = `_grad` + `_hist` + `_wd` |
| SGD rule | a plain gradient step with the same learning rate η and clip factor c — no momentum, no decay, no preconditioner: −η·c·∇L_B·∇L_A | `−update_lr · update_clip_factor · delta_loss` |
| −η·δL_P | AdamW's pre-step preconditioner P_t applied to the current gradient alone, without momentum or decay: −η·∇L_B·P_t∇L_A | `−update_lr · delta_loss_P` |

A is the batch the step was computed on. Each prediction is evaluated on the
step's own update batch (B = A) and on one held-out batch. The realized change
`L_B(θ_{t+1}) − L_B(θ_t)` (row `loss_change`) is shown too, so that what first
order itself misses is visible.

## What you need

- **A checkpoint bundle that is not part of this repository**: an AdamW run
  with fp32 weight files, in which every step used has its successor stored
  (the truth needs θ_t and θ_{t+1}), plus a directory of held-out
  `eval_batch_*.npy` files (the last one is used).
- The gradient clipping of the run. This bundle records the clip value but not
  the algorithm, so the command states it: `--clip-grad-norm 1.0`.
- A GPU is optional: 11 s on an RTX 3090 Ti for the 13 steps committed here,
  115 s on the CPU.

## How to run

```bash
python examples/adamw_vs_sgd_reconstruction/run.py --bundle <run>/bundle \
    --eval-batches <run>/eval_batches --clip-grad-norm 1.0 --device cuda \
    --steps 0 1 2 5 9 661 665 6611 6615 33057 33062 66105 66110
python examples/adamw_vs_sgd_reconstruction/evaluate.py   # table.md + reconstruction.png
```

In `run.py`, `cross_pairs=[(update_batch, held_out)]` is what makes
`delta_loss` for the pair (A, held-out batch) available, i.e. the SGD rule on
the held-out batch.

## What it shows

![Ratio of each prediction to the truth, per step, on the update batch and on the held-out batch](reconstruction.png)

*Ratio of each prediction to the truth, log scale; the horizontal line at 1
means exact. Hollow markers have the wrong sign and are drawn at |ratio|. Steps
are not to scale. Steps 0–9 are in the learning-rate warm-up, and the clip
engaged only at step 9 (c = 0.837).*

1. **The AdamW rule reproduces the truth** to within 2·10⁻⁵ relative at every
   step: within 1.6·10⁻⁵ of a ratio of one at step 0 (where the step is so small
   that fp32 rounding of the stored weights shows) and within 10⁻⁵ at the other
   twelve; the largest absolute difference is 1.4·10⁻⁷. This is the consistency
   check.
2. **The SGD rule is 400–1600 times too small** on the update batch (20–1400
   times on the held-out batch). That alone says little: η was set for
   preconditioned steps. Would one rescaling fix it on both batches? In steps
   0–9 it nearly would: its ratio to the truth is the same on the two batches
   to within 13%. But in those steps the gradients of the two batches are
   themselves strongly aligned (cosine 0.68–0.84), so this is weak evidence.
   From step 661 on (cosine −0.07 to 0.17) no rescaling fixes it: the two
   ratios differ by a factor of 2.7–21, and at step 33062 the sign is wrong on
   the held-out batch.
3. **−η·δL_P is 1.3–6 times the truth in steps 1–9 and 10–40 times from step
   661 on** (182 times at step 33062 on the held-out batch, where the truth is
   small). On the update batch the late step is almost entirely its gradient
   part (table below), and that part is 0.06–0.09 of −η·δL_P: the current
   gradient enters an AdamW step with weight 1 − β₁ = 0.1, and the step uses its
   own preconditioner, which already contains that gradient and is smaller than
   P_t where the gradient is large. On the held-out batch the momentum part has
   the opposite sign and cancels 44–94% of the gradient part at six of the eight
   late steps, which makes the factor larger there; at the other two it is
   again about 10. By the scale-free measure of point 2 (ratio on the
   held-out batch divided by ratio on the update batch) −η·δL_P does worse
   than the SGD rule in the warm-up (0.41–0.74 at steps 1–9, against 0.98–1.13
   at steps 0–9; −η·δL_P has no value at step 0, where the optimizer state is
   still empty, so there is no P_t) and better
   from step 661 on (0.84–3.2, 16 at step 33062, against 2.7–21).
4. **First order is not the whole step.** In steps 0–9 the realized loss change
   is within 1.6% of the truth (0.01–0.11% at steps 0–5, 1–1.6% at step 9).
   From step 661 on it differs by 3–13% on the update batch (13% at steps 661
   and 665, 3–5% later), and on the held-out batch it has the opposite sign at
   5 of 8 steps (661, 665, 6611, 33062, 66110): there, even an exact first-order prediction does not predict the loss
   change of one step.

The parts of the AdamW rule as fractions of their sum, at six of the steps (the
weight-decay part is below 0.2% of the sum everywhere except at step 33062 on
the held-out batch, where it is 2%):

| step | update batch: gradient | momentum | held-out batch: gradient | momentum |
|---|---|---|---|---|
| 1 | 0.61 | 0.39 | 0.51 | 0.49 |
| 9 | 0.23 | 0.77 | 0.19 | 0.81 |
| 661 | 1.06 | −0.06 | 2.50 | −1.50 |
| 6615 | 1.04 | −0.04 | 1.13 | −0.13 |
| 33062 | 1.00 | −0.00 | 17.2 | −16.2 |
| 66110 | 1.02 | −0.02 | 3.76 | −2.76 |

Every number per step and batch: [table.md](table.md), which also has the SGD
rule without the clip factor and the ratio on the held-out batch divided by the
ratio on the update batch.

## What it does not show

- Nothing about SGD-*trained* runs: the SGD rule here is a counterfactual step
  from a point an AdamW run reached.
- Nothing beyond this run: one seed, one model size, one learning-rate
  schedule, 13 of 66 116 steps, one held-out batch of 8 × 512 tokens. The
  factors above are not constants.
- Nothing about which optimizer trains better.
- The clip engaged at one step only, so how the SGD rule would look without the
  factor c rests on a single measurement.

## About the committed outputs

- Files: `results.parquet`, `run_summary.json`, `table.md`, `reconstruction.png`.
- The run: bundle `adamw_s0`, an 11M-parameter Llama (10,981,440 parameters)
  trained with AdamW — η rises from 1e-5 to 2e-3 over the first 100 steps and
  stays there, β = (0.9, 0.95), weight decay 0.01, batches of 32 × 512 tokens,
  pad-masked loss, gradient norm clipped at 1.0, fp32 weight files. 50 of its
  85 checkpoints have their successor stored (steps 0–9, 661–670, 6611–6620,
  33057–33066, 66105–66114); the 13 used are the first steps of the warm-up and
  two from each later group.
- Settings (from `run_summary.json` and `run.py`): fp32, `micro_batch_size=8`,
  `n_hutchinson=1` (`chi_net` is not used here), seed 0, `--device cuda` on an
  NVIDIA GeForce RTX 3090 Ti, torch 2.10.0+cu128; 11 s wall time.
- Reproducibility: two other runs of the same command on the shared card (13 s
  and 34 s) agree with the committed `loss_change*` rows to 3.1·10⁻¹⁰ in
  absolute terms. The same 13 steps on the CPU (115 s) give the same table:
  every `loss_change*` row agrees with the GPU value to within 1.1·10⁻⁷ in
  absolute terms (2·10⁻⁵ to 3·10⁻⁴ relative on the first-order rows, more only
  where a value is itself below 10⁻⁶), and the learning-rate and clip-factor
  rows to 3·10⁻⁷.
- Committed on 17 November 2026 in `2d6b92f` (#14).
