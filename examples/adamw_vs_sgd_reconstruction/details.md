# Reconstructing the AdamW step — details

The numbers and reasoning behind the four points of [README.md](README.md).
Every number per step and batch is in [table.md](table.md).

## The three predictions and their rows

At a checkpoint of an AdamW run the optimizer moved the weights from θ_t to
θ_{t+1}. To first order this changed the loss on a batch B by
`truth = ∇L_B · (θ_{t+1} − θ_t)`, read from the two stored weight files (row
`loss_change_first_order_weights`). A is the batch the step was computed on; η
is the step's learning rate (row `update_lr`); c ≤ 1 is the factor by which
gradient clipping scaled ∇L_A (row `update_clip_factor`); P_t is AdamW's
diagonal preconditioner, built from the second moments the optimizer had saved
before the step.

| prediction | what it is | from the rows |
|---|---|---|
| AdamW rule ("optimizer's rule" in the figure and in `table.md`) | the AdamW update rebuilt from the saved optimizer state: a gradient part, a momentum part and a weight-decay part | `loss_change_first_order` = `_grad` + `_hist` + `_wd` |
| SGD rule | a plain gradient step with the same learning rate η and clip factor c — no momentum, no decay, no preconditioner: −η·c·∇L_B·∇L_A | `−update_lr · update_clip_factor · delta_loss` |
| −η·δL_P | AdamW's pre-step preconditioner P_t applied to the current gradient alone, without momentum or decay: −η·∇L_B·P_t∇L_A | `−update_lr · delta_loss_P` |

Each prediction is evaluated on the step's own update batch (B = A) and on one
held-out batch. The realized change `L_B(θ_{t+1}) − L_B(θ_t)` (row
`loss_change`) is shown too, so that what first order itself misses is visible.

## 1. The AdamW rule

The ratio of the AdamW rule to the truth is within 1.6·10⁻⁵ of one at step 0,
where the step is so small that fp32 rounding of the stored weights shows, and
within 10⁻⁵ of one at the other twelve steps. The largest absolute difference
is 1.4·10⁻⁷.

## 2. The SGD rule

- **Size.** On the update batch it is 400–1600 times too small (ratio 0.0006 to
  0.0025); on the held-out batch 20–1400 times (ratio 0.0007 to 0.05). At step
  33062 on the held-out batch it has the wrong sign. At no step measured does
  it reconstruct the step better than the AdamW rule. That it is far too small
  is expected: the learning rate of this run was set for preconditioned steps.
- **Would one rescaling fix it?** The scale-free measure is the ratio on the
  held-out batch divided by the ratio on the update batch (1 = off by the same
  factor on both, so a rescaled learning rate would fix it). In steps 0–9 this
  quotient is 0.98 to 1.13: a plain-gradient step with a rescaled learning rate
  would have reconstructed both batches to within about 13%. But in those steps
  the gradients of the two batches are themselves strongly aligned (cosine
  0.68–0.84), so this is weak evidence. From step 661 on (cosine −0.07 to 0.17)
  the quotient is 2.7 to 21, and negative at step 33062, so no rescaling
  repairs it.
- **Without the clip factor.** "The SGD rule, unclipped" differs from the SGD
  rule only at step 9, the one step where the clip engaged (c = 0.837); both
  are in `table.md`.

## 3. −η·δL_P

- **Size.** It is 1.3–6 times the truth in steps 1–9 and 10–40 times from step
  661 on (182 times at step 33062 on the held-out batch, where the truth is
  small). It has no value at step 0: the optimizer state is still empty there,
  so there is no P_t. It is a different quantity, not an estimate of the step
  taken.
- **Why about ten times, late in training.** On the update batch the late step
  is almost entirely its gradient part (the split table in the README), and
  that part is 0.06–0.09 of −η·δL_P. The current gradient enters an AdamW step
  with weight 1 − β₁ = 0.1. In addition, the step is taken with the
  preconditioner after the second-moment update by that same gradient, not
  with P_t. By the way AdamW works (this is reasoning, not something read from
  the result file), that preconditioner is smaller than P_t where the gradient
  is large.
- **The held-out batch.** There the momentum part has the opposite sign and
  cancels 44–94% of the gradient part at six of the eight late steps, which
  makes the factor larger; at the other two it is again about 10.
- **Compared with the SGD rule, scale-free.** By the quotient of point 2,
  −η·δL_P does worse than the SGD rule in the warm-up (0.41–0.74 at steps 1–9,
  against 0.98–1.13 at steps 0–9) and better from step 661 on (0.84–3.2, 16 at
  step 33062, against 2.7–21).

## 4. First order against the realized change

During the warm-up the realized loss change `L_B(θ_{t+1}) − L_B(θ_t)` differs
from the truth by 0.01% to 0.11% (steps 0–5) and by 1% to 1.6% (step 9). At
the full learning rate it differs by 13% on the update batch at steps 661 and
665 and by 3% to 5% later. On the held-out batch the realized change then has
the opposite sign to the first-order change at five of the eight later steps
(661, 665, 6611, 33062, 66110): there, even an exact first-order prediction
does not predict the loss change of one step.

## The run

An 11M-parameter Llama (10,981,440 parameters), bundle `adamw_s0`, trained with
AdamW: η rises from 1e-5 to 2e-3 over the first 100 steps and stays there,
β = (0.9, 0.95), weight decay 0.01, batches of 32 × 512 tokens, pad-masked
loss, gradient norm clipped at 1.0, fp32 weight files. The truth needs θ_t and
θ_{t+1}, so every step used must have its successor stored: 50 of the bundle's
85 checkpoints do (steps 0–9, 661–670, 6611–6620, 33057–33066, 66105–66114).
The 13 used are 0, 1, 2, 5, 9 (the first steps of the warm-up) and two from
each later group. The held-out batch is `eval_batch_2.npy`, 8 × 512 tokens,
scored with the run's pad convention.

In `run.py`, `cross_pairs=[(update_batch, held_out)]` is what makes
`delta_loss` for the pair (A, held-out batch) available, i.e. the SGD rule on
the held-out batch.

## Reproducibility

The committed run took 11 s on one RTX 3090 Ti. Two other runs of the same
command on the shared card (13 s and 34 s) agree with the committed
`loss_change*` rows to 3.1·10⁻¹⁰ in absolute terms. The same 13 steps on the
CPU (115 s) give the same table: every `loss_change*` row agrees with the GPU
value to within 1.1·10⁻⁷ in absolute terms (2·10⁻⁵ to 3·10⁻⁴ relative on the
first-order rows, more only where a value is itself below 10⁻⁶), and the
learning-rate and clip-factor rows to 3·10⁻⁷.
