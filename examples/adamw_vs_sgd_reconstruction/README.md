# Which update rule reproduces the step an AdamW run took?

At a checkpoint of an AdamW run the optimizer moved the weights from θ_t to θ_{t+1}. To first order this changed the loss on a batch B by

    truth = ∇L_B · (θ_{t+1} − θ_t)

(row `loss_change_first_order_weights`, from the two stored weight files). We compare three predictions of this number that use only what was known *before* the step. Notation: A is the batch the step was computed on; ∇L_A and ∇L_B are the gradients of the mean loss on A and on B at θ_t; η is the step's learning rate (row `update_lr`); c ≤ 1 is the factor by which gradient clipping scaled ∇L_A (row `update_clip_factor`); P_t is AdamW's diagonal preconditioner, built from the second moments the optimizer had saved before the step.

| prediction | what it is | from the rows |
|---|---|---|
| AdamW rule | the AdamW update that vatis rebuilds from the saved optimizer state: a gradient part, a momentum part and a weight-decay part | `loss_change_first_order` = `_grad` + `_hist` + `_wd` |
| SGD rule | a plain gradient step with the same η and c — no momentum, no decay, no preconditioner: −η·c·∇L_B·∇L_A | `−update_lr · update_clip_factor · delta_loss` |
| −η·δL_P | P_t applied to the current gradient alone, without momentum or decay: −η·∇L_B·P_t∇L_A | `−update_lr · delta_loss_P` |

`delta_loss` is the row holding ∇L_B·∇L_A, and `delta_loss_P` the row holding ∇L_B·P_t∇L_A. Each prediction is evaluated on the step's own update batch (B = A) and on one held-out batch.

The size of the SGD rule means nothing by itself: η was tuned for preconditioned steps, so a plain gradient step with that η is far too small. The question that can be asked is whether one rescaling would repair it on both batches. −η·δL_P has no value at step 0: the optimizer state is still empty there, so there is no P_t.

## Answer

One run, one seed, 11M parameters, 13 steps, 2 batches.

1. **The AdamW rule reproduces the truth** to within 2·10⁻⁵ relative at every step. This checks that vatis and the bundle are consistent. It is expected, not a finding.
2. **The SGD rule is 400–1600 times too small** on the update batch (20–1400 times on the held-out batch). Would one rescaling fix it? In steps 0–9 it nearly would: its ratio to the truth is the same on the two batches to within 13%. But in those steps the gradients of the two batches are themselves strongly aligned (cosine 0.68–0.84), so this is weak evidence. From step 661 on (cosine −0.07 to 0.17) no rescaling fixes it: the two ratios differ by a factor of 2.7–21, and at step 33062 the sign is wrong on the held-out batch.
3. **−η·δL_P is 1.3–6 times the truth in steps 1–9 and 10–40 times from step 661 on** (182 times at step 33062 on the held-out batch, where the truth is small). On the update batch the late step is almost entirely its gradient part (table below), and that part is 0.06–0.09 of −η·δL_P: the current gradient enters an AdamW step with weight 1 − β₁ = 0.1, and the step uses its own preconditioner, which already contains that gradient and is smaller than P_t where the gradient is large. On the held-out batch the momentum part has the opposite sign and cancels 44–94% of the gradient part at six of the eight late steps, which makes the factor larger there; at the other two it is again about 10.
4. **First order is not the whole step.** In steps 0–9 the realized loss change L_B(θ_{t+1}) − L_B(θ_t) (row `loss_change`) is within 1.6% of the truth. From step 661 on it differs by 3–13% on the update batch, and on the held-out batch it has the opposite sign at 5 of 8 steps.

The parts of the AdamW rule as fractions of their sum, at six of the steps (the weight-decay part is below 0.2% of the sum everywhere except at step 33062 on the held-out batch, where it is 2%):

| step | update batch: gradient | momentum | held-out batch: gradient | momentum |
|---|---|---|---|---|
| 1 | 0.61 | 0.39 | 0.51 | 0.49 |
| 9 | 0.23 | 0.77 | 0.19 | 0.81 |
| 661 | 1.06 | −0.06 | 2.50 | −1.50 |
| 6615 | 1.04 | −0.04 | 1.13 | −0.13 |
| 33062 | 1.00 | −0.00 | 17.2 | −16.2 |
| 66110 | 1.02 | −0.02 | 3.76 | −2.76 |

![Ratio of each prediction to the truth, per step, on the update batch and on the held-out batch](reconstruction.png)

*Ratio of each prediction to the truth, log scale; the horizontal line at 1 means exact. Hollow markers have the wrong sign and are drawn at |ratio|. Steps are not to scale. Steps 0–9 are in the learning-rate warm-up, and the clip engaged only at step 9 (c = 0.837).*

## What it does not show

- Nothing about SGD-*trained* runs: the SGD rule here is a counterfactual step from a point an AdamW run reached.
- Nothing beyond this run: one seed, one model size, one learning-rate schedule, 13 of 66 116 steps, one held-out batch of 8 × 512 tokens. The factors above are not constants.
- Nothing about which optimizer trains better.
- The clip engaged at one step only, so how the SGD rule would look without the factor c rests on a single measurement.

## Data and command

An 11M-parameter Llama trained with AdamW: η rises from 1e-5 to 2e-3 over the first 100 steps and stays there, β = (0.9, 0.95), weight decay 0.01, batches of 32 × 512 tokens, pad-masked loss, gradient norm clipped at 1.0, fp32 weight files. The truth needs θ_t and θ_{t+1}, so every step used must have its successor stored; 50 of this bundle's 85 checkpoints do. The bundle records the clip value but not the algorithm, so the command states it. In the script, `cross_pairs=[(update_batch, held_out)]` is what makes `delta_loss` for the pair (A, held-out batch) available, i.e. the SGD rule on the held-out batch. The run takes 13 s on one RTX 3090 Ti.

```bash
python examples/adamw_vs_sgd_reconstruction.py --bundle <run>/bundle \
    --held-out <run>/eval_batches/eval_batch_2.npy --clip-grad-norm 1.0 --device cuda \
    --steps 0 1 2 5 9 661 665 6611 6615 33057 33062 66105 66110
python examples/adamw_vs_sgd_reconstruction_evaluate.py
```

Every number per step and batch, including all three parts at all steps: [table.md](table.md).
