Run `adamw_s0` (adamw, 10,981,440 parameters, gradient clipping `1.0`), fp32 on NVIDIA GeForce RTX 3090 Ti. Update batch 32 × 512 tokens, held-out batch 8 × 512 tokens. Symbols as in the README.

**Each prediction as a ratio to the truth** (the AdamW rule as ratio − 1):

| step | batch | truth | AdamW rule − 1 | SGD rule | −η·δL_P | realized / truth |
|---|---|---|---|---|---|---|
| 0 | update batch | -1.019e-02 | +1.4e-05 | 0.000643 | — | 1.000 |
| 0 | held-out batch | -8.231e-03 | +1.5e-05 | 0.000728 | — | 1.000 |
| 1 | update batch | -2.626e-02 | +2.9e-07 | 0.000689 | 5.94 | 1.001 |
| 1 | held-out batch | -2.410e-02 | +4.3e-07 | 0.000698 | 2.46 | 1.001 |
| 2 | update batch | -4.412e-02 | +1.6e-07 | 0.00076 | 2.19 | 1.001 |
| 2 | held-out batch | -4.130e-02 | +1.8e-07 | 0.000792 | 1.44 | 1.001 |
| 5 | update batch | -9.454e-02 | -1.1e-07 | 0.000879 | 1.78 | 1.000 |
| 5 | held-out batch | -9.353e-02 | -1.3e-07 | 0.000877 | 1.31 | 0.999 |
| 9 | update batch | -1.663e-01 | +2.2e-07 | 0.00136 | 2.16 | 0.990 |
| 9 | held-out batch | -1.680e-01 | +2.3e-07 | 0.00133 | 1.57 | 0.984 |
| 661 | update batch | -1.387e-01 | -6.4e-08 | 0.0017 | 11.4 | 0.871 |
| 661 | held-out batch | -8.340e-03 | -8.2e-08 | 0.00824 | 26.3 | -0.415 |
| 665 | update batch | -1.464e-01 | -1.0e-07 | 0.00158 | 18.7 | 0.866 |
| 665 | held-out batch | -8.011e-03 | -1.2e-06 | 0.0046 | 20.8 | -0.573 |
| 6611 | update batch | -7.942e-02 | -9.3e-08 | 0.0013 | 11.3 | 0.955 |
| 6611 | held-out batch | -2.090e-03 | -1.1e-07 | 0.00884 | 20.8 | -0.178 |
| 6615 | update batch | -8.357e-02 | -5.3e-08 | 0.00133 | 11.5 | 0.953 |
| 6615 | held-out batch | -5.646e-03 | +1.2e-06 | 0.00352 | 12 | 0.559 |
| 33057 | update batch | -6.271e-02 | -2.4e-08 | 0.00228 | 11.9 | 0.963 |
| 33057 | held-out batch | -2.308e-03 | -2.0e-06 | 0.0198 | 9.95 | 0.458 |
| 33062 | update batch | -6.606e-02 | +8.3e-09 | 0.00201 | 11.1 | 0.959 |
| 33062 | held-out batch | -1.278e-04 | -8.9e-06 | -0.15 | 182 | -7.567 |
| 66105 | update batch | -5.925e-02 | -7.8e-08 | 0.00246 | 12 | 0.967 |
| 66105 | held-out batch | -2.001e-03 | -3.3e-06 | 0.0165 | 19.1 | 0.431 |
| 66110 | update batch | -5.412e-02 | -6.8e-08 | 0.00233 | 12.4 | 0.964 |
| 66110 | held-out batch | -5.233e-04 | +6.6e-06 | 0.0492 | 39.5 | -0.850 |

**Per step.** η and c are the rows `update_lr` and `update_clip_factor`; the cosine is between the gradients of the two batches; the last two columns are the ratio on the held-out batch divided by the ratio on the update batch (1 = one rescaling would repair the prediction on both batches):

| step | η | c | cosine | SGD rule: held-out ÷ update | −η·δL_P: held-out ÷ update |
|---|---|---|---|---|---|
| 0 | 1e-05 | 1.000 | 0.70 | 1.13 | — |
| 1 | 2.99e-05 | 1.000 | 0.68 | 1.01 | 0.414 |
| 2 | 4.98e-05 | 1.000 | 0.75 | 1.04 | 0.655 |
| 5 | 0.00011 | 1.000 | 0.78 | 0.998 | 0.735 |
| 9 | 0.000189 | 0.837 | 0.84 | 0.978 | 0.728 |
| 661 | 0.002 | 1.000 | 0.15 | 4.84 | 2.31 |
| 665 | 0.002 | 1.000 | 0.08 | 2.91 | 1.12 |
| 6611 | 0.002 | 1.000 | 0.09 | 6.81 | 1.85 |
| 6615 | 0.002 | 1.000 | 0.09 | 2.65 | 1.04 |
| 33057 | 0.002 | 1.000 | 0.17 | 8.68 | 0.837 |
| 33062 | 0.002 | 1.000 | -0.07 | -74.4 | 16.4 |
| 66105 | 0.002 | 1.000 | 0.10 | 6.7 | 1.59 |
| 66110 | 0.002 | 1.000 | 0.09 | 21.1 | 3.18 |

**The three parts of the AdamW rule**, as fractions of their sum (rows `loss_change_first_order_grad`, `_hist`, `_wd`):

| step | batch | gradient | momentum | weight decay |
|---|---|---|---|---|
| 0 | update batch | 1.000 | 0.000 | 5.9e-06 |
| 0 | held-out batch | 1.000 | 0.000 | 7.2e-06 |
| 1 | update batch | 0.610 | 0.390 | 6.4e-06 |
| 1 | held-out batch | 0.507 | 0.493 | 6.8e-06 |
| 2 | update batch | 0.460 | 0.540 | 4.8e-06 |
| 2 | held-out batch | 0.411 | 0.589 | 5.1e-06 |
| 5 | update batch | 0.293 | 0.707 | -4.9e-06 |
| 5 | held-out batch | 0.243 | 0.757 | -5.5e-06 |
| 9 | update batch | 0.233 | 0.767 | -2.7e-05 |
| 9 | held-out batch | 0.187 | 0.813 | -2.8e-05 |
| 661 | update batch | 1.058 | -0.058 | 8.4e-06 |
| 661 | held-out batch | 2.502 | -1.502 | 6.7e-04 |
| 665 | update batch | 1.036 | -0.036 | 3.1e-05 |
| 665 | held-out batch | 1.986 | -0.987 | 8.7e-04 |
| 6611 | update batch | 1.012 | -0.012 | 3.0e-05 |
| 6611 | held-out batch | 1.966 | -0.966 | -7.4e-06 |
| 6615 | update batch | 1.038 | -0.038 | -7.0e-07 |
| 6615 | held-out batch | 1.126 | -0.126 | -2.8e-04 |
| 33057 | update batch | 1.012 | -0.012 | -2.7e-05 |
| 33057 | held-out batch | 0.938 | 0.061 | 1.2e-03 |
| 33062 | update batch | 1.004 | -0.004 | -2.6e-05 |
| 33062 | held-out batch | 17.180 | -16.201 | 2.2e-02 |
| 66105 | update batch | 1.035 | -0.035 | 4.2e-05 |
| 66105 | held-out batch | 1.794 | -0.795 | 1.2e-03 |
| 66110 | update batch | 1.015 | -0.015 | -1.9e-05 |
| 66110 | held-out batch | 3.759 | -2.759 | 7.8e-05 |
