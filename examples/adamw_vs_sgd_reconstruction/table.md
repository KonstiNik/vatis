Bundle `adamw_s0` (adamw, rows tagged `adamw_update`), 10,981,440 parameters, clipping stated as `1.0`, fp32 on NVIDIA GeForce RTX 3090 Ti. Update batch [32, 512] tokens, held-out batch `eval_batch_2.npy` [8, 512] tokens. Steps [0, 1, 2, 5, 9, 661, 665, 6611, 6615, 33057, 33062, 66105, 66110] of the 50 that have their successor stored. The truth is `loss_change_first_order_weights`; the optimizer's rule is `loss_change_first_order`.

**Each prediction as a ratio to the truth** (the optimizer's rule as ratio − 1):

| step | batch | truth | optimizer's rule − 1 | SGD rule | −lr·δL_P | realized |
|---|---|---|---|---|---|---|
| 0 | update | -0.0102 | +1e-05 | 0.000643 | — | 1 |
| 0 | held-out | -0.00823 | +2e-05 | 0.000728 | — | 1 |
| 1 | update | -0.0263 | +3e-07 | 0.000689 | 5.94 | 1 |
| 1 | held-out | -0.0241 | +4e-07 | 0.000698 | 2.46 | 1 |
| 2 | update | -0.0441 | +2e-07 | 0.00076 | 2.19 | 1 |
| 2 | held-out | -0.0413 | +2e-07 | 0.000792 | 1.44 | 1 |
| 5 | update | -0.0945 | -1e-07 | 0.000879 | 1.78 | 1 |
| 5 | held-out | -0.0935 | -1e-07 | 0.000877 | 1.31 | 0.999 |
| 9 | update | -0.166 | +2e-07 | 0.00136 | 2.16 | 0.99 |
| 9 | held-out | -0.168 | +2e-07 | 0.00133 | 1.57 | 0.984 |
| 661 | update | -0.139 | -6e-08 | 0.0017 | 11.4 | 0.871 |
| 661 | held-out | -0.00834 | -8e-08 | 0.00824 | 26.3 | -0.415 |
| 665 | update | -0.146 | -1e-07 | 0.00158 | 18.7 | 0.866 |
| 665 | held-out | -0.00801 | -1e-06 | 0.0046 | 20.8 | -0.573 |
| 6611 | update | -0.0794 | -9e-08 | 0.0013 | 11.3 | 0.955 |
| 6611 | held-out | -0.00209 | -1e-07 | 0.00884 | 20.8 | -0.178 |
| 6615 | update | -0.0836 | -5e-08 | 0.00133 | 11.5 | 0.953 |
| 6615 | held-out | -0.00565 | +1e-06 | 0.00352 | 12 | 0.559 |
| 33057 | update | -0.0627 | -2e-08 | 0.00228 | 11.9 | 0.963 |
| 33057 | held-out | -0.00231 | -2e-06 | 0.0198 | 9.95 | 0.458 |
| 33062 | update | -0.0661 | +8e-09 | 0.00201 | 11.1 | 0.959 |
| 33062 | held-out | -0.000128 | -9e-06 | -0.15 | 182 | -7.57 |
| 66105 | update | -0.0592 | -8e-08 | 0.00246 | 12 | 0.967 |
| 66105 | held-out | -0.002 | -3e-06 | 0.0165 | 19.1 | 0.431 |
| 66110 | update | -0.0541 | -7e-08 | 0.00233 | 12.4 | 0.964 |
| 66110 | held-out | -0.000523 | +7e-06 | 0.0492 | 39.5 | -0.85 |

**Per step**: the learning rate, the clip factor, and the ratio on the held-out batch divided by the ratio on the update batch (1 = off by the same factor on both, i.e. repairable by rescaling the step):

| step | lr | clip | SGD rule | −lr·δL_P |
|---|---|---|---|---|
| 0 | 1e-05 | 1.000 | 1.13 | — |
| 1 | 2.99e-05 | 1.000 | 1.01 | 0.414 |
| 2 | 4.98e-05 | 1.000 | 1.04 | 0.655 |
| 5 | 0.00011 | 1.000 | 0.998 | 0.735 |
| 9 | 0.000189 | 0.837 | 0.978 | 0.728 |
| 661 | 0.002 | 1.000 | 4.84 | 2.31 |
| 665 | 0.002 | 1.000 | 2.91 | 1.12 |
| 6611 | 0.002 | 1.000 | 6.81 | 1.85 |
| 6615 | 0.002 | 1.000 | 2.65 | 1.04 |
| 33057 | 0.002 | 1.000 | 8.68 | 0.837 |
| 33062 | 0.002 | 1.000 | -74.4 | 16.4 |
| 66105 | 0.002 | 1.000 | 6.7 | 1.59 |
| 66110 | 0.002 | 1.000 | 21.1 | 3.18 |

**The parts of the optimizer's rule** as fractions of their sum (rows `loss_change_first_order_grad`, `_hist`, `_wd`):

| step | batch | gradient | momentum | weight decay |
|---|---|---|---|---|
| 0 | update | 1.00 | 0.00 | 6e-06 |
| 0 | held-out | 1.00 | 0.00 | 7e-06 |
| 1 | update | 0.61 | 0.39 | 6e-06 |
| 1 | held-out | 0.51 | 0.49 | 7e-06 |
| 2 | update | 0.46 | 0.54 | 5e-06 |
| 2 | held-out | 0.41 | 0.59 | 5e-06 |
| 5 | update | 0.29 | 0.71 | -5e-06 |
| 5 | held-out | 0.24 | 0.76 | -6e-06 |
| 9 | update | 0.23 | 0.77 | -3e-05 |
| 9 | held-out | 0.19 | 0.81 | -3e-05 |
| 661 | update | 1.06 | -0.06 | 8e-06 |
| 661 | held-out | 2.50 | -1.50 | 7e-04 |
| 665 | update | 1.04 | -0.04 | 3e-05 |
| 665 | held-out | 1.99 | -0.99 | 9e-04 |
| 6611 | update | 1.01 | -0.01 | 3e-05 |
| 6611 | held-out | 1.97 | -0.97 | -7e-06 |
| 6615 | update | 1.04 | -0.04 | -7e-07 |
| 6615 | held-out | 1.13 | -0.13 | -3e-04 |
| 33057 | update | 1.01 | -0.01 | -3e-05 |
| 33057 | held-out | 0.94 | 0.06 | 1e-03 |
| 33062 | update | 1.00 | 0.00 | -3e-05 |
| 33062 | held-out | 17.18 | -16.20 | 2e-02 |
| 66105 | update | 1.03 | -0.03 | 4e-05 |
| 66105 | held-out | 1.79 | -0.79 | 1e-03 |
| 66110 | update | 1.01 | -0.01 | -2e-05 |
| 66110 | held-out | 3.76 | -2.76 | 8e-05 |
