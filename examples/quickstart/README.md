# What does a use of vatis look like, from start to finish?

The smallest complete use: [`run.py`](run.py) loads `EleutherAI/pythia-14m` at
its last training checkpoint (`step143000`), builds two short real-text batches
(a passage of prose and a few lines of Python, 2 × 32 tokens each), calls
`vatis.analyze` once, and prints the result rows as a table. About 40 lines of
code, nothing else to look up. The next step is
[`pythia_sweep/`](../pythia_sweep/README.md): the same measurement over 13
checkpoints, with figures.

## What you need

- vatis installed (`uv pip install .` in a checkout). No matplotlib.
- Internet access the first time: the checkpoint is downloaded from the
  HuggingFace hub (set `HF_HOME` to choose where it goes).
- A CPU is enough: 7 s on a 12th-gen Intel Core i7 with four threads, once the
  checkpoint is cached.

## How to run

```bash
python examples/quickstart/run.py
```

## What it shows

The printed table, from the run described under
[About the committed outputs](#about-the-committed-outputs):

```
observable           batch_a batch_b       value
chi_loss             prose   prose       0.01462
chi_loss_normalized  prose   prose        0.9065
chi_net              prose   prose     2.595e+12
chi_net_normalized   prose   prose     4.185e+10
delta_loss           prose   prose          6431
chi_pos              prose   prose     1.695e-07
chi_loss             code    code        0.01288
chi_loss_normalized  code    code         0.7985
chi_net              code    code      3.531e+12
chi_net_normalized   code    code      5.696e+10
delta_loss           code    code           9517
chi_pos              code    code      2.092e-07
chi_loss             prose   code        0.01372
chi_loss_normalized  prose   code         0.8508
chi_net              prose   code      3.027e+12
chi_net_normalized   prose   code      4.882e+10
delta_loss           prose   code          39.25
chi_pos              prose   code      9.449e-10
```

- **One row per observable and pair of batches.** `batch_a = batch_b` is a
  self pair (the batch with itself); `prose`, `code` is the cross pair asked
  for with `cross_pairs`. Passing `sink="results.parquet"` to `analyze` writes
  the same rows, with their settings, to a file.
- **What the cross pair adds.** `delta_loss(prose, code)` is the dot product of
  the two batches' loss gradients: positive here, so a small gradient step on
  the prose would also lower the loss on the code, to first order.
- What each observable is: the top-level `README.md` and `docs/SPEC.md`.

## What it does not show

- Anything about training: one checkpoint, two batches of 64 tokens. The
  Pythia sweep is the example that follows the observables over training.
- Exact `chi_net`, `chi_net_normalized` and `chi_pos` values: they are
  estimates from 8 random probes (`n_hutchinson=8`); another seed or more
  probes gives somewhat different numbers. `chi_loss`, `chi_loss_normalized`
  and `delta_loss` are exact.
- Run time on a GPU or on a cold cache: not measured.

## About the committed outputs

None; the output above is from a run on 17 November 2026 on the CPU (12th Gen
Intel Core i7-12700K, four threads, torch 2.10.0+cu128, the checkpoint already
cached), with `run.py` as committed. Two runs gave the same table.
