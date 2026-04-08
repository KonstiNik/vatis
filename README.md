# vatis

Compute LNA observables (`chi_loss`, `chi_net`, `chi_pos`) on **pretrained model
checkpoints**, scaled across multiple GPUs via DDP. Sister package to `perspic`,
which does the same thing during Lightning training. vatis is for the regime
where you can't pretrain yourself and instead probe published checkpoints
(Pythia, OLMo, ...).

See `CLAUDE.md` for the full design specification, math derivation, and
implementation conventions.

## Quick start

```python
from vatis import analyze

results = analyze(
    model="EleutherAI/pythia-160m",
    revisions=["step1000", "step2000", "step10000"],
    eval_batches={"val": my_batch},
    observables=["chi_loss", "chi_net", "chi_pos", "delta_loss"],
    num_gpus=4,
    sink="results.parquet",
)
```

## Install

```bash
uv pip install -e .[dev]
```
