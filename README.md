# vatis

Compute **LNP observables** — `chi_loss`, `chi_net`, `chi_pos` — on **pretrained
HF checkpoints** (Pythia, OLMo, GPT-NeoX), scaled across GPUs with DDP. Sister
package to `perspic`, which computes the same quantities *during* Lightning
training; vatis is for when you can only probe published weights.

The LNP decomposition factors the linearized loss change into three terms —
**L**oss, **N**etwork, **P**osition:

```
δL  =  χ_loss · χ_net · χ_pos
```

## What you get

Per `(checkpoint, eval-batch)`, four scalars:

| observable   | symbol             | meaning                                            | cost |
|--------------|--------------------|----------------------------------------------------|------|
| `chi_loss`   | ‖∇_f L‖²           | steepness of the loss in output space              | free (closed form from logits) |
| `chi_net`    | Tr(JJᵀ)            | eNTK trace — how much weight updates move outputs  | the expensive one (Hutchinson) |
| `chi_pos`    | δL/(χ_loss·χ_net)  | spectral **position**: bulk (→1) vs tail (→0)      | combinator |
| `delta_loss` | ⟨∇_θ Lᴬ, ∇_θ Lᴮ⟩   | gradient alignment of two batches (A=B → ‖∇_θ L‖²) | 1–2 backwards |

## Setup

vatis recommends [uv](https://docs.astral.sh/uv/) for environment management —
set it up first. Needs Python 3.11+.

```bash
uv venv --python 3.11      # creates ./.venv with py3.11
source .venv/bin/activate  # then python/pytest/... just work
```

**To use vatis** (probe checkpoints, run `analyze`):

```bash
uv pip install .
```

Smoke-check it works (no GPU needed):

```bash
python -c "from vatis import analyze; print('vatis ready')"
```

**To develop vatis or run the test suite** — editable, adds pytest, ruff, mypy:

```bash
uv pip install -e ".[dev]"
pytest tests/unit -q          # 50 CPU-only unit tests, ~30 s

# the cross-validation tier additionally compares against perspic:
uv pip install -e /path/to/perspic
pytest tests/cross_validation -m cross_validation -q
```

Optional logging backends: `uv pip install ".[wandb]"` or `".[tensorboard]"`.

## Quick start

```python
from vatis import analyze

analyze(
    model="EleutherAI/pythia-160m",
    revisions=["step1000", "step8000", "step143000"],
    eval_batches={"prose": prose_batch, "code": code_batch},
    cross_pairs=[("prose", "code")],   # adds δL(prose,code) + chi_pos, no extra backward
    sink="results.parquet",
)
```

Runnable end-to-end (downloads pythia-14m, ~20 s warm):

```bash
python examples/pythia_sweep.py        # compute  → results.parquet + plots
python examples/analyze_results.py     # analyze  → derived plots (zero vatis imports)
```

## Output

Long-format parquet: one row per `(checkpoint, batch_a, batch_b, observable)`,
each in raw and `_normalized` form. Load with pandas/pyarrow and pivot. The
schema is the contract between vatis and any downstream analysis — see
`CLAUDE.md → Result format`.

## Multi-GPU

```bash
python -m vatis run --model EleutherAI/pythia-160m --revisions step1000 step2000 \
    --eval-batch val=batch.pt --sink results.parquet --num-gpus 4
```

Wraps `torchrun`; each rank takes a batch shard and accumulators are all-reduced
once per `(checkpoint, batch)`. Tune `micro_batch_size` for memory.

## Choosing the χ_net estimator

Auto by default: `per_sequence_cv` for B ≤ 32 (lower variance, also returns the
cross-sample alignment matrix), `hutchinson` otherwise (always works). Override
with `chi_net_method=`.

## Learn more

- `CLAUDE.md` — full spec: math, architecture, conventions, known issues.
- `examples/` — runnable sweeps and the spectral-tail / DPO research experiments.
- `background_info.tex` — the LNP derivation.
```