# vatis

Compute  **Spectral Position** — and LNP observables `chi_loss`, `chi_net`, `chi_pos` — on **pretrained HF checkpoints** (Pythia, OLMo, GPT-NeoX), scaled across GPUs with DDP. Sister
package to [perspic](https://github.com/zincware/perspic), which computes the same quantities *during* Lightning
training; vatis is for when you want to probe existing checkpoints.

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
pytest tests/unit -q          # 52 CPU-only unit tests, ~30 s

# the cross-validation tier additionally compares against perspic (the
# unpublished sister package); skip this tier if you don't have it:
uv pip install -e /path/to/perspic
pytest tests/cross_validation -m cross_validation -q
```

The unit tier above is CPU-only, runs anywhere, and is the tier CI runs.

To run the example/benchmark scripts (they need matplotlib + tqdm, which the
package itself does not): `uv pip install ".[examples]"`.

## Quick start

```python
from vatis import analyze

analyze(
    model="EleutherAI/pythia-160m",
    revisions=["step1000", "step8000", "step143000"],
    eval_batches={"prose": prose_batch, "code": code_batch},
    cross_pairs=[("prose", "code")],   # adds δL(prose,code) + chi_pos, no extra backward
    cross_grad_storage="auto",         # offload cross-pair grads to host RAM on large models
    sink="results.parquet",
)
```

`cross_grad_storage="auto"` (the default) lets cross-pair analysis scale to
models whose two full gradients won't co-reside on the GPU: it offloads the
cached gradients to host RAM and runs the dot on the CPU, while small models
stay fully on-device.

Runnable end-to-end (downloads pythia-14m, ~20 s warm):

```bash
python examples/pythia_sweep.py        # compute  → results.parquet + plots
python examples/analyze_results.py     # analyze  → derived plots (zero vatis imports)
```

For a **production-scale** demonstration — the full LNP decomposition on a real
7B SFT checkpoint (`Olmo-3-7B-Think-SFT`) over real chat data, with
code-vs-English cross-pair analysis and wall-time / memory / GPU-activity
tracking — see [`examples/olmo_sft_analysis/`](examples/olmo_sft_analysis/README.md).

Models download to the HuggingFace cache (`~/.cache/huggingface` by default).
To relocate it, set `HF_HOME` in your shell or copy `.env.example` to `.env`
and edit it — the example scripts read it via `run_config.py`. SLURM benchmark
jobs are submitted with `examples/benchmark/submit.sh <script>.sbatch`, which
pulls `SBATCH_PARTITION`/`SBATCH_ACCOUNT` from the same `.env`.

## Output

Long-format parquet: one row per `(checkpoint, batch_a, batch_b, observable)`,
each in raw and `_normalized` form. Load with pandas/pyarrow and pivot. The
schema is the contract between vatis and any downstream analysis — see
[`docs/SPEC.md` → Result format](docs/SPEC.md).

## Multi-GPU

```bash
python -m vatis run --model EleutherAI/pythia-160m --revisions step1000 step2000 \
    --eval-batch val=batch.pt --sink results.parquet --num-gpus 4
```

Wraps `torchrun`; each rank takes a batch shard and accumulators are all-reduced
once per `(checkpoint, batch)`. Tune `micro_batch_size` for memory.

## Choosing the χ_net estimator

Auto by default: `per_sequence_cv` for B ≤ 32 (lower variance), `hutchinson` otherwise 
(always works). Override with `chi_net_method=`.

## Learn more

- [`docs/SPEC.md`](docs/SPEC.md) — full spec: architecture, public API, result
  format, distributed semantics, scope.
- `examples/` — runnable sweeps plus the single-GPU / DDP-scaling / accuracy
  benchmarks under `examples/benchmark/`.
- The LNP decomposition is derived in the paper:
  [arXiv:2605.31244](https://arxiv.org/abs/2605.31244).
```