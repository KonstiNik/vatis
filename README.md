# vatis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23475753.svg)](https://doi.org/10.5281/zenodo.23475753)

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

With a checkpoint that carries its optimizer state, three of them can
additionally be measured on the optimizer's preconditioned kernel
(`delta_loss_P`, `chi_net_P`, `chi_pos_P`; `chi_loss` does not depend on the
kernel) — see
[Optimizer-aware measurement](#optimizer-aware-measurement).

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
pytest tests/unit -q          # the unit tier (GPU-only cases skip without CUDA)

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

An eval batch is a dict with `input_ids` and `labels` (plus `attention_mask` if
the model takes one). The labels decide what enters the loss — a position counts
iff its label is not `-100` — and they are **pre-shifted**: `labels[:, i]` is
the target of `logits[:, i]`, i.e. `input_ids[:, i + 1]`, with the last position
`-100`. A batch collated the HuggingFace / TRL way (`labels` aligned with
`input_ids`, the model shifts) needs one line first:

```python
from vatis.data import shift_hf_labels

batch["labels"] = shift_hf_labels(batch["labels"])
```

Labels that are already shifted — what `transformers` calls `shift_labels` —
are in vatis's form and need no conversion. The shipped loss refuses labels that
are provably unshifted rather than scoring them one position off; it cannot
catch every case (a single labelled token at the very end of a sequence looks
the same in both conventions), so convert knowingly rather than rely on it
([`docs/SPEC.md` → Batch format and the label convention](docs/SPEC.md)).

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

## Optimizer-aware measurement

An optimizer with update `Δθ = −η P ∇L` trains on the effective kernel
`Θ_P = J P Jᵀ` rather than on the raw eNTK. Pass a diagonal `P` and vatis
measures the LNP triple on that kernel too, in the same pass and with no extra
backward:

| observable     | symbol                | meaning |
|----------------|-----------------------|---------|
| `delta_loss_P` | ∇_θ Lᴮ · P ∇_θ Lᴬ     | preconditioned gradient alignment |
| `chi_net_P`    | Tr(J P Jᵀ)            | trace of the effective kernel |
| `chi_pos_P`    | δL^P/(χ_loss·χ_net^P) | spectral position on the preconditioned kernel |

```python
from vatis import Analyzer, load_local_bundle

source = load_local_bundle(run_dir, "my_project.models:build_model")
analyzer = Analyzer(
    eval_batches={"train": source.batch_a_spec(), "val": val_batch},
    preconditioner=source.preconditioner(),  # AdamW: 1/(√v̂ + ε) per checkpoint; SGD: I
    cross_pairs=[("train", "val")],
    sink="results.parquet",
)
analyzer.run(source.model_specs())
```

`preconditioner` also accepts `"identity"` or any diagonal you build yourself
(`Preconditioner.from_diagonal(vector)`), which works with HF checkpoints as
well. The vector must be in the order of the gradient vatis flattens: the
concatenation of `ModelBundle.params`, each tensor flattened in its own
(row-major) order. For the hub loader that is `model.parameters()` order, in
which tied weights appear once; for a bundle it is `param_names.json` order.

Without the argument there are no P rows and no extra cost. One thing does
differ from earlier releases either way: for a given `seed`, `chi_net` and
`chi_pos` with the default Rademacher probes come out as a different draw,
because the Hutchinson probes are now drawn differently (see the
[changelog](CHANGELOG.md)). `chi_loss` and `delta_loss` are unchanged.

Three things to know before using it:

- **AdamW's `P` is read from vatis's own checkpoint-bundle format only** — a run
  directory with weights, `optim.pt` and the update batch per checkpoint
  ([`docs/SPEC.md` → Native checkpoint bundles](docs/SPEC.md)). There is no
  loader yet for optimizer states published with hub checkpoints.
- **`delta_loss_P` is not the first-order loss change of the AdamW step.** `P`
  is the geometry factor only: momentum and weight decay are excluded, and it is
  built from the second moment *before* the step
  ([`docs/SPEC.md` → What `delta_loss_P` is not](docs/SPEC.md)).
- Coordinates that never received a non-zero gradient sit at `P = 1/ε` and can
  dominate `chi_net_P` — on held-out data, and equally on the update batch
  itself when it contains a token the run has not seen before (`P` is the
  pre-step one). Nothing is clipped; instead every AdamW row comes with a
  `delta_loss_P_capped` / `chi_net_P_capped` companion giving the part that
  those coordinates contribute, so you can see when a value is carried by a
  handful of untouched coordinates
  ([`docs/SPEC.md` → Coordinates at the ceiling](docs/SPEC.md)).

## Output

Long-format parquet: one row per `(checkpoint, batch_a, batch_b, preconditioner,
observable)`; `chi_loss` and `chi_net` are written both unnormalized and as
`_normalized`. Load with pandas/pyarrow and pivot on `observable` to get one
line per checkpoint and batch pair with all values side by side (recipe in the
spec); the `preconditioner` column says which `P` the `_P` values used. The
schema is the contract between vatis and any downstream analysis — see
[`docs/SPEC.md` → Result format](docs/SPEC.md).

## Multi-GPU

```bash
python -m vatis run --model EleutherAI/pythia-160m --revisions step1000 step2000 \
    --eval-batch val=batch.pt --sink results.parquet --num-gpus 4
```

Wraps `torchrun`; each rank takes a batch shard and accumulators are all-reduced
once per `(checkpoint, batch)`. `micro_batch_size` trades memory against
noise — see [below](#micro_batch_size-is-not-only-a-memory-setting).

The sampled values of a multi-GPU run — `chi_net`, `chi_pos` and their `_P`
counterparts — depend on the `seed` **and on the number of ranks**: every rank
draws its own Hutchinson probe stream for its own shard, so the same seed gives
a different (equally valid, unbiased) estimate at 1, 2 or 4 ranks. Multi-rank
values for a given seed also differ from every earlier vatis version, which
seeded all ranks identically. `chi_loss` and `delta_loss` depend on neither.

## Choosing the χ_net estimator

Auto by default: `per_sequence_cv` for B ≤ 32 (lower variance), `hutchinson` otherwise 
(always works). Override with `chi_net_method=`.

## `micro_batch_size` is not only a memory setting

One Hutchinson probe spans the whole micro-batch, and the cross-terms between
its sequences add noise. `chi_net` (and `chi_pos` and the `_P` values built on
it) stays unbiased, but at fixed `n_hutchinson` its variance grows about
linearly with the micro-batch size, for both estimators, with a slope that
depends on the model. Measured in a single process at `B = 8`,
`n_hutchinson = 4`, relative to `micro_batch_size=1`:

| model | micro-batch 2 | micro-batch 4 | micro-batch 8 |
|---|---|---|---|
| toy transformer, trained | ×1.7 | ×3.4 | ×6.5 |
| `SimpleStories-1.25M` (pretrained hub model) | ×1.1 | ×1.6 | ×1.9 |
| toy transformer, untrained | ×2.3 | ×5.2 | ×10.6 |

These are three small CPU test models (≤ 1.25M parameters); nothing larger has
been measured, so do not read a number for your model off this table.

- The default `micro_batch_size=1` is the least noisy setting.
- The variance is proportional to `1/n_hutchinson`. If you raise
  `micro_batch_size` for speed, raising `n_hutchinson` by the same factor keeps
  the number of probe backwards unchanged and — on the trained toy and the hub
  model above, not quite on the untrained toy — at least restores the noise
  level.
- A multi-GPU run shards the batch before micro-batching, so at
  `micro_batch_size > 1` its noise differs from a single-process run's
  ([`docs/SPEC.md` → Probe vector details](docs/SPEC.md)).

## Learn more

- [`docs/SPEC.md`](docs/SPEC.md) — full spec: architecture, public API, result
  format, distributed semantics, scope.
- `examples/` — runnable sweeps plus the single-GPU / DDP-scaling / accuracy
  benchmarks under `examples/benchmark/`.
- The LNP decomposition is derived in the paper:
  [arXiv:2605.31244](https://arxiv.org/abs/2605.31244).
```