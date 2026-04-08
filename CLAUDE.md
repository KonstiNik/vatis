# vatis

Compute LNA observables (`chi_loss`, `chi_net`, `chi_pos`) on **pretrained model checkpoints**, scaled across multiple GPUs via DDP. Sister package to `perspic`, which does the same thing during Lightning training. vatis is for the regime where you can't pretrain yourself and instead probe published checkpoints (Pythia, OLMo, …).

## Core idea (math)

For a loss `L = (1/N) Σᵢ ℓᵢ` over `N` "samples" (= valid tokens for an LM), the LNA decomposition factors the linearized loss change:

```
δL(A, B) = χ_loss · χ_net · χ_pos
```

with

- `χ_loss = ‖∇_f L‖²`  — closed form from logits, **no backward through θ**.
- `χ_net  = ‖∇_θ f‖_F² = Tr(Θ)` where `Θ = ∇_θ f (∇_θ f)^T` is the eNTK.
- `χ_pos = δL / (χ_loss · χ_net)` — recovered indirectly, no eNTK eigendecomposition.
- `δL(A, B) = ⟨∇_θ L^A, ∇_θ L^B⟩` — two ordinary backwards plus a dot product.

The full derivation lives in `background_info.tex` (LNA-decomposition for mini-batch training). Read it before touching the math.

### How `χ_loss` and `δL` are computed (cheap, fixed cost)

These two are the same regardless of which `χ_net` estimator we use.

- `χ_loss` for cross-entropy is closed form from logits:
  ```
  ∂ℓ_{b,s} / ∂z_{b,s,v} = (softmax(z_{b,s})_v − onehot(y_{b,s})_v) / N_valid
  χ_loss = Σ_{b,s,v ∈ valid} (∂L/∂z_{b,s,v})²
  ```
  Zero backward passes through θ — pure tensor op on logits. For other losses we call `torch.autograd.grad(L, logits)`, which is one cheap backward through the loss head only.

- `δL(A, B)`: one full backward of `L^A` w.r.t. θ → flatten gradient `g_A`. Same for `g_B`. Then `δL = g_A · g_B`. **Two backwards total**, regardless of granularity. For the self case `δL(A, A) = ‖g_A‖²`, only one backward is needed.

Total fixed cost per `(checkpoint, eval-batch-pair)`: **2 backwards** (or 1 for self). Everything else goes into `χ_net`.

### `χ_net` estimation: three methods

`χ_net = Σ_b Tr(M_b)` where `M_b = J_b J_bᵀ` and `J_b = ∇_θ f(x_b)` is the per-sequence Jacobian. This is **the only expensive observable**, and the reason vatis exists. We support three methods, selected by the user (or auto-picked based on batch size and model compatibility). They share the same outer loop and produce the same observable; they differ only in *how* `Tr(M_b)` is estimated.

The math reason there's more than one method: a single VJP is a projection (`g = Jᵀ v` collapses information), so trace estimation needs either (a) deterministic basis vectors over the full output dim — exact but `O(S·V)` backwards, infeasible for LMs — or (b) random probes (Hutchinson) — `O(n)` backwards with controllable variance. The methods below differ in *how the random probes are arranged across the batch*.

#### Method 1: `HutchinsonEstimator` (default for large `B`)

Pure plain-torch Hutchinson at micro-batch level. For each micro-batch (sequences `[b₁, ..., b_M]`), draw `n_h` independent Rademacher vectors `v ∈ R^(M·S·V)`, each with the same shape as the micro-batch output. Per draw:

```python
out = model(x_micro)              # (M, S, V)
projected = (out * v).sum()       # scalar
g, = torch.autograd.grad(projected, model.parameters())
chi_net_accumulator += g.pow(2).sum()  # accumulates Σ_b ‖J_bᵀ v‖²
```

Cost: `n_h` backwards per micro-batch. No opacus, no vmap, no per-sample machinery. Variance is `~B× higher per draw` than the opacus path because the same `v` doesn't get reused across samples for free.

**When to use:** large `B` (≥ 32), or when other methods don't apply. This is the always-works fallback.

#### Method 2: `PerSequenceControlVariateEstimator` (default for small `B`)

For each sequence `b`, do **one extra backward** of `L_b` (per-sequence loss) w.r.t. `θ`. This gives `g_b = J_bᵀ u_b` where `u_b = ∇_f L_b` is known in closed form. From `(g_b, u_b)` we get exactly:

- `‖g_b‖²` — a Rayleigh quotient of `M_b` along the loss direction (free, deterministic).
- `⟨g_b, g_{b'}⟩` for all `b, b'` — the cross-sample alignment matrix, a bonus observable.

These are used as a **control variate** for the Hutchinson estimate of `Tr(M_b)`: the loss-direction projection is correlated with the trace, known exactly, has zero variance, and subtracting its noisy Hutchinson counterpart and adding back the exact value reduces total variance — substantially when the loss direction overlaps the top eigenmodes of `M_b`, which is exactly the high-`χ_pos` regime we care about.

Cost per (ckpt, batch): `n_h + B` backwards. Linear in `B`, so this method becomes unattractive for `B > ~64`.

**Bonus output:** the cross-sample alignment matrix `C_{bb'} = ⟨g_b, g_{b'}⟩` is written to the sink as an additional observable. It's exactly the eNTK projected onto the loss direction, sample-resolved. Useful for sample-level variance / interference analysis (the negative-`χ_pos` regime per `background_info.tex §A.2`).

**When to use:** moderate `B` (≤ 32), or when the user wants the per-sample alignment matrix.

#### Method 3: `OpacusEstimator` (opt-in fast path, v1.1)

When the model contains only opacus-supported layers and has no parameter tying, opacus + ghost clipping computes per-sample gradient norms in one backward, packing `B` samples per pass. With Hutchinson over `(S, V)` only, it's the fastest method **per backward count** — but the wall-time win only materializes when you can fit `M > 1` samples per micro-batch, which for an 8B model on an 80GB GPU is borderline. For smaller models (1B and below) it's a clear ~`B`× speedup.

Cost: `~n_h` backwards, where `n_h` is `~10–32`. Requires layer-compatibility check at startup; falls back with a clear error message if the model has tied embeddings (Pythia!), unsupported layers, or in-place ops that can't be neutralized.

**When to use:** explicitly opt-in via `chi_net_method="opacus"`, only after the user has verified it works on their model. **Not in v1.** Add in v1.1 once the other two are battle-tested.

#### Auto-selection rule (v1)

```python
def select_chi_net_method(B_total: int, requested: str | None) -> str:
    if requested is not None:
        return requested  # user override always wins
    if B_total <= 32:
        return "per_sequence_cv"
    return "hutchinson"
```

`opacus` is never auto-selected in v1.

### Compute scaling (8B / A100 80GB / 4 GPUs reference)

With `micro_batch_size=4` (activation checkpointing on, near memory ceiling), per-rank backward ≈ 1.8 s, ~25k tok/s/GPU. Per `(ckpt, eval-batch)` cost for `B=32`:

| method | backwards | wallclock | notes |
|---|---|---|---|
| `hutchinson` (n≈320, B-corrected) | ~320 | ~10 min | always applicable |
| `per_sequence_cv` (n≈32, +B=32) | ~64 | ~2 min | also gives cross-sample matrix |
| `opacus` (n≈10) | ~12 | ~22 s | not applicable to Pythia (tied embeddings) |

For a 50-checkpoint × 3-eval-batch sweep (150 pairs):

| method | total wallclock |
|---|---|
| `hutchinson` | ~25 h |
| `per_sequence_cv` | ~5 h |
| `opacus` | ~55 min |

These numbers are order-of-magnitude — actual perf depends heavily on the exact model, achievable micro-batch, and disk/network speed for checkpoint loading. For 1B-class models everything is 5–10× faster and `hutchinson` becomes interactive.

The real bottleneck for long sweeps is often **checkpoint loading**, not compute (~30 s/checkpoint cached, more from network). Cache aggressively; if multiple ranks share a node, share the HF cache directory.

### Sample unit and normalization

There is **one** observable, computed at token granularity, normalized by the number of valid (non-pad, non-ignored) tokens. The per-sequence vs per-token dichotomy is a normalization choice, not a different quantity:

- `χ_loss` and `χ_net` are sums over `(b, s, v)` regardless. Frobenius norms of Jacobians decompose additively across output dims.
- The `√|A|·|B|` factors from `background_info.tex §A.2` are batch-size normalization. We use **valid token counts**, not sequence counts. Track `n_valid_A`, `n_valid_B` per call.
- Normalized observables (default output):
  ```
  χ̃_loss = √(N_A · N_B) · χ_loss
  χ̃_net  = χ_net / √(N_A · N_B)
  ```
  where `N_A`, `N_B` are valid token counts. Both unnormalized and normalized values are written to the result sink so downstream analysis can pick.

## Scope (v1)

- **In:** HuggingFace causal-LM checkpoints (Pythia, OLMo, GPT-NeoX class). Tokenized data already in model-ready form. DDP across user-specified `num_gpus`. Two `χ_net` estimators (`hutchinson`, `per_sequence_cv`) with auto-selection. Parquet result sink (canonical) plus optional W&B and TensorBoard. CLI via `python -m vatis run …` that wraps `torchrun`.
- **Out (v1):** Tokenization. Checkpoint discovery / sweep scheduling. FSDP. Tensor / pipeline parallelism. `OpacusEstimator` (stub only — full implementation in v1.1). Per-sample variance observables beyond the cross-sample alignment matrix. Models that don't fit on one GPU.

Adding any of these later should be straightforward because the core (`vatis/core/`) is decoupled from model loading, data plumbing, and parallelism.

## Architecture

```
vatis/
├── __init__.py            # public API: analyze(), Analyzer
├── core/
│   ├── observables.py     # chi_loss, delta_loss, chi_pos (combine χ_net from estimator)
│   ├── chi_net/
│   │   ├── base.py        # ChiNetEstimator ABC
│   │   ├── hutchinson.py  # HutchinsonEstimator (always-works fallback)
│   │   ├── per_seq_cv.py  # PerSequenceControlVariateEstimator (default for small B)
│   │   └── opacus.py      # OpacusEstimator (opt-in, v1.1 — stub in v1)
│   ├── probes.py          # Rademacher / Gaussian probe vector generation, masking
│   └── normalization.py   # valid-token counting, √(N_A·N_B) factors
├── models/
│   ├── __init__.py        # load_model(spec) dispatch (registry-ready, single impl for now)
│   └── hf.py              # load_hf_model(name, revision) → nn.Module + loss_fn
├── data/
│   ├── batches.py         # EvalBatchSpec: "fixed" | "resample" | "callable"
│   └── collate.py         # micro-batch splitting, attention-mask aware
├── distributed/
│   ├── launcher.py        # `python -m vatis run` → torchrun wrapper
│   ├── ddp.py             # init_process_group, all_reduce of accumulators
│   └── sharding.py        # split eval batch across ranks; pad-and-mask if uneven
├── sinks/
│   ├── base.py            # ResultSink ABC
│   ├── parquet.py         # canonical: long-format rows
│   ├── wandb.py           # optional
│   └── tensorboard.py     # optional
├── analyzer.py            # Analyzer class: orchestrates checkpoint × eval_batch loop
├── cli.py                 # argparse / typer entrypoint
└── _version.py
```

### Public API

```python
from vatis import analyze

results = analyze(
    model="EleutherAI/pythia-160m",
    revisions=["step1000", "step2000", "step10000"],
    eval_batches={
        "val_fixed":   fixed_batch,             # tensor / dict, frozen
        "val_resample": val_dataloader,         # DataLoader, redrawn per checkpoint
        "train_at_step": lambda ckpt: ...,      # callable(checkpoint_id) -> batch
    },
    observables=["chi_loss", "chi_net", "chi_pos", "delta_loss"],
    chi_net_method=None,                        # None = auto: per_seq_cv if B≤32 else hutchinson
    n_hutchinson=32,                            # interpretation depends on method
    hutchinson_distribution="rademacher",       # or "gaussian"
    micro_batch_size=1,                         # safe default; user can grow it
    num_gpus=4,
    distributed="ddp",                          # only option for v1
    sink="results.parquet",                     # str → ParquetSink; or list of sinks
    dtype="bf16",                               # forward dtype; grads in fp32
)
```

`analyze()` is the one-shot convenience. The underlying `Analyzer` class is exposed for users who want to drive the loop themselves (e.g. interleave with their own logging).

### EvalBatchSpec

```python
from vatis.data import EvalBatchSpec

EvalBatchSpec.fixed(batch)            # frozen tensor or dict, reused at every checkpoint
EvalBatchSpec.resample(dataloader)    # next(iter(loader)) at each checkpoint
EvalBatchSpec.callable(fn)            # fn(checkpoint_id) -> batch
```

The `eval_batches` argument to `analyze()` is just `dict[str, EvalBatchSpec | Tensor | DataLoader | Callable]` — bare values are auto-wrapped. Multiple named eval batches are computed at every checkpoint and emitted as separate rows in the sink. This is the central feature: comparing observables across eval distributions at the same checkpoint is the primary use case.

For `δL(A, B)`, the user can specify `pairs=[("val_fixed", "train_at_step"), …]` to compute cross-batch alignments. By default `δL` is computed only for the "self" pair `(b, b)` for each named batch.

### Result format (parquet, long)

One row per `(checkpoint_id, eval_batch_name_A, eval_batch_name_B, observable, value)`. Long format because:
- It handles arbitrary numbers of eval batches and pair combinations without schema churn.
- Trivial to load into pandas and pivot.
- Append-friendly across DDP ranks (rank 0 writes after all-reduce).

Schema:
```
checkpoint_id      str
revision           str
batch_a            str
batch_b            str           # equals batch_a for self-observables
n_valid_a          int64
n_valid_b          int64
observable         str           # "chi_loss" | "chi_loss_normalized" | "chi_net" | ...
value              float64
n_hutchinson       int32         # null if not applicable
hutchinson_seed    int64         # null if not applicable
wallclock_s        float64
```

### Distributed semantics

- User passes `num_gpus=N`. CLI launches `torchrun --nproc_per_node=N -m vatis._worker …`.
- Each rank loads the same model checkpoint (v1 assumes the model fits on one GPU).
- Each eval batch is sharded across ranks along the batch dim. If `B % N != 0`, the last rank gets a smaller shard and contributes fewer valid tokens — the all-reduce handles this naturally because we accumulate token-weighted sums, not means.
- Per micro-batch on each rank: forward, compute logits → `χ_loss` accumulator. For `χ_net` Hutchinson: draw `n` probe vectors **with the same seed across ranks** so each rank computes a partial estimate of the same trace; all-reduce sums them. For `δL`: each rank computes its partial gradient `g_A^rank`, all-reduce-sum to get the full `g_A`, dot with `g_B` (same procedure).
- All-reduce happens **once per checkpoint × eval-batch** combination, not per micro-batch. Accumulators live on each rank as plain tensors.
- `micro_batch_size=1` is the always-safe fallback. The launcher does **not** attempt OOM auto-tuning in v1 — it's the user's responsibility to pick a size that fits.

### Probe vector details

- Default `n_hutchinson=32`, Rademacher distribution. Variance scales as `1/n`. For `HutchinsonEstimator` the effective variance is `~B×` higher per draw, so the auto-selection uses `per_sequence_cv` for small `B`.
- Probe seed is **shared across ranks** within one `(checkpoint, batch)` call so all ranks compute partial estimates of the same probe; different seed across calls.
- The probe vector shape matches the local micro-batch output: for HF causal LMs, `(M, S, V)`. **Padding positions are zeroed** in the probe vector so masked tokens contribute to neither `χ_net` nor `χ_loss`.
- No exact-vocab fallback in v1. If users want a deterministic check on a tiny test model, they crank `n_hutchinson` up.

### Loss handling

vatis ships with the standard HF causal-LM CE loss as default, but accepts a user-provided `loss_fn(logits, batch) -> scalar` for custom setups (e.g. masked LM, contrastive). The closed-form `χ_loss` shortcut is only used when `loss_fn` is the registered CE; otherwise we fall back to `torch.autograd.grad(loss, logits)` which is still cheap (one backward through the loss head only).

### Precision

- Forward in `bf16` by default (matches Pythia/OLMo training precision).
- Gradients accumulated in `fp32`. Hutchinson probes drawn in `fp32`, cast to forward dtype for the backward, gradient norm computed in `fp32`.
- `dtype="fp32"` is available for tiny test models.

## Dependencies

Hard:
- `torch >= 2.2`
- `transformers >= 4.40`
- `pyarrow` (parquet sink)
- `numpy`

Soft (extras):
- `wandb` → `vatis[wandb]`
- `tensorboard` → `vatis[tensorboard]`
- `pytest`, `ruff`, `mypy` → `vatis[dev]`

**No** `perspic`, `opacus`, `functorch`, or `pytorch-lightning`. The math is reimplemented in `vatis/core/observables.py` (~50 lines). This keeps vatis lightweight and lets it evolve independently of perspic.

## Tooling

- **Package manager:** `uv`. `pyproject.toml` is the single source of truth.
- **Python:** `>= 3.11`.
- **Lint + format:** `ruff` (for both — `ruff check` and `ruff format`). Black-compatible style. Drop-in for black, much faster, one tool.
- **Types:** `mypy --strict` on `vatis/`. Tests are unchecked.
- **Tests:**
  - `tests/unit/` — runs in CI. Uses a hand-built ~1M-param toy transformer (defined in `tests/fixtures/tiny_transformer.py`). Tests math correctness (Hutchinson convergence to exact trace as `n → ∞`, `χ_loss` closed-form vs autograd ground truth, `δL` symmetry, normalization invariants).
  - `tests/integration/` — gated behind `pytest -m integration`, **skipped in CI**, runnable locally. Uses `EleutherAI/pythia-14m` (smallest published Pythia) to verify the full HF loading + DDP path on real checkpoints.
  - `tests/cross_validation/` — gated behind `pytest -m cross_validation`, **skipped in CI** (requires perspic in the environment), runnable locally. **Cross-validates every observable against perspic** as the ground-truth reference implementation. The protocol:
    1. Build a small model that perspic can wrap (a `pl.LightningModule` with `criterion`) — e.g. a tiny MLP or the toy transformer with a Lightning shim.
    2. Run `perspic.analyzer(...)` on a single training step with a fixed batch and seed, capturing the logged `chi_loss`, `chi_net`, `chi_align`, `chi_coup`, and `grad_norm_squared` (= `δL`).
    3. Run `vatis.analyze(...)` on the **same model weights, same batch, same seed**, with `chi_net_method="hutchinson"` and a very large `n_hutchinson` (≥ 4096) so the Hutchinson noise is negligible.
    4. Assert that `vatis.chi_loss ≈ perspic.chi_loss`, `vatis.chi_net ≈ perspic.chi_net` (within Hutchinson std), `vatis.delta_loss ≈ perspic.grad_norm_squared`, and **`vatis.chi_pos ≈ perspic.chi_align ≈ perspic.chi_coup`**.
    5. Repeat with `chi_net_method="per_sequence_cv"` to verify the control-variate path agrees.
    6. Repeat with both opacus and functorch perspic backends to make sure normalization conventions match across the board.

    **Naming map for the cross-validation tests:**

    | perspic | vatis |
    | --- | --- |
    | `chi_loss` | `chi_loss` |
    | `chi_net` | `chi_net` |
    | `chi_align` / `chi_coup` | **`chi_pos`** |
    | `grad_norm_squared` | `delta_loss` |

    Any disagreement larger than the Hutchinson noise floor + ~1% numerical tolerance is a bug in vatis (perspic is the reference).
- **CI:** GitHub Actions, single matrix entry `python-3.11`, runs `ruff check`, `ruff format --check`, `mypy`, `pytest -m "not integration"`.

## Working conventions (for the agent building this)

- **Use subagents where it makes sense.** Spawn `Explore` subagents for codebase searches that span more than 2–3 files (e.g. "find all places perspic computes χ_loss" before writing the cross-validation tests). Spawn `Plan` subagents for non-trivial implementation steps where the design isn't obvious from CLAUDE.md alone. Spawn parallel `general-purpose` subagents when several independent investigations can run at once (e.g. checking HF API for `from_pretrained(revision=...)` while simultaneously checking how perspic's logger formats its outputs). Don't spawn subagents for trivial edits or single-file reads — that's overhead for no gain.
- **All installs go into the existing venv only. Nothing global, ever.** The venv lives at `.venv/` in the project root and already has perspic installed editable. Use `uv pip install …` (which respects the active venv) or `uv add …` for pyproject-tracked deps. **Never** run `pip install` without the venv activated, **never** use `--user`, **never** touch the system Python. If a tool seems to want a global install, stop and ask. Verify with `which python` → should resolve to `.venv/bin/python` before any install.
- **Run tests through the venv too:** `.venv/bin/pytest …` or `uv run pytest …`. Same for `ruff`, `mypy`, `python -m vatis`.

### Permission model (what's enforced vs convention)

The `settings.local.json` allow list bounds *some* of what you can do, but **most safety is behavioral, not technical.** Read this section before making any change that touches things outside the vatis project tree.

**Technically enforced (the system will block you):**
- `Write` and `Edit` tool calls are restricted to `/tikhome/knikolaou/PycharmProjects/vatis/**`. You cannot use these tools to modify files outside the project.
- Recursive `rm -r` / `rm -rf` is **not** in the allow list. Only flat `rm -- <files>` is allowed. Recursive deletes will prompt and stall in unattended mode.
- `git push`, `git reset --hard`, `git rebase`, `git checkout` of branches, `git branch -D`, `git clean`, anything that writes to remotes or rewrites history — **not allowed**. Only read commands and safe writes (`add`, `commit`, `restore`, `stash`) are.
- `sed`, `awk`, `find`, `source` — **removed from the allow list** because they each provided escape hatches around the file-write restriction. Use `Edit` for edits, `Glob`/`Grep` for searching, and never `source` arbitrary scripts.
- No network write tools (no `curl`, no `gh`, no `ssh`). `WebFetch`/`WebSearch` are read-only.

**Convention only — the system will NOT stop you, but you must not do these things:**
- **Arbitrary Python execution is the fundamental escape hatch.** `Bash(python:*)`, `Bash(uv:*)`, `Bash(.venv/bin/python:*)`, and `Bash(pytest:*)` all let you run Python code, which can do *anything Python can do* — including `open(path, 'w')` on any path, `os.remove`, `subprocess.run`, network calls. This bypasses every `Write`/`Edit` path restriction. **You must not use Python's standard library to write, delete, or move files outside the project tree.** If you find yourself writing `open("/some/path/outside/vatis", "w")` or `subprocess.run(["rm", "-rf", ...])` or `os.system(...)`, stop. The allow list doesn't catch this; the user trusts you not to do it.
- **Don't `pip install` random packages.** Stick to dependencies declared in `pyproject.toml`. If you genuinely need a new dep, add it via `uv add <pkg>` (which records it in `pyproject.toml` so the user can review later) — never via `pip install <pkg>` ad hoc.
- **Don't `mv` or `cp` files outside the project.** `Bash(mv:*)` and `Bash(cp:*)` are wide for convenience; the user accepted the residual risk on the understanding that you only operate inside the vatis tree.
- **Don't modify `settings.local.json` to grant yourself permissions you weren't given.** The allow list is the user's contract. If you need a permission you don't have, log it to `BLOCKED.md` and stop.
- **Don't modify files in `~/.ssh`, `~/.gitconfig`, `~/.aws`, `~/.config`, or any dotfile in `$HOME`.** Even though you technically *could* via Python, you must not.

**The real safety boundary is the host filesystem.** Treat anything outside `/tikhome/knikolaou/PycharmProjects/vatis/` as off-limits, regardless of which tool would let you reach it. The only exceptions are read-only operations (which are explicitly allowed everywhere via `Read(/**)` etc.) and the HuggingFace cache (which transformers will populate at `~/.cache/huggingface/` as a side effect of `from_pretrained` — that's expected and fine).

**When in doubt, log to `BLOCKED.md` and skip.** It's always better to leave a step undone with a clear note than to take a risky workaround.
- **Unattended-mode behavior (graceful fallback when blocked).** This project is built to run autonomously when the user is not around. If a tool call gets denied by the permission system:
  1. **Do not retry the same call.** Retrying just blocks again. There is no human to approve.
  2. **Try an allowed alternative.** Examples: if `Bash(curl …)` is denied, try `WebFetch`. If a `Bash` install command is denied, try `uv add` (which is allowed). If `Write` to a path outside the project is denied, write inside the project instead.
  3. **If no alternative exists, skip the blocked step and keep going on independent work.** Make a note in a `BLOCKED.md` file at the project root with: the exact tool call that was denied, what you were trying to do, what the user needs to do to unblock it, and which downstream tasks depend on it. Continue with everything that doesn't depend on the blocked step.
  4. **Never use destructive workarounds** to "make the obstacle go away" — no `rm -rf` to dodge a permission check, no `--no-verify` on git, no editing settings.local.json to grant yourself permissions you weren't given. The allow list is the user's contract; expanding it is the user's job, not yours.
  5. **At the end of an unattended session, summarize in `BLOCKED.md`** what got built, what got skipped and why, and what the user needs to approve to resume. This is the file the user reads first when they come back.

## Conventions

- One concept per module. Don't bundle Hutchinson with normalization with sharding "because they're all numerical".
- The math core (`vatis/core/`) takes plain tensors and `nn.Module`. It knows nothing about HF, DDP, or sinks. This is the only part that needs to be bulletproof — everything else is plumbing.
- All-reduce only at the boundaries of `(checkpoint, batch)` computations, never in inner loops.
- Never materialize per-sample gradients. If a future feature needs them, it goes behind a separate `vatis.advanced` module and is opt-in.
- Padding is handled at one place: `vatis/core/normalization.py::valid_token_mask`. Every accumulator multiplies by this mask. No ad-hoc masking elsewhere.

## Glossary

| symbol | code name | meaning |
| --- | --- | --- |
| `χ_loss` | `chi_loss` | `‖∇_f L‖²` |
| `χ_net` | `chi_net` | `‖∇_θ f‖_F² = Tr(Θ)` |
| `χ_pos` | `chi_pos` | `δL / (χ_loss · χ_net)` — "spectral position". **Naming note:** in vatis this is `chi_pos`. The same quantity is called `chi_align` (and sometimes `chi_coup`) in perspic and in `background_info.tex`. When cross-validating against perspic (see "Testing" below), map `perspic.chi_align ↔ vatis.chi_pos` and `perspic.chi_coup ↔ vatis.chi_pos`. |
| `δL` | `delta_loss` | `⟨∇_θ L^A, ∇_θ L^B⟩` |
| `Θ` | (not stored) | empirical NTK; never materialized |
| `N_A`, `N_B` | `n_valid_a`, `n_valid_b` | valid (non-pad) token counts in batches A and B |
| `n` | `n_hutchinson` | number of Hutchinson probe vectors |

## Implementation order

1. `vatis/core/normalization.py`, `vatis/core/probes.py` — valid-token masking, Rademacher/Gaussian probe generation. Trivial unit tests.
2. `vatis/core/observables.py` — closed-form `χ_loss`, `δL`, `χ_pos` combinator. Unit tests against toy transformer ground truth.
3. `vatis/core/chi_net/base.py` + `hutchinson.py` — `ChiNetEstimator` ABC and the always-works fallback. Convergence test: Hutchinson estimate → exact `χ_net` (computed by iterating output dims on the toy model) as `n → ∞`.
4. `vatis/core/chi_net/per_seq_cv.py` — per-sequence backward + control variate. Test that variance is reduced vs pure Hutchinson on toy model and that the estimator is unbiased.
5. `vatis/core/chi_net/opacus.py` — **stub only in v1**: raises `NotImplementedError("opacus path is v1.1; use 'hutchinson' or 'per_sequence_cv'")`.
6. `vatis/models/hf.py` — `load_hf_model(name, revision)`.
7. `vatis/data/batches.py` — `EvalBatchSpec` and auto-wrapping.
8. `vatis/sinks/parquet.py` — long-format writer, then base ABC, then optional sinks.
9. `vatis/analyzer.py` — single-GPU loop first. End-to-end test with toy transformer + a tiny synthetic dataset, no DDP. Verify auto-selection of `chi_net_method` based on `B`.
10. `vatis/distributed/` — DDP sharding + all-reduce. Test on 2-GPU loopback. Verify probe seed sharing across ranks.
11. `vatis/cli.py` + `python -m vatis` — torchrun wrapper.
12. Integration test on `pythia-14m` (gated, local). Both methods, both single-GPU and DDP.

**v1.1 (deferred):** full `OpacusEstimator` with layer-compatibility check, in-place-op neutralization, parameter-tying detection, and clear fallback messages. Reuse the patterns from `perspic/calculator/samplewise_opacus.py`.
