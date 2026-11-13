"""Write a real checkpoint bundle by running a tiny AdamW training loop.

This is the *writer* half of the WP0 golden acceptance gate (``03_estimator.md``
§Golden step 1). It runs ~100 CPU AdamW steps on a tiny model over a synthetic
but *learnable* token stream, bundling every ~10 steps in the frozen schema of
``00_context.md``, and hands the caller back everything needed to check the
bundle against the run that produced it.

Conventions this module commits to — the interface with
:mod:`vatis.models.local_bundle`, which is the *reader*:

- ``param_names.json`` is a bare JSON **list** of ``[name, [dim, ...]]`` pairs,
  in ``model.named_parameters()`` order, and the optimizer is constructed from
  ``model.parameters()`` so that saved optimizer-state index ``i`` is
  ``param_names[i]``. Nothing else goes in that file; the reader parses it
  strictly.
- **Pre-step semantics.** The checkpoint written for step ``t`` holds ``theta_t``
  and the optimizer state as it was *before* the step-``t`` update, and
  ``batch_a.npy`` is the batch step ``t`` is about to consume. Concretely the
  loop forwards, writes the checkpoint (with ``train_loss = L(theta_t; A_t)``
  from that forward), and only then backwards and steps.
- **``batch_a.npy`` stores the FULL token sequence**, int32
  ``[B_train, seq_len]``, not the model input alone. A consumer recovers the
  supervised pairs as ``x = ids[:, :-1]``, ``y = ids[:, 1:]``; equivalently (and
  this is what vatis does) it feeds all ``seq_len`` ids to the causal model and
  ignores the last position, which has no next-token target. The training loop
  here uses *exactly* that objective, so ``delta_loss(A_t, .)`` computed from
  the stored file is the gradient the optimizer actually applied — not a
  one-token-shorter approximation of it. :func:`labels_for` is the single place
  the shift is defined, and :attr:`BundleFixture.update_batches` lets the golden
  test assert the round trip.
- ``metadata.json`` records the sequence-length semantics explicitly
  (``seq_len``, ``batch_a_semantics``, ``seq_len_semantics``) so no consumer has
  to guess whether the stored length is the full sequence or the model input.

The architecture comes from a factory, configurable via the
``VATIS_TEST_MODEL_FACTORY`` environment variable as a ``"module:function"``
import string, defaulting to the self-contained :class:`GoldenTinyTransformer`
below. Nothing in this module imports a training stack.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

# The env var of ``03_estimator.md`` §Golden, and the default it falls back to.
MODEL_FACTORY_ENV = "VATIS_TEST_MODEL_FACTORY"
DEFAULT_MODEL_FACTORY = "tests.utils.bundle_fixture:build_default_model"

IGNORE_INDEX = -100

# Defaults for the fixture run. Small enough that the whole gate (train + three
# analyzer passes) is a matter of seconds on CPU.
DEFAULT_VOCAB_SIZE = 48
DEFAULT_SEQ_LEN = 16
DEFAULT_BATCH_SIZE = 4
DEFAULT_N_STEPS = 100
DEFAULT_BUNDLE_EVERY = 10
DEFAULT_LR = 3e-3
DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_EPS = 1e-8
DEFAULT_WEIGHT_DECAY = 0.01


# ------------------------------------------------------------- default model


class GoldenTinyTransformer(nn.Module):
    """A self-contained causal-LM transformer, ~20k parameters.

    Deliberately written out here rather than imported from
    ``tests/fixtures/tiny_transformer.py``: the golden fixture is the thing that
    has to keep working when this suite is read as the reference for how a bundle
    is produced, so it carries its own architecture.

    ``tie_head`` shares the output projection with the token embedding, the
    realistic case (and the one where ``named_parameters()`` reports the tensor
    once while ``state_dict()`` reports it twice).
    """

    def __init__(
        self,
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        seq_len: int = DEFAULT_SEQ_LEN,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 64,
        tie_head: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([_GoldenBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim >= 2:
                    p.copy_(torch.randn(p.shape, generator=generator) * 0.05)
                else:
                    p.zero_()
        if tie_head:
            self.lm_head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        positions = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.final_ln(h))


class _GoldenBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=-1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        scores = heads(q) @ heads(k).transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.full((s, s), float("-inf"), device=x.device, dtype=scores.dtype), diagonal=1
        )
        attn = torch.softmax(scores + causal, dim=-1)
        out = (attn @ heads(v)).transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(out)
        h = self.ln2(x)
        return x + self.ff2(F.gelu(self.ff1(h)))


def build_default_model(metadata: Any) -> nn.Module:
    """``model_factory(metadata) -> nn.Module`` for :class:`GoldenTinyTransformer`.

    Reads the architecture out of the bundle's own ``metadata.json``, so the
    bundle fully determines the model — the same contract a project factory has
    to honour. Works whether it is handed a ``BundleMetadata`` or a plain dict.
    """
    config = _model_config(metadata)
    return GoldenTinyTransformer(
        vocab_size=int(config.get("vocab_size", DEFAULT_VOCAB_SIZE)),
        seq_len=int(config.get("seq_len", DEFAULT_SEQ_LEN)),
        d_model=int(config.get("d_model", 32)),
        n_heads=int(config.get("n_heads", 2)),
        n_layers=int(config.get("n_layers", 2)),
        d_ff=int(config.get("d_ff", 64)),
        tie_head=bool(config.get("tie_head", False)),
        seed=int(config.get("init_seed", 0)),
    )


def _model_config(metadata: Any) -> dict[str, Any]:
    getter = getattr(metadata, "get", None)
    raw = getter("model_config", {}) if callable(getter) else {}
    return dict(raw) if isinstance(raw, dict) else {}


def configured_model_factory() -> str:
    """The ``"module:function"`` factory to use: ``VATIS_TEST_MODEL_FACTORY`` or
    the default tiny transformer."""
    return os.environ.get(MODEL_FACTORY_ENV) or DEFAULT_MODEL_FACTORY


# ------------------------------------------------------------------ the data


def token_stream(
    *,
    n_rows: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    noise: float = 0.1,
) -> np.ndarray:
    """A learnable synthetic token stream: ``next = perm[cur]`` with noise.

    Structured on purpose. Uniform random tokens have nothing to learn, so the
    loss would not move over 100 steps and the golden test's linearization
    comparison would be noise divided by noise. A fixed random permutation gives
    a bigram law the tiny model can partially fit, so ``loss(B)`` genuinely
    decreases. ``seed`` selects the *sequences*; the permutation is shared, so
    train and held-out rows come from the same distribution but are disjoint
    samples of it.
    """
    rng = np.random.default_rng(0xC0FFEE)
    permutation = rng.permutation(vocab_size)
    rng = np.random.default_rng(seed)
    out = np.empty((n_rows, seq_len), dtype=np.int32)
    out[:, 0] = rng.integers(0, vocab_size, size=n_rows)
    for s in range(1, seq_len):
        follow = permutation[out[:, s - 1]]
        random = rng.integers(0, vocab_size, size=n_rows)
        keep = rng.random(n_rows) >= noise
        out[:, s] = np.where(keep, follow, random)
    return out


def labels_for(input_ids: torch.Tensor) -> torch.Tensor:
    """The next-token targets of a full stored sequence.

    ``labels[:, :-1] = input_ids[:, 1:]``, ``labels[:, -1] = IGNORE_INDEX``. The
    supervised pairs are therefore exactly ``(ids[:, :-1] -> ids[:, 1:])``, which
    is what a consumer of ``batch_a.npy`` reconstructs. Identical to
    ``vatis.data.collate.shifted_lm_targets``, written out here so the fixture
    does not borrow the reader's notion of the convention it is supposed to be
    testing.
    """
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, :-1] = input_ids[:, 1:]
    return labels


def batch_from_tokens(ids: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
    """``[B, seq_len]`` token ids -> the vatis LM batch dict they encode."""
    tensor = (
        torch.from_numpy(np.ascontiguousarray(ids)) if isinstance(ids, np.ndarray) else ids.detach()
    ).to(dtype=torch.int64)
    return {"input_ids": tensor, "labels": labels_for(tensor)}


def lm_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean-reduced cross entropy over the valid (non-final) positions."""
    labels = batch["labels"]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )


# ---------------------------------------------------------------- the fixture


@dataclass
class BundleFixture:
    """What :func:`write_training_bundle` produced.

    ``update_batches`` maps a bundled step to the *exact* token array that was
    written to that checkpoint's ``batch_a.npy`` and consumed by that step, so a
    test can prove the bundle reproduces its own update batch.
    """

    run_dir: Path
    model_factory: str
    steps: tuple[int, ...]
    n_steps: int
    bundle_every: int
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    vocab_size: int
    seq_len: int
    batch_size: int
    update_batches: dict[int, np.ndarray] = field(default_factory=dict)
    train_losses: dict[int, float] = field(default_factory=dict)
    n_parameters: int = 0
    param_names: tuple[str, ...] = ()

    def expected_batch(self, step: int) -> dict[str, torch.Tensor]:
        """The batch dict a consumer must reconstruct for ``step``."""
        return batch_from_tokens(self.update_batches[step])


def write_training_bundle(
    run_dir: str | Path,
    *,
    model_factory: str | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    bundle_every: int = DEFAULT_BUNDLE_EVERY,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    betas: tuple[float, float] = DEFAULT_BETAS,
    eps: float = DEFAULT_EPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = 0,
    tie_head: bool = True,
    model_config_extra: dict[str, Any] | None = None,
) -> BundleFixture:
    """Run a tiny AdamW loop and write a bundle in the frozen schema.

    Pre-step semantics, full-sequence ``batch_a.npy``; see the module docstring.
    A final checkpoint is written *after* the loop, at step ``n_steps``, holding
    the fully trained ``theta`` and the batch a hypothetical step ``n_steps``
    would consume — so the bundle spans ``theta_0 ... theta_{n_steps}`` and the
    realized change in a held-out loss across the whole run is measurable from
    the bundle alone.

    Returns:
        A :class:`BundleFixture` describing what was written.
    """
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    factory_spec = model_factory or configured_model_factory()

    torch.manual_seed(seed)
    model_config: dict[str, Any] = {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "d_ff": 64,
        "tie_head": tie_head,
        "init_seed": seed,
    }
    if model_config_extra:
        model_config.update(model_config_extra)

    from vatis.models.local_bundle import resolve_model_factory

    model = resolve_model_factory(factory_spec)(_FactoryMetadata(model_config))
    model.train()
    params = list(model.parameters())
    param_records = [(name, tuple(p.shape)) for name, p in model.named_parameters()]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # metadata.json + param_names.json, written once before any checkpoint.
    _write_json(run_dir / "param_names.json", [[n, list(s)] for n, s in param_records])
    _write_json(
        run_dir / "metadata.json",
        _metadata(
            model=model,
            model_config=model_config,
            factory_spec=factory_spec,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            seed=seed,
            batch_size=batch_size,
            seq_len=seq_len,
            n_steps=n_steps,
            bundle_every=bundle_every,
            param_records=param_records,
        ),
    )

    tokens = token_stream(
        n_rows=batch_size * (n_steps + 1), seq_len=seq_len, vocab_size=vocab_size, seed=seed + 1
    )
    fixture = BundleFixture(
        run_dir=run_dir,
        model_factory=factory_spec,
        steps=(),
        n_steps=n_steps,
        bundle_every=bundle_every,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        n_parameters=sum(p.numel() for p in params),
        param_names=tuple(n for n, _ in param_records),
    )
    written: list[int] = []
    started = time.perf_counter()

    for step in range(n_steps + 1):
        ids_np = tokens[step * batch_size : (step + 1) * batch_size]
        batch = batch_from_tokens(ids_np)
        optimizer.zero_grad(set_to_none=True)
        # L(theta_t; A_t) — from theta_t, i.e. BEFORE the step-t update.
        loss = lm_loss(model(batch["input_ids"]), batch)
        if step % bundle_every == 0 or step == n_steps:
            _write_checkpoint(
                run_dir,
                step,
                model=model,
                optimizer=optimizer,
                batch_a=ids_np,
                train_loss=float(loss.detach()),
                tokens_seen=step * batch_size * seq_len,
                lr=lr,
                wall_time=time.perf_counter() - started,
            )
            written.append(step)
            fixture.update_batches[step] = np.array(ids_np, dtype=np.int32, copy=True)
            fixture.train_losses[step] = float(loss.detach())
        if step == n_steps:
            break  # nothing left to update: theta_{n_steps} is the final state
        loss.backward()
        optimizer.step()

    fixture.steps = tuple(written)
    return fixture


class _FactoryMetadata:
    """The minimal ``metadata``-like object the factory sees while *writing*.

    Reading a bundle hands the factory a real ``BundleMetadata``; here the
    metadata does not exist on disk yet, so the writer passes an object with the
    same ``get``/``[]``/``in`` surface. Keeping the shapes identical is what
    guarantees the model the loop trains is the model the reader rebuilds.
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        self.raw = {"model_config": model_config}

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw


def _metadata(
    *,
    model: nn.Module,
    model_config: dict[str, Any],
    factory_spec: str,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    seed: int,
    batch_size: int,
    seq_len: int,
    n_steps: int,
    bundle_every: int,
    param_records: list[tuple[str, tuple[int, ...]]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "written_by": "vatis tests/utils/bundle_fixture.py (WP0 component 03 golden gate)",
        "model_class": type(model).__name__,
        "model_config": model_config,
        "model_factory": factory_spec,
        "num_parameters": int(sum(p.numel() for _n, p in model.named_parameters())),
        "num_param_tensors": len(param_records),
        "optimizer": "adamw",
        "optimizer_class": "AdamW",
        "optimizer_hparams": {
            "lr": lr,
            "betas": [betas[0], betas[1]],
            "eps": eps,
            "weight_decay": weight_decay,
        },
        "lr_schedule": {"name": "constant", "lr": lr, "warmup_steps": 0},
        "seed": seed,
        "B_train": batch_size,
        "seq_len": seq_len,
        "dataset": "synthetic-bigram-permutation-stream",
        "total_steps_planned": n_steps,
        "bundle_every": bundle_every,
        "git_hashes": {"vatis": "0" * 40},
        # The two semantics notes the reader must not have to guess.
        "seq_len_semantics": (
            "seq_len is the number of token ids stored per row of batch_a.npy — the FULL "
            "sequence, not the model input alone. The causal model consumes all seq_len "
            "positions; the last has no next-token target and is ignored, so the supervised "
            "pairs are (ids[:, :-1] -> ids[:, 1:])."
        ),
        "batch_a_semantics": (
            "batch_a.npy is int32 [B_train, seq_len] holding the FULL token sequence of the "
            "update batch A_t that step t consumes at this theta_t (pre-step semantics). "
            "Recover x = ids[:, :-1], y = ids[:, 1:]."
        ),
        "optimizer_state_semantics": (
            "optim.pt is torch optimizer.state_dict() saved PRE-step: the state before the "
            "step-t update, so state['step'] is the number of updates already applied and is "
            "the bias-correction exponent matching the saved exp_avg_sq."
        ),
    }


def _write_checkpoint(
    run_dir: Path,
    step: int,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_a: np.ndarray,
    train_loss: float,
    tokens_seen: int,
    lr: float,
    wall_time: float,
) -> Path:
    ckpt_dir = run_dir / "checkpoints" / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        name: p.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        for name, p in model.named_parameters()
    }
    save_file(weights, str(ckpt_dir / "model.safetensors"))
    torch.save(_cpu_fp32(optimizer.state_dict()), ckpt_dir / "optim.pt")
    np.save(ckpt_dir / "batch_a.npy", np.asarray(batch_a, dtype=np.int32))
    _write_json(
        ckpt_dir / "meta.json",
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "lr": lr,
            "train_loss": train_loss,
            "wall_time": wall_time,
        },
    )
    return ckpt_dir


def _cpu_fp32(obj: Any) -> Any:
    """Move an optimizer ``state_dict`` to CPU and fp32, as the schema requires."""
    if isinstance(obj, torch.Tensor):
        out = obj.detach().to(device="cpu")
        return out.to(dtype=torch.float32) if out.is_floating_point() else out
    if isinstance(obj, dict):
        return {k: _cpu_fp32(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_fp32(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_fp32(v) for v in obj)
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
