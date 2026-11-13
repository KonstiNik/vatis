"""Hand-written synthetic checkpoint bundles for the local-bundle loader tests.

The bundle schema (``00_context.md`` §"Checkpoint bundle schema") is the
interface between the training stack (which writes bundles) and vatis (which
reads them). These helpers write bundles **by hand** — plain ``json.dump``,
``safetensors.torch.save_file``, ``torch.save``, ``numpy.save`` — and
deliberately do *not* import anything from ``vatis.models.local_bundle`` or
``vatis.core.param_layout``. If the writer used the reader's own notion of the
canonical order, an ordering bug would cancel out and the tests would pass on
a broken loader.

Two knobs matter for the alignment test:

- ``param_order`` — the order written into ``param_names.json``. The tests pass
  a *permutation* of ``model.named_parameters()`` order so that a loader which
  silently used the model's own iteration order produces a different vector.
- ``exp_avg_sq_constants`` — one distinct constant per parameter, so a
  permutation of same-shaped parameters is visible in the values.

The model deliberately contains two pairs of identically-shaped parameters
(``fc1.weight``/``fc2.weight`` and ``fc1.bias``/``fc2.bias``) so that swapping
them cannot be caught by a shape check alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

DEFAULT_VOCAB_SIZE = 16
DEFAULT_D_MODEL = 6
DEFAULT_SEQ_LEN = 5
DEFAULT_BATCH_SIZE = 3


class BundleTinyLM(nn.Module):
    """A tiny causal LM: ``(B, S) int ids -> (B, S, V) logits``.

    Two hidden layers of identical shape on purpose (see the module docstring).
    Sized in the hundreds of parameters by default so the tests stay exact and
    instant; the lazy-loading test scales it up via the constructor.

    ``tie_head`` shares ``head.weight`` with ``embed.weight`` — the project's
    Llama-style tied-embedding case, where ``named_parameters()`` reports the
    tensor once but ``state_dict()`` reports it under both names.
    """

    def __init__(
        self,
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        d_model: int = DEFAULT_D_MODEL,
        seq_len: int = DEFAULT_SEQ_LEN,
        seed: int = 0,
        tie_head: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=generator) * 0.1)
        if tie_head:
            self.head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        return self.head(h)


def tiny_lm_factory(metadata: Any) -> nn.Module:
    """``model_factory(metadata) -> nn.Module`` for :class:`BundleTinyLM`.

    Also the target of the import-string test
    (``"tests.fixtures.synthetic_bundle:tiny_lm_factory"``). Reads the
    architecture out of the bundle's ``metadata.json`` and works whether the
    loader hands it a ``BundleMetadata`` or a plain dict.
    """
    config = metadata.get("model_config", {})
    return BundleTinyLM(
        vocab_size=int(config.get("vocab_size", DEFAULT_VOCAB_SIZE)),
        d_model=int(config.get("d_model", DEFAULT_D_MODEL)),
        seq_len=int(config.get("seq_len", DEFAULT_SEQ_LEN)),
        tie_head=bool(config.get("tie_head", False)),
    )


def wrong_factory(metadata: Any) -> nn.Module:
    """A factory whose architecture does *not* match the bundle (error tests)."""
    del metadata
    return nn.Sequential(nn.Linear(4, 4))


# ------------------------------------------------------------------ orderings


def named_parameter_order(model: nn.Module) -> list[str]:
    """``model.named_parameters()`` order — the writer-side default."""
    return [name for name, _ in model.named_parameters()]


def permuted_parameter_order(model: nn.Module) -> list[str]:
    """A non-trivial permutation of the model's own parameter order.

    Reversing is enough: it moves every parameter *and* swaps the two pairs of
    identically-shaped parameters relative to each other, which is the case a
    shape check cannot catch.
    """
    return list(reversed(named_parameter_order(model)))


# --------------------------------------------------------------------- writing


def bundle_metadata(
    model: BundleTinyLM,
    *,
    optimizer: str = "adamw",
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int | None = None,
) -> dict[str, Any]:
    """A ``metadata.json`` payload with every field the schema names."""
    return {
        "model_config": {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "seq_len": model.seq_len,
            "tie_head": model.head.weight is model.embed.weight,
        },
        "model_class": type(model).__name__,
        "optimizer": optimizer,
        "optimizer_hparams": {
            "lr": lr,
            "betas": [betas[0], betas[1]],
            "eps": eps,
            "weight_decay": weight_decay,
        },
        "lr_schedule": {"name": "cosine", "warmup_steps": 0, "min_lr": lr / 10},
        "seed": seed,
        "B_train": batch_size,
        "seq_len": seq_len if seq_len is not None else model.seq_len,
        "dataset": "synthetic-random-tokens",
        "git_hashes": {"antonio_code": "0" * 40, "vatis": "1" * 40},
    }


def write_run_files(
    run_dir: Path,
    model: nn.Module,
    *,
    param_order: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    """Write ``metadata.json`` + ``param_names.json``; return the written order.

    ``param_names.json`` gets the frozen form ``[[name, [dim, ...]], ...]``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    order = list(param_order) if param_order is not None else named_parameter_order(model)
    unknown = [name for name in order if name not in shapes]
    if unknown:
        raise ValueError(f"param_order names not in the model: {unknown}")
    records = [[name, list(shapes[name])] for name in order]
    _write_json(run_dir / "param_names.json", records)
    payload: Mapping[str, Any]
    if metadata is not None:
        payload = metadata
    elif isinstance(model, BundleTinyLM):
        payload = bundle_metadata(model)
    else:  # pragma: no cover - defensive
        payload = {}
    _write_json(run_dir / "metadata.json", payload)
    return order


def handwritten_optim_state(
    model: nn.Module,
    param_order: Sequence[str],
    *,
    step: int,
    exp_avg_sq_constants: Mapping[str, float] | None = None,
    exp_avg_constants: Mapping[str, float] | None = None,
    param_indices: Sequence[int] | None = None,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> dict[str, Any]:
    """Build an AdamW-shaped ``state_dict`` by hand, in ``param_order``.

    ``state`` index ``i`` corresponds to ``param_order[i]`` — the alignment the
    schema mandates. ``param_indices`` overrides the ``param_groups`` index list
    (used to build a *deliberately broken* bundle for the rejection test).
    """
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    constants_sq = dict(exp_avg_sq_constants or {})
    constants_avg = dict(exp_avg_constants or {})
    state: dict[int, dict[str, Any]] = {}
    for index, name in enumerate(param_order):
        shape = shapes[name]
        state[index] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.full(shape, float(constants_avg.get(name, 0.0)), dtype=torch.float32),
            "exp_avg_sq": torch.full(
                shape, float(constants_sq.get(name, 1.0)), dtype=torch.float32
            ),
        }
    indices = list(param_indices) if param_indices is not None else list(range(len(param_order)))
    return {
        "state": state,
        "param_groups": [
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": False,
                "params": indices,
            }
        ],
    }


def run_adamw_steps(
    model: nn.Module,
    params: Sequence[torch.nn.Parameter],
    *,
    n_steps: int,
    batch: torch.Tensor,
    lr: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
) -> torch.optim.AdamW:
    """Run real AdamW steps and return the optimizer (real state to compare to).

    ``params`` is passed to the optimizer in the caller's order, so the
    resulting ``state_dict`` indices follow that order — which is exactly the
    convention ``param_names.json`` records.
    """
    optimizer = torch.optim.AdamW(
        list(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    targets = torch.roll(batch, shifts=-1, dims=1)
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def make_batch_a(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seed: int = 0,
) -> np.ndarray:
    """An int32 ``(B_train, seq_len)`` token-id array, as the schema requires."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)


def write_checkpoint(
    run_dir: Path,
    step: int,
    *,
    model: nn.Module,
    optim_state: Mapping[str, Any],
    batch_a: np.ndarray,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write one ``checkpoints/step_<08d>/`` directory per the frozen schema.

    Weights are stored bf16 (the loader upcasts to fp32), optimizer tensors
    fp32, ``batch_a`` int32.
    """
    ckpt_dir = run_dir / "checkpoints" / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        name: p.detach().to(dtype=torch.bfloat16).clone() for name, p in model.named_parameters()
    }
    save_file(weights, str(ckpt_dir / "model.safetensors"))
    torch.save(dict(optim_state), ckpt_dir / "optim.pt")
    np.save(ckpt_dir / "batch_a.npy", batch_a.astype(np.int32))
    payload = dict(meta) if meta is not None else {}
    payload.setdefault("step", step)
    payload.setdefault("tokens_seen", int(step * batch_a.size))
    payload.setdefault("lr", 1e-3)
    payload.setdefault("train_loss", 2.5 - 0.01 * step)
    payload.setdefault("wall_time", 0.5 * step)
    _write_json(ckpt_dir / "meta.json", payload)
    return ckpt_dir


def bf16_reference_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """What the loader must produce: the on-disk bf16 values, upcast to fp32.

    bf16 -> fp32 is exact, so the round-trip test can assert bit equality.
    """
    return {
        name: p.detach().to(dtype=torch.bfloat16).to(dtype=torch.float32).clone()
        for name, p in model.named_parameters()
    }


def write_simple_bundle(
    run_dir: Path,
    *,
    steps: Sequence[int] = (0, 10),
    param_order: Sequence[str] | None = None,
    exp_avg_sq_constants: Mapping[str, float] | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    d_model: int = DEFAULT_D_MODEL,
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    tie_head: bool = False,
) -> tuple[BundleTinyLM, list[str]]:
    """Write a complete multi-checkpoint bundle with hand-made optimizer state.

    Returns ``(model, param_order)`` where ``model`` holds the weights written
    at *every* step (the same theta at each step — these tests are about
    plumbing and ordering, not training dynamics), so a caller can compare
    against :func:`bf16_reference_weights`.
    """
    model = BundleTinyLM(
        vocab_size=vocab_size,
        d_model=d_model,
        seq_len=seq_len,
        seed=seed,
        tie_head=tie_head,
    )
    order = write_run_files(
        run_dir,
        model,
        param_order=param_order,
        metadata=bundle_metadata(model, batch_size=batch_size, seed=seed),
    )
    for step in steps:
        write_checkpoint(
            run_dir,
            step,
            model=model,
            optim_state=handwritten_optim_state(
                model,
                order,
                step=step,
                exp_avg_sq_constants=exp_avg_sq_constants,
            ),
            batch_a=make_batch_a(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                seed=seed + step,
            ),
        )
    return model, order


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
