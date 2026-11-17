"""Real optimizer steps, recorded, for the update-response tests.

Everything the ``update_response*`` rows claim is a statement about what a
``torch.optim`` optimizer did to the weights, so the reference in these tests is
always the optimizer itself: a real ``step()`` on a real model, with the weights
and the ``state_dict()`` captured immediately before it and the weights again
immediately after. Nothing here re-derives an update formula.

Two producers share one training loop (:func:`run_training`):

- :func:`record_steps` keeps the steps in memory (any dtype — the exactness
  tests run in fp64);
- :func:`write_bundle` writes them as a checkpoint bundle in the schema the
  loader reads (pre-step semantics, full-sequence ``batch_a.npy``), with the
  weight files in a chosen dtype, and returns the in-memory record next to it so
  a test can hold the analyzer's rows against what the run really did.

Parameter order: the optimizer's own — group after group, the order in which
``state_dict()`` numbers the per-parameter state — which is what
``param_names.json`` has to list. For a single group that is
``named_parameters()`` order.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

from tests.utils.bundle_fixture import (
    DEFAULT_MODEL_FACTORY,
    GoldenTinyTransformer,
    batch_from_tokens,
    lm_loss,
    token_stream,
)

VOCAB_SIZE = 48
SEQ_LEN = 16
BATCH_SIZE = 4
MODEL_CONFIG: dict[str, Any] = {
    "vocab_size": VOCAB_SIZE,
    "seq_len": SEQ_LEN,
    "d_model": 32,
    "n_heads": 2,
    "n_layers": 2,
    "d_ff": 64,
}
_WEIGHT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}

OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], Any]


def make_model(
    *, dtype: torch.dtype = torch.float64, tie_head: bool = True, seed: int = 0
) -> nn.Module:
    """The golden fixture's tiny transformer (non-degenerate at initialisation)."""
    model = GoldenTinyTransformer(
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, tie_head=tie_head, seed=seed
    ).to(dtype=dtype)
    model.train()
    return model


def held_out_batch(seed: int = 903, n_rows: int = 8) -> dict[str, torch.Tensor]:
    """A batch the training stream (seeds ``< 900``) never draws."""
    return batch_from_tokens(
        token_stream(n_rows=n_rows, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=seed)
    )


def decay_groups(
    model: nn.Module,
    *,
    weight_decay: float,
    lr_no_decay: float | None = None,
) -> list[dict[str, Any]]:
    """Two param groups: matrices with weight decay, biases and norm gains without.

    The common "no decay on biases / norms" split. ``lr_no_decay`` additionally
    gives the second group its own learning rate.
    """
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay: dict[str, Any] = {
        "params": [p for p in model.parameters() if p.ndim < 2],
        "weight_decay": 0.0,
    }
    if lr_no_decay is not None:
        no_decay["lr"] = lr_no_decay
    return [{"params": decay, "weight_decay": weight_decay}, no_decay]


def optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    """The optimizer's parameters in ``state_dict()`` index order."""
    return [p for group in optimizer.param_groups for p in group["params"]]


def flat(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate into one fp64 vector (detached copies)."""
    return torch.cat([t.detach().reshape(-1).to(torch.float64) for t in tensors])


@dataclass
class RecordedStep:
    """One real optimizer step, as it happened.

    ``index`` is the number of updates applied before it (the pre-step counter
    ``t``); everything "before" is at ``theta_t`` and from the optimizer as it
    stood before ``step()``.
    """

    index: int
    tokens: np.ndarray
    params_before: list[torch.Tensor]
    params_after: list[torch.Tensor]
    state_dict: dict[str, Any]
    lr: float
    grad_update: torch.Tensor  # flat, BEFORE clipping
    grad_seen: torch.Tensor  # flat, what the optimizer was handed
    train_loss: float

    @property
    def delta(self) -> torch.Tensor:
        """``theta_{t+1} - theta_t``, flat fp64."""
        return flat(self.params_after) - flat(self.params_before)


@dataclass
class TrainingRecord:
    """A recorded run: the model class' parameter names and every step."""

    names: list[str]
    shapes: list[tuple[int, ...]]
    steps: list[RecordedStep] = field(default_factory=list)
    final_params: list[torch.Tensor] = field(default_factory=list)
    dtype: torch.dtype = torch.float64
    tie_head: bool = True
    seed: int = 0

    def model_at(self, params: Sequence[torch.Tensor]) -> nn.Module:
        """A fresh model holding ``params`` (in the recorded order), same dtype."""
        model = make_model(dtype=self.dtype, tie_head=self.tie_head, seed=self.seed)
        by_name = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in zip(self.names, params, strict=True):
                by_name[name].copy_(value)
        model.eval()
        return model

    def loss_and_grad(
        self, params: Sequence[torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[float, torch.Tensor]:
        """Plain-torch loss and flat gradient (recorded order) at ``params``."""
        model = self.model_at(params)
        by_name = dict(model.named_parameters())
        ordered = [by_name[name] for name in self.names]
        loss = lm_loss(model(batch["input_ids"]), batch)
        grads = torch.autograd.grad(loss, ordered)
        return float(loss.detach().to(torch.float64)), flat(grads)


def run_training(
    make_optimizer: OptimizerFactory,
    n_steps: int,
    *,
    dtype: torch.dtype = torch.float64,
    tie_head: bool = True,
    seed: int = 0,
    clip_grad_norm: float | None = None,
    make_scheduler: SchedulerFactory | None = None,
    batch_size: int = BATCH_SIZE,
    ignore_token: int | None = None,
) -> TrainingRecord:
    """Train the tiny transformer for ``n_steps`` real steps and record each.

    ``clip_grad_norm`` clips with ``torch.nn.utils.clip_grad_norm_`` over the
    optimizer's parameters, between ``backward()`` and ``step()``;
    ``make_scheduler`` builds a scheduler that is stepped after every optimizer
    step; ``ignore_token`` makes the training loss ignore that target id (the
    pad convention a bundle records as ``loss_ignore_index``).
    """
    torch.manual_seed(seed)
    model = make_model(dtype=dtype, tie_head=tie_head, seed=seed)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer) if make_scheduler is not None else None
    params = optimizer_parameters(optimizer)
    name_of = {id(p): name for name, p in model.named_parameters()}
    names = [name_of[id(p)] for p in params]
    record = TrainingRecord(
        names=names,
        shapes=[tuple(p.shape) for p in params],
        dtype=dtype,
        tie_head=tie_head,
        seed=seed,
    )
    tokens = token_stream(
        n_rows=batch_size * n_steps, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=seed + 1
    )
    for t in range(n_steps):
        ids = tokens[t * batch_size : (t + 1) * batch_size]
        batch = training_batch(ids, ignore_token)
        optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(model(batch["input_ids"]), batch)
        loss.backward()
        before = [p.detach().clone() for p in params]
        state = copy.deepcopy(optimizer.state_dict())
        lr = float(optimizer.param_groups[0]["lr"])
        grad_update = flat([p.grad for p in params])
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, clip_grad_norm)
        grad_seen = flat([p.grad for p in params])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        record.steps.append(
            RecordedStep(
                index=t,
                tokens=np.array(ids, dtype=np.int32, copy=True),
                params_before=before,
                params_after=[p.detach().clone() for p in params],
                state_dict=state,
                lr=lr,
                grad_update=grad_update,
                grad_seen=grad_seen,
                train_loss=float(loss.detach()),
            )
        )
    record.final_params = [p.detach().clone() for p in params]
    return record


def training_batch(ids: np.ndarray | torch.Tensor, ignore_token: int | None) -> dict[str, Any]:
    """The LM batch of a token array under the run's pad convention."""
    batch = batch_from_tokens(ids)
    if ignore_token is not None:
        batch["labels"] = batch["labels"].masked_fill(batch["labels"] == ignore_token, -100)
    return batch


def record_steps(make_optimizer: OptimizerFactory, n_steps: int, **kwargs: Any) -> TrainingRecord:
    """:func:`run_training` — the in-memory producer (fp64 by default)."""
    return run_training(make_optimizer, n_steps, **kwargs)


# ------------------------------------------------------------------ the bundle


def write_bundle(
    run_dir: str | Path,
    make_optimizer: OptimizerFactory,
    *,
    n_steps: int,
    bundle_steps: Sequence[int],
    weight_dtype: str = "fp32",
    metadata_extra: dict[str, Any] | None = None,
    optimizer_name: str | None = "adamw",
    **training: Any,
) -> TrainingRecord:
    """Train in fp32 and write the checkpoints ``bundle_steps`` as a bundle.

    Step ``t`` holds ``theta_t``, the optimizer state before update ``t + 1`` and
    the batch that update consumes; ``bundle_steps`` may include ``n_steps``
    (the final weights, with an update batch the run never consumed).
    ``metadata_extra`` is merged into ``metadata.json`` last, so a test can
    record — or contradict — anything.
    """
    run_dir = Path(run_dir)
    # One step more than asked for: the checkpoint at ``n_steps`` needs the
    # optimizer state as it stands after the last update, which the extra
    # recorded step captures before it steps.
    record = run_training(make_optimizer, n_steps + 1, dtype=torch.float32, **training)
    ignore_token = training.get("ignore_token")
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "written_by": "vatis tests/fixtures/update_response_toys.py",
        "model_config": {**MODEL_CONFIG, "tie_head": record.tie_head, "init_seed": record.seed},
        "model_factory": DEFAULT_MODEL_FACTORY,
        "B_train": int(record.steps[0].tokens.shape[0]),
        "seq_len": SEQ_LEN,
        "weight_dtype": weight_dtype,
        "gradient_clip_val": training.get("clip_grad_norm"),
    }
    if training.get("clip_grad_norm") is not None:
        metadata["gradient_clip_algorithm"] = "norm"
    if optimizer_name is not None:
        metadata["optimizer"] = optimizer_name
    if ignore_token is not None:
        metadata["mask_pad_in_loss"] = True
        metadata["loss_ignore_index"] = int(ignore_token)
    metadata.update(metadata_extra or {})
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "param_names.json",
        [[name, list(shape)] for name, shape in zip(record.names, record.shapes, strict=True)],
    )
    for step in bundle_steps:
        recorded = record.steps[step]
        ckpt_dir = run_dir / "checkpoints" / f"step_{step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                name: value.to(dtype=_WEIGHT_DTYPES[weight_dtype]).clone()
                for name, value in zip(record.names, recorded.params_before, strict=True)
            },
            str(ckpt_dir / "model.safetensors"),
        )
        torch.save(recorded.state_dict, ckpt_dir / "optim.pt")
        np.save(ckpt_dir / "batch_a.npy", recorded.tokens)
        _write_json(
            ckpt_dir / "meta.json",
            {
                "step": step,
                "tokens_seen": step * int(recorded.tokens.size),
                "lr": recorded.lr,
                "train_loss": recorded.train_loss,
                "wall_time": 0.0,
            },
        )
    return record


def edit_json(path: Path, **changes: Any) -> None:
    """Rewrite a JSON object file with ``changes`` applied (``...`` removes a key)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is ...:
            payload.pop(key, None)
        else:
            payload[key] = value
    _write_json(path, payload)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
