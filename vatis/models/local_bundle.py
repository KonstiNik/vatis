"""Native checkpoint-bundle loader — vatis's local (non-HuggingFace) source.

A *bundle* is a directory written by a training run, in the frozen schema
documented in ``docs/SPEC.md`` §"Native checkpoint bundles"::

    <run_dir>/
      metadata.json                # model config, optimizer type + hparams, lr
                                   # schedule, seed, B_train, seq_len, dataset, git
      param_names.json             # ordered [ [name, shape], ... ] — THE canonical
                                   # flattening order; index-aligned with the
                                   # optimizer state_dict indices
      checkpoints/step_00000123/
        model.safetensors          # weights, bf16
        optim.pt                   # torch optimizer.state_dict(), fp32, saved PRE-step
        batch_a.npy                # int32 [B_train, seq_len] token ids of update batch A_t
        meta.json                  # step, tokens_seen, lr, train_loss, wall_time

**Pre-step semantics** (do not reinterpret): the checkpoint at step ``t`` holds
``theta_t`` *and the optimizer state as it was before the step-t update*;
``batch_a.npy`` is the batch that step ``t`` is about to consume. The
preconditioner built from ``optim.pt`` is therefore the geometry factor from
*before* the step-``t`` update; the step itself applies ``P_{t+1}`` (which also
folds in ``grad L(batch_a)``) to the first moment, not to the gradient. See
``docs/SPEC.md`` §"What ``delta_loss_P`` is not".

Relationship to the HF source (``vatis/models/hf.py``): that module turns
``(name, revision)`` into a :class:`~vatis.models.hf.ModelBundle` — the four
things the analyzer needs (model, params, forward_fn, loss_fn, valid_mask_fn).
This module does the same for one bundle checkpoint, and adds the two things a
bundle has and the hub does not: the saved optimizer state, flattened in the
canonical order, and the update batch. vatis stays project-agnostic: the
architecture comes from a **user-supplied factory**
``model_factory(metadata) -> nn.Module`` (a callable or a ``"module:function"``
import string), never from an import of the training stack.

Laziness: :class:`LocalBundleSource` only stats the bundle at construction
time. Tensors are read when a :class:`LocalBundleCheckpoint`'s ``model_bundle``
/ ``optimizer_state`` / ``batch_a`` is first touched, and iterating the source
releases the previous checkpoint before yielding the next, so a 100 x 0.5 GB
run needs one checkpoint's worth of memory (see ``release()``).

Alignment (the correctness point): ``param_names.json`` is parsed into a
:class:`~vatis.core.param_layout.ParameterLayout`, and that single object is
used for *both* ``ModelBundle.params`` (via
:meth:`~vatis.core.param_layout.ParameterLayout.parameters_in_order`) and
:meth:`OptimizerStateView.flat_state`. There is therefore one mapping
``optimizer state index i <-> param_names[i] <-> flat segment i`` and one code
path producing it.
"""

from __future__ import annotations

import importlib
import json
import re
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vatis.core.param_layout import LayoutMismatchError, ParameterLayout
from vatis.core.preconditioner import Preconditioner, resolve_adamw_hparams
from vatis.data.batches import EvalBatchSpec
from vatis.models.hf import (
    ForwardFn,
    LossFn,
    ModelBundle,
    ValidMaskFn,
    causal_lm_loss,
    causal_lm_valid_mask,
)

# ``model_factory(metadata) -> nn.Module``, or a ``"module:function"`` string.
ModelFactory = Callable[["BundleMetadata"], nn.Module]
ModelFactorySpec = ModelFactory | str

_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# ``checkpoints/step_00000123`` (the schema's form) and ``step123`` both parse.
_STEP_DIR_RE = re.compile(r"^step[_-]?(\d+)$")

BUNDLE_METADATA_FILE = "metadata.json"
BUNDLE_PARAM_NAMES_FILE = "param_names.json"
BUNDLE_CHECKPOINTS_DIR = "checkpoints"
CHECKPOINT_WEIGHTS_FILE = "model.safetensors"
CHECKPOINT_OPTIM_FILE = "optim.pt"

# Optimizers whose update divides by ``sqrt(v_hat) + eps`` exactly as
# ``Preconditioner.adamw`` assumes. Adam and AdamW differ only in how weight
# decay enters, which P excludes anyway.
_ADAM_FAMILY = frozenset({"adam", "adamw"})
CHECKPOINT_BATCH_A_FILE = "batch_a.npy"
CHECKPOINT_META_FILE = "meta.json"


class BundleSchemaError(ValueError):
    """Raised when a bundle does not match the frozen schema.

    Kept a ``ValueError`` (as is
    :class:`~vatis.core.param_layout.LayoutMismatchError`) so callers can catch
    both with one ``except ValueError``.
    """


# --------------------------------------------------------------------- metadata


@dataclass(frozen=True)
class BundleMetadata:
    """Parsed ``metadata.json`` — run-level facts, plus dict-style access.

    The frozen schema fixes *which* facts are recorded (model config, optimizer
    type + hparams, lr schedule, seed, ``B_train``, ``seq_len``, dataset id, git
    hashes) but not the exact JSON key names, so :attr:`raw` is authoritative
    and the convenience accessors below look through a documented list of
    candidate keys and return ``None`` when nothing matches. They never invent
    a value.

    This object is what a ``model_factory`` receives. It also supports
    ``metadata["key"]`` / ``metadata.get("key")`` / ``"key" in metadata`` so a
    factory can treat it as the raw dict.
    """

    raw: Mapping[str, Any]
    path: Path | None = None

    # -- dict-style passthrough (so dict-shaped factories work unchanged) ----

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw

    def get(self, key: str, default: Any = None) -> Any:
        """``dict.get`` on the raw JSON."""
        return self.raw.get(key, default)

    # -- documented candidate-key accessors ---------------------------------

    @property
    def model_config(self) -> Any:
        """Architecture config: ``model_config`` | ``model`` | ``model_args`` | ``config``."""
        return self._first("model_config", "model", "model_args", "config")

    @property
    def optimizer_name(self) -> str | None:
        """Optimizer type: ``optimizer_name`` | ``optimizer_type`` | ``optimizer``.

        If ``optimizer`` is a dict, its ``name``/``type`` entry is used.
        """
        value = self._first("optimizer_name", "optimizer_type", "optimizer")
        if isinstance(value, Mapping):
            inner = value.get("name", value.get("type"))
            return str(inner) if inner is not None else None
        return str(value) if value is not None else None

    @property
    def optimizer_hparams(self) -> Mapping[str, Any] | None:
        """Optimizer hyper-parameters as recorded in ``metadata.json``.

        ``optimizer_hparams`` | ``optim_hparams`` | ``optimizer_kwargs``, or the
        ``optimizer`` entry itself when it is a dict.

        For the preconditioner, prefer :attr:`OptimizerStateView.hparams` — the
        ``param_groups`` of the saved ``optim.pt`` are what the optimizer
        actually ran with. Use this as a cross-check / fallback.
        """
        value = self._first("optimizer_hparams", "optim_hparams", "optimizer_kwargs")
        if isinstance(value, Mapping):
            return value
        optimizer = self.raw.get("optimizer")
        if isinstance(optimizer, Mapping):
            return optimizer
        return None

    @property
    def lr_schedule(self) -> Any:
        """LR schedule spec: ``lr_schedule`` | ``lr_scheduler`` | ``scheduler``."""
        return self._first("lr_schedule", "lr_scheduler", "scheduler")

    @property
    def seed(self) -> int | None:
        """Run seed: ``seed`` | ``master_seed`` | ``training_seed``."""
        return _opt_int(self._first("seed", "master_seed", "training_seed"))

    @property
    def batch_size_train(self) -> int | None:
        """``B_train``: ``B_train`` | ``batch_size`` | ``batch_size_train``."""
        return _opt_int(self._first("B_train", "batch_size", "batch_size_train"))

    @property
    def seq_len(self) -> int | None:
        """Sequence length: ``seq_len`` | ``max_seq_len`` | ``sequence_length``."""
        return _opt_int(self._first("seq_len", "max_seq_len", "sequence_length"))

    @property
    def dataset(self) -> Any:
        """Dataset id: ``dataset`` | ``dataset_id`` | ``data``."""
        return self._first("dataset", "dataset_id", "data")

    @property
    def git_hashes(self) -> Any:
        """Code provenance: ``git_hashes`` | ``git_hash`` | ``git``."""
        return self._first("git_hashes", "git_hash", "git")

    def _first(self, *keys: str) -> Any:
        for key in keys:
            if key in self.raw:
                return self.raw[key]
        return None


@dataclass(frozen=True)
class CheckpointMeta:
    """Parsed ``meta.json`` of one checkpoint (per-step facts)."""

    step: int
    tokens_seen: int | None = None
    lr: float | None = None
    train_loss: float | None = None
    wall_time: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], *, source: str) -> CheckpointMeta:
        if "step" not in raw:
            raise BundleSchemaError(f"{source}: meta.json has no 'step' field")
        step = _opt_int(raw["step"])
        if step is None:
            raise BundleSchemaError(
                f"{source}: meta.json 'step' is not an integer: {raw['step']!r}"
            )
        return cls(
            step=step,
            tokens_seen=_opt_int(raw.get("tokens_seen")),
            lr=_opt_float(raw.get("lr")),
            train_loss=_opt_float(raw.get("train_loss")),
            wall_time=_opt_float(raw.get("wall_time")),
            raw=raw,
        )


# ------------------------------------------------------------- optimizer state


class OptimizerStateView:
    """The saved (pre-step) optimizer state, aligned to the canonical order.

    ``torch.optim.Optimizer.state_dict()`` keys its per-parameter ``state`` by
    an **integer index**, assigned by enumerating ``param_groups`` in order —
    which is the order the parameters were handed to the optimizer, i.e. the
    ``model.parameters()`` order that ``param_names.json`` records. So::

        state[i]  <->  layout.names[i]  <->  flat segment i

    is the one mapping, and this class is the one code path that applies it.
    Two invariants are checked at construction (both would otherwise be silent
    permutations of the flat vector):

    1. the ``params`` index lists of the saved ``param_groups``, concatenated in
       group order, must be exactly ``[0, 1, ..., len(layout) - 1]``. Anything
       else (a subset of the model, a re-ordering, a hole) means index ``i`` is
       not ``layout.names[i]``;
    2. each per-parameter state tensor must have the shape the canonical order
       records for that index.

    Bias correction, ``eps``, momentum handling and every other piece of
    preconditioner math belong to :mod:`vatis.core.preconditioner`; this class only
    hands out correctly-ordered vectors and the hparams the optimizer ran with.
    """

    __slots__ = ("_layout", "_param_groups", "_source", "_state")

    def __init__(
        self,
        state_dict: Mapping[str, Any],
        layout: ParameterLayout,
        *,
        source: str = "<optim.pt>",
    ) -> None:
        if "param_groups" not in state_dict or "state" not in state_dict:
            raise BundleSchemaError(
                f"{source}: not a torch optimizer state_dict "
                f"(expected 'state' and 'param_groups' keys, got {sorted(state_dict)})"
            )
        raw_groups = state_dict["param_groups"]
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
            raise BundleSchemaError(f"{source}: 'param_groups' is not a list")
        groups: list[dict[str, Any]] = []
        indices: list[int] = []
        for gi, group in enumerate(raw_groups):
            if not isinstance(group, Mapping):
                raise BundleSchemaError(f"{source}: param_groups[{gi}] is not a dict")
            groups.append(dict(group))
            group_params = group.get("params")
            if group_params is None:
                raise BundleSchemaError(f"{source}: param_groups[{gi}] has no 'params' list")
            indices.extend(int(i) for i in group_params)
        expected = list(range(len(layout)))
        if indices != expected:
            raise LayoutMismatchError(
                f"{source}: the optimizer's parameter indices are "
                f"{_abbrev(indices)}, but the canonical order ({layout.source}) has "
                f"{len(layout)} parameters, so they must be {_abbrev(expected)}. "
                "The optimizer state indices are therefore NOT index-aligned with "
                "param_names.json and no preconditioner can be built from them."
            )

        raw_state = state_dict["state"]
        if not isinstance(raw_state, Mapping):
            raise BundleSchemaError(f"{source}: 'state' is not a dict")
        state: dict[int, Mapping[str, Any]] = {}
        for key, entry in raw_state.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise BundleSchemaError(
                    f"{source}: optimizer state key {key!r} is not a parameter index"
                ) from exc
            if not 0 <= index < len(layout):
                raise LayoutMismatchError(
                    f"{source}: optimizer state has index {index} but the canonical "
                    f"order ({layout.source}) has only {len(layout)} parameters"
                )
            if not isinstance(entry, Mapping):
                raise BundleSchemaError(f"{source}: optimizer state[{index}] is not a dict")
            state[index] = entry
        # Invariant 2: per-parameter tensors must carry the canonical shape.
        for index, entry in state.items():
            shape = layout.shapes[index]
            for key, value in entry.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and tuple(value.shape) != shape
                ):
                    raise LayoutMismatchError(
                        f"{source}: state[{index}][{key!r}] has shape "
                        f"{tuple(value.shape)} but the canonical order says parameter "
                        f"{index} ({layout.names[index]!r}) has shape {shape}"
                    )
        self._state: dict[int, Mapping[str, Any]] = state
        self._param_groups: list[dict[str, Any]] = groups
        self._layout: ParameterLayout = layout
        self._source: str = source

    # ------------------------------------------------------------- accessors

    @property
    def layout(self) -> ParameterLayout:
        """The canonical order this state is aligned to."""
        return self._layout

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """The saved ``param_groups`` (copies; ``params`` indices included)."""
        return [dict(g) for g in self._param_groups]

    @property
    def is_empty(self) -> bool:
        """``True`` when no parameter has any state.

        This is the normal shape of a step-0 (initialization) checkpoint: with
        pre-step semantics, nothing has been updated yet, so
        ``optimizer.state`` is empty and no Adam preconditioner exists.
        """
        return not self._state

    def state_keys(self) -> tuple[str, ...]:
        """Sorted union of the state keys present on any parameter."""
        keys: set[str] = set()
        for entry in self._state.values():
            keys.update(entry)
        return tuple(sorted(keys))

    def has_state(self, key: str) -> bool:
        """Whether *every* parameter has ``key`` (what :meth:`flat_state` needs)."""
        if self.is_empty:
            return False
        return all(key in self._state.get(i, {}) for i in range(len(self._layout)))

    @property
    def hparams(self) -> dict[str, Any]:
        """Hyper-parameters from the saved ``param_groups``.

        These are authoritative: they are what the optimizer ran with (e.g.
        ``betas``, ``eps``, ``weight_decay``, ``lr``, ``momentum``). With
        several param groups only the keys on which all groups *agree* are
        returned; a key the groups disagree on is omitted, so a caller that
        needs it gets a loud ``KeyError`` and can go to :attr:`param_groups`
        instead of silently using group 0's value for the whole model.
        """
        out: dict[str, Any] = {}
        if not self._param_groups:
            return out
        for key, value in self._param_groups[0].items():
            if key == "params":
                continue
            if all(key in g and g[key] == value for g in self._param_groups[1:]):
                out[key] = value
        return out

    @property
    def step_count(self) -> int | None:
        """The saved per-parameter ``step`` counter, or ``None`` if absent.

        With pre-step semantics this is the number of updates already applied
        to ``theta_t`` — i.e. the exponent ``t`` of the bias correction that
        matches the *saved* ``exp_avg_sq`` (P is taken from the saved pre-step
        state). Raises if parameters disagree, which
        would make a single scalar meaningless.
        """
        if not self.has_state("step"):
            return None
        values = self.scalar_state("step")
        distinct = sorted(set(values))
        if len(distinct) != 1:
            raise BundleSchemaError(
                f"{self._source}: parameters have different 'step' counters "
                f"({_abbrev(distinct)}); there is no single step for bias correction"
            )
        return int(distinct[0])

    # ------------------------------------------------------------- flattening

    def flat_state(
        self,
        key: str,
        *,
        dtype: torch.dtype | None = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the state ``key`` as one flat vector in the canonical order.

        ``flat_state("exp_avg_sq")`` is segment-for-segment aligned with the
        gradient vector vatis flattens from ``ModelBundle.params``
        (``parameter_grad_vector``), because both are laid out by the same
        :class:`~vatis.core.param_layout.ParameterLayout`. That is the property
        the preconditioned ruler depends on.

        Args:
            key: e.g. ``"exp_avg_sq"``, ``"exp_avg"``, ``"momentum_buffer"``.
            dtype: dtype of the result; ``fp32`` by default (bundle optimizer
                tensors are already fp32). ``None`` keeps the stored dtype.
            device: device of the result; ``None`` keeps the stored device.

        Raises:
            BundleSchemaError: the key is missing for some parameter, or is not
                a per-element tensor (e.g. ``"step"``, a per-parameter scalar —
                use :meth:`scalar_state` / :attr:`step_count` for those).
            LayoutMismatchError: a stored tensor's shape contradicts the
                canonical order (also checked at construction).
        """
        if self.is_empty:
            raise BundleSchemaError(
                f"{self._source}: the optimizer state is empty, so {key!r} is not "
                "available. With pre-step semantics this is what a step-0 "
                "(initialization) checkpoint looks like."
            )
        tensors: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        for index, name in enumerate(self._layout.names):
            entry = self._state.get(index, {})
            if key not in entry:
                missing.append(name)
                continue
            value = entry[key]
            # A 0-dim tensor is a per-parameter *scalar* (AdamW's ``step``) unless
            # the parameter itself is 0-dim, in which case it is that parameter's
            # one-element state and belongs in the flat vector.
            is_scalar_parameter = self._layout.shape_of(name) == ()
            if not isinstance(value, torch.Tensor) or (value.ndim == 0 and not is_scalar_parameter):
                raise BundleSchemaError(
                    f"{self._source}: state[{index}][{key!r}] is a per-parameter "
                    f"scalar ({value!r}), not a per-element tensor; use "
                    f"scalar_state({key!r}) (or .step_count for 'step') instead"
                )
            tensors[name] = value
        if missing:
            raise BundleSchemaError(
                f"{self._source}: no {key!r} state for {_abbrev(missing)}; "
                f"available keys: {self.state_keys()}"
            )
        return self._layout.flatten_named(tensors, dtype=dtype, device=device)

    def scalar_state(self, key: str) -> list[float]:
        """Return a per-parameter scalar state (e.g. ``"step"``), canonical order.

        The list is indexed like :attr:`ParameterLayout.names`.
        """
        out: list[float] = []
        missing: list[str] = []
        for index, name in enumerate(self._layout.names):
            entry = self._state.get(index, {})
            if key not in entry:
                missing.append(name)
                continue
            value = entry[key]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise BundleSchemaError(
                        f"{self._source}: state[{index}][{key!r}] has "
                        f"{value.numel()} elements, not a scalar; use "
                        f"flat_state({key!r})"
                    )
                out.append(float(value.reshape(()).item()))
            elif isinstance(value, (int, float)):
                out.append(float(value))
            else:
                raise BundleSchemaError(
                    f"{self._source}: state[{index}][{key!r}] is not a scalar: {value!r}"
                )
        if missing:
            raise BundleSchemaError(
                f"{self._source}: no {key!r} state for {_abbrev(missing)}; "
                f"available keys: {self.state_keys()}"
            )
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OptimizerStateView(n_params={len(self._layout)}, "
            f"keys={self.state_keys()}, source={self._source!r})"
        )


# ------------------------------------------------------------------ checkpoint


def plain_lm_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    """Forward for a plain (non-HF) causal LM: ``model(input_ids) -> logits``.

    The HF counterpart (:func:`vatis.models.hf.causal_lm_forward`) calls
    ``model(**batch)`` because HF models take keyword arguments and return a
    ``ModelOutput``. A bundle's model comes from the user's factory and is an
    ordinary ``nn.Module`` taking token ids positionally, so we pass
    ``batch["input_ids"]`` (or the tensor itself) and accept either a tensor or
    a ``.logits`` carrier as the result.
    """
    if isinstance(batch, torch.Tensor):
        inputs = batch
    elif isinstance(batch, Mapping):
        if "input_ids" not in batch:
            raise BundleSchemaError(
                f"plain_lm_forward expects an 'input_ids' key, got {sorted(batch)}"
            )
        inputs = batch["input_ids"]
    elif isinstance(batch, Sequence) and batch and isinstance(batch[0], torch.Tensor):
        inputs = batch[0]
    else:
        raise TypeError(f"plain_lm_forward cannot read a batch of type {type(batch).__name__}")
    out = model(inputs)
    if isinstance(out, torch.Tensor):
        return out
    logits = getattr(out, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError(
        f"unexpected forward output type: {type(out).__name__}; expected a Tensor "
        "or an object with .logits"
    )


class LocalBundleCheckpoint:
    """One checkpoint of a bundle — a lazy handle.

    Nothing is read from disk until an attribute needs it; every loaded piece
    is cached until :meth:`release` drops it (after which the next access
    re-reads). Iterating the parent :class:`LocalBundleSource` releases the
    previous checkpoint automatically.
    """

    __slots__ = (
        "_batch_a",
        "_batch_a_tokens",
        "_meta",
        "_model_bundle",
        "_optimizer_state",
        "_source",
        "name",
        "path",
        "step",
    )

    def __init__(self, source: LocalBundleSource, path: Path, step: int) -> None:
        self._source = source
        self.path = path
        self.name = path.name
        self.step = step
        self._meta: CheckpointMeta | None = None
        self._model_bundle: ModelBundle | None = None
        self._optimizer_state: OptimizerStateView | None = None
        self._batch_a_tokens: torch.Tensor | None = None
        self._batch_a: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------- identifiers

    @property
    def revision(self) -> str:
        """Row label for the sink, in the repo's ``"step1000"`` convention.

        Parseable with ``examples/_helpers.py::step_from_revision``; use
        :attr:`name` for the on-disk directory name (``"step_00001000"``).
        """
        return f"step{self.step}"

    @property
    def identifier(self) -> str:
        """``"<run_dir name>@step<step>"`` — mirrors HF's ``"<name>@<revision>"``."""
        return f"{self._source.run_name}@{self.revision}"

    @property
    def is_loaded(self) -> bool:
        """Whether any tensor of this checkpoint is currently held in memory."""
        return (
            self._model_bundle is not None
            or self._optimizer_state is not None
            or self._batch_a_tokens is not None
        )

    # -------------------------------------------------------------- lazy loads

    @property
    def meta(self) -> CheckpointMeta:
        """Parsed ``meta.json`` (step, tokens_seen, lr, train_loss, wall_time)."""
        if self._meta is None:
            raw = _read_json(self.path / CHECKPOINT_META_FILE)
            if not isinstance(raw, Mapping):
                raise BundleSchemaError(f"{self.path / CHECKPOINT_META_FILE}: not a JSON object")
            meta = CheckpointMeta.from_raw(raw, source=str(self.path / CHECKPOINT_META_FILE))
            if meta.step != self.step:
                raise BundleSchemaError(
                    f"{self.path / CHECKPOINT_META_FILE}: meta.json says step "
                    f"{meta.step} but the directory is named {self.name!r}"
                )
            self._meta = meta
        return self._meta

    @property
    def model_bundle(self) -> ModelBundle:
        """The :class:`~vatis.models.hf.ModelBundle` for ``theta_t``.

        Built on first access: the factory instantiates the architecture, the
        bf16 weights of ``model.safetensors`` are upcast to the requested dtype
        (fp32 by default) and loaded, and ``params`` is looked up **by name in
        the canonical order** so vatis's gradient flattening matches
        :meth:`OptimizerStateView.flat_state` segment for segment.
        """
        if self._model_bundle is None:
            self._model_bundle = self._build_model_bundle()
        return self._model_bundle

    @property
    def optimizer_state(self) -> OptimizerStateView:
        """The pre-step optimizer state, aligned to the canonical order."""
        if self._optimizer_state is None:
            path = self.path / CHECKPOINT_OPTIM_FILE
            raw = torch.load(
                path,
                map_location=self._source.device,
                weights_only=True,
            )
            if not isinstance(raw, Mapping):
                raise BundleSchemaError(f"{path}: not an optimizer state_dict")
            self._optimizer_state = OptimizerStateView(raw, self._source.layout, source=str(path))
        return self._optimizer_state

    @property
    def batch_a_tokens(self) -> torch.Tensor:
        """``batch_a.npy`` as an int64 ``(B_train, seq_len)`` tensor.

        The update batch A_t: the batch step ``t`` is about to consume at this
        ``theta_t``. Stored as int32 on disk; returned as int64 because that is
        what ``nn.Embedding`` indexing requires (a lossless widening).
        """
        if self._batch_a_tokens is None:
            import numpy as np

            path = self.path / CHECKPOINT_BATCH_A_FILE
            array = np.load(path, allow_pickle=False)
            if array.ndim != 2:
                raise BundleSchemaError(
                    f"{path}: expected a 2-D [B_train, seq_len] token array, got shape "
                    f"{array.shape}"
                )
            if not np.issubdtype(array.dtype, np.integer):
                raise BundleSchemaError(f"{path}: expected integer token ids, got {array.dtype}")
            self._batch_a_tokens = torch.from_numpy(np.ascontiguousarray(array)).to(
                dtype=torch.int64, device=self._source.device
            )
        return self._batch_a_tokens

    @property
    def batch_a(self) -> dict[str, torch.Tensor]:
        """``batch_a.npy`` as a vatis LM batch dict.

        ``{"input_ids": ids, "labels": next-token targets}`` with the repo's
        pre-shifted convention (``labels[:, :-1] = input_ids[:, 1:]``,
        ``labels[:, -1] = -100``), i.e. exactly what
        :func:`vatis.models.hf.causal_lm_loss` and
        :func:`vatis.models.hf.causal_lm_valid_mask` expect.

        Two deliberate choices, both conservative:

        - no ``attention_mask``: the schema stores token ids only, so vatis has
          no padding information for the update batch and counts every position
          (the project's training criterion does the same);
        - the shift is applied *here* rather than assumed to be baked in: the
          schema says ``batch_a.npy`` holds the *token ids of the batch*, so if
          the writer stores the model inputs, the last position has no target
          and is ignored.

        Use :attr:`batch_a_tokens` if you need the raw ids.
        """
        if self._batch_a is None:
            from vatis.data.collate import shifted_lm_targets

            ids = self.batch_a_tokens
            self._batch_a = {"input_ids": ids, "labels": shifted_lm_targets(ids)}
        return self._batch_a

    # ----------------------------------------------------------------- release

    def release_optimizer_state(self) -> None:
        """Drop the loaded ``optim.pt`` tensors; the next access reloads them."""
        self._optimizer_state = None

    def release(self) -> None:
        """Drop every loaded tensor of this checkpoint.

        Makes the laziness contract enforceable: ``LocalBundleSource.__iter__``
        calls this on checkpoint ``i - 1`` before yielding checkpoint ``i``, so
        at most one checkpoint's tensors are reachable from the source at a
        time. Anything a caller still holds (e.g. a ``ModelBundle`` it kept)
        stays alive — Python frees memory by reference counting, not by decree.
        Cheap metadata (``meta.json``) is kept; the next attribute access
        re-reads whatever was dropped.
        """
        self._model_bundle = None
        self._optimizer_state = None
        self._batch_a_tokens = None
        self._batch_a = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LocalBundleCheckpoint(step={self.step}, path={str(self.path)!r}, "
            f"loaded={self.is_loaded})"
        )

    # ------------------------------------------------------------- internals

    def _build_model_bundle(self) -> ModelBundle:
        source = self._source
        model = source.build_model()
        params = source.layout.parameters_in_order(
            model, allow_unlisted=source.allow_unlisted_params
        )
        _load_weights(
            self.path / CHECKPOINT_WEIGHTS_FILE,
            model,
            layout=source.layout,
            dtype=source.torch_dtype,
        )
        model.eval()
        for p in params:
            p.requires_grad_(True)
        return ModelBundle(
            model=model,
            params=params,
            forward_fn=source.forward_fn,
            loss_fn=source.loss_fn,
            valid_mask_fn=source.valid_mask_fn,
            identifier=self.identifier,
        )


# ---------------------------------------------------------------------- source


class LocalBundleSource:
    """Iterate the checkpoints of one bundle ``run_dir`` as a vatis source.

    Constructing this reads only ``metadata.json`` and ``param_names.json`` and
    stats the checkpoint directories. Use :meth:`model_specs` to feed
    :meth:`vatis.analyzer.Analyzer.run` directly, or iterate the source when
    you also need the optimizer state / update batch per checkpoint::

        source = load_local_bundle(run_dir, model_factory="my_pkg.models:build")
        analyzer.run(source.model_specs())                 # HF-style loop

        for ckpt in source:                               # full access
            v = ckpt.optimizer_state.flat_state("exp_avg_sq")
            ...
    """

    def __init__(
        self,
        run_dir: str | Path,
        model_factory: ModelFactorySpec,
        *,
        dtype: str = "fp32",
        device: torch.device | str = "cpu",
        steps: Sequence[int] | None = None,
        forward_fn: ForwardFn | None = None,
        loss_fn: LossFn | None = None,
        valid_mask_fn: ValidMaskFn | None = None,
        allow_unlisted_params: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise BundleSchemaError(f"{self.run_dir}: not a directory")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        self.torch_dtype = _DTYPES[dtype]
        self.dtype = dtype
        self.device = torch.device(device)
        self.allow_unlisted_params = allow_unlisted_params
        self.forward_fn: ForwardFn = forward_fn if forward_fn is not None else plain_lm_forward
        self.loss_fn: LossFn = loss_fn if loss_fn is not None else causal_lm_loss
        self.valid_mask_fn: ValidMaskFn = (
            valid_mask_fn if valid_mask_fn is not None else causal_lm_valid_mask
        )
        self.model_factory: ModelFactory = resolve_model_factory(model_factory)

        raw_metadata = _read_json(self.run_dir / BUNDLE_METADATA_FILE)
        if not isinstance(raw_metadata, Mapping):
            raise BundleSchemaError(f"{self.run_dir / BUNDLE_METADATA_FILE}: not a JSON object")
        self.metadata = BundleMetadata(raw=raw_metadata, path=self.run_dir / BUNDLE_METADATA_FILE)
        self.layout = ParameterLayout.from_json(self.run_dir / BUNDLE_PARAM_NAMES_FILE)
        self.checkpoints: tuple[LocalBundleCheckpoint, ...] = self._discover(steps)

    # ------------------------------------------------------------- discovery

    @property
    def run_name(self) -> str:
        """The run directory's name — the ``"model"`` half of the identifier."""
        return self.run_dir.name

    @property
    def steps(self) -> tuple[int, ...]:
        """Steps of the selected checkpoints, ascending."""
        return tuple(c.step for c in self.checkpoints)

    def _discover(self, steps: Sequence[int] | None) -> tuple[LocalBundleCheckpoint, ...]:
        root = self.run_dir / BUNDLE_CHECKPOINTS_DIR
        if not root.is_dir():
            raise BundleSchemaError(f"{root}: no checkpoints/ directory in the bundle")
        found: dict[int, Path] = {}
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            match = _STEP_DIR_RE.match(child.name)
            if match is None:
                continue
            step = int(match.group(1))
            if step in found:
                raise BundleSchemaError(
                    f"{root}: two checkpoint directories for step {step} "
                    f"({found[step].name!r} and {child.name!r})"
                )
            found[step] = child
        if not found:
            raise BundleSchemaError(f"{root}: contains no step_<n> checkpoint directories")
        if steps is None:
            selected = sorted(found)
        else:
            selected = sorted(dict.fromkeys(int(s) for s in steps))
            unknown = [s for s in selected if s not in found]
            if unknown:
                raise BundleSchemaError(
                    f"{root}: requested step(s) {unknown} not in the bundle "
                    f"(available: {_abbrev(sorted(found))})"
                )
        checkpoints: list[LocalBundleCheckpoint] = []
        for step in selected:
            path = found[step]
            missing = [
                f
                for f in (
                    CHECKPOINT_WEIGHTS_FILE,
                    CHECKPOINT_OPTIM_FILE,
                    CHECKPOINT_BATCH_A_FILE,
                    CHECKPOINT_META_FILE,
                )
                if not (path / f).is_file()
            ]
            if missing:
                raise BundleSchemaError(f"{path}: checkpoint is missing {missing}")
            checkpoints.append(LocalBundleCheckpoint(self, path, step))
        return tuple(checkpoints)

    # ------------------------------------------------------------- iteration

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __iter__(self) -> Iterator[LocalBundleCheckpoint]:
        """Yield checkpoints in ascending step order, one loaded at a time.

        Before yielding checkpoint ``i`` the previous one is
        :meth:`~LocalBundleCheckpoint.release`\\ d, so the source never holds
        more than one checkpoint's tensors. Use :attr:`checkpoints` for random
        access without that behaviour.
        """
        previous: LocalBundleCheckpoint | None = None
        for checkpoint in self.checkpoints:
            if previous is not None:
                previous.release()
            previous = checkpoint
            yield checkpoint

    def model_specs(self) -> Iterator[tuple[ModelBundle, str]]:
        """Yield ``(ModelBundle, revision)`` pairs for :meth:`Analyzer.run`.

        The HF path passes ``(name, revision)`` strings and lets the analyzer
        load; a bundle needs the user's factory, so we pass the already-built
        bundle and use ``revision`` as the row label. Being a generator keeps
        the loading lazy: checkpoint ``i`` is read when the analyzer asks for
        it, and checkpoint ``i - 1`` is released at that moment.

        One honest caveat: ``Analyzer.run``'s loop variable still references the
        previous ``ModelBundle`` while the generator builds the next one, so the
        *weights* of two checkpoints coexist for that moment (the optimizer
        state and update batch of the previous one are already released).
        Memory stays O(1) in the number of checkpoints either way; iterate the
        source directly if you need the strict one-at-a-time bound.
        """
        for checkpoint in self:
            yield checkpoint.model_bundle, checkpoint.revision

    # ---------------------------------------------------------------- lookups

    def checkpoint_for(self, key: str | int) -> LocalBundleCheckpoint:
        """Find a checkpoint by step, identifier, revision or directory name."""
        if isinstance(key, int):
            for checkpoint in self.checkpoints:
                if checkpoint.step == key:
                    return checkpoint
            raise BundleSchemaError(f"no checkpoint for step {key} (have {_abbrev(self.steps)})")
        for checkpoint in self.checkpoints:
            if key in (checkpoint.identifier, checkpoint.revision, checkpoint.name):
                return checkpoint
        raise BundleSchemaError(
            f"no checkpoint matching {key!r} (identifiers: "
            f"{_abbrev([c.identifier for c in self.checkpoints])})"
        )

    def batch_a_spec(self) -> EvalBatchSpec:
        """The update batch A_t as a per-checkpoint :class:`EvalBatchSpec`.

        ``EvalBatchSpec.from_callable`` resolves the batch per checkpoint id,
        which is exactly the ``"train_at_step"`` pattern from vatis's public
        API. Pass it in ``eval_batches`` to measure at the batch the step was
        about to consume::

            Analyzer(eval_batches={"batch_a": source.batch_a_spec()}).run(
                source.model_specs()
            )
        """
        return EvalBatchSpec.from_callable(lambda ckpt_id: self.checkpoint_for(ckpt_id).batch_a)

    def adamw_preconditioner(
        self,
        *,
        beta2: float | None = None,
        eps: float | None = None,
        descriptor: str = "adamw",
    ) -> Callable[[str], Preconditioner | None]:
        """The ``adamw`` P source, as a per-checkpoint factory for the analyzer.

        Pass the result as ``Analyzer(preconditioner=...)`` /
        ``analyze(preconditioner=...)``: it is called once per checkpoint with
        that checkpoint's id and builds ``P = diag(1 / (sqrt(v_hat) + eps))``
        from *that* checkpoint's saved (pre-step) optimizer state, with
        ``v_hat = exp_avg_sq / (1 - beta2**step_count)``. The flat ``exp_avg_sq``
        comes from :meth:`OptimizerStateView.flat_state`, so it is
        segment-for-segment aligned with the gradient vector the analyzer builds
        from ``ModelBundle.params`` — both go through this source's one
        :class:`~vatis.core.param_layout.ParameterLayout`.

        ``beta2`` / ``eps`` default to the saved ``param_groups`` (what the
        optimizer actually ran with), cross-checked against
        ``metadata.json``'s ``optimizer_hparams`` — a disagreement is warned
        about, not silently resolved. Pass them explicitly to override.

        **Step-0 / empty-optimizer-state policy: the P rows are SKIPPED for that
        checkpoint** (a ``UserWarning`` says so, naming the checkpoint), and only
        the raw rows are emitted. A step-0 checkpoint legitimately has no
        optimizer state, and neither available fallback is honest:
        ``P = 1/(sqrt(0) + eps) = 1/eps`` is a garbage preconditioner (it scales
        every observable by ``1/eps^2``, e.g. ``1e16``), and ``P = I`` would
        publish ``chi_pos_P == chi_pos`` under the ``"adamw"`` descriptor — a
        plausible-looking "the preconditioner does nothing yet" reading that is
        an artifact of the fallback, not a measurement. An absent row cannot be
        mistaken for a measured one.

        Returns:
            ``fn(checkpoint_id) -> Preconditioner | None``.
        """

        def build(checkpoint_id: str) -> Preconditioner | None:
            checkpoint = self.checkpoint_for(checkpoint_id)
            state = checkpoint.optimizer_state
            if state.is_empty:
                warnings.warn(
                    f"{checkpoint.identifier}: the saved optimizer state is empty "
                    f"({checkpoint.path / CHECKPOINT_OPTIM_FILE}), which is what a step-0 "
                    "checkpoint looks like under pre-step semantics. No AdamW "
                    "preconditioner exists there, so the delta_loss_P / chi_net_P / "
                    "chi_pos_P rows are SKIPPED for this checkpoint (the raw rows are "
                    "unaffected). P = 1/(sqrt(0) + eps) = 1/eps would be a garbage "
                    "preconditioner, and P = I would misreport the identity as 'adamw'.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            if state.step_count is not None and state.step_count != checkpoint.step:
                warnings.warn(
                    f"{checkpoint.identifier}: the saved optimizer state has applied "
                    f"{state.step_count} update(s), but the checkpoint directory says step "
                    f"{checkpoint.step}. Under pre-step semantics the two are the same "
                    "number; a mismatch means the optimizer state and the weights may not "
                    "belong together. P is built from the saved state's own counter.",
                    UserWarning,
                    stacklevel=2,
                )
            precond = adamw_preconditioner_from_state(
                state,
                metadata=self.metadata,
                beta2=beta2,
                eps=eps,
                descriptor=descriptor,
                context=checkpoint.identifier,
            )
            # P is all the analyzer needs from optim.pt. Dropping the loaded state
            # here keeps two parameter-sized vectors (exp_avg, exp_avg_sq) from
            # staying resident for the whole checkpoint; a later access reloads it.
            checkpoint.release_optimizer_state()
            return precond

        return build

    def build_model(self) -> nn.Module:
        """Instantiate a fresh, un-initialized-by-us model via the factory."""
        model = self.model_factory(self.metadata)
        if not isinstance(model, nn.Module):
            raise TypeError(f"model_factory returned {type(model).__name__}, expected an nn.Module")
        model.to(device=self.device)
        # Cast only floating-point *parameters*: ``Module.to(dtype)`` would also
        # hit complex buffers (e.g. precomputed rotary freqs_cis), where casting
        # to a real dtype silently discards the imaginary part.
        for p in model.parameters():
            if p.is_floating_point() and p.dtype != self.torch_dtype:
                p.data = p.data.to(dtype=self.torch_dtype)
        return model

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LocalBundleSource(run_dir={str(self.run_dir)!r}, "
            f"n_checkpoints={len(self.checkpoints)}, steps={_abbrev(self.steps)})"
        )


def load_local_bundle(
    run_dir: str | Path,
    model_factory: ModelFactorySpec,
    *,
    dtype: str = "fp32",
    device: torch.device | str = "cpu",
    steps: Sequence[int] | None = None,
    forward_fn: ForwardFn | None = None,
    loss_fn: LossFn | None = None,
    valid_mask_fn: ValidMaskFn | None = None,
    allow_unlisted_params: bool = False,
) -> LocalBundleSource:
    """Open a checkpoint bundle as a vatis source (nothing heavy is read yet).

    The bundle counterpart of :func:`vatis.models.hf.load_hf_model`.

    Args:
        run_dir: the bundle root (contains ``metadata.json``,
            ``param_names.json``, ``checkpoints/``).
        model_factory: ``model_factory(metadata) -> nn.Module``, or a
            ``"module:function"`` import string. Receives the
            :class:`BundleMetadata` (which also behaves like the raw dict), so
            the architecture is described by the bundle, not by vatis. This is
            how vatis stays project-agnostic: the factory for a specific
            training stack lives in that project (or in tests), never here.
        dtype: dtype the weights are loaded *as*. Default ``"fp32"``: bundle
            weights are bf16 on disk and the ruler math wants fp32, so the
            upcast happens on load (lossless).
        device: device for the model, optimizer state and batches.
        steps: optional subset of steps to expose, in any order (the source
            always iterates ascending). Unknown steps raise.
        forward_fn / loss_fn / valid_mask_fn: override the defaults
            (:func:`plain_lm_forward`, :func:`vatis.models.hf.causal_lm_loss`,
            :func:`vatis.models.hf.causal_lm_valid_mask`) for non-causal-LM
            models.
        allow_unlisted_params: accept a model with parameters that
            ``param_names.json`` does not list. Off by default — such
            parameters are silently absent from every parameter-space vector.

    Returns:
        A :class:`LocalBundleSource`.
    """
    return LocalBundleSource(
        run_dir,
        model_factory,
        dtype=dtype,
        device=device,
        steps=steps,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        valid_mask_fn=valid_mask_fn,
        allow_unlisted_params=allow_unlisted_params,
    )


def adamw_preconditioner_from_state(
    state: OptimizerStateView,
    *,
    metadata: BundleMetadata | None = None,
    beta2: float | None = None,
    eps: float | None = None,
    descriptor: str = "adamw",
    state_key: str = "exp_avg_sq",
    context: str = "optimizer state",
) -> Preconditioner:
    """Build the AdamW ``P`` from one saved (pre-step) optimizer state.

    The plumbing half of the ``adamw`` P source; the math is
    :meth:`vatis.core.preconditioner.Preconditioner.adamw` (which documents the
    deliberate exclusion of momentum and decoupled weight decay — ``P`` is the
    geometry factor only).

    Where the numbers come from:

    - ``exp_avg_sq``: :meth:`OptimizerStateView.flat_state`, i.e. the canonical
      flat order;
    - ``step``: :attr:`OptimizerStateView.step_count`. Under pre-step semantics
      that is the number of updates already folded into the saved
      ``exp_avg_sq``, hence the matching bias-correction exponent;
    - ``beta2`` / ``eps``: the saved ``param_groups``
      (:attr:`OptimizerStateView.hparams`) — authoritative, because that is what
      the optimizer ran with. ``metadata.json`` is used as a *cross-check* only:
      a disagreement raises a ``UserWarning`` naming both values and the saved
      state wins. Explicit ``beta2`` / ``eps`` arguments override everything.

    Raises:
        BundleSchemaError: the state is empty, has no ``exp_avg_sq`` (e.g. an
            SGD run — its ruler is the raw one, ``preconditioner=None``), or has
            no ``step`` counter to bias-correct with.
        ValueError: ``betas``/``eps`` missing from the saved ``param_groups``, or
            ``amsgrad=True`` (which preconditions with ``max_exp_avg_sq``).
    """
    if state.is_empty:
        raise BundleSchemaError(
            f"{context}: the optimizer state is empty, so no AdamW preconditioner exists. "
            "Under pre-step semantics this is a step-0 checkpoint; the analyzer's "
            "adamw_preconditioner() skips the P rows for it."
        )
    optimizer_name = None if metadata is None else metadata.optimizer_name
    if optimizer_name is not None and optimizer_name.lower() not in _ADAM_FAMILY:
        raise BundleSchemaError(
            f"{context}: metadata.json names the optimizer {optimizer_name!r}. "
            f"P = 1/(sqrt(v_hat) + eps) is the preconditioner of {sorted(_ADAM_FAMILY)} "
            "only; other optimizers that happen to keep an 'exp_avg_sq' (RAdam, NAdam, "
            "...) apply a different update, and labelling their rows 'adamw' would be "
            "wrong. Build the P you mean with Preconditioner.from_diagonal(...)."
        )
    if not state.has_state(state_key):
        if state_key not in state.state_keys():
            raise BundleSchemaError(
                f"{context}: the saved optimizer state has no {state_key!r} at all "
                f"(present keys: {state.state_keys()}). An AdamW preconditioner needs the "
                "second moment; a plain-SGD run has none and its ruler is the raw one "
                "(preconditioner=None)."
            )
        raise BundleSchemaError(
            f"{context}: the saved optimizer state has {state_key!r} for some parameters "
            "but not for all. torch keeps no state for a parameter that never received a "
            "gradient, so at least one parameter in param_names.json was never updated. "
            "No P is defined for it under the adamw convention; exclude it from the "
            "bundle's parameter list or build P explicitly."
        )
    step = state.step_count
    if step is None:
        raise BundleSchemaError(
            f"{context}: the saved optimizer state has no 'step' counter, so "
            "exp_avg_sq cannot be bias-corrected. Refusing to guess the exponent."
        )
    disagreeing = [
        key
        for key in ("betas", "eps")
        if len({repr(group.get(key)) for group in state.param_groups}) > 1
    ]
    if disagreeing and (beta2 is None or eps is None):
        raise ValueError(
            f"{context}: the saved param_groups disagree on {disagreeing}. One diagonal "
            "P with a single beta2 / eps cannot represent per-group values; pass beta2= "
            "and eps= explicitly if one pair is meant for all parameters, or build P "
            "per group with Preconditioner.from_diagonal(...)."
        )
    if beta2 is not None and eps is not None:
        # Both given explicitly: nothing is read from the saved groups, which may
        # legitimately not agree on a single pair.
        saved_beta2, saved_eps = float(beta2), float(eps)
    else:
        saved_beta2, saved_eps = resolve_adamw_hparams(
            state.hparams, source=f"{context}: saved param_groups"
        )
        if metadata is not None:
            _cross_check_adamw_hparams(metadata, saved_beta2, saved_eps, context=context)
    # An override changes P, so it has to be visible in the descriptor: two runs
    # that differ only in an override must not both be written as "adamw".
    overrides = [
        f"{name}={value:g}" for name, value in (("beta2", beta2), ("eps", eps)) if value is not None
    ]
    if overrides and descriptor == "adamw":
        descriptor = f"adamw({','.join(overrides)})"
    return Preconditioner.adamw(
        state.flat_state(state_key),  # a fresh concatenation, safe to overwrite
        step=step,
        beta2=saved_beta2 if beta2 is None else float(beta2),
        eps=saved_eps if eps is None else float(eps),
        descriptor=descriptor,
        inplace=True,
    )


def _cross_check_adamw_hparams(
    metadata: BundleMetadata,
    beta2: float,
    eps: float,
    *,
    context: str,
) -> None:
    """Warn when ``metadata.json`` disagrees with the saved ``param_groups``.

    The saved state wins (it is what ran); the warning exists because a
    disagreement usually means the bundle's metadata was written from a config
    the run did not actually use — worth knowing before it shows up as a
    surprising ``chi_pos_P`` trajectory.
    """
    hparams = metadata.optimizer_hparams
    if hparams is None:
        return
    try:
        meta_beta2, meta_eps = resolve_adamw_hparams(hparams, source=f"{context}: metadata.json")
    except (ValueError, TypeError):
        # metadata.json is informational; an unusable entry is not a reason to
        # refuse a P that the authoritative param_groups fully specify.
        return
    mismatches = [
        f"{name}: metadata.json={meta_value!r} vs saved param_groups={saved_value!r}"
        for name, meta_value, saved_value in (
            ("beta2", meta_beta2, beta2),
            ("eps", meta_eps, eps),
        )
        if meta_value != saved_value
    ]
    if mismatches:
        warnings.warn(
            f"{context}: AdamW hyper-parameters disagree ({'; '.join(mismatches)}). "
            "Using the saved param_groups — they are what the optimizer actually ran "
            "with — but the bundle's metadata.json is out of sync with its own run.",
            UserWarning,
            stacklevel=3,
        )


def resolve_model_factory(spec: ModelFactorySpec) -> ModelFactory:
    """Turn a factory spec into a callable.

    Accepts a callable as-is, or a ``"module:function"`` /
    ``"module:Class.attr"`` import string (the same shape gunicorn / uvicorn
    use). Import strings are what makes a factory configurable from a config
    file, an env var, or a CLI flag without vatis importing the project.
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"model_factory must be a callable or a 'module:function' string, got "
            f"{type(spec).__name__}"
        )
    if ":" not in spec:
        raise ValueError(
            f"model_factory import string {spec!r} must be 'module:function' "
            "(e.g. 'my_pkg.models:build_model')"
        )
    module_name, _, attr_path = spec.partition(":")
    if not module_name or not attr_path:
        raise ValueError(f"model_factory import string {spec!r} must be 'module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"model_factory {spec!r}: cannot import module {module_name!r}") from exc
    target: Any = module
    for part in attr_path.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise AttributeError(
                f"model_factory {spec!r}: {module_name!r} has no attribute {attr_path!r}"
            ) from exc
    if not callable(target):
        raise TypeError(f"model_factory {spec!r} resolved to a non-callable: {target!r}")
    return target  # type: ignore[no-any-return]


# -------------------------------------------------------------------- helpers


def _load_weights(
    path: Path,
    model: nn.Module,
    *,
    layout: ParameterLayout,
    dtype: torch.dtype,
) -> None:
    """Load ``model.safetensors`` into ``model``, upcasting bf16 -> ``dtype``.

    Checks, in order:

    - every canonical parameter name is in the file (else the bundle and the
      factory disagree, or ``param_names.json`` is stale);
    - every *model parameter* is in the file, so ``theta`` is fully specified by
      the checkpoint and no weight silently keeps its factory initialization
      (this also covers parameters outside the canonical order when
      ``allow_unlisted_params`` was set);
    - no file key is unknown to the model.

    Missing *buffers* are tolerated: non-persistent buffers (e.g. precomputed
    rotary tables) are recomputed by the factory and are not in the file.
    """
    from safetensors.torch import load_file

    file_state: dict[str, torch.Tensor] = load_file(str(path), device="cpu")
    model_state = model.state_dict()
    model_param_names = set(dict(model.named_parameters()))

    missing_canonical = [name for name in layout.names if name not in file_state]
    if missing_canonical:
        raise LayoutMismatchError(
            f"{path}: no weights for canonical parameter(s) "
            f"{_abbrev(missing_canonical)}; the bundle and the model factory disagree"
        )
    missing_params = sorted(name for name in model_param_names if name not in file_state)
    if missing_params:
        raise BundleSchemaError(
            f"{path}: no weights for model parameter(s) {_abbrev(missing_params)}; "
            "theta would be left at the factory's initialization"
        )
    unexpected = sorted(key for key in file_state if key not in model_state)
    if unexpected:
        raise BundleSchemaError(
            f"{path}: contains tensor(s) {_abbrev(unexpected)} that the model does "
            "not have; the model factory does not match this bundle"
        )

    # The schema stores parameters. A *persistent* buffer (one that is part of
    # state_dict()) that is not in the file keeps whatever the factory gave it:
    # harmless for a constant (a causal mask, position ids), wrong for anything a
    # training run changes (running statistics, a learned scale kept as a buffer).
    # vatis cannot tell the two apart, so it says so instead of staying silent.
    stale_buffers = sorted(
        name for name, _ in model.named_buffers() if name in model_state and name not in file_state
    )
    if stale_buffers:
        warnings.warn(
            f"{path}: persistent buffer(s) {_abbrev(stale_buffers)} are not in the file and "
            "keep the model factory's values. That is correct only if they are constants; "
            "a buffer the training run modified is NOT restored.",
            UserWarning,
            stacklevel=2,
        )

    to_load = {
        key: (value.to(dtype=dtype) if value.is_floating_point() else value)
        for key, value in file_state.items()
    }
    # strict=False: non-persistent buffers are not in state_dict() at all and
    # persistent ones were reported above. Every parameter is checked above, and
    # load_state_dict still raises on a shape mismatch.
    model.load_state_dict(to_load, strict=False)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise BundleSchemaError(f"{path}: missing (required by the bundle schema)")
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise BundleSchemaError(f"{path}: invalid JSON ({exc})") from exc


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _abbrev(items: Sequence[Any], limit: int = 8) -> str:
    shown = list(items[:limit])
    suffix = f", ... (+{len(items) - limit})" if len(items) > limit else ""
    return f"{shown}{suffix}"
