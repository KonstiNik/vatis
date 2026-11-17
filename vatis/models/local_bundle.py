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

import dataclasses
import importlib
import json
import math
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
from vatis.core.update_response import (
    ADAM_FAMILY,
    SGD_FAMILY,
    AdamHyper,
    OptimizerUpdate,
    SGDHyper,
)
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
# Hyper-parameter keys that identify a ``torch.optim.Adam`` / ``AdamW`` param
# group. ``betas`` alone does not: RAdam, NAdam, Adamax and SparseAdam groups
# carry it too, and their updates are not ``m_hat / (sqrt(v_hat) + eps)``.
# ``amsgrad`` is the key only Adam and AdamW groups have.
_ADAM_GROUP_KEYS = frozenset({"betas", "amsgrad"})
# Hyper-parameter keys every ``torch.optim.SGD`` param group carries and no
# Adam-family or RMSprop group does (RMSprop has ``momentum`` but neither
# ``dampening`` nor ``nesterov``).
_SGD_GROUP_KEYS = frozenset({"momentum", "dampening", "nesterov"})
# Descriptor of the rows measured with SGD's geometry factor, ``P = I``.
SGD_DESCRIPTOR = "sgd"
# Appended to the pad-key refusals: what to do about it.
_PAD_KEYS_REMEDY = (
    " The bundle's data are untouched: correct the two keys in metadata.json to what the "
    "run did; if that is not known, each checkpoint's recorded train_loss decides it -- "
    "see docs/SPEC.md, 'Native checkpoint bundles', 'Repairing contradictory pad fields'."
)
# What a P source does when a checkpoint's directory step differs from the step
# counter of its saved optimizer state.
_STEP_MISMATCH_POLICIES = ("raise", "warn")
CHECKPOINT_BATCH_A_FILE = "batch_a.npy"
CHECKPOINT_META_FILE = "meta.json"

# ``update_response(clip_grad_norm=...)``: read the clipping from metadata.json.
CLIP_FROM_METADATA = "metadata"
# safetensors dtype tags of weight files that hold ``theta_t`` itself when the
# run trained in fp32 or fp64. Anything narrower (BF16, F16) holds a rounded
# ``theta_t``, and the difference of two such files is not the step.
_EXACT_WEIGHT_DTYPES = frozenset({"F32", "F64"})
# ... and the loader dtypes that keep such a file's values as they are.
_EXACT_LOADER_DTYPES = frozenset({torch.float32, torch.float64})
# Relative tolerance of the cross-check between a checkpoint's ``meta.json``
# ``lr`` and the learning rate saved in its optimizer state.
_LR_RTOL = 1e-6


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

        If ``optimizer`` is a dict, its ``name``/``type`` entry is used. When
        none of these yields a name, ``optimizer_class`` is read (the class
        name a writer records, e.g. ``"AdamW"``; a dotted path such as
        ``"torch.optim.adamw.AdamW"`` is reduced to its last component).
        """
        value = self._first("optimizer_name", "optimizer_type", "optimizer")
        if isinstance(value, Mapping):
            value = value.get("name", value.get("type"))
        if value is not None:
            return str(value)
        recorded_class = self.raw.get("optimizer_class")
        if recorded_class is None:
            return None
        return str(recorded_class).rpartition(".")[2]

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

    @property
    def loss_ignore_index(self) -> int | None:
        """The target id the *training* loss ignored, or ``None`` if it ignored none.

        Read from the two optional run-level keys a training run records about
        its pad convention:

        - ``mask_pad_in_loss`` (bool) — whether the training cross entropy was
          given an ``ignore_index``;
        - ``loss_ignore_index`` (int, or null when unmasked) — that index (the
          tokenizer's pad id).

        ``None`` means every target counted: both keys absent (a bundle written
        before they existed), or ``mask_pad_in_loss`` false/absent with
        ``loss_ignore_index`` null/absent. An integer means the training loss was
        a mean over the targets different from it, and
        :attr:`LocalBundleCheckpoint.batch_a` ignores the same targets.

        The two keys have to agree. ``mask_pad_in_loss: true`` without an index,
        an index while ``mask_pad_in_loss`` is false or absent, or a value of the
        wrong JSON type raise :class:`BundleSchemaError`: either reading of such
        metadata would put a number on a loss that may not be the one the run
        descended, and the two keys alone do not say which. So does an index
        that cannot be a target — negative, or not below the ``vocab_size`` of the
        metadata's model config when one is stated — because it would be
        accepted and then mask nothing. The remedy is to
        correct ``metadata.json``; ``docs/SPEC.md`` ("Repairing contradictory pad
        fields") describes how each checkpoint's recorded ``train_loss`` settles
        which entry is wrong.
        """
        where = f"{self.path}: " if self.path is not None else ""
        masked = self.raw.get("mask_pad_in_loss")
        index = self.raw.get("loss_ignore_index")

        def spelled(key: str) -> str:
            return "null" if key in self.raw else "absent"

        if masked is not None and not isinstance(masked, bool):
            raise BundleSchemaError(
                f"{where}'mask_pad_in_loss' must be true, false or absent, got {masked!r}"
            )
        if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
            raise BundleSchemaError(
                f"{where}'loss_ignore_index' must be an integer token id, null or absent, "
                f"got {index!r}"
            )
        if masked and index is None:
            raise BundleSchemaError(
                f"{where}'mask_pad_in_loss' is true but 'loss_ignore_index' is "
                f"{spelled('loss_ignore_index')}: the run says its "
                "loss ignored pad targets but not which token id that was, so the loss it "
                "descended cannot be reconstructed for the update batch. Record the pad id "
                "as 'loss_ignore_index'." + _PAD_KEYS_REMEDY
            )
        if index is not None and not masked:
            raise BundleSchemaError(
                f"{where}'loss_ignore_index' is {index} but 'mask_pad_in_loss' is "
                f"{'false' if masked is False else spelled('mask_pad_in_loss')}: the two keys "
                "contradict each "
                "other about whether the training loss ignored that target. Record "
                "'mask_pad_in_loss': true, or 'loss_ignore_index': null." + _PAD_KEYS_REMEDY
            )
        if index is not None:
            # An index no target can equal would be accepted and then mask
            # nothing: the rows would silently be those of the unmasked loss
            # under metadata that says "masked".
            config = self.model_config
            vocab_size = _opt_int(config.get("vocab_size")) if isinstance(config, Mapping) else None
            if index < 0:
                raise BundleSchemaError(
                    f"{where}'loss_ignore_index' is {index}: a token id cannot be negative, so "
                    "no target would be ignored although 'mask_pad_in_loss' is true. Record "
                    "the pad token id the training loss ignored." + _PAD_KEYS_REMEDY
                )
            if vocab_size is not None and index >= vocab_size:
                raise BundleSchemaError(
                    f"{where}'loss_ignore_index' is {index} but the model config states "
                    f"vocab_size = {vocab_size}: no target can equal it, so no target would "
                    "be ignored although 'mask_pad_in_loss' is true. Record the pad token "
                    "id the training loss ignored." + _PAD_KEYS_REMEDY
                )
        return index

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
        the preconditioned observables depend on.

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

    def parameter_states(self, key: str) -> list[torch.Tensor | None]:
        """The state ``key`` per parameter, canonical order, without copying.

        Entry ``i`` is the stored tensor of parameter ``layout.names[i]`` (its
        own dtype and device), or ``None`` when that parameter has no such
        state — no entry at all, or an explicit ``None`` (``torch.optim.SGD``
        stores ``momentum_buffer: None`` for a group without momentum).
        :meth:`flat_state` concatenates the same tensors into one vector and
        refuses a missing one; this accessor is for a consumer that walks the
        parameters one by one and has its own rule for a missing entry.
        """
        out: list[torch.Tensor | None] = []
        for index, name in enumerate(self._layout.names):
            value = self._state.get(index, {}).get(key)
            if value is None:
                out.append(None)
                continue
            if not isinstance(value, torch.Tensor) or (
                value.ndim == 0 and self._layout.shape_of(name) != ()
            ):
                raise BundleSchemaError(
                    f"{self._source}: state[{index}][{key!r}] is not a per-element tensor "
                    f"({value!r})"
                )
            out.append(value)
        return out

    def group_of_parameter(self) -> list[dict[str, Any]]:
        """The saved param group of each parameter, canonical order.

        Entry ``i`` is (a copy of) the ``param_groups`` entry whose ``params``
        list contains index ``i``; construction already checked that the lists
        cover ``0 .. n-1`` exactly once.
        """
        out: list[dict[str, Any]] = []
        for group in self._param_groups:
            out.extend(dict(group) for _ in group["params"])
        return out

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
        """The pre-step optimizer state, aligned to the canonical order.

        Loaded on the **CPU** whatever the source's ``device`` is: the state is
        two parameter-sized vectors (``exp_avg``, ``exp_avg_sq``) that only
        serve to build ``P``, and on a GPU source they would sit next to the
        model for nothing. :meth:`OptimizerStateView.flat_state` takes a
        ``device=`` for a caller that wants a vector elsewhere.
        """
        if self._optimizer_state is None:
            path = self.path / CHECKPOINT_OPTIM_FILE
            raw = torch.load(path, map_location="cpu", weights_only=True)
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

        Three deliberate choices:

        - no ``attention_mask``: the schema stores token ids only;
        - the pad convention is the one the training run **recorded**
          (:attr:`BundleMetadata.loss_ignore_index`). If the run's loss ignored a
          target id, the labels carry ``-100`` wherever the next-token target
          equals that id, so the loss, its gradient and every count describe the
          mean over the non-ignored targets the run descended. If nothing was
          recorded, every target counts. The labels are the only place this is
          expressed — no ``valid_mask_fn`` is involved;
        - the shift is applied *here* rather than assumed to be baked in: the
          schema says ``batch_a.npy`` holds the *token ids of the batch*, so if
          the writer stores the model inputs, the last position has no target
          and is ignored.

        Use :attr:`batch_a_tokens` if you need the raw ids.
        """
        if self._batch_a is None:
            from vatis.core.normalization import DEFAULT_IGNORE_INDEX
            from vatis.data.collate import shifted_lm_targets

            ids = self.batch_a_tokens
            labels = shifted_lm_targets(ids, ignore_index=DEFAULT_IGNORE_INDEX)
            loss_ignore_index = self._source.loss_ignore_index
            if loss_ignore_index is not None:
                labels = labels.masked_fill(labels == loss_ignore_index, DEFAULT_IGNORE_INDEX)
            self._batch_a = {"input_ids": ids, "labels": labels}
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
        # The pad convention of the training loss, resolved (and validated) once
        # at open time so contradictory metadata fails here, not at the first
        # ``batch_a`` access deep inside a run. See ``BundleMetadata.loss_ignore_index``.
        self.loss_ignore_index: int | None = self.metadata.loss_ignore_index
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
        # Every step on disk, also the ones a ``steps=`` selection leaves out:
        # the realized loss change at step t needs the weights of step t + 1
        # whether or not that checkpoint is analyzed itself.
        self._step_dirs: dict[int, Path] = dict(found)
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
        on_step_mismatch: str = "raise",
    ) -> Callable[[str], Preconditioner | None]:
        """The ``adamw`` P source, as a per-checkpoint factory for the analyzer.

        Pass the result as ``Analyzer(preconditioner=...)`` and sweep with
        ``analyzer.run(source.model_specs())`` (``analyze()`` takes a single hub
        model spec and cannot iterate a bundle). The factory is called once per
        checkpoint with
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

        A bundle written by ``torch.optim.SGD`` is refused here with that reason
        (its saved ``param_groups`` identify it even at step 0, where every
        optimizer's state is empty); use :meth:`preconditioner`, which handles
        both optimizers. A state written by any optimizer other than
        ``torch.optim.Adam`` / ``AdamW`` is refused as well, at every step and
        whether or not ``metadata.json`` names the optimizer: RAdam, NAdam and
        Adamax also save ``betas`` and an ``exp_avg_sq``, but their param groups
        lack the ``amsgrad`` key every Adam / AdamW group has.

        Param groups that disagree on ``lr`` are refused at every checkpoint
        (see :func:`adamw_preconditioner_from_state`); a ``beta2`` / ``eps``
        override does not change that.

        The optimizer state is read on the CPU; the returned ``P`` is on the
        source's ``device``.

        Args:
            beta2: override for Adam's second-moment decay.
            eps: override for Adam's epsilon.
            descriptor: the parquet descriptor of the resulting rows.
            on_step_mismatch: what to do when the saved optimizer state's own
                step counter differs from the checkpoint directory's step.
                Under pre-step semantics the two are the same number, so a
                difference means the optimizer state and the weights may not
                belong together. ``"raise"`` (default) raises
                :class:`BundleSchemaError`; ``"warn"`` emits a ``UserWarning``
                and builds ``P`` from the saved state's own counter — for a
                bundle whose step was merely stored wrongly.

        Returns:
            ``fn(checkpoint_id) -> Preconditioner | None``.
        """
        _check_step_mismatch_policy(on_step_mismatch)

        def build(checkpoint_id: str) -> Preconditioner | None:
            checkpoint = self.checkpoint_for(checkpoint_id)
            state = checkpoint.optimizer_state
            family = _optimizer_family(state)
            if family == "sgd":
                raise BundleSchemaError(
                    f"{checkpoint.identifier}: the saved param_groups are those of an SGD "
                    "optimizer, which has no second moment and hence no AdamW "
                    "preconditioner. Use source.preconditioner() to get SGD's geometry "
                    f"factor P = I under the {SGD_DESCRIPTOR!r} descriptor, or "
                    "preconditioner=None for the raw rows only."
                )
            if state.is_empty:
                if family != "adam":
                    # The groups identify the optimizer even without any state, so
                    # a non-Adam bundle is refused at step 0 as well, not skipped
                    # there and refused one checkpoint later.
                    raise BundleSchemaError(_not_adam_message(state, checkpoint.identifier))
                # Likewise for per-group learning rates: the groups are saved at
                # step 0 too.
                _refuse_per_group_lr(state, checkpoint.identifier)
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
                mismatch = (
                    f"{checkpoint.identifier}: the saved optimizer state has applied "
                    f"{state.step_count} update(s), but the checkpoint directory says step "
                    f"{checkpoint.step}. Under pre-step semantics the two are the same "
                    "number; a mismatch means the optimizer state and the weights may not "
                    "belong together."
                )
                if on_step_mismatch == "raise":
                    raise BundleSchemaError(
                        f"{mismatch} Refusing to build P from it. If only the step was "
                        'stored wrongly, pass on_step_mismatch="warn" to build P from the '
                        "saved state's own counter."
                    )
                warnings.warn(
                    f"{mismatch} P is built from the saved state's own counter.",
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
            # The state was read and P was built on the CPU; only the finished P
            # goes to the source's device.
            return precond.to(device=self.device)

        return build

    def preconditioner(
        self,
        *,
        beta2: float | None = None,
        eps: float | None = None,
        on_step_mismatch: str = "raise",
    ) -> Callable[[str], Preconditioner | None]:
        """The saved optimizer's geometry factor, whichever optimizer wrote the bundle.

        A per-checkpoint factory for ``Analyzer(preconditioner=...)`` that looks
        at the checkpoint's saved ``param_groups`` and returns

        - for Adam / AdamW: exactly what :meth:`adamw_preconditioner` returns
          (descriptor ``"adamw"``; no rows at a step-0 checkpoint);
        - for ``torch.optim.SGD``, with or without momentum: ``P = I`` under the
          descriptor ``"sgd"``, at every checkpoint including step 0.

        So a comparison across optimizers reads the same three observables
        (``delta_loss_P``, ``chi_net_P``, ``chi_pos_P``) for both, and the
        ``preconditioner`` column says which geometry each row used. For SGD the
        geometry factor is the identity, so its rows repeat the raw values
        exactly (``chi_pos_P == chi_pos``); they cost one parameter-sized vector
        of ones. Momentum and weight decay are excluded for SGD for the same
        reason as for AdamW: ``P`` is the map from the gradient to the update
        direction, not the update.

        Any other optimizer is refused. ``beta2`` / ``eps`` are AdamW overrides
        and are refused for an SGD bundle.

        Param groups that disagree on ``lr`` are refused for both optimizers:
        ``P`` leaves the learning rate out, which is only sound when every
        parameter has the same one. With per-group rates neither the AdamW
        ``P`` nor SGD's ``P = I`` is the optimizer's geometry.

        ``on_step_mismatch`` is passed to :meth:`adamw_preconditioner` (default
        ``"raise"``). An SGD state has no step counter to compare with the
        directory step, so it has no effect on an SGD bundle.
        """
        adamw = self.adamw_preconditioner(beta2=beta2, eps=eps, on_step_mismatch=on_step_mismatch)

        def build(checkpoint_id: str) -> Preconditioner | None:
            checkpoint = self.checkpoint_for(checkpoint_id)
            family = _optimizer_family(checkpoint.optimizer_state)
            recorded = self.metadata.optimizer_name
            if family == "adam":
                return adamw(checkpoint_id)
            if family == "sgd":
                if recorded is not None and recorded.lower() != "sgd":
                    raise BundleSchemaError(
                        f"{checkpoint.identifier}: the saved param_groups are those of an "
                        f"SGD optimizer but metadata.json names the optimizer {recorded!r}; "
                        "the bundle's metadata and its optimizer state do not belong together."
                    )
                if beta2 is not None or eps is not None:
                    raise ValueError(
                        f"{checkpoint.identifier}: beta2 / eps are AdamW overrides, but this "
                        "bundle was written by SGD, whose geometry factor is P = I."
                    )
                _refuse_per_group_lr(
                    checkpoint.optimizer_state, checkpoint.identifier, optimizer="SGD"
                )
                checkpoint.release_optimizer_state()
                return Preconditioner.identity(
                    self.layout.total_numel, device=self.device, descriptor=SGD_DESCRIPTOR
                )
            group_keys = sorted(
                {key for group in checkpoint.optimizer_state.param_groups for key in group}
                - {"params"}
            )
            raise BundleSchemaError(
                f"{checkpoint.identifier}: cannot tell which geometry factor belongs to "
                f"this optimizer (metadata.json names {recorded!r}; saved param_group keys: "
                f"{group_keys}). "
                "preconditioner() knows Adam / AdamW and torch.optim.SGD; for anything "
                "else build P with Preconditioner.from_diagonal(...)."
            )

        return build

    def update_response(
        self,
        *,
        clip_grad_norm: float | str | None = CLIP_FROM_METADATA,
        realized: bool = True,
        on_step_mismatch: str = "raise",
        unlisted_params_frozen: bool = False,
    ) -> Callable[[str], OptimizerUpdate | None]:
        """The update each step applied, as a per-checkpoint factory for the analyzer.

        Pass the result as ``Analyzer(update_response=...)`` together with
        :meth:`batch_a_spec` among the eval batches. For every checkpoint the
        analyzer then measures the first-order loss change under the update
        the optimizer took at that step, ``grad L^B . (theta_{t+1} - theta_t)``,
        split into the parts carried by the step's own gradient, by the
        gradient history and by weight decay (the ``loss_change_first_order*`` rows;
        definitions in :mod:`vatis.core.update_response` and ``docs/SPEC.md``
        §"The response to the update that was taken").

        The update is rebuilt from what the checkpoint saved **before** the
        step: the weights, the optimizer state (``exp_avg`` / ``exp_avg_sq`` /
        the step counter, or the momentum buffer) and the hyper-parameters of
        the saved ``param_groups`` — per parameter, so groups with different
        learning rates or weight decays are handled exactly. The learning rate
        is the saved group's ``lr``, i.e. the scheduled value the step applied;
        it is cross-checked against ``meta.json``'s ``lr`` when the groups share
        one rate. Unlike the P rows these rows exist at step 0 too: an empty
        optimizer state is a well-defined starting point for an update.

        Covered: ``torch.optim.AdamW``, ``Adam`` (coupled L2 decay) and ``SGD``
        (momentum, dampening, Nesterov, coupled decay). Refused: any other
        optimizer, ``amsgrad=True``, ``maximize=True``, a parameter that has no
        state while others have.

        **What the saved state cannot show, and what is done about it.** The
        gradient ``grad L^A`` is recomputed by the analyzer from ``batch_a.npy``
        at the stored weights; the rows are the response to the step the
        optimizer took only if that is the gradient it saw.

        - *Gradient clipping* changes it and leaves no trace in the state. With
          ``clip_grad_norm="metadata"`` (default) ``metadata.json`` must say what
          happened: ``gradient_clip_val`` null or 0 (not clipped), or a positive
          value together with ``gradient_clip_algorithm: "norm"``. A bundle that
          does not record the key at all, or records a value without the
          algorithm, is refused; state it yourself with ``clip_grad_norm=None``
          (not clipped) or ``clip_grad_norm=<max norm>`` (global-norm clipping as
          ``torch.nn.utils.clip_grad_norm_`` over all parameters does it). A
          stated value that contradicts the recorded one is refused as well.
          The norm is taken over the bundle's canonical parameters.
        - *Parameters outside the canonical list* (a source opened with
          ``allow_unlisted_params`` whose model has parameters that
          ``param_names.json`` does not name). They are in no gradient vector
          and in no saved optimizer state. If the run trained them, their part
          of the update is missing from every row while ``loss_change`` still
          contains its effect; if they merely had gradients when the norm was
          clipped, the clipping factor is wrong. Nothing in a bundle says
          either way, so such a bundle is **refused** unless you confirm with
          ``unlisted_params_frozen=True`` that those parameters received no
          gradient and no update during the run.
        - *A parameter that is not in the graph of the update batch* gets no
          gradient at all. torch then skips it entirely if its ``.grad`` is
          ``None`` (the default ``zero_grad()``), and applies momentum and
          weight decay to it if its ``.grad`` holds zeros
          (``zero_grad(set_to_none=False)``). The analyzer knows which
          parameters these are; the rows are refused when the saved state
          would move such a parameter, and are exact when it would not.
        - *Gradient accumulation, dropout, mixed precision, anything else the
          loop did to the gradient*: not detectable. The bundle contract is that
          one optimizer step on ``batch_a.npy`` at the stored weights reproduces
          the run's ``theta_{t+1}``; see the SPEC section for what that requires
          of a writer.

        **The realized loss change** (``realized=True``, default). Where the
        bundle also holds step ``t + 1`` (whether or not it is among the
        selected ``steps``), both weight files are stored as fp32 or wider and
        the source itself was opened with ``dtype="fp32"``,
        the factory hands the analyzer ``theta_{t+1}``, and three more rows are
        measured: ``loss_change_first_order_weights = grad L^B . (theta_{t+1} -
        theta_t)`` from the two weight files, ``loss_change = L_B(theta_{t+1}) -
        L_B(theta_t)`` and their difference ``loss_change_residual``. With bf16
        (or fp16) weight files — or with a source that loads the weights as
        bf16 / fp16, which rounds them in memory just the same — those rows are
        **skipped with a warning**: the
        difference of two rounded sets of weights is dominated by the rounding
        early in training, and a residual computed from it would be an
        artefact. A checkpoint without a stored successor has no such rows and
        no warning.

        Args:
            clip_grad_norm: ``"metadata"`` (default), ``None`` or a positive
                max norm; see above.
            realized: measure the realized loss change where the bundle allows.
            on_step_mismatch: as for :meth:`adamw_preconditioner` — what to do
                when the optimizer state's own step counter differs from the
                checkpoint directory's step (``"raise"`` or ``"warn"``).
            unlisted_params_frozen: only matters for a source opened with
                ``allow_unlisted_params``: confirms that the model's parameters
                outside ``param_names.json`` received no gradient and no update
                during the run. Without it such a bundle is refused.

        Returns:
            ``fn(checkpoint_id) -> OptimizerUpdate``.
        """
        _check_step_mismatch_policy(on_step_mismatch)
        clip = _resolve_clip_grad_norm(self.metadata, clip_grad_norm)

        def build(checkpoint_id: str) -> OptimizerUpdate | None:
            checkpoint = self.checkpoint_for(checkpoint_id)
            if self.allow_unlisted_params and not unlisted_params_frozen:
                bundle = checkpoint.model_bundle
                listed = {id(p) for p in bundle.params}
                unlisted = [
                    name for name, p in bundle.model.named_parameters() if id(p) not in listed
                ]
                if unlisted:
                    raise ValueError(
                        f"{checkpoint.identifier}: the model has parameters outside "
                        f"param_names.json ({_abbrev(unlisted)}; the source was opened with "
                        "allow_unlisted_params). They are in no gradient vector and in no "
                        "saved optimizer state: if the run trained them, their part of the "
                        "update is missing from every loss_change_first_order* row while loss_change "
                        "still contains its effect, and if they had gradients when the norm "
                        "was clipped, the clipping factor is wrong. The bundle does not say. "
                        "Pass unlisted_params_frozen=True to confirm that they received no "
                        "gradient and no update during the run."
                    )
            update = optimizer_update_from_state(
                checkpoint.optimizer_state,
                checkpoint_step=checkpoint.step,
                metadata=self.metadata,
                recorded_lr=checkpoint.meta.lr,
                clip_grad_norm=clip,
                on_step_mismatch=on_step_mismatch,
                context=checkpoint.identifier,
            )
            return dataclasses.replace(
                update,
                update_batch=checkpoint.batch_a,
                next_weights=self._next_weights(checkpoint) if realized else None,
            )

        return build

    def _next_weights(self, checkpoint: LocalBundleCheckpoint) -> Callable[[], ModelBundle] | None:
        """A loader for ``theta_{t+1}``, or ``None`` when it must not be used.

        ``None`` without a word when the bundle holds no step ``t + 1``; ``None``
        with a ``UserWarning`` when it does but the two sets of weights the
        analyzer would compare are rounded: one of the weight files is stored
        narrower than fp32, or this source loads the weights narrower than fp32
        (``dtype="bf16"`` / ``"fp16"``), which rounds unrounded files in memory.
        """
        step = checkpoint.step + 1
        path = self._step_dirs.get(step)
        if path is None or not (path / CHECKPOINT_WEIGHTS_FILE).is_file():
            return None
        narrow = {
            f"step {c_step}": sorted(dtypes - _EXACT_WEIGHT_DTYPES)
            for c_step, c_path in ((checkpoint.step, checkpoint.path), (step, path))
            if (
                dtypes := _stored_weight_dtypes(c_path / CHECKPOINT_WEIGHTS_FILE, self.layout.names)
            )
            - _EXACT_WEIGHT_DTYPES
        }
        reasons = []
        if narrow:
            reasons.append(f"the weight files are stored as {narrow}")
        if self.torch_dtype not in _EXACT_LOADER_DTYPES:
            reasons.append(
                f'the source was opened with dtype="{self.dtype}", which rounds the weights '
                "when they are loaded"
            )
        if reasons:
            warnings.warn(
                f"{checkpoint.identifier}: the bundle holds step {step} as well, but "
                f"{' and '.join(reasons)}. The analyzer would then compare two rounded "
                "thetas, and the difference of two rounded thetas is dominated by the "
                "rounding early in training, so the realized loss change and its residual "
                "(loss_change_first_order_weights, loss_change, loss_change_residual) are SKIPPED "
                "for this checkpoint. They need fp32 weight files, loaded with "
                'dtype="fp32". The loss_change_first_order* rows built from the optimizer state '
                "are still reported.",
                UserWarning,
                stacklevel=3,
            )
            return None
        if step in self.steps:
            successor = self.checkpoint_for(step)
        else:
            # Not among the selected steps: a handle of its own, dropped with the
            # closure once the analyzer is done with this checkpoint.
            successor = LocalBundleCheckpoint(self, path, step)
        return lambda: successor.model_bundle

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
            weights are bf16 on disk and the analysis math wants fp32, so the
            upcast happens on load. The upcast itself adds no error, but it
            does not undo the rounding that writing bf16 applied: the loaded
            values are the run's ``theta_t`` rounded to bf16, not ``theta_t``
            (see ``docs/SPEC.md`` §"Native checkpoint bundles" for what that
            does to single-step comparisons).
        device: device for the model and the batches, and for the ``P`` the
            preconditioner factories return. The optimizer state itself is
            read and kept on the CPU (see
            :attr:`LocalBundleCheckpoint.optimizer_state`).
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


def _optimizer_family(state: OptimizerStateView) -> str | None:
    """``"adam"``, ``"sgd"`` or ``None``, read off the saved ``param_groups``.

    The per-parameter state cannot answer this: it is empty at step 0 for every
    optimizer, and at every step for SGD without momentum. The hyper-parameter
    keys of the groups are always saved and differ by optimizer class.

    ``"adam"`` means ``torch.optim.Adam`` or ``AdamW`` and nothing else: the
    groups must carry ``amsgrad`` next to ``betas`` (:data:`_ADAM_GROUP_KEYS`).
    RAdam, NAdam, Adamax and SparseAdam groups have ``betas`` but no
    ``amsgrad`` and come back as ``None``.
    """
    groups = state.param_groups
    if not groups:
        return None
    if all(group.keys() >= _ADAM_GROUP_KEYS for group in groups):
        return "adam"
    if all(group.keys() >= _SGD_GROUP_KEYS and "betas" not in group for group in groups):
        return "sgd"
    return None


def _check_step_mismatch_policy(on_step_mismatch: str) -> None:
    if on_step_mismatch not in _STEP_MISMATCH_POLICIES:
        raise ValueError(
            f"on_step_mismatch must be one of {list(_STEP_MISMATCH_POLICIES)}, "
            f"got {on_step_mismatch!r}"
        )


def _rate_key(rate: Any) -> Any:
    """A learning rate as a comparable value (``1`` and ``1.0`` are the same rate)."""
    try:
        return float(rate)
    except (TypeError, ValueError):
        return repr(rate)


def _refuse_per_group_lr(
    state: OptimizerStateView, context: str, *, optimizer: str = "AdamW"
) -> None:
    """Raise unless every saved param group has the same learning rate.

    ``P`` deliberately leaves the learning rate out: for one rate ``eta`` it is
    a common factor, ``dtheta = -eta * P grad L``, and ``chi_pos_P`` does not
    depend on it. With per-group rates the update is ``-eta_g * P grad L`` on
    group ``g``, so the rates are part of the geometry and the documented ``P``
    (or ``P = I`` for SGD) is not the optimizer's.

    The saved ``lr`` is the scheduled value at that checkpoint. Groups that
    share one schedule agree on it at every checkpoint, so the saved values are
    compared as they are.
    """
    rates = [group.get("lr") for group in state.param_groups]
    if len({_rate_key(rate) for rate in rates}) <= 1:
        return
    if optimizer == "SGD":
        remedy = (
            "a diagonal that holds each group's rate relative to a reference rate in place of P = I"
        )
    else:
        remedy = (
            "the AdamW P = 1/(sqrt(v_hat) + eps) scaled per group by its rate relative "
            "to a reference rate"
        )
    raise ValueError(
        f"{context}: the saved param_groups have different learning rates "
        f"({_abbrev(rates)}). P leaves the learning rate out, which is only sound when "
        "it is the same for every parameter: per-group rates change the optimizer's "
        "geometry (the update is -lr_g * P grad L on group g), and one P without the "
        f"rate cannot represent that. For {optimizer}, build the P you mean with "
        f"Preconditioner.from_diagonal(...), e.g. {remedy}. A beta2 / eps override does "
        "not lift this."
    )


def _not_adam_message(state: OptimizerStateView, context: str) -> str:
    """Why a state whose ``param_groups`` are not Adam's / AdamW's is refused."""
    group_keys = sorted({key for group in state.param_groups for key in group} - {"params"})
    return (
        f"{context}: the saved param_groups are not those of torch.optim.Adam / AdamW "
        f"(hyper-parameter keys: {group_keys}; an Adam / AdamW group carries "
        f"{sorted(_ADAM_GROUP_KEYS)}). P = 1/(sqrt(v_hat) + eps) is the preconditioner "
        "of those two only; other optimizers that keep 'betas' and an 'exp_avg_sq' "
        "(RAdam, NAdam, Adamax, ...) apply a different update, and labelling their rows "
        "'adamw' would be wrong. Build the P you mean with "
        "Preconditioner.from_diagonal(...)."
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
            SGD run — no second moment exists, use ``preconditioner=None``), has
            no ``step`` counter to bias-correct with, or was not written by
            ``torch.optim.Adam`` / ``AdamW`` — judged by ``metadata.json``'s
            optimizer name when there is one and, independently of it, by the
            saved ``param_groups`` (RAdam, NAdam and Adamax keep an
            ``exp_avg_sq`` too, but their groups have no ``amsgrad`` key).
        ValueError: ``betas``/``eps`` missing from the saved ``param_groups``,
            ``amsgrad=True`` (which preconditions with ``max_exp_avg_sq``), or
            param groups that disagree on ``lr``. ``P`` leaves the learning rate
            out, which is only sound when it is the same for every parameter;
            per-group rates are part of the optimizer's geometry. Explicit
            ``beta2`` / ``eps`` do not bypass this check (nor the ``amsgrad``
            one).
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
    # Looked up on every saved group, not on the collapsed ``state.hparams`` (which
    # drops a key the groups disagree on) and regardless of a beta2 / eps override:
    # an AMSGrad group divides by max_exp_avg_sq, which no choice of beta2 / eps
    # turns into the documented P.
    amsgrad_groups = [
        index for index, group in enumerate(state.param_groups) if bool(group.get("amsgrad", False))
    ]
    if amsgrad_groups:
        raise ValueError(
            f"{context}: saved param_groups {amsgrad_groups} have amsgrad=True. AMSGrad "
            "preconditions with max_exp_avg_sq, so the documented v_hat = exp_avg_sq / "
            "(1 - beta2**t) convention would build the wrong P for those parameters. "
            "Refusing rather than reporting a plausible wrong number."
        )
    if not state.has_state(state_key):
        if state_key not in state.state_keys():
            raise BundleSchemaError(
                f"{context}: the saved optimizer state has no {state_key!r} at all "
                f"(present keys: {state.state_keys()}). An AdamW preconditioner needs the "
                "second moment; a plain-SGD run has none and its geometry factor is P = I "
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
    # Decided by the saved groups, not by metadata.json: a bundle that names no
    # optimizer (or names it under a key vatis does not read) must not turn a
    # RAdam / NAdam state into rows labelled "adamw".
    if _optimizer_family(state) != "adam":
        raise BundleSchemaError(_not_adam_message(state, context))
    # Before, and independent of, the beta2 / eps handling below: an override
    # must not be a way around it (as it once was for amsgrad).
    _refuse_per_group_lr(state, context)
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


# ------------------------------------------------------------ update response


def _resolve_clip_grad_norm(
    metadata: BundleMetadata, clip_grad_norm: float | str | None
) -> float | None:
    """The global-norm clipping threshold of the run, or ``None`` for "not clipped".

    Clipping rescales the gradient before the optimizer sees it and leaves no
    trace in the optimizer state, so it has to be *stated* — by the bundle's
    ``metadata.json`` or by the caller — and is never assumed either way.
    See :meth:`LocalBundleSource.update_response`.
    """
    where = f"{metadata.path}: " if metadata.path is not None else ""
    has_key = "gradient_clip_val" in metadata
    recorded = metadata.get("gradient_clip_val")
    algorithm = metadata.get("gradient_clip_algorithm")
    if recorded is not None and (
        isinstance(recorded, bool) or not isinstance(recorded, (int, float))
    ):
        raise BundleSchemaError(
            f"{where}'gradient_clip_val' must be a number or null, got {recorded!r}"
        )
    if recorded is not None and not (math.isfinite(recorded) and recorded >= 0):
        raise BundleSchemaError(
            f"{where}'gradient_clip_val' must be a finite number >= 0 or null, got {recorded!r}"
        )
    if algorithm is not None and algorithm not in ("norm", "value"):
        raise BundleSchemaError(
            f"{where}'gradient_clip_algorithm' must be 'norm', 'value' or absent, got {algorithm!r}"
        )
    recorded_norm = float(recorded) if recorded else None  # null and 0 both mean "off"

    if isinstance(clip_grad_norm, str):
        if clip_grad_norm != CLIP_FROM_METADATA:
            raise ValueError(
                f"clip_grad_norm must be {CLIP_FROM_METADATA!r}, None or a positive number, "
                f"got {clip_grad_norm!r}"
            )
        if not has_key:
            raise BundleSchemaError(
                f"{where}no 'gradient_clip_val' is recorded, so the bundle does not say "
                "whether the gradient was clipped before the optimizer saw it. Clipping "
                "leaves no trace in the saved optimizer state and changes every "
                "loss_change_first_order* row, so it is not assumed either way: pass "
                "clip_grad_norm=None if the run did not clip, or clip_grad_norm=<max norm> "
                "if it clipped the global gradient norm (torch.nn.utils.clip_grad_norm_ "
                "over all parameters)."
            )
        if recorded_norm is None:
            return None
        if algorithm == "norm":
            return recorded_norm
        if algorithm == "value":
            raise BundleSchemaError(
                f"{where}the run clipped gradients by value (gradient_clip_algorithm = "
                f"'value', gradient_clip_val = {recorded}). Only global-norm clipping is "
                "implemented for the update response."
            )
        raise BundleSchemaError(
            f"{where}'gradient_clip_val' is {recorded} but no 'gradient_clip_algorithm' is "
            "recorded, so the bundle does not say HOW the gradient was clipped. If it was "
            "the global gradient norm (torch.nn.utils.clip_grad_norm_ over all parameters, "
            f"also Lightning's default), pass clip_grad_norm={recorded}; clipping by value "
            "is not implemented."
        )

    if clip_grad_norm is None:
        if recorded_norm is not None:
            raise ValueError(
                f"{where}clip_grad_norm=None says the run did not clip, but metadata.json "
                f"records gradient_clip_val = {recorded}. Pass clip_grad_norm={recorded} "
                "(global-norm clipping) or correct the metadata."
            )
        return None
    if isinstance(clip_grad_norm, bool) or not isinstance(clip_grad_norm, (int, float)):
        raise ValueError(
            f"clip_grad_norm must be {CLIP_FROM_METADATA!r}, None or a positive number, got "
            f"{clip_grad_norm!r}"
        )
    stated = float(clip_grad_norm)
    if not (math.isfinite(stated) and stated > 0.0):
        raise ValueError(f"clip_grad_norm must be a finite positive number, got {clip_grad_norm}")
    if has_key and recorded_norm is None:
        raise ValueError(
            f"{where}clip_grad_norm={stated} says the run clipped, but metadata.json "
            f"records gradient_clip_val = {recorded!r} (not clipped)."
        )
    if recorded_norm is not None and recorded_norm != stated:
        raise ValueError(
            f"{where}clip_grad_norm={stated} contradicts the recorded gradient_clip_val = "
            f"{recorded}."
        )
    if algorithm == "value":
        raise ValueError(
            f"{where}clip_grad_norm={stated} states global-norm clipping, but metadata.json "
            "records gradient_clip_algorithm = 'value'."
        )
    return stated


def optimizer_update_from_state(
    state: OptimizerStateView,
    *,
    checkpoint_step: int | None = None,
    metadata: BundleMetadata | None = None,
    recorded_lr: float | None = None,
    clip_grad_norm: float | None = None,
    on_step_mismatch: str = "raise",
    context: str = "optimizer state",
) -> OptimizerUpdate:
    """Turn one saved (pre-step) optimizer state into the update rule of that step.

    The plumbing half of the ``loss_change_first_order*`` rows; the math is
    :class:`vatis.core.update_response.OptimizerUpdate`. Hyper-parameters are
    read per parameter from the saved ``param_groups`` (what the optimizer ran
    with), never from ``metadata.json``; the metadata's optimizer name is only a
    cross-check, and the tie-breaker for one question the groups of an old
    torch cannot answer (see below).

    Args:
        state: the saved optimizer state.
        checkpoint_step: the checkpoint directory's step, compared with the
            state's own notion of how many updates were applied.
        metadata: the bundle's ``metadata.json``.
        recorded_lr: the checkpoint's ``meta.json`` ``lr``; when every group has
            the same saved ``lr`` the two must agree.
        clip_grad_norm: global-norm clipping threshold, already resolved
            (``None`` = not clipped).
        on_step_mismatch: ``"raise"`` or ``"warn"``.
        context: prefix of every message (the checkpoint identifier).

    Raises:
        BundleSchemaError: an optimizer other than Adam / AdamW / SGD (judged by
            the saved groups, and by ``metadata.json``'s name when there is
            one), a state that has moments for some parameters only, a state
            without a step counter, a step or learning-rate mismatch.
        ValueError: ``amsgrad=True`` or ``maximize=True`` in any group, a group
            that lacks a hyper-parameter, or an Adam-family group with non-zero
            weight decay for which neither the group (its
            ``decoupled_weight_decay`` entry) nor ``metadata.json`` says whether
            the decay was decoupled.
    """
    _check_step_mismatch_policy(on_step_mismatch)
    family = _optimizer_family(state)
    recorded_name = None if metadata is None else metadata.optimizer_name
    if family is None:
        group_keys = sorted({key for group in state.param_groups for key in group} - {"params"})
        raise BundleSchemaError(
            f"{context}: the update response is implemented for torch.optim.Adam, AdamW "
            f"and SGD, and the saved param_groups are none of these (metadata.json names "
            f"{recorded_name!r}; hyper-parameter keys: {group_keys}). RAdam, NAdam, Adamax, "
            "RMSprop and the others apply a different update; no rows rather than rows "
            "from the wrong formula."
        )
    allowed = _ADAM_FAMILY if family == "adam" else frozenset({"sgd"})
    if recorded_name is not None and recorded_name.lower() not in allowed:
        raise BundleSchemaError(
            f"{context}: the saved param_groups are those of "
            f"{'Adam / AdamW' if family == 'adam' else 'SGD'} but metadata.json names the "
            f"optimizer {recorded_name!r}; the bundle's metadata and its optimizer state do "
            "not belong together."
        )
    groups = state.group_of_parameter()
    for flag, why in (
        ("amsgrad", "AMSGrad divides by max_exp_avg_sq, which the Adam formula does not use"),
        ("maximize", "the step ascends the gradient; that sign convention is not implemented"),
    ):
        flagged = [i for i, g in enumerate(state.param_groups) if bool(g.get(flag, False))]
        if flagged:
            raise ValueError(
                f"{context}: saved param_groups {flagged} have {flag}=True ({why}). "
                "Refusing rather than reporting a plausible wrong number."
            )

    def hyper(group: Mapping[str, Any], key: str) -> Any:
        if group.get(key) is None:
            raise ValueError(
                f"{context}: a saved param_group has no {key!r}; the update cannot be "
                "rebuilt without it"
            )
        return group[key]

    rates = {_rate_key(g.get("lr")) for g in state.param_groups}
    if recorded_lr is not None and len(rates) == 1:
        saved_lr = float(hyper(state.param_groups[0], "lr"))
        if abs(saved_lr - recorded_lr) > _LR_RTOL * max(abs(saved_lr), abs(recorded_lr)):
            raise BundleSchemaError(
                f"{context}: the saved optimizer state holds lr = {saved_lr!r} but the "
                f"checkpoint's meta.json records lr = {recorded_lr!r}. Under pre-step "
                "semantics both are the learning rate this step applies; a difference "
                "means one of them is not (a scheduler stepped between the two writes, or "
                "the state and the checkpoint do not belong together), and every "
                "loss_change_first_order* row is proportional to it."
            )

    hypers: list[AdamHyper] | list[SGDHyper]
    first: list[torch.Tensor | None] | None
    second: list[torch.Tensor] | None = None
    if family == "adam":
        adam_hypers: list[AdamHyper] = []
        decoupled_flags: set[bool] = set()
        for group in groups:
            betas = hyper(group, "betas")
            if isinstance(betas, (str, bytes)) or not isinstance(betas, Sequence) or len(betas) < 2:
                raise ValueError(f"{context}: 'betas' is not a (beta1, beta2) pair: {betas!r}")
            weight_decay = float(hyper(group, "weight_decay"))
            decoupled = _decoupled_weight_decay(group, weight_decay, recorded_name, context)
            if weight_decay != 0.0:
                decoupled_flags.add(decoupled)
            adam_hypers.append(
                AdamHyper(
                    lr=float(hyper(group, "lr")),
                    beta1=float(betas[0]),
                    beta2=float(betas[1]),
                    eps=float(hyper(group, "eps")),
                    weight_decay=weight_decay,
                    decoupled_weight_decay=decoupled,
                )
            )
        hypers = adam_hypers
        has_first, has_second = state.has_state("exp_avg"), state.has_state("exp_avg_sq")
        if state.is_empty:
            step = 0
            first = None
        elif not (has_first and has_second):
            raise BundleSchemaError(
                f"{context}: the saved optimizer state has 'exp_avg' / 'exp_avg_sq' for some "
                "parameters but not for all (present keys: "
                f"{state.state_keys()}). torch keeps no state for a parameter that never "
                "received a gradient, and whether it receives one at this step — and is "
                "then decayed — cannot be told from the bundle. Exclude it from the "
                "bundle's parameter list."
            )
        else:
            counter = state.step_count
            if counter is None:
                raise BundleSchemaError(
                    f"{context}: the saved optimizer state has no 'step' counter, so the "
                    "bias corrections of the step cannot be formed. Refusing to guess."
                )
            step = counter
            first = state.parameter_states("exp_avg")
            second = [t for t in state.parameter_states("exp_avg_sq") if t is not None]
        applied = f"has applied {step} update(s)"
        mismatch = checkpoint_step is not None and step != checkpoint_step
        # "adam" when the decay is coupled, or when there is none and the bundle
        # names Adam; "adamw" otherwise (with zero decay the two rules coincide).
        named_adam = not decoupled_flags and (recorded_name or "").lower() == "adam"
        descriptor = "adam_update" if decoupled_flags == {False} or named_adam else "adamw_update"
    else:
        sgd_hypers = [
            SGDHyper(
                lr=float(hyper(group, "lr")),
                momentum=float(hyper(group, "momentum")),
                dampening=float(hyper(group, "dampening")),
                weight_decay=float(hyper(group, "weight_decay")),
                nesterov=bool(hyper(group, "nesterov")),
            )
            for group in groups
        ]
        hypers = sgd_hypers
        buffers = state.parameter_states("momentum_buffer")
        with_momentum = [i for i, h in enumerate(sgd_hypers) if h.momentum != 0.0]
        have = [i for i in with_momentum if buffers[i] is not None]
        if have and len(have) != len(with_momentum):
            raise BundleSchemaError(
                f"{context}: {len(have)} of the {len(with_momentum)} parameters updated "
                "with momentum have a momentum buffer and the others have none. torch "
                "creates the buffer at a parameter's first gradient, so some parameter "
                "never received one, and whether it does at this step cannot be told from "
                "the bundle. Exclude it from the bundle's parameter list."
            )
        first = buffers if have else None
        step = 0
        applied = (
            "holds momentum buffers, so at least one update was applied"
            if have
            else "holds no momentum buffer, so no update was applied yet"
        )
        mismatch = (
            checkpoint_step is not None
            and bool(with_momentum)
            and (bool(have) != (checkpoint_step > 0))
        )
        descriptor = "sgd_update"

    if mismatch:
        message = (
            f"{context}: the saved optimizer state {applied}, but the checkpoint "
            f"directory says step {checkpoint_step}. Under pre-step semantics the two "
            "must agree; a mismatch means the optimizer state and the weights may not "
            "belong together."
        )
        if on_step_mismatch == "raise":
            raise BundleSchemaError(
                f"{message} Refusing to rebuild the update from it. If only the step was "
                'stored wrongly, pass on_step_mismatch="warn" to use the saved state as it is.'
            )
        warnings.warn(f"{message} The saved state is used as it is.", UserWarning, stacklevel=2)

    return OptimizerUpdate(
        family=ADAM_FAMILY if family == "adam" else SGD_FAMILY,
        hypers=hypers,
        step=step,
        first_moment=first,
        second_moment=second,
        clip_grad_norm=clip_grad_norm,
        descriptor=descriptor,
    )


def _decoupled_weight_decay(
    group: Mapping[str, Any],
    weight_decay: float,
    recorded_name: str | None,
    context: str,
) -> bool:
    """Whether an Adam-family group applies its weight decay decoupled (AdamW).

    Recent torch saves the answer in the group (``decoupled_weight_decay``;
    present in the groups written by torch 2.9 and 2.10). A group without that
    entry does not say whether it belonged to ``Adam`` or ``AdamW``; then
    ``metadata.json``'s optimizer name decides, and without one a non-zero
    decay is refused. With zero decay the two coincide and the answer is moot.
    """
    flag = group.get("decoupled_weight_decay")
    if flag is not None:
        return bool(flag)
    if weight_decay == 0.0:
        return True
    name = (recorded_name or "").lower()
    if name == "adamw":
        return True
    if name == "adam":
        return False
    raise ValueError(
        f"{context}: a saved param_group has weight_decay = {weight_decay} but neither the "
        "group (no 'decoupled_weight_decay' entry; older torch versions do not save one) nor "
        "metadata.json (no optimizer name) says whether the optimizer was Adam (decay "
        "added to the gradient) or AdamW (decoupled decay). The two give different "
        "updates; record the optimizer name in metadata.json."
    )


def _stored_weight_dtypes(path: Path, names: Sequence[str]) -> set[str]:
    """The safetensors dtype tags (``"F32"``, ``"BF16"``, ...) of ``names`` in ``path``.

    Read from the file header only, so no tensor is loaded.
    """
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        try:
            header = json.loads(handle.read(size))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BundleSchemaError(f"{path}: not a safetensors file ({exc})") from exc
    missing = [name for name in names if name not in header]
    if missing:
        raise LayoutMismatchError(
            f"{path}: no weights for canonical parameter(s) {_abbrev(missing)}"
        )
    return {str(header[name]["dtype"]) for name in names}


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
