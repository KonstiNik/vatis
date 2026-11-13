"""The canonical parameter flattening order — one mapping, one code path.

Every vatis quantity that lives in *parameter space* is a flat 1-D vector:
the gradient (``vatis.core.observables.parameter_grad_vector``, the analyzer's
``_zero_param_vector`` / ``_add_grads_into_flat``), and — for the
preconditioned ruler — the diagonal preconditioner built from a saved
optimizer state. Those vectors are only comparable if **every one of them uses
the same order and the same segment boundaries**.

This module owns that order.

    layout = ParameterLayout.from_json(run_dir / "param_names.json")
    params = layout.parameters_in_order(model)   # -> ModelBundle.params
    v      = layout.flatten_named(state_tensors) # -> 1-D vector, same segments

A :class:`ParameterLayout` is an ordered list of ``(name, shape)`` records.
Concatenating ``reshape(-1)`` of each entry, in list order, defines:

- the *flat index* of every parameter element, and hence
- the segment ``[offset(name), offset(name) + numel(name))`` that belongs to
  each named parameter.

For a checkpoint bundle (``docs/SPEC.md`` §"Native checkpoint bundles") the records
come from
``param_names.json``, which is *by definition* the canonical order: the saved
optimizer ``state_dict`` indices are index-aligned with it. vatis therefore
builds ``ModelBundle.params`` by looking parameters up **by name in that
order** (:meth:`ParameterLayout.parameters_in_order`) rather than by iterating
``model.named_parameters()`` — the two coincide for a bundle written by our
own trainer, but relying on that coincidence is exactly the ordering bug that
would silently corrupt every chi_pos^P number.

The module knows nothing about HF, safetensors, DDP or sinks — it takes plain
tensors, ``(name, shape)`` records and ``nn.Module``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LayoutMismatchError(ValueError):
    """Raised when something does not line up with the canonical order.

    Every raise site describes *which* invariant broke and names the offending
    parameter(s). This is deliberately a hard failure: a silent fallback here
    would produce plausible-looking numbers computed on a permuted vector.
    """


class ParameterLayout:
    """An ordered ``(name, shape)`` list defining THE flattening order.

    Construct from ``param_names.json`` (:meth:`from_json`), from raw records
    (:meth:`from_records`), or from a model (:meth:`from_named_parameters`,
    which reproduces the writer-side convention
    ``[(n, tuple(p.shape)) for n, p in model.named_parameters()]``).

    Instances are immutable and cheap to copy around.
    """

    __slots__ = ("_names", "_offsets", "_shapes", "_source", "_total_numel")

    def __init__(
        self,
        records: Sequence[tuple[str, tuple[int, ...]]],
        *,
        source: str = "<records>",
    ) -> None:
        names: list[str] = []
        shapes: list[tuple[int, ...]] = []
        offsets: list[int] = []
        seen: dict[str, int] = {}
        offset = 0
        for position, (name, shape) in enumerate(records):
            if name in seen:
                raise LayoutMismatchError(
                    f"{source}: duplicate parameter name {name!r} at positions "
                    f"{seen[name]} and {position}; the canonical order must list "
                    "each parameter exactly once"
                )
            seen[name] = position
            names.append(name)
            shapes.append(shape)
            offsets.append(offset)
            offset += _numel(shape)
        if not names:
            raise LayoutMismatchError(f"{source}: the canonical order is empty")
        self._names: tuple[str, ...] = tuple(names)
        self._shapes: tuple[tuple[int, ...], ...] = tuple(shapes)
        self._offsets: tuple[int, ...] = tuple(offsets)
        self._total_numel: int = offset
        self._source: str = source

    # --------------------------------------------------------------- builders

    @classmethod
    def from_records(cls, records: Any, *, source: str = "<records>") -> ParameterLayout:
        """Build a layout from the frozen ``[[name, [dim, ...]], ...]`` form.

        The record list is validated strictly — a malformed
        ``param_names.json`` is a broken interface between the bundle writer
        and vatis, not something to guess around.
        """
        if isinstance(records, Mapping) or not isinstance(records, Sequence):
            raise LayoutMismatchError(
                f"{source}: expected a JSON list of [name, shape] pairs, got "
                f"{type(records).__name__}"
            )
        parsed: list[tuple[str, tuple[int, ...]]] = []
        for position, record in enumerate(records):
            if isinstance(record, str) or not isinstance(record, Sequence) or len(record) != 2:
                raise LayoutMismatchError(
                    f"{source}: entry {position} is not a [name, shape] pair: {record!r}"
                )
            name, raw_shape = record[0], record[1]
            if not isinstance(name, str):
                raise LayoutMismatchError(
                    f"{source}: entry {position} has a non-string name: {name!r}"
                )
            if isinstance(raw_shape, str) or not isinstance(raw_shape, Sequence):
                raise LayoutMismatchError(
                    f"{source}: entry {position} ({name!r}) has a non-list shape: {raw_shape!r}"
                )
            dims: list[int] = []
            for dim in raw_shape:
                if isinstance(dim, bool) or not isinstance(dim, int):
                    raise LayoutMismatchError(
                        f"{source}: entry {position} ({name!r}) has a non-integer "
                        f"dimension: {raw_shape!r}"
                    )
                dims.append(int(dim))
            parsed.append((name, tuple(dims)))
        return cls(parsed, source=source)

    @classmethod
    def from_json(cls, path: str | Path) -> ParameterLayout:
        """Read ``param_names.json`` (the frozen bundle schema)."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        return cls.from_records(records, source=str(p))

    @classmethod
    def from_named_parameters(cls, model: nn.Module) -> ParameterLayout:
        """Build the layout a bundle writer would record for ``model``.

        This is the writer-side convention of the bundle schema: ordered
        ``[name, shape]`` from ``model.named_parameters()``, which iterates in
        the same order as ``model.parameters()`` — hence index-aligned with a
        ``torch.optim`` ``state_dict`` built from ``model.parameters()``. Tied
        parameters appear once (``named_parameters`` de-duplicates shared
        storages), which is also what the optimizer sees.
        """
        records = [(name, tuple(p.shape)) for name, p in model.named_parameters()]
        return cls(records, source=f"{type(model).__name__}.named_parameters()")

    # -------------------------------------------------------------- accessors

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names, in canonical order."""
        return self._names

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Parameter shapes, in canonical order."""
        return self._shapes

    @property
    def total_numel(self) -> int:
        """Length of the flat vector this layout describes."""
        return self._total_numel

    @property
    def source(self) -> str:
        """Where the layout came from (used in error messages)."""
        return self._source

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self) -> Iterator[tuple[str, tuple[int, ...]]]:
        return iter(zip(self._names, self._shapes, strict=True))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ParameterLayout(n_params={len(self._names)}, "
            f"total_numel={self._total_numel}, source={self._source!r})"
        )

    def index_of(self, name: str) -> int:
        """Position of ``name`` in the canonical order."""
        try:
            return self._names.index(name)
        except ValueError as exc:
            raise LayoutMismatchError(f"{self._source}: unknown parameter name {name!r}") from exc

    def shape_of(self, name: str) -> tuple[int, ...]:
        """Recorded shape of ``name``."""
        return self._shapes[self.index_of(name)]

    def numel_of(self, name: str) -> int:
        """Number of elements of ``name``."""
        return _numel(self.shape_of(name))

    def offset_of(self, name: str) -> int:
        """Flat-vector offset at which ``name``'s segment starts."""
        return self._offsets[self.index_of(name)]

    def slice_of(self, name: str) -> slice:
        """Half-open flat-vector slice belonging to ``name``."""
        i = self.index_of(name)
        return slice(self._offsets[i], self._offsets[i] + _numel(self._shapes[i]))

    def segment(self, flat: torch.Tensor, name: str) -> torch.Tensor:
        """Return the (view of the) segment of ``flat`` belonging to ``name``.

        ``flat`` must be 1-D with ``numel == total_numel`` — i.e. produced by
        this layout. That is checked, so passing a vector from a different
        layout fails loudly instead of silently slicing the wrong elements.
        """
        self.validate_flat(flat, context=f"segment({name!r})")
        return flat[self.slice_of(name)]

    # ------------------------------------------------------------ validation

    def validate_flat(self, flat: torch.Tensor, *, context: str = "flat vector") -> None:
        """Raise unless ``flat`` is a 1-D vector of length :attr:`total_numel`."""
        if flat.ndim != 1 or flat.numel() != self._total_numel:
            raise LayoutMismatchError(
                f"{context}: expected a 1-D vector of length {self._total_numel} "
                f"({self._source}), got shape {tuple(flat.shape)}"
            )

    def validate_params(
        self,
        params: Sequence[torch.nn.Parameter] | Sequence[torch.Tensor],
        *,
        context: str = "parameter list",
    ) -> None:
        """Raise unless ``params`` is element-for-element this layout.

        This is the assertion that ties vatis's gradient flattening to the
        canonical order: ``parameter_grad_vector(params)`` concatenates
        ``params`` in list order, so if the list matches the layout position by
        position (count *and* shape), the resulting flat gradient has exactly
        the segments this layout describes.
        """
        if len(params) != len(self._names):
            raise LayoutMismatchError(
                f"{context}: has {len(params)} parameters but the canonical order "
                f"({self._source}) has {len(self._names)}"
            )
        for i, (param, (name, shape)) in enumerate(zip(params, self, strict=True)):
            if tuple(param.shape) != shape:
                raise LayoutMismatchError(
                    f"{context}: parameter {i} ({name!r}) has shape "
                    f"{tuple(param.shape)}, canonical order says {shape}"
                )

    def parameters_in_order(
        self,
        model: nn.Module,
        *,
        allow_unlisted: bool = False,
    ) -> list[torch.nn.Parameter]:
        """Look up ``model``'s parameters **by name, in canonical order**.

        This is the single place where a model is turned into the parameter
        list vatis flattens (``ModelBundle.params``). Errors:

        - a canonical name the model does not have → :class:`LayoutMismatchError`
          (wrong factory / stale ``param_names.json``);
        - a shape disagreement → :class:`LayoutMismatchError`;
        - a *model* parameter the canonical order does not list → also an error,
          because such a parameter would be silently excluded from every
          parameter-space vector. Pass ``allow_unlisted=True`` to accept that
          (the unlisted parameters are then simply not measured).

        Tied weights need no special handling but are a likely source of the
        first error: ``named_parameters()`` (and the optimizer) see a shared
        tensor once, under its first name, while ``state_dict()`` lists it under
        every name. A canonical order recorded from ``state_dict()`` keys
        therefore names a parameter the model does not have — reported below.
        """
        available: dict[str, torch.nn.Parameter] = dict(model.named_parameters())
        missing = [name for name in self._names if name not in available]
        if missing:
            shared = sorted(set(missing) & set(model.state_dict()))
            hint = (
                f" ({shared[:4]} exist in state_dict() but not in named_parameters(), "
                "i.e. they are tied to another parameter and must be listed once)"
                if shared
                else "; the model factory does not match this bundle"
            )
            raise LayoutMismatchError(
                f"{self._source}: model {type(model).__name__} has no parameter(s) "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}{hint}"
            )
        if not allow_unlisted:
            unlisted = [name for name in available if name not in set(self._names)]
            if unlisted:
                raise LayoutMismatchError(
                    f"{self._source}: model {type(model).__name__} has parameter(s) "
                    f"{unlisted[:8]}{' ...' if len(unlisted) > 8 else ''} that the "
                    "canonical order does not list; they would be excluded from "
                    "every parameter-space vector. Pass allow_unlisted=True only "
                    "if you intend not to measure them."
                )
        params = [available[name] for name in self._names]
        self.validate_params(params, context=f"{type(model).__name__} parameters")
        return params

    # ------------------------------------------------------------- flattening

    def flatten_named(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Concatenate per-parameter tensors into one flat vector.

        Args:
            tensors: mapping name → tensor. Every canonical name must be
                present and every tensor must have the recorded shape;
                anything else raises :class:`LayoutMismatchError`. Extra keys
                are ignored (they are not part of the canonical order).
            dtype: dtype of the result. ``None`` keeps the inputs' dtype
                (which must then be uniform, as ``torch.cat`` requires).
            device: device of the result. ``None`` keeps the inputs' device.

        Returns:
            A 1-D tensor of length :attr:`total_numel`, segment-aligned with
            this layout.
        """
        missing = [name for name in self._names if name not in tensors]
        if missing:
            raise LayoutMismatchError(
                f"{self._source}: no tensor supplied for {missing[:8]}"
                f"{' ...' if len(missing) > 8 else ''}"
            )
        flats: list[torch.Tensor] = []
        for name, shape in self:
            t = tensors[name]
            if tuple(t.shape) != shape:
                raise LayoutMismatchError(
                    f"{self._source}: tensor for {name!r} has shape {tuple(t.shape)}, "
                    f"canonical order says {shape}"
                )
            flat = t.detach().reshape(-1)
            if dtype is not None:
                flat = flat.to(dtype=dtype)
            if device is not None:
                flat = flat.to(device=device)
            flats.append(flat)
        return torch.cat(flats)

    def to_records(self) -> list[list[Any]]:
        """Serialize back to the frozen ``[[name, [dim, ...]], ...]`` form."""
        return [[name, list(shape)] for name, shape in self]


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n
