"""``EvalBatchSpec`` — a uniform handle on the three eval-batch flavors.

The user can pass any of:
    - a frozen tensor / dict / tuple, reused at every checkpoint;
    - a DataLoader, redrawn at each checkpoint;
    - a callable ``fn(checkpoint_id) -> batch``, computed per checkpoint.

The analyzer iterates over named EvalBatchSpecs and asks each one for "the
batch for this checkpoint" — the spec hides whether that means returning the
fixed tensor, draining one minibatch from the loader, or invoking the user
callable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch

# A "raw" batch as accepted from the user. Tensors / dicts / tuples are all
# valid; the analyzer normalizes them via ``vatis.data.collate``.
RawBatch = torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | list[torch.Tensor]


class _DataLoaderLike(Protocol):
    def __iter__(self) -> Iterator[Any]: ...


@dataclass
class EvalBatchSpec:
    """A handle on one named eval batch.

    Use the factory methods rather than the raw constructor:

    - :meth:`fixed` — frozen, reused.
    - :meth:`resample` — drains the loader once per checkpoint.
    - :meth:`from_callable` — invokes ``fn(checkpoint_id)``.

    The internal representation is one of three modes; only one of
    ``_fixed_batch``, ``_loader``, ``_callable`` is set.
    """

    mode: str  # one of {"fixed", "resample", "callable"}
    _fixed_batch: RawBatch | None = None
    _loader: _DataLoaderLike | None = None
    _callable: Callable[[str], RawBatch] | None = None
    # Internal iter cached for "resample" mode so we can advance through the
    # loader across checkpoints without re-creating it.
    _loader_iter: Iterator[Any] | None = None

    @classmethod
    def fixed(cls, batch: RawBatch) -> EvalBatchSpec:
        """Return a frozen-batch spec, reused at every checkpoint."""
        return cls(mode="fixed", _fixed_batch=batch)

    @classmethod
    def resample(cls, loader: _DataLoaderLike) -> EvalBatchSpec:
        """Return a spec that draws ``next(iter(loader))`` per checkpoint."""
        return cls(mode="resample", _loader=loader)

    @classmethod
    def from_callable(cls, fn: Callable[[str], RawBatch]) -> EvalBatchSpec:
        """Return a spec that invokes ``fn(checkpoint_id)`` per checkpoint."""
        return cls(mode="callable", _callable=fn)

    @classmethod
    def auto(cls, value: Any) -> EvalBatchSpec:
        """Auto-wrap a user value into the right spec.

        - already-an-EvalBatchSpec: passthrough.
        - tensor / dict / tuple / list: ``fixed``.
        - callable: ``from_callable``.
        - object with ``__iter__`` (DataLoader-like): ``resample``.
        """
        if isinstance(value, EvalBatchSpec):
            return value
        if isinstance(value, (torch.Tensor, dict, tuple, list)):
            return cls.fixed(value)
        if callable(value):
            return cls.from_callable(value)
        if hasattr(value, "__iter__"):
            return cls.resample(value)
        raise TypeError(
            f"cannot auto-wrap value of type {type(value).__name__} into "
            "EvalBatchSpec; use the factory methods explicitly."
        )

    def get_batch(self, checkpoint_id: str) -> RawBatch:
        """Return the concrete batch to use for this checkpoint."""
        if self.mode == "fixed":
            assert self._fixed_batch is not None
            return self._fixed_batch
        if self.mode == "resample":
            assert self._loader is not None
            # Each checkpoint pulls one fresh minibatch. We cache the iterator
            # so we don't reset to the loader's beginning every time. When the
            # iterator is exhausted, restart.
            if self._loader_iter is None:
                self._loader_iter = iter(self._loader)
            try:
                return next(self._loader_iter)  # type: ignore[no-any-return]
            except StopIteration:
                self._loader_iter = iter(self._loader)
                return next(self._loader_iter)  # type: ignore[no-any-return]
        if self.mode == "callable":
            assert self._callable is not None
            return self._callable(checkpoint_id)
        raise ValueError(f"unknown EvalBatchSpec mode: {self.mode!r}")


def normalize_eval_batches(
    eval_batches: dict[str, Any],
) -> dict[str, EvalBatchSpec]:
    """Wrap each value in ``eval_batches`` into an :class:`EvalBatchSpec`."""
    return {name: EvalBatchSpec.auto(v) for name, v in eval_batches.items()}
