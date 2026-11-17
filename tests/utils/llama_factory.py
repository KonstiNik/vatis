"""The project's Llama architecture as a bundle model factory — when available.

``03_estimator.md`` §Golden asks the golden gate to run over two architectures:
the self-contained tiny transformer of :mod:`tests.utils.bundle_fixture` (always)
and the project's Llama (tied weights, rotary embeddings — the realistic case),
**skipped cleanly when the path is unavailable** so the vatis suite stays
self-contained after merge.

This is the one place in the test suite that reaches outside the repository, and
only in the way ``00_context.md`` §Workspace permits: "vatis *package* code must
never depend on antonio_code or the RO repo; vatis **test fixtures may** import
the Llama model via a configurable factory path". The location is configurable
via ``VATIS_LLAMA_MODEL_PATH``; nothing is imported at module import time, so a
machine without the training stack pays nothing.

Two details that matter for the bundle interface:

- **The factory wraps the ``Transformer`` in a module with a ``model``
  attribute**, so ``named_parameters()`` yields ``model.tok_embeddings.weight``
  and friends. The project's own bundles are written from a LightningModule and
  therefore record ``model.``-prefixed names in ``param_names.json``; a factory
  returning a bare ``Transformer`` makes the loader raise
  ``LayoutMismatchError`` (correctly). Mirroring the prefix here means the golden
  gate exercises the naming a real project bundle has.
- **Tying is applied by the factory.** ``llama_model.model.Transformer`` ships
  with the ``output.weight = tok_embeddings.weight`` line commented out, so
  tying is switched on here (driven by the bundle's own
  ``model_config["tie_head"]``) rather than by editing the read-only training
  stack. That is what makes the tied-weight case — one tensor,
  ``named_parameters()`` reporting it once and ``state_dict()`` twice — actually
  covered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch.nn as nn

LLAMA_PATH_ENV = "VATIS_LLAMA_MODEL_PATH"
DEFAULT_LLAMA_PATH = "/workspace/antonio_code"

# Kept small: the golden gate trains this for ~100 CPU steps.
DEFAULT_LLAMA_CONFIG: dict[str, Any] = {
    "dim": 64,
    "n_layers": 2,
    "n_heads": 2,
    "vocab_size": 48,
    "multiple_of": 32,
    "max_seq_len": 32,
    "norm_eps": 1e-5,
    "dropout": 0.0,
}


def llama_repo_path() -> Path | None:
    """The directory containing ``llama_model/``, or ``None`` if unavailable."""
    candidate = Path(os.environ.get(LLAMA_PATH_ENV) or DEFAULT_LLAMA_PATH)
    return candidate if (candidate / "llama_model" / "model.py").is_file() else None


def llama_available() -> bool:
    """Whether the project Llama can be imported on this machine."""
    return llama_repo_path() is not None


class LlamaBundleModel(nn.Module):
    """``self.model = llama_model.Transformer(...)`` — the ``model.``-prefixed shape.

    ``forward(input_ids) -> logits`` so it works with
    :func:`vatis.models.local_bundle.plain_lm_forward` unchanged.
    """

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.model = transformer

    def forward(self, input_ids: Any) -> Any:
        return self.model(input_ids)


def build_llama_model(metadata: Any) -> nn.Module:
    """``model_factory(metadata) -> nn.Module`` for the project Llama.

    Raises ``ImportError`` when the training stack is not on disk — callers
    should gate on :func:`llama_available` and skip.
    """
    root = llama_repo_path()
    if root is None:
        raise ImportError(
            f"the project Llama is not available (looked for {DEFAULT_LLAMA_PATH}/llama_model; "
            f"override with {LLAMA_PATH_ENV})"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from llama_model import ModelArgs, Transformer  # type: ignore[import-not-found]

    getter = getattr(metadata, "get", None)
    raw = getter("model_config", {}) if callable(getter) else {}
    config = dict(raw) if isinstance(raw, dict) else {}
    args = ModelArgs(
        **{key: config.get(key, default) for key, default in DEFAULT_LLAMA_CONFIG.items()}
    )
    transformer = Transformer(args)
    if bool(config.get("tie_head", True)):
        transformer.output.weight = transformer.tok_embeddings.weight
    return LlamaBundleModel(transformer)


# The ``model_config`` a bundle must record for :func:`build_llama_model`.
def llama_model_config(*, vocab_size: int, seq_len: int, tie_head: bool = True) -> dict[str, Any]:
    """A ``model_config`` block for the Llama factory, sized for the gate."""
    config = dict(DEFAULT_LLAMA_CONFIG)
    config["vocab_size"] = vocab_size
    # Rotary tables are precomputed for max_seq_len * 2 positions; keep headroom.
    config["max_seq_len"] = max(int(DEFAULT_LLAMA_CONFIG["max_seq_len"]), seq_len)
    config["tie_head"] = tie_head
    config["seq_len"] = seq_len
    return config
