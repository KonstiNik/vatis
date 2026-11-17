"""The inputs of the Llama example: the model factory and the measurement batches.

Support code for ``run.py``, moved out of it so that the script shows the vatis
calls. Nothing here calls the analyzer.

- ``llama_factory`` rebuilds the training repository's Llama for
  ``vatis.load_local_bundle``. Code outside this repository imports it from
  here: put this folder on ``sys.path`` and ``from llama_bundle import
  llama_factory``.
- ``load_measurement_batches`` reads the held-out ``eval_batch_*.npy`` files
  (or falls back, loudly, to random tokens) and scores them with the pad
  convention the bundle records.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vatis.data.collate import shifted_lm_targets  # noqa: E402

# The three held-out batches, in the order the training script dumps them:
# two measurement batches and one eval batch. Names land in the parquet.
BATCH_NAMES = ("A1", "A2", "B")
EVAL_BATCH_GLOB = "eval_batch_*.npy"
FALLBACK_MARKER = "DATA_FALLBACK"


# --------------------------------------------------------------- model factory


class WrappedLlama(torch.nn.Module):
    """The project Llama under the ``model.`` parameter-name prefix.

    The bundle's ``param_names.json`` records the *training* LightningModule's
    names (``model.tok_embeddings.weight``, ...) because that is the module the
    optimizer was built from. A factory returning a bare ``Transformer`` is
    therefore rejected by the loader's canonical-layout check — correctly, since
    a silent name mismatch would mean a permuted P. The fix belongs in the
    factory, which is exactly what this wrapper is.

    ``forward(tokens) -> logits`` keeps
    :func:`vatis.models.local_bundle.plain_lm_forward` working unchanged.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.model = inner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(tokens)
        return out


def llama_factory(metadata: Any) -> torch.nn.Module:
    """``model_factory(metadata) -> nn.Module`` for the project Llama.

    vatis stays project-agnostic: the architecture is described by the bundle's
    own ``model_config`` and instantiated by user code (this function), never by
    vatis. The import needs the training repo's *pure-torch model definition*
    only — no lightning, no perspic, nothing that would pull the training stack's
    torch build into this process.
    """
    llama_src = os.environ.get("WP0_LLAMA_SRC", "/workspace/antonio_code")
    if llama_src not in sys.path:
        sys.path.insert(0, llama_src)
    try:
        from llama_model import ModelArgs, Transformer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"cannot import the project Llama from {llama_src!r} (needs llama_model/model.py). "
            "Point WP0_LLAMA_SRC at the training repo."
        ) from exc
    config = dict(metadata["model_config"])
    return WrappedLlama(Transformer(ModelArgs(**config)))


# ------------------------------------------------------------------- batches


def lm_batch_from_tokens(
    ids: torch.Tensor, loss_ignore_index: int | None
) -> dict[str, torch.Tensor]:
    """``[B, seq_len]`` token ids -> a vatis LM batch under the run's pad convention.

    The same construction the bundle loader applies to ``batch_a.npy``
    (``LocalBundleCheckpoint.batch_a``): pre-shifted next-token labels, last
    position ignored, and — when the training run recorded that its loss ignored
    a target id (``loss_ignore_index``, the pad id) — ``-100`` wherever the target
    equals that id. With ``loss_ignore_index=None`` every target counts.
    """
    labels = shifted_lm_targets(ids)
    if loss_ignore_index is not None:
        labels = labels.masked_fill(labels == loss_ignore_index, -100)
    return {"input_ids": ids, "labels": labels}


def load_measurement_batches(
    directory: Path,
    *,
    seq_len: int,
    vocab_size: int,
    batch_size: int,
    device: torch.device,
    loss_ignore_index: int | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], bool]:
    """The held-out measurement batches, or a loud random-token fallback.

    Files are ``[B, seq_len]`` int32 token ids in the same whole-sequence
    convention as the bundle's ``batch_a.npy``, so a batch is turned into an LM
    batch the same way the loader does it (``labels`` = shift by one, last
    position ignored).

    Padding follows the **bundle's recorded pad convention**
    (``loss_ignore_index = source.loss_ignore_index``), exactly as the loader
    applies it to the update batch — so the held-out rows and the update-batch
    rows describe the same objective, the one the run descended. A bundle whose
    run masked padding in its loss gets the pad targets ignored here too; a
    bundle that records no such index gets every target counted, as before.
    Measuring a different objective than
    the one that was trained would be the larger error in either direction.

    Returns ``(batches, used_random_fallback)``.
    """
    paths = sorted(directory.glob(EVAL_BATCH_GLOB)) if directory.is_dir() else []
    if len(paths) >= len(BATCH_NAMES):
        batches: dict[str, dict[str, torch.Tensor]] = {}
        for name, path in zip(BATCH_NAMES, paths[: len(BATCH_NAMES)], strict=True):
            array = np.load(path, allow_pickle=False)
            ids = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.int64, device=device)
            if ids.ndim != 2:
                raise ValueError(
                    f"{path}: expected a 2-D [B, seq_len] token array, got {ids.shape}"
                )
            batches[name] = lm_batch_from_tokens(ids, loss_ignore_index)
            print(f"    {name}: {path.name} shape={tuple(ids.shape)}")
        return batches, False

    # Spec 05 §1: no shard -> random tokens, LOUDLY, and logged. This path is a
    # pipeline demo only; chi_net is the Jacobian norm *evaluated at the input*,
    # so off-distribution tokens make every value scientifically empty.
    banner = "#" * 78
    message = (
        f"{banner}\n"
        f"##  WARNING: RANDOM-TOKEN FALLBACK -- THIS ANALYSIS IS SCIENTIFICALLY EMPTY.\n"
        f"##  No {EVAL_BATCH_GLOB} files under {directory} (found {len(paths)}, "
        f"need {len(BATCH_NAMES)}), so the measurement batches are uniformly\n"
        f"##  random token ids. The pipeline is exercised end to end; the numbers\n"
        f"##  mean nothing. Re-run the training script to dump the held-out batches.\n"
        f"{banner}"
    )
    print(message, flush=True)
    warnings.warn(
        "random-token measurement batches: no held-out shard found, values are "
        "scientifically empty (pipeline demo only)",
        UserWarning,
        stacklevel=2,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260728)
    fallback: dict[str, dict[str, torch.Tensor]] = {}
    for name in BATCH_NAMES:
        ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.int64
        ).to(device)
        fallback[name] = lm_batch_from_tokens(ids, loss_ignore_index)
    return fallback, True
