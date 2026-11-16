"""The bundle loader applies the pad convention the training run recorded.

A training run that masks padding in its loss — ``F.cross_entropy(logits,
targets, ignore_index=<pad id>)``, a mean over the non-pad targets — records
that in ``metadata.json`` as ``mask_pad_in_loss: true`` and ``loss_ignore_index:
<pad id>``. ``LocalBundleCheckpoint.batch_a`` used to ignore both keys and count
every target, so for such a run the update-batch rows described a loss that
includes the pad targets: not the loss the run descended.

The loader now writes ``-100`` into the ``batch_a`` labels wherever the
next-token target equals the recorded index — the labels are the single source
of truth, no ``valid_mask_fn`` is involved.

References here share no code with the loader: the bundles are written by
``tests/fixtures/synthetic_bundle.py`` (which imports nothing from it), the
tokens are read back from ``batch_a.npy`` with numpy, and the masked loss and its
gradient are recomputed with plain ``torch.nn.functional.cross_entropy`` in the
training run's own form, ``inputs = ids[:, :-1]``, ``targets = ids[:, 1:]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.fixtures.synthetic_bundle import (
    DEFAULT_VOCAB_SIZE,
    BundleTinyLM,
    bundle_metadata,
    handwritten_optim_state,
    tiny_lm_factory,
    write_checkpoint,
    write_run_files,
)
from vatis import Analyzer, load_local_bundle
from vatis.models.local_bundle import BundleSchemaError

PAD_ID = 3
STEPS = (1, 2)
_ABSENT = object()


def _padded_batch(step: int, *, batch_size: int = 4, seq_len: int = 7) -> np.ndarray:
    """Right-padded token rows; ``PAD_ID`` also occurs as an ordinary input token.

    Row 0 is unpadded, the others end in 1-3 pad tokens, and one interior token of
    row 0 is set to ``PAD_ID`` so that "pad as a target in the middle of a
    sequence" is exercised too (the training criterion ignores a target by value,
    wherever it sits).
    """
    rng = np.random.default_rng(100 + step)
    ids = rng.integers(0, DEFAULT_VOCAB_SIZE, size=(batch_size, seq_len), dtype=np.int32)
    ids[ids == PAD_ID] = PAD_ID + 1
    for row in range(1, batch_size):
        ids[row, seq_len - row :] = PAD_ID
    ids[0, 2] = PAD_ID
    return ids


def _write_bundle(
    run_dir: Path,
    *,
    mask_pad_in_loss: Any = _ABSENT,
    loss_ignore_index: Any = _ABSENT,
) -> BundleTinyLM:
    model = BundleTinyLM(seq_len=7, seed=0)
    metadata = bundle_metadata(model, batch_size=4)
    if mask_pad_in_loss is not _ABSENT:
        metadata["mask_pad_in_loss"] = mask_pad_in_loss
    if loss_ignore_index is not _ABSENT:
        metadata["loss_ignore_index"] = loss_ignore_index
    order = write_run_files(run_dir, model, metadata=metadata)
    for step in STEPS:
        write_checkpoint(
            run_dir,
            step,
            model=model,
            optim_state=handwritten_optim_state(model, order, step=step),
            batch_a=_padded_batch(step),
        )
    return model


def _tokens_on_disk(run_dir: Path, step: int) -> torch.Tensor:
    path = run_dir / "checkpoints" / f"step_{step:08d}" / "batch_a.npy"
    return torch.from_numpy(np.load(path)).to(torch.int64)


def _plain_torch_reference(
    model: torch.nn.Module, ids: torch.Tensor, *, ignore_index: int | None
) -> dict[str, float]:
    """Loss, ``n``, ``||dL/dz||^2`` and ``||dL/dtheta||^2`` in the training run's form.

    ``inputs = ids[:, :-1]``, ``targets = ids[:, 1:]`` and
    ``F.cross_entropy(..., ignore_index=<pad id>)`` when the run masked padding,
    plain ``F.cross_entropy`` when it did not. Feeding the truncated inputs is
    exact for any causal model; ``BundleTinyLM`` is position-wise.
    """
    inputs, targets = ids[:, :-1], ids[:, 1:]
    logits = model(inputs)
    flat_logits, flat_targets = logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    if ignore_index is None:
        loss = F.cross_entropy(flat_logits, flat_targets)
        n = int(flat_targets.numel())
    else:
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)
        n = int((flat_targets != ignore_index).sum())
    (grad_logits,) = torch.autograd.grad(loss, logits, retain_graph=True)
    grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return {
        "loss": float(loss.detach()),
        "n": float(n),
        "chi_loss": float((grad_logits.double() ** 2).sum()),
        "delta_loss": sum(float((g.double() ** 2).sum()) for g in grads if g is not None),
    }


# -------------------------------------------------------------- the labels


def test_recorded_pad_id_is_ignored_in_the_update_batch_labels(tmp_path: Path) -> None:
    _write_bundle(tmp_path / "run", mask_pad_in_loss=True, loss_ignore_index=PAD_ID)
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    assert source.loss_ignore_index == PAD_ID
    assert source.metadata.loss_ignore_index == PAD_ID
    for checkpoint in source:
        ids = _tokens_on_disk(tmp_path / "run", checkpoint.step)
        batch = checkpoint.batch_a
        assert set(batch) == {"input_ids", "labels"}
        assert torch.equal(batch["input_ids"], ids)  # inputs untouched, pads included
        assert torch.equal(checkpoint.batch_a_tokens, ids)
        n_ignored = 0
        for b in range(ids.shape[0]):
            for s in range(ids.shape[1]):
                target = int(ids[b, s + 1]) if s + 1 < ids.shape[1] else None
                expected = -100 if target is None or target == PAD_ID else target
                assert int(batch["labels"][b, s]) == expected, (checkpoint.step, b, s)
                n_ignored += int(target == PAD_ID)
        assert n_ignored == 1 + 2 + 3 + 1  # the padded tails + the interior pad of row 0
        # A pad token that is merely an *input* still has its own target counted.
        assert int(batch["labels"][0, 2]) == int(ids[0, 3]) != PAD_ID

        # Released and re-read: the same labels again.
        checkpoint.release()
        assert torch.equal(checkpoint.batch_a["labels"], batch["labels"])


@pytest.mark.parametrize(
    "keys",
    [
        {},
        {"mask_pad_in_loss": False},
        {"loss_ignore_index": None},
        {"mask_pad_in_loss": False, "loss_ignore_index": None},
        {"mask_pad_in_loss": None, "loss_ignore_index": None},
    ],
    ids=["absent", "false", "null-index", "false+null", "null+null"],
)
def test_without_a_recorded_index_every_target_counts(tmp_path: Path, keys: dict[str, Any]) -> None:
    """Absent or null: exactly the labels the loader produced before the keys existed."""
    _write_bundle(tmp_path / "run", **keys)
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    # ``getattr`` so this test also runs against a loader that predates the
    # attribute: "nothing recorded" must behave exactly as it did then.
    assert getattr(source, "loss_ignore_index", None) is None
    for checkpoint in source:
        ids = _tokens_on_disk(tmp_path / "run", checkpoint.step)
        expected = torch.full_like(ids, -100)
        expected[:, :-1] = ids[:, 1:]
        assert torch.equal(checkpoint.batch_a["labels"], expected)
        assert int((checkpoint.batch_a["labels"] != -100).sum()) == ids.shape[0] * (
            ids.shape[1] - 1
        )


# ---------------------------------------- the observables of the update batch


@pytest.mark.parametrize("micro", [1, 2, 4])
def test_update_batch_rows_describe_the_masked_loss_the_run_descended(
    tmp_path: Path, micro: int
) -> None:
    """``n_valid``, ``chi_loss`` and ``delta_loss`` against plain torch on the tokens.

    Pre-fix the rows matched the *unmasked* reference instead (also computed
    here, and asserted to be a different number so the test can tell the two
    apart).
    """
    _write_bundle(tmp_path / "run", mask_pad_in_loss=True, loss_ignore_index=PAD_ID)
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    analyzer = Analyzer(
        eval_batches={"train": source.batch_a_spec()},
        chi_net_method="hutchinson",
        n_hutchinson=1,
        micro_batch_size=micro,
        sink=None,
    )
    results = analyzer.run(source.model_specs())
    assert len(results) == len(STEPS)
    for result, checkpoint in zip(results, source.checkpoints, strict=True):
        ids = _tokens_on_disk(tmp_path / "run", checkpoint.step)
        model = checkpoint.model_bundle.model
        masked = _plain_torch_reference(model, ids, ignore_index=PAD_ID)
        unmasked = _plain_torch_reference(model, ids, ignore_index=None)
        assert masked["n"] == unmasked["n"] - 7
        assert abs(masked["delta_loss"] - unmasked["delta_loss"]) > 0.05 * masked["delta_loss"]
        assert abs(masked["chi_loss"] - unmasked["chi_loss"]) > 0.05 * masked["chi_loss"]

        values = {r.observable: r.value for r in result.rows}
        assert {r.n_valid_a for r in result.rows} == {int(masked["n"])}
        assert values["chi_loss"] == pytest.approx(masked["chi_loss"], rel=1e-5)
        assert values["delta_loss"] == pytest.approx(masked["delta_loss"], rel=1e-5)

        # And the loss the shipped loss_fn computes on the loader's batch is the
        # training criterion's value.
        bundle = checkpoint.model_bundle
        batch = checkpoint.batch_a
        with torch.no_grad():
            loss = bundle.loss_fn(bundle.forward_fn(bundle.model, batch), batch)
        assert float(loss) == pytest.approx(masked["loss"], rel=1e-6)


def test_an_unmasked_run_keeps_its_unmasked_rows(tmp_path: Path) -> None:
    """No recorded index: the rows are those of the plain mean over all targets."""
    _write_bundle(tmp_path / "run")
    source = load_local_bundle(tmp_path / "run", tiny_lm_factory)
    analyzer = Analyzer(
        eval_batches={"train": source.batch_a_spec()},
        chi_net_method="hutchinson",
        n_hutchinson=1,
        micro_batch_size=2,
        sink=None,
    )
    for result, checkpoint in zip(
        analyzer.run(source.model_specs()), source.checkpoints, strict=True
    ):
        ids = _tokens_on_disk(tmp_path / "run", checkpoint.step)
        unmasked = _plain_torch_reference(checkpoint.model_bundle.model, ids, ignore_index=None)
        values = {r.observable: r.value for r in result.rows}
        assert {r.n_valid_a for r in result.rows} == {int(unmasked["n"])}
        assert values["chi_loss"] == pytest.approx(unmasked["chi_loss"], rel=1e-5)
        assert values["delta_loss"] == pytest.approx(unmasked["delta_loss"], rel=1e-5)


# ------------------------------------------------------ inconsistent metadata


@pytest.mark.parametrize(
    ("keys", "match"),
    [
        ({"mask_pad_in_loss": True}, "is true but 'loss_ignore_index' is absent"),
        (
            {"mask_pad_in_loss": True, "loss_ignore_index": None},
            "is true but 'loss_ignore_index' is null",
        ),
        ({"loss_ignore_index": PAD_ID}, "'mask_pad_in_loss' is absent"),
        ({"mask_pad_in_loss": False, "loss_ignore_index": PAD_ID}, "'mask_pad_in_loss' is false"),
        ({"mask_pad_in_loss": None, "loss_ignore_index": PAD_ID}, "contradict"),
        ({"mask_pad_in_loss": "yes", "loss_ignore_index": PAD_ID}, "must be true, false or absent"),
        ({"mask_pad_in_loss": 1, "loss_ignore_index": PAD_ID}, "must be true, false or absent"),
        ({"mask_pad_in_loss": True, "loss_ignore_index": "3"}, "must be an integer token id"),
        ({"mask_pad_in_loss": True, "loss_ignore_index": 3.0}, "must be an integer token id"),
        ({"mask_pad_in_loss": True, "loss_ignore_index": True}, "must be an integer token id"),
    ],
)
def test_contradictory_pad_metadata_is_refused_when_the_bundle_is_opened(
    tmp_path: Path, keys: dict[str, Any], match: str
) -> None:
    """Neither reading of contradictory keys is safe, so there is no fallback."""
    _write_bundle(tmp_path / "run", **keys)
    with pytest.raises(BundleSchemaError, match=match):
        load_local_bundle(tmp_path / "run", tiny_lm_factory)
