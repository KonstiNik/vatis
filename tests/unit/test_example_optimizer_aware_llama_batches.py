"""``examples/two_rulers_llama.py``: held-out batches follow the bundle's pad convention.

The example measures on held-out batches it builds itself from ``.npy`` token
files. It used to count every target there unconditionally, so for a bundle
whose run masked padding in its loss the held-out rows described a different
objective than the one the run descended (and than the loader's update batch).
The script now applies ``source.loss_ignore_index`` to those labels.

The full script needs a bundle of the project Llama and is not exercised here;
what *is* testable without one is the batch-building function, on the hand-written
bundles of ``tests/fixtures/synthetic_bundle.py``: its labels against an explicit
loop, its agreement with what the loader produces for the same tokens, and the
analyzer's rows on such a batch against plain ``torch`` cross entropy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
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

PAD_ID = 3
SEQ_LEN = 7
EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "two_rulers_llama.py"


@pytest.fixture(scope="module")
def example() -> ModuleType:
    """The example script, imported as a module (it has no package to import from)."""
    name = "_two_rulers_llama_under_test"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    sys.modules[name] = module  # its dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path  # the script prepends the repo root; undo it
    return module


def _padded_tokens(seed: int, *, batch_size: int = 4) -> np.ndarray:
    """Right-padded rows (0-3 trailing pads) plus one pad as an interior token."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, DEFAULT_VOCAB_SIZE, size=(batch_size, SEQ_LEN), dtype=np.int32)
    ids[ids == PAD_ID] = PAD_ID + 1
    for row in range(1, batch_size):
        ids[row, SEQ_LEN - row :] = PAD_ID
    ids[0, 2] = PAD_ID
    return ids


def _write_bundle_and_eval_batches(root: Path, *, masked: bool) -> tuple[Path, Path]:
    """A one-checkpoint bundle and three held-out ``eval_batch_*.npy`` files.

    ``eval_batch_0.npy`` holds the very tokens of the checkpoint's ``batch_a.npy``,
    so the example's labels can be compared with the loader's for the same input.
    """
    run_dir, eval_dir = root / "bundle", root / "eval_batches"
    model = BundleTinyLM(seq_len=SEQ_LEN, seed=0)
    metadata: dict[str, Any] = bundle_metadata(model, batch_size=4)
    if masked:
        metadata["mask_pad_in_loss"] = True
        metadata["loss_ignore_index"] = PAD_ID
    order = write_run_files(run_dir, model, metadata=metadata)
    write_checkpoint(
        run_dir,
        1,
        model=model,
        optim_state=handwritten_optim_state(model, order, step=1),
        batch_a=_padded_tokens(0),
    )
    eval_dir.mkdir(parents=True)
    for index in range(3):
        np.save(eval_dir / f"eval_batch_{index}.npy", _padded_tokens(index))
    return run_dir, eval_dir


def _build(example: ModuleType, eval_dir: Path, loss_ignore_index: int | None) -> Any:
    batches, used_fallback = example.load_measurement_batches(
        eval_dir,
        seq_len=SEQ_LEN,
        vocab_size=DEFAULT_VOCAB_SIZE,
        batch_size=4,
        device=torch.device("cpu"),
        loss_ignore_index=loss_ignore_index,
    )
    assert not used_fallback
    assert tuple(batches) == example.BATCH_NAMES
    return batches


@pytest.mark.parametrize("masked", [True, False], ids=["pad-masked-run", "unmasked-run"])
def test_held_out_labels_follow_the_recorded_pad_convention(
    example: ModuleType, tmp_path: Path, masked: bool
) -> None:
    run_dir, eval_dir = _write_bundle_and_eval_batches(tmp_path, masked=masked)
    source = load_local_bundle(run_dir, tiny_lm_factory)
    assert source.loss_ignore_index == (PAD_ID if masked else None)
    batches = _build(example, eval_dir, source.loss_ignore_index)

    for index, name in enumerate(example.BATCH_NAMES):
        ids = torch.from_numpy(np.load(eval_dir / f"eval_batch_{index}.npy")).to(torch.int64)
        assert torch.equal(batches[name]["input_ids"], ids)  # inputs keep their pads
        for b in range(ids.shape[0]):
            for s in range(SEQ_LEN):
                target = int(ids[b, s + 1]) if s + 1 < SEQ_LEN else None
                ignored = target is None or (masked and target == PAD_ID)
                expected = -100 if ignored else target
                assert int(batches[name]["labels"][b, s]) == expected, (name, b, s)
        n_counted = int((batches[name]["labels"] != -100).sum())
        assert n_counted == 4 * (SEQ_LEN - 1) - (7 if masked else 0)

    # Same tokens, same labels as the loader gives the update batch: one objective.
    update_batch = source.checkpoints[0].batch_a
    assert torch.equal(batches["A1"]["input_ids"], update_batch["input_ids"])
    assert torch.equal(batches["A1"]["labels"], update_batch["labels"])


def test_without_a_recorded_index_the_labels_are_what_they_always_were(
    example: ModuleType, tmp_path: Path
) -> None:
    """``None`` (also the default): plain next-token labels, every target counted."""
    _run_dir, eval_dir = _write_bundle_and_eval_batches(tmp_path, masked=False)
    explicit = _build(example, eval_dir, None)
    default, _ = example.load_measurement_batches(
        eval_dir,
        seq_len=SEQ_LEN,
        vocab_size=DEFAULT_VOCAB_SIZE,
        batch_size=4,
        device=torch.device("cpu"),
    )
    for index, name in enumerate(example.BATCH_NAMES):
        ids = torch.from_numpy(np.load(eval_dir / f"eval_batch_{index}.npy")).to(torch.int64)
        expected = torch.full_like(ids, -100)
        expected[:, :-1] = ids[:, 1:]
        assert torch.equal(explicit[name]["labels"], expected)
        assert torch.equal(default[name]["labels"], expected)


@pytest.mark.parametrize("masked", [True, False], ids=["pad-masked-run", "unmasked-run"])
def test_held_out_rows_match_plain_torch_under_the_runs_objective(
    example: ModuleType, tmp_path: Path, masked: bool
) -> None:
    """``n_valid``, ``chi_loss`` and ``delta_loss`` of a held-out batch.

    Reference: ``F.cross_entropy`` on ``inputs = ids[:, :-1]``, ``targets =
    ids[:, 1:]``, with ``ignore_index=<pad id>`` iff the run masked padding — the
    training criterion's own form, sharing nothing with the example or the loader.
    """
    run_dir, eval_dir = _write_bundle_and_eval_batches(tmp_path, masked=masked)
    source = load_local_bundle(run_dir, tiny_lm_factory)
    batches = _build(example, eval_dir, source.loss_ignore_index)
    analyzer = Analyzer(
        eval_batches={"B": batches["B"]},
        chi_net_method="hutchinson",
        n_hutchinson=1,
        micro_batch_size=2,
        sink=None,
    )
    (result,) = analyzer.run(source.model_specs())

    ids = torch.from_numpy(np.load(eval_dir / "eval_batch_2.npy")).to(torch.int64)
    model = source.checkpoints[0].model_bundle.model
    logits = model(ids[:, :-1])
    flat_logits, flat_targets = logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    if masked:
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=PAD_ID)
        n = int((flat_targets != PAD_ID).sum())
    else:
        loss = F.cross_entropy(flat_logits, flat_targets)
        n = int(flat_targets.numel())
    (grad_logits,) = torch.autograd.grad(loss, logits, retain_graph=True)
    grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    chi_loss = float((grad_logits.double() ** 2).sum())
    delta_loss = sum(float((g.double() ** 2).sum()) for g in grads if g is not None)

    values = {r.observable: r.value for r in result.rows}
    assert {r.n_valid_a for r in result.rows} == {n}
    assert values["chi_loss"] == pytest.approx(chi_loss, rel=1e-5)
    assert values["delta_loss"] == pytest.approx(delta_loss, rel=1e-5)


def test_the_random_token_fallback_applies_the_same_convention(
    example: ModuleType, tmp_path: Path
) -> None:
    """No ``.npy`` files: random tokens (loudly), scored under the same convention."""
    kwargs = {"seq_len": 32, "vocab_size": 5, "batch_size": 6, "device": torch.device("cpu")}
    with pytest.warns(UserWarning, match="random-token measurement batches"):
        masked, used_fallback = example.load_measurement_batches(
            tmp_path / "nothing-here", loss_ignore_index=PAD_ID, **kwargs
        )
    assert used_fallback
    with pytest.warns(UserWarning, match="random-token measurement batches"):
        unmasked, _ = example.load_measurement_batches(tmp_path / "nothing-here", **kwargs)
    for name in example.BATCH_NAMES:
        ids = masked[name]["input_ids"]
        assert torch.equal(ids, unmasked[name]["input_ids"])  # same seeded tokens
        plain = torch.full_like(ids, -100)
        plain[:, :-1] = ids[:, 1:]
        assert torch.equal(unmasked[name]["labels"], plain)
        expected = plain.clone()
        expected[plain == PAD_ID] = -100
        assert torch.equal(masked[name]["labels"], expected)
        assert int((plain == PAD_ID).sum()) > 0  # the fixture does contain such targets
