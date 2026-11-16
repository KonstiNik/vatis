"""HuggingFace/TRL-style (unshifted) labels: converted, or refused — never mis-scored.

Two conventions for causal-LM labels differ by one position:

- HuggingFace ``Trainer`` / TRL collate ``labels`` **aligned with**
  ``input_ids`` (a copy of them with ``-100`` at ignored positions) and the
  model shifts inside its loss;
- vatis expects **pre-shifted** labels, ``labels[:, i] = input_ids[:, i + 1]``
  with the last position ignored, and compares ``logits[:, i]`` with
  ``labels[:, i]`` as given.

Handing the first form to vatis unchanged used to score every target one
position off — each logit against the token the model had just been given — in
``causal_lm_loss`` and in the closed-form ``chi_loss`` alike, with no error.
Measured on the batch of ``test_an_unshifted_batch_is_refused_...`` below before
the guard existed: the loss read 2.955 instead of HuggingFace's 3.129, over 32
targets instead of 28.

The reference throughout is HuggingFace's own loss: ``model(**batch).loss`` of a
randomly initialised ``GPT2LMHeadModel`` (built from a config, no download).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from tests.fixtures.tiny_transformer import make_tiny_lm_batch
from tests.utils.bundle_fixture import batch_from_tokens, token_stream
from vatis import analyze
from vatis.data import shift_hf_labels
from vatis.data.collate import shifted_lm_targets, unshifted_label_rows
from vatis.models.hf import (
    ModelBundle,
    causal_lm_forward,
    causal_lm_loss,
    causal_lm_valid_mask,
)

VOCAB, BATCH, SEQ = 23, 4, 10


def _tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(config).eval()


def _hf_batch() -> dict[str, torch.Tensor]:
    """What an HF/TRL collator produces: labels = input_ids, -100 on padding."""
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=torch.Generator().manual_seed(3))
    attention_mask = torch.ones_like(ids)
    attention_mask[1, 7:] = 0
    attention_mask[3, 5:] = 0
    labels = ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": ids, "attention_mask": attention_mask, "labels": labels}


def _bundle(model: GPT2LMHeadModel) -> ModelBundle:
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=causal_lm_forward,
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="gpt2-tiny@step0",
    )


# ------------------------------------------------------------------ the converter


def test_shift_hf_labels_is_the_one_position_shift() -> None:
    """Against an explicit loop: ``out[b, s] = labels[b, s + 1]``, last ignored."""
    gen = torch.Generator().manual_seed(1)
    labels = torch.randint(0, 50, (5, 9), generator=gen)
    labels[torch.rand(5, 9, generator=gen) < 0.3] = -100
    before = labels.clone()
    out = shift_hf_labels(labels)
    assert out.shape == labels.shape and out.dtype == labels.dtype
    for b in range(5):
        for s in range(8):
            assert int(out[b, s]) == int(labels[b, s + 1])
        assert int(out[b, 8]) == -100
    assert torch.equal(labels, before)  # not modified in place

    # Any number of leading dimensions; a custom ignore index only fills the tail.
    assert shift_hf_labels(torch.tensor([4, 5, 6])).tolist() == [5, 6, -100]
    assert shift_hf_labels(torch.tensor([[4, 5, 6]]), ignore_index=-1).tolist() == [[5, 6, -1]]
    with pytest.raises(ValueError, match="sequence dimension"):
        shift_hf_labels(torch.tensor(3))


def test_shift_hf_labels_equals_shifted_lm_targets_on_unmasked_input_ids() -> None:
    ids = torch.randint(0, 50, (3, 7), generator=torch.Generator().manual_seed(2))
    assert torch.equal(shift_hf_labels(ids), shifted_lm_targets(ids))


def test_converted_labels_reproduce_the_huggingface_loss() -> None:
    """``causal_lm_loss`` on converted labels == the HF model's own loss."""
    model = _tiny_gpt2()
    batch = _hf_batch()
    with torch.no_grad():
        out = model(**batch)
    converted = {**batch, "labels": shift_hf_labels(batch["labels"])}
    ours = causal_lm_loss(out.logits, converted)
    assert float(ours) == pytest.approx(float(out.loss), rel=1e-6)
    # Same targets counted: the HF loss is a mean over labels[:, 1:] != -100.
    assert int((converted["labels"] != -100).sum()) == int((batch["labels"][:, 1:] != -100).sum())


def test_analyze_on_a_converted_batch_measures_the_huggingface_gradient() -> None:
    """``delta_loss`` is ``||grad_theta L||^2`` of the loss HF itself computes."""
    model = _tiny_gpt2()
    batch = _hf_batch()
    params = list(model.parameters())
    loss = model(**batch).loss
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    exact = sum(float((g.double() ** 2).sum()) for g in grads if g is not None)

    converted = {**batch, "labels": shift_hf_labels(batch["labels"])}
    for micro in (1, 2, 4):
        rows = analyze(
            model=_bundle(model),
            revisions=["step0"],
            eval_batches={"v": converted},
            chi_net_method="hutchinson",
            n_hutchinson=1,
            micro_batch_size=micro,
            sink=None,
        )[0].rows
        values = {r.observable: r.value for r in rows}
        assert values["delta_loss"] == pytest.approx(exact, rel=1e-5), micro
        assert rows[0].n_valid_a == int((batch["labels"][:, 1:] != -100).sum())


# ------------------------------------------------------------------------ the guard


def test_an_unshifted_batch_is_refused_by_the_shipped_loss_and_by_analyze() -> None:
    model = _tiny_gpt2()
    batch = _hf_batch()
    with torch.no_grad():
        logits = model(**batch).logits
    with pytest.raises(ValueError, match="unshifted"):
        causal_lm_loss(logits, batch)
    for method in ("hutchinson", "per_sequence_cv"):
        for micro in (1, 4):
            with pytest.raises(ValueError, match="shift_hf_labels"):
                analyze(
                    model=_bundle(model),
                    revisions=["step0"],
                    eval_batches={"v": batch},
                    chi_net_method=method,
                    n_hutchinson=1,
                    micro_batch_size=micro,
                    sink=None,
                )


def test_what_the_guard_prevents_is_a_different_number() -> None:
    """The unshifted reading is not a rounding issue: it is another loss.

    Computed here without vatis's loss so the statement survives the guard.
    """
    model = _tiny_gpt2()
    batch = _hf_batch()
    with torch.no_grad():
        out = model(**batch)
    one_position_off = F.cross_entropy(
        out.logits.reshape(-1, VOCAB), batch["labels"].reshape(-1), ignore_index=-100
    )
    assert abs(float(one_position_off) - float(out.loss)) > 0.01 * float(out.loss)


def test_a_preshifted_row_is_never_flagged() -> None:
    """No false positive is possible on a correctly pre-shifted batch.

    Swept over vocabulary sizes down to 2 (where a label coincides with the
    input token at its own position half the time), padding, and random extra
    masking as a completion-only objective produces.
    """
    gen = torch.Generator().manual_seed(0)
    for vocab in (2, 3, 5, 50):
        for _ in range(200):
            ids = torch.randint(0, vocab, (6, 9), generator=gen)
            labels = shifted_lm_targets(ids)
            labels[torch.rand(6, 9, generator=gen) < 0.4] = -100
            assert not bool(unshifted_label_rows(ids, labels).any())
            # ... also when the last target is known (inputs = seq[:-1], labels = seq[1:]).
            labels[:, -1] = torch.randint(0, vocab, (6,), generator=gen)
            assert not bool(unshifted_label_rows(ids, labels).any())


def test_only_unambiguous_rows_are_flagged() -> None:
    ids = torch.tensor(
        [
            [5, 6, 7, 8, 9],  # 0: labels == ids, not next-token      -> flagged
            [4, 4, 4, 4, 4],  # 1: constant: both readings hold       -> ambiguous
            [1, 2, 3, 4, 5],  # 2: only the last position is labelled -> ambiguous
            [1, 2, 3, 4, 5],  # 3: correctly pre-shifted              -> fine
            [1, 2, 2, 2, 9],  # 4: labelled only inside the 2-run     -> ambiguous
            [7, 8, 9, 1, 2],  # 5: nothing labelled                   -> fine
            [3, 1, 4, 1, 5],  # 6: aligned, padded tail               -> flagged
        ]
    )
    labels = torch.full_like(ids, -100)
    labels[0] = ids[0]
    labels[1] = ids[1]
    labels[2, -1] = ids[2, -1]
    labels[3, :-1] = ids[3, 1:]
    labels[4, 1:3] = ids[4, 1:3]
    labels[6, :3] = ids[6, :3]
    assert unshifted_label_rows(ids, labels).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    # Per row, so the verdict cannot depend on how a batch is cut into micro-batches.
    for row in range(ids.shape[0]):
        single = unshifted_label_rows(ids[row : row + 1], labels[row : row + 1])
        assert bool(single[0]) == (row in (0, 6))


def test_an_ambiguous_batch_is_scored_not_refused() -> None:
    """A constant sequence satisfies both conventions; that is not an error."""
    ids = torch.full((2, SEQ), 4)
    labels = ids.clone()
    logits = torch.randn(2, SEQ, VOCAB, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(causal_lm_loss(logits, {"input_ids": ids, "labels": labels}))


def test_the_guard_only_looks_at_token_batches_it_can_compare() -> None:
    """No ``input_ids``, a shape mismatch or float inputs: nothing to compare."""
    logits = torch.randn(3, 4, 7, generator=torch.Generator().manual_seed(0))
    labels = torch.randint(0, 7, (3, 4), generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(causal_lm_loss(logits, {"labels": labels}))
    assert torch.isfinite(
        causal_lm_loss(logits, {"input_ids": torch.zeros(3, 9, dtype=torch.long), "labels": labels})
    )
    assert torch.isfinite(
        causal_lm_loss(logits, {"input_ids": labels.to(torch.float32), "labels": labels})
    )
    with pytest.raises(ValueError, match="equal shape"):
        unshifted_label_rows(torch.zeros(3, 9, dtype=torch.long), labels)


def test_the_repo_fixtures_do_not_trip_the_guard() -> None:
    """Every LM batch builder the suite uses produces pre-shifted labels."""
    for seed in range(20):
        for pad_fraction in (0.0, 0.25, 0.375, 0.5):
            batch = make_tiny_lm_batch(
                batch_size=4, seq_len=8, vocab_size=16, pad_fraction=pad_fraction, seed=seed
            )
            assert not bool(unshifted_label_rows(batch["input_ids"], batch["labels"]).any())
        golden = batch_from_tokens(token_stream(n_rows=4, seq_len=16, vocab_size=48, seed=seed))
        assert not bool(unshifted_label_rows(golden["input_ids"], golden["labels"]).any())
        ids = torch.randint(0, 16, (3, 5), generator=torch.Generator().manual_seed(seed))
        assert not bool(unshifted_label_rows(ids, shifted_lm_targets(ids)).any())
