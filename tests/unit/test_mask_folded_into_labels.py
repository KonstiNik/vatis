"""A ``valid_mask_fn`` *stricter* than the labels must restrict the loss too.

``tests/unit/test_masking_intersection.py`` covers a mask that is **wider** than
the labels (it accepts positions whose label is ``ignore_index``). This file
covers the other direction: a mask that **rejects positions whose label is
valid**. The mask restricted ``n_valid``, ``chi_loss`` and the ``chi_net`` probes,
but ``loss_fn`` only ever sees the batch, so the gradient was that of the
unrestricted loss:

- ``delta_loss`` (and hence ``chi_pos``) belonged to a different loss than the
  other components, and changed with ``micro_batch_size`` whenever the accepted
  count differed between rows, because the per-micro-batch weights were counted
  over the mask while each micro-batch loss was a mean over the labels;
- ``per_sequence_cv`` built its control variate from the gradient of that
  unrestricted loss, which biased ``chi_net`` by an amount no number of probes
  removes (``hutchinson`` uses no loss and was unaffected).

The fix is the HuggingFace ``Trainer`` / TRL procedure: the effective mask is
written into the labels the loss is evaluated on
(``analyzer.py::_label_folded_loss_fn``), so the labels are the single source of
truth for what enters the loss. Every reference below is independent of that
code: the same restriction written into the labels by hand, a plain-torch
gradient, and the explicit per-token Jacobian.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from tests.fixtures.tiny_transformer import (
    TinyMLP,
    TinyTransformer,
    causal_lm_loss,
    make_tiny_lm_batch,
    make_tiny_mlp_batch,
)
from tests.unit.test_chi_net import _build_full_jacobian_and_u
from vatis import analyze
from vatis.analyzer import _intersected_valid_mask_fn
from vatis.core.chi_net import HutchinsonEstimator, PerSequenceControlVariateEstimator
from vatis.core.preconditioner import Preconditioner
from vatis.models.hf import ModelBundle
from vatis.models.hf import causal_lm_loss as shipped_causal_lm_loss
from vatis.models.hf import causal_lm_valid_mask as shipped_causal_lm_valid_mask

FP64 = torch.float64
VOCAB = 16
SEQ_LEN = 8
BATCH_SIZE = 4


def _fixture() -> tuple[TinyTransformer, dict[str, torch.Tensor], list[torch.nn.Parameter]]:
    torch.manual_seed(42)
    model = TinyTransformer(
        vocab_size=VOCAB, seq_len=SEQ_LEN, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    batch = make_tiny_lm_batch(
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB, pad_fraction=0.25, seed=3
    )
    return model, batch, list(model.parameters())


def _accepted(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The strict mask: valid label AND among the first ``k`` positions of its row.

    ``k = first token id mod 4`` is a function of the row's own content, so the
    mask of a row does not depend on how the batch is sliced into micro-batches.
    On the fixture it is row-dependent (so the micro-batch weights matter) and
    rejects one row entirely although its labels are valid (so a micro-batch can
    be left with no accepted position) — asserted in the first test below.
    """
    labels = batch["labels"]
    keep_first = batch["input_ids"][:, 0] % 4
    position = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    return (labels != -100) & (position < keep_first.unsqueeze(1))


def _strict_mask(batch: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return _accepted(batch)


def _labels_mask(batch: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return batch["labels"] != -100


def _restricted(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same restriction, written into the labels by hand."""
    labels = batch["labels"].clone()
    labels[~_accepted(batch)] = -100
    return {**batch, "labels": labels}


def _run(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    batch: Any,
    mask_fn: Any,
    *,
    method: str,
    micro: int,
    n_hutchinson: int = 4,
    preconditioner: Preconditioner | None = None,
    forward_fn: Any = None,
    loss_fn: Any = causal_lm_loss,
    seed: int = 5,
) -> dict[str, float]:
    bundle = ModelBundle(
        model=model,
        params=params,
        forward_fn=forward_fn or (lambda m, b: m(b["input_ids"])),
        loss_fn=loss_fn,
        valid_mask_fn=mask_fn,
        identifier="toy@step0",
    )
    rows = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"lm": batch},
        preconditioner=preconditioner,
        chi_net_method=method,
        n_hutchinson=n_hutchinson,
        micro_batch_size=micro,
        sink=None,
        seed=seed,
    )[0].rows
    out = {r.observable: r.value for r in rows}
    out["n_valid"] = float(rows[0].n_valid_a)
    return out


def test_the_fixture_mask_is_genuinely_stricter_than_the_labels() -> None:
    """Otherwise the tests below prove nothing."""
    _model, batch, _params = _fixture()
    accepted = _accepted(batch)
    valid = batch["labels"] != -100
    assert bool((valid & ~accepted).any())
    assert not bool((accepted & ~valid).any())
    per_row = accepted.sum(dim=1).tolist()
    assert len(set(per_row)) >= 3, per_row  # row-dependent
    assert 0 in per_row and int(valid[per_row.index(0)].sum()) > 0  # a fully rejected row
    assert sum(per_row) >= 4, per_row


# ------------------------------------------- equal to the restriction in labels


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
@pytest.mark.parametrize("micro", [1, 2, 4])
def test_strict_mask_gives_exactly_what_the_same_restriction_in_the_labels_gives(
    method: str, micro: int
) -> None:
    """Every observable, raw and preconditioned, bit for bit.

    Before the fix ``delta_loss`` (both estimators) and ``chi_net``
    (``per_sequence_cv``) differed from the reference at every ``micro`` here;
    the sizes are in the two tests below.
    """
    model, batch, params = _fixture()
    gen = torch.Generator().manual_seed(3)
    p_diag = torch.exp(torch.randn(sum(p.numel() for p in params), generator=gen) * 2.0)
    precond = Preconditioner.from_diagonal(p_diag, descriptor="explicit")

    strict = _run(
        model, params, batch, _strict_mask, method=method, micro=micro, preconditioner=precond
    )
    reference = _run(
        model,
        params,
        _restricted(batch),
        _labels_mask,
        method=method,
        micro=micro,
        preconditioner=precond,
    )
    assert strict["n_valid"] == float(_accepted(batch).sum())
    assert set(strict) == set(reference)
    assert {"delta_loss", "chi_pos", "delta_loss_P", "chi_net_P", "chi_pos_P"} <= set(strict)
    for observable, value in reference.items():
        assert strict[observable] == value, (method, micro, observable, strict[observable], value)


def _exact_restricted_delta_loss(
    model: TinyTransformer, batch: dict[str, torch.Tensor], params: list[torch.nn.Parameter]
) -> float:
    """``||grad_theta L||^2`` of the mean CE over the accepted positions, in plain torch."""
    logits = model(batch["input_ids"])
    accepted = _accepted(batch)
    per_position = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["labels"].clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape(accepted.shape)
    loss = (per_position * accepted).sum() / accepted.sum()
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return sum(float((g.detach().to(FP64) ** 2).sum()) for g in grads if g is not None)


@pytest.mark.parametrize("micro", [1, 2, 4])
def test_delta_loss_under_a_strict_mask_is_exact_at_every_micro_batch_size(micro: int) -> None:
    """Against a gradient that never touches ``ignore_index`` or vatis's fold.

    Pre-fix this fixture gave 0.383x / 0.213x / 0.218x of the exact value at
    micro = 1 / 2 / 4 (measured): wrong, and not the same wrong at every
    micro-batch size.
    """
    model, batch, params = _fixture()
    exact = _exact_restricted_delta_loss(model, batch, params)
    got = _run(model, params, batch, _strict_mask, method="hutchinson", micro=micro, n_hutchinson=1)
    assert got["delta_loss"] == pytest.approx(exact, rel=1e-5), (
        f"delta_loss {got['delta_loss']} at micro={micro}, exact {exact} "
        f"(ratio {got['delta_loss'] / exact:.6f})"
    )


# ------------------------------------------------ chi_net against the exact trace


def _exact_trace_and_noise(
    model: TinyTransformer, restricted: dict[str, torch.Tensor], *, control_variate: bool
) -> tuple[float, float]:
    """``Tr(J J^T)`` over the accepted positions and the per-probe Rademacher std.

    For a symmetric ``A`` and Rademacher ``v``, ``Var[v^T A v] = 2 sum_{i != j}
    A_ij^2``. Plain Hutchinson estimates the trace of ``A = J J^T``; the
    control-variate estimator estimates the trace of ``(I - Pi) A (I - Pi)`` with
    probes and adds the rest exactly, where ``Pi`` projects each sequence's block
    onto its loss direction.
    """
    j_full, u_full, _n_valid = _build_full_jacobian_and_u(model, restricted)
    a = j_full @ j_full.T
    trace = float(torch.diagonal(a).sum())
    if control_variate:
        v = VOCAB
        rows_per_sequence = [
            int(n) * v for n in (restricted["labels"] != -100).sum(dim=1).tolist() if n
        ]
        projector = torch.zeros_like(a)
        start = 0
        for n_rows in rows_per_sequence:
            u_b = u_full[start : start + n_rows]
            projector[start : start + n_rows, start : start + n_rows] = torch.outer(u_b, u_b) / (
                u_b @ u_b
            )
            start += n_rows
        assert start == a.shape[0]
        complement = torch.eye(a.shape[0], dtype=a.dtype) - projector
        a = complement @ a @ complement
    off_diagonal = a - torch.diag(torch.diagonal(a))
    return trace, float((2.0 * (off_diagonal**2).sum()).sqrt())


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_chi_net_under_a_strict_mask_converges_to_the_trace_over_accepted_positions(
    method: str,
) -> None:
    """``chi_net`` is ``Tr(J J^T)`` over the accepted positions for both estimators.

    The tolerance is four standard deviations of the estimator, derived from the
    explicit Jacobian (see ``_exact_trace_and_noise``): 0.68% of the trace at
    this probe count. Pre-fix, ``per_sequence_cv`` read +2.8% to +3.3% here
    (measured over three seeds at 16384 probes, one standard deviation 0.12%):
    its control variate used the gradient of the unrestricted loss, a bias that
    does not shrink with the number of probes. ``hutchinson`` evaluates no loss
    and passed before as well.
    """
    n_hutchinson = 8192
    model, batch, params = _fixture()
    exact, per_probe_std = _exact_trace_and_noise(
        model, _restricted(batch), control_variate=method == "per_sequence_cv"
    )
    four_sigma = 4.0 * per_probe_std / n_hutchinson**0.5
    # The whole batch is one micro-batch, which is the case the variance formula
    # above describes (probes are drawn per micro-batch).
    got = _run(
        model,
        params,
        batch,
        _strict_mask,
        method=method,
        micro=BATCH_SIZE,
        n_hutchinson=n_hutchinson,
        seed=7,
    )
    assert four_sigma < 0.02 * exact, (four_sigma, exact)  # the test has resolving power
    assert abs(got["chi_net"] - exact) <= four_sigma + 1e-5 * exact, (
        f"{method}: chi_net {got['chi_net']}, exact {exact} "
        f"(ratio {got['chi_net'] / exact:.4f}, 4 sigma = {four_sigma / exact:.4f} relative)"
    )


# ------------------------------------------------------------ the other batch forms


def _mlp_loss(logits: torch.Tensor, batch: Any) -> torch.Tensor:
    return F.cross_entropy(logits, batch[1], reduction="mean", ignore_index=-100)


@pytest.mark.parametrize("form", ["tuple", "list"])
@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
@pytest.mark.parametrize("micro", [1, 3])
def test_sequence_batches_are_folded_too(form: str, method: str, micro: int) -> None:
    """``(input, target)`` batches: the mask is written into element 1."""
    torch.manual_seed(1)
    model = TinyMLP().eval()
    params = list(model.parameters())
    x, y = make_tiny_mlp_batch(batch_size=8, seed=2)
    y = y.clone()
    y[5] = -100  # one label ignored by the labels themselves
    rejected = torch.tensor([False, True, False, False, True, False, False, True])
    y_restricted = y.clone()
    y_restricted[rejected] = -100

    def strict(batch: Any, logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        # Identify samples by content so the mask survives micro-batch slicing.
        row_rejected = (batch[0].unsqueeze(1) == x[rejected].unsqueeze(0)).all(dim=2).any(dim=1)
        return (batch[1] != -100) & ~row_rejected

    def make(labels: torch.Tensor) -> Any:
        return (x, labels) if form == "tuple" else [x, labels]

    common = {
        "method": method,
        "micro": micro,
        "forward_fn": lambda m, b: m(b[0]),
        "loss_fn": _mlp_loss,
    }
    got = _run(model, params, make(y), strict, **common)
    reference = _run(model, params, make(y_restricted), lambda b, _l: b[1] != -100, **common)
    assert got["n_valid"] == 4.0
    for observable, value in reference.items():
        assert got[observable] == value, (form, method, micro, observable)


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_a_dict_batch_with_a_targets_key_is_folded_into_that_key(method: str) -> None:
    model, batch, params = _fixture()
    as_targets = {"input_ids": batch["input_ids"], "targets": batch["labels"]}
    restricted = {"input_ids": batch["input_ids"], "targets": _restricted(batch)["labels"]}

    def loss(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return causal_lm_loss(logits, {"labels": b["targets"]})

    def strict(b: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        return _strict_mask({"input_ids": b["input_ids"], "labels": b["targets"]}, logits)

    got = _run(model, params, as_targets, strict, method=method, micro=2, loss_fn=loss)
    reference = _run(
        model,
        params,
        restricted,
        lambda b, _l: b["targets"] != -100,
        method=method,
        micro=2,
        loss_fn=loss,
    )
    for observable, value in reference.items():
        assert got[observable] == value, (method, observable)


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_a_dict_batch_without_labels_gains_masked_next_token_labels(method: str) -> None:
    """Shipped ``causal_lm_loss`` on ``{"input_ids"}`` alone: targets are derived.

    There is no label tensor to write into, so the fold supplies one — the
    next-token targets with the mask applied — and the result must equal the
    batch that states those labels explicitly.
    """
    model, batch, params = _fixture()
    ids = batch["input_ids"]
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    keep = torch.zeros_like(ids, dtype=torch.bool)
    keep[:, :3] = True

    got = _run(
        model,
        params,
        {"input_ids": ids},
        lambda b, _l: keep[: b["input_ids"].shape[0]],
        method=method,
        micro=4,
        loss_fn=shipped_causal_lm_loss,
    )
    explicit = labels.clone()
    explicit[~keep] = -100
    reference = _run(
        model,
        params,
        {"input_ids": ids, "labels": explicit},
        shipped_causal_lm_valid_mask,
        method=method,
        micro=4,
        loss_fn=shipped_causal_lm_loss,
    )
    assert got["n_valid"] == float(keep.sum())
    for observable, value in reference.items():
        assert got[observable] == value, (method, observable)


def test_a_bare_tensor_batch_is_refused_rather_than_run_unmasked() -> None:
    """No labels to read, none to write into: the analyzer must not guess."""
    model, batch, params = _fixture()
    with pytest.raises(ValueError, match="cannot extract targets"):
        _run(
            model,
            params,
            batch["input_ids"],
            lambda _b, logits: torch.ones(logits.shape[:-1], dtype=torch.bool),
            method="hutchinson",
            micro=2,
            forward_fn=lambda m, b: m(b),
        )


# ---------------------------------------------------- no-op for the shipped masks


def test_fold_returns_the_same_batch_object_when_the_mask_adds_nothing() -> None:
    """Shipped loaders (mask == ``labels != ignore_index``) and wider masks: untouched.

    Identity, not equality — the loss then sees the very object it always saw,
    which is why nothing can change bit for bit for those loaders.
    """
    from vatis.analyzer import _fold_mask_into_labels

    _model, batch, _params = _fixture()
    labels = batch["labels"]
    all_true = torch.ones_like(labels, dtype=torch.bool)
    effective = labels != -100  # what _intersected_valid_mask_fn yields for both masks
    assert _fold_mask_into_labels(batch, effective, -100) is batch
    assert _fold_mask_into_labels(batch, effective & all_true, -100) is batch
    as_tuple = (batch["input_ids"], labels)
    assert _fold_mask_into_labels(as_tuple, effective, -100) is as_tuple
    as_list = [batch["input_ids"], labels]
    assert _fold_mask_into_labels(as_list, effective, -100) is as_list


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_loss_fn_receives_the_untouched_labels_under_the_shipped_mask(method: str) -> None:
    """End to end: with ``labels != -100`` as the mask, no label tensor is rewritten."""
    model, batch, params = _fixture()
    seen: list[bool] = []

    def spying_loss(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        # A micro-batch is a slice of the caller's batch, so its labels share the
        # caller's storage unless something rebuilt them.
        seen.append(
            b["labels"].untyped_storage().data_ptr() == batch["labels"].untyped_storage().data_ptr()
        )
        return causal_lm_loss(logits, b)

    _run(model, params, batch, _labels_mask, method=method, micro=2, loss_fn=spying_loss)
    assert seen and all(seen), seen


def test_fold_never_modifies_the_callers_batch() -> None:
    from vatis.analyzer import _fold_mask_into_labels

    model, batch, params = _fixture()
    labels_before = batch["labels"].clone()
    keys_before = set(batch)
    _run(model, params, batch, _strict_mask, method="per_sequence_cv", micro=2)
    assert torch.equal(batch["labels"], labels_before)
    assert set(batch) == keys_before

    folded = _fold_mask_into_labels(batch, _accepted(batch), -100)
    assert folded is not batch
    assert torch.equal(folded["labels"], _restricted(batch)["labels"])
    assert folded["input_ids"] is batch["input_ids"]
    assert torch.equal(batch["labels"], labels_before)


# ------------------------------------------------- refusing what cannot be masked


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_a_loss_fn_that_does_not_honour_the_folded_labels_is_refused(method: str) -> None:
    """Never silently unmasked.

    This ``loss_fn`` reads its targets from somewhere vatis cannot write to, so
    writing the mask into the batch's labels does not restrict it. Pre-fix the
    run completed with the unrestricted gradient and no warning.
    """
    model, batch, params = _fixture()
    rows_by_content = {
        tuple(row.tolist()): batch["labels"][i] for i, row in enumerate(batch["input_ids"])
    }

    def deaf_loss(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        own_labels = torch.stack([rows_by_content[tuple(row.tolist())] for row in b["input_ids"]])
        return causal_lm_loss(logits, {"labels": own_labels})

    with pytest.raises(ValueError, match="rejected positions"):
        _run(model, params, batch, _strict_mask, method=method, micro=2, loss_fn=deaf_loss)
    # The same loss with a mask that adds nothing to the labels is fine.
    _run(model, params, batch, _labels_mask, method=method, micro=2, loss_fn=deaf_loss)


@pytest.mark.parametrize("which", ["first", "middle", "last"])
def test_a_leak_at_one_single_rejected_position_is_refused(which: str) -> None:
    """Every rejected position is inspected, not a sample of them.

    This loss honours the folded labels everywhere and additionally depends on
    the logits of exactly ONE position the mask rejects — the first, a middle
    or the last one in row-major order. A check that looked only at the first
    rejected position would pass the other two.
    """
    model, batch, params = _fixture()
    rejected = torch.nonzero((batch["labels"] != -100) & ~_accepted(batch))
    assert len(rejected) >= 3
    assert len({int(b) for b, _s in rejected}) >= 3  # they span several rows
    index = {"first": 0, "middle": len(rejected) // 2, "last": len(rejected) - 1}[which]
    row, position = (int(i) for i in rejected[index])

    def leaky_loss(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        # The whole batch is one micro-batch below, so ``row`` indexes it directly.
        return causal_lm_loss(logits, b) + 1e-3 * logits[row, position, 0]

    with pytest.raises(ValueError, match=r"\(1 non-zero gradient entries at rejected positions\)"):
        _run(
            model,
            params,
            batch,
            _strict_mask,
            method="hutchinson",
            micro=BATCH_SIZE,
            loss_fn=leaky_loss,
        )


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
@pytest.mark.parametrize("honours_the_fold", [False, True], ids=["deaf", "honours-the-fold"])
def test_a_loss_fn_that_ignores_the_logits_it_is_handed_is_refused(
    method: str, honours_the_fold: bool
) -> None:
    """ "Could not verify" must not pass for "verified".

    This ``loss_fn`` runs its own forward and never touches its ``logits``
    argument, so no gradient reaches them and the leak check has nothing to
    inspect. It used to skip itself in that case: with the deaf variant below
    (which also ignores the rewritten labels) ``delta_loss`` came out at 0.213x
    the restricted value under ``hutchinson`` at ``micro_batch_size=2`` on this
    fixture (measured), with no error.
    Now the run is refused — also for the variant that does read the rewritten
    labels, because the analyzer cannot tell the two apart.
    """
    model, batch, params = _fixture()
    rows_by_content = {
        tuple(row.tolist()): batch["labels"][i] for i, row in enumerate(batch["input_ids"])
    }

    def own_forward_loss(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: ARG001
        if honours_the_fold:
            labels = b["labels"]
        else:
            labels = torch.stack([rows_by_content[tuple(row.tolist())] for row in b["input_ids"]])
        return causal_lm_loss(model(b["input_ids"]), {"labels": labels})

    with pytest.raises(ValueError, match="cannot verify that loss_fn honours the mask"):
        _run(model, params, batch, _strict_mask, method=method, micro=2, loss_fn=own_forward_loss)
    if method == "hutchinson":
        # Without a stricter mask nothing was rewritten and nothing needs
        # verifying: such a loss keeps working where it worked before
        # (``per_sequence_cv`` needs the gradient w.r.t. the logits regardless).
        _run(model, params, batch, _labels_mask, method=method, micro=2, loss_fn=own_forward_loss)


def test_folded_loss_closure_matches_the_restricted_loss_and_its_gradient() -> None:
    """The closure itself, against plain torch on hand-restricted labels."""
    from vatis.analyzer import _label_folded_loss_fn

    model, batch, params = _fixture()
    mask_fn = _intersected_valid_mask_fn(_strict_mask, -100)
    loss_fn = _label_folded_loss_fn(causal_lm_loss, mask_fn, -100)

    logits = model(batch["input_ids"])
    got = loss_fn(logits, batch)
    got_grads = torch.autograd.grad(got, params, allow_unused=True)

    logits_ref = model(batch["input_ids"])
    reference = F.cross_entropy(
        logits_ref.reshape(-1, VOCAB), _restricted(batch)["labels"].reshape(-1), ignore_index=-100
    )
    reference_grads = torch.autograd.grad(reference, params, allow_unused=True)
    assert float(got.detach()) == float(reference.detach())
    for g, r in zip(got_grads, reference_grads, strict=True):
        assert (g is None) == (r is None)
        if g is not None and r is not None:
            assert torch.equal(g, r)


# ------------------------------------------------------- the estimator's own guard


def test_per_sequence_cv_refuses_a_loss_that_counts_positions_its_mask_rejects() -> None:
    """Called directly (no analyzer, so no fold), the inconsistency is an error.

    Pre-fix this returned a ``chi_net`` biased by the control variate, with no
    signal (the size of the bias on this fixture is in
    ``test_chi_net_under_a_strict_mask_converges_...`` above).
    """
    model, batch, params = _fixture()
    mask_fn = _intersected_valid_mask_fn(_strict_mask, -100)
    estimator = PerSequenceControlVariateEstimator(n_hutchinson=2)
    kwargs = {
        "valid_mask_fn": mask_fn,
        "params": params,
        "micro_batch_size": 2,
        "generator": torch.Generator().manual_seed(0),
    }
    with pytest.raises(ValueError, match="positions valid_mask_fn rejects"):
        estimator.compute(
            model, batch, lambda m, b: m(b["input_ids"]), loss_fn=causal_lm_loss, **kwargs
        )
    # With the loss evaluated on labels that carry the mask it runs ...
    from vatis.analyzer import _label_folded_loss_fn

    folded_loss = _label_folded_loss_fn(causal_lm_loss, mask_fn, -100)
    result = estimator.compute(
        model, batch, lambda m, b: m(b["input_ids"]), loss_fn=folded_loss, **kwargs
    )
    assert result.n_valid_tokens == int(_accepted(batch).sum())
    # ... and so does a mask that is merely *wider* than the loss, as before.
    all_true = _intersected_valid_mask_fn(
        lambda _b, logits: torch.ones(logits.shape[:-1], dtype=torch.bool), -100
    )
    estimator.compute(
        model,
        batch,
        lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        **{**kwargs, "valid_mask_fn": all_true},
    )
    # Hutchinson evaluates no loss and needs no such guard.
    HutchinsonEstimator(n_hutchinson=2).compute(
        model, batch, lambda m, b: m(b["input_ids"]), loss_fn=causal_lm_loss, **kwargs
    )


def test_per_sequence_cv_refuses_a_loss_that_leaks_at_one_single_position() -> None:
    """The estimator's guard has no tolerance: one leaking position is one too many.

    The mask rejects exactly one position whose label is valid, so the loss
    direction leaves the masked subspace in ``V = 16`` logit entries and nowhere
    else. A guard that tolerated "a few" non-zero entries would let this through
    and return a biased ``chi_net``.
    """
    model, batch, params = _fixture()
    valid = batch["labels"] != -100
    row, position = (int(i) for i in torch.nonzero(valid)[-1])

    def all_but_one(b: dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        mask = (b["labels"] != -100).clone()
        mask[row, position] = False  # one micro-batch below: ``row`` indexes it directly
        return mask

    estimator = PerSequenceControlVariateEstimator(n_hutchinson=1)
    with pytest.raises(ValueError, match=rf"positions valid_mask_fn rejects \({VOCAB} entries\)"):
        estimator.compute(
            model,
            batch,
            lambda m, b: m(b["input_ids"]),
            loss_fn=causal_lm_loss,
            valid_mask_fn=_intersected_valid_mask_fn(all_but_one, -100),
            params=params,
            micro_batch_size=BATCH_SIZE,
            generator=torch.Generator().manual_seed(0),
        )
