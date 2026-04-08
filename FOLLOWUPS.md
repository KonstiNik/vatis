# vatis follow-ups

Items noticed during the v1.1 hardening pass that are **not** done and
**not** committed as code. The user decides what to act on.

## CLAUDE.md sections that may need updates (forbidden by task 8 to edit)

### `## Architecture > ### Loss handling`

Currently says:

> vatis ships with the standard HF causal-LM CE loss as default, but
> accepts a user-provided `loss_fn(logits, batch) -> scalar` for custom
> setups (e.g. masked LM, contrastive). The closed-form `χ_loss`
> shortcut is only used when `loss_fn` is the registered CE; otherwise
> we fall back to `torch.autograd.grad(loss, logits)` which is still
> cheap (one backward through the loss head only).

Task 6 of the v1.1 hardening pass deleted the autograd fallback
(`chi_loss_from_autograd`) because it was unused outside its own tests.
The analyzer now always uses the closed-form CE path. The "Loss
handling" paragraph is therefore stale: there is no `loss_fn`
fall-back path for chi_loss; non-CE losses are out of scope until v1.2.

This section was not in the allowed-edit list for task 8 (and arguably
sits inside the architecture description). Suggested edit, for the
user to apply:

> vatis ships with the standard HF causal-LM CE loss as default. v1
> only supports the closed-form CE path for `chi_loss`; if a user
> passes a non-CE `loss_fn`, the analyzer will still use it for
> `delta_loss`, but `chi_loss` will be incorrect because it always
> goes through the closed-form CE shortcut. Custom loss functions are
> out of scope until v1.2; the v1.1 hardening pass removed the
> partially-implemented autograd fallback to avoid silent
> wrong-answer modes.

### `## Implementation order`

Step 7 lists `vatis/data/batches.py`, but in practice `data/collate.py`
had to land alongside the chi_net estimators in steps 3–4 because the
estimators import `iter_micro_batches`. SESSION_SUMMARY.md documents
this. The implementation order is presumably descriptive of the
finished build, so no edit is strictly needed; flagging it only because
a future re-build agent following the spec literally would hit the
same dependency-order issue.

## Latent issues that aren't bugs but could surprise a user

### Bundle's `valid_mask_fn` is not consulted by `chi_loss`

The analyzer's `chi_loss` accumulator uses `_extract_targets(micro)`
and `chi_loss_cross_entropy_unnormalized` directly, which check
`valid_token_mask(targets, ignore_index=...)` internally — i.e. they
honor `labels=-100` but **not** the bundle's `valid_mask_fn`. The
n_valid count for normalization, on the other hand, comes from
`bundle.valid_mask_fn(micro, logits).sum()`. If a user supplies a
`valid_mask_fn` that disagrees with the labels-vs-ignore_index
convention (e.g. all-True for an MLP with -100 labels, as the default
`mlp_valid_mask` does), `chi_loss` and the n_valid count will use
different masks, producing a subtly wrong normalization.

The fix from task 4 sidesteps this by having the test use a custom
`valid_mask_fn` that honors `-100`. The right systematic fix is for
the analyzer to use the bundle's `valid_mask_fn` to construct the
attention mask passed into `chi_loss_cross_entropy_unnormalized`, so
the two paths agree by construction. This is a one-line change in
`vatis/analyzer.py::_compute_self_pair` and a one-line change in
`vatis/core/observables.py::chi_loss_cross_entropy_unnormalized` to
take the explicit mask.

Cost of leaving it: every user with a non-trivial `valid_mask_fn`
needs to make sure it matches `(targets != -100)` exactly; otherwise
silent wrong answers. Cost of fixing it: ~5 lines + a test.

### `ParquetSink` truncates on re-open within the same path

Calling `analyze()` multiple times in a row with the same `sink="..."`
path **truncates** the file on each call (the second call's Analyzer
constructs a fresh `ParquetSink` which opens the file fresh).
Discovered while writing `examples/pythia_sweep.py` — the workaround
is to call `analyze()` once with `revisions=[...]` instead of looping.

The example uses the workaround. The right systematic fix is one of:

1. `ParquetSink(path, mode="append")` — appends a row group if the
   file already exists. pyarrow supports this; the `ParquetWriter`
   would need to be opened in append mode and the schema would need
   to match.
2. Keep the user's looping pattern working by re-opening the
   `ParquetWriter` from where it left off, but parquet's column-major
   format makes this expensive.
3. Just document the contract: "one ParquetSink instance per output
   file; pass it explicitly if you need to call `analyze()` multiple
   times". This is the cheapest fix and matches what the example
   already does implicitly.

Cost of leaving it: easy footgun for any user who loops over
revisions manually instead of letting `analyze()` handle it.

## Stale numbers in CLAUDE.md compute scaling table

The 8B/A100 numbers in `## Compute scaling` are order-of-magnitude
guesses and are kept by my CLAUDE.md.proposed (because they're the
production reference target). If a user later runs an actual 8B sweep,
the table should be updated with measured numbers.
