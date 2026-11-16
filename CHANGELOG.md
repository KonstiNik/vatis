# Changelog

All notable changes to vatis are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Optimizer-aware measurement (preconditioned observables).** `Analyzer` / `analyze()` gained **`preconditioner`**
  (`None` | `"identity"` | `Preconditioner` | `callable(checkpoint_id)`). With a
  diagonal `P`, three more observables are measured on the effective kernel
  `Θ_P = J P Jᵀ` in the same pass, with no extra backward: `delta_loss_P`,
  `chi_net_P`, `chi_pos_P`. Supported by both `χ_net` estimators, cross pairs
  and DDP. `P = I` reproduces the raw triple.
- `vatis.Preconditioner` (`identity`, `from_diagonal`, `adamw`) and
  `vatis.core.preconditioner.epsilon_cap_mask`.
- **Native checkpoint bundles**: `vatis.load_local_bundle(run_dir,
  model_factory)` reads a run directory holding weights, the pre-step
  `torch.optim` state and the update batch per checkpoint, and exposes the
  optimizer state flattened in one canonical parameter order
  (`vatis.core.param_layout.ParameterLayout`). `source.adamw_preconditioner()`
  builds `P = diag(1/(√v̂ + ε))` per checkpoint; `source.preconditioner()` picks
  the geometry factor from the saved optimizer state — that AdamW `P`, or `P = I`
  under the descriptor `"sgd"` for a bundle written by `torch.optim.SGD`.
  A checkpoint whose directory step differs from its optimizer state's own step
  counter is refused (`BundleSchemaError`); both methods take
  `on_step_mismatch="warn"` to build `P` from the saved counter with a warning
  instead (default `"raise"`). A state whose param groups have different
  learning rates is refused as well: `P` leaves the rate out, which is only
  sound when every parameter has the same one.
- **`delta_loss_P_capped` / `chi_net_P_capped`** rows next to the AdamW P
  observables: the part contributed by coordinates at the `1/ε` ceiling of `P`
  (second moment still zero). A few such coordinates can carry nearly all of a
  P observable — on held-out tokens, or on an update batch containing a token
  the run has not seen before — and these rows make that visible per value
  instead of clipping `P`.
- Parquet rows carry a **`preconditioner`** column (appended last): `"none"` on
  raw rows, the `P` descriptor on the new rows.
- A warning when a published `chi_pos` / `chi_pos_P` leaves `[−1, 1]` by more
  than 0.5 (a Cauchy–Schwarz violation, i.e. inconsistent inputs such as a
  sum-reduced `loss_fn`).
- `golden` test tier (`pytest -m golden`): trains a tiny transformer, writes a
  bundle, runs the raw and preconditioned observables end to end.
- New hard dependency: `safetensors` (to read bundle weights).
- `analyze()` accepts `ignore_index` (default `-100`) and passes it to
  `Analyzer`, which already did.
- `vatis.data.shift_hf_labels(labels)`: converts HuggingFace / TRL-style labels
  (aligned with `input_ids`, shifted by the model) to the pre-shifted labels
  vatis expects, with the same one-position shift `transformers`' causal-LM loss
  applies. The label convention is now documented in `docs/SPEC.md` ("Batch
  format and the label convention") and the README.
- Checkpoint bundles: two optional `metadata.json` fields, `mask_pad_in_loss`
  (bool) and `loss_ignore_index` (int or null), record whether the training loss
  ignored a target id. Exposed as `BundleMetadata.loss_ignore_index` and
  `LocalBundleSource.loss_ignore_index`.

### Changed (breaking)
The labels are the single source of truth for what enters the loss
(`docs/SPEC.md`: "Batch format and the label convention", "Loss handling",
"Native checkpoint bundles"). For code that ran before:
- **A `valid_mask_fn` stricter than the labels now restricts the loss**, so `delta_loss` and `chi_pos` (and `chi_net` under `per_sequence_cv`) change for such a mask. *To do:* nothing if the restricted loss is what the mask meant; the result now equals the same restriction written into the labels.
- **A loss that cannot be verified under a stricter mask raises** a `ValueError` (it still depends on a rejected position, or no gradient reaches the logits it was handed). *To do:* make `loss_fn` read its targets from the batch, honour `ignore_index`, and compute the loss from its `logits` argument.
- **The analyzer's `ignore_index` must be the value `loss_fn` ignores.** With a different value and a stricter mask the run is refused with a message naming the value (it used to work when the mask happened to equal that loss's own ignore mark). *To do:* pass `ignore_index=<that value>` to `Analyzer(...)` or `analyze(...)` (which now accepts it).
- **A direct `PerSequenceControlVariateEstimator.compute` call raises** a `ValueError` when `loss_fn` counts positions its `valid_mask_fn` rejects (it used to return a biased `chi_net`). *To do:* write the mask into the labels the loss reads, as `Analyzer` does.
- **Unshifted (HuggingFace / TRL-style) labels are refused** by the shipped `causal_lm_loss` when they are provably unshifted (they used to be scored one position off). *To do:* `batch["labels"] = vatis.data.shift_hf_labels(batch["labels"])`.
- **The update-batch rows of a bundle that records pad masking change**: `batch_a` ignores the targets equal to the recorded `loss_ignore_index` (they used to be counted), and `examples/two_rulers_llama.py` scores its held-out batches the same way. *To do:* re-run analyses of such bundles; bundles without the two fields are read exactly as before.
- **Contradictory or impossible pad metadata is refused** with `BundleSchemaError` when the bundle is opened, also for uses that never touch the update batch (`mask_pad_in_loss: true` without an index, an index without it, a wrong JSON type, a negative index or one not below the model config's `vocab_size`). *To do:* correct the two fields in `metadata.json`; "Repairing contradictory pad fields" in the SPEC says how, including how the recorded `train_loss` settles which entry is wrong.

### Changed
- **Hutchinson probes are drawn on the CPU** and copied to the device, so a seed
  now identifies one probe set on every device, and the default Rademacher draw
  is bit-packed. Consequence: for a given `seed`, `chi_net` (and therefore
  `chi_pos`) with Rademacher probes is a *different draw* than in 0.2.x — same
  estimator, same expectation, different random numbers. `chi_loss` and
  `delta_loss` are bit-identical to 0.2.x, and so is `chi_net` with Gaussian
  probes on a CPU run. `make_probe` refuses a CUDA generator.
- `chi_pos` is computed in fp64 and returned as an fp64 scalar. The absolute
  `1e-30` floor on its denominator is gone: only an exactly-zero denominator
  yields `0.0`.
- **Custom `ChiNetEstimator` subclasses must accept a `preconditioner` keyword**
  in `compute()`; the analyzer always passes it (`None` without a P).
- The analyzer raises a `RuntimeError` if the estimator's valid-token count
  differs from the one `chi_loss` was normalized with.
- `cross_grad_storage="auto"` looks at the device the model is on rather than at
  `Analyzer.device`.
- `observables=None` is the new default and means "all raw observables, plus the
  P observables when a preconditioner is set". Passing an explicit list behaves
  as before.

### Fixed
- The `valid_mask_fn ∧ labels != ignore_index` intersection is now applied to
  the token counts, the gradient weights and the probes, not only to `chi_loss`.
  With a `valid_mask_fn` *wider* than the labels (one that accepts positions
  whose label is `ignore_index`), `delta_loss` was rescaled and depended on
  `micro_batch_size`, and `chi_net` included the Jacobian rows of ignored
  positions. That direction is fixed by the intersection. A mask alone cannot
  reach the loss itself; the *stricter* direction is fixed separately (next
  entry but one). The shipped loaders were not affected (their mask already
  equals `labels != ignore_index`).
- The analyzer restores the `requires_grad` flags it changes on the model's parameters.
- A `valid_mask_fn` **stricter** than the labels (it rejects a position whose
  label is valid) now restricts the loss as well. It used to restrict `n_valid`,
  `chi_loss` and the `chi_net` probes only: `delta_loss` and `chi_pos` were those
  of the unrestricted loss and changed with `micro_batch_size`, and
  `per_sequence_cv` biased `chi_net` through its control variate. The analyzer
  now writes the mask into the labels the loss is evaluated on, which gives
  exactly what the same restriction written into the labels gives. The shipped
  loaders were not affected (their mask equals `labels != ignore_index`); every
  observable is bit-identical for them.

### Known limitations
- `delta_loss_P` is **not** the first-order loss change of the AdamW step:
  momentum, weight decay and the step's own second-moment update are excluded.
- AdamW's `P` can only be built from native checkpoint bundles; optimizer states
  published with hub checkpoints are not read.
- **DDP: every rank draws the same probe stream for its own shard.** The ranks'
  estimation errors are therefore correlated, and a multi-rank `chi_net` /
  `chi_net_P` (hence `chi_pos` / `chi_pos_P`) has a larger variance than a
  single-process run with the same `n_hutchinson`, by a factor that depends on
  the model, the data and the number of ranks. The estimate stays unbiased;
  `chi_loss` and `delta_loss` are not affected. Fixed in a follow-up pull
  request.
- The unshifted-label guard of `causal_lm_loss` is not exhaustive: a single
  labelled token at the very end of a sequence, or a label equal to the token
  after it, looks the same in both conventions and is scored as given; a custom
  `loss_fn` is not guarded.

## [0.2.1] — 2026-10-19

### Added
- `CITATION.cff` for software citation.

## [0.2.0] — 2026-06-17

### Added
- `Analyzer` / `analyze()` gained **`cross_grad_storage`** (`"auto"` | `"gpu"` |
  `"cpu"`, default `"auto"`): cross-pair gradients can be offloaded to host RAM
  so cross-pair observables fit on large models whose two full gradients won't
  co-reside on the GPU; small models stay on-device (no transfer, no numeric
  change). `"auto"` offloads only when an on-GPU cache wouldn't leave room for a
  self-pair backward.
- `load_hf_model(..., attn_implementation=...)` — passthrough to
  `from_pretrained` (e.g. `"sdpa"`, `"flash_attention_2"`, `"eager"`).
- `examples/olmo_sft_analysis/` — a production-scale deployment example: the LNP
  decomposition on `Olmo-3-7B-Think-SFT` over real SFT chat data with
  completion-only loss masking, code-vs-English cross-pair analysis, and
  wall-time / peak-memory / GPU-engine-activity tracking.

### Fixed
- The analyzer no longer caches a batch's full parameter-space gradient unless a
  cross pair consumes it. Caching it unconditionally kept a P-sized vector
  (~29 GB for a 7B model) resident across batches and could OOM large models
  even when no cross pair was requested.

## [0.1.0] — initial public release

First public release.

### Added
- `analyze()` / `Analyzer`: compute LNP observables (`chi_loss`, `chi_net`,
  `chi_pos`, `delta_loss`) on pretrained HuggingFace causal-LM checkpoints.
- Two `χ_net` estimators with auto-selection — `HutchinsonEstimator` and
  `PerSequenceControlVariateEstimator` (the latter can also return the
  cross-sample alignment matrix via `compute_alignment_matrix=True`).
  `OpacusEstimator` is present as a stub.
- Self-pair and cross-pair observables; long-format Parquet result sink
  (canonical) plus optional W&B and TensorBoard sinks.
- DDP scaling via `python -m vatis run …` (wraps `torchrun`).
- Examples: end-to-end `pythia_sweep.py` + `analyze_results.py`, and
  single-GPU / DDP-scaling / accuracy benchmarks under `examples/benchmark/`.
- Specification in `docs/SPEC.md`; LNP derivation in
  [arXiv:2605.31244](https://arxiv.org/abs/2605.31244).

[0.2.1]: https://github.com/KonstiNik/vatis/releases/tag/v0.2.1
[0.2.0]: https://github.com/KonstiNik/vatis/releases/tag/v0.2.0
[0.1.0]: https://github.com/KonstiNik/vatis/releases/tag/v0.1.0
