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

### Changed
- **Hutchinson probes are drawn on the CPU** and copied to the device, so a seed
  now identifies one probe set on every device, and the default Rademacher draw
  is bit-packed. Consequence: for a given `seed`, `chi_net` (and therefore
  `chi_pos`) with Rademacher probes is a *different draw* than in 0.2.x — same
  estimator, same expectation, different random numbers. `chi_loss` and
  `delta_loss` are bit-identical to 0.2.x, and so is `chi_net` with Gaussian
  probes on a CPU run. `make_probe` refuses a CUDA generator.
- **Multi-rank (DDP) runs draw different Hutchinson probes for a given `seed`**:
  rank 0 keeps the per-call stream, every other rank now has its own (see
  "Fixed"). `chi_net`, `chi_net_P`, `chi_pos` and `chi_pos_P` (and
  `chi_net_P_capped`) of a run with more than one rank are therefore a different
  draw than in **every earlier version** — same estimator, same expectation —
  and they depend on the number of ranks as well as on the seed. Single-process
  results are unchanged *by this fix* (the CPU-probe entry above still applies
  to them), and so are `chi_loss` and `delta_loss` everywhere. The
  `hutchinson_seed` column keeps its meaning: it is the per-call seed, the
  stream of rank 0, and is the same on every rank.
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
- **DDP ranks no longer replay one Hutchinson probe stream.** Every rank seeded
  its generator with the same per-call seed and drew the probes for its own
  shard, so the same random vectors landed on the `k`-th sample of every shard.
  The shards' estimation errors were then positively correlated (the
  output-space Gram matrices of different inputs are not orthogonal), and the
  variance of the all-reduced `chi_net` — hence of `chi_pos`, and to a
  model-dependent degree of `chi_net_P` / `chi_pos_P` — was between 1× and
  world-size× that of independent rank streams on the same shards. Where in
  that range depends on the model and on the probe distribution, and it has
  been measured **only on CPU test models of at most 1.25M parameters** (two toy
  transformers and one small hub model; 2 / 4 ranks, one sequence per
  micro-batch, where independent rank streams and a single process have the
  same variance): 1.3× / 2.1–2.6× with the default Rademacher probes,
  1.4–1.9× / 2.6–3.3× with Gaussian probes. Nothing larger has been measured.
  The estimate was unbiased throughout. Rank `r > 0` now seeds its generator
  from the per-call seed and `r`; the same measurements then give 0.8–1.2×,
  consistent with 1.
- The `valid_mask_fn ∧ labels != ignore_index` intersection is now applied to
  the token counts, the gradient weights and the probes, not only to `chi_loss`.
  With a `valid_mask_fn` *wider* than the labels (one that accepts positions
  whose label is `ignore_index`), `delta_loss` was rescaled and depended on
  `micro_batch_size`, and `chi_net` included the Jacobian rows of ignored
  positions. That direction is fixed. The intersection does not reach the loss
  itself, so a `valid_mask_fn` *stricter* than the labels is not fixed (see
  Known limitations). The shipped loaders were not affected (their mask already
  equals `labels != ignore_index`).
- The analyzer restores the `requires_grad` flags it changes on the model's parameters.

### Documentation
- `micro_batch_size` is documented as a noise setting, not only a memory
  setting (README, `docs/SPEC.md`): one probe spans the whole micro-batch, so at
  fixed `n_hutchinson` the variance of `chi_net` grows about linearly with the
  micro-batch size for both estimators — measured ×1.7 / ×3.4 / ×6.5 at
  micro-batch 2 / 4 / 8 on a trained toy transformer and ×1.1 / ×1.6 / ×1.9 on
  `SimpleStories-1.25M` (small CPU test models only). This is existing
  behaviour, not a change; the estimate is unbiased at every setting.
- README "Multi-GPU" and `docs/SPEC.md` state that the sampled values of a
  multi-rank run depend on the seed and on the number of ranks, and how the
  batch is sharded when `B % N != 0` (the first `B % N` ranks get one sequence
  more; the docs said the last rank gets a smaller shard).

### Known limitations
- `delta_loss_P` is **not** the first-order loss change of the AdamW step:
  momentum, weight decay and the step's own second-moment update are excluded.
- AdamW's `P` can only be built from native checkpoint bundles; optimizer states
  published with hub checkpoints are not read.
- **A `valid_mask_fn` stricter than the labels does not reach the loss.** It
  restricts the counts, the gradient weights and the probes only, so for such a
  mask `delta_loss` is wrong and depends on `micro_batch_size`, and
  `per_sequence_cv`'s `chi_net` is biased. Exclude positions through the labels
  (`ignore_index`). Fixed in a follow-up pull request.
- **Unshifted labels are not detected.** vatis expects pre-shifted labels; a
  batch whose `labels` equal its `input_ids` (the HuggingFace / TRL convention)
  is scored one position off without an error. Fixed in a follow-up pull
  request.
- **A bundle's recorded pad convention is not applied to the update batch**
  (`mask_pad_in_loss` / `loss_ignore_index` in `metadata.json` are not read):
  update-batch rows of a run whose loss ignored pad targets describe a
  different loss than the one that was descended. Fixed in a follow-up pull
  request.

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
