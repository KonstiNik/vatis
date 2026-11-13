# Changelog

All notable changes to vatis are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Preconditioned ruler.** `Analyzer` / `analyze()` gained **`preconditioner`**
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
  builds `P = diag(1/(√v̂ + ε))` per checkpoint.
- Parquet rows carry a **`preconditioner`** column (appended last): `"none"` on
  raw rows, the `P` descriptor on the new rows.
- A warning when a published `chi_pos` / `chi_pos_P` leaves `[−1, 1]` by more
  than 0.5 (a Cauchy–Schwarz violation, i.e. inconsistent inputs such as a
  sum-reduced `loss_fn`).
- `golden` test tier (`pytest -m golden`): trains a tiny transformer, writes a
  bundle, runs both rulers end to end.
- New hard dependency: `safetensors` (to read bundle weights).

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
- The `valid_mask_fn ∧ labels != ignore_index` intersection is now applied by
  **every** accumulator, not only `chi_loss`. With a `valid_mask_fn` that accepts
  positions whose label is `ignore_index`, `delta_loss` was rescaled and depended
  on `micro_batch_size`, and `chi_net` included the Jacobian rows of ignored
  positions. The shipped loaders were not affected (their mask already equals
  `labels != ignore_index`).
- The analyzer restores the `requires_grad` flags it changes on the model's parameters.

### Known limitations
- `delta_loss_P` is **not** the first-order loss change of the AdamW step:
  momentum, weight decay and the step's own second-moment update are excluded.
- AdamW's `P` can only be built from native checkpoint bundles; optimizer states
  published with hub checkpoints are not read.

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
