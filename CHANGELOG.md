# Changelog

All notable changes to vatis are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/KonstiNik/vatis/releases/tag/v0.2.0
[0.1.0]: https://github.com/KonstiNik/vatis/releases/tag/v0.1.0
