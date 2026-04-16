# Spectral tail overlap hypothesis

Detecting when training resolves shared semantic structure between two eval batches via a chi_pos bump in the LNA decomposition.

## Core idea

If two eval batches A and B share similarity only in the "tail" of the NTK spectrum (i.e. deeper/later-learned features), then:

- **Early training**: gradients are orthogonal (chi_pos near zero) because the model hasn't resolved that spectral band yet.
- **Mid training**: when the optimizer's "frontier" reaches the shared band, chi_pos should bump up — the cross coupling becomes visible.
- **Late training**: the shared band is resolved and the bump subsides, or the model specializes further and the coupling is consumed.

This would appear as a transient bump in `chi_pos(A, B)` at the checkpoint where training "arrives at" the spectral band where A and B overlap.

## Probe design

Model of what LMs learn in order:
1. Token combinations into words (surface statistics)
2. Words into sentences (syntax)
3. Semantic content (what the code *does*)

Good probes should **differ at stages 1-2 but share stage 3**. This is why Python vs C++ implementing the same algorithm is a better probe than two Python styles — Python vs C++ eliminates almost all syntactic overlap, isolating the semantic layer.

Key design principles:
- **Repetition is signal, not redundancy.** Multiple implementations of the same algorithm within one text creates within-context pressure for the model to build abstract representations.
- **Parallel structure matters.** Both texts should implement the same operations in the same order (matmul, transpose, dot, trace, Frobenius) — not divergent operations.
- **B=1 with long S is optimal.** Longer sequences give the model more context to build semantic representations. B=1, S=1024 maximizes context per position.

## Findings (2026-04-16)

**At 14m–410m scale, we do not observe the predicted spectral tail bump.** All cross chi_pos trajectories show a monotonic decline after early training, with some scale-dependent late-training structure (secondary rises at 70m and 410m) that is not clearly separated from the negative control. The positive signal (python x cpp, shared algorithm) runs 2–4x above the negative control (python x cpp_str, different algorithm) in the late-training window, but the effect is modest and not the transient bump the hypothesis predicts.

The most likely explanation is that models at this scale do not develop cross-domain abstract representations of "matrix multiplication" that bridge programming languages. The feature we are probing for may require larger models.

One additional observation: when plotted over estimated compute (6ND FLOPs) instead of training steps, the initial chi_pos decline collapses across all model scales onto a single curve (`--compute` flag). This early phase reflects generic token-statistics learning and is compute-determined, not scale-dependent.

### Follow-ups

1. **Larger models (1B+).** Cross-lingual code understanding is a capability that emerges at scale. These models may actually build the shared representations we are probing for.
2. **Simpler semantic overlap.** Instead of probing for a highly abstract feature (cross-lingual algorithm equivalence), probe for something these models plausibly do learn — e.g. repeated structural patterns, same topic in different registers. This tests whether the method works at all before scaling up.

Doing (2) first is the more cautious path: if a simpler feature doesn't produce a bump even at 410m, the method itself needs rethinking, not just the scale.

## Experiment setup

Code: `examples/spectral_tail_experiment.py` (compute), `examples/spectral_tail_evaluate.py` (plots).
Results: `examples/spectral_tail/results_pythia-{14m,31m,70m,160m,410m}.parquet`.
Figures: `examples/spectral_tail/figures/`.
Key plots: `figures/chi_pos_signal_vs_controls_pythia-410m.png`, `figures/chi_pos_cross_scale_python_x_cpp.png`.

Current probes (4 eval batches, all describing 7 aligned operations):
- **python**: Python matmul code (verbose, type-hinted, docstrings)
- **cpp**: C++ matmul code (raw pointers, C-style)
- **prose**: English prose describing matmul (no code syntax)
- **cpp_str**: C++ string-processing code (negative control — same language as cpp, different algorithm)

Cross pairs measured:
- `python x cpp` — positive signal (cross-language, same algorithm)
- `python x prose` — positive control (cross-genre, same algorithm)
- `cpp x cpp_str` — negative control (same language, different algorithm)
- `python x cpp_str` — secondary control (cross-language, different algorithm)

B=1, S=1024, n_hutchinson=32. Models: pythia-14m, 31m, 70m, 160m, 410m across 13 training checkpoints. Plots available in step-based and compute-scaled (6ND FLOPs, `--compute` flag) versions.

### Probe design iterations

1. Python verbose vs Python terse — too much syntactic overlap, couldn't isolate semantic layer
2. Python vs C++ with divergent operations — cleaner separation but misaligned operations added noise
3. Python vs C++ with aligned operations — tightest probe, same ops in same order
4. Added prose probe as positive control — tracks python x cpp but weaker, confirms signal is robust to surface form
5. Added cpp_str negative control — shows code-structure baseline; python x cpp signal is 2-4x above python x cpp_str in the late-training window, suggesting partial algorithm specificity
