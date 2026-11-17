# Spectral tail overlap — details

The hypothesis and the probe design behind [README.md](README.md): detecting
when training resolves shared semantic structure between two eval batches via
a `chi_pos` bump in the LNP decomposition.

## The hypothesis

If two eval batches A and B share similarity only in the "tail" of the NTK
spectrum (i.e. deeper, later-learned features), then:

- **Early training**: gradients are orthogonal (`chi_pos` near zero) because
  the model has not resolved that spectral band yet.
- **Mid training**: when the optimizer's "frontier" reaches the shared band,
  `chi_pos` should bump up — the cross coupling becomes visible.
- **Late training**: the shared band is resolved and the bump subsides, or the
  model specializes further and the coupling is consumed.

This would appear as a transient bump in `chi_pos(A, B)` at the checkpoint
where training "arrives at" the spectral band where A and B overlap.

## Probe design

A model of what language models learn, in order:

1. Token combinations into words (surface statistics)
2. Words into sentences (syntax)
3. Semantic content (what the code *does*)

Good probes should **differ at stages 1–2 but share stage 3**. This is why
Python vs C++ implementing the same algorithm is a better probe than two Python
styles: Python vs C++ removes almost all syntactic overlap, isolating the
semantic layer.

Design principles:

- **Repetition is signal, not redundancy.** Several implementations of the same
  algorithm within one text create within-context pressure for the model to
  build abstract representations.
- **Parallel structure matters.** Both texts should implement the same
  operations in the same order (matmul, transpose, dot, trace, Frobenius) — not
  divergent operations.
- **B = 1 with long S is optimal.** Longer sequences give the model more
  context to build semantic representations. B = 1, S = 1024 maximizes context
  per position.

## The probes and pairs measured

Four eval batches, all describing 7 aligned operations:

- **python**: Python matmul code (verbose, type-hinted, docstrings)
- **cpp**: C++ matmul code (raw pointers, C-style)
- **prose**: English prose describing matmul (no code syntax)
- **cpp_str**: C++ string-processing code (negative control — same language as
  cpp, different algorithm)

Cross pairs:

- `python x cpp` — positive signal (cross-language, same algorithm)
- `python x prose` — positive control (cross-genre, same algorithm)
- `cpp x cpp_str` — negative control (same language, different algorithm)
- `python x cpp_str` — secondary control (cross-language, different algorithm)

B = 1, S = 1024, `n_hutchinson=32`. Models: pythia-14m, 31m, 70m, 160m, 410m
across 13 training checkpoints. Plots exist in step-based and compute-scaled
(6ND FLOPs, `--compute` flag) versions. Key plots:
[`figures/chi_pos_signal_vs_controls_pythia-410m.png`](figures/chi_pos_signal_vs_controls_pythia-410m.png),
[`figures/chi_pos_cross_scale_python_x_cpp.png`](figures/chi_pos_cross_scale_python_x_cpp.png).

## Probe design iterations

1. Python verbose vs Python terse — too much syntactic overlap, could not
   isolate the semantic layer.
2. Python vs C++ with divergent operations — cleaner separation, but the
   misaligned operations added noise.
3. Python vs C++ with aligned operations — the tightest probe, same operations
   in the same order.
4. Added the prose probe as a positive control — tracks `python x cpp` but
   weaker, which confirms the signal is robust to surface form.
5. Added the `cpp_str` negative control — shows the code-structure baseline;
   the `python x cpp` signal is 2–4× above `python x cpp_str` in the
   late-training window, suggesting partial algorithm specificity.
