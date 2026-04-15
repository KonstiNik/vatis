# Spectral tail overlap hypothesis

Detecting when training resolves shared semantic structure between two eval batches via a chi_pos bump in the LNA decomposition.

## Core idea

If two eval batches A and B share similarity only in the "tail" of the NTK spectrum (i.e. deeper/later-learned features), then:

- **Early training**: gradients are orthogonal (chi_pos near zero) because the model hasn't resolved that spectral band yet.
- **Mid training**: when the optimizer's "frontier" reaches the shared band, chi_pos should bump up — the cross coupling becomes visible.
- **Late training**: the shared band is resolved and the bump subsides, or the model specializes further and the coupling is consumed.

This would appear as a transient bump in `chi_pos(A, B)` at the checkpoint where training "arrives at" the spectral band where A and B overlap.

## Probe design intuition

Model of what LMs learn in order:
1. Token combinations into words (surface statistics)
2. Words into sentences (syntax)
3. Semantic content (what the code *does*)

Good probes should **differ at stages 1-2 but share stage 3**. This is why Python vs C++ implementing the same algorithm is a better probe than two Python styles — Python vs C++ eliminates almost all syntactic overlap, isolating the semantic layer.

Key design principles discovered:
- **Repetition is signal, not redundancy.** Multiple implementations of the same algorithm within one text creates within-context pressure for the model to build abstract representations. A single function plus filler doesn't create that pressure.
- **Parallel structure matters.** Both texts should implement the same operations in the same order (matmul, transpose, dot, trace, Frobenius) — not divergent operations (Strassen in one, Gram-Schmidt in the other).
- **B=1 with long S is optimal.** Longer sequences give the model more context to build semantic representations. B doesn't affect statistical quality (same total tokens, Hutchinson noise depends on n_hutchinson not B). B=1, S=1024 maximizes context per position.

## Why raw cross chi_pos, not a normalized ratio

An earlier version used `chi_pos(A,B) / sqrt(chi_pos(A,A) * chi_pos(B,B))` to factor out the shared chi_net magnitude trend. This turned out to suppress the signal we're looking for.

The reason: each self pair (Python, Python) already contains multiple matmul implementations, so the "matmul" spectral band contributes to the self chi_pos. Dividing the cross by the self divides out the very band we're trying to detect. The normalized ratio isolates only the language-switch effect, not the semantic-resolution effect.

The raw cross chi_pos is the right quantity. It measures total spectral overlap between Python-matmul and C++-matmul. Language-specific features contribute to the self pairs but not to the cross term, so they don't inflate the signal.

## Findings on model scale

Larger models decorrelate the two languages faster and more completely:
- 14m: cross chi_pos stays positive throughout training
- 31m, 70m: decline more steeply, 70m shows a secondary rise around step16000-64000
- 160m: declines fastest, touches zero/negative at step143000

All models show a peak at step64 in the raw cross chi_pos, driven by the shared chi_net dip (the Jacobian norm shrinks in the first ~100 steps). This is a real effect but reflects early restructuring, not semantic resolution.

The late-training structure (step8000+) is where the semantic signal lives. The 70m secondary rise around step16000-64000 is particularly interesting — it could be the model resolving a spectral band where Python and C++ matmul implementations overlap.

## Experiment status (2026-04-15)

Code: `examples/spectral_tail_experiment.py` (compute), `examples/spectral_tail_evaluate.py` (plots).
Results: `examples/spectral_tail/results_pythia-{14m,31m,70m,160m}.parquet`.
Key plot: `chi_pos_cross_scale.png`.

Current probe: Python vs C++ implementing the same 7 operations (naive matmul, accumulator matmul, transpose-and-dot matmul, transpose, dot product, trace, Frobenius norm) in the same order. B=1, S=1024, n_hutchinson=32. Models: pythia-14m, 31m, 70m, 160m across 13 training checkpoints.

Iterations that led here:
1. Python verbose vs Python terse — too much syntactic overlap, couldn't isolate semantic layer
2. Python vs C++ with divergent operations — cleaner separation but misaligned operations added noise
3. Python vs C++ with aligned operations (current) — tightest probe, same ops in same order
