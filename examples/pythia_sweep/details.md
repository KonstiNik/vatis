# Pythia-14m sweep — details

The longer notes behind [README.md](README.md): what the example evaluates on,
how it is sized, and the early-phase findings in full.

## What it evaluates on

`run.py` uses **two real-text eval batches**:

- **prose**: the opening of Jane Austen's *Pride and Prejudice* (public
  domain), tokenized with the model's own tokenizer and sliced into
  `B × S = 8 × 128 = 1024` non-overlapping tokens.
- **code**: a small self-contained Python module (linear-algebra primitives,
  sorting, primality), the same `B × S = 1024` tokens after tokenization.

Real text matters here. `chi_net` is the squared Frobenius norm of the
parameter Jacobian **evaluated at the input**. Random tokens put that
evaluation at an off-distribution point in input space, and the trajectory
across checkpoints then reflects nothing about Pythia's actual training.
Tokenized prose puts the evaluation back on the data the model was trained on.
(Performance benchmarks are different: wall-clock and memory can be measured
on random integers, which is what [`../benchmark/`](../benchmark/README.md)
does.)

The example also computes the **cross pair**
`δL(prose, code) = ⟨∇_θ L^prose, ∇_θ L^code⟩` and the matching
`chi_pos(prose, code)`. This is almost free — one extra dot product per
checkpoint and no extra backward pass, because the gradients are already
cached from the self pairs — and it is the most LNP-relevant quantity in the
example: it measures whether a gradient step on prose helps or hurts the model
on code.

## Sizing

The example is sized to finish well within 15 minutes. As committed
(`per_sequence_cv`, `n_hutchinson=32`, `B=8`, `S=128`, 13 revisions × (2 self
pairs + 1 cross pair) = 39 measurements), it takes about **76 s** on an
RTX 3090 Ti the first time (the 13 checkpoints are downloaded) and **52–55 s**
once they are cached.

Most of that time is the analysis, about 1.5 s per measurement. About 0.6 s of
it is drawing the random Hutchinson probes on the CPU: one probe at this size
(8 × 128 × 50,304 entries) takes about 18 ms that way, against about 1 ms when
drawn on the GPU, and each measurement draws 32. Before the probes were drawn
on the CPU (April 2026), a measurement took about 0.5 s and a cached run about
20 s. See [`../benchmark/single_gpu/`](../benchmark/single_gpu/results_pythia-160m.md)
for the wall-clock and memory sweep on an A100.

The analysis script is separate from `run.py` on purpose: the compute is the
expensive part (about a minute on an RTX 3090 Ti), the analysis is cheap (under
a second), and `evaluate.py` imports no vatis code. That makes the
parquet schema the contract — the right pattern for any downstream consumer.

## Revisions and early-phase findings

The example uses **13 revisions** spanning the published Pythia training
schedule, with four log-spaced early checkpoints:

```
step1, step8, step64, step512,                          # log-2 phase
step1000, step2000, step4000, step8000, step16000,      # main phase,
step32000, step64000, step128000, step143000            # roughly log-2
```

The early phase is essential. Without it the trajectories look deceptively
smooth and monotonic; with it several non-trivial features are visible
(`chi_loss` and `chi_net` below are the normalized values the plots show):

1. **`chi_loss` is flat at 1.0 through step 64** on both batches. The model
   does not improve on next-token prediction in the first 64 steps — those
   steps are warm-up, not learning.
2. **`chi_net` has a U-shape in the first ~100 steps.** It starts at ~1e8 at
   step 1 (the random-initialization Frobenius norm), drops by ~3× to ~3e7 at
   step 64, then climbs back and grows monotonically to ~5e10 by step 143000.
   The dip is not noise: it shows up the same way on prose and on code, and in
   the self `delta_loss` too. A plausible reading: the very first steps smooth
   out high-magnitude initialization noise before the model starts building
   structured representations.
3. **The cross `delta_loss` has non-monotonic structure.** It starts at
   **+3.25 at step 1** (prose and code gradients well aligned on a random
   model), drops to **+0.95 at step 64** and to its lowest positive value,
   **+0.67, at step 512**, climbs back to **+2.62 at step 1000**, then
   declines, **crosses zero around step 4000** and ends at **−18 at step
   143000**. The drop to step 64 is mostly the gradients shrinking (the self
   `delta_loss` falls from 7.9 to 2.5 on prose and from 13.8 to 5.5 on code;
   the cosine is still 0.25 at step 64); by step 512 the gradients are nearly
   orthogonal (cosine 0.012). Without the early checkpoints the
   trajectory looks like a clean monotonic decline from positive to negative —
   a simpler, and partly wrong, story.

The negative end of the cross `delta_loss` trajectory is the negative-`chi_pos`
interference regime (arXiv:2605.31244 §A.2; not re-checked against the paper
here): a gradient step that decreases prose loss *increases*
code loss, and vice versa. None of this would be visible on random-integer
inputs, where every batch is statistically equivalent to every other.

## Why the cosine view

`evaluate.py` reads `results.parquet` and produces `cos_similarity.png` — the
normalized cross-batch gradient correlation
`cos(g_A, g_B) = δL(A,B) / sqrt(δL(A,A) · δL(B,B))` — plus a printed summary.
The raw cross `delta_loss` plot is visually misleading: it shows a
step 1 → step 512 decline followed by a "rebound" at step 1000. The rebound
itself, from step 512 to step 1000, is about the same size in both views (3.9×
raw, 3.6× in the cosine; the product of the two gradient norms grows only 1.07×
between those steps). What the raw plot exaggerates is the level: the raw cross
`delta_loss` at step 1000 is back to 81% of its step-1 value (2.62 against
3.25), the cosine only to 14% (0.043 against 0.311), because the product of the
gradient norms grew from 10.5 at step 1 to 56.6 at step 512. The dominant
feature is the early decline (`+0.31 → +0.012`, ~26×): **most
of the prose-vs-code gradient decorrelation is over by step 512**, well before
the absolute cross `delta_loss` even starts looking interesting.

## Outputs

All under `examples/pythia_sweep/`:

- `results.parquet` — the long-format result table (234 rows: 13 checkpoints ×
  6 observables × (2 self pairs + 1 cross pair)).
- `chi_loss.png`, `chi_net.png` — self-pair trajectories (one line per eval
  batch).
- `delta_loss.png`, `chi_pos.png` — self pairs and the prose × code cross pair
  (symlog, since the cross values cross zero).
- `cos_similarity.png` (from `evaluate.py`) — the normalized cross-batch
  correlation, with gradient magnitudes divided out.

`run.py` needs the `examples` extra for matplotlib
(`uv pip install ".[examples]"`). Set `HF_HOME` in your shell or in the
repo-root `.env` to relocate the model cache.
