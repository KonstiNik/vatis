# vatis deployment example — findings & sizing (pythia-14m)

What the `examples/pythia_sweep.py` deployment example evaluates, why it's sized
the way it is, and the non-trivial dynamics it surfaces on real Pythia
checkpoints.

> **Performance benchmarks** (wallclock, memory, DDP scaling, estimator
> variance) live in [`examples/benchmark/`](benchmark/README.md), measured on
> A100s — the representative hardware. This document is about the *example* and
> its *observable-value findings*, which are method- and hardware-independent.

## What the deployment example evaluates on

`examples/pythia_sweep.py` uses **two real-text eval batches**:

- **prose**: the opening of Jane Austen's *Pride and Prejudice* (public
  domain), tokenized with the model's own tokenizer and sliced into
  `B × S = 8 × 128 = 1024` non-overlapping tokens.
- **code**: a small self-contained Python module (linear-algebra
  primitives, sorting, primality), same `B × S = 1024` tokens after
  tokenization.

Real text matters here. The observable values must be theory-relevant: the
chi_net term is the squared Frobenius norm of the parameter Jacobian
**evaluated at the input**; feeding random tokens puts the evaluation at an
off-distribution point in input-space and the resulting trajectory across
checkpoints reflects nothing about Pythia's actual training dynamics. Tokenized
prose puts the evaluation back on the data manifold the model was trained on.
(Shape-only *performance* benchmarks are different — measuring wallclock/memory
is fine on synthetic random integers, which is what `examples/benchmark/` does.)

The example also computes the **cross-pair** observable
`δL(prose, code) = ⟨∇_θ L^prose, ∇_θ L^code⟩` and the corresponding
`chi_pos(prose, code)`. This is essentially free (one extra dot product per
checkpoint, zero extra backward passes — the gradients are already cached from
the self-pair work) and is the most LNP-relevant quantity in the example: it
measures whether a gradient step on prose helps or hurts the model on code.

## Sizing

The example is deliberately sized to run end-to-end in **well under 15 minutes**
on modest hardware, so a maintainer on a slow GPU or a cold HF cache still gets a
quick turnaround. As committed (`per_sequence_cv`, `n_h=32`, `B=8`, `S=128`, 13
revisions × 2 self pairs + 1 cross pair = 39 measurements) the measured wallclock
is:

- **~50 s on the first run** (cold HF cache — the 4 log-spaced early-phase
  checkpoints are fresh ~3–4 s downloads),
- **~20 s on subsequent runs** (warm cache).

The bottleneck is checkpoint loading, not analysis — the per-call compute is a
fraction of a second at these parameters (see `examples/benchmark/single_gpu/`
for the wallclock/memory sweep). Bumping `n_hutchinson`, adding eval batches, or
running both methods would all stay comfortably within budget.

## Revisions and early-phase findings

The example uses **13 revisions** spanning the published Pythia training
schedule, with four log-spaced early checkpoints:

```
step1, step8, step64, step512,                          # log-2 phase
step1000, step2000, step4000, step8000, step16000,      # main phase,
step32000, step64000, step128000, step143000            # roughly log-2
```

The early phase is essential. Without it the trajectories look deceptively
smooth and monotonic; with it several non-trivial features are visible:

1. **chi_loss is flat at 1.0 through step64** on both batches. The model
   literally doesn't improve on next-token prediction in the first 64 SGD
   steps — those steps are warmup, not learning.
2. **chi_net has a U-shape in the first ~100 steps.** It starts at ~1e8 at
   step1 (the random-init Frobenius norm), drops by ~3× to ~3e7 at step64, then
   climbs back up and grows monotonically to ~5e10 by step143000. The dip is not
   noise — it shows up identically on both prose and code, and in self
   delta_loss too. Plausible reading: the very first SGD steps "smooth out"
   high-magnitude initialization noise before the model starts building
   structured representations.
3. **The cross delta_loss has non-monotonic structure.** It starts at **+3.25
   at step1** (prose and code gradients well-aligned on a random model), drops to
   **+0.95 at step64** (essentially orthogonal), climbs back to **+2.62 at
   step1000**, then declines and **crosses zero around step4000**, ending at
   **−18 by step143000**. Without the early checkpoints the trajectory looks
   like a clean monotonic decline from positive to negative — a simpler, and
   partially wrong, story.

The negative end of the cross delta_loss trajectory is the negative-`chi_pos`
interference regime discussed in the paper (arXiv:2605.31244 §A.2): a gradient
step that decreases prose loss *increases* code loss, and vice versa. None of
this would be visible on random-integer inputs, where every batch is
statistically equivalent to every other.

## Reproducing

```bash
.venv/bin/python examples/pythia_sweep.py      # compute → results.parquet + plots
.venv/bin/python examples/analyze_results.py   # analyze → cos_similarity.png (zero vatis imports)
```

(Needs the `examples` extra for matplotlib: `uv pip install ".[examples]"`. Set
`HF_HOME` via your shell or the repo-root `.env` to relocate the model cache.)

`analyze_results.py` reads `results.parquet` and produces `cos_similarity.png` —
the normalized cross-batch gradient correlation
`cos(g_A, g_B) = δL(A,B) / sqrt(δL(A,A) · δL(B,B))` — plus a printed summary.
The cosine view exists because the raw cross `delta_loss` plot is visually
misleading: it shows a step1 → step512 decline followed by a "rebound" at
step1000, but the rebound is amplified by gradient magnitudes growing
simultaneously. In normalized terms the rebound is real but ~10× smaller than
the absolute number suggests, and the dominant feature is the early-phase
decline (`+0.31 → +0.012`, ~26×) — i.e. **most of the prose-vs-code gradient
decorrelation is over by step512**, well before the absolute cross delta_loss
even starts looking interesting.

The analysis script is intentionally separate from `pythia_sweep.py`: compute is
expensive (~50 s cold), analysis is cheap (sub-second), and the analysis script
imports only `pyarrow` (not vatis) — making the parquet schema the contract, the
right pattern for any downstream consumer.

Outputs (under `examples/pythia_sweep/`):

- `results.parquet` — canonical long-format table (234 rows: 13 ckpts × 6
  observables × (2 self pairs + 1 cross pair)).
- `chi_loss.png`, `chi_net.png` — self-pair trajectories (one line per eval
  batch).
- `delta_loss.png`, `chi_pos.png` — self pairs + the prose×code cross pair
  (symlog, since the cross values cross zero).
- `cos_similarity.png` (from `analyze_results.py`) — the normalized cross-batch
  correlation, with gradient magnitudes divided out.
