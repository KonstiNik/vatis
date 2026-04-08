"""vatis deployment example: a chi_pos trajectory across pythia-14m revisions.

Single-GPU end-to-end example. Loads ``EleutherAI/pythia-14m`` at nine
training checkpoints, runs a single ``analyze()`` call per checkpoint
with two distinct fixed eval batches, writes the result to a long-format
parquet file, and produces three plots showing each observable as a
function of training step.

Sized to fit comfortably under the 15-minute budget on the reference
hardware (RTX 3090 Ti, see ``examples/BENCHMARK.md`` for the budget
arithmetic). On that hardware the script wallclock is ~75 s. Slower
hardware or a cold HF cache will take longer; the rate-limiting step is
the per-checkpoint download/load, not the chi_net computation.

Run with::

    HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/pythia_sweep.py

Outputs are written next to this script:

- ``examples/results.parquet``  — canonical long-format result table
- ``examples/chi_loss.png``      — chi_loss_normalized vs step
- ``examples/chi_net.png``       — chi_net_normalized vs step
- ``examples/chi_pos.png``       — chi_pos vs step
"""

from __future__ import annotations

import os

# Pin the HF cache before transformers / vatis are imported. This keeps
# the home filesystem from filling up with checkpoint shards.
os.environ.setdefault("HF_HOME", "/data/knikolaou/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/knikolaou/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/knikolaou/huggingface")

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display backend

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import torch

from vatis import analyze
from vatis.models.hf import load_hf_model  # noqa: F401  (kept for the docstring example)

# ---------------------------------------------------------------- config

MODEL = "EleutherAI/pythia-14m"

# Nine pythia checkpoints spanning the published training schedule
# (pythia-14m has revisions at every 1000 steps from 1000 to 143000).
# A coarse log-spaced subset gives a clean trajectory without spending
# too much time on checkpoint loading.
REVISIONS: list[str] = [
    "step1000",
    "step2000",
    "step4000",
    "step8000",
    "step16000",
    "step32000",
    "step64000",
    "step128000",
    "step143000",
]

BATCH_SIZE = 8
SEQ_LEN = 128
N_HUTCHINSON = 32
SEED = 0

# Two distinct eval batches at the same shape — different seeds, fixed
# at script start so each checkpoint sees the same content. The pair
# gives us two trajectories per observable, which is the minimal version
# of "compare across eval distributions at fixed checkpoint" — the use
# case vatis exists for.
EVAL_BATCH_NAMES = ("eval_a", "eval_b")
EVAL_BATCH_SEEDS = (0, 1)

OUT_DIR = Path(__file__).parent
PARQUET_PATH = OUT_DIR / "results.parquet"
PLOTS = {
    "chi_loss_normalized": OUT_DIR / "chi_loss.png",
    "chi_net_normalized": OUT_DIR / "chi_net.png",
    "chi_pos": OUT_DIR / "chi_pos.png",
}

# ---------------------------------------------------------------- helpers


def make_synthetic_lm_batch(
    *, vocab_size: int, batch_size: int, seq_len: int, seed: int
) -> dict[str, torch.Tensor]:
    """Reproducible HF-style causal-LM batch.

    Synthetic on purpose: this example is about demonstrating vatis on
    real published checkpoints, not about loading a real eval set.
    Tokenized real corpora would add a download step and an extra
    dependency without changing the observables that the example is
    plotting.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, :-1] = input_ids[:, 1:]  # standard HF shift convention
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def step_from_revision(rev: str) -> int:
    """Parse the trailing integer step out of a Pythia revision tag."""
    return int(rev.removeprefix("step"))


# ---------------------------------------------------------------- main


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"vatis deployment example — model={MODEL}")
    print(f"  device={device}")
    print(f"  revisions={REVISIONS}")
    print(f"  B={BATCH_SIZE}, S={SEQ_LEN}, n_hutchinson={N_HUTCHINSON}")
    print(f"  eval_batches={EVAL_BATCH_NAMES}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PARQUET_PATH.exists():
        PARQUET_PATH.unlink()  # fresh write — don't append to a stale file

    # ---- determine the vocab size from one revision before building batches
    # (so we can size the synthetic batches to match the model's vocab).
    from vatis.models.hf import load_hf_model as _load

    bundle0 = _load(MODEL, revision=REVISIONS[0], dtype="fp32", device=device)
    vocab_size = int(bundle0.model.config.vocab_size)
    del bundle0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_batches = {
        name: make_synthetic_lm_batch(
            vocab_size=vocab_size,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            seed=seed,
        )
        for name, seed in zip(EVAL_BATCH_NAMES, EVAL_BATCH_SEEDS, strict=True)
    }

    # ---- single sweep call: analyze() iterates over revisions and writes
    # one row per (checkpoint, batch, observable) into the same parquet
    # sink. This is the canonical way to drive vatis on a checkpoint
    # series — calling analyze() once per revision would create a fresh
    # ParquetSink each time and truncate the file.
    t_total = time.perf_counter()
    analyze(
        model=MODEL,
        revisions=REVISIONS,
        eval_batches=eval_batches,
        chi_net_method="per_sequence_cv",
        n_hutchinson=N_HUTCHINSON,
        micro_batch_size=BATCH_SIZE,
        seed=SEED,
        device=device,
        dtype="fp32",
        sink=str(PARQUET_PATH),
    )
    total_s = time.perf_counter() - t_total

    # ---- read back, plot
    table = pq.read_table(PARQUET_PATH)
    print(f"\nresult table: {table.num_rows} rows, {table.num_columns} columns")

    rows = table.to_pylist()
    # Index rows by (observable, batch) → sorted list of (step, value)
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["observable"], row["batch_a"])
        series.setdefault(key, []).append((step_from_revision(row["revision"]), row["value"]))
    for key in series:
        series[key].sort()

    for obs_name, plot_path in PLOTS.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for batch_name in EVAL_BATCH_NAMES:
            xs_ys = series.get((obs_name, batch_name), [])
            if not xs_ys:
                continue
            xs = [p[0] for p in xs_ys]
            ys = [p[1] for p in xs_ys]
            ax.plot(xs, ys, marker="o", label=batch_name)
        ax.set_xlabel("training step")
        ax.set_ylabel(obs_name)
        ax.set_xscale("log")
        if obs_name == "chi_net_normalized":
            ax.set_yscale("log")
        ax.set_title(f"{obs_name} vs training step  ({MODEL})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"wrote {plot_path}")

    print(
        "\nDone in "
        f"{total_s:.1f}s. Ran {len(REVISIONS)} checkpoints × "
        f"{len(EVAL_BATCH_NAMES)} eval batches × 1 chi_net_method "
        f"({len(REVISIONS) * len(EVAL_BATCH_NAMES)} analyze() calls). "
        f"Outputs at {OUT_DIR}/."
    )


if __name__ == "__main__":
    main()
