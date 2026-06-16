"""Stage 2 (OFFLINE, GPU node): LNP observables on Olmo-3-7B-Think-SFT.

A realistic single-checkpoint SFT analysis. Loads ``Olmo-3-7B-Think-SFT`` from
the (pre-populated, offline) HF cache, builds two real eval batches from the
samples ``prefetch.py`` saved — 10 *code* SFT examples and 10 *English* SFT
examples, each tokenized with the model's chat template and **completion-only
loss masking** (the actual SFT objective: cross-entropy on assistant tokens
only) — and computes the LNP decomposition with vatis:

- ``chi_pos`` — the **spectral position** of the SFT loss gradient, in [0, 1].
  Near 1 = the gradient exploits the bulk (well-resolved, dominant eigenmodes);
  near 0 = it lives in the tail (weak, fine-grained directions).
- ``chi_loss`` / ``chi_net`` (and their normalized forms), ``delta_loss``.
- the **cross pair** ``(english, code)``: ``delta_loss`` and ``chi_pos`` between
  the two batches' gradients — do an SFT step on English and one on code help or
  fight each other at this checkpoint? (positive ``delta_loss`` = transfer,
  negative = interference.)

Resource usage (wall time, peak GPU memory, GPU engine activity via DCGM) is
tracked throughout and written to ``resources.json``; see ``resource_tracker``.

Run offline on a GPU node (the cache must already hold the model — run
``prefetch.py`` on a login node first). Via SLURM::

    examples/benchmark/submit.sh examples/olmo_sft_analysis/run.sbatch

or interactively on a GPU::

    .venv/bin/python examples/olmo_sft_analysis/olmo_sft_analyze.py

Outputs (next to this script):

- ``results.parquet``  — canonical long-format LNP table
- ``resources.json``   — time / memory / GPU-activity summary + timeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_EXAMPLES_DIR = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))  # run_config
sys.path.insert(0, str(_EXAMPLES_DIR))  # _helpers

from run_config import configure_hf_cache  # noqa: E402

# On the compute node this pins HF_HOME and (via .env) HF_HUB_OFFLINE=1 so HF
# only ever touches the pre-populated cache. Call before importing torch/transformers.
configure_hf_cache()

import time  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from _helpers import stack_lm_batch, tokenize_chat_sft_sample  # noqa: E402
from resource_tracker import ResourceTracker  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vatis import analyze  # noqa: E402
from vatis.models.hf import load_hf_model  # noqa: E402

SAMPLES_PATH = _HERE.parent / "samples.json"
PARQUET_PATH = _HERE.parent / "results.parquet"
RESOURCES_PATH = _HERE.parent / "resources.json"

CODE_NAME = "code"
ENGLISH_NAME = "english"
CROSS_PAIRS = [(ENGLISH_NAME, CODE_NAME)]


def build_batch(samples: list[dict], tokenizer, seq_len: int) -> dict[str, torch.Tensor]:
    """Tokenize a list of chat samples into one stacked completion-only LM batch."""
    return stack_lm_batch(
        [tokenize_chat_sft_sample(s["messages"], tokenizer, seq_len=seq_len) for s in samples]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(SAMPLES_PATH))
    parser.add_argument(
        "--seq-len", type=int, default=None, help="override the seq_len recorded in samples.json"
    )
    parser.add_argument("--n-hutchinson", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--chi-net-method", default="per_sequence_cv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler-interval-s", type=float, default=0.25)
    parser.add_argument(
        "--no-cross-pairs",
        action="store_true",
        help="skip the (english, code) cross pair. With --cross-grad-storage "
        "auto the cross pair fits even on a 7B model (the cached gradients "
        "offload to host RAM and the dot runs on CPU), so this is opt-out.",
    )
    parser.add_argument(
        "--cross-grad-storage",
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="where cached cross-pair gradients live. 'auto' (default) offloads "
        "to host RAM for large models so the cross dot fits, keeps small models "
        "on-device; 'gpu'/'cpu' force it. See vatis.Analyzer.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="optional attention-backend passthrough to from_pretrained "
        "(e.g. 'sdpa', 'flash_attention_2', 'eager'). Default None = the model's "
        "own default (sdpa for OLMo3). NOTE: for OLMo3 sdpa does NOT reduce the "
        "O(S^2) attention memory, so it does not raise the seq_len ceiling here.",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.exists():
        raise SystemExit(
            f"{samples_path} not found. Run prefetch.py on a login node first "
            "(see this example's README)."
        )
    payload = json.loads(samples_path.read_text())
    meta = payload["meta"]
    model_name = meta["model"]
    seq_len = args.seq_len or meta["seq_len_target"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = "bf16"  # matches OLMo training precision; grads accumulate in fp32
    else:
        device = torch.device("cpu")
        dtype = "fp32"
        print("WARNING: no CUDA device — a 7B model will not realistically run on CPU.")

    cross_pairs = [] if args.no_cross_pairs else CROSS_PAIRS

    print(f"olmo SFT analysis — model={model_name}")
    print(f"  device={device}, dtype={dtype}, seq_len={seq_len}, attn={args.attn_implementation}")
    print(
        f"  chi_net_method={args.chi_net_method}, n_hutchinson={args.n_hutchinson}, "
        f"micro_batch_size={args.micro_batch_size}, cross_pairs={cross_pairs}, "
        f"cross_grad_storage={args.cross_grad_storage}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_batches = {
        CODE_NAME: build_batch(payload["batches"][CODE_NAME], tokenizer, seq_len),
        ENGLISH_NAME: build_batch(payload["batches"][ENGLISH_NAME], tokenizer, seq_len),
    }
    for name, batch in eval_batches.items():
        n_tokens = int(batch["input_ids"].numel())
        n_valid = int((batch["labels"] != -100).sum().item())
        b, s = tuple(batch["input_ids"].shape)
        print(
            f"  {name}: B={b} S={s}, {n_valid} valid completion tokens "
            f"({n_valid / n_tokens:.1%} of {n_tokens})"
        )

    if PARQUET_PATH.exists():
        PARQUET_PATH.unlink()  # fresh write; ParquetSink truncates on re-open anyway

    tracker = ResourceTracker(device=device, interval_s=args.sampler_interval_s)
    with tracker:
        with tracker.measure("load_model"):
            t0 = time.perf_counter()
            bundle = load_hf_model(
                model_name,
                dtype=dtype,
                device=device,
                attn_implementation=args.attn_implementation,
            )
            n_params = sum(p.numel() for p in bundle.params)
            print(f"  loaded {n_params:,} params in {time.perf_counter() - t0:.1f}s")

        with tracker.measure("analyze"):
            analyze(
                model=bundle,
                revisions=["sft"],  # single checkpoint; label only
                eval_batches=eval_batches,
                cross_pairs=cross_pairs,
                cross_grad_storage=args.cross_grad_storage,
                chi_net_method=args.chi_net_method,
                n_hutchinson=args.n_hutchinson,
                micro_batch_size=args.micro_batch_size,
                seed=args.seed,
                device=device,
                dtype=dtype,
                sink=str(PARQUET_PATH),
            )

    tracker.save(str(RESOURCES_PATH))
    print(f"\nwrote {PARQUET_PATH}")
    print(f"wrote {RESOURCES_PATH}")

    # ---- compact console summary of the spectral-position story ----
    table = pq.read_table(PARQUET_PATH).to_pylist()
    print("\n=== LNP observables ===")
    wanted = ("chi_pos", "delta_loss", "chi_loss_normalized", "chi_net_normalized")
    for obs in wanted:
        for row in table:
            if row["observable"] != obs:
                continue
            a, b = row["batch_a"], row["batch_b"]
            pair = a if a == b else f"{a}×{b}"
            print(f"  {obs:<20} {pair:<16} {row['value']:+.4e}")

    print("\n=== resources ===")
    print(tracker.format_summary())


if __name__ == "__main__":
    main()
