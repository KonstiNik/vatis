"""Stage 1 (ONLINE, login node): prefetch the model + select the eval samples.

The compute nodes on this cluster have **no internet** (``HF_HUB_OFFLINE=1`` in
the repo ``.env``), so everything the offline analysis needs must be pulled into
the HF cache and written to a local file first. This script does both:

1. **Selects the eval data** — streams ``allenai/Dolci-Think-SFT`` (the dataset
   ``Olmo-3-7B-Think-SFT`` was trained on, so the observables are evaluated
   on-distribution) and picks the first ``--n-per-batch`` *code* samples and
   ``--n-per-batch`` *English* samples whose prompt is short enough to leave
   real completion tokens after truncation to ``--seq-len``. The raw chat
   ``messages`` are written to ``samples.json`` so the offline step needs no
   dataset dependency at all.

2. **Warms the model into the HF cache** — downloads the tokenizer and (unless
   ``--skip-model-download``) the model weights so ``olmo_sft_analyze.py`` can
   load them offline.

This is the only part of the example that needs the ``datasets`` package and a
network connection. ``datasets`` is intentionally **not** a repo dependency
(vatis is dataset-agnostic); install it into the venv just for this step::

    uv pip install --python .venv/bin/python datasets

Run on a login node (which has internet)::

    .venv/bin/python examples/olmo_sft_analysis/prefetch.py

Set ``HF_HOME`` (shell or repo-root ``.env``) to relocate the cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_EXAMPLES_DIR = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))  # run_config
sys.path.insert(0, str(_EXAMPLES_DIR))  # _helpers

# Force ONLINE before run_config loads .env (which pins offline=1 for compute
# nodes). configure_hf_cache() uses os.environ.setdefault, so pre-setting these
# to "0" wins while HF_HOME is still taken from .env.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from run_config import configure_hf_cache  # noqa: E402

configure_hf_cache()

from _helpers import sft_prompt_completion_lengths  # noqa: E402

# ---------------------------------------------------------------- config

MODEL = "allenai/Olmo-3-7B-Think-SFT"
DATASET = "allenai/Dolci-Think-SFT"
SPLIT = "train"
OUT_PATH = _HERE.parent / "samples.json"

# Code samples: source name names a programming/algorithms split.
CODE_SOURCE_RE = re.compile(r"(Code|Python|Algorithms|Nemotron)", re.IGNORECASE)
# English natural-language chat: curated non-code, non-math, English sources.
# (Aya is excluded — it is multilingual; math/STEM splits are excluded — they
# are English text but not natural-*language* in the prose sense we want as the
# contrast with code.)
ENGLISH_SOURCES = {
    "Persona Precise IF",
    "OpenAssistant",
    "CoCoNot",
    "WildChat",
    "Dolci Think Precise IF",
}


def classify(source: str) -> str | None:
    if CODE_SOURCE_RE.search(source or ""):
        return "code"
    if source in ENGLISH_SOURCES:
        return "english"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--n-per-batch", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--max-prompt-frac",
        type=float,
        default=0.5,
        help="reject a sample if its prompt exceeds this fraction of seq_len "
        "(guarantees real completion tokens survive truncation).",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=20000,
        help="max dataset rows to stream before giving up on filling a batch.",
    )
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="select data and fetch the tokenizer only; skip the ~14GB weights.",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"prefetch — model={args.model}  dataset={args.dataset}")
    print(f"  HF_HOME={os.environ.get('HF_HOME', '(default)')}")
    print(
        f"  selecting {args.n_per_batch} code + {args.n_per_batch} english, "
        f"seq_len={args.seq_len}, max_prompt_frac={args.max_prompt_frac}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_prompt = int(args.max_prompt_frac * args.seq_len)

    ds = load_dataset(args.dataset, split=SPLIT, streaming=True)
    batches: dict[str, list[dict]] = {"code": [], "english": []}
    selected_meta: dict[str, list[dict]] = {"code": [], "english": []}
    n_scanned = 0

    for ex in ds:
        n_scanned += 1
        if n_scanned > args.scan_limit:
            break
        kind = classify(ex.get("source", ""))
        if kind is None or len(batches[kind]) >= args.n_per_batch:
            continue
        messages = ex["messages"]
        # Need a final assistant turn to have a completion to learn.
        if not messages or messages[-1].get("role") != "assistant":
            continue
        try:
            prompt_len, total_len = sft_prompt_completion_lengths(messages, tokenizer)
        except Exception:  # noqa: BLE001 — skip samples the template can't render
            continue
        if prompt_len > max_prompt or total_len <= prompt_len:
            continue  # no usable completion tokens at this seq_len

        batches[kind].append({"id": ex.get("id"), "source": ex.get("source"), "messages": messages})
        selected_meta[kind].append(
            {
                "id": ex.get("id"),
                "source": ex.get("source"),
                "prompt_len": prompt_len,
                "total_len": total_len,
                "completion_len_at_seq_len": max(0, min(total_len, args.seq_len) - prompt_len),
            }
        )
        if all(len(batches[k]) >= args.n_per_batch for k in batches):
            break

    for k in batches:
        got = len(batches[k])
        if got < args.n_per_batch:
            print(
                f"  WARNING: only found {got}/{args.n_per_batch} '{k}' samples "
                f"in {n_scanned} rows. Raise --scan-limit or relax the filters."
            )
        print(f"  {k}: {got} samples; sources={sorted({m['source'] for m in selected_meta[k]})}")

    payload = {
        "meta": {
            "model": args.model,
            "dataset": args.dataset,
            "split": SPLIT,
            "seq_len_target": args.seq_len,
            "n_per_batch": args.n_per_batch,
            "max_prompt_frac": args.max_prompt_frac,
            "code_source_regex": CODE_SOURCE_RE.pattern,
            "english_sources": sorted(ENGLISH_SOURCES),
            "n_rows_scanned": n_scanned,
            "selected": selected_meta,
        },
        "batches": batches,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    if args.skip_model_download:
        print("skipped model weight download (--skip-model-download); tokenizer cached.")
        return

    print(
        f"downloading model weights for {args.model} into the HF cache (this is the ~14GB step) ..."
    )
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  cached model: {n_params:,} parameters. Ready for offline analysis.")
    del model


if __name__ == "__main__":
    main()
