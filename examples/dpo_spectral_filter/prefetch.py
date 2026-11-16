"""Stage 1 (ONLINE, login node): select eval preference pairs + warm both checkpoints.

Implements the data side of the diagnostics in [`README.md`](README.md):
``R_dir`` and ``R_spec`` need, for each of a small set of (prompt, chosen,
rejected) triples, both a completion-only SFT-style CE loss on ``chosen`` and
a DPO loss on ``(chosen, rejected)`` — evaluated at two checkpoints,
``allenai/Olmo-3-7B-Instruct-SFT`` and ``allenai/Olmo-3-7B-Instruct-DPO`` (DPO
continues training from the SFT checkpoint, and is also the fixed reference
policy throughout DPO training).

This streams ``allenai/Dolci-Instruct-DPO`` and selects ``--n-per-batch``
*code* and ``--n-per-batch`` *english* preference pairs (classified by a regex
over ``prompt_id``, the only field that names the underlying source dataset)
whose prompt is short enough to leave room for both completions after
truncation to ``--seq-len``. Raw ``chosen``/``rejected`` message lists are
written to ``samples.json``; both checkpoints' weights are warmed into the HF
cache.

Needs ``datasets`` (not a vatis dependency; install into the venv just for
this step)::

    uv pip install --python .venv/bin/python datasets

Run on a login node::

    .venv/bin/python examples/dpo_spectral_filter/prefetch.py

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

SFT_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
DPO_MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
DATASET = "allenai/Dolci-Instruct-DPO"
SPLIT = "train"
OUT_PATH = _HERE.parent / "samples.json"

# allenai/Dolci-Instruct-DPO has no clean domain column; `prompt_id` embeds the
# underlying source-dataset slug (e.g. "correct-python-sft-187k-request-...",
# "valpy_if_qwq_reasoning_verified-...") — the same code/english contrast used
# in olmo_sft_analysis, ported to this dataset's naming.
CODE_RE = re.compile(r"(correct-python-sft|codealpaca|persona.*python)", re.IGNORECASE)
ENGLISH_RE = re.compile(r"(valpy_if|flan_v2|coconot)", re.IGNORECASE)


def classify(prompt_id: str) -> str | None:
    if CODE_RE.search(prompt_id or ""):
        return "code"
    if ENGLISH_RE.search(prompt_id or ""):
        return "english"
    return None


def _last_assistant_ok(messages: list[dict]) -> bool:
    return bool(messages) and messages[-1].get("role") == "assistant"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", default=SFT_MODEL)
    parser.add_argument("--dpo-model", default=DPO_MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--n-per-batch", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--max-prompt-frac",
        type=float,
        default=0.5,
        help="reject a pair if its prompt exceeds this fraction of seq_len "
        "(guarantees real completion tokens survive truncation, for BOTH "
        "chosen and rejected).",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=40000,
        help="max dataset rows to stream before giving up on filling a batch.",
    )
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="select data and fetch the tokenizer only; skip the ~14GB-per-model weights.",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"prefetch — sft={args.sft_model}  dpo={args.dpo_model}  dataset={args.dataset}")
    print(f"  HF_HOME={os.environ.get('HF_HOME', '(default)')}")
    print(
        f"  selecting {args.n_per_batch} code + {args.n_per_batch} english pairs, "
        f"seq_len={args.seq_len}, max_prompt_frac={args.max_prompt_frac}"
    )

    # Both checkpoints share the SFT tokenizer/chat template (DPO continues
    # training from the SFT checkpoint) — tokenization happens once, downstream.
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    max_prompt = int(args.max_prompt_frac * args.seq_len)

    ds = load_dataset(args.dataset, split=SPLIT, streaming=True)
    pairs: dict[str, list[dict]] = {"code": [], "english": []}
    selected_meta: dict[str, list[dict]] = {"code": [], "english": []}
    n_scanned = 0

    for ex in ds:
        n_scanned += 1
        if n_scanned > args.scan_limit:
            break
        kind = classify(ex.get("prompt_id", ""))
        if kind is None or len(pairs[kind]) >= args.n_per_batch:
            continue
        chosen, rejected = ex["chosen"], ex["rejected"]
        if not (_last_assistant_ok(chosen) and _last_assistant_ok(rejected)):
            continue
        # chosen/rejected share the same prompt prefix; require both completions
        # to leave real tokens after truncation to seq_len.
        try:
            prompt_len_w, total_len_w = sft_prompt_completion_lengths(chosen, tokenizer)
            prompt_len_l, total_len_l = sft_prompt_completion_lengths(rejected, tokenizer)
        except Exception:  # noqa: BLE001 — skip pairs the template can't render
            continue
        if (
            prompt_len_w > max_prompt
            or total_len_w <= prompt_len_w
            or prompt_len_l > max_prompt
            or total_len_l <= prompt_len_l
        ):
            continue

        pairs[kind].append(
            {
                "prompt_id": ex.get("prompt_id"),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        selected_meta[kind].append(
            {
                "prompt_id": ex.get("prompt_id"),
                "chosen_model": ex.get("chosen_model"),
                "rejected_model": ex.get("rejected_model"),
                "prompt_len_chosen": prompt_len_w,
                "total_len_chosen": total_len_w,
                "prompt_len_rejected": prompt_len_l,
                "total_len_rejected": total_len_l,
            }
        )
        if all(len(pairs[k]) >= args.n_per_batch for k in pairs):
            break

    for k in pairs:
        got = len(pairs[k])
        if got < args.n_per_batch:
            print(
                f"  WARNING: only found {got}/{args.n_per_batch} '{k}' pairs "
                f"in {n_scanned} rows. Raise --scan-limit or relax the filters."
            )
        else:
            print(f"  {k}: {got} pairs")

    payload = {
        "meta": {
            "sft_model": args.sft_model,
            "dpo_model": args.dpo_model,
            "dataset": args.dataset,
            "split": SPLIT,
            "seq_len_target": args.seq_len,
            "n_per_batch": args.n_per_batch,
            "max_prompt_frac": args.max_prompt_frac,
            "code_regex": CODE_RE.pattern,
            "english_regex": ENGLISH_RE.pattern,
            "n_rows_scanned": n_scanned,
            "selected": selected_meta,
        },
        "pairs": pairs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    if args.skip_model_download:
        print("skipped model weight download (--skip-model-download); tokenizer cached.")
        return

    from transformers import AutoModelForCausalLM

    for model_name in (args.sft_model, args.dpo_model):
        print(f"downloading model weights for {model_name} into the HF cache (~14GB) ...")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  cached {model_name}: {n_params:,} parameters.")
        del model
    print("both checkpoints cached. Ready for offline analysis.")


if __name__ == "__main__":
    main()
