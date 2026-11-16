"""Stage 1 (ONLINE, login node): select eval preference pairs + warm both checkpoints.

Implements the data side of the diagnostics in [`README.md`](README.md):
``R_dir`` and ``R_spec`` need, for each of a small set of (prompt, chosen,
rejected) triples, both a completion-only SFT-style CE loss on ``chosen`` and
a DPO loss on ``(chosen, rejected)`` — evaluated at two checkpoints,
``allenai/Olmo-3-7B-Instruct-SFT`` and ``allenai/Olmo-3-7B-Instruct-DPO`` (DPO
continues training from the SFT checkpoint, and is also the fixed reference
policy throughout DPO training).

By default this streams ``allenai/reward-bench`` — a HELD-OUT reward-model eval
set that Olmo-3's DPO stage did *not* train on, so both checkpoints see it as
unseen (unlike ``allenai/Dolci-Instruct-DPO``, which *is* the DPO training data;
still selectable via ``--dataset`` for the seen-vs-unseen ablation). It selects
``--n-per-batch`` *code* and ``--n-per-batch`` *english* preference pairs
(RewardBench: ``hep-*`` HumanEvalPack → code, ``alpacaeval``/``mt-bench`` chat →
english, via the ``subset`` column; Dolci: a regex over ``prompt_id``) whose
prompt is short enough to leave room for both completions after truncation to
``--seq-len``. RewardBench's single-turn string rows are wrapped into the
user/assistant message-list shape (see ``normalize_pair``) that the Olmo chat
template and downstream tokenizer expect. Selected ``chosen``/``rejected``
message lists are written to ``samples.json``; both checkpoints' weights are
warmed into the HF cache.

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
# Default to allenai/reward-bench: a HELD-OUT reward-model eval set (disjoint
# from Olmo-3's DPO training mix by construction), so BOTH checkpoints see it as
# unseen. allenai/Dolci-Instruct-DPO is *the* DPO training data (260k pairs, one
# `train` split, no holdout) — evaluating the DPO checkpoint on it confounds
# "objective geometry changed" with "model memorized these exact pairs". Still
# selectable via --dataset for the truncation/seen-vs-unseen ablation.
DATASET = "allenai/reward-bench"
OUT_PATH = _HERE.parent / "runs" / "default" / "samples.json"

# --- domain classification, per dataset ---------------------------------------
# Dolci has no domain column; `prompt_id` embeds the source-dataset slug
# (e.g. "correct-python-sft-187k-request-...", "valpy_if_...").
DOLCI_CODE_RE = re.compile(r"(correct-python-sft|codealpaca|persona.*python)", re.IGNORECASE)
DOLCI_ENGLISH_RE = re.compile(r"(valpy_if|flan_v2|coconot)", re.IGNORECASE)
# RewardBench names its source in the `subset` column: HumanEvalPack (hep-*) is
# code (all six languages — correct vs subtly-buggy solutions); the english side
# takes every chat-domain subset (AlpacaEval easy/length/hard, MT-Bench, LLMBar
# natural + adversarial) to maximize pair count. Per-pair provenance is kept in
# prompt_id ("subset::id"), so downstream analysis can slice by subset.
RB_CODE_RE = re.compile(r"^hep-", re.IGNORECASE)
RB_ENGLISH_RE = re.compile(r"^(alpacaeval|mt-bench|llmbar)", re.IGNORECASE)


def _is_reward_bench(dataset: str) -> bool:
    return "reward-bench" in dataset


def default_split(dataset: str) -> str:
    # RewardBench ships "raw" and the leaderboard-curated "filtered"; use filtered.
    return "filtered" if _is_reward_bench(dataset) else "train"


def code_english_res(dataset: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    if _is_reward_bench(dataset):
        return RB_CODE_RE, RB_ENGLISH_RE
    return DOLCI_CODE_RE, DOLCI_ENGLISH_RE


def classify(ex: dict, dataset: str) -> str | None:
    code_re, english_re = code_english_res(dataset)
    key = (ex.get("subset") if _is_reward_bench(dataset) else ex.get("prompt_id")) or ""
    if code_re.search(key):
        return "code"
    if english_re.search(key):
        return "english"
    return None


def _last_assistant_ok(messages: list[dict]) -> bool:
    return bool(messages) and messages[-1].get("role") == "assistant"


def normalize_pair(ex: dict, dataset: str) -> tuple[str, list, list] | None:
    """Return ``(prompt_id, chosen_messages, rejected_messages)`` or None to skip.

    RewardBench rows are single-turn *plain strings* (``prompt``/``chosen``/
    ``rejected``); wrap them into the user/assistant message-list shape the rest
    of the pipeline (and the Olmo chat template) expects — matching Dolci's
    native format, which is already message lists.
    """
    if _is_reward_bench(dataset):
        prompt, chosen, rejected = ex.get("prompt"), ex.get("chosen"), ex.get("rejected")
        if not all(isinstance(s, str) and s for s in (prompt, chosen, rejected)):
            return None
        pid = f"{ex.get('subset', '?')}::{ex.get('id', '?')}"
        chosen_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        rejected_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]
        return pid, chosen_msgs, rejected_msgs
    chosen, rejected = ex.get("chosen"), ex.get("rejected")
    if not (_last_assistant_ok(chosen) and _last_assistant_ok(rejected)):
        return None
    return ex.get("prompt_id", "") or "", chosen, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", default=SFT_MODEL)
    parser.add_argument("--dpo-model", default=DPO_MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--split",
        default=None,
        help="dataset split (default: 'filtered' for reward-bench, else 'train').",
    )
    parser.add_argument("--n-per-batch", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=2048)
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

    split = args.split or default_split(args.dataset)
    ds = load_dataset(args.dataset, split=split, streaming=True)
    pairs: dict[str, list[dict]] = {"code": [], "english": []}
    selected_meta: dict[str, list[dict]] = {"code": [], "english": []}
    n_scanned = 0

    for ex in ds:
        n_scanned += 1
        if n_scanned > args.scan_limit:
            break
        kind = classify(ex, args.dataset)
        if kind is None or len(pairs[kind]) >= args.n_per_batch:
            continue
        normalized = normalize_pair(ex, args.dataset)
        if normalized is None:
            continue
        prompt_id, chosen, rejected = normalized
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
                "prompt_id": prompt_id,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        selected_meta[kind].append(
            {
                "prompt_id": prompt_id,
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
            "split": split,
            "seq_len_target": args.seq_len,
            "n_per_batch": args.n_per_batch,
            "max_prompt_frac": args.max_prompt_frac,
            "code_regex": code_english_res(args.dataset)[0].pattern,
            "english_regex": code_english_res(args.dataset)[1].pattern,
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
