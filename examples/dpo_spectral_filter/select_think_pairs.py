"""Stage 1 for the Think SFT-trajectory sweep: pick a fixed, stratified pair set.

Selects preference pairs from ``allenai/Dolci-Think-DPO-7B`` (the delta-learning
pairs Olmo-3-7B-Think-DPO was trained on: chosen = Qwen3-32B-reasoning,
rejected = Qwen3-0.6B-reasoning) for evaluating ``R_dir`` / ``R_spec`` at every
``allenai/Olmo-3-7B-Think-SFT`` checkpoint. The same pairs are reused at every
checkpoint, so differences along the trajectory come from the weights alone.

Two steps, both CPU-only (run on a debug/interactive node, not a login node):

1. **Index** — render every pair through the Think-SFT chat template and record
   prompt / chosen / rejected token lengths (``index.parquet``, cached; reused
   on re-runs). Pairs whose raw text is obviously too long are skipped cheaply
   before rendering.
2. **Sample** — keep pairs where *both* full sequences fit in ``--seq-len``
   (no truncation, so the chosen/rejected difference is never cut off), then
   draw ``--n-pairs`` stratified by ``dataset_source`` (proportional,
   largest-remainder), with a fixed seed, in shuffled order (the tiered
   Hutchinson boost in ``diagnostics.py`` assumes the leading pairs are a
   random subset).

Writes ``runs/{tag}/samples.json`` in the format ``diagnostics.py`` reads, with
one group per domain (``--stratify domain``, the default) or a single group
``"all"`` (``--stratify source``), and per-pair ``dataset_source`` / ``domain``
tags for offline slicing.

The dataset parquet files must already be in the HF cache (or the node needs
internet to fetch them, ~1.4 GB)::

    .venv/bin/python examples/dpo_spectral_filter/select_think_pairs.py \\
        --tag think_smoke --n-pairs 40
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_EXAMPLES_DIR = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))  # run_config
sys.path.insert(0, str(_EXAMPLES_DIR))  # _helpers

from run_config import configure_hf_cache  # noqa: E402

configure_hf_cache()

from _helpers import sft_prompt_completion_lengths  # noqa: E402

DATASET = "allenai/Dolci-Think-DPO-7B"
SFT_MODEL = "allenai/Olmo-3-7B-Think-SFT"
DPO_MODEL = "allenai/Olmo-3-7B-Think-DPO"
_RUNS_DIR = _HERE.parent / "runs"

# Coarse domain per dataset_source, for slicing results offline.
DOMAIN_OF_SOURCE = {
    "correct-python-sft-187k": "code",
    "evol_codealpaca_heval_decontaminated": "code",
    "personahub_code_v2_34999": "code",
    "tulu-3-sft-personas-math": "math",
    "tulu-3-sft-personas-math-grade": "math",
    "tulu-3-sft-personas-algebra": "math",
    "tulu_v3.9_open_math_2_gsm8k_50k": "math",
    "OpenThoughts3-full-filtered-science-no-cot": "science",
    "tulu_v3.9_sciriff_10k": "science",
    "tulu_v3.9_wildjailbreak_decontaminated_50k": "safety",
    "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k": "safety",
    "tulu-3-sft-coconot-regenerated": "safety",
    "valpy_if_qwq_reasoning_verified_no_reasoning": "if",
    "IF_sft_data_verified_permissive": "if",
    "tulu-3-sft-personas-instruction-following-o3": "if",
}
# ~3 chars/token is a safe lower bound for these texts, so anything longer
# than this many characters cannot fit and is skipped before rendering.
CHARS_PER_TOKEN_FLOOR = 3.0

_tokenizer = None


def _init_worker(model_name: str) -> None:
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_name)


def _lengths(row: tuple[str, list, list]) -> tuple[str, int, int, int]:
    """(id, prompt_len, total_len_chosen, total_len_rejected); -1 if unrenderable."""
    pid, chosen, rejected = row
    try:
        p_w, t_w = sft_prompt_completion_lengths(chosen, _tokenizer)
        _, t_l = sft_prompt_completion_lengths(rejected, _tokenizer)
    except Exception:  # noqa: BLE001 — skip pairs the template can't render
        return pid, -1, -1, -1
    return pid, p_w, t_w, t_l


def _dataset_files() -> list[str]:
    from huggingface_hub import snapshot_download

    path = snapshot_download(DATASET, repo_type="dataset")
    return sorted(glob.glob(os.path.join(path, "data", "*.parquet")))


def build_index(files: list[str], max_chars: int, workers: int) -> pa.Table:
    ids, sources, p_lens, w_lens, l_lens = [], [], [], [], []
    with Pool(workers, initializer=_init_worker, initargs=(SFT_MODEL,)) as pool:
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["id", "chosen", "rejected", "dataset_source"])
                rows = []
                for r in t.to_pylist():
                    n_chars = max(
                        sum(len(m["content"]) for m in r["chosen"]),
                        sum(len(m["content"]) for m in r["rejected"]),
                    )
                    if n_chars <= max_chars:
                        rows.append((r["id"], r["chosen"], r["rejected"]))
                    else:  # too long to ever fit: record as such without rendering
                        ids.append(r["id"])
                        sources.append(r["dataset_source"])
                        p_lens.append(-1)
                        w_lens.append(-1)
                        l_lens.append(-1)
                src = dict(
                    zip(
                        t.column("id").to_pylist(),
                        t.column("dataset_source").to_pylist(),
                        strict=True,
                    )
                )
                for pid, p_len, w_len, l_len in pool.imap(_lengths, rows, chunksize=64):
                    ids.append(pid)
                    sources.append(src[pid])
                    p_lens.append(p_len)
                    w_lens.append(w_len)
                    l_lens.append(l_len)
                del t, rows
            print(f"  indexed {os.path.basename(f)}: {len(ids)} rows so far", flush=True)
    return pa.table(
        {
            "id": ids,
            "dataset_source": sources,
            "prompt_len": p_lens,
            "total_len_chosen": w_lens,
            "total_len_rejected": l_lens,
        }
    )


def stratified_sample(
    sources: np.ndarray, n: int, rng: np.random.Generator, *, balanced: bool = False
) -> np.ndarray:
    """Indices of ``n`` items drawn per stratum (largest remainder).

    ``balanced=False``: quotas proportional to stratum size. ``balanced=True``:
    equal quota per stratum (uniform within each stratum); fails if a stratum
    has fewer items than its quota.
    """
    uniq, counts = np.unique(sources, return_counts=True)
    exact = (np.ones_like(counts) / len(counts) if balanced else counts / counts.sum()) * n
    quota = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - quota), kind="stable")[: n - quota.sum()]:
        quota[i] += 1
    short = [(s, int(q), int(c)) for s, q, c in zip(uniq, quota, counts, strict=True) if q > c]
    if short:
        raise SystemExit(f"strata smaller than their quota (name, quota, available): {short}")
    picked = [
        rng.choice(np.flatnonzero(sources == s), size=q, replace=False)
        for s, q in zip(uniq, quota, strict=True)
        if q > 0
    ]
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def fetch_rows(files: list[str], wanted: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for f in files:
        t = pq.read_table(f, columns=["id"])
        mask = np.isin(np.asarray(t.column("id").to_pylist()), list(wanted))
        if not mask.any():
            continue
        full = pq.read_table(f, columns=["id", "chosen", "rejected", "dataset_source"])
        for r in full.filter(pa.array(mask)).to_pylist():
            rows[r["id"]] = r
        del full
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="output goes to runs/{tag}/samples.json")
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stratify",
        choices=["domain", "source"],
        default="domain",
        help="domain: equal pairs per domain, uniform within it (default); "
        "source: proportional to dataset_source sizes.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--index",
        default=str(_RUNS_DIR / "think_index" / "index.parquet"),
        help="token-length index cache (built on first use, shared across tags).",
    )
    args = parser.parse_args()

    files = _dataset_files()
    index_path = Path(args.index)
    if index_path.exists():
        index = pq.read_table(index_path)
        print(f"loaded index {index_path} ({index.num_rows} rows)")
    else:
        # The index is built for the largest seq_len we'd plausibly use, so
        # one index serves every --seq-len up to it.
        max_chars = int(8192 * CHARS_PER_TOKEN_FLOOR)
        print(f"building index over {len(files)} files with {args.workers} workers ...")
        index = build_index(files, max_chars, args.workers)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(index, index_path)
        print(f"wrote {index_path}")

    df = index.to_pandas()
    fits = (
        (df.prompt_len > 0)
        & (df.total_len_chosen > df.prompt_len)
        & (df.total_len_rejected > df.prompt_len)
        & (df.total_len_chosen <= args.seq_len)
        & (df.total_len_rejected <= args.seq_len)
    )
    pool = df[fits].reset_index(drop=True)
    print(f"{len(pool)} / {len(df)} pairs fit in seq_len={args.seq_len}")
    if len(pool) < args.n_pairs:
        raise SystemExit(f"only {len(pool)} pairs fit; lower --n-pairs or raise --seq-len")

    rng = np.random.default_rng(args.seed)
    pool["domain"] = pool.dataset_source.map(lambda s: DOMAIN_OF_SOURCE.get(s, "chat"))
    # domain: equal pairs per domain, uniform within each domain
    strata = (pool.domain if args.stratify == "domain" else pool.dataset_source).to_numpy()
    chosen_idx = stratified_sample(strata, args.n_pairs, rng, balanced=args.stratify == "domain")
    sel = pool.iloc[chosen_idx]
    rows = fetch_rows(files, set(sel.id))

    # --stratify domain: one group per domain (diagnostics.py computes every
    # observable per group, plus a pooled view); --stratify source: one group.
    pairs: dict[str, list] = {}
    selected: dict[str, list] = {}
    for r in sel.itertuples():
        row = rows[r.id]
        domain = DOMAIN_OF_SOURCE.get(r.dataset_source, "chat")
        group = domain if args.stratify == "domain" else "all"
        pairs.setdefault(group, []).append(
            {
                "prompt_id": r.id,
                "dataset_source": r.dataset_source,
                "domain": domain,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )
        selected.setdefault(group, []).append(
            {
                "prompt_id": r.id,
                "dataset_source": r.dataset_source,
                "domain": domain,
                "prompt_len": int(r.prompt_len),
                "total_len_chosen": int(r.total_len_chosen),
                "total_len_rejected": int(r.total_len_rejected),
            }
        )

    by_source = sel.dataset_source.value_counts().to_dict()
    by_domain = sel.domain.value_counts().to_dict()
    pool_by_domain = pool.domain.value_counts().to_dict()
    payload = {
        "meta": {
            "sft_model": SFT_MODEL,
            "dpo_model": DPO_MODEL,
            "dataset": DATASET,
            "split": "train",
            "seq_len_target": args.seq_len,
            "n_pairs": args.n_pairs,
            "seed": args.seed,
            "n_pool": len(pool),
            "stratify": args.stratify,
            "n_per_source": {k: int(v) for k, v in by_source.items()},
            "n_per_domain": {k: int(v) for k, v in by_domain.items()},
            # pool sizes per domain, for reweighting pooled numbers offline
            "n_pool_per_domain": {k: int(v) for k, v in pool_by_domain.items()},
            "selected": {g: selected[g] for g in sorted(selected)},
        },
        "pairs": {g: pairs[g] for g in sorted(pairs)},
    }
    out = _RUNS_DIR / args.tag / "samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    for k, v in sorted(by_domain.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  [{k}]  (pool {pool_by_domain[k]})")
    for k, v in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
