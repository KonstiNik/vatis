"""Spectral tail overlap experiment: does ``chi_pos`` show a transient bump when
training reaches what two texts share?

A research note with a negative result; the question, the findings and the
probe design are in ``README.md`` and ``details.md`` next to this script.

Hypothesis: two texts that implement the same computation (matrix operations)
with almost no surface overlap should have nearly orthogonal gradients early
in training and develop shared structure later, when the model resolves what
the code *does*. That would show up as a **transient bump in the cross
chi_pos** at the checkpoint where training reaches the spectral band where the
two texts overlap.

The four probes (eval batches, one sequence of 1024 tokens each, B = 1, S =
1024), in ``probe_texts.py``:

- ``python``  — Python implementing matrix multiply, transpose, dot product,
  trace and squared Frobenius norm, with verbose names, docstrings, type hints;
- ``cpp``     — C++ implementing the same operations in the same order;
- ``prose``   — English prose describing the same operations (control);
- ``cpp_str`` — an unrelated C++ string-processing program (negative control).

Cross pairs: ``python × cpp`` (the signal), ``python × prose``,
``cpp × cpp_str`` and ``python × cpp_str``. Each model is measured at the same
13 Pythia checkpoints as the Pythia sweep, and its rows go to
``results_<model>.parquet`` next to this script; ``evaluate.py`` draws the
figures into ``figures/``.

Run with::

    python examples/spectral_tail/run.py                                 # pythia-14m
    python examples/spectral_tail/run.py --model EleutherAI/pythia-70m   # one other size
    python examples/spectral_tail/run.py --all                           # all five sizes in turn

``--all`` skips a model whose result file already exists. Set ``HF_HOME`` (in
your shell or the repo-root ``.env``) to relocate the model cache; see
``run_config.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for _helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for probe_texts
from run_config import configure_hf_cache  # noqa: E402

configure_hf_cache()

import argparse  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from _helpers import tokenize_into_lm_batch  # noqa: E402
from probe_texts import STYLE_A_TEXT, STYLE_B_TEXT, STYLE_C_TEXT, STYLE_D_TEXT  # noqa: E402
from transformers import AutoTokenizer

from vatis import analyze

# ---------------------------------------------------------------- config

DEFAULT_MODEL = "EleutherAI/pythia-14m"

MODELS_FOR_COMPARISON = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
]

REVISIONS: list[str] = [
    "step1",
    "step8",
    "step64",
    "step512",
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

BATCH_SIZE = 1
SEQ_LEN = 1024
N_HUTCHINSON = 32
SEED = 0

STYLE_A_NAME = "python"
STYLE_B_NAME = "cpp"
STYLE_C_NAME = "prose"
STYLE_D_NAME = "cpp_str"
CROSS_PAIRS: list[tuple[str, str]] = [
    (STYLE_A_NAME, STYLE_B_NAME),
    (STYLE_A_NAME, STYLE_C_NAME),
    (STYLE_B_NAME, STYLE_D_NAME),
    (STYLE_A_NAME, STYLE_D_NAME),
]

OUT_DIR = Path(__file__).parent


def model_tag(model_name: str) -> str:
    """Extract a short tag from a HF model name, e.g. 'pythia-14m'."""
    return model_name.split("/")[-1]


def parquet_path(model_name: str) -> Path:
    return OUT_DIR / f"results_{model_tag(model_name)}.parquet"


# The probe texts STYLE_A_TEXT ... STYLE_D_TEXT are in probe_texts.py.

# ---------------------------------------------------------------- main


def pick_chi_net_method(model_name: str) -> str:
    """Choose chi_net method. per_sequence_cv caches a full gradient vector
    per sequence (~n_params * 4 bytes) which OOMs on larger models.
    Fall back to hutchinson for 410m+."""
    if "410m" in model_name or "1b" in model_name:
        return "hutchinson"
    return "per_sequence_cv"


def run_single_model(model_name: str) -> None:
    """Run the spectral tail experiment for one model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pq_path = parquet_path(model_name)
    chi_net_method = pick_chi_net_method(model_name)

    print(f"\nspectral tail experiment -- model={model_name}")
    print(f"  device={device}, chi_net_method={chi_net_method}")
    print(f"  revisions={REVISIONS}")
    print(f"  B={BATCH_SIZE}, S={SEQ_LEN}, n_hutchinson={N_HUTCHINSON}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if pq_path.exists():
        pq_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_batches = {
        STYLE_A_NAME: tokenize_into_lm_batch(
            STYLE_A_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
        STYLE_B_NAME: tokenize_into_lm_batch(
            STYLE_B_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
        STYLE_C_NAME: tokenize_into_lm_batch(
            STYLE_C_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
        STYLE_D_NAME: tokenize_into_lm_batch(
            STYLE_D_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
    }
    for name, batch in eval_batches.items():
        n_tokens = int(batch["input_ids"].numel())
        n_valid = int((batch["labels"] != -100).sum().item())
        print(f"  {name}: {n_tokens} tokens, {n_valid} valid ({n_valid / n_tokens:.2%})")

    t_total = time.perf_counter()
    analyze(
        model=model_name,
        revisions=REVISIONS,
        eval_batches=eval_batches,
        cross_pairs=CROSS_PAIRS,
        chi_net_method=chi_net_method,
        n_hutchinson=N_HUTCHINSON,
        micro_batch_size=BATCH_SIZE,
        seed=SEED,
        device=device,
        dtype="fp32",
        sink=str(pq_path),
    )
    total_s = time.perf_counter() - t_total

    n_self = len(eval_batches)
    n_cross = len(CROSS_PAIRS)
    n_calls = len(REVISIONS) * (n_self + n_cross)
    print(
        f"\nDone in {total_s:.1f}s. "
        f"{len(REVISIONS)} checkpoints x "
        f"({n_self} self pairs + {n_cross} cross pairs) = "
        f"{n_calls} measurements. Wrote {pq_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral tail overlap experiment")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Run all models ({', '.join(MODELS_FOR_COMPARISON)}) sequentially.",
    )
    args = parser.parse_args()

    if args.all:
        for model_name in MODELS_FOR_COMPARISON:
            pq_path = parquet_path(model_name)
            if pq_path.exists():
                print(f"results for {model_name} already exist at {pq_path}, skipping compute")
            else:
                run_single_model(model_name)
    else:
        run_single_model(args.model)


if __name__ == "__main__":
    main()
