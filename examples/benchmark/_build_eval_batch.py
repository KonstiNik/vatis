"""Build a fixed real-text eval batch and save it to disk (run on login node).

The multi-GPU scaling job runs offline on a compute node, so we tokenize here
(where the tokenizer can be fetched) and torch.save the batch. Using one
pre-built batch keeps content identical across every (world_size, batch_size)
config — the scaling numbers then reflect parallelism only.

Real on-manifold text (Pride and Prejudice) rather than random ids: the batch
is short enough that sequences repeat to fill B*S tokens, which is fine for a
timing/scaling benchmark (content doesn't change wallclock) while keeping the
reported observables on the model's training manifold.

    .venv/bin/python examples/benchmark/_build_eval_batch.py

Set ``HF_HOME`` (in your shell or the repo-root ``.env``) to relocate the
model cache; see ``run_config.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_config import configure_hf_cache  # noqa: E402

configure_hf_cache()

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def tokenize_into_lm_batch(text, tokenizer, *, batch_size, seq_len):
    """Tokenize ``text`` into a (batch_size, seq_len) causal-LM batch.

    Inlined copy of ``examples._helpers.tokenize_into_lm_batch`` so this
    login-node prep script has no intra-repo import dependency.
    """
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    needed = batch_size * seq_len
    while ids.shape[0] < needed:
        ids = torch.cat([ids, ids], dim=0)
    ids = ids[:needed].reshape(batch_size, seq_len).contiguous()
    attention_mask = torch.ones_like(ids)
    labels = torch.full_like(ids, fill_value=-100)
    labels[:, :-1] = ids[:, 1:]
    return {"input_ids": ids, "attention_mask": attention_mask, "labels": labels}

PROSE = (
    "It is a truth universally acknowledged, that a single man in possession of "
    "a good fortune, must be in want of a wife. However little known the feelings "
    "or views of such a man may be on his first entering a neighbourhood, this "
    "truth is so well fixed in the minds of the surrounding families, that he is "
    "considered the rightful property of some one or other of their daughters. "
    "My dear Mr. Bennet, said his lady to him one day, have you heard that "
    "Netherfield Park is let at last? Mr. Bennet replied that he had not. But it "
    "is, returned she; for Mrs. Long has just been here, and she told me all about "
    "it. Mr. Bennet made no answer. Do you not want to know who has taken it? "
    "cried his wife impatiently. You want to tell me, and I have no objection to "
    "hearing it. This was invitation enough."
)

MODEL = "EleutherAI/pythia-1.4b"
BATCH_SIZE = 64
SEQ_LEN = 512


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tokenize_into_lm_batch(PROSE, tok, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    out = Path(__file__).resolve().parent / f"eval_batch_b{BATCH_SIZE}_s{SEQ_LEN}.pt"
    torch.save(batch, out)
    print(f"saved {out}")
    print({k: tuple(v.shape) for k, v in batch.items()})


if __name__ == "__main__":
    main()
