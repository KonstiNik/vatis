"""Quickstart: the smallest complete use of vatis.

Loads EleutherAI/pythia-14m at its last training checkpoint, measures every
observable on two short real-text batches and on the pair of them, and prints
the result rows as a table. Runs on the CPU; see README.md next to this file.

    python examples/quickstart/run.py
"""

import torch
from transformers import AutoTokenizer

from vatis import analyze

MODEL, REVISION = "EleutherAI/pythia-14m", "step143000"
BATCH_SIZE, SEQ_LEN = 2, 32  # each batch: 2 sequences of 32 tokens
TEXTS = {
    "prose": "It is a truth universally acknowledged, that a single man in possession of a "
    "good fortune, must be in want of a wife. However little known the feelings or views of "
    "such a man may be on his first entering a neighbourhood, this truth is so well fixed in "
    "the minds of the surrounding families, that he is considered as the rightful property "
    "of some one or other of their daughters.",
    "code": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n"
    "        a, b = b, a + b\n\n\ndef is_prime(n):\n    if n < 2:\n        return False\n"
    "    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n"
    "            return False\n    return True\n",
}

# An eval batch is a dict of (batch, sequence) tensors. vatis expects the labels
# already shifted: labels[:, i] is the token at position i + 1; -100 = not scored.
tokenizer = AutoTokenizer.from_pretrained(MODEL)
eval_batches = {}
for name, text in TEXTS.items():
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0][: BATCH_SIZE * SEQ_LEN]
    ids = ids.reshape(BATCH_SIZE, SEQ_LEN)  # fails if the text is shorter than that
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    eval_batches[name] = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
    }

results = analyze(
    model=MODEL,
    revisions=[REVISION],
    eval_batches=eval_batches,
    cross_pairs=[("prose", "code")],  # also measure the pair: does a step on prose help code?
    n_hutchinson=8,  # random probes for the chi_net estimate: more = less noise, slower
    seed=0,
    device="cpu",
    dtype="fp32",
)

print(f"\n{'observable':<20} {'batch_a':<7} {'batch_b':<7} {'value':>11}")
for row in results[0].rows:
    print(f"{row.observable:<20} {row.batch_a:<7} {row.batch_b:<7} {row.value:>11.4g}")
