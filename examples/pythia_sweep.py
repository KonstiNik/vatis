"""vatis deployment example: LNA observables across pythia-14m revisions.

Single-GPU end-to-end example. Loads ``EleutherAI/pythia-14m`` at nine
training checkpoints, evaluates two distinct fixed eval batches at each
checkpoint, computes both self-pair and cross-pair LNA observables,
writes the result to a long-format parquet file, and produces four
plots showing each observable as a function of training step.

The two eval batches are real text — one English prose passage (the
opening of *Pride and Prejudice*) and one Python source file — both
tokenized at runtime with the model's own tokenizer. Real text is
essential for the observables to be theory-relevant: the chi_net term
is the squared Frobenius norm of the model's parameter Jacobian
evaluated **at the input you feed it**, so feeding random tokens
puts you at an off-distribution point in input-space and the resulting
trajectory says nothing about Pythia's training dynamics. Tokenized
prose puts the evaluation back on-distribution.

The cross-pair observables ``delta_loss(prose, code)`` and
``chi_pos(prose, code)`` are the most theory-relevant outputs of the
example: ``delta_loss(A, B) = <∇_θ L^A, ∇_θ L^B>`` measures whether a
gradient step on prose helps or hurts the model on code (positive →
transfer, negative → interference, zero → orthogonal). Watching this
across checkpoints answers a real LNA question: do the prose and code
loss directions become more or less aligned as Pythia trains?

Sized to fit comfortably under the 15-minute budget on the reference
hardware (RTX 3090 Ti, see ``examples/BENCHMARK.md`` for the budget
arithmetic). On that hardware the script wallclock is ~15 s
including model loads. Slower hardware or a cold HF cache will take
longer; the rate-limiting step is the per-checkpoint download/load,
not the chi_net computation.

Run with::

    HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/pythia_sweep.py

Outputs are written next to this script:

- ``examples/results.parquet``  — canonical long-format result table
- ``examples/chi_loss.png``      — chi_loss_normalized vs step (self pairs only)
- ``examples/chi_net.png``       — chi_net_normalized vs step (self pairs only)
- ``examples/delta_loss.png``    — delta_loss vs step (self + cross)
- ``examples/chi_pos.png``       — chi_pos vs step (self + cross)
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
from transformers import AutoTokenizer

from vatis import analyze

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

# Two distinct, real-text eval batches. The names are also the keys in
# the cross-pair list below; keep them in sync.
PROSE_NAME = "prose"
CODE_NAME = "code"
EVAL_BATCH_NAMES = (PROSE_NAME, CODE_NAME)

# The cross pair we want δL and chi_pos for. Order matters only for the
# row label; ⟨g_A, g_B⟩ is symmetric.
CROSS_PAIRS: list[tuple[str, str]] = [(PROSE_NAME, CODE_NAME)]

OUT_DIR = Path(__file__).parent
PARQUET_PATH = OUT_DIR / "results.parquet"
PLOTS = {
    "chi_loss_normalized": OUT_DIR / "chi_loss.png",
    "chi_net_normalized": OUT_DIR / "chi_net.png",
    "delta_loss": OUT_DIR / "delta_loss.png",
    "chi_pos": OUT_DIR / "chi_pos.png",
}

# ---------------------------------------------------------------- data
#
# Two pieces of real text, embedded in the script so the example has no
# data download dependency. Both are clearly representative of distinct
# distributions Pythia was trained on (The Pile contains both English
# prose and Python source code from GitHub).

# Public-domain prose: opening of Jane Austen's *Pride and Prejudice*.
PROSE_TEXT = """
It is a truth universally acknowledged, that a single man in possession
of a good fortune, must be in want of a wife. However little known the
feelings or views of such a man may be on his first entering a
neighbourhood, this truth is so well fixed in the minds of the
surrounding families, that he is considered as the rightful property of
some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard
that Netherfield Park is let at last?" Mr. Bennet replied that he had
not. "But it is," returned she; "for Mrs. Long has just been here, and
she told me all about it." Mr. Bennet made no answer. "Do you not want
to know who has taken it?" cried his wife impatiently. "You want to
tell me, and I have no objection to hearing it." This was invitation
enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken
by a young man of large fortune from the north of England; that he came
down on Monday in a chaise and four to see the place, and was so much
delighted with it that he agreed with Mr. Morris immediately; that he
is to take possession before Michaelmas, and some of his servants are
to be in the house by the end of next week."

"What is his name?" "Bingley." "Is he married or single?" "Oh, single,
my dear, to be sure! A single man of large fortune; four or five
thousand a year. What a fine thing for our girls!" "How so? How can it
affect them?" "My dear Mr. Bennet," replied his wife, "how can you be
so tiresome! You must know that I am thinking of his marrying one of
them." "Is that his design in settling here?" "Design! Nonsense, how
can you talk so! But it is very likely that he may fall in love with
one of them, and therefore you must visit him as soon as he comes."

"I see no occasion for that. You and the girls may go, or you may send
them by themselves, which perhaps will be still better; for, as you are
as handsome as any of them, Mr. Bingley might like you the best of the
party." "My dear, you flatter me. I certainly have had my share of
beauty, but I do not pretend to be anything extraordinary now. When a
woman has five grown-up daughters, she ought to give over thinking of
her own beauty." "In such cases, a woman has not often much beauty to
think of." "But, my dear, you must indeed go and see Mr. Bingley when
he comes into the neighbourhood."

"It is more than I engage for, I assure you." "But consider your
daughters. Only think what an establishment it would be for one of
them. Sir William and Lady Lucas are determined to go, merely on that
account, for in general, you know, they visit no newcomers. Indeed you
must go, for it will be impossible for us to visit him if you do not."
"You are over-scrupulous, surely. I dare say Mr. Bingley will be very
glad to see you; and I will send a few lines by you to assure him of
my hearty consent to his marrying whichever he chuses of the girls;
though I must throw in a good word for my little Lizzy."

"I desire you will do no such thing. Lizzy is not a bit better than the
others; and I am sure she is not half so handsome as Jane, nor half so
good-humoured as Lydia. But you are always giving her the preference."
"They have none of them much to recommend them," replied he; "they are
all silly and ignorant like other girls; but Lizzy has something more
of quickness than her sisters." "Mr. Bennet, how can you abuse your own
children in such a way? You take delight in vexing me. You have no
compassion for my poor nerves." "You mistake me, my dear. I have a
high respect for your nerves. They are my old friends. I have heard
you mention them with consideration these last twenty years at least."
"""

# Self-contained Python module: linear algebra primitives plus a few
# classic algorithms. Long enough that the BPE tokenizer produces well
# over `BATCH_SIZE * SEQ_LEN` tokens.
CODE_TEXT = '''
"""Small linear algebra and algorithm utilities for the vatis example."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Vector:
    """An immutable 3D vector with the usual arithmetic operations."""

    x: float
    y: float
    z: float

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vector":
        n = self.norm()
        if n == 0.0:
            raise ValueError("zero vector cannot be normalized")
        return Vector(self.x / n, self.y / n, self.z / n)


def cross(a: Vector, b: Vector) -> Vector:
    """The right-handed cross product of two 3-vectors."""
    return Vector(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def angle_between(a: Vector, b: Vector) -> float:
    """Return the angle between two non-zero vectors, in radians."""
    cos_theta = a.dot(b) / (a.norm() * b.norm())
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def quicksort(items: list[int]) -> list[int]:
    """Pure-Python quicksort. Stable on equal elements via the middle bucket."""
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    left = [x for x in items if x < pivot]
    middle = [x for x in items if x == pivot]
    right = [x for x in items if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def binary_search(sorted_items: list[int], target: int) -> int:
    """Return the index of `target` in `sorted_items`, or -1 if not present."""
    lo, hi = 0, len(sorted_items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        value = sorted_items[mid]
        if value == target:
            return mid
        if value < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def fibonacci(n: int) -> Iterator[int]:
    """Yield the first `n` Fibonacci numbers, starting at 0."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


def is_prime(n: int) -> bool:
    """Return True iff `n` is a prime number. Trial division up to sqrt(n)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.isqrt(n))
    for d in range(3, limit + 1, 2):
        if n % d == 0:
            return False
    return True


def primes_below(limit: int) -> list[int]:
    """Sieve of Eratosthenes returning all primes strictly below `limit`."""
    if limit <= 2:
        return []
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit, i):
                sieve[j] = False
    return [i for i, prime in enumerate(sieve) if prime]


if __name__ == "__main__":
    v1 = Vector(1.0, 2.0, 3.0)
    v2 = Vector(4.0, 5.0, 6.0)
    print(f"sum:    {v1 + v2}")
    print(f"dot:    {v1.dot(v2)}")
    print(f"cross:  {cross(v1, v2)}")
    print(f"angle:  {angle_between(v1, v2):.4f} radians")
    print(f"sorted: {quicksort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])}")
    print(f"fib:    {list(fibonacci(10))}")
    print(f"primes: {primes_below(50)}")
'''

# ---------------------------------------------------------------- helpers


def tokenize_into_lm_batch(
    text: str,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Tokenize ``text`` and reshape into a ``(batch_size, seq_len)`` LM batch.

    The text is tokenized with no special tokens, then truncated or
    cyclically repeated to produce exactly ``batch_size * seq_len``
    tokens, which are reshaped row-major into the batch. Labels are the
    HF causal-LM convention: ``labels[:, :-1] = input_ids[:, 1:]``,
    ``labels[:, -1] = -100`` so the last position of every sequence is
    ignored (no next-token target available there).
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]  # (T,)
    needed = batch_size * seq_len

    # If the text is shorter than needed, repeat it. We never expect
    # this to fire in this example (the embedded texts are sized to
    # comfortably exceed `needed` for both prose and code), but the
    # fallback keeps the helper robust to future text edits.
    while ids.shape[0] < needed:
        ids = torch.cat([ids, ids], dim=0)

    ids = ids[:needed].reshape(batch_size, seq_len).contiguous()
    attention_mask = torch.ones_like(ids)
    labels = torch.full_like(ids, fill_value=-100)
    labels[:, :-1] = ids[:, 1:]
    return {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def step_from_revision(rev: str) -> int:
    """Parse the trailing integer step out of a Pythia revision tag."""
    return int(rev.removeprefix("step"))


def cross_label(name_a: str, name_b: str) -> str:
    """Render a cross-pair label for plot legends."""
    return f"{name_a} × {name_b}"


# ---------------------------------------------------------------- main


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"vatis deployment example — model={MODEL}")
    print(f"  device={device}")
    print(f"  revisions={REVISIONS}")
    print(f"  B={BATCH_SIZE}, S={SEQ_LEN}, n_hutchinson={N_HUTCHINSON}")
    print(f"  eval_batches={EVAL_BATCH_NAMES}")
    print(f"  cross_pairs={CROSS_PAIRS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PARQUET_PATH.exists():
        PARQUET_PATH.unlink()  # fresh write — don't append to a stale file

    # ---- tokenize the two real-text eval batches with the model's own
    # tokenizer. We do this once, before any checkpoint is loaded, so
    # that every revision sees byte-identical input.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    eval_batches = {
        PROSE_NAME: tokenize_into_lm_batch(
            PROSE_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
        CODE_NAME: tokenize_into_lm_batch(
            CODE_TEXT, tokenizer, batch_size=BATCH_SIZE, seq_len=SEQ_LEN
        ),
    }
    for name, batch in eval_batches.items():
        n_tokens = int(batch["input_ids"].numel())
        n_valid = int((batch["labels"] != -100).sum().item())
        print(
            f"  {name}: {n_tokens} tokens, {n_valid} valid (n_valid/total = {n_valid / n_tokens:.2%})"
        )

    # ---- single sweep call: analyze() iterates over revisions and writes
    # one row per (checkpoint, batch_a, batch_b, observable) into the
    # same parquet sink. Self pairs and the prose×code cross pair are
    # both emitted because we passed `cross_pairs`.
    t_total = time.perf_counter()
    analyze(
        model=MODEL,
        revisions=REVISIONS,
        eval_batches=eval_batches,
        cross_pairs=CROSS_PAIRS,
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
    # Index rows by (observable, batch_a, batch_b) → sorted [(step, value)].
    # batch_a == batch_b for self pairs; otherwise it's the cross pair.
    series: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["observable"], row["batch_a"], row["batch_b"])
        series.setdefault(key, []).append((step_from_revision(row["revision"]), row["value"]))
    for key in series:
        series[key].sort()

    # Lines we want on each plot. For chi_loss / chi_net the cross-pair
    # values are derived (geometric means of the two self values), so we
    # only plot the self pairs there. For delta_loss / chi_pos the
    # cross-pair values are independent measurements, so we plot them
    # alongside the self pairs.
    self_lines = [(name, name) for name in EVAL_BATCH_NAMES]
    cross_lines = [(a, b) for a, b in CROSS_PAIRS]

    plot_recipes = {
        "chi_loss_normalized": self_lines,
        "chi_net_normalized": self_lines,
        "delta_loss": self_lines + cross_lines,
        "chi_pos": self_lines + cross_lines,
    }

    # Per-observable y-axis scaling. delta_loss and chi_pos can take
    # negative values (cross-pair gradient interference), so a plain log
    # scale would clip them. We use symlog with a per-observable
    # linthresh chosen to put the sign-change region in the linear band.
    yscales: dict[str, tuple[str, dict[str, float]]] = {
        "chi_loss_normalized": ("linear", {}),
        "chi_net_normalized": ("log", {}),
        "delta_loss": ("symlog", {"linthresh": 1.0}),
        "chi_pos": ("symlog", {"linthresh": 1e-9}),
    }

    for obs_name, plot_path in PLOTS.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for batch_a, batch_b in plot_recipes[obs_name]:
            xs_ys = series.get((obs_name, batch_a, batch_b), [])
            if not xs_ys:
                continue
            xs = [p[0] for p in xs_ys]
            ys = [p[1] for p in xs_ys]
            label = batch_a if batch_a == batch_b else cross_label(batch_a, batch_b)
            linestyle = "-" if batch_a == batch_b else "--"
            marker = "o" if batch_a == batch_b else "s"
            ax.plot(xs, ys, marker=marker, linestyle=linestyle, label=label)
        ax.set_xlabel("training step")
        ax.set_ylabel(obs_name)
        ax.set_xscale("log")
        scale_kind, scale_kwargs = yscales[obs_name]
        ax.set_yscale(scale_kind, **scale_kwargs)
        # Mark zero on symlog plots so the sign change is unambiguous.
        if scale_kind == "symlog":
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{obs_name} vs training step  ({MODEL})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"wrote {plot_path}")

    n_calls = len(REVISIONS) * (len(EVAL_BATCH_NAMES) + len(CROSS_PAIRS))
    print(
        "\nDone in "
        f"{total_s:.1f}s. Ran {len(REVISIONS)} checkpoints × "
        f"({len(EVAL_BATCH_NAMES)} self pairs + {len(CROSS_PAIRS)} cross pair) = "
        f"{n_calls} (ckpt, pair) measurements. Outputs at {OUT_DIR}/."
    )


if __name__ == "__main__":
    main()
