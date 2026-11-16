"""Stage 2 (OFFLINE, GPU node): R_dir and R_spec at the SFT and DPO checkpoints.

Implements the two diagnostics from [`README.md`](README.md) for
``allenai/Olmo-3-7B-Instruct-{SFT,DPO}`` on preference pairs selected by
``prefetch.py`` into ``samples.json`` (RewardBench by default; Dolci for the
seen-data control arm):

- **R_dir** — cosine between the parameter-space gradients of the SFT-style
  CE loss (on ``chosen``) and the DPO loss (on ``(chosen, rejected)``), at a
  given checkpoint. Cheap: two backward passes per pair plus a dot product.
- **R_spec** — ``chi_pos(L_DPO, L_DPO) / chi_pos(L_CE, L_CE)`` at a given
  checkpoint: the spectral position of the DPO loss gradient relative to the
  CE loss gradient, in the *same* model's NTK.

Both diagnostics need a DPO loss, which is not the closed-form CE path vatis
v1 ships (see ``vatis/core/observables.py``), so this script builds the DPO
loss by hand from vatis's lower-level, loss-agnostic primitives; ``chi_loss``
(``||grad_f L||^2``) and ``delta_loss`` (``||grad_theta L||^2``) come from
plain autograd — one joint ``autograd.grad`` against ``[logits, *params]``
per loss per pair.

**Error instrumentation** (added for the n_hutchinson / batch-size sweeps —
the whole point is that every noise/sampling question becomes an *offline*
analysis of one run's raw records):

- ``chi_net`` is estimated by a local per-sequence Hutchinson loop (built on
  ``vatis.core.probes``) that records **every per-probe estimate**
  ``||J_b^T v||^2`` per sequence. Offline, subsampling probes gives the
  error-vs-``n_hutchinson`` law; subsampling sequences (bootstrap) gives the
  error-vs-batch-size law and error bars on chi_net / chi_pos / R_spec. The
  first ``--boost-pairs`` pairs per domain get ``--n-hutchinson-boost`` probes
  (default 64) so the 1/sqrt(n) law is measurable over two decades; the rest
  get ``--n-hutchinson`` (default 8 — at B ~ 10^3 sequences the *total* probe
  count is what controls the error, so a small per-sequence n suffices).
- ``Tr(J J^T)`` is additive over sequences, so ``chi_net_dpo`` is assembled as
  ``chi_net(chosen) + chi_net(rejected)`` — the chosen-batch estimate is
  *shared* with ``chi_net_ce`` (its noise partially cancels in the R_spec
  ratio) and each sequence is probed exactly once (1/3 fewer backwards than
  stacking chosen+rejected and re-probing).
- The gradient passes run in **chunks** whose boundaries are the
  ``--grad-snapshots`` prefix sizes: two running flat-gradient accumulators
  (fp32, host RAM — ~28 GB each at 7B) are updated per chunk and the scalars
  ``<g_CE, g_DPO>``, ``||g_CE||^2``, ``||g_DPO||^2``, chi_loss sums are
  snapshotted at each boundary, giving **R_dir / delta_loss / chi_loss at
  nested batch sizes** with zero gradient recomputation. Raw (unnormalized)
  sums plus per-pair valid-token counts are stored so any normalization can
  be re-derived offline.

The DPO loss's reference policy is the (fixed, frozen) **SFT checkpoint**,
matching how Olmo's actual DPO training was set up. To avoid ever holding two
7B models in memory at once, the SFT pass is run first and the *policy*
log-probabilities computed there are also cached as the reference (`ref ==
policy` is correct at the SFT checkpoint, since DPO training begins with
`ref = policy`); the DPO pass then reuses that cache as its frozen reference
instead of loading the SFT model a second time.

Run offline on a GPU node (the cache must already hold both models — run
``prefetch.py`` on a login node first). Via SLURM::

    examples/benchmark/submit.sh examples/dpo_spectral_filter/run.sbatch

or interactively on a GPU::

    .venv/bin/python examples/dpo_spectral_filter/diagnostics.py

Outputs (in ``runs/{tag}/`` next to this script — one directory per run, so
e.g. a RewardBench run cannot clobber the Dolci results):

- ``results.json``     — one record per (checkpoint, domain) with all raw
  observables plus ``R_dir`` and ``R_spec`` and the grad-pass snapshots.
- ``sweep.json``       — per-pair raw records (per-probe chi_net estimates,
  per-pair grad norms / margins / valid-token counts) for offline error
  analysis.
- ``resources.json``   — time / memory / GPU-activity summary + timeline.

Results are flushed after every (checkpoint, domain) so a crash keeps
finished work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_EXAMPLES_DIR = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))  # run_config
sys.path.insert(0, str(_EXAMPLES_DIR))  # _helpers
sys.path.insert(0, str(_EXAMPLES_DIR / "olmo_sft_analysis"))  # resource_tracker

from run_config import configure_hf_cache  # noqa: E402

configure_hf_cache()

from _helpers import tokenize_chat_sft_sample  # noqa: E402
from resource_tracker import ResourceTracker  # noqa: E402

from vatis.analyzer import (  # noqa: E402
    _add_grads_into_flat,
    _dot_fp64,
    _sum_sq_fp64,
    _zero_param_vector,
)
from vatis.core.observables import chi_pos as chi_pos_fn  # noqa: E402
from vatis.core.probes import make_probe, mask_probe  # noqa: E402
from vatis.models.hf import causal_lm_valid_mask, load_hf_model  # noqa: E402

_RUNS_DIR = _HERE.parent / "runs"


def _run_dir(tag: str) -> Path:
    """Per-run directory holding samples.json + all outputs for one --tag."""
    return _RUNS_DIR / (tag or "default")


SFT_REVISION = "sft"
DPO_REVISION = "dpo"

# A pair's tokenized chosen/rejected sequences, each a dict of (S,) tensors.
TokenizedPair = dict[str, dict[str, torch.Tensor]]

POOLED_DOMAIN = "pooled"


class PooledAccumulator:
    """Cross-group running sums of the summed CE / DPO gradients (host, fp32).

    Groups are processed one after another; after each group its summed
    gradients ``G_d`` are folded into ``P = sum_{d' < d} G_d'``. The pooled
    scalars ``||P_ce||^2``, ``||P_dpo||^2``, ``<P_ce, P_dpo>`` follow from the
    same rank-1 recurrence as the within-group snapshots, and each group's
    cross terms against the earlier groups are recorded so any pooled scalar is
    reconstructable offline. The first group's tensors are adopted, not
    copied, so peak host memory is 2 pooled + 2 running buffers (~112 GB at 7B).
    """

    def __init__(self) -> None:
        self.p_ce: torch.Tensor | None = None
        self.p_dpo: torch.Tensor | None = None
        self.ce_sq = 0.0
        self.dpo_sq = 0.0
        self.dot = 0.0
        self.terms: list[dict[str, float]] = []

    def add(self, g_ce: torch.Tensor, g_dpo: torch.Tensor) -> None:
        t = {
            "g_ce_sq": float(_sum_sq_fp64(g_ce)),
            "g_dpo_sq": float(_sum_sq_fp64(g_dpo)),
            "g_ce_dot_g_dpo": float(_dot_fp64(g_ce, g_dpo)),
        }
        if self.p_ce is None or self.p_dpo is None:
            t |= {"p_ce_dot_g_ce": 0.0, "p_dpo_dot_g_dpo": 0.0}
            t |= {"p_ce_dot_g_dpo": 0.0, "g_ce_dot_p_dpo": 0.0}
            self.p_ce, self.p_dpo = g_ce, g_dpo
        else:
            t["p_ce_dot_g_ce"] = float(_dot_fp64(self.p_ce, g_ce))
            t["p_dpo_dot_g_dpo"] = float(_dot_fp64(self.p_dpo, g_dpo))
            t["p_ce_dot_g_dpo"] = float(_dot_fp64(self.p_ce, g_dpo))
            t["g_ce_dot_p_dpo"] = float(_dot_fp64(g_ce, self.p_dpo))
            self.p_ce += g_ce
            self.p_dpo += g_dpo
        self.ce_sq += 2.0 * t["p_ce_dot_g_ce"] + t["g_ce_sq"]
        self.dpo_sq += 2.0 * t["p_dpo_dot_g_dpo"] + t["g_dpo_sq"]
        self.dot += t["p_ce_dot_g_dpo"] + t["g_ce_dot_p_dpo"] + t["g_ce_dot_g_dpo"]
        self.terms.append(t)

    def free(self) -> None:
        self.p_ce = self.p_dpo = None


def tokenize_pairs(raw_pairs: list[dict], tokenizer, seq_len: int) -> list[TokenizedPair]:
    out = []
    for p in raw_pairs:
        out.append(
            {
                "chosen": tokenize_chat_sft_sample(p["chosen"], tokenizer, seq_len=seq_len),
                "rejected": tokenize_chat_sft_sample(p["rejected"], tokenizer, seq_len=seq_len),
            }
        )
    return out


def _move(item: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.unsqueeze(0).to(device) for k, v in item.items()}


def _forward_logits(model: torch.nn.Module, item: dict[str, torch.Tensor]) -> torch.Tensor:
    """Forward one moved ``(1, S)`` example and return its ``(1, S, V)`` logits."""
    return model(input_ids=item["input_ids"], attention_mask=item["attention_mask"]).logits


def _logp_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, *, average: bool = True
) -> torch.Tensor:
    """Sequence log-prob over valid (``labels != -100``) positions.

    If ``average`` (default), returns the per-token **mean** log-prob — the
    length-normalized quantity open-instruct's ``dpo_norm`` loss uses
    (``_get_batch_logps(average_log_prob=True)``), which is what Olmo-3's DPO
    stage trained with. If not, returns the **sum** (vanilla ``dpo``). The
    switch only changes the reduction: ``F.cross_entropy(reduction="mean", ...)``
    divides the summed per-token loss by the number of non-ignored targets,
    i.e. exactly ``sum / n_valid_tokens``.

    **Computed in fp32.** The model runs in bf16, but a completion's log-prob is
    a sum of hundreds of per-token terms (magnitude ~hundreds of nats); the DPO
    margin then *differences* two such sums, where the real signal is fractions
    of a nat. In bf16 the log-prob's quantization step at that magnitude is tens
    of nats, so the margin would be pure noise. We upcast the logits to fp32
    before the reduction (the gradient still flows back to the bf16 logits, so
    chi_loss / delta_loss are unaffected); only the forward log-prob value —
    which drives the margin — is made precise.
    """
    return -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean" if average else "sum",
    )


def _grads_sq_fp64(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    """``sum(g**2)`` over a per-pair grad tuple, accumulated in fp64."""
    total: torch.Tensor | None = None
    for g in grads:
        if g is None:
            continue
        sq = (g.to(dtype=torch.float64) ** 2).sum()
        total = sq if total is None else total + sq
    if total is None:
        return torch.zeros((), dtype=torch.float64)
    return total


def ce_chunk_pass(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict]]:
    """RAW (sum-reduced, unnormalized) CE gradients over one chunk of pairs.

    Returns the chunk's flat parameter gradient of ``sum_i CE_raw_i`` (still
    on ``device``) plus per-pair records. All values are unnormalized sums —
    the caller (and offline analysis) divide by valid-token counts.
    """
    flat = _zero_param_vector(params, device)
    recs: list[dict] = []

    for p in pairs:
        chosen = _move(p["chosen"], device)
        logits = _forward_logits(model, chosen)
        labels = chosen["labels"]
        ce_raw = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        outs = torch.autograd.grad(ce_raw, [logits, *params], allow_unused=True)
        g_logits, grads = outs[0], outs[1:]
        chi_pair = float((g_logits.to(dtype=torch.float64) ** 2).sum())
        grad_sq_pair = float(_grads_sq_fp64(grads))
        _add_grads_into_flat(flat, grads, params)
        recs.append(
            {
                "n_valid": int((labels != -100).sum().item()),
                "ce_raw": float(ce_raw.detach()),
                "chi_loss_raw": chi_pair,
                "grad_sq_raw": grad_sq_pair,
            }
        )
        del logits, outs, grads
    return flat, recs


def dpo_chunk_pass(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    device: torch.device,
    *,
    beta: float,
    ref_slice: list[tuple[float, float]] | None,
    dpo_norm: bool,
) -> tuple[torch.Tensor, list[dict], list[tuple[float, float]]]:
    """RAW (per-pair, un-averaged) DPO gradients over one chunk of pairs.

    ``L_DPO`` per pair is the standard Rafailov et al. (2023) sigmoid loss;
    the chunk gradient is the plain sum over pairs (the caller / offline
    analysis divide by the pair count). ``dpo_norm`` (default) uses
    length-normalized sequence log-probs (open-instruct's ``dpo_norm`` /
    Olmo-3's recipe). If ``ref_slice`` is ``None`` (the SFT checkpoint — its
    own reference), the policy's own detached log-probs are used as the
    reference and returned as a fresh cache slice; otherwise ``ref_slice[i]``
    supplies the frozen ``(logp_w_ref, logp_l_ref)`` for chunk pair ``i``.
    """
    flat = _zero_param_vector(params, device)
    recs: list[dict] = []
    new_refs: list[tuple[float, float]] = []

    for i, p in enumerate(pairs):
        chosen = _move(p["chosen"], device)
        rejected = _move(p["rejected"], device)
        z_w = _forward_logits(model, chosen)
        z_l = _forward_logits(model, rejected)
        logp_w = _logp_from_logits(z_w, chosen["labels"], average=dpo_norm)
        logp_l = _logp_from_logits(z_l, rejected["labels"], average=dpo_norm)

        if ref_slice is None:
            logp_w_ref = logp_w.detach()
            logp_l_ref = logp_l.detach()
            new_refs.append((float(logp_w_ref), float(logp_l_ref)))
        else:
            ref_w, ref_l = ref_slice[i]
            logp_w_ref = torch.tensor(ref_w, device=device, dtype=logp_w.dtype)
            logp_l_ref = torch.tensor(ref_l, device=device, dtype=logp_l.dtype)

        margin = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
        l_dpo = -F.logsigmoid(margin)

        outs = torch.autograd.grad(l_dpo, [z_w, z_l, *params], allow_unused=True)
        g_zw, g_zl, grads = outs[0], outs[1], outs[2:]
        chi_pair = 0.0
        if g_zw is not None:
            chi_pair += float((g_zw.to(dtype=torch.float64) ** 2).sum())
        if g_zl is not None:
            chi_pair += float((g_zl.to(dtype=torch.float64) ** 2).sum())
        grad_sq_pair = float(_grads_sq_fp64(grads))
        _add_grads_into_flat(flat, grads, params)
        n_valid_w = int((chosen["labels"] != -100).sum().item())
        n_valid_l = int((rejected["labels"] != -100).sum().item())
        del z_w, z_l, outs, grads

        # ||s_l||^2 = ||grad_theta logp_l||^2: one extra forward+backward on the
        # rejected sequence (its graph was freed by the DPO grad above). With
        # ||s_w||^2 (= CE grad_sq_raw / n_valid_w^2 under dpo_norm) and
        # ||s_w - s_l||^2 (= DPO grad_sq_raw / (beta*w)^2) this gives
        # <s_w, s_l> and the per-pair chosen/rejected cosine offline.
        z_l = _forward_logits(model, rejected)
        logp_l_again = _logp_from_logits(z_l, rejected["labels"], average=dpo_norm)
        score_l = torch.autograd.grad(logp_l_again, params, allow_unused=True)
        score_l_sq = float(_grads_sq_fp64(score_l))
        del z_l, logp_l_again, score_l

        recs.append(
            {
                "l_dpo": float(l_dpo.detach()),
                "margin": float(margin.detach()),
                "logp_w": float(logp_w.detach()),
                "logp_l": float(logp_l.detach()),
                "n_valid_w": n_valid_w,
                "n_valid_l": n_valid_l,
                "chi_loss_raw": chi_pair,
                "grad_sq_raw": grad_sq_pair,
                "score_l_sq_raw": score_l_sq,
            }
        )
    return flat, recs, new_refs


def grad_passes(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    device: torch.device,
    *,
    beta: float,
    ref_cache: list[tuple[float, float]] | None,
    dpo_norm: bool,
    snapshot_sizes: list[int],
    perturb_rel_steps: list[float] | None = None,
    return_grads: bool = False,
) -> dict[str, Any]:
    """CE + DPO gradient passes over all pairs, chunked at snapshot boundaries.

    Two running flat gradients live in HOST RAM (fp32, ~28 GB each at 7B —
    never both on the GPU); at every boundary the prefix scalars
    ``<g_CE, g_DPO>``, ``||g_CE||^2``, ``||g_DPO||^2`` and the chi_loss raw
    sums are recorded via the rank-1 update recurrence, so R_dir / delta_loss
    / chi_loss are available at each nested batch size for free. All stored
    sums are unnormalized; snapshot rows carry the prefix pair / valid-token
    counts needed to normalize offline.
    """
    n_pairs = len(pairs)
    boundaries = sorted({b for b in snapshot_sizes if 0 < b < n_pairs} | {n_pairs})

    g_ce_run = _zero_param_vector(params, torch.device("cpu"))
    g_dpo_run = _zero_param_vector(params, torch.device("cpu"))
    dot_run = 0.0
    nce_sq = 0.0
    ndpo_sq = 0.0
    chi_ce_raw_run = 0.0
    chi_dpo_raw_run = 0.0
    n_valid_run = 0
    per_pair_ce: list[dict] = []
    per_pair_dpo: list[dict] = []
    all_new_refs: list[tuple[float, float]] = []
    snapshots: list[dict] = []

    prev = 0
    for b in boundaries:
        chunk = pairs[prev:b]

        flat_ce_gpu, recs_ce = ce_chunk_pass(model, params, chunk, device)
        flat_ce = flat_ce_gpu.to("cpu")
        del flat_ce_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Prefix recurrence: update the CE-side scalars BEFORE folding the
        # chunk into the running vector (they need the k-1 prefix), and the
        # cross dot against the *not yet updated* DPO running vector.
        nce_sq += 2.0 * float(_dot_fp64(g_ce_run, flat_ce)) + float(_sum_sq_fp64(flat_ce))
        dot_run += float(_dot_fp64(flat_ce, g_dpo_run))
        g_ce_run += flat_ce
        del flat_ce

        ref_slice = ref_cache[prev:b] if ref_cache is not None else None
        flat_dpo_gpu, recs_dpo, new_refs = dpo_chunk_pass(
            model, params, chunk, device, beta=beta, ref_slice=ref_slice, dpo_norm=dpo_norm
        )
        flat_dpo = flat_dpo_gpu.to("cpu")
        del flat_dpo_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ndpo_sq += 2.0 * float(_dot_fp64(g_dpo_run, flat_dpo)) + float(_sum_sq_fp64(flat_dpo))
        # g_ce_run already contains this chunk, so this one term completes
        # dot(G_ce_k, G_dpo_k) = dot(G_ce_{k-1}, G_dpo_{k-1})
        #                        + dot(gce_k, G_dpo_{k-1}) + dot(G_ce_k, gdpo_k).
        dot_run += float(_dot_fp64(g_ce_run, flat_dpo))
        g_dpo_run += flat_dpo
        del flat_dpo

        chi_ce_raw_run += sum(r["chi_loss_raw"] for r in recs_ce)
        chi_dpo_raw_run += sum(r["chi_loss_raw"] for r in recs_dpo)
        n_valid_run += sum(r["n_valid"] for r in recs_ce)
        per_pair_ce.extend(recs_ce)
        per_pair_dpo.extend(recs_dpo)
        all_new_refs.extend(new_refs)

        r_dir_b = (
            dot_run / (nce_sq**0.5 * ndpo_sq**0.5) if nce_sq > 0 and ndpo_sq > 0 else float("nan")
        )
        snapshots.append(
            {
                "n_pairs": b,
                "n_valid_chosen": n_valid_run,
                "dot_raw": dot_run,
                "grad_ce_sq_raw": nce_sq,
                "grad_dpo_sq_raw": ndpo_sq,
                "chi_loss_ce_raw": chi_ce_raw_run,
                "chi_loss_dpo_raw": chi_dpo_raw_run,
                "r_dir": r_dir_b,
            }
        )
        print(f"    grad snapshot B={b}: R_dir={r_dir_b:+.4f}", flush=True)
        prev = b

    perturbation = None
    if perturb_rel_steps:
        refs = ref_cache if ref_cache is not None else all_new_refs
        perturbation = perturbation_pass(
            model,
            params,
            pairs,
            device,
            grads={"ce": g_ce_run, "dpo": g_dpo_run},
            rel_steps=perturb_rel_steps,
            refs=refs,
            beta=beta,
            dpo_norm=dpo_norm,
        )

    # Free the two ~28 GB host buffers before the chi_net pass, unless the
    # caller folds them into a pooled (cross-group) sum.
    grads_out = (g_ce_run, g_dpo_run) if return_grads else None
    del g_ce_run, g_dpo_run

    n_valid_total = max(n_valid_run, 1)
    return {
        "chi_loss_ce": chi_ce_raw_run / n_valid_total**2,
        "delta_loss_ce": nce_sq / n_valid_total**2,
        "chi_loss_dpo": chi_dpo_raw_run / max(n_pairs, 1) ** 2,
        "delta_loss_dpo": ndpo_sq / max(n_pairs, 1) ** 2,
        "r_dir": snapshots[-1]["r_dir"],
        "n_valid": n_valid_run,
        "snapshots": snapshots,
        "per_pair_ce": per_pair_ce,
        "per_pair_dpo": per_pair_dpo,
        "new_ref_cache": all_new_refs if ref_cache is None else None,
        "perturbation": perturbation,
        "grads": grads_out,
    }


def _seq_ce_and_logp(
    logits: torch.Tensor, labels: torch.Tensor, *, dpo_norm: bool
) -> tuple[float, float]:
    """(sum-reduced CE, DPO log-prob) of one sequence from fp32 logits."""
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")
    logp = -F.cross_entropy(
        flat_logits, flat_labels, ignore_index=-100, reduction="mean" if dpo_norm else "sum"
    )
    return float(ce), float(logp)


def _dpo_loss_value(logp_w: float, logp_l: float, ref: tuple[float, float], beta: float) -> float:
    margin = beta * ((logp_w - ref[0]) - (logp_l - ref[1]))
    return float(-F.logsigmoid(torch.tensor(margin, dtype=torch.float64)))


@torch.no_grad()
def perturbation_pass(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    device: torch.device,
    *,
    grads: dict[str, torch.Tensor],
    rel_steps: list[float],
    refs: list[tuple[float, float]],
    beta: float,
    dpo_norm: bool,
) -> dict[str, Any]:
    """Finite-difference probes along the full-batch loss gradients.

    For each direction ``d = g_X / ||g_X||`` (X in ``grads``: the raw summed CE
    and DPO gradients, host fp32) and each relative step ``r``, the weights are
    set to ``theta +- eps*d`` with ``eps = r * ||theta||`` and every pair is
    re-evaluated (forward only). Recorded per (direction, step):

    - ``pred``: the first-order loss change ``<g_A, Delta>`` for every loss A,
      using the *realized* bf16 step ``Delta`` (weights are bf16, so small steps
      round away element-wise; predicting from the realized step keeps the
      linearity check exact to first order).
    - ``loss_plus`` / ``loss_minus``: summed CE and DPO losses at +-eps. With
      ``loss_zero`` (same code at theta) this gives the realized change
      (linearity / one-step transfer) and the curvature along d.
    - ``jd_sq_*``: ``sum ||f(theta+Delta) - f(theta-Delta)||^2`` over valid
      positions of chosen / rejected sequences, i.e. ``~4 ||J Delta||^2`` —
      the second spectral moment ``u^T Theta^2 u`` along the gradient.
    """
    theta0 = [p.detach().clone() for p in params]
    theta_norm = sum(float(_sum_sq_fp64(t.reshape(-1).float())) for t in theta0) ** 0.5

    def set_params(src: list[torch.Tensor]) -> None:
        for p, s in zip(params, src, strict=True):
            p.copy_(s)

    def eval_losses() -> dict[str, float]:
        ce_total, dpo_total = 0.0, 0.0
        for i, pr in enumerate(pairs):
            lp = {}
            for side in ("chosen", "rejected"):
                item = _move(pr[side], device)
                ce, lp[side] = _seq_ce_and_logp(
                    _forward_logits(model, item), item["labels"], dpo_norm=dpo_norm
                )
                if side == "chosen":
                    ce_total += ce
            dpo_total += _dpo_loss_value(lp["chosen"], lp["rejected"], refs[i], beta)
        return {"ce": ce_total, "dpo": dpo_total}

    loss_zero = eval_losses()
    records = []
    for d_name, g_dir in grads.items():
        g_norm = float(_sum_sq_fp64(g_dir)) ** 0.5
        for r in rel_steps:
            eps = r * theta_norm
            plus, minus = [], []
            pred = {a: {"plus": 0.0, "minus": 0.0} for a in grads}
            changed = 0
            delta_diff_sq = 0.0
            offset = 0
            for t0 in theta0:
                n = t0.numel()
                d = g_dir[offset : offset + n].to(device).view_as(t0) * (eps / g_norm)
                t_plus = (t0.float() + d).to(t0.dtype)
                t_minus = (t0.float() - d).to(t0.dtype)
                del d
                delta_plus = (t_plus.float() - t0.float()).reshape(-1)
                delta_minus = (t_minus.float() - t0.float()).reshape(-1)
                changed += int((delta_plus != 0).sum())
                delta_diff_sq += float(_sum_sq_fp64(delta_plus - delta_minus))
                for a, g_a in grads.items():
                    g_slice = g_a[offset : offset + n].to(device)
                    pred[a]["plus"] += float(_dot_fp64(g_slice, delta_plus))
                    pred[a]["minus"] += float(_dot_fp64(g_slice, delta_minus))
                    del g_slice
                del delta_plus, delta_minus
                plus.append(t_plus)
                minus.append(t_minus)
                offset += n

            totals = {s: {"ce": 0.0, "dpo": 0.0} for s in ("plus", "minus")}
            jd_sq = {"chosen": 0.0, "rejected": 0.0}
            for i, pr in enumerate(pairs):
                lp: dict[tuple[str, str], float] = {}
                for side in ("chosen", "rejected"):
                    item = _move(pr[side], device)
                    set_params(plus)
                    z_p = _forward_logits(model, item).float()
                    set_params(minus)
                    z_m = _forward_logits(model, item).float()
                    vmask = causal_lm_valid_mask(item, z_p)
                    diff = z_p - z_m
                    if vmask is not None:
                        diff = diff[vmask.bool()]
                    jd_sq[side] += float((diff.double() ** 2).sum())
                    del diff
                    for sign, z in (("plus", z_p), ("minus", z_m)):
                        ce, lp[(sign, side)] = _seq_ce_and_logp(
                            z, item["labels"], dpo_norm=dpo_norm
                        )
                        if side == "chosen":
                            totals[sign]["ce"] += ce
                    del z_p, z_m
                for sign in ("plus", "minus"):
                    totals[sign]["dpo"] += _dpo_loss_value(
                        lp[(sign, "chosen")], lp[(sign, "rejected")], refs[i], beta
                    )
            set_params(theta0)
            del plus, minus
            if device.type == "cuda":
                torch.cuda.empty_cache()
            records.append(
                {
                    "direction": d_name,
                    "rel_step": r,
                    "eps": eps,
                    "theta_norm": theta_norm,
                    "g_norm_raw": g_norm,
                    "frac_params_changed": changed / max(sum(t.numel() for t in theta0), 1),
                    "delta_diff_sq": delta_diff_sq,
                    "pred": pred,
                    "loss_plus": totals["plus"],
                    "loss_minus": totals["minus"],
                    "jd_sq_chosen": jd_sq["chosen"],
                    "jd_sq_rejected": jd_sq["rejected"],
                }
            )
            print(
                f"    perturb d={d_name} r={r:g}: dCE={totals['plus']['ce'] - loss_zero['ce']:+.4e}"
                f" (pred {pred['ce']['plus']:+.4e})  dDPO="
                f"{totals['plus']['dpo'] - loss_zero['dpo']:+.4e} (pred {pred['dpo']['plus']:+.4e})",
                flush=True,
            )
    del theta0
    return {"loss_zero": loss_zero, "records": records}


def chi_net_probe_pass(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    device: torch.device,
    *,
    n_base: int,
    n_boost: int,
    boost_pairs: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Per-sequence Hutchinson with every per-probe estimate recorded.

    Each sequence (chosen and rejected of every pair) is probed independently;
    a probe contributes one unbiased estimate ``||J_b^T v||^2`` of that
    sequence's ``Tr(J_b J_b^T)``. ``chi_net`` of any sub-batch at any
    ``n <= n_probes`` is then a pure offline reduction over the records. The
    first ``boost_pairs`` pairs get ``n_boost`` probes, the rest ``n_base``.
    """
    per_pair: list[dict] = []
    for i, p in enumerate(pairs):
        n_probes = n_boost if i < boost_pairs else n_base
        rec: dict[str, Any] = {"n_probes": n_probes}
        for side in ("chosen", "rejected"):
            item = _move(p[side], device)
            vals: list[float] = []
            n_valid = 0
            for _ in range(n_probes):
                logits = _forward_logits(model, item)
                vmask = causal_lm_valid_mask(item, logits)
                probe = make_probe(
                    tuple(logits.shape),
                    distribution="rademacher",
                    device=logits.device,
                    dtype=torch.float32,
                    generator=generator,
                )
                probe = mask_probe(probe, vmask)
                projected = (logits * probe.to(dtype=logits.dtype)).sum()
                grads = torch.autograd.grad(projected, params, allow_unused=True)
                vals.append(float(_grads_sq_fp64(grads)))
                n_valid = int(vmask.sum().item()) if vmask is not None else 0
                del logits, grads, probe
            rec[f"probes_{side}"] = vals
            rec[f"n_valid_{side}"] = n_valid
        per_pair.append(rec)
        if (i + 1) % 50 == 0:
            print(f"    chi_net probes: {i + 1}/{len(pairs)} pairs", flush=True)

    chi_net_chosen = sum(
        sum(r["probes_chosen"]) / len(r["probes_chosen"]) for r in per_pair if r["probes_chosen"]
    )
    chi_net_rejected = sum(
        sum(r["probes_rejected"]) / len(r["probes_rejected"])
        for r in per_pair
        if r["probes_rejected"]
    )
    return {
        "chi_net_chosen": chi_net_chosen,
        "chi_net_rejected": chi_net_rejected,
        "n_valid_chosen": sum(r["n_valid_chosen"] for r in per_pair),
        "n_valid_rejected": sum(r["n_valid_rejected"] for r in per_pair),
        "per_pair": per_pair,
    }


def compute_domain_observables(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    *,
    device: torch.device,
    beta: float,
    n_base: int,
    n_boost: int,
    boost_pairs: int,
    seed: int,
    snapshot_sizes: list[int],
    ref_cache: list[tuple[float, float]] | None,
    dpo_norm: bool = True,
    perturb_rel_steps: list[float] | None = None,
    pooled: PooledAccumulator | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns ``(summary_record, raw_record)`` for one (checkpoint, domain).

    With ``pooled``, the domain's summed CE / DPO gradients are folded into the
    cross-domain running sums (and freed) before the chi_net pass.
    """
    g = grad_passes(
        model,
        params,
        pairs,
        device,
        beta=beta,
        ref_cache=ref_cache,
        dpo_norm=dpo_norm,
        snapshot_sizes=snapshot_sizes,
        perturb_rel_steps=perturb_rel_steps,
        return_grads=pooled is not None,
    )
    if pooled is not None:
        g_ce_dom, g_dpo_dom = g.pop("grads")
        pooled.add(g_ce_dom, g_dpo_dom)
        del g_ce_dom, g_dpo_dom
    g.pop("grads", None)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    cn = chi_net_probe_pass(
        model,
        params,
        pairs,
        device,
        n_base=n_base,
        n_boost=n_boost,
        boost_pairs=boost_pairs,
        generator=gen,
    )
    chi_net_ce = cn["chi_net_chosen"]
    chi_net_dpo = cn["chi_net_chosen"] + cn["chi_net_rejected"]

    chi_pos_ce = float(chi_pos_fn(g["delta_loss_ce"], g["chi_loss_ce"], chi_net_ce))
    chi_pos_dpo = float(chi_pos_fn(g["delta_loss_dpo"], g["chi_loss_dpo"], chi_net_dpo))
    r_spec = chi_pos_dpo / chi_pos_ce if chi_pos_ce != 0.0 else float("nan")

    summary = {
        "n_pairs": len(pairs),
        "n_valid_chosen_tokens": g["n_valid"],
        "chi_loss_ce": g["chi_loss_ce"],
        "delta_loss_ce": g["delta_loss_ce"],
        "chi_net_ce": chi_net_ce,
        "n_valid_chi_net_ce": cn["n_valid_chosen"],
        "chi_pos_ce": chi_pos_ce,
        "chi_loss_dpo": g["chi_loss_dpo"],
        "delta_loss_dpo": g["delta_loss_dpo"],
        "chi_net_dpo": chi_net_dpo,
        "n_valid_chi_net_dpo": cn["n_valid_chosen"] + cn["n_valid_rejected"],
        "chi_pos_dpo": chi_pos_dpo,
        "r_spec": r_spec,
        "r_dir": g["r_dir"],
        "grad_snapshots": g["snapshots"],
        "_new_ref_cache": g["new_ref_cache"],
    }
    raw = {
        "per_pair_ce": g["per_pair_ce"],
        "per_pair_dpo": g["per_pair_dpo"],
        "per_pair_chi_net": cn["per_pair"],
        "perturbation": g["perturbation"],
    }
    return summary, raw


def pooled_summary(pooled: PooledAccumulator, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled observables over all groups, normalized exactly like one group.

    chi_loss and chi_net are additive across groups (no cross terms); the
    gradient norms and the CE-DPO dot come from the pooled recurrence.
    """
    last = [s["grad_snapshots"][-1] for s in summaries]
    n_pairs = sum(s["n_pairs"] for s in summaries)
    n_valid = max(sum(x["n_valid_chosen"] for x in last), 1)
    chi_loss_ce = sum(x["chi_loss_ce_raw"] for x in last) / n_valid**2
    chi_loss_dpo = sum(x["chi_loss_dpo_raw"] for x in last) / max(n_pairs, 1) ** 2
    delta_loss_ce = pooled.ce_sq / n_valid**2
    delta_loss_dpo = pooled.dpo_sq / max(n_pairs, 1) ** 2
    chi_net_ce = sum(s["chi_net_ce"] for s in summaries)
    chi_net_dpo = sum(s["chi_net_dpo"] for s in summaries)
    chi_pos_ce = float(chi_pos_fn(delta_loss_ce, chi_loss_ce, chi_net_ce))
    chi_pos_dpo = float(chi_pos_fn(delta_loss_dpo, chi_loss_dpo, chi_net_dpo))
    denom = (pooled.ce_sq * pooled.dpo_sq) ** 0.5
    return {
        "n_pairs": n_pairs,
        "n_valid_chosen_tokens": n_valid,
        "chi_loss_ce": chi_loss_ce,
        "delta_loss_ce": delta_loss_ce,
        "chi_net_ce": chi_net_ce,
        "chi_pos_ce": chi_pos_ce,
        "chi_loss_dpo": chi_loss_dpo,
        "delta_loss_dpo": delta_loss_dpo,
        "chi_net_dpo": chi_net_dpo,
        "chi_pos_dpo": chi_pos_dpo,
        "r_spec": chi_pos_dpo / chi_pos_ce if chi_pos_ce != 0.0 else float("nan"),
        "r_dir": pooled.dot / denom if denom > 0 else float("nan"),
        "grad_ce_sq_raw": pooled.ce_sq,
        "grad_dpo_sq_raw": pooled.dpo_sq,
        "dot_raw": pooled.dot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        default=None,
        help="path to prefetch.py's samples.json; default runs/{tag}/samples.json.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=None, help="override the seq_len recorded in samples.json"
    )
    parser.add_argument("--beta", type=float, default=0.1, help="DPO loss temperature.")
    parser.add_argument(
        "--dpo-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Length-normalize DPO sequence log-probs (open-instruct dpo_norm, "
        "Olmo-3's recipe). Use --no-dpo-norm for the vanilla summed-logp dpo loss.",
    )
    parser.add_argument(
        "--n-hutchinson",
        type=int,
        default=8,
        help="base probes per sequence; the total error is set by n * B, so at "
        "B ~ 10^3 sequences a small n suffices.",
    )
    parser.add_argument(
        "--n-hutchinson-boost",
        type=int,
        default=64,
        help="probes per sequence for the first --boost-pairs pairs (n-scaling arm).",
    )
    parser.add_argument(
        "--boost-pairs",
        type=int,
        default=20,
        help="number of leading pairs per domain that get the boosted probe count.",
    )
    parser.add_argument(
        "--grad-snapshots",
        default="10,40,160,640",
        help="comma-separated nested prefix sizes at which R_dir/delta_loss/"
        "chi_loss are snapshotted (the full batch is always included).",
    )
    parser.add_argument(
        "--perturb-rel-steps",
        default="",
        help="comma-separated relative step sizes r (eps = r * ||theta||) for the "
        "finite-difference pass along the CE and DPO gradients (one-step transfer, "
        "curvature, second spectral moment). Empty (default) skips it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pooled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with more than one pair group, also report pooled (all-group) "
        "observables; keeps two extra ~28 GB host buffers at 7B.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="MODEL[@REVISION]",
        help="evaluate this checkpoint with itself as the DPO reference (margin 0); "
        "repeatable. Default: the SFT then DPO model from samples.json, with the "
        "SFT checkpoint as the DPO model's reference.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="run name: inputs/outputs live in runs/{tag}/ (empty = runs/default/).",
    )
    parser.add_argument("--sampler-interval-s", type=float, default=0.25)
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="optional attention-backend passthrough to from_pretrained.",
    )
    args = parser.parse_args()

    run_dir = _run_dir(args.tag)
    samples_path = Path(args.samples) if args.samples else run_dir / "samples.json"
    if not samples_path.exists():
        raise SystemExit(
            f"{samples_path} not found. Run prefetch.py on a login node first "
            "(see this example's README)."
        )
    payload = json.loads(samples_path.read_text())
    meta = payload["meta"]
    sft_model = meta["sft_model"]
    dpo_model = meta["dpo_model"]
    seq_len = args.seq_len or meta["seq_len_target"]
    snapshot_sizes = [int(x) for x in args.grad_snapshots.split(",") if x.strip()]
    perturb_rel_steps = [float(x) for x in args.perturb_rel_steps.split(",") if x.strip()]

    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    sweep_path = run_dir / "sweep.json"
    resources_path = run_dir / "resources.json"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = "bf16"
    else:
        device = torch.device("cpu")
        dtype = "fp32"
        print("WARNING: no CUDA device — a 7B model will not realistically run on CPU.")

    print(f"dpo_spectral_filter diagnostics — sft={sft_model}  dpo={dpo_model}")
    print(
        f"  device={device}, dtype={dtype}, seq_len={seq_len}, beta={args.beta}, "
        f"dpo_norm={args.dpo_norm}"
    )
    print(
        f"  n_hutchinson={args.n_hutchinson} (boost {args.n_hutchinson_boost} on first "
        f"{args.boost_pairs} pairs), snapshots={snapshot_sizes}, "
        f"attn={args.attn_implementation}"
    )
    print(f"  outputs: {results_path.name}, {sweep_path.name}, {resources_path.name}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(sft_model)
    domains = list(payload["pairs"])
    tokenized_pairs = {
        domain: tokenize_pairs(payload["pairs"][domain], tokenizer, seq_len) for domain in domains
    }
    for domain, pairs in tokenized_pairs.items():
        print(f"  {domain}: {len(pairs)} pairs")

    run_config = {
        "n_hutchinson": args.n_hutchinson,
        "n_hutchinson_boost": args.n_hutchinson_boost,
        "boost_pairs": args.boost_pairs,
        "grad_snapshots": snapshot_sizes,
        "perturb_rel_steps": perturb_rel_steps,
        "seed": args.seed,
        "beta": args.beta,
        "dpo_norm": args.dpo_norm,
        "seq_len": seq_len,
        "pooled": args.pooled and len(domains) > 1,
    }
    tracker = ResourceTracker(device=device, interval_s=args.sampler_interval_s)
    results: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    ref_cache_by_domain: dict[str, list[tuple[float, float]]] = {}

    def flush() -> None:
        results_path.write_text(
            json.dumps({"meta": meta, "run_config": run_config, "results": results}, indent=2)
        )
        sweep_path.write_text(
            json.dumps({"meta": meta, "run_config": run_config, "records": raw_records})
        )

    # (label, model, hf_revision, uses the SFT pass's cached reference log-probs)
    if args.checkpoint:
        checkpoints = []
        for spec in args.checkpoint:
            model_name, _, hf_rev = spec.partition("@")
            checkpoints.append((hf_rev or "main", model_name, hf_rev or None, False))
    else:
        checkpoints = [
            (SFT_REVISION, sft_model, None, False),
            (DPO_REVISION, dpo_model, None, True),
        ]

    with tracker:
        for revision, model_name, hf_revision, ref_from_sft in checkpoints:
            with tracker.measure(revision):
                t0 = time.perf_counter()
                bundle = load_hf_model(
                    model_name,
                    revision=hf_revision,
                    dtype=dtype,
                    device=device,
                    attn_implementation=args.attn_implementation,
                )
                print(f"  loaded {model_name}@{revision} in {time.perf_counter() - t0:.1f}s")

                pooled = PooledAccumulator() if run_config["pooled"] else None
                ckpt_summaries: list[dict[str, Any]] = []
                for domain in domains:
                    ref_cache = ref_cache_by_domain.get(domain) if ref_from_sft else None
                    summary, raw = compute_domain_observables(
                        bundle.model,
                        bundle.params,
                        tokenized_pairs[domain],
                        device=device,
                        beta=args.beta,
                        n_base=args.n_hutchinson,
                        n_boost=args.n_hutchinson_boost,
                        boost_pairs=args.boost_pairs,
                        seed=args.seed,
                        snapshot_sizes=snapshot_sizes,
                        ref_cache=ref_cache,
                        dpo_norm=args.dpo_norm,
                        perturb_rel_steps=perturb_rel_steps,
                        pooled=pooled,
                    )
                    ckpt_summaries.append(summary)
                    new_cache = summary.pop("_new_ref_cache")
                    if revision == SFT_REVISION:
                        ref_cache_by_domain[domain] = new_cache
                    results.append(
                        {
                            "checkpoint": model_name,
                            "revision": revision,
                            "hf_revision": hf_revision,
                            "ref": "sft" if ref_from_sft else "self",
                            "domain": domain,
                        }
                        | summary
                    )
                    raw_records.append({"revision": revision, "domain": domain} | raw)
                    flush()
                    print(
                        f"  [{revision}] {domain}: R_dir={summary['r_dir']:+.4f} "
                        f"R_spec={summary['r_spec']:.4f} "
                        f"chi_pos_ce={summary['chi_pos_ce']:.4e} "
                        f"chi_pos_dpo={summary['chi_pos_dpo']:.4e}"
                    )

                if pooled is not None:
                    pooled.free()
                    psum = pooled_summary(pooled, ckpt_summaries)
                    results.append(
                        {
                            "checkpoint": model_name,
                            "revision": revision,
                            "hf_revision": hf_revision,
                            "ref": "sft" if ref_from_sft else "self",
                            "domain": POOLED_DOMAIN,
                        }
                        | psum
                    )
                    raw_records.append(
                        {
                            "revision": revision,
                            "domain": POOLED_DOMAIN,
                            "groups": domains,
                            "pooled_terms": pooled.terms,
                        }
                    )
                    flush()
                    print(
                        f"  [{revision}] {POOLED_DOMAIN}: R_dir={psum['r_dir']:+.4f} "
                        f"R_spec={psum['r_spec']:.4f}"
                    )

                del bundle
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    flush()
    tracker.save(str(resources_path))
    print(f"\nwrote {results_path}")
    print(f"wrote {sweep_path}")
    print(f"wrote {resources_path}")
    print("\n=== resources ===")
    print(tracker.format_summary())


if __name__ == "__main__":
    main()
