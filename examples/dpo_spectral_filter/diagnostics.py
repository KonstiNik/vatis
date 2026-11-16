"""Stage 2 (OFFLINE, GPU node): R_dir and R_spec at the SFT and DPO checkpoints.

Implements the two diagnostics from [`README.md`](README.md) for
``allenai/Olmo-3-7B-Instruct-{SFT,DPO}`` on ``allenai/Dolci-Instruct-DPO``
preference pairs (selected by ``prefetch.py`` into ``samples.json``):

- **R_dir** — cosine between the parameter-space gradients of the SFT-style
  CE loss (on ``chosen``) and the DPO loss (on ``(chosen, rejected)``), at a
  given checkpoint. Cheap: two backward passes per pair plus a dot product.
- **R_spec** — ``chi_pos(L_DPO, L_DPO) / chi_pos(L_CE, L_CE)`` at a given
  checkpoint: the spectral position of the DPO loss gradient relative to the
  CE loss gradient, in the *same* model's NTK.

Both diagnostics need a DPO loss, which is not the closed-form CE path vatis
v1 ships (see ``vatis/core/observables.py``), so this script builds the DPO
loss by hand from vatis's lower-level, loss-agnostic primitives:

- ``chi_net`` (``Tr(J J^T)`` over valid output positions) is loss-independent
  — :class:`vatis.core.chi_net.HutchinsonEstimator` is reused directly for
  both the CE batch (``chosen`` only) and the DPO batch (``chosen`` +
  ``rejected`` stacked, each masked to its own completion tokens).
- ``chi_loss`` (``||grad_f L||^2``) and ``delta_loss`` (``||grad_theta
  L||^2``) for the DPO loss are computed by plain autograd (one extra
  ``autograd.grad`` call against the logits, then a backward against the
  parameters) instead of the CE closed form — mathematically the same
  definition, just without the softmax-onehot shortcut.

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

Outputs (next to this script):

- ``results.json``     — one record per (checkpoint, domain) with all raw
  observables plus ``R_dir`` and ``R_spec``.
- ``resources.json``   — time / memory / GPU-activity summary + timeline.
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

from _helpers import stack_lm_batch, tokenize_chat_sft_sample  # noqa: E402
from resource_tracker import ResourceTracker  # noqa: E402

from vatis.analyzer import (  # noqa: E402
    _add_grads_into_flat,
    _dot_fp64,
    _sum_sq_fp64,
    _zero_param_vector,
)
from vatis.core.chi_net import HutchinsonEstimator  # noqa: E402
from vatis.core.observables import chi_pos as chi_pos_fn  # noqa: E402
from vatis.models.hf import (  # noqa: E402
    causal_lm_forward,
    causal_lm_loss,
    causal_lm_valid_mask,
    load_hf_model,
)

SAMPLES_PATH = _HERE.parent / "samples.json"
RESULTS_PATH = _HERE.parent / "results.json"
RESOURCES_PATH = _HERE.parent / "resources.json"

DOMAINS = ("code", "english")
SFT_REVISION = "sft"
DPO_REVISION = "dpo"

# A pair's tokenized chosen/rejected sequences, each a dict of (S,) tensors.
TokenizedPair = dict[str, dict[str, torch.Tensor]]


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


def _logp_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``sum_t log p(y_t | x)`` over valid (``labels != -100``) positions.

    ``-cross_entropy(..., reduction="sum")`` is exactly the sum of per-token
    log-probabilities at the target ids, restricted to valid positions.

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
        reduction="sum",
    )


def ce_pass(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Accumulate ``chi_loss_CE`` and the flat ``grad_theta L_CE`` over all pairs.

    ``L_CE`` is the standard mean-reduced completion-only cross entropy on
    each pair's ``chosen`` sequence — the SFT objective.
    """
    n_valid_total = sum(int((p["chosen"]["labels"] != -100).sum().item()) for p in pairs)
    chi_loss_acc = torch.zeros((), dtype=torch.float64, device=device)
    flat_grad = _zero_param_vector(params, device)

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
        (g_logits,) = torch.autograd.grad(ce_raw, [logits], retain_graph=True)
        chi_loss_acc = chi_loss_acc + (g_logits.to(dtype=torch.float64) ** 2).sum()

        grads = torch.autograd.grad(
            ce_raw / max(n_valid_total, 1), params, retain_graph=False, allow_unused=True
        )
        _add_grads_into_flat(flat_grad, grads, params)
        del logits

    chi_loss_ce = chi_loss_acc / float(max(n_valid_total, 1) ** 2)
    delta_loss_ce = _sum_sq_fp64(flat_grad)
    # Offload immediately: the caller holds both this and the DPO pass's
    # ~28 GB (7B params, fp32) flat-grad buffer at once for the R_dir dot,
    # plus runs the chi_net Hutchinson pass afterwards. Keeping both on-GPU
    # would crowd out that headroom (see run.sbatch for the memory budget).
    flat_grad_cpu = flat_grad.to("cpu")
    del flat_grad
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "chi_loss": chi_loss_ce.to(dtype=torch.float32),
        "delta_loss": delta_loss_ce.to(dtype=torch.float32),
        "grad": flat_grad_cpu,
        "n_valid": n_valid_total,
    }


def dpo_pass(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    device: torch.device,
    *,
    beta: float,
    ref_cache: list[tuple[float, float]] | None,
) -> dict[str, Any]:
    """Accumulate ``chi_loss_DPO`` and the flat ``grad_theta L_DPO`` over all pairs.

    ``L_DPO`` is the standard Rafailov et al. (2023) sigmoid loss, mean-reduced
    over pairs. If ``ref_cache`` is ``None`` (the SFT checkpoint — its own
    reference), the policy's own (detached) log-probabilities are used as the
    reference and also returned as a fresh cache for reuse at the DPO
    checkpoint; otherwise ``ref_cache[i]`` supplies the frozen
    ``(logp_w_ref, logp_l_ref)`` for pair ``i``.
    """
    n_pairs = len(pairs)
    chi_loss_acc = torch.zeros((), dtype=torch.float64, device=device)
    flat_grad = _zero_param_vector(params, device)
    new_ref_cache: list[tuple[float, float]] = []

    for i, p in enumerate(pairs):
        chosen = _move(p["chosen"], device)
        rejected = _move(p["rejected"], device)
        z_w = _forward_logits(model, chosen)
        z_l = _forward_logits(model, rejected)
        logp_w = _logp_from_logits(z_w, chosen["labels"])
        logp_l = _logp_from_logits(z_l, rejected["labels"])

        if ref_cache is None:
            logp_w_ref = logp_w.detach()
            logp_l_ref = logp_l.detach()
            new_ref_cache.append((float(logp_w_ref), float(logp_l_ref)))
        else:
            ref_w, ref_l = ref_cache[i]
            logp_w_ref = torch.tensor(ref_w, device=device, dtype=logp_w.dtype)
            logp_l_ref = torch.tensor(ref_l, device=device, dtype=logp_l.dtype)

        margin = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
        l_dpo = -F.logsigmoid(margin)

        g_zw, g_zl = torch.autograd.grad(l_dpo, [z_w, z_l], retain_graph=True, allow_unused=True)
        if g_zw is not None:
            chi_loss_acc = chi_loss_acc + (g_zw.to(dtype=torch.float64) ** 2).sum()
        if g_zl is not None:
            chi_loss_acc = chi_loss_acc + (g_zl.to(dtype=torch.float64) ** 2).sum()

        grads = torch.autograd.grad(
            l_dpo / max(n_pairs, 1), params, retain_graph=False, allow_unused=True
        )
        _add_grads_into_flat(flat_grad, grads, params)
        del z_w, z_l

    chi_loss_dpo = chi_loss_acc / float(max(n_pairs, 1) ** 2)
    delta_loss_dpo = _sum_sq_fp64(flat_grad)
    flat_grad_cpu = flat_grad.to("cpu")
    del flat_grad
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "chi_loss": chi_loss_dpo.to(dtype=torch.float32),
        "delta_loss": delta_loss_dpo.to(dtype=torch.float32),
        "grad": flat_grad_cpu,
        "n_pairs": n_pairs,
        "new_ref_cache": new_ref_cache if ref_cache is None else None,
    }


def chi_net_for_batch(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    batch: dict[str, torch.Tensor],
    *,
    n_hutchinson: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """chi_net (``Tr(J J^T)``, unnormalized) over a stacked ``(N, S)`` batch."""
    estimator = HutchinsonEstimator(n_hutchinson=n_hutchinson)
    result = estimator.compute(
        model,
        batch,
        causal_lm_forward,
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        params=params,
        micro_batch_size=1,
        generator=generator,
    )
    return result.chi_net, result.n_valid_tokens


def compute_domain_observables(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    pairs: list[TokenizedPair],
    *,
    device: torch.device,
    beta: float,
    n_hutchinson: int,
    seed: int,
    ref_cache: list[tuple[float, float]] | None,
) -> dict[str, Any]:
    ce = ce_pass(model, params, pairs, device)
    dpo = dpo_pass(model, params, pairs, device, beta=beta, ref_cache=ref_cache)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    chosen_batch = stack_lm_batch([p["chosen"] for p in pairs])
    chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
    chi_net_ce, n_valid_ce = chi_net_for_batch(
        model, params, chosen_batch, n_hutchinson=n_hutchinson, generator=gen
    )

    rejected_batch = stack_lm_batch([p["rejected"] for p in pairs])
    rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}
    dpo_batch = {k: torch.cat([chosen_batch[k], rejected_batch[k]], dim=0) for k in chosen_batch}
    chi_net_dpo, n_valid_dpo = chi_net_for_batch(
        model, params, dpo_batch, n_hutchinson=n_hutchinson, generator=gen
    )

    chi_pos_ce = float(chi_pos_fn(ce["delta_loss"], ce["chi_loss"], chi_net_ce))
    chi_pos_dpo = float(chi_pos_fn(dpo["delta_loss"], dpo["chi_loss"], chi_net_dpo))
    r_spec = chi_pos_dpo / chi_pos_ce if chi_pos_ce != 0.0 else float("nan")

    g_ce = ce["grad"].to(dtype=torch.float64)
    g_dpo = dpo["grad"].to(dtype=torch.float64)
    dot = float(_dot_fp64(g_ce, g_dpo))
    norm_ce = float(_sum_sq_fp64(g_ce)) ** 0.5
    norm_dpo = float(_sum_sq_fp64(g_dpo)) ** 0.5
    r_dir = dot / (norm_ce * norm_dpo) if norm_ce > 0 and norm_dpo > 0 else float("nan")

    return {
        "n_pairs": len(pairs),
        "n_valid_chosen_tokens": ce["n_valid"],
        "chi_loss_ce": float(ce["chi_loss"]),
        "delta_loss_ce": float(ce["delta_loss"]),
        "chi_net_ce": float(chi_net_ce),
        "n_valid_chi_net_ce": n_valid_ce,
        "chi_pos_ce": chi_pos_ce,
        "chi_loss_dpo": float(dpo["chi_loss"]),
        "delta_loss_dpo": float(dpo["delta_loss"]),
        "chi_net_dpo": float(chi_net_dpo),
        "n_valid_chi_net_dpo": n_valid_dpo,
        "chi_pos_dpo": chi_pos_dpo,
        "r_spec": r_spec,
        "r_dir": r_dir,
        "_new_ref_cache": dpo["new_ref_cache"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(SAMPLES_PATH))
    parser.add_argument(
        "--seq-len", type=int, default=None, help="override the seq_len recorded in samples.json"
    )
    parser.add_argument("--beta", type=float, default=0.1, help="DPO loss temperature.")
    parser.add_argument("--n-hutchinson", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler-interval-s", type=float, default=0.25)
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="optional attention-backend passthrough to from_pretrained.",
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
    sft_model = meta["sft_model"]
    dpo_model = meta["dpo_model"]
    seq_len = args.seq_len or meta["seq_len_target"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = "bf16"
    else:
        device = torch.device("cpu")
        dtype = "fp32"
        print("WARNING: no CUDA device — a 7B model will not realistically run on CPU.")

    print(f"dpo_spectral_filter diagnostics — sft={sft_model}  dpo={dpo_model}")
    print(f"  device={device}, dtype={dtype}, seq_len={seq_len}, beta={args.beta}")
    print(f"  n_hutchinson={args.n_hutchinson}, attn={args.attn_implementation}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(sft_model)
    tokenized_pairs = {
        domain: tokenize_pairs(payload["pairs"][domain], tokenizer, seq_len) for domain in DOMAINS
    }
    for domain, pairs in tokenized_pairs.items():
        print(f"  {domain}: {len(pairs)} pairs")

    tracker = ResourceTracker(device=device, interval_s=args.sampler_interval_s)
    results: list[dict[str, Any]] = []
    ref_cache_by_domain: dict[str, list[tuple[float, float]]] = {}

    with tracker:
        for revision, model_name in ((SFT_REVISION, sft_model), (DPO_REVISION, dpo_model)):
            with tracker.measure(revision):
                t0 = time.perf_counter()
                bundle = load_hf_model(
                    model_name,
                    dtype=dtype,
                    device=device,
                    attn_implementation=args.attn_implementation,
                )
                print(f"  loaded {model_name} in {time.perf_counter() - t0:.1f}s")

                for domain in DOMAINS:
                    ref_cache = (
                        ref_cache_by_domain.get(domain) if revision == DPO_REVISION else None
                    )
                    out = compute_domain_observables(
                        bundle.model,
                        bundle.params,
                        tokenized_pairs[domain],
                        device=device,
                        beta=args.beta,
                        n_hutchinson=args.n_hutchinson,
                        seed=args.seed,
                        ref_cache=ref_cache,
                    )
                    new_cache = out.pop("_new_ref_cache")
                    if revision == SFT_REVISION:
                        ref_cache_by_domain[domain] = new_cache
                    results.append(
                        {"checkpoint": model_name, "revision": revision, "domain": domain, **out}
                    )
                    print(
                        f"  [{revision}] {domain}: R_dir={out['r_dir']:+.4f} "
                        f"R_spec={out['r_spec']:.4f} "
                        f"chi_pos_ce={out['chi_pos_ce']:.4e} chi_pos_dpo={out['chi_pos_dpo']:.4e}"
                    )

                del bundle
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    RESULTS_PATH.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    tracker.save(str(RESOURCES_PATH))
    print(f"\nwrote {RESULTS_PATH}")
    print(f"wrote {RESOURCES_PATH}")
    print("\n=== resources ===")
    print(tracker.format_summary())


if __name__ == "__main__":
    main()
