"""The first-order loss response to the update an optimizer actually applied.

``delta_loss_P`` (:mod:`vatis.core.preconditioner`) answers a question about
geometry: how does the loss on ``B`` respond to the *hypothetical* update
``-eta * P_t grad L^A``. This module answers the other question: how does it
respond, to first order, to the update the optimizer **took**,

    loss_change_first_order(A, B) = grad L^B . dtheta,        dtheta = theta_{t+1} - theta_t,

with ``dtheta`` rebuilt from the optimizer state saved *before* the step
(``docs/SPEC.md`` §"The response to the update that was taken"). The update is a
sum of three parts, and so is the response:

    grad   the part carried by the step's own gradient ``grad L^A``
    hist   the part carried by the gradient history (the first moment / the
           momentum buffer as it was before the step)
    wd     the part carried by weight decay

For ``torch.optim.AdamW`` at a checkpoint that has already applied ``t``
updates (``m_t``, ``v_t`` saved, ``g = grad L^A`` as the optimizer saw it)::

    v_{t+1} = beta2 * v_t + (1 - beta2) * g**2
    P_{t+1} = 1 / (sqrt(v_{t+1} / (1 - beta2**(t+1))) + eps)
    grad = -lr * (1 - beta1) / (1 - beta1**(t+1)) * P_{t+1} * g
    hist = -lr *      beta1  / (1 - beta1**(t+1)) * P_{t+1} * m_t
    wd   = -lr * weight_decay * theta_t

``P_{t+1}`` is the preconditioner the step really used: it already contains the
step's own gradient, unlike the pre-step ``P_t`` of the ``adamw`` P source. The
other covered update rules are spelled out on :class:`AdamHyper` (Adam with
coupled L2 weight decay) and :class:`SGDHyper` (SGD with momentum, dampening,
Nesterov and coupled weight decay).

Everything here is per parameter: each parameter carries the hyper-parameters
of its own optimizer param group, so learning rates and weight decays that
differ between groups (no decay on biases and norms is common) are exact, not
approximated by one value. That is also why this is not a
:class:`~vatis.core.preconditioner.Preconditioner`: the response is a dot
product with an update vector, which exists for any per-parameter rule, while a
single diagonal ``P`` without the learning rate does not.

A parameter the optimizer was handed **no gradient** for (``.grad is None``:
it is not in the graph of the update batch) is a special case: torch skips it
altogether — no momentum, no weight decay — whereas a parameter holding a zero
gradient gets both. The flat gradient vector cannot tell the two apart, so the
caller says which parameters received none (``no_gradient``), and the rule
refuses when its update for such a parameter would not be zero.

What this module cannot know is whether ``g`` really is the gradient the
optimizer saw. It takes the gradient it is handed, applies global-norm clipping
if told to (:attr:`OptimizerUpdate.clip_grad_norm`), and that is all; the
caller (:meth:`vatis.models.local_bundle.LocalBundleSource.update_response`)
documents what a bundle has to guarantee.

Pure math on tensors: no HF, no DDP, no sinks, no files.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

# Chunk size of the fp64 reductions, as in :mod:`vatis.core.preconditioner`.
_CHUNK = 1 << 25

# ``torch.nn.utils.clip_grad_norm_`` divides by ``total_norm + 1e-6``.
_CLIP_NORM_EPS = 1e-6

ADAM_FAMILY = "adam"
SGD_FAMILY = "sgd"


@dataclass(frozen=True)
class AdamHyper:
    """Hyper-parameters one parameter was updated with by ``torch.optim.Adam`` / ``AdamW``.

    With ``g`` the (possibly clipped) gradient of the update batch, ``m``, ``v``
    the saved moments and ``t`` the number of updates already applied::

        decoupled_weight_decay = True   (AdamW)
            g_eff = g
            wd    = -lr * weight_decay * theta
        decoupled_weight_decay = False  (Adam, L2 penalty added to the gradient)
            g_eff = g + weight_decay * theta
            wd    = -lr * (1 - beta1) / (1 - beta1**(t+1)) * P_{t+1} * weight_decay * theta

        v_{t+1} = beta2 * v + (1 - beta2) * g_eff**2
        P_{t+1} = 1 / (sqrt(v_{t+1} / (1 - beta2**(t+1))) + eps)
        grad    = -lr * (1 - beta1) / (1 - beta1**(t+1)) * P_{t+1} * g
        hist    = -lr *      beta1  / (1 - beta1**(t+1)) * P_{t+1} * m

    In the coupled case the decay also enters ``P_{t+1}`` through ``g_eff``, so
    ``wd`` there is the part of the update that is *linear* in ``theta`` at the
    ``P_{t+1}`` the step used, not the whole effect of switching the decay on.
    """

    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float = 0.0
    decoupled_weight_decay: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {self.eps}")
        _check_finite("lr", self.lr)
        _check_finite("weight_decay", self.weight_decay)


@dataclass(frozen=True)
class SGDHyper:
    """Hyper-parameters one parameter was updated with by ``torch.optim.SGD``.

    Weight decay is coupled (``g_eff = g + weight_decay * theta``). With ``b``
    the saved momentum buffer::

        momentum = 0                    d = g_eff
        first step with momentum        b' = g_eff                       (no dampening)
        later steps                     b' = momentum * b + (1 - dampening) * g_eff
        nesterov                        d = g_eff + momentum * b'
        otherwise                       d = b'
        dtheta = -lr * d

    which is linear in ``(g, b, theta)``; the three coefficients are the three
    parts. "First step" means the parameter has no momentum buffer yet.
    """

    lr: float
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    def __post_init__(self) -> None:
        for name in ("lr", "momentum", "dampening", "weight_decay"):
            _check_finite(name, getattr(self, name))

    def coefficients(self, *, has_buffer: bool) -> tuple[float, float]:
        """``(c_g, c_b)`` with ``dtheta = -lr * (c_g * g_eff + c_b * b)``."""
        if self.momentum == 0.0:
            return 1.0, 0.0
        if has_buffer:
            on_grad, on_buffer = 1.0 - self.dampening, self.momentum
        else:
            on_grad, on_buffer = 1.0, 0.0
        if self.nesterov:
            return 1.0 + self.momentum * on_grad, self.momentum * on_buffer
        return on_grad, on_buffer


@dataclass(frozen=True)
class UpdateResponseTerms:
    """``grad L^B . dtheta``, split into the three parts of ``dtheta``.

    Loss units, with the sign of a loss change: negative means the update
    lowers the loss on ``B`` to first order.
    """

    grad: float
    hist: float
    wd: float

    @property
    def total(self) -> float:
        """``grad + hist + wd`` — the first-order loss change under the whole update."""
        return self.grad + self.hist + self.wd


@dataclass(frozen=True, eq=False)
class OptimizerUpdate:
    """The update one optimizer step applied, as a rule that can be evaluated.

    Attributes:
        family: :data:`ADAM_FAMILY` or :data:`SGD_FAMILY`.
        hypers: one :class:`AdamHyper` / :class:`SGDHyper` per parameter, in the
            canonical parameter order (the entry of that parameter's group).
        step: number of updates already applied to ``theta_t`` (the pre-step
            counter ``t``); the step being described is update number ``t + 1``
            and its bias corrections use that exponent. Unused for SGD.
        first_moment: per-parameter ``exp_avg`` (Adam) / ``momentum_buffer``
            (SGD) as saved before the step, or ``None`` when no update has been
            applied yet (then it is zero for Adam and "no buffer" for SGD). For
            SGD a single entry may be ``None``: a parameter without a buffer (its
            group has ``momentum = 0``, or this is its first step).
        second_moment: per-parameter ``exp_avg_sq`` saved before the step (Adam
            only), or ``None`` together with ``first_moment``.
        clip_grad_norm: if set, the gradient was clipped to this global L2 norm
            before the optimizer saw it, exactly as
            ``torch.nn.utils.clip_grad_norm_(params, clip_grad_norm)`` does:
            scaled by ``min(1, clip_grad_norm / (norm + 1e-6))``, the norm
            taken over the whole gradient.
        descriptor: what the result file's ``preconditioner`` column records
            for the rows computed from this rule (``"adamw_update"``, ...).
        update_batch: the batch the step consumed. The analyzer uses it to find
            which of its eval batches is the update batch.
        next_weights: returns the model at ``theta_{t+1}`` (a
            :class:`~vatis.models.hf.ModelBundle`) when the realized loss change
            can be measured, else ``None``.
    """

    family: str
    hypers: Sequence[AdamHyper] | Sequence[SGDHyper]
    step: int
    first_moment: Sequence[torch.Tensor | None] | None
    second_moment: Sequence[torch.Tensor] | None
    clip_grad_norm: float | None = None
    descriptor: str = "update"
    update_batch: Any = None
    next_weights: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        if self.family not in (ADAM_FAMILY, SGD_FAMILY):
            raise ValueError(f"family must be 'adam' or 'sgd', got {self.family!r}")
        if not self.hypers:
            raise ValueError("an OptimizerUpdate needs at least one parameter")
        expected: type = AdamHyper if self.family == ADAM_FAMILY else SGDHyper
        if not all(isinstance(h, expected) for h in self.hypers):
            raise TypeError(f"every entry of hypers must be a {expected.__name__}")
        if int(self.step) < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        for name, moment in (
            ("first_moment", self.first_moment),
            ("second_moment", self.second_moment),
        ):
            if moment is not None and len(moment) != len(self.hypers):
                raise ValueError(
                    f"{name} has {len(moment)} entries for {len(self.hypers)} parameters"
                )
        if self.family == ADAM_FAMILY:
            if (self.first_moment is None) != (self.second_moment is None):
                raise ValueError("Adam needs both moments or neither (no update applied yet)")
            if self.first_moment is None and int(self.step) != 0:
                raise ValueError(
                    f"step is {self.step} but no moments were given; an Adam state "
                    "without moments is the state before the first update (step 0)"
                )
            if self.first_moment is not None and any(m is None for m in self.first_moment):
                raise ValueError("Adam needs a first moment for every parameter")
        elif self.second_moment is not None:
            raise ValueError("SGD has no second moment")
        if self.clip_grad_norm is not None and not (
            math.isfinite(self.clip_grad_norm) and self.clip_grad_norm > 0.0
        ):
            raise ValueError(
                f"clip_grad_norm must be a finite positive number or None, got "
                f"{self.clip_grad_norm}"
            )
        if not isinstance(self.descriptor, str) or not self.descriptor:
            raise ValueError(f"descriptor must be a non-empty str, got {self.descriptor!r}")

    # ---------------------------------------------------------------- clipping

    def clip_coefficient(self, grad_update: torch.Tensor) -> float:
        """The factor the optimizer's gradient was scaled by before the step.

        ``1.0`` without clipping; otherwise ``min(1, clip / (||g|| + 1e-6))``
        with the L2 norm over the whole flat gradient, as
        ``torch.nn.utils.clip_grad_norm_`` computes it.
        """
        if self.clip_grad_norm is None:
            return 1.0
        norm = math.sqrt(float(_sum_sq_fp64(grad_update)))
        return min(1.0, float(self.clip_grad_norm) / (norm + _CLIP_NORM_EPS))

    def learning_rates(self) -> tuple[float, float]:
        """The smallest and the largest learning rate any parameter was updated with.

        Equal when every parameter group has the same rate, which is the only
        case in which "the learning rate of the step" is one number.
        """
        rates = [float(h.lr) for h in self.hypers]
        return min(rates), max(rates)

    # ------------------------------------------------------------------- parts

    def iter_parts(
        self,
        grad_update: torch.Tensor,
        params: Sequence[torch.Tensor],
        *,
        chunk: int = _CHUNK,
        no_gradient: Mapping[int, str] | None = None,
    ) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield the update, chunk by chunk, split into its three parts.

        Each item is ``(start, stop, d_grad, d_hist, d_wd)``: the slice of the
        flat parameter vector and the fp64 pieces of ``dtheta`` on it
        (``dtheta[start:stop] = d_grad + d_hist + d_wd``), on the device of
        ``grad_update``.

        Args:
            grad_update: flat ``grad L^A`` of the update batch in the canonical
                order, **before clipping** (clipping is applied here).
            params: ``theta_t``, one tensor per parameter in the same order.
            chunk: maximal number of elements per item; bounds the fp64
                temporaries.
            no_gradient: parameter index -> display name, for the parameters
                that received **no gradient at all** on the update batch (not a
                zero one: none). torch skips such a parameter entirely when its
                ``.grad`` is ``None`` and applies momentum and weight decay to
                it when it holds zeros; which of the two the run did is not in
                the saved state. The rule applies momentum and decay (it sees a
                zero gradient), so it is only right for such a parameter when
                that update is zero anyway — otherwise this raises.

        Raises:
            ValueError: a size mismatch, or a parameter named in ``no_gradient``
                whose rebuilt update is not zero.
        """
        if grad_update.ndim != 1:
            raise ValueError(
                f"grad_update must be a flat vector, got shape {tuple(grad_update.shape)}"
            )
        if len(params) != len(self.hypers):
            raise ValueError(
                f"the update rule covers {len(self.hypers)} parameters but {len(params)} "
                "were given; both must follow the canonical parameter order"
            )
        total = sum(int(p.numel()) for p in params)
        if total != grad_update.numel():
            raise ValueError(
                f"grad_update has {grad_update.numel()} elements but the parameters "
                f"flatten to {total}"
            )
        for name, moment in (
            ("first_moment", self.first_moment),
            ("second_moment", self.second_moment),
        ):
            if moment is None:
                continue
            for index, (state, p) in enumerate(zip(moment, params, strict=True)):
                if state is not None and state.numel() != p.numel():
                    raise ValueError(
                        f"{name}[{index}] has {state.numel()} elements, parameter {index} "
                        f"has {p.numel()}"
                    )
        device = grad_update.device
        clip = self.clip_coefficient(grad_update)
        offset = 0
        for index, p in enumerate(params):
            n = int(p.numel())
            theta_flat = p.detach().reshape(-1)
            first = None if self.first_moment is None else self.first_moment[index]
            first = None if first is None else first.reshape(-1)
            second = None if self.second_moment is None else self.second_moment[index].reshape(-1)
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                g = grad_update[offset + lo : offset + hi].to(dtype=torch.float64)
                if clip != 1.0:
                    g = g * clip
                theta = theta_flat[lo:hi].to(dtype=torch.float64, device=device)
                m = None if first is None else first[lo:hi].to(dtype=torch.float64, device=device)
                v = None if second is None else second[lo:hi].to(dtype=torch.float64, device=device)
                hyper = self.hypers[index]
                if isinstance(hyper, AdamHyper):
                    parts = _adam_parts(hyper, int(self.step), g, theta, m, v)
                else:
                    parts = _sgd_parts(hyper, g, theta, m)
                if no_gradient is not None and index in no_gradient:
                    moved = max(float(part.abs().max()) for part in parts)
                    if moved != 0.0:
                        raise ValueError(
                            f"parameter {index} ({no_gradient[index]}) receives no gradient on "
                            "the update batch (it is not in the graph), yet the saved state "
                            "gives it a non-zero update through momentum or weight decay "
                            f"(largest entry {moved:.3e}). torch skips a parameter whose "
                            ".grad is None altogether, and applies momentum and decay to one "
                            "whose .grad holds zeros (zero_grad(set_to_none=False)); the "
                            "bundle does not say which the run did, so the update of this "
                            "parameter cannot be rebuilt. Exclude it from the bundle's "
                            "parameter list."
                        )
                yield (offset + lo, offset + hi, *parts)
            offset += n

    def predicted_delta(
        self,
        grad_update: torch.Tensor,
        params: Sequence[torch.Tensor],
        *,
        no_gradient: Mapping[int, str] | None = None,
    ) -> torch.Tensor:
        """``theta_{t+1} - theta_t`` as this rule predicts it: a flat fp64 vector.

        Materializes one parameter-sized fp64 vector; meant for tests and
        diagnostics on small models. :meth:`response` never builds it.
        """
        out = torch.empty(grad_update.numel(), dtype=torch.float64, device=grad_update.device)
        for start, stop, d_grad, d_hist, d_wd in self.iter_parts(
            grad_update, params, no_gradient=no_gradient
        ):
            out[start:stop] = d_grad + d_hist + d_wd
        return out

    # ---------------------------------------------------------------- response

    def response(
        self,
        grad_update: torch.Tensor,
        grads_eval: Mapping[str, torch.Tensor],
        params: Sequence[torch.Tensor],
        *,
        no_gradient: Mapping[int, str] | None = None,
    ) -> dict[str, UpdateResponseTerms]:
        """``grad L^B . dtheta`` for every evaluation gradient, split in three.

        Args:
            grad_update: flat ``grad L^A`` of the update batch, before clipping.
            grads_eval: name -> flat ``grad L^B``. The update batch itself may be
                one of them (its entry is the response on the update batch). A
                gradient may live on another device than ``grad_update`` (a
                host-offloaded one): chunks are moved as needed.
            params: ``theta_t``, one tensor per parameter, canonical order.
            no_gradient: the parameters that received no gradient at all on the
                update batch; see :meth:`iter_parts`.

        Returns:
            name -> :class:`UpdateResponseTerms`, accumulated in fp64.

        Raises:
            ValueError: a size mismatch, a parameter without a gradient whose
                rebuilt update is not zero, or a non-finite result (for instance
                ``eps = 0`` on a coordinate whose second moment is zero, where
                the optimizer's own update is ``0 / 0``).
        """
        for name, g in grads_eval.items():
            if g.numel() != grad_update.numel():
                raise ValueError(
                    f"gradient {name!r} has {g.numel()} elements, the update gradient "
                    f"has {grad_update.numel()}"
                )
        device = grad_update.device
        sums = {name: torch.zeros(3, dtype=torch.float64, device=device) for name in grads_eval}
        for start, stop, d_grad, d_hist, d_wd in self.iter_parts(
            grad_update, params, no_gradient=no_gradient
        ):
            for name, g in grads_eval.items():
                ge = g[start:stop].to(dtype=torch.float64, device=device)
                acc = sums[name]
                acc[0] += (ge * d_grad).sum()
                acc[1] += (ge * d_hist).sum()
                acc[2] += (ge * d_wd).sum()
        out: dict[str, UpdateResponseTerms] = {}
        for name, acc in sums.items():
            grad, hist, wd = (float(x) for x in acc.tolist())
            if not all(math.isfinite(x) for x in (grad, hist, wd)):
                raise ValueError(
                    f"the update response on {name!r} is not finite (grad={grad}, "
                    f"hist={hist}, wd={wd}); the saved state does not describe a finite "
                    "update (eps = 0 with a zero second moment gives 0 / 0)"
                )
            out[name] = UpdateResponseTerms(grad=grad, hist=hist, wd=wd)
        return out


# ------------------------------------------------------------------ the rules


def _adam_parts(
    hyper: AdamHyper,
    step: int,
    g: torch.Tensor,
    theta: torch.Tensor,
    m: torch.Tensor | None,
    v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three parts of one Adam / AdamW update on a chunk (all fp64)."""
    # The step being described is update number ``step + 1``: both bias
    # corrections take that exponent, and the second moment is the one that
    # already absorbed this step's gradient.
    exponent = step + 1
    bias1 = 1.0 - hyper.beta1**exponent
    bias2 = 1.0 - hyper.beta2**exponent
    if not (bias1 > 0.0 and bias2 > 0.0):
        raise ValueError(
            f"Adam bias correction underflowed (beta1={hyper.beta1}, beta2={hyper.beta2}, "
            f"update number {exponent})"
        )
    decay = hyper.weight_decay * theta
    coupled = not hyper.decoupled_weight_decay and hyper.weight_decay != 0.0
    g_eff = g + decay if coupled else g
    second = (1.0 - hyper.beta2) * g_eff * g_eff
    if v is not None:
        second = second + hyper.beta2 * v
    precond = 1.0 / ((second / bias2).sqrt() + hyper.eps)
    scale = hyper.lr / bias1
    d_grad = (-scale * (1.0 - hyper.beta1)) * precond * g
    d_hist = torch.zeros_like(g) if m is None else (-scale * hyper.beta1) * precond * m
    d_wd = (-scale * (1.0 - hyper.beta1)) * precond * decay if coupled else (-hyper.lr) * decay
    return d_grad, d_hist, d_wd


def _sgd_parts(
    hyper: SGDHyper,
    g: torch.Tensor,
    theta: torch.Tensor,
    buffer: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three parts of one SGD update on a chunk (all fp64)."""
    on_grad, on_buffer = hyper.coefficients(has_buffer=buffer is not None)
    d_grad = (-hyper.lr * on_grad) * g
    d_hist = (
        torch.zeros_like(g)
        if buffer is None or on_buffer == 0.0
        else (-hyper.lr * on_buffer) * buffer
    )
    d_wd = (-hyper.lr * on_grad * hyper.weight_decay) * theta
    return d_grad, d_hist, d_wd


# -------------------------------------------------------------------- helpers


def weight_delta_dot_fp64(
    grad: torch.Tensor,
    params_after: Sequence[torch.Tensor],
    params_before: Sequence[torch.Tensor],
    *,
    chunk: int = _CHUNK,
) -> torch.Tensor:
    """``grad . (theta_after - theta_before)`` accumulated in fp64, chunked.

    The first-order loss change under the weight difference of two stored
    checkpoints — no optimizer formula involved. ``grad`` is flat in the
    canonical order; the two parameter lists follow the same order.
    """
    if len(params_after) != len(params_before):
        raise ValueError(
            f"the two checkpoints have {len(params_after)} and {len(params_before)} parameters"
        )
    total = torch.zeros((), dtype=torch.float64, device=grad.device)
    offset = 0
    for after, before in zip(params_after, params_before, strict=True):
        if tuple(after.shape) != tuple(before.shape):
            raise ValueError(
                f"parameter shapes differ between the two checkpoints: {tuple(after.shape)} "
                f"vs {tuple(before.shape)}"
            )
        n = int(before.numel())
        a_flat = after.detach().reshape(-1)
        b_flat = before.detach().reshape(-1)
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            delta = a_flat[lo:hi].to(dtype=torch.float64, device=grad.device) - b_flat[lo:hi].to(
                dtype=torch.float64, device=grad.device
            )
            total = total + (grad[offset + lo : offset + hi].to(dtype=torch.float64) * delta).sum()
        offset += n
    if offset != grad.numel():
        raise ValueError(f"grad has {grad.numel()} elements but the parameters flatten to {offset}")
    return total


def _sum_sq_fp64(flat: torch.Tensor, chunk: int = _CHUNK) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=flat.device)
    for i in range(0, flat.numel(), chunk):
        c = flat[i : i + chunk].to(dtype=torch.float64)
        total = total + (c * c).sum()
    return total


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")
