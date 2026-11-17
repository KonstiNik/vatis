"""chi_net estimators: Hutchinson, per-sequence control variate, opacus stub."""

from vatis.core.chi_net.base import ChiNetEstimator, ChiNetResult
from vatis.core.chi_net.hutchinson import HutchinsonEstimator
from vatis.core.chi_net.opacus import OpacusEstimator
from vatis.core.chi_net.per_seq_cv import PerSequenceControlVariateEstimator

__all__ = [
    "ChiNetEstimator",
    "ChiNetResult",
    "HutchinsonEstimator",
    "OpacusEstimator",
    "PerSequenceControlVariateEstimator",
    "select_chi_net_method",
    "build_estimator",
]


def select_chi_net_method(b_total: int, requested: str | None) -> str:
    """Return the chi_net estimation method to use.

    An explicit request always wins; without one the method is ``"hutchinson"``
    at every batch size. Through 0.2.x the default for ``B <= 32`` was
    ``"per_sequence_cv"``; that estimator is now opt-in only (see
    ``docs/SPEC.md``, "Default estimator").

    Args:
        b_total: total batch size (sum across ranks if DDP). No longer affects
            the result; kept so existing callers keep working.
        requested: user-supplied method name, or None for the default.

    Returns:
        The selected method name. ``"opacus"`` and ``"per_sequence_cv"`` are
        returned only when requested.
    """
    del b_total
    if requested is not None:
        return requested
    return "hutchinson"


def build_estimator(method: str, **kwargs: object) -> ChiNetEstimator:
    """Construct an estimator by method name."""
    if method == "hutchinson":
        return HutchinsonEstimator(**kwargs)  # type: ignore[arg-type]
    if method == "per_sequence_cv":
        return PerSequenceControlVariateEstimator(**kwargs)  # type: ignore[arg-type]
    if method == "opacus":
        return OpacusEstimator(**kwargs)
    raise ValueError(
        f"unknown chi_net method: {method!r}. "
        "expected one of: 'hutchinson', 'per_sequence_cv', 'opacus'"
    )
