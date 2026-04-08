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
    """Auto-select the chi_net estimation method based on batch size.

    Per CLAUDE.md, the rule is:
        - explicit user request always wins
        - per_sequence_cv for small batches (B <= 32)
        - hutchinson for everything else

    Args:
        b_total: total batch size (sum across ranks if DDP).
        requested: user-supplied method name, or None for auto.

    Returns:
        The selected method name. Never returns "opacus" from auto-selection.
    """
    if requested is not None:
        return requested
    if b_total <= 32:
        return "per_sequence_cv"
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
