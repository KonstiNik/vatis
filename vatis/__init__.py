"""vatis — compute LNP observables on pretrained model checkpoints.

Public API:
    analyze: one-shot convenience function.
    Analyzer: class for users who want to drive the loop themselves.

See CLAUDE.md at the project root for the complete spec.
"""

from vatis._version import __version__

# We import the heavy public surface lazily to keep `import vatis` cheap
# and to avoid import-time cycles with the optional submodules. Tests and
# users who want the convenience API should `from vatis import analyze` or
# `from vatis import Analyzer` directly — both go through __getattr__ below.

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Analyzer": ("vatis.analyzer", "Analyzer"),
    "analyze": ("vatis.analyzer", "analyze"),
    "EvalBatchSpec": ("vatis.data.batches", "EvalBatchSpec"),
    "load_local_bundle": ("vatis.models.local_bundle", "load_local_bundle"),
    "Preconditioner": ("vatis.core.preconditioner", "Preconditioner"),
}

__all__ = [
    "Analyzer",
    "EvalBatchSpec",
    "Preconditioner",
    "__version__",
    "analyze",
    "load_local_bundle",
]


def __getattr__(name: str) -> object:  # pragma: no cover - thin shim
    import importlib

    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'vatis' has no attribute {name!r}")
