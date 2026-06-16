"""Pytest configuration: makes tests/fixtures importable as a normal package."""

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``@pytest.mark.gpu`` when no CUDA device exists.

    Keeps GPU-only tests from failing on CPU-only machines and CI runners
    (which have no GPU) while still running them locally on a GPU box. The
    import of torch is deferred into this hook so plain collection stays cheap.
    """
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
