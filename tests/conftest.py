"""Pytest configuration: makes tests/fixtures importable as a normal package."""

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``@pytest.mark.gpu`` without CUDA, and ``@pytest.mark.golden``
    unless it was explicitly asked for.

    - ``gpu``: keeps GPU-only tests from failing on CPU-only machines and CI
      runners while still running them locally on a GPU box. The import of torch
      is deferred into this hook so plain collection stays cheap.
    - ``golden``: the WP0 acceptance gate writes a whole checkpoint bundle (a
      ~100-step training run) and analyzes it three times, so it does not belong
      in the unit tier. Unlike ``integration`` / ``cross_validation`` — which
      callers deselect by passing ``-m "not integration and not
      cross_validation"`` — ``golden`` postdates those call sites, so it opts
      *itself* out: it runs only when the ``-m`` expression mentions ``golden``
      (e.g. ``pytest -m golden``). A skipped-with-reason entry stays visible in
      the unit tier's summary rather than silently vanishing.
    """
    import torch

    marker_expression = config.getoption("-m", default="") or ""
    if "golden" not in marker_expression:
        skip_golden = pytest.mark.skip(
            reason="WP0 golden acceptance gate: run explicitly with `-m golden`"
        )
        for item in items:
            if "golden" in item.keywords:
                item.add_marker(skip_golden)

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
