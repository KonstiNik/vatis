"""``examples/_plots.py``: the common diagnostic figure of the examples, on a tiny table.

The helper needs matplotlib, which is the ``examples`` extra and not a
dependency of vatis, so the tests skip where it is not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PLOTS_PATH = Path(__file__).resolve().parents[2] / "examples" / "_plots.py"


@pytest.fixture(scope="module")
def plots() -> ModuleType:
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("_example_plots_under_test", PLOTS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(steps: list[int]) -> list[dict[str, Any]]:
    """Two batches and their cross pair; the cross delta_loss changes sign."""
    rows: list[dict[str, Any]] = []
    for step in steps:
        for a, b in (("x", "x"), ("y", "y"), ("x", "y")):
            for observable, value in (
                ("chi_loss", 1.0 / (step + 1)),
                ("delta_loss", (2.0 - step) if a != b else 3.0 + step),
                ("my_observable", 0.5),
            ):
                rows.append(
                    {
                        "revision": f"step{step}",
                        "batch_a": a,
                        "batch_b": b,
                        "observable": observable,
                        "value": value,
                    }
                )
    return rows


def test_step_of_reads_the_trailing_number(plots: ModuleType) -> None:
    assert plots.step_of("step0") == 0
    assert plots.step_of("bundle@step143000") == 143000
    with pytest.raises(ValueError):
        plots.step_of("main")


@pytest.mark.parametrize("steps", [[0, 1, 4, 9], [1, 8, 64]], ids=["with-step-0", "log-steps"])
def test_one_panel_per_observable_plus_the_loss(
    plots: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: list[int]
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    saved: list[Any] = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **k: saved.append(fig))
    out = plots.plot_observables_over_training(
        _rows(steps),
        {"x": {s: 3.0 - 0.1 * s for s in steps}, "train loss": {s: 2.0 for s in steps}},
        tmp_path / "overview.png",
        title="t",
        loss_label="loss (nats per token)",
        note="n",
    )
    assert out == tmp_path / "overview.png"
    (fig,) = saved
    visible = [ax for ax in fig.axes if ax.get_visible()]
    titles = [ax.get_title(loc="left") for ax in visible]
    # Loss first, then the known observables in their fixed order, then the rest.
    assert titles == ["loss", "chi_loss", "delta_loss", "my_observable"]
    scales = {ax.get_title(loc="left"): ax.get_yscale() for ax in visible}
    assert scales["chi_loss"] == "log"  # all positive
    assert scales["delta_loss"] == "symlog"  # the cross pair turns negative
    assert visible[0].get_xscale() == ("symlog" if steps[0] == 0 else "log")
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert labels == ["x (self pair)", "y (self pair)", "x × y (cross pair)"]
    plt.close(fig)


def test_more_batches_than_colours_is_refused(plots: ModuleType, tmp_path: Path) -> None:
    rows = [
        {"revision": "step1", "batch_a": n, "batch_b": n, "observable": "chi_loss", "value": 1.0}
        for n in "abcde"
    ]
    with pytest.raises(ValueError, match="colours"):
        plots.plot_observables_over_training(
            rows, {}, tmp_path / "x.png", title="t", loss_label="l"
        )
