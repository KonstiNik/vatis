"""The default chi_net estimator, at the places a run actually picks it.

``select_chi_net_method`` has its own test in ``test_chi_net.py``. That test
alone does not notice an analyzer that applies a rule of its own, or a CLI whose
option default names an estimator. These tests pin the default where a user
reaches it: ``analyze()`` and the ``Analyzer`` class without ``chi_net_method``,
the ``vatis run`` parser, and what ``vatis run`` hands to the analyzer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tests.fixtures.tiny_transformer import TinyTransformer, make_tiny_lm_batch
from vatis import analyze, cli
from vatis.analyzer import Analyzer
from vatis.cli import build_parser
from vatis.models.hf import ModelBundle, causal_lm_loss, causal_lm_valid_mask


def _rows(
    batch_size: int, micro_batch_size: int, method: str | None, *, through_class: bool = False
) -> dict[str, float]:
    torch.manual_seed(0)
    model = TinyTransformer().eval()
    bundle = ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )
    batch = make_tiny_lm_batch(batch_size=batch_size, seed=0, pad_fraction=0.25, device="cpu")
    kwargs: dict[str, Any] = {} if method is None else {"chi_net_method": method}
    if through_class:
        # analyze() always hands chi_net_method on, so the class default is only
        # reached by constructing Analyzer directly.
        analyzer = Analyzer(
            eval_batches={"val": batch},
            n_hutchinson=4,
            micro_batch_size=micro_batch_size,
            sink=None,
            device="cpu",
            preconditioner="identity",
            **kwargs,
        )
        results = analyzer.run([(bundle, "step0")])
    else:
        results = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": batch},
            n_hutchinson=4,
            micro_batch_size=micro_batch_size,
            sink=None,
            device="cpu",
            preconditioner="identity",
            **kwargs,
        )
    return {row.observable: row.value for row in results[0].rows}


@pytest.mark.parametrize(
    ("batch_size", "micro_batch_size"), [(1, 1), (4, 1), (8, 2), (32, 4), (33, 4)]
)
def test_analyze_without_a_method_is_hutchinson_bit_for_bit(
    batch_size: int, micro_batch_size: int
) -> None:
    default = _rows(batch_size, micro_batch_size, None)
    hutchinson = _rows(batch_size, micro_batch_size, "hutchinson")
    per_sequence_cv = _rows(batch_size, micro_batch_size, "per_sequence_cv")
    assert default == hutchinson
    # the comparison above would be vacuous if both estimators returned the same number
    assert default["chi_net"] != per_sequence_cv["chi_net"]
    assert default["chi_net_P"] != per_sequence_cv["chi_net_P"]


@pytest.mark.parametrize(
    ("batch_size", "micro_batch_size"), [(1, 1), (4, 1), (8, 2), (32, 4), (33, 4)]
)
def test_the_analyzer_class_without_a_method_is_hutchinson_bit_for_bit(
    batch_size: int, micro_batch_size: int
) -> None:
    default = _rows(batch_size, micro_batch_size, None, through_class=True)
    hutchinson = _rows(batch_size, micro_batch_size, "hutchinson", through_class=True)
    per_sequence_cv = _rows(batch_size, micro_batch_size, "per_sequence_cv", through_class=True)
    assert default == hutchinson
    assert default["chi_net"] != per_sequence_cv["chi_net"]


def test_vatis_run_hands_no_estimator_to_the_analyzer_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the parser leaves unset must reach the analyzer unset.

    ``run_from_args`` serves the CLI and the DDP worker; a substitution there
    (``args.chi_net_method or "..."``) is invisible to a test of the parser.
    """
    batch_path = tmp_path / "batch.pt"
    torch.save({"input_ids": torch.zeros((2, 4), dtype=torch.long)}, batch_path)
    seen: dict[str, Any] = {}

    class _Spy:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def run(self, specs: Any) -> list[Any]:
            return []

    monkeypatch.setattr(cli, "Analyzer", _Spy)
    argv = ["run", "--model", "m", "--eval-batch", f"v={batch_path}", "--sink", "out.parquet"]
    assert cli.main(argv) == 0
    assert "chi_net_method" in seen
    assert seen["chi_net_method"] is None

    seen.clear()
    assert cli.main([*argv, "--chi-net-method", "per_sequence_cv"]) == 0
    assert seen["chi_net_method"] == "per_sequence_cv"


def test_the_cli_leaves_the_estimator_unset_by_default() -> None:
    args = build_parser().parse_args(
        ["run", "--model", "m", "--eval-batch", "v=x.pt", "--sink", "out.parquet"]
    )
    assert args.chi_net_method is None
