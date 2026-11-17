"""The default chi_net estimator, at the places a run actually picks it.

``select_chi_net_method`` has its own test in ``test_chi_net.py``. That test
alone does not notice an analyzer that applies a rule of its own, or a CLI whose
option default names an estimator. These tests pin the default where a user
reaches it: ``analyze()`` without ``chi_net_method``, and the ``vatis run``
parser.
"""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.tiny_transformer import TinyTransformer, make_tiny_lm_batch
from vatis import analyze
from vatis.cli import build_parser
from vatis.models.hf import ModelBundle, causal_lm_loss, causal_lm_valid_mask


def _rows(batch_size: int, micro_batch_size: int, method: str | None) -> dict[str, float]:
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
    kwargs = {} if method is None else {"chi_net_method": method}
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


def test_the_cli_leaves_the_estimator_unset_by_default() -> None:
    args = build_parser().parse_args(
        ["run", "--model", "m", "--eval-batch", "v=x.pt", "--sink", "out.parquet"]
    )
    assert args.chi_net_method is None
