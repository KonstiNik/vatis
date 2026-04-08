"""Unit tests for vatis.core.probes."""

import torch

from vatis.core.probes import make_probe, mask_probe


def test_rademacher_probe_has_unit_variance_in_expectation() -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    probe = make_probe((8, 16, 64), distribution="rademacher", generator=g)
    # Each entry is exactly ±1.
    assert torch.all((probe == 1.0) | (probe == -1.0))
    # Mean over many entries should be near 0.
    assert abs(float(probe.mean())) < 0.05


def test_gaussian_probe_has_unit_variance_in_expectation() -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    probe = make_probe((4, 1024), distribution="gaussian", generator=g)
    assert abs(float(probe.mean())) < 0.1
    assert abs(float(probe.var()) - 1.0) < 0.2


def test_unknown_probe_distribution_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_probe((2, 2), distribution="bogus")  # type: ignore[arg-type]


def test_mask_probe_zeros_invalid_positions_lm_shape() -> None:
    # probe shape (M, S, V) and mask shape (M, S).
    probe = torch.ones(2, 3, 4)
    valid = torch.tensor([[True, True, False], [False, True, True]])
    out = mask_probe(probe, valid)
    expected = torch.tensor(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(out, expected)


def test_mask_probe_passthrough_when_mask_is_none() -> None:
    probe = torch.randn(2, 3, 4)
    out = mask_probe(probe, None)
    assert torch.equal(out, probe)


def test_mask_probe_handles_classifier_shape() -> None:
    # probe shape (B, V) and mask shape (B,).
    probe = torch.ones(3, 4)
    valid = torch.tensor([True, False, True])
    out = mask_probe(probe, valid)
    expected = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_make_probe_reproducible_with_generator() -> None:
    g1 = torch.Generator(device="cpu")
    g1.manual_seed(42)
    p1 = make_probe((2, 3, 4), generator=g1)
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(42)
    p2 = make_probe((2, 3, 4), generator=g2)
    assert torch.equal(p1, p2)
