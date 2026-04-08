"""Unit tests for vatis.core.normalization."""

import torch

from vatis.core.normalization import (
    DEFAULT_IGNORE_INDEX,
    count_valid,
    normalization_factor,
    valid_token_mask,
)


def test_default_ignore_index_is_minus_100() -> None:
    assert DEFAULT_IGNORE_INDEX == -100


def test_valid_token_mask_no_attention_mask() -> None:
    targets = torch.tensor([[0, 1, -100], [-100, 2, 3]])
    mask = valid_token_mask(targets)
    expected = torch.tensor([[True, True, False], [False, True, True]])
    assert torch.equal(mask, expected)


def test_valid_token_mask_with_attention_mask() -> None:
    targets = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    mask = valid_token_mask(targets, attention_mask=attention_mask)
    expected = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.equal(mask, expected)


def test_valid_token_mask_combined() -> None:
    targets = torch.tensor([[0, -100, 2], [3, 4, -100]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    mask = valid_token_mask(targets, attention_mask=attention_mask)
    expected = torch.tensor([[True, False, False], [True, True, False]])
    assert torch.equal(mask, expected)


def test_count_valid_matches_sum() -> None:
    mask = torch.tensor([[True, False, True], [True, True, True]])
    assert count_valid(mask) == 5


def test_normalization_factor_self() -> None:
    f = normalization_factor(8, 8)
    assert abs(f - 8.0) < 1e-12


def test_normalization_factor_cross() -> None:
    f = normalization_factor(4, 16)
    assert abs(f - 8.0) < 1e-12  # sqrt(4 * 16)


def test_normalization_factor_rejects_zero() -> None:
    import pytest

    with pytest.raises(ValueError):
        normalization_factor(0, 5)
    with pytest.raises(ValueError):
        normalization_factor(5, 0)
