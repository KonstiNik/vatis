"""Unit tests for vatis.core.probes."""

import pytest
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


def test_cuda_generator_is_refused_rather_than_silently_ignored() -> None:
    """Accepting one would reintroduce device-dependent probes."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    with pytest.raises(ValueError, match="CPU generator"):
        make_probe((2, 3), generator=g, device="cuda")


@pytest.mark.gpu
@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_same_seed_gives_bit_identical_probes_on_cpu_and_cuda(distribution: str) -> None:
    """A seed must identify one probe set, whatever device the probe lands on.

    ``torch.Generator("cpu")`` and ``torch.Generator("cuda")`` produce different
    sequences from the same seed, so before probes were drawn on the CPU a CPU
    reference run and a GPU production run of the same ``analyze()`` call agreed
    on ``chi_net`` only to within the Hutchinson noise floor (~4e-2 relative at
    ``n_h = 400``), while every deterministic observable agreed to ~5e-08.
    """
    g_cpu = torch.Generator()
    g_cpu.manual_seed(20260729)
    on_cpu = make_probe((3, 5, 7), distribution=distribution, generator=g_cpu)
    g_gpu = torch.Generator()
    g_gpu.manual_seed(20260729)
    on_gpu = make_probe((3, 5, 7), distribution=distribution, generator=g_gpu, device="cuda")
    assert on_gpu.device.type == "cuda"
    assert torch.equal(on_cpu, on_gpu.cpu())


@pytest.mark.gpu
def test_probe_lands_on_the_requested_device() -> None:
    probe = make_probe((2, 2), device="cuda")
    assert probe.device.type == "cuda"
    assert probe.dtype == torch.float32
