# Contributing to vatis

Thanks for your interest in vatis. This is a research tool; contributions that
improve correctness, performance, documentation, or test coverage are welcome.

## Development setup

vatis uses [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

If you run the example or benchmark scripts, point the HuggingFace cache at a
suitable location by exporting `HF_HOME`, or copy `.env.example` to `.env` and
edit it (see `run_config.py`). The installed package itself needs none of this.

## Checks before opening a PR

CI runs these against `uv sync --extra dev`; run them locally first:

```bash
ruff check vatis/ tests/
ruff format vatis/ tests/ --check
mypy vatis/
pytest -m "not integration and not cross_validation"
```

- **Lint/format:** `ruff` for both. Fix with `ruff check --fix` and
  `ruff format`.
- **Types:** `mypy --strict` on `vatis/`. Tests are not type-checked.
- **Tests:** the unit tier is CPU-only and runs anywhere. GPU-only tests are
  marked `@pytest.mark.gpu` and auto-skip without CUDA. The `integration` and
  `cross_validation` tiers are skipped in CI (the latter needs the unpublished
  `perspic`); run them locally where applicable.

## Conventions

- The math core (`vatis/core/`) takes plain tensors and `nn.Module` and knows
  nothing about HuggingFace, DDP, or sinks. Keep it that way — it's the only
  part that must be bulletproof.
- One concept per module; padding is handled only in
  `vatis/core/normalization.py`.
- If a change contradicts the spec, update [`docs/SPEC.md`](docs/SPEC.md) in the
  same PR.
- Keep examples runnable: after changing the analyzer or an estimator, re-run
  `examples/pythia_sweep.py` and sanity-check the plots.

## Math reference

The LNP decomposition is derived in [arXiv:2605.31244](https://arxiv.org/abs/2605.31244).
`docs/SPEC.md` documents how each observable is computed in code.
