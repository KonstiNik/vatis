# vatis — publish cleanup plan

Working document for getting the repo ready to publish on GitHub
(`github.com/KonstiNik/vatis`). This file is build scaffolding — it lives on
`dev`, never on the clean public `main`.

## Decisions (locked)

- **Repo is currently private.** Going public after cleanup.
- **Clean slate on `main`:** a fresh orphan branch with a single
  `initial public release` commit. The current 30-commit history does **not**
  carry over to public `main`.
- **`dev` = continued-development branch, public is fine.** Branches off the
  current `main` so the full history is preserved, then its tip is cleaned up
  (see below). This is where active research continues.
- **`background_info.tex` is removed everywhere** — it is part of an
  already-published paper (arXiv:2605.31244) and was only a math reference for
  building the package. README/docs link the arXiv paper instead.
- **CI must auto-skip GPU tests** rather than fail them on the runner.

## Branch layout after cleanup

```
main  (orphan, 1 commit, PUBLIC)        dev  (full history, PUBLIC)
└─ initial public release               └─ …30 commits… → cleanup commit
   clean package + demos + docs            package + live research, scaffolding stripped
```

---

## `main` — clean public release (orphan, 1 commit)

### Keep
- `vatis/` — the package
- `tests/` — unit + gated integration/cross_validation tiers
- `pyproject.toml`, `uv.lock`
- `LICENSE`, `README.md`
- `.github/workflows/ci.yml`
- `.gitignore`
- `docs/SPEC.md` — durable math/architecture/API, extracted from `CLAUDE.md`,
  internal sections stripped, math derivation links arXiv:2605.31244
- `CONTRIBUTING.md`, `CHANGELOG.md` (new)
- `examples/` — **tool demos + benchmarks:**
  - `pythia_sweep.py`, `analyze_results.py`, `_helpers.py`, `BENCHMARK.md`
    (reframed as deployment-example findings & sizing), `pythia_sweep/` outputs
  - (`_probe.py` and `_benchmark.py` were deleted — the RTX-3090 timing/probe
    tables were stale and are superseded by `examples/benchmark/single_gpu/`)
  - `examples/benchmark/` — **scaling/validation results are public** (the
    `accuracy/`, `ddp_scaling/`, `single_gpu/` suites + `_build_eval_batch.py`
    + `eval_batch_b64_s512.pt`), after the scrub in edit #9 below. Excludes
    `__pycache__/` and `*.log` (gitignored).

### Do NOT include on `main`
- `background_info.tex` (published — link arXiv instead)
- `CLAUDE.md`, `SESSION_SUMMARY.md`, `TASKS_NEXT.md`, `launch_build.sh`,
  `CLEANUP_PLAN.md` — build scaffolding
- `.claude/settings.local.json` — leaks home path + permission config
- `examples/dpo_spectral_filter/`, `examples/spectral_tail/`,
  `examples/spectral_tail_experiment.py`, `examples/spectral_tail_evaluate.py`
  — unpublished / WIP research

### Edits applied before sealing the clean commit
1. **Scrub `/data/knikolaou/huggingface`** hardcoded HF-cache default from
   `examples/_benchmark.py`, `_probe.py`, `pythia_sweep.py`, `BENCHMARK.md`.
   Drop the hardcoded default; rely on HF's own `~/.cache/huggingface` or an
   env var the user already set.
   **✅ UNBLOCKED** (benchmarks finished). Apply now.
2. **`pyproject.toml`:** fix description `LNA → LNP`; set real `authors`
   (Konstantin Nikolaou); add `[project.urls]` (Homepage / Repository / Issues).
3. **Version + tag:** confirm `0.1.0` in `pyproject.toml` and
   `vatis/_version.py`; tag `v0.1.0` on the clean commit.
4. **`.gitignore`:** add `.claude/`, `build.log`, `BLOCKED.md`, `*.pt`,
   example run logs (`__pycache__/` already covered).
5. **CI GPU handling:** register a `gpu` marker in `pyproject.toml` and add a
   `pytest_collection_modifyitems` hook in `tests/conftest.py` that skips
   `@pytest.mark.gpu` tests when `torch.cuda.is_available()` is False. Fix the
   stale doc note that CI triggers on `master` (it triggers on `main`).
6. **`README.md`:** replace the `background_info.tex` reference with the arXiv
   link; add a note that the `cross_validation` test tier needs the
   unpublished sister package `perspic`.
7. **New:** `CONTRIBUTING.md`, `CHANGELOG.md`.
8. **`docs/SPEC.md`:** extract math/architecture/result-format/API/distributed
   sections from `CLAUDE.md`; drop the "Working conventions", "Permission
   model", "Unattended-mode", "v1 build history", "v1.1 hardening pass"
   sections and the `/tikhome/knikolaou/PycharmProjects/vatis` path.
9. **Scrub `examples/benchmark/` for public release:**
   - HF cache path `/data/horse/ws/koni010i-dpo_sft_transition/hf_cache` →
     drop hardcoded `_DEFAULT_HF_HOME` defaults + docstrings in
     `_build_eval_batch.py`, `accuracy/accuracy_sweep.py`,
     `single_gpu/single_gpu_sweep.py` (same treatment as edit #1).
   - `.sbatch` files (`accuracy/`, `ddp_scaling/`, `single_gpu/`): remove
     `--account=p_neurasearch` and `--partition=alpha` (or make them
     `# set for your cluster` placeholders); replace absolute
     `--output=` / `REPO=` / `HF_HOME=` paths with repo-relative /
     env-derived values.
   - Absolute paths in `DDP_SCALING_RESULTS.md` (and any committed `.md`
     results) → relative or removed.
   - Exclude `__pycache__/` and `*.log` (gitignored).
   - Decision: keep `eval_batch_b64_s512.pt` (901 KB, regenerable) committed
     for reproducibility — confirm or drop in favor of regenerate-on-run.

### Pre-flight gate (before sealing)
- `ruff check`, `ruff format --check`, `mypy --strict`, and
  `pytest -m "not integration and not cross_validation"` all green.
- All suite tests currently pin `device="cpu"`, so the CI tier is GPU-free;
  the `gpu` marker handles any GPU test added later.

---

## `dev` — continued development (full history preserved)

Branch off current `main` (keeps all 30 commits), then a single cleanup commit
at the tip that **removes the package-build scaffolding** and keeps the live
work.

### Remove from `dev` tip (build scaffolding — its job is done)
- `background_info.tex` (published)
- `SESSION_SUMMARY.md`, `TASKS_NEXT.md`, `launch_build.sh`
- `.claude/settings.local.json`
- (`CLAUDE.md` is **kept** on `dev` as ongoing repo context)

### Keep on `dev` (live dev work)
- `vatis/`, `tests/`, build config
- `examples/` in full, including the research:
  `dpo_spectral_filter/`, `spectral_tail/` (+ scripts)
- `examples/benchmark/` — commit it here (drop `__pycache__/`); it is the
  current diagnostic/benchmark work in progress
- `CLEANUP_PLAN.md` (this file)

> Note: removing scaffolding only cleans the `dev` *tip*. The files still exist
> in the preserved history (old commits). That is fine — `dev` is public and
> the content is acceptable to be public.

---

## Resolved

1. **`CLAUDE.md` stays on `dev`** (kept off `main`). Removed from the
   `dev`-tip scaffolding-strip list below.
2. **`docs/SPEC.md`** = architecture + API + result-format only; links
   arXiv:2605.31244 for the math derivation.

---

## Execution order (once confirmed)

1. Safe local edits on a scratch state (no git ops): path scrub, pyproject,
   gitignore, CI/GPU, README, CONTRIBUTING, CHANGELOG, `docs/SPEC.md`.
2. Run the pre-flight gate; get it green.
3. `git branch dev` (preserve current state) → cleanup commit on `dev` →
   push `dev`.
4. `git checkout --orphan` fresh root → stage only the `main` keep-list →
   `initial public release` commit → tag `v0.1.0`.
5. Force-push the new `main` to the remote; verify CI green.
6. Flip the repo to public.

**Irreversible / outward-facing steps (3–6) wait for explicit go-ahead.**
