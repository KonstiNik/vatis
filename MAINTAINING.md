# Maintaining vatis — branch model & release workflow

This file is **dev-only** — it documents the maintainer process and must never
be promoted to `main`. (Public contributor guidance lives in `CONTRIBUTING.md`.)

## Two branches, two purposes

| branch | role | contents |
|---|---|---|
| **`dev`** | source of truth + working branch | full history, the package, tests, **all** examples incl. research (`spectral_tail/`, `dpo_spectral_filter/`), and maintainer docs (`CLAUDE.md`, `MAINTAINING.md`, `CLEANUP_PLAN.md`) |
| **`main`** | the published public release | a **curated subset** of `dev`: package, tests, public examples + A100 benchmarks, `docs/SPEC.md`, README/CONTRIBUTING/CHANGELOG/LICENSE, configs |

`main` was created as an **orphan** branch (a clean single-commit history with no
research and no build scaffolding). **`main` and `dev` therefore share no common
ancestor** — their histories are deliberately unrelated.

### The hard invariant

> **Never merge `dev` into `main` (or vice versa).** They have no merge base, so
> a merge treats every file as "added on both sides" → conflict soup.
>
> You may only merge a **`main`-descended** branch into `main`, and a
> **`dev`-descended** branch into `dev`. Move content *across* the two with
> cherry-pick or path-scoped checkout (content operations — ancestry-independent),
> never with merge.

## Working-directory layout (git worktree)

Because the two branches have different file sets, switching between them in one
directory leaves stale untracked files and can abort checkouts. We keep each
branch in its own worktree (shared `.git`):

```
.../vatis        → dev   (primary working dir — develop here)
.../vatis-main   → main  (release/promotion only — never develop here)
```

Set up (one-time): `git worktree add ../vatis-main main`
List / remove: `git worktree list` · `git worktree remove ../vatis-main`

## Day-to-day development (on `dev`)

1. Branch off `dev`: `git checkout -b feat/<x> dev`.
2. **Keep public changes and research/dev-only changes in separate commits** —
   this is what makes promotion to `main` clean. A commit that touches both
   `vatis/...` and `examples/spectral_tail/...` is painful to promote.
3. Open a PR into `dev`. CI runs (it triggers on push to any branch). Merge.

## Promoting public changes to `main`

Work in the `main` worktree so `dev` is never disturbed:

```bash
cd ../vatis-main
git checkout -b promote/<x> main
```

Then bring the content over — **two options**:

- **Cherry-pick** (preferred when the `dev` commits are public-only — preserves
  message/authorship):
  ```bash
  git cherry-pick <dev-commit> [<dev-commit> ...]
  ```
  For a commit that *mixes* public + dev-only paths, stage then drop the
  dev-only parts:
  ```bash
  git cherry-pick -n <commit>
  git restore --staged --worktree \
      CLAUDE.md MAINTAINING.md CLEANUP_PLAN.md \
      examples/spectral_tail examples/spectral_tail_experiment.py \
      examples/spectral_tail_evaluate.py examples/dpo_spectral_filter
  git commit
  ```

- **Path-scoped checkout** (preferred when `dev` history is messy — copies the
  final file state, no commit replay):
  ```bash
  git checkout dev -- vatis/ tests/ docs/SPEC.md examples/<public paths> ...
  git commit -m "..."
  ```

Then push and (optionally) open a PR into `main` for review + CI:
```bash
git push -u origin promote/<x>
# review, then:
git checkout main && git merge promote/<x> && git push origin main
```

### Never promote (dev-only)
`CLAUDE.md`, `MAINTAINING.md`, `CLEANUP_PLAN.md`, `examples/spectral_tail*`,
`examples/dpo_spectral_filter/`, and anything else research/scaffolding. When in
doubt, check what `main` currently tracks: `git ls-tree -r --name-only main`.

### The public surface (what `main` should contain)
`vatis/`, `tests/`, `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`,
`.github/`, `.gitignore`, `docs/SPEC.md`, `CONTRIBUTING.md`, `CHANGELOG.md`,
`.env.example`, `run_config.py`, and `examples/` minus the research dirs
(`_helpers.py`, `pythia_sweep.py`, `analyze_results.py`, `BENCHMARK.md`,
`pythia_sweep/`, `benchmark/`).

## Cutting a release

1. On `dev`, bump the version in **both** `pyproject.toml` and
   `vatis/_version.py`, and add a `CHANGELOG.md` section.
2. Promote the release-ready content to `main` (above).
3. Tag on `main` and push:
   ```bash
   cd ../vatis-main
   git tag -a vX.Y.Z -m "vatis vX.Y.Z"
   git push origin main --tags
   ```
4. **Public-release commits carry no AI co-authorship trailer** (fine on `dev`).
5. Confirm GitHub Actions is green on `main`.

## Config & secrets

Machine-specific settings live in a **gitignored** `.env` at the repo root (see
`run_config.py` / `.env.example`): `HF_HOME` for the model cache, and
`SBATCH_PARTITION` / `SBATCH_ACCOUNT` for benchmark submits via
`examples/benchmark/submit.sh`. Never commit `.env`.

## Long-term note

Maintaining two divergent histories in one repo is workable but requires the
discipline above. If promotion ever becomes a chore — or once external
contributors open PRs against the public `main` — consider splitting into two
repos: a public repo with a normal `main`-based flow, and a separate **private**
repo for the research. That removes the divergent-history dance entirely.
