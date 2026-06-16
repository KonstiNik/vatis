"""Repo-local run configuration for the example and benchmark scripts.

**Not part of the installed package.** This file lives at the repo root and is
imported only by scripts under ``examples/``. The installed ``vatis`` package
reads the standard HuggingFace environment variables directly (via
``transformers``) and needs none of this.

Single source of truth for machine-specific settings (currently just the
HuggingFace cache location). Resolution order for each value:

  1. a variable already set in the environment (e.g. ``HF_HOME`` exported in
     your shell or by an sbatch script) — always wins;
  2. a ``KEY=VALUE`` line in a gitignored ``.env`` file at the repo root;
  3. unset → the underlying tool falls back to its own default
     (``~/.cache/huggingface`` for HuggingFace).

So to point the examples at a different cache on a new machine you change
*one* thing — ``export HF_HOME=...`` in your shell, or one line in ``.env`` —
and never touch the individual scripts. Copy ``.env.example`` to ``.env`` to
get started.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _REPO_ROOT / ".env"


def load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from the repo-root ``.env`` into ``os.environ``.

    Existing environment variables are never overwritten, so an exported shell
    variable always takes precedence over the file. Lines that are blank,
    comments (``#``), or lack an ``=`` are ignored. Surrounding quotes on the
    value are stripped. No-op if ``.env`` does not exist.
    """
    if not _ENV_FILE.is_file():
        return
    for raw in _ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def configure_hf_cache() -> None:
    """Pin the HuggingFace cache environment variables from ``HF_HOME``.

    Call this **before** importing ``torch`` / ``transformers`` so the cache
    location is in place when those libraries initialize. If ``HF_HOME`` is not
    set (neither in the shell nor in ``.env``) this is a no-op and HuggingFace
    uses its default cache.
    """
    load_dotenv()
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", hf_home)
