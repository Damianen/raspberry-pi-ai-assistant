"""Filesystem locations for runtime artifacts: data/ (state) and models/ (downloads)."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV_VAR = "ASSISTANT_DATA_DIR"
MODELS_DIR_ENV_VAR = "ASSISTANT_MODELS_DIR"


def _resolve(env_var: str, dirname: str) -> Path:
    """Env override, repo checkout (src layout), then cwd; created on access."""
    override = os.environ.get(env_var)
    if override:
        path = Path(override)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        base = repo_root if (repo_root / "pyproject.toml").is_file() else Path.cwd()
        path = base / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    return _resolve(DATA_DIR_ENV_VAR, "data")


def models_dir() -> Path:
    return _resolve(MODELS_DIR_ENV_VAR, "models")
