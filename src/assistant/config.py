"""Profile-based configuration: loads config/<profile>.yaml per ASSISTANT_PROFILE."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILE = "laptop"
PROFILE_ENV_VAR = "ASSISTANT_PROFILE"
CONFIG_DIR_ENV_VAR = "ASSISTANT_CONFIG_DIR"


@dataclass(frozen=True)
class DisplayConfig:
    width: int
    height: int
    fullscreen: bool


@dataclass(frozen=True)
class CameraConfig:
    index: int


@dataclass(frozen=True)
class Config:
    profile: str
    display: DisplayConfig
    camera: CameraConfig
    stt: dict[str, Any]
    tts: dict[str, Any]
    llm: dict[str, Any]


def config_dir() -> Path:
    """Resolve the config directory: env override, repo checkout, then cwd."""
    override = os.environ.get(CONFIG_DIR_ENV_VAR)
    if override:
        return Path(override)
    repo_config = Path(__file__).resolve().parents[2] / "config"
    if repo_config.is_dir():
        return repo_config
    return Path.cwd() / "config"


def load_config(profile: str | None = None) -> Config:
    profile = profile or os.environ.get(PROFILE_ENV_VAR) or DEFAULT_PROFILE
    path = config_dir() / f"{profile}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no config file for profile {profile!r}: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    display = raw["display"]
    camera = raw["camera"]
    return Config(
        profile=profile,
        display=DisplayConfig(
            width=int(display["width"]),
            height=int(display["height"]),
            fullscreen=bool(display["fullscreen"]),
        ),
        camera=CameraConfig(index=int(camera["index"])),
        stt=raw.get("stt") or {},
        tts=raw.get("tts") or {},
        llm=raw.get("llm") or {},
    )
