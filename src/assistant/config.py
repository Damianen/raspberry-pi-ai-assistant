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
    device: str | None  # V4L2 device path; takes precedence over index (Linux only)
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class PerceptionConfig:
    detect_every_n_frames: int
    detector_score_threshold: float
    detector_nms_threshold: float
    detector_top_k: int
    presence_frames: int
    absence_timeout_s: float
    gaze_throttle_ms: float
    gaze_mirror: bool
    match_threshold: float
    vote_samples: int
    reverify_interval_s: float
    show_debug: bool
    enroll_samples: int
    enroll_timeout_s: float
    enroll_min_gap_ms: float


@dataclass(frozen=True)
class BlinkConfig:
    min_interval_s: float
    max_interval_s: float
    close_open_ms: float
    double_chance: float
    double_gap_ms: float


@dataclass(frozen=True)
class GazeConfig:
    smoothing: float
    idle_after_s: float
    drift_amount: float
    drift_interval_s: float
    glance_min_s: float
    glance_max_s: float
    glance_margin: float


@dataclass(frozen=True)
class FaceConfig:
    transition_ms: float
    breathing_hz: float
    bounce_hz: float
    drift_hz: float
    debug_controls: bool
    debug_gaze_hz: float
    blink: BlinkConfig
    gaze: GazeConfig


@dataclass(frozen=True)
class Config:
    profile: str
    display: DisplayConfig
    camera: CameraConfig
    perception: PerceptionConfig
    face: FaceConfig
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


def _load_face(raw: dict[str, Any]) -> FaceConfig:
    blink = raw["blink"]
    gaze = raw["gaze"]
    return FaceConfig(
        transition_ms=float(raw["transition_ms"]),
        breathing_hz=float(raw["breathing_hz"]),
        bounce_hz=float(raw["bounce_hz"]),
        drift_hz=float(raw["drift_hz"]),
        debug_controls=bool(raw["debug_controls"]),
        debug_gaze_hz=float(raw["debug_gaze_hz"]),
        blink=BlinkConfig(
            min_interval_s=float(blink["min_interval_s"]),
            max_interval_s=float(blink["max_interval_s"]),
            close_open_ms=float(blink["close_open_ms"]),
            double_chance=float(blink["double_chance"]),
            double_gap_ms=float(blink["double_gap_ms"]),
        ),
        gaze=GazeConfig(
            smoothing=float(gaze["smoothing"]),
            idle_after_s=float(gaze["idle_after_s"]),
            drift_amount=float(gaze["drift_amount"]),
            drift_interval_s=float(gaze["drift_interval_s"]),
            glance_min_s=float(gaze["glance_min_s"]),
            glance_max_s=float(gaze["glance_max_s"]),
            glance_margin=float(gaze["glance_margin"]),
        ),
    )


def _load_perception(raw: dict[str, Any]) -> PerceptionConfig:
    return PerceptionConfig(
        detect_every_n_frames=int(raw["detect_every_n_frames"]),
        detector_score_threshold=float(raw["detector_score_threshold"]),
        detector_nms_threshold=float(raw["detector_nms_threshold"]),
        detector_top_k=int(raw["detector_top_k"]),
        presence_frames=int(raw["presence_frames"]),
        absence_timeout_s=float(raw["absence_timeout_s"]),
        gaze_throttle_ms=float(raw["gaze_throttle_ms"]),
        gaze_mirror=bool(raw["gaze_mirror"]),
        match_threshold=float(raw["match_threshold"]),
        vote_samples=int(raw["vote_samples"]),
        reverify_interval_s=float(raw["reverify_interval_s"]),
        show_debug=bool(raw["show_debug"]),
        enroll_samples=int(raw["enroll_samples"]),
        enroll_timeout_s=float(raw["enroll_timeout_s"]),
        enroll_min_gap_ms=float(raw["enroll_min_gap_ms"]),
    )


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
        camera=CameraConfig(
            index=int(camera["index"]),
            device=str(camera["device"]) if camera.get("device") else None,
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["fps"]),
        ),
        perception=_load_perception(raw["perception"]),
        face=_load_face(raw["face"]),
        stt=raw.get("stt") or {},
        tts=raw.get("tts") or {},
        llm=raw.get("llm") or {},
    )
