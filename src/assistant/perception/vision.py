"""OpenCV detector/recognizer wrappers and model auto-download, shared with enrollment.

The opencv_zoo repo stores its models via git-lfs, so downloads go through
media.githubusercontent.com; a tiny file on disk means we received an LFS
pointer instead of the model, which is treated as an error.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from assistant.config import PerceptionConfig
from assistant.paths import models_dir

log = logging.getLogger("assistant.perception")

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MIN_MODEL_BYTES = 100 * 1024  # an LFS pointer is ~130 bytes; the real models are MBs


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str


YUNET = ModelSpec(
    filename="face_detection_yunet_2023mar.onnx",
    url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
SFACE = ModelSpec(
    filename="face_recognition_sface_2021dec.onnx",
    url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)


def ensure_model(spec: ModelSpec) -> Path:
    """Return the local model path, downloading it on first use."""
    path = models_dir() / spec.filename
    if path.is_file() and path.stat().st_size > MIN_MODEL_BYTES:
        return path
    log.info("downloading %s from %s", spec.filename, spec.url)
    partial = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(spec.url, partial)
    size = partial.stat().st_size
    if size <= MIN_MODEL_BYTES:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.filename}: downloaded only {size} bytes — looks like a git-lfs "
            f"pointer, not the model ({spec.url})"
        )
    partial.replace(path)
    log.info("downloaded %s (%.1f MB)", spec.filename, size / 1e6)
    return path


def create_recognizer() -> cv2.FaceRecognizerSF:
    return cv2.FaceRecognizerSF.create(str(ensure_model(SFACE)), "")


def embed_face(recognizer: cv2.FaceRecognizerSF, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
    """SFace embedding of the aligned crop for one YuNet detection row."""
    aligned = recognizer.alignCrop(frame, face)
    # feature() may return a view of an internal buffer reused by the next call.
    return np.asarray(recognizer.feature(aligned), dtype=np.float32).copy().ravel()


class FaceFinder:
    """YuNet wrapper that tracks the input size and returns only the largest face."""

    def __init__(self, pcfg: PerceptionConfig, input_size: tuple[int, int]) -> None:
        self._detector = cv2.FaceDetectorYN.create(
            str(ensure_model(YUNET)),
            "",
            input_size,
            pcfg.detector_score_threshold,
            pcfg.detector_nms_threshold,
            pcfg.detector_top_k,
        )
        self._input_size = input_size

    def largest(self, frame: np.ndarray) -> np.ndarray | None:
        """Largest detected face as the full 15-float YuNet row, or None.

        Row layout: x, y, w, h, five landmark (x, y) pairs, score — alignCrop
        needs the landmarks, so the row is passed around intact.
        """
        height, width = frame.shape[:2]
        if (width, height) != self._input_size:
            self._input_size = (width, height)
            self._detector.setInputSize(self._input_size)
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: float(f[2]) * float(f[3]))
