"""cv2.VideoCapture setup: V4L2 on Linux, MJPG, low-latency single-frame buffer."""

from __future__ import annotations

import logging
import sys

import cv2

from assistant.config import CameraConfig

log = logging.getLogger("assistant.perception")


def open_capture(cfg: CameraConfig) -> cv2.VideoCapture:
    """Open and configure the webcam; raises RuntimeError if it cannot be opened.

    A configured device path wins over the numeric index: V4L2 indexes shuffle
    between boots, while /dev/v4l/by-id/ symlinks stay pinned to one camera —
    no silent fallback, so the wrong camera is never opened quietly.
    """
    if cfg.device and sys.platform.startswith("linux"):
        source: str | int = cfg.device
        cap = cv2.VideoCapture(cfg.device, cv2.CAP_V4L2)
    elif sys.platform.startswith("linux"):
        source = cfg.index
        cap = cv2.VideoCapture(cfg.index, cv2.CAP_V4L2)
    else:
        source = cfg.index
        cap = cv2.VideoCapture(cfg.index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open camera {source!r}")
    # FOURCC must be set before the frame size — the other order reverts many
    # UVC cameras to YUYV at a low fps. All props are advisory on some
    # backends, so the actual values are logged rather than asserted.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info(
        "camera %r open: %.0fx%.0f @ %.0f fps",
        source,
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FPS),
    )
    return cap
