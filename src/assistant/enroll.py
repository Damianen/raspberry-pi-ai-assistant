"""Console script: enroll a face as the normalized mean of SFace embeddings.

Captures the largest detected face over up to perception.enroll_timeout_s
seconds and stores the L2-normalized mean embedding in data/brain.db under
the given name (re-enrolling replaces the previous embedding).
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from assistant.config import Config, load_config
from assistant.memory.db import connect, db_path
from assistant.memory.people import PeopleStore
from assistant.perception.camera import open_capture
from assistant.perception.vision import FaceFinder, create_recognizer, embed_face

READ_FAILURE_SLEEP_S = 0.1
DEBUG_WINDOW = "assistant-enroll"
DEBUG_BOX_COLOR = (0, 255, 0)  # BGR


def _collect_embeddings(config: Config, args: argparse.Namespace) -> list[np.ndarray]:
    pcfg = config.perception
    finder = FaceFinder(pcfg, (config.camera.width, config.camera.height))
    recognizer = create_recognizer()
    cap = open_capture(config.camera)

    print(
        f"Enrolling {args.name!r}: look at the camera "
        f"(collecting {pcfg.enroll_samples} samples, up to {pcfg.enroll_timeout_s:.0f}s)..."
    )
    embeddings: list[np.ndarray] = []
    deadline = time.monotonic() + pcfg.enroll_timeout_s
    last_sample_at = 0.0
    frame_index = 0
    try:
        while len(embeddings) < pcfg.enroll_samples and time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(READ_FAILURE_SLEEP_S)
                continue
            frame_index += 1
            if frame_index % pcfg.detect_every_n_frames != 0:
                continue
            face = finder.largest(frame)
            now = time.monotonic()
            if face is not None and now - last_sample_at >= pcfg.enroll_min_gap_ms / 1000.0:
                last_sample_at = now
                embedding = embed_face(recognizer, frame, face)
                norm = float(np.linalg.norm(embedding))
                if norm > 0.0:
                    embeddings.append(embedding / norm)
                    print(f"  sample {len(embeddings)}/{pcfg.enroll_samples}")
            if args.show:
                if face is not None:
                    x, y, w, h = (int(v) for v in face[:4])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), DEBUG_BOX_COLOR, 2)
                cv2.imshow(DEBUG_WINDOW, frame)
                cv2.waitKey(1)
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
    return embeddings


def run() -> None:
    parser = argparse.ArgumentParser(
        prog="assistant-enroll",
        description="Enroll a face for recognition by the assistant.",
    )
    parser.add_argument("name", help="name to store the face under")
    parser.add_argument(
        "--show", action="store_true", help="show the camera feed with the detection box"
    )
    args = parser.parse_args()

    config = load_config()
    embeddings = _collect_embeddings(config, args)
    if not embeddings:
        raise SystemExit("no face captured before the timeout — nothing stored.")

    mean = np.mean(np.stack(embeddings), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise SystemExit("degenerate embeddings (zero mean) — nothing stored.")
    PeopleStore(connect()).upsert(args.name, mean / norm)

    if len(embeddings) < config.perception.enroll_samples:
        print(f"warning: timed out after {len(embeddings)} samples; enrolled anyway.")
    print(f"Enrolled {args.name!r} from {len(embeddings)} samples -> {db_path()}")
