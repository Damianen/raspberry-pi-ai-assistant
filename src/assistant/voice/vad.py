"""Streaming Silero VAD: one speech probability per 32 ms chunk of 16 kHz mono.

Runs the Silero VAD v6 ONNX model that faster-whisper bundles (the official
silero-vad package would pull in torch, which neither the laptop nor the Pi
needs) through onnxruntime with explicit LSTM state, so probabilities stream
chunk by chunk instead of going through faster-whisper's batch-only wrapper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime
from faster_whisper.vad import get_assets_path

# Model contract (Silero VAD v6 at 16 kHz) — these are not tunables.
SAMPLE_RATE = 16_000  # whisper expects 16 kHz mono too
CHUNK_SAMPLES = 512  # fixed model chunk: 32 ms at 16 kHz
CHUNK_MS = 1000.0 * CHUNK_SAMPLES / SAMPLE_RATE
_CONTEXT_SAMPLES = 64  # the model input carries this much of the previous chunk
_STATE_SHAPE = (1, 1, 128)
_MODEL_FILENAME = "silero_vad_v6.onnx"


class StreamingVad:
    def __init__(self) -> None:
        path = Path(get_assets_path()) / _MODEL_FILENAME
        if not path.is_file():
            raise RuntimeError(
                f"bundled silero model not found at {path} — "
                "did a faster-whisper upgrade rename its VAD asset?"
            )
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        """Forget LSTM state and context, e.g. after a gap in the audio stream."""
        self._h = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        """Speech probability for one float32 chunk of exactly CHUNK_SAMPLES samples."""
        x = np.concatenate([self._context, chunk]).reshape(1, -1)
        out, self._h, self._c = self._session.run(
            None, {"input": x, "h": self._h, "c": self._c}
        )
        self._context = chunk[-_CONTEXT_SAMPLES:].copy()
        return float(out.reshape(-1)[0])
