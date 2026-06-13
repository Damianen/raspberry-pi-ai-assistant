"""STT worker: mic capture → Silero VAD segmentation → faster-whisper transcripts.

Audio blocks are timestamped at capture and checked against the half-duplex
gate, so anything recorded while the robot speaks (or during the cooldown
tail) is discarded even when transcription lags behind the microphone.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from assistant.bus import EventBus
from assistant.config import SttConfig
from assistant.paths import models_dir
from assistant.voice.logic import HalfDuplexGate, UtteranceSegmenter, chunks_for_ms
from assistant.voice.vad import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE, StreamingVad

log = logging.getLogger("assistant.voice")

# Internal plumbing — behavioral tunables live in config.
BLOCK_POLL_S = 0.1  # stop-flag responsiveness while waiting for mic blocks
STATUS_LOG_INTERVAL_S = 30.0  # rate limit for capture over/underflow warnings


class SpeechToText:
    def __init__(self, cfg: SttConfig, bus: EventBus, gate: HalfDuplexGate) -> None:
        self._cfg = cfg
        self._bus = bus
        self._gate = gate
        self._blocks: queue.SimpleQueue[tuple[np.ndarray, float]] = queue.SimpleQueue()
        self._status_count = 0
        self._status_logged_at = 0.0
        self._status_logged_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="assistant-stt", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    # --- worker thread -----------------------------------------------------

    def _run(self) -> None:
        # Everything stateful (whisper, vad, the input stream) lives on this thread.
        try:
            log.info("loading whisper model %r (%s)", self._cfg.model, self._cfg.compute_type)
            self._model = WhisperModel(
                self._cfg.model,
                device="cpu",
                compute_type=self._cfg.compute_type,
                download_root=str(models_dir() / "whisper"),
            )
            self._vad = StreamingVad()
            self._segmenter = UtteranceSegmenter(
                threshold=self._cfg.vad_threshold,
                end_silence_chunks=chunks_for_ms(self._cfg.silence_end_ms, CHUNK_MS),
                min_speech_chunks=chunks_for_ms(self._cfg.min_speech_ms, CHUNK_MS),
                max_chunks=chunks_for_ms(self._cfg.max_utterance_s * 1000.0, CHUNK_MS),
            )
            self._pre_roll: deque[np.ndarray] = deque(
                maxlen=chunks_for_ms(self._cfg.pre_roll_ms, CHUNK_MS)
            )
            self._utterance: list[np.ndarray] = []
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SAMPLES,
                channels=1,
                dtype="float32",
                device=self._cfg.input_device,
                callback=self._on_audio,
            )
        except Exception:
            log.exception("stt setup failed; speech recognition disabled")
            return
        log.info("listening on %r", sd.query_devices(stream.device)["name"])
        try:
            with stream:
                self._loop()
        except Exception:
            log.exception("stt crashed; speech recognition disabled")

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # PortAudio callback thread: hand off and get out.
        if status:
            self._status_count += 1
        self._blocks.put((indata[:, 0].copy(), time.monotonic()))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._log_capture_status()
            try:
                chunk, captured_at = self._blocks.get(timeout=BLOCK_POLL_S)
            except queue.Empty:
                continue
            if not self._gate.accepts(captured_at):
                self._discard_pending()
                continue
            self._process(chunk)

    def _discard_pending(self) -> None:
        """The robot is (or just was) speaking: drop everything the mic heard."""
        self._segmenter.reset()
        self._vad.reset()
        self._pre_roll.clear()
        self._utterance = []

    def _process(self, chunk: np.ndarray) -> None:
        decision = self._segmenter.feed(self._vad.prob(chunk))
        if decision == "begin":
            # VAD onset trails the actual voice start; keep the pre-roll audio.
            self._utterance = list(self._pre_roll)
            self._pre_roll.clear()
        if self._segmenter.active or decision in ("end", "abort"):
            self._utterance.append(chunk)
        else:
            self._pre_roll.append(chunk)
        if decision == "end":
            audio = np.concatenate(self._utterance)
            self._utterance = []
            self._transcribe(audio)
        elif decision == "abort":
            log.debug("segment dropped: too little voiced audio")
            self._utterance = []

    def _transcribe(self, audio: np.ndarray) -> None:
        started = time.monotonic()
        segments, _info = self._model.transcribe(
            audio,
            language=self._cfg.language,
            beam_size=self._cfg.beam_size,
            without_timestamps=True,
        )
        texts = [
            seg.text.strip()
            for seg in segments
            if seg.no_speech_prob <= self._cfg.no_speech_threshold
        ]
        text = " ".join(t for t in texts if t)
        audio_s = len(audio) / SAMPLE_RATE
        if not text:
            log.info("discarded segment: no confident speech in %.1fs of audio", audio_s)
            return
        log.info(
            "heard %r (%.1fs audio, %.1fs transcribe)",
            text,
            audio_s,
            time.monotonic() - started,
        )
        self._bus.emit("speech_heard", {"text": text})

    def _log_capture_status(self) -> None:
        if self._status_count == self._status_logged_count:
            return
        now = time.monotonic()
        if now - self._status_logged_at < STATUS_LOG_INTERVAL_S:
            return
        log.warning("mic capture reported %d over/underflows so far", self._status_count)
        self._status_logged_at = now
        self._status_logged_count = self._status_count
