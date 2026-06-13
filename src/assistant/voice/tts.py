"""TTS worker: say events → sentence-by-sentence piper synthesis → speakers.

The shared half-duplex gate closes before any audio reaches the speakers and
is released — starting the mute cooldown — only after the output stream has
fully drained, so the mic provably never hears the robot's own voice.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import sounddevice as sd
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

from assistant.bus import EventBus
from assistant.config import TtsConfig
from assistant.paths import models_dir
from assistant.voice.logic import HalfDuplexGate, split_sentences

log = logging.getLogger("assistant.voice")

# Internal plumbing — behavioral tunables live in config.
SAY_POLL_S = 0.1  # stop-flag responsiveness while waiting for say events
WRITE_BLOCK_S = 0.25  # playback write granularity; bounds stop() latency


def ensure_voice(voice_id: str) -> Path:
    """Return the local piper voice path, downloading it on first use."""
    voice_dir = models_dir() / "piper"
    onnx = voice_dir / f"{voice_id}.onnx"
    config = voice_dir / f"{voice_id}.onnx.json"
    if onnx.is_file() and config.is_file():
        return onnx
    voice_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading piper voice %s to %s", voice_id, voice_dir)
    download_voice(voice_id, voice_dir)
    if not (onnx.is_file() and config.is_file()):
        raise RuntimeError(f"piper voice download produced no model files: {voice_id}")
    log.info("downloaded %s (%.1f MB)", voice_id, onnx.stat().st_size / 1e6)
    return onnx


class TextToSpeech:
    def __init__(self, cfg: TtsConfig, bus: EventBus, gate: HalfDuplexGate) -> None:
        self._cfg = cfg
        self._bus = bus
        self._gate = gate
        self._inbox = bus.open_inbox("say")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="assistant-tts", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self._inbox.close()

    # --- worker thread -----------------------------------------------------

    def _run(self) -> None:
        try:
            self._voice = PiperVoice.load(ensure_voice(self._cfg.voice))
            self._syn_config = SynthesisConfig(length_scale=self._cfg.length_scale)
            self._write_block = int(self._voice.config.sample_rate * WRITE_BLOCK_S)
        except Exception:
            log.exception("tts setup failed; speech output disabled")
            return
        try:
            self._loop()
        except Exception:
            log.exception("tts crashed; speech output disabled")

    def _loop(self) -> None:
        while not self._stop.is_set():
            event = self._inbox.get(timeout=SAY_POLL_S)
            if event is None:
                continue
            sentences = split_sentences(str(event.payload.get("text") or ""))
            if sentences:
                self._speak(sentences)

    def _speak(self, sentences: list[str]) -> None:
        self._gate.close()
        self._bus.emit("speaking_started")
        self._bus.emit("face_state", {"state": "speaking"})
        try:
            # Context exit stops the stream, which blocks until playback drains.
            with sd.OutputStream(
                samplerate=self._voice.config.sample_rate,
                channels=1,
                dtype="int16",
                device=self._cfg.output_device,
            ) as stream:
                for sentence in sentences:
                    if self._stop.is_set():
                        break
                    log.info("speaking: %r", sentence)
                    self._play_sentence(stream, sentence)
        finally:
            self._gate.release(time.monotonic())
            self._bus.emit("speaking_finished")

    def _play_sentence(self, stream: sd.OutputStream, sentence: str) -> None:
        for chunk in self._voice.synthesize(sentence, self._syn_config):
            samples = chunk.audio_int16_array
            # Write in small blocks so a stop request never waits long.
            for start in range(0, len(samples), self._write_block):
                if self._stop.is_set():
                    return
                stream.write(samples[start : start + self._write_block])
