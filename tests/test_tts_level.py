"""Tests for the mouth-sync level plumbing (slice 7).

The cosmetic face changes are eyeballed on the panel, but turning playback into a
0..1 level is real logic worth pinning: (1) the rolling-peak normalization makes a
quiet voice animate as fully as a loud one, (2) silence drives the level to ~0 so
the mouth closes in the gaps, and (3) speak() always resets the level to 0 on exit
— normal end OR a playback error — so the mouth never freezes mid-flap.

No audio device and no Piper model: we fake the voice and stub audio_io.play, so
this runs anywhere pytest does (the real chunk loop is exercised on the Pi).
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from assistant import audio_io, tts
from assistant.state import SharedState


class _FakeChunk:
    def __init__(self, arr: np.ndarray) -> None:
        self.audio_float_array = arr


class _FakeConfig:
    sample_rate = 22050


class _FakeVoice:
    config = _FakeConfig()

    def synthesize(self, text: str):
        yield _FakeChunk(np.full(256, 0.5, dtype=np.float32))


def _loud(rms: float = 0.3, n: int = 1024) -> np.ndarray:
    return np.full(n, rms, dtype=np.float32)


def test_level_normalizes_against_rolling_peak():
    # A quiet voice (rms 0.02) must still hit a full-open mouth, because the peak
    # tracks ITS loudest chunk — absolute volume must not matter.
    shared = SharedState()
    tts.configure("x", output_device=None, shared=shared)
    cb = tts._make_level_cb()
    cb(_loud(0.02))
    assert shared.snapshot().meta["level"] == pytest.approx(1.0, abs=1e-3)


def test_silence_closes_the_mouth():
    # After a loud passage, sustained silence must decay the level to ~0 (the
    # between-sentence pause that proves the level is real, not a constant flap).
    shared = SharedState()
    tts.configure("x", output_device=None, shared=shared)
    cb = tts._make_level_cb()
    cb(_loud(0.4))
    for _ in range(200):
        cb(np.zeros(1024, dtype=np.float32))
    assert shared.snapshot().meta["level"] < 0.02


def test_speak_resets_level_on_clean_finish(monkeypatch):
    shared = SharedState()
    tts.configure("x", shared=shared)
    monkeypatch.setattr(tts, "_get_voice", lambda: _FakeVoice())

    def fake_play(samples, rate, device=None, *, stop_event=None, level_cb=None):
        if level_cb is not None:
            level_cb(_loud(0.4))                    # mouth opens mid-playback...

    monkeypatch.setattr(audio_io, "play", fake_play)
    tts.speak("hello there", stop_event=threading.Event())
    assert shared.snapshot().meta["level"] == 0.0   # ...then closes on finish


def test_speak_resets_level_on_playback_error(monkeypatch):
    shared = SharedState()
    tts.configure("x", shared=shared)
    monkeypatch.setattr(tts, "_get_voice", lambda: _FakeVoice())

    def boom(*a, **k):
        raise RuntimeError("playback blew up")

    monkeypatch.setattr(audio_io, "play", boom)
    with pytest.raises(RuntimeError):
        tts.speak("hello there", stop_event=threading.Event())
    assert shared.snapshot().meta["level"] == 0.0   # finally still ran


def test_no_shared_means_no_crash(monkeypatch):
    # Offline/test path: configure with shared=None -> speak must not touch state.
    tts.configure("x", shared=None)
    monkeypatch.setattr(tts, "_get_voice", lambda: _FakeVoice())
    monkeypatch.setattr(audio_io, "play", lambda *a, **k: None)
    tts.speak("hello there", stop_event=threading.Event())   # no exception
