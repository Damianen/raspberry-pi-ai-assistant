"""The Voice module: speech in (STT) and speech out (TTS) with half-duplex turn-taking.

Two worker threads share one HalfDuplexGate: the TTS side closes it before
audio reaches the speakers and releases it only after playback drains plus a
cooldown tail; the STT side discards every mic block captured while it is
closed. A voice failure disables only this module — the face never freezes
because of it.
"""

from __future__ import annotations

from assistant.bus import EventBus
from assistant.config import Config
from assistant.voice.logic import HalfDuplexGate
from assistant.voice.stt import SpeechToText
from assistant.voice.tts import TextToSpeech


class Voice:
    def __init__(self, config: Config, bus: EventBus) -> None:
        gate = HalfDuplexGate(config.stt.mute_tail_ms / 1000.0)
        self._stt = SpeechToText(config.stt, bus, gate)
        self._tts = TextToSpeech(config.tts, bus, gate)

    def start(self) -> None:
        self._stt.start()
        self._tts.start()

    def stop(self) -> None:
        # TTS first: it stops feeding the speakers that the mic must ignore.
        self._tts.stop()
        self._stt.stop()
