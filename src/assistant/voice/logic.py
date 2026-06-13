"""Pure voice logic — sentence splitting, half-duplex gate, utterance segmentation."""

from __future__ import annotations

import math
import re

# Sentence-ending punctuation followed by whitespace, or any line break.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into speakable sentences so long answers start playing quickly.

    Splits after ., !, ? or … when followed by whitespace, and on line breaks;
    decimals like "3.14" stay intact because the period has no trailing space.
    """
    return [part for part in (p.strip() for p in _SENTENCE_BOUNDARY.split(text)) if part]


def chunks_for_ms(ms: float, chunk_ms: float) -> int:
    """Smallest chunk count covering the given duration (at least 1)."""
    return max(1, math.ceil(ms / chunk_ms))


class HalfDuplexGate:
    """Mic gate: closed while the robot speaks, plus a cooldown tail after.

    Written from the TTS thread, read on the STT path. Plain attribute writes
    are atomic under the GIL and `close()` always runs before any audio
    reaches the speakers, so no lock is needed.
    """

    def __init__(self, cooldown_s: float) -> None:
        self._cooldown_s = cooldown_s
        self._closed = False
        self._cooldown_until = float("-inf")

    def close(self) -> None:
        """Speech playback is about to start: stop accepting mic input."""
        self._closed = True

    def release(self, now: float) -> None:
        """Playback fully drained: reopen after the cooldown tail."""
        self._closed = False
        self._cooldown_until = now + self._cooldown_s

    def accepts(self, now: float) -> bool:
        return not self._closed and now >= self._cooldown_until


class UtteranceSegmenter:
    """Turns a stream of per-chunk VAD probabilities into utterance boundaries.

    Feed one Silero speech probability per fixed-size audio chunk. Returns
    "begin" when voice starts, "end" when enough trailing silence accumulates
    (or the max length is hit) and the utterance had enough voiced chunks,
    "abort" when it did not, and None otherwise.
    """

    def __init__(
        self,
        *,
        threshold: float,
        end_silence_chunks: int,
        min_speech_chunks: int,
        max_chunks: int,
    ) -> None:
        self._threshold = threshold
        self._end_silence_chunks = end_silence_chunks
        self._min_speech_chunks = min_speech_chunks
        self._max_chunks = max_chunks
        self.active = False
        self._speech = 0
        self._silence = 0
        self._total = 0

    def feed(self, prob: float) -> str | None:
        voiced = prob >= self._threshold
        if not self.active:
            if not voiced:
                return None
            self.active = True
            self._speech = 1
            self._silence = 0
            self._total = 1
            return "begin"
        self._total += 1
        if voiced:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self._end_silence_chunks or self._total >= self._max_chunks:
            self.active = False
            return "end" if self._speech >= self._min_speech_chunks else "abort"
        return None

    def reset(self) -> None:
        """Discard any in-progress utterance (e.g. the mic gate closed)."""
        self.active = False
