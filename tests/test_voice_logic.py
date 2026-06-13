"""Sentence splitting, half-duplex gating, and VAD utterance segmentation."""

from __future__ import annotations

from assistant.voice.logic import (
    HalfDuplexGate,
    UtteranceSegmenter,
    chunks_for_ms,
    split_sentences,
)

# --- split_sentences ---------------------------------------------------------


def test_split_empty_and_whitespace() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_split_single_sentence_without_punctuation() -> None:
    assert split_sentences("hello there") == ["hello there"]


def test_split_multiple_sentences() -> None:
    assert split_sentences("Hi. How are you? Great!") == ["Hi.", "How are you?", "Great!"]


def test_split_keeps_ellipsis_together() -> None:
    assert split_sentences("Wait... okay.") == ["Wait...", "okay."]


def test_split_keeps_decimals_together() -> None:
    assert split_sentences("Pi is 3.14 exactly.") == ["Pi is 3.14 exactly."]


def test_split_on_line_breaks() -> None:
    assert split_sentences("first line\nsecond line") == ["first line", "second line"]


def test_split_strips_whitespace() -> None:
    assert split_sentences("  One.   Two.  ") == ["One.", "Two."]


# --- chunks_for_ms -----------------------------------------------------------


def test_chunks_for_ms_rounds_up() -> None:
    assert chunks_for_ms(700.0, 32.0) == 22  # ceil(21.875)
    assert chunks_for_ms(320.0, 32.0) == 10  # exact


def test_chunks_for_ms_is_at_least_one() -> None:
    assert chunks_for_ms(0.0, 32.0) == 1
    assert chunks_for_ms(1.0, 32.0) == 1


# --- HalfDuplexGate ----------------------------------------------------------

COOLDOWN_S = 0.3


def test_gate_accepts_initially() -> None:
    assert HalfDuplexGate(COOLDOWN_S).accepts(0.0)


def test_gate_rejects_while_closed() -> None:
    gate = HalfDuplexGate(COOLDOWN_S)
    gate.close()
    assert not gate.accepts(0.0)
    assert not gate.accepts(100.0)  # closed has no timeout; only release reopens


def test_gate_cooldown_after_release() -> None:
    gate = HalfDuplexGate(COOLDOWN_S)
    gate.close()
    gate.release(10.0)
    assert not gate.accepts(10.0)
    assert not gate.accepts(10.0 + COOLDOWN_S - 0.01)
    assert gate.accepts(10.0 + COOLDOWN_S)


def test_gate_reclose_during_cooldown() -> None:
    gate = HalfDuplexGate(COOLDOWN_S)
    gate.close()
    gate.release(10.0)
    gate.close()  # next utterance queued back-to-back
    assert not gate.accepts(10.0 + COOLDOWN_S + 1.0)
    gate.release(20.0)
    assert gate.accepts(20.0 + COOLDOWN_S)


# --- UtteranceSegmenter ------------------------------------------------------

THRESHOLD = 0.5
VOICE = 0.9
SILENCE = 0.1


def make_segmenter(
    *, end_silence: int = 3, min_speech: int = 2, max_chunks: int = 100
) -> UtteranceSegmenter:
    return UtteranceSegmenter(
        threshold=THRESHOLD,
        end_silence_chunks=end_silence,
        min_speech_chunks=min_speech,
        max_chunks=max_chunks,
    )


def test_segmenter_stays_idle_on_silence() -> None:
    seg = make_segmenter()
    for _ in range(50):
        assert seg.feed(SILENCE) is None
    assert not seg.active


def test_segmenter_begins_on_voice() -> None:
    seg = make_segmenter()
    assert seg.feed(VOICE) == "begin"
    assert seg.active


def test_segmenter_threshold_is_inclusive() -> None:
    assert make_segmenter().feed(THRESHOLD) == "begin"


def test_segmenter_ends_after_trailing_silence() -> None:
    seg = make_segmenter(end_silence=3, min_speech=2)
    assert seg.feed(VOICE) == "begin"
    assert seg.feed(VOICE) is None
    assert seg.feed(SILENCE) is None
    assert seg.feed(SILENCE) is None
    assert seg.feed(SILENCE) == "end"
    assert not seg.active


def test_segmenter_aborts_short_utterance() -> None:
    seg = make_segmenter(end_silence=3, min_speech=2)
    assert seg.feed(VOICE) == "begin"  # a single voiced chunk: a click, a cough
    seg.feed(SILENCE)
    seg.feed(SILENCE)
    assert seg.feed(SILENCE) == "abort"
    assert not seg.active


def test_segmenter_voice_resets_silence_count() -> None:
    seg = make_segmenter(end_silence=3, min_speech=2)
    seg.feed(VOICE)
    seg.feed(SILENCE)
    seg.feed(SILENCE)
    assert seg.feed(VOICE) is None  # pause survived; still the same utterance
    seg.feed(SILENCE)
    seg.feed(SILENCE)
    assert seg.feed(SILENCE) == "end"


def test_segmenter_force_ends_at_max_length() -> None:
    seg = make_segmenter(end_silence=100, min_speech=2, max_chunks=5)
    assert seg.feed(VOICE) == "begin"
    for _ in range(3):
        assert seg.feed(VOICE) is None
    assert seg.feed(VOICE) == "end"
    assert not seg.active


def test_segmenter_force_end_of_garbage_aborts() -> None:
    seg = make_segmenter(end_silence=100, min_speech=3, max_chunks=4)
    assert seg.feed(VOICE) == "begin"
    seg.feed(SILENCE)
    seg.feed(SILENCE)
    assert seg.feed(SILENCE) == "abort"  # hit max with too little voice


def test_segmenter_reset_discards_utterance() -> None:
    seg = make_segmenter()
    seg.feed(VOICE)
    seg.reset()
    assert not seg.active
    assert seg.feed(VOICE) == "begin"  # ready for a fresh utterance


def test_segmenter_reusable_after_end() -> None:
    seg = make_segmenter(end_silence=2, min_speech=1)
    seg.feed(VOICE)
    seg.feed(SILENCE)
    assert seg.feed(SILENCE) == "end"
    assert seg.feed(VOICE) == "begin"
