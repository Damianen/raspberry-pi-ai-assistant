"""Local, deterministic intent parsing.

This is the reliability backbone. Commands (alarm/timer/reminder/time/date) are
parsed here with plain rules — instant, offline, and never hallucinated. Anything
that ISN'T a recognised command falls through to `QUERY`, which the caller routes
to the LLM. Alarms must NEVER depend on the network, so they must be parsed here.

Pure functions, no I/O — fully unit-testable (see tests/test_intent.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto


class IntentType(Enum):
    SET_ALARM = auto()
    SET_TIMER = auto()
    SET_REMINDER = auto()
    GET_TIME = auto()
    GET_DATE = auto()
    QUERY = auto()      # not a local command -> send to LLM


@dataclass
class Intent:
    type: IntentType
    # when the thing should fire (absolute), if applicable
    fire_at: datetime | None = None
    # human label, e.g. "take the bread out", or the raw query text
    label: str = ""
    # original transcript, always preserved
    raw: str = ""
    # True for a long-form request ("tell me a story/poem"): the caller hands this
    # to brain.ask(longform=...) for a different system prompt + bigger token
    # budget. Only ever set on QUERY; every command leaves it False.
    longform: bool = False


_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60,
    "half": 30, "quarter": 15, "a": 1, "an": 1,
}


# Words valid as a clock HOUR (deliberately excludes 'a'/'an'/'half'/'quarter'
# so "set an alarm for seven" doesn't read "an" as 1).
_HOUR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "noon": 12, "midnight": 0,
}


# Long-form requests ("tell me a story", "tell a poem"). Matched BEFORE the
# command rules so "tell me a story about waking up at seven" is a story, not a
# 7 o'clock alarm. The explicit "a/an story|poem|tale" keeps a normal "tell me
# about the moon" out of long-form mode — that still routes to a short answer.
_LONGFORM_RE = re.compile(r"\btell\s+(?:me\s+)?an?\s+(?:story|poem|tale)\b")


def _word_to_int(tok: str) -> int | None:
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok)


def _parse_clock_time(text: str, now: datetime) -> datetime | None:
    """Parse an absolute time-of-day like '7', '7:30', '6 am', 'half past seven'."""
    t = text.lower()

    # "half past seven", "quarter past six", "quarter to eight"
    m = re.search(r"(half|quarter|\w+)\s+(past|to)\s+(\w+)", t)
    if m:
        frac, rel, hour_w = m.group(1), m.group(2), m.group(3)
        mins = _word_to_int(frac)
        hour = _word_to_int(hour_w)
        if mins is not None and hour is not None:
            if rel == "to":
                hour = (hour - 1) % 24
                mins = 60 - mins
            return _build_time(now, hour, mins, t)

    # "7:30", "07:05", optional am/pm
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", t)
    if m:
        hour, mins = int(m.group(1)), int(m.group(2))
        return _build_time(now, hour, mins, t, m.group(3))

    # bare hour: digits first ("for 7 am"), then hour words ("at seven").
    for m in re.finditer(r"\b(\d{1,2})\b\s*(am|pm)?", t):
        hour = int(m.group(1))
        if 0 <= hour <= 23:
            return _build_time(now, hour, 0, t, m.group(2))
    for m in re.finditer(r"\b([a-z]+)\b\s*(am|pm)?", t):
        hour = _HOUR_WORDS.get(m.group(1))
        if hour is not None:
            return _build_time(now, hour, 0, t, m.group(2))
    return None


def _build_time(now: datetime, hour: int, mins: int, ctx: str,
                ampm: str | None = None) -> datetime:
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif ampm is None and hour < 7:
        # no am/pm and an early-looking hour: assume the *next* occurrence,
        # which _roll_forward handles below. (Keep dumb + predictable.)
        pass
    target = now.replace(hour=hour % 24, minute=mins, second=0, microsecond=0)
    return _roll_forward(target, now)


def _roll_forward(target: datetime, now: datetime) -> datetime:
    """If the time already passed today, push it to tomorrow."""
    if target <= now:
        target += timedelta(days=1)
    return target


def _parse_duration(text: str) -> timedelta | None:
    """Parse 'in 10 minutes', '5 min', 'two hours', '90 seconds'."""
    total = timedelta()
    found = False
    for value, unit in re.findall(
        r"(\d+|[a-z]+)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)", text.lower()
    ):
        n = _word_to_int(value)
        if n is None:
            continue
        found = True
        if unit.startswith(("hour", "hr")):
            total += timedelta(hours=n)
        elif unit.startswith(("min",)):
            total += timedelta(minutes=n)
        else:
            total += timedelta(seconds=n)
    return total if found else None


def _normalize(t: str) -> str:
    """Tidy a lowercased transcript before rule-matching.

    whisper writes "8 p.m." with dots, but the am/pm rules only match "pm" — so
    without this, "8 p.m." fails the (am|pm) match, is read as a bare hour, and
    becomes 8 AM. Collapse dotted/spaced "a.m."/"p.m." to "am"/"pm" and drop
    trailing sentence punctuation. Applied to the match text only; Intent.raw
    keeps the untouched transcript.
    """
    t = re.sub(r"\b([ap])\.\s*m\.?", r"\1m", t)
    return t.rstrip(" .,!?;:")


def parse(text: str, now: datetime | None = None) -> Intent:
    now = now or datetime.now()
    raw = text.strip()
    t = _normalize(raw.lower())

    if not t:
        return Intent(IntentType.QUERY, raw=raw, label=raw)

    # --- long-form story/poem -> LLM (with the long-form prompt). MUST stay above
    # the command rules so a story request is never swallowed as an alarm/timer. ---
    if _LONGFORM_RE.search(t):
        return Intent(IntentType.QUERY, raw=raw, label=raw, longform=True)

    # --- time / date questions ---
    if re.search(r"\bwhat('?s| is)?\s+the\s+time\b|\bwhat time is it\b", t):
        return Intent(IntentType.GET_TIME, raw=raw)
    if re.search(r"\bwhat('?s| is)?\s+(the\s+)?date\b|\bwhat day is it\b", t):
        return Intent(IntentType.GET_DATE, raw=raw)

    # --- timer: "set a timer for 10 minutes", "timer 5 min" ---
    if "timer" in t:
        dur = _parse_duration(t)
        if dur:
            return Intent(IntentType.SET_TIMER, fire_at=now + dur, raw=raw,
                          label=_strip_label(t, ("timer",)))

    # --- reminder: "remind me to X in/at Y" ---
    if "remind" in t:
        label = _extract_reminder_label(t)
        dur = _parse_duration(t)
        if dur:
            return Intent(IntentType.SET_REMINDER, fire_at=now + dur, raw=raw, label=label)
        clk = _parse_clock_time(t, now)
        if clk:
            return Intent(IntentType.SET_REMINDER, fire_at=clk, raw=raw, label=label)

    # --- alarm: "set an alarm for 7", "wake me at 6:30 am" ---
    if "alarm" in t or "wake me" in t or "wake up" in t:
        clk = _parse_clock_time(t, now)
        if clk:
            return Intent(IntentType.SET_ALARM, fire_at=clk, raw=raw, label="alarm")
        dur = _parse_duration(t)
        if dur:
            return Intent(IntentType.SET_ALARM, fire_at=now + dur, raw=raw, label="alarm")

    # --- fallthrough -> LLM ---
    return Intent(IntentType.QUERY, raw=raw, label=raw)


def _strip_label(t: str, drop: tuple[str, ...]) -> str:
    out = t
    for d in drop:
        out = out.replace(d, "")
    return out.strip()


def _extract_reminder_label(t: str) -> str:
    m = re.search(r"remind me to (.+?)(?:\s+(?:in|at)\s+.+)?$", t)
    if m:
        return m.group(1).strip()
    m = re.search(r"remind me (.+?)(?:\s+(?:in|at)\s+.+)?$", t)
    return m.group(1).strip() if m else "reminder"
