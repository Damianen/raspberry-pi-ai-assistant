"""Tests for the local intent parser. Run: pytest -q (or python -m pytest)."""
from datetime import datetime

from assistant.intent import IntentType, parse

NOW = datetime(2026, 6, 3, 14, 0, 0)  # Wed 3 Jun 2026, 14:00


def test_alarm_with_ampm():
    i = parse("set an alarm for 7 am", NOW)
    assert i.type is IntentType.SET_ALARM
    assert i.fire_at.hour == 7 and i.fire_at.minute == 0
    # 07:00 already passed today -> rolls to tomorrow
    assert i.fire_at.day == 4


def test_alarm_pm_with_dots():
    # whisper writes "p.m." with dots; without normalization this parsed as 8 AM.
    i = parse("set an alarm for 8 p.m.", NOW)
    assert i.type is IntentType.SET_ALARM
    assert (i.fire_at.hour, i.fire_at.minute) == (20, 0)


def test_alarm_pm_with_dots_uppercase():
    i = parse("set an alarm for 8 P.M.", NOW)
    assert i.type is IntentType.SET_ALARM
    assert (i.fire_at.hour, i.fire_at.minute) == (20, 0)


def test_alarm_hhmm():
    i = parse("wake me at 6:30 am", NOW)
    assert i.type is IntentType.SET_ALARM
    assert (i.fire_at.hour, i.fire_at.minute) == (6, 30)


def test_alarm_half_past():
    i = parse("set an alarm for half past seven", NOW)
    assert i.type is IntentType.SET_ALARM
    assert (i.fire_at.hour, i.fire_at.minute) == (7, 30)


def test_timer_minutes():
    i = parse("set a timer for 10 minutes", NOW)
    assert i.type is IntentType.SET_TIMER
    assert i.fire_at == NOW.replace(hour=14, minute=10)


def test_reminder_in_duration():
    i = parse("remind me to take the bread out in 20 minutes", NOW)
    assert i.type is IntentType.SET_REMINDER
    assert "bread" in i.label
    assert i.fire_at == NOW.replace(minute=20)


def test_get_time():
    assert parse("what time is it", NOW).type is IntentType.GET_TIME


def test_get_date():
    assert parse("what's the date", NOW).type is IntentType.GET_DATE


def test_query_fallthrough():
    i = parse("why is the sky blue", NOW)
    assert i.type is IntentType.QUERY
    assert i.raw == "why is the sky blue"
    assert i.longform is False


def test_empty_is_query():
    assert parse("   ", NOW).type is IntentType.QUERY


# --- long-form (slice 6) ---

def test_longform_story_beats_alarm():
    # The whole point of ordering long-form first: this must be a story, NOT a
    # 7 o'clock alarm.
    i = parse("tell me a story about waking up at seven", NOW)
    assert i.type is IntentType.QUERY
    assert i.longform is True
    assert i.fire_at is None


def test_longform_plain_story():
    i = parse("tell me a story about trees", NOW)
    assert i.type is IntentType.QUERY
    assert i.longform is True


def test_longform_poem_no_me():
    i = parse("tell a poem", NOW)
    assert i.type is IntentType.QUERY
    assert i.longform is True


def test_normal_command_still_parses_and_not_longform():
    # "tell" only triggers long-form for a/an story|poem|tale — commands are intact.
    i = parse("set a timer for one minute", NOW)
    assert i.type is IntentType.SET_TIMER
    assert i.longform is False
    assert i.fire_at == NOW.replace(minute=1)


def test_tell_me_about_is_a_normal_query():
    # A plain "tell me about X" is NOT long-form: short spoken answer, 120-token cap.
    i = parse("tell me about the moon", NOW)
    assert i.type is IntentType.QUERY
    assert i.longform is False
