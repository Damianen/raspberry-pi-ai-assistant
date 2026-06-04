"""Tests for the OpenRouter fallback. No network: requests.post is monkeypatched.

These pin the failure-mode contract that keeps the device from staring into
space when offline (see brain.py): exactly two attempts (cheap model, then the
fallback model) and a short connect timeout paired with the configured read
timeout. If someone re-adds a third attempt or collapses the timeout back to a
scalar, the worst-case latency regresses — these tests catch that.
"""
from __future__ import annotations

import pytest

from assistant import brain


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def test_no_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        brain.ask("hi", model="m", fallback_model="f")


def test_success_returns_stripped_content(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = {}

    def fake_post(url, *, headers, timeout, json):
        seen["timeout"] = timeout
        seen["model"] = json["model"]
        seen["max_tokens"] = json["max_tokens"]
        return _FakeResponse("  The sky is blue.  ")

    monkeypatch.setattr(brain.requests, "post", fake_post)
    out = brain.ask("why is the sky blue", model="cheap", fallback_model="f",
                    timeout=7.0, max_tokens=120)
    assert out == "The sky is blue."
    assert seen["model"] == "cheap"            # first attempt uses the cheap model
    assert seen["max_tokens"] == 120
    # (connect, read): connect capped short, read = configured timeout.
    assert seen["timeout"] == (brain._CONNECT_TIMEOUT, 7.0)


def test_offline_tries_two_models_then_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[str] = []

    def fake_post(url, *, headers, timeout, json):
        calls.append(json["model"])
        raise brain.requests.exceptions.ConnectionError("wifi off")

    monkeypatch.setattr(brain.requests, "post", fake_post)
    with pytest.raises(Exception):
        brain.ask("why is the sky blue", model="cheap", fallback_model="backup")
    # Exactly two attempts, cheap then fallback — NOT (model, model, fallback).
    assert calls == ["cheap", "backup"]


def test_longform_uses_story_prompt_budget_and_timeout(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = {}

    def fake_post(url, *, headers, timeout, json):
        seen["timeout"] = timeout
        seen["max_tokens"] = json["max_tokens"]
        seen["system"] = json["messages"][0]["content"]
        return _FakeResponse("Once upon a time...")

    monkeypatch.setattr(brain.requests, "post", fake_post)
    out = brain.ask("tell me a story about trees", model="cheap", fallback_model="f",
                    timeout=7.0, max_tokens=120, longform=True)
    assert out == "Once upon a time..."
    # Long-form overrides BOTH the prompt and the token budget...
    assert seen["system"] == brain.LONGFORM_SYSTEM
    assert seen["max_tokens"] == brain.LONGFORM_MAX_TOKENS
    # ...and lifts the READ timeout floor (connect stays short -> offline fails fast).
    assert seen["timeout"] == (brain._CONNECT_TIMEOUT, brain._LONGFORM_READ_TIMEOUT)


def test_falls_back_to_second_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[str] = []

    def fake_post(url, *, headers, timeout, json):
        calls.append(json["model"])
        if len(calls) == 1:
            raise brain.requests.exceptions.ConnectionError("primary blip")
        return _FakeResponse("answer from backup")

    monkeypatch.setattr(brain.requests, "post", fake_post)
    out = brain.ask("q", model="cheap", fallback_model="backup")
    assert out == "answer from backup"
    assert calls == ["cheap", "backup"]
