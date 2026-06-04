"""LLM fallback via OpenRouter (OpenAI-compatible). FINISH + TEST with your key.

Only called when the local intent parser returns QUERY. Engineered so a network
hiccup degrades gracefully instead of freezing the voice loop:
- bounded time per attempt -> raises -> caller speaks a fallback line
- try the cheap model, then the fallback model, then give up (2 attempts)
- max_tokens cap so the model can't ramble into a 30-second monologue

Worst-case latency (the thing that makes a device feel broken): the `timeout`
argument is the READ timeout and is paired with a short CONNECT timeout, so an
offline/no-upstream tap fails the TCP connect fast (~2 x _CONNECT_TIMEOUT)
instead of burning a full read budget per attempt, while a working call still
gets the whole read window to answer.
CAVEAT: a socket timeout does NOT bound DNS resolution. If the interface is up
but the resolver is dead (captive portal), getaddrinfo can block on the system
resolver well past this budget. "Wifi off" (interface down) fails fast; a dead
resolver is the lurking case. A true wall-clock cap would need a watchdog
thread (out of v1 scope) -- TODO if the failure drills surface it on the Pi.

Key comes from the OPENROUTER_API_KEY env var, never from disk.
Privacy: STT is local, so only the transcribed TEXT leaves the device. Disable
provider data retention in your OpenRouter account if you care about that.
"""
from __future__ import annotations

import os

import requests

_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cap the TCP handshake so an offline tap fails fast. 3.05 is the requests idiom
# (just over a 3s kernel SYN-retransmit multiple). The read budget stays separate.
_CONNECT_TIMEOUT = 3.05

SYSTEM = (
    "You are a small desk assistant. Answer in one or two short spoken sentences. "
    "No markdown, no lists — this will be read aloud."
)

# Long-form mode ("tell me a story/poem"), selected by ask(longform=True). Kept
# entirely separate from the SYSTEM/120-token path so the normal 1-2 sentence
# answer is unchanged.
LONGFORM_SYSTEM = (
    "You are a small desk assistant telling a story out loud. Tell one engaging, "
    "self-contained short story — about a minute when read aloud. Use plain spoken "
    "prose: no markdown, no lists, no headings, no title. Begin the story directly."
)
LONGFORM_MAX_TOKENS = 400
# 400 tokens take longer to generate than 1-2 sentences, so the READ timeout gets
# a higher floor for long-form (the CONNECT timeout is untouched — offline still
# fails fast). Without this, a legitimately slow story trips the 7s read budget
# and degrades to the fallback line for no good reason.
_LONGFORM_READ_TIMEOUT = 15.0


def ask(text: str, *, model: str, fallback_model: str,
        timeout: float = 7.0, max_tokens: int = 120, longform: bool = False) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    if longform:
        system = LONGFORM_SYSTEM
        max_tokens = LONGFORM_MAX_TOKENS
        timeout = max(timeout, _LONGFORM_READ_TIMEOUT)
    else:
        system = SYSTEM

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": text}]

    attempts = (model, fallback_model)
    timeouts = (_CONNECT_TIMEOUT, timeout)  # (connect, read)
    for i, mdl in enumerate(attempts):
        try:
            r = requests.post(
                _URL, headers=headers, timeout=timeouts,
                json={"model": mdl, "messages": messages, "max_tokens": max_tokens},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            if i == len(attempts) - 1:
                raise
    raise RuntimeError("unreachable")
