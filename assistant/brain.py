"""LLM fallback via OpenRouter (OpenAI-compatible). FINISH + TEST with your key.

Only called when the local intent parser returns QUERY. Engineered so a network
hiccup degrades gracefully instead of freezing the voice loop:
- hard timeout (config) -> raises -> caller speaks a fallback line
- one retry, then try the fallback model, then give up
- max_tokens cap so the model can't ramble into a 30-second monologue

Key comes from the OPENROUTER_API_KEY env var, never from disk.
Privacy: STT is local, so only the transcribed TEXT leaves the device. Disable
provider data retention in your OpenRouter account if you care about that.
"""
from __future__ import annotations

import os

import requests

_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = (
    "You are a small desk assistant. Answer in one or two short spoken sentences. "
    "No markdown, no lists — this will be read aloud."
)


def ask(text: str, *, model: str, fallback_model: str,
        timeout: float = 7.0, max_tokens: int = 300) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}]

    for attempt, mdl in enumerate((model, model, fallback_model)):
        try:
            r = requests.post(
                _URL, headers=headers, timeout=timeout,
                json={"model": mdl, "messages": messages, "max_tokens": max_tokens},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")
