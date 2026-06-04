"""The pipeline: record -> STT -> intent -> act -> speak. Runs in a worker thread.

This is the ONLY place AppState is driven during an interaction. It writes state;
the UI reads it. Guards against overlapping taps with a busy flag.

Wiring is mostly here already; it depends on audio_io / stt / tts being
implemented on the Pi. Test it slice by slice (see CLAUDE.md build plan).
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from . import audio_io, brain, stt, tts
from .intent import Intent, IntentType, parse
from .state import AppState, SharedState
from .store import Store

# One line per interaction: timestamp | transcript | intent type | fire_at.
# This is the dataset we mine to fix the parser, so it's written for EVERY run,
# right after parsing — before acting, so a downstream failure can't drop the row.
LOG_PATH = Path("logs/interactions.log")


def _oneline(s: str) -> str:
    """Collapse anything that would break the pipe-delimited, one-row-per-line
    log format (field separators and newlines)."""
    return s.replace("|", "/").replace("\n", " ").replace("\r", " ")


class Pipeline:
    def __init__(self, shared: SharedState, store: Store, brain_cfg: dict, *,
                 input_device: int | str | None = None,
                 stt_model: str = "base.en") -> None:
        self.shared = shared
        self.store = store
        self.brain_cfg = brain_cfg
        self.input_device = input_device   # injected from config by run.py
        self.stt_model = stt_model
        self._busy = threading.Lock()

    def on_tap(self) -> None:
        """Non-blocking entry point for the UI. Ignores taps while busy."""
        if self._busy.locked():
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        with self._busy:
            try:
                self.shared.set(AppState.LISTENING)
                audio = audio_io.record_until_silence(
                    self.brain_cfg.get("max_record_seconds", 8),
                    device=self.input_device)

                self.shared.set(AppState.THINKING)
                text = stt.transcribe(audio, model=self.stt_model).strip()
                intent = parse(text, datetime.now())
                print(f"[pipeline] transcript: {text!r}")
                print(f"[pipeline] intent:     {intent.type.name}")

                # Commands are logged BEFORE acting so a store/TTS failure can't
                # drop the parser-mining row. A QUERY's answer isn't known until
                # the LLM returns, so its row is logged AFTER _handle; _ask_llm
                # never raises, so only a hard kill mid-request could lose it.
                if intent.type is not IntentType.QUERY:
                    self._log_interaction(text, intent)
                answer = self._handle(intent)
                if intent.type is IntentType.QUERY:
                    self._log_interaction(text, intent, answer or "")
            except Exception as exc:
                print(f"[pipeline] error: {exc!r}")
                self._error("Sorry, something went wrong.")
            finally:
                self.shared.set(AppState.IDLE)

    def _handle(self, intent: Intent) -> str | None:
        """Act on the intent. Returns the spoken LLM answer for QUERY (so the
        caller can log it), None for every local command."""
        if intent.type in (IntentType.SET_ALARM, IntentType.SET_TIMER,
                            IntentType.SET_REMINDER):
            kind = intent.type.name.replace("SET_", "").lower()
            self.store.add(kind, intent.fire_at, intent.label)
            self._confirm(self._confirm_phrase(kind, intent))
        elif intent.type is IntentType.GET_TIME:
            self._speak(f"It's {datetime.now():%H:%M}.")
        elif intent.type is IntentType.GET_DATE:
            self._speak(f"It's {datetime.now():%A, %B %d}.")
        else:  # QUERY -> LLM
            answer = self._ask_llm(intent.raw)
            self._speak(answer)
            return answer
        return None

    @staticmethod
    def _log_interaction(transcript: str, intent: Intent, answer: str = "") -> None:
        """Append one pipe-delimited row to the interaction log. Best-effort: a
        logging failure must never break the interaction, so it's swallowed.

        Columns: timestamp | transcript | intent | fire_at | answer. `answer` is
        the spoken LLM reply for QUERY intents (empty otherwise); only its first
        80 chars are kept so the log stays greppable and one-row-per-line."""
        fire_at = intent.fire_at.isoformat(timespec="seconds") if intent.fire_at else ""
        row = (f"{datetime.now().isoformat(timespec='seconds')} | {_oneline(transcript)} | "
               f"{intent.type.name} | {fire_at} | {_oneline(answer)[:80]}\n")
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(row)
        except Exception as exc:
            print(f"[pipeline] could not write interaction log: {exc!r}")

    # ---- effects ----
    def _confirm(self, phrase: str) -> None:
        self.shared.set(AppState.CONFIRM)
        tts.speak(phrase)

    def _speak(self, phrase: str) -> None:
        self.shared.set(AppState.SPEAKING)
        tts.speak(phrase)

    def _error(self, phrase: str) -> None:
        self.shared.set(AppState.ERROR)
        try:
            tts.speak(phrase)
        except Exception:
            pass

    def _ask_llm(self, text: str) -> str:
        try:
            return brain.ask(
                text,
                model=self.brain_cfg["model"],
                fallback_model=self.brain_cfg["fallback_model"],
                timeout=self.brain_cfg.get("timeout_seconds", 7),
                max_tokens=self.brain_cfg.get("max_tokens", 300),
            )
        except Exception:
            return "I can't reach the internet right now."

    @staticmethod
    def _confirm_phrase(kind: str, intent: Intent) -> str:
        when = f"{intent.fire_at:%H:%M}"
        if kind == "timer":
            return f"Timer set for {when}."
        if kind == "reminder":
            return f"Okay, I'll remind you to {intent.label} at {when}."
        return f"Alarm set for {when}."
