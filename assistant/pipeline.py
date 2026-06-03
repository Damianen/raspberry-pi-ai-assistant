"""The pipeline: record -> STT -> intent -> act -> speak. Runs in a worker thread.

This is the ONLY place AppState is driven during an interaction. It writes state;
the UI reads it. Guards against overlapping taps with a busy flag.

Wiring is mostly here already; it depends on audio_io / stt / tts being
implemented on the Pi. Test it slice by slice (see CLAUDE.md build plan).
"""
from __future__ import annotations

import threading
from datetime import datetime

from . import audio_io, brain, stt, tts
from .intent import Intent, IntentType, parse
from .state import AppState, SharedState
from .store import Store


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

                # --- SLICE 2 ONLY: prove capture + STT. parse() is side-effect-
                # free, so printing the intent type is safe. Do NOT call the
                # action handler, store, tts, or brain yet (that's slice 3+).
                intent = parse(text, datetime.now())
                print(f"[slice2] transcript: {text!r}")
                print(f"[slice2] intent:     {intent.type.name}")
            except Exception as exc:
                # No tts in this slice — surface failures on the console + ERROR eyes.
                self.shared.set(AppState.ERROR)
                print(f"[slice2] pipeline error: {exc!r}")
            finally:
                self.shared.set(AppState.IDLE)

    def _handle(self, intent: Intent) -> None:
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
            self._speak(self._ask_llm(intent.raw))

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
