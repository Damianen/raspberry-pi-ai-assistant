"""Entrypoint. Loads config, wires components, runs the UI on the main thread.

  python run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from assistant.pipeline import Pipeline
from assistant.scheduler import Scheduler
from assistant.state import AppState, SharedState
from assistant.store import Event, Store
from assistant.ui import run_ui
from assistant import audio_io, tts

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def load_config() -> dict:
    path = Path("config.toml")
    if not path.exists():
        print("config.toml not found — copy config.example.toml and fill it in.")
        sys.exit(1)
    with path.open("rb") as f:
        return tomllib.load(f)


def main() -> None:
    cfg = load_config()
    shared = SharedState()
    store = Store(cfg["storage"]["db_path"])

    brain_cfg = {**cfg["brain"], "max_record_seconds": cfg["audio"]["max_record_seconds"]}
    pipeline = Pipeline(
        shared, store, brain_cfg,
        input_device=cfg["audio"]["input_device"] or None,
        stt_model=cfg["stt"]["model"],
    )

    output_device = cfg["audio"]["output_device"] or None

    def on_fire(ev: Event) -> None:
        shared.set(AppState.SPEAKING)
        # Per-kind announcement. A timer's stored label is the stripped command
        # text ("set a  for one minute") — never speak that; timers are anonymous.
        if ev.kind == "reminder":
            phrase = f"Reminder: {ev.label}."
        elif ev.kind == "timer":
            phrase = "Timer's up."
        else:  # alarm
            phrase = "Alarm."
        try:
            audio_io.beep(device=output_device)   # audible chime before the announcement
            tts.speak(phrase)
        except Exception as exc:
            # A fire-time audio failure must NOT propagate: the scheduler loop has
            # no except around on_fire, so an exception here would kill the thread
            # and silently stop every future alarm.
            print(f"[fire] audio failed for {ev.kind} #{ev.id}: {exc!r}")
        finally:
            shared.set(AppState.IDLE)

    scheduler = Scheduler(store, on_fire)
    scheduler.start()

    try:
        run_ui(shared, pipeline.on_tap, fullscreen=cfg["ui"]["fullscreen"],
               fps=cfg["ui"]["fps"])
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
