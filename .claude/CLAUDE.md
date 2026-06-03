# CLAUDE.md — Desk Assistant

Steering doc for Claude Code. Read this fully before writing or changing code.

## What this is
A local AI desk assistant on a Raspberry Pi 5 (8GB) in a Pironman 5 Pro Max case.
Pixel "eyes" on the 4.3" DSI touchscreen give visual feedback. You tap the screen
to talk to it; it sets alarms/reminders/timers locally and answers open questions
via a cheap cloud LLM.

## v1 scope — DO NOT EXCEED
Tap screen → eyes LISTENING → record until silence → THINKING → then either:
- a local command (alarm / timer / reminder / time / date) → store it → CONFIRM
  (happy eyes) + spoken confirmation, **or**
- anything else → QUERY → cheap LLM via OpenRouter → spoken answer (SPEAKING)
→ back to IDLE. Scheduled events fire on time and survive a reboot.

**Explicitly OUT of v1** (do not build unless asked): wake word, acoustic echo
cancellation, multi-turn conversation, rich recurring schedules, any GUI beyond
the eyes. Resist scope creep.

## The one architectural rule (non-negotiable)
`assistant/state.py::AppState` is the single source of truth.
- The UI/eyes ONLY READ state (`SharedState.snapshot()`/`.state`). They contain
  no logic and never mutate state.
- The pipeline/scheduler ONLY WRITE state. They never draw.
If you're tempted to put logic in the eyes or drawing in the pipeline, stop —
you're breaking the design.

## Concurrency model
- **Main thread:** pygame loop (`assistant/ui.py`) — renders eyes, polls touch.
  Must never block. The tap handler kicks work onto a worker thread and returns.
- **Worker thread:** the pipeline (`assistant/pipeline.py`) — record → STT →
  intent → act → TTS. Writes AppState as it goes. One at a time (busy lock).
- **Scheduler thread:** (`assistant/scheduler.py`) — polls the DB for due events.
Shared state is behind a lock in `SharedState`. Don't add global mutable state.

## Stack (locked — don't substitute without a reason)
- UI: pygame, fullscreen on the 800x480 DSI. Eyes = rects on a grid.
- STT: whisper.cpp via `pywhispercpp`, `base.en`.
- TTS: Piper.
- LLM: OpenRouter (OpenAI-compatible REST), cheap model, fallback model.
- Storage: SQLite (stdlib `sqlite3`).
- Config: `config.toml` (TOML). Secrets via env (`OPENROUTER_API_KEY`) only.

## Layout
```
run.py                  entrypoint; wires everything, runs UI on main thread
config.example.toml     copy to config.toml (gitignored) and fill in
assistant/
  state.py      AppState + thread-safe SharedState        [DONE, tested]
  intent.py     local rule parser -> Intent               [DONE, tested]
  store.py      SQLite events (alarm/reminder/timer)       [DONE]
  scheduler.py  polls store, fires due events              [DONE]
  eyes.py       pygame pixel-eye renderer (reads state)    [DONE, tune on Pi]
  ui.py         pygame loop, fullscreen, tap trigger       [DONE, tune on Pi]
  pipeline.py   listen->think->act->speak orchestration    [WIRED, needs audio]
  brain.py      OpenRouter LLM fallback                    [WIRED, needs key]
  audio_io.py   record + playback                          [STUB — do on Pi]
  stt.py        whisper.cpp wrapper                         [STUB — do on Pi]
  tts.py        Piper wrapper                               [STUB — do on Pi]
tests/test_intent.py    pytest for the parser              [DONE, 9 passing]
```

## Build plan — vertical slices, test each ON THE PI before the next
0. **Hardware sanity (done):** `arecord` from USB mic + `aplay` through the case
   speakers. Confirm the speaker/HDMI/3.5mm jumper is set for speakers.
1. **Face:** `python -m assistant.ui` shows the eyes; tapping/keys cycle states.
   Tune CELL size, colours, glow, proportions on the real panel.
2. **Capture + STT:** implement `audio_io.record_until_silence` + `stt.transcribe`.
   Tap → record → print transcript to console. No actions yet.
3. **Commands + persistence:** route transcript through `intent.parse`, write to
   `store`, let `scheduler` fire. Verify an alarm fires AND survives a restart.
4. **Voice out:** implement `tts.speak` (Piper). Confirmations + announcements
   are now spoken. Eyes show CONFIRM/SPEAKING.
5. **LLM fallback:** set `OPENROUTER_API_KEY`, finish `brain.ask`. QUERY intents
   get spoken answers; network failure degrades gracefully (it must not hang).

## Conventions
- Python 3.11+, type hints, small pure functions where possible.
- No network in the command path — alarms must work fully offline.
- Never commit `config.toml`, `*.db`, models, or keys (see `.gitignore`).
- LLM calls: always a timeout + max_tokens; degrade to a spoken fallback line.
- Add a test when you add parser cases. Keep `pytest` green.

## Hardware notes (Pironman 5 Pro Max)
- Pi 5 has no analog jack — audio is via the case adapter; speakers conflict with
  HDMI0 + 3.5mm (jumper).
- USB mic dongle uses one USB 2.0 port; it's a single far-field mic (expect rough
  noise rejection — that's why echo cancellation/wake word are deferred).
- Use the RTC battery so the clock is correct after a power loss.
- Cooling is strong; sustained CPU is fine. Use the 27W PSU.
