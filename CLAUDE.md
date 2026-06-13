# raspberry-pi-ai-assitant — project memory

## What this is
A desk companion robot: USB webcam, USB mic, speakers, and a screen showing
an animated pixel face. It perceives presence and faces, holds spoken
conversations, has a deterministic emotion engine, persistent SQLite memory,
an idle inner life, and a nightly sleep cycle that consolidates,
deduplicates, and prunes its memory ("dreaming"). It develops on a laptop
(ASSISTANT_PROFILE=laptop) and runs on a Raspberry Pi 5 (ASSISTANT_PROFILE=pi).

## Repo facts
- Distribution / repo name: raspberry-pi-ai-assitant.
- Import package: src/assistant/ (hyphens are invalid in imports).
- Console scripts: assistant, assistant-enroll, assistant-nightjob,
  assistant-memory, assistant-doctor.
- The `legacy` branch holds the pre-rebuild project. Never read, merge, or
  copy from it. All work happens on main and feature branches off main.

## Architecture: System 1 / System 2
- System 1 (reflexes): everything visible reacts instantly and locally —
  face state changes, gaze tracking, blinks, emotion drift. Never blocked by
  network or LLM calls.
- System 2 (cognition): LLM calls via OpenRouter for conversation, idle
  thoughts, and night jobs. Always async, never on the render path.
- The face must never freeze. If any instruction conflicts with this rule,
  this rule wins.

## Modules and communication
Modules: face, perception, voice, brain, memory, scheduler.
They communicate ONLY through the async event bus (src/assistant/bus.py).
No module imports another module's internals.

Canonical bus events — never invent new names without updating this list:
- person_appeared, person_left
- face_recognized {name, score}, face_unknown
- gaze {x, y}                      (normalized 0..1, throttled)
- speech_heard {text}
- speaking_started, speaking_finished
- idle_tick, sleep_start, sleep_end
Command events (published by brain, consumed by face/voice):
- face_state {state}, face_gaze {x, y}, say {text}

Face states: sleeping, drowsy, neutral, alert, listening, thinking,
speaking, happy, curious.

Emotion vector (brain-owned, deterministic): energy, mood, curiosity,
social — floats in [0, 1].

## Conventions
- Python 3.11+, uv, pyproject.toml, src layout, type hints, dataclasses.
- pygame render loop owns the MAIN thread at 60 fps. Everything else runs in
  an asyncio loop on a background thread; blocking work (camera, STT, TTS,
  LLM, embeddings) goes to worker threads/executors.
- All tunables live in config/laptop.yaml and config/pi.yaml, selected by
  ASSISTANT_PROFILE (default: laptop). No magic numbers in code.
- data/ (gitignored) holds brain.db and runtime state.
  models/ (gitignored) holds downloaded model files.
  .env (gitignored) holds OPENROUTER_API_KEY.
- pytest covers pure logic (bus, emotion math, memory operations). Hardware
  paths get manual acceptance steps documented in README.md.
- Commit at the end of every slice with a clear message.

## Hardware profiles
- laptop: built-in webcam/mic/speakers, windowed 800x480 display.
- pi: Pi 5 in a Pironman case — USB webcam, USB mic, case speakers, DSI
  touchscreen fullscreen (target 800x480; verify real resolution on device).

## Slice status (update as slices land)
- [x] 0 scaffold
- [x] 1 face
- [x] 2 perception
- [x] 3 voice
- [ ] 4 brain v1 (emotion + conversation)
- [ ] 5 memory
- [ ] 6 inner life (idle thoughts + sleep cycle)
- [ ] 7 pi deployment
