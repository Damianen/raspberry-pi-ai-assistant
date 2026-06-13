# raspberry-pi-ai-assitant

A desk companion robot with an animated pixel face: it sees you through a
webcam, talks with you through a mic and speakers, runs a deterministic
emotion engine, remembers things in SQLite, and dreams at night. Develops on
a laptop, deploys to a Raspberry Pi 5.

## Install

```sh
uv sync
```

## Run

```sh
uv run assistant
```

## Profiles

Configuration comes from `config/<profile>.yaml`, selected by the
`ASSISTANT_PROFILE` environment variable (default: `laptop`):

```sh
ASSISTANT_PROFILE=pi uv run assistant
```

## Legacy code

The pre-rebuild project lives on the `legacy` branch. It is kept for
reference only — nothing is read, merged, or copied from it.

## Acceptance log

Manual checks for hardware/UI paths that pytest cannot cover. Record each
slice's checks here as it lands.

### Slice 1 — procedural face (2026-06-13)

- [x] `uv run pytest` green (25 tests: bus + face logic/styles/frame/config).
- [x] `uv run assistant` shows two rounded-rect eyes with irises and
      highlights on a dark background; window title shows the current state.
- [x] Keys 1–9 switch through all nine states (verified via
      `hyprctl dispatch sendshortcut` + screenshots); every state is visually
      distinct: sleeping closed lids, drowsy half-lids, neutral cyan, alert
      bright/enlarged, listening teal, thinking narrowed violet with upward
      gaze, speaking warm amber, happy golden upward arcs, curious pink with
      one larger tilted eye. Transitions ease without pops.
- [x] Mouse position moves the pupils smoothly (verified by moving the
      cursor to window corners via `hyprctl dispatch movecursor`).
- [x] Idle wander: pupils drift on their own after 5 s without gaze input.
- [x] Blinks close and reopen smoothly (screenshot burst caught a full arc:
      eye height 201 → 119 → 49 → 8 → 85 → 201 px). B key delivered without
      side effects; forced-blink logic covered by pytest.
- [x] Sleeping breathes: closed-lid width oscillates at ~0.1 Hz.
- [x] Mouse motion publishes throttled `face_gaze` events logged at DEBUG
      only — no INFO spam; `face_state` events visible at INFO.
- [x] Window close and SIGTERM both exit cleanly ("clean shutdown" logged).

### Slice 0 — scaffold (2026-06-13)

- [x] `uv run pytest` green (4 tests).
- [x] `uv run assistant` opens an 800x480 window with a placeholder circle.
- [x] Heartbeat events logged every 5 s with timestamps.
- [x] Ctrl+C (SIGINT) exits cleanly ("clean shutdown" logged, exit code 0).
- [x] Window close (QUIT event) exits cleanly.
