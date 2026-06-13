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

## Face enrollment

Teach the assistant your face (10 webcam samples, up to 30 s):

```sh
uv run assistant-enroll <name>          # add --show for a camera preview
```

The embedding lands in `data/brain.db`; the running assistant picks up new
enrollments at its next identity vote. `uv run assistant --show` opens an
OpenCV debug window with the detection box and the decided name. Models
(YuNet detector, SFace recognizer) auto-download to `models/` on first use.

## Legacy code

The pre-rebuild project lives on the `legacy` branch. It is kept for
reference only — nothing is read, merged, or copied from it.

## Acceptance log

Manual checks for hardware/UI paths that pytest cannot cover. Record each
slice's checks here as it lands.

### Slice 2 — perception (2026-06-13)

Verified without a person in frame (run at night, nobody at the desk):

- [x] `uv run pytest` green (55 tests: +30 for cosine matching, majority
      vote, re-verify timing, presence debounce, gaze throttle).
- [x] First `uv run assistant` auto-downloads both models to `models/`
      (YuNet 232 KB, SFace 38.7 MB — both pass the >100 KB LFS-pointer
      check) and creates `data/brain.db` with the `people` table.
- [x] Camera opens via V4L2 (logged `640x480 @ 30 fps` — the cam ignores
      the 15 fps request; properties are advisory, detection cadence is
      per-frame so behavior is unaffected).
- [x] Vision pipeline verified offline against a sample face photo:
      YuNet detects (score 0.91), SFace yields a 128-dim float32 embedding,
      enrollment-style normalized mean round-trips through SQLite,
      self-match scores 0.99 (threshold 0.363), random noise does not match.
- [x] `uv run assistant-enroll testperson` with nobody in frame exits 1
      after the 30 s timeout with "no face captured" and stores nothing.
- [x] `uv run assistant --show` opens the OpenCV debug window next to the
      pygame face (Qt-over-XWayland warnings are harmless); SIGINT still
      shuts down cleanly with no Python errors.
- [x] App keeps running (face animates) regardless of perception state.

Still needs a human in frame (run these when at the desk):

- [ ] Sitting down: `person_appeared` within ~1 s, face goes alert, pupils
      follow you via throttled `gaze` events; `face_unknown` logged once.
- [ ] Leaving for 25 s: `person_left`, face goes drowsy.
- [ ] `uv run assistant-enroll <yourname>` prints `sample i/10` progress and
      stores a row; next `uv run assistant` logs
      `face_recognized {name, score}` once per decision (re-verify ~90 s).
- [ ] `--show` window draws the detection box with your decided name.

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
