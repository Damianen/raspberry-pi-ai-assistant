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

### Slice 3 — voice (2026-06-13)

- [x] `uv run pytest` green (80 tests: +25 for sentence splitting, the
      half-duplex gate, VAD utterance segmentation, and blocking Inbox.get).
- [x] First run auto-downloads the piper voice `en_US-lessac-medium`
      (63.2 MB) to `models/piper/` and faster-whisper `small` to
      `models/whisper/`; subsequent startups load from disk and the mic is
      live ~2 s after launch ("listening on 'default'").
- [x] Offline pipeline check: piper synthesizes 2.9 s of speech in 0.11 s;
      the streaming Silero VAD (v6 ONNX bundled with faster-whisper — no
      torch) peaks at 0.994 on speech vs 0.076 on room noise; the segmenter
      brackets the utterance (onset 88 ms into speech, covered by the
      200 ms pre-roll); whisper transcribes it back verbatim with
      no-speech probability 0.004.
- [x] End-to-end loop on a zero-attenuation digital loopback (PipeWire
      null sink as default output, its monitor as default mic — the robot's
      own voice feeds straight back into its mic at full level):
      `paplay` of a spoken phrase → `heard 'Testing 123. Can you hear me?'`
      (1.2 s transcribe) → `say` echo → `speaking_started` →
      sentence-by-sentence playback ("You said: Testing 123." then
      "Can you hear me?") → `speaking_finished`. Repeated three times.
- [x] No feedback loop: across all echo cycles the robot never transcribed
      its own reply (no "You said: You said:" — zero `speech_heard` during
      or after playback), even with its voice looped back at 0 dB.
- [x] Half-duplex semantics: a phrase played while the robot was replying
      was discarded entirely (no partial transcript), as designed — no
      barge-in until a future slice.
- [x] `face_state speaking` published when playback starts; placeholder
      sets neutral on `speaking_finished`. Face kept animating throughout
      (whisper transcription runs off the render thread).
- [x] Clean SIGINT shutdown with the voice module running (both sessions
      logged "clean shutdown"; audio devices released).
- [ ] Real-microphone conversation at the desk: speak, hear the echo from
      the actual speakers, confirm the mic ignores them. (The laptop's
      default sink is a headphone DAC the Yeti mic cannot hear, so the
      acoustic path needs speakers — verify when on the Pi or with desk
      speakers selected.)

### Slice 2 — perception (2026-06-13)

- [x] `uv run pytest` green (55 tests: +30 for cosine matching, majority
      vote, re-verify timing, presence debounce, gaze throttle).
- [x] First `uv run assistant` auto-downloads both models to `models/`
      (YuNet 232 KB, SFace 38.7 MB — both pass the >100 KB LFS-pointer
      check) and creates `data/brain.db` with the `people` table.
- [x] Dock webcam (Logitech C920) opens via its stable /dev/v4l/by-id path
      at `640x480 @ 15 fps` — `camera.device` beats `camera.index` because
      V4L2 indexes shuffle between boots and the closed laptop's lid cam
      sees nothing. (The lid cam also worked but negotiated 30 fps.)
- [x] Vision pipeline verified offline against a sample face photo:
      YuNet detects (score 0.91), SFace yields a 128-dim float32 embedding,
      enrollment-style normalized mean round-trips through SQLite,
      self-match scores 0.99 (threshold 0.363), random noise does not match.
- [x] Sitting at the desk: `person_appeared` ~1.3 s after start, face goes
      alert (window title `assistant — alert`), then `face_unknown` exactly
      once with nobody enrolled.
- [x] Gaze events flow at 5/s (under the 150 ms throttle cap) with sane
      normalized coordinates that match the seating position.
- [x] `uv run assistant-enroll damian` printed `sample 1..10/10` in a few
      seconds and stored one row; next run logged
      `face_recognized {name: damian, score: 0.79}` and updated last_seen.
- [x] `uv run assistant-enroll` with nobody in frame exits 1 after the 30 s
      timeout with "no face captured" and stores nothing.
- [x] `uv run assistant --show` opens the OpenCV debug window next to the
      pygame face (Qt-over-XWayland warnings are harmless); screenshot shows
      the green detection box labeled `damian`. Clean SIGINT shutdown.
- [x] App keeps running (face animates) regardless of perception state.
- [ ] Leaving the desk for 25 s: `person_left`, face goes drowsy. (Needs an
      actual absence — verify on a normal break.)

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
