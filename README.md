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

### Slice 0 — scaffold (2026-06-13)

- [x] `uv run pytest` green (4 tests).
- [x] `uv run assistant` opens an 800x480 window with a placeholder circle.
- [x] Heartbeat events logged every 5 s with timestamps.
- [x] Ctrl+C (SIGINT) exits cleanly ("clean shutdown" logged, exit code 0).
- [x] Window close (QUIT event) exits cleanly.
