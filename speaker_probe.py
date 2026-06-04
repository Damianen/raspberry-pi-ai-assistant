"""Speaker diagnostic — run on the Pi:  python speaker_probe.py

The output-side counterpart to mic_probe.py. It finds the right device for
[audio].output_device in config.toml, and isolates the alarm-fire -> beep path
from speech recognition so you can tell *which* half is broken.

It uses the app's OWN audio_io.beep(), so whatever works here works in run.py.

  python speaker_probe.py              # play a chime on the default output, then
                                       # on each output device in turn; you tell
                                       # it which one you hear -> that goes in config
  python speaker_probe.py --fire-test  # insert a real timer due in 30s, then run
                                       #   python run.py
                                       # and wait. If you hear the beep, the whole
                                       # fire->beep path works and ONLY voice
                                       # recognition is the problem.

Why this exists: "can't reach the internet" means your command fell through to
QUERY, so no event was stored, so nothing ever fires, so beep() never runs. The
beep being silent in the program is almost always "no event fired", not audio.

Throwaway diagnostic; delete once audio works.
"""
from __future__ import annotations

import sys

import sounddevice as sd

from assistant import audio_io


def list_devices() -> list[int]:
    print("=== all devices (sounddevice / PortAudio) ===")
    print(sd.query_devices())

    devs = sd.query_devices()
    outs = [i for i, d in enumerate(devs) if d["max_output_channels"] > 0]
    print(f"\n=== output-capable devices (max_output_channels > 0): {outs} ===")
    try:
        default_out = sd.default.device[1]
        print(f"=== current PortAudio default output: index {default_out} "
              f"({sd.query_devices(default_out)['name']!r}) ===")
        print("    NOTE: this default is NOT necessarily ALSA's default — that's "
              "why `aplay` working doesn't prove this path works.")
    except Exception as e:
        print(f"(could not read default output: {e!r})")
    return outs


def walk_outputs(outs: list[int]) -> None:
    print("\n--- Playing a 1.5s chime on the DEFAULT output ---")
    try:
        audio_io.beep()
        print("  If you heard that, `output_device = \"\"` (empty) in config works.")
    except Exception as e:
        print(f"  default beep FAILED: {e!r}")

    for dev in outs:
        name = sd.query_devices(dev)["name"]
        try:
            input(f"\n[{dev}] {name!r} — press Enter to chime THIS device (Ctrl-C to stop)...")
        except (EOFError, KeyboardInterrupt):
            print("\nstopped.")
            break
        try:
            audio_io.beep(device=str(dev))
            print(f"  -> played on device {dev}.")
        except Exception as e:
            print(f"  -> device {dev} FAILED: {e!r}")

    print("\nDone. Put the index (or a name substring) of the device you HEARD into "
          "config.toml:\n"
          "    [audio]\n"
          '    output_device = "<that index>"   # or "" to use the default\n')


def fire_test(seconds: int = 30) -> None:
    from datetime import datetime, timedelta

    from assistant.store import Store

    # Uses the same Store/format the app uses, so this is a faithful fire test.
    eid = Store("assistant.db").add(
        "timer", datetime.now() + timedelta(seconds=seconds), "")
    print(f"Inserted timer #{eid} into assistant.db, fires in ~{seconds}s.")
    print("Now run:   python run.py   and wait. Expected at fire time:")
    print("  - you HEAR the beep")
    print("  - console shows:  TTS(stub): Timer's up.")
    print("If that works, the fire->beep path is fine and the ONLY problem is "
          "voice -> intent recognition (check the [pipeline] transcript/intent "
          "lines when you speak a command).")
    print("If it's silent, check the console for:  [fire] audio failed ...")
    print("\n(Make sure run.py's [storage].db_path points at this same assistant.db.)")


if __name__ == "__main__":
    if "--fire-test" in sys.argv:
        fire_test()
    else:
        walk_outputs(list_devices())
