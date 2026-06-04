"""Mic diagnostic — run on the Pi:  python mic_probe.py

Finds every input-capable device, probes which sample rates each one accepts,
then records a short clip from each to confirm it captures real signal (not
silence). Use the output to decide:
  - if the USB mic shows "16000 Hz: OK" -> set [audio] input_device to its index
    in config.toml; no code change needed.
  - if 16000 is "no" but a higher rate is OK -> the mic can't capture at 16k and
    audio_io needs a capture-at-native-then-downsample fix. Send me this output.

Throwaway diagnostic; delete once audio works.
"""
import numpy as np
import sounddevice as sd

RATES = (16000, 22050, 32000, 44100, 48000)
RECORD_SECS = 3

print("=== all devices ===")
print(sd.query_devices())

devices = sd.query_devices()
inputs = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
print(f"\n=== input-capable devices: {inputs} ===")
if not inputs:
    raise SystemExit("No input devices found — is the USB mic plugged in?")

for dev in inputs:
    d = sd.query_devices(dev)
    print(f"\n[{dev}] {d['name']!r}  (native {d['default_samplerate']:.0f} Hz, "
          f"{d['max_input_channels']} in)")
    ok = []
    for sr in RATES:
        try:
            sd.check_input_settings(device=dev, samplerate=sr, channels=1, dtype="float32")
            ok.append(sr)
            print(f"    {sr:>6} Hz: OK")
        except Exception:
            print(f"    {sr:>6} Hz: no")
    if not ok:
        print("    (no usable rate — skipping record)")
        continue

    sr = 16000 if 16000 in ok else ok[0]   # prefer 16k, else lowest that works
    print(f"    -> recording {RECORD_SECS}s @ {sr} Hz ... speak now")
    try:
        audio = sd.rec(int(RECORD_SECS * sr), samplerate=sr, channels=1,
                       dtype="float32", device=dev)
        sd.wait()
        audio = audio[:, 0]
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.abs(audio).max())
        verdict = "SILENT — wrong device or muted?" if rms < 0.005 else "signal present."
        print(f"    rms={rms:.5f}  peak={peak:.4f}  -> {verdict}")
    except Exception as e:
        print(f"    record failed: {e}")
