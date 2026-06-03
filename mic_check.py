"""THROWAWAY mic sanity check — NOT part of slice 2, delete when done.

Run ON THE PI:  python mic_check.py
Proves the USB mic is the right device and capturing real signal (not silence).
If the default device is wrong, read the printed list and set DEVICE to the mic's
index (or a substring of its name), then re-run.
"""
import numpy as np
import sounddevice as sd

SR, SECS = 16_000, 3
DEVICE = None  # None = default; set to the USB mic's index or name if wrong

print(sd.query_devices())  # <- find your USB mic here
print(f"\nRecording {SECS}s @ {SR} Hz from device={DEVICE or 'default'} ... speak now")
audio = sd.rec(int(SECS * SR), samplerate=SR, channels=1, dtype="float32", device=DEVICE)
sd.wait()
audio = audio[:, 0]

rms = float(np.sqrt(np.mean(audio**2)))
peak = float(np.abs(audio).max())
print(f"samplerate={SR}  samples={audio.size}  peak={peak:.4f}  rms={rms:.5f}")
print("LOOKS SILENT — wrong device or mic muted?" if rms < 0.005 else "Signal present. Mic is live.")
