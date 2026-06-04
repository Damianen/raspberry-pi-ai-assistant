"""Audio capture + playback.

Slice 2: capture is IMPLEMENTED; `play()` is still a stub (do it in the TTS slice).

`record_until_silence` returns a 1-D numpy float32 array, 16 kHz mono, in [-1, 1]
(PortAudio's native float range — we do NOT peak-normalize; whisper.cpp wants
natural levels, and rescaling would just amplify room noise). Endpointing is a
simple energy threshold + trailing-silence timeout, with a fixed-window fallback
so the slice always completes even if the mic never trips the threshold.

The USB mic can't capture at 16 kHz (only 44.1k/48k), and pywhispercpp does NOT
resample an ndarray you pass it — so we capture at CAPTURE_RATE (48 kHz, an exact
3:1 ratio to 16 kHz) and downsample in software. The downsample is a windowed-sinc
low-pass + stride: the low-pass is mandatory, because a naive `[::3]` would alias
everything from 8–24 kHz (incl. broadband room noise) down into the speech band
and wreck transcription. Fixed 3:1 only — a 44.1k-only mic would need rework.

`sounddevice` is imported LAZILY inside the function: importing it loads native
PortAudio, which is absent on dev machines. Keeping the import local means this
module (and the offline command path + tests) load fine without an audio stack.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable

import numpy as np

# --- Rates. We capture at CAPTURE_RATE and return TARGET_RATE (what whisper wants). ---
TARGET_RATE = 16_000
CAPTURE_RATE = 48_000          # the USB mic supports this; 48k/16k = exact 3:1
CHANNELS = 1
assert CAPTURE_RATE % TARGET_RATE == 0, "decimator only handles integer rate ratios"
DECIM = CAPTURE_RATE // TARGET_RATE   # 3

FRAME_MS = 30                                       # energy-analysis frame length
FRAME_SAMPLES = CAPTURE_RATE * FRAME_MS // 1000     # 1440 samples/frame @ 48 kHz
_FRAME_SEC = FRAME_MS / 1000.0                      # rate-independent frame duration

# --- TUNE THESE ON THE MIC (single far-field USB mic, expect a noisy room) ---
# RMS is computed on a float32 frame in [-1, 1]. START_RMS must be > SILENCE_RMS:
# that gap is hysteresis, so a frame hovering near the threshold can't flip us
# between "speech" and "silence" every 30 ms. (Thresholds are rate-independent.)
START_RMS = 0.020            # frame energy that counts as "speech has started"
SILENCE_RMS = 0.010          # frame energy below this counts as silence
TRAILING_SILENCE_SEC = 1.2   # stop once we see this much silence after speech
PREROLL_SEC = 0.3            # audio kept BEFORE onset so word 1 isn't clipped
LISTEN_TIMEOUT_SEC = 6.0     # no speech within this -> fixed-window fallback
FALLBACK_SECONDS = 5.0       # length of that fixed-window fallback

# --- Anti-aliasing low-pass for the 48k->16k decimation (built once at import) ---
_DECIM_TAPS = 127            # odd; Blackman-windowed sinc
_DECIM_CUTOFF_HZ = 7600      # just under the 8 kHz output Nyquist


def _design_lowpass(num_taps: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """Windowed-sinc low-pass FIR, unity DC gain, linear phase."""
    m = num_taps - 1
    n = np.arange(num_taps)
    h = np.sinc(2.0 * (cutoff_hz / fs) * (n - m / 2.0)) * np.blackman(num_taps)
    return (h / h.sum()).astype(np.float32)


_DECIM_FIR = _design_lowpass(_DECIM_TAPS, _DECIM_CUTOFF_HZ, CAPTURE_RATE)

# --- Beep (slice 3): a short sine chime so a firing alarm is audible before the
# spoken announcement exists. Also the first thing to actually exercise the
# OUTPUT path in code. 44.1 kHz is universally supported by output devices. ---
BEEP_RATE = 44_100     # fallback rate only; we use the device's advertised rate
BEEP_FREQ = 880.0      # A5 — clearly audible, not piercing
BEEP_AMPLITUDE = 0.3   # well below clipping; speakers in a quiet room


def record_until_silence(
    max_seconds: float = 8.0, device: int | str | None = None
) -> np.ndarray:
    """Record one utterance; return 16 kHz mono float32 in [-1, 1].

    Capture begins when frame energy crosses START_RMS (keeping PREROLL_SEC of
    lead-in) and stops after TRAILING_SILENCE_SEC of trailing silence OR at
    max_seconds, whichever comes first. If no speech starts within
    LISTEN_TIMEOUT_SEC, falls back to a fixed FALLBACK_SECONDS window and says so.
    Audio is captured at CAPTURE_RATE and downsampled to TARGET_RATE before return.

    `device` may be a PortAudio index, a name substring, or None (default device).
    """
    import sounddevice as sd  # lazy: needs native PortAudio (absent on dev boxes)

    device = _resolve_device(device)
    info = sd.query_devices(device, "input")
    label = "default" if device is None else device
    print(f"[audio] input device [{label}]: {info['name']!r} "
          f"(capturing {CAPTURE_RATE} Hz -> downsampling to {TARGET_RATE} Hz)")

    preroll_frames = max(1, round(PREROLL_SEC / _FRAME_SEC))
    preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
    collected: list[np.ndarray] = []
    started = False
    silence_sec = 0.0
    waited = 0.0

    with sd.InputStream(samplerate=CAPTURE_RATE, channels=CHANNELS, dtype="float32",
                        device=device, blocksize=FRAME_SAMPLES) as stream:
        while True:
            frame, _overflow = stream.read(FRAME_SAMPLES)
            frame = frame[:, 0]                       # (N, 1) -> (N,) mono
            rms = float(np.sqrt(np.mean(frame * frame)))

            if not started:
                preroll.append(frame)                 # rolling lead-in buffer
                waited += _FRAME_SEC
                if rms >= START_RMS:
                    started = True
                    collected.extend(preroll)         # keep the lead-in + onset
                elif waited >= LISTEN_TIMEOUT_SEC:
                    print(f"[audio] no speech in {LISTEN_TIMEOUT_SEC:.0f}s — "
                          f"falling back to a fixed {FALLBACK_SECONDS:.0f}s window.")
                    collected.extend(preroll)
                    _fill_fixed_window(stream, collected)
                    break
                continue

            collected.append(frame)
            if rms < SILENCE_RMS:
                silence_sec += _FRAME_SEC
                if silence_sec >= TRAILING_SILENCE_SEC:
                    break
            else:
                silence_sec = 0.0
            if len(collected) * _FRAME_SEC >= max_seconds:
                break

    if not collected:
        return np.zeros(0, dtype=np.float32)
    captured = np.concatenate(collected).astype(np.float32, copy=False)
    return _downsample(captured)


def _downsample(audio: np.ndarray) -> np.ndarray:
    """Anti-aliased 48k -> 16k (low-pass then 3:1 stride). Returns float32 in [-1,1]."""
    filtered = np.convolve(audio, _DECIM_FIR, mode="same")
    out = filtered[::DECIM].astype(np.float32, copy=False)
    return np.clip(out, -1.0, 1.0)


def _fill_fixed_window(stream, collected: list[np.ndarray]) -> None:
    """Top up `collected` to ~FALLBACK_SECONDS total (includes existing preroll)."""
    target = round(FALLBACK_SECONDS / _FRAME_SEC)
    while len(collected) < target:
        frame, _overflow = stream.read(FRAME_SAMPLES)
        collected.append(frame[:, 0])


def _resolve_device(device: int | str | None) -> int | str | None:
    """Normalize config's device value: ''/None -> default; digit-string -> index."""
    if device is None:
        return None
    if isinstance(device, str):
        s = device.strip()
        if not s:
            return None
        return int(s) if s.isdigit() else s
    return device


# --- Output path. beep() and play() (TTS) share ONE device path and ONE lock so
# a chime, an alarm announcement, and a spoken answer can never open the single
# PortAudio output stream at the same time and garble each other. The lock means
# an alarm that fires mid-answer WAITS for the answer to finish rather than
# talking over it — the right call for a one-speaker desk device. ---
_PLAY_LOCK = threading.Lock()


def _resolve_output(device: int | str | None):
    """Resolve the output device and read ITS native (rate, channels).

    Channel count and sample rate come FROM THE DEVICE, not assumed: many outputs
    (HDMI, USB DACs) reject a mono stream (PaErrorCode -9998) or a fixed rate
    (-9997). Callers build/resample audio to these values so it plays on whatever
    is actually openable.
    """
    import sounddevice as sd  # lazy: needs native PortAudio (absent on dev boxes)

    device = _resolve_device(device)
    info = sd.query_devices(device, "output")
    rate = int(info["default_samplerate"]) or BEEP_RATE
    channels = max(1, min(2, int(info["max_output_channels"])))   # 1 or 2; prefer stereo
    return device, rate, channels


def _play_blocking(mono: np.ndarray, rate: int, channels: int,
                   device: int | str | None) -> None:
    """Tile a mono float32 signal to `channels` and play it (blocking, serialized)."""
    import sounddevice as sd  # lazy: needs native PortAudio (absent on dev boxes)

    wave = np.tile(mono[:, None], (1, channels))   # mono column -> (n, channels)
    with _PLAY_LOCK:
        sd.play(wave, samplerate=rate, device=device)
        sd.wait()


# Interruptible playback (tap-to-interrupt during SPEAKING). We check the stop
# event between fixed-size chunks; ~100 ms bounds the tap-to-silence latency to a
# single chunk while keeping the per-write Python overhead negligible.
_INTERRUPT_CHUNK_MS = 100


def _play_interruptible(mono: np.ndarray, rate: int, channels: int,
                        device: int | str | None,
                        stop_event: "threading.Event",
                        level_cb: "Callable[[np.ndarray], None] | None" = None) -> None:
    """Play a mono float32 signal in chunks, bailing the instant `stop_event` is set.

    Backs the interruptible SPEAKING path: a tap sets the event, and at most the
    current chunk still plays before we ABORT — abort discards everything PortAudio
    has buffered, so the speaker goes silent at once instead of draining a second
    of queued story. On a normal finish we stop() (which drains the queued tail)
    before closing, so the last words aren't clipped.

    `level_cb`, if given, is called with each chunk's MONO samples just before it's
    queued — that's where TTS turns playback into the live mouth-sync level (the
    eyes read it back via SharedState). We hand it the pre-tiled mono so the caller
    sees the true signal, and call it before write() so the mouth opens roughly as
    the chunk becomes audible. A callback raising would kill playback, so the chunk
    size (_INTERRUPT_CHUNK_MS) bounds how often it runs; keep the callback cheap.

    Shares _PLAY_LOCK with beep()/_play_blocking, so two streams can never open the
    single output device at the same time. We drive an OutputStream directly here
    (rather than sd.play/sd.wait) precisely so the write loop can poll the event;
    the non-interruptible paths keep the proven sd.play/sd.wait path unchanged.
    """
    import sounddevice as sd  # lazy: needs native PortAudio (absent on dev boxes)

    wave = np.tile(mono[:, None], (1, channels))   # mono column -> (n, channels)
    chunk = max(1, int(rate * _INTERRUPT_CHUNK_MS / 1000))
    with _PLAY_LOCK:
        stream = sd.OutputStream(samplerate=rate, channels=channels,
                                 dtype="float32", device=device)
        stream.start()
        try:
            for start in range(0, len(wave), chunk):
                if stop_event.is_set():
                    stream.abort()     # drop buffered frames -> instant silence
                    return
                if level_cb is not None:
                    level_cb(mono[start:start + chunk])
                stream.write(wave[start:start + chunk])
            stream.stop()              # drain the queued tail before closing
        finally:
            stream.close()


def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a mono float32 signal by linear interpolation.

    Used to match Piper's voice rate (e.g. 22050) to the output device's native
    rate (e.g. 48000). We only ever UPSAMPLE here (device rate >= voice rate),
    where linear interp adds NO aliasing and only a benign top-octave roll-off —
    inaudible on 3W speakers, and far cheaper than a polyphase sinc on the Pi.
    If a device ever advertised a LOWER rate than the voice, this would alias and
    we'd need an anti-alias low-pass first (see the 48k->16k capture path); guard
    for that day, don't build it now.
    """
    if x.size == 0 or src_rate == dst_rate:
        return x
    n_out = int(round(x.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    # Output sample i maps to source position i * src_rate / dst_rate (in samples).
    pos = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    grid = np.arange(x.size, dtype=np.float64)
    return np.interp(pos, grid, x).astype(np.float32)


def beep(duration: float = 1.5, device: int | str | None = None,
         freq: float = BEEP_FREQ) -> None:
    """Play a short sine chime on the output device (blocking).

    `device` takes the same forms as record_until_silence (PortAudio index, name
    substring, or ''/None for the default). Called by the scheduler's on_fire so
    a due alarm is audible before its spoken announcement. Blocking is intentional:
    it runs on the scheduler thread, which has nothing else to do meanwhile.

    The tone is built at the device's advertised rate (so no resample is needed)
    and routed through the shared playback path.
    """
    device, rate, channels = _resolve_output(device)

    n = max(1, int(rate * duration))
    t = np.arange(n, dtype=np.float32) / rate
    tone = (BEEP_AMPLITUDE * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    # 10 ms raised-cosine fade in/out so the tone doesn't click on start/stop.
    fade = min(n // 2, max(1, int(rate * 0.01)))
    ramp = (0.5 * (1 - np.cos(np.linspace(0.0, np.pi, fade)))).astype(np.float32)
    tone[:fade] *= ramp
    tone[-fade:] *= ramp[::-1]

    _play_blocking(tone, rate, channels, device)


def play(samples: np.ndarray, src_rate: int,
         device: int | str | None = None, *,
         stop_event: "threading.Event | None" = None,
         level_cb: "Callable[[np.ndarray], None] | None" = None) -> None:
    """Play a mono float32 signal in [-1, 1] through the output device (blocking).

    Resamples from `src_rate` to the device's native rate and tiles to its channel
    count — the SAME proven path as beep(). This is what tts.speak() hands its
    synthesized PCM to; tts never touches sounddevice itself.

    `stop_event`: when given (the SPEAKING / tap-to-interrupt path), play in chunks
    and bail the moment it's set. When None (beep, confirmations, alarm
    announcements), use the unchanged blocking path — those are not interruptible.

    `level_cb`: per-chunk mouth-sync hook. Only meaningful on the interruptible
    (chunked) path, so it's ignored when stop_event is None — the blocking path is
    one PortAudio write with no chunk boundaries to report on. Confirmations/alarms
    don't drive the talking mouth, so that's fine.
    """
    if samples is None or samples.size == 0:
        return
    device, rate, channels = _resolve_output(device)
    mono = _resample_linear(samples.astype(np.float32, copy=False), src_rate, rate)
    mono = np.clip(mono, -1.0, 1.0)
    if stop_event is None:
        _play_blocking(mono, rate, channels, device)
    else:
        _play_interruptible(mono, rate, channels, device, stop_event, level_cb)
