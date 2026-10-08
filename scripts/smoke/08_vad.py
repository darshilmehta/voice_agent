# /// script
# requires-python = ">=3.12"
# dependencies = ["silero-vad>=5.1", "torch", "numpy", "soundfile", "scipy"]
# ///
"""Smoke test 08 — Silero VAD (server side).

Builds a synthetic "conversation" track from the speech clips with known
speech/silence boundaries, then checks:

  1. model loads with no network (bundled in the pip package)
  2. offline segmentation finds each utterance; a short 300 ms mid-sentence
     pause does NOT split a turn, a 1 s gap does
  3. background noise alone does not trigger speech
  4. streaming (32 ms chunks): per-chunk compute, speech-start detection delay
     (what barge-in reacts to) and end-of-turn delay at 600 ms silence

Needs clips from 09_kokoro.py.   Run:  uv run scripts/smoke/08_vad.py
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "data/smoke/audio"
SR = 16_000
CHUNK = 512  # 32 ms — the window Silero expects at 16 kHz
END_OF_TURN_MS = 600
results: list[tuple[str, bool, str]] = []
rng = np.random.default_rng(0)


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def load(name: str) -> np.ndarray:
    a, sr = sf.read(AUDIO / name, dtype="float32")
    if sr != SR:
        g = np.gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    a = a / max(1e-6, np.abs(a).max()) * 0.6
    # Trim the clip's own leading/trailing silence so the ground truth is where speech really is.
    env = np.convolve(np.abs(a), np.ones(160) / 160, mode="same")  # 10 ms envelope
    voiced = np.flatnonzero(env > 0.02)
    return a[voiced[0]:voiced[-1] + 1]


def noise(seconds: float, level: float = 0.01) -> np.ndarray:
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def main() -> int:
    print("\n[1] Load")
    t0 = time.perf_counter()
    model = load_silero_vad()
    check("model loaded (bundled, no download)", True, f"{(time.perf_counter() - t0) * 1000:.0f} ms")

    # Track: [1.5 noise] utt A [1.5 noise] utt B-part1 [0.3 noise] utt B-part2 [1.5 noise] utt C [2.0 noise]
    # Gaps ≥ 1.5 s: background noise right after speech holds Silero's lower "off" threshold for an extra 200-400 ms.
    a, b, c = load("say_en_0.wav"), load("kokoro_hi_1.wav"), load("kokoro_en_2.wav")
    half = len(b) // 2
    parts = [noise(1.5, 0.003), a, noise(1.5, 0.003), b[:half], noise(0.3, 0.003), b[half:], noise(1.5, 0.003), c, noise(2.0, 0.003)]
    track = np.concatenate(parts)
    bounds, pos = [], 0
    for p in parts:
        bounds.append((pos / SR, (pos + len(p)) / SR)); pos += len(p)
    truth = [(bounds[1][0], bounds[1][1]), (bounds[3][0], bounds[5][1]), (bounds[7][0], bounds[7][1])]
    print(f"        track {len(track) / SR:.1f}s; expected turns: " + ", ".join(f"{s:.2f}–{e:.2f}s" for s, e in truth))

    print(f"\n[2] Offline segmentation (min silence {END_OF_TURN_MS} ms)")
    segs = get_speech_timestamps(torch.from_numpy(track), model, sampling_rate=SR, threshold=0.5,
                                 min_silence_duration_ms=END_OF_TURN_MS, min_speech_duration_ms=250, return_seconds=True)
    print("        detected: " + ", ".join(f"{s['start']:.2f}–{s['end']:.2f}s" for s in segs))
    check("3 turns detected (300 ms pause not split, 1 s gaps split)", len(segs) == 3, f"{len(segs)} segments")
    if len(segs) == 3:
        err = max(max(abs(s["start"] - t[0]), abs(s["end"] - t[1])) for s, t in zip(segs, truth))
        check("boundaries within 450 ms of truth (incl. padding + noise tail)", err < 0.45, f"max error {err * 1000:.0f} ms")

    print("\n[3] Noise only")
    for level in (0.01, 0.03):
        n = get_speech_timestamps(torch.from_numpy(noise(5.0, level)), model, sampling_rate=SR, return_seconds=True)
        check(f"white noise at {level} amplitude → no speech", len(n) == 0, f"{len(n)} false segments")

    print("\n[4] Streaming, 32 ms chunks")
    it = VADIterator(model, threshold=0.5, sampling_rate=SR, min_silence_duration_ms=END_OF_TURN_MS, speech_pad_ms=30)
    events, compute = [], []
    for i in range(0, len(track) - CHUNK + 1, CHUNK):
        t0 = time.perf_counter()
        ev = it(torch.from_numpy(track[i:i + CHUNK]), return_seconds=False)
        compute.append((time.perf_counter() - t0) * 1000)
        if ev:
            # the event becomes known at the END of the chunk we just fed
            events.append((list(ev)[0], (i + CHUNK) / SR, list(ev.values())[0] / SR))
    it.reset_states()
    starts = [e for e in events if e[0] == "start"]
    ends = [e for e in events if e[0] == "end"]
    check("per-chunk compute ≪ 32 ms budget", statistics.median(compute) < 5,
          f"p50 {statistics.median(compute):.2f} ms, max {max(compute):.2f} ms")
    if len(starts) == 3 and len(ends) == 3:
        start_delay = [known - t[0] for (_, known, _), t in zip(starts, truth)]
        end_delay = [known - t[1] for (_, known, _), t in zip(ends, truth)]
        check("speech-start detected within 150 ms (barge-in trigger)", max(start_delay) < 0.15,
              "delays " + ", ".join(f"{d * 1000:.0f} ms" for d in start_delay))
        check(f"end-of-turn fires within 1.1 s of speech stopping ({END_OF_TURN_MS} ms setting + noise tail)", all(0.4 < d < 1.1 for d in end_delay),
              "delays " + ", ".join(f"{d * 1000:.0f} ms" for d in end_delay))
    else:
        check("streaming produced 3 start + 3 end events", False, f"{len(starts)} starts, {len(ends)} ends")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
