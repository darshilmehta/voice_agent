# /// script
# requires-python = ">=3.12"
# dependencies = ["faster-whisper>=1.1", "mlx-whisper>=0.4", "jiwer", "numpy", "soundfile", "scipy"]
# ///
"""Smoke tests 06–07 — Whisper variants for speech-to-text.

Compares, on 20 synthetic clips (Kokoro + macOS `say`, English + Hindi):

  faster-whisper small / large-v3-turbo   (CTranslate2, CPU int8)
  mlx-whisper    small / large-v3-turbo   (MLX, Apple GPU)

Measures: load time, warm latency per clip, real-time factor, language
detection, Devanagari output for Hindi, WER (English) / CER (Hindi), memory.
Each variant runs in its own subprocess so memory numbers are clean.

Needs clips from 09_kokoro.py.   Run:  uv run scripts/smoke/06_whisper.py
"""

import json
import os
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT / "data/models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

AUDIO = ROOT / "data/smoke/audio"
SENT = json.loads((Path(__file__).parent / "fixtures/speech_sentences.json").read_text())
VARIANTS = {
    "fw-small": ("faster", "Systran/faster-whisper-small"),
    "fw-turbo": ("faster", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    "mlx-small": ("mlx", "mlx-community/whisper-small-mlx"),
    "mlx-turbo": ("mlx", "mlx-community/whisper-large-v3-turbo"),
}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def local_snapshot(repo_id: str) -> str:
    snaps = Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return str(sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime)[-1])


def normalize(text: str) -> str:
    text = text.lower().replace("twenty twenty four", "2024").replace("ebidta", "ebitda")
    text = re.sub(r"[^\w\sऀ-ॿ']", " ", text)  # drop punctuation incl. । ? ,
    text = re.sub(r"[।॥]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clips() -> list[tuple[str, str, str]]:
    out = []
    for src in ("kokoro", "say"):
        for lang in ("en", "hi"):
            for i, ref in enumerate(SENT[lang]):
                f = AUDIO / f"{src}_{lang}_{i}.wav"
                if f.exists():
                    out.append((str(f), lang, ref))
    return out


def load_pcm16k(f: str):
    """Float32 mono PCM at 16 kHz — the same shape the backend gets from the WebSocket."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(f, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        g = np.gcd(sr, 16_000)
        audio = resample_poly(audio, 16_000 // g, sr // g).astype(np.float32)
    return audio


def run_variant(name: str) -> dict:
    """Runs inside a subprocess: load one model, transcribe every clip."""
    import jiwer

    kind, repo = VARIANTS[name]
    path = local_snapshot(repo)
    t0 = time.perf_counter()
    if kind == "faster":
        from faster_whisper import WhisperModel
        model = WhisperModel(path, device="cpu", compute_type="int8")

        def transcribe(f: str) -> tuple[str, str]:
            segs, info = model.transcribe(load_pcm16k(f), beam_size=1, condition_on_previous_text=False)
            return "".join(s.text for s in segs).strip(), info.language
    else:
        import mlx.core as mx
        import mlx_whisper

        def transcribe(f: str) -> tuple[str, str]:
            r = mlx_whisper.transcribe(load_pcm16k(f), path_or_hf_repo=path, fp16=True, verbose=None,
                                       condition_on_previous_text=False)
            return r["text"].strip(), r["language"]
        transcribe(clips()[0][0])  # mlx loads lazily on first call
    load_s = time.perf_counter() - t0
    transcribe(clips()[0][0])  # warm

    rows = []
    for f, lang, ref in clips():
        dur = len(load_pcm16k(f)) / 16_000
        t0 = time.perf_counter()
        hyp, detected = transcribe(f)
        dt = time.perf_counter() - t0
        n_ref, n_hyp = normalize(ref), normalize(hyp)
        err = jiwer.wer(n_ref, n_hyp) if lang == "en" else jiwer.cer(n_ref, n_hyp)
        deva = len(DEVANAGARI.findall(hyp)) / max(1, len(re.sub(r"[\s\d.,?!'\"\u0964\u0965-]", "", hyp))) if lang == "hi" else None
        rows.append({"clip": Path(f).name, "lang": lang, "detected": detected, "latency": dt, "rtf": dt / dur,
                     "err": err, "devanagari": deva, "hyp": hyp})
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    gpu = None
    if kind == "mlx":
        gpu = (mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()) / 1e9
    return {"variant": name, "load_s": load_s, "rss_gb": mem, "gpu_gb": gpu, "rows": rows}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--variant":
        print(json.dumps(run_variant(sys.argv[2]), ensure_ascii=False))
        return 0

    n = len(clips())
    if n < 20:
        print(f"Only {n} clips in {AUDIO}; run 09_kokoro.py first."); return 1
    print(f"{n} clips (kokoro + say × en + hi)\n")
    summary = []
    selected = [v for v in os.environ.get("WHISPER_VARIANTS", ",".join(VARIANTS)).split(",") if v in VARIANTS]
    for name in selected:
        print(f"=== {name} ===")
        p = subprocess.run([sys.executable, __file__, "--variant", name], capture_output=True, text=True)
        if p.returncode != 0:
            print("  FAILED\n" + p.stderr[-1500:]); summary.append((name, None)); continue
        r = json.loads(p.stdout.strip().splitlines()[-1])
        for row in r["rows"]:
            ok_lang = row["detected"] == row["lang"]
            extra = f" deva={row['devanagari']:.0%}" if row["devanagari"] is not None else ""
            print(f"  {'✓' if ok_lang and row['err'] < 0.2 else '✗'} {row['clip']:<14} {row['latency'] * 1000:5.0f} ms "
                  f"lang={row['detected']:<3} {'WER' if row['lang'] == 'en' else 'CER'}={row['err']:.2f}{extra}  {row['hyp'][:60]!r}")
        summary.append((name, r))
        print()

    print("=== summary (warm) ===")
    print(f"  {'variant':<10} {'load':>6} {'p50 ms':>7} {'RTF':>5} {'EN WER':>7} {'HI CER':>7} {'lang ok':>8} {'HI deva':>8} {'mem':>16}")
    for name, r in summary:
        if not r:
            print(f"  {name:<10} FAILED"); continue
        rows = r["rows"]
        en = [x for x in rows if x["lang"] == "en"]; hi = [x for x in rows if x["lang"] == "hi"]
        mem = f"{r['rss_gb']:.2f} GB rss" + (f", {r['gpu_gb']:.2f} GB gpu" if r["gpu_gb"] else "")
        print(f"  {name:<10} {r['load_s']:5.1f}s {statistics.median(x['latency'] for x in rows) * 1000:7.0f} "
              f"{statistics.median(x['rtf'] for x in rows):5.2f} {statistics.mean(x['err'] for x in en):7.2f} "
              f"{statistics.mean(x['err'] for x in hi):7.2f} {sum(x['detected'] == x['lang'] for x in rows):>5}/{len(rows)} "
              f"{statistics.mean(x['devanagari'] for x in hi):8.0%} {mem:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
