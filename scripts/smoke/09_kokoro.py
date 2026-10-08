# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kokoro>=0.9.4",
#   "misaki[en]>=0.9.4",
#   "transformers>=4.45",  # without this, resolution backtracks to a 2021 release that needs Rust
#   "soundfile",
#   "torch",
#   "numpy",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""Smoke test 09 — Kokoro TTS, plus generation of STT test clips.

  1. loads fully offline from local files (model, config, voice packs, spaCy model bundled)
  2. English + Hindi synthesis, CPU vs Apple GPU (MPS)
  3. time to first audio per sentence (what streaming TTS will feel like) and real-time factor
  4. writes listenable samples and the STT test set:
       data/smoke/audio/kokoro_{en,hi}_{i}.wav  (Kokoro voices)
       data/smoke/audio/say_{en,hi}_{i}.wav     (macOS `say`: Rishi en-IN, Lekha hi-IN)

Run:  uv run scripts/smoke/09_kokoro.py
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT / "data/models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # a few Kokoro ops have no MPS kernel

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from kokoro import KModel, KPipeline  # noqa: E402

SR = 24_000
OUT = ROOT / "data/smoke/audio"
SENT = json.loads((Path(__file__).parent / "fixtures/speech_sentences.json").read_text())
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def local_snapshot(repo_id: str) -> Path:
    snaps = Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime)[-1]


def synth(pipe: KPipeline, text: str, voice: str) -> tuple[np.ndarray, float]:
    """Return (audio, seconds until the first audio chunk was ready)."""
    t0 = time.perf_counter()
    first, chunks = None, []
    for _, _, audio in pipe(text, voice=voice, speed=1.0):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio))
    return np.concatenate(chunks), first or 0.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = local_snapshot("hexgrad/Kokoro-82M")
    voice = lambda name: str(snap / "voices" / f"{name}.pt")  # noqa: E731

    timings: dict[str, dict] = {}
    for device in ("cpu", "mps"):
        print(f"\n[{device}] load + synthesize")
        t0 = time.perf_counter()
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(snap / "config.json"),
                       model=str(snap / "kokoro-v1_0.pth")).to(device).eval()
        pipes = {"en": KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model),
                 "hi": KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M", model=model)}
        check(f"{device}: model + EN/HI pipelines loaded offline", True, f"{time.perf_counter() - t0:.1f}s")
        synth(pipes["en"], "Warm up.", voice(SENT["voices"]["kokoro"]["en"]))

        for lang in ("en", "hi"):
            firsts, rtfs = [], []
            for i, text in enumerate(SENT[lang]):
                audio, first = synth(pipes[lang], text, voice(SENT["voices"]["kokoro"][lang]))
                dur = len(audio) / SR
                firsts.append(first); rtfs.append(first / dur if dur else 9)
                if device == "cpu":  # write the clips once
                    sf.write(OUT / f"kokoro_{lang}_{i}.wav", audio, SR)
                ok = dur > 0.5 and np.abs(audio).max() > 0.01
                if not ok:
                    check(f"{device}: {lang} sentence {i} produced audio", False, f"{dur:.2f}s")
            timings[f"{device}_{lang}"] = {"first": statistics.median(firsts), "rtf": statistics.median(rtfs)}
            check(f"{device}: {lang} full-sentence time to first audio (median of {len(firsts)})",
                  statistics.median(firsts) < 1.0,
                  f"{statistics.median(firsts) * 1000:.0f} ms, real-time factor {statistics.median(rtfs):.2f} (lower = faster)")
        if device == "mps":
            check("mps memory", True, f"allocated {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        del pipes, model

    best = min(("cpu", "mps"), key=lambda d: timings[f"{d}_en"]["first"] + timings[f"{d}_hi"]["first"])
    print(f"\n  → faster device for Kokoro: {best}")

    print(f"\n[{best}] latency vs chunk length (streaming can start on the first clause)")
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(snap / "config.json"),
                   model=str(snap / "kokoro-v1_0.pth")).to(best).eval()
    pipes = {"en": KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model),
             "hi": KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M", model=model)}
    synth(pipes["en"], "Warm up.", voice(SENT["voices"]["kokoro"]["en"]))
    probes = {
        "en": ["Sure.", "No wait, I meant", "FY24 revenue was four thousand crore,",
               "FY24 revenue was four thousand two hundred crore, up thirty four percent from the previous year."],
        "hi": ["ज़रूर।", "रुको, मेरा मतलब था", "पिछले साल कंपनी का मुनाफा,",
               "पिछले साल कंपनी का मुनाफा चार हज़ार करोड़ रुपये था, जो पिछले वर्ष से चौंतीस प्रतिशत अधिक है।"],
    }
    clause_firsts = []
    for lang, texts in probes.items():
        for t in texts:
            runs = [synth(pipes[lang], t, voice(SENT["voices"]["kokoro"][lang]))[1] for _ in range(3)]
            audio, _ = synth(pipes[lang], t, voice(SENT["voices"]["kokoro"][lang]))
            print(f"        {lang} {len(t.split()):>2} words → first audio {statistics.median(runs) * 1000:4.0f} ms for {len(audio) / SR:.1f}s of speech  {t[:40]!r}")
            if len(t.split()) <= 6:
                clause_firsts.append(statistics.median(runs))
    check("clause-level (≤6 words) time to first audio < 400 ms", max(clause_firsts) < 0.4,
          f"max {max(clause_firsts) * 1000:.0f} ms over EN/HI short chunks")

    print("\n[macOS say] second synthetic voice source for STT tests")
    for lang in ("en", "hi"):
        v = SENT["voices"]["macos_say"][lang]
        for i, text in enumerate(SENT[lang]):
            f = OUT / f"say_{lang}_{i}.wav"
            subprocess.run(["say", "-v", v, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(f), text], check=True)
        check(f"say {v} ({lang}) clips written", all((OUT / f"say_{lang}_{i}.wav").stat().st_size > 10_000 for i in range(len(SENT[lang]))))

    print(f"\n  Listen: open {OUT}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
