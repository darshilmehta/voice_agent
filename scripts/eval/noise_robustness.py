"""Noise robustness of the voice loop, before vs after the noise work (docs/DESIGN.md §9.7, §3.10 "Noisy rooms").

    cd backend && uv run --group ml python ../scripts/eval/noise_robustness.py [--quick] [--out DIR] [--models DIR]

Everything is generated here, nothing is downloaded and nothing is played (audio is only written to files under
--out, default <repo>/data/eval/noise, git-ignored):

- clean questions (10 English, 4 Hindi) and background chatter (16 English, 4 Hindi lines written for this test) are
  spoken by macOS `say` into 16 kHz WAV files; the user's questions are set to an active speech level of -26 dBFS
  (near field, after the browser's gain control)
- noise: white, pink and brown noise; babble (six chatter talkers overlapped, with a little room reverb); café (babble
  plus dish clatter); TV (chatter from two voices, band-limited like a TV speaker and in a synthetic reverberant room)
- "before" is main's pipeline: no denoiser, the plain Silero endpointer, every transcribed utterance answered (garbled
  ones asked again). "after" runs the browser's RNNoise over the audio first (scripts/eval/denoise.mjs: the exact
  worklet and WASM the page loads, at 48 kHz) and then the noise-floor gate and the background-speech check
  (backend/app/services/voice/noise.py) with the thresholds in config/local.config.json

The server's decisions are simulated on audio time with the session's own parts (Silero VAD, ``Endpointer``,
mlx-whisper, ``addressed``, ``barge_in_verdict``, ``is_backchannel``): every SpeechEnded utterance is transcribed and
classified as the session would; while the agent is "speaking", every speech start is a ``barge_in_start`` (the
browser's VAD runs the same Silero v5 with the same gate) and the barge-in decision follows §3.10 (snapshot transcript
once min_speech_ms of speech is in, assumed to arrive STT_LATENCY_S later; deadline 700 ms, cap 1.4 s; transcripts
every 400 ms of further speech after a "resume"; the utterance's own end). The session's glue is covered by the unit
tests; this measures the audio-dependent parts.

Measured, per noise type and level, before → after:
- false turn starts per minute of noise-only audio (speech starts the server reports: the UI shows "listening")
- background replies per minute: noise-only utterances that would get an answer or a "say again"
- false barge-ins per minute: noise-only audio while the agent speaks, answers stopped
- question outcomes and STT word error rate at SNR clean / 20 / 10 / 5 / 0 dB (answered, asked again, dropped,
  missed); after, both before the user's first answered turn (their level unknown) and after it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.offline import apply_runtime_env  # noqa: E402
from app.providers.registry import build_container  # noqa: E402
from app.services.voice.noise import (  # noqa: E402
    NoiseGate,
    SpeechEvidence,
    UserLevel,
    addressed,
)
from app.services.voice.speech_text import (  # noqa: E402
    is_backchannel,
    is_filler,
    normalize_utterance,
    real_words,
    transcript_garbled,
)
from app.services.voice.turn_taking import (  # noqa: E402
    FRAME_SAMPLES,
    BargeInEvidence,
    Endpointer,
    SpeechEnded,
    SpeechStarted,
    barge_in_verdict,
)
from app.settings import Settings, load_settings  # noqa: E402

SR = 16_000
FRAME_S = FRAME_SAMPLES / SR
USER_DBFS = -26.0  # the user's active speech level
MIC_FLOOR_DBFS = -75.0  # a microphone's own hiss under everything
STT_LATENCY_S = 0.4  # a barge-in snapshot's transcript arrives this long after it was taken (measured 0.3-0.7 s)
WATCH_STEP_S = 0.4
SNRS: list[float | None] = [None, 20, 10, 5, 0]  # None: no added noise
NOISE_TYPES = ["white", "pink", "brown", "babble", "cafe", "tv"]
NOISE_ONLY_SNRS = [20, 10, 5]  # noise-only streams at the level that would give this SNR against the user
NOISE_ONLY_S = 60.0

QUESTIONS: list[tuple[str, str, str]] = [  # (language, voice, text)
    ("en", "Samantha", "What was the company's revenue last year?"),
    ("en", "Daniel", "How many employees did the company have at the end of the year?"),
    ("en", "Rishi", "Can you summarize the termination clause in simple words?"),
    ("en", "Karen", "What is the EBITDA margin for the specialty chemicals segment?"),
    ("en", "Tara", "Which facility had the highest carbon emissions?"),
    ("en", "Aman", "Show me the revenue over the last five years as a chart."),
    ("en", "Moira", "Who is the chief financial officer of the company?"),
    ("en", "Samantha", "What does the policy say about remote work?"),
    ("en", "Daniel", "Compare the operating profit of the two companies."),
    ("en", "Rishi", "Tell me about the dividend instead."),
    ("hi", "Lekha", "कंपनी का पिछले साल का मुनाफा कितना था?"),
    ("hi", "Lekha", "अनुबंध की समाप्ति की शर्तें आसान भाषा में बताइए।"),
    ("hi", "Lekha", "कंपनी ने कार्बन उत्सर्जन कितना कम किया?"),
    ("hi", "Lekha", "कर्मचारियों की कुल संख्या कितनी है?"),
]
CHATTER: list[tuple[str, str]] = [  # (language, text): background talk, written for this test
    ("en", "Did you see the match last night?"),
    ("en", "I think we should order another coffee."),
    ("en", "The traffic on the way here was terrible."),
    ("en", "She said the meeting moved to Thursday afternoon."),
    ("en", "Can you pass me the sugar, please?"),
    ("en", "We are going to the beach this weekend."),
    ("en", "My phone battery is almost dead again."),
    ("en", "That new restaurant near the station is really good."),
    ("en", "Honestly, I have no idea what he meant by that."),
    ("en", "Let me know when you get home tonight."),
    ("en", "The weather has been strange all week."),
    ("en", "I left my keys in the car, can you believe it?"),
    ("en", "In other news, the city council approved the new budget today."),
    ("en", "Stay tuned for the weather forecast after the break."),
    ("en", "Our next guest has travelled the world for twenty years."),
    ("en", "Scientists say the discovery could change everything we know."),
    ("hi", "आज बाहर बहुत गर्मी है।"),
    ("hi", "चलो, चाय पीते हैं।"),
    ("hi", "कल रात का मैच देखा तुमने?"),
    ("hi", "मेरी ट्रेन दस मिनट लेट है।"),
]
CHATTER_VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tara", "Aman", "Fred", "Kathy", "Ralph"]
TV_VOICES = ["Daniel", "Karen"]


# ------------------------------------------------------------------ audio helpers


def say(text: str, voice: str, path: Path) -> np.ndarray:
    """``text`` spoken by macOS ``say`` into a 16 kHz WAV (cached); nothing is played."""
    import soundfile as sf

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True)
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    assert rate == SR, (path, rate)
    return trim(audio)


def trim(x: np.ndarray) -> np.ndarray:
    """Without leading and trailing digital silence (kept: 50 ms)."""
    loud = np.flatnonzero(np.abs(x) > 1e-4)
    if not loud.size:
        return x
    pad = SR // 20
    return x[max(0, loud[0] - pad) : loud[-1] + pad]


def active_dbfs(x: np.ndarray) -> float:
    """Active speech level: the mean power of the 32 ms frames within 25 dB of the loudest one."""
    n = len(x) // FRAME_SAMPLES * FRAME_SAMPLES
    power = np.mean(np.square(x[:n].reshape(-1, FRAME_SAMPLES), dtype=np.float64), axis=1)
    top = power.max()
    return 10 * math.log10(float(power[power >= top / 10**2.5].mean()))


def rms_dbfs(x: np.ndarray) -> float:
    return 10 * math.log10(float(np.mean(np.square(x, dtype=np.float64))) + 1e-20)


def at_level(x: np.ndarray, dbfs: float, *, active: bool = True) -> np.ndarray:
    current = active_dbfs(x) if active else rms_dbfs(x)
    return (x * 10 ** ((dbfs - current) / 20)).astype(np.float32)


def colored(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Noise with a 1/f^exponent power spectrum (0 white, 1 pink, 2 brown)."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    spectrum /= f ** (exponent / 2)
    x = np.fft.irfft(spectrum, n)
    return (x / np.sqrt(np.mean(x**2))).astype(np.float32)


def room(x: np.ndarray, rt60: float, rng: np.random.Generator, direct: float = 1.0) -> np.ndarray:
    """``x`` in a synthetic reverberant room: a direct path plus an exponentially decaying noise tail."""
    from scipy.signal import fftconvolve

    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * 0.08
    tail[: int(0.004 * SR)] = 0  # early-reflection gap
    ir = tail
    ir[0] = direct
    return fftconvolve(x, ir)[: len(x)].astype(np.float32)


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def talker(lines: list[np.ndarray], seconds: float, rng: np.random.Generator, gap: tuple[float, float]) -> np.ndarray:
    """One talker: lines in random order with pauses between them, ``seconds`` long."""
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    at = int(rng.uniform(0, 1.5) * SR)
    while at < len(out):
        line = lines[rng.integers(len(lines))]
        line = at_level(line, -26)
        end = min(len(out), at + len(line))
        out[at:end] += line[: end - at]
        at = end + int(rng.uniform(*gap) * SR)
    return out


@dataclass
class Material:
    questions: list[tuple[str, str, np.ndarray]]  # (language, text, audio at USER_DBFS)
    chatter: dict[str, list[np.ndarray]]  # voice → its lines


def material(out: Path) -> Material:
    clips = out / "clips"
    questions = []
    for i, (lang, voice, text) in enumerate(QUESTIONS):
        audio = say(text, voice, clips / f"q{i:02d}_{lang}_{voice}.wav")
        questions.append((lang, text, at_level(audio, USER_DBFS)))
    chatter: dict[str, list[np.ndarray]] = {}
    for voice in [*CHATTER_VOICES, "Lekha"]:
        lines = [(j, text) for j, (lang, text) in enumerate(CHATTER) if (lang == "hi") == (voice == "Lekha")]
        chatter[voice] = [say(text, voice, clips / f"chatter_{voice}_{j:02d}.wav") for j, text in lines]
    return Material(questions, chatter)


def noise(kind: str, seconds: float, m: Material, rng: np.random.Generator) -> np.ndarray:
    """``kind`` of noise, ``seconds`` long, at 0 dBFS RMS for steady noise and babble (scaled by the caller); the TV at
    -26 dBFS active level."""
    n = int(seconds * SR)
    if kind in ("white", "pink", "brown"):
        return colored(n, {"white": 0, "pink": 1, "brown": 2}[kind], rng)
    if kind in ("babble", "cafe"):
        voices = rng.choice(CHATTER_VOICES, size=6, replace=False)
        mix = sum(talker(m.chatter[v], seconds, rng, (0.1, 0.8)) for v in voices)
        mix = room(np.asarray(mix, dtype=np.float32), 0.5, rng, direct=0.6)
        mix = mix / np.sqrt(np.mean(mix**2))
        if kind == "cafe":  # dish and cutlery clatter: short bright bursts, 0.3-2 s apart, peaks above the babble
            clatter = np.zeros(n, dtype=np.float32)
            at = 0
            while at < n:
                at += int(rng.uniform(0.3, 2.0) * SR)
                k = int(rng.uniform(0.02, 0.12) * SR)
                if at + k >= n:
                    break
                burst = rng.standard_normal(k) * np.exp(-np.arange(k) / (k / 4)) * rng.uniform(2, 5)
                clatter[at : at + k] += burst.astype(np.float32)
            mix = mix + bandpass(clatter, 1500, 7500)
            mix = mix / np.sqrt(np.mean(mix**2))
        return mix.astype(np.float32)
    if kind == "tv":
        lines = [*m.chatter[TV_VOICES[0]], *m.chatter[TV_VOICES[1]]]
        out = np.zeros(n, dtype=np.float32)
        at = 0
        while at < n:
            line = at_level(lines[rng.integers(len(lines))], -26)
            end = min(n, at + len(line))
            out[at:end] += line[: end - at]
            at = end + int(rng.uniform(0.15, 0.6) * SR)
        tv = bandpass(room(out, 0.6, rng, direct=0.5), 180, 6000)
        return at_level(tv, -26)
    raise ValueError(kind)


def mic(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (x + colored(len(x), 0, rng) * 10 ** (MIC_FLOOR_DBFS / 20)).astype(np.float32)


# ------------------------------------------------------------------ the browser's denoiser


def denoise_batch(clips: list[np.ndarray], work: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    """RNNoise over each clip as the page runs it: 16 kHz → 48 kHz (the AudioContext) → RNNoise → 16 kHz."""
    from scipy.signal import resample_poly

    work.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    for i, x in enumerate(clips):
        up = resample_poly(x, 3, 1).astype("<f4")
        (work / f"in{i}.f32").write_bytes(up.tobytes())
        args += [str(work / f"in{i}.f32"), str(work / f"out{i}.f32")]
    result = subprocess.run(
        ["node", str(REPO / "scripts/eval/denoise.mjs"), *args], check=True, capture_output=True, text=True
    )
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    out = []
    for i, x in enumerate(clips):
        y = np.frombuffer((work / f"out{i}.f32").read_bytes(), dtype="<f4")
        down = resample_poly(y, 1, 3).astype(np.float32)[: len(x)]
        out.append(np.pad(down, (0, len(x) - len(down))))
        (work / f"in{i}.f32").unlink()
        (work / f"out{i}.f32").unlink()
    return out, stats


# ------------------------------------------------------------------ the session, simulated on audio time


@dataclass
class Utt:
    start: float
    end: float
    text: str
    outcome: str  # answer | say_again | drop | ignored | stop | resume
    reason: str = ""
    level: float | None = None
    floor: float | None = None
    avg_logprob: float | None = None


@dataclass
class Run:
    starts: int = 0  # speech starts reported to the client
    utterances: list[Utt] = field(default_factory=list)
    barge_stops: int = 0
    barge_resumes: int = 0


class Simulator:
    def __init__(self, settings: Settings, vad: Any, stt: Any, *, after: bool) -> None:
        self.settings = settings
        self.vad = vad
        self.stt = stt
        self.after = after
        noise_cfg = settings.voice.noise
        self.noise = noise_cfg if after else replace_ns(noise_cfg, adaptive_gating=False, drop_background_speech=False)

    def endpointer(self, user: UserLevel) -> Endpointer:
        v = self.settings.vad
        gate = NoiseGate(self.noise, threshold=v.threshold, min_speech_ms=v.min_speech_ms, user=user)
        return Endpointer(
            threshold=v.threshold, min_speech_ms=v.min_speech_ms, end_of_turn_ms=v.end_of_turn_ms, gate=gate
        )

    async def transcribe(self, audio: np.ndarray) -> Any:
        return await self.stt.transcribe(audio, ["en", "hi"])

    def classify(self, t: Any, level: float | None, floor: float | None, user: UserLevel) -> tuple[str, str]:
        text = t.text.strip()
        evidence = SpeechEvidence(text, level, floor, user.dbfs, transcript_garbled(t), t.avg_logprob, t.no_speech_prob)
        decision = addressed(evidence, self.noise)
        if decision.verdict == "drop":
            return "drop", decision.reason
        if not normalize_utterance(text) or is_filler(text):
            return "ignored", "no words"
        return decision.verdict, decision.reason

    def background(self, t: Any, level: float | None, floor: float | None, user: UserLevel) -> bool:
        if not self.after:
            return False
        e = SpeechEvidence(t.text, level, floor, user.dbfs, transcript_garbled(t), t.avg_logprob, t.no_speech_prob)
        return addressed(e, self.noise).verdict == "drop"

    async def run(self, audio: np.ndarray, *, speaking: bool, user_dbfs: float | None = None) -> Run:
        """The session over ``audio``: idle (every utterance a possible turn) or while the agent speaks throughout
        (every speech start a barge-in to decide)."""
        b = self.settings.voice.barge_in
        stream = self.vad.open_stream()
        user = UserLevel()
        if user_dbfs is not None:
            user.update(user_dbfs)
        ep = self.endpointer(user)
        n = len(audio) // FRAME_SAMPLES
        frames = audio[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES)
        probabilities = await stream(frames)
        run = Run()
        start_t = 0.0
        pending: dict[str, Any] | None = None  # a barge-in being decided
        watch: dict[str, Any] | None = None  # speech after a "resume"
        for i, (frame, p) in enumerate(zip(frames, probabilities, strict=True)):
            t = (i + 1) * FRAME_S
            for event in ep.push(frame, p):
                if isinstance(event, SpeechStarted):
                    run.starts += 1
                    start_t = t
                    watch = None
                    if speaking:
                        pending = {"deadline": t + b.decision_timeout_ms / 1000, "transcript": None, "snap": None}
                        pending["cap"] = t + 2 * b.decision_timeout_ms / 1000
                elif isinstance(event, SpeechEnded):
                    u = event.utterance
                    tr = await self.transcribe(u.audio)
                    if speaking:
                        outcome = "resume"
                        if not self.background(tr, u.level_dbfs, u.floor_dbfs, user) and not is_backchannel(
                            tr.text, b.backchannel_max_words
                        ):
                            outcome = "stop"
                        if pending is not None or watch is not None or outcome == "stop":
                            run.barge_stops += outcome == "stop"
                            run.barge_resumes += outcome == "resume" and pending is not None
                        run.utterances.append(Utt(start_t, t, tr.text, outcome, "", u.level_dbfs, u.floor_dbfs))
                        pending = watch = None
                    else:
                        outcome, reason = self.classify(tr, u.level_dbfs, u.floor_dbfs, user)
                        if outcome == "answer" and self.after:
                            user.update(u.level_dbfs)
                        run.utterances.append(
                            Utt(start_t, t, tr.text, outcome, reason, u.level_dbfs, u.floor_dbfs, tr.avg_logprob)
                        )
            if not speaking or not ep.in_utterance:
                continue
            level, floor = ep.level_dbfs, ep.floor_at_start
            if pending is not None:
                if pending["snap"] is None and ep.speech_ms >= ep.min_speech:
                    began = time.perf_counter()
                    tr = await self.transcribe(ep.snapshot())
                    pending["snap"] = (t + max(STT_LATENCY_S, time.perf_counter() - began), tr)
                transcribing = pending["snap"] is not None and pending["snap"][0] > t
                if pending["snap"] is not None and pending["snap"][0] <= t and pending["transcript"] is None:
                    tr = pending["snap"][1]
                    bg = self.background(tr, level, floor, user)
                    pending["transcript"] = tr.text
                    pending["backchannel"] = True if bg else is_backchannel(tr.text, b.backchannel_max_words)
                    pending["real"] = 0 if bg else real_words(tr.text)
                evidence = BargeInEvidence(
                    speech_ms=ep.speech_ms,
                    speaking=ep.speaking,
                    transcript=pending["transcript"],
                    ended=False,
                    deadline_passed=t >= pending["deadline"],
                    real_words=pending.get("real", 0),
                    cap_passed=t >= pending["cap"],
                    transcribing=transcribing,
                )
                verdict = barge_in_verdict(
                    evidence, min_speech_ms=ep.min_speech, is_backchannel=pending.get("backchannel")
                )
                if verdict is None and t >= pending["cap"]:
                    verdict = "resume"
                if verdict == "stop":
                    run.barge_stops += 1
                    pending = None
                    ep.reset()  # the answer is cut; the agent "starts again" on the next noise
                elif verdict == "resume":
                    run.barge_resumes += 1
                    pending = None
                    watch = {"checked": ep.speech_ms}
            elif (
                watch is not None
                and ep.speaking
                and t >= watch.get("free_at", 0.0)
                and ep.speech_ms >= watch["checked"] + WATCH_STEP_S * 1000
            ):
                # As the session: the next transcription only once the last one is done (audio arrives in real time)
                watch["checked"] = ep.speech_ms
                began = time.perf_counter()
                tr = await self.transcribe(ep.snapshot())
                watch["free_at"] = t + time.perf_counter() - began
                if (
                    not self.background(tr, level, floor, user)
                    and not is_backchannel(tr.text, b.backchannel_max_words)
                    and real_words(tr.text) >= 2
                ):
                    run.barge_stops += 1
                    watch = None
                    ep.reset()
        return run


def replace_ns(cfg: Any, **changes: Any) -> Any:
    data = cfg.model_dump() if hasattr(cfg, "model_dump") else vars(cfg)
    return SimpleNamespace(**{**data, **changes})


# ------------------------------------------------------------------ scoring

_PUNCT = re.compile(r"[^\w\sऀ-ॿ']|[।॥]")


def words(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.casefold().replace("\u2019", "'")).split()


def wer(reference: str, hypothesis: str) -> float:
    r, h = words(reference), words(hypothesis)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


@dataclass
class QuestionResult:
    noise: str
    snr: float | None
    mode: str  # before | after (first turn) | after (known user)
    language: str
    outcome: str  # answer | say_again | drop | ignored | missed | split
    wer: float
    text: str
    reason: str = ""
    background_replies: int = 0  # utterances outside the question that got an answer or a "say again"


def question_outcome(run: Run, q_start: float, q_end: float) -> tuple[str, str, str, int]:
    """(outcome, transcript, reason, background replies) of a question trial."""
    inside = [u for u in run.utterances if u.end > q_start + 0.1 and u.start < q_end + 0.3]
    outside = [u for u in run.utterances if u not in inside]
    replies = sum(u.outcome in ("answer", "say_again") for u in outside)
    if not inside:
        return "missed", "", "", replies
    text = " ".join(u.text for u in inside)
    main = max(inside, key=lambda u: u.end - u.start)
    outcome = main.outcome if len(inside) == 1 else ("split" if main.outcome == "answer" else main.outcome)
    return outcome, text, main.reason, replies


# ------------------------------------------------------------------ the run


def models_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if (REPO / "data/models/huggingface").is_dir():
        return REPO / "data/models"
    common = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent / "data/models"  # a worktree: the main checkout's models (read only)


def settings_for(models: Path, root: Path) -> Settings:
    env = {
        **os.environ,
        "APP_ROOT_DIR": str(root),
        "MODEL_CACHE__HF_HOME": str(models / "huggingface"),
        "VECTOR_STORE__COLLECTION": "noise_eval_unused",  # never opened: only the VAD and STT are used
    }
    s = load_settings(REPO / "config/local.config.json", env)
    apply_runtime_env(s)
    return s


def mix(speech: np.ndarray, noise_: np.ndarray, start: int) -> np.ndarray:
    out = noise_.copy()
    out[start : start + len(speech)] += speech
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data/eval/noise"))
    ap.add_argument("--models", default=None)
    ap.add_argument("--quick", action="store_true", help="4 questions, 20 s noise-only streams")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    m = material(out)
    questions = m.questions[:2] + m.questions[-2:] if args.quick else m.questions
    noise_s = 20.0 if args.quick else NOISE_ONLY_S

    root = Path(tempfile.mkdtemp(prefix="noise-eval-"))
    settings = settings_for(models_root(args.models), root)
    container = build_container(settings)
    vad, stt = container["vad"], container["stt"]
    t0 = time.perf_counter()
    await stt.transcribe(np.zeros(SR, dtype=np.float32), ["en"])  # load the model
    print(f"models ready in {time.perf_counter() - t0:.1f} s", flush=True)
    sims = {"before": Simulator(settings, vad, stt, after=False), "after": Simulator(settings, vad, stt, after=True)}

    # ---- noise-only streams: false turn starts, background replies, false barge-ins
    streams: list[tuple[str, int, np.ndarray]] = []
    for kind in NOISE_TYPES:
        base = noise(kind, noise_s, m, rng)
        for snr in NOISE_ONLY_SNRS:
            level = USER_DBFS - snr
            x = at_level(base, level, active=kind == "tv") if kind == "tv" else base * 10 ** (level / 20)
            streams.append((kind, snr, mic(np.asarray(x, dtype=np.float32), rng)))
    denoised, cpu = denoise_batch([x for _, _, x in streams], out / "work")
    print(f"RNNoise: {json.dumps(cpu)}", flush=True)
    noise_rows = []
    for (kind, snr, raw), clean in zip(streams, denoised, strict=True):
        minutes = len(raw) / SR / 60
        row: dict[str, Any] = {"noise": kind, "snr": snr}
        for mode, audio, user_dbfs in (
            ("before", raw, None),
            ("after", clean, None),
            ("after_known", clean, USER_DBFS),
        ):
            sim = sims["before" if mode == "before" else "after"]
            idle = await sim.run(audio, speaking=False, user_dbfs=user_dbfs)
            talking = await sim.run(audio, speaking=True, user_dbfs=user_dbfs)
            replies = [u for u in idle.utterances if u.outcome in ("answer", "say_again")]
            row[mode] = {
                "starts_per_min": round(idle.starts / minutes, 1),
                "replies_per_min": round(len(replies) / minutes, 1),
                "barge_ins_per_min": round(talking.barge_stops / minutes, 1),
                "replies": [u.text for u in replies][:6],
                "dropped": [f"{u.text} ({u.reason})" for u in idle.utterances if u.outcome == "drop"][:6],
            }
        noise_rows.append(row)
        print(
            f"{kind:7s} {snr:>3} dB  starts/min {row['before']['starts_per_min']:5.1f}"
            f" → {row['after']['starts_per_min']:5.1f}  replies/min {row['before']['replies_per_min']:5.1f}"
            f" → {row['after']['replies_per_min']:5.1f}"
            f" ({row['after_known']['replies_per_min']:.1f} known)"
            f"  barge-ins/min {row['before']['barge_ins_per_min']:5.1f} → {row['after']['barge_ins_per_min']:5.1f}",
            flush=True,
        )

    # ---- questions in noise
    lead, tail = 4.0, 2.5
    trials: list[tuple[str, float | None, int, np.ndarray, float, float]] = []
    for kind in NOISE_TYPES:
        for snr in SNRS:
            for qi, (_lang, _text, audio) in enumerate(questions):
                total = int((lead + tail) * SR) + len(audio)
                if snr is None:
                    bed = np.zeros(total, dtype=np.float32)
                else:
                    base = noise(kind, total / SR, m, rng)
                    level = USER_DBFS - snr
                    bed = at_level(base, level) if kind == "tv" else base * 10 ** (level / 20)
                x = mic(mix(audio, np.asarray(bed, dtype=np.float32), int(lead * SR)), rng)
                trials.append((kind, snr, qi, x, lead, lead + len(audio) / SR))
            if snr is None and kind != NOISE_TYPES[0]:
                trials = [tr for tr in trials if not (tr[0] == kind and tr[1] is None)]  # clean once
    denoised_trials, _ = denoise_batch([tr[3] for tr in trials], out / "work")
    results: list[QuestionResult] = []
    for (kind, snr, qi, raw, qs, qe), clean in zip(trials, denoised_trials, strict=True):
        lang, text, _ = questions[qi]
        for mode, audio, user_dbfs in (
            ("before", raw, None),
            ("after", clean, None),
            ("after_known", clean, USER_DBFS),
        ):
            sim = sims["before" if mode == "before" else "after"]
            run = await sim.run(audio, speaking=False, user_dbfs=user_dbfs)
            outcome, said, reason, replies = question_outcome(run, qs, qe)
            results.append(
                QuestionResult(
                    "clean" if snr is None else kind, snr, mode, lang, outcome, wer(text, said), said, reason, replies
                )
            )
    print(f"{len(results)} question runs", flush=True)

    summary = summarize(noise_rows, results, cpu)
    (out / "results.json").write_text(
        json.dumps(
            {"noise_only": noise_rows, "questions": [asdict(r) for r in results], "summary": summary},
            indent=1,
            ensure_ascii=False,
        )
    )
    report = render(summary)
    (out / "report.md").write_text(report)
    print(report)
    os._exit(0)  # onnxruntime / MLX threads can abort during interpreter shutdown on macOS (§9.6)


def summarize(noise_rows: list[dict[str, Any]], results: list[QuestionResult], cpu: dict[str, Any]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[QuestionResult]] = {}
    for r in results:
        key = ("clean" if r.snr is None else f"{r.snr:g} dB", r.mode)
        by.setdefault(key, []).append(r)
    q = {}
    for (snr, mode), rs in by.items():
        n = len(rs)
        q.setdefault(snr, {})[mode] = {
            "n": n,
            "answered": sum(r.outcome in ("answer", "split") for r in rs) / n,
            "say_again": sum(r.outcome == "say_again" for r in rs) / n,
            "dropped": sum(r.outcome == "drop" for r in rs) / n,
            "missed": sum(r.outcome in ("missed", "ignored") for r in rs) / n,
            "wer": float(np.mean([r.wer for r in rs])),
            "background_replies": sum(r.background_replies for r in rs),
        }
    per_noise_wer: dict[str, dict[str, float]] = {}
    for r in results:
        if r.snr is not None and r.snr <= 10:
            per_noise_wer.setdefault(r.noise, {}).setdefault(r.mode, []).append(r.wer)  # type: ignore[arg-type]
    return {
        "noise_only": noise_rows,
        "questions": q,
        "wer_by_noise_at_10_to_0_db": {
            k: {mode: float(np.mean(v)) for mode, v in modes.items()} for k, modes in per_noise_wer.items()
        },
        "rnnoise_cpu": cpu,
    }


def render(s: dict[str, Any]) -> str:
    lines = ["## Noise-only audio (per minute): before → after (after, user's level known)", ""]
    lines.append("| noise | level vs user | false turn starts | background replies | false barge-ins |")
    lines.append("|---|---|---|---|---|")
    for row in s["noise_only"]:
        b, a, k = row["before"], row["after"], row["after_known"]
        lines.append(
            f"| {row['noise']} | {row['snr']} dB below | {b['starts_per_min']} → {a['starts_per_min']} | "
            f"{b['replies_per_min']} → {a['replies_per_min']} ({k['replies_per_min']}) | "
            f"{b['barge_ins_per_min']} → {a['barge_ins_per_min']} ({k['barge_ins_per_min']}) |"
        )
    lines += ["", "## Questions in noise: answered / asked again / dropped / missed, WER", ""]
    lines.append("| SNR | before | after (first turn) | after (user's level known) |")
    lines.append("|---|---|---|---|")

    def cell(d: dict[str, Any]) -> str:
        return (
            f"{d['answered']:.0%} / {d['say_again']:.0%} / {d['dropped']:.0%} / {d['missed']:.0%}, WER {d['wer']:.2f}"
        )

    for snr in ["clean", "20 dB", "10 dB", "5 dB", "0 dB"]:
        if snr in s["questions"]:
            row = s["questions"][snr]
            lines.append(f"| {snr} | {cell(row['before'])} | {cell(row['after'])} | {cell(row['after_known'])} |")
    lines += [
        "",
        "WER by noise at 10-0 dB SNR (before → after): "
        + ", ".join(f"{k} {v['before']:.2f} → {v['after']:.2f}" for k, v in s["wer_by_noise_at_10_to_0_db"].items()),
    ]
    c = s["rnnoise_cpu"]
    lines.append(
        f"RNNoise CPU: {c['us_per_quantum']} µs per 128-sample render quantum (2.67 ms of audio), real-time factor "
        f"{c['realtime_factor']} over {c['audio_s']:.0f} s"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    asyncio.run(main())
