"""Speech providers: VAD, STT, TTS and the audio transport (docs/DESIGN.md §3.3, §9.3-§9.4).

    VoiceActivityDetector  silero         speech probability per 32 ms frame; one streaming state per voice session
    SpeechRecognizer       mlx_whisper    Apple GPU (default on this Mac: small ≈ 0.45 s per utterance, §9.3)
                           faster_whisper CPU/CUDA (Linux, cloud)
    SpeechSynthesizer      kokoro         PCM16 at 24 kHz; voices per language from the config

Rules learned in the smoke tests (§9.2-§9.4) and applied here:

- Models load from local snapshot directories, never repo ids, so no library can reach the Hub.
- Whisper gets 16 kHz float32 PCM arrays, never files (faster-whisper's file path breaks on current PyAV).
- Whisper's language detection is restricted to the configured languages (alone it may pick e.g. Urdu for Hindi).
- Silero runs on onnxruntime with the model bundled in the ``silero-vad`` wheel; importing the ``silero_vad`` package
  itself is avoided because it sets torch's global thread count to 1 (which would slow Kokoro on the CPU).
- Kokoro runs on the configured device (``cpu`` locally to keep GPU memory for the LLM, §9.5); its English G2P needs
  the bundled spaCy model, Hindi uses espeak-ng.

Heavy libraries come from the optional ``ml`` group and are imported lazily; inference runs off the event loop.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np
from pydantic import BaseModel

from . import models
from .base import HealthStatus, PlaceholderProvider, Provider, ProviderContext, ProviderHealth, local_snapshot
from .models import LazyModelProvider, ModelUnavailableError, require_modules
from .registry import register

if TYPE_CHECKING:
    from ..settings import Language, STTSection, TTSSection, VADSection

SAMPLE_RATE = 16_000  # microphone audio: PCM mono 16 kHz (what Silero and Whisper take)
MLX_CACHE_LIMIT_BYTES = 256 * 2**20  # MLX's buffer cache (mlx-whisper is the process's only MLX user), §8
VAD_FRAME_SAMPLES = 512  # 32 ms: the window Silero v5 expects at 16 kHz
KOKORO_SAMPLE_RATE = 24_000


# ------------------------------------------------------------------ VAD


class VADStream(Protocol):
    """One voice session's VAD state (Silero is recurrent: every session needs its own)."""

    async def __call__(self, frames: np.ndarray) -> list[float]:
        """Speech probability (0..1) of each frame of ``frames``: float32, shape (n, 512), 16 kHz, in order."""
        ...

    def reset(self) -> None: ...


class VoiceActivityDetector(Provider):
    capability = "vad"
    frame_samples: ClassVar[int] = VAD_FRAME_SAMPLES

    def open_stream(self) -> VADStream:
        """A new streaming state for one session."""
        raise NotImplementedError(f"{type(self).__name__}.open_stream")


def silero_model_path() -> Path | None:
    """The ONNX model inside the installed ``silero-vad`` wheel, found without importing the package."""
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).parent / "data" / "silero_vad.onnx"
    return path if path.is_file() else None


@register
class SileroVAD(VoiceActivityDetector):
    """Silero VAD v5+ on onnxruntime (CPU, one thread): one shared inference session, per-session recurrent state."""

    name = "silero"
    required_modules = ("onnxruntime", "silero_vad")
    CONTEXT_SAMPLES = 64  # samples of the previous frame the model sees in front of each frame (16 kHz)

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._lock = threading.Lock()
        self._session: Any = None

    @property
    def cfg(self) -> VADSection:
        return self.config  # type: ignore[return-value]

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def session(self) -> Any:
        if self._session is None:
            with self._lock:
                if self._session is None:
                    require_modules("vad provider 'silero'", self.required_modules)
                    path = silero_model_path()
                    if path is None:
                        raise ModelUnavailableError("the silero-vad package has no data/silero_vad.onnx")
                    import onnxruntime

                    opts = onnxruntime.SessionOptions()
                    opts.inter_op_num_threads = 1
                    opts.intra_op_num_threads = 1
                    self._session = onnxruntime.InferenceSession(
                        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
                    )
        return self._session

    def open_stream(self) -> VADStream:
        return _SileroStream(self)

    async def preload(self) -> None:
        await asyncio.to_thread(self.session)

    async def close(self) -> None:
        self._session = None

    async def health(self) -> ProviderHealth:
        missing = models.missing_modules(self.required_modules)
        if missing:
            return self._health(HealthStatus.DEGRADED, f"{', '.join(missing)} not installed ({models.ML_INSTALL_HINT})")
        state = "loaded" if self.loaded else "loads on first use"
        return self._health(HealthStatus.OK, f"Silero VAD (bundled with silero-vad; browser: vad-web), {state}")


class _SileroStream:
    def __init__(self, vad: SileroVAD) -> None:
        self._vad = vad
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, SileroVAD.CONTEXT_SAMPLES), dtype=np.float32)

    async def __call__(self, frames: np.ndarray) -> list[float]:
        if len(frames) == 0:
            return []
        return await asyncio.to_thread(self._run, frames)

    def _run(self, frames: np.ndarray) -> list[float]:
        session = self._vad.session()
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        probs: list[float] = []
        for frame in np.asarray(frames, dtype=np.float32).reshape(-1, VAD_FRAME_SAMPLES):
            x = np.concatenate([self._context, frame[None, :]], axis=1)
            out, self._state = session.run(None, {"input": x, "state": self._state, "sr": sr})
            self._context = x[:, -SileroVAD.CONTEXT_SAMPLES :]
            probs.append(float(np.asarray(out).reshape(-1)[0]))
        return probs


# ------------------------------------------------------------------ STT


@dataclass(frozen=True, slots=True)
class Transcript:
    """One transcribed utterance. The confidence fields are Whisper's own (None when the recognizer can't tell, or
    nothing was decoded): ``avg_logprob`` is the mean log probability of the decoded tokens (clear speech is above
    about -0.5; Whisper itself retries a decode below -1.0), ``no_speech_prob`` the probability that the audio held no
    speech at all. ``likely_misheard`` combines them."""

    text: str
    language: Language  # spoken language: detected among the allowed ones, or the only one allowed
    language_probability: float | None = None  # None when the language was given, not detected
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    prompted: bool = False  # decoded with a vocabulary prompt (the chat's names and terms)

    @property
    def likely_misheard(self) -> bool:
        """The transcript is probably garbled (worth asking the user to repeat rather than answering it): words were
        decoded, but with a low mean log probability, or from audio Whisper thinks held no speech. Thresholds from
        the synthetic-clip benchmark (DESIGN §9.3): clean questions -0.05 to -0.45, garbled ones below -0.6."""
        if not self.text.strip():
            return False
        if self.avg_logprob is not None and self.avg_logprob < MISHEARD_LOGPROB:
            return True
        return self.no_speech_prob is not None and self.no_speech_prob > MISHEARD_NO_SPEECH


MISHEARD_LOGPROB = -0.7
MISHEARD_NO_SPEECH = 0.6  # Whisper's own no-speech threshold

# Whisper conditions on at most 223 prompt tokens (half its 448-token text context); a vocabulary prompt longer than
# this is cut by the model from its start. Callers keep prompts well under it (it is also decoding time).
PROMPT_MAX_TOKENS = 223

TranscriptionPrompts = Mapping[str, str]  # language → vocabulary prompt (the chat's names and terms, §9.3)


class SpeechRecognizer(Provider):
    capability = "stt"

    async def transcribe(
        self, pcm16k: np.ndarray, languages: Sequence[Language], *, prompts: TranscriptionPrompts | None = None
    ) -> Transcript:
        """Transcribe one utterance: float32 mono PCM at 16 kHz. ``languages`` restricts language detection (one
        language: no detection). ``prompts``: a vocabulary prompt per language (names and terms the speaker may use);
        the one for the spoken language, once known, biases the decoding toward those spellings."""
        raise NotImplementedError(f"{type(self).__name__}.transcribe")

    def count_tokens(self, text: str) -> int | None:
        """How many tokens the recognizer's tokenizer makes of ``text`` (None: unknown). A word the model knows well
        is one token; a name it has never seen is several, which is what a vocabulary prompt should carry."""
        return None


def pick_language(probabilities: Mapping[str, float], languages: Sequence[Language]) -> tuple[Language, float]:
    """The most probable of the allowed languages."""
    best = max(languages, key=lambda lang: probabilities.get(lang, 0.0))
    return best, float(probabilities.get(best, 0.0))


def segment_confidence(segments: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """(mean token log probability, highest no-speech probability) over decoded segments (dicts with ``avg_logprob``,
    ``no_speech_prob`` and ``tokens``); (None, None) without segments."""
    weighted, count, no_speech = 0.0, 0, None
    for s in segments:
        logprob = s.get("avg_logprob")
        n = max(1, len(s.get("tokens") or ()))
        if logprob is not None:
            weighted, count = weighted + float(logprob) * n, count + n
        p = s.get("no_speech_prob")
        if p is not None:
            no_speech = float(p) if no_speech is None else max(no_speech, float(p))
    return (weighted / count if count else None), no_speech


def _squash(text: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.casefold()).split())


def echoes_prompt(text: str, prompt: str | None) -> bool:
    """The transcript only repeats (part of) the prompt: Whisper's known failure on near-silence when prompted
    ("Valmora Industries, Zephyra Logistics" for a breath)."""
    if not prompt:
        return False
    words = _squash(text)
    return len(words.split()) >= 2 and words in _squash(prompt)


# A prompted decode is one greedy pass, its length capped by the audio's: a prompt can tip Whisper into a repetition
# loop ("ॐ ॐ ॐ …" to the 224-token limit, measured on Hindi clips), and its own temperature fallback then re-decoded
# it up to five times (3.9-5.8 s instead of 0.5 s). Loops, and other unreliable prompted decodes, are decoded again
# without the prompt, exactly as an unprompted transcription (fallback included).
PROMPTED_TOKENS_BASE = 16
PROMPTED_TOKENS_PER_S = 30  # Devanagari costs ~1.3 tokens per letter: fast Hindi speech is ~20 tokens/s
COMPRESSION_RATIO_MAX = 2.4  # Whisper's own threshold for a looping decode
LOGPROB_MIN = -1.0  # Whisper's own threshold for a failed decode


def prompted_token_cap(samples: int) -> int:
    return min(PROMPT_MAX_TOKENS, PROMPTED_TOKENS_BASE + int(PROMPTED_TOKENS_PER_S * samples / SAMPLE_RATE))


def prompted_decode_failed(text: str, segments: Sequence[Mapping[str, Any]], prompt: str) -> bool:
    """The prompted decode can't be trusted: it echoes the prompt, loops, decoded poorly, or came from audio Whisper
    thinks held no speech (with a prompt Whisper turns noise into "Thank you." where it otherwise says nothing)."""
    if echoes_prompt(text, prompt):
        return True
    for s in segments:
        if float(s.get("compression_ratio") or 0.0) > COMPRESSION_RATIO_MAX:
            return True
    logprob, no_speech = segment_confidence(segments)
    if logprob is not None and logprob < LOGPROB_MIN:
        return True
    return no_speech is not None and no_speech > MISHEARD_NO_SPEECH


class _Whisper(SpeechRecognizer, LazyModelProvider):
    uses_torch = False  # MLX / CTranslate2: not under the torch gate (MLX measured safe alongside torch MPS)

    @property
    def cfg(self) -> STTSection:
        return self.config  # type: ignore[return-value]

    async def transcribe(
        self, pcm16k: np.ndarray, languages: Sequence[Language], *, prompts: TranscriptionPrompts | None = None
    ) -> Transcript:
        allowed = list(dict.fromkeys(languages or self.cfg.languages))
        if not allowed:
            raise ValueError("no language allowed for transcription")
        audio = np.ascontiguousarray(pcm16k, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return Transcript("", allowed[0])
        return await self._run(self._transcribe_sync, audio, allowed, dict(prompts or {}))

    def _transcribe_sync(self, audio: np.ndarray, languages: list[Language], prompts: dict[str, str]) -> Transcript:
        with self._lock:
            model = self._model_locked()
            language, probability = self._language_locked(model, audio, languages)
            prompt = (prompts.get(language) or "").strip() or None
            if prompt is not None:
                text, segments = self._decode_locked(model, audio, language, prompt)
                if prompted_decode_failed(text, segments, prompt):
                    prompt = None
            if prompt is None:
                text, segments = self._decode_locked(model, audio, language, None)
            logprob, no_speech = segment_confidence(segments)
            return Transcript(text, language, probability, logprob, no_speech, prompted=prompt is not None)

    def _language_locked(
        self, model: Any, audio: np.ndarray, languages: list[Language]
    ) -> tuple[Language, float | None]:
        """The spoken language among ``languages`` (no detection for one) and its probability (None if given)."""
        raise NotImplementedError

    def _decode_locked(
        self, model: Any, audio: np.ndarray, language: Language, prompt: str | None
    ) -> tuple[str, list[dict[str, Any]]]:
        """(text, segments as dicts with avg_logprob, no_speech_prob, compression_ratio, tokens). With a prompt: one
        greedy pass of at most ``prompted_token_cap`` tokens; without: Whisper's usual decode (temperature fallback)."""
        raise NotImplementedError

    def _warm(self, model: Any) -> None:
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        language, _ = self._language_locked(model, silence, list(self.cfg.languages))
        self._decode_locked(model, silence, language, None)


@register
class MlxWhisper(_Whisper):
    """mlx-whisper on the Apple GPU. MLX work stays on one dedicated thread (MLX streams are per thread)."""

    name = "mlx_whisper"
    required_modules = ("mlx_whisper", "mlx")

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._executor: ThreadPoolExecutor | None = None
        self._path = ""

    def extra_problems(self) -> list[str]:
        return [] if models.APPLE_SILICON else ["mlx-whisper runs only on Apple Silicon; use faster_whisper elsewhere"]

    def count_tokens(self, text: str) -> int | None:
        try:  # Whisper's multilingual BPE (tiktoken, bundled with mlx-whisper): no model needed
            from mlx_whisper.tokenizer import get_tokenizer

            return len(get_tokenizer(multilingual=True).encode(text))
        except Exception:
            return None

    async def _run[T](self, fn: Any, *args: Any) -> T:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-whisper")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _load(self, snapshot: Path) -> Any:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        self._path = str(snapshot)
        # MLX keeps freed Metal buffers for reuse, without limit by default: 751 MiB after the first transcriptions
        # beside the 462 MiB model (measured). Capped, a transcription re-allocates some of them (+17 ms p50).
        mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        # transcribe() looks the model up in this holder by path: loading through it keeps a single copy in memory.
        return ModelHolder.get_model(self._path, mx.float16)

    def _language_locked(
        self, model: Any, audio: np.ndarray, languages: list[Language]
    ) -> tuple[Language, float | None]:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        if len(languages) == 1:
            return languages[0], None
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
        return pick_language(probs, languages)

    def _decode_locked(
        self, model: Any, audio: np.ndarray, language: Language, prompt: str | None
    ) -> tuple[str, list[dict[str, Any]]]:
        import mlx_whisper

        prompted: dict[str, Any] = {}
        if prompt is not None:
            prompted = {"initial_prompt": prompt, "temperature": 0.0, "sample_len": prompted_token_cap(audio.size)}
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._path,
            language=language,
            fp16=True,
            verbose=None,
            condition_on_previous_text=False,
            **prompted,
        )
        return str(result.get("text", "")).strip(), list(result.get("segments") or [])

    def _release(self, model: Any) -> None:
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model, ModelHolder.model_path = None, None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # best effort
            pass

    async def close(self) -> None:
        await super().close()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


@register
class FasterWhisper(_Whisper):
    """faster-whisper (CTranslate2) on the CPU or CUDA. CPU-only on a Mac, so the Linux/cloud provider (§9.3)."""

    name = "faster_whisper"
    required_modules = ("faster_whisper",)

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._tokenizer: Any = None

    def extra_problems(self) -> list[str]:
        return [] if self.cfg.device in ("cpu", "cuda") else [f"device {self.cfg.device!r}: use cpu or cuda"]

    def count_tokens(self, text: str) -> int | None:
        try:  # the model's own tokenizer.json, without loading the model
            if self._tokenizer is None:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(self.snapshot_path() / "tokenizer.json"))
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        except Exception:
            return None

    def _load(self, snapshot: Path) -> Any:
        from faster_whisper import WhisperModel

        cfg = self.cfg
        return WhisperModel(
            str(snapshot), device=cfg.device, compute_type=cfg.compute_type or "default", local_files_only=True
        )

    def _language_locked(
        self, model: Any, audio: np.ndarray, languages: list[Language]
    ) -> tuple[Language, float | None]:
        if len(languages) == 1:
            return languages[0], None
        _, _, all_probs = model.detect_language(audio)
        return pick_language(dict(all_probs), languages)

    def _decode_locked(
        self, model: Any, audio: np.ndarray, language: Language, prompt: str | None
    ) -> tuple[str, list[dict[str, Any]]]:
        prompted: dict[str, Any] = {}
        if prompt is not None:
            prompted = {"initial_prompt": prompt, "temperature": 0.0, "max_new_tokens": prompted_token_cap(audio.size)}
        segments, _ = model.transcribe(
            audio, language=language, beam_size=1, condition_on_previous_text=False, vad_filter=False, **prompted
        )
        decoded = [  # segments is lazy: decoding happens here
            {
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "compression_ratio": s.compression_ratio,
                "tokens": s.tokens,
            }
            for s in segments
        ]
        return "".join(s["text"] for s in decoded).strip(), decoded


# ------------------------------------------------------------------ TTS


class SpeechSynthesizer(Provider):
    capability = "tts"

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio ``synthesize`` returns."""
        return int(self.config.sample_rate)  # type: ignore[attr-defined]

    async def synthesize(self, text: str, language: Language) -> bytes:
        """Speech for ``text`` in ``language``: PCM16 little-endian mono at ``sample_rate``."""
        raise NotImplementedError(f"{type(self).__name__}.synthesize")


def to_pcm16(audio: np.ndarray) -> bytes:
    """Float audio in [-1, 1] → PCM16 little-endian bytes."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


ESPEAK_PATH_MAX = 150  # espeak-ng copies its data path into a 160-byte buffer


def shorten_espeak_data_path() -> None:
    """espeak-ng (Kokoro's Hindi G2P, via misaki) keeps its data path in a 160-byte buffer. A longer path, as in a deep
    virtualenv (git worktrees), is cut silently; espeak then falls back to a path compiled into the wheel and *exits
    the process*. When the bundled data path is that long, point ``ESPEAK_DATA_PATH`` (espeak's next choice, which
    phonemizer doesn't resolve) at a short symlink to it, made in a fresh private directory (``mkdtemp``: mode 0700,
    unpredictable name), so no other user can plant or swap the link."""
    if "ESPEAK_DATA_PATH" in os.environ:
        return
    spec = importlib.util.find_spec("espeakng_loader")
    if spec is None or spec.origin is None:
        return
    data = Path(spec.origin).parent / "espeak-ng-data"
    if len(str(data)) < ESPEAK_PATH_MAX or not data.is_dir():
        return
    link = Path(tempfile.mkdtemp(prefix="espeak-")) / "data"
    link.symlink_to(data, target_is_directory=True)
    if len(str(link)) >= ESPEAK_PATH_MAX:  # an unusually long TMPDIR: nothing short to point at
        return
    os.environ["ESPEAK_DATA_PATH"] = str(link)


@dataclass(slots=True)
class _Kokoro:
    pipelines: dict[str, Any]  # language → KPipeline (sharing one KModel)
    voices: dict[str, str]  # language → voice pack path
    model: Any


@register
class KokoroTTS(SpeechSynthesizer, LazyModelProvider):
    name = "kokoro"
    required_modules = ("kokoro", "misaki", "torch", "en_core_web_sm")

    @property
    def cfg(self) -> TTSSection:
        return self.config  # type: ignore[return-value]

    @property
    def sample_rate(self) -> int:
        return KOKORO_SAMPLE_RATE  # the model's native rate; health flags a config that says otherwise

    def extra_problems(self) -> list[str]:
        problems = []
        snap = local_snapshot(self.ctx.hf_home, self.model_repo())
        voices: dict[str, str] = self.cfg.voices
        missing = [f"{lang}:{v}" for lang, v in voices.items() if snap and not (snap / "voices" / f"{v}.pt").is_file()]
        if missing:
            problems.append(f"voice packs missing: {', '.join(missing)}")
        if self.cfg.sample_rate != KOKORO_SAMPLE_RATE:
            problems.append(f"tts.sample_rate is {self.cfg.sample_rate}, Kokoro speaks at {KOKORO_SAMPLE_RATE}")
        return problems

    async def synthesize(self, text: str, language: Language) -> bytes:
        if language not in self.cfg.voices:
            raise ValueError(f"no Kokoro voice configured for {language!r} (tts.voices)")
        if not text.strip():
            return b""
        return await self._run(self._synthesize_sync, text, language)

    def _load(self, snapshot: Path) -> Any:
        cfg = self.cfg
        if cfg.device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a few Kokoro ops have no MPS kernel
        shorten_espeak_data_path()
        from kokoro import KModel, KPipeline

        weights = sorted(snapshot.glob("*.pth"))
        if not weights:
            raise ModelUnavailableError(f"no Kokoro weights (*.pth) in {snapshot}")
        repo = self.model_repo()
        model = KModel(repo_id=repo, config=str(snapshot / "config.json"), model=str(weights[-1])).to(cfg.device).eval()
        # A voice's first letter is its language: a/b = American/British English, h = Hindi (Kokoro's lang codes).
        pipelines = {
            lang: KPipeline(lang_code=voice[0], repo_id=repo, model=model) for lang, voice in cfg.voices.items()
        }
        voices = {lang: str(snapshot / "voices" / f"{voice}.pt") for lang, voice in cfg.voices.items()}
        return _Kokoro(pipelines, voices, model)

    def _warm(self, model: Any) -> None:
        for language in model.pipelines:
            self._synthesize_locked(model, "Ready." if language == "en" else "ठीक है।", language)

    def _synthesize_sync(self, text: str, language: Language) -> bytes:
        with self._lock:
            return self._synthesize_locked(self._model_locked(), text, language)

    def _synthesize_locked(self, kokoro: _Kokoro, text: str, language: str) -> bytes:
        import torch

        parts = []
        with self._torch_use(), torch.inference_mode():
            for result in kokoro.pipelines[language](text, voice=kokoro.voices[language], speed=1.0):
                audio = result.audio
                if audio is None:
                    continue
                parts.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        return to_pcm16(np.concatenate(parts)) if parts else b""


# ------------------------------------------------------------------ transport


class AudioTransport(Provider):
    capability = "audio_transport"


@register
class WebSocketAudio(AudioTransport):
    """Audio over the voice WebSocket (``/ws/chats/{id}/voice``): PCM16 frames both ways plus JSON control messages."""

    name = "websocket"


@register
class WebRTCAudio(AudioTransport, PlaceholderProvider):
    name = "webrtc"
