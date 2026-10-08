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
import contextlib
import hashlib
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
    text: str
    language: Language  # spoken language: detected among the allowed ones, or the only one allowed
    language_probability: float | None = None  # None when the language was given, not detected


class SpeechRecognizer(Provider):
    capability = "stt"

    async def transcribe(self, pcm16k: np.ndarray, languages: Sequence[Language]) -> Transcript:
        """Transcribe one utterance: float32 mono PCM at 16 kHz. ``languages`` restricts language detection (one
        language: no detection)."""
        raise NotImplementedError(f"{type(self).__name__}.transcribe")


def pick_language(probabilities: Mapping[str, float], languages: Sequence[Language]) -> tuple[Language, float]:
    """The most probable of the allowed languages."""
    best = max(languages, key=lambda lang: probabilities.get(lang, 0.0))
    return best, float(probabilities.get(best, 0.0))


class _Whisper(SpeechRecognizer, LazyModelProvider):
    @property
    def cfg(self) -> STTSection:
        return self.config  # type: ignore[return-value]

    async def transcribe(self, pcm16k: np.ndarray, languages: Sequence[Language]) -> Transcript:
        allowed = list(dict.fromkeys(languages or self.cfg.languages))
        if not allowed:
            raise ValueError("no language allowed for transcription")
        audio = np.ascontiguousarray(pcm16k, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return Transcript("", allowed[0])
        return await self._run(self._transcribe_sync, audio, allowed)

    def _transcribe_sync(self, audio: np.ndarray, languages: list[Language]) -> Transcript:
        with self._lock:
            return self._transcribe_locked(self._model_locked(), audio, languages)

    def _transcribe_locked(self, model: Any, audio: np.ndarray, languages: list[Language]) -> Transcript:
        raise NotImplementedError

    def _warm(self, model: Any) -> None:
        self._transcribe_locked(model, np.zeros(SAMPLE_RATE, dtype=np.float32), list(self.cfg.languages))


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

    async def _run[T](self, fn: Any, *args: Any) -> T:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-whisper")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _load(self, snapshot: Path) -> Any:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        self._path = str(snapshot)
        # transcribe() looks the model up in this holder by path: loading through it keeps a single copy in memory.
        return ModelHolder.get_model(self._path, mx.float16)

    def _transcribe_locked(self, model: Any, audio: np.ndarray, languages: list[Language]) -> Transcript:
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        probability: float | None = None
        if len(languages) == 1:
            language = languages[0]
        else:
            mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
            _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
            language, probability = pick_language(probs, languages)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._path,
            language=language,
            fp16=True,
            verbose=None,
            condition_on_previous_text=False,
        )
        return Transcript(str(result.get("text", "")).strip(), language, probability)

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

    def extra_problems(self) -> list[str]:
        return [] if self.cfg.device in ("cpu", "cuda") else [f"device {self.cfg.device!r}: use cpu or cuda"]

    def _load(self, snapshot: Path) -> Any:
        from faster_whisper import WhisperModel

        cfg = self.cfg
        return WhisperModel(
            str(snapshot), device=cfg.device, compute_type=cfg.compute_type or "default", local_files_only=True
        )

    def _transcribe_locked(self, model: Any, audio: np.ndarray, languages: list[Language]) -> Transcript:
        probability: float | None = None
        if len(languages) == 1:
            language = languages[0]
        else:
            _, _, all_probs = model.detect_language(audio)
            language, probability = pick_language(dict(all_probs), languages)
        segments, _ = model.transcribe(
            audio, language=language, beam_size=1, condition_on_previous_text=False, vad_filter=False
        )
        text = "".join(s.text for s in segments).strip()  # segments is lazy: decoding happens here
        return Transcript(text, language, probability)


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
    phonemizer doesn't resolve) at a short symlink to it."""
    if "ESPEAK_DATA_PATH" in os.environ:
        return
    spec = importlib.util.find_spec("espeakng_loader")
    if spec is None or spec.origin is None:
        return
    data = Path(spec.origin).parent / "espeak-ng-data"
    if len(str(data)) < ESPEAK_PATH_MAX or not data.is_dir():
        return
    link = Path(tempfile.gettempdir()) / f"espeak-ng-data-{hashlib.sha256(str(data).encode()).hexdigest()[:12]}"
    if not link.exists():
        with contextlib.suppress(FileExistsError):  # created concurrently
            link.symlink_to(data, target_is_directory=True)
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
        with torch.inference_mode():
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
