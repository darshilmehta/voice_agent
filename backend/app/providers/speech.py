"""Speech providers: STT, VAD, TTS and the audio transport (docs/DESIGN.md §3.3). Methods arrive in phases 4 to 6."""

from __future__ import annotations

from . import models
from .base import HealthStatus, PlaceholderProvider, Provider, ProviderHealth, local_snapshot
from .models import LocalModelProvider
from .registry import register


class SpeechRecognizer(Provider):
    capability = "stt"


@register
class MlxWhisper(SpeechRecognizer, LocalModelProvider):
    name = "mlx_whisper"

    def extra_problems(self) -> list[str]:
        return [] if models.APPLE_SILICON else ["mlx-whisper runs only on Apple Silicon; use faster_whisper elsewhere"]


@register
class FasterWhisper(SpeechRecognizer, LocalModelProvider):
    name = "faster_whisper"


class VoiceActivityDetector(Provider):
    capability = "vad"


@register
class SileroVAD(VoiceActivityDetector):
    name = "silero"

    async def health(self) -> ProviderHealth:
        return self._health(HealthStatus.OK, "model ships inside silero-vad (server) and @ricky0123/vad-web (browser)")


class SpeechSynthesizer(Provider):
    capability = "tts"


@register
class KokoroTTS(SpeechSynthesizer, LocalModelProvider):
    name = "kokoro"

    def extra_problems(self) -> list[str]:
        snap = local_snapshot(self.ctx.hf_home, self.model_repo())
        voices: dict[str, str] = self.config.voices  # type: ignore[attr-defined]
        missing = [f"{lang}:{v}" for lang, v in voices.items() if snap and not (snap / "voices" / f"{v}.pt").is_file()]
        return [f"voice packs missing: {', '.join(missing)}"] if missing else []


class AudioTransport(Provider):
    capability = "audio_transport"


@register
class WebSocketAudio(AudioTransport):
    name = "websocket"


@register
class WebRTCAudio(AudioTransport, PlaceholderProvider):
    name = "webrtc"
