"""GET /api/config/public: everything the browser needs, and nothing secret.

Built field by field from the settings (never by dumping a section) so a secret added to the config
later can't leak through this endpoint.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from .. import __version__
from ..settings import Language, Settings

router = APIRouter(prefix="/api/config", tags=["config"])


class PublicClient(BaseModel):
    app_title: str
    languages: list[Language]
    default_language: Language


class PublicAuth(BaseModel):
    provider: str
    issuer_url: str | None
    client_id: str | None
    audience: str | None


class PublicFeatures(BaseModel):
    voice: bool
    web_search: bool
    debug_panel: bool


class PublicVoiceInput(BaseModel):
    """The browser's side of noisy rooms (docs/DESIGN.md §3.10): its denoiser and the same noise-floor gate as the
    server's (``voice.noise``), applied to its VAD before it ducks the agent."""

    denoise: Literal["rnnoise", "off"]
    adaptive_gating: bool
    threshold: float  # vad.threshold, in a quiet room
    min_speech_ms: int  # vad.min_speech_ms, in a quiet room
    floor_window_ms: int
    floor_percentile: float
    start_snr_db: float
    quiet_floor_dbfs: float
    loud_floor_dbfs: float
    noisy_threshold: float
    noisy_min_speech_ms: int
    assumed_user_dbfs: float  # the near-field check (noise.near_field_min): the user's level until it is learnt
    assumed_margin_db: float
    far_field_hard_db: float


class PublicLimits(BaseModel):
    max_upload_mb: int
    allowed_extensions: list[str]


class PublicConfig(BaseModel):
    app_name: str
    version: str
    profile: str
    client: PublicClient
    auth: PublicAuth
    features: PublicFeatures
    limits: PublicLimits
    voice_input: PublicVoiceInput


def public_config(s: Settings) -> PublicConfig:
    noise = s.voice.noise
    return PublicConfig(
        app_name=s.app.name,
        version=__version__,
        profile=s.profile,
        client=PublicClient(
            app_title=s.client.app_title,
            languages=s.client.languages,
            default_language=s.client.default_language,
        ),
        auth=PublicAuth(
            provider=s.auth.provider,
            issuer_url=s.auth.issuer_url,
            client_id=s.auth.client_id,
            audience=s.auth.audience,
        ),
        features=PublicFeatures(
            voice=s.client.voice_enabled,
            web_search=s.tools.web_search.enabled,
            debug_panel=s.client.show_debug_panel,
        ),
        limits=PublicLimits(
            max_upload_mb=s.server.max_upload_mb,
            allowed_extensions=s.ingestion.allowed_extensions,
        ),
        voice_input=PublicVoiceInput(
            denoise=noise.denoise,
            adaptive_gating=noise.adaptive_gating,
            threshold=s.vad.threshold,
            min_speech_ms=s.vad.min_speech_ms,
            floor_window_ms=noise.floor_window_ms,
            floor_percentile=noise.floor_percentile,
            start_snr_db=noise.start_snr_db,
            quiet_floor_dbfs=noise.quiet_floor_dbfs,
            loud_floor_dbfs=noise.loud_floor_dbfs,
            noisy_threshold=noise.noisy_threshold,
            noisy_min_speech_ms=noise.noisy_min_speech_ms,
            assumed_user_dbfs=noise.assumed_user_dbfs,
            assumed_margin_db=noise.assumed_margin_db,
            far_field_hard_db=noise.far_field_hard_db,
        ),
    )


@router.get("/public", response_model=PublicConfig)
async def get_public_config(request: Request) -> PublicConfig:
    return public_config(request.app.state.container.settings)
