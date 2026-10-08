"""GET /api/config/public: everything the browser needs, and nothing secret.

Built field by field from the settings (never by dumping a section) so a secret added to the config
later can't leak through this endpoint.
"""

from __future__ import annotations

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


def public_config(s: Settings) -> PublicConfig:
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
    )


@router.get("/public", response_model=PublicConfig)
async def get_public_config(request: Request) -> PublicConfig:
    return public_config(request.app.state.container.settings)
