"""GET /health: liveness plus the state of every configured provider."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from .. import __version__
from ..providers.base import HealthStatus, ProviderHealth

router = APIRouter(tags=["health"])

_HEALTHY = {HealthStatus.OK, HealthStatus.DISABLED}


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    profile: str
    strict_offline: bool
    providers: list[ProviderHealth]


@router.get("/health", response_model=HealthReport)
async def health(request: Request) -> HealthReport:
    container = request.app.state.container
    providers = await container.health()
    return HealthReport(
        status="ok" if all(p.status in _HEALTHY for p in providers) else "degraded",
        version=__version__,
        profile=container.settings.profile,
        strict_offline=container.settings.strict_offline,
        providers=providers,
    )
