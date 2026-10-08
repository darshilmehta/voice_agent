"""FastAPI application factory.

uv run python -m app                                   # host/port from the config
uv run uvicorn --factory app.main:create_app --reload  # development
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import health, public_config
from .logging_setup import configure_logging
from .offline import apply_runtime_env
from .providers.registry import Container, build_container
from .settings import Settings, load_settings

log = logging.getLogger("app")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app. Configuration problems raise ConfigError here, before the server accepts traffic."""
    settings = settings or load_settings()
    configure_logging(settings)
    container = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        apply_runtime_env(settings)
        await container.start()
        app.state.container = container
        log.info(
            "started %s %s profile=%s strict_offline=%s config=%s",
            settings.app.name,
            __version__,
            settings.profile,
            settings.strict_offline,
            settings.source,
        )
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(title=settings.client.app_title, version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_allowed_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    app.include_router(health.router)
    app.include_router(public_config.router)
    return app
