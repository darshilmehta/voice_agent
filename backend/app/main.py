"""FastAPI application factory.

uv run python -m app                                   # host/port from the config
uv run uvicorn --factory app.main:create_app --reload  # development
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import chats, documents, health, pins, projects, public_config, voice
from .api.deps import install_error_handlers
from .logging_setup import configure_logging
from .offline import apply_runtime_env
from .providers.registry import Container, build_container
from .services.chat_turns import wait_for_background
from .services.document_pipeline import DocumentPipeline
from .services.preload import ModelPreloader
from .services.voice import VoiceSessions
from .settings import Settings, load_settings

log = logging.getLogger("app")

SHUTDOWN_SAVE_WAIT_S = 10.0  # at shutdown, time given to saves still running before the database closes


def create_app(
    settings: Settings | None = None, container: Container | None = None, *, preload_models: bool = True
) -> FastAPI:
    """Build the app. Configuration problems raise ConfigError here, before the server accepts traffic.

    ``preload_models``: load the conversation models (VAD, STT, TTS, embedder, reranker, LLM) in the background at
    startup, reported in ``/health``. Tests that build the real providers without models turn it off."""
    settings = settings or load_settings()
    configure_logging(settings)
    container = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        apply_runtime_env(settings)
        await container.start()
        app.state.container = container
        app.state.document_pipeline = DocumentPipeline.from_container(container)
        await app.state.document_pipeline.start()  # re-queues ingestions a restart interrupted
        app.state.voice_sessions = VoiceSessions(container)
        app.state.preloader = ModelPreloader(container)
        if preload_models:
            app.state.preloader.start()
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
            await app.state.voice_sessions.close_all()
            with contextlib.suppress(TimeoutError):  # saves of stopped answers, voice session clean-ups
                await asyncio.wait_for(wait_for_background(), SHUTDOWN_SAVE_WAIT_S)
            await app.state.preloader.stop()
            await container.close()

    app = FastAPI(title=settings.client.app_title, version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_allowed_origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    install_error_handlers(app)
    for module in (health, public_config, projects, documents, chats, pins, voice):
        app.include_router(module.router)
    return app
