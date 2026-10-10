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
from .api import canvas as canvas_api
from .api import chats, documents, health, pins, projects, public_config, revisit, voice
from .api.deps import install_error_handlers
from .logging_setup import configure_logging
from .offline import apply_runtime_env
from .providers.registry import Container, build_container
from .services.canvas.service import CanvasService
from .services.chat_summary import ChatSummarizer
from .services.chat_turns import wait_for_background
from .services.document_pipeline import DocumentPipeline
from .services.host_checks import HostChecks
from .services.messages import add_agent_message_hook
from .services.preload import ModelPreloader
from .services.titles import TitleService
from .services.voice import VoiceSessions
from .services.voice.vocabulary import VocabularyWarmer
from .settings import Settings, load_settings

log = logging.getLogger("app")

SHUTDOWN_SAVE_WAIT_S = 10.0  # at shutdown, time given to saves still running before the database closes


def create_app(
    settings: Settings | None = None,
    container: Container | None = None,
    *,
    preload_models: bool = True,
    auto_titles: bool = True,
    host_checks: HostChecks | None = None,
) -> FastAPI:
    """Build the app. Configuration problems raise ConfigError here, before the server accepts traffic.

    ``preload_models``: load the conversation models (VAD, STT, TTS, embedder, reranker, LLM) in the background at
    startup, reported in ``/health``. Tests that build the real providers without models turn it off.

    ``auto_titles=False`` leaves chats with their placeholder title: no background LLM call follows the first answer
    (tests that count LLM calls turn it off).

    ``host_checks``: the read-only checks of this machine reported in ``/health`` (default: the real ones, §8)."""
    settings = settings or load_settings()
    configure_logging(settings)
    container = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        apply_runtime_env(settings)
        await container.start()
        app.state.container = container
        app.state.preloader = ModelPreloader(container)
        # Read-only checks of this machine (§8: Ollama's prompt cache), logged now and reported in /health.
        app.state.host_checks = host_checks or HostChecks(settings)
        await app.state.host_checks.warnings(log_them=True)
        if preload_models:
            app.state.preloader.start()
        app.state.document_pipeline = DocumentPipeline.from_container(container)
        # Ingestion starts once the preload is done: a conversion would otherwise hold off its model loads (and the
        # questions queued behind them, see models.TorchGate). Set before start(), which re-queues interrupted jobs.
        app.state.document_pipeline.wait_before_ingesting = app.state.preloader.wait
        # The canvas types each READY document's tables and keeps the project overviews (§12.1); documents ingested
        # before it get typed by a backfill job queued after any re-queued ingestion.
        canvas = app.state.canvas = CanvasService.from_container(container)
        canvas.wait_before_backfill = app.state.preloader.wait
        app.state.document_pipeline.listeners.append(canvas)
        await app.state.document_pipeline.start()  # re-queues ingestions a restart interrupted
        await canvas.schedule_backfill()
        app.state.voice_sessions = VoiceSessions(container, canvas=canvas)
        # The document names' Devanagari spellings for the Hindi speech prompt, asked of the model while it is idle
        # (after the preload, and when a document becomes READY), not as a voice session opens (polish round, item 5).
        warmer = None
        if (vocabulary := app.state.voice_sessions.vocabulary) is not None:
            warmer = VocabularyWarmer(vocabulary, canvas.store.projects_with_documents, app.state.preloader.wait)
            app.state.document_pipeline.listeners.append(warmer)
            if preload_models:
                warmer.warm_all()
        app.state.summarizer = ChatSummarizer.from_container(container)
        titles = app.state.titles = TitleService.from_container(container)
        # Titles follow the first saved agent answer (text or voice) as a background job, not as part of the turn.
        unhook = add_agent_message_hook(titles.db, titles.on_agent_message) if auto_titles else None
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
            if warmer is not None:
                await warmer.close()
            with contextlib.suppress(TimeoutError):  # saves of stopped answers, voice session clean-ups
                await asyncio.wait_for(wait_for_background(), SHUTDOWN_SAVE_WAIT_S)
            if unhook is not None:
                unhook()
            await app.state.preloader.stop()
            await container.close()

    app = FastAPI(title=settings.client.app_title, version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_allowed_origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["Content-Disposition"],  # the export's file name, read by the frontend on another origin
    )
    install_error_handlers(app)
    for module in (health, public_config, projects, documents, chats, pins, voice, revisit, canvas_api):
        app.include_router(module.router)
    return app
