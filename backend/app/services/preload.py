"""Model preloading at startup (docs/DESIGN.md §9.5): the conversation models load in the background while the app
already serves requests, so the first spoken question isn't slowed by loading them. Reported in ``/health``.

Models load one after another (no memory spikes or GPU contention on a 16 GB machine, §8): VAD, STT, TTS, embedder,
reranker, then the LLM is asked to load its chat model. A model that can't load (``ml`` group not installed, files
missing) is reported and still loads on first use if that becomes possible; nothing here fails startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Literal

from pydantic import BaseModel

from ..providers.base import PlaceholderProvider
from ..providers.models import ModelUnavailableError
from ..providers.registry import Container

log = logging.getLogger(__name__)

PRELOAD_ORDER = ("vad", "stt", "tts", "embeddings", "reranker", "llm")

LoadState = Literal["pending", "loading", "ready", "unavailable", "failed"]


class ModelLoad(BaseModel):
    capability: str
    provider: str
    state: LoadState
    seconds: float | None = None
    detail: str = ""


class PreloadReport(BaseModel):
    """``off``: not started (tests, or disabled by the caller); ``loading``; ``ready``: everything loaded;
    ``degraded``: something is unavailable or failed (see ``models``)."""

    state: Literal["off", "loading", "ready", "degraded"]
    models: list[ModelLoad]


class ModelPreloader:
    def __init__(self, container: Container) -> None:
        self.container = container
        self._loads = [
            ModelLoad(capability=cap, provider=container[cap].name, state="pending")
            for cap in PRELOAD_ORDER
            if cap in container.providers and not isinstance(container[cap], PlaceholderProvider)
        ]
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="model-preload")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def wait(self) -> None:
        """Until the preload is over (at once if it never started). Never raises, also if it was cancelled."""
        if self._task is not None:
            await asyncio.wait([self._task])

    def report(self) -> PreloadReport:
        if self._task is None:
            return PreloadReport(state="off", models=self._loads)
        states = {m.state for m in self._loads}
        if states & {"pending", "loading"}:
            state = "loading"
        elif states <= {"ready"}:
            state = "ready"
        else:
            state = "degraded"
        return PreloadReport(state=state, models=self._loads)

    async def _run(self) -> None:
        started = time.perf_counter()
        for load in self._loads:
            provider = self.container[load.capability]
            load.state = "loading"
            t0 = time.perf_counter()
            try:
                await provider.preload()
            except asyncio.CancelledError:
                load.state, load.detail = "pending", "cancelled"
                raise
            except ModelUnavailableError as e:
                load.state, load.detail = "unavailable", str(e)
                log.info("preload %s (%s): unavailable: %s", load.capability, load.provider, e)
            except Exception as e:
                load.state, load.detail = "failed", f"{type(e).__name__}: {e}"
                log.warning("preload %s (%s) failed: %s", load.capability, load.provider, load.detail)
            else:
                load.state = "ready"
            load.seconds = round(time.perf_counter() - t0, 2)
        log.info(
            "model preload finished in %.1fs: %s",
            time.perf_counter() - started,
            ", ".join(f"{m.capability} {m.state} {m.seconds}s" for m in self._loads),
        )
