"""Auth, job queue, session store and event bus providers (docs/DESIGN.md §6.4)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from .base import HealthStatus, PlaceholderProvider, Provider, ProviderContext, ProviderHealth
from .registry import register

log = logging.getLogger(__name__)

Job = Callable[[], Awaitable[Any]]
IdleCallback = Callable[[], Awaitable[None]]


class AuthProvider(Provider):
    capability = "auth"


@register
class NoAuth(AuthProvider):
    name = "none"

    async def health(self) -> ProviderHealth:
        return self._health(HealthStatus.OK, "authentication disabled (local profile)")


@register
class OIDCAuth(AuthProvider, PlaceholderProvider):
    name = "oidc"


class JobQueue(Provider):
    """Background work (document ingestion). Jobs are coroutine functions; their outcome is recorded by the job itself
    (e.g. a document's status), so the queue only runs them and never retries."""

    capability = "job_queue"

    async def submit(self, name: str, job: Job) -> None:
        """Queue ``job`` to run in the background. ``name`` is for logs."""
        raise NotImplementedError(f"{type(self).__name__}.submit")

    def on_idle(self, callback: IdleCallback) -> None:
        """Call ``callback`` each time the queue drains (no job queued or running), e.g. to unload ingestion models."""
        raise NotImplementedError(f"{type(self).__name__}.on_idle")

    async def join(self) -> None:
        """Wait until every submitted job, and the idle callbacks after the last one, have finished."""
        raise NotImplementedError(f"{type(self).__name__}.join")


@register
class InProcessJobQueue(JobQueue):
    """An asyncio queue with ``job_queue.concurrency`` worker tasks in the app's event loop (one backend process).

    Shutdown (``close``) is graceful: new submissions are refused, jobs not started yet are dropped, running jobs get
    ``SHUTDOWN_GRACE_S`` to finish and are then cancelled. Whoever submits jobs must be able to pick up dropped or
    interrupted work at the next startup (ingestion re-queues documents left PENDING/PROCESSING).
    """

    name = "in_process"
    SHUTDOWN_GRACE_S = 5.0

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._queue: asyncio.Queue[tuple[str, Job]] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._idle_callbacks: list[IdleCallback] = []
        self._running = 0
        self._closed = False

    @property
    def concurrency(self) -> int:
        return self.config.concurrency  # type: ignore[attr-defined]

    @property
    def queued(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    @property
    def running(self) -> int:
        return self._running

    async def start(self) -> None:
        self._ensure_workers()

    def _ensure_workers(self) -> asyncio.Queue[tuple[str, Job]]:
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._workers = [asyncio.create_task(self._work(), name=f"job-worker-{i}") for i in range(self.concurrency)]
        return self._queue

    async def submit(self, name: str, job: Job) -> None:
        if self._closed:
            raise RuntimeError(f"job queue is shut down; {name!r} not queued")
        self._ensure_workers().put_nowait((name, job))

    def on_idle(self, callback: IdleCallback) -> None:
        self._idle_callbacks.append(callback)

    async def join(self) -> None:
        if self._queue is not None:
            await self._queue.join()

    async def _work(self) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            name, job = await queue.get()
            self._running += 1
            try:
                await job()
            except asyncio.CancelledError:
                raise
            except Exception:  # a failing job must not stop the worker; the job records its own outcome
                log.exception("job %s failed", name)
            finally:
                self._running -= 1
                if self._running == 0 and queue.empty() and not self._closed:
                    await self._run_idle_callbacks()
                queue.task_done()

    async def _run_idle_callbacks(self) -> None:
        for callback in self._idle_callbacks:
            try:
                await callback()
            except Exception:
                log.exception("job queue idle callback failed")

    async def close(self) -> None:
        self._closed = True
        queue, workers = self._queue, self._workers
        if queue is None:
            return
        dropped = 0
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
            dropped += 1
        if dropped:
            log.info("job queue shutting down: %d queued job(s) not started", dropped)
        if self._running:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(queue.join(), self.SHUTDOWN_GRACE_S)
        for task in workers:
            task.cancel()
        results = await asyncio.gather(*workers, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                log.warning("job worker ended with %r", r)
        self._queue, self._workers = None, []

    async def health(self) -> ProviderHealth:
        state = "stopped" if self._closed else f"{self._running} running, {self.queued} queued"
        return self._health(HealthStatus.OK, f"{state} (concurrency {self.concurrency})")


@register
class RedisJobQueue(JobQueue, PlaceholderProvider):
    name = "redis"


class SessionStore(Provider):
    capability = "session_store"


@register
class InMemorySessionStore(SessionStore):
    name = "in_memory"


@register
class RedisSessionStore(SessionStore, PlaceholderProvider):
    name = "redis"


class EventBus(Provider):
    capability = "event_bus"


@register
class InProcessEventBus(EventBus):
    name = "in_process"


@register
class RedisEventBus(EventBus, PlaceholderProvider):
    name = "redis"
