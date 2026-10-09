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


# Lanes of the job queue. A lane has its own workers, so a job in one never waits for a job in another.
LANE_LONG = "long"  # heavy, serialized work (document ingestion); the default
LANE_SHORT = "short"  # quick jobs that must not wait behind a long one (automatic chat titles)


class JobQueue(Provider):
    """Background work: document ingestion and automatic chat titles. Jobs are coroutine functions; their outcome is
    recorded by the job itself (e.g. a document's status), so the queue only runs them and never retries.

    Jobs run in named lanes (``LANE_LONG``, ``LANE_SHORT``). Within a lane they run in order, ``concurrency`` at a time
    for the long lane; lanes don't wait for each other. Whoever submits a job picks the lane that fits it: anything
    that takes seconds and shouldn't be held up by an ingestion (it may sit through a whole PDF conversion and the
    startup model preload) belongs in ``LANE_SHORT``.
    """

    capability = "job_queue"

    async def submit(self, name: str, job: Job, *, lane: str = LANE_LONG) -> None:
        """Queue ``job`` to run in the background in ``lane``. ``name`` is for logs."""
        raise NotImplementedError(f"{type(self).__name__}.submit")

    def on_idle(self, callback: IdleCallback, *, lane: str = LANE_LONG) -> None:
        """Call ``callback`` each time ``lane`` drains (no job of it queued or running), e.g. to unload the ingestion
        models. A short job finishing never triggers the long lane's callbacks."""
        raise NotImplementedError(f"{type(self).__name__}.on_idle")

    async def join(self, lane: str | None = None) -> None:
        """Wait until every job submitted to ``lane`` (all lanes by default), and the idle callbacks after the last
        one, have finished."""
        raise NotImplementedError(f"{type(self).__name__}.join")


class _Lane:
    """One lane of an ``InProcessJobQueue``: a FIFO queue, its worker tasks and its idle callbacks."""

    def __init__(self, name: str, workers: int) -> None:
        self.name = name
        self.worker_count = workers
        self.queue: asyncio.Queue[tuple[str, Job]] | None = None
        self.workers: list[asyncio.Task[None]] = []
        self.idle_callbacks: list[IdleCallback] = []
        self.running = 0

    @property
    def queued(self) -> int:
        return self.queue.qsize() if self.queue is not None else 0


@register
class InProcessJobQueue(JobQueue):
    """asyncio queues, one per lane, with worker tasks in the app's event loop (one backend process): the long lane has
    ``job_queue.concurrency`` workers (1 by default: ingestions run one at a time), the short lane
    ``SHORT_LANE_WORKERS``, so a title is written while a PDF is still being converted.

    Shutdown (``close``) is graceful: new submissions are refused, jobs not started yet are dropped, running jobs get
    ``SHUTDOWN_GRACE_S`` to finish (all lanes together) and are then cancelled. Whoever submits jobs must be able to
    pick up dropped or interrupted work at the next startup (ingestion re-queues documents left PENDING/PROCESSING; a
    chat's title is retried at its next agent message).
    """

    name = "in_process"
    SHUTDOWN_GRACE_S = 5.0
    SHORT_LANE_WORKERS = 2  # titles only call the LLM server, which queues requests itself; no model contention here

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._lanes = {
            LANE_LONG: _Lane(LANE_LONG, self.concurrency),
            LANE_SHORT: _Lane(LANE_SHORT, self.SHORT_LANE_WORKERS),
        }
        self._closed = False

    @property
    def concurrency(self) -> int:
        """Workers of the long lane (``job_queue.concurrency``)."""
        return self.config.concurrency  # type: ignore[attr-defined]

    @property
    def queued(self) -> int:
        return sum(lane.queued for lane in self._lanes.values())

    @property
    def running(self) -> int:
        return sum(lane.running for lane in self._lanes.values())

    def _lane(self, name: str) -> _Lane:
        try:
            return self._lanes[name]
        except KeyError:
            raise ValueError(f"unknown job lane {name!r} (available: {', '.join(self._lanes)})") from None

    async def start(self) -> None:
        for lane in self._lanes.values():
            self._ensure_workers(lane)

    def _ensure_workers(self, lane: _Lane) -> asyncio.Queue[tuple[str, Job]]:
        if lane.queue is None:
            lane.queue = asyncio.Queue()
            lane.workers = [
                asyncio.create_task(self._work(lane), name=f"job-worker-{lane.name}-{i}")
                for i in range(lane.worker_count)
            ]
        return lane.queue

    async def submit(self, name: str, job: Job, *, lane: str = LANE_LONG) -> None:
        target = self._lane(lane)
        if self._closed:
            raise RuntimeError(f"job queue is shut down; {name!r} not queued")
        self._ensure_workers(target).put_nowait((name, job))

    def on_idle(self, callback: IdleCallback, *, lane: str = LANE_LONG) -> None:
        self._lane(lane).idle_callbacks.append(callback)

    async def join(self, lane: str | None = None) -> None:
        lanes = self._lanes.values() if lane is None else [self._lane(lane)]
        await asyncio.gather(*(entry.queue.join() for entry in lanes if entry.queue is not None))

    async def _work(self, lane: _Lane) -> None:
        queue = lane.queue
        assert queue is not None
        while True:
            name, job = await queue.get()
            lane.running += 1
            try:
                await job()
            except asyncio.CancelledError:
                raise
            except Exception:  # a failing job must not stop the worker; the job records its own outcome
                log.exception("job %s failed", name)
            finally:
                lane.running -= 1
                if lane.running == 0 and queue.empty() and not self._closed:
                    await self._run_idle_callbacks(lane)
                queue.task_done()

    async def _run_idle_callbacks(self, lane: _Lane) -> None:
        for callback in lane.idle_callbacks:
            try:
                await callback()
            except Exception:
                log.exception("job queue idle callback failed (%s lane)", lane.name)

    async def close(self) -> None:
        self._closed = True
        started = [lane for lane in self._lanes.values() if lane.queue is not None]
        dropped = 0
        for lane in started:
            assert lane.queue is not None
            while not lane.queue.empty():
                lane.queue.get_nowait()
                lane.queue.task_done()
                dropped += 1
        if dropped:
            log.info("job queue shutting down: %d queued job(s) not started", dropped)
        busy = [lane.queue.join() for lane in started if lane.running and lane.queue is not None]
        if busy:  # one grace period for all lanes
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*busy), self.SHUTDOWN_GRACE_S)
        workers = [task for lane in started for task in lane.workers]
        for task in workers:
            task.cancel()
        results = await asyncio.gather(*workers, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                log.warning("job worker ended with %r", r)
        for lane in started:
            lane.queue, lane.workers = None, []

    async def health(self) -> ProviderHealth:
        if self._closed:
            return self._health(HealthStatus.OK, "stopped")
        lanes = "; ".join(
            f"{lane.name}: {lane.running} running, {lane.queued} queued (workers {lane.worker_count})"
            for lane in self._lanes.values()
        )
        return self._health(HealthStatus.OK, lanes)


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
