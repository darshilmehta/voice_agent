"""InProcessJobQueue: background jobs on the event loop, bounded concurrency, idle hook, graceful shutdown."""

from __future__ import annotations

import asyncio

import pytest

from app.providers.base import ProviderContext
from app.providers.runtime import InProcessJobQueue

from .conftest import mock_http


@pytest.fixture
def make_queue(load_local):
    def make(concurrency: int = 1) -> InProcessJobQueue:
        settings = load_local(JOB_QUEUE__CONCURRENCY=str(concurrency))
        return InProcessJobQueue(settings.job_queue, ProviderContext(settings=settings, http=mock_http()))

    return make


async def test_jobs_run_in_order_and_idle_hook_fires_once_drained(make_queue):
    q = make_queue()
    await q.start()
    done: list[str] = []
    idle: list[list[str]] = []

    async def on_idle() -> None:
        idle.append(list(done))

    q.on_idle(on_idle)

    def job(name: str):
        async def run() -> None:
            await asyncio.sleep(0.01)
            done.append(name)

        return run

    for name in ("a", "b", "c"):
        await q.submit(name, job(name))
    await q.join()
    assert done == ["a", "b", "c"]
    assert idle == [["a", "b", "c"]]  # once, after the last job, not between jobs
    await q.close()


async def test_concurrency_is_bounded(make_queue):
    q = make_queue(concurrency=2)
    active = peak = 0

    async def job() -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1

    for i in range(6):
        await q.submit(f"job{i}", job)  # workers start lazily on first submit
    await q.join()
    assert peak == 2
    await q.close()


async def test_a_failing_job_does_not_stop_the_queue(make_queue, caplog):
    q = make_queue()
    ran: list[str] = []

    async def boom() -> None:
        raise RuntimeError("kaboom")

    async def ok() -> None:
        ran.append("ok")

    await q.submit("boom", boom)
    await q.submit("ok", ok)
    await q.join()
    assert ran == ["ok"]
    assert any("job boom failed" in r.getMessage() for r in caplog.records)
    assert (await q.health()).status == "ok"
    await q.close()


async def test_close_drops_queued_jobs_and_cancels_running_ones_after_the_grace(make_queue, monkeypatch):
    q = make_queue()
    monkeypatch.setattr(q, "SHUTDOWN_GRACE_S", 0.05)
    started = asyncio.Event()
    cancelled: list[str] = []
    ran: list[str] = []

    async def slow() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append("slow")
            raise

    async def never() -> None:
        ran.append("never")

    await q.submit("slow", slow)
    await q.submit("never", never)
    await started.wait()
    await q.close()
    assert cancelled == ["slow"] and ran == []
    with pytest.raises(RuntimeError, match="shut down"):
        await q.submit("late", never)
    assert "stopped" in (await q.health()).detail


async def test_close_lets_a_short_running_job_finish(make_queue):
    q = make_queue()
    finished: list[str] = []

    async def short() -> None:
        await asyncio.sleep(0.02)
        finished.append("short")

    await q.submit("short", short)
    await asyncio.sleep(0)  # let the worker pick it up
    await q.close()
    assert finished == ["short"]


async def test_close_without_start_is_a_no_op(make_queue):
    await make_queue().close()
