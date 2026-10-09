"""InProcessJobQueue: background jobs in named lanes on the event loop, bounded concurrency, idle hook, shutdown."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from app.providers.base import ProviderContext
from app.providers.runtime import LANE_LONG, LANE_SHORT, InProcessJobQueue

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


# ------------------------------------------------------------------ lanes (B4: a title must not wait for an ingestion)


def _worker_tasks() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t.get_name().startswith("job-worker-") and not t.done()]


async def test_a_short_job_completes_while_a_long_job_is_running(make_queue):
    q = make_queue()  # job_queue.concurrency = 1, like the shipped configs
    release, started = asyncio.Event(), asyncio.Event()
    finished: list[str] = []

    async def ingest() -> None:
        started.set()
        await release.wait()
        finished.append("ingest")

    async def another_ingest() -> None:
        finished.append("ingest 2")

    async def title() -> None:
        finished.append("title")

    await q.submit("ingest", ingest, lane=LANE_LONG)
    await q.submit("ingest 2", another_ingest)  # the default lane is the long one: queued behind the first
    await started.wait()
    await q.submit("title", title, lane=LANE_SHORT)
    await asyncio.wait_for(q.join(LANE_SHORT), 1)  # the old shared worker would have kept it waiting here
    assert finished == ["title"]
    assert (q.running, q.queued) == (1, 1)  # the conversion is still going, the next one still waits

    release.set()
    await q.join()
    assert finished == ["title", "ingest", "ingest 2"]
    await q.close()


@pytest.mark.parametrize("concurrency", [1, 2])
async def test_the_long_lane_keeps_its_concurrency_and_short_jobs_do_not_count_against_it(make_queue, concurrency):
    q = make_queue(concurrency=concurrency)
    active = {LANE_LONG: 0, LANE_SHORT: 0}
    peak = dict(active)
    overlapped = False

    def job(lane: str):
        async def run() -> None:
            nonlocal overlapped
            active[lane] += 1
            peak[lane] = max(peak[lane], active[lane])
            overlapped = overlapped or (active[LANE_LONG] > 0 and active[LANE_SHORT] > 0)
            await asyncio.sleep(0.02)
            active[lane] -= 1

        return run

    for i in range(5):
        await q.submit(f"ingest {i}", job(LANE_LONG), lane=LANE_LONG)
    for i in range(6):
        await q.submit(f"title {i}", job(LANE_SHORT), lane=LANE_SHORT)
    await q.join()
    assert peak == {LANE_LONG: concurrency, LANE_SHORT: q.SHORT_LANE_WORKERS}  # ingestion stays as serialized as ever
    assert overlapped
    await q.close()


async def test_idle_hooks_belong_to_their_lane(make_queue):
    """The parser's release hook is for the long lane: a title finishing never triggers it, and an ingestion still
    running keeps it from firing."""
    q = make_queue()
    idle: list[str] = []

    async def long_idle() -> None:
        idle.append("long")

    async def short_idle() -> None:
        idle.append("short")

    q.on_idle(long_idle)  # long by default
    q.on_idle(short_idle, lane=LANE_SHORT)
    release, started = asyncio.Event(), asyncio.Event()

    async def ingest() -> None:
        started.set()
        await release.wait()

    async def title() -> None:
        pass

    await q.submit("title", title, lane=LANE_SHORT)
    await q.join()
    assert idle == ["short"]  # a title alone doesn't unload the ingestion models

    await q.submit("ingest", ingest, lane=LANE_LONG)
    await started.wait()
    await q.submit("title", title, lane=LANE_SHORT)
    await q.join(LANE_SHORT)
    assert idle == ["short", "short"]  # the long lane is busy: its hook waits
    release.set()
    await q.join()
    assert idle == ["short", "short", "long"]
    await q.close()


async def test_join_waits_for_one_lane_or_all(make_queue):
    q = make_queue()
    release = asyncio.Event()

    async def slow() -> None:
        await release.wait()

    await q.submit("ingest", slow, lane=LANE_LONG)
    await asyncio.wait_for(q.join(LANE_SHORT), 1)  # an empty or never-started lane returns at once
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(q.join(), 0.05)
    release.set()
    await asyncio.wait_for(q.join(), 1)
    await q.close()


async def test_an_unknown_lane_is_refused(make_queue):
    q = make_queue()

    async def job() -> None:
        pass

    with pytest.raises(ValueError, match=r"unknown job lane 'bulk' \(available: long, short\)"):
        await q.submit("x", job, lane="bulk")
    with pytest.raises(ValueError, match="unknown job lane"):
        q.on_idle(job, lane="bulk")
    with pytest.raises(ValueError, match="unknown job lane"):
        await q.join("bulk")
    await q.close()


async def test_a_failing_short_job_leaves_both_lanes_working(make_queue, caplog):
    q = make_queue()
    ran: list[str] = []

    async def boom() -> None:
        raise RuntimeError("kaboom")

    async def ok(name: str) -> None:
        ran.append(name)

    await q.submit("title", boom, lane=LANE_SHORT)
    await q.submit("title ok", lambda: ok("title"), lane=LANE_SHORT)
    await q.submit("ingest ok", lambda: ok("ingest"), lane=LANE_LONG)
    await q.join()
    assert sorted(ran) == ["ingest", "title"]
    assert any("job title failed" in r.getMessage() for r in caplog.records)
    await q.close()


async def test_health_reports_every_lane(make_queue):
    q = make_queue(concurrency=3)
    detail = (await q.health()).detail
    assert "long: 0 running, 0 queued (workers 3)" in detail
    assert f"short: 0 running, 0 queued (workers {q.SHORT_LANE_WORKERS})" in detail


async def test_close_drops_queued_jobs_and_cancels_running_ones_in_both_lanes(make_queue, monkeypatch):
    q = make_queue()
    monkeypatch.setattr(q, "SHUTDOWN_GRACE_S", 0.05)
    started = asyncio.Event()
    running = 0
    cancelled: list[str] = []
    ran: list[str] = []

    def hang(name: str):
        async def run() -> None:
            nonlocal running
            running += 1
            if running == 1 + q.SHORT_LANE_WORKERS:
                started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        return run

    async def never() -> None:
        ran.append("never")

    await q.submit("ingest", hang("ingest"), lane=LANE_LONG)
    await q.submit("ingest 2", never, lane=LANE_LONG)  # queued behind the running one
    for i in range(q.SHORT_LANE_WORKERS):
        await q.submit(f"title {i}", hang(f"title {i}"), lane=LANE_SHORT)
    await q.submit("title queued", never, lane=LANE_SHORT)  # every short worker is busy: queued
    await asyncio.wait_for(started.wait(), 1)

    await q.close()
    assert sorted(cancelled) == ["ingest", "title 0", "title 1"] and ran == []
    assert _worker_tasks() == []  # no worker of either lane outlives the queue
    for lane in (LANE_LONG, LANE_SHORT):
        with pytest.raises(RuntimeError, match="shut down"):
            await q.submit("late", never, lane=lane)
    assert "stopped" in (await q.health()).detail


async def test_close_lets_running_jobs_of_both_lanes_finish_within_the_grace(make_queue):
    q = make_queue()
    finished: list[str] = []

    def short(name: str):
        async def run() -> None:
            await asyncio.sleep(0.02)
            finished.append(name)

        return run

    await q.submit("ingest", short("ingest"), lane=LANE_LONG)
    await q.submit("title", short("title"), lane=LANE_SHORT)
    await asyncio.sleep(0)  # let the workers pick them up
    await q.close()
    assert sorted(finished) == ["ingest", "title"]
    assert _worker_tasks() == []


async def test_work_dropped_at_shutdown_is_resubmitted_to_a_new_queue_and_runs(make_queue, monkeypatch):
    """What a restart does: ingestion re-queues the documents left PENDING/PROCESSING (long lane); a chat whose title
    was dropped gets a new title job at its next answer (short lane). A fresh queue runs both."""
    first = make_queue()
    monkeypatch.setattr(first, "SHUTDOWN_GRACE_S", 0.05)
    interrupted = True
    attempts: Counter[str] = Counter()
    completed: list[str] = []
    both_running = asyncio.Event()

    def job(name: str):
        async def run() -> None:
            attempts[name] += 1
            if interrupted:
                if sum(attempts.values()) >= 1 + first.SHORT_LANE_WORKERS:
                    both_running.set()
                await asyncio.sleep(10)
            completed.append(name)

        return run

    plan = [
        ("ingest a", LANE_LONG),
        ("ingest b", LANE_LONG),
        ("title a", LANE_SHORT),
        ("title b", LANE_SHORT),
        ("title c", LANE_SHORT),
    ]
    for name, lane in plan:
        await first.submit(name, job(name), lane=lane)
    await asyncio.wait_for(both_running.wait(), 1)
    await first.close()
    assert completed == []

    interrupted = False
    second = make_queue()
    await second.start()
    for name, lane in plan:
        await second.submit(name, job(name), lane=lane)
    await asyncio.wait_for(second.join(), 1)
    assert sorted(completed) == sorted(name for name, _ in plan)
    await second.close()
    assert _worker_tasks() == []
