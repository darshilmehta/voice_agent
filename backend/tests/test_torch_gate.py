"""PyTorch work is serialized process-wide (app/providers/models.py TorchGate): a crash found in the real end-to-end
run (a PDF uploaded while models preloaded aborted the backend) and in stress tests (ingestion while questions run).

Without the ml group: the gate itself, and the real embedder and reranker classes with fake models that record every
overlap the gate must prevent.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from app.providers import models
from app.providers.models import TorchGate
from app.providers.registry import build_container

from .conftest import make_model_assets


class Overlaps:
    """Records what runs at once; a load next to anything else, or two MPS users, is a violation."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = {"load": 0, "mps": 0, "cpu": 0}
        self.violations: list[dict[str, int]] = []
        self.peak = {"load": 0, "mps": 0, "cpu": 0}

    @contextlib.contextmanager
    def work(self, kind: str, seconds: float = 0.01) -> Iterator[None]:
        with self.lock:
            self.active[kind] += 1
            now = dict(self.active)
            self.peak = {k: max(self.peak[k], now[k]) for k in now}
            if now["mps"] > 1 or (now["load"] and (now["load"] > 1 or now["mps"] or now["cpu"])):
                self.violations.append(now)
        try:
            time.sleep(seconds)
            yield
        finally:
            with self.lock:
                self.active[kind] -= 1


def hammer(*jobs, seconds: float = 0.6) -> None:
    """Run each job in a loop in its own thread for ``seconds``."""
    stop = time.monotonic() + seconds
    errors: list[BaseException] = []

    def loop(job):
        try:
            while time.monotonic() < stop:
                job()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=loop, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


# ------------------------------------------------------------------ the gate


def test_loads_exclude_everything_and_mps_users_take_turns_while_cpu_users_share():
    gate, seen = TorchGate(), Overlaps()

    def load():
        with gate.load(), seen.work("load"):
            pass

    def mps():
        with gate.use("mps"), seen.work("mps"):
            pass

    def cpu():
        with gate.use("cpu"), seen.work("cpu", 0.03):
            pass

    hammer(load, mps, mps, mps, cpu, cpu, cpu)
    assert seen.violations == []
    assert seen.peak["mps"] == 1 and seen.peak["cpu"] >= 2  # CPU work ran alongside other work


def test_cpu_work_runs_alongside_mps_work():
    gate, seen = TorchGate(), Overlaps()
    started = threading.Event()

    def long_mps():
        with gate.use("mps"), seen.work("mps", 0.3):
            started.set()

    t = threading.Thread(target=long_mps)
    t.start()
    started.wait(1)
    t0 = time.monotonic()
    with gate.use("cpu"):  # doesn't wait for the MPS user
        waited = time.monotonic() - t0
    t.join()
    assert waited < 0.1


def test_a_pending_load_goes_before_new_users():
    gate = TorchGate()
    order: list[str] = []
    first_user_in = threading.Event()
    release_first = threading.Event()

    def first_user():
        with gate.use("cpu"):
            first_user_in.set()
            release_first.wait(2)
            order.append("first user done")

    def load():
        with gate.load():
            order.append("load")

    def late_user():
        with gate.use("cpu"):
            order.append("late user")

    threads = [threading.Thread(target=first_user)]
    threads[0].start()
    first_user_in.wait(1)
    threads.append(threading.Thread(target=load))
    threads[-1].start()
    time.sleep(0.05)  # the load is now waiting for the first user
    threads.append(threading.Thread(target=late_user))
    threads[-1].start()
    time.sleep(0.05)
    release_first.set()
    for t in threads:
        t.join(2)
    assert order == ["first user done", "load", "late user"]  # the late user didn't overtake the waiting load


# ------------------------------------------------------------------ the providers use it


class FakeBgeM3:
    def __init__(self, seen: Overlaps) -> None:
        self.seen = seen
        self.batches: list[int] = []

    def encode(self, texts, **kw):
        with self.seen.work("mps"):
            self.batches.append(len(texts))
            return {"dense_vecs": np.zeros((len(texts), 4)), "lexical_weights": [{1: 0.5} for _ in texts]}


class FakeCrossEncoder:
    def __init__(self, seen: Overlaps) -> None:
        self.seen = seen

    def predict(self, pairs, **kw):
        with self.seen.work("mps"):
            return [0.5] * len(pairs)


@pytest.fixture
def gated(load_local, tmp_path, monkeypatch):
    """The real embedder and reranker (MPS, as configured) on fake models whose loads and inferences are recorded; a
    fresh gate."""
    make_model_assets(tmp_path)
    container = build_container(load_local())
    monkeypatch.setattr(models, "TORCH", TorchGate())
    seen = Overlaps()
    embedder, reranker = container["embeddings"], container["reranker"]

    def slow_load(fake):
        def load(self, snapshot):
            with seen.work("load", 0.05):
                return fake

        return load

    monkeypatch.setattr(type(embedder), "_load", slow_load(FakeBgeM3(seen)))
    monkeypatch.setattr(type(reranker), "_load", slow_load(FakeCrossEncoder(seen)))
    return embedder, reranker, seen


def test_questions_during_ingestion_and_model_loads_never_overlap_on_the_gpu(gated):
    embedder, reranker, seen = gated

    def ingest():  # a document's chunks
        asyncio.run(embedder.embed([f"chunk {i}" for i in range(40)]))

    def ask():  # a question: query embedding, then reranking
        asyncio.run(embedder.embed(["What was the EBITDA margin?"]))
        asyncio.run(reranker.score("What was the EBITDA margin?", ["a", "b", "c"]))

    def reload():  # e.g. the startup preload loading models while the others run
        embedder.unload()
        reranker.unload()
        asyncio.run(embedder.preload())
        asyncio.run(reranker.preload())

    hammer(ingest, ask, ask, reload, seconds=1.0)
    assert seen.violations == []
    assert seen.peak["mps"] == 1 and seen.peak["load"] == 1


def test_a_document_is_embedded_one_batch_at_a_time(gated):
    embedder, _, _ = gated
    vectors = asyncio.run(embedder.embed([f"chunk {i}" for i in range(40)]))  # batch_size 16 (config)
    assert len(vectors) == 40
    assert embedder._model.batches == [16, 16, 8]  # the GPU is free between batches for a question's query
