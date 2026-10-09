"""Model loads, document ingestion and questions at the same time, with the real stack. A PDF uploaded while the
models preloaded aborted the backend (exit 134), and ingesting while questions ran did too in stress tests; see
app/providers/models.py TorchGate.

    RUN_INTEGRATION=1 uv run --group ml pytest tests/integration/test_concurrency_e2e.py -s

Needs what test_voice_e2e needs (models, smoke documents and speech clips, Qdrant, Ollama). The smoke PDF is uploaded
the moment the app starts, while the models preload: it waits (PENDING) and is ingested once they are loaded. Then a
text question and a spoken question are timed while idle, and again while a long PDF (the smoke PDF 20 times) is
being ingested; the summary compares them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.registry import build_container

from .conftest import METRICS, integration_settings, require_chat_model
from .test_chat_e2e import EN, READY_TIMEOUT_S, ask
from .test_voice_e2e import PRELOAD_TIMEOUT_S, LiveClient, first, frame_of, is_type, latency_row, pcm16k

pytestmark = pytest.mark.integration

LONG_COPIES = 20  # 60 pages, ~30 s of ingestion: longer than both questions together


def upload(api: TestClient, project: str, path: Path) -> str:
    r = api.post(f"/api/projects/{project}/documents", files={"file": (path.name, path.read_bytes())})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def document(api: TestClient, doc_id: str) -> dict[str, Any]:
    return api.get(f"/api/documents/{doc_id}").json()


def wait_ready(api: TestClient, doc_id: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    while (doc := document(api, doc_id))["status"] not in ("READY", "FAILED"):
        assert time.perf_counter() - t0 < READY_TIMEOUT_S, doc
        time.sleep(0.2)
    assert doc["status"] == "READY", doc["error"]
    return doc


def long_pdf(source: Path, dest: Path, copies: int) -> Path:
    import pypdfium2 as pdfium

    src, out = pdfium.PdfDocument(source), pdfium.PdfDocument.new()
    for _ in range(copies):
        out.import_pages(src)
    out.save(dest)
    return dest


def voice_question(api: TestClient, project: str, audio: np.ndarray) -> dict[str, Any]:
    """A spoken question in a new chat, answered in speech; ms from the end of speech to each stage (latency_row)."""
    chat = api.post(f"/api/projects/{project}/chats", json={}).json()["id"]
    with api.websocket_connect(f"/ws/chats/{chat}/voice") as ws:
        c = LiveClient(ws)
        c.send("start", language=None)
        c.wait(is_type("ready"))
        end = c.stream(audio, then_silence_s=2.0, stop_silence_on=frame_of(1))
        agent = c.wait(is_type("agent_message")).item["message"]  # after the turn's last audio frame
        user = first(c.drain(), is_type("user_message")).item["message"]
        assert "revenue" in user["text"].lower() and agent["citations"], (user, agent)
        c.send("playback_done", turn_id=1)
        c.send("end")
    return latency_row(c.log, end, 1, user, agent)


def test_upload_during_preload_then_questions_during_ingestion(models_root, smoke_docs, smoke_audio, tmp_path_factory):
    root = tmp_path_factory.mktemp("concurrency")
    settings = integration_settings(models_root, root)
    require_chat_model(settings)
    container = build_container(settings)
    pdf = smoke_docs / "annual_report.pdf"
    spoken = pcm16k(smoke_audio / "say_en_0.wav")  # a question about revenue
    with TestClient(create_app(settings, container)) as api:  # the models start preloading in the background
        try:
            # ---- an upload while the models preload: ingested after them, never alongside
            project = api.post("/api/projects", json={"name": "Concurrency"}).json()["id"]
            t0 = time.perf_counter()
            doc_id = upload(api, project, pdf)
            seen: set[tuple[str, str]] = set()
            preloaded = None
            while (doc := document(api, doc_id))["status"] not in ("READY", "FAILED"):
                preload = api.get("/health").json()["preload"]["state"]
                seen.add((preload, doc["status"]))
                if preload != "loading" and preloaded is None:
                    preloaded = time.perf_counter()
                assert time.perf_counter() - t0 < PRELOAD_TIMEOUT_S + READY_TIMEOUT_S, (preload, doc)
                time.sleep(0.2)
            ready = time.perf_counter()
            assert doc["status"] == "READY", doc["error"]
            assert ("loading", "PENDING") in seen and ("loading", "PROCESSING") not in seen, seen
            report = api.get("/health").json()["preload"]
            assert report["state"] == "ready", report
            preloaded = preloaded or ready
            METRICS["concurrency: smoke PDF uploaded at startup"] = (
                f"preload over after {preloaded - t0:.1f}s, READY {ready - preloaded:.1f}s later "
                f"({doc['page_count']} pages, {doc['chunk_count']} chunks)"
            )

            # ---- questions while idle
            chat = api.post(f"/api/projects/{project}/chats", json={}).json()["id"]
            ask(api, chat, EN, "concurrency: warm-up")
            _, idle = ask(api, chat, EN, "concurrency: idle")
            assert "18.2%" in idle["text"], idle
            METRICS["concurrency: voice, idle (ms after end of speech)"] = voice_question(api, project, spoken)

            # ---- the same questions while a long PDF is being ingested
            long = long_pdf(pdf, root / "annual_report_x20.pdf", LONG_COPIES)
            t1 = time.perf_counter()
            long_id = upload(api, project, long)
            while document(api, long_id)["status"] == "PENDING":
                time.sleep(0.05)
            busy_voice = voice_question(api, project, spoken)
            _, busy = ask(api, chat, EN, "concurrency: during ingestion")
            still = document(api, long_id)["status"]
            assert still == "PROCESSING", f"the long PDF was {still} before the questions were answered"
            assert "18.2%" in busy["text"], busy
            METRICS["concurrency: voice, during ingestion (ms after end of speech)"] = busy_voice
            done = wait_ready(api, long_id)
            METRICS["concurrency: long PDF upload → READY (questions meanwhile)"] = (
                f"{time.perf_counter() - t1:.1f}s, {done['page_count']} pages, {done['chunk_count']} chunks"
            )
        finally:
            api.portal.call(container["vector_store"].drop_collection)  # type: ignore[attr-defined]
