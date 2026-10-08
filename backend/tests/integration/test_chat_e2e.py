"""Phase 1 acceptance, end to end over HTTP with the real stack: upload → Docling → BGE-M3 → Qdrant → reranker → Ollama.

    RUN_INTEGRATION=1 uv run --group ml pytest tests/integration/test_chat_e2e.py -s

Needs, besides the models and Qdrant (see conftest): Ollama at 127.0.0.1:11434 with qwen3:4b-instruct pulled.
Checks: fact questions are answered and cited (English and Hindi), out-of-document questions abstain, two documents
are told apart, and documents never leak across projects. Timings and answers print in the summary.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.registry import build_container
from app.services.language import message_language
from app.services.prompts import ABSTENTIONS

from .conftest import METRICS, integration_settings

pytestmark = pytest.mark.integration

EN = "What was the EBITDA margin in FY24?"
HI = "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?"
OUT_OF_DOCUMENT = "What is the CEO's salary?"
SUPPLIER = "Which product depends on a single supplier?"
READY_TIMEOUT_S = 600


@pytest.fixture(scope="module")
def api(models_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = integration_settings(models_root, tmp_path_factory.mktemp("e2e"))
    try:
        r = httpx.get(f"{settings.llm.base_url}/api/tags", timeout=2)
        pulled = {m["name"] for m in r.json().get("models", [])}
    except httpx.HTTPError as e:
        pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
    if settings.llm.chat_model not in pulled:
        pytest.skip(f"{settings.llm.chat_model} not pulled in Ollama")
    container = build_container(settings)
    with TestClient(create_app(settings, container)) as client:
        try:
            yield client
        finally:
            client.portal.call(container["vector_store"].drop_collection)  # type: ignore[attr-defined]


def project(api: TestClient, name: str) -> str:
    return api.post("/api/projects", json={"name": name}).json()["id"]


def upload_ready(api: TestClient, project_id: str, path: Path, label: str) -> dict:
    t0 = time.perf_counter()
    r = api.post(f"/api/projects/{project_id}/documents", files={"file": (path.name, path.read_bytes())})
    assert r.status_code == 202, r.text
    doc_id = r.json()["id"]
    while True:
        doc = api.get(f"/api/documents/{doc_id}").json()
        if doc["status"] in ("READY", "FAILED"):
            break
        assert time.perf_counter() - t0 < READY_TIMEOUT_S, f"{path.name} still {doc['status']}"
        time.sleep(0.5)
    METRICS[f"upload → READY ({label})"] = f"{time.perf_counter() - t0:.1f}s, {doc['page_count']} pages, " + (
        f"{doc['chunk_count']} chunks"
    )
    assert doc["status"] == "READY", doc["error"]
    return doc


def ask(api: TestClient, chat_id: str, text: str, label: str) -> tuple[dict, dict]:
    """(sources event, agent message); records the answer and its timings."""
    t0 = time.perf_counter()
    r = api.post(f"/api/chats/{chat_id}/messages", json={"text": text})
    wall = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    events = []
    for block in r.text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    order = [e for e, _ in events]
    assert order[0] == "user_message" and order[-1] == "agent_message", events[-1]
    sources = next(d for e, d in events if e == "sources")
    agent = events[-1][1]
    lat = agent["latency"]
    cites = ", ".join(f"{c['source_id']} {c['filename']} p{c['page_start']}" for c in agent["citations"])
    METRICS[f"Q {label}"] = (
        f"{agent['text']!r} | cites: {cites or '-'} | abstained={sources['abstained']} "
        f"top={sources['confidence']['top_score']:.3f} | retrieval {lat['retrieval_ms']:.0f} ms "
        f"(rerank {lat['rerank_ms'] or 0:.0f}), first delta {lat['first_delta_ms']:.0f} ms, "
        f"total {lat['total_ms']:.0f} ms (HTTP {wall * 1000:.0f} ms)"
    )
    return sources, agent


def cites_page(agent: dict, document_id: str, page: int) -> bool:
    return any(
        c["document_id"] == document_id and (c["page_start"] or 0) <= page <= (c["page_end"] or c["page_start"] or 0)
        for c in agent["citations"]
    )


@pytest.fixture(scope="module")
def report(api: TestClient, smoke_docs: Path) -> dict:
    p = project(api, "Annual report FY24")
    doc = upload_ready(api, p, smoke_docs / "annual_report.pdf", "annual_report.pdf, cold models")
    assert doc["page_count"] == 3 and doc["chunk_count"] >= 4
    chat = api.post(f"/api/projects/{p}/chats", json={}).json()["id"]
    return {"project": p, "doc": doc, "chat": chat}


def test_english_fact_question_is_answered_and_cites_page_2(api, report):
    for label in ("EN fact (cold: reranker and LLM load)", "EN fact (warm)"):
        sources, agent = ask(api, report["chat"], EN, label)
        assert not sources["abstained"]
        assert "18.2%" in agent["text"]
        assert cites_page(agent, report["doc"]["id"], 2), agent["citations"]
        assert agent["language"] == "en"


def test_hindi_question_gets_a_hindi_answer_citing_page_2(api, report):
    sources, agent = ask(api, report["chat"], HI, "HI fact")
    assert not sources["abstained"]
    assert agent["language"] == "hi" and message_language(agent["text"]) == "hi", agent["text"]
    assert cites_page(agent, report["doc"]["id"], 2), agent["citations"]


def test_out_of_document_question_abstains(api, report):
    sources, agent = ask(api, report["chat"], OUT_OF_DOCUMENT, "out of document")
    assert sources["abstained"] and sources["sources"] == []
    assert agent["text"] == ABSTENTIONS["not_covered"]["en"] and agent["citations"] == []


def test_a_second_project_never_sees_the_first_projects_document(api, report, smoke_docs):
    other = project(api, "Board decks")
    deck = upload_ready(api, other, smoke_docs / "board_deck.pptx", "board_deck.pptx")
    chat = api.post(f"/api/projects/{other}/chats", json={}).json()["id"]
    for question, label in ((SUPPLIER, "project B: report-only fact"), (EN, "project B: fact in both")):
        sources, agent = ask(api, chat, question, label)
        seen = {s["document_id"] for s in sources["sources"]} | {c["document_id"] for c in agent["citations"]}
        assert seen <= {deck["id"]}, seen
        assert report["doc"]["id"] not in agent["text"]
    supplier_sources, supplier_answer = ask(api, chat, SUPPLIER, "project B: report-only fact (again)")
    assert supplier_sources["abstained"] or "SKU-48213" not in supplier_answer["text"]


def test_two_documents_in_one_project_are_told_apart(api, report, smoke_docs):
    deck = upload_ready(api, report["project"], smoke_docs / "board_deck.pptx", "board_deck.pptx into project A")
    chat = api.post(f"/api/projects/{report['project']}/chats", json={}).json()["id"]
    _, agent = ask(api, chat, "When was the investor presentation?", "two docs: deck fact")
    assert "April 2025" in agent["text"]
    assert {c["document_id"] for c in agent["citations"]} == {deck["id"]}, agent["citations"]
    _, agent = ask(api, chat, SUPPLIER, "two docs: report fact")
    assert "SKU-48213" in agent["text"]
    assert cites_page(agent, report["doc"]["id"], 3), agent["citations"]
