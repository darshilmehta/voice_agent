"""Phase 8 on the real stack (docs/DESIGN.md §3.7): live web search through SearXNG and answers from
qwen3:4b-instruct, on the text path (``ChatTurnService``, the pipeline the SSE endpoint serializes).

    docker compose -f infra/docker-compose.yml --profile websearch up -d searxng
    RUN_INTEGRATION=1 uv run pytest tests/integration/test_web_search_e2e.py -s

Needs only SearXNG (WEB_SEARCH_URL, default http://127.0.0.1:8888) and Ollama with qwen3:4b-instruct. Document
retrieval is faked (no embedder, reranker or Qdrant: a passage the documents "contain"), so the timings are the
router, the search and the answer. One English question that is half about the report and half live, one Hindi live
question. Public engines rate-limit and block automated traffic (CAPTCHAs), so a search may come back empty: the turn
must then answer from the documents, saying live data couldn't be fetched.

Reported per question: router, tool start → first web result, → sources, → first answer delta, → first speakable
chunk (the voice loop's first TTS input), whole turn; and the voice estimate "first words after the filler" =
first speakable chunk + TTS first audio - (filler start + filler length), with Kokoro's measured CPU first-audio time
(about 500 ms for a first chunk of ≤ 5 words, §9.5 and voice/speech_text.py) and the filler's length at Kokoro's
speaking rate (FILLER_MS). Both are estimates: no TTS model is loaded here.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.registry import build_container
from app.providers.retrieval import IndexedChunk
from app.services.chat_turns import (
    AgentMessageEvent,
    ChatTurnService,
    DeltaEvent,
    SourcesEvent,
    ToolEvent,
    wait_for_background,
)
from app.services.chats import ChatService
from app.services.projects import ProjectService
from app.services.prompts import LIVE_NOTICES
from app.services.voice.speech_text import SpeechChunker
from app.settings import PROJECT_ROOT, load_settings

from ..conftest import add_document
from ..fakes import FakeEmbedder, FakeReranker, FakeStore, keyword_scorer, make_chunk, vector_for
from .conftest import METRICS

pytestmark = pytest.mark.integration

TTS_FIRST_AUDIO_MS = 500  # Kokoro on CPU, a first chunk of ≤ 5 words (DESIGN §9.5: 627 ms for ≤ 8)
FILLER_MS = 1300  # "Let me look that up." at Kokoro's ~4 words/s with its pauses (estimate; not synthesized here)
PASSAGE = "Infosys revenue grew 34% in FY24, led by the enterprise segment."
QUESTIONS = {
    "EN doc + live": "What does the report say about Infosys revenue growth, and how is the Infosys stock doing today?",
    "HI live": "आज डॉलर का रेट क्या है?",
}


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, ChatTurnService, str]]:
    root = tmp_path_factory.mktemp("web")
    raw = json.loads((PROJECT_ROOT / "config/local.config.json").read_text())
    raw["strict_offline_exceptions"] = ["web_search"]
    (root / "config.json").write_text(json.dumps(raw))
    url = os.environ.get("WEB_SEARCH_URL", "http://127.0.0.1:8888")
    settings = load_settings(
        root / "config.json",
        {**os.environ, "APP_ROOT_DIR": str(root), "TOOLS__WEB_SEARCH__ENABLED": "true", "TOOLS__WEB_SEARCH__URL": url},
    )
    try:
        httpx.get(f"{url}/healthz", timeout=2).raise_for_status()
    except httpx.HTTPError as e:
        pytest.skip(f"SearXNG not reachable at {url} (docker compose --profile websearch up -d searxng): {e}")
    try:
        tags = httpx.get(f"{settings.llm.base_url}/api/tags", timeout=2).json()
    except httpx.HTTPError as e:
        pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
    if settings.llm.chat_model not in {m["name"] for m in tags.get("models", [])}:
        pytest.skip(f"{settings.llm.chat_model} not pulled")
    container = build_container(settings)
    store = FakeStore(search_points=True)
    container.providers.update(
        embeddings=FakeEmbedder(), reranker=FakeReranker(keyword_scorer), vector_store=store
    )  # no ML models: only Ollama and SearXNG are real
    with TestClient(create_app(settings, container, preload_models=False)) as client:

        async def setup() -> tuple[ChatTurnService, str]:
            db = container["metadata_db"]
            project = await ProjectService(db).create("Infosys annual report")
            doc = await add_document(db, project.id, filename="annual_report.pdf", status="READY", page_count=1)
            chunk = make_chunk(0, project_id=project.id, document_id=doc, text=PASSAGE, page_start=4, page_end=4)
            await store.upsert([IndexedChunk(chunk, vector_for(PASSAGE))])
            await container["llm"].preload()  # loaded with the answers' context, as the app does at startup
            return ChatTurnService.from_container(container), project.id

        service, project_id = client.portal.call(setup)  # type: ignore[union-attr]
        yield client, service, project_id


async def run_turn(service: ChatTurnService, project_id: str, text: str) -> dict[str, Any]:
    """The question as the first message of a new chat."""
    chat = await ChatService(service.chats.db).create(project_id)
    turn = await service.begin(chat.id, text)
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    chunker = SpeechChunker()
    events: list[Any] = []

    def mark(name: str) -> None:
        marks.setdefault(name, (time.perf_counter() - t0) * 1000)

    async for event in service.run(turn):
        events.append(event)
        if isinstance(event, ToolEvent):
            mark(f"tool_{event.phase}")
        elif isinstance(event, SourcesEvent):
            mark("sources")
        elif isinstance(event, DeltaEvent):
            mark("first_delta")
            if chunker.feed(event.text):
                mark("first_chunk")
        elif isinstance(event, AgentMessageEvent):
            mark("agent_message")
    await wait_for_background()
    return {"events": events, "marks": marks, "agent": events[-1].message}


def report(label: str, result: dict[str, Any]) -> None:
    m, agent = result["marks"], result["agent"]
    lat, route = agent.latency, agent.route

    def at(name: str) -> str:
        return f"{m[name]:.0f}" if name in m else "-"

    start = m.get("tool_start")
    chunk = m.get("first_chunk") or m.get("first_delta")
    gap = None
    if start is not None and chunk is not None:
        gap = chunk + TTS_FIRST_AUDIO_MS - (start + FILLER_MS)
    web = route.get("web_search") or {}
    cites = ", ".join(f"{c.source_id}:{c.site or c.filename}" for c in agent.citations) or "-"
    METRICS[f"web {label}"] = (
        f"{agent.text!r}\n    cites: {cites} | search {web.get('query')!r} → {web.get('status')}, "
        f"{web.get('results')} results, {web.get('pages')} page(s) | continuations {route.get('continuations')}\n"
        f"    ms from the turn's start: router {lat.get('router_ms')}, tool start {at('tool_start')}, "
        f"results/terminal {at('tool_results')}/{at('tool_done') if 'tool_done' in m else at('tool_timeout')}, "
        f"sources {at('sources')}, first delta {at('first_delta')}, first speakable chunk {at('first_chunk')}, "
        f"agent_message {at('agent_message')}\n"
        f"    web: first result {lat.get('web_first_result_ms')} ms, search {lat.get('web_search_ms')} ms after start"
        + (
            f"\n    voice estimate: first words {gap:.0f} ms after the filler ends "
            f"(chunk + {TTS_FIRST_AUDIO_MS} ms TTS - {FILLER_MS} ms filler)"
            if gap is not None
            else ""
        )
    )


@pytest.mark.parametrize("label", list(QUESTIONS))
def test_live_question_end_to_end(stack, label):
    client, service, project_id = stack
    result = client.portal.call(run_turn, service, project_id, QUESTIONS[label])  # type: ignore[union-attr]
    report(label, result)
    events, agent = result["events"], result["agent"]
    language = "hi" if label.startswith("HI") else "en"
    phases = [e.phase for e in events if isinstance(e, ToolEvent)]
    if not phases:  # the router failed (timeout): a Hindi turn has no English query, and its words never leave
        assert language == "hi" and agent.route["router"]["source"] == "fallback", agent.route["router"]
        assert agent.route["live_note"] == "failed" and agent.text.startswith(LIVE_NOTICES["failed"]["hi"])
        return
    assert phases[0] == "start" and phases[-1] in ("done", "timeout", "failed"), phases
    query = next(e for e in events if isinstance(e, ToolEvent)).query
    assert query and not any("ऀ" <= ch <= "ॿ" for ch in query)  # only an English query leaves
    assert PASSAGE not in query and "34%" not in query  # never document text
    web = [c for c in agent.citations if c.kind == "web"]
    assert agent.language == language
    if web:
        assert all(c.url and c.url.startswith("http") for c in web)
    else:  # the engines gave nothing in time: said so, answered without live data
        assert agent.text.startswith(LIVE_NOTICES["failed"][language]), agent.text
    if label.startswith("EN"):  # the document passage was offered next to the web results
        sources = next(e for e in events if isinstance(e, SourcesEvent))
        assert sources.sources[0].source_id == "S1"


def test_partial_results_timing(stack):
    """The same EN question again (warm): how much of the turn the search costs."""
    client, service, project_id = stack
    result = client.portal.call(run_turn, service, project_id, QUESTIONS["EN doc + live"])  # type: ignore[union-attr]
    report("EN doc + live (again, warm)", result)
    assert result["marks"].get("first_delta") is not None
