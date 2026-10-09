"""Time to the answer's first token on the real model, with cached retrieval (docs/DESIGN.md §9.5).

    RUN_INTEGRATION=1 uv run pytest tests/integration/test_answer_latency.py -s

Needs only Ollama (127.0.0.1:11434, qwen3:4b-instruct pulled): no embedder, reranker, speech models, Docling or
Qdrant. Retrieval is "cached": real passages of the eval corpus (``latency_passages.json``: the Valmora annual report
and the Zephyra investor deck as the ingestion pipeline chunked them). For each turn the passages on the pages the eval
set expects (``evals/retrieval/questions.jsonl``) rank first with a confident score, then keyword matches as
distractors, after the measured idle-machine search and rerank times (~0.15 s + ~0.25 s). Everything else is the real
turn pipeline: ``ChatTurnService`` with the router model, speculative retrieval, the answer prompt, Ollama's prompt
cache, the automatic title after the first answer and the memory summary when it is due.

Three voice chats (``modality="voice"``, ``length="short"``) of five different turns each: document questions (some
routed, some not), a follow-up, a Hindi or Hinglish question, an unanswerable one and a general one. Reported: the
LLM's first token, both the wall clock (which includes waiting behind other clients' requests on a shared Ollama) and
the model's own time (reading the uncached part of the prompt, then one token: what an idle machine pays); the answer
prompt's size; the router's time; whether each answer carries the figure the eval set expects. A model reload during
the run (``load_duration`` above 0.5 s) is reported too.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import statistics
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.providers.base import ProviderContext
from app.providers.ingestion import Chunk
from app.providers.llm import OllamaLLM
from app.providers.registry import Container
from app.providers.retrieval import DenseSparse, Embedder, IndexedChunk, Reranker, RetrievalFilters, SearchHit
from app.providers.storage import SqliteDB
from app.services.chat_turns import ChatTurnService, DeltaEvent, SourcesEvent, wait_for_background
from app.services.chats import ChatService
from app.services.messages import MessageService
from app.services.preload import ModelPreloader
from app.services.projects import ProjectService
from app.services.retrieval import RetrievalService
from app.services.titles import TitleService
from app.settings import PROJECT_ROOT, load_settings

from ..conftest import add_document
from ..fakes import FakeStore, vector_for
from .conftest import LOCAL_CONFIG, METRICS

pytestmark = pytest.mark.integration

PASSAGES = json.loads((Path(__file__).parent / "latency_passages.json").read_text(encoding="utf-8"))
_LINES = (PROJECT_ROOT / "evals/retrieval/questions.jsonl").read_text(encoding="utf-8").splitlines()
QUESTIONS = {q["id"]: q for q in map(json.loads, _LINES)}
REPORT, DECK = "eval-valmora-annual-report-fy24", "eval-zephyra-investor-deck-q4fy24"
FILENAMES = {REPORT: "valmora_annual_report_fy24.pdf", DECK: "zephyra_investor_deck_q4fy24.pptx"}
SEARCH_S, RERANK_S = 0.15, 0.25  # idle-machine embed + search and rerank of 8 candidates (§9.5)


@dataclass(frozen=True)
class T:
    kind: str
    text: str
    pages: tuple[int, ...] = ()  # where the answer is (empty: nowhere)
    expect: tuple[str, ...] = ()  # each item: alternatives separated by "|", one must be in the answer


def q(kind: str, qid: str) -> T:
    item = QUESTIONS[qid]
    pages = tuple(item["expected"][0]["pages"]) if item["expected"] else ()
    return T(kind, item["question"], pages, tuple(item["answer_contains"]))


CHATS = [
    [
        q("document", "q001"),
        T("follow-up", "And what is the record date for it?", (28,), ("16 August 2024",)),
        q("hindi", "q127"),
        q("routed fact", "q087"),
        q("unanswerable", "q157"),
    ],
    [
        q("document", "q006"),
        T("follow-up", "And how many customers did it serve?", (6,), ("2,100",)),
        q("hindi", "q145"),
        T("routed fact", "How many employees did the company have at the end of the year?", (12,), ("9,842",)),
        q("document", "q012"),
    ],
    [
        q("document", "q007"),
        q("routed fact", "q018"),
        q("hindi", "q136"),
        q("document", "q011"),
        T("general", "What is EBITDA?"),
    ],
]
_WORD = re.compile(r"[\wऀ-ॿ%.,₹-]+")


def _words(text: str) -> set[str]:
    return {w.strip(".,?").casefold() for w in _WORD.findall(text) if len(w.strip(".,?")) > 2}


class CachedIndex(Embedder, Reranker):
    """Embedder, vector store and reranker over the fixture passages for the current turn: the passages on its pages
    first (reranker score ~0.9), then keyword matches (at most 0.35), after the measured search and rerank times."""

    name = "cached"
    dim = 8

    def __init__(self) -> None:  # no config needed
        self.queries: dict[int, str] = {}
        self.pages: tuple[int, ...] = ()

    async def embed(self, texts: Sequence[str]) -> list[DenseSparse]:
        await asyncio.sleep(SEARCH_S / 2)
        vectors = [vector_for(t) for t in texts]
        for text, vector in zip(texts, vectors, strict=True):
            self.queries[id(vector)] = text
        return vectors

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        await asyncio.sleep(RERANK_S)
        words = _words(query)
        pages = {c["text"]: c["page_start"] for c in PASSAGES if c["document_id"] == REPORT}
        page_of = [next((page for text, page in pages.items() if p.endswith(text)), None) for p in passages]
        return [
            0.9 if page in self.pages else min(0.35, 0.5 * len(words & _words(p)) / max(len(words), 1))
            for p, page in zip(passages, page_of, strict=True)
        ]


class CachedStore(FakeStore):
    def __init__(self, index: CachedIndex) -> None:
        super().__init__(search_points=True)
        self.index = index

    async def hybrid_search(
        self, query: DenseSparse, filters: RetrievalFilters, *, limit: int | None = None
    ) -> list[SearchHit]:
        await asyncio.sleep(SEARCH_S / 2)
        words = _words(self.index.queries.get(id(query), ""))
        points = [
            p
            for p in self.points.values()
            if p.chunk.project_id == filters.project_id
            and (filters.document_ids is None or p.chunk.document_id in filters.document_ids)
        ]
        ranked = sorted(
            points,
            key=lambda p: (p.chunk.page_start not in self.index.pages, -len(words & _words(p.chunk.text))),
        )
        return [SearchHit(p.chunk, 1.0 / (i + 1), 0.5) for i, p in enumerate(ranked[: limit or 8])]


class OllamaStats(httpx.AsyncBaseTransport):
    """Passes requests to Ollama and keeps the statistics of each finished generation (the last line of a stream or
    the body of a non-streamed reply): prompt tokens, prompt reading time, load time."""

    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.done: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        if not request.url.path.startswith("/api/"):
            return response
        kind = "json" if json.loads(request.content or b"{}").get("format") else "text"
        stats, stream = self.done, response.stream

        async def tee() -> AsyncIterator[bytes]:
            """Every chunk passes through; the line saying ``"done":true`` is kept before it is passed on (the client
            stops reading there)."""
            buffer = b""
            async for chunk in stream:  # type: ignore[union-attr]
                buffer += chunk
                if b'"done":true' in buffer:
                    for line in buffer.split(b"\n"):
                        if b'"done":true' in line:
                            with contextlib.suppress(ValueError):
                                stats.append({"kind": kind, "path": request.url.path, **json.loads(line)})
                    buffer = b""
                else:
                    buffer = buffer[buffer.rfind(b"\n") + 1 :]
                yield chunk

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                async for chunk in tee():
                    yield chunk

            async def aclose(self) -> None:
                await stream.aclose()  # type: ignore[union-attr]

        return httpx.Response(
            response.status_code, headers=response.headers, stream=Stream(), extensions=response.extensions
        )

    async def aclose(self) -> None:
        await self.inner.aclose()


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _spent(c: dict[str, Any]) -> float:
    """A whole model call's own time (ms): loading, reading the prompt, generating."""
    return (c.get("load_duration", 0) + c.get("prompt_eval_duration", 0) + c.get("eval_duration", 0)) / 1e6


def _model_first(a: dict[str, Any]) -> float:
    """The model's own time to the first token (ms): loading, reading the uncached prompt, one token."""
    return (a.get("load_duration", 0) + a.get("prompt_eval_duration", 0)) / 1e6 + a.get("eval_duration", 0) / 1e6 / max(
        a.get("eval_count", 1), 1
    )


async def test_time_to_the_first_token_of_voice_answers(tmp_path):
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    stats = OllamaStats()
    async with httpx.AsyncClient(transport=stats) as http:
        try:
            tags = (await http.get(f"{settings.llm.base_url}/api/tags", timeout=2)).json()
        except httpx.HTTPError as e:
            pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
        if settings.llm.chat_model not in {m["name"] for m in tags.get("models", [])}:
            pytest.skip(f"{settings.llm.chat_model} not pulled")
        ctx = ProviderContext(settings=settings, http=http)
        llm = OllamaLLM(settings.llm, ctx)
        db = SqliteDB(settings.metadata_db, ctx)
        await db.start()
        rows: list[dict[str, Any]] = []
        router_calls: list[dict[str, Any]] = []
        try:
            await ModelPreloader(Container(settings, http, {"llm": llm}))._run()  # as at the app's startup
            preload_calls = len(stats.done)
            index = CachedIndex()
            store = CachedStore(index)
            retrieval = RetrievalService(index, index, store, settings.retrieval)
            service = ChatTurnService(db, retrieval=retrieval, llm=llm, settings=settings)
            titles = TitleService(db, llm=llm, settings=settings)
            project = await ProjectService(db).create("Latency")
            run = f"run {time.time_ns() % 10**9}"
            ids = {}
            for doc in (REPORT, DECK):
                ids[doc] = await add_document(db, project.id, FILENAMES[doc], status="READY", page_count=30)
            for i, p in enumerate(PASSAGES):
                chunk = Chunk(
                    chunk_id=f"{ids[p['document_id']]}:v1:{p['chunk_index']:04d}",
                    project_id=project.id,
                    document_id=ids[p["document_id"]],
                    version=1,
                    chunk_index=p["chunk_index"],
                    chunking_version="v1",
                    page_start=p["page_start"],
                    page_end=p["page_end"],
                    # A run tag in each passage's heading: earlier runs' prompts in Ollama's cache don't make the
                    # evidence look cheap (the system prompt and the history are cached legitimately).
                    heading_path=[*p["heading_path"], run],
                    content_type=p["content_type"],
                    language="en",
                    text=p["text"],
                    token_count=len(p["text"]) // 4,
                )
                await store.upsert([IndexedChunk(chunk, vector_for(f"{i}"))])

            for turns in CHATS:
                chat = await ChatService(db).create(project.id, document_scope=[ids[REPORT]])
                for n, t in enumerate(turns):
                    await wait_for_background()  # the previous turn's memory summary, as between spoken turns
                    index.pages = t.pages
                    before = len(stats.done)
                    turn = await service.begin(chat.id, t.text, modality="voice", length="short")
                    t0 = time.perf_counter()
                    marks: dict[str, float] = {}
                    async for event in service.run(turn):
                        if isinstance(event, SourcesEvent):
                            marks["sources"] = time.perf_counter()
                        elif isinstance(event, DeltaEvent):
                            marks.setdefault("delta", time.perf_counter())
                    agent = (await MessageService(db).list(chat.id)).items[-1]
                    texts = [s for s in stats.done[before:] if s["kind"] == "text"]
                    router_calls += [s for s in stats.done[before:] if s["kind"] == "json"]
                    a = texts[-1] if texts else {}
                    grounded = all(any(alt in agent.text for alt in e.split("|")) for e in t.expect)
                    rows.append(
                        {
                            "kind": t.kind,
                            "text": t.text,
                            "answer": agent.text,
                            "mode": agent.route["answer"],
                            "abstained": agent.route["abstained"],
                            "language": agent.language,
                            "ok": grounded if t.expect else None,
                            "first_token": (marks["delta"] - marks["sources"]) * 1000 if a else None,
                            "first_delta": (marks["delta"] - t0) * 1000 if "delta" in marks else None,
                            "router": (agent.latency or {}).get("router_ms"),
                            "router_source": ((agent.route or {}).get("router") or {}).get("source"),
                            "prompt": a.get("prompt_eval_count"),
                            "read_ms": a.get("prompt_eval_duration", 0) / 1e6,
                            # a retried answer (B5) pays for its first attempt too
                            "model_first": _model_first(a) + sum(_spent(c) for c in texts[:-1]) if a else None,
                            "calls": len(texts),
                            "retry": (agent.route or {}).get("language_retry"),
                        }
                    )
                    if n == 0:
                        await titles.generate(chat.id)  # the automatic title, right after the first answer
        finally:
            await wait_for_background()
            await db.close()

    reloads = [s for s in stats.done[preload_calls:] if s.get("load_duration", 0) > 5e8]
    model = [r for r in rows if r["model_first"] is not None]
    walls = [r["first_token"] for r in model]
    firsts = [r["model_first"] for r in model]
    for kind in dict.fromkeys(r["kind"] for r in rows):
        rs = [r for r in model if r["kind"] == kind]
        if not rs:
            continue
        METRICS[f"LLM first token, {kind} (p50 model / wall)"] = (
            f"{statistics.median(r['model_first'] for r in rs):.0f} / "
            f"{statistics.median(r['first_token'] for r in rs):.0f} ms; "
            f"prompt {statistics.median(r['prompt'] for r in rs):.0f} tokens"
        )
    METRICS["LLM first token, model's own time, all answers (p50 / p95 / max)"] = (
        f"{statistics.median(firsts):.0f} / {_pct(firsts, 0.95):.0f} / {max(firsts):.0f} ms over {len(firsts)} answers"
    )
    METRICS["LLM first token, wall clock, all answers (p50 / p95 / max)"] = (
        f"{statistics.median(walls):.0f} / {_pct(walls, 0.95):.0f} / {max(walls):.0f} ms"
    )
    METRICS["answer prompt tokens (p50 / max), uncached part read (p50 ms)"] = (
        f"{statistics.median(r['prompt'] for r in model):.0f} / {max(r['prompt'] for r in model)}, "
        f"{statistics.median(r['read_ms'] for r in model):.0f}"
    )
    routed = [r for r in rows if r["router"] is not None]
    METRICS["router time (p50 / max) and sources"] = (
        f"{statistics.median(r['router'] for r in routed):.0f} / {max(r['router'] for r in routed):.0f} ms; "
        f"{[r['router_source'] for r in rows]}"
    )
    if router_calls:
        METRICS["router model calls: prompt tokens, reading, generation (p50)"] = (
            f"{statistics.median(s['prompt_eval_count'] for s in router_calls):.0f} tokens, "
            f"{statistics.median(s['prompt_eval_duration'] / 1e6 for s in router_calls):.0f} ms, "
            f"{statistics.median(s['eval_duration'] / 1e6 for s in router_calls):.0f} ms ({len(router_calls)} calls)"
        )
    checked = [r for r in rows if r["ok"] is not None]
    METRICS["answers carrying the expected figure"] = f"{sum(r['ok'] for r in checked)}/{len(checked)}"
    METRICS["model reloads during the run (load_duration > 0.5 s)"] = len(reloads)
    METRICS["turns"] = "\n  " + "\n  ".join(
        f"{r['kind']}: {r['text'][:60]!r} → {r['mode']}{' (abstained)' if r['abstained'] else ''} [{r['language']}] "
        f"first token {r['model_first'] or 0:.0f} ms ({r['prompt'] or 0} tokens){' RETRIED' if r['retry'] else ''} "
        f"{'' if r['ok'] is None else 'OK' if r['ok'] else 'MISSING'} {r['answer'][:110]!r}"
        for r in rows
    )
