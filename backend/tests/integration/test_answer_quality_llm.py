"""The quality round's answer fixes on the real model (qwen3:4b-instruct), with cached retrieval (opt-in).

    CANVAS_PARSE_CACHE=<folder> RUN_INTEGRATION=1 uv run pytest tests/integration/test_answer_quality_llm.py -s

Needs Ollama and the eval corpus's parsed documents (Docling, or a parse cache, as the canvas tests): no embedder,
reranker, speech models or Qdrant. Retrieval is cached with an explicit ranking per question, so each scenario
reproduces what the final real run saw (for the chart: the quarterly tables ranked 4th and 5th, below the three
passages a voice answer keeps). Everything else is the real turn: router, answer, canvas draft, the answer's checks.

Scenarios, voice (``length="short"``): the chart and the answer (item 1), a live rate with web search off (2), the CIN
(3), the dividend with no period named (4), FY25 revenue (5), a misheard name (10), and the length of six ordinary
answers (7). Reported per answer: its text, words, abstention, citations, the checks that fired.
"""

from __future__ import annotations

import asyncio
import re
import statistics
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from app.db import models as orm
from app.domain.canvas import VisualEvent
from app.providers.base import ProviderContext
from app.providers.ingestion import Chunk
from app.providers.llm import OllamaLLM
from app.providers.retrieval import DenseSparse, Embedder, IndexedChunk, Reranker, RetrievalFilters, SearchHit
from app.providers.storage import SqliteDB
from app.services.canvas.conversation import settled_callbacks
from app.services.canvas.service import CanvasService
from app.services.chat_turns import AgentMessageEvent, ChatTurnService, SourcesEvent, wait_for_background
from app.services.chats import ChatService
from app.services.preload import warm_prompts
from app.services.projects import ProjectService
from app.services.prompts import LIVE_FIGURE_TEXTS
from app.services.retrieval import RetrievalService
from app.services.sources import says_not_covered
from app.settings import load_settings

from ..conftest import add_document
from ..fakes import FakeStore, vector_for
from .canvas_corpus import CorpusDocument, corpus_fixture, stored_tables  # noqa: F401  (the "corpus" fixture)
from .conftest import LOCAL_CONFIG, METRICS
from .test_answer_latency import DECK, FILENAMES, PASSAGES, REPORT

pytestmark = pytest.mark.integration

POLICY = "eval-valmora-travel-policy"
CIN = "L24119GJ1994PLC023871"


def policy_passages(markdown: str) -> list[dict[str, Any]]:
    """The travel policy (a DOCX: no pages) cut at its headings, tables kept whole, as the pipeline chunks it."""
    out: list[dict[str, Any]] = []
    path: list[str] = []
    block: list[str] = []

    def flush() -> None:
        parts, current, table = [], [], None
        for line in "\n".join(block).strip().splitlines():
            row = line.strip().startswith("|")
            if table is not None and row != table and current:
                parts.append("\n".join(current))
                current = []
            table = row
            current.append(line)
        if current:
            parts.append("\n".join(current))
        for part in (p.strip() for p in parts):
            if part:
                kind = "table" if part.startswith("|") else "paragraph"
                out.append(
                    {"document_id": POLICY, "chunk_index": len(out), "page_start": None, "page_end": None,
                     "heading_path": list(path), "content_type": kind, "text": part}
                )  # fmt: skip
        block.clear()

    for line in markdown.splitlines():
        if m := re.match(r"^(#+)\s+(.*)", line):
            flush()
            path[:] = [*path[: len(m.group(1)) - 1], m.group(2)]
        else:
            block.append(line)
    flush()
    return out


def key(doc: str, index: int) -> str:
    return f"{doc}#{index}"


class RankedIndex(Embedder, Reranker):
    """Ranks the passages in ``ranking`` (doc#index → score) first, the rest at 0.01."""

    name = "ranked"
    dim = 8

    def __init__(self) -> None:  # no config needed
        self.ranking: dict[str, float] = {}
        self.by_text: dict[str, str] = {}

    async def embed(self, texts: Sequence[str]) -> list[DenseSparse]:
        await asyncio.sleep(0.07)
        return [vector_for(t) for t in texts]

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        await asyncio.sleep(0.25)
        keys = [next((k for t, k in self.by_text.items() if p.endswith(t)), None) for p in passages]
        return [self.ranking.get(k, 0.01) if k else 0.01 for k in keys]


class RankedStore(FakeStore):
    def __init__(self, index: RankedIndex) -> None:
        super().__init__(search_points=True)
        self.index = index
        self.keys: dict[str, str] = {}

    async def hybrid_search(
        self, query: DenseSparse, filters: RetrievalFilters, *, limit: int | None = None
    ) -> list[SearchHit]:
        await asyncio.sleep(0.07)
        points = [
            p
            for p in self.points.values()
            if p.chunk.project_id == filters.project_id
            and (filters.document_ids is None or p.chunk.document_id in filters.document_ids)
        ]
        ranked = sorted(points, key=lambda p: -self.index.ranking.get(self.keys[p.chunk.chunk_id], 0.0))
        return [SearchHit(p.chunk, 1.0 / (i + 1), 0.5) for i, p in enumerate(ranked[: limit or 8])]


def rk(*items: tuple[str, int, float]) -> dict[str, float]:
    return {key(d, i): s for d, i, s in items}


R, D = REPORT, DECK
QUARTERLY = rk((R, 4, 0.93), (D, 9, 0.9), (R, 44, 0.88), (R, 45, 0.86), (R, 48, 0.85), (R, 47, 0.6))
LENGTH = [
    ("What was Valmora's profit after tax in FY24 and how did it change from FY23?", rk((R, 4, 0.95), (R, 39, 0.8))),
    ("How did each segment do in FY24?", rk((R, 4, 0.7), (R, 6, 0.9), (R, 18, 0.8))),
    ("What are the main risks the company faces?", rk((R, 32, 0.95), (R, 31, 0.9))),
    ("Tell me about Valmora's quarterly performance in FY24", rk((R, 44, 0.95), (R, 45, 0.9), (R, 46, 0.85))),
    ("What did the chairperson say about the year?", rk((R, 6, 0.95), (R, 9, 0.6))),
    ("What is the record date for the FY24 dividend?", rk((R, 74, 0.95), (R, 73, 0.9))),
]


async def test_the_answer_fixes_on_the_real_model(corpus: dict[str, CorpusDocument], tmp_path):
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    async with httpx.AsyncClient() as http:
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
        rows: dict[str, list[dict[str, Any]]] = {}
        try:
            await llm.preload()
            await warm_prompts(llm, settings)
            index = RankedIndex()
            store = RankedStore(index)
            canvas = CanvasService(db, settings=settings, llm=llm)
            service = ChatTurnService(
                db,
                retrieval=RetrievalService(index, index, store, settings.retrieval),
                llm=llm,
                settings=settings,
                canvas=canvas,
            )
            project = await ProjectService(db).create("Answer quality")
            policy = policy_passages(corpus["valmora_travel_expense_policy.docx"].parsed.markdown)
            names = {**FILENAMES, POLICY: "valmora_travel_expense_policy.docx"}
            ids = {d: await add_document(db, project.id, names[d], status="READY", page_count=30) for d in names}
            chunks: dict[str, list[Chunk]] = {d: [] for d in ids}
            for i, p in enumerate([*PASSAGES, *policy]):
                chunk = Chunk(
                    chunk_id=f"{ids[p['document_id']]}:v1:{p['chunk_index']:04d}",
                    project_id=project.id,
                    document_id=ids[p["document_id"]],
                    version=1,
                    chunk_index=p["chunk_index"],
                    chunking_version="v1",
                    page_start=p["page_start"],
                    page_end=p["page_end"],
                    heading_path=p["heading_path"],
                    content_type=p["content_type"],
                    language="en",
                    text=p["text"],
                    token_count=len(p["text"]) // 4,
                )
                await store.upsert([IndexedChunk(chunk, vector_for(f"{i}"))])
                store.keys[chunk.chunk_id] = key(p["document_id"], p["chunk_index"])
                index.by_text[p["text"]] = key(p["document_id"], p["chunk_index"])
                chunks[p["document_id"]].append(chunk)
            for doc in (REPORT, DECK):  # typed tables, linked to their passages (the chart's cells cite them)
                parsed = corpus[FILENAMES[doc]].parsed
                async with db.session() as s:
                    for t in stored_tables(parsed, ids[doc]):
                        fields = t.model_dump(exclude={"id", "document_id", "version", "created_at"})
                        s.add(orm.DocumentTable(document_id=ids[doc], version=1, **fields))
                linked: list[Chunk] = []
                for table in parsed.tables:
                    used = {c.chunk_id for c in linked}
                    match = next(
                        (
                            c
                            for c in chunks[doc]
                            if c.content_type == "table" and c.page_start == table.page_start and c.chunk_id not in used
                        ),
                        None,
                    )
                    if match is not None:
                        linked.append(match.model_copy(update={"table_index": table.index}))
                await canvas.type_document(ids[doc], version=1, parsed=parsed, chunks=linked)

            async def ask(kind: str, docs: Sequence[str], text: str, ranking: dict[str, float]) -> dict[str, Any]:
                index.ranking = ranking
                chat = await ChatService(db).create(project.id, document_scope=[ids[d] for d in docs])
                turn = await service.begin(chat.id, text, modality="voice", length="short")
                sources, visual, agent = [], None, None
                async for event in service.run(turn):
                    if isinstance(event, SourcesEvent):
                        sources = event.sources
                    elif isinstance(event, VisualEvent) and event.visual is not None:
                        visual = event.visual
                    elif isinstance(event, AgentMessageEvent):
                        agent = event.message
                await wait_for_background()
                await settled_callbacks()
                assert agent is not None
                route = agent.route or {}
                row = {
                    "text": text,
                    "answer": agent.text,
                    "words": len(re.sub(r"\[[SW]\d+\]", "", agent.text).split()),
                    "abstained": route.get("abstained"),
                    "abstained_by": route.get("abstained_by"),
                    "citations": [c.source_id for c in agent.citations],
                    "sources": [(c.source_id, c.chunk_id) for c in sources],
                    "visual": visual,
                    "checks": [f"{c['check']}:{c['action']}" for c in route.get("checks") or []],
                }
                rows.setdefault(kind, []).append(row)
                return row

            for _ in range(2):
                row = await ask(
                    "chart", (R, D), "Show me Valmora's quarterly revenue and EBITDA for FY23 and FY24", QUARTERLY
                )
                if row["visual"] is not None:  # item 1: the chart's tables are the answer's evidence; no denial
                    assert {c.chunk_id for c in row["visual"].sources} <= {cid for _, cid in row["sources"]}
                assert not says_not_covered(row["answer"]), row["answer"]
                row = await ask("live", (R,), "USD to INR today", rk())
                assert row["answer"] == LIVE_FIGURE_TEXTS["en"][0]  # item 2
                row = await ask("cin", (R, D), "What is Valmora's CIN?", rk((R, 0, 0.95), (R, 1, 0.7), (R, 69, 0.6)))
                assert CIN in row["answer"]  # item 3
                row = await ask("dividend", (R,), "What is the dividend per share?", rk((R, 64, 0.93), (R, 73, 0.9)))
                assert "15.00" in row["answer"] and "FY24" in row["answer"]  # item 4
                row = await ask("fy25", (R, D), "What was Valmora's revenue in FY25?", rk((R, 6, 0.85), (R, 4, 0.8)))
                assert row["abstained"] is True and row["citations"] == []  # item 5
                row = await ask("name", (R, D), "What was Wall Mora's revenue in FY24?", rk((R, 4, 0.95)))
                assert "Mora" not in row["answer"].replace("Valmora", "")  # item 10
            for text, ranking in LENGTH:
                row = await ask("length", (R,), text, ranking)
                assert row["words"] <= 70, row  # item 7: ~45 words, never past the sentence that reaches it
        finally:
            await wait_for_background()
            await db.close()

    for kind, rs in rows.items():
        for r in rs:
            METRICS[f"{kind}: {r['text'][:40]}"] = f"{r['answer'][:160]!r} ({r['words']} words, {r['checks']})"
    words = [r["words"] for r in rows["length"]]
    METRICS["length: words p50 / max"] = f"{statistics.median(words)} / {max(words)}"
