"""The canvas in the conversation, end to end on the real model (docs/DESIGN.md §12.1, opt-in).

    CANVAS_PARSE_CACHE=<folder> RUN_INTEGRATION=1 uv run pytest tests/integration/test_canvas_conversation_e2e.py -s

Needs Ollama (qwen3:4b-instruct) and the eval corpus's tables (Docling, or a parse cache). Retrieval is cached as in
``test_answer_latency.py`` (the corpus's passages as the pipeline chunked them; the pages a turn names rank first,
after the measured idle-machine search and rerank times), so no embedder, reranker or Qdrant: everything else is the
real text path, ``ChatTurnService`` with the router, the answer, the canvas service, its planner, the builder and
the grounding check, over the typed tables of the Valmora annual report and the Zephyra deck.

A conversation of turns, each consumed as the SSE endpoint does (the stream stays open for the visual): a requested
chart, an edit, a question about the chart on screen, a suggested chart, Hindi (a requested chart and an edit),
removing it. Then the scheduling question asked of the whole pipeline: a question asked right after an answer whose
visual is still being planned (the planner gives way) against the same question asked once it is done.

Reported per turn: route and answer mode, the first delta, the answer complete, agent_message, the visual ready (and
how long after the answer), what it is (kind, table), its grounding, and for edits what changed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from app.db import models as orm
from app.domain.canvas import CanvasEvent, Visual, VisualEvent
from app.providers.base import ProviderContext
from app.providers.ingestion import Chunk
from app.providers.llm import OllamaLLM
from app.providers.retrieval import IndexedChunk
from app.providers.storage import SqliteDB
from app.services.canvas.builder import check_grounding
from app.services.canvas.conversation import settled_callbacks
from app.services.canvas.service import CanvasService
from app.services.chat_turns import AgentMessageEvent, ChatTurnService, DeltaEvent, wait_for_background
from app.services.chats import ChatService
from app.services.messages import MessageService
from app.services.preload import warm_prompts
from app.services.projects import ProjectService
from app.services.retrieval import RetrievalService
from app.settings import load_settings

from ..conftest import add_document
from ..fakes import vector_for
from .canvas_corpus import CorpusDocument, corpus_fixture, stored_tables  # noqa: F401  (the "corpus" fixture)
from .conftest import LOCAL_CONFIG, METRICS
from .test_answer_latency import DECK, FILENAMES, PASSAGES, REPORT, CachedIndex, CachedStore, OllamaStats

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class T:
    kind: str
    text: str
    pages: tuple[int, ...] = ()
    table: str | None = None  # the table the visual should come from (a title's start, lower case)
    edit: str | None = None  # the kind an edit should leave


CONVERSATION = [
    T("requested", "Show me Valmora's revenue by quarter for FY24", (19,), table="quarterly performance: fy24"),
    T("edit", "make it a bar chart", edit="bar"),
    T("about the chart", "What's the second bar on the chart?", (19,)),
    T("suggested", "How did each segment's revenue change from FY23 to FY24?", (18,), table="segment results"),
    T("requested (hi)", "वालमोरा की FY24 की तिमाही आय का चार्ट दिखाओ", (19,), table="quarterly performance: fy24"),
    T("edit (hi)", "इसे टेबल में दिखाओ", edit="table"),
    T("remove (hi)", "हटा दो", edit="removed"),
]
FOLLOW_UP = "What was the EBITDA margin in FY24?"


async def test_the_canvas_in_a_real_conversation(corpus: dict[str, CorpusDocument], tmp_path):
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
        try:
            await llm.preload()
            await warm_prompts(llm, settings)  # as at the app's startup (router, answer and planner prompts)
            index = CachedIndex()
            store = CachedStore(index)
            retrieval = RetrievalService(index, index, store, settings.retrieval)
            canvas = CanvasService(db, settings=settings, llm=llm)
            service = ChatTurnService(db, retrieval=retrieval, llm=llm, settings=settings, canvas=canvas)
            project = await ProjectService(db).create("Canvas e2e")
            ids = await load_documents(db, project.id, corpus, store, canvas)
            titles = {d.table_id: d.title for d in await canvas.store.project_datasets(project.id)}

            chat = await ChatService(db).create(project.id, document_scope=[ids[REPORT]])
            for t in CONVERSATION:
                index.pages = t.pages
                rows.append(await sse_turn(service, chat.id, t, titles))
                await wait_for_background()
            await settled_callbacks()
            transcript = (await MessageService(db).list(chat.id)).items
            follow = await follow_up_scheduling(service, db, project.id, ids, index)
        finally:
            await wait_for_background()
            await db.close()

    for r in rows:
        METRICS[f"turn {r['kind']}"] = (
            f"{r['text'][:48]!r} → {r['intent']}/{r['mode']}; first delta {r['first_delta']}, answer done "
            f"{r['answer_done']}, agent_message {r['agent_message']} ms; visual {r['visual']}; "
            f"answer {r['answer'][:140]!r}"
        )
    tabled = [r for r in rows if r["table_ok"] is not None]
    METRICS["visuals from the expected table"] = f"{sum(r['table_ok'] for r in tabled)}/{len(tabled)}"
    ready = [r for r in rows if r["ready_after_answer_ms"] is not None]
    if ready:
        METRICS["visual ready after the answer's text (ms, each)"] = [r["ready_after_answer_ms"] for r in ready]
        METRICS["visual ready after the first delta (ms, each)"] = [r["ready_after_first_ms"] for r in ready]
        METRICS["the draft after the first delta (ms, each)"] = [r["draft_after_first_ms"] for r in ready]
        METRICS["visual plans"] = [r["plan"] for r in ready]
    METRICS["follow-up first delta: visual cancelled vs done (ms)"] = follow
    marked = [m.route.get("visual_id") for m in transcript if m.role == "agent" and m.route]
    METRICS["agent messages with a visual id"] = f"{sum(1 for v in marked if v)}/{len(marked)}"
    for r in rows:
        assert r["grounded"] in (True, None), r  # every number on screen is a cell (or a calculation of cells)
    edits = [r for r in rows if r["kind"].startswith(("edit", "remove"))]
    assert all(r["edit_ok"] for r in edits), edits


async def load_documents(
    db: SqliteDB, project_id: str, corpus: dict[str, CorpusDocument], store: CachedStore, canvas: CanvasService
) -> dict[str, str]:
    """The report and the deck as READY documents: their passages in the cached index, their tables typed into
    datasets (linked to their table passages, so a visual's cells cite the turn's own [S#])."""
    ids: dict[str, str] = {}
    run = f"run {time.time_ns() % 10**9}"
    for doc in (REPORT, DECK):
        ids[doc] = await add_document(db, project_id, FILENAMES[doc], status="READY", page_count=30)
    chunks: dict[str, list[Chunk]] = {doc: [] for doc in ids}
    for i, p in enumerate(PASSAGES):
        chunk = Chunk(
            chunk_id=f"{ids[p['document_id']]}:v1:{p['chunk_index']:04d}",
            project_id=project_id,
            document_id=ids[p["document_id"]],
            version=1,
            chunk_index=p["chunk_index"],
            chunking_version="v1",
            page_start=p["page_start"],
            page_end=p["page_end"],
            heading_path=[*p["heading_path"], run],
            content_type=p["content_type"],
            language="en",
            text=p["text"],
            token_count=len(p["text"]) // 4,
        )
        await store.upsert([IndexedChunk(chunk, vector_for(f"{i}"))])
        chunks[p["document_id"]].append(chunk)
    for doc, document_id in ids.items():
        parsed = corpus[FILENAMES[doc]].parsed
        async with db.session() as s:
            for t in stored_tables(parsed, document_id):
                fields = t.model_dump(exclude={"id", "document_id", "version", "created_at"})
                s.add(orm.DocumentTable(document_id=document_id, version=1, **fields))
        # each table's chunk: the next table passage on its page
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
        await canvas.type_document(document_id, version=1, parsed=parsed, chunks=linked)
    return ids


async def sse_turn(service: ChatTurnService, chat_id: str, t: T, titles: dict[str, str]) -> dict[str, Any]:
    turn = await service.begin(chat_id, t.text, modality="text", length="short")
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    visual: Visual | None = None
    panels: list[Visual] | None = None
    agent = None
    async for event in service.run(turn):  # as the SSE endpoint: the stream stays open for the visual
        now = (time.perf_counter() - t0) * 1000
        if isinstance(event, DeltaEvent):
            marks.setdefault("first_delta", now)
            marks["answer_done"] = now
        elif isinstance(event, AgentMessageEvent):
            marks["agent_message"] = now
            agent = event.message
        elif isinstance(event, VisualEvent):
            marks[f"visual_{event.phase}"] = now  # the last one: the planner's, when it replaced the draft
            if event.phase == "ready":
                marks.setdefault("visual_first_ready", now)  # the draft (or the planner's, without one)
            if event.visual is not None:
                visual = event.visual
        elif isinstance(event, CanvasEvent):
            panels = event.panels
    assert agent is not None
    route = agent.route or {}
    row: dict[str, Any] = {
        "kind": t.kind,
        "text": t.text,
        "intent": route.get("intent"),
        "mode": route.get("answer"),
        "answer": agent.text,
        "first_delta": round(marks.get("first_delta", 0)),
        "answer_done": round(marks.get("answer_done", 0)),
        "agent_message": round(marks.get("agent_message", 0)),
        "ready_after_answer_ms": None,
        "ready_after_first_ms": None,
        "draft_after_first_ms": None,
        "plan": route.get("visual_plan"),
        "grounded": None,
        "visual": "none",
        "table_ok": None,
    }
    if visual is not None:
        row["grounded"] = check_grounding(visual) == []
        cells = [c for r in visual.rows for c in r.cells.values() if c] + [x.cell for x in visual.tiles if x.cell]
        tables = sorted({titles.get(c.table_id, "?") for c in cells})
        row["visual"] = f"{visual.kind} {visual.title!r} ({len(visual.rows) or len(visual.tiles)} points) from {tables}"
        if t.table is not None:
            row["table_ok"] = bool(tables) and all(title.casefold().startswith(t.table) for title in tables)
        if "visual_ready" in marks and t.edit is None:
            row["ready_after_answer_ms"] = round(marks["visual_ready"] - marks["answer_done"])
            row["ready_after_first_ms"] = round(marks["visual_ready"] - marks["first_delta"])
            row["draft_after_first_ms"] = round(marks["visual_first_ready"] - marks["first_delta"])
    elif "visual_failed" in marks:
        row["visual"] = "failed"
    if t.edit is not None:
        edit = route.get("canvas_edit") or {}
        if t.edit == "removed":
            row["edit_ok"] = edit.get("op") == "remove" and edit.get("outcome") == "done"
        else:  # done, or "it's already shown that way" when the planner had picked that kind
            row["edit_ok"] = edit.get("outcome") == "same" or (visual is not None and visual.kind == t.edit)
        row["visual"] += f"; edit {edit}"
    if t.table is not None:
        row["visual"] += f"; expected from {t.table!r}"
    row["panels"] = len(panels) if panels is not None else None
    return row


async def follow_up_scheduling(
    service: ChatTurnService, db: SqliteDB, project_id: str, ids: dict[str, str], index: CachedIndex
) -> list[str]:
    """A question asked right after an answer whose visual is being planned, against the same asked once the visual
    is done: the first delta of each (the planner gives way, so the first should not wait for it)."""
    out = []
    for wait in (False, True):
        chat = await ChatService(db).create(project_id, document_scope=[ids[REPORT]])
        index.pages = (19,)
        sink: list[Any] = []

        async def collect(event: VisualEvent | CanvasEvent, into: list[Any] = sink) -> None:
            into.append(event)

        first = await service.begin(chat.id, "Show me Valmora's revenue by quarter for FY24")
        [_ async for _ in service.run(first, on_visual=collect)]
        if wait:
            while not any(isinstance(e, VisualEvent) and e.phase in ("ready", "failed") for e in sink):
                await asyncio.sleep(0.05)
        index.pages = (3,)
        follow = await service.begin(chat.id, FOLLOW_UP)
        t0 = time.perf_counter()
        first_delta = None
        async for event in service.run(follow, on_visual=collect):
            if isinstance(event, DeltaEvent) and first_delta is None:
                first_delta = (time.perf_counter() - t0) * 1000
        await wait_for_background()
        phases = [e.phase for e in sink if isinstance(e, VisualEvent)]
        out.append(f"{'after the visual' if wait else 'while planning'}: {first_delta:.0f} ms (visual: {phases})")
    return out
