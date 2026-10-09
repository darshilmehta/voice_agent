"""Quality-round behaviours on the real model (qwen3:4b-instruct), with the cached retrieval of test_answer_latency:

    RUN_INTEGRATION=1 uv run pytest tests/integration/test_quality_round_llm.py -s

- B5: a Hinglish question, then "answer in English please": the answer is in English, saved and spoken as English.
- UX5: two documents with the same metrics (the Valmora annual report, the Zephyra investor deck); "What was revenue
  from operations in FY24?" with passages of both: the answer says which document each figure comes from.

Needs only Ollama: no embedder, reranker, speech models, Docling or Qdrant.
"""

from __future__ import annotations

import httpx
import pytest

from app.providers.base import ProviderContext
from app.providers.ingestion import Chunk
from app.providers.llm import OllamaLLM
from app.providers.retrieval import IndexedChunk
from app.providers.storage import SqliteDB
from app.services.chat_turns import ChatTurnService, wait_for_background
from app.services.chats import ChatService
from app.services.language import script_language
from app.services.projects import ProjectService
from app.services.retrieval import RetrievalService
from app.settings import load_settings

from ..conftest import add_document
from ..fakes import vector_for
from .conftest import LOCAL_CONFIG, METRICS
from .test_answer_latency import DECK, FILENAMES, PASSAGES, REPORT, CachedIndex, CachedStore

pytestmark = pytest.mark.integration


@pytest.fixture
async def world(tmp_path):
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    async with httpx.AsyncClient() as http:
        try:
            tags = (await http.get(f"{settings.llm.base_url}/api/tags", timeout=2)).json()
        except httpx.HTTPError as e:
            pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
        if settings.llm.chat_model not in {m["name"] for m in tags.get("models", [])}:
            pytest.skip(f"{settings.llm.chat_model} not pulled")
        ctx = ProviderContext(settings=settings, http=http)
        db = SqliteDB(settings.metadata_db, ctx)
        await db.start()
        try:
            index = CachedIndex()
            store = CachedStore(index)
            llm = OllamaLLM(settings.llm, ctx)
            service = ChatTurnService(
                db, retrieval=RetrievalService(index, index, store, settings.retrieval), llm=llm, settings=settings
            )
            project = await ProjectService(db).create("Two companies")
            ids = {d: await add_document(db, project.id, FILENAMES[d], status="READY") for d in (REPORT, DECK)}
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
                    heading_path=p["heading_path"],
                    content_type=p["content_type"],
                    language="en",
                    text=p["text"],
                    token_count=len(p["text"]) // 4,
                )
                await store.upsert([IndexedChunk(chunk, vector_for(f"{i}"))])
            yield service, project, ids, index
        finally:
            await wait_for_background()
            await db.close()


async def ask(service: ChatTurnService, chat_id: str, text: str):
    turn = await service.begin(chat_id, text, modality="voice", length="short")
    events = [e async for e in service.run(turn)]
    await wait_for_background()
    return events[-1].message


async def test_b5_answer_in_english_after_a_hinglish_question(world):
    service, project, ids, index = world
    chat = await ChatService(db_of(service)).create(project.id, document_scope=[ids[REPORT]])
    index.pages = (3, 17)
    first = await ask(service, chat.id, "Valmora ka FY24 mein revenue kitna tha?")
    english = await ask(service, chat.id, "answer in English please")
    METRICS["B5 Hinglish question"] = f"[{first.language}] {first.text!r}"
    METRICS["B5 then 'answer in English please'"] = (
        f"[{english.language}] {english.text!r}; asked {english.route['rewritten_query']!r}; "
        f"retried: {english.route['language_retry']}"
    )
    assert first.language == "hi"
    assert english.language == "en" == script_language(english.text) and "7,365" in english.text


async def test_ux5_an_answer_from_two_documents_says_which_is_which(world):
    service, project, _, index = world
    chat = await ChatService(db_of(service)).create(project.id)  # both documents
    index.pages = (3, 17)  # the report's financial highlights; the deck's table comes in by keywords
    answer = await ask(service, chat.id, "What was revenue from operations in FY24?")
    METRICS["UX5 answer from two documents"] = (
        f"{answer.text!r}; named documents: {answer.route['named_documents']}; "
        f"cited: {sorted({c.filename for c in answer.citations})}"
    )
    assert answer.route["named_documents"] is True
    files = {c.filename for c in answer.citations}
    if len(files) > 1 or FILENAMES[DECK] in files:  # it used the deck: it must say so
        assert "zephyra" in answer.text.casefold() or "deck" in answer.text.casefold()
    assert (
        "valmora" in answer.text.casefold() or "report" in answer.text.casefold() or "zephyra" in answer.text.casefold()
    )


def db_of(service: ChatTurnService):
    return service.chats.db
