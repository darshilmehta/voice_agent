"""Title and summary quality with the real model, to judge the prompts by eye. Opt-in:

    RUN_INTEGRATION=1 uv run pytest tests/integration/test_revisit_llm.py -s

Needs only Ollama at 127.0.0.1:11434 with qwen3:4b-instruct pulled: no BGE-M3, reranker, Whisper, Kokoro, Docling or
Qdrant (retrieval is replaced by citations written into the scripted chats). Four short model calls in all; titles and
summaries print in the terminal summary.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest

from app.providers.base import ProviderContext
from app.providers.llm import OllamaLLM
from app.services.chat_summary import ChatSummarizer
from app.services.chats import ChatService
from app.services.messages import MessageService
from app.services.projects import ProjectService
from app.services.titles import TitleService

from ..revisit_helpers import ABSTAINED, ANSWERED, cite
from .conftest import METRICS

pytestmark = pytest.mark.integration

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
REPORT_P2 = cite("S1", "annual_report.pdf", 2, document_id="doc_report")
REPORT_P3 = cite("S1", "annual_report.pdf", 3, document_id="doc_report", snippet="Net debt fell to 1.2x EBITDA.")
DECK_P7 = cite("S1", "investor_deck.pdf", 7, document_id="doc_deck", snippet="Revenue grew 34% year on year.")

ENGLISH_CHAT = [
    ("user", "What was the EBITDA margin in FY24?", None, []),
    ("agent", "The EBITDA margin in FY24 was 18.2%, up from 16.9% in FY23 [S1].", ANSWERED, [REPORT_P2]),
    ("user", "How much did revenue grow?", None, []),
    ("agent", "Revenue grew 34% year on year [S1].", ANSWERED, [DECK_P7]),
    ("user", "What happened to net debt?", None, []),
    ("agent", "Net debt fell to 1.2x EBITDA [S1].", ANSWERED, [REPORT_P3]),
    ("user", "What is the CEO's salary?", None, []),
    ("agent", "I couldn't find that in this chat's documents, so I can't answer it from them.", ABSTAINED, []),
]
HINDI_CHAT = [
    ("user", "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?", None, []),
    ("agent", "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था, जो वित्त वर्ष 2023 के 16.9% से अधिक है [S1]।", ANSWERED, [REPORT_P2]),
    ("user", "राजस्व कितना बढ़ा?", None, []),
    ("agent", "राजस्व में साल-दर-साल 34% की वृद्धि हुई [S1]।", ANSWERED, [DECK_P7]),
]


@pytest.fixture
async def real(db, load_local):
    settings = load_local()
    try:
        tags = httpx.get(f"{settings.llm.base_url}/api/tags", timeout=2).json()
    except httpx.HTTPError as e:
        pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
    if settings.llm.chat_model not in {m["name"] for m in tags.get("models", [])}:
        pytest.skip(f"{settings.llm.chat_model} not pulled in Ollama")
    async with httpx.AsyncClient() as http:
        llm = OllamaLLM(settings.llm, ProviderContext(settings=settings, http=http))
        yield db, settings, llm


async def seeded_chat(db, script, language: str) -> str:
    project = await ProjectService(db).create("Annual report FY24")
    chat = await ChatService(db).create(project.id)
    messages = MessageService(db)
    for role, text, route, citations in script:
        await messages.append(
            chat.id, role=role, text=text, language=language, citations=citations, route=route, modality="voice"
        )
    return chat.id


@pytest.mark.parametrize(("script", "language"), [(ENGLISH_CHAT, "en"), (HINDI_CHAT, "hi")], ids=["en", "hi"])
async def test_title_and_summary_of_a_scripted_chat(real, script, language):
    db, settings, llm = real
    chat_id = await seeded_chat(db, script, language)

    t0 = time.perf_counter()
    chat = await TitleService(db, llm=llm, settings=settings).generate(chat_id)
    title_s = time.perf_counter() - t0
    METRICS[f"title [{language}]"] = f"{chat.title!r} ({title_s:.1f}s)"
    assert chat.title_is_auto and chat.title != "New chat"
    assert 1 <= len(chat.title.split()) <= 6 and not chat.title.endswith((".", "?", "!", "।")) and '"' not in chat.title
    if language == "hi":
        assert DEVANAGARI.search(chat.title)

    t0 = time.perf_counter()
    summary = await ChatSummarizer(db, llm=llm, settings=settings).generate(chat_id)
    METRICS[f"summary [{language}]"] = f"{time.perf_counter() - t0:.1f}s\n{summary.content}"
    assert summary.language == language and summary.overview and not summary.stale
    assert 1 <= len(summary.key_points) <= 8
    real_sources = {("annual_report.pdf", 2), ("annual_report.pdf", 3), ("investor_deck.pdf", 7)}
    cited = {(r.filename, r.page_start) for p in summary.key_points for r in p.sources}
    assert cited and cited <= real_sources  # grounded: only pages the chat's answers cited
    if language == "en":
        assert [q.question for q in summary.unanswered_questions] == ["What is the CEO's salary?"]
    else:
        assert DEVANAGARI.search(summary.overview)


async def test_map_reduce_with_the_real_model(real):
    """The same chat in tiny windows, so the reduce prompt (combining partial summaries) runs on the real model too."""
    db, settings, llm = real
    chat_id = await seeded_chat(db, ENGLISH_CHAT, "en")
    summarizer = ChatSummarizer(db, llm=llm, settings=settings, window_tokens=110)

    t0 = time.perf_counter()
    summary = await summarizer.generate(chat_id)
    stored = await summarizer.summaries.get(chat_id, "user")
    windows = stored.data["windows"]  # type: ignore[index]
    METRICS["summary [map-reduce]"] = f"{windows} windows, {time.perf_counter() - t0:.1f}s\n{summary.content}"
    assert windows >= 2  # more than one window, so the reduce prompt ran
    cited = {(r.filename, r.page_start) for p in summary.key_points for r in p.sources}
    assert cited <= {("annual_report.pdf", 2), ("annual_report.pdf", 3), ("investor_deck.pdf", 7)}
    assert [q.question for q in summary.unanswered_questions] == ["What is the CEO's salary?"]
