"""The last round's backend fixes in whole turns (``ChatTurnService`` on fakes, scripted model), from the post-fix
real-model verification: a year-end date is the end of a fiscal year (item 3), no invented figures after a misheard
question (item 1), a rewrite that names another document's subject (item 4), Hindi short replies cut mid-sentence
(item 5) and the say-again line (item 6)."""

from __future__ import annotations

from typing import Any

import pytest

from app.providers.retrieval import IndexedChunk
from app.services.chat_turns import AgentMessageEvent, ChatTurnService, DeltaEvent, wait_for_background
from app.services.chats import ChatService
from app.services.projects import ProjectService
from app.services.retrieval import RetrievalService

from .conftest import add_document
from .fakes import make_chunk, vector_for

LABEL = "valmora annual report fy24: Valmora Industries Limited - Annual Report 2023-24"
REPORT = [
    (
        ["Management discussion and analysis", "Recap of FY23, the comparative year"],
        "In FY23, Valmora's revenue from operations was ₹ 6,482 crore. Net debt stood at ₹ 1,188 crore at 31 March "
        "2023, or 0.93x EBITDA.",
    ),
    (
        ["Management discussion and analysis: FY24 financial performance"],
        "Net debt declined to ₹ 831 crore from ₹ 1,188 crore, and ROCE rose to 19.6% from 17.9%.",
    ),
    (["Highlights"], "| Metric | FY24 | FY23 |\n|---|---|---|\n| Revenue from operations (₹ crore) | 7,365 | 6,482 |"),
]


@pytest.fixture
async def world(db, load_local, fakes):
    """Valmora's annual report (three passages) in the fake index, a chat, the turn service on fakes."""
    settings = load_local()
    project = await ProjectService(db).create("Valmora")
    doc = await add_document(db, project.id, "valmora_annual_report_fy24.pdf", status="READY", page_count=30)
    for i, (heading, text) in enumerate(REPORT):
        chunk = make_chunk(
            i,
            project_id=project.id,
            document_id=doc,
            text=text,
            heading_path=heading,
            document_label=LABEL,
            page_start=16 + i,
            page_end=16 + i,
        )
        await fakes.store.upsert([IndexedChunk(chunk, vector_for(text))])
    chat = await ChatService(db).create(project.id)
    retrieval = RetrievalService(fakes.embedder, fakes.reranker, fakes.store, settings.retrieval)
    service = ChatTurnService(db, retrieval=retrieval, llm=fakes.llm, settings=settings)
    return service, chat.id


async def turn(service: ChatTurnService, chat_id: str, text: str, **kw: Any) -> list[Any]:
    t = await service.begin(chat_id, text, **kw)
    events = [e async for e in service.run(t)]
    await wait_for_background()
    return events


def saved(events: list[Any]):
    return next(e.message for e in events if isinstance(e, AgentMessageEvent))


def spoken(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, DeltaEvent))


def answer_prompts(llm) -> list[str]:
    return [c["messages"][-1].content for c in llm.calls if "Question:" in c["messages"][-1].content]


# ------------------------------------------------------------------ item 3: the end of a fiscal year

NET_DEBT = "What was Valmora's net debt on 31 March 2024?"


def recap_first(q: str, p: str) -> float:
    """What the real reranker did: the recap of FY23 ("… at 31 March 2023") just above the FY24 paragraph."""
    return 0.93 if "Recap of FY23" in p else 0.81 if "831" in p else 0.0


async def test_a_year_end_date_is_answered_from_that_fiscal_years_passage(world, fakes):
    service, chat_id = world
    fakes.reranker.scorer = recap_first
    fakes.llm.route = {"intent": "document_qa", "query": None}
    fakes.llm.reply = "Net debt was ₹ 831 crore at 31 March 2024 [S1]."
    events = await turn(service, chat_id, NET_DEBT, modality="voice")
    prompt = answer_prompts(fakes.llm)[-1]
    first = prompt.split("[S2]")[0]
    assert "831" in first and "Recap of FY23" not in first  # the FY24 paragraph is S1
    assert "31 March 2024 is the end of FY24" in prompt
    assert saved(events).text == "Net debt was ₹ 831 crore at 31 March 2024 [S1]."
    assert saved(events).route["abstained"] is False


async def test_without_the_fiscal_years_passage_it_abstains_rather_than_give_another_years_figure(world, fakes):
    service, chat_id = world
    fakes.reranker.scorer = lambda q, p: 0.93 if "Recap of FY23" in p else 0.0
    fakes.llm.route = {"intent": "document_qa", "query": None}
    fakes.llm.reply = "Net debt was ₹ 1,188 crore [S1]."
    events = await turn(service, chat_id, NET_DEBT, modality="voice")
    message = saved(events)
    assert message.route["abstained"] is True and "1,188" not in message.text
    assert answer_prompts(fakes.llm) == []  # the gate abstained: no answer model call


@pytest.mark.parametrize("asked", ["How much net debt was there as at March 31, 2024?", "Net debt on 31.03.2024?"])
async def test_a_general_proposal_for_a_year_end_question_is_checked_against_the_documents(world, fakes, asked):
    """The B1 check: a year-end date is the documents' subject, like "FY24" (a match below the strong 0.6 makes a
    question that isn't about the documents' subject a mixed one)."""
    service, chat_id = world
    fakes.reranker.scorer = lambda q, p: 0.45 if "Recap of FY23" in p else 0.4 if "831" in p else 0.0
    fakes.llm.route = {"intent": "general_qa", "query": None}
    fakes.llm.reply = "Net debt was ₹ 831 crore at 31 March 2024 [S1]."
    events = await turn(service, chat_id, asked, modality="voice")
    assert saved(events).route["intent"] == "document_qa"
    assert "31 March 2024 is the end of FY24" in answer_prompts(fakes.llm)[-1]
