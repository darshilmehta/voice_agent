"""The quality round's answer checks in whole turns (``ChatTurnService`` on fakes, scripted model): the chart on screen
and the answer agree (item 1), live figures without live data (2), codes copied exactly (3), the latest period when
none is named (4), "not covered" answers recorded as abstentions (5), garbled transcripts (8), general answers in a
documents chat (9), misheard names (10) and the length of short answers (7)."""

from __future__ import annotations

import re
from functools import partial
from typing import Any

import pytest

from app.domain.canvas import VisualEvent
from app.providers.retrieval import IndexedChunk
from app.services.canvas.conversation import settled_callbacks
from app.services.chat_turns import (
    SHORT_ANSWER_WORDS,
    AgentMessageEvent,
    ChatTurnService,
    DeltaEvent,
    SourcesEvent,
    wait_for_background,
)
from app.services.chats import ChatService
from app.services.messages import MessageService
from app.services.projects import ProjectService
from app.services.prompts import ACK_TEXTS, LIVE_FIGURE_TEXTS, NOT_FROM_DOCUMENTS
from app.services.retrieval import RetrievalService

from .canvas_turn_helpers import Script, choose, report_chat
from .conftest import add_document
from .fakes import make_chunk, vector_for

CIN = "L24119GJ1994PLC023871"
PASSAGES = [
    f"Valmora Industries Limited. CIN: {CIN}. Registered office: Vapi, Gujarat.",
    "| Financial year | Dividend per share (₹) |\n|---|---|\n| FY24 (recommended) | 15.00 |\n| FY23 | 12.00 |",
    "The final dividend for FY23 of ₹ 150 crore was equal to ₹ 12.00 per share.",
    "| Metric | FY24 | FY23 |\n|---|---|---|\n| Revenue from operations (₹ crore) | 7,365 | 6,482 |",
    "Hotel limits per night.\n| Grade | Tier-1 city (₹ per night) | Tier-2 city (₹ per night) |\n|---|---|---|\n"
    "| L3 to L4 | 7,500 | 5,200 |",
    "In FY24 the Board approved capital expenditure of ₹ 1,100 crore over FY25 and FY26.",
]


def prefer(*needles: str) -> Any:
    """A reranker that finds the passages with one of ``needles`` strong (0.9) and the rest weak (0.05, over the
    gate): what the real reranker does for a question those passages answer."""
    return lambda q, p: 0.9 if any(n in p for n in needles) else 0.05


@pytest.fixture
async def world(db, load_local, fakes):
    """One READY document (Valmora's annual report) in the fake index, a chat per test, the turn service on fakes."""
    settings = load_local()
    project = await ProjectService(db).create("Valmora")
    doc = await add_document(db, project.id, "valmora_annual_report_fy24.pdf", status="READY", page_count=30)
    for i, text in enumerate(PASSAGES):
        chunk = make_chunk(i, project_id=project.id, document_id=doc, text=text, page_start=i + 1, page_end=i + 1)
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


# ------------------------------------------------------------------ item 1: no denial of what a strong passage says


def source_of(prompt: str, needle: str) -> str:
    """The [S#] id the answer prompt gives the passage with ``needle``."""
    return next(sid for sid, text in re.findall(r"\[(S\d+)\][^\n]*\n([^\n]*(?:\n\|[^\n]*)*)", prompt) if needle in text)


async def test_a_denial_of_a_strong_passage_is_asked_again(world, fakes):
    service, chat_id = world
    replies = iter(
        [
            "The documents do not cover hotel provisions for L3 employees in tier 2 cities. [{sid}]",
            "The hotel limit for L3 employees in tier-2 cities is ₹5,200 a night [{sid}].",
        ]
    )
    fakes.llm.reply = lambda m: next(replies).format(sid=source_of(m[-1].content, "Hotel limits"))
    fakes.reranker.scorer = prefer("Hotel limits")
    events = await turn(service, chat_id, "What are the hotel provisions for L3 employees in tier 2 cities?")
    agent = saved(events)
    sid = source_of(answer_prompts(fakes.llm)[0], "Hotel limits")
    assert spoken(events) == f"The hotel limit for L3 employees in tier-2 cities is ₹5,200 a night [{sid}]."  # only
    assert agent.route["abstained"] is False and [c.source_id for c in agent.citations] == [sid]
    assert agent.route["checks"][0]["check"] == "coverage" and agent.route["checks"][0]["action"] == "retry"
    second = fakes.llm.calls[-1]["messages"][-1].content
    assert f"IMPORTANT: the sources do contain this: [{sid}]" in second  # told what holds it


async def test_a_second_denial_is_replaced_by_a_pointer_to_the_passage(world, fakes):
    service, chat_id = world
    fakes.llm.reply = "The policy does not cover hotel provisions."
    fakes.reranker.scorer = prefer("Hotel limits")
    events = await turn(service, chat_id, "What are the hotel provisions for L3 employees in tier 2 cities?")
    agent = saved(events)
    sid = source_of(answer_prompts(fakes.llm)[0], "Hotel limits")
    assert len(answer_prompts(fakes.llm)) == 2  # asked once more, then corrected
    assert "does not cover" not in spoken(events) and spoken(events).endswith(f"[{sid}].")
    assert agent.text.startswith("The Valmora annual report covers this") and agent.route["abstained"] is False
    assert [c["action"] for c in agent.route["checks"]] == ["retry", "corrected"]


# ------------------------------------------------------------------ item 1: the chart on screen and the answer


@pytest.fixture
def report(make_app, fakes):
    with make_app() as api:
        _, chat_id, _ = report_chat(api)
        yield api, chat_id, fakes


def run(api, fn, *args, **kwargs):
    return api.portal.call(partial(fn, *args, **kwargs))


async def canvas_turn(api, chat_id: str, text: str) -> list[Any]:
    service = ChatTurnService.from_container(api.app.state.container, canvas=api.app.state.canvas)
    t = await service.begin(chat_id, text, modality="voice", length="short")
    events = [e async for e in service.run(t)]
    await wait_for_background()
    await settled_callbacks()
    return events


def test_the_drafts_table_joins_the_voice_answers_evidence_and_the_answer_is_told_what_is_on_screen(report):
    """The final real run: the chart showed all the quarters, the answer (best three passages, without the quarterly
    table) said the report doesn't give them."""
    api, chat_id, fakes = report
    Script(planner=choose("line", "Revenue")).install(fakes.llm)
    # the quarterly table ranks last: a voice answer's best three passages leave it out
    fakes.reranker.scorer = lambda q, p: 0.1 if "Q1 FY24" in p else 0.9
    fakes.llm.reply = "The chart shows revenue rising in every quarter of FY24, to 1,933 in Q4 [S4]."
    events = run(api, canvas_turn, api, chat_id, "Show me revenue by quarter")
    ready = next(e.visual for e in events if isinstance(e, VisualEvent) and e.phase == "ready")
    table = ready.sources[0]
    sources = next(e for e in events if isinstance(e, SourcesEvent)).sources
    assert table.chunk_id in {s.chunk_id for s in sources}  # in the answer's evidence, with the chart's own id
    assert next(s for s in sources if s.chunk_id == table.chunk_id).source_id == table.source_id
    prompt = answer_prompts(fakes.llm)[-1]
    assert "| Q1 FY24 | 1,742 |" in prompt  # the table itself
    assert f"A chart of this is on screen now, drawn by the app from [{table.source_id}]" in prompt
    agent = saved(events)
    assert agent.route["visual_status"] == "ready" and [c.source_id for c in agent.citations] == [table.source_id]


def test_an_answer_that_denies_the_chart_is_asked_again_then_corrected(report):
    api, chat_id, fakes = report
    Script(planner=choose("line", "Revenue")).install(fakes.llm)
    fakes.reranker.scorer = lambda q, p: 0.1 if "Q1 FY24" in p else 0.9
    fakes.llm.reply = "The report does not provide quarterly revenue figures for FY24 [S1]."
    events = run(api, canvas_turn, api, chat_id, "Show me revenue by quarter")
    agent = saved(events)
    assert len(answer_prompts(fakes.llm)) == 2
    assert "does not provide" not in spoken(events)
    assert agent.text.startswith("The chart on screen shows") and agent.route["abstained"] is False
    assert agent.route["visual_status"] == "ready"  # the chart stays: the answer now agrees with it


# ------------------------------------------------------------------ item 2: live figures without live data


async def test_a_live_rate_without_web_search_gets_the_honest_line_not_a_guess(world, fakes):
    service, chat_id = world
    fakes.llm.route = {"intent": "general_qa", "query": None}
    fakes.llm.reply = "As of now, the USD to INR exchange rate is approximately 83.50."
    events = await turn(service, chat_id, "USD to INR today")
    agent = saved(events)
    assert agent.text == LIVE_FIGURE_TEXTS["en"][0] and "83" not in spoken(events)
    assert fakes.llm.calls == []  # never asked: the 4B model guessed whatever the prompt said
    assert agent.route["live_fixed"] is True and agent.route["basis"] == []
    events = await turn(service, chat_id, "आज डॉलर का भाव क्या है?", language="hi")
    assert saved(events).text == LIVE_FIGURE_TEXTS["hi"][0]


async def test_a_document_answer_with_a_live_part_keeps_its_figures_and_drops_a_guessed_one(world, fakes):
    service, chat_id = world
    fakes.llm.route = {"intent": "mixed", "query": None}
    fakes.llm.reply = "Valmora's FY24 revenue was ₹7,365 crore [S1]. Its share price today is about ₹1,240."
    events = await turn(service, chat_id, "What was Valmora's revenue in FY24 and what is its share price today?")
    agent = saved(events)
    assert agent.text == f"Valmora's FY24 revenue was ₹7,365 crore [S1]. {LIVE_FIGURE_TEXTS['en'][0]}"
    assert agent.route["checks"][0]["check"] == "live_figure"


# ------------------------------------------------------------------ item 3: codes


async def test_a_code_one_digit_off_is_copied_from_the_source(world, fakes):
    service, chat_id = world
    fakes.llm.reply = "Valmora's CIN is L24119GJ1994PLC023971 [S1]."
    fakes.reranker.scorer = prefer("CIN")
    events = await turn(service, chat_id, "What is Valmora's CIN?")
    assert CIN in spoken(events) and saved(events).text == f"Valmora's CIN is {CIN} [S1]."
    assert saved(events).route["checks"][0] == {
        "check": "identifier",
        "action": "corrected",
        "text": "L24119GJ1994PLC023971",
        "to": CIN,
    }


# ------------------------------------------------------------------ item 4: no period named


async def test_a_question_without_a_period_is_answered_for_the_latest_one(world, fakes):
    service, chat_id = world
    fakes.reranker.scorer = prefer("dividend")
    await turn(service, chat_id, "What is the dividend per share?")
    assert "answer for the latest, FY24, and say that it is for FY24" in answer_prompts(fakes.llm)[-1]
    await turn(service, chat_id, "What was the dividend per share in FY23?")
    assert "names no period" not in answer_prompts(fakes.llm)[-1]


# ------------------------------------------------------------------ item 5: "not covered" answers


async def test_an_answer_saying_the_documents_dont_cover_it_is_an_abstention_without_chips(world, fakes):
    service, chat_id = world
    fakes.llm.reply = "The Valmora annual report does not mention Valmora's revenue for FY25. [S1]"
    # FY25 passes the gate (the best passage names it); the answer says the documents don't give it
    fakes.reranker.scorer = lambda q, p: 0.95 if "capital expenditure" in p else 0.9 if "Revenue" in p else 0.05
    events = await turn(service, chat_id, "What was Valmora's revenue in FY25 and FY24?")
    assert saved(events).route["abstained"] is False  # FY24 is not denied: a partial answer
    events = await turn(service, chat_id, "What was Valmora's revenue in FY25?")
    agent = saved(events)
    assert agent.route["abstained"] is True and agent.route["abstained_by"] == "answer"
    assert (
        agent.citations == [] and agent.text == "The Valmora annual report does not mention Valmora's revenue for FY25."
    )
    fakes.llm.reply = "FY25 के राजस्व की जानकारी दस्तावेज़ों में नहीं दी गई है [S1]।"
    events = await turn(service, chat_id, "FY25 में Valmora का राजस्व कितना था?", language="hi")
    assert saved(events).route["abstained"] is True and saved(events).citations == []


# ------------------------------------------------------------------ item 8: garbled transcripts


async def test_a_garbled_transcript_is_asked_again_once(world, fakes):
    service, chat_id = world
    garbled = "आब आब आब आब आब आब आब आब"
    events = await turn(service, chat_id, garbled, modality="voice", garbled=True)
    agent = saved(events)
    assert agent.text == ACK_TEXTS["repeat"]["hi"] and agent.route["intent"] == "clarification"
    assert fakes.llm.calls == [] and fakes.llm.json_calls == []  # no router, no answer: nothing to understand
    events = await turn(
        service, chat_id, "Just 1 employee sign here... Just 1 employee sign here...", modality="voice", garbled=True
    )
    assert saved(events).role == "event" and spoken(events) == ""  # not asked twice in a row: silence
    events = await turn(service, chat_id, "stop", modality="voice", garbled=True)
    assert saved(events).route["intent"] == "stop"


# ------------------------------------------------------------------ item 9: general answers in a documents chat


async def test_a_spoken_general_answer_to_a_fact_question_says_it_isnt_from_the_documents(world, fakes):
    service, chat_id = world
    fakes.llm.route = {"intent": "general_qa", "query": None}
    fakes.llm.reply = "The population of Mumbai is about 21 million."
    fakes.reranker.scorer = lambda q, p: 0.0  # the documents don't match: the general answer stands
    events = await turn(service, chat_id, "How many people live in Mumbai?", modality="voice")
    agent = saved(events)
    assert agent.text == "Not from your documents, but the population of Mumbai is about 21 million."
    assert agent.route["not_from_documents"] is True
    assert "go straight on with the answer" in fakes.llm.calls[-1]["messages"][0].content
    fakes.llm.reply = "EBITDA is earnings before interest, taxes, depreciation and amortisation."
    events = await turn(service, chat_id, "What is EBITDA?", modality="voice")  # a definition: no prefix
    assert not saved(events).text.startswith(NOT_FROM_DOCUMENTS["en"])
    fakes.llm.reply = "The population of Mumbai is about 21 million."
    events = await turn(service, chat_id, "How many people live in Mumbai?")  # typed: the label says it
    assert saved(events).text == "The population of Mumbai is about 21 million."


# ------------------------------------------------------------------ item 10: misheard names


async def test_a_misheard_name_is_asked_and_answered_as_the_documents_spell_it(world, fakes):
    service, chat_id = world
    fakes.llm.reply = "Wall Mora's revenue in FY24 was ₹7,365 crore [S1]."
    events = await turn(service, chat_id, "What was Wall Mora's revenue in FY24?", modality="voice")
    prompt = answer_prompts(fakes.llm)[-1]
    assert "Question: What was Valmora's revenue in FY24?" in prompt
    assert '"Wall Mora" is Valmora' in prompt
    assert saved(events).text == "Valmora's revenue in FY24 was ₹7,365 crore [S1]."
    user = (await MessageService(service.messages.db).list(chat_id)).items[-2]
    assert user.text == "What was Wall Mora's revenue in FY24?"  # what was said stays as it was heard


# ------------------------------------------------------------------ item 7: short answers


async def test_a_short_answer_ends_at_the_sentence_that_reaches_the_word_limit(world, fakes):
    service, chat_id = world
    sentence = (
        "Revenue grew strongly in every quarter of the year on higher volumes, better prices and a richer mix of "
        "specialty products sold to automotive and pharma customers [S1]."
    )  # 28 words
    fakes.llm.reply = " ".join([sentence] * 6)
    fakes.llm.delay = 0.0
    events = await turn(service, chat_id, "How did revenue do in FY24?", modality="voice")
    text = saved(events).text
    words = len(text.replace("[S1]", "").split())
    assert words == 56 >= SHORT_ANSWER_WORDS and text.endswith("[S1].")  # two whole sentences, not three
    assert fakes.llm.closed  # generation stopped there
    assert saved(events).route["checks"][-1]["check"] == "length"
