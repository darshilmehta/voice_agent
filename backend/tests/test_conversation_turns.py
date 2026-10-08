"""Phase 3 + 7 through ``ChatTurnService`` (no HTTP, fake retrieval, a scripted LLM): each kind of turn, corrections
from what was heard, the conversation state across a restart, the memory summary and when it runs, the language
rules, and a scripted drift conversation: document → general → Hindi → back to the document."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from app.providers.llm import LLMMessage
from app.providers.retrieval import IndexedChunk
from app.services.chat_turns import (
    AgentMessageEvent,
    AnswerStop,
    ChatTurnService,
    DeltaEvent,
    SourcesEvent,
    wait_for_background,
)
from app.services.chats import ChatService
from app.services.conversation import ConversationStateService, StateWrites
from app.services.memory import ChatActivity, chat_activity
from app.services.messages import MessageService
from app.services.projects import ProjectService
from app.services.prompts import ACK_TEXTS
from app.services.retrieval import RetrievalService
from app.services.summaries import SummaryService

from .conftest import add_document
from .fakes import keyword_scorer, make_chunk, vector_for

PASSAGES = [  # (text, page)
    ("EBITDA margin was 18.2% in FY24.", 2),
    ("EBITDA margin was 16.9% in FY23.", 2),
    ("Revenue grew 34% in FY24 on the enterprise segment.", 1),
    ("Net debt fell to 310 crore in FY24.", 3),
]
MEMORY = "- The user asked about the EBITDA margin: 18.2% in FY24 (annual_report.pdf p.2)."


# ------------------------------------------------------------------ a deterministic model


RERANKED: list[str] = []  # the reranker's last query (the English one for Hindi turns), set by the fixture


def scripted_reply(messages: list[LLMMessage]) -> str:
    """Answers by prompt kind: grounded answers quote and cite the source that best matches what was searched."""
    system = messages[0].content
    hindi = "in Hindi" in system
    if system.startswith("You keep the memory"):
        return MEMORY
    if system.startswith("You answer questions about the user's documents"):
        prompt = messages[-1].content
        question = RERANKED[-1] if RERANKED else prompt.split("Question: ", 1)[1].split("\n", 1)[0]
        sources = re.findall(r"\[(S\d+)\][^\n]*\n([^\n]+)", prompt)
        sid, text = max(sources, key=lambda s: keyword_scorer(question, s[1]))
        return f"{'दस्तावेज़ के अनुसार: ' if hindi else ''}{text} [{sid}]"
    if "Ask exactly one short question" in system:
        return "आप किस वर्ष की बात कर रहे हैं?" if hindi else "Which year do you mean?"
    if system.startswith("You are a friendly voice assistant"):
        return "नमस्ते!" if hindi else "Happy to help."
    return "सामान्य जानकारी: नई दिल्ली।" if hindi else "From general knowledge: Paris."


def routes(table: dict[str, dict[str, Any]]) -> Callable[[list[LLMMessage]], dict[str, Any] | None]:
    """The router model's proposals by utterance (anything else: no route → the turn falls back)."""

    def route(messages: list[LLMMessage]) -> dict[str, Any] | None:
        utterance = messages[-1].content.rsplit("Utterance: ", 1)[-1]
        return table.get(utterance)

    return route


@pytest.fixture
async def world(db, load_local, fakes):
    settings = load_local()
    project = await ProjectService(db).create("Annual report FY24")
    doc_id = await add_document(
        db, project.id, filename="annual_report.pdf", status="READY", page_count=3, chunk_count=len(PASSAGES)
    )
    for i, (text, page) in enumerate(PASSAGES):
        chunk = make_chunk(i, project_id=project.id, document_id=doc_id, text=text, page_start=page, page_end=page)
        await fakes.store.upsert([IndexedChunk(chunk, vector_for(text))])
    chat = await ChatService(db).create(project.id)
    retrieval = RetrievalService(fakes.embedder, fakes.reranker, fakes.store, settings.retrieval)
    RERANKED.clear()

    def scorer(query: str, passage: str) -> float:
        if not RERANKED or RERANKED[-1] != query:
            RERANKED.append(query)
        return keyword_scorer(query, passage)

    fakes.reranker.scorer = scorer
    fakes.llm.reply = scripted_reply
    fakes.llm.route = routes({})
    service = ChatTurnService(db, retrieval=retrieval, llm=fakes.llm, settings=settings)
    return SimpleNamespace(
        db=db, fakes=fakes, settings=settings, retrieval=retrieval, service=service, chat_id=chat.id, doc_id=doc_id
    )


async def say(world, text: str, **kw: Any) -> tuple[list[Any], Any]:
    turn = await world.service.begin(world.chat_id, text, **kw)
    events = [e async for e in world.service.run(turn)]
    await wait_for_background()
    assert isinstance(events[-1], AgentMessageEvent), events[-1]
    return events, events[-1].message


def answer_calls(world) -> list[dict[str, Any]]:
    """The model's answer calls (not memory summaries)."""
    return [c for c in world.fakes.llm.calls if not c["messages"][0].content.startswith("You keep the memory")]


def text_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, DeltaEvent))


def sources_of(events) -> SourcesEvent:
    (s,) = [e for e in events if isinstance(e, SourcesEvent)]
    return s


# ------------------------------------------------------------------ each kind of turn


async def test_a_document_question_is_answered_from_the_documents(world):
    world.fakes.llm.route = routes({"What was the EBITDA margin in FY24?": {"intent": "document_qa", "query": None}})
    events, agent = await say(world, "What was the EBITDA margin in FY24?")
    assert [e.name for e in events][:2] == ["user_message", "sources"] and not sources_of(events).abstained
    assert agent.text == "EBITDA margin was 18.2% in FY24. [S1]" and [c.page_start for c in agent.citations] == [2]
    r = agent.route
    assert (r["intent"], r["needs_retrieval"], r["answer"], r["abstained"], r["stopped"]) == (
        "document_qa",
        True,
        "grounded",
        False,
        False,
    )
    assert (r["topic"], r["rewritten_query"], r["query_en"], r["language"]) == ("ebitda margin", None, None, "en")
    assert r["router"]["source"] == "llm" and r["speculation"] == "used" and r["prompt"] == "answer-v1"
    assert agent.latency["router_ms"] is not None and agent.latency["router_llm_ms"] is not None


async def test_an_unrelated_question_skips_retrieval_and_says_so(world):
    world.fakes.llm.route = routes({"What's the capital of France?": {"intent": "general_qa", "query": None}})
    events, agent = await say(world, "What's the capital of France?")
    s = sources_of(events)
    assert s.sources == [] and not s.abstained  # skipped on purpose: not an abstention
    assert agent.text == "From general knowledge: Paris." and agent.citations == []
    r = agent.route
    assert (r["intent"], r["needs_retrieval"], r["answer"], r["abstained"], r["prompt"]) == (
        "general_qa",
        False,
        "general",
        False,
        "general-v1",
    )
    assert r["speculation"] == "discarded" and r["sources"] == 0 and r["candidates"] == 0
    (call,) = answer_calls(world)
    system = call["messages"][0].content
    assert "not about the user's documents" in system and "Never claim" in system and "[S1]" in system


async def test_acknowledgements_and_thanks_get_short_fixed_replies_never_an_abstention(world):
    _, first = await say(world, "okay")
    assert (first.role, first.text, first.route["intent"], first.route["answer"]) == (
        "agent",
        ACK_TEXTS["ack"]["en"],
        "backchannel",
        "ack",
    )
    events, second = await say(world, "okay")  # right after "Anything else?": nothing more is said
    assert (second.role, second.text, text_of(events)) == ("event", "Acknowledged", "")
    _, thanks = await say(world, "theek hai, shukriya")  # Hinglish thanks: the router isn't needed
    assert thanks.text == ACK_TEXTS["thanks"]["hi"] and thanks.language == "hi"
    for m in (first, second, thanks):
        assert m.route["abstained"] is False and m.route["needs_retrieval"] is False
    assert answer_calls(world) == [] and world.fakes.llm.json_calls == []  # no model at all
    assert world.fakes.reranker.calls == []  # nothing searched


async def test_stop_says_nothing(world):
    events, notice = await say(world, "stop")
    assert [e.name for e in events] == ["user_message", "sources", "agent_message"]
    assert (notice.role, notice.text, notice.route["intent"], notice.route["stopped"]) == (
        "event",
        "Stopped",
        "stop",
        False,
    )
    roles = [m.role for m in (await MessageService(world.db).list(world.chat_id)).items]
    assert roles == ["user", "event"]


async def test_an_unclear_request_gets_a_question_back(world):
    world.fakes.llm.route = routes({"What about that one?": {"intent": "clarification", "query": None}})
    _, agent = await say(world, "What about that one?")
    assert agent.text == "Which year do you mean?" and agent.route["answer"] == "clarification"
    assert answer_calls(world)[-1]["max_tokens"] == 96
    _, reply = await say(world, "Hello there, who are you?")  # a conversation turn via the model (fallback here)
    assert reply.route["intent"] == "document_qa"  # no route scripted: phase-1 fallback, abstains honestly
    assert reply.route["router"]["source"] == "fallback" and reply.route["abstained"] is True


async def test_mixed_questions_cite_documents_or_fall_back_to_general_knowledge(world):
    world.fakes.llm.route = routes(
        {
            "Is the FY24 EBITDA margin good?": {"intent": "mixed", "query": None},
            "Is that typical for the sector?": {"intent": "mixed", "query": "Is that typical for the sector?"},
        }
    )
    _, agent = await say(world, "Is the FY24 EBITDA margin good?")
    assert agent.route["answer"] == "mixed" and agent.citations and agent.route["prompt"] == "mixed-v1"
    assert "general knowledge" in answer_calls(world)[-1]["messages"][0].content
    events, agent = await say(world, "Is that typical for the sector?")  # nothing in the documents
    assert agent.route["answer"] == "general" and agent.route["general_note"] == "not_covered"
    assert agent.route["abstained"] is False and not sources_of(events).abstained
    assert "don't cover this" in answer_calls(world)[-1]["messages"][0].content


async def test_resume_returns_to_the_document_topic_without_the_model(world):
    world.fakes.llm.route = routes(
        {
            "What was the EBITDA margin in FY24?": {"intent": "document_qa", "query": None},
            "What's the capital of France?": {"intent": "general_qa", "query": None},
            # §9.5: the model once took this for a backchannel; validation makes it a return to the documents
            "Let's go back to the annual report": {"intent": "backchannel", "query": None},
        }
    )
    await say(world, "What was the EBITDA margin in FY24?")
    await say(world, "What's the capital of France?")
    calls = len(answer_calls(world))
    _, agent = await say(world, "Let's go back to the annual report")
    assert (
        agent.text
        == "Sure, back to the annual report. We were talking about ebitda margin. What would you like to know?"
    )
    assert len(answer_calls(world)) == calls  # a fixed text, never "I can't access documents"
    r = agent.route
    assert (r["intent"], r["answer"], r["topic"], r["is_topic_shift"]) == (
        "resume_document",
        "resume",
        "ebitda margin",
        True,
    )
    assert r["router"]["overrides"] == ["backchannel→resume_document: resume phrase"]
    state = await ConversationStateService(world.db).get(world.chat_id)
    assert (state.active_topic, state.previous_topic) == ("ebitda margin", "capital france")


async def test_a_correction_after_a_barge_in_uses_what_was_heard(world):
    world.fakes.llm.route = routes(
        {"No, I meant FY23": {"intent": "correction", "query": "What was the EBITDA margin in FY23?"}}
    )
    world.fakes.llm.delay = 0.01
    turn = await world.service.begin(world.chat_id, "What was the EBITDA margin in FY24?", modality="voice")
    stop = AnswerStop()
    stop.heard_text, stop.reason = "EBITDA margin", "barge_in"

    async def consume() -> None:
        async with contextlib.aclosing(world.service.run(turn, stop=stop)) as events:
            async for event in events:
                if isinstance(event, DeltaEvent):
                    started.set()

    started = asyncio.Event()
    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    await asyncio.wait([task])
    await wait_for_background()
    assert stop.saved is not None and stop.saved.heard_text == "EBITDA margin"
    state = await ConversationStateService(world.db).get(world.chat_id)
    assert state.last_interrupted_message_id == stop.saved.id

    world.fakes.llm.delay = 0.0
    _, agent = await say(world, "No, I meant FY23", modality="voice")
    router_prompt = world.fakes.llm.json_calls[-1]["messages"][-1].content
    assert (
        "they heard only" not in router_prompt
        and '(The user cut that answer off after hearing: "EBITDA margin")' in router_prompt
    )
    assert "Assistant: EBITDA margin" in router_prompt  # the history shows what was heard, not the whole answer
    assert world.fakes.reranker.calls[-1][0] == "What was the EBITDA margin in FY23?"
    assert agent.text == "EBITDA margin was 16.9% in FY23. [S2]"
    assert (agent.route["intent"], agent.route["rewritten_query"]) == (
        "correction",
        "What was the EBITDA margin in FY23?",
    )
    state = await ConversationStateService(world.db).get(world.chat_id)
    assert state.last_interrupted_message_id is None  # answered


# ------------------------------------------------------------------ state, memory, languages


async def test_the_conversation_state_survives_a_restart(world):
    world.fakes.llm.route = routes({"What was the EBITDA margin in FY24?": {"intent": "document_qa", "query": None}})
    await say(world, "What was the EBITDA margin in FY24?")
    await say(world, "Please answer in Hindi")  # a request: it sticks
    restarted = ChatTurnService(world.db, retrieval=world.retrieval, llm=world.fakes.llm, settings=world.settings)
    state = await restarted.states.get(world.chat_id)
    assert (state.document_topic, state.active_document_ids, state.preferred_language) == (
        "ebitda margin",
        [world.doc_id],
        "hi",
    )
    assert (await restarted.begin(world.chat_id, "What about the debt?")).language == "hi"
    await ConversationStateService(world.db).set_retrieval_enabled(world.chat_id, False)
    _, agent = await say(world, "What does the report say about debt?")
    assert agent.route["answer"] == "general" and agent.route["general_note"] == "retrieval_off"


async def test_hinglish_hindi_and_switching_languages(world):
    world.fakes.llm.route = routes(
        {
            "FY24 mein revenue kitna tha?": {"intent": "document_qa", "query": "How much was revenue in FY24?"},
            "वित्त वर्ष 2024 में कर्ज कितना था?": {"intent": "document_qa", "query": "What was the net debt in FY24?"},
        }
    )
    _, agent = await say(world, "FY24 mein revenue kitna tha?")
    assert agent.language == "hi" and agent.route["input_language"] == "hi"
    assert agent.route["query_en"] == "How much was revenue in FY24?" and agent.route["speculation"] == "reused_search"
    assert world.fakes.reranker.calls[-1][0] == "How much was revenue in FY24?"  # the English query is scored
    assert agent.text.startswith("दस्तावेज़ के अनुसार") and [c.source_id for c in agent.citations] == ["S3"]
    assert "Answer in Hindi" in answer_calls(world)[-1]["messages"][0].content
    json_calls = len(world.fakes.llm.json_calls)
    _, english = await say(world, "Now answer in English please")  # asked: English from now on
    assert english.language == "en" and len(world.fakes.llm.json_calls) == json_calls  # no router call
    r = english.route  # the previous question again, in English, from the documents
    assert (r["intent"], r["answer"], r["rewritten_query"], r["query_en"]) == (
        "correction",
        "grounded",
        "FY24 mein revenue kitna tha?",
        "How much was revenue in FY24?",
    )
    assert r["router"]["overrides"] == ["language request: the previous question again"]
    assert r["speculation"] == "reused_search" and english.text.startswith("Revenue grew 34%")
    _, agent = await say(world, "वित्त वर्ष 2024 में कर्ज कितना था?")
    assert agent.language == "en" and agent.citations  # the user's request beats the utterance's language
    _, agent = await say(world, "हिंदी में बताइए")  # asked again
    assert agent.language == "hi"


async def test_the_memory_summary_is_refreshed_in_the_background_and_used(world):
    world.fakes.llm.route = routes({"What was the EBITDA margin in FY24?": {"intent": "document_qa", "query": None}})
    for _ in range(3):  # 3 exchanges = 6 messages: all still in the prompt's window, no summary needed
        await say(world, "What was the EBITDA margin in FY24?")
    assert await SummaryService(world.db).get(world.chat_id, "memory") is None
    await say(world, "What was the EBITDA margin in FY24?")  # 8 messages: older ones would drop out of the window
    summary = await SummaryService(world.db).get(world.chat_id, "memory")
    assert summary is not None and summary.content == MEMORY and summary.data["messages"] == 8
    memory_call = next(c for c in world.fakes.llm.calls if c["messages"][0].content.startswith("You keep the memory"))
    assert "User: What was the EBITDA margin in FY24?" in memory_call["messages"][1].content
    assert "(sources: annual_report.pdf p.2)" in memory_call["messages"][1].content

    _, agent = await say(world, "What was the EBITDA margin in FY24?")
    system = answer_calls(world)[-1]["messages"][0].content
    assert f"Earlier in this conversation (summary; the recent messages follow):\n{MEMORY}" in system
    assert agent.route["memory"] is True
    assert len(answer_calls(world)[-1]["messages"]) == 1 + 6 + 1  # system, the recent window, the question


async def test_memory_summaries_never_run_while_a_turn_is_answering():
    activity = ChatActivity()
    ran: list[str] = []
    release = asyncio.Event()

    async def summary() -> None:
        ran.append("start")
        await release.wait()
        ran.append("done")

    activity.turn_started("a")
    activity.turn_started("b")
    activity.turn_finished("a", summary)
    await asyncio.sleep(0)
    assert ran == [] and activity.jobs == {} and set(activity.waiting) == {"a"}  # b is still answering: queued
    activity.turn_finished("b", summary)
    await asyncio.sleep(0)
    assert ran == ["start", "start"] and set(activity.jobs) == {"a", "b"} and activity.waiting == {}
    activity.turn_started("b")  # the next turn: the summaries give way
    await asyncio.sleep(0)
    await activity.idle()
    assert ran == ["start", "start"] and activity.jobs == {}  # cancelled, never finished


async def test_state_writes_are_applied_in_order_and_awaited_by_readers():
    writes = StateWrites()
    done: list[str] = []
    gate = asyncio.Event()

    async def slow() -> None:
        await gate.wait()
        done.append("first")

    async def fast() -> None:
        done.append("second")

    writes.schedule("c", slow, spawn=asyncio.ensure_future)
    writes.schedule("c", fast, spawn=asyncio.ensure_future)
    await asyncio.sleep(0.01)
    assert done == []  # the second waits for the first
    reader = asyncio.ensure_future(writes.settled("c"))
    await asyncio.sleep(0.01)
    assert not reader.done()  # a reader waits for both
    gate.set()
    await reader
    assert done == ["first", "second"] and writes.pending == {}


async def test_a_turn_stopped_while_routing_records_no_state(world):
    world.fakes.llm.json_delay = 5.0  # the router is still thinking
    world.fakes.reranker.scorer = lambda q, passage: 0.1  # and retrieval can't settle the route alone
    turn = await world.service.begin(world.chat_id, "And what about the margin?", modality="voice")
    stop = AnswerStop(heard_text="", reason="stop")

    async def consume() -> None:
        async with contextlib.aclosing(world.service.run(turn, stop=stop)) as events:
            async for event in events:
                if event.name == "user_message":
                    started.set()

    started = asyncio.Event()
    task = asyncio.create_task(consume())
    await started.wait()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.wait([task])
    await wait_for_background()
    assert world.fakes.llm.json_cancelled == 1  # the router call was cancelled with the turn
    assert stop.saved is not None and stop.saved.route["stopped"] is True
    state = await ConversationStateService(world.db).get(world.chat_id)
    assert state.updated_at is None and state.last_intent is None  # no row: nothing was decided, nothing recorded


async def test_a_summary_cancelled_before_it_started_doesnt_block_the_next_one():
    activity = ChatActivity()
    ran: list[str] = []

    def summary(chat: str) -> Callable[[], Any]:
        async def job() -> None:
            ran.append(chat)

        return job

    activity.turn_started("x")
    activity.turn_finished("x", summary("x"))  # spawned, not started yet…
    activity.turn_started("y")  # …and cancelled in the same loop iteration by another chat's turn
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert ran == [] and activity.jobs == {}  # its entry is gone even though its coroutine never ran
    activity.turn_finished("y", summary("y"))
    await activity.idle()
    activity.turn_started("x")
    activity.turn_finished("x", summary("x"))
    await activity.idle()
    assert ran == ["y", "x"]  # x's summary runs again


async def test_a_turn_cancels_a_running_memory_summary(world):
    llm = world.fakes.llm
    llm.route = routes({"What was the EBITDA margin in FY24?": {"intent": "document_qa", "query": None}})
    stall = True

    def reply(messages: list[LLMMessage]) -> str:
        is_memory = messages[0].content.startswith("You keep the memory")
        llm.hold_after = 1 if is_memory and stall else None  # the summary's stream stalls after one piece
        return scripted_reply(messages)

    llm.reply = reply
    for _ in range(3):
        await say(world, "What was the EBITDA margin in FY24?")
    turn = await world.service.begin(world.chat_id, "What was the EBITDA margin in FY24?")
    [e async for e in world.service.run(turn)]  # its answer completes; then the summary starts and stalls
    activity = chat_activity()
    for _ in range(100):
        if any(c["messages"][0].content.startswith("You keep the memory") for c in llm.calls):
            break
        await asyncio.sleep(0.01)
    assert activity.jobs  # the summary is generating

    stall = False
    turn = await world.service.begin(world.chat_id, "What was the EBITDA margin in FY24?")
    events = world.service.run(turn)
    assert (await anext(events)).name == "user_message"  # the next turn has started…
    await asyncio.sleep(0.05)
    assert llm.closed and not activity.jobs  # …and the summary's generation was cancelled
    assert await SummaryService(world.db).get(world.chat_id, "memory") is None
    rest = [e async for e in events]
    assert rest[-1].name == "agent_message"
    await wait_for_background()  # the summary runs again after this turn, now that the model is free
    assert (await SummaryService(world.db).get(world.chat_id, "memory")).content == MEMORY


# ------------------------------------------------------------------ drift: document → general → Hindi → document


async def test_drift_document_general_hindi_and_back(world):
    world.fakes.llm.route = routes(
        {
            "What was the EBITDA margin in FY24?": {"intent": "document_qa", "query": None},
            "By the way, what's the capital of France?": {"intent": "general_qa", "query": None},
            "वित्त वर्ष 2024 में राजस्व कितना बढ़ा?": {
                "intent": "document_qa",
                "query": "How much did revenue grow in FY24?",
            },
            "भारत की राजधानी क्या है?": {"intent": "general_qa", "query": "What is the capital of India?"},
            "Okay, let's go back to the annual report.": {"intent": "resume_document", "query": None},
            "And what about the margin in FY23?": {
                "intent": "document_qa",
                "query": "What was the EBITDA margin in FY23?",
            },
        }
    )
    script = [
        ("What was the EBITDA margin in FY24?", "document_qa", "en", ["S1"]),
        ("By the way, what's the capital of France?", "general_qa", "en", []),
        ("वित्त वर्ष 2024 में राजस्व कितना बढ़ा?", "document_qa", "hi", ["S3"]),
        ("भारत की राजधानी क्या है?", "general_qa", "hi", []),
        ("Okay, let's go back to the annual report.", "resume_document", "en", []),
        ("And what about the margin in FY23?", "document_qa", "en", ["S2"]),
    ]
    transcript = []
    for text, intent, language, cited in script:
        events, agent = await say(world, text)
        r = agent.route
        transcript.append((r["intent"], agent.language, [c.source_id for c in agent.citations], r["abstained"]))
        assert (r["intent"], agent.language, transcript[-1][2]) == (intent, language, cited), text
        assert r["needs_retrieval"] == bool(cited) and sources_of(events).abstained is False
    assert all(not abstained for *_, abstained in transcript)

    messages = (await MessageService(world.db).list(world.chat_id)).items
    resume = messages[9]
    assert resume.text == "Sure, back to the annual report. We were talking about revenue. What would you like to know?"
    hindi = messages[5]
    assert hindi.route["query_en"] == "How much did revenue grow in FY24?" and hindi.text.startswith("दस्तावेज़")
    assert [m.route["is_topic_shift"] for m in messages if m.role == "agent"] == [False, True, True, True, True, True]
    state = await ConversationStateService(world.db).get(world.chat_id)
    assert (state.active_topic, state.document_topic, state.document_query) == (
        "ebitda margin",
        "ebitda margin",
        "What was the EBITDA margin in FY23?",
    )
    assert (state.input_language, state.response_language, state.last_intent) == ("en", "en", "document_qa")
