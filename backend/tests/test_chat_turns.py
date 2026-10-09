"""ChatTurnService without HTTP: stopping mid-answer, voice modality and answer length.

A client that goes away mid-answer stops generation, and the partial answer is saved with route.stopped = true.
Starlette (ASGI spec < 2.4, as uvicorn speaks) cancels the response's anyio task group on http.disconnect; the
cancellation then fires again at every await. The first test reproduces exactly that around the app's SSE generator.
"""

from __future__ import annotations

import asyncio
import contextlib

import anyio
import pytest

from app.api.chats import sse
from app.providers.retrieval import IndexedChunk
from app.services.base import InvalidInput
from app.services.chat_turns import STRONG_PASSAGE, AnswerStop, ChatTurnService, DeltaEvent, wait_for_background
from app.services.chats import ChatService
from app.services.messages import MessageService
from app.services.projects import ProjectService
from app.services.prompts import ANSWER_LENGTHS
from app.services.retrieval import RetrievalService

from .conftest import add_document
from .fakes import keyword_scorer, make_chunk, vector_for

REPLY = "Margin 18.2% [S1] while revenue grew 34% [S2] on a strong enterprise segment."  # 6-char pieces


@pytest.fixture
async def setup(db, load_local, fakes):
    """A project with one READY document indexed in the fake store, a chat, and the pipeline on fakes."""
    settings = load_local()
    project = await ProjectService(db).create("Annual report FY24")
    doc_id = await add_document(db, project.id, status="READY", page_count=3, chunk_count=2)
    for i, text in enumerate(["EBITDA margin was 18.2% in FY24.", "Revenue grew 34% in FY24, EBITDA margin up."]):
        chunk = make_chunk(i, project_id=project.id, document_id=doc_id, text=text, page_start=2, page_end=2)
        await fakes.store.upsert([IndexedChunk(chunk, vector_for(text))])
    chat = await ChatService(db).create(project.id)
    # Passages that answer, but not "strong" ones (reranker < STRONG_PASSAGE): the answer streams as the model writes
    # it, which is what these tests stop. (Over strong passages the answer's checks hold its sentences until each is
    # checked: tests/test_answer_checks.py.)
    fakes.reranker.scorer = lambda q, p: min(keyword_scorer(q, p), STRONG_PASSAGE - 0.05)
    retrieval = RetrievalService(fakes.embedder, fakes.reranker, fakes.store, settings.retrieval)
    pipeline = ChatTurnService(db, retrieval=retrieval, llm=fakes.llm, settings=settings)
    fakes.llm.reply = REPLY
    fakes.llm.delay = 0.01
    return pipeline, chat.id


async def transcript(db, chat_id):
    return (await MessageService(db).list(chat_id)).items


async def test_disconnect_mid_answer_stops_generation_and_saves_the_partial_answer(setup, db, fakes):
    pipeline, chat_id = setup
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?")
    received: list[bytes] = []

    async def client(scope: anyio.CancelScope) -> None:
        async for chunk in sse(pipeline.run(turn)):
            received.append(chunk)
            if sum(c.startswith(b"event: delta") for c in received) == 3:
                scope.cancel()  # what Starlette does on http.disconnect

    async with anyio.create_task_group() as tg:
        tg.start_soon(client, tg.cancel_scope)
    await wait_for_background()

    # generation cancelled right away: 6 pieces for 3 deltas, the first 4 sent together once they showed the answer
    # is in English (B5)
    assert fakes.llm.closed and fakes.llm.sent == 6
    user, agent = await transcript(db, chat_id)
    assert user.role == "user" and agent.role == "agent"
    assert agent.text == "Margin 18.2% [S1] while revenue grew"  # what was streamed
    assert [c.source_id for c in agent.citations] == ["S1"]  # [S2] was never generated
    assert agent.route["stopped"] is True and agent.route["abstained"] is False
    assert agent.latency["llm_ms"] is not None


async def test_closing_the_event_stream_saves_the_partial_answer(setup, db, fakes):
    pipeline, chat_id = setup
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?")
    events = pipeline.run(turn)
    deltas = 0
    async for event in events:
        deltas += event.name == "delta"
        if deltas == 2:
            break
    await events.aclose()  # e.g. the response being closed instead of cancelled
    await wait_for_background()
    assert fakes.llm.closed
    agent = (await transcript(db, chat_id))[-1]
    # two deltas: the first words, held until they showed the answer is in English (B5), then "revenu"
    assert agent.text == "Margin 18.2% [S1] while revenu" and [c.source_id for c in agent.citations] == ["S1"]
    assert agent.route["stopped"] is True


async def test_a_marker_cut_in_half_is_dropped(setup, db, fakes):
    pipeline, chat_id = setup
    fakes.llm.reply = "Margin was 18.2% [S1]"
    fakes.llm.piece_chars = 19  # "Margin was 18.2% [S" | "1]"
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?")
    events = pipeline.run(turn)
    async for event in events:
        if event.name == "delta":
            break
    await events.aclose()
    await wait_for_background()
    agent = (await transcript(db, chat_id))[-1]
    assert agent.text == "Margin was 18.2%" and agent.citations == []


async def test_disconnect_before_any_text_saves_no_answer(setup, db, fakes):
    pipeline, chat_id = setup
    fakes.llm.delay = 5.0  # the model is still reading the prompt when the client leaves
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?")
    seen: list[bytes] = []

    async def client(scope: anyio.CancelScope) -> None:
        async for chunk in sse(pipeline.run(turn)):
            seen.append(chunk)
            if chunk.startswith(b"event: sources"):
                scope.cancel()

    with anyio.fail_after(2):
        async with anyio.create_task_group() as tg:
            tg.start_soon(client, tg.cancel_scope)
    await wait_for_background()
    assert [m.role for m in await transcript(db, chat_id)] == ["user"]  # the user message stays, no answer


async def test_completed_and_abstained_answers_are_not_stopped(setup, db, fakes):
    pipeline, chat_id = setup
    fakes.llm.delay = 0.0
    for question in ("What was the EBITDA margin in FY24?", "Who audits the accounts?"):
        turn = await pipeline.begin(chat_id, question)
        events = [e async for e in pipeline.run(turn)]
        assert events[-1].name == "agent_message"
    answered, abstained = (m for m in await transcript(db, chat_id) if m.role == "agent")
    assert (answered.route["stopped"], answered.route["abstained"]) == (False, False)
    assert (abstained.route["stopped"], abstained.route["abstained"]) == (False, True)


async def test_voice_turns_and_answer_length_are_parameters_of_the_service(setup, db, fakes):
    pipeline, chat_id = setup
    fakes.llm.delay = 0.0
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?", modality="voice")
    events = [e async for e in pipeline.run(turn)]
    assert [e.name for e in events][:2] == ["user_message", "sources"]
    deltas = [e.text for e in events if isinstance(e, DeltaEvent)]  # what the voice loop feeds to TTS
    assert "".join(deltas) == REPLY
    user, agent = await transcript(db, chat_id)
    assert (user.modality, agent.modality, agent.route["length"]) == ("voice", "voice", "short")
    short_call = fakes.llm.calls[-1]
    assert short_call["max_tokens"] == ANSWER_LENGTHS["short"].max_tokens
    assert ANSWER_LENGTHS["short"].instruction in short_call["messages"][0].content

    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?", length="full")
    assert [e.name async for e in pipeline.run(turn)][-1] == "agent_message"
    full_call = fakes.llm.calls[-1]
    assert full_call["max_tokens"] == ANSWER_LENGTHS["full"].max_tokens > ANSWER_LENGTHS["short"].max_tokens
    assert ANSWER_LENGTHS["full"].instruction in full_call["messages"][0].content
    with pytest.raises(InvalidInput, match="length"):
        await pipeline.begin(chat_id, "hi", length="essay")  # type: ignore[arg-type]


# ------------------------------------------------------------------ voice: what was heard


async def _stop_after(pipeline, turn, stop, *, deltas: int) -> None:
    """Consume the turn until ``deltas`` deltas arrived (0: right after the user message), then cancel it."""
    seen = 0

    async def consume() -> None:
        nonlocal seen
        async with contextlib.aclosing(pipeline.run(turn, stop=stop)) as events:
            async for event in events:
                seen += event.name == "delta"
                if event.name == ("delta" if deltas else "user_message") and seen >= deltas:
                    started.set()

    started = asyncio.Event()
    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    await asyncio.wait([task])


async def test_a_stopped_voice_answer_is_saved_with_what_was_heard(setup, db, fakes):
    pipeline, chat_id = setup
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?", modality="voice")
    stop = AnswerStop()
    stop.heard_text, stop.reason = "Margin", "barge_in"
    await _stop_after(pipeline, turn, stop, deltas=3)
    assert stop.saved is not None and stop.user is not None and stop.completed is None
    user, agent = await transcript(db, chat_id)
    assert stop.user == user and stop.saved == agent
    assert agent.text == "Margin 18.2% [S1] while revenue grew" and agent.heard_text == "Margin" and agent.interrupted
    assert agent.route["stopped"] is True and agent.route["interrupted"] == "barge_in"


async def test_a_voice_answer_stopped_before_any_text_is_saved_empty(setup, db, fakes):
    pipeline, chat_id = setup
    fakes.llm.hold_after = 0  # still "thinking"
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?", modality="voice")
    stop = AnswerStop(heard_text="", reason="stop")
    await _stop_after(pipeline, turn, stop, deltas=0)
    _, agent = await transcript(db, chat_id)
    assert stop.saved == agent and (agent.text, agent.heard_text, agent.citations) == ("", "", [])
    assert agent.route["stopped"] is True and agent.route["interrupted"] == "stop"
    assert agent.latency["first_delta_ms"] is None and agent.latency["llm_ms"] is None  # stopped before retrieval


async def test_voice_spoken_language_and_stt_timings_are_saved_on_the_user_message(setup, db, fakes):
    pipeline, chat_id = setup
    fakes.llm.delay = 0.0
    stt = {"speech_ms": 2100, "stt_ms": 410.5, "speculative_stt": True}
    turn = await pipeline.begin(
        chat_id, "kya EBITDA margin badha?", modality="voice", language="hi", input_language="hi", input_latency=stt
    )
    [e async for e in pipeline.run(turn)]
    user, agent = await transcript(db, chat_id)
    assert (user.language, user.latency) == ("hi", stt)  # Hinglish, spoken as Hindi
    assert agent.route["language"] == "hi" and "Answer in Hindi" in fakes.llm.calls[0]["messages"][-1].content
    assert agent.language == "en"  # the fake answered in English (twice: B5): saved as what it is


async def test_an_answer_nobody_heard_is_left_out_of_the_next_prompt(setup, db, fakes):
    """Review item 12: heard_text "" (cut before any audio played) means nothing was heard, not "not interrupted"."""
    pipeline, chat_id = setup
    fakes.llm.delay = 0.0
    messages = MessageService(db)
    await messages.append(chat_id, role="user", text="First question?", modality="voice")
    await messages.append(chat_id, role="agent", text="An answer nobody heard [S1].", modality="voice", heard_text="")
    await messages.append(chat_id, role="user", text="Second question?", modality="voice")
    await messages.append(chat_id, role="agent", text="Partly heard answer.", modality="voice", heard_text="Partly")
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?", modality="voice")
    [e async for e in pipeline.run(turn)]
    history = [(m.role, m.content) for m in fakes.llm.calls[-1]["messages"][1:-1]]
    assert history == [("user", "First question?"), ("user", "Second question?"), ("assistant", "Partly")]


async def test_the_caller_can_save_the_user_message_first(setup, db, fakes):
    """Review item 2: the voice session saves the user message before the answer task, so a stop can't lose it."""
    pipeline, chat_id = setup
    fakes.llm.delay = 0.0
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?", modality="voice")
    user = await pipeline.save_user_message(turn)
    stop = AnswerStop()
    events = [e async for e in pipeline.run(turn, stop=stop, user=user)]
    assert events[0].name == "user_message" and events[0].message == user and stop.user == user
    assert [m.role for m in await transcript(db, chat_id)] == ["user", "agent"]  # saved once, not twice


async def test_a_stop_during_the_final_save_waits_for_it_and_hands_the_answer_back(setup, db, fakes, monkeypatch):
    """Review item 3: the complete answer's save can't be rolled back by a stop; the stop gets it as ``completed``."""
    pipeline, chat_id = setup
    fakes.llm.delay = 0.0
    original = MessageService.append
    saving = asyncio.Event()

    async def slow_append(self, chat_id_, **kw):
        if kw.get("role") == "agent":
            saving.set()
            await asyncio.sleep(0.2)
        return await original(self, chat_id_, **kw)

    monkeypatch.setattr(MessageService, "append", slow_append)
    turn = await pipeline.begin(chat_id, "What was the EBITDA margin in FY24?", modality="voice")
    stop = AnswerStop(heard_text="Margin", reason="stop")

    async def consume() -> None:
        async with contextlib.aclosing(pipeline.run(turn, stop=stop)) as events:
            async for _ in events:
                pass

    task = asyncio.create_task(consume())
    await saving.wait()
    task.cancel()
    await asyncio.wait([task])
    assert stop.saved is None and stop.completed is not None and stop.completed.text == REPLY
    _, agent = await transcript(db, chat_id)
    assert agent == stop.completed  # saved once, complete (the session then records what was heard)
