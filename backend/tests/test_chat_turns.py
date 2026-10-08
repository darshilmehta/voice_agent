"""ChatTurnService without HTTP: stopping mid-answer, voice modality and answer length.

A client that goes away mid-answer stops generation, and the partial answer is saved with route.stopped = true.
Starlette (ASGI spec < 2.4, as uvicorn speaks) cancels the response's anyio task group on http.disconnect; the
cancellation then fires again at every await. The first test reproduces exactly that around the app's SSE generator.
"""

from __future__ import annotations

import anyio
import pytest

from app.api.chats import sse
from app.providers.retrieval import IndexedChunk
from app.services.base import InvalidInput
from app.services.chat_turns import ChatTurnService, DeltaEvent, wait_for_background
from app.services.chats import ChatService
from app.services.messages import MessageService
from app.services.projects import ProjectService
from app.services.prompts import ANSWER_LENGTHS
from app.services.retrieval import RetrievalService

from .conftest import add_document
from .fakes import make_chunk, vector_for

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

    assert fakes.llm.closed and fakes.llm.sent == 3  # generation cancelled right away
    user, agent = await transcript(db, chat_id)
    assert user.role == "user" and agent.role == "agent"
    assert agent.text == "Margin 18.2% [S1]"  # the three pieces that were streamed: "Margin", " 18.2%", " [S1] "
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
    assert agent.text == "Margin 18.2%" and agent.citations == []
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
