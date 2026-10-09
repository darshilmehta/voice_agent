"""Chats, their transcripts, and new messages answered as a Server-Sent Events stream (docs/DESIGN.md §3.9)."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, StringConstraints

from ..domain.projects import Chat, MessagePage
from ..services.chat_turns import TEXT_MAX, ChatEvent
from ..services.messages import PAGE_DEFAULT, PAGE_MAX
from ..settings import Language
from .deps import Chats, ChatTurns, Messages
from .schemas import Body, Name, Patch

router = APIRouter(tags=["chats"])


class ChatCreate(Body):
    title: Name | None = None  # omitted: "New chat", replaced by an automatic title later
    document_scope: list[str] | None = None  # omitted or null: all of the project's documents
    language: Language | None = None


class ChatUpdate(Patch):
    NOT_NULL = ("title", "pinned", "archived")

    title: Name | None = None
    pinned: bool | None = None
    archived: bool | None = None
    document_scope: list[str] | None = None
    language: Language | None = None


class ChatList(BaseModel):
    items: list[Chat]


class MessageCreate(Body):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=TEXT_MAX)]
    language: Language | None = None  # null: the language of the text (Devanagari → Hindi)


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
TEXT_ANSWER_LENGTH = "short"  # same style as voice for now; "full" gives text chats longer written answers

_SSE_DOC = {
    200: {
        "description": (
            "Server-Sent Events, in order: user_message (the saved user Message), sources ({sources: [Citation], "
            "confidence, abstained}), delta ({text}) zero or more times, agent_message (the saved agent Message). "
            "On failure: error ({detail, stage: retrieval | llm | storage}) and the stream ends. Closing the "
            "connection mid-answer stops generation; the partial answer is saved with route.stopped = true. "
            "A turn that searches the web (live data) also sends tool events ({name: web_search, phase: start | "
            "results | done | timeout | failed, query, …}) between user_message and agent_message: start before "
            "sources; results (web Citations, kind: web, [W1]…) before sources and again before a continuation's "
            "deltas; one of done / timeout / failed when the search ends. "
            "The live visual canvas (docs/DESIGN.md §12.1): an answer from the documents that calls for a visual gets "
            "visual events ({phase: preparing | ready | failed, visual_id, visual?, detail?}) and a canvas snapshot "
            "({panels}) once its text is complete: preparing (a requested visual's skeleton) just before "
            "agent_message, ready + canvas (or failed) after it; the stream stays open for them, at most "
            "canvas.planner_timeout_ms + 2 s. A canvas edit ('make it a bar chart') sends its visual / canvas events "
            "before its delta ('Done.')."
        ),
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}


async def sse(events: AsyncGenerator[ChatEvent, None]) -> AsyncIterator[bytes]:
    """``event: <name>`` + one ``data:`` line of compact JSON + a blank line, per event. When the client disconnects
    the response is cancelled or closed, and so is the pipeline: it stops generating and keeps the partial answer."""
    async with contextlib.aclosing(events):
        async for e in events:
            data = json.dumps(e.payload(), ensure_ascii=False, separators=(",", ":"))
            yield f"event: {e.name}\ndata: {data}\n\n".encode()


@router.get("/api/projects/{project_id}/chats")
async def list_chats(project_id: str, chats: Chats, include_archived: bool = False) -> ChatList:
    """The project's chats, most recent activity first."""
    return ChatList(items=await chats.list(project_id, include_archived=include_archived))


@router.post("/api/projects/{project_id}/chats", status_code=status.HTTP_201_CREATED)
async def create_chat(project_id: str, body: ChatCreate, chats: Chats) -> Chat:
    return await chats.create(project_id, title=body.title, document_scope=body.document_scope, language=body.language)


@router.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, chats: Chats) -> Chat:
    return await chats.get(chat_id)


@router.patch("/api/chats/{chat_id}")
async def update_chat(chat_id: str, body: ChatUpdate, chats: Chats) -> Chat:
    """Rename, pin/unpin, archive/unarchive, narrow to some documents (``document_scope``; null = all)."""
    return await chats.update(chat_id, **body.changes())


@router.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str, chats: Chats) -> Response:
    """Delete the chat with its messages and summaries."""
    await chats.delete(chat_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/api/chats/{chat_id}/messages")
async def list_messages(
    chat_id: str,
    messages: Messages,
    after: Annotated[int | None, Query(ge=0, description="read forward: messages with seq > after")] = None,
    before: Annotated[int | None, Query(ge=1, description="read backward: the last messages with seq < before")] = None,
    limit: Annotated[int, Query(ge=1, le=PAGE_MAX)] = PAGE_DEFAULT,
) -> MessagePage:
    """The transcript, one page at a time, always in chronological order. Without a cursor it starts at the first
    message; to open at the latest, pass ``before=<message_count + 1>``. ``next_cursor`` continues in the same
    direction."""
    if after is not None and before is not None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "pass either 'after' or 'before', not both")
    return await messages.list(chat_id, after=after, before=before, limit=limit)


@router.post("/api/chats/{chat_id}/messages", response_class=StreamingResponse, responses=_SSE_DOC)
async def post_message(chat_id: str, body: MessageCreate, turns: ChatTurns) -> StreamingResponse:
    """Send a text message and stream the answer. Unknown chat → 404 and invalid text → 422 before the stream
    starts; after that, failures arrive as an ``error`` event (the user message stays saved). The turn itself is
    ``ChatTurnService`` (shared with voice); this endpoint only serialises its events."""
    turn = await turns.begin(chat_id, body.text, language=body.language, modality="text", length=TEXT_ANSWER_LENGTH)
    return StreamingResponse(sse(turns.run(turn)), media_type="text/event-stream", headers=SSE_HEADERS)
