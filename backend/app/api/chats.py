"""Chats and their transcripts (docs/DESIGN.md §3.9). Messages are written by the chat pipeline, not over REST."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel

from ..domain.projects import Chat, MessagePage
from ..services.messages import PAGE_DEFAULT, PAGE_MAX
from ..settings import Language
from .deps import Chats, Messages
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
