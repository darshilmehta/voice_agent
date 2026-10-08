"""Projects, documents, chats, messages and summaries as the rest of the app sees them (docs/DESIGN.md §3.9).

Services return these, never ORM rows, so callers can't trigger lazy database access after a session is closed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field

Role = Literal["user", "agent", "event"]
Modality = Literal["voice", "text"]
SummaryKind = Literal["user", "memory"]


class Record(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)


class Project(Record):
    id: str
    name: str
    description: str | None
    pinned_at: datetime | None
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime
    chat_count: int = 0
    document_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pinned(self) -> bool:
        return self.pinned_at is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def archived(self) -> bool:
        return self.archived_at is not None


class Document(Record):
    id: str
    project_id: str
    filename: str
    mime: str
    size_bytes: int
    sha256: str
    version: int
    status: str  # PENDING | PROCESSING | READY | FAILED (owned by ingestion)
    page_count: int | None
    chunk_count: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class Chat(Record):
    id: str
    project_id: str
    title: str
    title_is_auto: bool
    pinned_at: datetime | None
    archived_at: datetime | None
    document_scope: list[str] | None  # None = all of the project's documents
    language: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pinned(self) -> bool:
        return self.pinned_at is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def archived(self) -> bool:
        return self.archived_at is not None


class PinnedChat(Chat):
    project_name: str


class PinnedItems(BaseModel):
    """Everything pinned, for the sidebar's PINNED section: most recently pinned first."""

    projects: list[Project]
    chats: list[PinnedChat]


class Message(Record):
    id: str
    chat_id: str
    seq: int  # 1, 2, 3, … within the chat: transcript order and pagination cursor
    role: Role
    modality: Modality
    text: str
    heard_text: str | None  # agent answers cut off by a barge-in: what was actually played
    language: str | None
    citations: list[dict[str, Any]]
    route: dict[str, Any] | None
    latency: dict[str, Any] | None
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def interrupted(self) -> bool:
        return self.heard_text is not None


class MessagePage(BaseModel):
    """A page of a transcript, always in chronological order.

    ``next_cursor`` is the value to pass as the same parameter (``after`` or ``before``) to get the next page in
    that direction; it is None when there are no more messages that way.
    """

    items: list[Message]
    total: int
    has_more: bool
    next_cursor: int | None


class ChatSummary(Record):
    id: str
    chat_id: str
    kind: SummaryKind
    content: str
    data: dict[str, Any] | None
    covers_message_id: str | None
    model: str | None
    created_at: datetime
    stale: bool = False  # messages were added after the last one it covers
