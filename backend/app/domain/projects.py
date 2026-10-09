"""Projects, documents, chats, messages and summaries as the rest of the app sees them (docs/DESIGN.md §3.9).

Services return these, never ORM rows, so callers can't trigger lazy database access after a session is closed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    computed_field,
    field_validator,
    model_serializer,
)

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


class DocumentTable(Record):
    """A parsed table of a document version as a cell-level dataset (§12.1); cells as in ``ParsedTable``."""

    id: str
    document_id: str
    version: int
    table_index: int
    page_start: int | None
    page_end: int | None
    bbox: list[float] | None
    heading_path: list[str]
    caption: str | None
    num_rows: int
    num_cols: int
    markdown: str
    cells: list[dict[str, Any]]
    created_at: datetime


SNIPPET_CHARS = 300


CitationKind = Literal["document", "web"]
_WEB_FIELDS = ("kind", "url", "title", "site", "published")


class CitationJSON(TypedDict):
    """A citation as JSON (the API schema): the web fields appear on web citations only."""

    source_id: str
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_id: str
    snippet: str
    kind: NotRequired[CitationKind]
    url: NotRequired[str | None]
    title: NotRequired[str | None]
    site: NotRequired[str | None]
    published: NotRequired[str | None]


class Citation(BaseModel):
    """A source an agent answer cites. ``source_id`` is the marker used in the answer text: ``[S1]`` for a passage
    of the user's documents, ``[W1]`` for a live web search result (docs/DESIGN.md §3.7); ``snippet`` is up to ~300
    characters of the cited chunk or result.

    Web citations carry ``kind: "web"``, ``url``, ``title``, ``site`` (host) and ``published`` (when the engine gave
    a date); their ``filename`` is the site, ``document_id`` and ``chunk_id`` are empty and there are no pages.
    Document citations serialize exactly as before web search existed (no ``kind`` or web fields): a citation without
    ``kind`` is a document passage."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_id: str
    snippet: str
    kind: CitationKind = "document"
    url: str | None = None
    title: str | None = None
    site: str | None = None
    published: str | None = None

    @model_serializer(mode="wrap")
    def _without_web_fields(self, handler: SerializerFunctionWrapHandler) -> CitationJSON:
        data = handler(self)
        if self.kind == "document":
            for name in _WEB_FIELDS:
                data.pop(name, None)
        return data

    @classmethod
    def coerce(cls, value: Any, position: int) -> Citation | None:
        """A Citation from a stored value. Rows written before citations were typed hold free-form JSON
        (e.g. ``{"document_id", "page", "chunk_id"}``): missing fields get neutral defaults, ``page`` fills both page
        fields, and values that aren't objects are dropped (None)."""
        if isinstance(value, Citation):
            return value
        if not isinstance(value, dict):
            return None
        page = _int_or_none(value.get("page"))
        start = _int_or_none(value.get("page_start"))
        end = _int_or_none(value.get("page_end"))
        web: dict[str, Any] = {}
        if value.get("kind") == "web":
            web = {"kind": "web", **{k: _str_or_none(value.get(k)) for k in ("url", "title", "site", "published")}}
        return cls(
            source_id=str(value.get("source_id") or f"{'W' if web else 'S'}{position}"),
            document_id=str(value.get("document_id") or ""),
            filename=str(value.get("filename") or ""),
            page_start=start if start is not None else page,
            page_end=end if end is not None else (start if start is not None else page),
            chunk_id=str(value.get("chunk_id") or ""),
            snippet=str(value.get("snippet") or value.get("text") or "")[:SNIPPET_CHARS],
            **web,
        )


def coerce_citations(values: Any) -> list[Citation]:
    if not isinstance(values, list | tuple):
        return []
    coerced = (Citation.coerce(v, i) for i, v in enumerate(values, start=1))
    return [c for c in coerced if c is not None]


def _str_or_none(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


class Message(Record):
    id: str
    chat_id: str
    seq: int  # 1, 2, 3, … within the chat: transcript order and pagination cursor
    role: Role
    modality: Modality
    text: str
    heard_text: str | None  # agent answers cut off by a barge-in: what was actually played
    language: str | None
    citations: list[Citation]  # agent answers: only the sources the text cites ([S1] …)
    route: dict[str, Any] | None
    latency: dict[str, Any] | None
    created_at: datetime

    @field_validator("citations", mode="before")
    @classmethod
    def _tolerant_citations(cls, value: Any) -> list[Citation]:
        return coerce_citations(value)

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
    covers_seq: int = 0  # ``seq`` of the last message it includes (0: none)
    message_count: int = 0  # the chat's messages now
    stale: bool = False  # messages were added after the last one it covers (covers_seq < message_count)
