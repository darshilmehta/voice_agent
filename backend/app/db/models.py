"""SQLAlchemy models for projects, documents, chats, messages and summaries (docs/DESIGN.md §3.9).

Every child row references its parent with ``ON DELETE CASCADE``, so deleting a project deletes its documents
(with their versions and ingestion jobs) and its chats (with their messages and summaries) inside the database.
On SQLite this needs ``PRAGMA foreign_keys=ON``, which the engine sets on every connection (``app.db.engine``).

The models deliberately have no ORM ``relationship()``s: services query explicitly (no lazy loading under asyncio)
and deletes cascade in the database. One consequence: when adding a parent and its child in the same session
(a document and its first version), ``await session.flush()`` after adding the parent, since SQLAlchemy only orders
INSERTs across tables through relationships.

Schema changes go through Alembic (``app/db/migrations``); a test fails if these models and the migrations drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .types import JSONType, UTCDateTime, new_id, utcnow

# Deterministic constraint names: Postgres needs them to alter constraints, and SQLite batch migrations need them
# to recreate tables.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

ID = String(40)  # "<prefix>_" + 26 characters, see new_id()


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012  (SQLAlchemy reads this class attribute)
        datetime: UTCDateTime(),
        dict[str, Any]: JSONType,
        list[str]: JSONType,
        list[dict[str, Any]]: JSONType,
    }


def _parent(table: str) -> Any:
    return ForeignKey(f"{table}.id", ondelete="CASCADE")


class Project(Base):
    """A container of documents and chats about one subject.

    ``updated_at`` changes when the project is edited and when there is activity inside it (a chat created, a
    message added), so "recent first" ordering follows what the user worked on. Pinning and archiving don't
    change it.
    """

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(ID, primary_key=True, default=lambda: new_id("prj"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    pinned_at: Mapped[datetime | None] = mapped_column(index=True)
    archived_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


class Document(Base):
    """An uploaded file, ingested once, belonging to one project (§3.1).

    ``version``, ``sha256``, ``filename``, ``mime`` and ``size_bytes`` describe the current version; every version
    is kept in ``document_versions``. ``status`` follows the ingestion lifecycle: PENDING → PROCESSING → READY,
    or FAILED with ``error``.
    """

    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_project_id_sha256", "project_id", "sha256"),)  # dedupe lookups

    id: Mapped[str] = mapped_column(ID, primary_key=True, default=lambda: new_id("doc"))
    project_id: Mapped[str] = mapped_column(ID, _parent("projects"))
    filename: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    page_count: Mapped[int | None]
    chunk_count: Mapped[int | None]
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


class DocumentVersion(Base):
    """One uploaded version of a document; ``storage_key`` locates the original file in the object store."""

    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version"),)

    id: Mapped[str] = mapped_column(ID, primary_key=True, default=lambda: new_id("docv"))
    document_id: Mapped[str] = mapped_column(ID, _parent("documents"))
    version: Mapped[int]
    filename: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class IngestionJob(Base):
    """One attempt to ingest a document version. ``status``: QUEUED → RUNNING → SUCCEEDED | FAILED | CANCELLED;
    ``stage`` is the pipeline step it is in or failed at (validate, parse, chunk, embed, index, qa)."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_status_created_at", "status", "created_at"),)  # queue polling

    id: Mapped[str] = mapped_column(ID, primary_key=True, default=lambda: new_id("job"))
    document_id: Mapped[str] = mapped_column(ID, _parent("documents"), index=True)
    version: Mapped[int]
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    stage: Mapped[str | None] = mapped_column(String(32))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class Chat(Base):
    """One conversation inside a project, by voice and/or text.

    ``document_scope`` is a JSON list of document ids the chat answers from; NULL means all of the project's
    documents. ``message_count`` doubles as the sequence counter for the chat's messages (``messages.seq``).
    """

    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(ID, primary_key=True, default=lambda: new_id("cht"))
    project_id: Mapped[str] = mapped_column(ID, _parent("projects"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    title_is_auto: Mapped[bool] = mapped_column(default=True)
    pinned_at: Mapped[datetime | None] = mapped_column(index=True)
    archived_at: Mapped[datetime | None]
    document_scope: Mapped[list[str] | None]
    language: Mapped[str | None] = mapped_column(String(8))
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_message_at: Mapped[datetime | None]


class Message(Base):
    """One utterance in a chat. ``seq`` (1, 2, 3, … per chat) gives the transcript order and the pagination cursor.

    For an interrupted agent answer, ``text`` is the full generated answer and ``heard_text`` what was actually
    played before the barge-in (§3.3).
    """

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("chat_id", "seq"),
        CheckConstraint("role IN ('user', 'agent', 'event')", name="role"),
        CheckConstraint("modality IN ('voice', 'text')", name="modality"),
    )

    id: Mapped[str] = mapped_column(ID, primary_key=True, default=lambda: new_id("msg"))
    chat_id: Mapped[str] = mapped_column(ID, _parent("chats"))
    seq: Mapped[int]
    role: Mapped[str] = mapped_column(String(8))
    modality: Mapped[str] = mapped_column(String(8), default="text")
    text: Mapped[str] = mapped_column(Text)
    heard_text: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(8))
    citations: Mapped[list[dict[str, Any]]] = mapped_column(default=list)
    route: Mapped[dict[str, Any] | None]
    latency: Mapped[dict[str, Any] | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ChatSummary(Base):
    """The latest summary of a chat per kind: ``user`` (the on-demand digest) or ``memory`` (the compact context
    that keeps prompts short, §3.5). ``covers_message_id`` is the last message it includes."""

    __tablename__ = "chat_summaries"
    __table_args__ = (
        UniqueConstraint("chat_id", "kind"),
        CheckConstraint("kind IN ('user', 'memory')", name="kind"),
    )

    id: Mapped[str] = mapped_column(ID, primary_key=True, default=lambda: new_id("sum"))
    chat_id: Mapped[str] = mapped_column(ID, _parent("chats"))
    kind: Mapped[str] = mapped_column(String(8))
    content: Mapped[str] = mapped_column(Text)  # markdown
    data: Mapped[dict[str, Any] | None]  # structured form (overview, topics, key answers, …) when the generator has it
    covers_message_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("messages.id", ondelete="SET NULL"), index=True
    )
    model: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
