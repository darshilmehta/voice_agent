"""initial schema: projects, documents (+ versions, ingestion jobs), chats, messages, chat summaries

Revision ID: 0001
Revises:
Create Date: 2026-10-09 00:01:05.589264
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID = sa.String(length=40)


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def _parent(column: str, table: str, child: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{table}.id"], name=op.f(f"fk_{child}_{column}_{table}"), ondelete="CASCADE"
    )


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", ID, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        _ts("pinned_at", nullable=True),
        _ts("archived_at", nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
    )
    op.create_index(op.f("ix_projects_pinned_at"), "projects", ["pinned_at"])

    op.create_table(
        "documents",
        sa.Column("id", ID, nullable=False),
        sa.Column("project_id", ID, nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("mime", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
        _parent("project_id", "projects", "documents"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
    )
    op.create_index("ix_documents_project_id_sha256", "documents", ["project_id", "sha256"])

    op.create_table(
        "document_versions",
        sa.Column("id", ID, nullable=False),
        sa.Column("document_id", ID, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("mime", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.String(length=1024), nullable=False),
        _ts("created_at"),
        _parent("document_id", "documents", "document_versions"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_versions")),
        sa.UniqueConstraint("document_id", "version", name=op.f("uq_document_versions_document_id_version")),
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", ID, nullable=False),
        sa.Column("document_id", ID, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        _ts("created_at"),
        _ts("started_at", nullable=True),
        _ts("finished_at", nullable=True),
        _parent("document_id", "documents", "ingestion_jobs"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_jobs")),
    )
    op.create_index(op.f("ix_ingestion_jobs_document_id"), "ingestion_jobs", ["document_id"])
    op.create_index("ix_ingestion_jobs_status_created_at", "ingestion_jobs", ["status", "created_at"])

    op.create_table(
        "chats",
        sa.Column("id", ID, nullable=False),
        sa.Column("project_id", ID, nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("title_is_auto", sa.Boolean(), nullable=False),
        _ts("pinned_at", nullable=True),
        _ts("archived_at", nullable=True),
        sa.Column("document_scope", _json(), nullable=True),
        sa.Column("language", sa.String(length=8), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        _ts("last_message_at", nullable=True),
        _parent("project_id", "projects", "chats"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chats")),
    )
    op.create_index(op.f("ix_chats_pinned_at"), "chats", ["pinned_at"])
    op.create_index(op.f("ix_chats_project_id"), "chats", ["project_id"])

    op.create_table(
        "messages",
        sa.Column("id", ID, nullable=False),
        sa.Column("chat_id", ID, nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=8), nullable=False),
        sa.Column("modality", sa.String(length=8), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heard_text", sa.Text(), nullable=True),
        sa.Column("language", sa.String(length=8), nullable=True),
        sa.Column("citations", _json(), nullable=False),
        sa.Column("route", _json(), nullable=True),
        sa.Column("latency", _json(), nullable=True),
        _ts("created_at"),
        sa.CheckConstraint("modality IN ('voice', 'text')", name=op.f("ck_messages_modality")),
        sa.CheckConstraint("role IN ('user', 'agent', 'event')", name=op.f("ck_messages_role")),
        _parent("chat_id", "chats", "messages"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint("chat_id", "seq", name=op.f("uq_messages_chat_id_seq")),
    )

    op.create_table(
        "chat_summaries",
        sa.Column("id", ID, nullable=False),
        sa.Column("chat_id", ID, nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("data", _json(), nullable=True),
        sa.Column("covers_message_id", ID, nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        _ts("created_at"),
        sa.CheckConstraint("kind IN ('user', 'memory')", name=op.f("ck_chat_summaries_kind")),
        _parent("chat_id", "chats", "chat_summaries"),
        sa.ForeignKeyConstraint(
            ["covers_message_id"],
            ["messages.id"],
            name=op.f("fk_chat_summaries_covers_message_id_messages"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_summaries")),
        sa.UniqueConstraint("chat_id", "kind", name=op.f("uq_chat_summaries_chat_id_kind")),
    )
    op.create_index(op.f("ix_chat_summaries_covers_message_id"), "chat_summaries", ["covers_message_id"])


def downgrade() -> None:
    # Children first; dropping a table drops its indexes.
    for table in (
        "chat_summaries",
        "messages",
        "chats",
        "ingestion_jobs",
        "document_versions",
        "documents",
        "projects",
    ):
        op.drop_table(table)
