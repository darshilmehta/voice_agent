"""chat states: each chat's conversation state for the router (topics, languages, interruption; docs/DESIGN.md §3.5)

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-09 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID = sa.String(length=40)


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "chat_states",
        sa.Column("chat_id", ID, nullable=False),
        sa.Column("active_topic", sa.String(length=200), nullable=True),
        sa.Column("previous_topic", sa.String(length=200), nullable=True),
        sa.Column("document_topic", sa.String(length=200), nullable=True),
        sa.Column("document_query", sa.Text(), nullable=True),
        sa.Column("active_document_ids", _json(), nullable=False),
        sa.Column("input_language", sa.String(length=8), nullable=True),
        sa.Column("response_language", sa.String(length=8), nullable=True),
        sa.Column("preferred_language", sa.String(length=8), nullable=True),
        sa.Column("last_intent", sa.String(length=32), nullable=True),
        sa.Column("last_interrupted_message_id", ID, nullable=True),
        sa.Column("retrieval_enabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_id"], ["chats.id"], name=op.f("fk_chat_states_chat_id_chats"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("chat_id", name=op.f("pk_chat_states")),
    )


def downgrade() -> None:
    op.drop_table("chat_states")
