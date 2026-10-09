"""canvas: typed table datasets, canvas and overview visuals, overview build markers (docs/DESIGN.md §12.1)

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-09 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID = sa.String(length=40)


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _parent(column: str, table: str, child: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{table}.id"], name=op.f(f"fk_{child}_{column}_{table}"), ondelete="CASCADE"
    )


def upgrade() -> None:
    op.create_table(
        "table_datasets",
        sa.Column("id", ID, nullable=False),
        sa.Column("table_id", ID, nullable=False),
        sa.Column("document_id", ID, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("typer_version", sa.String(length=16), nullable=False),
        sa.Column("chart_kind", sa.String(length=16), nullable=False),
        sa.Column("chart_confidence", sa.Float(), nullable=False),
        sa.Column("data", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        _parent("table_id", "document_tables", "table_datasets"),
        _parent("document_id", "documents", "table_datasets"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_table_datasets")),
        sa.UniqueConstraint("table_id", name=op.f("uq_table_datasets_table_id")),
    )
    op.create_index(op.f("ix_table_datasets_document_id"), "table_datasets", ["document_id"])

    op.create_table(
        "canvas_visuals",
        sa.Column("id", ID, nullable=False),
        sa.Column("project_id", ID, nullable=False),
        sa.Column("chat_id", ID, nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("document_ids", _json(), nullable=False),
        sa.Column("spec", _json(), nullable=False),
        sa.Column("visual", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        _parent("project_id", "projects", "canvas_visuals"),
        _parent("chat_id", "chats", "canvas_visuals"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_canvas_visuals")),
    )
    op.create_index(op.f("ix_canvas_visuals_chat_id"), "canvas_visuals", ["chat_id"])
    op.create_index(
        "ix_canvas_visuals_project_id_chat_id_position", "canvas_visuals", ["project_id", "chat_id", "position"]
    )

    op.create_table(
        "project_overviews",
        sa.Column("project_id", ID, nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
        _parent("project_id", "projects", "project_overviews"),
        sa.PrimaryKeyConstraint("project_id", name=op.f("pk_project_overviews")),
    )


def downgrade() -> None:
    for table in ("project_overviews", "canvas_visuals", "table_datasets"):
        op.drop_table(table)
