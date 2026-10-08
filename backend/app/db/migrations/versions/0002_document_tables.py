"""document tables: every parsed table of a document version as a cell-level dataset (docs/DESIGN.md §3.1, §12.1)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-09 01:00:04.072586
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID = sa.String(length=40)


def _json() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "document_tables",
        sa.Column("id", ID, nullable=False),
        sa.Column("document_id", ID, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("table_index", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("bbox", _json(), nullable=True),
        sa.Column("heading_path", _json(), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("num_rows", sa.Integer(), nullable=False),
        sa.Column("num_cols", sa.Integer(), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("cells", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name=op.f("fk_document_tables_document_id_documents"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_tables")),
        # Its index also serves lookups by document_id (the leading column).
        sa.UniqueConstraint(
            "document_id", "version", "table_index", name=op.f("uq_document_tables_document_id_version_table_index")
        ),
    )


def downgrade() -> None:
    op.drop_table("document_tables")
