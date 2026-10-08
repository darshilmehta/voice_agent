"""Documents of a project, read-only here. Upload, ingestion and deletion belong to the ingestion pipeline (§3.1)."""

from __future__ import annotations

from sqlalchemy import select

from ..db import models as orm
from ..domain.projects import Document
from .base import Service, get_or_404


class DocumentService(Service):
    async def list(self, project_id: str) -> list[Document]:
        """The project's documents, newest first."""
        stmt = (
            select(orm.Document)
            .where(orm.Document.project_id == project_id)
            .order_by(orm.Document.created_at.desc(), orm.Document.id.desc())
        )
        async with self.db.session() as s:
            await get_or_404(s, orm.Project, project_id, "project")
            return [Document.model_validate(d) for d in (await s.scalars(stmt)).all()]
