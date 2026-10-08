"""Projects: create, list (pinned first), rename, pin, archive, delete with everything inside (docs/DESIGN.md §3.9)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models as orm
from ..domain.projects import Project
from .base import UNSET, Maybe, NotFound, Service, blank_to_none, clean_text, get_or_404

NAME_MAX = 200


def project_query() -> Select[Any]:
    """Projects with their chat and document counts."""
    chats = select(func.count()).select_from(orm.Chat).where(orm.Chat.project_id == orm.Project.id)
    documents = select(func.count()).select_from(orm.Document).where(orm.Document.project_id == orm.Project.id)
    return select(
        orm.Project,
        chats.scalar_subquery().label("chat_count"),
        documents.scalar_subquery().label("document_count"),
    )


def to_projects(rows: Sequence[Any]) -> list[Project]:
    return [
        Project.model_validate(p).model_copy(update={"chat_count": chats, "document_count": documents})
        for p, chats, documents in rows
    ]


class ProjectService(Service):
    async def create(self, name: str, description: str | None = None) -> Project:
        now = self.now()
        row = orm.Project(
            name=clean_text(name, "name", NAME_MAX),
            description=blank_to_none(description),
            created_at=now,
            updated_at=now,
        )
        async with self.db.session() as s:
            s.add(row)
        return Project.model_validate(row)

    async def get(self, project_id: str) -> Project:
        async with self.db.session() as s:
            return await self._load(s, project_id)

    async def list(self, *, include_archived: bool = False) -> list[Project]:
        """Pinned projects first (most recently pinned on top), then the rest by most recent activity."""
        stmt = project_query().order_by(
            orm.Project.pinned_at.desc().nulls_last(), orm.Project.updated_at.desc(), orm.Project.id.desc()
        )
        if not include_archived:
            stmt = stmt.where(orm.Project.archived_at.is_(None))
        async with self.db.session() as s:
            return to_projects((await s.execute(stmt)).all())

    async def update(
        self,
        project_id: str,
        *,
        name: Maybe[str] = UNSET,
        description: Maybe[str | None] = UNSET,
        pinned: Maybe[bool] = UNSET,
        archived: Maybe[bool] = UNSET,
    ) -> Project:
        """Change only the given fields. Pinning keeps the original pin time if already pinned (same for archiving);
        pinning and archiving don't count as activity, so they don't move the project in the recent list."""
        async with self.db.session() as s:
            row = await get_or_404(s, orm.Project, project_id, "project")
            now = self.now()
            if name is not UNSET:
                row.name = clean_text(name, "name", NAME_MAX)
            if description is not UNSET:
                row.description = blank_to_none(description)
            if name is not UNSET or description is not UNSET:
                row.updated_at = now
            if pinned is not UNSET:
                row.pinned_at = (row.pinned_at or now) if pinned else None
            if archived is not UNSET:
                row.archived_at = (row.archived_at or now) if archived else None
            await s.flush()
            return await self._load(s, project_id)

    async def delete(self, project_id: str) -> list[str]:
        """Delete the project and, through ON DELETE CASCADE, its documents (versions, ingestion jobs), chats,
        messages and summaries. Returns the deleted document ids so callers can remove their vectors and files."""
        async with self.db.session() as s:
            document_ids = (await s.scalars(select(orm.Document.id).where(orm.Document.project_id == project_id))).all()
            result = await s.execute(delete(orm.Project).where(orm.Project.id == project_id))
            if result.rowcount == 0:  # type: ignore[attr-defined]
                raise NotFound("project", project_id)
        return list(document_ids)

    async def _load(self, s: AsyncSession, project_id: str) -> Project:
        rows = (await s.execute(project_query().where(orm.Project.id == project_id))).all()
        if not rows:
            raise NotFound("project", project_id)
        return to_projects(rows)[0]
