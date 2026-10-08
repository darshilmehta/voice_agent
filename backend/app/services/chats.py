"""Chats inside a project: create, list (newest activity first), rename, pin, document scope, delete (§3.9)."""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models as orm
from ..domain.projects import Chat
from .base import UNSET, InvalidInput, Maybe, NotFound, Service, clean_text, get_or_404

TITLE_MAX = 200
DEFAULT_TITLE = "New chat"  # replaced by an automatic title from the first question in phase 3


class ChatService(Service):
    async def create(
        self,
        project_id: str,
        *,
        title: str | None = None,
        document_scope: list[str] | None = None,
        language: str | None = None,
    ) -> Chat:
        """Without a title the chat gets a placeholder and ``title_is_auto`` stays true until someone renames it."""
        async with self.db.session() as s:
            project = await get_or_404(s, orm.Project, project_id, "project")
            now = self.now()
            row = orm.Chat(
                project_id=project_id,
                title=DEFAULT_TITLE if title is None else clean_text(title, "title", TITLE_MAX),
                title_is_auto=title is None,
                document_scope=await _checked_scope(s, project_id, document_scope),
                language=language,
                created_at=now,
                updated_at=now,
            )
            s.add(row)
            project.updated_at = now
        return Chat.model_validate(row)

    async def get(self, chat_id: str) -> Chat:
        async with self.db.session() as s:
            return Chat.model_validate(await get_or_404(s, orm.Chat, chat_id, "chat"))

    async def list(self, project_id: str, *, include_archived: bool = False) -> list[Chat]:
        """The project's chats, most recent activity (last message, or creation) first."""
        stmt = (
            select(orm.Chat)
            .where(orm.Chat.project_id == project_id)
            .order_by(func.coalesce(orm.Chat.last_message_at, orm.Chat.created_at).desc(), orm.Chat.id.desc())
        )
        if not include_archived:
            stmt = stmt.where(orm.Chat.archived_at.is_(None))
        async with self.db.session() as s:
            await get_or_404(s, orm.Project, project_id, "project")
            return [Chat.model_validate(c) for c in (await s.scalars(stmt)).all()]

    async def update(
        self,
        chat_id: str,
        *,
        title: Maybe[str] = UNSET,
        pinned: Maybe[bool] = UNSET,
        archived: Maybe[bool] = UNSET,
        document_scope: Maybe[list[str] | None] = UNSET,
        language: Maybe[str | None] = UNSET,
    ) -> Chat:
        """Change only the given fields. A new title is the user's (``title_is_auto`` becomes false);
        ``document_scope=None`` means all of the project's documents."""
        async with self.db.session() as s:
            row = await get_or_404(s, orm.Chat, chat_id, "chat")
            now = self.now()
            if title is not UNSET:
                row.title = clean_text(title, "title", TITLE_MAX)
                row.title_is_auto = False
            if document_scope is not UNSET:
                row.document_scope = await _checked_scope(s, row.project_id, document_scope)
            if language is not UNSET:
                row.language = language
            if title is not UNSET or document_scope is not UNSET or language is not UNSET:
                row.updated_at = now
            if pinned is not UNSET:
                row.pinned_at = (row.pinned_at or now) if pinned else None
            if archived is not UNSET:
                row.archived_at = (row.archived_at or now) if archived else None
            return Chat.model_validate(row)

    async def delete(self, chat_id: str) -> None:
        """Delete the chat with its messages and summaries (ON DELETE CASCADE)."""
        async with self.db.session() as s:
            result = await s.execute(delete(orm.Chat).where(orm.Chat.id == chat_id))
            if result.rowcount == 0:  # type: ignore[attr-defined]
                raise NotFound("chat", chat_id)


async def _checked_scope(s: AsyncSession, project_id: str, scope: list[str] | None) -> list[str] | None:
    """None (all documents) or a de-duplicated list of documents that belong to this project."""
    if scope is None:
        return None
    scope = list(dict.fromkeys(scope))
    if not scope:
        raise InvalidInput("document_scope must list at least one document; use null for all documents")
    stmt = select(orm.Document.id).where(orm.Document.project_id == project_id, orm.Document.id.in_(scope))
    found = set((await s.scalars(stmt)).all())
    unknown = [d for d in scope if d not in found]
    if unknown:
        raise InvalidInput(f"document_scope: not documents of project {project_id!r}: {', '.join(unknown)}")
    return scope
