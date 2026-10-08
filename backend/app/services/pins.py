"""Pinned projects and chats for the sidebar's PINNED section (docs/DESIGN.md §3.9)."""

from __future__ import annotations

from sqlalchemy import select

from ..db import models as orm
from ..domain.projects import Chat, PinnedChat, PinnedItems
from .base import Service
from .projects import project_query, to_projects


class PinService(Service):
    async def list(self) -> PinnedItems:
        """Everything pinned and not archived, most recently pinned first. Chats of archived projects are left out."""
        projects = (
            project_query()
            .where(orm.Project.pinned_at.is_not(None), orm.Project.archived_at.is_(None))
            .order_by(orm.Project.pinned_at.desc(), orm.Project.id.desc())
        )
        chats = (
            select(orm.Chat, orm.Project.name)
            .join(orm.Project, orm.Project.id == orm.Chat.project_id)
            .where(orm.Chat.pinned_at.is_not(None), orm.Chat.archived_at.is_(None), orm.Project.archived_at.is_(None))
            .order_by(orm.Chat.pinned_at.desc(), orm.Chat.id.desc())
        )
        async with self.db.session() as s:
            pinned_projects = to_projects((await s.execute(projects)).all())
            pinned_chats = [
                PinnedChat(**Chat.model_validate(chat).model_dump(), project_name=name)
                for chat, name in (await s.execute(chats)).all()
            ]
        return PinnedItems(projects=pinned_projects, chats=pinned_chats)
