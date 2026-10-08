"""Messages: append (used by the chat pipeline) and read a transcript page by page (docs/DESIGN.md §3.9)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, get_args

from sqlalchemy import select, update

from ..db import models as orm
from ..domain.projects import Citation, Message, MessagePage, Modality, Role, coerce_citations
from .base import InvalidInput, NotFound, Service

PAGE_DEFAULT = 50
PAGE_MAX = 200


class MessageService(Service):
    async def append(
        self,
        chat_id: str,
        *,
        role: Role,
        text: str,
        modality: Modality = "text",
        heard_text: str | None = None,
        language: str | None = None,
        citations: Sequence[Citation | dict[str, Any]] | None = None,
        route: dict[str, Any] | None = None,
        latency: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message at the end of the chat.

        The chat's counter is incremented and read back in one UPDATE … RETURNING, which serializes concurrent
        appends to the same chat on both SQLite and Postgres, so ``seq`` is gap-free and unique per chat. The chat's
        and project's activity times move to now. ``citations`` are stored typed (``Citation``); plain dicts are
        coerced the way old rows are read.
        """
        if role not in get_args(Role):
            raise InvalidInput(f"role must be one of {get_args(Role)}, got {role!r}")
        if modality not in get_args(Modality):
            raise InvalidInput(f"modality must be one of {get_args(Modality)}, got {modality!r}")
        now = self.now()
        async with self.db.session() as s:
            counter = await s.execute(
                update(orm.Chat)
                .where(orm.Chat.id == chat_id)
                .values(message_count=orm.Chat.message_count + 1, last_message_at=now, updated_at=now)
                .returning(orm.Chat.message_count, orm.Chat.project_id)
                .execution_options(synchronize_session=False)
            )
            row = counter.first()
            if row is None:
                raise NotFound("chat", chat_id)
            seq, project_id = row
            message = orm.Message(
                chat_id=chat_id,
                seq=seq,
                role=role,
                modality=modality,
                text=text,
                heard_text=heard_text,
                language=language,
                citations=[c.model_dump(mode="json") for c in coerce_citations(list(citations or []))],
                route=route,
                latency=latency,
                created_at=now,
            )
            s.add(message)
            await s.execute(
                update(orm.Project)
                .where(orm.Project.id == project_id)
                .values(updated_at=now)
                .execution_options(synchronize_session=False)
            )
        return Message.model_validate(message)

    async def list(
        self,
        chat_id: str,
        *,
        after: int | None = None,
        before: int | None = None,
        limit: int = PAGE_DEFAULT,
    ) -> MessagePage:
        """One page of the transcript, in chronological order.

        - ``after=N`` (default 0): the first ``limit`` messages with ``seq > N``: read forward from the start.
        - ``before=N``: the last ``limit`` messages with ``seq < N``: read backward from the end. To open a chat at
          its latest messages pass ``before=message_count + 1``.
        """
        if after is not None and before is not None:
            raise InvalidInput("pass either 'after' or 'before', not both")
        if not 1 <= limit <= PAGE_MAX:
            raise InvalidInput(f"limit must be between 1 and {PAGE_MAX}")
        backward = before is not None
        stmt = select(orm.Message).where(orm.Message.chat_id == chat_id)
        if backward:
            stmt = stmt.where(orm.Message.seq < before).order_by(orm.Message.seq.desc())
        else:
            stmt = stmt.where(orm.Message.seq > (after or 0)).order_by(orm.Message.seq)
        async with self.db.session() as s:
            total = await s.scalar(select(orm.Chat.message_count).where(orm.Chat.id == chat_id))
            if total is None:
                raise NotFound("chat", chat_id)
            rows = list((await s.scalars(stmt.limit(limit + 1))).all())
        has_more = len(rows) > limit
        rows = rows[:limit]
        if backward:
            rows.reverse()
        items = [Message.model_validate(r) for r in rows]
        next_cursor = None
        if has_more and items:
            next_cursor = items[0].seq if backward else items[-1].seq
        return MessagePage(items=items, total=total, has_more=has_more, next_cursor=next_cursor)
