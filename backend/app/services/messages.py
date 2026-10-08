"""Messages: append (used by the chat pipeline) and read a transcript page by page (docs/DESIGN.md §3.9).

Work that should follow a saved agent answer (the automatic chat title) subscribes with ``add_agent_message_hook``
instead of being called from the chat pipeline: hooks are registered per metadata database, so every
``MessageService`` on that database fires them, whoever created it (text turns, voice turns, tests).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, get_args
from weakref import WeakKeyDictionary

from sqlalchemy import select, update

from ..db import models as orm
from ..domain.projects import Citation, Message, MessagePage, Modality, Role, coerce_citations
from ..providers.storage import MetadataDB
from .base import InvalidInput, NotFound, Service

log = logging.getLogger(__name__)

PAGE_DEFAULT = 50
PAGE_MAX = 200

AgentMessageHook = Callable[[Message], Awaitable[None]]
_AGENT_HOOKS: WeakKeyDictionary[MetadataDB, list[AgentMessageHook]] = WeakKeyDictionary()


def add_agent_message_hook(db: MetadataDB, hook: AgentMessageHook) -> Callable[[], None]:
    """Call ``await hook(message)`` after every agent message saved through ``db``, once its transaction has
    committed. Hooks must be quick (they run before the saver continues): anything slow goes to the job queue. A hook
    that raises is logged and ignored, so it can never fail a chat turn. Returns a function that removes the hook."""
    _AGENT_HOOKS.setdefault(db, []).append(hook)

    def remove() -> None:
        hooks = _AGENT_HOOKS.get(db, [])
        if hook in hooks:
            hooks.remove(hook)

    return remove


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
        saved = Message.model_validate(message)
        if role == "agent":
            await self._run_agent_hooks(saved)
        return saved

    async def _run_agent_hooks(self, message: Message) -> None:
        for hook in list(_AGENT_HOOKS.get(self.db, ())):
            try:
                await hook(message)
            except Exception:
                log.exception("agent-message hook %r failed (chat %s)", hook, message.chat_id)

    async def record_interruption(self, message_id: str, *, heard_text: str, reason: str) -> Message:
        """Mark an agent answer as cut short after it was saved (voice: the user barged in or said stop while it was
        still playing): ``heard_text`` is what was actually played; the route gets ``stopped = true`` and
        ``interrupted = reason``."""
        async with self.db.session() as s:
            row = await s.get(orm.Message, message_id)
            if row is None:
                raise NotFound("message", message_id)
            if row.role != "agent":
                raise InvalidInput(f"message {message_id!r} is not an agent answer")
            row.heard_text = heard_text
            row.route = {**(row.route or {}), "stopped": True, "interrupted": reason}
            await s.flush()
            return Message.model_validate(row)

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
