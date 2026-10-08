"""Chat summaries: store and read the latest one per kind. Generating them (map-reduce with the LLM) is phase 3."""

from __future__ import annotations

from typing import Any, get_args

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models as orm
from ..domain.projects import ChatSummary, SummaryKind
from .base import InvalidInput, Service, get_or_404


class SummaryService(Service):
    async def save(
        self,
        chat_id: str,
        *,
        kind: SummaryKind,
        content: str,
        data: dict[str, Any] | None = None,
        covers_message_id: str | None = None,
        model: str | None = None,
    ) -> ChatSummary:
        """Store a summary, replacing the previous one of the same kind.

        ``covers_message_id`` is the last message the summary includes; by default the chat's latest message.
        """
        if kind not in get_args(SummaryKind):
            raise InvalidInput(f"kind must be one of {get_args(SummaryKind)}, got {kind!r}")
        async with self.db.session() as s:
            chat = await get_or_404(s, orm.Chat, chat_id, "chat")
            if covers_message_id is None:
                covers_seq = chat.message_count
                covers_message_id = await s.scalar(
                    select(orm.Message.id).where(orm.Message.chat_id == chat_id, orm.Message.seq == covers_seq)
                )
            else:
                covers_seq = await _seq_of(s, chat_id, covers_message_id)
            previous = delete(orm.ChatSummary).where(orm.ChatSummary.chat_id == chat_id, orm.ChatSummary.kind == kind)
            await s.execute(previous)
            row = orm.ChatSummary(
                chat_id=chat_id,
                kind=kind,
                content=content,
                data=data,
                covers_message_id=covers_message_id,
                model=model,
                created_at=self.now(),
            )
            s.add(row)
            await s.flush()  # assigns the id
            return _summary(row, covers_seq, chat.message_count)

    async def get(self, chat_id: str, kind: SummaryKind = "user") -> ChatSummary | None:
        """The chat's latest summary of this kind (None if there is none yet), flagged ``stale`` when messages were
        added after the last one it covers."""
        async with self.db.session() as s:
            chat = await get_or_404(s, orm.Chat, chat_id, "chat")
            found = (
                await s.execute(
                    select(orm.ChatSummary, orm.Message.seq)
                    .outerjoin(orm.Message, orm.Message.id == orm.ChatSummary.covers_message_id)
                    .where(orm.ChatSummary.chat_id == chat_id, orm.ChatSummary.kind == kind)
                )
            ).first()
            if found is None:
                return None
            row, covers_seq = found
            return _summary(row, covers_seq or 0, chat.message_count)


async def _seq_of(s: AsyncSession, chat_id: str, message_id: str) -> int:
    seq = await s.scalar(select(orm.Message.seq).where(orm.Message.id == message_id, orm.Message.chat_id == chat_id))
    if seq is None:
        raise InvalidInput(f"covers_message_id: {message_id!r} is not a message of chat {chat_id!r}")
    return seq


def _summary(row: orm.ChatSummary, covers_seq: int, message_count: int) -> ChatSummary:
    return ChatSummary.model_validate(row).model_copy(update={"stale": covers_seq < message_count})
