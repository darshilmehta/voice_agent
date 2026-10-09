"""Automatic chat titles (docs/DESIGN.md §3.9): a short title from the first question, off the answer's critical path.

    first agent message saved ──hook──► chat still untitled (auto title = the placeholder)?
                                         └─► job queue, short lane (never behind an ingestion): ask the LLM for
                                              ≤ 6 words in the conversation's language
                                              (small token cap, timeout, one retry)  ─ failed ─► cleaned first question
                                              └─► write the title only if nobody renamed the chat meanwhile

A title is generated only while the chat still has the placeholder title and ``title_is_auto`` is true, so it happens
once (a retry after a dropped job or an LLM failure falls to the next agent message), and a title the user set is never
touched: the write is a compare-and-set on ``(title_is_auto, title)`` inside its own transaction, which fails if the
user renamed the chat while the model was thinking.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from functools import partial
from typing import Literal

from sqlalchemy import select, update

from ..db import models as orm
from ..db.types import utcnow
from ..domain.projects import Chat, Message
from ..providers.llm import LLMClient
from ..providers.registry import Container
from ..providers.runtime import LANE_SHORT, JobQueue
from ..providers.storage import MetadataDB
from ..settings import Language, Settings
from .base import InvalidInput, NotFound, Service, ServiceError, Unavailable, get_or_404
from .chats import DEFAULT_TITLE, ChatService
from .language import message_language
from .revisit_prompts import clean_title, fallback_title, title_messages

log = logging.getLogger(__name__)

TITLE_MAX_TOKENS = 64  # six Hindi words are ~40 tokens (§9.1); the model stops by itself long before
TITLE_TEMPERATURE = 0.2
TITLE_TIMEOUT_S = 20.0  # per attempt
TITLE_ATTEMPTS = 2  # one try and one retry
JOB_NAME = "chat-title"

Replace = Literal["placeholder", "auto", "any"]


class TitleLocked(ServiceError):
    """The chat's title was set by the user and the caller didn't ask to replace it."""


class TitleService(Service):
    def __init__(
        self,
        db: MetadataDB,
        *,
        llm: LLMClient,
        settings: Settings,
        queue: JobQueue | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(db, clock=clock)
        self.chats = ChatService(db, clock=clock)
        self.llm = llm
        self.settings = settings
        self.queue = queue
        self._pending: set[str] = set()  # chats with a title job queued or running

    @classmethod
    def from_container(cls, container: Container) -> TitleService:
        db, llm, queue = container["metadata_db"], container["llm"], container["job_queue"]
        if not isinstance(db, MetadataDB) or not isinstance(llm, LLMClient) or not isinstance(queue, JobQueue):
            raise TypeError("expected MetadataDB, LLMClient and JobQueue providers")
        return cls(db, llm=llm, settings=container.settings, queue=queue)

    # -------------------------------------------------------------- the hook and its job

    async def on_agent_message(self, message: Message) -> None:
        """``add_agent_message_hook`` callback: queue a title job if this chat is still untitled. One small read."""
        chat_id = message.chat_id
        if message.role != "agent" or self.queue is None or chat_id in self._pending:
            return
        async with self.db.session() as s:
            row = (
                await s.execute(select(orm.Chat.title, orm.Chat.title_is_auto).where(orm.Chat.id == chat_id))
            ).first()
        if row is None or not row.title_is_auto or row.title != DEFAULT_TITLE:
            return
        self._pending.add(chat_id)
        try:
            await self.queue.submit(JOB_NAME, partial(self._job, chat_id), lane=LANE_SHORT)
        except Exception as e:  # queue shut down or a placeholder provider: the next agent message tries again
            self._pending.discard(chat_id)
            log.warning("chat %s: title job not queued: %s", chat_id, e)

    async def _job(self, chat_id: str) -> None:
        try:
            await self.generate(chat_id)
        except NotFound:
            log.info("chat %s: deleted before its title was written", chat_id)
        except Exception:
            log.exception("chat %s: title generation failed", chat_id)
        finally:
            self._pending.discard(chat_id)

    # -------------------------------------------------------------- generation

    async def generate(self, chat_id: str, *, replace: Replace = "placeholder", fallback: bool = True) -> Chat:
        """Title the chat from its first question (and the answer) and return it.

        ``replace`` says which titles may be replaced: ``"placeholder"`` (the job: only a chat still called "New chat"
        with an automatic title; otherwise nothing happens), ``"auto"`` (regenerate: any automatic title; a title the
        user set raises ``TitleLocked``) or ``"any"`` (the user insists: replaced, and the title is automatic again).
        The chat must have a user message to be titled (InvalidInput, except for ``"placeholder"``, which waits). When
        the model gives no usable title, ``fallback`` uses the cleaned first question; without it ``Unavailable`` is
        raised and nothing changes.
        """
        async with self.db.session() as s:
            chat = await get_or_404(s, orm.Chat, chat_id, "chat")
            current = Chat.model_validate(chat)
            if replace == "placeholder" and not (chat.title_is_auto and chat.title == DEFAULT_TITLE):
                return current
            if replace == "auto" and not chat.title_is_auto:
                raise TitleLocked("the title was set by the user")
            seen = (chat.title_is_auto, chat.title)
            question = await s.scalar(
                select(orm.Message)
                .where(orm.Message.chat_id == chat_id, orm.Message.role == "user")
                .order_by(orm.Message.seq)
                .limit(1)
            )
            if question is None:
                if replace != "placeholder":
                    raise InvalidInput("the chat has no message to title yet")
                return current
            answer = await s.scalar(
                select(orm.Message)
                .where(orm.Message.chat_id == chat_id, orm.Message.role == "agent", orm.Message.seq > question.seq)
                .order_by(orm.Message.seq)
                .limit(1)
            )
            question_text = question.text
            answer_text = None if answer is None or (answer.route or {}).get("abstained") else answer.text
            language = self._language(answer.language if answer else None, question_text, chat.language)

        title = await self._suggest(chat_id, language, question_text, answer_text)
        if title is None:
            if not fallback:
                raise Unavailable("could not generate a title; try again")
            title = fallback_title(question_text)
            if title is None:
                return current
            log.info("chat %s: title from the first question (model gave none)", chat_id)
        return await self._write(chat_id, title, seen)

    def _language(self, answer_language: str | None, question: str, chat_language: str | None) -> Language:
        for candidate in (answer_language, message_language(question), chat_language):
            if candidate in ("en", "hi"):
                return candidate  # type: ignore[return-value]
        return self.settings.client.default_language

    async def _suggest(self, chat_id: str, language: Language, question: str, answer: str | None) -> str | None:
        """The model's title, cleaned; None if both attempts fail or give nothing usable."""
        prompt = title_messages(language, question, answer)
        for attempt in range(1, TITLE_ATTEMPTS + 1):
            try:
                raw = await asyncio.wait_for(
                    self.llm.generate(prompt, max_tokens=TITLE_MAX_TOKENS, temperature=TITLE_TEMPERATURE),
                    TITLE_TIMEOUT_S,
                )
            except Exception as e:  # LLM down, timeout, placeholder provider: all mean "no title from the model"
                log.warning("chat %s: title attempt %d failed: %s: %s", chat_id, attempt, type(e).__name__, e)
                continue
            if title := clean_title(raw):
                return title
            log.warning("chat %s: title attempt %d gave nothing usable: %r", chat_id, attempt, raw[:80])
        return None

    async def _write(self, chat_id: str, title: str, seen: tuple[bool, str]) -> Chat:
        """Set the title if the chat still has the title and flag the model saw (a compare-and-set in one statement,
        so a rename that committed meanwhile wins); returns the chat as it is afterwards."""
        was_auto, was_title = seen
        async with self.db.session() as s:
            result = await s.execute(
                update(orm.Chat)
                .where(orm.Chat.id == chat_id, orm.Chat.title_is_auto == was_auto, orm.Chat.title == was_title)
                .values(title=title, title_is_auto=True, updated_at=self.now())
                .execution_options(synchronize_session=False)
            )
            if result.rowcount == 0:  # type: ignore[attr-defined]
                log.info("chat %s: changed while its title was being generated; title not written", chat_id)
            chat = await get_or_404(s, orm.Chat, chat_id, "chat")
            return Chat.model_validate(chat)
