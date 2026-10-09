"""The chat's memory summary (docs/DESIGN.md §3.5): a compact, internal digest of the conversation, so prompts carry
recent messages + this summary + evidence + the utterance, never the whole transcript. (The user-facing summary,
``SummaryKind`` "user", is a different thing.)

When: after a turn completes, in the background, once ``llm.memory_summary_every_turns`` exchanges or
``llm.memory_summary_token_budget`` (estimated) tokens of messages are not covered yet, and only when the chat has
more messages than the prompt's recent window (otherwise the window already holds everything). Each refresh folds the
new messages into the previous summary, so its cost doesn't grow with the chat.

Never on the critical path: one local LLM serves turns and summaries, so a summary starts only when no turn is
answering, and gives way (is cancelled, to be retried after the next turn) the moment one starts: ``ChatActivity``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select

from ..db import models as orm
from ..domain.projects import Message
from ..providers.llm import LLMClient, LLMMessage
from ..providers.storage import MetadataDB
from ..settings import Settings
from .base import NotFound, Service
from .conversation import loop_local
from .messages import MessageService
from .prompts import MEMORY_PROMPT_VERSION, memory_system_prompt, memory_user_prompt
from .revisit_prompts import INTERRUPTED_LINE
from .router import heard
from .sources import estimate_tokens, strip_markers
from .summaries import SummaryService

log = logging.getLogger(__name__)

MEMORY_MAX_TOKENS = 200  # ~6 s of generation at 34 tok/s: short enough to finish while the answer is being spoken
MEMORY_TEMPERATURE = 0.1
MESSAGE_CHARS = 600  # per message in the summary prompt
MAX_NEW_MESSAGES = 60  # a long chat without a summary yet: fold in its latest messages only


class MemoryKeeper(Service):
    """Reads and refreshes a chat's memory summary."""

    def __init__(self, db: MetadataDB, llm: LLMClient, settings: Settings, *, window: int) -> None:
        super().__init__(db)
        self.llm = llm
        self.settings = settings
        self.window = window  # messages the answer prompt already carries
        self.summaries = SummaryService(db)
        self.messages = MessageService(db)

    async def current(self, chat_id: str) -> str | None:
        """The memory summary's text for prompts (None before the first one)."""
        summary = await self.summaries.get(chat_id, "memory")
        return summary.content if summary is not None else None

    async def pending(self, chat_id: str) -> tuple[str | None, list[Message]] | None:
        """(previous summary, messages to fold in) when a refresh is due, else None."""
        previous = await self.summaries.get(chat_id, "memory")
        covered = await self._seq(previous.covers_message_id) if previous is not None else 0
        page = await self.messages.list(chat_id, after=covered, limit=MAX_NEW_MESSAGES)
        if page.total <= self.window:
            return None
        if page.has_more:  # far behind (a long chat without a summary yet): fold in the latest messages
            page = await self.messages.list(chat_id, before=page.total + 1, limit=MAX_NEW_MESSAGES)
        new = [m for m in page.items if m.role in ("user", "agent")][-MAX_NEW_MESSAGES:]
        exchanges = sum(m.role == "user" for m in new)
        tokens = sum(estimate_tokens(heard(m)) for m in new)
        cfg = self.settings.llm
        if exchanges < cfg.memory_summary_every_turns and tokens < cfg.memory_summary_token_budget:
            return None
        if not new or new[-1].role != "agent":  # a turn still in flight (or failed): wait for its answer
            return None
        return (previous.content if previous is not None else None), new

    async def refresh_if_due(self, chat_id: str) -> bool:
        """Fold the uncovered messages into the summary if it is due. True when a new summary was saved."""
        try:
            due = await self.pending(chat_id)
        except NotFound:  # the chat was deleted meanwhile
            return False
        if due is None:
            return False
        previous, new = due
        prompt = [
            LLMMessage("system", memory_system_prompt()),
            LLMMessage("user", memory_user_prompt(previous, transcript(new))),
        ]
        model = self.settings.llm.chat_model
        text = (
            await self.llm.generate(prompt, model=model, temperature=MEMORY_TEMPERATURE, max_tokens=MEMORY_MAX_TOKENS)
        ).strip()
        if not text:
            return False
        save = self.summaries.save(
            chat_id,
            kind="memory",
            content=text,
            data={"prompt": MEMORY_PROMPT_VERSION, "messages": len(new)},
            covers_message_id=new[-1].id,
            model=model,
        )
        with contextlib.suppress(NotFound):
            await asyncio.shield(save)  # the model's work is done: a turn starting now doesn't throw it away
            log.info("chat %s: memory summary updated (%d new messages)", chat_id, len(new))
            return True
        return False

    async def _seq(self, message_id: str | None) -> int:
        if message_id is None:
            return 0
        async with self.db.session() as s:
            return await s.scalar(select(orm.Message.seq).where(orm.Message.id == message_id)) or 0


def transcript(messages: list[Message]) -> str:
    """Messages as "User: …" / "Assistant: …" lines for the summary prompt, with the pages answers cited."""
    lines = []
    for m in messages:
        if m.role == "agent" and m.heard_text is not None:
            # Cut off (B7): a fragment ("Product…") isn't a fact to remember, only that the answer was cut.
            if m.heard_text.strip():
                lines.append(f"Assistant: {INTERRUPTED_LINE}")
            continue
        text = " ".join(strip_markers(heard(m)).split())
        if not text:  # an answer nobody heard
            continue
        if len(text) > MESSAGE_CHARS:
            text = text[: MESSAGE_CHARS - 1] + "…"
        if m.role == "agent":
            pages = [c for c in m.citations if c.kind == "document" and c.page_start is not None]
            cited = sorted({f"{c.filename} p.{c.page_start}" for c in pages})
            web = sorted({f"web: {c.site or c.url}" for c in m.citations if c.kind == "web"})  # live data, §3.7
            if cited or web:
                text += f" (sources: {', '.join([*cited, *web])})"
        lines.append(f"{'User' if m.role == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


# ------------------------------------------------------------------ scheduling

Job = Callable[[], Awaitable[Any]]
Spawn = Callable[[Awaitable[Any]], "asyncio.Task[Any]"]


class ChatActivity:
    """Which chats have a turn answering, and the memory summaries waiting for the LLM to be free.

    ``turn_started`` cancels any summary still running (the turn has priority on the shared model);
    ``turn_finished`` queues the chat's summary job, and queued jobs start once no turn is answering anywhere (one per
    chat at a time). One instance per event loop (``chat_activity()``)."""

    def __init__(self) -> None:
        self.active: dict[str, int] = {}
        self.jobs: dict[str, asyncio.Task[Any]] = {}  # running
        self.waiting: dict[str, tuple[Job, Spawn]] = {}  # queued until no turn is answering

    def turn_started(self, chat_id: str) -> None:
        self.active[chat_id] = self.active.get(chat_id, 0) + 1
        self.interrupt()

    def interrupt(self) -> None:
        """Cancel running summaries now (they are retried after the next turn). For a caller that knows a turn is
        coming before it starts, e.g. the voice session when the user starts speaking (speech recognition shares the
        GPU with the model)."""
        for task in self.jobs.values():
            task.cancel()

    def turn_finished(self, chat_id: str, job: Job, *, spawn: Spawn = asyncio.ensure_future) -> None:
        count = self.active.get(chat_id, 0) - 1
        if count > 0:
            self.active[chat_id] = count
        else:
            self.active.pop(chat_id, None)
        self.waiting[chat_id] = (job, spawn)
        if self.active:
            return
        for waiting_chat in [c for c in self.waiting if c not in self.jobs or self.jobs[c].done()]:
            job_, spawn_ = self.waiting.pop(waiting_chat)
            task = spawn_(self._run(waiting_chat, job_))
            self.jobs[waiting_chat] = task
            # Removed when the task ends, even if it was cancelled before it ever ran.
            task.add_done_callback(lambda t, c=waiting_chat: self.jobs.pop(c) if self.jobs.get(c) is t else None)

    @staticmethod
    async def _run(chat_id: str, job: Job) -> None:
        try:
            await job()
        except asyncio.CancelledError:
            log.debug("chat %s: memory summary gave way to a turn", chat_id)
        except Exception as e:  # off the critical path: a failed summary only means a longer prompt later
            log.warning("chat %s: memory summary failed: %s: %s", chat_id, type(e).__name__, e)

    async def idle(self) -> None:
        """Wait until no summary job is running (tests)."""
        while self.jobs:
            await asyncio.wait(list(self.jobs.values()))


_ACTIVITY: dict[int, tuple[asyncio.AbstractEventLoop, Any]] = {}


def chat_activity() -> ChatActivity:
    """The running event loop's ``ChatActivity`` (shared by every ``ChatTurnService`` built on it)."""
    return loop_local(_ACTIVITY, ChatActivity)
