"""Conversation state (docs/DESIGN.md §3.5): read before a turn, advanced from the turn's validated route after it.

Only application code changes the state, and only through ``advance`` (a pure function of the state and what the
turn did) or the explicit setters here. The router model proposes routes; it never writes state.

How a turn moves the state (``advance``):

- **stop / backchannel**: nothing changes, except that an answer the user had just cut off stays the interrupted one
  (so "stop" followed by "no, I meant FY23" still corrects it).
- **topic**: document, general, mixed and correction turns set the topic. A topic shift moves the active topic to
  ``previous_topic``; ``resume_document`` makes the last document topic active again. Conversation (thanks, small
  talk) and clarification turns keep the topic.
- **documents**: a turn that searched the documents records its topic and standalone question as the document topic
  and question ("back to the report" returns to them) and, when its answer cites documents, those documents as the
  active ones.
- **languages**: the user's input language and the answer's language; a language the user asked for becomes the
  preferred one and stays until they ask for another (``services/language.py``).
- **interruption**: a turn whose answer was stopped (barge-in, stop, client gone) becomes the last interrupted answer;
  a later answer that completes clears it.

Writes are ordered per chat (``StateWrites``): each turn's update runs as its own task after the previous one and is
applied to the state that one left, and a turn reads the state only once pending updates are written, so concurrent
turns (a barge-in's stopped answer, then the next question) neither read a stale state nor overwrite a newer one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.exc import IntegrityError

from ..db import models as orm
from ..domain.conversation import SILENT_INTENTS, TOPIC_NEUTRAL_INTENTS, ConversationState, TurnRoute
from ..settings import Language
from .base import Service, get_or_404

log = logging.getLogger(__name__)

TOPIC_MAX = 200

_FIELDS = tuple(f for f in ConversationState.model_fields if f not in ("chat_id", "updated_at"))


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """What a finished turn did, as far as the conversation state is concerned."""

    route: TurnRoute
    query: str | None = None  # the standalone question the turn answered (document turns)
    input_language: Language | None = None
    asked_language: Language | None = None  # the user asked for this language in the turn
    searched: bool = False  # the documents were searched for this turn
    cited_document_ids: list[str] = field(default_factory=list)
    completed: bool = True  # False: the answer was stopped part-way
    message_id: str | None = None  # the saved answer
    interrupted_message_id: str | None = None  # the answer the user had cut off before this turn, if any


def _topic(value: str | None) -> str | None:
    value = (value or "").strip()
    return value[:TOPIC_MAX] or None


def advance(state: ConversationState, outcome: TurnOutcome) -> ConversationState:
    """The state after a turn (see the module docstring). Pure: returns a new state."""
    route = outcome.route
    changes: dict[str, object] = {}
    if route.intent in SILENT_INTENTS:
        if outcome.interrupted_message_id is not None:
            changes["last_interrupted_message_id"] = outcome.interrupted_message_id
        if outcome.asked_language is not None:
            changes["preferred_language"] = outcome.asked_language
        return state.model_copy(update=changes)

    topic = _topic(route.topic)
    active, previous = state.active_topic, state.previous_topic
    if route.intent == "resume_document":
        target = state.document_topic or topic or active
        if target != active:
            active, previous = target, active
    elif route.intent not in TOPIC_NEUTRAL_INTENTS and topic is not None:
        if active is None:
            active = topic
        elif topic != active and route.is_topic_shift:
            active, previous = topic, active
    changes.update(active_topic=active, previous_topic=previous)

    if outcome.searched:
        changes["document_topic"] = active
        if outcome.query:
            changes["document_query"] = outcome.query
        if outcome.cited_document_ids:
            changes["active_document_ids"] = list(dict.fromkeys(outcome.cited_document_ids))

    if outcome.input_language is not None:
        changes["input_language"] = outcome.input_language
    changes["response_language"] = route.response_language
    if outcome.asked_language is not None:
        changes["preferred_language"] = outcome.asked_language
    changes["last_intent"] = route.intent
    changes["last_interrupted_message_id"] = None if outcome.completed else outcome.message_id
    return state.model_copy(update=changes)


class ConversationStateService(Service):
    async def get(self, chat_id: str) -> ConversationState:
        """The chat's state; a chat without one yet gets the defaults (nothing is written)."""
        async with self.db.session() as s:
            row = await s.get(orm.ChatState, chat_id)
            if row is None:
                await get_or_404(s, orm.Chat, chat_id, "chat")
                return ConversationState(chat_id=chat_id)
            return ConversationState.model_validate(row)

    async def save(self, state: ConversationState) -> ConversationState:
        """Write the whole state (insert or update)."""
        values = {f: getattr(state, f) for f in _FIELDS}
        for attempt in range(2):
            try:
                async with self.db.session() as s:
                    row = await s.get(orm.ChatState, state.chat_id)
                    if row is None:
                        row = orm.ChatState(chat_id=state.chat_id)
                        s.add(row)
                    for name, value in values.items():
                        setattr(row, name, value)
                    row.updated_at = self.now()
                    await s.flush()
                    return ConversationState.model_validate(row)
            except IntegrityError:  # another turn created the row first: update it instead
                if attempt:
                    raise
        raise AssertionError("unreachable")

    async def apply(self, chat_id: str, outcome: TurnOutcome) -> ConversationState:
        """Advance the chat's current state by a finished turn and save it."""
        return await self.save(advance(await self.get(chat_id), outcome))

    async def set_retrieval_enabled(self, chat_id: str, enabled: bool) -> ConversationState:
        """Turn document search on or off for the chat (off: document questions get general answers that say so)."""
        await state_writes().settled(chat_id)
        state = await self.get(chat_id)
        return await self.save(state.model_copy(update={"retrieval_enabled": enabled}))


# ------------------------------------------------------------------ ordering of state writes


class StateWrites:
    """A chat's state updates run off the answer's critical path (as their own tasks, so a stop can't cancel them),
    one after the other, each applied to the state the previous one left; a turn reads the state only once they are
    done (``settled``), so it never sees a stale one and never overwrites a newer one."""

    def __init__(self) -> None:
        self.pending: dict[str, asyncio.Task[Any]] = {}

    def schedule(self, chat_id: str, update: Callable[[], Awaitable[Any]], *, spawn: Spawn) -> asyncio.Task[Any]:
        previous = self.pending.get(chat_id)

        async def run() -> None:
            if previous is not None:
                await asyncio.wait([previous])
            try:
                await update()
            except Exception:
                log.exception("chat %s: saving the conversation state failed", chat_id)

        task = spawn(run())
        self.pending[chat_id] = task
        task.add_done_callback(lambda t: self.pending.pop(chat_id) if self.pending.get(chat_id) is t else None)
        return task

    async def settled(self, chat_id: str) -> None:
        task = self.pending.get(chat_id)
        if task is not None:
            await asyncio.wait([task])


Spawn = Callable[[Coroutine[Any, Any, Any]], "asyncio.Task[Any]"]


def loop_local[T](registry: dict[int, tuple[asyncio.AbstractEventLoop, Any]], factory: Callable[[], T]) -> T:
    """The running event loop's instance in ``registry`` (created by ``factory``); entries of closed loops (tests)
    are dropped."""
    loop = asyncio.get_running_loop()
    for key, (other, _) in list(registry.items()):
        if other.is_closed():
            del registry[key]
    entry = registry.get(id(loop))
    if entry is None or entry[0] is not loop:
        entry = registry[id(loop)] = (loop, factory())
    return entry[1]


_WRITES: dict[int, tuple[asyncio.AbstractEventLoop, Any]] = {}


def state_writes() -> StateWrites:
    return loop_local(_WRITES, StateWrites)
