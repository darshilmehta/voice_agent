"""One turn of a chat (text or voice) as a stream of typed events (docs/DESIGN.md §3.2-§3.6, §3.9).

    save the user message                                        → UserMessageEvent
    plan the turn: route + retrieval policy (services/planning.py: a keyword fast path, else the router model
      alongside a speculative retrieval; a timeout or failure falls back to a document question, §3.4)
    document turns: retrieve within the chat's project and document scope (READY documents only)
      abstention gate: no evidence above the threshold → fixed answer in the user's language, no LLM
      numbered sources [S1]… (dedupe, sections, context budget)  → SourcesEvent
    other turns: no retrieval                                    → SourcesEvent (no sources, not abstained)
    the answer streamed from the LLM (grounded, mixed, general, conversation or clarification prompt), or a fixed
      text ("back to the report…", "Anything else?"), or nothing (stop) → DeltaEvent …
    citations validated, message saved with route and latency, conversation state advanced → AgentMessageEvent
    any failure                                                  → ErrorEvent (stage: retrieval | llm | storage), end

Live data (§3.7): a route with ``tools=["web_search"]`` starts the web search (services/web_search.py) right after
planning, so it runs beside document retrieval                   → ToolEvent start (the query that leaves the machine)
    the first web results (partial_wait_ms after the first one, bounded by timeout_s) → ToolEvent results [W1]…
    sources: document passages [S#] and web results [W#] (``kind: "web"``) → SourcesEvent
    the answer from both (live prompt: each fact cited and said as from the report or from the web) → DeltaEvent …
    results that arrived meanwhile: at most ``max_continuations`` short continuations → ToolEvent results, DeltaEvent
    the search ends                                              → ToolEvent done | timeout | failed
    no web results (timeout, failure) or the tool unavailable: the answer starts with a fixed notice ("I couldn't get
    live data just now.") and answers from the documents (or abstains, or answers from general knowledge) as before.
Stopping the turn cancels the search and its page fetches with it.

Every turn that doesn't fail ends with ``AgentMessageEvent``. A "stop" turn says nothing: no ``DeltaEvent``, and the
message it ends with has role ``event`` (a short notice for the transcript, carrying the route).

Prompts carry recent messages + the chat's memory summary + evidence + the utterance, never the whole transcript
(§3.5); the memory summary is refreshed in the background after turns, never while one is answering
(services/memory.py). The conversation state (services/conversation.py) is read before the turn and advanced from
its validated route when its answer is saved.

Transport-agnostic: ``ChatTurnService`` knows nothing about HTTP. The SSE endpoint (``api/chats.py``) only serialises
the events (``event.name`` + ``event.payload()``); the voice WebSocket (phase 4-6) runs the same turn with
``modality="voice"`` and feeds the ``DeltaEvent`` texts to TTS. The answer's length is a parameter
(``length="short"``: 1-3 speakable sentences, details left to the on-screen citations; ``"full"``: fuller text).

The user message stays saved when a later stage fails; a failed answer is not saved. To stop an answer (client gone,
barge-in), cancel the task consuming the events or ``await events.aclose()``: generation is cancelled (the LLM stream
is closed) and the text generated so far is saved with ``route.stopped = true``, citing only what that text cites;
nothing is saved if no text was generated yet. Every saved agent message's ``route`` carries top-level ``abstained``
and ``stopped`` booleans (``abstained``: a document question the documents couldn't answer; never a turn that
deliberately didn't search) and the router's decision (intent, rewritten and English queries, topic, shift,
language). A caller that knows what the user actually heard of the answer (the voice session, §3.3 c) passes an
``AnswerStop`` to ``run``: the stopped answer is saved with that ``heard_text`` and ``route.interrupted`` (the
reason), and the saved message is handed back on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import re
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ..domain.conversation import ConversationState
from ..domain.projects import Chat, Citation, Message, Modality
from ..providers.base import PlaceholderProvider
from ..providers.llm import LLMClient, LLMMessage
from ..providers.registry import Container
from ..providers.storage import MetadataDB
from ..providers.web_search import WebSearch
from ..settings import Language, Settings
from .base import InvalidInput
from .chats import ChatService
from .conversation import ConversationStateService, TurnOutcome, state_writes
from .documents import DocumentService
from .language import asked_language, decide_language, message_language
from .memory import MemoryKeeper, chat_activity
from .messages import MessageService
from .planning import DocumentQARouter, TurnPlan, TurnPlanner, interrupted_answer
from .prompts import (
    ANSWER_LENGTHS,
    CONTINUATION_NOTHING,
    CONTINUATION_PROMPT_VERSION,
    LIVE_PROMPT_VERSION,
    PROMPT_IDS,
    SILENT_NOTICES,
    AbstainReason,
    AnswerLength,
    DocumentsPart,
    abstention,
    ack_text,
    answer_system_prompt,
    answer_user_prompt,
    clarification_system_prompt,
    continuation_user_prompt,
    conversation_system_prompt,
    general_system_prompt,
    general_user_prompt,
    live_notice,
    live_system_prompt,
    live_user_prefix,
    live_user_prompt,
    resume_text,
    with_memory,
)
from .retrieval import Confidence, RetrievalResult, RetrievalService, SpeculationOutcome
from .router import LLMTurnRouter, RouteRequest, TurnRouter, heard
from .sources import Source, build_sources, finalize_answer, strip_markers, trim_open_marker
from .web_search import ToolEvent, WebSearchRun, WebSource

log = logging.getLogger(__name__)

__all__ = [  # the turn API used by the transports, and the router slot (phase 1's stand-in is kept for tests)
    "AgentMessageEvent",
    "AnswerStop",
    "ChatEvent",
    "ChatTurnService",
    "DeltaEvent",
    "DocumentQARouter",
    "ErrorEvent",
    "SourcesEvent",
    "ToolEvent",
    "Turn",
    "TurnPlan",
    "TurnRouter",
    "UserMessageEvent",
    "detach",
    "wait_for_background",
]

TEXT_MAX = 4000
HISTORY_MESSAGES = 6  # recent messages sent as conversation context (three exchanges)
HISTORY_CHARS = 1000  # per message
SHORT_REPLY_TOKENS = 96  # conversation replies and clarifying questions: one sentence
CONTINUATION_TOKENS = 96  # one sentence about results that arrived after the answer started (§3.7)
# The end of a continuation's first sentence: a full stop before a capitalised word or a Devanagari one ("Rs. 997"
# and "U.S. dollar" don't end it).
_FIRST_SENTENCE = re.compile("(?<=[.!?।])\\s+(?=[A-Z\"'ऀ-ॿ])")

Stage = Literal["retrieval", "llm", "storage"]

# Saves of stopped answers, voice session clean-ups and memory summaries run as their own tasks, so a cancelled
# request can't interrupt them; keep them referenced.
_BACKGROUND: set[asyncio.Task[Any]] = set()


def detach(coro: Any) -> asyncio.Task[Any]:
    """Run ``coro`` as its own task that the caller's cancellation can't interrupt (await it with asyncio.shield).
    Needed under anyio cancel scopes, whose cancellation fires again at every await of the cancelled task."""
    task = asyncio.ensure_future(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return task


async def wait_for_background() -> None:
    """Wait for pending saves of stopped answers, session clean-ups and memory summaries of this event loop (tests,
    shutdown)."""
    loop = asyncio.get_running_loop()
    while pending := [t for t in _BACKGROUND if t.get_loop() is loop and not t.done()]:
        await asyncio.wait(pending)


# ------------------------------------------------------------------ events


def confidence_json(c: Confidence | None) -> dict[str, Any]:
    if c is None:
        return {"top_score": 0.0, "gap": 0.0, "dense_similarity": 0.0, "above_threshold": False}
    return {
        "top_score": round(c.top_score, 4),
        "gap": round(c.gap, 4),
        "dense_similarity": round(c.dense_similarity or 0.0, 4),
        "above_threshold": c.above_threshold,
    }


@dataclass(frozen=True, slots=True)
class UserMessageEvent:
    """The user's message, saved."""

    message: Message
    name: ClassVar[str] = "user_message"

    def payload(self) -> dict[str, Any]:
        return self.message.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """The numbered sources the answer may cite (none when abstaining or not searching) and the retrieval
    confidence. Document passages [S#] first, then the web results the answer starts with [W#] (``kind: "web"``)."""

    sources: list[Citation]
    confidence: Confidence | None
    abstained: bool
    name: ClassVar[str] = "sources"

    def payload(self) -> dict[str, Any]:
        return {
            "sources": [c.model_dump(mode="json") for c in self.sources],
            "confidence": confidence_json(self.confidence),
            "abstained": self.abstained,
        }


@dataclass(frozen=True, slots=True)
class DeltaEvent:
    """The next piece of the answer's text."""

    text: str
    name: ClassVar[str] = "delta"

    def payload(self) -> dict[str, Any]:
        return {"text": self.text}


@dataclass(frozen=True, slots=True)
class AgentMessageEvent:
    """The agent's answer, saved: final text and the citations it actually makes. (A "stop" turn: a message with role
    ``event`` and no answer.)"""

    message: Message
    name: ClassVar[str] = "agent_message"

    def payload(self) -> dict[str, Any]:
        return self.message.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """The turn failed at ``stage``; no event follows."""

    stage: Stage
    detail: str
    name: ClassVar[str] = "error"

    def payload(self) -> dict[str, Any]:
        return {"detail": self.detail, "stage": self.stage}


ChatEvent = UserMessageEvent | SourcesEvent | DeltaEvent | AgentMessageEvent | ErrorEvent | ToolEvent


# ------------------------------------------------------------------ turn


@dataclass(frozen=True, slots=True)
class Turn:
    """A validated user message for a chat, before anything is saved."""

    chat: Chat
    text: str
    language: Language  # the answer's language
    requested_language: Language | None
    modality: Modality = "text"
    length: AnswerLength = "short"
    input_language: Language | None = None  # the language the user spoke (STT); None: read from the text's script
    input_latency: dict[str, Any] | None = None  # saved as the user message's latency (voice: VAD and STT timings)


# Why a voice answer was cut short: the user talked over it, pressed stop, or the session ended (client gone, replaced
# by a newer session for the chat, server shutdown).
InterruptReason = Literal["barge_in", "stop", "disconnect"]


@dataclass(eq=False)
class AnswerStop:
    """Lets a caller that cuts an answer short (voice: barge-in or stop, §3.3 c) record what the user heard of it.

    Set ``heard_text`` and ``reason`` before cancelling the consumer of ``run``. The stopped answer is then saved with
    ``heard_text`` and ``route.interrupted = reason`` (with the text generated so far, which may be empty: a stopped
    voice turn always ends with an agent message), and once the cancelled consumer has finished:

    - ``saved`` is that message (None if the save failed or the answer had already been saved complete);
    - ``completed`` is the complete answer when it was saved (or its save had already started) before the stop;
    - ``user`` is the turn's user message (set as soon as it is saved, or passed to ``run``).
    """

    heard_text: str | None = None
    reason: InterruptReason | None = None
    saved: Message | None = None
    completed: Message | None = None
    user: Message | None = None


@dataclass
class _Progress:
    """How far a turn's answer got, so a stopped answer can be saved with what it had, and what the turn read before
    planning (for the route and the conversation state)."""

    plan: TurnPlan
    result: RetrievalResult | None = None
    sources: list[Source] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)  # answer text so far
    abstained: bool = False
    reason: AbstainReason | None = None
    retrieval_ms: float | None = None
    first_delta_ms: float | None = None
    llm_start: float | None = None
    saving: bool = False  # the complete answer is being saved: a stop must not save a second copy
    final_save: asyncio.Task[Message] | None = None  # that save, as its own task: a stop waits for it
    final: Message | None = None  # the complete answer, saved
    state: ConversationState | None = None  # None: stopped before it was read
    memory: str | None = None  # the chat's memory summary
    ready: Mapping[str, str] = field(default_factory=dict)  # READY documents: id → filename
    speculation: SpeculationOutcome = "none"
    web: WebSearchRun | None = None  # the turn's web search (§3.7)
    web_sources: list[WebSource] = field(default_factory=list)  # web results given to the model, [W1]…
    documents_part: DocumentsPart = "none"  # what the live prompt says about the documents
    continuations: int = 0  # continuation sentences added after the answer
    warm: asyncio.Task[Any] | None = None  # the live prompt's prefix, read by the model while the web is searched

    @property
    def citable(self) -> list[Source | WebSource]:
        return [*self.sources, *self.web_sources]


@dataclass
class _Clock:
    start: float = field(default_factory=time.perf_counter)

    def ms(self, since: float | None = None) -> float:
        return round((time.perf_counter() - (self.start if since is None else since)) * 1000, 1)


def _describe(e: BaseException) -> str:
    text = str(e)
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


# ------------------------------------------------------------------ service


class ChatTurnService:
    """Runs chat turns for any transport::

    turn = await service.begin(chat_id, text, language=None, modality="voice", length="short")  # may raise
    async for event in service.run(turn):  # UserMessageEvent, SourcesEvent, DeltaEvent…, AgentMessageEvent
        ...                                # or ErrorEvent; cancel or aclose() to stop the answer
    """

    def __init__(
        self,
        db: MetadataDB,
        *,
        retrieval: RetrievalService,
        llm: LLMClient,
        settings: Settings,
        router: TurnRouter | None = None,
        web_search: WebSearch | None = None,
    ) -> None:
        self.chats = ChatService(db)
        self.messages = MessageService(db)
        self.documents = DocumentService(db)
        self.states = ConversationStateService(db)
        self.retrieval = retrieval
        self.llm = llm
        self.settings = settings
        self.web_search = web_search  # None: no live-data tool (questions that want live data say so)
        self.router: TurnRouter = router if router is not None else LLMTurnRouter(llm)
        self.planner = TurnPlanner(retrieval, self.router, timeout_s=settings.llm.router_timeout_ms / 1000)
        self.memory = MemoryKeeper(db, llm, settings, window=HISTORY_MESSAGES)

    @classmethod
    def from_container(cls, container: Container) -> ChatTurnService:
        db, llm = container["metadata_db"], container["llm"]
        if not isinstance(db, MetadataDB) or not isinstance(llm, LLMClient):
            raise TypeError(
                f"expected MetadataDB and LLMClient providers, got {type(db).__name__}, {type(llm).__name__}"
            )
        web = container.providers.get("web_search")
        web_search = web if isinstance(web, WebSearch) and not isinstance(web, PlaceholderProvider) else None
        return cls(
            db,
            retrieval=RetrievalService.from_container(container),
            llm=llm,
            settings=container.settings,
            web_search=web_search,
        )

    def available_tools(self) -> frozenset[str]:
        """The live-data tools that can run now (§3.7): web search when configured, allowed and reachable."""
        if self.web_search is None or self.web_search.unavailable_reason() is not None:
            return frozenset()
        return frozenset({"web_search"})

    async def begin(
        self,
        chat_id: str,
        text: str,
        *,
        language: Language | None = None,
        modality: Modality = "text",
        length: AnswerLength = "short",
        input_language: Language | None = None,
        input_latency: dict[str, Any] | None = None,
    ) -> Turn:
        """Check the chat and the message before any event is produced (unknown chat → NotFound, bad text →
        InvalidInput). The answer language follows ``services/language.py``: a language the user asks for (now or
        earlier in the chat), ``language`` when the caller forces one, else the language of the message (the spoken
        one for voice), else the previous answer's. ``modality`` is recorded on both messages; ``length`` sets the
        answer style and token cap. Voice turns pass the spoken language (``input_language``, saved on the user
        message) and the STT timings (``input_latency``)."""
        chat = await self.chats.get(chat_id)
        text = text.strip()
        if not text:
            raise InvalidInput("text must not be empty")
        if len(text) > TEXT_MAX:
            raise InvalidInput(f"text is longer than {TEXT_MAX} characters")
        if length not in ANSWER_LENGTHS:
            raise InvalidInput(f"length must be one of {', '.join(ANSWER_LENGTHS)}")
        await state_writes().settled(chat.id)  # the previous turn's state update, if still being written
        state = await self.states.get(chat.id)
        decision = decide_language(
            text,
            requested=language,
            spoken=input_language,
            preferred=state.preferred_language,
            fallback=self._fallback_language(chat, state),
        )
        return Turn(chat, text, decision.language, language, modality, length, input_language, input_latency)

    def _fallback_language(self, chat: Chat, state: ConversationState) -> Language:
        if state.response_language is not None:
            return state.response_language
        return chat.language if chat.language in ("en", "hi") else self.settings.client.default_language  # type: ignore[return-value]

    async def save_user_message(self, turn: Turn) -> Message:
        """Save the turn's user message. ``run`` does it itself unless the caller saved it first (voice: so that a
        stop arriving while the turn starts can't lose it)."""
        return await self.messages.append(
            turn.chat.id,
            role="user",
            text=turn.text,
            modality=turn.modality,
            language=turn.input_language or message_language(turn.text) or turn.language,
            latency=turn.input_latency,
        )

    async def save_unanswered(self, turn: Turn, *, heard_text: str, reason: InterruptReason) -> Message:
        """Close a turn that was stopped before its answer had any text (voice): an empty agent message with
        ``route.stopped``, so the transcript shows the question was stopped, not lost."""
        p = _Progress(TurnPlan(query=turn.text, language=turn.language))
        route = self._route(turn, p, abstained=False, reason=None, stopped=True)
        route["interrupted"] = reason
        return await self.messages.append(
            turn.chat.id,
            role="agent",
            text="",
            modality=turn.modality,
            heard_text=heard_text,
            language=turn.language,
            route=route,
        )

    async def run(
        self, turn: Turn, *, stop: AnswerStop | None = None, user: Message | None = None
    ) -> AsyncGenerator[ChatEvent, None]:
        """The turn's events. ``stop``: see ``AnswerStop`` (callers that know what was heard of a stopped answer).
        ``user``: the user message, if the caller already saved it with ``save_user_message``. While the turn runs,
        the chat's memory summary waits (or gives way if it is running): one LLM serves both."""
        activity = chat_activity()
        activity.turn_started(turn.chat.id)
        try:
            async with contextlib.aclosing(self._run(turn, stop, user)) as events:
                async for event in events:
                    yield event
        finally:
            job = functools.partial(self.memory.refresh_if_due, turn.chat.id)
            activity.turn_finished(turn.chat.id, job, spawn=detach)

    async def _run(self, turn: Turn, stop: AnswerStop | None, user: Message | None) -> AsyncGenerator[ChatEvent, None]:
        clock = _Clock()
        chat = turn.chat
        if user is None:
            try:
                user = await self.save_user_message(turn)
            except Exception as e:
                log.exception("chat %s: saving the user message failed", chat.id)
                yield ErrorEvent("storage", f"could not save the message: {_describe(e)}")
                return
        if stop is not None:
            stop.user = user
        try:
            history = await self._history(chat.id, before=user.seq)
        except Exception as e:
            log.exception("chat %s: reading the chat history failed", chat.id)
            yield ErrorEvent("storage", f"could not read the chat history: {_describe(e)}")
            return
        p = _Progress(TurnPlan(query=turn.text, language=turn.language))
        try:
            yield UserMessageEvent(user)
            async with contextlib.aclosing(self._answer(turn, history, clock, p)) as answer:
                async for event in answer:
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            # Stopped (client gone, barge-in, "stop"): keep what was generated. Saves run as their own tasks, so a
            # consumer whose cancellation keeps re-firing (anyio cancel scopes) can't interrupt them; this frame waits
            # if it can. A caller passing ``stop`` (voice) always gets a saved answer, even an empty one.
            if p.final_save is not None:  # the complete answer was being saved: wait for it instead
                with contextlib.suppress(BaseException):
                    await asyncio.shield(p.final_save)
                if p.final_save.done() and not p.final_save.cancelled() and p.final_save.exception() is None:
                    p.final = p.final_save.result()
            if p.final is not None:
                if stop is not None:
                    stop.completed = p.final
            elif not p.saving and (p.parts or stop is not None):
                save = detach(self._save_stopped(turn, p, clock, stop))
                with contextlib.suppress(BaseException):
                    await asyncio.shield(save)
            raise

    async def _answer(
        self, turn: Turn, history: Sequence[Message], clock: _Clock, p: _Progress
    ) -> AsyncGenerator[ChatEvent, None]:
        """Everything after the user message, recording its progress in ``p`` so that ``run`` can save a stopped
        answer. ``run`` closes this generator when it is closed, which closes the LLM stream (stops generation)."""
        chat = turn.chat
        try:
            await state_writes().settled(chat.id)
            p.state = await self.states.get(chat.id)
            p.memory = await self.memory.current(chat.id)
        except Exception as e:
            log.exception("chat %s: reading the conversation state failed", chat.id)
            yield ErrorEvent("storage", f"could not read the conversation state: {_describe(e)}")
            return
        try:
            ready = p.ready = await self.documents.ready_documents(chat.project_id, chat.document_scope)
        except Exception as e:
            log.warning("chat %s: listing the documents failed: %s", chat.id, _describe(e))
            yield ErrorEvent("retrieval", f"document search failed: {_describe(e)}")
            return
        state = p.state
        request = RouteRequest(
            turn.text,
            turn.language,
            history,
            state,
            list(ready.values()),
            interrupted_answer(history),
            available_tools=self.available_tools(),
        )
        plan = p.plan = await self.planner.plan(
            request, project_id=chat.project_id, ready=ready, retrieval_enabled=state.retrieval_enabled
        )
        if plan.speculated and plan.speculation is None:
            p.speculation = "discarded"  # the route needs no retrieval

        if plan.mode == "silent":
            yield SourcesEvent([], None, abstained=False)
            async for event in self._save_silent(turn, p, clock):
                yield event
            return
        if plan.mode in ("resume", "ack"):
            answer = self._fixed_reply(plan, p)
            yield SourcesEvent([], None, abstained=False)
            p.parts.append(answer)
            yield DeltaEvent(answer)
            route = self._route(turn, p, abstained=False, reason=None, model_used=False)
            latency = self._latency(clock, p, first_delta_ms=clock.ms(), llm_ms=None)
            async for event in self._save_answer(turn, answer, [], route, latency, p):
                yield event
            return

        # The web search (if any) starts now and runs while the documents are searched; it ends with the turn.
        web = p.web = self._start_web(plan)
        try:
            if web is not None:
                yield ToolEvent("start", web.query)
            async with contextlib.aclosing(self._sourced_answer(turn, history, clock, p)) as answer:
                async for event in answer:
                    yield event
        finally:
            if p.warm is not None and not p.warm.done():  # stopped while waiting for the web
                p.warm.cancel()
            if web is not None:
                await web.aclose()

    async def _sourced_answer(
        self, turn: Turn, history: Sequence[Message], clock: _Clock, p: _Progress
    ) -> AsyncGenerator[ChatEvent, None]:
        """Retrieval (and the web's first results), the abstention gate, the sources, the answer and its
        continuation from later web results, the save."""
        chat, plan, ready, web = turn.chat, p.plan, p.ready, p.web
        confidence: Confidence | None = None
        if plan.needs_retrieval:
            try:
                result = p.result = await self._retrieve(plan, p, chat) if ready else None
            except Exception as e:
                log.warning("chat %s: retrieval failed: %s", chat.id, _describe(e))
                yield ErrorEvent("retrieval", f"document search failed: {_describe(e)}")
                return
            p.retrieval_ms = clock.ms()
            confidence = result.confidence if result is not None else None
        covered = plan.needs_retrieval and confidence is not None and confidence.above_threshold
        if covered:
            assert p.result is not None
            p.sources = build_sources(
                p.result.chunks, ready, budget_tokens=self.settings.retrieval.context_token_budget
            )
            p.documents_part = "sources"
        elif plan.needs_retrieval:
            p.documents_part = "not_covered"

        if web is not None:  # the answer starts on the first web results (or without them, saying so)
            if not web.results:  # waiting for the web: have the model read the documents meanwhile
                p.warm = self._warm_up(turn, plan, history, p)
            p.web_sources = await web.first_batch()
            if p.web_sources:
                yield web.results_event(p.web_sources)
            if (end := web.terminal_event()) is not None:
                yield end
            if not p.web_sources:
                plan = p.plan = plan.without_live_data("failed")
        web_citations = [w.citation() for w in p.web_sources]

        if plan.needs_retrieval and not covered:
            if p.web_sources:  # the documents don't answer it, the web may: say so and answer from the web
                yield SourcesEvent(web_citations, confidence, abstained=False)
            elif plan.mode == "mixed":  # the document part isn't covered: general knowledge, saying so
                plan = p.plan = plan.as_general("not_covered" if ready else "no_documents")
                yield SourcesEvent([], confidence, abstained=False)
            else:
                reason: AbstainReason = "no_documents" if not ready else "not_covered"
                p.abstained, p.reason = True, reason
                yield SourcesEvent([], confidence, abstained=True)
                notice = live_notice(plan.live_note, plan.language) + " " if plan.live_note else ""
                answer = notice + abstention(plan.language, reason)
                p.parts.append(answer)
                yield DeltaEvent(answer)
                route = self._route(turn, p, abstained=True, reason=reason, model_used=False)
                latency = self._latency(clock, p, first_delta_ms=clock.ms(), llm_ms=None)
                async for event in self._save_answer(turn, answer, [], route, latency, p):
                    yield event
                return
        elif covered:
            yield SourcesEvent([*(s.citation() for s in p.sources), *web_citations], confidence, abstained=False)
        else:
            yield SourcesEvent(web_citations, None, abstained=False)

        if plan.live_note is not None and not p.web_sources:  # live data asked for, none to give: say so first
            notice = live_notice(plan.live_note, plan.language) + " "
            p.first_delta_ms = clock.ms()
            p.parts.append(notice)
            yield DeltaEvent(notice)

        p.llm_start = time.perf_counter()
        prompt = self._prompt(turn, plan, history, p.sources, p.memory, p)
        short = plan.mode in ("conversation", "clarification")
        stream = self.llm.stream(
            prompt, max_tokens=SHORT_REPLY_TOKENS if short else ANSWER_LENGTHS[turn.length].max_tokens
        )
        model_parts: list[str] = []
        try:
            async with contextlib.aclosing(stream):  # type: ignore[type-var]  (closing the stream stops generation)
                async for piece in stream:
                    if p.first_delta_ms is None:
                        p.first_delta_ms = clock.ms()
                    p.parts.append(piece)
                    model_parts.append(piece)
                    yield DeltaEvent(piece)
                    # The search ended with nothing left to add: the "searching the web" badge can go now.
                    if web is not None and web.taken == len(web.results) and (end := web.terminal_event()):
                        yield end
        except Exception as e:  # cancellation (BaseException) goes to run's handler
            log.warning("chat %s: answer generation failed: %s", chat.id, _describe(e))
            yield ErrorEvent("llm", f"answer generation failed: {_describe(e)}")
            return
        if not "".join(model_parts).strip():
            yield ErrorEvent("llm", "the model returned an empty answer")
            return
        llm_ms = clock.ms(p.llm_start)
        if web is not None:
            async for event in self._continue(turn, prompt, "".join(model_parts), p, web):
                yield event
            if (end := web.stop()) is not None:
                yield end
        answer, citations = finalize_answer("".join(p.parts), p.citable)
        if not answer:
            yield ErrorEvent("llm", "the model returned an empty answer")
            return
        if (p.sources or p.web_sources) and not citations:
            log.info("chat %s: answer cites no source", chat.id)
        route = self._route(turn, p, abstained=False, reason=None)
        latency = self._latency(clock, p, first_delta_ms=p.first_delta_ms, llm_ms=llm_ms)
        async for event in self._save_answer(turn, answer, citations, route, latency, p):
            yield event

    # -------------------------------------------------------------- live data (§3.7)

    def _start_web(self, plan: TurnPlan) -> WebSearchRun | None:
        """Start the plan's web search (only its query leaves the machine)."""
        if plan.web_query is None or self.web_search is None or "web_search" not in plan.tools:
            return None
        return WebSearchRun(self.web_search, plan.web_query, self.settings.tools.web_search)

    async def _continue(
        self, turn: Turn, prompt: list[LLMMessage], answer: str, p: _Progress, web: WebSearchRun
    ) -> AsyncGenerator[ChatEvent, None]:
        """Results (and page texts) that arrived after the answer started: at most ``max_continuations`` short
        continuations, each one more model call, kept only if it cites what is new."""
        cfg = self.settings.tools.web_search
        if not p.web_sources or not cfg.stream_partial_results:
            return
        for i in range(cfg.max_continuations):
            new = await web.next_batch(last=i == cfg.max_continuations - 1)
            pages = web.fresh_pages()
            if new:
                p.web_sources.extend(new)
                yield web.results_event(new)
            if not new and not pages:
                return
            text = await self._continuation(turn, prompt, answer, new, pages, p)
            if text:
                piece = " " + text
                p.parts.append(piece)
                answer += piece
                p.continuations += 1
                yield DeltaEvent(piece)
            if web.finished and not web.pending_pages:
                return

    async def _continuation(
        self,
        turn: Turn,
        prompt: list[LLMMessage],
        answer: str,
        new: Sequence[WebSource],
        pages: Sequence[WebSource],
        p: _Progress,
    ) -> str | None:
        """One continuation sentence (buffered: it is kept only if it cites a new result or page), or None."""
        for source in (*new, *pages):
            source.content_given = source.content_given or bool(source.content)
        messages = [
            *prompt,
            LLMMessage("assistant", answer),
            LLMMessage("user", continuation_user_prompt(new, pages, p.plan.language)),
        ]
        pieces: list[str] = []
        try:
            stream = self.llm.stream(messages, max_tokens=CONTINUATION_TOKENS)
            async with contextlib.aclosing(stream):  # type: ignore[type-var]
                async for piece in stream:
                    pieces.append(piece)
        except Exception as e:  # the answer stands without it
            log.warning("chat %s: continuation failed: %s", turn.chat.id, _describe(e))
            return None
        text = _FIRST_SENTENCE.split(" ".join("".join(pieces).split()), maxsplit=1)[0]  # one sentence, as asked
        fresh = {s.source_id for s in (*new, *pages)}
        _, cited = finalize_answer(text, p.citable)
        keep = not text.startswith(CONTINUATION_NOTHING) and any(c.source_id in fresh for c in cited)
        verdict = "kept" if keep else "dropped"
        log.info("chat %s: continuation from %s %s: %r", turn.chat.id, sorted(fresh), verdict, text)
        return text if keep else None

    # -------------------------------------------------------------- helpers

    async def _retrieve(self, plan: TurnPlan, p: _Progress, chat: Chat) -> RetrievalResult:
        """The plan's retrieval: the speculative one when the route kept its query, else a new one."""
        if plan.speculation is not None:
            result, p.speculation = await plan.speculation.result_for(plan.query, plan.query_en)
            return result
        return await self.retrieval.retrieve(
            plan.query, project_id=chat.project_id, document_ids=list(p.ready), query_en=plan.query_en
        )

    @staticmethod
    def _fixed_reply(plan: TurnPlan, p: _Progress) -> str:
        if plan.mode == "ack":
            return ack_text(plan.ack or "ack", plan.language)
        state = p.state
        active = [p.ready[d] for d in (state.active_document_ids if state is not None else []) if d in p.ready]
        topic = state.document_topic if state is not None else None
        return resume_text(plan.language, topic, active or list(p.ready.values()))

    async def _history(self, chat_id: str, *, before: int) -> list[Message]:
        """Recent user and agent messages, oldest first, without "stop" turns (they got no answer)."""
        if before <= 1:
            return []
        items = (await self.messages.list(chat_id, before=before, limit=HISTORY_MESSAGES)).items
        kept = []
        for i, m in enumerate(items):
            if m.role == "event":
                continue
            if m.role == "user" and i + 1 < len(items) and items[i + 1].role == "event":
                continue
            kept.append(m)
        return kept

    def _prompt(
        self,
        turn: Turn,
        plan: TurnPlan,
        history: Sequence[Message],
        sources: Sequence[Source],
        memory: str | None,
        p: _Progress | None = None,
    ) -> list[LLMMessage]:
        language, length = plan.language, turn.length
        web = p.web_sources if p is not None else []
        if web and p is not None and p.web is not None:  # live data (§3.7): documents [S#] and web results [W#]
            # Page texts are long (prefill ~3 ms per token): the first answer has the snippets, a continuation the
            # pages, unless no continuation will come.
            cfg = self.settings.tools.web_search
            pages = not (cfg.stream_partial_results and cfg.max_continuations > 0)
            system = live_system_prompt(language, length, documents=p.documents_part)
            question = live_user_prompt(
                plan.query, sources, web, language, search_query=p.web.query, with_content=pages
            )
            for source in web:
                source.content_given = source.content_given or (pages and bool(source.content))
        elif plan.mode in ("grounded", "mixed"):
            system = answer_system_prompt(language, length, mixed=plan.mode == "mixed", live_note=plan.live_note)
            question = answer_user_prompt(plan.query, sources, language)
        elif plan.mode == "general":
            system = general_system_prompt(language, length, plan.general_note, live_note=plan.live_note)
            question = general_user_prompt(plan.query, language)
        elif plan.mode == "clarification":
            system, question = clarification_system_prompt(language), turn.text
        else:
            system, question = conversation_system_prompt(language), turn.text
        return [*self._context(system, history, memory), LLMMessage("user", question)]

    @staticmethod
    def _context(system: str, history: Sequence[Message], memory: str | None) -> list[LLMMessage]:
        """The system prompt (with the memory summary) and the recent messages."""
        messages = [LLMMessage("system", with_memory(system, memory))]
        for m in history:
            # Interrupted voice answers: only what was heard ("" when nothing was: the answer is left out).
            text = strip_markers(heard(m))[:HISTORY_CHARS]
            if text:
                messages.append(LLMMessage("user" if m.role == "user" else "assistant", text))
        return messages

    def _warm_up(
        self, turn: Turn, plan: TurnPlan, history: Sequence[Message], p: _Progress
    ) -> asyncio.Task[Any] | None:
        """While the web is searched, have the model read what the live prompt starts with (system prompt, history,
        document passages): the model server keeps that prefix, so the answer then only reads the web results and
        the question (§3.7; prefill is ~3 ms per token on this machine, most of the time to the first word).
        A one-token request in the background, cancelled with the turn; its failure changes nothing."""
        if p.web is None:
            return None
        system = live_system_prompt(plan.language, turn.length, documents=p.documents_part)
        messages = self._context(system, history, p.memory)
        if p.sources:
            messages.append(LLMMessage("user", live_user_prefix(p.sources)))

        async def warm() -> None:
            try:
                await self.llm.generate(messages, max_tokens=1)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.info("chat %s: warming the live prompt failed: %s", turn.chat.id, _describe(e))

        task = asyncio.ensure_future(warm())
        task.set_name("live-prompt-warm-up")
        return task

    def _route(
        self,
        turn: Turn,
        p: _Progress,
        *,
        abstained: bool,
        reason: AbstainReason | None,
        stopped: bool = False,
        model_used: bool = True,
    ) -> dict[str, Any]:
        """The message's ``route`` (§3.9): the validated route (§3.4), how it was decided, and what the answer did.
        ``abstained`` and ``stopped`` stay top-level booleans."""
        plan, result = p.plan, p.result
        route = plan.route
        prompt = PROMPT_IDS[plan.mode] if model_used else None
        if model_used and p.web_sources:
            prompt = LIVE_PROMPT_VERSION + (f"+{CONTINUATION_PROMPT_VERSION}" if p.continuations else "")
        return {
            "intent": plan.intent,
            "needs_retrieval": plan.needs_retrieval,
            "rewritten_query": route.rewritten_query if route is not None else None,
            "query_en": plan.query_en,
            "tools": list(route.tools) if route is not None else [],
            "web_search": p.web.record() if p.web is not None else None,
            "live_note": plan.live_note,
            "web_sources": len(p.web_sources),
            "continuations": p.continuations,
            "topic": route.topic if route is not None else None,
            "is_topic_shift": route.is_topic_shift if route is not None else False,
            "response_language": plan.language,
            "language": plan.language,
            "input_language": turn.input_language or message_language(turn.text),
            "route_confidence": route.confidence if route is not None else None,
            "answer": plan.mode,
            "general_note": plan.general_note,
            "router": plan.decision.record() if plan.decision is not None else None,
            "speculation": p.speculation,
            "memory": bool(p.memory) and prompt is not None,
            "abstained": abstained,
            "abstain_reason": reason,
            "stopped": stopped,
            "confidence": confidence_json(result.confidence if result is not None else None),
            "candidates": result.candidate_count if result is not None else 0,
            "sources": len(p.sources),
            "length": turn.length,
            "model": self.settings.llm.chat_model if prompt is not None else None,
            "prompt": prompt,
        }

    @staticmethod
    def _latency(clock: _Clock, p: _Progress, *, first_delta_ms: float | None, llm_ms: float | None) -> dict[str, Any]:
        timings = p.result.timings_ms if p.result is not None else {}
        decision = p.plan.decision
        latency: dict[str, Any] = {
            "router_ms": p.plan.router_ms,
            "router_llm_ms": decision.llm_ms if decision is not None else None,
            "retrieval_ms": p.retrieval_ms,
            "embed_ms": timings.get("embed"),
            "search_ms": timings.get("search"),
            "rerank_ms": timings.get("rerank"),
            "first_delta_ms": first_delta_ms,
            "llm_ms": llm_ms,
            "total_ms": clock.ms(),
        }
        if p.web is not None:  # from the web search's start (right after routing)
            latency.update(p.web.latency())
        return latency

    def _advance_state(
        self,
        turn: Turn,
        p: _Progress,
        citations: Sequence[Citation],
        *,
        completed: bool,
        message_id: str | None = None,
    ) -> asyncio.Task[Any] | None:
        """Move the conversation state on from this turn's validated route (application code owns the state). The
        write runs as its own task (a stop can't cancel it), after any earlier one for the chat (``StateWrites``); the
        next turn reads the state only once it is written. ``message_id``: a stopped answer's (it becomes the
        interrupted one)."""
        plan = p.plan
        if p.state is None or plan.route is None:  # stopped before the route was decided: nothing to record
            return None
        outcome = TurnOutcome(
            route=plan.route.model_copy(update={"needs_retrieval": plan.needs_retrieval}),
            query=plan.query if plan.needs_retrieval else None,
            input_language=turn.input_language or message_language(turn.text),
            asked_language=asked_language(turn.text),
            searched=plan.needs_retrieval and bool(p.ready),
            cited_document_ids=[c.document_id for c in citations],
            completed=completed,
            message_id=message_id,
            interrupted_message_id=plan.interrupted.message_id if plan.interrupted is not None else None,
        )
        chat_id = turn.chat.id
        return state_writes().schedule(chat_id, functools.partial(self.states.apply, chat_id, outcome), spawn=detach)

    async def _save_answer(
        self,
        turn: Turn,
        answer: str,
        citations: list[Citation],
        route: dict[str, Any],
        latency: dict[str, Any],
        p: _Progress,
        *,
        role: Literal["agent", "event"] = "agent",
    ) -> AsyncGenerator[ChatEvent, None]:
        # The state is written before the answer, so nothing of this turn is written after its agent_message (the voice
        # session's own saves, e.g. what was heard, then never wait on it). A stop arriving meanwhile saves the answer
        # as stopped (its state follows, chained after this one).
        write = self._advance_state(turn, p, citations, completed=True)
        if write is not None:
            await asyncio.shield(write)
        p.saving = True  # from here on a stop must not save a second copy
        # Its own task: a stop arriving mid-save can't roll it back, and run's stop handler waits for it.
        p.final_save = detach(
            self.messages.append(
                turn.chat.id,
                role=role,
                text=answer,
                modality=turn.modality,
                language=p.plan.language,
                citations=citations,
                route=route,
                latency=latency,
            )
        )
        try:
            agent = await asyncio.shield(p.final_save)
        except Exception as e:
            log.exception("chat %s: saving the answer failed", turn.chat.id)
            yield ErrorEvent("storage", f"could not save the answer: {_describe(e)}")
            return
        p.final = agent
        yield AgentMessageEvent(agent)

    async def _save_silent(self, turn: Turn, p: _Progress, clock: _Clock) -> AsyncGenerator[ChatEvent, None]:
        """A turn that gets no answer ("stop", a repeated "okay"): an ``event`` message records its route."""
        route = self._route(turn, p, abstained=False, reason=None, model_used=False)
        latency = self._latency(clock, p, first_delta_ms=None, llm_ms=None)
        notice = SILENT_NOTICES.get(p.plan.intent, "Acknowledged")
        async for event in self._save_answer(turn, notice, [], route, latency, p, role="event"):
            yield event

    async def _save_stopped(self, turn: Turn, p: _Progress, clock: _Clock, stop: AnswerStop | None) -> None:
        """Save an answer that was stopped: the text generated so far (a marker cut in half dropped), citing only the
        sources that text cites, with what was heard of it when the caller knows (``stop``), and remember it as the
        interrupted answer. Without ``stop``, nothing is saved when no text remains; with it (voice) an empty answer
        is saved, so the turn is closed."""
        text, citations = finalize_answer(trim_open_marker("".join(p.parts)), p.citable)
        if not text and stop is None:
            return
        route = self._route(turn, p, abstained=p.abstained, reason=p.reason, stopped=True)
        if stop is not None and stop.reason is not None:
            route["interrupted"] = stop.reason
        llm_ms = clock.ms(p.llm_start) if p.llm_start is not None else None
        latency = self._latency(clock, p, first_delta_ms=p.first_delta_ms, llm_ms=llm_ms)
        try:
            message = await self.messages.append(
                turn.chat.id,
                role="agent",
                text=text,
                modality=turn.modality,
                heard_text=stop.heard_text if stop is not None else None,
                language=p.plan.language,
                citations=citations,
                route=route,
                latency=latency,
            )
        except Exception:
            log.exception("chat %s: saving the stopped answer failed", turn.chat.id)
            return
        if stop is not None:
            stop.saved = message
        self._advance_state(turn, p, message.citations, completed=False, message_id=message.id)
        log.info("chat %s: answer stopped after %d characters", turn.chat.id, len(text))
