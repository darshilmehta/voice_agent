"""One turn of a chat (text or voice) as a stream of typed events (docs/DESIGN.md §3.2-§3.6, §3.9).

    save the user message                                        → UserMessageEvent
    plan the turn (phase 1: always document Q&A; phase 3 slots the router in here, §3.4)
    retrieve within the chat's project and document scope (READY documents only)
    abstention gate: no evidence above the threshold → fixed answer in the user's language, no LLM
    numbered sources [S1]… (dedupe, sections, context budget)    → SourcesEvent
    grounded answer streamed from the LLM                        → DeltaEvent …
    citations validated, agent message saved with latency         → AgentMessageEvent
    any failure                                                  → ErrorEvent (stage: retrieval | llm | storage), end

Transport-agnostic: ``ChatTurnService`` knows nothing about HTTP. The SSE endpoint (``api/chats.py``) only serialises
the events (``event.name`` + ``event.payload()``); the voice WebSocket (phase 4-6) runs the same turn with
``modality="voice"`` and feeds the ``DeltaEvent`` texts to TTS. The answer's length is a parameter
(``length="short"``: 1-3 speakable sentences, details left to the on-screen citations; ``"full"``: fuller text).

The user message stays saved when a later stage fails; a failed answer is not saved. To stop an answer (client gone,
barge-in), cancel the task consuming the events or ``await events.aclose()``: generation is cancelled (the LLM stream
is closed) and the text generated so far is saved with ``route.stopped = true``, citing only what that text cites;
nothing is saved if no text was generated yet. Every saved agent message's ``route`` carries top-level ``abstained``
and ``stopped`` booleans. A caller that knows what the user actually heard of the answer (the voice session, §3.3 c)
passes an ``AnswerStop`` to ``run``: the stopped answer is saved with that ``heard_text`` and ``route.interrupted``
(the reason), and the saved message is handed back on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

from ..domain.projects import Chat, Citation, Message, Modality
from ..providers.llm import LLMClient, LLMMessage
from ..providers.registry import Container
from ..providers.storage import MetadataDB
from ..settings import Language, Settings
from .base import InvalidInput
from .chats import ChatService
from .documents import DocumentService
from .language import choose_language, message_language
from .messages import MessageService
from .prompts import (
    ANSWER_LENGTHS,
    PROMPT_VERSION,
    AbstainReason,
    AnswerLength,
    abstention,
    answer_system_prompt,
    answer_user_prompt,
)
from .retrieval import Confidence, RetrievalResult, RetrievalService
from .sources import Source, build_sources, finalize_answer, strip_markers, trim_open_marker

log = logging.getLogger(__name__)

TEXT_MAX = 4000
HISTORY_MESSAGES = 6  # recent messages sent as conversation context (three exchanges)
HISTORY_CHARS = 1000  # per message

Stage = Literal["retrieval", "llm", "storage"]

# Saves of stopped answers (and voice session clean-ups) run as their own tasks, so a cancelled request can't interrupt
# them; keep them referenced.
_BACKGROUND: set[asyncio.Task[Any]] = set()


def detach(coro: Any) -> asyncio.Task[Any]:
    """Run ``coro`` as its own task that the caller's cancellation can't interrupt (await it with asyncio.shield).
    Needed under anyio cancel scopes, whose cancellation fires again at every await of the cancelled task."""
    task = asyncio.ensure_future(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return task


async def wait_for_background() -> None:
    """Wait for pending saves of stopped answers and session clean-ups of this event loop (tests, shutdown)."""
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
    """The numbered sources the answer may cite (none when abstaining) and the retrieval confidence."""

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
    """The agent's answer, saved: final text and the citations it actually makes."""

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


ChatEvent = UserMessageEvent | SourcesEvent | DeltaEvent | AgentMessageEvent | ErrorEvent


# ------------------------------------------------------------------ turn and plan


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


InterruptReason = Literal["barge_in", "stop"]


@dataclass(eq=False)
class AnswerStop:
    """Lets a caller that cuts an answer short (voice: barge-in or stop, §3.3 c) record what the user heard of it.

    Set ``heard_text`` and ``reason`` before cancelling the consumer of ``run``. The stopped answer is then saved with
    ``heard_text`` and ``route.interrupted = reason`` (with the text generated so far, which may be empty: a stopped
    voice turn always ends with an agent message), and once the cancelled consumer has finished:

    - ``saved`` is that message (None if the save failed or the answer had already been saved complete);
    - ``completed`` is the complete answer when it was saved before the stop took effect;
    - ``user`` is the turn's user message (set as soon as it is saved), even if its event was never consumed.
    """

    heard_text: str | None = None
    reason: InterruptReason | None = None
    saved: Message | None = None
    completed: Message | None = None
    user: Message | None = None


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """What to do with a user message. Phase 3's router (§3.4) produces this from the LLM (intent, rewritten and
    English queries, tools); phase 1 always plans document Q&A behind the abstention gate."""

    query: str
    language: Language
    intent: Literal["document_qa"] = "document_qa"
    needs_retrieval: bool = True
    query_en: str | None = None


@dataclass
class _Progress:
    """How far a turn's answer got, so a stopped answer can be saved with what it had."""

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
    final: Message | None = None  # the complete answer, saved


class TurnRouter(Protocol):
    async def plan(self, turn: Turn, history: Sequence[Message]) -> TurnPlan: ...


class DocumentQARouter:
    """Phase 1: every turn is a document question."""

    async def plan(self, turn: Turn, history: Sequence[Message]) -> TurnPlan:
        return TurnPlan(query=turn.text, language=turn.language)


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
    ) -> None:
        self.chats = ChatService(db)
        self.messages = MessageService(db)
        self.documents = DocumentService(db)
        self.retrieval = retrieval
        self.llm = llm
        self.settings = settings
        self.router = router or DocumentQARouter()

    @classmethod
    def from_container(cls, container: Container) -> ChatTurnService:
        db, llm = container["metadata_db"], container["llm"]
        if not isinstance(db, MetadataDB) or not isinstance(llm, LLMClient):
            raise TypeError(
                f"expected MetadataDB and LLMClient providers, got {type(db).__name__}, {type(llm).__name__}"
            )
        return cls(db, retrieval=RetrievalService.from_container(container), llm=llm, settings=container.settings)

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
        InvalidInput). ``language`` forces the answer language (else the message's script decides); ``modality`` is
        recorded on both messages; ``length`` sets the answer style and token cap. Voice turns pass the spoken
        language (``input_language``, saved on the user message) and the STT timings (``input_latency``)."""
        chat = await self.chats.get(chat_id)
        text = text.strip()
        if not text:
            raise InvalidInput("text must not be empty")
        if len(text) > TEXT_MAX:
            raise InvalidInput(f"text is longer than {TEXT_MAX} characters")
        if length not in ANSWER_LENGTHS:
            raise InvalidInput(f"length must be one of {', '.join(ANSWER_LENGTHS)}")
        fallback: Language = chat.language if chat.language in ("en", "hi") else self.settings.client.default_language  # type: ignore[assignment]
        language_ = choose_language(text, language, fallback)
        return Turn(chat, text, language_, language, modality, length, input_language, input_latency)

    async def run(self, turn: Turn, *, stop: AnswerStop | None = None) -> AsyncGenerator[ChatEvent, None]:
        """The turn's events. ``stop``: see ``AnswerStop`` (callers that know what was heard of a stopped answer)."""
        clock = _Clock()
        chat = turn.chat
        try:
            user = await self.messages.append(
                chat.id,
                role="user",
                text=turn.text,
                modality=turn.modality,
                language=turn.input_language or message_language(turn.text) or turn.language,
                latency=turn.input_latency,
            )
            history = await self._history(chat.id, before=user.seq)
        except Exception as e:
            log.exception("chat %s: saving the user message failed", chat.id)
            yield ErrorEvent("storage", f"could not save the message: {_describe(e)}")
            return
        if stop is not None:
            stop.user = user
        p = _Progress(TurnPlan(query=turn.text, language=turn.language))
        try:
            yield UserMessageEvent(user)
            async with contextlib.aclosing(self._answer(turn, history, clock, p)) as answer:
                async for event in answer:
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            # Stopped (client gone, barge-in, "stop"): keep what was generated. The save runs as its own task, so a
            # consumer whose cancellation keeps re-firing (anyio cancel scopes) can't interrupt it; this frame waits if
            # it can. A caller passing ``stop`` (voice) always gets a saved answer, even an empty one.
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
        plan = p.plan = await self.router.plan(turn, history)
        try:
            ready = await self.documents.ready_documents(chat.project_id, chat.document_scope)
            result = (
                await self.retrieval.retrieve(
                    plan.query, project_id=chat.project_id, document_ids=list(ready), query_en=plan.query_en
                )
                if ready
                else None
            )
        except Exception as e:
            log.warning("chat %s: retrieval failed: %s", chat.id, _describe(e))
            yield ErrorEvent("retrieval", f"document search failed: {_describe(e)}")
            return
        p.result = result
        retrieval_ms = p.retrieval_ms = clock.ms()
        confidence = result.confidence if result is not None else None

        if confidence is None or not confidence.above_threshold:
            reason: AbstainReason = "no_documents" if not ready else "not_covered"
            p.abstained, p.reason = True, reason
            yield SourcesEvent([], confidence, abstained=True)
            answer = abstention(plan.language, reason)
            p.parts.append(answer)
            yield DeltaEvent(answer)
            route = self._route(turn, plan, result, abstained=True, reason=reason, sources=0)
            latency = self._latency(clock, result, retrieval_ms, first_delta_ms=clock.ms(), llm_ms=None)
            async for event in self._save_answer(turn, plan, answer, [], route, latency, p):
                yield event
            return

        assert result is not None
        sources = p.sources = build_sources(
            result.chunks, ready, budget_tokens=self.settings.retrieval.context_token_budget
        )
        yield SourcesEvent([s.citation() for s in sources], confidence, abstained=False)

        p.llm_start = time.perf_counter()
        prompt = self._prompt(turn, plan, history, sources)
        stream = self.llm.stream(prompt, max_tokens=ANSWER_LENGTHS[turn.length].max_tokens)
        try:
            async with contextlib.aclosing(stream):  # type: ignore[type-var]  (closing the stream stops generation)
                async for piece in stream:
                    if p.first_delta_ms is None:
                        p.first_delta_ms = clock.ms()
                    p.parts.append(piece)
                    yield DeltaEvent(piece)
        except Exception as e:  # cancellation (BaseException) goes to run's handler
            log.warning("chat %s: answer generation failed: %s", chat.id, _describe(e))
            yield ErrorEvent("llm", f"answer generation failed: {_describe(e)}")
            return
        answer, citations = finalize_answer("".join(p.parts), sources)
        if not answer:
            yield ErrorEvent("llm", "the model returned an empty answer")
            return
        if not citations:
            log.info("chat %s: answer cites no source", chat.id)
        route = self._route(turn, plan, result, abstained=False, reason=None, sources=len(sources))
        latency = self._latency(
            clock, result, retrieval_ms, first_delta_ms=p.first_delta_ms, llm_ms=clock.ms(p.llm_start)
        )
        async for event in self._save_answer(turn, plan, answer, citations, route, latency, p):
            yield event

    # -------------------------------------------------------------- helpers

    async def _history(self, chat_id: str, *, before: int) -> list[Message]:
        if before <= 1:
            return []
        page = await self.messages.list(chat_id, before=before, limit=HISTORY_MESSAGES)
        return [m for m in page.items if m.role in ("user", "agent")]

    def _prompt(
        self, turn: Turn, plan: TurnPlan, history: Sequence[Message], sources: Sequence[Source]
    ) -> list[LLMMessage]:
        messages = [LLMMessage("system", answer_system_prompt(plan.language, turn.length))]
        for m in history:
            text = strip_markers(m.heard_text or m.text)[:HISTORY_CHARS]  # interrupted voice answers: what was heard
            if text:
                messages.append(LLMMessage("user" if m.role == "user" else "assistant", text))
        messages.append(LLMMessage("user", answer_user_prompt(plan.query, sources, plan.language)))
        return messages

    def _route(
        self,
        turn: Turn,
        plan: TurnPlan,
        result: RetrievalResult | None,
        *,
        abstained: bool,
        reason: AbstainReason | None,
        sources: int,
        stopped: bool = False,
    ) -> dict[str, Any]:
        return {
            "intent": plan.intent,
            "needs_retrieval": plan.needs_retrieval,
            "language": plan.language,
            "abstained": abstained,
            "abstain_reason": reason,
            "stopped": stopped,
            "confidence": confidence_json(result.confidence if result is not None else None),
            "candidates": result.candidate_count if result is not None else 0,
            "sources": sources,
            "length": turn.length,
            "model": None if abstained else self.settings.llm.chat_model,
            "prompt": PROMPT_VERSION,
        }

    @staticmethod
    def _latency(
        clock: _Clock,
        result: RetrievalResult | None,
        retrieval_ms: float | None,
        *,
        first_delta_ms: float | None,
        llm_ms: float | None,
    ) -> dict[str, Any]:
        timings = result.timings_ms if result is not None else {}
        return {
            "retrieval_ms": retrieval_ms,
            "embed_ms": timings.get("embed"),
            "search_ms": timings.get("search"),
            "rerank_ms": timings.get("rerank"),
            "first_delta_ms": first_delta_ms,
            "llm_ms": llm_ms,
            "total_ms": clock.ms(),
        }

    async def _save_answer(
        self,
        turn: Turn,
        plan: TurnPlan,
        answer: str,
        citations: list[Citation],
        route: dict[str, Any],
        latency: dict[str, Any],
        p: _Progress,
    ) -> AsyncGenerator[ChatEvent, None]:
        p.saving = True  # from here on a stop must not save a second copy
        try:
            agent = await self.messages.append(
                turn.chat.id,
                role="agent",
                text=answer,
                modality=turn.modality,
                language=plan.language,
                citations=citations,
                route=route,
                latency=latency,
            )
        except Exception as e:
            log.exception("chat %s: saving the answer failed", turn.chat.id)
            yield ErrorEvent("storage", f"could not save the answer: {_describe(e)}")
            return
        p.final = agent
        yield AgentMessageEvent(agent)

    async def _save_stopped(self, turn: Turn, p: _Progress, clock: _Clock, stop: AnswerStop | None) -> None:
        """Save an answer that was stopped: the text generated so far (a marker cut in half dropped), citing only the
        sources that text cites, with what was heard of it when the caller knows (``stop``). Without ``stop``,
        nothing is saved when no text remains; with it (voice) an empty answer is saved, so the turn is closed."""
        text, citations = finalize_answer(trim_open_marker("".join(p.parts)), p.sources)
        if not text and stop is None:
            return
        route = self._route(
            turn, p.plan, p.result, abstained=p.abstained, reason=p.reason, sources=len(p.sources), stopped=True
        )
        if stop is not None and stop.reason is not None:
            route["interrupted"] = stop.reason
        llm_ms = clock.ms(p.llm_start) if p.llm_start is not None else None
        latency = self._latency(clock, p.result, p.retrieval_ms, first_delta_ms=p.first_delta_ms, llm_ms=llm_ms)
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
        log.info("chat %s: answer stopped after %d characters", turn.chat.id, len(text))
