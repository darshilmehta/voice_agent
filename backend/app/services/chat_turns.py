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
    no web results (timeout, failure): the answer starts with a fixed notice ("I couldn't get live data just now.")
    and answers from the documents (or abstains, or answers from general knowledge) as before; the tool unavailable
    now: the notice ("I can't look up live data right now.") only when the documents don't answer it; turned off:
    nothing is said, the prompt only says never to guess current figures.
Stopping the turn cancels the search, its page fetches and the warm-up with it. Once the answer's own text is
complete (``AnswerStop.answered``), a stop saves it complete: only its continuation was still to come.

Every saved agent message has ``route.basis`` (``answer_basis``): what it drew on, documents / web / general.

The live visual canvas (§12.1): with a ``CanvasService``, the canvas is read before routing (``RouteRequest.screen``:
the router sees what is on screen; a question about a chart there gets that chart's tables first among its sources).
A document answer whose question calls for a visual (``route.visual``) gets its draft as soon as its retrieval returns
(``_start_draft``: code, no model call, so nothing queues ahead of the answer on the serial model) → VisualEvent
preparing / ready, CanvasEvent, in this stream with the answer's first delta, or, with ``run(on_visual=…)`` (voice), to
the caller as they come (the session holds them until the answer's first audio). A draft that isn't confident is
refined by the planner once the answer's text is complete (``_start_visual``), replaced in place (another ready with
its id) or withdrawn; an answer whose own figures suggest a visual starts one then. The stream stays open for the
refinement (bounded). A visual still being refined is cancelled by the next turn that needs the model (the draft
stays), waited for by an edit. A ``canvas_edit`` turn applies the edit (its events in the stream) and says "Done." /
"हो गया।".

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
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ..db.types import new_id
from ..domain.canvas import CanvasEvent, Visual, VisualEvent
from ..domain.conversation import ConversationState, VisualWant
from ..domain.projects import Chat, Citation, Message, Modality
from ..providers.base import PlaceholderProvider
from ..providers.ingestion import Chunk, document_label
from ..providers.llm import LLMClient, LLMMessage
from ..providers.registry import Container
from ..providers.storage import MetadataDB
from ..providers.web_search import WebSearch
from ..settings import Language, Settings
from .answer_guard import (
    AnswerGuard,
    Coverage,
    Unit,
    evidence_units,
    figures_in,
    identifiers_in,
    number_facts,
    terms,
)
from .base import InvalidInput
from .canvas.conversation import (
    CanvasEdit,
    EditResult,
    TurnVisual,
    describe,
    edit_reply,
    parse_edit,
    point_reference,
    refers_to_screen,
    resolve_target,
    screen_lines,
    visual_in_progress,
    visual_want,
)
from .canvas.planner import visual_intent
from .chats import ChatService
from .conversation import ConversationStateService, TurnOutcome, state_writes
from .documents import DocumentService
from .language import (
    ScriptCheck,
    asked_language,
    decide_language,
    message_language,
    script_language,
)
from .live_data import LiveNote, asks_live_figure
from .memory import MemoryKeeper, chat_activity
from .messages import MessageService
from .planning import VISUAL_INTENTS, DocumentQARouter, TurnPlan, TurnPlanner, interrupted_answer
from .prompts import (
    ANSWER_LENGTHS,
    CONTINUATION_NOTHING,
    CONTINUATION_PROMPT_VERSION,
    GENERAL_FIGURE_TEXTS,
    LIVE_FIGURE_TEXTS,
    LIVE_NOTICES,
    LIVE_PROMPT_VERSION,
    NOT_FROM_DOCUMENTS,
    PROMPT_IDS,
    SILENT_NOTICES,
    AbstainReason,
    AnswerLength,
    DocumentsPart,
    abstention,
    ack_text,
    answer_system_prompt,
    answer_user_prompt,
    chart_correction,
    clarification_system_prompt,
    continuation_user_prompt,
    conversation_system_prompt,
    coverage_note,
    fiscal_year_end_note,
    general_system_prompt,
    general_user_prompt,
    heard_note,
    insist_on_language,
    language_request_note,
    latest_period_note,
    live_figure_text,
    live_notice,
    live_system_prompt,
    live_user_prefix,
    live_user_prompt,
    named_document,
    names_note,
    on_screen_note,
    own_subject_note,
    passage_correction,
    resume_text,
    short_document_name,
    with_memory,
    with_note,
)
from .retrieval import (
    Confidence,
    RankedChunk,
    RetrievalResult,
    RetrievalService,
    SpeculationOutcome,
    asked_periods,
    fiscal_year_ends,
    scope_of,
)
from .router import (
    LLMTurnRouter,
    RouteRequest,
    TurnRouter,
    asks_about_facts,
    asks_for_figure,
    fast_route,
    heard,
    same_text,
)
from .sources import (
    Source,
    answer_declines,
    build_sources,
    estimate_tokens,
    finalize_answer,
    section_of,
    strip_markers,
    trim_open_marker,
)
from .subjects import (
    DEVANAGARI_WORD,
    HINDI_FUNCTION_WORDS,
    label_words,
    misheard_names,
    misheard_words,
    respell,
    subject_names,
)
from .web_search import ToolEvent, WebSearchRun, WebSource

if TYPE_CHECKING:
    from .canvas.service import CanvasService

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
# Recent messages sent as conversation context: at least HISTORY_MIN, and the window's start moves HISTORY_STEP
# messages at a time, so between moves the prompt's history only grows at its end and Ollama's prompt cache keeps
# all of it (a window sliding by one message every turn made the model read the whole history again, §9.5).
HISTORY_MIN = 4  # two exchanges
HISTORY_STEP = 4
HISTORY_MESSAGES = HISTORY_MIN + HISTORY_STEP - 1  # the most the window holds (7)
HISTORY_CHARS = 1000  # per message
# Conversation replies and clarifying questions: a sentence or two. Devanagari takes about three times the tokens of
# English for the same words (~0.9 a character), so a Hindi reply gets three times the cap: at 96 the last real run's
# Hindi clarifications ended "क्या आप किसी विशिष्ट विषय" and "या इसक" (last round, item 5). A reply cut by the cap still
# ends at its last whole sentence (``AnswerGuard.whole_sentences``).
SHORT_REPLY_TOKENS: dict[Language, int] = {"en": 96, "hi": 288}
CAP_SLACK = 2  # an answer that streamed this close to its token cap was cut by it
VISUAL_BUILD_S = 2.0  # after the planner's timeout: a heuristic fallback, building from the cells, storing (§12.1)
CONTINUATION_TOKENS = 96  # one sentence about results that arrived after the answer started (§3.7)
# Short answers (voice, and text chats' default) stop at the end of the sentence that reaches this many words, or at
# their third sentence (quality round, item 7: three long sentences were 36-45 s of speech).
SHORT_ANSWER_WORDS = 45
SHORT_ANSWER_SENTENCES = 3
# A passage whose reranker score is at least this "covers" the question for the answer's checks: an answer saying the
# documents don't have what it states is asked again (quality round, item 1).
STRONG_PASSAGE = 0.5
DRAFT_WAIT_S = 0.3  # how long the sources wait for the visual's draft (built in ~20 ms) to add its tables
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


ChatEvent = (
    UserMessageEvent
    | SourcesEvent
    | DeltaEvent
    | AgentMessageEvent
    | ErrorEvent
    | ToolEvent
    | VisualEvent
    | CanvasEvent
)
# Where a turn's visual events go when they don't travel in its own stream (voice: they may follow agent_message).
VisualSink = Callable[[VisualEvent | CanvasEvent], Awaitable[None]]


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
    garbled: bool = False  # the transcript looks garbled (voice): the user is asked to say it again (item 8)
    # Speech recognition wasn't sure of the transcript (voice: a low average log probability) without it looking
    # garbled: a question the documents don't answer is asked again rather than declined (last round, item 1).
    unsure: bool = False


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
    - ``completed`` is the complete answer when it was saved (or its save had already started) before the stop, or
      when the answer itself was complete and only its live-data continuation was still to come (§3.7): it is then
      saved complete, without the continuation;
    - ``user`` is the turn's user message (set as soon as it is saved, or passed to ``run``).

    ``answered`` is True once the answer's text is complete (a continuation may still follow), and ``on_answered``,
    if set, is called at that moment (voice: speak the last sentence now, not after the continuation).
    """

    heard_text: str | None = None
    reason: InterruptReason | None = None
    saved: Message | None = None
    completed: Message | None = None
    user: Message | None = None
    answered: bool = False
    on_answered: Callable[[], None] | None = None
    visual: TurnVisual | None = None  # the turn's visual (§12.1), once started


@dataclass
class _Progress:
    """How far a turn's answer got, so a stopped answer can be saved with what it had, and what the turn read before
    planning (for the route and the conversation state)."""

    plan: TurnPlan
    stop: AnswerStop | None = None
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
    answered: bool = False  # the answer's text is complete (a live-data continuation may still follow)
    llm_ms: float | None = None  # the answer's generation time, once complete
    notice: LiveNote | None = None  # the live-data notice the answer started with
    language_retry: bool = False  # the answer came out in the wrong script and was asked again (B5)
    pieces: int = 0  # pieces the model streamed for the current attempt (one a token): did it reach its cap?
    name_documents: bool = False  # the answer says which document its figures come from (UX5)
    # The canvas (§12.1)
    panels: list[Visual] = field(default_factory=list)  # on screen when the turn started
    visual_sink: VisualSink | None = None  # the transport's channel for the visual (None: the turn's own stream)
    visual: TurnVisual | None = None  # the answer's visual, prepared after its text is complete
    visual_want: VisualWant = "none"  # route.visual, with the answer's own figures counted
    screen_visual: Visual | None = None  # the visual a question is about ("what's the second bar?")
    screen: list[str] = field(default_factory=list)  # that visual, described for the answer prompt
    edit: dict[str, Any] | None = None  # route.canvas_edit of an edit turn
    labels: dict[str, str] = field(default_factory=dict)  # document id → label, for the company a question names
    # The answer's checks (services/answer_guard.py, quality round)
    draft: Visual | None = None  # the visual's draft, on screen from the answer's first words
    draft_sources: list[Source] = field(default_factory=list)  # its tables, among the answer's sources
    draft_confident: bool = False
    renames: dict[str, str] = field(default_factory=dict)  # misheard name → the documents' spelling (item 10)
    checks: list[dict[str, Any]] = field(default_factory=list)  # route.checks: what the checks changed
    fixed_live: bool = False  # a live figure answered with the fixed honest line (item 2)
    prefix: str | None = None  # "Not from your documents, but" (item 9)

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


def with_evidence(result: RetrievalResult | None, shown: Sequence[RankedChunk], query: str) -> RetrievalResult:
    """The retrieval with the tables of the visual a question is about put first (§12.1): the user pointed at them,
    so they pass the gate whatever the search scored."""
    ids = {c.chunk.chunk_id for c in shown}
    if result is None:
        confidence = Confidence(top_score=1.0, gap=1.0, dense_similarity=None, above_threshold=True)
        return RetrievalResult(query, query, list(shown), len(shown), confidence)
    chunks = [*shown, *(c for c in result.chunks if c.chunk.chunk_id not in ids)]
    confidence = (
        replace(result.confidence, above_threshold=True, missing_periods=(), missing_subjects=())
        if result.confidence is not None
        else Confidence(top_score=1.0, gap=1.0, dense_similarity=None, above_threshold=True)
    )
    return replace(result, chunks=chunks, confidence=confidence)


def _delivered(visual: TurnVisual, events: Sequence[VisualEvent | CanvasEvent]) -> list[VisualEvent | CanvasEvent]:
    """Events of a turn's visual on their way to the client (SSE): a ready visual among them is on screen now, so
    a cut no longer withdraws it."""
    if any(isinstance(e, VisualEvent) and e.phase == "ready" for e in events):
        visual.delivered = True
    return list(events)


Basis = Literal["documents", "web", "general"]
_SENTENCE_END = re.compile("(?<=[.!?।])\\s+")
# A question that names a period without a fiscal-year tag: a quarter, a calendar year, "last year", "पिछले साल".
_NAMES_A_PERIOD = re.compile(
    r"\b(?:q[1-4]|h[12]|(?:19|20)\d\d|(?:last|this|previous|current|next|prior)\s+(?:fiscal\s+|financial\s+)?"
    r"(?:year|quarter|half)|pichhle\s+saal|pichle\s+saal|is\s+saal)\b|पिछले\s+साल|इस\s+साल|पिछली\s+तिमाही",
    re.IGNORECASE,
)
_FIGURE_IN_LINE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CITED = re.compile(r"\[\s*[SW]\d+", re.IGNORECASE)
_NOTICES = tuple(text for texts in LIVE_NOTICES.values() for text in texts.values())


def answer_basis(p: _Progress, text: str, citations: Sequence[Citation]) -> list[Basis]:
    """``route.basis``: what an answer drew on, a sorted subset of ``["documents", "web", "general"]``.

    - "documents": it cites a document passage ([S#]); "web": it cites a web result ([W#]);
    - "general": a general-knowledge answer; a mixed answer with a sentence that cites nothing (the general
      knowledge it adds; the live-data notice doesn't count); or an answer without citations in a mode that isn't
      about the documents (conversation, clarification, the fixed "back to the report" and "Anything else?").

    An answer from web results counts only what it cites; an abstention, a grounded answer citing nothing and a
    "stop" are []."""
    mode = p.plan.mode
    if p.fixed_live:  # "I can't look up live data…": nothing was drawn on
        return []
    kinds = {c.kind for c in citations}
    basis: set[Basis] = set()
    if "document" in kinds:
        basis.add("documents")
    if "web" in kinds:
        basis.add("web")
    if p.web_sources or p.abstained or mode == "silent":
        return sorted(basis)
    if mode == "general":
        basis.add("general")
    elif mode == "mixed":
        body = text
        for notice in _NOTICES:
            body = body.replace(notice, " ")
        if any(s.strip() and not _CITED.search(s) for s in _SENTENCE_END.split(body)):
            basis.add("general")
    elif mode in ("conversation", "clarification", "resume", "ack") and not citations:
        basis.add("general")
    return sorted(basis)


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
        canvas: CanvasService | None = None,
    ) -> None:
        self.canvas = canvas  # None: no visuals in turns (§12.1)
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
        # The memory summary starts once the chat outgrows six messages: by the time the window's start moves past
        # a message (the 9th message moves it to the 5th), the summary has covered it.
        self.memory = MemoryKeeper(db, llm, settings, window=6)

    @classmethod
    def from_container(cls, container: Container, *, canvas: CanvasService | None = None) -> ChatTurnService:
        """``canvas``: the app's long-lived canvas service (``app.state.canvas``); without it turns make no visuals."""
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
            canvas=canvas,
        )

    def available_tools(self) -> frozenset[str]:
        """The live-data tools that can run now (§3.7): web search when configured, allowed and reachable."""
        if self.web_search is None or self.web_search.unavailable_reason() is not None:
            return frozenset()
        return frozenset({"web_search"})

    def enabled_tools(self) -> frozenset[str]:
        """The live-data tools turned on in the config (``tools.web_search.enabled``), reachable or not."""
        if self.web_search is None or not self.settings.tools.web_search.enabled:
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
        garbled: bool = False,
        unsure: bool = False,
    ) -> Turn:
        """Check the chat and the message before any event is produced (unknown chat → NotFound, bad text →
        InvalidInput). The answer language follows ``services/language.py``: a language the user asks for (now or
        earlier in the chat), ``language`` when the caller forces one, else the language of the message (the spoken
        one for voice), else the previous answer's. ``modality`` is recorded on both messages; ``length`` sets the
        answer style and token cap. Voice turns pass the spoken language (``input_language``, saved on the user
        message), the STT timings (``input_latency``) and whether the transcript looks garbled (``garbled``: the
        user is asked to say it again, nothing is answered) or unsure (``unsure``: a question the documents don't
        answer is asked again)."""
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
        return Turn(
            chat, text, decision.language, language, modality, length, input_language, input_latency, garbled, unsure
        )

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
        route["basis"] = []
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
        self,
        turn: Turn,
        *,
        stop: AnswerStop | None = None,
        user: Message | None = None,
        on_visual: VisualSink | None = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        """The turn's events. ``stop``: see ``AnswerStop`` (callers that know what was heard of a stopped answer).
        ``user``: the user message, if the caller already saved it with ``save_user_message``. While the turn runs,
        the chat's memory summary waits (or gives way if it is running): one LLM serves both.

        The answer's visual (§12.1) is drafted when the retrieval returns and, unless the draft is confident, refined
        by the planner once the answer's text is complete. Its events (``VisualEvent``, ``CanvasEvent``) travel in this
        stream unless ``on_visual`` is given: the draft's with the answer's first delta, the refinement's after
        ``AgentMessageEvent`` (the stream stays open until the visual is settled or given up:
        ``canvas.planner_timeout_ms`` and a little more). With ``on_visual`` (voice) they go there instead, as they
        come, and the stream ends with the answer. A canvas edit's events are always in the stream (they come before
        its reply). The memory summary and the next prompt's warm-up wait for the planner too: they would queue
        behind it on the model."""
        activity = chat_activity()
        activity.turn_started(turn.chat.id)
        p = _Progress(TurnPlan(query=turn.text, language=turn.language), stop=stop, visual_sink=on_visual)
        try:
            async with contextlib.aclosing(self._run(turn, stop, user, p)) as events:
                async for event in events:
                    yield event
        finally:
            job = functools.partial(self._after_turn, turn)
            visual = p.visual
            if visual is not None:  # (a no-op once it was given the answer's text)
                visual.answer_cut()  # never left waiting for an answer that won't come
            if visual is not None and visual.task is not None and not visual.done:
                chat_id = turn.chat.id
                visual.task.add_done_callback(lambda _: activity.turn_finished(chat_id, job, spawn=detach))
            else:
                activity.turn_finished(turn.chat.id, job, spawn=detach)

    async def _run(
        self, turn: Turn, stop: AnswerStop | None, user: Message | None, p: _Progress
    ) -> AsyncGenerator[ChatEvent, None]:
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
        try:
            yield UserMessageEvent(user)
            async with contextlib.aclosing(self._answer(turn, history, clock, p)) as answer:
                async for event in answer:
                    yield event
            if p.visual is not None and not p.visual.answer_known:  # the answer failed: no planner, unseen draft goes
                p.visual.answer_cut()
                if not p.visual.delivered:
                    p.visual.withdraw()
            if p.visual is not None and p.visual_sink is None:
                # The answer is saved; the stream stays open for its visual (the planner refining the draft, or
                # planning one), bounded (the planner's own timeout, then building it from the cells).
                async for event in p.visual.events(timeout=self._visual_wait_s()):
                    yield _delivered(p.visual, [event])[0]
        except (asyncio.CancelledError, GeneratorExit):
            if p.visual is not None:
                if not p.answered:  # cut while it was written: the draft (if shown) stays, without the planner
                    p.visual.answer_cut()
                if p.visual_sink is None:  # the stream that would carry it is gone
                    p.visual.cancel()
                    if not p.visual.delivered:
                        p.visual.withdraw()
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
            elif p.answered and not p.saving:  # complete; only the live-data continuation was still to come
                save = detach(self._save_complete(turn, p, clock, stop))
                with contextlib.suppress(BaseException):
                    await asyncio.shield(save)
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
            enabled_tools=self.enabled_tools(),
            garbled=turn.garbled,
        )
        request = await self._with_screen(chat.id, request, p)
        plan = p.plan = await self.planner.plan(
            request, project_id=chat.project_id, ready=ready, retrieval_enabled=state.retrieval_enabled
        )
        if plan.speculated and plan.speculation is None:
            p.speculation = "discarded"  # the route needs no retrieval

        if plan.mode == "canvas":
            async for event in self._canvas_edit(turn, p, clock):
                yield event
            return
        if plan.needs_retrieval and p.panels and refers_to_screen(turn.text):
            # "What's the second bar?", "why did it dip there?": answered from the tables of the visual it means.
            target = p.screen_visual = resolve_target(turn.text, p.panels, infer_kind=True)
            if target is not None:
                point = point_reference(turn.text, target)
                p.screen = [describe(target) + ".", *([f"It points at {point}."] if point else [])]

        if plan.mode == "silent":
            yield SourcesEvent([], None, abstained=False)
            async for event in self._save_silent(turn, p, clock):
                yield event
            return
        if plan.mode in ("resume", "ack"):
            answer = self._fixed_reply(plan, p, turn.text)
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
            if p.screen_visual is not None and ready and (shown := await self._screen_evidence(chat, p)):
                result = p.result = with_evidence(result, shown, plan.query)
            p.retrieval_ms = clock.ms()
            confidence = result.confidence if result is not None else None
        covered = plan.needs_retrieval and confidence is not None and confidence.above_threshold
        if covered:
            assert p.result is not None
            style = ANSWER_LENGTHS[turn.length]
            budget = self.settings.retrieval.context_token_budget
            p.sources = build_sources(
                self._own_documents(turn, p, p.result.chunks),
                ready,
                budget_tokens=min(budget, style.context_tokens or budget),
                max_sources=style.max_sources,
                best_budget_tokens=budget,
            )
            p.documents_part = "sources"
            p.name_documents = self._name_documents(p)
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
                if p.warm is not None and not p.warm.done():  # the live prompt won't be used: free the model
                    p.warm.cancel()
        web_citations = [w.citation() for w in p.web_sources]

        if plan.needs_retrieval and not covered:
            if p.web_sources:  # the documents don't answer it, the web may: say so and answer from the web
                yield SourcesEvent(web_citations, confidence, abstained=False)
            elif plan.mode == "mixed" and not (ready and self._withholds_figure(turn, p)):
                # the document part isn't covered: general knowledge, saying so (never a figure for a question that
                # asks for one: that is the documents' or nobody's, last round item 1)
                plan = p.plan = plan.as_general("not_covered" if ready else "no_documents")
                yield SourcesEvent([], confidence, abstained=False)
            else:
                reason: AbstainReason = "no_documents" if not ready else "not_covered"
                async for event in self._abstain(turn, p, clock, confidence, reason):
                    yield event
                return
        elif covered:
            await self._start_draft(turn, p)  # the visual's draft, built while the answer is written (§12.1)
            await self._draft_evidence(turn, p)  # its tables among the answer's sources (quality round, item 1)
            yield SourcesEvent([*(s.citation() for s in p.sources), *web_citations], confidence, abstained=False)
        elif plan.mode == "general" and not p.web_sources and self._withholds_figure(turn, p):
            # A general answer to a question for an amount, a number, a limit, a rate or a date, in a chat with
            # documents: they weren't searched (the router said general and the B1 check couldn't run), and a figure
            # from general knowledge would pass for theirs. Declined (or asked again), never guessed (last round, 1).
            async for event in self._abstain(turn, p, clock, None, "not_covered"):
                yield event
            return
        else:
            yield SourcesEvent(web_citations, None, abstained=False)

        if plan.mode == "general" and not p.web_sources and self._asks_live_figure(turn, p):
            # A live figure (a rate, a price, the news) with no live data and no document that answers it: a fixed
            # honest line, never the model's guess (quality round, item 2: "USD to INR today" got "about 83.50").
            p.fixed_live = True
            p.notice = plan.live_note
            answer = live_figure_text(plan.language, plan.live_note)
            p.parts.append(answer)
            yield DeltaEvent(answer)
            route = self._route(turn, p, abstained=False, reason=None, model_used=False)
            route["live_fixed"] = True
            latency = self._latency(clock, p, first_delta_ms=clock.ms(), llm_ms=None)
            async for event in self._save_answer(turn, answer, [], route, latency, p):
                yield event
            return

        # Live data asked for, none to give: say so first, after a failed search (the user heard the filler), or with
        # the tool unavailable when the documents don't answer it. Otherwise the prompt only says never to guess
        # current figures (live_hint), and web search turned off says nothing.
        if plan.live_note == "failed" or (plan.live_note == "unavailable" and not p.sources):
            p.notice = plan.live_note
            notice = live_notice(plan.live_note, plan.language) + " "
            p.first_delta_ms = clock.ms()
            p.parts.append(notice)
            yield DeltaEvent(notice)
        elif self._not_from_documents(turn, p):
            # A general answer spoken in a chat with documents says so first (quality round, item 9): the "general
            # knowledge" label is on screen only.
            p.prefix = NOT_FROM_DOCUMENTS[plan.language]
            p.first_delta_ms = clock.ms()
            p.parts.append(p.prefix)
            yield DeltaEvent(p.prefix)

        p.llm_start = time.perf_counter()
        p.renames = self._renames(turn, p)
        prompt = self._prompt(turn, plan, history, p.sources, p.memory, p)
        short = plan.mode in ("conversation", "clarification")
        max_tokens = SHORT_REPLY_TOKENS[plan.language] if short else ANSWER_LENGTHS[turn.length].max_tokens
        guard = self._guard(turn, p, prompt)
        guard.whole_sentences = short  # (a reply cut by the cap ends at its last whole sentence, never mid-word)
        stream = self._guarded_stream(prompt, plan.language, max_tokens, p, guard)
        model_parts: list[str] = []
        try:
            async with contextlib.aclosing(stream):  # closing the stream stops generation
                async for piece in stream:
                    if p.first_delta_ms is None:
                        p.first_delta_ms = clock.ms()
                    p.parts.append(piece)
                    model_parts.append(piece)
                    yield DeltaEvent(piece)
                    if p.visual is not None and p.visual_sink is None:  # the draft, once the answer has started
                        for event in _delivered(p.visual, p.visual.pending()):
                            yield event
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
        llm_ms = p.llm_ms = clock.ms(p.llm_start)
        # The answer is complete: stopped from here on (waiting for slower engines, or the continuation), it is saved
        # complete, not as interrupted.
        p.answered = True
        if p.stop is not None:
            p.stop.answered = True
            if p.stop.on_answered is not None:
                p.stop.on_answered()
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
        p.checks = guard.checks
        if (
            plan.mode == "grounded"
            and not p.web_sources
            and answer_declines(answer, [turn.text, plan.query, plan.query_en])
        ):
            # The documents passed the gate but the answer says they don't cover what was asked (B9; quality round,
            # item 5: also when it cites what it says around it): an abstention, listed among the summary's
            # unanswered questions, with no source chips.
            p.abstained, p.reason = True, "not_covered"
            answer, citations = strip_markers(answer), []
        self._start_visual(turn, p, answer)
        if p.visual is not None and p.visual_sink is None:
            # the draft (or a skeleton) if no delta carried it yet: it shows while the answer is saved
            for event in _delivered(p.visual, p.visual.pending()):
                yield event
        route = self._route(turn, p, abstained=p.abstained, reason=p.reason)
        if p.abstained:
            route["abstained_by"] = "answer"  # the gate let it through; the answer itself said so
        if p.prefix:
            route["not_from_documents"] = True
        latency = self._latency(clock, p, first_delta_ms=p.first_delta_ms, llm_ms=llm_ms)
        async for event in self._save_answer(turn, answer, citations, route, latency, p):
            yield event

    async def _abstain(
        self, turn: Turn, p: _Progress, clock: _Clock, confidence: Confidence | None, reason: AbstainReason
    ) -> AsyncGenerator[ChatEvent, None]:
        """The documents don't answer it: a fixed answer, no model. Spoken, a question that speech recognition wasn't
        sure of (``Turn.unsure``), or with a word the turn's passages have in another spelling that sounds the same
        ("बेख रिन" for "बैंक ऋण"), is asked again instead: "Sorry, I didn't catch that…" (last round, item 1)."""
        plan = p.plan
        p.abstained, p.reason = True, reason
        yield SourcesEvent([], confidence, abstained=True)
        p.notice = plan.live_note
        notice = live_notice(plan.live_note, plan.language) + " " if plan.live_note else ""
        spoken = reason == "not_covered" and turn.modality == "voice"
        misheard = self._misheard_words(turn, p) if spoken else {}
        say_again = spoken and (turn.unsure or bool(misheard))
        answer = notice + (ack_text("repeat", plan.language) if say_again else abstention(plan.language, reason))
        p.parts.append(answer)
        yield DeltaEvent(answer)
        route = self._route(turn, p, abstained=True, reason=reason, model_used=False)
        if say_again:
            route["say_again"] = {"unsure": turn.unsure, "misheard": misheard}
        latency = self._latency(clock, p, first_delta_ms=clock.ms(), llm_ms=None)
        async for event in self._save_answer(turn, answer, [], route, latency, p):
            yield event

    def _figure_question(self, turn: Turn, p: _Progress) -> bool:
        """The question asks for an amount, a number, a limit, a rate or a date (as said, or in its standalone or
        English form)."""
        plan = p.plan
        route = plan.route
        proposal = plan.decision.proposal if plan.decision is not None else None
        texts = (
            turn.text,
            plan.query,
            plan.query_en,
            route.rewritten_query if route is not None else None,
            proposal.query if proposal is not None else None,
        )
        return any(asks_for_figure(t) for t in texts)

    def _withholds_figure(self, turn: Turn, p: _Progress) -> bool:
        """A general (or not-from-the-documents) answer would give a figure from general knowledge for a question that
        asks for one, in a chat whose documents are searched: it is declined instead (last round, item 1). Live data
        has its own fixed line (§3.7), and with document search turned off general answers are what the user chose."""
        plan = p.plan
        if (
            not p.ready
            or plan.general_note == "retrieval_off"
            or (p.state is not None and not p.state.retrieval_enabled)
        ):
            return False
        if plan.live_hint or plan.live_note is not None or (plan.decision is not None and plan.decision.live):
            return False
        return self._figure_question(turn, p)

    @staticmethod
    def _misheard_words(turn: Turn, p: _Progress) -> dict[str, str]:
        """Words of the question that the turn's passages have in another spelling that sounds the same: likely
        misheard ("बेख" → "बैंक"). Only the passages retrieval found for it (the best few, whatever their score)."""
        passages = [r.chunk.text for r in (p.result.chunks if p.result is not None else [])]
        return misheard_words(turn.text, passages) if passages else {}

    async def _answer_stream(
        self, prompt: list[LLMMessage], language: Language, max_tokens: int, p: _Progress
    ) -> AsyncGenerator[str, None]:
        """The model's answer, in the answer language (B5). Its first letters are held back until they tell the script
        (``ScriptCheck``: two words or so of English, the first Devanagari word of Hindi) and then sent as one piece;
        in the wrong script the stream is closed and the model asked once more, insisting on the language. If it
        still answers in the other script, that answer stands, and is saved (and spoken) as what it is
        (``script_language``)."""
        for attempt in (1, 2):
            check = ScriptCheck(language)
            held: list[str] = []
            wrong = False
            stream = self.llm.stream(prompt, max_tokens=max_tokens)
            p.pieces = 0
            async with contextlib.aclosing(stream):  # type: ignore[type-var]  (closing the stream stops generation)
                async for piece in stream:
                    p.pieces += 1
                    if check.verdict is not None:
                        yield piece
                        continue
                    held.append(piece)
                    if check.feed(piece) is None:
                        continue
                    if check.verdict is False and attempt == 1:
                        wrong = True
                        break
                    yield "".join(held)
                    held = []
            if not wrong and attempt == 1 and check.finish() is False:
                wrong = True
            if wrong:
                log.info("answer in the wrong script for %s (%r): asking again", language, "".join(held)[:80])
                p.language_retry = True
                prompt = insist_on_language(prompt, language)
                continue
            if held:
                yield "".join(held)
            return

    # -------------------------------------------------------------- the answer's checks (quality round)

    async def _guarded_stream(
        self, prompt: list[LLMMessage], language: Language, max_tokens: int, p: _Progress, guard: AnswerGuard
    ) -> AsyncGenerator[str, None]:
        """The model's answer through its checks (services/answer_guard.py): only checked text comes out. A first
        sentence that says the documents don't cover what the chart on screen or a strong passage states closes the
        stream, and the model is asked once more with the evidence named (``route.checks``)."""
        if not guard.active:
            async for piece in self._answer_stream(prompt, language, max_tokens, p):
                yield piece
            return
        for attempt in (1, 2):
            retry = False
            stream = self._answer_stream(prompt, language, max_tokens, p)
            async with contextlib.aclosing(stream):  # closing the stream stops generation
                async for piece in stream:
                    out = guard.feed(piece)
                    if out.text:
                        yield out.text
                    if out.verdict == "retry":
                        retry = True
                        break
                    if out.verdict == "stop":
                        return
            if not retry:
                out = guard.finish(truncated=p.pieces >= max_tokens - CAP_SLACK)
                if out.text:
                    yield out.text
                retry = out.verdict == "retry"
            if not retry or attempt == 2 or guard.coverage is None:
                return
            log.info("answer denies what its evidence states: asking again")
            guard.restart()
            prompt = with_note(prompt, coverage_note(guard.coverage.retry_note))

    def _guard(self, turn: Turn, p: _Progress, prompt: Sequence[LLMMessage] = ()) -> AnswerGuard:
        """The checks of this answer: coverage (a draft on screen, strong passages), live figures (a live question
        answered without live data), codes in the sources, misheard names, numbers the model corrupts (against what
        its ``prompt`` says: sources, history, question; every mode, conversation replies too), the length of a
        short answer."""
        plan = p.plan
        documents = plan.mode in ("grounded", "mixed") and not p.web_sources
        figures = None
        line = LIVE_FIGURE_TEXTS[plan.language][0 if p.notice is None else 1]
        check = "live_figure"
        said = [turn.text, plan.query, plan.query_en or ""]
        if documents and (plan.live_hint or plan.live_note is not None) and p.sources:
            figures = figures_in([*(s.chunk.text for s in p.sources), *said])
        elif self._general_about_facts(turn, p):
            # A general answer about facts in a chat with documents: a figure that isn't in the question is replaced
            # (once; another is dropped), as a live figure is (last round, item 1): it would pass for the documents'.
            figures = figures_in(said)
            line = GENERAL_FIGURE_TEXTS[plan.language][0 if p.prefix is None else 1]
            check = "general_figure"
        short = turn.length == "short" and plan.mode in ("grounded", "mixed", "general")
        return AnswerGuard(
            coverage=self._coverage(turn, p) if documents else None,
            identifiers=identifiers_in(s.chunk.text for s in p.sources),
            figures=figures,
            live_line=line,
            figure_check=check,
            renames=p.renames,
            max_words=SHORT_ANSWER_WORDS if short else None,
            max_sentences=SHORT_ANSWER_SENTENCES if short else None,
            lower_first=p.prefix is not None and plan.language == "en",
            after_disclaimer=p.prefix is not None,
            numbers=number_facts(m.content for m in prompt),
        )

    @staticmethod
    def _general_about_facts(turn: Turn, p: _Progress) -> bool:
        """A general answer, in a chat with READY documents that are searched, to a question about facts (not a
        definition, a how-to or small talk) or to one the documents were searched for and don't cover."""
        plan = p.plan
        if plan.mode != "general" or p.web_sources or not p.ready or plan.general_note == "retrieval_off":
            return False
        if plan.live_hint or plan.live_note is not None:
            return False
        return plan.general_note == "not_covered" or asks_about_facts(turn.text)

    def _coverage(self, turn: Turn, p: _Progress) -> Coverage | None:
        """What answers the question already (the draft's tables, the strong passages), as the coverage check's units,
        with the sentence that replaces a denial of it and the note for asking again."""
        plan = p.plan
        strong = [s for s in p.sources if s.rerank_score >= STRONG_PASSAGE and s not in p.draft_sources]
        if not strong and not p.draft_sources:
            return None
        units: list[Unit] = []
        for s in [*p.draft_sources, *strong]:
            units += evidence_units(s.chunk.text, " ".join(s.chunk.heading_path))
        names = frozenset(w for label in self._labels_now(p).values() for w in label.casefold().split())
        question = terms(" ".join(t for t in (turn.text, plan.query, plan.query_en) if t), ignore=names)
        if p.draft is not None and p.draft_sources:
            ids = [s.source_id for s in p.draft_sources]
            document = short_document_name(p.draft_sources[0].filename)
            correction = chart_correction(p.draft.title, document, ids, plan.language)
            note = f"{''.join(f'[{i}]' for i in ids)}, the table(s) of the chart on screen ({describe(p.draft)})"
        else:
            best = max(strong, key=lambda s: s.rerank_score)
            heading = best.chunk.heading_path[-1] if best.chunk.heading_path else None
            where = heading or (best.pages or None)
            document = short_document_name(best.filename)
            correction = passage_correction(document, where, best.source_id, plan.language)
            note = f"[{best.source_id}] ({section_of(best.chunk.heading_path) or best.pages or best.filename})"
        return Coverage(units, question, correction, note, names=names, visual=p.draft is not None)

    def _asks_live_figure(self, turn: Turn, p: _Progress) -> bool:
        """A question for a live figure that gets no live data (web search off, unavailable, or empty)."""
        plan = p.plan
        if not (plan.live_hint or plan.live_note is not None):
            return False
        route = plan.route
        texts = (turn.text, route.rewritten_query if route is not None else None, plan.query_en)
        documents = list(p.ready.values())
        return any(asks_live_figure(t, documents=documents) for t in texts if t)

    def _not_from_documents(self, turn: Turn, p: _Progress) -> bool:
        """A spoken general answer in a chat with READY documents to a question about facts (not a definition, a
        how-to or small talk), or to a question the documents were searched for and don't cover: it starts by saying
        it isn't from them (quality round, item 9)."""
        plan = p.plan
        if plan.mode != "general" or turn.modality != "voice" or not p.ready or p.notice is not None:
            return False
        return plan.general_note == "not_covered" or asks_about_facts(turn.text)

    def _renames(self, turn: Turn, p: _Progress) -> dict[str, str]:
        """Names the user was misheard as, and the documents' spelling ("Wall Mora" → "Valmora", item 10); spoken, also
        Hindi words the answer's passages spell differently but that sound the same ("बेख रिन" → "बैंक ऋण", last round,
        item 1): the answer model reads the question as the documents spell it."""
        if not p.ready or p.plan.mode not in ("grounded", "mixed", "general"):
            return {}
        plan = p.plan
        renames = misheard_names([turn.text, plan.query, plan.query_en], self._labels_now(p))
        if turn.modality == "voice" and p.sources:
            renames.update(misheard_words(turn.text, [s.chunk.text for s in p.sources]))
        return renames

    async def _draft_evidence(self, turn: Turn, p: _Progress) -> None:
        """The visual's draft is on screen from the answer's first words (§12.1): its tables must be among the
        answer's sources, or the answer may deny what the chart shows (quality round, item 1: a voice answer gets the
        best three passages, and the quarterly table the chart was drawn from was the fourth). Waits for the draft
        (~20 ms of code), adds its tables (with the [S#] ids the visual's cells already use), and keeps the
        evidence within the answer's budget by dropping the weakest other passages."""
        visual = p.visual
        if visual is None:
            return
        shown = await visual.draft(DRAFT_WAIT_S)
        if shown is None:
            return
        p.draft, p.draft_confident = shown, visual.trace.draft == "confident"
        have = {s.chunk.chunk_id: s for s in p.sources}
        ranked = {r.chunk.chunk_id: r for r in (p.result.chunks if p.result is not None else [])}
        added: list[Source] = []
        tables: dict[str, Chunk] | None = None
        for citation in shown.sources:
            if citation.chunk_id in have:
                p.draft_sources.append(have[citation.chunk_id])
                continue
            r = ranked.get(citation.chunk_id)
            chunk = r.chunk if r is not None else None
            if chunk is None:
                if tables is None:
                    tables = {c.chunk.chunk_id: c.chunk for c in await self._table_chunks(turn.chat, shown)}
                chunk = tables.get(citation.chunk_id)
            if chunk is None or chunk.document_id not in p.ready:
                continue
            filename = p.ready[chunk.document_id]
            added.append(Source(citation.source_id, chunk, filename, r.rerank_score if r is not None else 1.0))
        p.draft_sources += added
        if not added:
            return
        style = ANSWER_LENGTHS[turn.length]
        budget = self.settings.retrieval.context_token_budget
        limit = min(budget, style.context_tokens or budget)  # the passages besides the best one (build_sources)
        best = max(p.sources, key=lambda s: s.rerank_score) if p.sources else None
        keep = {s.chunk.chunk_id for s in p.draft_sources} | ({best.chunk.chunk_id} if best is not None else set())

        def cost(s: Source) -> int:
            return estimate_tokens(s.chunk.text) + 24

        used = sum(cost(s) for s in [*p.sources, *added] if s is not best)
        weakest = sorted((s for s in p.sources if s.chunk.chunk_id not in keep), key=lambda s: s.rerank_score)
        dropped: set[str] = set()
        while used > limit and weakest:
            s = weakest.pop(0)
            dropped.add(s.chunk.chunk_id)
            used -= cost(s)
        p.sources = [*(s for s in p.sources if s.chunk.chunk_id not in dropped), *added]
        p.name_documents = self._name_documents(p)
        log.info(
            "chat %s: the draft's table(s) %s added to the answer's sources (%d dropped)",
            turn.chat.id,
            [s.source_id for s in added],
            len(dropped),
        )

    @staticmethod
    def _latest_period(question: Sequence[str | None], sources: Sequence[Source]) -> str | None:
        """For a question that names no period, the latest fiscal year the sources' tables give figures for, when they
        give figures for more than one ("What is the dividend per share?": FY24, not FY23; quality round, item 4).
        Tables only (their header and rows): a paragraph's years are as often plans ("capex over FY25 and FY26") as
        figures."""
        if any(t and (asked_periods(t) or _NAMES_A_PERIOD.search(t)) for t in question):
            return None
        found: set[int] = set()
        for s in sources:
            rows = [line for line in s.chunk.text.splitlines() if line.strip().startswith("|")]
            plain = [re.sub(r"(?i)\b(?:q[1-4]\s*)?fy\s?'?\d{2,4}|(?:19|20)\d\d", " ", r) for r in rows]
            if any(_FIGURE_IN_LINE.search(r) for r in plain):
                for row in rows:
                    found |= asked_periods(row)
        return f"FY{max(found) % 100:02d}" if len(found) >= 2 else None

    async def _table_chunks(self, chat: Chat, visual: Visual) -> list[RankedChunk]:
        """The tables a visual was built from, as passages (``chunk_id`` as the visual's citations name them)."""
        if self.canvas is None:
            return []
        try:
            tables = await self.canvas.visual_tables(chat.id, visual)
        except Exception as e:
            log.warning("chat %s: reading the tables of %s failed: %s", chat.id, visual.id, _describe(e))
            return []
        out = []
        for ds, table in tables:
            chunk = Chunk(
                chunk_id=ds.chunk_id or f"{ds.document_id}:v{ds.version}:table{ds.table_index}",
                project_id=chat.project_id,
                document_id=ds.document_id,
                version=ds.version,
                chunk_index=-1,
                chunking_version="canvas",
                page_start=ds.page_start,
                page_end=ds.page_end,
                heading_path=[*table.heading_path[-1:], ds.title] if table.heading_path else [ds.title],
                content_type="table",
                language="hi" if any("ऀ" <= ch <= "ॿ" for ch in table.markdown) else "en",
                text=table.markdown,
                token_count=len(table.markdown) // 4,
            )
            out.append(RankedChunk(chunk, rerank_score=1.0, fused_score=1.0, dense_score=None, search_rank=0))
        return out

    # -------------------------------------------------------------- the canvas (§12.1)

    def _visual_wait_s(self) -> float:
        """How long a turn waits for a visual still being prepared: the planner's timeout, then building it."""
        return self.settings.canvas.planner_timeout_ms / 1000 + VISUAL_BUILD_S

    async def _panels(self, chat_id: str) -> list[Visual]:
        if self.canvas is None:
            return []
        try:
            return (await self.canvas.canvas(chat_id)).panels
        except Exception as e:  # the canvas is optional: the conversation goes on without it
            log.warning("chat %s: reading the canvas failed: %s", chat_id, _describe(e))
            return []

    async def _with_screen(self, chat_id: str, request: RouteRequest, p: _Progress) -> RouteRequest:
        """The canvas as context for the route (``RouteRequest.screen``), and the previous turn's visual if it is
        still being prepared: an edit waits for it ("make it a bar chart" said while it is drawn); a turn that needs
        the model cancels it (the model serves one request at a time: the new question comes first); an
        acknowledgement, thanks or a greeting leaves it alone."""
        if self.canvas is None:
            return request
        previous = visual_in_progress(chat_id)
        p.panels = await self._panels(chat_id)
        request = replace(request, screen=screen_lines(p.panels, request.utterance))
        if previous is not None:
            if parse_edit(request.utterance) is not None:  # about the visual being drawn (the canvas may be empty)
                await previous.wait(self._visual_wait_s())
                p.panels = await self._panels(chat_id)
                return replace(request, screen=screen_lines(p.panels, request.utterance))
            fast = fast_route(request)
            if fast is None or fast.route.intent == "stop" or fast.route.needs_retrieval or fast.language_request:
                previous.cancel()
        return request

    async def _canvas_edit(self, turn: Turn, p: _Progress, clock: _Clock) -> AsyncGenerator[ChatEvent, None]:
        """A canvas edit: applied (``CanvasService.edit``: its canvas events in this stream), then a fixed reply,
        "Done." / "हो गया।" or why not ("There's no chart on screen yet."). Never an abstention."""
        plan = p.plan
        edit = parse_edit(turn.text) or CanvasEdit("model")
        yield SourcesEvent([], None, abstained=False)
        result = EditResult("nothing", edit.op)
        if self.canvas is not None:
            route = plan.route
            english = plan.query_en or (route.rewritten_query if route is not None else None)
            try:
                async for event in self.canvas.edit(
                    turn.chat.id, edit, utterance=turn.text, language=plan.language, query_en=english
                ):
                    if isinstance(event, EditResult):
                        result = event
                    else:
                        yield event
            except Exception as e:  # a bug in an edit must not end the turn without a reply
                log.exception("chat %s: the canvas edit failed", turn.chat.id)
                result = EditResult("failed", edit.op, detail=_describe(e)[:300])
        p.edit = result.record(edit)
        answer = edit_reply(result.outcome, plan.language)
        p.parts.append(answer)
        yield DeltaEvent(answer)
        route_json = self._route(turn, p, abstained=False, reason=None, model_used=False)
        latency = self._latency(clock, p, first_delta_ms=clock.ms(), llm_ms=None)
        async for event in self._save_answer(turn, answer, [], route_json, latency, p):
            yield event

    def _wants_visual(self, p: _Progress, answer: str | None = None) -> VisualWant:
        """``route.visual`` for an answer from the documents (document, mixed or correction turns with sources, not
        abstained, not about a chart on screen), with the answer's own figures counted once there is one; a suggestion
        needs a table among the sources."""
        plan = p.plan
        route = plan.route
        if self.canvas is None or route is None:
            return "none"
        if plan.intent not in VISUAL_INTENTS or plan.mode not in ("grounded", "mixed"):
            return "none"
        if p.abstained or not p.sources or p.screen_visual is not None:  # (a question about a chart on screen)
            return "none"
        want = route.visual
        if want == "none" and answer is not None:
            want = visual_want(answer=answer)
        if want == "suggest" and not any(s.chunk.content_type == "table" for s in p.sources):
            return "none"  # a suggestion needs a table behind the answer
        return want

    async def _start_draft(self, turn: Turn, p: _Progress) -> None:
        """The answer's visual (§12.1, "Instant draft, then refine"), started as soon as the turn's retrieval returns
        when its words ask for one (``route.visual``): the draft is code (milliseconds, no model call, so nothing
        queues in front of the answer on the serial model) and is on screen when the answer starts (voice: with its
        first audio; SSE: with its first delta). The planner, when the draft isn't confident, waits for the answer's
        text (``_start_visual``)."""
        want = self._wants_visual(p)
        if want == "none" or self.canvas is None:
            return
        p.visual_want = want
        question = p.plan.query
        if want == "suggest" and visual_intent(question) == "none":
            question = turn.text  # the words that suggested it
        self._launch_visual(turn, p, question, want, answer=None, labels=await self._document_labels(turn, p))

    def _start_visual(self, turn: Turn, p: _Progress, answer: str) -> None:
        """The answer's text is complete (and not yet saved). A visual started with the draft gets it: the planner
        may refine the draft now (Ollama serves one request at a time, so a planner sent earlier would delay the
        answer's first token: 0.8 → 4.1 s measured; §12.1), or, for an answer that turned out to abstain, the draft is
        withdrawn. Otherwise the answer's own figures may suggest a visual (three or more, with a table among the
        sources): its draft and, if needed, the planner run now, back to back."""
        spoken = strip_markers(answer)
        if p.visual is not None:
            if p.abstained:
                p.visual.answer_abstained()
            else:
                p.visual.answered(spoken)
            return
        want = self._wants_visual(p, spoken)
        if want == "none" or self.canvas is None:
            if p.visual_want != "none" and p.abstained:
                p.visual_want = "none"
            return
        p.visual_want = want
        question = p.plan.query
        if want == "suggest" and visual_intent(question, spoken) == "none":
            question = turn.text  # the words that suggested it
        # (only the answer's figures suggest it: a draft that isn't sure stays hidden, the planner decides)
        self._launch_visual(turn, p, question, want, answer=spoken, labels=self._labels_now(p), show_unsure=False)

    def _launch_visual(
        self,
        turn: Turn,
        p: _Progress,
        question: str,
        want: VisualWant,
        *,
        answer: str | None,
        labels: Mapping[str, str] | None,
        show_unsure: bool = True,
    ) -> None:
        assert self.canvas is not None
        canvas, chat_id = self.canvas, turn.chat.id
        visual = p.visual = TurnVisual(chat_id, new_id("vis"), want, announce=want == "requested")
        visual_id = visual.visual_id
        visual.remove = lambda: canvas.remove_visual(chat_id, visual_id)
        if answer is not None:
            visual.answered(answer)
        if p.stop is not None:
            p.stop.visual = visual
        visual.start(
            canvas.prepare_visual(
                chat_id,
                question,
                language=p.plan.language,
                answer=answer,
                sources=[s.citation() for s in p.sources],
                query_en=p.plan.query_en,
                force=want == "requested",
                visual_id=visual_id,
                turn=visual,
                labels=labels,
                show_unsure=show_unsure,
            )
        )
        if p.visual_sink is not None:
            detach(self._forward_visual(visual, p.visual_sink))

    async def _document_labels(self, turn: Turn, p: _Progress) -> dict[str, str]:
        """The chat's documents' labels (file name and title) as retrieval knows them (fetched with the search and
        remembered, so this costs nothing), for which company a question names; file names otherwise."""
        labels = self._labels_now(p)
        if len(p.ready) >= 2:
            with contextlib.suppress(Exception):
                labels |= await self.retrieval.document_labels(scope_of(turn.chat.project_id, list(p.ready)))
        p.labels = labels
        return labels

    @staticmethod
    def _labels_now(p: _Progress) -> dict[str, str]:
        return p.labels or {d: document_label(name, None) for d, name in p.ready.items()}

    @staticmethod
    async def _forward_visual(visual: TurnVisual, sink: VisualSink) -> None:
        async for event in visual.events():
            try:
                await sink(event)
            except Exception as e:  # the transport is gone: the visual is still on the canvas
                log.info("a visual event couldn't be sent: %s", _describe(e))

    async def _screen_evidence(self, chat: Chat, p: _Progress) -> list[RankedChunk]:
        """The tables of the visual a question is about, as passages that rank first (a question about what is on
        screen is answered from the cells it shows, cited like any source)."""
        if p.screen_visual is None:
            return []
        return await self._table_chunks(chat, p.screen_visual)

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
        """The plan's retrieval: the one already run while routing (B1), the speculative one when the route kept its
        query, else a new one."""
        if plan.prefetched is not None:
            result, p.speculation = await plan.prefetched
            return result
        if plan.speculation is not None:
            result, p.speculation = await plan.speculation.result_for(plan.query, plan.query_en)
            return result
        return await self.retrieval.retrieve(
            plan.query, project_id=chat.project_id, document_ids=list(p.ready), query_en=plan.query_en
        )

    @staticmethod
    def _name_documents(p: _Progress) -> bool:
        """Should the answer say which document its figures come from (UX5)? When its sources come from more than one
        document, or from one that isn't the obvious one: the chat searches several documents and the conversation
        wasn't about this one (two reports with the same metrics: "revenue FY24" answered from the other company's
        report without saying so)."""
        documents = list(dict.fromkeys(s.chunk.document_id for s in p.sources))
        if len(documents) > 1:
            return True
        active = p.state.active_document_ids if p.state is not None else []
        return len(p.ready) > 1 and bool(documents) and documents[0] not in active

    @staticmethod
    def _fixed_reply(plan: TurnPlan, p: _Progress, text: str) -> str:
        if plan.mode == "ack":
            return ack_text(plan.ack or "ack", plan.language)
        state = p.state
        active_ids = [d for d in (state.active_document_ids if state is not None else []) if d in p.ready]
        named = named_document(text, p.ready) if len(p.ready) > 1 else None
        if named is not None and named not in active_ids:
            # "Back to the annual report" after a question about another document: that report, not the last topic
            return resume_text(plan.language, None, [p.ready[named]])
        topic = state.document_topic if state is not None else None
        return resume_text(plan.language, topic, [p.ready[d] for d in active_ids] or list(p.ready.values()))

    async def _after_turn(self, turn: Turn) -> None:
        """After a turn, while its answer is spoken and nobody is asking yet (``ChatActivity``: cancelled the moment
        the next turn starts): refresh the memory summary if it is due, then have the model read the next answer's
        prompt prefix (system prompt, memory, history up to this answer), so the next answer reads only its evidence
        and question (§9.5)."""
        await self.memory.refresh_if_due(turn.chat.id)
        await self._warm_next_prompt(turn)

    async def _warm_next_prompt(self, turn: Turn) -> None:
        chat = await self.chats.get(turn.chat.id)
        state = await self.states.get(chat.id)
        language = state.preferred_language or self._fallback_language(chat, state)
        history = await self._history(chat.id, before=chat.message_count + 1)
        if not history:
            return
        system = answer_system_prompt(language, turn.length)  # most next turns are document questions
        messages = [*self._context(system, history, await self.memory.current(chat.id)), LLMMessage("user", "Sources:")]
        await self.llm.warm_up([messages])

    async def _history(self, chat_id: str, *, before: int) -> list[Message]:
        """Recent user and agent messages, oldest first, without "stop" turns (they got no answer): the messages from
        the window's start (``HISTORY_MIN``/``HISTORY_STEP`` above) up to ``before``."""
        if before <= 1:
            return []
        start = max(1, (before - 1 - HISTORY_MIN) // HISTORY_STEP * HISTORY_STEP + 1)
        items = [
            m
            for m in (await self.messages.list(chat_id, before=before, limit=HISTORY_MESSAGES)).items
            if m.seq >= start
        ]
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
        renames = p.renames if p is not None else {}
        query = respell(plan.query, renames)  # the documents' spelling of a misheard name (item 10)
        reading = self._english_reading(turn, plan, sources, renames) if p is not None else None
        if reading is not None:  # a garbled Hindi transcript: asked in the router's reading (polish round, item 3)
            query = f'{reading} (asked in Hindi by voice; speech recognition heard: "{turn.text}")'
        if web and p is not None and p.web is not None:  # live data (§3.7): documents [S#] and web results [W#]
            # Page texts are long (prefill ~3 ms per token): the first answer has the snippets, a continuation the
            # pages, unless no continuation will come.
            cfg = self.settings.tools.web_search
            pages = not (cfg.stream_partial_results and cfg.max_continuations > 0)
            system = live_system_prompt(language, length, documents=p.documents_part)
            question = live_user_prompt(query, sources, web, language, search_query=p.web.query, with_content=pages)
            for source in web:
                source.content_given = source.content_given or (pages and bool(source.content))
        elif plan.mode in ("grounded", "mixed"):
            notice = p.notice if p is not None else None
            system = answer_system_prompt(
                language,
                length,
                mixed=plan.mode == "mixed",
                live_note=notice,
                live_hint=plan.live_hint and notice is None,
            )
            route = plan.route
            notes = []
            if (latest := self._latest_period([turn.text, plan.query, plan.query_en], sources)) is not None:
                notes.append(latest_period_note(latest))
            year_ends = list(
                dict.fromkeys(y for t in (turn.text, query, plan.query_en) if t for y in fiscal_year_ends(t))
            )
            if year_ends:  # "31 March 2024" is the end of FY24
                notes.append(fiscal_year_end_note(year_ends))
            if renames:
                notes.append(names_note(renames))
            if (own := self._own_subject(turn, plan, p)) is not None:
                notes.append(own_subject_note(own))
                # Without the conversation: with the old topic in its history the 4B model answered the age rule
                # again however the question and its sources read (3 of 3 runs, with the note and Valmora's passages).
                history = []
            on_screen = None
            if p is not None and p.draft is not None and p.draft_sources:
                ids = [s.source_id for s in p.draft_sources]
                on_screen = on_screen_note(describe(p.draft), ids, confident=p.draft_confident)
            question = answer_user_prompt(
                query,
                sources,
                language,
                name_documents=p is not None and p.name_documents,
                visual_requested=route is not None and route.visual == "requested" and self.canvas is not None,
                screen=p.screen if p is not None else (),
                on_screen=on_screen,
                notes=notes,
            )
        elif plan.mode == "general":
            notice = p.notice if p is not None else None
            prefixed = p is not None and p.prefix is not None
            system = general_system_prompt(language, length, plan.general_note, live_note=notice, prefixed=prefixed)
            question = general_user_prompt(query, language)
        elif plan.mode == "clarification":
            system, question = clarification_system_prompt(language), turn.text
        else:
            system, question = conversation_system_prompt(language), turn.text
        if turn.modality == "voice" and plan.mode in ("grounded", "mixed", "general"):
            # Answers repeated misheard words ("Morris Revenue", "एट्वाई चाँबीस"); a question shown as transcribed
            # gets the router's reading of it too (polish round, item 3)
            understood = None if reading is not None else self._understood_as(turn, plan, renames)
            question = f"{question}\n\n{heard_note(understood)}"
        if plan.language_request:
            question = f"{question}\n\n{language_request_note(language)}"
        return [*self._context(system, history, memory), LLMMessage("user", question)]

    @classmethod
    def _english_reading(
        cls, turn: Turn, plan: TurnPlan, sources: Sequence[Source], renames: Mapping[str, str]
    ) -> str | None:
        """For a spoken Devanagari question most of whose words are in no source (Whisper's Hindi for English terms
        and names: "वाल्मोरा का एट्वाई चाँबीस में रेवेन योग कितना था?"; the answer copied "एट्वाई चाँबीस में रेवेन योग"
        even with the router's reading beside it), the router's English reading, to be asked instead; else None (a
        Hindi question the Hindi sources spell out keeps its own words)."""
        understood = cls._understood_as(turn, plan, renames)
        if understood is None or turn.modality != "voice":
            return None
        words = [
            w
            for w in DEVANAGARI_WORD.findall(respell(turn.text, renames))
            if len(w) >= 3 and w not in HINDI_FUNCTION_WORDS
        ]
        if not words:
            return None
        written = {w for s in sources for w in DEVANAGARI_WORD.findall(s.chunk.text)}
        unknown = sum(1 for w in words if w not in written)
        return understood if unknown * 2 >= len(words) else None

    @staticmethod
    def _understood_as(turn: Turn, plan: TurnPlan, renames: Mapping[str, str]) -> str | None:
        """The router's English reading of a spoken question, when the answer is shown the transcript itself (a Hindi
        or Hinglish question keeps its own words; "Valmora Kar FY-24 May Revenue Kitna Tha?" was answered "The
        documents do not cover the revenue for Valmora Kar FY24 May" with "What was the revenue for Valmora in FY24?"
        in hand). None when the question shown is already the router's (a rewrite) or there is no reading."""
        english = plan.query_en
        if not english or not same_text(plan.query, turn.text) or same_text(english, turn.text):
            return None
        return respell(english, renames)

    @staticmethod
    def _own_subject(turn: Turn, plan: TurnPlan, p: _Progress | None) -> list[str] | None:
        """What the question names when the router's rewrite was dropped for losing it or carrying the last topic over
        (last round, item 4): the answer is told to keep to it (the history still holds the old topic, and the 4B model
        answered "Valmuraka, FY24, Meerajesh" with the age rule again)."""
        if plan.decision is None or not any(o.startswith("rewrite dropped") for o in plan.decision.overrides):
            return None
        labels = {d: document_label(f, None) for d, f in (p.ready if p is not None else {}).items()}
        names = [n[:1].upper() + n[1:] for n in sorted(subject_names(turn.text, labels))]
        periods = [f"FY{y:02d}" for y in sorted(asked_periods(turn.text))]
        return [*names, *periods] or None

    def _own_documents(self, turn: Turn, p: _Progress, chunks: Sequence[RankedChunk]) -> list[RankedChunk]:
        """The passages for the answer: when the router's rewrite was dropped (``_own_subject``) and the question names
        documents of the chat, only theirs (if any were found). With the old topic still in the history, a passage of
        the other document among the sources was what the model answered from (the age rule, 3 of 3 runs)."""
        if self._own_subject(turn, p.plan, p) is None:
            return list(chunks)
        labels = {d: document_label(f, None) for d, f in p.ready.items()}
        names = subject_names(turn.text, labels)
        docs = {d for d, label in labels.items() if label_words(label) & names}
        own = [c for c in chunks if c.chunk.document_id in docs]
        return own or list(chunks)

    @staticmethod
    def _context(system: str, history: Sequence[Message], memory: str | None) -> list[LLMMessage]:
        """The system prompt (with the memory summary) and the recent messages."""
        messages = [LLMMessage("system", with_memory(system, memory))]
        for m in history:
            # Interrupted voice answers: only what was heard ("" when nothing was: the answer is left out).
            text = strip_markers(heard(m))[:HISTORY_CHARS]
            if text and m.role == "agent" and any(c.kind == "web" for c in m.citations):
                text += " (Partly from live web results, not from the documents.)"  # §3.7: never the report's
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
        canvas: dict[str, Any] = {
            # §12.1: what the user's words (or the answer's figures) asked for; visual_status / visual_id only when a
            # visual was started (none for an abstention, a general answer, …)
            "visual": p.visual_want if p.visual_want != "none" else (route.visual if route is not None else "none"),
            **(p.visual.record() if p.visual is not None else {}),
        }
        if p.screen_visual is not None:
            canvas["screen_visual_id"] = p.screen_visual.id  # the visual a question was about
        if p.edit is not None:
            canvas["canvas_edit"] = p.edit
        if p.checks:
            canvas["checks"] = list(p.checks)  # what the answer's checks changed (quality round)
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
            "language_retry": p.language_retry,
            "named_documents": p.name_documents,
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
            **canvas,
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
        route["basis"] = answer_basis(p, answer, citations) if role == "agent" else []
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
                language=script_language(answer) or p.plan.language,
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
        self._record_visual(p, agent)
        yield AgentMessageEvent(agent)

    def _record_visual(self, p: _Progress, message: Message) -> None:
        """The turn's visual on its saved message (``route.visual_id`` once ready, §12.1): written when the visual
        settles, if that is after the save (the transcript then shows "chart added")."""
        visual = p.visual
        if visual is None:
            return
        saved = dict(message.route or {})

        async def write(v: TurnVisual) -> None:
            record = v.record()
            changes = {k: value for k, value in record.items() if saved.get(k) != value}
            if saved.get("visual_id") and record.get("visual_status") != "ready":  # a draft shown, then withdrawn
                changes["visual_id"] = None
                changes.setdefault("visual_status", v.status)
            if changes:
                saved.update(changes)
                await self.messages.update_route(message.id, changes)

        visual.when_settled(write)

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
        route["basis"] = answer_basis(p, text, citations)
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
                language=script_language(text) or p.plan.language,
                citations=citations,
                route=route,
                latency=latency,
            )
        except Exception:
            log.exception("chat %s: saving the stopped answer failed", turn.chat.id)
            return
        if stop is not None:
            stop.saved = message
        self._record_visual(p, message)
        self._advance_state(turn, p, message.citations, completed=False, message_id=message.id)
        log.info("chat %s: answer stopped after %d characters", turn.chat.id, len(text))

    async def _save_complete(self, turn: Turn, p: _Progress, clock: _Clock, stop: AnswerStop | None) -> None:
        """Save an answer whose text was complete when the turn was stopped, with only its live-data continuation
        still to come (§3.7): complete, not interrupted (a voice session records a barge-in during its playback
        itself, as for any complete answer), without the continuation."""
        text, citations = finalize_answer("".join(p.parts), p.citable)
        route = self._route(turn, p, abstained=False, reason=None)
        route["basis"] = answer_basis(p, text, citations)
        latency = self._latency(clock, p, first_delta_ms=p.first_delta_ms, llm_ms=p.llm_ms)
        write = self._advance_state(turn, p, citations, completed=True)
        if write is not None:
            await asyncio.wait([write])
        try:
            message = await self.messages.append(
                turn.chat.id,
                role="agent",
                text=text,
                modality=turn.modality,
                language=script_language(text) or p.plan.language,
                citations=citations,
                route=route,
                latency=latency,
            )
        except Exception:
            log.exception("chat %s: saving the answer failed", turn.chat.id)
            return
        p.final = message
        if stop is not None:
            stop.completed = message
        self._record_visual(p, message)
        log.info("chat %s: answer saved complete, its live-data continuation dropped (stopped)", turn.chat.id)
