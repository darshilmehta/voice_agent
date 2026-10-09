"""Planning a turn (docs/DESIGN.md §3.3 d, §3.4, §9.5): its route and what to do with it, latency first.

    keyword fast path ─────► route, no model: stop, backchannel, thanks / greeting, an English standalone question
                             naming the documents
    otherwise, at the same time:
      speculative retrieval of the raw utterance (search; + rerank when it is an English standalone question)
      router model (JSON, two short fields, timeout)
        ├─ a standalone question whose speculative retrieval is confident before the router answers
        │     → document question, router call cancelled (no model wait)
        ├─ the router's proposal, validated (services/router.py)
        └─ timeout / invalid JSON / model down → fallback: a document question on the raw utterance (phase 1)
    retrieval policy (application code, from intent + state):
      document_qa, correction of a document question, resume with a question  → search, grounded answer
      mixed                                                                  → search, grounded + general knowledge
      general_qa, correction of a general question                           → no search, general answer
      conversation / clarification                                           → no search, short reply / question
      thanks / greeting (fast path), backchannel                             → no search, fixed short reply
      resume without a question                                              → no search, fixed text
      stop, or a backchannel right after "Anything else?"                    → nothing is said
      document search off for the chat → general answer saying so; no READY documents → abstain (mixed: general)
    live data (§3.7, router.with_live_tools): a question with a live-data cue gets tools=["web_search"] when the
      tool is available, and the search query built from its English standalone question (live_data.web_query);
      unavailable tool, or no English question to search for → no search, and the answer says live data isn't
      available (``live_note``)

The speculative retrieval is handed to the turn when the route keeps its query (see ``SpeculativeRetrieval``) and
discarded otherwise, so a document turn that agrees with the raw utterance pays for retrieval and the router once,
in parallel, not one after the other.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ..domain.conversation import Intent, TurnRoute
from ..domain.projects import Message
from ..settings import Language
from .language import message_language
from .live_data import LiveNote, web_query
from .prompts import AckKind, AnswerMode, GeneralNote
from .retrieval import RetrievalService, SpeculativeRetrieval
from .router import (
    InterruptedAnswer,
    RouteDecision,
    RouteRequest,
    RouterProposal,
    TurnRouter,
    fallback_route,
    fast_route,
    retrieval_route,
    standalone_question,
    validate,
    with_live_tools,
)

log = logging.getLogger(__name__)

# A standalone question whose speculative retrieval scores at least this (reranker, 0-1) is a document question
# without waiting for the router; well above the abstention threshold, so only clear matches skip the model.
FAST_ACCEPT_SCORE = 0.6


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """What to do with a user message: the validated route and the answer it gets."""

    query: str  # the standalone question to search for and answer (rewritten, or the utterance)
    language: Language
    intent: Intent = "document_qa"
    needs_retrieval: bool = True
    query_en: str | None = None
    mode: AnswerMode = "grounded"
    general_note: GeneralNote | None = None  # why a turn that wanted the documents gets a general answer
    decision: RouteDecision | None = None  # how the route was decided (None: no router, phase 1)
    speculation: SpeculativeRetrieval | None = None
    router_ms: float | None = None  # time spent deciding the route
    interrupted: InterruptedAnswer | None = None
    ack: AckKind | None = None  # the fixed reply of an "ack" turn
    speculated: bool = False  # a speculative retrieval was started (and, without ``speculation``, discarded)
    web_query: str | None = None  # the web search to run (§3.7): only this leaves the machine
    live_note: LiveNote | None = None  # the question wants live data the answer won't have: say so

    @property
    def route(self) -> TurnRoute | None:
        return self.decision.route if self.decision is not None else None

    @property
    def tools(self) -> list[str]:
        return list(self.route.tools) if self.route is not None else []

    def as_general(self, note: GeneralNote) -> TurnPlan:
        """A mixed question the documents don't cover: answered from general knowledge, saying so."""
        return replace(self, mode="general", general_note=note)

    def without_live_data(self, note: LiveNote) -> TurnPlan:
        """The web search gave nothing (timeout, failure, no results): answer without it, saying so."""
        return replace(self, live_note=note)


class DocumentQARouter:
    """Phase 1's router: every turn is a document question (tests; or to switch the router model off)."""

    async def propose(self, request: RouteRequest) -> RouterProposal:
        return RouterProposal(intent="document_qa", query=None)


def interrupted_answer(history: Sequence[Message]) -> InterruptedAnswer | None:
    """The agent's latest answer if the user cut it off (voice ``heard_text``, or a stopped answer), with the
    question it answered."""
    agents = [i for i, m in enumerate(history) if m.role == "agent"]
    if not agents:
        return None
    i = agents[-1]
    answer = history[i]
    if answer.heard_text is None and not (answer.route or {}).get("stopped"):
        return None
    question = next((m.text for m in reversed(history[:i]) if m.role == "user"), None)
    heard = answer.heard_text if answer.heard_text is not None else answer.text
    return InterruptedAnswer(answer.id, heard, question)


def _corrected_mode(history: Sequence[Message]) -> AnswerMode:
    """How the question a correction corrects was answered: a correction repeats that kind of answer (§9.1: a
    correction inherits the retrieval decision of the turn it corrects). Replies that answered no question (a
    clarifying question, "Anything else?", "back to the report", small talk) are skipped; with no answered question
    before (or an answer saved before routing existed), it is a document question."""
    for m in reversed(history):
        mode = (m.route or {}).get("answer") if m.role == "agent" else None
        if mode == "general":
            return "general"
        if mode == "mixed":
            return "mixed"
        if mode == "grounded" or (m.role == "agent" and mode is None):
            return "grounded"
    return "grounded"


_MODES: dict[Intent, tuple[AnswerMode, bool]] = {
    "document_qa": ("grounded", True),
    "mixed": ("mixed", True),
    "general_qa": ("general", False),
    "conversation": ("conversation", False),
    "clarification": ("clarification", False),
    "stop": ("silent", False),
    "backchannel": ("ack", False),  # "Anything else?"; nothing if that was just said (below)
    "resume_document": ("resume", False),  # with a question: grounded (below)
    "correction": ("grounded", True),  # of a general question: general (below)
}


@dataclass(frozen=True, slots=True)
class Policy:
    decision: RouteDecision  # with ``needs_retrieval`` set by the policy
    mode: AnswerMode
    note: GeneralNote | None = None
    ack: AckKind | None = None


def policy(
    decision: RouteDecision,
    req: RouteRequest,
    *,
    has_documents: bool,
    retrieval_enabled: bool,
    speculation: SpeculativeRetrieval | None = None,
    fast_accept: float = FAST_ACCEPT_SCORE,
) -> Policy:
    """Retrieval policy: the answer mode and whether to search, from the validated route and the chat's state."""
    route = decision.route
    intent = route.intent
    mode, search = _MODES[intent]
    ack: AckKind | None = None
    overrides: list[str] = []
    if intent == "backchannel":
        last = req.last_answer
        if last is not None and (last.route or {}).get("answer") == "ack":
            mode = "silent"  # "okay" after "Anything else?": don't nag
        else:
            ack = "ack"
    elif intent == "conversation" and decision.reply is not None:
        mode, ack = "ack", decision.reply
    elif intent == "resume_document" and route.rewritten_query is not None:
        mode, search = "grounded", True
    elif intent == "correction":
        mode = _corrected_mode(req.history)
        search = mode != "general"
    elif intent == "general_qa" and speculation is not None and route.rewritten_query is None:
        ranked = speculation.peek()  # only if already there: never wait for it here
        if ranked is not None and ranked.confidence is not None and ranked.confidence.top_score >= fast_accept:
            mode, search = "mixed", True
            overrides.append(f"general_qa→mixed: documents match ({ranked.confidence.top_score:.2f})")
    note: GeneralNote | None = None
    if search and not retrieval_enabled:
        mode, search, note = "general", False, "retrieval_off"
    elif search and not has_documents and mode == "mixed":
        mode, search, note = "general", False, "no_documents"
    route = route.model_copy(update={"needs_retrieval": search})
    decision = replace(decision, route=route, overrides=decision.overrides + tuple(overrides))
    return Policy(decision, mode, note, ack)


def live_search(decision: RouteDecision, req: RouteRequest) -> tuple[RouteDecision, str | None, LiveNote | None]:
    """The web search a live question runs (its query), or why it runs none (§3.7): the tool isn't available, or
    there is no English question to search for (a Hindi turn the router couldn't translate: the utterance itself
    never leaves the machine)."""
    if decision.live is None:
        return decision, None, None
    route = decision.route
    if "web_search" not in route.tools:
        return decision, None, "unavailable"
    english = route.query_en or route.rewritten_query or req.utterance
    query = web_query(english, documents=req.documents, utterance=req.utterance)
    if query is None:
        overrides = (*decision.overrides, "web_search dropped: no English search query")
        return replace(decision, route=route.model_copy(update={"tools": []}), overrides=overrides), None, "failed"
    return decision, query, None


class TurnPlanner:
    def __init__(
        self,
        retrieval: RetrievalService,
        router: TurnRouter,
        *,
        timeout_s: float,
        fast_accept: float = FAST_ACCEPT_SCORE,
    ) -> None:
        self.retrieval = retrieval
        self.router = router
        self.timeout_s = timeout_s
        self.fast_accept = fast_accept

    async def plan(
        self,
        req: RouteRequest,
        *,
        project_id: str,
        ready: Mapping[str, str],
        retrieval_enabled: bool = True,
    ) -> TurnPlan:
        """The turn's plan. ``ready``: the chat's READY documents (id → filename), the retrieval scope."""
        t0 = time.perf_counter()
        decision = fast_route(req)
        speculation: SpeculativeRetrieval | None = None
        if ready and retrieval_enabled and (decision is None or decision.route.needs_retrieval):
            # Rerank speculatively only what will likely be kept: the fast path's query, or an English standalone
            # question (a Hindi one gets an English query from the router, which the reranker should score instead).
            rerank = decision is not None or (standalone_question(req) and message_language(req.utterance) == "en")
            query = (decision.route.rewritten_query if decision is not None else None) or req.utterance
            speculation = SpeculativeRetrieval(
                self.retrieval, query, project_id=project_id, document_ids=list(ready), rerank=rerank
            )
        try:
            if decision is None:
                decision = await self._ask_router(req, speculation)
            decision = with_live_tools(decision, req)
            p = policy(
                decision,
                req,
                has_documents=bool(ready),
                retrieval_enabled=retrieval_enabled,
                speculation=speculation,
                fast_accept=self.fast_accept,
            )
        except BaseException:
            if speculation is not None:
                speculation.discard()
            raise
        route = p.decision.route
        if speculation is not None and not route.needs_retrieval:
            speculation.discard()
        decision, query, note = live_search(p.decision, req)
        return TurnPlan(
            query=route.rewritten_query or req.utterance,
            language=req.language,
            intent=route.intent,
            needs_retrieval=route.needs_retrieval,
            query_en=route.query_en,
            mode=p.mode,
            general_note=p.note,
            decision=decision,
            speculation=speculation if route.needs_retrieval else None,
            router_ms=round((time.perf_counter() - t0) * 1000, 1),
            interrupted=req.interrupted,
            ack=p.ack,
            speculated=speculation is not None,
            web_query=query,
            live_note=note,
        )

    async def _ask_router(self, req: RouteRequest, speculation: SpeculativeRetrieval | None) -> RouteDecision:
        """The router model's validated proposal, raced against a confident speculative retrieval for standalone
        questions; falls back to a document question when the model fails or is too slow."""
        t0 = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - t0) * 1000, 1)

        call = asyncio.ensure_future(asyncio.wait_for(self.router.propose(req), self.timeout_s))
        waiter: asyncio.Future[object] | None = None
        try:
            if speculation is not None and speculation.reranks and standalone_question(req):
                waiter = asyncio.ensure_future(speculation.wait_ranked())
                done, _ = await asyncio.wait({call, waiter}, return_when=asyncio.FIRST_COMPLETED)
                if call not in done:
                    ranked = speculation.peek()
                    score = ranked.confidence.top_score if ranked is not None and ranked.confidence else 0.0
                    if score >= self.fast_accept:
                        call.cancel()
                        return replace(retrieval_route(req, score), llm_ms=elapsed())
            proposal = await call
        except TimeoutError:
            log.warning("router timed out after %.0f ms; answering as a document question", elapsed())
            return fallback_route(req, f"router timed out after {self.timeout_s * 1000:.0f} ms", elapsed())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            log.warning("router failed (%s); answering as a document question", detail)
            return fallback_route(req, detail, elapsed())
        finally:
            # A router call still running is cancelled (closing its request stops Ollama's generation). asyncio.wait
            # doesn't raise the tasks' errors, and a cancellation of this turn still propagates.
            pending = [t for t in (call, waiter) if t is not None and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(pending)
        try:
            return validate(proposal, req, llm_ms=elapsed())
        except Exception as e:  # a bug in validation must not end the turn without an answer
            log.exception("router proposal %s could not be validated; answering as a document question", proposal)
            return fallback_route(req, f"validation failed: {type(e).__name__}: {e}", elapsed())
