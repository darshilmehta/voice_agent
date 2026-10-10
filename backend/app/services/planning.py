"""Planning a turn (docs/DESIGN.md §3.3 d, §3.4, §9.5): its route and what to do with it, latency first.

    keyword fast path ─────► route, no model: stop, backchannel, thanks / greeting, an English standalone question
                             naming the documents
    otherwise, at the same time:
      speculative retrieval of the raw utterance (search; + rerank when it is an English standalone question)
      router model (JSON, two short fields, timeout)
        ├─ a standalone question whose speculative retrieval is confident before the router answers
        │     → document question, router call cancelled (no model wait)
        ├─ the router's proposal, validated (services/router.py)
        │     "general", "conversation" or "clarification" for a question about facts, in a chat with READY documents
        │     (B1; a clarification only when the question needs no conversation to be understood): wait for
        │     the speculative retrieval (at most ``FACT_WAIT_S`` more); it passes the confidence gate → a document
        │     question (strong match, or the question names the documents' subject) or a mixed one; it doesn't, but
        │     the question is about the documents' subject ("the company", "FY24") → a document question that
        │     abstains; otherwise general, as proposed. Definitions and how-tos ("What is EBITDA?") stay general.
        └─ timeout / invalid JSON / model down → fallback: a document question on the raw utterance (phase 1)
    retrieval policy (application code, from intent + state):
      document_qa, correction of a document question, resume with a question  → search, grounded answer
      mixed                                                                  → search, grounded + general knowledge
      general_qa, correction of a general question                           → no search, general answer
      conversation / clarification                                           → no search, short reply / question
      thanks / greeting (fast path), backchannel                             → no search, fixed short reply
      resume without a question                                              → no search, fixed text
      stop, or a backchannel right after "Anything else?"                    → nothing is said
      a garbled voice transcript (fast path, ``RouteRequest.garbled``)        → "Sorry, I didn't catch that. Could
                                                                               you say it again?"; nothing after one
      canvas_edit ("make that a bar chart", "हटा दो", §12.1)                  → no search, the edit, "Done."
      document search off for the chat → general answer saying so; no READY documents → abstain (mixed: general)
    route.visual (§12.1): "requested" / "suggest" from the user's words for document, mixed and correction turns
      answered from the documents, else "none"
    live data (§3.7, router.with_live_tools): a question with a live-data cue gets tools=["web_search"] when the
      tool is available, and the search query built from its English standalone question (live_data.web_query);
      no search otherwise (``live_hint``: the answer never guesses current figures); with web search turned on but
      unavailable now, ``live_note`` lets the answer say so (turned off: no note, nothing to apologise for); no
      usable search query (a document question that merely contains a cue: "as of today, per the report…") → no
      search and no note, only the hint, as with the tool off

The speculative retrieval is handed to the turn when the route keeps its query (see ``SpeculativeRetrieval``) and
discarded otherwise, so a document turn that agrees with the raw utterance pays for retrieval and the router once,
in parallel, not one after the other. A retrieval waited for to check a "general" proposal is handed to the turn
too (``TurnPlan.prefetched``): it is never run twice.
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
from .canvas.conversation import refers_to_screen, visual_want
from .language import message_language
from .live_data import LiveNote, web_query
from .prompts import AckKind, AnswerMode, GeneralNote
from .retrieval import RetrievalResult, RetrievalService, SpeculationOutcome, SpeculativeRetrieval
from .router import (
    CONFIDENCE,
    InterruptedAnswer,
    RouteDecision,
    RouteRequest,
    RouterProposal,
    TurnRouter,
    about_the_documents,
    asks_about_facts,
    asks_for_figure,
    asks_for_judgement,
    fallback_route,
    fast_route,
    leans_on_the_conversation,
    names_a_subject,
    retrieval_route,
    standalone_question,
    validate,
    with_live_tools,
)

log = logging.getLogger(__name__)

# A standalone question whose speculative retrieval scores at least this (reranker, 0-1) is a document question
# without waiting for the router; well above the abstention threshold, so only clear matches skip the model.
FAST_ACCEPT_SCORE = 0.6
# A "general" proposal for a question about facts waits at most this long after the router for the speculative
# retrieval (B1): measured under load, retrieval takes 1.0-2.5 s and the router 0.9-1.4 s.
FACT_WAIT_S = 1.5

Prefetched = asyncio.Future[tuple[RetrievalResult, SpeculationOutcome]]


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
    # Web search is on, but this answer won't have live data (unavailable, nothing to search for, or it failed): the
    # answer may say so (``ChatTurnService`` decides when). None when web search is turned off.
    live_note: LiveNote | None = None
    live_hint: bool = False  # the question wants live data and gets no search: never guess current figures
    # The turn's retrieval, already started (or done) while the route was decided (B1): awaited instead of searching
    # again. Cancelled with the plan when the route needs no retrieval.
    prefetched: Prefetched | None = None

    @property
    def language_request(self) -> bool:
        """The user asked for this answer language and nothing else: the previous question again (B5)."""
        return self.decision is not None and self.decision.language_request

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


def said_again(req: RouteRequest) -> bool:
    """The previous reply was the "say it again" line and was heard in full: another garbled transcript gets nothing
    (a TV keeps talking). Cut short, it is said again (last round, item 6: the reversed Hindi clip went on over it, so
    the user heard none of it, and the next part got only a silent "Not understood")."""
    last = req.last_answer
    if last is None:
        return False
    route = last.route or {}
    if route.get("answer") != "ack" or route.get("intent") != "clarification" or route.get("stopped"):
        return False
    return last.heard_text is None or last.heard_text.strip() == last.text.strip()


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
    "canvas_edit": ("canvas", False),  # the edit applied, then "Done." / "हो गया।" (§12.1)
}
# Turns whose answer may get a visual on the canvas (§12.1), when answered from the documents.
VISUAL_INTENTS: frozenset[Intent] = frozenset({"document_qa", "mixed", "correction"})


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
) -> Policy:
    """Retrieval policy: the answer mode and whether to search, from the validated route and the chat's state. (A
    "general" proposal for a question about facts was checked against the documents before, ``TurnPlanner``.)"""
    route = decision.route
    intent = route.intent
    mode, search = _MODES[intent]
    ack: AckKind | None = None
    if decision.reply == "repeat":  # a garbled transcript: "Sorry, I didn't catch that…", once (then nothing)
        mode, ack = ("silent" if said_again(req) else "ack"), "repeat"
    elif intent == "backchannel":
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
    note: GeneralNote | None = None
    if search and not retrieval_enabled:
        mode, search, note = "general", False, "retrieval_off"
    elif search and not has_documents and mode == "mixed":
        mode, search, note = "general", False, "no_documents"
    # route.visual (§12.1): a document answer may get a visual when the user's words call for one (the answer's own
    # figures may still suggest one later, services/chat_turns.py).
    # A question about a chart on screen ("what's the second bar on the chart?") is answered, not drawn again.
    about_the_screen = bool(req.screen) and refers_to_screen(req.utterance)
    visual = (
        visual_want(req.utterance, route.rewritten_query, route.query_en)
        if intent in VISUAL_INTENTS and mode in ("grounded", "mixed") and not about_the_screen
        else "none"
    )
    route = route.model_copy(update={"needs_retrieval": search, "visual": visual})
    return Policy(replace(decision, route=route), mode, note, ack)


def live_search(decision: RouteDecision, req: RouteRequest) -> tuple[RouteDecision, str | None, LiveNote | None]:
    """The web search a live question runs (its query), or why it runs none (§3.7): the tool isn't available (a
    note, when it is turned on), or there is no usable search query (no note: a document question that merely
    contains a cue, "as of today, how many employees… per the report?", or a Hindi turn the router couldn't
    translate; the answer only gets the "never guess current figures" hint, as with the tool off)."""
    if decision.live is None:
        return decision, None, None
    route = decision.route
    if "web_search" not in route.tools:  # turned off (no note: nothing to apologise for), or can't run now
        return decision, None, "unavailable" if "web_search" in req.enabled_tools | req.available_tools else None
    english = route.query_en or route.rewritten_query or req.utterance
    query = web_query(english, documents=req.documents, utterance=req.utterance)
    if query is None:
        overrides = (*decision.overrides, "web_search dropped: no usable search query")
        return replace(decision, route=route.model_copy(update={"tools": []}), overrides=overrides), None, None
    return decision, query, None


class TurnPlanner:
    def __init__(
        self,
        retrieval: RetrievalService,
        router: TurnRouter,
        *,
        timeout_s: float,
        fast_accept: float = FAST_ACCEPT_SCORE,
        fact_wait_s: float = FACT_WAIT_S,
    ) -> None:
        self.retrieval = retrieval
        self.router = router
        self.timeout_s = timeout_s
        self.fast_accept = fast_accept
        self.fact_wait_s = fact_wait_s

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
        prefetched: Prefetched | None = None
        try:
            if decision is None:
                decision = await self._ask_router(req, speculation)
                decision, prefetched = await self._check_facts(decision, req, speculation)
            decision = with_live_tools(decision, req)
            p = policy(decision, req, has_documents=bool(ready), retrieval_enabled=retrieval_enabled)
        except BaseException:
            if prefetched is not None:
                prefetched.cancel()
            if speculation is not None:
                speculation.discard()
            raise
        route = p.decision.route
        if not route.needs_retrieval:
            if prefetched is not None:
                prefetched.cancel()
                prefetched = None
            if speculation is not None:
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
            live_hint=decision.live is not None and query is None,
            prefetched=prefetched,
        )

    async def _check_facts(
        self, decision: RouteDecision, req: RouteRequest, speculation: SpeculativeRetrieval | None
    ) -> tuple[RouteDecision, Prefetched | None]:
        """A "general" (or "conversation") proposal for a question about facts, in a chat with READY documents: the
        documents may well answer it (the 4B router labels "How many employees did the company have at year end?"
        general), so wait for the speculative retrieval, at most ``fact_wait_s`` (module docstring, B1). The same for a
        "clarification" of a question that stands on its own: the router answers "What is the hotel limit per night for
        level 3 and level 4 employees in tier 1 cities?" with "which organization do you mean?", while the documents
        have the answer (a clarification is for a reference to something never mentioned: "what about that one?", which
        ``leans_on_the_conversation``). Returns the decision (upgraded or not) and the retrieval to hand to the turn."""
        route, proposal = decision.route, decision.proposal
        if (
            decision.source != "llm"
            or proposal is None
            or route.intent not in ("general_qa", "conversation", "clarification")
            or (route.intent == "clarification" and leans_on_the_conversation(req.utterance))
            or speculation is None
            or not (asks_about_facts(req.utterance) or names_a_subject(req.utterance, req.documents))
            or req.live_cue(route.rewritten_query, proposal.query) is not None  # live data: the web search decides
        ):
            return decision, None
        about = (
            about_the_documents(req.utterance, req.documents)
            or names_a_subject(req.utterance, req.documents)
            or (route.rewritten_query is not None and about_the_documents(route.rewritten_query, req.documents))
        )
        # An amount, a number, a limit, a rate or a date: the documents answer it or nothing does. A general answer
        # would give a figure from general knowledge that passes for theirs (last round, item 1: "तूलकेच के लिए कितनी
        # सहायता मिलती है?", misheard "टूलकिट…", got "₹ 1,500 प्रति माह"; the notice says ₹ 10,000).
        figure = not about and any(asks_for_figure(t) for t in (req.utterance, route.rewritten_query, proposal.query))
        about = about or figure
        searched = validate(proposal.model_copy(update={"intent": "document_qa"}), req, llm_ms=decision.llm_ms)
        query = searched.route.rewritten_query or req.utterance
        t0 = time.perf_counter()
        lookup: Prefetched = asyncio.ensure_future(speculation.result_for(query, searched.route.query_en))
        try:
            await asyncio.wait({lookup}, timeout=self.fact_wait_s)
        except BaseException:
            lookup.cancel()
            raise
        waited = round((time.perf_counter() - t0) * 1000, 1)
        found = lookup.done() and not lookup.cancelled() and lookup.exception() is None
        confidence = lookup.result()[0].confidence if found else None
        score = f"{confidence.top_score:.2f}" if confidence is not None else "none"
        was = route.intent
        intent: Intent
        if confidence is not None and confidence.above_threshold:
            strong = about or confidence.top_score >= self.fast_accept
            intent = "document_qa" if strong and not asks_for_judgement(req.utterance) else "mixed"
            why = f"{was}→{intent}: the documents match ({score})"
        elif about:  # about the documents' subject: searched, and if they don't say, the answer abstains
            intent = "document_qa"
            found_part = f"best match {score}" if lookup.done() else "retrieval still running"
            asks = "asks for a figure" if figure else "asks about the documents' subject"
            why = f"{was}→document_qa: {asks} ({found_part})"
        else:
            lookup.cancel()
            kept = f"{was} kept: " + ("the documents don't match" if lookup.done() else "retrieval too slow")
            overrides = (*decision.overrides, f"{kept} ({score})")
            return replace(decision, overrides=overrides, retrieval_wait_ms=waited), None
        upgraded = searched.route.model_copy(update={"intent": intent, "confidence": CONFIDENCE["llm_overridden"]})
        overrides = (*decision.overrides, *searched.overrides, why)
        return replace(searched, route=upgraded, overrides=overrides, retrieval_wait_ms=waited), lookup

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
