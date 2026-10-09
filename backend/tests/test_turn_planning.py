"""Planning a turn (services/planning.py) on fake retrieval and a scripted router model: the fast path, speculative
retrieval used / reused / discarded, the race against a confident retrieval, timeouts and failures falling back to a
document question, the retrieval policy, and (B1) "general" proposals for questions about facts checked against the
documents."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.domain.conversation import ConversationState
from app.providers.llm import LLMUnavailableError
from app.providers.retrieval import IndexedChunk
from app.services.planning import TurnPlanner, interrupted_answer, policy
from app.services.retrieval import RetrievalService
from app.services.router import LLMTurnRouter, RouteDecision, RouteRequest, validate

from .fakes import FakeEmbedder, FakeLLM, FakeReranker, FakeStore, keyword_scorer, make_chunk, vector_for
from .test_routing import MARGIN, history, proposal

PASSAGES = [
    "EBITDA margin was 18.2% in FY24.",
    "EBITDA margin was 16.9% in FY23.",
    "Revenue grew 34% in FY24 on the enterprise segment.",
]
READY = {"doc1": "annual_report.pdf"}


@pytest.fixture
async def world(load_local):
    store = FakeStore(search_points=True)
    for i, text in enumerate(PASSAGES):
        await store.upsert([IndexedChunk(make_chunk(i, text=text), vector_for(text))])
    embedder, reranker, llm = FakeEmbedder(), FakeReranker(keyword_scorer), FakeLLM()
    retrieval = RetrievalService(embedder, reranker, store, load_local().retrieval)
    planner = TurnPlanner(retrieval, LLMTurnRouter(llm), timeout_s=1.0)
    return SimpleNamespace(store=store, embedder=embedder, reranker=reranker, llm=llm, planner=planner)


def req(text: str, hist=(), *, language: str = "en", state: ConversationState | None = None) -> RouteRequest:
    hist = list(hist)
    return RouteRequest(
        text, language, hist, state or ConversationState(chat_id="c"), list(READY.values()), interrupted_answer(hist)
    )  # type: ignore[arg-type]


async def plan(world, request: RouteRequest, **kw):
    return await world.planner.plan(request, project_id="proj1", ready=kw.pop("ready", READY), **kw)


# ------------------------------------------------------------------ fast path and speculation


async def test_a_fast_path_question_never_waits_for_the_router(world):
    p = await plan(world, req("What does the annual report say about the EBITDA margin?"))
    assert p.decision.source == "heuristic" and p.intent == "document_qa" and p.mode == "grounded"
    assert world.llm.json_calls == []  # no model call
    result, outcome = await p.speculation.result_for(p.query, p.query_en)
    assert outcome == "used" and result.chunks[0].chunk.text.startswith("EBITDA margin was 18.2%")
    assert len(world.embedder.calls) == 1 and len(world.reranker.calls) == 1  # retrieval ran once


async def test_a_confident_retrieval_beats_a_slow_router(world):
    world.llm.json_delay = 0.5  # the router is still thinking when retrieval is done
    world.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(world, req("What was the EBITDA margin in FY24?"))
    assert p.decision.source == "retrieval" and p.intent == "document_qa"
    assert world.llm.json_cancelled == 1  # the router call was cancelled, not waited for
    assert p.router_ms < 400 and p.decision.route.confidence >= 0.6
    _, outcome = await p.speculation.result_for(p.query, p.query_en)
    assert outcome == "used"


async def test_an_unsure_retrieval_waits_for_the_router(world):
    world.reranker.scorer = lambda q, passage: 0.3  # above the abstention threshold, below the fast-accept score
    world.llm.json_delay = 0.05
    world.llm.route = {"intent": "document_qa", "query": None}
    p = await plan(world, req("What was the EBITDA margin in FY24?"))
    assert p.decision.source == "llm" and world.llm.json_cancelled == 0
    _, outcome = await p.speculation.result_for(p.query, p.query_en)
    assert outcome == "used" and len(world.reranker.calls) == 1  # the speculative ranking is the turn's


async def test_a_rewritten_query_discards_the_speculation(world):
    world.llm.route = {"intent": "document_qa", "query": "What was the EBITDA margin in FY23?"}
    p = await plan(world, req("And in FY23?", MARGIN))
    assert p.query == "What was the EBITDA margin in FY23?" and p.decision.route.rewritten_query == p.query
    result, outcome = await p.speculation.result_for(p.query, p.query_en)
    assert outcome == "discarded"
    assert world.embedder.calls == [["And in FY23?"], ["What was the EBITDA margin in FY23?"]]
    assert [q for q, _ in world.reranker.calls] == ["What was the EBITDA margin in FY23?"]  # not the raw follow-up
    assert result.chunks[0].chunk.text == "EBITDA margin was 16.9% in FY23."


async def test_a_hindi_question_reuses_the_speculative_search_and_reranks_in_english(world):
    hindi = "वित्त वर्ष 2024 में कंपनी का मुनापा कितना था?"  # misspelled transcript
    world.llm.route = {"intent": "document_qa", "query": "What was the EBITDA margin in FY24?"}
    p = await plan(world, req(hindi, language="hi"))
    assert p.query == hindi and p.query_en == "What was the EBITDA margin in FY24?" and p.language == "hi"
    result, outcome = await p.speculation.result_for(p.query, p.query_en)
    assert outcome == "reused_search"
    assert world.embedder.calls == [[hindi], ["What was the EBITDA margin in FY24?"]]
    assert [q for q, _ in world.reranker.calls] == ["What was the EBITDA margin in FY24?"]  # EN-EN, as §9.2 asks
    assert result.rerank_query == p.query_en and result.confidence.above_threshold


async def test_turns_that_need_no_documents_skip_retrieval(world):
    world.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(world, req("What's the capital of France?", MARGIN))
    assert (p.mode, p.needs_retrieval, p.speculation) == ("general", False, None)
    for text, intent, mode in [
        ("hello", "conversation", "ack"),
        ("okay", "backchannel", "ack"),
        ("stop", "stop", "silent"),
    ]:
        p = await plan(world, req(text, MARGIN))
        assert (p.intent, p.mode, p.needs_retrieval) == (intent, mode, False)


# ------------------------------------------------------------------ fallbacks


async def test_a_slow_router_times_out_into_a_document_question(world):
    world.planner.timeout_s = 0.05
    world.llm.json_delay = 2.0
    world.reranker.scorer = lambda q, passage: 0.3  # no fast accept either
    p = await plan(world, req("And what about debt?", MARGIN))
    assert p.decision.source == "fallback" and "timed out after 50 ms" in p.decision.error
    assert (p.intent, p.mode, p.needs_retrieval, p.query) == ("document_qa", "grounded", True, "And what about debt?")
    assert world.llm.json_cancelled == 1 and p.router_ms < 1000
    assert p.decision.record()["source"] == "fallback"


@pytest.mark.parametrize(
    ("script", "error"),
    [
        ('{"intent": "chit_chat", "query": null}', "doesn't match RouterProposal"),
        ('{"intent": "document_qa"', "doesn't match RouterProposal"),
        (LLMUnavailableError("Ollama unreachable"), "LLMUnavailableError: Ollama unreachable"),
        (None, "no route scripted"),
    ],
)
async def test_invalid_output_or_a_model_down_falls_back(world, script, error):
    world.llm.route = script
    p = await plan(world, req("And what about debt?", MARGIN))
    assert p.decision.source == "fallback" and error in p.decision.error
    assert p.intent == "document_qa" and p.needs_retrieval


# ------------------------------------------------------------------ policy


async def test_document_search_turned_off_answers_from_general_knowledge(world):
    p = await plan(world, req("What does the annual report say about debt?"), retrieval_enabled=False)
    assert (p.intent, p.mode, p.general_note, p.needs_retrieval) == ("document_qa", "general", "retrieval_off", False)
    assert world.embedder.calls == []  # nothing searched, not even speculatively


async def test_mixed_without_documents_is_a_general_answer(world):
    world.llm.route = {"intent": "mixed", "query": None}
    p = await plan(world, req("Is an 18% margin good for a manufacturer?"), ready={})
    assert (p.mode, p.general_note, p.needs_retrieval) == ("general", "no_documents", False)
    world.llm.route = {"intent": "document_qa", "query": None}  # a document question still abstains
    p = await plan(world, req("Is the margin in the report good?"), ready={})
    assert (p.mode, p.needs_retrieval) == ("grounded", True)


async def test_a_correction_repeats_the_kind_of_answer_it_corrects(world):
    general = [
        *history(("user", "What's the capital of France?")),
        history(("agent", "Paris."))[0].model_copy(update={"route": {"answer": "general"}, "seq": 2}),
    ]
    world.llm.route = {"intent": "correction", "query": "What's the capital of Germany?"}
    p = await plan(world, req("no, I meant Germany", general))
    assert (p.intent, p.mode, p.needs_retrieval) == ("correction", "general", False)
    p = await plan(world, req("no, I meant FY23", MARGIN))  # MARGIN's answer was a document answer
    assert (p.intent, p.mode, p.needs_retrieval) == ("correction", "grounded", True)


async def test_a_correction_skips_replies_that_answered_nothing(world):
    def answered(*items: tuple[str, str, str | None]) -> list:
        out = history(*[(role, text) for role, text, _ in items])
        return [
            m.model_copy(update={"route": {"answer": mode}}) if mode else m
            for m, (*_, mode) in zip(out, items, strict=True)
        ]

    world.llm.route = {"intent": "correction", "query": "What was the debt in the annual report?"}
    # the agent asked which report it was: the correction is the first real question, a document one
    asked = answered(("user", "Tell me the number", None), ("agent", "Which report do you mean?", "clarification"))
    p = await plan(world, req("I meant debt in the annual report", asked))
    assert (p.mode, p.needs_retrieval) == ("grounded", True)
    # a general answer, then "okay" → "Anything else?": the correction corrects the general question
    general = answered(
        ("user", "What's the capital of France?", None),
        ("agent", "Paris.", "general"),
        ("user", "okay", None),
        ("agent", "Anything else?", "ack"),
    )
    world.llm.route = {"intent": "correction", "query": "What's the capital of Germany?"}
    p = await plan(world, req("no wait, I meant Germany", general))
    assert (p.mode, p.needs_retrieval) == ("general", False)


async def test_a_validation_bug_falls_back_instead_of_ending_the_turn(world, monkeypatch):
    def broken(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr("app.services.planning.validate", broken)
    world.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(world, req("And what about debt?", MARGIN))
    assert p.decision.source == "fallback" and "validation failed: KeyError" in p.decision.error


# ------------------------------------------------------------------ B1: "general" questions about facts


class SlowReranker(FakeReranker):
    """Retrieval under load: the reranker takes ``delay`` seconds (the router model is faster)."""

    def __init__(self, delay: float) -> None:
        super().__init__(keyword_scorer)
        self.delay = delay

    async def score(self, query, passages):
        await asyncio.sleep(self.delay)
        return await super().score(query, passages)


EMPLOYEES = "How many employees did the company have at year end?"


@pytest.fixture
async def b1(world):
    """The world with a headcount passage and a reranker slower than the router (the B1 race)."""
    text = "The company had 9,842 employees at year end, up from 9,310."
    await world.store.upsert([IndexedChunk(make_chunk(9, text=text, page_start=12, page_end=12), vector_for(text))])
    world.reranker = SlowReranker(0.3)
    world.planner.retrieval.reranker = world.reranker
    world.llm.json_delay = 0.05
    return world


async def test_b1_a_fact_question_routed_general_waits_for_retrieval_and_is_answered_from_the_documents(b1):
    b1.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(b1, req(EMPLOYEES))
    assert (p.intent, p.mode, p.needs_retrieval) == ("document_qa", "grounded", True)
    assert p.decision.overrides[-1].startswith("general_qa→document_qa: the documents match (")
    wait = p.decision.retrieval_wait_ms
    assert 150 < wait < 600 and p.decision.record()["retrieval_wait_ms"] == wait  # the reranker's 0.3 s, not more
    result, outcome = await p.prefetched  # the turn's retrieval: not run again
    assert outcome == "used" and result.chunks[0].chunk.text.startswith("The company had 9,842 employees")
    assert len(b1.reranker.calls) == 1


async def test_b1_no_wait_when_retrieval_finished_first(b1):
    b1.reranker.delay, b1.llm.json_delay = 0.0, 0.3
    b1.planner.fast_accept = 1.1  # no shortcut: the router answers, then its proposal is checked
    b1.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(b1, req(EMPLOYEES))
    assert p.intent == "document_qa" and p.decision.retrieval_wait_ms < 50


async def test_b1_the_wait_is_bounded(b1):
    b1.planner.fact_wait_s, b1.reranker.delay = 0.1, 0.5
    b1.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(b1, req(EMPLOYEES))  # about the documents' subject ("the company"): searched anyway
    assert p.intent == "document_qa" and "retrieval still running" in p.decision.overrides[-1]
    assert 90 < p.decision.retrieval_wait_ms < 250 and not p.prefetched.done()
    result, _ = await p.prefetched  # the turn waits for the rest of it, without searching again
    assert result.chunks and len(b1.reranker.calls) == 1
    p = await plan(b1, req("Who won the cricket world cup in 2011?"))  # nothing points at the documents
    assert (p.intent, p.mode, p.prefetched) == ("general_qa", "general", None)
    assert p.decision.overrides[-1] == "general_qa kept: retrieval too slow (none)"


async def test_b1_definitions_and_how_tos_stay_general_without_waiting(b1):
    text = "EBITDA means earnings before interest, tax, depreciation and amortisation."
    await b1.store.upsert([IndexedChunk(make_chunk(10, text=text), vector_for(text))])
    b1.reranker.scorer = lambda q, passage: 0.95  # the glossary matches very well
    for utterance in ("What is EBITDA?", "What does EBITDA stand for?", "How is EBITDA calculated?"):
        b1.llm.route = {"intent": "general_qa", "query": None}
        p = await plan(b1, req(utterance))
        assert (p.mode, p.needs_retrieval, p.prefetched) == ("general", False, None), utterance
        assert p.decision.retrieval_wait_ms is None


async def test_b1_a_general_fact_the_documents_dont_match_stays_general(b1):
    b1.reranker.delay = 0.05
    b1.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(b1, req("Who won the cricket world cup in 2011?"))
    assert (p.intent, p.mode, p.needs_retrieval, p.prefetched) == ("general_qa", "general", False, None)
    assert p.decision.overrides[-1].startswith("general_qa kept: the documents don't match")


async def test_b1_a_weaker_match_without_the_documents_subject_is_mixed(b1):
    b1.reranker.scorer = lambda q, passage: 0.3 if "supplier" in passage else 0.0
    text = "Fluoropolymer resin depends on a single supplier in Japan."
    await b1.store.upsert([IndexedChunk(make_chunk(11, text=text), vector_for(text))])
    b1.llm.route = {"intent": "conversation", "query": None}
    p = await plan(b1, req("Which product depends on a single supplier?"))
    assert (p.intent, p.mode, p.needs_retrieval) == ("mixed", "mixed", True)
    assert p.decision.overrides[-1] == "conversation→mixed: the documents match (0.30)"


async def test_b1_a_hindi_profit_question_routed_general_searches_in_english(b1):
    hindi = "कंपनी का मुनाफा कितना था?"
    text = "Profit after tax was 871 crore in FY24."
    await b1.store.upsert([IndexedChunk(make_chunk(12, text=text), vector_for(text))])
    b1.llm.route = {"intent": "general_qa", "query": "What was the company's profit after tax?"}
    p = await plan(b1, req(hindi, language="hi"))
    assert (p.intent, p.mode, p.query, p.query_en) == (
        "document_qa",
        "grounded",
        hindi,
        "What was the company's profit after tax?",
    )
    result, outcome = await p.prefetched
    assert outcome == "reused_search" and result.rerank_query == p.query_en
    assert result.chunks[0].chunk.text == text


async def test_b1_a_question_about_the_company_the_documents_dont_answer_abstains(b1):
    b1.reranker.scorer = lambda q, passage: 0.0
    b1.llm.route = {"intent": "general_qa", "query": None}
    p = await plan(b1, req("Who was the company's statutory auditor in FY24?"))
    assert (p.intent, p.mode, p.needs_retrieval) == ("document_qa", "grounded", True)
    assert p.decision.overrides[-1] == "general_qa→document_qa: asks about the documents' subject (best match 0.00)"


def test_a_second_acknowledgement_in_a_row_gets_nothing():
    ack = history(("user", "okay"), ("agent", "Anything else?"))
    ack[1] = ack[1].model_copy(update={"route": {"answer": "ack"}})
    r = req("okay", [*MARGIN, *ack])
    decision = RouteDecision(validate(proposal("backchannel"), r).route, "heuristic")
    assert policy(decision, r, has_documents=True, retrieval_enabled=True).mode == "silent"
    r = req("okay", MARGIN)
    p = policy(
        RouteDecision(validate(proposal("backchannel"), r).route, "heuristic"),
        r,
        has_documents=True,
        retrieval_enabled=True,
    )
    assert (p.mode, p.ack) == ("ack", "ack")
