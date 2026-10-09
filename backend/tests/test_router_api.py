"""The router over HTTP: every intent through POST /api/chats/{id}/messages (the SSE contract stays the same for all
of them: user_message, sources, delta…, agent_message), with fake retrieval and a scripted router model."""

from __future__ import annotations

from typing import Any

import pytest

from .test_chat_api import ask, names, new_chat, payload, project_with_report
from .test_conversation_turns import routes, scripted_reply

PREVIOUS = "What was the EBITDA margin in FY24?"

CASES: list[dict[str, Any]] = [
    # utterance, the router model's proposal (None: not asked), what the turn does
    {
        "say": PREVIOUS,
        "route": {"intent": "document_qa", "query": None},
        "intent": "document_qa",
        "answer": "grounded",
        "sources": True,
    },
    {
        "say": "What's the capital of France?",
        "route": {"intent": "general_qa", "query": None},
        "intent": "general_qa",
        "answer": "general",
    },
    {
        "say": "Is that margin good for a manufacturer?",
        "route": {"intent": "mixed", "query": "Is an 18.2% EBITDA margin good for a manufacturer?"},
        "intent": "mixed",
        "answer": "mixed",
        "sources": True,
    },
    {
        "say": "Who are you?",
        "route": {"intent": "conversation", "query": None},
        "intent": "conversation",
        "answer": "conversation",
    },
    {"say": "Thanks a lot!", "route": None, "intent": "conversation", "answer": "ack", "text": "You're welcome."},
    {
        "say": "Let's go back to the report",
        "route": {"intent": "resume_document", "query": None},
        "intent": "resume_document",
        "answer": "resume",
    },
    {
        "say": "No, I meant FY23",
        "route": {"intent": "correction", "query": "What was the EBITDA margin in FY23?"},
        "intent": "correction",
        "answer": "grounded",
        "sources": True,
    },
    {
        "say": "What about that one?",
        "route": {"intent": "clarification", "query": None},
        "intent": "clarification",
        "answer": "clarification",
    },
    {"say": "okay", "route": None, "intent": "backchannel", "answer": "ack", "text": "Anything else?"},
    {"say": "stop", "route": None, "intent": "stop", "answer": "silent", "silent": True},
]


@pytest.mark.parametrize("case", CASES, ids=[c["intent"] + ":" + c["say"] for c in CASES])
def test_every_intent_over_sse(app, fakes, case):
    project, _ = project_with_report(app)
    chat = new_chat(app, project)
    fakes.llm.reply = scripted_reply
    fakes.llm.route = routes({PREVIOUS: {"intent": "document_qa", "query": None}, case["say"]: case["route"]})
    if case["say"] != PREVIOUS:
        ask(app, chat, PREVIOUS)  # a document answer before: something to correct, resume or acknowledge
    json_calls = len(fakes.llm.json_calls)

    events = ask(app, chat, case["say"])
    assert names(events)[:2] == ["user_message", "sources"] and names(events)[-1] == "agent_message"
    deltas = [d["text"] for e, d in events if e == "delta"]
    sources = payload(events, "sources")
    agent = payload(events, "agent_message")
    route = agent["route"]
    assert (route["intent"], route["answer"]) == (case["intent"], case["answer"])
    assert route["abstained"] is False and sources["abstained"] is False  # never an abstention here
    assert route["stopped"] is False and isinstance(route["is_topic_shift"], bool)
    assert {"rewritten_query", "query_en", "topic", "response_language", "router", "speculation"} <= set(route)
    assert bool(sources["sources"]) == case.get("sources", False) == route["needs_retrieval"]
    assert {"router_ms", "router_llm_ms", "retrieval_ms", "first_delta_ms", "total_ms"} <= set(agent["latency"])
    if case.get("silent"):
        assert deltas == [] and agent["role"] == "event"
    else:
        assert agent["role"] == "agent" and "".join(deltas).strip()
    if "text" in case:
        assert agent["text"] == case["text"]
    if case["route"] is None:
        assert len(fakes.llm.json_calls) == json_calls  # the fast path: no router model call
    if case.get("sources"):
        assert agent["citations"]
    else:
        assert agent["citations"] == []
    transcript = app.get(f"/api/chats/{chat}/messages").json()["items"]
    assert transcript[-1] == agent  # persisted as streamed


def test_a_document_question_the_documents_dont_cover_is_the_only_abstention(app, fakes):
    project, _ = project_with_report(app)
    chat = new_chat(app, project)
    fakes.llm.route = routes(
        {
            "What is the CEO's salary?": {"intent": "document_qa", "query": None},
            "What is the tallest mountain?": {"intent": "general_qa", "query": None},
        }
    )
    abstained = payload(ask(app, chat, "What is the CEO's salary?"), "agent_message")["route"]
    general = payload(ask(app, chat, "What is the tallest mountain?"), "agent_message")["route"]
    assert (abstained["intent"], abstained["abstained"], abstained["abstain_reason"]) == (
        "document_qa",
        True,
        "not_covered",
    )
    assert (general["intent"], general["abstained"], general["abstain_reason"]) == ("general_qa", False, None)


def test_a_failing_router_answers_as_in_phase_one(app, fakes):
    project, _ = project_with_report(app)
    chat = new_chat(app, project)
    fakes.llm.route = None  # the router model fails on every call
    events = ask(app, chat, PREVIOUS)
    agent = payload(events, "agent_message")
    assert agent["route"]["intent"] == "document_qa" and agent["route"]["router"]["source"] in ("fallback", "retrieval")
    assert payload(events, "sources")["sources"]
