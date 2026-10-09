"""Routing to the live-data tool (docs/DESIGN.md §3.7), no model: live-data cues in EN / HI / Hinglish, the search
query that may leave the machine, and how the validated route gets ``tools=["web_search"]``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.conversation import ConversationState
from app.services.live_data import live_data_cue, web_query
from app.services.planning import live_search
from app.services.router import (
    RouteDecision,
    RouteRequest,
    RouterProposal,
    _route,
    fallback_route,
    fast_route,
    validate,
    with_live_tools,
)

CASES = json.loads((Path(__file__).parent / "integration/router_cases.json").read_text(encoding="utf-8"))
WEB = frozenset({"web_search"})


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("How is the stock doing today?", "doing today"),
        ("The report says revenue grew 34%; how is the stock doing today?", "doing today"),
        ("What's the latest news about Infosys?", "latest news"),
        ("What is the USD to INR exchange rate?", "usd to inr"),
        ("What is the current share price of TCS?", "current share price"),
        ("Is Infosys trading higher right now?", "right now"),
        ("What is the current repo rate?", "current repo rate"),
        ("How are the shares doing?", "shares doing"),
        ("What's the weather in Mumbai?", "weather"),
        ("आज डॉलर का रेट क्या है?", "आज"),
        ("इंफोसिस के शेयर का भाव क्या है?", "शेयर का भाव"),
        ("ताज़ा ख़बर क्या है?", "ताज़ा"),
        ("abhi Infosys ka share price kya hai?", "abhi"),
        ("aaj Sensex kitna upar gaya?", "aaj"),
        ("Infosys ki koi khabar hai kya?", "khabar"),
    ],
)
def test_questions_that_need_live_data(text, cue):
    assert live_data_cue(text) == cue


@pytest.mark.parametrize(
    "text",
    [
        "What was the EBITDA margin in FY24?",
        "What is the current ratio?",  # accounting terms
        "What were the current assets?",
        "What were the current year figures?",
        "What are the current debt levels?",
        "What is the current interest rate on the term loan?",
        "What does the latest annual report say about debt?",
        "What does the report say about the share price?",  # a topic cue that points at the documents
        "Summarize the news section of the report.",
        "What is the revenue forecast for FY25?",
        "अभी तक कितना राजस्व हुआ?",  # "so far"
        "आजकल लोग क्या पढ़ते हैं?",  # "nowadays"
        "रिपोर्ट में शेयर की कीमत क्या बताई गई है?",
        "FY24 mein revenue kitna tha?",
        "Abhi Sharma is the CFO, right?",  # an English sentence: Hinglish words don't count
    ],
)
def test_questions_that_dont(text):
    assert live_data_cue(text) is None


def test_naming_a_document_points_at_the_documents():
    assert live_data_cue("What does annual report say about headlines?") == "headlines"
    assert live_data_cue("What does annual report say about headlines?", documents=["annual_report.pdf"]) is None


@pytest.mark.parametrize(
    ("question", "query"),
    [
        ("The report says revenue grew 34%; how is the stock doing today?", "how is the stock doing today?"),
        (
            "Given revenue grew 34% per the report, how is Infosys stock doing today?",
            "how is Infosys stock doing today?",
        ),
        ("And what is the latest news about Infosys?", "what is the latest news about Infosys?"),
        ("What is the latest news? Mail it to me at ceo@example.com", "What is the latest news?"),
        ("Call +91 98765 43210 and tell me the share price today", "Call and tell me the share price today"),
        ("आज डॉलर का रेट क्या है?", None),  # never the Hindi utterance itself
        ("abhi Infosys ka share price kya hai?", None),  # nor romanized Hindi
        ("", None),
    ],
)
def test_the_search_query_is_only_the_english_live_part(question, query):
    assert web_query(question) == query


def test_long_questions_are_cut():
    assert len(web_query("What is the latest news about " + "very " * 60 + "big companies?") or "") <= 160


# ------------------------------------------------------------------ the route


def request(text: str, *, tools: frozenset[str] = WEB, history=(), documents=("annual_report.pdf",)) -> RouteRequest:
    return RouteRequest(text, "en", list(history), ConversationState(chat_id="c"), list(documents), None, tools)


def decision_for(req: RouteRequest, intent: str, **kw) -> RouteDecision:
    return RouteDecision(_route(req, intent, confidence=0.8, **kw), "llm")  # type: ignore[arg-type]


def test_a_live_question_gets_the_tool_when_it_is_available():
    req = request("What's the latest news about Infosys?")
    d = with_live_tools(decision_for(req, "general_qa"), req)
    assert d.route.tools == ["web_search"] and d.live == "latest news"
    assert d.record()["live_cue"] == "latest news"
    _, query, note = live_search(d, req)
    assert (query, note) == ("What's the latest news about Infosys?", None)


def test_unavailable_tool_records_the_cue_so_the_answer_can_say_so():
    req = request("What's the latest news about Infosys?", tools=frozenset())
    d = with_live_tools(decision_for(req, "general_qa"), req)
    assert d.route.tools == [] and d.live == "latest news"
    assert live_search(d, req)[1:] == (None, "unavailable")


@pytest.mark.parametrize("intent", ["conversation", "clarification", "stop", "backchannel"])
def test_turns_that_ask_nothing_never_search(intent):
    req = request("today is a good day, thanks")
    d = with_live_tools(decision_for(req, intent), req)
    assert d.route.tools == [] and d.live is None


def test_a_non_live_question_never_searches():
    req = request("What was the EBITDA margin in FY24?")
    d = with_live_tools(decision_for(req, "document_qa"), req)
    assert d.route.tools == [] and d.live is None and live_search(d, req) == (d, None, None)


def test_a_follow_up_is_live_through_its_standalone_question():
    req = request("and Wipro?")
    d = with_live_tools(decision_for(req, "general_qa", query="How is Wipro's stock doing today?"), req)
    assert d.route.tools == ["web_search"]
    assert live_search(d, req)[1] == "How is Wipro's stock doing today?"


def test_a_live_question_about_the_documents_skips_the_document_fast_path():
    text = "The report says revenue grew 34%; how is the stock doing today?"
    assert fast_route(request(text, tools=frozenset())) is not None  # without the tool: a document question
    assert fast_route(request(text)) is None  # with it: the router writes the standalone search question


def test_hindi_live_questions_keep_the_english_query_for_the_web():
    req = RouteRequest("आज डॉलर का रेट क्या है?", "hi", [], ConversationState(chat_id="c"), [], None, WEB)
    proposal = RouterProposal(intent="general_qa", query="What is the dollar rate today?")
    d = with_live_tools(validate(proposal, req), req)
    assert d.route.query_en == "What is the dollar rate today?" and d.route.tools == ["web_search"]
    assert live_search(d, req)[1] == "What is the dollar rate today?"
    # a non-live Hindi general question still drops it (nothing to search)
    plain = RouteRequest("भारत की राजधानी क्या है?", "hi", [], ConversationState(chat_id="c"), [], None, WEB)
    assert validate(RouterProposal(intent="general_qa", query="What is India's capital?"), plain).route.query_en is None


def test_a_hindi_live_question_without_an_english_query_searches_nothing():
    """The router failed: the Hindi utterance itself must not leave the machine."""
    req = RouteRequest("आज डॉलर का रेट क्या है?", "hi", [], ConversationState(chat_id="c"), [], None, WEB)
    d = with_live_tools(fallback_route(req, "router timed out"), req)
    assert d.route.tools == ["web_search"]
    dropped, query, note = live_search(d, req)
    assert (query, note, dropped.route.tools) == (None, "failed", [])
    assert "no English search query" in dropped.overrides[-1]


@pytest.mark.parametrize("case", [c for c in CASES["cases"] if "tools" in c], ids=lambda c: c["id"])
def test_router_eval_cases_tool_decisions(case):
    """The eval set's live / non-live cases (tests/integration/router_cases.json), decided without the model: the
    cue is in the utterance itself, whatever intent the router picks among the accepted ones."""
    req = RouteRequest(
        case["utterance"], "en", [], ConversationState(chat_id="c"), CASES["documents"], None, available_tools=WEB
    )
    for intent in (case["intent"], *case.get("also", [])):
        d = with_live_tools(decision_for(req, intent), req)
        assert d.route.tools == case["tools"], intent
