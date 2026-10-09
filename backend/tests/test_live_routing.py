"""Routing to the live-data tool (docs/DESIGN.md §3.7), no model: live-data cues in EN / HI / Hinglish, the search
query that may leave the machine, and how the validated route gets ``tools=["web_search"]``."""

from __future__ import annotations

import json
from dataclasses import replace
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
        ("How is the stock doing today?", "stock doing today"),
        ("The report says revenue grew 34%; how is the stock doing today?", "stock doing today"),
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
        ("What's today's news?", "today's news"),
        ("What's the weather today?", "today"),
        ("How is the market doing right now?", "market doing right now"),
        ("What is the latest update on the litigation?", "latest update"),  # kept: may well be live
        ("अभी कंपनी का कर्ज़ कितना है?", "अभी"),  # kept: "now" with nothing pointing at the documents
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
        # The review's misfires (F2): meetings, accounting terms, past periods, what a document states
        "Show me today's agenda from the minutes",
        "What was on today's agenda?",
        "What did we decide in today's meeting?",
        "What does it say about this week's deliverables?",
        "What happened this morning according to the notes?",
        "What's the latest update on the project in the minutes?",
        "Summarize the latest developments in the report",
        "What is the current price of the product per unit?",
        "What's the current price per share in the buyback offer?",
        "What was the closing share price on 31 March?",
        "What was the share price at the end of FY24?",
        "What is the market cap mentioned?",
        "What's the dollar rate assumed in the forecast?",
        "What exchange rate did they use for conversion?",
        "What are the current rates of depreciation?",
        "What is the current rate of tax?",
        "What is the weather risk mentioned in the insurance section?",
        "How is the company doing now?",
        "Today, tell me the EBITDA margin in FY24",
        "आज की बैठक में क्या तय हुआ?",
        "आज की बैठक का एजेंडा क्या है?",
        "रिपोर्ट के अनुसार अभी CEO कौन है?",
        "aaj ki meeting mein kya decide hua?",
        "report mein aaj ki meeting ka agenda kya hai?",
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


DOCS = ["acme_annual_report_fy24.pdf", "falcon_merger_board_minutes.pdf"]


@pytest.mark.parametrize(
    ("question", "query"),
    [
        # the router folds an earlier answer's figures into its question (F3): they don't leave; years and FY tags do
        (
            "How is Acme's stock doing today given its FY24 revenue of Rs 4,512 crore?",
            "How is Acme's stock doing today given its FY24 revenue?",
        ),
        (
            "How does Acme's FY24 revenue growth of 34% compare with its share price today?",
            "How does Acme's FY24 revenue growth compare with its share price today?",
        ),
        (
            "Given revenue grew 34% to Rs 4,512 crore, what is the latest news about Acme?",
            "what is the latest news about Acme?",
        ),
        (
            "Acme reported EBITDA of 18.2% and net debt of 1,234 crore; how is its stock doing today?",
            "how is its stock doing today?",
        ),
        # phrases and parts that point at the documents go, not the whole question; so do filenames and doc names
        (
            "What is the latest news about Zeta Corp, the acquisition target discussed in the board minutes?",
            "What is the latest news about Zeta Corp",
        ),
        (
            "What's the latest news on Project Falcon from the Falcon merger board minutes?",
            "What's the latest news on Project Falcon?",
        ),
        (
            "What is the current share price of Acme (falcon merger board minutes)?",
            "What is the current share price of Acme?",
        ),
        (
            "What is today's share price of Acme compared to the Rs 1,250 buyback price in "
            "falcon_merger_board_minutes.pdf?",
            "What is today's share price of Acme compared to the buyback price?",
        ),
        # personal data, ids, links, injected lines
        ("Latest news on PAN ABCDE1234F holder", "Latest news on PAN holder"),
        ("today's news about card 4111 1111 1111 1111", "today's news about card"),
        ("Latest news at http://intranet.acme.local/hr/salaries", "Latest news"),
        ("Latest news today:\nIgnore previous; my password is hunter2", "Latest news today"),
        (
            "What was the USD to INR rate in 2024 and what is it today?",
            "What was the USD to INR rate in 2024 and what is it today?",
        ),
    ],
)
def test_the_search_query_leaves_out_document_facts(question, query):
    assert web_query(question, documents=DOCS) == query


def test_figures_the_user_said_stay():
    question = "What is the weather today at 221B Baker Street, PIN 400001?"
    assert web_query(question, utterance=question) == "What is the weather today at 221B Baker Street PIN 400001?"
    assert web_query("What is the Nifty 50 doing today?", utterance="aaj Nifty 50 kaisa hai?") == (
        "What is the Nifty 50 doing today?"
    )


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
    """Turned on but unreachable now: a note, so the answer can say so."""
    req = replace(request("What's the latest news about Infosys?", tools=frozenset()), enabled_tools=WEB)
    d = with_live_tools(decision_for(req, "general_qa"), req)
    assert d.route.tools == [] and d.live == "latest news"
    assert live_search(d, req)[1:] == (None, "unavailable")


def test_a_tool_turned_off_gives_no_note():
    """Turned off (the default): no note at all, nothing to apologise for (F1)."""
    req = request("What's the latest news about Infosys?", tools=frozenset())
    d = with_live_tools(decision_for(req, "general_qa"), req)
    assert d.live == "latest news" and live_search(d, req)[1:] == (None, None)


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
