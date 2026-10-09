"""The router with a canvas (docs/DESIGN.md §3.4, §12.1): the canvas-edit fast path (only with something on screen),
the screen in the router's input (only then), questions about a chart, requests to see the documents' figures that
the model takes for small talk, and route.visual."""

from __future__ import annotations

from app.services.planning import policy
from app.services.router import (
    SCREEN_HINT,
    RouteRequest,
    RouterProposal,
    fast_route,
    router_messages,
    standalone_question,
    validate,
)

SCREEN = ['1. line "Revenue by quarter" — x: Q1 FY24, Q2 FY24, Q3 FY24, Q4 FY24 — series: Revenue (₹ crore) (latest)']
DOCS = ["valmora_annual_report_fy24.pdf"]


def req(text: str, *, screen: list[str] | None = None, language: str = "en") -> RouteRequest:
    return RouteRequest(text, language, documents=DOCS, screen=screen or [])  # type: ignore[arg-type]


def test_canvas_edits_take_the_fast_path_only_with_something_on_screen():
    for text in ("make that a bar chart", "हटा दो", "pin this", "isko table mein dikhao", "put FY23 next to it"):
        decision = fast_route(req(text, screen=SCREEN))
        assert decision is not None and decision.route.intent == "canvas_edit", text
        assert decision.route.needs_retrieval is False and decision.route.topic == ""
    assert fast_route(req("make that a bar chart")) is None  # nothing on screen: the router decides
    assert fast_route(req("What's the second bar?", screen=SCREEN)) is None


def test_the_screen_is_in_the_routers_input_only_when_there_is_one():
    with_screen = router_messages(req("why did it dip there?", screen=SCREEN))[-1].content
    assert "On screen (charts the app drew from the documents' tables):" in with_screen
    assert SCREEN[0] in with_screen and SCREEN_HINT in with_screen
    assert with_screen.index(SCREEN_HINT) < with_screen.index("Utterance: why did it dip there?")
    without = router_messages(req("why did it dip there?"))
    assert "On screen" not in without[-1].content and SCREEN_HINT not in without[-1].content
    assert router_messages(req("x", screen=SCREEN))[0] == without[0]  # the cached system prompt is the same


def test_a_question_about_the_chart_on_screen_is_a_document_question():
    proposal = RouterProposal(intent="clarification", query="What was revenue in Q2 FY24?")
    decision = validate(proposal, req("What's the second bar?", screen=SCREEN))
    assert decision.route.intent == "document_qa" and decision.route.rewritten_query == "What was revenue in Q2 FY24?"
    assert "clarification→document_qa: asks about a chart on screen" in decision.overrides
    # without a canvas the clarification stands
    assert validate(proposal, req("What's the second bar?")).route.intent == "clarification"
    # and it is never answered from a speculative retrieval of its raw words: the router reads the screen
    assert standalone_question(req("What is the second bar in the revenue chart?"))
    assert not standalone_question(req("What is the second bar in the revenue chart?", screen=SCREEN))


def test_an_edit_is_never_a_question():
    decision = validate(RouterProposal(intent="canvas_edit", query=None), req("can you show the chart as a table?"))
    assert decision.route.intent == "document_qa"


def test_a_hindi_edit_keeps_the_routers_english_for_the_planner():
    proposal = RouterProposal(intent="canvas_edit", query="Add EBITDA to the chart")
    decision = validate(proposal, req("isme EBITDA bhi jodo", screen=SCREEN, language="hi"))
    assert decision.route.intent == "canvas_edit" and decision.route.query_en == "Add EBITDA to the chart"


def test_asking_to_see_the_documents_figures_is_never_small_talk():
    text = "वालमोरा की FY24 की तिमाही आय का चार्ट दिखाओ"
    proposal = RouterProposal(intent="conversation", query="Show Valmora's quarterly revenue chart for FY24")
    decision = validate(proposal, req(text, language="hi"))
    assert decision.route.intent == "document_qa"
    assert "conversation→document_qa: asks to see the documents' figures" in decision.overrides
    # a how-to that says "show" is still general knowledge
    general = validate(RouterProposal(intent="general_qa", query=None), req("show me how to make masala chai"))
    assert general.route.intent == "general_qa"


def test_route_visual_is_set_by_the_retrieval_policy():
    def visual_of(text: str, intent: str = "document_qa", screen: list[str] | None = None) -> str:
        decision = validate(RouterProposal(intent=intent, query=None), req(text, screen=screen))  # type: ignore[arg-type]
        r = req(text, screen=screen)
        return policy(decision, r, has_documents=True, retrieval_enabled=True).decision.route.visual

    assert visual_of("Show me revenue by quarter") == "requested"
    assert visual_of("How did revenue move over the years?") == "suggest"
    assert visual_of("What was revenue in FY24?") == "none"
    assert visual_of("What was EBITDA in Q3 FY24?") == "none"  # one period, not two
    assert visual_of("Show me revenue by quarter", "general_qa") == "none"  # not from the documents
    assert visual_of("What does the second bar of this chart show?", screen=SCREEN) == "none"  # answered, not drawn
