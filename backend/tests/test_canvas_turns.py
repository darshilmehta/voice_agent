"""The canvas in text turns (docs/DESIGN.md §12.1): an answer's visual on the SSE stream after the answer, never
before it; route.visual and the visual's id on the message; failures that leave the answer alone; no visual where
none belongs; spoken canvas edits (English, Hindi); questions about what is on screen; a newer turn that cancels a
visual still being planned, an acknowledgement that doesn't, an edit that waits for it."""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.domain.canvas import CanvasEvent, VisualEvent
from app.providers.llm import LLMError
from app.services.canvas.conversation import settled_callbacks
from app.services.canvas.draft import Draft
from app.services.chat_turns import AgentMessageEvent, ChatTurnService, DeltaEvent, wait_for_background
from app.services.prompts import VISUAL_NOTE

from .canvas_turn_helpers import (
    Script,
    add_visual,
    choose,
    choose_table,
    donut_of_segments,
    line_of_revenue,
    planner_calls,
    report_chat,
)
from .test_chat_api import names, parse_sse, payload

SHOW = "Show me revenue by quarter"  # a confident draft: the line of quarterly revenue
UNSURE = "Show me revenue"  # which table? a draft (segment revenue), for the planner to check


def run(api: TestClient, fn, *args, **kwargs):
    return api.portal.call(partial(fn, *args, **kwargs))  # type: ignore[union-attr]


def ask(api: TestClient, chat_id: str, text: str) -> list[tuple[str, dict[str, Any]]]:
    r = api.post(f"/api/chats/{chat_id}/messages", json={"text": text})
    assert r.status_code == 200, r.text
    events = parse_sse(r.text)
    run(api, settled_callbacks)  # the visual's id written onto the saved message
    return events


def messages(api: TestClient, chat_id: str) -> list[dict[str, Any]]:
    return api.get(f"/api/chats/{chat_id}/messages").json()["items"]


def canvas(api: TestClient, chat_id: str) -> list[dict[str, Any]]:
    return api.get(f"/api/chats/{chat_id}/canvas").json()["panels"]


def visuals(events) -> list[tuple[str, str | None]]:
    return [(d["phase"], d.get("detail")) for e, d in events if e == "visual"]


@pytest.fixture
def app_with_report(make_app, fakes):
    with make_app() as api:
        _, chat_id, ds = report_chat(api)
        yield api, chat_id, ds, fakes


# ------------------------------------------------------------------ the answer's visual on the SSE stream


def test_a_requested_visuals_draft_arrives_with_the_answers_first_delta(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose("line", "Revenue")).install(fakes.llm)
    fakes.llm.delay = 0.05  # the answer's first words take a moment (0.8 s on the real model): the draft is ready
    events = ask(api, chat_id, SHOW)
    order = names(events)
    # the draft (code, built when the retrieval returned) comes with the answer's first words, never before them
    assert order[:6] == ["user_message", "sources", "delta", "visual", "visual", "canvas"]
    assert order[-1] == "agent_message" and set(order[6:-1]) == {"delta"}
    assert visuals(events) == [("preparing", None), ("ready", None)]
    preparing, ready = (d for e, d in events if e == "visual")
    assert preparing["visual_id"] == ready["visual_id"] == ready["visual"]["id"]
    visual = ready["visual"]
    assert visual["kind"] == "line" and visual["chat_id"] == chat_id
    assert [r["x"] for r in visual["rows"]] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    # the chart's cells cite the answer's own [S#] ids
    turn_sources = {s["chunk_id"]: s["source_id"] for s in payload(events, "sources")["sources"]}
    for source in visual["sources"]:
        if source["chunk_id"] in turn_sources:
            assert source["source_id"] == turn_sources[source["chunk_id"]]
    assert payload(events, "canvas")["panels"][0]["id"] == visual["id"]
    # a confident draft is the visual: no model call for it at all
    assert planner_calls(fakes.llm) == []
    # persisted: what was asked for, the visual it got, and how it was made
    agent = messages(api, chat_id)[-1]
    assert agent["route"]["visual"] == "requested"
    assert agent["route"]["visual_status"] == "ready" and agent["route"]["visual_id"] == visual["id"]
    plan = agent["route"]["visual_plan"]
    assert plan["draft"] == "confident" and plan["planner"] == "skipped"
    assert [v["id"] for v in canvas(api, chat_id)] == [visual["id"]]
    # the answer prompt knew a chart was asked for: no "I can't show charts", no promise
    assert VISUAL_NOTE in fakes.llm.calls[-1]["messages"][-1].content


async def timed_turn(api, text: str, chat_id: str) -> tuple[list[tuple[float, Any]], float]:
    """A turn's events with when each came (s from the start), and when the planner was called."""
    service = ChatTurnService.from_container(api.app.state.container, canvas=api.app.state.canvas)
    t = await service.begin(chat_id, text)
    t0 = time.perf_counter()
    got = [(time.perf_counter() - t0, event) async for event in service.run(t)]
    await wait_for_background()
    return got, t0


def test_an_unsure_draft_is_refined_after_the_answer_and_replaced_in_place(app_with_report):
    api, chat_id, _, fakes = app_with_report
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(fakes.llm)
    script.planner_delay = 0.3
    fakes.llm.delay = 0.02  # every piece of the answer takes a moment
    got, t0 = run(api, timed_turn, api, UNSURE, chat_id)
    deltas = [t for t, e in got if isinstance(e, DeltaEvent)]
    readies = [(t, e) for t, e in got if isinstance(e, VisualEvent) and e.phase == "ready"]
    (agent_at,) = [t for t, e in got if isinstance(e, AgentMessageEvent)]
    planner_at = script.planner_at[0] - t0
    (draft_at, draft), (refined_at, refined) = readies
    assert deltas[0] <= draft_at < deltas[1]  # the draft with the first delta, not after the answer
    assert draft.visual is not None and draft.visual.kind == "donut"  # (the words don't tell: segments, a guess)
    assert planner_at >= deltas[-1]  # Ollama is serial: the planner never goes ahead of the answer's request
    assert agent_at < planner_at + 0.2  # the answer is saved and sent without waiting for the planner
    assert refined_at >= planner_at + 0.3
    # replaced in place: the same visual, now the planner's chart; one panel for the turn, never two
    assert refined.visual_id == draft.visual_id and refined.visual is not None and refined.visual.kind == "line"
    panels = canvas(api, chat_id)
    assert [(p["id"], p["kind"]) for p in panels] == [(draft.visual_id, "line")]
    run(api, settled_callbacks)
    route = messages(api, chat_id)[-1]["route"]
    assert route["visual_id"] == draft.visual_id
    assert route["visual_plan"]["draft"] == "refine" and route["visual_plan"]["planner"] == "changed"


def test_a_planner_that_agrees_with_the_draft_changes_nothing(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose_table("donut", "Specialty Chemicals", "Revenue FY24")).install(fakes.llm)
    events = ask(api, chat_id, UNSURE)
    assert len(planner_calls(fakes.llm)) == 1
    assert visuals(events) == [("preparing", None), ("ready", None)]  # nothing to replace
    assert messages(api, chat_id)[-1]["route"]["visual_plan"]["planner"] == "same"


def test_a_planner_finding_no_table_withdraws_an_unsure_draft(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner={"kind": "none", "datasets": ["D1"], "series": []}).install(fakes.llm)
    events = ask(api, chat_id, UNSURE)
    assert visuals(events) == [("preparing", None), ("ready", None), ("failed", "no table fits this question")]
    snapshots = [d["panels"] for e, d in events if e == "canvas"]
    assert len(snapshots[0]) == 1 and snapshots[-1] == []  # the draft, then the canvas without it
    assert payload(events, "agent_message")["text"] == "The answer [S1]."
    route = messages(api, chat_id)[-1]["route"]
    assert route["visual"] == "requested" and route["visual_status"] == "failed" and not route.get("visual_id")
    assert canvas(api, chat_id) == []


def test_a_planner_that_is_down_leaves_the_draft(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner=LLMError("model unavailable")).install(fakes.llm)
    events = ask(api, chat_id, UNSURE)
    assert visuals(events) == [("preparing", None), ("ready", None)]
    assert messages(api, chat_id)[-1]["route"]["visual_plan"]["planner"] == "failed"
    assert len(canvas(api, chat_id)) == 1


@pytest.fixture
def no_draft(monkeypatch):
    """Drafts that can't be built: the planner alone, as before the draft existed."""
    import app.services.canvas.service as service

    monkeypatch.setattr(service, "draft_visual", lambda *a, **kw: Draft(None, False, ["nothing builds"]))


def test_without_a_draft_the_planner_draws_it_after_the_answer(app_with_report, no_draft):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(fakes.llm)
    events = ask(api, chat_id, SHOW)
    order = names(events)
    assert order[:4] == ["user_message", "sources", "delta", "visual"]  # the skeleton with the first delta
    assert order[-3:] == ["agent_message", "visual", "canvas"]  # the planner's visual after the answer
    assert visuals(events) == [("preparing", None), ("ready", None)]
    plan = messages(api, chat_id)[-1]["route"]["visual_plan"]
    assert plan["draft"] == "none" and plan["planner"] == "planned"


def test_without_a_draft_a_planner_that_is_down_still_draws_a_requested_chart(app_with_report, no_draft):
    api, chat_id, _, fakes = app_with_report
    Script(planner=LLMError("model unavailable")).install(fakes.llm)
    events = ask(api, chat_id, SHOW)
    assert visuals(events) == [("preparing", None), ("ready", None)]  # the planner's heuristic fallback
    assert names(events)[-3:] == ["agent_message", "visual", "canvas"]


def test_without_a_draft_a_planner_finding_nothing_is_a_failed_event(app_with_report, no_draft):
    api, chat_id, _, fakes = app_with_report
    Script(planner={"kind": "none", "datasets": ["D1"], "series": []}).install(fakes.llm)
    events = ask(api, chat_id, SHOW)
    assert visuals(events) == [("preparing", None), ("failed", "no table fits this question")]
    assert "canvas" not in names(events) and canvas(api, chat_id) == []


@pytest.mark.parametrize(
    ("utterance", "route"),
    [
        ("What is EBITDA in general terms?", {"intent": "general_qa", "query": None}),  # general knowledge
        ("okay", None),  # an acknowledgement (fast path)
        ("stop", None),  # silent
        ("Show me the market share of our competitors", None),  # abstained: the documents don't cover it
    ],
)
def test_no_visual_for_general_ack_stop_or_abstained_turns(app_with_report, utterance, route):
    api, chat_id, _, fakes = app_with_report
    Script(router={utterance: route} if route else {}, planner=choose("line", "Revenue")).install(fakes.llm)
    events = ask(api, chat_id, utterance)
    assert "visual" not in names(events) and "canvas" not in names(events)
    assert planner_calls(fakes.llm) == []
    agent = messages(api, chat_id)[-1]
    assert "visual_id" not in (agent["route"] or {}) and "visual_status" not in (agent["route"] or {})
    assert canvas(api, chat_id) == []


def test_an_answer_that_says_the_documents_dont_cover_it_withdraws_its_draft(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose("line", "Revenue")).install(fakes.llm)
    fakes.llm.reply = "The documents don't cover quarterly revenue."  # the gate let it through; the answer abstains
    events = ask(api, chat_id, UNSURE)
    assert visuals(events) == [("preparing", None), ("ready", None), ("failed", "cancelled")]  # no note: just gone
    assert [d["panels"] for e, d in events if e == "canvas"][-1] == []
    assert planner_calls(fakes.llm) == []
    route = messages(api, chat_id)[-1]["route"]
    assert route["abstained"] is True and not route.get("visual_id") and route["visual_status"] == "cancelled"
    assert canvas(api, chat_id) == []


def test_an_answer_that_fails_takes_its_unseen_draft_with_it(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose("line", "Revenue")).install(fakes.llm)
    fakes.llm.fail_with = LLMError("model crashed")  # before any of the answer's words
    events = ask(api, chat_id, SHOW)
    assert "error" in names(events) and "visual" not in names(events)
    assert planner_calls(fakes.llm) == [] and canvas(api, chat_id) == []


def test_an_answers_own_figures_suggest_a_visual_that_appears_quietly(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose("line", "Revenue")).install(fakes.llm)
    fakes.llm.reply = "Revenue was 1,742 in Q1, 1,801 in Q2 and 1,933 in Q4 FY24 [S1]."
    events = ask(api, chat_id, "What was revenue in the quarters of FY24?")
    assert visuals(events) == [("ready", None)]  # suggested: no skeleton first, nothing if it fails
    assert messages(api, chat_id)[-1]["route"]["visual"] == "suggest"


# ------------------------------------------------------------------ canvas edits


def edit(api: TestClient, chat_id: str, text: str) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    events = ask(api, chat_id, text)
    return events, payload(events, "agent_message")


def test_make_that_a_bar_chart_changes_the_latest_visual_in_place(app_with_report):
    api, chat_id, ds, fakes = app_with_report
    line = line_of_revenue(api, chat_id, ds)
    donut = donut_of_segments(api, chat_id, ds)
    events, agent = edit(api, chat_id, "Make that a bar chart.")
    assert names(events) == ["user_message", "sources", "visual", "canvas", "delta", "agent_message"]
    (ready,) = (d for e, d in events if e == "visual")
    assert ready["phase"] == "ready" and ready["visual_id"] == donut["id"]  # "that": the one touched last
    assert ready["visual"]["kind"] == "bar"
    assert [r["x"] for r in ready["visual"]["rows"]] == [r["x"] for r in donut["rows"]]  # the same cells
    assert agent["text"] == "Done." and agent["route"]["intent"] == "canvas_edit"
    assert agent["route"]["canvas_edit"] == {
        "op": "kind", "kind": "bar", "outcome": "done", "source": "rules", "visual_id": donut["id"],
    }  # fmt: skip
    panels = canvas(api, chat_id)
    assert [(v["id"], v["kind"], v["position"]) for v in panels] == [(line["id"], "line", 0), (donut["id"], "bar", 1)]
    assert fakes.llm.calls == [] and fakes.llm.json_calls == []  # the fast path: no model at all


def test_remove_the_pie_pin_this_and_show_it_as_a_table(app_with_report):
    api, chat_id, ds, _ = app_with_report
    line = line_of_revenue(api, chat_id, ds)
    donut = donut_of_segments(api, chat_id, ds)
    events, agent = edit(api, chat_id, "pin this")
    assert agent["text"] == "Done." and [v["pinned"] for v in payload(events, "canvas")["panels"]] == [False, True]
    events, agent = edit(api, chat_id, "show the line chart as a table")
    assert agent["route"]["canvas_edit"]["visual_id"] == line["id"]  # a named kind picks the visual
    assert {v["id"]: v["kind"] for v in canvas(api, chat_id)} == {line["id"]: "table", donut["id"]: "donut"}
    events, agent = edit(api, chat_id, "remove the pie")
    assert [v["id"] for v in payload(events, "canvas")["panels"]] == [line["id"]]
    assert agent["route"]["canvas_edit"]["visual_id"] == donut["id"]


def test_hindi_edits(app_with_report):
    api, chat_id, ds, _ = app_with_report
    line = line_of_revenue(api, chat_id, ds)
    _, agent = edit(api, chat_id, "इसे टेबल में दिखाओ")
    assert agent["text"] == "हो गया।" and agent["route"]["canvas_edit"]["kind"] == "table"
    assert canvas(api, chat_id)[0]["kind"] == "table"
    _, agent = edit(api, chat_id, "isko bar chart mein dikhao")
    # typed Hinglish this short reads as English (services/language.py): the reply follows the turn's language
    assert agent["text"] == "Done." and canvas(api, chat_id)[0]["kind"] == "bar"
    _, agent = edit(api, chat_id, "हटा दो")
    assert agent["route"]["canvas_edit"] == {
        "op": "remove", "outcome": "done", "source": "rules", "visual_id": line["id"],
    }  # fmt: skip
    assert canvas(api, chat_id) == []


def test_put_fy23_next_to_it_adds_the_period_from_the_same_table(app_with_report):
    api, chat_id, ds, _ = app_with_report
    h = ds["p2"]["id"]
    kpi = add_visual(api, chat_id, kind="bar", datasets=[h], series=[f"{h}:revenue_from_operations"], periods=["FY24"])
    assert [r["x"] for r in kpi["rows"]] == ["FY24"]
    _, agent = edit(api, chat_id, "put FY23 next to it")
    assert agent["text"] == "Done."
    assert [r["x"] for r in canvas(api, chat_id)[0]["rows"]] == ["FY23", "FY24"]
    _, agent = edit(api, chat_id, "add FY24 too")
    assert agent["text"] == "That's already on the chart." and agent["route"]["canvas_edit"]["outcome"] == "already"


def test_an_edit_the_rules_cant_read_goes_to_the_planner_and_keeps_the_visual(app_with_report):
    api, chat_id, ds, fakes = app_with_report
    line = line_of_revenue(api, chat_id, ds)
    Script(planner=choose("line", "Revenue", "EBITDA")).install(fakes.llm)
    events, agent = edit(api, chat_id, "add EBITDA to that chart")
    assert visuals(events) == [("preparing", None), ("ready", None)]  # "Updating…" on the panel, then the new one
    ready = next(d for e, d in events if e == "visual" and d["phase"] == "ready")
    assert ready["visual_id"] == line["id"] and [s["label"] for s in ready["visual"]["series"]] == ["Revenue", "EBITDA"]
    assert agent["text"] == "Done." and agent["route"]["canvas_edit"]["source"] == "model"
    prompt = planner_calls(fakes.llm)[0]["messages"][-1].content
    assert 'Change the chart on screen (line "Revenue by quarter"' in prompt


def test_an_edit_with_nothing_on_screen_says_so(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(router={"make that a bar chart": {"intent": "canvas_edit", "query": None}}).install(fakes.llm)
    _, agent = edit(api, chat_id, "make that a bar chart")
    assert agent["text"] == "There's no chart on screen yet." and agent["route"]["answer"] == "canvas"
    assert agent["route"]["abstained"] is False  # never an abstention


# ------------------------------------------------------------------ visuals as conversation context


def test_a_question_about_the_chart_on_screen_is_answered_from_its_table(app_with_report):
    api, chat_id, ds, fakes = app_with_report
    line = line_of_revenue(api, chat_id, ds)
    question = "What's the second point on the chart?"
    Script(router={question: {"intent": "clarification", "query": "What was revenue in Q2 FY24?"}}).install(fakes.llm)
    fakes.llm.reply = "Revenue in Q2 FY24 was ₹1,801 crore [S1]."
    events = ask(api, chat_id, question)
    router_prompt = fakes.llm.json_calls[0]["messages"][-1].content
    assert "On screen (charts the app drew" in router_prompt and '1. line "Revenue by quarter"' in router_prompt
    assert 'The user points at the second item of "Revenue by quarter": Q2 FY24.' in router_prompt
    agent = payload(events, "agent_message")
    # "clarification" about the chart → a document question, answered from the chart's table, cited
    assert agent["route"]["intent"] == "document_qa" and agent["route"]["screen_visual_id"] == line["id"]
    assert "asks about a chart on screen" in " ".join(agent["route"]["router"]["overrides"])
    answer_prompt = fakes.llm.calls[-1]["messages"][-1].content
    assert "The user is asking about a chart on screen" in answer_prompt and "Q2 FY24" in answer_prompt
    first_source = payload(events, "sources")["sources"][0]
    assert "Q2 FY24 | 1,801" in first_source["snippet"] and agent["citations"][0]["source_id"] == "S1"


def test_chats_without_a_canvas_keep_the_routers_prompt(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script().install(fakes.llm)
    ask(api, chat_id, "and what about the dividend")
    assert "On screen" not in fakes.llm.json_calls[0]["messages"][-1].content


# ------------------------------------------------------------------ the next turn and a visual still being refined


async def turn_with_sink(api, chat_id: str, text: str) -> tuple[ChatTurnService, list[Any], list[Any]]:
    """A turn whose visual goes to a sink (as voice does): its stream ends with the answer."""
    service = ChatTurnService.from_container(api.app.state.container, canvas=api.app.state.canvas)
    sink: list[Any] = []

    async def collect(event: VisualEvent | CanvasEvent) -> None:
        sink.append(event)

    events = [e async for e in service.run(await service.begin(chat_id, text), on_visual=collect)]
    return service, events, sink


def phases(sink: list[Any]) -> list[tuple[str, str | None]]:
    return [(e.phase, e.detail) for e in sink if isinstance(e, VisualEvent)]


def test_a_new_question_cancels_the_refinement_and_the_draft_stays(app_with_report):
    api, chat_id, _, fakes = app_with_report
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(fakes.llm)
    script.planner_delay = 5.0

    async def scenario() -> tuple[list[Any], list[Any]]:
        _, first, sink = await turn_with_sink(api, chat_id, UNSURE)
        assert isinstance(first[-1], AgentMessageEvent)  # the answer is done, the refinement isn't
        await asyncio.sleep(0.05)
        _, second, _ = await turn_with_sink(api, chat_id, "Who is the chairperson of the board?")
        await wait_for_background()
        return sink, second

    sink, second = run(api, scenario)
    assert phases(sink) == [("preparing", None), ("ready", None)]  # no "failed": the draft is the turn's visual
    assert isinstance(second[-1], AgentMessageEvent)
    assert fakes.llm.json_cancelled >= 1  # the planner's request was closed: the model is free for the question
    assert [p["kind"] for p in canvas(api, chat_id)] == ["donut"]
    run(api, settled_callbacks)
    route = messages(api, chat_id)[1]["route"]
    assert route["visual_status"] == "ready" and route["visual_plan"]["planner"] == "cancelled"


def test_without_a_draft_a_new_question_cancels_the_visual(app_with_report, no_draft):
    api, chat_id, _, fakes = app_with_report
    script = Script(planner=choose("line", "Revenue")).install(fakes.llm)
    script.planner_delay = 5.0

    async def scenario() -> list[Any]:
        _, _, sink = await turn_with_sink(api, chat_id, SHOW)
        await asyncio.sleep(0.05)
        await turn_with_sink(api, chat_id, "Who is the chairperson of the board?")
        await wait_for_background()
        return sink

    sink = run(api, scenario)
    assert phases(sink) == [("preparing", None), ("failed", "cancelled")]
    assert canvas(api, chat_id) == []
    run(api, settled_callbacks)
    assert messages(api, chat_id)[1]["route"]["visual_status"] == "cancelled"


def test_an_acknowledgement_leaves_the_refinement_alone(app_with_report):
    api, chat_id, _, fakes = app_with_report
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(fakes.llm)
    script.planner_delay = 0.3

    async def scenario() -> list[Any]:
        _, _, sink = await turn_with_sink(api, chat_id, UNSURE)
        _, ack, _ = await turn_with_sink(api, chat_id, "okay")
        assert ack[-2].text == "Anything else?"  # type: ignore[attr-defined]
        await wait_for_background()
        return sink

    sink = run(api, scenario)
    assert [e.phase for e in sink if isinstance(e, VisualEvent)] == ["preparing", "ready", "ready"]
    assert [p["kind"] for p in canvas(api, chat_id)] == ["line"]


def test_an_edit_waits_for_the_refinement(app_with_report):
    api, chat_id, _, fakes = app_with_report
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(fakes.llm)
    script.planner_delay = 0.3

    async def scenario() -> tuple[list[Any], list[Any]]:
        _, _, sink = await turn_with_sink(api, chat_id, UNSURE)
        _, edit_events, _ = await turn_with_sink(api, chat_id, "make it a bar chart")  # said while it is refined
        await wait_for_background()
        return sink, edit_events

    sink, edit_events = run(api, scenario)
    refined = [e for e in sink if isinstance(e, VisualEvent) and e.phase == "ready"][-1]
    assert refined.visual is not None and refined.visual.kind == "line"
    edited = next(e for e in edit_events if isinstance(e, VisualEvent))
    assert edited.visual_id == refined.visual_id and edited.visual is not None and edited.visual.kind == "bar"
    assert [r.x for r in edited.visual.rows] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]  # the refined one, edited
    assert edit_events[-1].message.text == "Done."  # type: ignore[union-attr]


def test_the_next_prompts_warm_up_waits_for_the_refinement(app_with_report):
    api, chat_id, _, fakes = app_with_report
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(fakes.llm)
    script.planner_delay = 0.3

    async def scenario() -> tuple[int, int]:
        warmed_at_start = len(fakes.llm.warmed)
        await turn_with_sink(api, chat_id, UNSURE)
        await asyncio.sleep(0.1)
        during = len(fakes.llm.warmed) - warmed_at_start  # the draft is shown, the planner still refining it
        await wait_for_background()
        await asyncio.sleep(0.05)
        await wait_for_background()
        return during, len(fakes.llm.warmed) - warmed_at_start

    during, after = run(api, scenario)
    assert during == 0 and after == 1  # it would queue behind the planner on the one model


def test_a_confident_draft_lets_the_warm_up_go_at_once(app_with_report):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose("line", "Revenue")).install(fakes.llm)

    async def scenario() -> int:
        warmed_at_start = len(fakes.llm.warmed)
        await turn_with_sink(api, chat_id, SHOW)
        await wait_for_background()
        await asyncio.sleep(0.05)
        await wait_for_background()
        return len(fakes.llm.warmed) - warmed_at_start

    assert run(api, scenario) == 1
    assert planner_calls(fakes.llm) == []
