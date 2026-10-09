"""The canvas in the conversation, pure parts (services/canvas/conversation.py): route.visual, the canvas edit grammar
(English, Hindi, Hinglish), which visual "that" means, what a question points at, the canvas as router context, and
``TurnVisual`` (events, cancellation, the bound on waiting)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from app.domain.canvas import CanvasEvent, Visual, VisualEvent
from app.services.canvas.builder import build_visual
from app.services.canvas.conversation import (
    CanvasEdit,
    TurnVisual,
    describe,
    parse_edit,
    point_reference,
    refers_to_screen,
    resolve_target,
    screen_lines,
    visual_in_progress,
    visual_want,
)
from app.services.canvas.spec import VisualSpec, resolve

from .canvas_helpers import report_datasets

DS = report_datasets()
BY_ID = {d.id: d for d in DS.values()}
T0 = datetime(2026, 10, 9, 10, 0, tzinfo=UTC)


def visual(vid: str, kind: str, dataset: str, series: list[str], *, minutes: int = 0, position: int = 0, **spec):
    ds = DS[dataset]
    s = VisualSpec(kind=kind, datasets=[ds.id], series=[f"{ds.id}:{k}" for k in series], **spec)  # type: ignore[arg-type]
    v = build_visual(
        resolve(s, BY_ID), visual_id=vid, project_id="p", chat_id="c", filenames={}, now=T0 + timedelta(minutes=minutes)
    )
    return v.model_copy(update={"position": position, "updated_at": T0 + timedelta(minutes=minutes)})


LINE = visual("vis_line", "line", "q_fy24", ["revenue"], minutes=0, position=0, title="Revenue by quarter")
DONUT = visual("vis_donut", "donut", "segments", ["revenue_fy24"], minutes=1, position=1, title="Segment revenue")
KPI = visual("vis_kpi", "kpi", "highlights", ["revenue_from_operations", "ebitda"], minutes=2, position=2)
PANELS = [LINE, DONUT, KPI]


# ------------------------------------------------------------------ route.visual


@pytest.mark.parametrize(
    ("texts", "want"),
    [
        (["Show me revenue by quarter"], "requested"),
        (["Revenue ka chart dikhao"], "requested"),
        (["तिमाही राजस्व का ग्राफ़ दिखाओ"], "requested"),
        (["How did revenue move over the years?"], "suggest"),
        (["Compare FY23 and FY24 EBITDA"], "suggest"),
        (["What is the registered office address?"], "none"),
        (["What was revenue in FY24?", None], "none"),
        (["isko compare karo", "Compare revenue in FY23 and FY24"], "suggest"),  # the standalone form counts too
    ],
)
def test_route_visual_from_the_users_words(texts, want):
    assert visual_want(*texts) == want


def test_an_answer_with_three_figures_suggests_a_visual():
    assert visual_want("What was revenue?", answer="₹7,365 crore in FY24, up from ₹6,482 crore, 13.6% more.") == (
        "suggest"
    )
    assert visual_want("What was revenue?", answer="₹7,365 crore [S1].") == "none"


# ------------------------------------------------------------------ the edit grammar


@pytest.mark.parametrize(
    ("utterance", "edit"),
    [
        # kind changes
        ("Make that a bar chart.", CanvasEdit("kind", kind="bar")),
        ("show it as a table", CanvasEdit("kind", kind="table")),
        ("bar chart instead", CanvasEdit("kind", kind="bar")),
        ("make the pie a bar chart", CanvasEdit("kind", kind="bar", target_kind="donut")),
        ("turn it into a line chart please", CanvasEdit("kind", kind="line")),
        ("isko bar chart mein dikhao", CanvasEdit("kind", kind="bar")),
        ("pie chart ko table mein badlo", CanvasEdit("kind", kind="table", target_kind="donut")),
        ("इसे बार चार्ट में दिखाओ", CanvasEdit("kind", kind="bar")),
        ("इसे टेबल में दिखाओ", CanvasEdit("kind", kind="table")),
        # remove, pin, unpin, clear
        ("remove the pie", CanvasEdit("remove", target_kind="donut")),
        ("delete that chart", CanvasEdit("remove")),
        ("हटा दो", CanvasEdit("remove")),
        ("isko hata do", CanvasEdit("remove")),
        ("pin this", CanvasEdit("pin")),
        ("pin kar do", CanvasEdit("pin")),
        ("इसे पिन करो", CanvasEdit("pin")),
        ("unpin it", CanvasEdit("unpin")),
        ("pin hata do", CanvasEdit("unpin")),
        ("clear the canvas", CanvasEdit("clear")),
        ("सब हटा दो", CanvasEdit("clear")),
        # periods
        ("put FY23 next to it", CanvasEdit("periods", periods=("FY23",))),
        ("FY23 bhi dikhao", CanvasEdit("periods", periods=("FY23",))),
        ("FY23 भी जोड़ो", CanvasEdit("periods", periods=("FY23",))),
        ("only FY24", CanvasEdit("only", periods=("FY24",))),
        # edits the grammar can't read: the planner, with the visual as context
        ("add EBITDA to that chart", CanvasEdit("model")),
        ("show profit on it instead", CanvasEdit("model")),
    ],
)
def test_the_edit_grammar(utterance, edit):
    assert parse_edit(utterance) == edit


@pytest.mark.parametrize(
    "utterance",
    [
        "Show me revenue as a bar chart",  # names a metric: a new visual
        "show the table",  # may be a document's table
        "What does the chart show?",  # a question
        "what caused that drop",  # a question without its question mark
        "Why did it dip there",
        "okay",
        "What was revenue in FY24",
        "How has revenue grown since FY20 compared with the plan for the next five years and the market?",
    ],
)
def test_questions_and_new_visuals_are_not_edits(utterance):
    assert parse_edit(utterance) is None


# ------------------------------------------------------------------ which visual, which point


def test_that_means_the_visual_touched_last():
    assert resolve_target("make that a bar chart", PANELS) is KPI
    assert resolve_target("हटा दो", PANELS) is KPI
    pinned_line = LINE.model_copy(update={"updated_at": T0 + timedelta(minutes=5)})  # pinned (edited) just now
    assert resolve_target("make it a table", [pinned_line, DONUT, KPI]) is pinned_line
    assert resolve_target("remove anything", []) is None


def test_a_named_kind_an_ordinal_or_title_words_pick_the_visual():
    assert resolve_target("remove the pie", PANELS, kind_hint="donut") is DONUT
    assert resolve_target("remove the line chart", PANELS, kind_hint="line") is LINE
    assert resolve_target("delete the first chart", PANELS) is LINE
    assert resolve_target("दूसरा चार्ट हटा दो", PANELS) is DONUT
    assert resolve_target("remove the last chart", PANELS) is KPI
    assert resolve_target("pin the segment revenue chart", PANELS) is DONUT


def test_questions_about_the_screen():
    for text in (
        "What's the second bar?",
        "why did it dip there?",
        "what does this chart show",
        "is graph mein sabse zyada kya hai",
        "दूसरा बार क्या है?",
        "What is on screen?",
    ):
        assert refers_to_screen(text), text
    for text in (
        "What was the highest revenue?",  # a document question
        "What do the product lines include?",
        "What was revenue in the last quarter?",
        "What does the table on page 5 say?",
    ):
        assert not refers_to_screen(text), text


def test_what_a_question_points_at_is_found_by_code():
    assert point_reference("what's the second point?", LINE) == 'the second item of "Revenue by quarter": Q2 FY24'
    assert point_reference("and the last bar?", LINE) == 'the last item of "Revenue by quarter": Q4 FY24'
    assert point_reference("दूसरा बार क्या है", DONUT) == 'the second item of "Segment revenue": Engineered Plastics'
    assert point_reference("where is the peak?", LINE) == 'the highest point of "Revenue by quarter": Q4 FY24'
    falling = visual("vis_fall", "line", "q_fy24", ["revenue"], title="Revenue")
    rows = [r.model_copy(update={"values": {"revenue": v}}) for r, v in zip(falling.rows, [10, 7, 9, 12], strict=True)]
    falling = falling.model_copy(update={"rows": rows})
    assert (
        point_reference("why did it dip there?", falling)
        == 'the dip in "Revenue": Q2 FY24 (Revenue fell from the point before)'
    )
    assert point_reference("what was the margin?", LINE) is None


def test_the_canvas_as_router_context_has_labels_not_numbers():
    lines = screen_lines(PANELS, "what's the second bar of the pie?")
    assert lines[0].startswith('1. line "Revenue by quarter" — x: Q1 FY24, Q2 FY24, Q3 FY24, Q4 FY24')
    assert "series: Revenue (₹ crore)" in lines[0]
    assert lines[2].startswith('3. kpi "') and lines[2].endswith("(latest)")
    assert lines[-1] == 'The user points at the second item of "Segment revenue": Engineered Plastics.'
    assert not any(
        ch.isdigit()
        for ch in " ".join(lines)
        .replace("FY24", "")
        .replace("Q1", "")
        .replace("Q2", "")
        .replace("Q3", "")
        .replace("Q4", "")
        .replace("1.", "")
        .replace("2.", "")
        .replace("3.", "")
    )
    assert screen_lines([], "anything") == []
    assert screen_lines(PANELS, "What was revenue in FY24?")[-1].endswith("(latest)")  # no pointing line
    many = [
        visual(f"vis_{n}", "line", "q_fy24", ["revenue"], minutes=n, position=n, title=f"Chart {'ABCDE'[n]}")
        for n in range(5)
    ]
    assert [line.split('"')[1] for line in screen_lines(many)] == ["Chart C", "Chart D", "Chart E"]  # the latest 3


def test_describe_a_visual():
    assert describe(DONUT).startswith('donut "Segment revenue" — x: Specialty Chemicals, Engineered Plastics, Digital')


# ------------------------------------------------------------------ TurnVisual


async def events_of(*items: VisualEvent | CanvasEvent, hang: float = 0.0) -> AsyncIterator[VisualEvent | CanvasEvent]:
    for item in items:
        yield item
    await asyncio.sleep(hang)


def ready(vid: str) -> VisualEvent:
    return VisualEvent(phase="ready", visual_id=vid, visual=LINE.model_copy(update={"id": vid}))


async def test_a_requested_visual_announces_itself_and_ends_ready():
    v = TurnVisual("c1", "vis_a", "requested", announce=True)
    v.start(events_of(VisualEvent(phase="preparing", visual_id="vis_a"), ready("vis_a"), CanvasEvent(panels=[])))
    assert visual_in_progress("c1") is v
    got = [e async for e in v.events()]
    assert [(e.name, getattr(e, "phase", None)) for e in got] == [
        ("visual", "preparing"),  # once: announced at the start, the generator's own is dropped
        ("visual", "ready"),
        ("canvas", None),
    ]
    assert v.status == "ready" and v.record() == {"visual_status": "ready", "visual_id": "vis_a"}
    await v.wait()
    assert visual_in_progress("c1") is None


async def test_a_suggested_visual_that_fails_shows_nothing():
    v = TurnVisual("c2", "vis_b", "suggest", announce=False)
    v.start(
        events_of(VisualEvent(phase="preparing", visual_id="vis_b"), VisualEvent(phase="failed", visual_id="vis_b"))
    )
    assert [e async for e in v.events()] == []
    assert v.status == "failed" and v.record() == {}


async def test_cancelling_a_requested_visual_clears_its_skeleton():
    v = TurnVisual("c3", "vis_c", "requested", announce=True)
    v.start(events_of(hang=60))
    assert [e.phase for e in v.pending()] == ["preparing"]  # type: ignore[union-attr]
    v.cancel()
    rest = [e async for e in v.events()]
    assert rest == [VisualEvent(phase="failed", visual_id="vis_c", detail="cancelled")]
    assert v.status == "cancelled" and v.record() == {"visual_status": "cancelled", "visual_detail": "cancelled"}


async def test_waiting_for_a_visual_is_bounded():
    v = TurnVisual("c4", "vis_d", "requested", announce=True)
    v.start(events_of(hang=60))
    got = [e async for e in v.events(timeout=0.05)]
    assert [(e.phase, e.detail) for e in got] == [("preparing", None), ("failed", "timed out")]  # type: ignore[union-attr]


async def test_settled_callbacks_run_once_even_when_registered_late():
    v = TurnVisual("c5", "vis_e", "requested", announce=True)
    seen: list[str] = []

    async def note(tv: TurnVisual) -> None:
        seen.append(tv.status)

    v.when_settled(note)
    v.start(events_of(ready("vis_e")))
    await v.wait()
    v.when_settled(note)  # after it settled: at once
    for _ in range(5):
        await asyncio.sleep(0)
    assert seen == ["ready", "ready"]


async def test_preparing_that_yields_nothing_ends_as_none():
    v = TurnVisual("c6", "vis_f", "requested", announce=True)
    v.start(events_of())  # prepare_visual decided no visual was worth it
    got = [e async for e in v.events()]
    assert [(e.phase, e.detail) for e in got] == [("preparing", None), ("failed", "no visual")]  # type: ignore[union-attr]
    assert v.status == "none"


def test_visual_fixture_sanity():
    assert isinstance(LINE, Visual) and [r.x for r in LINE.rows] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
