"""Edits that rebuild a visual keep what was on screen and its language (docs/DESIGN.md §12.1, "Canvas edits"): a kind
change keeps every table, series and period of the chart (found in the final real-model run: "isko table mein dikhao"
on an 8-quarter FY23 + FY24 chart came back as a table of FY23), what the planner chooses for an edit the rules can't
read is merged with what was on screen, and a Hindi edit said in an English chat doesn't turn the visual's labels and
units into Hindi."""

from __future__ import annotations

from typing import Any

import pytest

from app.db import models as orm
from app.services.canvas.builder import build_visual
from app.services.canvas.edits import change_kind, edit_language, keeps_data, merge_planned
from app.services.canvas.planner import builds
from app.services.canvas.spec import VisualSpec, resolve
from app.services.canvas.store import CanvasStore

from .canvas_helpers import report_datasets
from .canvas_turn_helpers import Script, add_visual, choose, line_of_revenue, planner_calls, report_chat
from .test_canvas_api import chat, db, drain, project, run, upload
from .test_canvas_turns import TWO_TABLES, canvas, edit


@pytest.fixture
def app_with_report(make_app, fakes):
    with make_app() as api:
        _, chat_id, ds = report_chat(api)
        yield api, chat_id, ds, fakes


FY23, FY24 = "ds_q_fy23", "ds_q_fy24"
BOTH_YEARS = [f"{FY23}:revenue", f"{FY24}:revenue", f"{FY23}:ebitda", f"{FY24}:ebitda"]


def two_year_chart(kind: str = "line", **extra: Any) -> tuple[VisualSpec, dict]:
    """The eight quarters of FY23 and FY24, revenue and EBITDA: a chart of two tables' four columns."""
    pool = {d.id: d for d in report_datasets().values()}
    return VisualSpec(kind=kind, datasets=[FY23, FY24], series=BOTH_YEARS, **extra), pool  # type: ignore[arg-type]


def built(spec: VisualSpec, pool: dict):
    from datetime import UTC, datetime

    return build_visual(
        resolve(spec, pool), visual_id="vis_t", project_id="p", chat_id="c", filenames={}, now=datetime.now(UTC)
    )


# ------------------------------------------------------------------ a kind change keeps all of the data


def test_a_chart_of_two_tables_as_a_table_has_both_tables_all_series_and_every_quarter():
    spec, pool = two_year_chart()
    table = change_kind(spec, "table", pool, builds(pool))
    assert table is not None and table.kind == "table"
    assert table.datasets == [FY23, FY24] and table.series == BOTH_YEARS
    visual = built(table, pool)
    assert [s.label for s in visual.series] == ["Revenue", "EBITDA"]
    quarters = [r.x for r in visual.rows if r.x.startswith("Q")]
    assert quarters == [f"Q{q} FY{y}" for y in (23, 24) for q in (1, 2, 3, 4)]
    assert len(visual.rows) == 10  # a table also has the "Full year" rows
    # the table is named by what it shows, not by the first table's own title ("Q Fy23")
    assert visual.title == "Revenue · EBITDA, Q1 FY23–Q4 FY24"  # noqa: RUF001
    assert "FY23" in visual.summary or "Fy23" in visual.summary
    assert "Fy24" in visual.summary or "FY24" in visual.summary


@pytest.mark.parametrize("kind", ["bar", "grouped_bar", "line", "table"])
def test_every_kind_a_chart_of_two_tables_can_take_keeps_its_data(kind):
    spec, pool = two_year_chart("grouped_bar")
    changed = change_kind(spec, kind, pool, builds(pool))
    assert changed is not None and changed.kind == kind and keeps_data(spec, changed, pool)
    assert set(changed.series) == set(BOTH_YEARS)


def test_a_kind_that_would_give_up_part_of_the_chart_is_not_drawn_at_all():
    """A donut shows one series at one moment: of this chart that is the first series alone (what ``first_valid``
    would settle for) or a bar chart (its kind fallback): neither is what was asked, and both lose data."""
    spec, pool = two_year_chart()
    assert change_kind(spec, "donut", pool, builds(pool)) is None
    assert change_kind(spec, "waterfall", pool, builds(pool)) is None


def test_a_table_keeps_the_periods_it_was_filtered_to():
    spec, pool = two_year_chart(periods=["Q1 FY24", "Q2 FY24"])
    visual = built(change_kind(spec, "table", pool, builds(pool)), pool)  # type: ignore[arg-type]
    assert [r.x for r in visual.rows] == ["Q1 FY24", "Q2 FY24"]  # the filter used to be ignored by tables with totals


def test_a_table_filtered_to_a_fiscal_years_quarters_leaves_out_the_full_year_row():
    spec, pool = two_year_chart("table", periods=["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"])
    assert [r.x for r in built(spec, pool).rows] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    annual = two_year_chart("table", periods=["FY23", "FY24"])[0]
    assert [r.x for r in built(annual, pool).rows] == ["Full year FY23", "Full year FY24"]


def test_keeps_data_compares_tables_series_and_x_items_not_the_total_rows_of_a_table():
    spec, pool = two_year_chart()
    table = spec.model_copy(update={"kind": "table"})
    assert keeps_data(spec, table, pool)  # a table has more rows: the totals
    assert keeps_data(table, spec, pool)  # and a chart of them leaves the totals out without losing the quarters
    assert not keeps_data(spec, spec.model_copy(update={"datasets": [FY23], "series": BOTH_YEARS[::2]}), pool)
    assert not keeps_data(spec, spec.model_copy(update={"series": BOTH_YEARS[:3]}), pool)
    assert not keeps_data(spec, spec.model_copy(update={"periods": ["Q1 FY23", "Q2 FY23"]}), pool)


# ------------------------------------------------------------------ what the planner chooses is merged with the screen


def test_a_planner_that_drops_a_table_for_a_kind_change_gets_it_back():
    spec, pool = two_year_chart()
    planned = VisualSpec(kind="table", datasets=[FY23], series=[f"{FY23}:revenue"])  # FY23's revenue: a table of FY23
    merged = merge_planned(spec, planned, "put the figures of that chart into a table", pool, builds(pool))
    assert merged is not None and merged.kind == "table"
    assert merged.datasets == [FY23, FY24] and set(merged.series) == set(BOTH_YEARS)


def test_an_addition_keeps_what_was_there_and_a_drop_the_user_asked_for_is_respected():
    pool = {d.id: d for d in report_datasets().values()}
    old = VisualSpec(kind="line", datasets=[FY24], series=[f"{FY24}:revenue"])
    planned = VisualSpec(kind="line", datasets=[FY24], series=[f"{FY24}:ebitda"])  # only the new series
    added = merge_planned(old, planned, "add EBITDA to that chart", pool, builds(pool))
    assert added is not None and added.series == [f"{FY24}:revenue", f"{FY24}:ebitda"]
    swapped = merge_planned(old, planned, "show EBITDA instead on that chart", pool, builds(pool))
    assert swapped == planned  # "instead": revenue is meant to go


def test_a_selection_of_periods_is_kept_and_extended():
    pool = {d.id: d for d in report_datasets().values()}
    old = VisualSpec(
        kind="line", datasets=[FY24], series=[f"{FY24}:revenue"], periods=["Q1 FY24", "Q2 FY24", "Q3 FY24"]
    )
    planned = VisualSpec(
        kind="line", datasets=[FY24], series=[f"{FY24}:revenue", f"{FY24}:ebitda"], periods=["Q3 FY24", "Q4 FY24"]
    )
    merged = merge_planned(old, planned, "add EBITDA and Q4 to that chart", pool, builds(pool))
    assert merged is not None and merged.periods == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]


# ------------------------------------------------------------------ in the conversation


def two_years_chat(api, fakes):
    """A chat whose canvas holds the line chart of FY23's and FY24's quarterly tables (revenue and EBITDA)."""
    p = project(api)
    upload(api, p, "two_tables.txt", TWO_TABLES)
    drain(api)
    chat_id = chat(api, p)
    a, b = (d["id"] for d in api.get(f"/api/projects/{p}/datasets").json()["items"])  # FY24's table, FY23's
    line = add_visual(
        api,
        chat_id,
        kind="line",
        datasets=[b, a],
        series=[f"{b}:revenue", f"{a}:revenue", f"{b}:ebitda", f"{a}:ebitda"],
    )
    assert len(line["rows"]) == 8 and [s["label"] for s in line["series"]] == ["Revenue", "EBITDA"]
    return chat_id, line


def test_isko_table_mein_dikhao_on_the_two_years_chart_keeps_both_years(make_app, fakes):
    with make_app() as api:
        chat_id, line = two_years_chat(api, fakes)
        _, agent = edit(api, chat_id, "isko table mein dikhao")
        assert agent["route"]["canvas_edit"]["outcome"] == "done" and agent["route"]["canvas_edit"]["source"] == "rules"
        (table,) = canvas(api, chat_id)
        assert table["id"] == line["id"] and table["kind"] == "table"
        assert [s["label"] for s in table["series"]] == ["Revenue", "EBITDA"]
        quarters = [r["x"] for r in table["rows"] if r["x"].startswith("Q")]
        assert quarters == [f"Q{q} FY{y}" for y in (23, 24) for q in (1, 2, 3, 4)]
        assert "FY23" in table["title"] and "FY24" in table["title"]
        assert planner_calls(fakes.llm) == []


def test_a_donut_of_the_two_years_chart_is_refused_not_degraded(make_app, fakes):
    with make_app() as api:
        chat_id, _line = two_years_chat(api, fakes)
        _, agent = edit(api, chat_id, "make that a pie chart")
        assert agent["text"] == "I can't show that chart that way."
        assert agent["route"]["canvas_edit"]["outcome"] == "cannot"
        (still,) = canvas(api, chat_id)
        assert still["kind"] == "line" and len(still["rows"]) == 8  # untouched


def test_a_kind_change_the_planner_has_to_read_keeps_both_tables(make_app, fakes):
    """The router calls it a canvas edit, the grammar can't read it, and a small model picks one table."""
    with make_app() as api:
        chat_id, _line = two_years_chat(api, fakes)
        said = "put the figures of that chart into a table"
        Script(planner=choose("table", "Revenue")).install(fakes.llm)  # the first offered table, revenue only
        _, agent = edit(api, chat_id, said)
        assert agent["route"]["canvas_edit"]["source"] == "model" and agent["route"]["canvas_edit"]["outcome"] == "done"
        (table,) = canvas(api, chat_id)
        assert table["kind"] == "table" and [s["label"] for s in table["series"]] == ["Revenue", "EBITDA"]
        assert len([r for r in table["rows"] if r["x"].startswith("Q")]) == 8


# ------------------------------------------------------------------ the language of a rebuilt visual


def test_a_visual_follows_the_chat_not_the_edit_utterance():
    assert edit_language("इसे टेबल में दिखाओ", "en", "en") == "en"  # a Hindi edit in an English chat
    assert edit_language("make that a table", "hi", "en") == "hi"  # an English edit in a Hindi chat
    assert edit_language("make that a table", None, "hi") == "hi"  # nothing known of the chat: the visual's own
    assert edit_language("हिंदी में बताओ", "en", "en") == "hi"  # the user asked to switch
    assert edit_language("answer in English", "hi", "hi") == "en"


def test_a_hindi_edit_in_an_english_chat_keeps_the_visuals_labels_and_units_english(app_with_report):
    api, chat_id, ds, fakes = app_with_report
    line = line_of_revenue(api, chat_id, ds)
    assert line["language"] == "en" and line["unit"]["label"] == "₹ crore"
    # the router calls it an edit; the grammar can't read it; the planner redraws it (the model path takes the turn's
    # language: Hindi, from the utterance)
    said = "इसमें EBITDA भी दिखाओ"
    Script(
        router={said: {"intent": "canvas_edit", "query": None}}, planner=choose("line", "Revenue", "EBITDA")
    ).install(fakes.llm)
    _, agent = edit(api, chat_id, said)
    assert agent["text"] == "हो गया।" and agent["route"]["canvas_edit"]["source"] == "model"  # the reply is Hindi
    (rebuilt,) = canvas(api, chat_id)
    assert [s["label"] for s in rebuilt["series"]] == ["Revenue", "EBITDA"]
    assert rebuilt["language"] == "en" and rebuilt["unit"]["label"] == "₹ crore"
    assert "करोड़" not in str(rebuilt["subtitle"]) and "तक" not in rebuilt["summary"]
    prompt = planner_calls(fakes.llm)[0]["messages"][-1].content
    assert "Language for the title: English" in prompt


def test_asking_for_another_language_in_the_edit_switches_the_visual(app_with_report):
    api, chat_id, ds, fakes = app_with_report
    line_of_revenue(api, chat_id, ds)
    said = "add EBITDA to that chart, answer in Hindi"
    Script(
        router={said: {"intent": "canvas_edit", "query": None}}, planner=choose("line", "Revenue", "EBITDA")
    ).install(fakes.llm)
    _, agent = edit(api, chat_id, said)
    assert agent["route"]["canvas_edit"]["outcome"] == "done"
    (rebuilt,) = canvas(api, chat_id)
    assert rebuilt["language"] == "hi" and rebuilt["unit"]["label"] == "₹ crore"  # (the unit's Hindi label is the UI's)


def test_the_chats_response_language_is_the_preference_then_the_last_answer_then_the_chats_own(app):
    p = project(app)
    chat_id = chat(app, p, language="hi")
    store = CanvasStore(db(app))

    async def message(seq: int, language: str, route: dict | None) -> None:
        async with db(app).session() as s:
            s.add(orm.Message(chat_id=chat_id, seq=seq, role="agent", text="x", language=language, route=route))

    assert run(app, store.response_language, chat_id) == "hi"  # the chat's own
    run(app, message, 1, "en", {"answer": "grounded"})
    assert run(app, store.response_language, chat_id) == "en"  # the last answer
    run(app, message, 2, "hi", {"answer": "canvas", "canvas_edit": {"op": "kind", "outcome": "done"}})
    assert run(app, store.response_language, chat_id) == "en"  # an edit's fixed reply says nothing about the chat

    async def prefer() -> None:
        async with db(app).session() as s:
            s.add(orm.ChatState(chat_id=chat_id, preferred_language="hi"))

    run(app, prefer)
    assert run(app, store.response_language, chat_id) == "hi"  # "answer in Hindi" sticks
