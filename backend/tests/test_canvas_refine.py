"""The planner refines the draft; it doesn't swap a good chart for a worse one (services/canvas/draft.py
``refinement_loses``, docs/DESIGN.md §12.1). Found in the final real-model run: with a misheard company name the
planner's refinement replaced a correct 8-quarter draft with a 2-point bar chart about 7 s later."""

from __future__ import annotations

import pytest

from app.domain.projects import Citation
from app.services.canvas.draft import cues, draft_visual, refinement_loses, turn_context
from app.services.canvas.spec import VisualSpec

from .canvas_helpers import TWO_FILES, TWO_LABELS, V, report_datasets, two_companies
from .canvas_turn_helpers import Script, choose, choose_table, planner_calls, report_chat
from .test_canvas_turns import ask, canvas, messages, visuals

DS = report_datasets()
POOL_LIST = list(DS.values())
POOL = {d.id: d for d in POOL_LIST}
QUESTION = "Show me quarterly revenue and EBITDA for FY23 and FY24"


def loses(planned: VisualSpec, question: str = QUESTION, pool_list=POOL_LIST, **ctx) -> str | None:
    drafted = draft_visual(question, "en", pool_list, **{k: v for k, v in ctx.items() if k != "turn"})
    assert drafted.spec is not None, drafted.reasons
    turn = ctx.get("turn") or turn_context(question, None, [], {})
    return refinement_loses(
        planned, drafted, {d.id: d for d in pool_list}, cues(question, None, names=turn.names), turn
    )


# ------------------------------------------------------------------ what the planner may not do to the draft


def test_the_draft_is_eight_quarters_of_two_tables():
    drafted = draft_visual(QUESTION, "en", POOL_LIST)
    assert drafted.spec is not None and drafted.spec.kind == "line"
    assert drafted.spec.datasets == ["ds_q_fy23", "ds_q_fy24"] and len(drafted.spec.series) == 4


def test_a_two_point_bar_chart_of_a_yearly_table_does_not_replace_an_eight_quarter_draft():
    two_points = VisualSpec(kind="bar", datasets=["ds_highlights"], series=["ds_highlights:revenue_from_operations"])
    why = loses(two_points)
    assert why is not None and "the draft's table matches the question better" in why


def test_nor_the_same_tables_cut_down_to_fewer_quarters_than_the_draft_has():
    both_years = ["ds_q_fy23:revenue", "ds_q_fy24:revenue", "ds_q_fy23:ebitda", "ds_q_fy24:ebitda"]
    fewer = VisualSpec(
        kind="line", datasets=["ds_q_fy23", "ds_q_fy24"], series=both_years, periods=["Q3 FY24", "Q4 FY24"]
    )
    why = loses(fewer, "Show me quarterly revenue and EBITDA")  # no period named: the whole trend is the question
    assert why == "the planner's chart has fewer periods (2 against 8)"
    # when the question names periods, what counts is how many of them the chart shows
    assert loses(fewer) == "the planner's chart shows fewer of the periods asked for"  # FY23 and FY24; only FY24 left


def test_nor_a_chart_with_fewer_of_the_series_the_question_names():
    revenue_only = VisualSpec(
        kind="line", datasets=["ds_q_fy23", "ds_q_fy24"], series=["ds_q_fy23:revenue", "ds_q_fy24:revenue"]
    )
    assert "fewer of the series" in (loses(revenue_only) or "")


@pytest.mark.parametrize("kind", ["bar", "grouped_bar", "table"])
def test_another_kind_of_the_same_data_may_replace_it(kind):
    same_data = VisualSpec(
        kind=kind,
        datasets=["ds_q_fy23", "ds_q_fy24"],
        series=["ds_q_fy23:revenue", "ds_q_fy24:revenue", "ds_q_fy23:ebitda", "ds_q_fy24:ebitda"],
    )
    assert loses(same_data) is None


def test_a_narrower_selection_is_fine_when_the_question_asks_for_it():
    question = "Show revenue for Q3 FY24 and Q4 FY24"
    narrow = VisualSpec(
        kind="bar", datasets=["ds_q_fy24"], series=["ds_q_fy24:revenue"], periods=["Q3 FY24", "Q4 FY24"]
    )
    assert loses(narrow, question) is None


def test_the_planner_may_choose_a_better_series_than_the_drafts_guess():
    question = "Show me the EBITDA margin"  # the draft has the table's first measure, the planner the one asked for
    margin = VisualSpec(kind="line", datasets=["ds_q_fy24"], series=["ds_q_fy24:ebitda_margin"])
    assert loses(margin, question) is None


# ------------------------------------------------------------------ the company's documents

TWO = two_companies()
ASK = "Show Valmora's segment revenue breakdown"


def sources(*document_ids: str) -> list[Citation]:
    return [
        Citation(
            source_id=f"S{n}",
            document_id=d,
            filename=TWO_FILES[d],
            page_start=1,
            page_end=1,
            chunk_id=f"{d}:c",
            snippet="",
        )
        for n, d in enumerate(document_ids, start=1)
    ]


def test_another_companys_lookalike_table_never_replaces_the_named_companys_draft():
    ctx = turn_context(ASK, None, sources(V), TWO_LABELS)
    zephyra = VisualSpec(kind="donut", datasets=["ds_z_seg"], series=["ds_z_seg:revenue_fy24"])
    drafted = draft_visual(
        ASK,
        "en",
        TWO,
        documents=ctx.documents,
        names=ctx.names,
        source_documents=ctx.source_documents,
        filenames=TWO_FILES,
    )
    assert drafted.spec is not None and drafted.spec.datasets == ["ds_v_seg"]
    by_id = {d.id: d for d in TWO}
    why = refinement_loses(zephyra, drafted, by_id, cues(ASK, None, names=ctx.names), ctx, TWO_FILES)
    assert why == "the planner's table is not from the company's documents"


def test_with_a_misheard_company_name_the_answers_own_documents_decide():
    """ "Valmora" came out as something the app doesn't know: nothing names a company, but the answer was retrieved from
    Valmora's report, and the chart must come from there too."""
    heard = "Show Valmorra's segment revenue breakdown"
    ctx = turn_context(heard, None, sources(V), TWO_LABELS)
    assert ctx.documents is None and ctx.source_documents == {V}
    drafted = draft_visual(heard, "en", TWO, source_documents=ctx.source_documents, filenames=TWO_FILES)
    assert drafted.spec is not None and drafted.spec.datasets == ["ds_v_seg"]
    zephyra = VisualSpec(kind="donut", datasets=["ds_z_seg"], series=["ds_z_seg:revenue_fy24"])
    by_id = {d.id: d for d in TWO}
    why = refinement_loses(zephyra, drafted, by_id, cues(heard, None), ctx, TWO_FILES)
    assert why == "the planner's table is not from the documents the answer came from"
    own = VisualSpec(kind="bar", datasets=["ds_v_seg"], series=["ds_v_seg:revenue_fy24"])
    assert refinement_loses(own, drafted, by_id, cues(heard, None), ctx, TWO_FILES) is None


def test_a_table_that_matches_the_question_clearly_worse_does_not_replace_the_draft():
    drafted = draft_visual(ASK, "en", TWO, filenames=TWO_FILES)
    assert drafted.spec is not None
    by_id = {d.id: d for d in TWO}
    ctx = turn_context(ASK, None, [], {})
    worse = VisualSpec(kind="bar", datasets=["ds_z_glance"], series=["ds_z_glance:ebitda"])
    why = refinement_loses(worse, drafted, by_id, cues(ASK, None), ctx, TWO_FILES)
    assert why is not None and "matches the question better" in why


# ------------------------------------------------------------------ in a turn


@pytest.fixture
def unsure(monkeypatch):
    """Every draft isn't sure of itself, so that the planner runs (its chart is what is under test)."""
    import app.services.canvas.service as service

    real = service.draft_visual

    def doubtful(*args, **kwargs):
        d = real(*args, **kwargs)
        d.confident = False
        return d

    monkeypatch.setattr(service, "draft_visual", doubtful)


@pytest.fixture
def app_with_report(make_app, fakes):
    with make_app() as api:
        _, chat_id, ds = report_chat(api)
        yield api, chat_id, ds, fakes


def test_a_planner_chart_with_fewer_periods_leaves_the_draft_on_screen(app_with_report, unsure):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose("bar", "Revenue from operations")).install(fakes.llm)  # the yearly table: FY23, FY24
    events = ask(api, chat_id, "Show me revenue by quarter")
    assert len(planner_calls(fakes.llm)) == 1
    assert visuals(events) == [("preparing", None), ("ready", None)]  # the draft; no second "ready", no "failed"
    (panel,) = canvas(api, chat_id)
    assert panel["kind"] == "line" and [r["x"] for r in panel["rows"]] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    plan = messages(api, chat_id)[-1]["route"]["visual_plan"]
    assert plan["draft"] == "refine" and plan["planner"] == "kept"
    assert plan["reasons"][-1].startswith("planner's chart not used: the draft's table matches the question better")
    snapshots = [d["panels"] for e, d in events if e == "canvas"]
    assert len(snapshots) == 1  # the canvas was sent once, with the draft


def test_a_planner_chart_that_covers_the_question_as_well_still_replaces_it_in_place(app_with_report, unsure):
    api, chat_id, _, fakes = app_with_report
    Script(planner=choose_table("bar", "Q1 FY24", "Revenue", "EBITDA")).install(fakes.llm)
    events = ask(api, chat_id, "Show me revenue by quarter")
    assert visuals(events) == [("preparing", None), ("ready", None), ("ready", None)]
    (panel,) = canvas(api, chat_id)
    assert panel["kind"] == "bar" and [s["label"] for s in panel["series"]] == ["Revenue", "EBITDA"]
    assert messages(api, chat_id)[-1]["route"]["visual_plan"]["planner"] == "changed"
