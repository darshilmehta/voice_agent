"""The planner refines the draft; it doesn't swap a good chart for a worse one (services/canvas/draft.py
``refinement_loses``, docs/DESIGN.md §12.1). Found in the final real-model run: with a misheard company name the
planner's refinement replaced a correct 8-quarter draft with a 2-point bar chart about 7 s later."""

from __future__ import annotations

import re

import pytest

from app.domain.projects import Citation
from app.services.canvas.draft import cues, draft_visual, refinement_loses, turn_context
from app.services.canvas.spec import VisualSpec

from .canvas_helpers import TWO_FILES, TWO_LABELS, V, Z, report_datasets, two_companies, two_companies_quarterly
from .canvas_turn_helpers import Script, choose, choose_table, planner_calls, report_chat
from .test_canvas_api import chat, drain, project, upload
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


REPRO = "Show me Valmora's quarterly revenue and EBITDA for FY23 and FY24"
BOTH_YEARS = ["ds_q_fy23:revenue", "ds_q_fy24:revenue", "ds_q_fy23:ebitda", "ds_q_fy24:ebitda"]


def test_the_final_runs_two_point_refinement_of_the_eight_quarter_draft_loses():
    """The last real-model run (2 of 4): the 8-quarter draft was replaced ~6 s later by a grouped bar of Q4 FY23 and
    Q4 FY24. "FY23 and FY24" over quarterly tables means all eight quarters; "Q4 FY23" doesn't show "FY23"."""
    drafted = draft_visual(REPRO, "en", POOL_LIST)
    assert drafted.spec is not None and drafted.spec.datasets == ["ds_q_fy23", "ds_q_fy24"]
    q4_only = VisualSpec(
        kind="grouped_bar", datasets=["ds_q_fy23", "ds_q_fy24"], series=BOTH_YEARS, periods=["Q4 FY23", "Q4 FY24"]
    )
    assert loses(q4_only, REPRO) == "the planner's chart shows fewer of the periods asked for"
    # the same chart of all eight quarters, as bars, is fine
    eight = VisualSpec(kind="grouped_bar", datasets=["ds_q_fy23", "ds_q_fy24"], series=BOTH_YEARS)
    assert loses(eight, REPRO) is None


def test_a_chart_of_fewer_quarters_loses_even_when_every_year_named_is_on_it():
    one_year = "Show me quarterly revenue and EBITDA for FY24"
    two = VisualSpec(
        kind="grouped_bar", datasets=["ds_q_fy24"], series=["ds_q_fy24:revenue", "ds_q_fy24:ebitda"],
        periods=["Q3 FY24", "Q4 FY24"],
    )  # fmt: skip
    assert loses(two, one_year) == "the planner's chart shows fewer of the periods asked for"


@pytest.mark.parametrize(
    "question",
    [
        "Show me only Q4 revenue and EBITDA for FY23 and FY24",
        "Just show the Q4 revenue and EBITDA of FY23 and FY24",
        "Show me Q4 FY23 and Q4 FY24 revenue and EBITDA",
    ],
)
def test_fewer_points_are_fine_when_the_question_asks_for_fewer(question):
    q4_only = VisualSpec(
        kind="grouped_bar", datasets=["ds_q_fy23", "ds_q_fy24"], series=BOTH_YEARS, periods=["Q4 FY23", "Q4 FY24"]
    )
    assert cues(question).narrow
    assert loses(q4_only, question) is None


def test_the_latest_quarter_asked_for_is_narrow_and_a_whole_year_is_not():
    assert cues("What was revenue in the latest quarter?").narrow
    assert cues("सिर्फ़ Q4 FY24 का राजस्व दिखाओ").narrow
    assert not cues(REPRO).narrow
    assert not cues("Show quarterly revenue").narrow


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


# ------------------------------------------------------------------ one document's tables, unless the question compares
#
# Found in the final real run: "the company's quarterly revenue" in a chat of both companies' documents drew the deck's
# Q4 table next to Valmora's FY23 table in one chart. The draft and the planner's refinement draw from one document.

QUARTERLY = two_companies_quarterly()
COMPANY = "Show the company's quarterly revenue"
MIXED = VisualSpec(
    kind="line", datasets=["ds_z_q24", "ds_q_fy23"], series=["ds_z_q24:revenue", "ds_q_fy23:revenue"], periods=[]
)


def test_a_planner_chart_that_mixes_two_documents_never_replaces_the_draft():
    ctx = turn_context(COMPANY, None, sources(Z, V), TWO_LABELS)
    drafted = draft_visual(
        COMPANY, "en", QUARTERLY, documents=ctx.scope, source_documents=ctx.source_documents, filenames=TWO_FILES
    )
    assert drafted.spec is not None and drafted.spec.datasets == ["ds_z_q23", "ds_z_q24"]
    by_id = {d.id: d for d in QUARTERLY}
    why = refinement_loses(MIXED, drafted, by_id, cues(COMPANY, None, names=ctx.names), ctx, TWO_FILES)
    assert why == "the planner's chart mixes tables of different documents"


def test_a_question_that_compares_the_companies_lets_the_planner_draw_from_both():
    question = "Show both companies' quarterly revenue"
    ctx = turn_context(question, None, sources(Z, V), TWO_LABELS)
    assert ctx.compare
    drafted = draft_visual(
        question, "en", QUARTERLY, documents=ctx.scope, source_documents=ctx.source_documents, filenames=TWO_FILES
    )
    assert drafted.spec is not None
    by_id = {d.id: d for d in QUARTERLY}
    assert refinement_loses(MIXED, drafted, by_id, cues(question, None, names=ctx.names), ctx, TWO_FILES) is None


TWO_COMPANY_TEXTS = {
    "valmora_annual_report_fy24.txt": (
        "Valmora annual report FY24\n\nQuarterly results.\f"
        "| Quarter | Revenue (₹ crore) | EBITDA (₹ crore) |\n"
        "| Q1 FY24 | 1,742 | 352 |\n| Q2 FY24 | 1,801 | 372 |\n| Q3 FY24 | 1,889 | 399 |\n| Q4 FY24 | 1,933 | 422 |"
    ),
    "zephyra_investor_deck_q4fy24.txt": (
        "Zephyra investor deck Q4FY24\n\nQuarterly results.\f"
        "| Quarter | Revenue (₹ crore) | EBITDA (₹ crore) |\n"
        "| Q1 FY24 | 880 | 128 |\n| Q2 FY24 | 930 | 140 |\n| Q3 FY24 | 990 | 156 |\n| Q4 FY24 | 1,070 | 172 |"
    ),
}


@pytest.fixture
def two_company_chat(make_app, fakes):
    """A project with both companies' documents, each with a quarterly table; the reranker finds Zephyra's table the
    best match, Valmora's the next, so Zephyra's deck is the top source."""
    with make_app() as api:
        p = project(api, "Two companies")
        for name, text in TWO_COMPANY_TEXTS.items():
            upload(api, p, name, text.encode())
        drain(api)
        fakes.reranker.scorer = lambda q, passage: 0.9 if "880" in passage else 0.6 if "1,742" in passage else 0.1
        yield api, chat(api, p), fakes


def both_documents(messages) -> dict:
    """A planner that takes the first table offered of each document (the 4B model did: "the company's quarterly
    revenue" got the deck's Q4 table and the other company's FY23 table)."""
    first: dict[str, str] = {}
    for line in messages[-1].content.splitlines():
        if m := re.match(r'(D\d+) ".*" \((\S+)', line):
            first.setdefault(m.group(2), m.group(1))
    aliases = list(first.values())
    return {"kind": "line", "datasets": aliases, "series": [f"{a} · Revenue" for a in aliases]}


def files_of(panel: dict) -> set[str]:
    return {s["filename"] for s in panel["sources"]}


def test_a_turn_naming_no_company_draws_one_chart_from_the_top_sources_document(two_company_chat, unsure):
    api, chat_id, fakes = two_company_chat
    Script(planner=both_documents).install(fakes.llm)
    ask(api, chat_id, COMPANY)
    (panel,) = canvas(api, chat_id)
    assert files_of(panel) == {"zephyra_investor_deck_q4fy24.txt"}
    assert [r["x"] for r in panel["rows"]] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    offered = planner_calls(fakes.llm)[0]["messages"][-1].content  # the planner saw nothing of Valmora's
    assert "zephyra_investor_deck_q4fy24.txt" in offered and "valmora" not in offered.lower()


def test_a_turn_comparing_the_companies_still_gets_both(two_company_chat, unsure):
    api, chat_id, fakes = two_company_chat
    Script(planner=both_documents).install(fakes.llm)
    ask(api, chat_id, "Show both companies' quarterly revenue")
    (panel,) = canvas(api, chat_id)
    assert files_of(panel) == set(TWO_COMPANY_TEXTS)
    assert messages(api, chat_id)[-1]["route"]["visual_plan"]["planner"] == "changed"
