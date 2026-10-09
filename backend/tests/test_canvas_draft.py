"""The instant draft of a turn's visual (services/canvas/draft.py, docs/DESIGN.md §12.1): the chart code picks and
builds without the model, how sure it is, a question naming Valmora never drawn from Zephyra's lookalike table, two
companies side by side."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.providers.ingestion import document_label
from app.services.canvas.builder import build_visual, check_grounding
from app.services.canvas.draft import (
    cues,
    draft_visual,
    filename_labels,
    names_table,
    same_choice,
    turn_context,
)
from app.services.canvas.spec import VisualSpec, resolve
from app.services.subjects import documents_by_name, named_documents

from .canvas_helpers import (
    DATES_HI,
    DISTRICTS_HI,
    TWO_FILES,
    TWO_LABELS,
    V,
    Z,
    report_datasets,
    two_companies,
    two_companies_quarterly,
    typed,
)
from .test_canvas_refine import sources

DS = report_datasets()
DS["dates_hi"] = typed(DATES_HI, heading=("5. महत्वपूर्ण तिथियाँ",), page=None, dataset_id="ds_dates_hi")
DS["districts_hi"] = typed(DISTRICTS_HI, heading=("8. जिलेवार लक्ष्य एवं बजट",), page=None, dataset_id="ds_districts_hi")
POOL = list(DS.values())
NAMES = {"doc_1": "valmora_annual_report_fy24.pdf"}


def draft(question: str, *, language: str = "en", pool=POOL, **kw):
    return draft_visual(question, language, pool, filenames=NAMES, **kw)


def grounded(spec: VisualSpec, pool=POOL, names=NAMES) -> bool:
    by_id = {d.id: d for d in pool}
    v = build_visual(
        resolve(spec, by_id, filenames=names),
        visual_id="vis_d",
        project_id="p",
        chat_id="c",
        filenames=names,
        now=datetime(2026, 10, 9, tzinfo=UTC),
    )
    return check_grounding(v) == []


@pytest.mark.parametrize(
    ("question", "query_en", "kind", "datasets", "series", "confident"),
    [
        ("Show me quarterly revenue for FY24", None, "line", ["ds_q_fy24"], ["ds_q_fy24:revenue"], True),
        (
            "How did the EBITDA margin move across the quarters of FY23?",
            None,
            "line",
            ["ds_q_fy23"],
            ["ds_q_fy23:ebitda_margin"],
            True,
        ),
        (
            "mujhe quarterly EBITDA ka graph dikhao",
            "Show me a graph of quarterly EBITDA",
            "line",
            ["ds_q_fy23", "ds_q_fy24"],
            ["ds_q_fy23:ebitda", "ds_q_fy24:ebitda"],
            None,
        ),
        ("Show quarterly profit after tax as a bar chart", None, "bar", ["ds_q_fy23", "ds_q_fy24"], None, True),
        ("Show the key financial highlights", None, "kpi", ["ds_highlights"], None, True),
        ("Show the segment revenue breakdown", None, "donut", ["ds_segments"], ["ds_segments:revenue_fy24"], True),
        (
            "Compare segment EBITDA in FY23 and FY24",
            None,
            "grouped_bar",
            ["ds_segments"],
            ["ds_segments:ebitda_fy23", "ds_segments:ebitda_fy24"],
            True,
        ),
        ("Show the cash flow bridge for FY24", None, "waterfall", ["ds_cash_flow"], ["ds_cash_flow:fy24"], True),
        ("तिमाही राजस्व दिखाओ", "Show quarterly revenue", "line", ["ds_q_fy23", "ds_q_fy24"], None, None),
        (
            "योजना की महत्वपूर्ण तिथियाँ दिखाइए",
            "Show the scheme's important dates",
            "timeline",
            ["ds_dates_hi"],
            [],
            True,
        ),
        ("जिलेवार बजट दिखाओ", "Show the budget by district", "donut", ["ds_districts_hi"], None, True),
        # the words tell neither the table nor the series: a draft all the same, for the planner to check
        ("Show me the numbers", None, None, None, None, False),
    ],
)
def test_the_draft_chart_for_a_question(question, query_en, kind, datasets, series, confident):
    d = draft(question, language="hi" if query_en else "en", query_en=query_en)
    assert d.spec is not None, d.reasons
    if kind is not None:
        assert d.spec.kind == kind, d.reasons
    if datasets is not None:
        assert sorted(d.spec.datasets) == sorted(datasets)
    if series is not None:
        assert d.spec.series == series
    if confident is not None:  # (None: either; the margin over the highlights' rows is close)
        assert d.confident is confident, d.reasons
    assert grounded(d.spec)  # every number of a draft is a cell, as for the planner's choice
    assert d.latency_ms < 100  # code: milliseconds


def test_a_draft_for_every_question_of_the_report_is_grounded():
    for question in (
        "Show revenue by segment",
        "How has the number of employees changed?",
        "Show the balance sheet",
        "Show inventories against last year",
        "Which district has the highest target?",
        "Show the budget and the target of each district",
        "Show EBITDA margin by quarter as a table",
        "Put the FY24 headline numbers up against FY23",
    ):
        d = draft(question)
        assert d.spec is not None, (question, d.reasons)
        assert grounded(d.spec), question


def test_the_kind_asked_for_wins_or_the_draft_isnt_sure():
    assert draft("Show quarterly revenue as a pie chart").confident is False  # a donut can't show quarters
    asked = draft("Show segment revenue for FY24 as a bar chart")
    assert asked.spec is not None and asked.spec.kind == "bar" and asked.confident


def test_cues_of_a_question():
    c = cues("Compare Valmora's segment EBITDA in FY23 and FY24 as a bar chart", names=["valmora"])
    assert c.kind == "bar" and {"comparison"} <= c.shapes
    assert c.periods == ("FY23", "FY24") and "valmora" not in c.words and "ebitda" in c.words
    assert cues("business line growth").kind is None  # a bare "line" isn't a line chart
    assert cues("I need the exact dividend numbers").kind == "table"
    assert "composition" in cues("Who owns Valmora? Break the shareholders down").shapes
    assert "ranking" in cues("कौन से जिले का लक्ष्य सबसे ज़्यादा है?").shapes
    assert cues("Kaun se district ka budget sabse zyada hai", "Which district has the highest budget").words >= {"बजट"}


def test_a_question_that_names_the_table_wants_all_of_it():
    sheet = DS["balance_sheet"]
    assert names_table(sheet, cues("Show the balance sheet"))
    assert not names_table(sheet, cues("Show inventories on the balance sheet"))  # a row of it
    whole = draft("Show the balance sheet")
    assert whole.spec is not None and whole.spec.kind == "table" and whole.confident


def test_same_choice_ignores_titles_and_highlights():
    a = VisualSpec(kind="line", datasets=["d"], series=["d:revenue"], title="Revenue")
    assert same_choice(a, a.model_copy(update={"title": "राजस्व", "highlight": ["Q4 FY24"]}))
    assert not same_choice(a, a.model_copy(update={"kind": "bar"}))
    assert not same_choice(a, a.model_copy(update={"series": ["d:ebitda"]}))


# ------------------------------------------------------------------ company-aware candidates

TWO = two_companies()
LABELS, FILES = TWO_LABELS, TWO_FILES
QUESTION = "Show Valmora's segment revenue breakdown"


def test_a_question_naming_valmora_never_gets_zephyras_table():
    named = named_documents((QUESTION,), LABELS)
    assert named is not None and named.document_ids == {V}
    d = draft_visual(QUESTION, "en", TWO, documents=named.document_ids, names=named.names, filenames=FILES)
    assert d.spec is not None and d.spec.datasets == ["ds_v_seg"] and d.spec.kind == "donut"
    assert {c.document_id for c in d.candidates} == {V}


def test_the_hindi_question_names_the_company_through_its_english_query():
    ctx = turn_context("वालमोरा के सेगमेंट का राजस्व दिखाओ", "Show Valmora's segment revenue", [], LABELS)
    assert ctx.documents == {V} and ctx.names == ("valmora",)


def test_two_companies_side_by_side():
    q = "Compare Valmora and Zephyra revenue"
    named = named_documents((q,), LABELS)
    assert named is not None
    companies = documents_by_name(named, LABELS)
    assert companies == {"valmora": frozenset({V}), "zephyra": frozenset({Z})}
    d = draft_visual(
        q, "en", TWO, documents=named.document_ids, names=named.names, companies=companies, filenames=FILES
    )
    assert d.spec is not None and set(d.spec.datasets) == {"ds_v_hl", "ds_z_glance"}
    assert d.spec.series == ["ds_v_hl:revenue_from_operations", "ds_z_glance:revenue_from_operations"]
    assert d.spec.kind == "grouped_bar" and d.confident
    assert grounded(d.spec, TWO, FILES)


def test_labels_from_file_names():
    assert filename_labels(FILES) == {V: document_label(FILES[V], None), Z: document_label(FILES[Z], None)}


# ------------------------------------------------------------------ one document's tables, unless the question compares

QUARTERLY = two_companies_quarterly()
COMPANY = "Show the company's quarterly revenue"  # names no company: both companies' quarterly tables fit


def documents_of(spec: VisualSpec, pool=QUARTERLY) -> set[str]:
    return {d.document_id for d in pool if d.id in spec.datasets}


@pytest.mark.parametrize(
    ("top", "tables"),
    [(V, {"ds_q_fy23", "ds_q_fy24"}), (Z, {"ds_z_q23", "ds_z_q24"})],
    ids=["valmora", "zephyra"],
)
def test_a_question_naming_no_company_draws_from_the_document_of_the_turns_top_source(top, tables):
    """Found in the final real run: "the company's quarterly revenue" in a chat of both companies' documents drew the
    deck's Q4 table next to Valmora's FY23 table. The answer comes from the top source's document: so does the chart."""
    other = Z if top == V else V
    ctx = turn_context(COMPANY, None, sources(top, other), TWO_LABELS)
    assert ctx.documents is None and not ctx.compare and ctx.scope == {top}
    d = draft_visual(
        COMPANY,
        "en",
        QUARTERLY,
        documents=ctx.scope,
        source_documents=ctx.source_documents,
        filenames=TWO_FILES,
    )
    assert d.spec is not None and set(d.spec.datasets) == tables and documents_of(d.spec) == {top}
    assert d.confident  # (the other document's tables fit as well by words: without it, the draft was not sure)
    # the same pool without the top source's say: by words and pages alone Zephyra's tables win, Valmora's source or not
    unscoped = draft_visual(COMPANY, "en", QUARTERLY, source_documents=ctx.source_documents, filenames=TWO_FILES)
    assert unscoped.spec is not None and documents_of(unscoped.spec) == {Z}


def test_a_draft_without_a_top_source_still_comes_from_one_documents_tables():
    ctx = turn_context(COMPANY, None, [], TWO_LABELS)
    assert ctx.scope is None
    d = draft_visual(COMPANY, "en", QUARTERLY, documents=ctx.scope, filenames=TWO_FILES)
    assert d.spec is not None and len(documents_of(d.spec)) == 1


def test_a_web_source_has_no_document_and_is_not_the_top_one():
    web = sources(V)[0].model_copy(update={"document_id": "", "chunk_id": "", "source_id": "W1"})
    ctx = turn_context(COMPANY, None, [web, *sources(Z)], TWO_LABELS)
    assert ctx.top_document == Z and ctx.scope == {Z}


@pytest.mark.parametrize(
    ("question", "query_en", "order", "compare", "scope"),
    [
        (COMPANY, None, (Z, V), False, {Z}),
        ("Show Valmora's quarterly revenue", None, (Z, V), False, {V}),  # a name beats the top source
        ("Compare Valmora and Zephyra revenue", None, (V, Z), True, {V, Z}),
        ("Valmora vs Zephyra: whose EBITDA margin is better?", None, (V, Z), True, {V, Z}),
        ("Show both companies' quarterly revenue", None, (Z, V), True, None),
        ("What was each company's revenue in FY24?", None, (Z, V), True, None),
        ("दोनों कंपनियों का तिमाही राजस्व दिखाओ", "Show both companies' quarterly revenue", (Z, V), True, None),
        ("Compare FY23 and FY24 revenue", None, (Z, V), False, {Z}),  # periods, not companies
        (COMPANY, None, (), False, None),
    ],
)
def test_which_documents_a_visual_may_draw_from(question, query_en, order, compare, scope):
    ctx = turn_context(question, query_en, sources(*order), TWO_LABELS)
    assert ctx.compare is compare and ctx.scope == (frozenset(scope) if scope else None)


def test_naming_both_companies_still_merges_them_beside_each_other():
    q = "Compare Valmora and Zephyra revenue"
    ctx = turn_context(q, None, sources(V, Z), TWO_LABELS)  # Valmora's passage is the top source
    d = draft_visual(
        q,
        "en",
        TWO,
        documents=ctx.scope,
        names=ctx.names,
        companies=ctx.companies,
        source_documents=ctx.source_documents,
        filenames=FILES,
    )
    assert d.spec is not None and set(d.spec.datasets) == {"ds_v_hl", "ds_z_glance"}
    assert d.spec.kind == "grouped_bar" and d.confident
