"""The visual planner with a scripted model (services/canvas/planner.py): when a visual is worth it, which tables
are offered, the per-request schema, mapping the model's choice back to a spec, and its failure modes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.providers.llm import LLMError
from app.services.canvas.planner import (
    VisualPlanner,
    bridges,
    build_catalog,
    builds,
    first_valid,
    planner_messages,
    planner_schema,
    rank_candidates,
    to_spec,
    transposed,
    variants,
    visual_intent,
)
from app.services.canvas.spec import VisualSpec, resolve

from .canvas_helpers import report_datasets
from .fakes import FakeLLM

DS = report_datasets()
POOL = list(DS.values())
NAMES = {"doc_1": "valmora_annual_report_fy24.pdf"}


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Show me revenue over the last four quarters", "requested"),
        ("Can you chart EBITDA by segment?", "requested"),
        ("Plot the margin trend", "requested"),
        ("Put that on screen", "requested"),
        ("revenue ka chart dikhao", "requested"),
        ("quarterly revenue dikhaiye", "requested"),
        ("तिमाही राजस्व दिखाओ", "requested"),
        ("सेगमेंट का ग्राफ़ बनाइए", "requested"),
        ("How has revenue moved over the years?", "suggested"),
        ("Compare FY23 and FY24 EBITDA", "suggested"),
        ("What is the revenue breakdown by segment?", "suggested"),
        ("Revenue in FY23 vs FY24", "suggested"),
        ("Q3 aur Q4 ki tulna karo", "suggested"),
        ("पिछले साल से राजस्व में कितनी वृद्धि हुई?", "suggested"),
        ("Who is the company secretary?", "none"),
        ("What is the ISIN?", "none"),
        ("thanks", "none"),
        ("कंपनी का पता क्या है?", "none"),
        # asking to see, in other words
        ("Put this year's headline numbers up against last year", "requested"),
        ("I'd like to see the segment split", "requested"),
        ("Can I see the borrowings?", "requested"),
        ("मुझे तिमाही मुनाफ़ा दिखाइये", "requested"),
        ("EBITDA ka graph banao", "requested"),
        # a trend, a comparison, a ranking, a breakdown, a bridge, headline numbers, dates (EN, HI, Hinglish)
        ("How has the EBITDA margin trended?", "suggested"),
        ("How did the company do this year compared with last year?", "suggested"),
        ("How does Valmora stack up against its peers?", "suggested"),
        ("Which segment grew the fastest?", "suggested"),
        ("Who has the stronger balance sheet, Valmora or Zephyra?", "suggested"),
        ("Break the borrowings down for me", "suggested"),
        ("Walk me through how revenue turns into profit", "suggested"),
        ("What are the key financial highlights?", "suggested"),
        ("What are the important dates of the scheme?", "suggested"),
        ("किस ज़िले का बजट सबसे ज़्यादा है?", "suggested"),
        ("पिछले साल की तुलना में कर्मचारी कितने बढ़े?", "suggested"),
        ("Kaunsa segment sabse zyada revenue laata hai?", "suggested"),
        ("Debt ka breakdown chahiye", "suggested"),
        ("Zephyra ka performance pichle saal ke mukable kaisa raha?", "suggested"),
        ("Revenue har quarter kaisa raha?", "suggested"),
        # plain facts and chat: no chart (the gate's old false positives among them)
        ("What share of revenue came from exports?", "none"),
        ("What dividend per share did the board recommend?", "none"),
        ("How much is the meal allowance each day?", "none"),
        ("What is the revenue growth guidance for next year?", "none"),
        ("How many shareholders does the company have?", "none"),
        ("What does the report show about debt?", "none"),
        ("How do I claim travel expenses?", "none"),
        ("Which company audits the accounts?", "none"),
        ("महिलाओं की हिस्सेदारी कितनी है?", "none"),
        ("परिवार की सालाना आय की सीमा कितनी है?", "none"),
        ("FY24 mein revenue kitna tha?", "none"),
        ("okay", "none"),
    ],
)
def test_visual_intent(question, intent):
    assert visual_intent(question) == intent


def test_an_answer_full_of_numbers_suggests_a_visual():
    assert visual_intent("What were the results?") == "none"
    assert (
        visual_intent("What were the results?", "Revenue ₹7,365 crore, EBITDA ₹1,545 crore, margin 21.0%.")
        == "suggested"
    )


def test_candidates_ranked_by_the_question():
    ranked = rank_candidates("Show quarterly revenue for FY24", POOL, limit=3)
    assert ranked[0].id == "ds_q_fy24"
    assert "ds_glossary" not in [d.id for d in rank_candidates("glossary terms", POOL, limit=10)]  # not chartable
    boosted = rank_candidates("show me the numbers", POOL, source_chunks=[DS["cash_flow"].chunk_id], limit=2)  # type: ignore[list-item]
    assert boosted[0].id == "ds_cash_flow"
    hindi = rank_candidates("segment revenue dikhao", POOL, query_en="show revenue by segment", limit=2)
    assert hindi[0].id == "ds_segments"


def test_catalog_and_schema_offer_only_real_references():
    cat = build_catalog([DS["q_fy24"], DS["highlights"]], NAMES, "en")
    assert list(cat.aliases) == ["D1", "D2"]
    assert cat.series["D1 · Revenue"] == "ds_q_fy24:revenue"
    assert cat.series["D2 · EBITDA margin"] == "ds_highlights:ebitda_margin"
    assert cat.periods == ["FY23", "Q1 FY24", "Q2 FY24", "Q3 FY24", "FY24", "Q4 FY24"]  # chronological
    assert any("periods in rows: Q1 FY24, Q2 FY24, Q3 FY24, Q4 FY24" in line for line in cat.lines)
    assert any("Revenue from operations [₹ crore]" in line for line in cat.lines)
    schema = planner_schema(cat)
    js = schema.model_json_schema()
    series_enum = js["properties"]["series"]["items"]["enum"]
    assert "D1 · Revenue" in series_enum and all(s.startswith(("D1 · ", "D2 · ")) for s in series_enum)
    assert js["properties"]["periods"]["items"]["enum"] == cat.periods
    assert "none" in js["properties"]["kind"]["enum"]
    schema.model_validate({"kind": "line", "datasets": ["D1"], "series": ["D1 · Revenue"]})
    for bad in (
        {"kind": "line", "datasets": ["D1"], "series": ["D1 · Market share"]},  # invented series
        {"kind": "line", "datasets": ["D3"], "series": []},  # invented dataset
        {"kind": "line", "datasets": ["D1"], "series": [], "periods": ["FY21"]},  # invented period
        {"kind": "sankey", "datasets": ["D1"], "series": []},  # unknown kind
        {"kind": "line", "datasets": ["D1"], "series": ["D1 · Revenue"] * 7},  # too many
    ):
        with pytest.raises(ValidationError):
            schema.model_validate(bad)


def test_duplicate_labels_get_told_apart():
    cat = build_catalog([DS["q_fy24"], DS["q_fy23"]], NAMES, "en")
    assert cat.series["D1 · Revenue"] == "ds_q_fy24:revenue" and cat.series["D2 · Revenue"] == "ds_q_fy23:revenue"
    hi = build_catalog([DS["highlights"]], NAMES, "hi")
    assert any("[₹ करोड़]" in line for line in hi.lines)


def test_choice_maps_back_to_a_spec():
    cat = build_catalog([DS["q_fy24"], DS["highlights"]], NAMES, "hi")
    spec = to_spec(
        {
            "kind": "bar",
            "datasets": ["D1"],
            "series": ["D1 · Revenue", "D1 · EBITDA"],
            "periods": ["Q3 FY24", "Q4 FY24"],
            "categories": ["D2 · EBITDA"],  # of a dataset the visual doesn't use: dropped
            "highlight": ["Q4 FY24"],
            "calculations": [{"op": "growth", "series": "D1 · Revenue"}, {"op": "share", "series": "D2 · EBITDA"}],
            "title": "तिमाही राजस्व",
        },
        cat,
        "hi",
    )
    assert spec == VisualSpec(
        kind="bar",
        datasets=["ds_q_fy24"],
        series=["ds_q_fy24:revenue", "ds_q_fy24:ebitda"],
        periods=["Q3 FY24", "Q4 FY24"],
        highlight=["Q4 FY24"],
        calculations=[{"op": "growth", "series": "ds_q_fy24:revenue"}],  # type: ignore[list-item]
        title="तिमाही राजस्व",
        language="hi",
    )
    assert to_spec({"kind": "none", "datasets": ["D1"], "series": []}, cat, "en") is None


def test_variants_fall_back_to_compatible_kinds():
    spec = VisualSpec(kind="line", datasets=["ds_segments"], series=["ds_segments:revenue_fy24"])
    kinds = [v.kind for v in variants(spec)]
    assert kinds[:3] == ["line", "line", "line"] and "bar" in kinds


BY_ID = {d.id: d for d in POOL}
SEG = "ds_segments"


def test_rows_chosen_as_a_donuts_series_become_its_slices():
    # what qwen3:4b does: the slices as "series"
    spec = VisualSpec(
        kind="donut", datasets=[SEG], series=[f"{SEG}:specialty_chemicals", f"{SEG}:digital_services"], periods=["FY24"]
    )
    flipped = transposed(spec, BY_ID)
    assert flipped is not None
    assert (flipped.series, flipped.categories) == (
        [f"{SEG}:revenue_fy24"],
        [f"{SEG}:specialty_chemicals", f"{SEG}:digital_services"],
    )
    assert first_valid(spec, BY_ID, builds(BY_ID)) == flipped
    no_period = spec.model_copy(update={"periods": []})
    assert transposed(no_period, BY_ID).series == [f"{SEG}:revenue_fy24"]  # type: ignore[union-attr]  # the latest


def test_a_waterfall_that_doesnt_add_up_becomes_one_that_does():
    cf = "ds_cash_flow"
    rows = [
        "net_cash_generated_from_operating_activities",
        "net_cash_used_in_investing_activities",
        "net_cash_used_in_financing_activities",
        "net_increase_in_cash_and_cash_equivalents",
        "cash_and_cash_equivalents_at_the_beginning_of",
    ]
    spec = VisualSpec(kind="waterfall", datasets=[cf], series=[f"{cf}:{r}" for r in rows], periods=["FY24"])
    valid = first_valid(spec, BY_ID, builds(BY_ID))
    assert valid is not None and valid.kind == "waterfall" and valid.series == [f"{cf}:fy24"]
    assert valid.categories == [f"{cf}:{r}" for r in rows[:4]]  # the flows, then the total they add up to
    assert bridges(valid, BY_ID)[0].categories == valid.categories


def test_choices_lose_what_the_table_doesnt_have():
    cat = build_catalog([DS["segments"], DS["q_fy24"]], NAMES, "en")
    spec = to_spec(
        {"kind": "bar", "datasets": ["D1"], "series": ["D1 · Revenue FY24"], "periods": ["Q3 FY24", "FY24"]}, cat, "en"
    )
    assert spec is not None and spec.periods == ["FY24"]  # Q3 FY24 is another table's period
    from .canvas_helpers import DATES_HI, typed

    dates = typed(DATES_HI, dataset_id="ds_dates")
    cat = build_catalog([dates], NAMES, "hi")
    spec = to_spec({"kind": "timeline", "datasets": ["D1"], "series": list(cat.series)[:2]}, cat, "hi")
    assert spec is not None and spec.series == [] and spec.datasets == ["ds_dates"]


def test_a_column_mixing_units_is_charted_in_its_main_unit():
    from .canvas_helpers import typed

    plants = typed(
        [
            ["Facility", "Installed capacity"],
            ["Dahej Complex", "2,10,000 tpa"],
            ["Vapi Unit I", "90,000 tpa"],
            ["Bengaluru Centre", "1,300 seats"],
        ],
        dataset_id="ds_plants",
    )
    pool = {"ds_plants": plants}
    spec = VisualSpec(kind="bar", datasets=["ds_plants"], series=["ds_plants:installed_capacity"])
    valid = first_valid(spec, pool, builds(pool))
    assert valid is not None and valid.categories == ["ds_plants:dahej_complex", "ds_plants:vapi_unit_i"]


# ------------------------------------------------------------------ planning with the model


def planner(llm: FakeLLM, **kw) -> VisualPlanner:
    return VisualPlanner(llm, model="qwen3:4b-instruct", timeout_s=kw.pop("timeout_s", 2.0), max_tokens=160, **kw)


async def test_plan_with_the_model():
    llm = FakeLLM()
    llm.route = {
        "kind": "line",
        "datasets": ["D1"],
        "series": ["D1 · Revenue", "D1 · EBITDA"],
        "title": "Revenue and EBITDA",
    }
    result = await planner(llm).plan_detailed("Show me revenue and EBITDA by quarter", "en", POOL, filenames=NAMES)
    assert result.source == "model" and result.intent == "requested"
    assert result.spec == VisualSpec(
        kind="line",
        datasets=["ds_q_fy24"],
        series=["ds_q_fy24:revenue", "ds_q_fy24:ebitda"],
        title="Revenue and EBITDA",
    )
    (call,) = llm.json_calls
    assert call["model"] == "qwen3:4b-instruct" and call["max_tokens"] == 160
    system, user = call["messages"]
    assert "never write numbers" in system.content and "Question: Show me revenue and EBITDA by quarter" in user.content
    assert resolve(result.spec, {d.id: d for d in POOL})  # type: ignore[arg-type]


async def test_an_unresolvable_choice_is_simplified():
    llm = FakeLLM()
    # a line over segments (categories, not periods) with a highlight: falls back to a bar of the same series
    llm.route = {"kind": "line", "datasets": ["D1"], "series": ["D1 · Revenue FY24"], "highlight": ["D1 · Total"]}
    result = await planner(llm).plan_detailed("show segment revenue", "en", [DS["segments"]], filenames=NAMES)
    assert result.source == "model" and result.spec is not None and result.spec.kind == "bar"
    assert result.spec.highlight == []


async def test_no_visual_when_the_question_doesnt_call_for_one():
    llm = FakeLLM()
    assert await planner(llm).plan("Who audits the company?", "en", POOL) is None
    assert llm.json_calls == []
    llm.route = {"kind": "kpi", "datasets": ["D1"], "series": ["D1 · EBITDA"]}
    assert await planner(llm).plan("Who audits the company?", "en", POOL, force=True) is not None


async def test_timeouts_and_failures():
    slow = FakeLLM()
    slow.json_delay = 1.0
    slow.route = {"kind": "line", "datasets": ["D1"], "series": ["D1 · Revenue"]}
    result = await planner(slow, timeout_s=0.05).plan_detailed("show quarterly revenue", "en", POOL)
    assert result.source == "heuristic" and "timed out" in result.reason  # asked to see something: a default chart
    assert result.spec is not None and result.spec.kind == "line" and result.latency_ms < 1000
    result = await planner(slow, timeout_s=0.05).plan_detailed("how did revenue grow?", "en", POOL)
    assert result.spec is None and result.source == "none"  # only suggested: nothing
    broken = FakeLLM()
    broken.route = LLMError("model output doesn't match")
    result = await planner(broken).plan_detailed("chart the cash flow", "en", POOL)
    assert result.source == "heuristic" and "planner failed" in result.reason
    nothing = FakeLLM()
    nothing.route = {"kind": "none", "datasets": ["D1"], "series": []}
    result = await planner(nothing).plan_detailed("show me the auditors' fees", "en", POOL)
    assert result.spec is None and result.reason == "the model chose no visual"  # the model's "none" is respected


async def test_no_candidates():
    llm = FakeLLM()
    assert (
        await planner(llm).plan_detailed("show the glossary", "en", [DS["glossary"]])
    ).reason == "no chartable table"


def test_prompt_in_hindi_keeps_the_catalog():
    cat = build_catalog([DS["q_fy24"]], NAMES, "hi")
    _system, user = planner_messages("तिमाही राजस्व दिखाओ", "hi", cat, answer="राजस्व बढ़ा।")
    assert "Language for the title: Hindi" in user.content and "Spoken answer: राजस्व बढ़ा।" in user.content
    assert 'D1 "Q Fy24"' in user.content
