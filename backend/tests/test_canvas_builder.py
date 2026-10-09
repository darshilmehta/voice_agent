# ruff: noqa: RUF001  (documents print en dashes and minus signs: the tests use them on purpose)
"""Building visuals from resolved specs, and the grounding invariant (services/canvas/builder.py).

Every number in a Visual is a cell value (its CellRef's printed text parses back to it) or a Calculation listed in
``calculations`` whose inputs are cells. ``test_every_buildable_spec_is_grounded`` builds every spec the test datasets
allow (each kind, each series, filters, highlights, every calculation) and checks that with an independent checker,
besides the builder's own ``check_grounding``.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from datetime import UTC, datetime

import pytest

from app.domain.canvas import CALC_OPS, VISUAL_KINDS, Calculation, CellRef, Visual
from app.domain.projects import Citation
from app.services.canvas import calculator as calc
from app.services.canvas.builder import build_visual, check_grounding
from app.services.canvas.chartability import series_columns
from app.services.canvas.parsing import Quantity, parse_quantity
from app.services.canvas.spec import SpecError, VisualSpec, ref, resolve

from .canvas_helpers import DATES_HI, DISTRICTS_HI, report_datasets, typed

NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)
DS = report_datasets()
DS["districts"] = typed(
    DISTRICTS_HI, dataset_id="ds_districts", document_id="doc_hi", heading=("जिलेवार बजट",), page=None
)
DS["dates"] = typed(DATES_HI, dataset_id="ds_dates", document_id="doc_hi", heading=("तिथियाँ",), page=None)
BY_ID = {d.id: d for d in DS.values()}
FILENAMES = {"doc_1": "valmora_annual_report_fy24.pdf", "doc_hi": "suryodaya_notice.docx"}
H, SEG, Q24, Q23, CF = "ds_highlights", "ds_segments", "ds_q_fy24", "ds_q_fy23", "ds_cash_flow"


def build(sources: list[Citation] | None = None, **kw) -> Visual:
    r = resolve(VisualSpec.model_validate(kw), BY_ID, filenames=FILENAMES)
    return build_visual(
        r,
        visual_id="vis_t",
        project_id="prj_t",
        chat_id="cht_t",
        filenames=FILENAMES,
        now=NOW,
        existing_sources=sources or [],
    )


# ------------------------------------------------------------------ an independent grounding check

_NUM = re.compile(r"(?<![\w.])[−\-+]?\d[\d,]*(?:\.\d+)?(?!\d|[.,]\d)")


def cell_value(ref_: CellRef) -> float:
    q = parse_quantity(ref_.text)
    assert isinstance(q, Quantity), f"cell text {ref_.text!r} is not a number"
    return q.value


def assert_grounded(v: Visual) -> None:
    calc_values = [c.value for c in v.calculations]
    for c in v.calculations:
        inputs = [cell_value(i) for i in c.inputs]
        assert calc.close(calc.recompute(c, inputs), c.value, 2), c
    for row in v.rows:
        for key, value in row.values.items():
            if value is None:
                continue
            series = next(s for s in v.series if s.key == key)
            if series.calculated:
                assert value in calc_values
                continue
            ref_ = row.cells[key]
            assert ref_ is not None, (row.x, key)
            printed = cell_value(ref_)
            assert value == printed or (v.kind == "waterfall" and value == abs(printed)), (row.x, key, value, ref_.text)
    for t in v.tiles:
        assert (t.cell is not None and cell_value(t.cell) == t.value) or t.value in calc_values
        if t.delta is not None:
            assert t.delta.calculation in v.calculations and t.delta.value == t.delta.calculation.value
    shown = [x for r in v.rows for x in r.values.values() if x is not None] + [t.value for t in v.tiles] + calc_values
    labels = " ".join(
        [r.x for r in v.rows]
        + [s.label for s in v.series]
        + [t.label for t in v.tiles]
        + [e.date for e in v.events]
        + [e.label for e in v.events]
        + [c.label for c in v.calculations]
        + [str(p) for s in v.sources for p in (s.page_start, s.page_end) if p is not None]
    )
    label_numbers = {t.replace(",", "").replace("−", "-").lstrip("+") for t in _NUM.findall(labels)}
    for token in _NUM.findall(v.summary):
        raw = token.replace(",", "").replace("−", "-").lstrip("+")
        if raw in label_numbers:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        assert any(calc.close(abs(float(raw)), abs(x), decimals) for x in shown), f"summary {token} in {v.summary!r}"
    ids = {s.source_id for s in v.sources}
    refs = [c for r in v.rows for c in r.cells.values() if c] + [t.cell for t in v.tiles if t.cell]
    refs += [i for c in v.calculations for i in c.inputs] + [e.cell for e in v.events if e.cell]
    assert {r.source_id for r in refs} <= ids


def all_specs():
    """Every combination worth trying on the test datasets."""
    for ds in DS.values():
        rows = [r.key for r in ds.rows if r.type != "section"]
        cols = [c.key for c in series_columns(ds)]
        periods = list(
            dict.fromkeys(p.label for p in [*(c.period for c in ds.columns), *(r.period for r in ds.rows)] if p)
        )
        series_options = [[k] for k in rows + cols] + [rows[:2], rows[:3], cols[:2], cols[:3], rows[1:4], cols[1:3]]
        for kind, series in itertools.product(VISUAL_KINDS, series_options):
            if not series and kind != "timeline":
                continue
            base = {"kind": kind, "datasets": [ds.id], "series": [ref(ds.id, k) for k in series]}
            variants = [base, {**base, "periods": periods[-1:]}, {**base, "periods": periods[-2:]}]
            variants.append({**base, "categories": [ref(ds.id, k) for k in rows[:3]]})
            variants.append({**base, "categories": [ref(ds.id, k) for k in cols[:2]]})
            variants.append({**base, "highlight": periods[-1:]})
            variants.append({**base, "title": "Revenue in 2099 was huge"})  # a number not in the data: replaced
            if series:
                calcs = [{"op": op, "series": base["series"][0]} for op in CALC_OPS if op != "ratio"]
                variants += [{**base, "calculations": [c]} for c in calcs]
            if len(series) >= 2:
                variants.append(
                    {**base, "calculations": [{"op": "ratio", "series": base["series"][0], "other": base["series"][1]}]}
                )
            for language in ("en", "hi"):
                for v in variants:
                    yield {**v, "language": language}
    yield {
        "kind": "line",
        "datasets": [Q23, Q24],
        "series": [f"{Q23}:revenue", f"{Q24}:revenue", f"{Q23}:ebitda", f"{Q24}:ebitda"],
    }
    yield {"kind": "timeline", "datasets": ["ds_dates"]}
    cf = DS["cash_flow"]
    yield {
        "kind": "waterfall",
        "datasets": [CF],
        "series": [f"{CF}:fy24"],
        "categories": [
            ref(CF, k)
            for k in (
                "cash_and_cash_equivalents_at_the_beginning_of",
                "net_cash_generated_from_operating_activities",
                "net_cash_used_in_investing_activities",
                "net_cash_used_in_financing_activities",
                "cash_and_cash_equivalents_at_the_end_of_the_year",
            )
            if cf.row(k)
        ],
    }


def test_every_buildable_spec_is_grounded():
    built: Counter[str] = Counter()
    rejected = 0
    for raw in all_specs():
        try:
            v = build(**raw)
        except SpecError:
            rejected += 1
            continue
        assert check_grounding(v) == [], (raw, check_grounding(v))
        assert_grounded(v)
        assert "2099" not in v.title
        built[v.kind] += 1
    assert set(built) == set(VISUAL_KINDS), built  # every kind was exercised
    assert sum(built.values()) > 500, built
    assert rejected > 0


# ------------------------------------------------------------------ the check catches tampering


def tampered(v: Visual, **changes) -> list[str]:
    return check_grounding(v.model_copy(update=changes))


def test_tampering_is_detected():
    v = build(
        kind="line",
        datasets=[Q24],
        series=[f"{Q24}:revenue"],
        calculations=[{"op": "growth", "series": f"{Q24}:revenue"}],
    )
    assert check_grounding(v) == []
    row = v.rows[0]
    assert tampered(v, rows=[row.model_copy(update={"values": {"revenue": 9999.0}}), *v.rows[1:]])
    ref_ = row.cells["revenue"]
    assert ref_ is not None
    bad_cell = ref_.model_copy(update={"text": "1,999"})
    assert tampered(v, rows=[row.model_copy(update={"cells": {"revenue": bad_cell}}), *v.rows[1:]])
    assert tampered(v, rows=[row.model_copy(update={"cells": {"revenue": None}}), *v.rows[1:]])
    assert tampered(v, summary=v.summary + " Revenue will reach ₹9,000 crore.")
    c = v.calculations[0]
    assert tampered(v, calculations=[c.model_copy(update={"value": c.value + 1})])
    assert tampered(v, sources=[])
    assert tampered(v, title="Revenue 4,321 next year")


def test_tile_deltas_must_be_listed_calculations():
    v = build(kind="kpi", datasets=[H], series=[f"{H}:revenue_from_operations", f"{H}:ebitda_margin"])
    assert check_grounding(v) == []
    assert tampered(v, calculations=[])
    tile = v.tiles[0]
    assert tampered(v, tiles=[tile.model_copy(update={"value": 7000.0}), *v.tiles[1:]])


# ------------------------------------------------------------------ what each kind looks like


def test_line_over_merged_quarterly_tables():
    v = build(kind="line", datasets=[Q23, Q24], series=[f"{Q23}:revenue", f"{Q24}:revenue"])
    assert [r.x for r in v.rows] == [
        "Q1 FY23",
        "Q2 FY23",
        "Q3 FY23",
        "Q4 FY23",
        "Q1 FY24",
        "Q2 FY24",
        "Q3 FY24",
        "Q4 FY24",
    ]
    assert (v.x.key, v.x.label, v.x.type, v.unit.label) == ("quarter", "Quarter", "period", "₹ crore")  # type: ignore[union-attr]
    assert v.summary == "Revenue went from ₹1,512 crore in Q1 FY23 to ₹1,933 crore in Q4 FY24 (+27.8%)."
    assert [c.op for c in v.calculations] == ["growth"]
    assert [s.source_id for s in v.sources] == ["S1", "S2"]
    first = v.rows[0].cells["revenue"]
    assert first is not None and (first.text, first.row, first.col, first.page) == ("1,512", 1, 1, 20)


def test_kpi_tiles_with_deltas():
    v = build(
        kind="kpi",
        datasets=[H],
        series=[f"{H}:revenue_from_operations", f"{H}:ebitda_margin", f"{H}:net_debt_to_ebitda"],
    )
    assert v.x is None and v.rows == [] and len(v.tiles) == 3
    rev, margin, nd = v.tiles
    assert (rev.label, rev.value, rev.delta.kind, rev.delta.value) == (
        "Revenue from operations (FY24)",
        7365,
        "pct",
        13.6,
    )  # type: ignore[union-attr]
    assert (margin.delta.kind, margin.delta.value) == ("pp", 1.2)  # type: ignore[union-attr]
    assert (nd.delta.kind, nd.delta.value) == ("abs", -0.39)  # type: ignore[union-attr]
    assert v.summary.startswith("Revenue from operations (FY24): ₹7,365 crore (+13.6%)")


def test_donut_shares_are_calculations():
    v = build(kind="donut", datasets=[SEG], series=[f"{SEG}:revenue_fy24"])
    assert [r.x for r in v.rows] == ["Specialty Chemicals", "Engineered Plastics", "Digital Services"]
    shares = {c.label: c.value for c in v.calculations}
    assert shares == {
        "Specialty Chemicals share of Revenue FY24": 48.4,
        "Engineered Plastics share of Revenue FY24": 31.3,
        "Digital Services share of Revenue FY24": 20.3,
    }
    assert all(c.inputs[1].text == "7,365" for c in v.calculations)  # of the table's own total


def test_waterfall_reconciles_and_uses_magnitudes():
    cf = DS["cash_flow"]
    keys = [r.key for r in cf.rows]
    begin = next(k for k in keys if k.startswith("cash_and_cash_equivalents_at_the_beginning"))
    end = next(k for k in keys if k.startswith("cash_and_cash_equivalents_at_the_end"))
    order = [
        begin,
        "net_cash_generated_from_operating_activities",
        "net_cash_used_in_investing_activities",
        "net_cash_used_in_financing_activities",
        end,
    ]
    v = build(kind="waterfall", datasets=[CF], series=[f"{CF}:fy24"], categories=[ref(CF, k) for k in order])
    assert [s.key for s in v.series] == ["total", "increase", "decrease"]
    assert [{k: x for k, x in r.values.items() if x is not None} for r in v.rows] == [
        {"total": 486},
        {"increase": 1284},
        {"decrease": 712},
        {"decrease": 446},
        {"total": 612},
    ]
    assert [r.kind for r in v.rows] == ["total", "delta", "delta", "delta", "total"]
    assert v.rows[2].cells["decrease"].text == "(712)"  # type: ignore[union-attr]
    (bridge,) = v.calculations
    assert bridge.formula_text == "486 + 1,284 − 712 − 446 = 612"
    with pytest.raises(SpecError, match="don't take"):
        build(kind="waterfall", datasets=[CF], series=[f"{CF}:fy24"], categories=[ref(CF, k) for k in order[:-1]])


def test_waterfall_subtotals_are_totals():
    rows = [
        ["Particulars", "FY24"],
        ["Cash at the beginning of the year", "486"],
        ["Net cash from operating activities", "1,284"],
        ["Net cash used in investing activities", "(712)"],
        ["Subtotal before financing", "1,058"],
        ["Net cash used in financing activities", "(446)"],
        ["Cash at the end of the year", "612"],
    ]
    ds = typed(rows, dataset_id="ds_bridge", before=["(₹ crore)"])
    keys = [r.key for r in ds.rows]
    spec = VisualSpec(
        kind="waterfall",
        datasets=["ds_bridge"],
        series=["ds_bridge:fy24"],
        categories=[ref("ds_bridge", k) for k in keys],
    )
    v = build_visual(
        resolve(spec, {"ds_bridge": ds}), visual_id="v", project_id="p", chat_id=None, filenames={}, now=NOW
    )
    assert [(r.x, r.kind) for r in v.rows] == [
        ("Cash at the beginning of the year", "total"),
        ("Net cash from operating activities", "delta"),
        ("Net cash used in investing activities", "delta"),
        ("Subtotal before financing", "total"),
        ("Net cash used in financing activities", "delta"),
        ("Cash at the end of the year", "total"),
    ]
    assert v.rows[3].values["total"] == 1058
    assert v.calculations[0].formula_text == "486 + 1,284 − 712 − 446 = 612"  # the subtotal is shown, not added
    assert check_grounding(v) == []


def test_donut_of_printed_percentages_keeps_them():
    rows = [["Category", "Share of equity capital"], ["Promoters", "54.8%"], ["FPIs", "17.3%"], ["Public", "27.9%"]]
    ds = typed(rows, dataset_id="ds_holding")
    spec = VisualSpec(kind="donut", datasets=["ds_holding"], series=["ds_holding:share_of_equity_capital"])
    v = build_visual(
        resolve(spec, {"ds_holding": ds}), visual_id="v", project_id="p", chat_id=None, filenames={}, now=NOW
    )
    assert [r.values["share_of_equity_capital"] for r in v.rows] == [54.8, 17.3, 27.9]  # 54.8 means 54.8%
    assert v.unit is not None and v.unit.kind == "percent" and v.calculations == []


def test_comparison_of_two_periods_and_hindi_summary():
    v = build(kind="comparison", datasets=[H], series=[f"{H}:ebitda", f"{H}:ebitda_margin"], language="hi")
    # the sides are the series, the metrics the rows, each metric's change the tile with the row's label
    assert v.x is None
    assert [(s.key, s.label, s.unit) for s in v.series] == [("fy23", "FY23", None), ("fy24", "FY24", None)]
    assert [(r.x, r.values) for r in v.rows] == [
        ("EBITDA", {"fy23": 1283, "fy24": 1545}),
        ("EBITDA margin", {"fy23": 19.8, "fy24": 21.0}),
    ]
    assert [t.label for t in v.tiles] == [r.x for r in v.rows]
    assert [(t.value, t.delta.kind, t.delta.value) for t in v.tiles] == [(1545, "pct", 20.4), (21.0, "pp", 1.2)]  # type: ignore[union-attr]
    assert v.highlight is None
    assert "बनाम" in v.summary and "करोड़" in v.summary
    assert v.language == "hi" and "पृ." in (v.subtitle or "")


def test_comparison_of_two_documents():
    other = typed(
        [["Metric", "FY24", "FY23"], ["Revenue from operations (₹ crore)", "4,986", "4,437"]],
        dataset_id="ds_zephyra",
        document_id="doc_z",
        heading=("Zephyra at a glance",),
    )
    pool = {**BY_ID, "ds_zephyra": other}
    names = {**FILENAMES, "doc_z": "zephyra_investor_deck_q4fy24.pptx"}
    spec = VisualSpec(
        kind="comparison",
        datasets=[H, "ds_zephyra"],
        series=[f"{H}:revenue_from_operations", "ds_zephyra:revenue_from_operations"],
        periods=["FY24"],
    )
    v = build_visual(
        resolve(spec, pool, filenames=names), visual_id="v", project_id="p", chat_id=None, filenames=names, now=NOW
    )
    assert [s.label for s in v.series] == ["Valmora annual report fy24", "Zephyra investor deck q4fy24"]
    assert [s.better for s in v.series] == ["up", "up"]
    (row,) = v.rows
    assert row.x == "Revenue from operations (FY24)" and list(row.values.values()) == [7365, 4986]
    (tile,) = v.tiles
    assert tile.label == row.x and tile.value == 4986
    assert tile.delta is not None and (tile.delta.kind, tile.delta.value) == ("abs", -2379)
    assert tile.delta.calculation.formula_text == "4,986 − 7,365 = −₹2,379 crore"
    assert check_grounding(v) == []


def test_kpi_lists_a_series_per_tile_with_its_direction():
    v = build(
        kind="kpi",
        datasets=[H],
        series=[f"{H}:revenue_from_operations", f"{H}:net_debt_to_ebitda", f"{H}:number_of_employees"],
    )
    assert [(s.label, s.better) for s in v.series] == [
        ("Revenue from operations (FY24)", "up"),
        ("Net debt to EBITDA (FY24)", "down"),
        ("Number of employees (FY24)", None),
    ]
    assert [s.label for s in v.series] == [t.label for t in v.tiles] and v.rows == []


def test_better_direction():
    from app.services.canvas.builder import better_direction

    up = ["Revenue from operations", "EBITDA margin", "Profit after tax", "Return on capital employed", "राजस्व"]
    down = [
        "Net debt to EBITDA",
        "Cost of materials consumed",
        "Finance costs",
        "Voluntary attrition rate",
        "Borrowings",
    ]
    unclear = ["Total assets", "Number of employees", "Net cash used in investing activities", "Inventories"]
    assert [better_direction(m) for m in up] == ["up"] * len(up)
    assert [better_direction(m) for m in down] == ["down"] * len(down)
    assert [better_direction(m) for m in unclear] == [None] * len(unclear)


def test_timeline_events_in_date_order():
    v = build(kind="timeline", datasets=["ds_dates"])
    assert [e.date for e in v.events] == ["2024-09-20", "2024-11-30", "2025-01-06"]
    assert v.events[0].label == "ऑनलाइन आवेदन प्रारंभ" and v.events[0].cell.text == "20 सितंबर 2024"  # type: ignore[union-attr]
    assert v.sources[0].page_start is None and v.sources[0].filename == "suryodaya_notice.docx"


def test_highlight_note_and_titles():
    v = build(kind="bar", datasets=[Q24], series=[f"{Q24}:revenue"], highlight=["Q4 FY24"], title="Quarterly revenue")
    assert v.highlight is not None and v.highlight.x == ["Q4 FY24"] and v.highlight.note == "Highest"
    assert v.title == "Quarterly revenue"
    v = build(kind="bar", datasets=[Q24], series=[f"{Q24}:revenue"], title="Revenue up 50% in FY24")
    assert v.title == "Revenue, Q1 FY24–Q4 FY24"  # the model's number isn't in the data: replaced


def test_sources_reuse_the_turns_ids():
    existing = [
        Citation(
            source_id="S1",
            document_id="doc_1",
            filename="r.pdf",
            page_start=7,
            page_end=7,
            chunk_id="doc_1:v1:0099",
            snippet="…",
        ),
        Citation(
            source_id="S2",
            document_id="doc_1",
            filename="r.pdf",
            page_start=19,
            page_end=19,
            chunk_id=DS["q_fy24"].chunk_id,
            snippet="…",
        ),
    ]
    v = build(sources=existing, kind="line", datasets=[Q23, Q24], series=[f"{Q23}:revenue", f"{Q24}:revenue"])
    by_page = {c.page_start: c.source_id for c in v.sources}
    assert by_page == {19: "S2", 20: "S3"}  # the turn's [S2] is the FY24 table; the FY23 table gets the next id
    assert {r.cells["revenue"].source_id for r in v.rows if r.x.endswith("FY24")} == {"S2"}  # type: ignore[union-attr]


def test_requested_calculations_appear_in_the_summary():
    v = build(
        kind="bar",
        datasets=[H],
        series=[f"{H}:revenue_from_operations"],
        calculations=[{"op": "diff", "series": f"{H}:revenue_from_operations", "from": "FY23", "to": "FY24"}],
    )
    assert any(c.op == "diff" and c.value == 883 for c in v.calculations)
    assert "+₹883 crore" in v.summary


def test_contract_shape():
    v = build(kind="bar", datasets=[Q24], series=[f"{Q24}:revenue"]).model_dump(mode="json")
    assert set(v) == {
        "id", "chat_id", "project_id", "created_at", "updated_at", "kind", "title", "subtitle", "language", "summary",
        "unit", "x", "series", "rows", "tiles", "events", "highlight", "calculations", "sources", "pinned", "position",
    }  # fmt: skip
    assert set(v["unit"]) == {"kind", "currency", "scale", "label"}
    assert set(v["x"]) == {"key", "label", "type"}
    assert set(v["series"][0]) == {"key", "label", "unit", "calculated", "better"}  # better: additive
    assert set(v["rows"][0]) == {"x", "values", "cells", "kind"}  # kind: additive, waterfall only
    assert v["rows"][0]["kind"] is None
    assert set(v["rows"][0]["cells"]["revenue"]) == {
        "source_id",
        "document_id",
        "table_id",
        "page",
        "row",
        "col",
        "text",
    }
    assert set(v["sources"][0]) == {
        "source_id",
        "document_id",
        "filename",
        "page_start",
        "page_end",
        "chunk_id",
        "snippet",
        "section",
    }
    assert set(Calculation.model_fields) == {"label", "op", "value", "unit", "inputs", "formula_text"}
