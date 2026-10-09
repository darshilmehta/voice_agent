"""Typing tables into datasets and classifying them (services/canvas/datasets.py, chartability.py)."""

from __future__ import annotations

from app.domain.datasets import TableContext
from app.providers.ingestion import ParsedItem
from app.services.canvas.datasets import document_unit, table_contexts
from tests.fakes import make_parsed

from .canvas_helpers import (
    BALANCE_SHEET,
    CASH_FLOW,
    DATES_HI,
    DISTRICTS_HI,
    GLOSSARY,
    HIGHLIGHTS,
    QUARTERS_FY24,
    SEGMENTS,
    SPAN,
    report_datasets,
    table,
    typed,
    typed_document,
)


def values(ds, row):
    return {v.col: v.value for v in ds.values if v.row == row}


def units(ds, row):
    return {v.col: (v.unit.label if v.unit else None) for v in ds.values if v.row == row}


# ------------------------------------------------------------------ headers


def test_flagged_single_header_row():
    ds = typed(HIGHLIGHTS, before=["Amounts are in ₹ crore unless stated otherwise."])
    assert ds.header_rows == 1 and ds.label_column == 0
    assert [c.key for c in ds.columns] == ["metric", "fy24", "fy23", "change"]
    assert [c.type for c in ds.columns] == ["label", "period", "period", "change"]
    assert [c.period.label if c.period else None for c in ds.columns] == [None, "FY24", "FY23", None]


def test_multi_row_header_with_spanning_cells():
    rows = [
        ["Segment", ("Revenue", 1, 2), SPAN, ("EBITDA", 1, 2), SPAN],
        ["", "FY24", "FY23", "FY24", "FY23"],
        ["Chemicals", "3,568", "3,214", "862", "745"],
        ["Plastics", "2,303", "2,126", "346", "331"],
        ["Total", "5,871", "5,340", "1,208", "1,076"],
    ]
    ds = typed(rows, header_rows=2, before=["(₹ crore)"])
    assert ds.header_rows == 2
    cols = {c.key: c for c in ds.columns}
    assert list(cols) == ["segment", "revenue_fy24", "revenue_fy23", "ebitda_fy24", "ebitda_fy23"]
    assert cols["revenue_fy23"].header == ["Revenue", "FY23"]
    assert (cols["ebitda_fy24"].measure, cols["ebitda_fy24"].period.label) == ("EBITDA", "FY24")  # type: ignore[union-attr]
    assert values(ds, "chemicals") == {"revenue_fy24": 3568, "revenue_fy23": 3214, "ebitda_fy24": 862, "ebitda_fy23": 745}
    assert ds.row("total").parts == ["chemicals", "plastics"]  # type: ignore[union-attr]
    assert ds.chartability.period_order == ["FY23", "FY24"]


def test_flagged_header_extended_by_a_row_of_periods():
    rows = [
        ["Particulars", ("Revenue", 1, 2), SPAN],
        ["", "FY24", "FY23"],
        ["India", "4,000", "3,500"],
        ["Exports", "3,365", "2,982"],
    ]
    ds = typed(rows, header_rows=1)  # Docling flagged only the first row
    assert ds.header_rows == 2
    assert [c.label for c in ds.columns][1:] == ["Revenue FY24", "Revenue FY23"]


def test_unflagged_headers_found_above_the_first_number():
    rows = [
        ["", "Revenue", "EBITDA"],
        ["", "₹ crore", "₹ crore"],
        ["Chemicals", "3,568", "862"],
        ["Plastics", "2,303", "346"],
    ]
    ds = typed(rows, flag_headers=False)
    assert ds.header_rows == 2
    assert ds.columns[1].unit is not None and ds.columns[1].unit.label == "₹ crore"


def test_section_rows_are_not_headers():
    ds = typed(BALANCE_SHEET, before=["All amounts are in ₹ crore."])
    assert ds.header_rows == 1
    assert [(r.key, r.type) for r in ds.rows if r.type == "section"] == [
        ("non_current_assets", "section"),
        ("current_assets", "section"),
    ]
    assert ds.row("inventories").section == "Current assets"  # type: ignore[union-attr]
    assert ds.column("note").type == "note"  # type: ignore[union-attr]
    assert "note" not in {v.col for v in ds.values}


def test_text_table_without_numbers():
    ds = typed(GLOSSARY)
    assert ds.header_rows == 1 and ds.values == []
    assert ds.chartability.kind == "none"
    assert ds.row("scope_1_and_2") is not None  # "Scope 1 and 2" keeps its numbers (not a footnote)


def test_identifier_first_column_is_the_label():
    rows = [
        ["Version", "Effective date", "Change"],
        ["3.2", "1 April 2024", "Hotel limits raised"],
        ["3.1", "1 April 2023", "Per diem raised"],
    ]
    ds = typed(rows)
    assert ds.label_column == 0
    assert [c.type for c in ds.columns] == ["label", "date", "text"]
    assert ds.chartability.kind == "timeline"


# ------------------------------------------------------------------ units


def test_row_units_override_the_table_unit():
    ds = typed(HIGHLIGHTS, before=["Amounts are in ₹ crore unless stated otherwise."])
    assert ds.unit is not None and ds.unit.label == "₹ crore"
    assert units(ds, "revenue_from_operations")["fy24"] == "₹ crore"
    assert units(ds, "ebitda_margin")["fy24"] == "%"
    assert units(ds, "basic_earnings_per_share")["fy24"] == "₹"  # "(₹)" in the label, not crore
    assert units(ds, "net_debt_to_ebitda")["fy24"] == "x"
    emp = ds.row("number_of_employees")
    assert emp is not None and emp.unit is not None and emp.unit.kind == "count"
    # labels lose their unit (kept as the row's unit); keys are readable
    assert ds.row("revenue_from_operations").label == "Revenue from operations"  # type: ignore[union-attr]


def test_unit_from_caption_header_cell_and_note():
    rows = [["₹ crore", "FY24", "FY23"], ["Revenue", "4,986", "4,437"], ["EBITDA", "633", "528"]]
    assert typed(rows).unit.label == "₹ crore"  # type: ignore[union-attr]
    rows = [["Item", "FY24", "FY23"], ["Revenue", "4,986", "4,437"], ["EBITDA", "633", "528"]]
    assert typed(rows, caption="Profit and loss (₹ in crore)").unit.label == "₹ crore"  # type: ignore[union-attr]
    assert typed(rows, after=["Revenue in ₹ crore."]).unit.label == "₹ crore"  # type: ignore[union-attr]
    assert typed(rows, before=["Rs. in lakhs"]).unit.label == "₹ lakh"  # type: ignore[union-attr]
    assert typed(rows, heading=("Results (US$ million)",)).unit.label == "$ million"  # type: ignore[union-attr]
    assert typed(rows).unit is None


def test_document_unit_only_for_amount_rows():
    report = table(HIGHLIGHTS, index=0)
    note = table([["Matter", "31 Mar 2024", "31 Mar 2023"], ["Disputed tax demands", "38", "31"], ["Employees", "9", "8"]], index=1)
    contexts = {0: TableContext(before=["Amounts are in ₹ crore unless stated otherwise."])}
    assert document_unit([report, note], contexts).label == "₹ crore"  # type: ignore[union-attr]
    _, ds = typed_document([report, note], contexts)
    assert units(ds, "disputed_tax_demands") == {"31_mar_2024": "₹ crore", "31_mar_2023": "₹ crore"}
    assert units(ds, "employees") == {"31_mar_2024": None, "31_mar_2023": None}
    assert ds.warnings == ["unit ₹ crore assumed from the rest of the document"]


def test_column_units_from_values_and_headers():
    ds = typed(SEGMENTS, before=["Segment revenue and EBITDA are shown below (₹ crore)."])
    cols = {c.key: c for c in ds.columns}
    assert cols["revenue_fy24"].type == "currency" and cols["revenue_fy24"].unit.label == "₹ crore"  # type: ignore[union-attr]
    assert cols["ebitda_margin_fy24"].type == "percent"
    ds = typed(DISTRICTS_HI)
    cols = {c.index: c for c in ds.columns}
    assert cols[3].unit.label == "₹ lakh" and cols[3].label == "आवंटित बजट"  # type: ignore[union-attr]


def test_mixed_unit_words_in_a_column():
    rows = [
        ["Facility", "Installed capacity", "Commissioned"],
        ["Dahej Complex", "2,10,000 tpa", "2009"],
        ["Vapi Unit I", "90,000 tpa", "1996"],
        ["Bengaluru Centre", "1,300 seats", "2017"],
    ]
    ds = typed(rows)
    assert ds.row("vapi_unit_i") is not None  # "Unit I" is a name, not a footnote
    assert values(ds, "dahej_complex") == {"installed_capacity": 210000}
    assert units(ds, "bengaluru_centre") == {"installed_capacity": "seats"}
    assert ds.column("commissioned").type == "year"  # type: ignore[union-attr]


# ------------------------------------------------------------------ values, rows, totals


def test_negatives_nulls_and_footnotes():
    rows = [
        ["Instrument", "31 Mar 2024", "31 Mar 2023"],
        ["Term loans", "752", "906"],
        ["Commercial paper", "-", "45"],
        ["Other¹", "(12)", "n.a."],
    ]
    ds = typed(rows, before=["(₹ crore)"])
    assert values(ds, "commercial_paper") == {"31_mar_2024": None, "31_mar_2023": 45}
    assert values(ds, "other") == {"31_mar_2024": -12, "31_mar_2023": None}
    assert ds.row("other").footnote == "1"  # type: ignore[union-attr]
    v = ds.value("other", "31_mar_2024")
    assert v is not None and (v.text, v.cell_row, v.cell_col) == ("(12)", 3, 1)


def test_footnote_numbers_after_labels():
    ds = typed(SEGMENTS, after=["1. Digital Services revenue is net of inter-segment sales."])
    assert ds.row("digital_services").footnote == "1"  # type: ignore[union-attr]
    plain = typed([["Plant", "Output"], ["Plant 2", "40"], ["Unit 1", "30"]])
    assert [r.label for r in plain.rows] == ["Plant 2", "Unit 1"]


def test_totals_are_checked_arithmetically_and_nested():
    ds = typed(BALANCE_SHEET, before=["All amounts are in ₹ crore."])
    assert ds.row("total_non_current_assets").parts == ["property_plant_and_equipment", "capital_work_in_progress"]  # type: ignore[union-attr]
    assert ds.row("total_assets").parts == ["total_non_current_assets", "total_current_assets"]  # type: ignore[union-attr]
    cf = typed(CASH_FLOW, before=["All amounts are in ₹ crore."])
    assert cf.row("net_increase_in_cash_and_cash_equivalents").parts == [  # type: ignore[union-attr]
        "net_cash_generated_from_operating_activities",
        "net_cash_used_in_investing_activities",
        "net_cash_used_in_financing_activities",
    ]
    q = typed(QUARTERS_FY24, before=["(₹ crore)"])
    full = q.row("full_year_fy24")
    assert full is not None and full.type == "total" and full.parts == ["q1_fy24", "q2_fy24", "q3_fy24", "q4_fy24"]


def test_small_counts_must_add_up_exactly():
    rows = [["Committee", "Members"], ["Audit", "4"], ["Risk", "3"], ["CSR", "4"], ["Board", "6"]]
    ds = typed(rows)
    assert ds.row("board").type == "data" and ds.row("board").parts == []  # type: ignore[union-attr]


def test_rounding_tolerance_for_printed_totals():
    rows = [["Segment", "Revenue"], ["North", "10.4"], ["South", "20.4"], ["East", "30.4"], ["Total", "61.3"]]
    ds = typed(rows)  # 61.2 printed as 61.3 after rounding the unrounded parts
    assert ds.row("total").parts == ["north", "south", "east"]  # type: ignore[union-attr]
    rows[-1][1] = "62.0"
    assert typed(rows).row("total").parts == []  # type: ignore[union-attr]


def test_hindi_table():
    ds = typed(DISTRICTS_HI)
    assert ds.rows[-1].type == "total" and ds.rows[-1].label == "कुल"
    assert len(ds.rows[-1].parts) == 3
    assert [r.key for r in ds.rows] == ["r1", "r2", "r3", "r4"]  # no Latin letters to make keys from


# ------------------------------------------------------------------ chartability


def test_chartability_of_the_report():
    ds = report_datasets()
    kinds = {name: d.chartability.kind for name, d in ds.items()}
    assert kinds == {
        "highlights": "kpi",
        "segments": "composition",
        "q_fy24": "time_series",
        "q_fy23": "time_series",
        "cash_flow": "composition",
        "balance_sheet": "composition",
        "glossary": "none",
    }
    q = ds["q_fy24"].chartability
    assert (q.period_axis, q.granularity, q.period_order) == ("rows", "quarter", ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"])
    h = ds["highlights"].chartability
    assert h.period_axis == "columns" and h.period_order == ["FY23", "FY24"]  # printed FY24 first
    assert {o.kind for o in h.options} >= {"kpi", "time_series"}
    assert ds["balance_sheet"].chartability.period_order == ["31 Mar 2023", "31 Mar 2024"]


def test_shares_adding_to_100_are_a_composition():
    rows = [["Category", "Share of equity capital"], ["Promoters", "54.8%"], ["FPIs", "17.3%"], ["Mutual funds", "14.6%"], ["Public", "13.3%"]]
    assert typed(rows).chartability.kind == "composition"
    rows[1][1] = "60.0%"
    assert typed(rows).chartability.kind == "categorical"


def test_dates_make_a_timeline():
    ds = typed(DATES_HI)
    assert ds.chartability.kind == "timeline"
    assert ds.columns[1].type == "date"
    assert [t.text for t in ds.texts if t.col == ds.columns[1].key][0] == "20 सितंबर 2024"


def test_confidence_grows_with_periods():
    two = typed([["Metric", "FY24", "FY23"], ["Revenue", "10", "8"], ["Cost", "6", "5"]])
    five = typed([["Year", "Revenue"], ["FY20", "5"], ["FY21", "6"], ["FY22", "7"], ["FY23", "8"], ["FY24", "10"]])
    conf = {o.kind: o.confidence for o in two.chartability.options}
    assert conf["time_series"] == 0.5
    assert five.chartability.kind == "time_series" and five.chartability.confidence >= 0.85


def test_table_contexts_from_the_parsed_document():
    items = [
        ParsedItem(ref="#/texts/0", label="section_header", text="Segment results", page=18, heading_path=["Segment results"]),
        ParsedItem(ref="#/texts/1", label="text", text="Shown below (₹ crore).", page=18, heading_path=["Segment results"]),
        ParsedItem(ref="#/tables/0", label="table", text="|…|", page=18, heading_path=["Segment results"], table_index=0),
        ParsedItem(ref="#/texts/2", label="text", text="1. Net of inter-segment sales.", page=18, heading_path=[]),
        ParsedItem(ref="#/texts/3", label="section_header", text="Next", page=19, heading_path=["Next"]),
    ]
    contexts = table_contexts(make_parsed(items=items))
    assert contexts == {0: TableContext(before=["Shown below (₹ crore)."], after=["1. Net of inter-segment sales."])}


def test_cell_positions_point_into_the_grid():
    ds = typed(HIGHLIGHTS)
    v = ds.value("ebitda", "fy23")
    assert v is not None and (v.cell_row, v.cell_col, v.text) == (2, 2, "1,283")
