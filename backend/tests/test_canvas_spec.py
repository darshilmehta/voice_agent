# ruff: noqa: RUF001  (documents print en dashes and minus signs: the tests use them on purpose)
"""VisualSpec validation against real datasets, and the calculator (services/canvas/spec.py, calculator.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.canvas import CellRef, Unit
from app.domain.datasets import PeriodInfo
from app.services.canvas import calculator as calc
from app.services.canvas.calculator import CalculationError, Operand, format_number, format_value
from app.services.canvas.parsing import PERCENT, RATIO, currency_unit
from app.services.canvas.spec import SpecError, VisualSpec, resolve

from .canvas_helpers import report_datasets

DS = report_datasets()
BY_ID = {d.id: d for d in DS.values()}
H, SEG, Q24, Q23, CF, BS = "ds_highlights", "ds_segments", "ds_q_fy24", "ds_q_fy23", "ds_cash_flow", "ds_balance_sheet"


def spec(**kw) -> VisualSpec:
    return VisualSpec.model_validate(kw)


def problems(**kw) -> list[str]:
    with pytest.raises(SpecError) as e:
        resolve(spec(**kw), BY_ID)
    return e.value.problems


# ------------------------------------------------------------------ resolving


def test_row_series_run_across_period_columns_chronologically():
    r = resolve(spec(kind="bar", datasets=[H], series=[f"{H}:revenue_from_operations"]), BY_ID)
    assert [i.label for i in r.x_items] == ["FY23", "FY24"]  # the table prints FY24 first
    assert r.x_type == "period" and r.x_key == "period"
    s = r.series[0]
    assert (s.label, s.unit.label) == ("Revenue from operations", "₹ crore")  # type: ignore[union-attr]
    assert {x: p.value.value for x, p in s.points.items()} == {"FY23": 6482, "FY24": 7365}


def test_column_series_run_down_period_rows_without_the_full_year():
    r = resolve(spec(kind="line", datasets=[Q24], series=[f"{Q24}:revenue", f"{Q24}:ebitda"]), BY_ID)
    assert [i.label for i in r.x_items] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    assert [s.key for s in r.series] == ["revenue", "ebitda"]


def test_same_named_series_of_continuing_tables_merge():
    r = resolve(
        spec(kind="line", datasets=[Q23, Q24], series=[f"{Q23}:revenue", f"{Q24}:revenue"]),
        BY_ID,
    )
    assert len(r.series) == 1 and len(r.series[0].points) == 8
    assert [i.label for i in r.x_items][:2] == ["Q1 FY23", "Q2 FY23"]


def test_measure_by_period_columns_pick_one_measure():
    r = resolve(
        spec(kind="stacked_bar", datasets=[SEG], series=[f"{SEG}:specialty_chemicals", f"{SEG}:engineered_plastics"]),
        BY_ID,
    )
    assert [i.label for i in r.x_items] == ["FY23", "FY24"]
    assert r.series[0].label == "Specialty Chemicals · Revenue"
    r = resolve(
        spec(
            kind="grouped_bar",
            datasets=[SEG],
            series=[f"{SEG}:specialty_chemicals", f"{SEG}:engineered_plastics"],
            categories=[f"{SEG}:ebitda_fy24", f"{SEG}:ebitda_fy23"],
        ),
        BY_ID,
    )
    assert [i.label for i in r.x_items] == ["FY23", "FY24"]
    assert r.series[0].points["FY24"].value.value == 862


def test_categories_pick_and_order_x_values():
    r = resolve(
        spec(
            kind="donut",
            datasets=[SEG],
            series=[f"{SEG}:revenue_fy24"],
            categories=[f"{SEG}:digital_services", f"{SEG}:specialty_chemicals"],
        ),
        BY_ID,
    )
    assert [i.label for i in r.x_items] == ["Digital Services", "Specialty Chemicals"]
    assert r.series[0].total is not None and r.series[0].total.value.value == 7365


def test_period_filter_and_x_reference():
    r = resolve(spec(kind="kpi", datasets=[H], series=[f"{H}:ebitda"], periods=["FY24"]), BY_ID)
    assert [i.label for i in r.x_items] == ["FY24"]
    r = resolve(spec(kind="table", datasets=[BS], x=f"{BS}:rows", series=[f"{BS}:31_mar_2024"]), BY_ID)
    assert "Total assets" in [i.label for i in r.x_items]  # tables keep totals
    assert r.x_label == "Particulars"


def test_timeline_resolution():
    from .canvas_helpers import DATES_HI, typed

    ds = typed(DATES_HI, dataset_id="ds_dates")
    r = resolve(spec(kind="timeline", datasets=["ds_dates"]), {"ds_dates": ds})
    assert len(r.timeline) == 3 and r.timeline[0].date_col == ds.columns[1].key


# ------------------------------------------------------------------ rejecting


def test_rejects_unknown_datasets_and_invented_rows():
    assert problems(kind="bar", datasets=["ds_nope"], series=["ds_nope:revenue"]) == ["unknown dataset 'ds_nope'"]
    (p,) = problems(kind="bar", datasets=[H], series=[f"{H}:market_share"])
    assert "'market_share' is not a row or a measured column" in p
    (p,) = problems(kind="bar", datasets=[H], series=[f"{SEG}:total"])
    assert "not in the spec's datasets" in p
    (p,) = problems(kind="bar", datasets=[H], series=["revenue"])
    assert "is not a '<dataset>:<key>' reference" in p


def test_rejects_invented_periods_and_categories():
    (p,) = problems(kind="bar", datasets=[H], series=[f"{H}:ebitda"], periods=["FY22"])
    assert "period 'FY22' is not in the datasets" in p
    (p,) = problems(kind="bar", datasets=[H], series=[f"{H}:ebitda"], categories=[f"{H}:fy22"])
    assert "not on the x axis" in p
    (p,) = problems(kind="bar", datasets=[Q24], series=[f"{Q24}:revenue"], categories=[f"{Q24}:revenue"])
    assert "not on the x axis" in p  # a column is not an x value of a column series


def test_rejects_the_x_axis_contradicting_the_series():
    (p,) = problems(kind="bar", datasets=[H], x=f"{H}:rows", series=[f"{H}:ebitda"])
    assert "is not a measured column" in p
    (p,) = problems(kind="bar", datasets=[H], x=f"{H}:somewhere", series=[f"{H}:ebitda"])
    assert "is not an axis" in p


def test_mixed_units_only_where_the_renderer_draws_small_multiples():
    (p,) = problems(kind="grouped_bar", datasets=[Q24], series=[f"{Q24}:revenue", f"{Q24}:ebitda_margin"])
    assert "plots one unit" in p and "%" in p and "₹ crore" in p
    assert any(
        "plots one unit" in p
        for p in problems(kind="stacked_bar", datasets=[Q24], series=[f"{Q24}:revenue", f"{Q24}:ebitda_margin"])
    )
    for kind in ("line", "bar", "table"):  # each series keeps its own unit
        r = resolve(spec(kind=kind, datasets=[Q24], series=[f"{Q24}:revenue", f"{Q24}:ebitda_margin"]), BY_ID)
        assert [s.unit.label for s in r.series] == ["₹ crore", "%"]  # type: ignore[union-attr]


def test_kind_specific_rules():
    assert any("needs periods" in p for p in problems(kind="line", datasets=[SEG], series=[f"{SEG}:revenue_fy24"]))
    assert any("not periods" in p for p in problems(kind="donut", datasets=[Q24], series=[f"{Q24}:revenue"]))
    assert any(
        "negative" in p for p in problems(kind="stacked_bar", datasets=[CF], series=[f"{CF}:fy24", f"{CF}:fy23"])
    )
    assert any("at least 2 series" in p for p in problems(kind="grouped_bar", datasets=[H], series=[f"{H}:ebitda"]))
    assert any(
        "needs at least 2 x values" in p
        for p in problems(kind="line", datasets=[H], series=[f"{H}:ebitda"], periods=["FY24"])
    )


def test_size_limits():
    with pytest.raises(ValidationError):
        spec(kind="table", datasets=[H], series=[f"{H}:r{n}" for n in range(9)])
    with pytest.raises(ValidationError):
        spec(kind="bar", datasets=[H, SEG, Q24, CF], series=[f"{H}:ebitda"])
    with pytest.raises(ValidationError):
        spec(kind="bar", datasets=[H], series=[f"{H}:ebitda"], title="x" * 81)
    rows = [f"{H}:{r.key}" for r in DS["highlights"].rows][:5]
    (p,) = problems(kind="line", datasets=[H], series=rows)
    assert "at most 4 series" in p
    with pytest.raises(ValidationError):
        VisualSpec.model_validate({"kind": "bar", "datasets": [H], "series": [f"{H}:ebitda"], "colour": "red"})


def test_rejects_unresolvable_highlights_and_calculations():
    (p,) = problems(kind="bar", datasets=[H], series=[f"{H}:ebitda"], highlight=["FY21"])
    assert "highlight 'FY21'" in p
    (p,) = problems(
        kind="bar",
        datasets=[H],
        series=[f"{H}:ebitda"],
        calculations=[{"op": "growth", "series": f"{H}:revenue_from_operations"}],
    )
    assert "not one of the visual's series" in p
    (p,) = problems(
        kind="bar", datasets=[H], series=[f"{H}:ebitda"], calculations=[{"op": "ratio", "series": f"{H}:ebitda"}]
    )
    assert "needs 'other'" in p
    (p,) = problems(
        kind="bar",
        datasets=[H],
        series=[f"{H}:ebitda"],
        calculations=[{"op": "growth", "series": f"{H}:ebitda", "from": "FY20"}],
    )
    assert "'FY20' is not an x value" in p


# ------------------------------------------------------------------ calculator


def ref(text: str, row: int = 1, col: int = 1) -> CellRef:
    return CellRef(source_id="S1", document_id="doc", table_id="tbl", page=3, row=row, col=col, text=text)


CRORE = currency_unit("INR", "crore")


def op(value: float, text: str, unit: Unit | None = CRORE, label: str = "", period: str | None = None) -> Operand:
    p = None
    if period:
        from app.services.canvas.parsing import parse_period

        pp = parse_period(period)
        assert pp is not None
        p = PeriodInfo(label=pp.label, kind=pp.kind, end=pp.end, months=pp.months)
    decimals = len(text.split(".")[1].rstrip("%x")) if "." in text else 0
    return Operand(value=value, unit=unit, cell=ref(text), decimals=decimals, label=label or (period or ""), period=p)


def test_growth_matches_the_documents_own_change_column():
    for new, old, printed in [(7365, 6482, 13.6), (1545, 1283, 20.4), (871, 664, 31.2), (9842, 9310, 5.7)]:
        c = calc.growth(op(old, f"{old:,}", period="FY23"), op(new, f"{new:,}", period="FY24"), series="x")
        assert c.value == printed and c.unit == PERCENT and c.op == "growth"
    c = calc.growth(op(6482, "6,482", period="FY23"), op(7365, "7,365", period="FY24"), series="Revenue")
    assert c.formula_text == "(7,365 − 6,482) ÷ 6,482 × 100 = 13.6%"
    assert c.label == "Revenue growth, FY23 to FY24"
    assert [i.text for i in c.inputs] == ["6,482", "7,365"]
    hi = calc.growth(op(6482, "6,482", period="FY23"), op(7365, "7,365", period="FY24"), series="राजस्व", language="hi")
    assert hi.label == "राजस्व वृद्धि, FY23 से FY24"


def test_cagr_uses_the_periods():
    c = calc.cagr(op(4000, "4,000", period="FY21"), op(7365, "7,365", period="FY24"), series="Revenue")
    assert c.value == pytest.approx(round(((7365 / 4000) ** (1 / 3) - 1) * 100, 1))
    assert "^(1/3)" in c.formula_text
    with pytest.raises(CalculationError, match="a year apart"):
        calc.cagr(op(1742, "1,742", period="Q1 FY24"), op(1933, "1,933", period="Q4 FY24"), series="x")
    with pytest.raises(CalculationError, match="periods"):
        calc.cagr(op(1, "1"), op(2, "2"), series="x")


def test_diff_in_percentage_points_and_units():
    c = calc.diff(
        op(19.8, "19.8%", PERCENT, period="FY23"), op(21.0, "21.0%", PERCENT, period="FY24"), series="EBITDA margin"
    )
    assert (c.value, c.unit.label, c.formula_text) == (1.2, "pp", "21.0 − 19.8 = +1.2 pp")  # type: ignore[union-attr]
    c = calc.diff(op(0.93, "0.93x", RATIO), op(0.54, "0.54x", RATIO), series="ND/EBITDA")
    assert c.value == -0.39 and c.unit == RATIO


def test_ratio_share_and_sum():
    c = calc.ratio(op(1545, "1,545"), op(7365, "7,365"), series="EBITDA", other="Revenue", at="FY24")
    assert (c.value, c.unit) == (0.21, RATIO)
    parts = [
        op(3568, "3,568", label="Chemicals"),
        op(2303, "2,303", label="Plastics"),
        op(1494, "1,494", label="Digital"),
    ]
    s = calc.share(parts[0], parts, total_label="Revenue")
    assert s.value == 48.4 and len(s.inputs) == 4  # the part, then the whole it is a share of
    assert s.formula_text == "3,568 ÷ (3,568 + 2,303 + 1,494) × 100 = 48.4%"
    s = calc.share(parts[0], op(7365, "7,365", label="Total"), total_label="Revenue")
    assert s.value == 48.4 and s.formula_text == "3,568 ÷ 7,365 × 100 = 48.4%"
    t = calc.total([op(486, "486"), op(1284, "1,284"), op(-712, "(712)"), op(-446, "(446)")], series="Cash")
    assert t.value == 612 and t.formula_text == "486 + 1,284 − 712 − 446 = 612"


def test_calculations_refuse_bad_inputs():
    with pytest.raises(CalculationError, match="different units"):
        calc.growth(op(10, "10", PERCENT), op(12, "12", CRORE), series="x")
    with pytest.raises(CalculationError, match="positive"):
        calc.growth(op(0, "0"), op(12, "12"), series="x")
    with pytest.raises(CalculationError, match="positive"):
        calc.growth(op(-5, "(5)"), op(12, "12"), series="x")
    with pytest.raises(CalculationError, match="zero"):
        calc.ratio(op(1, "1"), op(0, "0"), series="a", other="b")
    with pytest.raises(CalculationError, match="at least two"):
        calc.total([op(1, "1")], series="x")


def test_recompute_matches_every_operation():
    a, b = op(6482, "6,482", period="FY21"), op(7365, "7,365", period="FY24")
    for c in [
        calc.growth(a, b, series="x"),
        calc.cagr(a, b, series="x"),
        calc.diff(a, b, series="x"),
        calc.ratio(a, b, series="x", other="y"),
        calc.share(a, [a, b], total_label="t"),
        calc.total([a, b], series="x"),
    ]:
        values = [float(i.text.replace(",", "")) for i in c.inputs]
        assert calc.close(calc.recompute(c, values), c.value, 2), c


def test_number_formatting():
    assert format_number(421000, 0) == "4,21,000"
    assert format_number(421000, 0, indian=False) == "421,000"
    assert format_number(-712, 0) == "−712"
    assert format_number(69.684, 2) == "69.68"
    assert format_value(7365, CRORE, 0) == "₹7,365 crore"
    assert format_value(7365, CRORE, 0, "hi") == "₹7,365 करोड़"
    assert format_value(-712, CRORE, 0) == "−₹712 crore"
    assert format_value(21.0, PERCENT, 1) == "21.0%"
    assert format_value(0.54, RATIO, 2) == "0.54x"
    assert format_value(1.2, calc.PP, 1) == "1.2 pp"
    assert format_value(1500000, currency_unit("USD", None), 0) == "$1,500,000"
