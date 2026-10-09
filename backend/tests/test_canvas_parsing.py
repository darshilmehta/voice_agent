# ruff: noqa: RUF001  (documents print en dashes and minus signs: the tests use them on purpose)
"""Numbers, units and periods as documents print them (services/canvas/parsing.py)."""

from __future__ import annotations

from datetime import date

import pytest

from app.services.canvas.parsing import (
    NULL,
    Quantity,
    detect_amount_unit,
    detect_unit,
    find_period,
    is_null,
    parse_period,
    parse_quantity,
    split_footnote,
)


def q(text: str) -> Quantity:
    value = parse_quantity(text)
    assert isinstance(value, Quantity), f"{text!r} → {value!r}"
    return value


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("7,365", 7365),
        ("4,21,000", 421000),  # Indian grouping
        ("1,00,000", 100000),
        ("4,86,12,500", 48612500),
        ("2,10,000 tpa", 210000),
        ("421,000", 421000),  # Western grouping
        ("1,234,567.89", 1234567.89),
        ("12,34,567.5", 1234567.5),
        ("0.54", 0.54),
        (".5", 0.5),
        ("16", 16),
        ("(712)", -712),  # accounting negative
        ("(1,234.5)", -1234.5),
        ("-446", -446),
        ("−1,234", -1234),  # unicode minus
        ("–12", -12),  # en dash used as a minus
        ("+13.6%", 13.6),
        ("(12.5%)", -12.5),
        ("1,234*", 1234),  # footnote markers
        ("12.5%¹", 12.5),
        ("21.0% 1", 21.0),  # Docling's "<super>1</super>" after a unit
        ("1,284†", 1284),
        ("98[a]", 98),
        ("1,234.", 1234),
        ("१२३", 123),  # Devanagari digits
        ("२,४००", 2400),
    ],
)
def test_numbers(text, value):
    assert q(text).value == pytest.approx(value)


@pytest.mark.parametrize(
    ("text", "kind", "currency", "scale"),
    [
        ("₹ 4,210 crore", "currency", "INR", "crore"),
        ("₹4,210 cr", "currency", "INR", "crore"),
        ("Rs. 1.2 lakh", "currency", "INR", "lakh"),
        ("Rs 500", "currency", "INR", None),
        ("INR 4,210 crore", "currency", "INR", "crore"),
        ("₹ 1,200 करोड़", "currency", "INR", "crore"),
        ("₹ (1,234)", "currency", "INR", None),
        ("$3.4m", "currency", "USD", "million"),
        ("US$ 12 mn", "currency", "USD", "million"),
        ("USD 85", "currency", "USD", None),
        ("EUR 70", "currency", "EUR", None),
        ("€ 2.5 bn", "currency", "EUR", "billion"),
        ("£40", "currency", "GBP", None),
        ("21.0%", "percent", None, None),
        ("16.1 per cent", "percent", None, None),
        ("80 प्रतिशत", "percent", None, None),
        ("0.54x", "ratio", None, None),
        ("1.07 times", "ratio", None, None),
        ("+118 bps", "bps", None, None),
        ("+2.7 pts", "pp", None, None),
        ("1.2 pp", "pp", None, None),
        ("12.5 crore", "number", None, "crore"),  # shares, not rupees: the table decides
        ("4.8 lakh tpa", "number", None, "lakh"),
    ],
)
def test_units_printed_in_cells(text, kind, currency, scale):
    value = q(text)
    assert (value.kind, value.currency, value.scale) == (kind, currency, scale)


def test_unit_words_signs_and_decimals():
    assert q("1,450 seats").unit_word == "seats"
    assert q("5 meetings").unit_word == "meetings"
    assert q("500 MW").unit_word == "MW"
    assert q("7.85% p.a.").unit_word == "p.a."
    assert q("+13.6%").signed and not q("13.6%").signed
    assert not q("(712)").signed  # parentheses are a negative level, not a change
    assert q("15.00").decimals == 2 and q("7,365").decimals == 0
    assert q("12.5%¹").footnote == "1" and q("1,234*").footnote == "*"


@pytest.mark.parametrize("text", ["", "-", "–", "—", "--", "n.a.", "N.A.", "NA", "n/a", "Nil", "NM", "None", "…"])
def test_not_available_markers_are_null(text):
    assert parse_quantity(text) is NULL
    assert is_null(text)


@pytest.mark.parametrize(
    "text",
    [
        "FY24",
        "Q3 FY24",
        "2023-24",
        "Up to ₹ 25,000",
        "₹ 25,001 to ₹ 1,00,000",
        "10 per cent for dependent parents",
        "30 days before and 60 days after",
        "GHI/2024/00418377",
        "INE958R01014",
        "24x7",
        "1 April 2024",
        "कक्षा 10",
        "Floating rate; repayable by FY30",
        "1,23",  # not a valid grouping
        "Specialty Chemicals",
    ],
)
def test_text_is_not_a_quantity(text):
    assert parse_quantity(text) is None


def test_split_footnote():
    assert split_footnote("12.5%¹") == ("12.5%", "1")
    assert split_footnote("Revenue*") == ("Revenue", "*")
    assert split_footnote("*Revenue") == ("Revenue", "*")
    assert split_footnote("Total") == ("Total", None)


@pytest.mark.parametrize(
    ("text", "label", "currency", "scale"),
    [
        ("All amounts are in ₹ crore.", "₹ crore", "INR", "crore"),
        ("Amounts are in ₹ crore unless stated otherwise.", "₹ crore", "INR", "crore"),
        ("(₹ in crore)", "₹ crore", "INR", "crore"),
        ("Revenue from operations (₹ crore)", "₹ crore", "INR", "crore"),
        ("Revenue from operations (INR crore)", "₹ crore", "INR", "crore"),
        ("Rs. in lakhs", "₹ lakh", "INR", "lakh"),
        ("₹ करोड़ में", "₹ crore", "INR", "crore"),
        ("आवंटित बजट (₹ लाख)", "₹ lakh", "INR", "lakh"),
        ("(US$ million)", "$ million", "USD", "million"),
        ("Revenue in ₹ crore.", "₹ crore", "INR", "crore"),
        ("(in crores)", "₹ crore", "INR", "crore"),  # Indian scales are rupees
        ("₹ crore", "₹ crore", "INR", "crore"),
    ],
)
def test_amount_units(text, label, currency, scale):
    unit = detect_amount_unit(text)
    assert unit is not None and (unit.label, unit.currency, unit.scale, unit.kind) == (
        label,
        currency,
        scale,
        "currency",
    )


@pytest.mark.parametrize(
    "text",
    [
        "Other income for FY24 includes a one-time gain of ₹ 41 crore on the sale of surplus land",
        "EBITDA margin rose to 21.8% in the fourth quarter",
        "Directors crore",
        "Exports contributed 37 per cent of revenue",
    ],
)
def test_stated_amounts_are_not_unit_statements(text):
    assert detect_amount_unit(text) is None


@pytest.mark.parametrize(
    ("text", "kind", "label"),
    [
        ("Basic earnings per share (₹)", "currency", "₹"),
        ("Tier-1 city (₹ per night)", "currency", "₹"),
        ("EBITDA margin (%)", "percent", "%"),
        ("Net debt to EBITDA (x)", "ratio", "x"),
        ("Number of employees", "count", ""),
        ("Headcount", "count", ""),
        ("Average age of employees (years)", "duration", "years"),
        ("अवधि (सप्ताह)", "duration", "सप्ताह"),
        ("Warehouse capacity (million sq ft)", "none", "million sq ft"),
    ],
)
def test_header_units(text, kind, label):
    unit = detect_unit(text)
    assert unit is not None and (unit.kind, unit.label) == (kind, label)


def test_headers_without_units():
    assert detect_unit("EBITDA margin") is None
    assert detect_unit("Return on capital employed (ROCE)") is None


@pytest.mark.parametrize(
    ("text", "label", "kind", "end", "months"),
    [
        ("FY24", "FY24", "fy", date(2024, 3, 31), 12),
        ("FY 2023-24", "FY24", "fy", date(2024, 3, 31), 12),
        ("FY2023-24", "FY24", "fy", date(2024, 3, 31), 12),
        ("FY 23-24", "FY24", "fy", date(2024, 3, 31), 12),
        ("FY'24", "FY24", "fy", date(2024, 3, 31), 12),
        ("F.Y. 2023-24", "FY24", "fy", date(2024, 3, 31), 12),
        ("FY24-25", "FY25", "fy", date(2025, 3, 31), 12),
        ("2023-24", "FY24", "fy", date(2024, 3, 31), 12),
        ("वित्त वर्ष 2023-24", "FY24", "fy", date(2024, 3, 31), 12),
        ("Year ended 31 March 2024", "FY24", "fy", date(2024, 3, 31), 12),
        ("Q1 FY24", "Q1 FY24", "quarter", date(2023, 6, 30), 3),
        ("Q3 FY24", "Q3 FY24", "quarter", date(2023, 12, 31), 3),
        ("Q3FY24", "Q3 FY24", "quarter", date(2023, 12, 31), 3),
        ("3QFY24", "Q3 FY24", "quarter", date(2023, 12, 31), 3),
        ("Q4 FY23", "Q4 FY23", "quarter", date(2023, 3, 31), 3),
        ("Q1 2024", "Q1 2024", "quarter", date(2024, 3, 31), 3),
        ("H1 FY24", "H1 FY24", "half", date(2023, 9, 30), 6),
        ("H2 FY24", "H2 FY24", "half", date(2024, 3, 31), 6),
        ("9M FY24", "9M FY24", "nine_months", date(2023, 12, 31), 9),
        ("2024", "2024", "year", date(2024, 12, 31), 12),
        ("CY2023", "2023", "year", date(2023, 12, 31), 12),
        ("31 Mar 2024", "31 Mar 2024", "date", date(2024, 3, 31), 0),
        ("31 March 2024", "31 Mar 2024", "date", date(2024, 3, 31), 0),
        ("March 31, 2024", "31 Mar 2024", "date", date(2024, 3, 31), 0),
        ("31.03.2024", "31 Mar 2024", "date", date(2024, 3, 31), 0),
        ("2024-03-31", "31 Mar 2024", "date", date(2024, 3, 31), 0),
        ("20 सितंबर 2024", "20 Sep 2024", "date", date(2024, 9, 20), 0),
        ("Mar 2024", "Mar 2024", "month", date(2024, 3, 31), 1),
        ("Mar-24", "Mar 2024", "month", date(2024, 3, 31), 1),
        ("FY24¹", "FY24", "fy", date(2024, 3, 31), 12),
    ],
)
def test_period_labels(text, label, kind, end, months):
    p = parse_period(text)
    assert p is not None, text
    assert (p.label, p.kind, p.end, p.months) == (label, kind, end, months)


def test_period_labels_with_remarks():
    assert parse_period("Full year FY24").label == "FY24"  # type: ignore[union-attr]
    assert parse_period("FY24 (recommended)").label == "FY24"  # type: ignore[union-attr]
    assert parse_period("FY23 (paid in FY24)").label == "FY23"  # type: ignore[union-attr]
    assert parse_period("As at 31 March 2023").label == "31 Mar 2023"  # type: ignore[union-attr]
    assert parse_period("Revenue FY24") is None  # a measure of a period, not a period label
    assert parse_period("Specialty Chemicals") is None


def test_unanchored_quarters_keep_their_order():
    q1, q4 = parse_period("Q1"), parse_period("Q4")
    assert q1 is not None and q4 is not None and q1.end is None and q1.sort_key < q4.sort_key


def test_periods_inside_headers():
    assert find_period("Revenue FY24")[0].label == "FY24"  # type: ignore[index]
    assert find_period("Revenue FY24")[1] == "Revenue"  # type: ignore[index]
    assert find_period("Share of FY24 revenue")[1] == "Share of revenue"  # type: ignore[index]
    assert find_period("Meetings in FY24")[1] == "Meetings"  # type: ignore[index]
    assert find_period("EBITDA margin FY23")[0].label == "FY23"  # type: ignore[index]
    assert find_period("Specialty Chemicals") is None


def test_chronological_order_of_mixed_labels():
    labels = ["FY24", "FY23", "Q4 FY24", "Q1 FY24", "31 Mar 2023", "H1 FY24"]
    periods = sorted((parse_period(t) for t in labels), key=lambda p: p.sort_key)  # type: ignore[union-attr]
    assert [p.label for p in periods] == ["FY23", "31 Mar 2023", "Q1 FY24", "H1 FY24", "FY24", "Q4 FY24"]  # type: ignore[union-attr]
