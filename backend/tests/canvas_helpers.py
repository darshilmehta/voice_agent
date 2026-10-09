"""Synthetic tables for the canvas tests: cell grids as Docling stores them (``document_tables.cells``)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from app.domain.datasets import TableContext, TypedDataset
from app.domain.projects import DocumentTable
from app.services.canvas.datasets import type_document, type_table

_ids = itertools.count(1)

SPAN = object()  # marks a grid position covered by a spanning cell (no cell of its own)

Cell = str | tuple[str, int, int] | object  # text, (text, row_span, col_span), or SPAN


def table(
    rows: Sequence[Sequence[Cell]],
    *,
    header_rows: int = 1,
    flag_headers: bool = True,
    caption: str | None = None,
    heading: Sequence[str] = ("Financial highlights",),
    page: int | None = 3,
    document_id: str = "doc_1",
    index: int = 0,
    table_id: str | None = None,
) -> DocumentTable:
    """A stored table from rows of cells. ``("Revenue", 1, 2)`` spans two columns; ``SPAN`` marks the positions a
    spanning cell covers. The first ``header_rows`` rows carry Docling's column-header flag unless
    ``flag_headers`` is False."""
    cells: list[dict[str, Any]] = []
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is SPAN:
                continue
            text, rs, cs = (cell if isinstance(cell, tuple) else (cell, 1, 1))  # type: ignore[misc]
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "row_span": rs,
                    "col_span": cs,
                    "text": text,
                    "column_header": flag_headers and r < header_rows,
                    "row_header": False,
                }
            )
    return DocumentTable(
        id=table_id or f"tbl_{next(_ids)}",
        document_id=document_id,
        version=1,
        table_index=index,
        page_start=page,
        page_end=page,
        bbox=None,
        heading_path=list(heading),
        caption=caption,
        num_rows=len(rows),
        num_cols=max(len(r) for r in rows),
        markdown="",
        cells=cells,
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
    )


def typed(
    rows: Sequence[Sequence[Cell]],
    *,
    before: Sequence[str] = (),
    after: Sequence[str] = (),
    dataset_id: str | None = None,
    chunk_id: str | None = None,
    **kw: Any,
) -> TypedDataset:
    t = table(rows, **kw)
    return type_table(
        t,
        TableContext(before=list(before), after=list(after)),
        dataset_id=dataset_id or f"ds_{next(_ids)}",
        chunk_id=chunk_id,
    )


def typed_document(tables: Sequence[DocumentTable], contexts: dict[int, TableContext] | None = None) -> list[TypedDataset]:
    return type_document(tables, contexts or {}, new_id=lambda: f"ds_{next(_ids)}")


# ------------------------------------------------------------------ an annual report in miniature

HIGHLIGHTS = [
    ["Metric", "FY24", "FY23", "Change"],
    ["Revenue from operations (₹ crore)", "7,365", "6,482", "+13.6%"],
    ["EBITDA (₹ crore)", "1,545", "1,283", "+20.4%"],
    ["EBITDA margin", "21.0%", "19.8%", "+118 bps"],
    ["Profit after tax (₹ crore)", "871", "664", "+31.2%"],
    ["Basic earnings per share (₹)", "69.68", "53.12", "+31.2%"],
    ["Net debt to EBITDA", "0.54x", "0.93x", "-0.39x"],
    ["Number of employees", "9,842", "9,310", "+5.7%"],
]
SEGMENTS = [
    ["Segment", "Revenue FY24", "Revenue FY23", "EBITDA FY24", "EBITDA FY23", "EBITDA margin FY24"],
    ["Specialty Chemicals", "3,568", "3,214", "862", "745", "24.2%"],
    ["Engineered Plastics", "2,303", "2,126", "346", "331", "15.0%"],
    ["Digital Services 1", "1,494", "1,142", "337", "207", "22.6%"],
    ["Total", "7,365", "6,482", "1,545", "1,283", "21.0%"],
]
QUARTERS_FY24 = [
    ["Quarter", "Revenue", "EBITDA", "EBITDA margin", "Profit after tax"],
    ["Q1 FY24", "1,742", "352", "20.2%", "196"],
    ["Q2 FY24", "1,801", "372", "20.7%", "212"],
    ["Q3 FY24", "1,889", "399", "21.1%", "227"],
    ["Q4 FY24", "1,933", "422", "21.8%", "236"],
    ["Full year FY24", "7,365", "1,545", "21.0%", "871"],
]
QUARTERS_FY23 = [
    ["Quarter", "Revenue", "EBITDA", "EBITDA margin", "Profit after tax"],
    ["Q1 FY23", "1,512", "285", "18.8%", "148"],
    ["Q2 FY23", "1,598", "312", "19.5%", "163"],
    ["Q3 FY23", "1,661", "331", "19.9%", "171"],
    ["Q4 FY23", "1,711", "355", "20.7%", "182"],
    ["Full year FY23", "6,482", "1,283", "19.8%", "664"],
]
CASH_FLOW = [
    ["Particulars", "FY24", "FY23"],
    ["Net cash generated from operating activities", "1,284", "1,052"],
    ["Net cash used in investing activities", "(712)", "(655)"],
    ["Net cash used in financing activities", "(446)", "(293)"],
    ["Net increase in cash and cash equivalents", "126", "104"],
    ["Cash and cash equivalents at the beginning of the year", "486", "382"],
    ["Cash and cash equivalents at the end of the year", "612", "486"],
]
BALANCE_SHEET = [
    ["Particulars", "Note", "31 Mar 2024", "31 Mar 2023"],
    ["Non-current assets", "", "", ""],
    ["Property, plant and equipment", "3", "3,612", "3,240"],
    ["Capital work-in-progress", "4", "358", "412"],
    ["Total non-current assets", "", "3,970", "3,652"],
    ["Current assets", "", "", ""],
    ["Inventories", "8", "1,248", "1,126"],
    ["Cash and bank balances", "10", "612", "486"],
    ["Total current assets", "", "1,860", "1,612"],
    ["Total assets", "", "5,830", "5,264"],
]
GLOSSARY = [
    ["Term", "Meaning in this report"],
    ["EBITDA", "Earnings before interest, tax, depreciation and amortisation."],
    ["ROCE", "Return on capital employed."],
    ["Scope 1 and 2", "Direct and indirect emissions."],
]
DATES_HI = [
    ["गतिविधि", "तिथि"],
    ["ऑनलाइन आवेदन प्रारंभ", "20 सितंबर 2024"],
    ["आवेदन की अंतिम तिथि", "30 नवंबर 2024"],
    ["प्रशिक्षण का प्रारंभ", "6 जनवरी 2025"],
]
DISTRICTS_HI = [
    ["जिला", "प्रशिक्षण केंद्र", "लक्ष्य (युवा)", "आवंटित बजट (₹ लाख)"],
    ["कमलपुर", "6", "2,800", "650"],
    ["हरिपुर", "5", "2,400", "560"],
    ["सूर्यगढ़", "7", "3,100", "720"],
    ["कुल", "18", "8,300", "1,930"],
]


def report_datasets(document_id: str = "doc_1") -> dict[str, TypedDataset]:
    """The miniature annual report typed, by name."""
    specs = {
        "highlights": (HIGHLIGHTS, ["Amounts are in ₹ crore unless stated otherwise."], [], 3),
        "segments": (SEGMENTS, ["Segment revenue and EBITDA are shown below (₹ crore)."], ["1. Net of inter-segment sales."], 18),
        "q_fy24": (QUARTERS_FY24, ["The fourth quarter was the strongest (₹ crore)."], [], 19),
        "q_fy23": (QUARTERS_FY23, ["FY23 started slowly (₹ crore)."], [], 20),
        "cash_flow": (CASH_FLOW, ["All amounts are in ₹ crore."], [], 25),
        "balance_sheet": (BALANCE_SHEET, ["All amounts are in ₹ crore."], [], 23),
        "glossary": (GLOSSARY, [], [], 29),
    }
    out = {}
    for n, (name, (rows, before, after, page)) in enumerate(specs.items()):
        out[name] = typed(
            rows,
            before=before,
            after=after,
            page=page,
            index=n,
            document_id=document_id,
            heading=(name.replace("_", " ").title(),),
            dataset_id=f"ds_{name}",
            chunk_id=f"{document_id}:v1:{n:04d}",
        )
    return out
