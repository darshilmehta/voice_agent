"""Typing a parsed table into a dataset (docs/DESIGN.md §12.1, workstream 1). Pure: tables in, ``TypedDataset`` out.

    cells → grid (spanning cells fill every position they cover)
          → header rows (Docling's column-header flags, else: the rows above the first row with a number, that have
            text beyond the label column; a multi-row header joins its rows: "Revenue" over "FY24" → "Revenue FY24")
          → label column (the leftmost column of text or period labels) and column types (period, currency, percent,
            ratio, count, number, change, year, date, text, note)
          → rows: section headings (a label and nothing else), totals (a row that is the sum of the rows above it,
            checked on every additive column, nested: "Total assets" = the two subtotals), periods ("Q1 FY24")
          → every value as a number with its unit: the cell's own marker (%, x, ₹, crore) → the row label's
            ("(₹ crore)", "Number of …") → the column header's → the table's (caption, a "₹ crore" header cell, "All
            amounts are in ₹ crore" just before the table) → the document's most common amount unit, for rows that
            name an amount (revenue, borrowings, …) only
          → chartability (``chartability.py``)

Every value keeps the grid position of its cell, which is what a visual's CellRef points to.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...domain.canvas import Unit
from ...domain.datasets import (
    NUMERIC_COLUMN_TYPES,
    Chartability,
    ColumnType,
    DatasetColumn,
    DatasetRow,
    DatasetText,
    DatasetValue,
    PeriodInfo,
    RowType,
    TableContext,
    TypedDataset,
)
from ...providers.ingestion import ParsedDocument
from .chartability import classify
from .parsing import (
    COUNT,
    NULL,
    Period,
    Quantity,
    currency_unit,
    detect_amount_unit,
    detect_unit,
    find_period,
    is_aggregate_period_label,
    is_year_value,
    normalize_space,
    parse_period,
    parse_quantity,
    split_footnote,
)

TYPER_VERSION = "t1"
CONTEXT_ITEMS = 2  # paragraphs read before and after a table
CONTEXT_CHARS = 400
ADDITIVE_UNIT_KINDS = frozenset({"currency", "count", "none"})


class TableLike(Protocol):
    """What typing reads from a stored table (``domain.projects.DocumentTable``)."""

    id: str
    document_id: str
    version: int
    table_index: int
    page_start: int | None
    page_end: int | None
    caption: str | None
    heading_path: list[str]
    num_rows: int
    num_cols: int
    cells: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _Cell:
    text: str
    row: int  # origin (top-left) of the cell
    col: int
    column_header: bool
    row_header: bool

    def at(self, r: int, c: int) -> bool:
        return self.row == r and self.col == c


@dataclass(slots=True)
class _Grid:
    rows: int
    cols: int
    cells: list[list[_Cell | None]]

    def text(self, r: int, c: int) -> str:
        cell = self.cells[r][c]
        return cell.text if cell is not None else ""

    def own(self, r: int, c: int) -> _Cell | None:
        """The cell whose origin is (r, c); None for an empty position or one covered by a spanning cell."""
        cell = self.cells[r][c]
        return cell if cell is not None and cell.at(r, c) else None


def _grid(table: TableLike) -> _Grid:
    rows, cols = max(table.num_rows, 0), max(table.num_cols, 0)
    cells: list[list[_Cell | None]] = [[None] * cols for _ in range(rows)]
    for raw in table.cells:
        r, c = int(raw.get("row", 0)), int(raw.get("col", 0))
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        cell = _Cell(
            text=normalize_space(str(raw.get("text") or "")),
            row=r,
            col=c,
            column_header=bool(raw.get("column_header")),
            row_header=bool(raw.get("row_header")),
        )
        for rr in range(r, min(r + max(int(raw.get("row_span") or 1), 1), rows)):
            for cc in range(c, min(c + max(int(raw.get("col_span") or 1), 1), cols)):
                if cells[rr][cc] is None:
                    cells[rr][cc] = cell
    return _Grid(rows, cols, cells)


# ------------------------------------------------------------------ cell classification


def _quantity(text: str) -> Quantity | None:
    q = parse_quantity(text)
    return q if isinstance(q, Quantity) else None


def _is_number(text: str) -> bool:
    """A value cell: a quantity that is not also a period label ("2024" in a header row is a period)."""
    return _quantity(text) is not None and parse_period(text) is None


def _header_rows(grid: _Grid) -> int:
    """Docling's column-header flags when it set them (extended by a following row of bare periods under a spanning
    header); otherwise the rows above the first row with a number that have text beyond the label column."""
    if grid.rows < 2:
        return 0
    flagged = 0
    for r in range(grid.rows - 1):
        texts = [c for c in _row_cells(grid, r) if c.text]
        if texts and sum(c.column_header for c in texts) * 2 >= len(texts):
            flagged = r + 1
        else:
            break
    if flagged:
        while flagged < min(3, grid.rows - 1) and _period_row(grid, flagged):
            flagged += 1
        return flagged
    first_number = next(
        (r for r in range(grid.rows) if any(_is_number(grid.text(r, c)) for c in range(1, grid.cols))), None
    )
    if first_number is None:  # a table of text: its first row
        return 1
    heuristic = 0
    for r in range(min(first_number, 3)):
        beyond_label = [grid.text(r, c) for c in range(1, grid.cols) if grid.text(r, c)]
        if not beyond_label:
            break  # a section heading ("Non-current assets"), not a header
        heuristic = r + 1
    return min(heuristic, grid.rows - 1)


def _period_row(grid: _Grid, r: int) -> bool:
    """A header row of bare periods ("FY24 | FY23" under "Revenue" spanning both), nothing in the label column."""
    if grid.text(r, 0):
        return False
    texts = [cell.text for c in range(1, grid.cols) if (cell := grid.own(r, c)) is not None and cell.text]
    return len(texts) >= 2 and all(parse_period(t) is not None for t in texts)


def _row_cells(grid: _Grid, r: int) -> list[_Cell]:
    seen: list[_Cell] = []
    for c in range(grid.cols):
        cell = grid.cells[r][c]
        if cell is not None and cell not in seen:
            seen.append(cell)
    return seen


_IDENTIFIER_HEADER = re.compile(
    r"^(?:version|ver\.?|sr\.?\s*no\.?|s\.?\s*no\.?|sl\.?\s*no\.?|no\.?|#|id|code|item\s*no\.?|क्र\.?\s*सं\.?)$", re.I
)


def _label_column(grid: _Grid, header_rows: int) -> int | None:
    """The leftmost column of text or period labels; a first column of identifiers ("Version": 3.2, 3.1) counts
    too when its header says so, or when no other column holds numbers."""
    for c in range(grid.cols):
        texts = [grid.text(r, c) for r in range(header_rows, grid.rows) if grid.text(r, c)]
        if not texts:
            continue
        labels = sum(1 for t in texts if not _is_number(t) or parse_period(t) is not None)
        if labels * 2 >= len(texts):
            return c
        header = " ".join(grid.text(r, c) for r in range(header_rows)).strip()
        others_numeric = any(
            _is_number(grid.text(r, k)) for k in range(c + 1, grid.cols) for r in range(header_rows, grid.rows)
        )
        if _IDENTIFIER_HEADER.match(header) or (not others_numeric and len(set(texts)) == len(texts)):
            return c
        return None
    return None


_NOTE_HEADER = re.compile(r"^(?:notes?|note\s*no\.?|notes?\s*ref\.?|schedule|ref\.?|sch\.?|टिप्पणी)$", re.I)
_CHANGE_HEADER = re.compile(
    r"^(?:change|chg\.?|%\s*change|change\s*\(%\)|yoy|y-o-y|y-?o-?y\s*change|growth|growth\s*\(%\)|var(?:iance)?\.?|"
    r"increase|increase\s*/\s*\(decrease\)|\+/-|बदलाव|परिवर्तन|वृद्धि)$",
    re.I,
)
_TOTAL_LABEL = re.compile(r"^(?:total|grand\s+total|sub-?total|net\s+total|कुल|योग)\b", re.I)
_NUMBERED_WORDS = frozenset(
    {"unit", "plant", "phase", "line", "tier", "grade", "level", "note", "schedule", "class", "type", "block", "sector",
     "zone", "stage", "part", "version", "building", "tower", "wing", "floor", "centre", "center", "site", "cluster",
     "group", "team", "batch", "round", "series", "tranche", "and", "or", "to", "of", "&", "scope", "step", "item"}
)  # fmt: skip
_NOTE_MARK = re.compile(r"(?:^|\s)([1-9])\.\s+\S")
_MONETARY = re.compile(
    r"revenue|income|profit|loss|expense|cost|borrowing|loan|debt|asset|liabilit|equity|cash|capital|payable|"
    r"receivable|inventor|investment|tax|dividends?\s+paid|sales|ebitda|ebit\b|\bpat\b|\bpbt\b|debenture|paper|"
    r"demand|claim|outflow|inflow|capex|depreciation|amortisation|amortization|finance|provision|reserve|premium|"
    r"budget|spend|turnover|margin\s+money|bank\s+balance|बजट|राजस्व|लाभ|व्यय|आय|राशि",
    re.I,
)


def _clean_label(text: str, notes: set[str]) -> tuple[str, str | None]:
    """A row label without footnote markers: "Digital Services 1" → "Digital Services" when note 1 is under the
    table (or, without the notes, when the number can't be part of the name: "Unit 1", "Scope 1 and 2" keep it)."""
    label, footnote = split_footnote(text)
    m = re.fullmatch(r"(.*[^\W\d_)])\s+([1-9])", label)
    if m:
        prefix, number = m.group(1), m.group(2)
        last = prefix.split()[-1].lower().strip(".,")
        known = number in notes
        plausible = not notes and last not in _NUMBERED_WORDS and not re.search(r"\d", prefix)
        if (known or plausible) and parse_period(label) is None:
            return prefix.strip(), footnote or number
    return label, footnote


def _strip_unit(label: str) -> str:
    """ "Revenue from operations (₹ crore)" → "Revenue from operations"; "(ROCE)" stays."""
    m = re.fullmatch(r"(.*?)\s*\(([^()]*)\)\s*", label)
    if m and m.group(1) and detect_unit(f"({m.group(2)})") is not None:
        return m.group(1).strip()
    return label


_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str, fallback: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    slug = _SLUG.sub("_", ascii_text).strip("_")
    if len(slug) > 48:
        slug = slug[:48].rsplit("_", 1)[0]
    return slug if slug and not slug.isdigit() and len(slug) >= max(2, len(text) // 4) else fallback


def _unique(key: str, used: set[str]) -> str:
    out, n = key, 2
    while out in used:
        out, n = f"{key}_{n}", n + 1
    used.add(out)
    return out


def _period_info(p: Period | None) -> PeriodInfo | None:
    if p is None:
        return None
    return PeriodInfo(label=p.label, kind=p.kind, end=p.end, months=p.months)


# ------------------------------------------------------------------ typing


@dataclass(slots=True)
class _Col:
    index: int
    header: list[str]
    label: str
    type: ColumnType
    unit: Unit | None
    period: Period | None
    measure: str | None
    stated_unit: Unit | None  # what the header says


@dataclass(slots=True)
class _Row:
    index: int
    label: str
    type: RowType
    section: str | None
    period: Period | None
    stated_unit: Unit | None
    footnote: str | None
    key: str = ""
    parts: list[str] = field(default_factory=list)


def _column(grid: _Grid, c: int, header_rows: int, label_column: int | None) -> _Col:
    header: list[str] = []
    for r in range(header_rows):
        text = grid.text(r, c)
        if text and (not header or header[-1] != text):
            header.append(text)
    label = normalize_space(" ".join(header))
    if c == label_column:
        return _Col(c, header, label, "label", None, None, None, None)
    stated = detect_unit(label)
    found = find_period(label)
    period, measure = (found[0], found[1] or None) if found else (None, None)
    texts = [cell.text for r in range(header_rows, grid.rows) if (cell := grid.own(r, c)) is not None and cell.text]
    quantities: list[tuple[Quantity, str]] = []
    text_cells = dates = 0
    for t in texts:
        q = parse_quantity(t)
        if q is NULL:
            continue
        if isinstance(q, Quantity):
            quantities.append((q, t))
        elif parse_period(t) is not None or (
            (found_date := find_period(t)) is not None and found_date[0].kind == "date"
        ):
            dates += 1
        else:
            text_cells += 1
    if _NOTE_HEADER.match(label):
        return _Col(c, header, label, "note", None, None, None, stated)
    numbers = len(quantities)
    if numbers == 0 and text_cells == 0 and dates == 0:
        return _Col(c, header, label, "empty", stated, period, measure, stated)
    if numbers and numbers * 5 >= (numbers + text_cells + dates) * 3:
        qs = [q for q, _ in quantities]
        kinds = Counter(q.kind for q in qs)
        if _CHANGE_HEADER.match(_strip_unit(measure or label)) or (
            numbers >= 2 and all(q.signed or q.value == 0 for q in qs) and kinds.most_common(1)[0][0] != "number"
        ):
            return _Col(c, header, label, "change", stated, period, measure, stated)
        if stated is None and all(is_year_value(q, t) for q, t in quantities):
            return _Col(c, header, label, "year", None, period, measure, stated)
        unit = stated or _common_unit(qs)
        ctype: ColumnType
        if period is not None and not measure:
            ctype = "period"
        elif unit is not None and unit.kind in ("percent", "ratio", "currency", "count"):
            ctype = unit.kind
        else:
            ctype = "number"
        return _Col(c, header, label, ctype, unit, period, measure, stated)
    if dates and dates * 2 >= dates + text_cells + numbers:
        return _Col(c, header, label, "date", None, period, measure, stated)
    return _Col(c, header, label, "text", None, period, measure, stated)


def _common_unit(quantities: Sequence[Quantity]) -> Unit | None:
    units = Counter(q.unit() for q in quantities)
    unit, n = units.most_common(1)[0]
    return unit if n * 5 >= len(quantities) * 3 else None


def _notes(context: TableContext) -> set[str]:
    return {m.group(1) for text in context.after for m in _NOTE_MARK.finditer(text)}


def table_unit(table: TableLike, grid: _Grid, header_rows: int, context: TableContext) -> Unit | None:
    """The unit stated for the whole table: caption, a header cell that is only a unit ("₹ crore"), the paragraph
    just before the table, a note just after it, then the nearest heading."""
    candidates: list[str | None] = [table.caption]
    for r in range(header_rows):
        for c in range(grid.cols):
            text = grid.text(r, c)
            if text and detect_amount_unit(text) is not None and len(text) <= 24:
                candidates.append(text)
    candidates += list(reversed(context.before)) + list(context.after) + list(reversed(table.heading_path))
    for text in candidates:
        unit = detect_amount_unit(text)
        if unit is not None:
            return unit
    return None


def type_table(
    table: TableLike,
    context: TableContext | None = None,
    *,
    dataset_id: str,
    document_unit: Unit | None = None,
    chunk_id: str | None = None,
) -> TypedDataset:
    """Type one table. ``document_unit`` (the document's usual amount unit) is the last resort for rows that name
    an amount but whose table states no unit."""
    context = context or TableContext()
    grid = _grid(table)
    warnings: list[str] = []
    header_rows = _header_rows(grid)
    label_column = _label_column(grid, header_rows)
    cols = [_column(grid, c, header_rows, label_column) for c in range(grid.cols)]
    tunit = table_unit(table, grid, header_rows, context)
    notes = _notes(context)

    # rows
    rows: list[_Row] = []
    section: str | None = None
    value_cols = [c for c in cols if c.type in NUMERIC_COLUMN_TYPES or c.type == "change"]
    other_cols = [c for c in cols if c.type in ("text", "date", "year")]
    for r in range(header_rows, grid.rows):
        raw_label = grid.text(r, label_column) if label_column is not None else ""
        label, footnote = _clean_label(raw_label, notes) if raw_label else ("", None)
        has_values = any(grid.own(r, c.index) and grid.own(r, c.index).text for c in value_cols)  # type: ignore[union-attr]
        has_other = any(grid.own(r, c.index) and grid.own(r, c.index).text for c in other_cols)  # type: ignore[union-attr]
        if not label and not has_values and not has_other:
            continue  # blank row
        if label and not has_values and not has_other and value_cols:
            rows.append(_Row(r, label, "section", None, None, None, footnote))
            section = label
            continue
        period = parse_period(label) if label else None
        rtype: RowType = (
            "total" if _TOTAL_LABEL.match(label) or (period and is_aggregate_period_label(label)) else "data"
        )
        # the label without its unit ("Revenue from operations (₹ crore)" → "Revenue from operations"): the unit is kept
        rows.append(_Row(r, _strip_unit(label), rtype, section, period, detect_unit(label), footnote))

    used: set[str] = set()
    for row in rows:
        base = _slug(_strip_unit(row.label), f"r{row.index}")
        if base in used and row.section:
            base = _slug(f"{row.section} {_strip_unit(row.label)}", base)
        row.key = _unique(base, used)
    col_used: set[str] = set()
    col_keys = {c.index: _unique(_slug(_strip_unit(c.label), f"c{c.index}"), col_used) for c in cols}

    # values
    values: list[DatasetValue] = []
    by_cell: dict[tuple[str, str], DatasetValue] = {}
    inferred_from_document = False
    for row in rows:
        if row.type == "section":
            continue
        for col in value_cols:
            cell = grid.own(row.index, col.index)
            if cell is None or not cell.text:
                continue
            q = parse_quantity(cell.text)
            if q is None:
                continue  # text in a numeric column ("Matured in FY24")
            unit = None if q is NULL else q.unit()  # type: ignore[union-attr]
            if q is not NULL and (unit is None or (unit.kind == "none" and unit.scale and not unit.label.strip())):
                unit = _resolve_unit(row, col, tunit)
                if unit is None and document_unit is not None and _MONETARY.search(row.label or col.label):
                    unit, inferred_from_document = document_unit, True
            elif q is not NULL and unit is not None and unit.kind == "none" and unit.scale and q.kind == "number":
                ctx = _resolve_unit(row, col, tunit)  # "12.5 crore" in a ₹ table stays a count with a scale
                if ctx is not None and ctx.kind == "currency" and ctx.scale == unit.scale and not q.unit_word:
                    unit = ctx
            if q is NULL:
                unit = _resolve_unit(row, col, tunit)
            v = DatasetValue(
                row=row.key,
                col=col_keys[col.index],
                value=None if q is NULL else q.value,  # type: ignore[union-attr]
                text=cell.text,
                unit=unit,
                cell_row=cell.row,
                cell_col=cell.col,
                decimals=0 if q is NULL else q.decimals,  # type: ignore[union-attr]
                signed=False if q is NULL else q.signed,  # type: ignore[union-attr]
                footnote=None if q is NULL else q.footnote,  # type: ignore[union-attr]
            )
            values.append(v)
            by_cell[(row.key, v.col)] = v
    if inferred_from_document and document_unit is not None:
        warnings.append(f"unit {document_unit.label} assumed from the rest of the document")
    texts = [
        DatasetText(row=row.key, col=col_keys[col.index], text=cell.text, cell_row=cell.row, cell_col=cell.col)
        for row in rows
        if row.type != "section"
        for col in cols
        if col.type in ("label", "text", "date", "year")
        and (cell := grid.own(row.index, col.index)) is not None
        and cell.text
    ]

    _detect_totals(rows, [c for c in value_cols if c.type != "change"], col_keys, by_cell)

    columns = []
    for c in cols:
        ctype, cunit = c.type, (c.unit if c.type != "change" else None)
        if ctype in NUMERIC_COLUMN_TYPES and ctype != "period":
            resolved = _common_value_unit(col_keys[c.index], values)
            cunit = cunit if cunit is not None and cunit.kind != "none" else resolved or cunit
            if ctype == "number" and cunit is not None and cunit.kind in ("currency", "percent", "ratio", "count"):
                ctype = cunit.kind  # type: ignore[assignment]
        columns.append(
            DatasetColumn(
                key=col_keys[c.index],
                index=c.index,
                label=_strip_unit(c.label) or (c.period.label if c.period else ""),
                header=c.header,
                type=ctype,
                unit=cunit,
                period=_period_info(c.period),
                measure=_strip_unit(c.measure) if c.measure else None,
            )
        )
    measured = {col_keys[c.index] for c in cols if c.type in NUMERIC_COLUMN_TYPES}
    out_rows = [
        DatasetRow(
            key=r.key,
            index=r.index,
            label=r.label,
            type=r.type,
            section=r.section,
            period=_period_info(r.period),
            unit=r.stated_unit or _row_unit(r.key, values, measured),
            parts=r.parts,
            footnote=r.footnote,
        )
        for r in rows
    ]
    title = _title(table, cols, header_rows)
    dataset = TypedDataset(
        id=dataset_id,
        table_id=table.id,
        document_id=table.document_id,
        version=table.version,
        table_index=table.table_index,
        page_start=table.page_start,
        page_end=table.page_end,
        title=title,
        unit=tunit,
        header_rows=header_rows,
        label_column=label_column,
        columns=columns,
        rows=out_rows,
        values=values,
        texts=texts,
        chartability=Chartability(kind="none", confidence=0.0),
        context=context,
        chunk_id=chunk_id,
        warnings=warnings,
        typer_version=TYPER_VERSION,
    )
    return dataset.model_copy(update={"chartability": classify(dataset)})


def _resolve_unit(row: _Row, col: _Col, table_unit_: Unit | None) -> Unit | None:
    """The unit of a bare number: the row label's, the column header's, the column's cells', then the table's."""
    for unit in (row.stated_unit, col.stated_unit):
        if unit is not None:
            return unit
    if col.unit is not None and col.type not in ("period",):
        return col.unit
    if row.stated_unit is None and table_unit_ is not None:
        return table_unit_
    return None


def _row_unit(key: str, values: Iterable[DatasetValue], value_keys: set[str] | None = None) -> Unit | None:
    """The unit all of a row's measured values share (None when they differ)."""
    units = Counter(
        v.unit for v in values if v.row == key and v.value is not None and (value_keys is None or v.col in value_keys)
    )
    if not units:
        return None
    unit, n = units.most_common(1)[0]
    return unit if n == sum(units.values()) else None


def _common_value_unit(col_key: str, values: Iterable[DatasetValue]) -> Unit | None:
    units = Counter(v.unit for v in values if v.col == col_key and v.value is not None)
    if not units:
        return None
    unit, n = units.most_common(1)[0]
    return unit if n == sum(units.values()) else None


def _additive(unit: Unit | None) -> bool:
    return unit is None or unit.kind in ADDITIVE_UNIT_KINDS


def _detect_totals(
    rows: list[_Row], cols: Sequence[_Col], col_keys: Mapping[int, str], by_cell: Mapping[tuple[str, str], DatasetValue]
) -> None:
    """Mark rows that are the sum of the rows just above them (after collapsing totals already found, so a grand
    total is the sum of its subtotals). A match needs every additive column where both sides have numbers to agree
    within the rounding of the printed figures, and either a "Total" label, two such columns, or three parts."""
    working: list[_Row] = []
    for row in rows:
        if row.type == "section":
            continue
        found: list[_Row] | None = None
        checked_best = 0
        for k in range(2, len(working) + 1):
            parts = working[-k:]
            checked = 0
            ok = True
            for col in cols:
                key = col_keys[col.index]
                total = by_cell.get((row.key, key))
                if total is None or total.value is None or not _additive(total.unit):
                    continue
                part_values = [by_cell.get((p.key, key)) for p in parts]
                present = [v for v in part_values if v is not None and v.value is not None]
                if len(present) < 2:
                    continue
                if any(v.unit != total.unit or not _additive(v.unit) for v in present):
                    ok = False
                    break
                decimals = max([total.decimals, *(v.decimals for v in present)])
                # Printed figures are rounded, so parts may miss their total by up to half a unit each, but never by
                # more than 0.2% of it (small counts must add up exactly: 3 + 4 is not 6).
                tolerance = min(0.5 * 10**-decimals * len(present), 0.002 * abs(total.value)) + 1e-9
                if abs(sum(v.value for v in present) - total.value) > tolerance:  # type: ignore[misc]
                    ok = False
                    break
                checked += 1
            if ok and checked and (_TOTAL_LABEL.match(row.label) or row.type == "total" or checked >= 2 or k >= 3):
                found, checked_best = parts, checked
                break
        if found is not None and checked_best:
            row.type = "total"
            row.parts = [p.key for p in found]
            working = [*working[: len(working) - len(found)], row]
        else:
            working.append(row)


def _title(table: TableLike, cols: Sequence[_Col], header_rows: int) -> str:
    if table.caption and table.caption.strip():
        return normalize_space(table.caption)
    if table.heading_path:
        return normalize_space(table.heading_path[-1])
    labels = [c.label for c in cols if c.label]
    return " · ".join(labels[:3]) or f"Table {table.table_index + 1}"


# ------------------------------------------------------------------ documents


def table_contexts(parsed: ParsedDocument) -> dict[int, TableContext]:
    """The paragraphs just before each table (within its section) and the notes just after it."""
    items = parsed.items
    out: dict[int, TableContext] = {}
    text_labels = {"text", "paragraph", "caption", "footnote", "list_item"}
    for i, item in enumerate(items):
        if item.table_index is None:
            continue
        before: list[str] = []
        for prev in reversed(items[max(0, i - 6) : i]):
            if prev.table_index is not None or prev.label not in text_labels:
                break
            before.insert(0, prev.text[:CONTEXT_CHARS])
            if len(before) == CONTEXT_ITEMS:
                break
        after: list[str] = []
        for nxt in items[i + 1 : i + 7]:
            if nxt.table_index is not None or nxt.label not in text_labels:
                break
            after.append(nxt.text[:CONTEXT_CHARS])
            if len(after) == CONTEXT_ITEMS:
                break
        out[item.table_index] = TableContext(before=before, after=after)
    return out


def document_unit(tables: Sequence[TableLike], contexts: Mapping[int, TableContext]) -> Unit | None:
    """The amount unit most of the document's tables state ("₹ crore" in an Indian annual report)."""
    units: Counter[Unit] = Counter()
    for t in tables:
        grid = _grid(t)
        unit = table_unit(t, grid, _header_rows(grid), contexts.get(t.table_index, TableContext()))
        if unit is not None and unit.kind == "currency":
            units[unit] += 1
        for r in range(grid.rows):  # row labels such as "Revenue from operations (₹ crore)"
            label_unit = detect_amount_unit(grid.text(r, 0))
            if label_unit is not None and label_unit.kind == "currency" and label_unit.scale:
                units[label_unit] += 1
    if not units:
        return None
    return units.most_common(1)[0][0]


def type_document(
    tables: Sequence[TableLike],
    contexts: Mapping[int, TableContext] | None = None,
    *,
    new_id: Any,
    chunk_ids: Mapping[int, str] | None = None,
) -> list[TypedDataset]:
    """Type every table of one document version; ``new_id()`` gives dataset ids."""
    contexts = contexts or {}
    chunk_ids = chunk_ids or {}
    doc_unit = document_unit(tables, contexts)
    return [
        type_table(
            t,
            contexts.get(t.table_index),
            dataset_id=new_id(),
            document_unit=doc_unit,
            chunk_id=chunk_ids.get(t.table_index),
        )
        for t in tables
    ]


__all__ = [
    "COUNT",
    "TYPER_VERSION",
    "TableLike",
    "currency_unit",
    "document_unit",
    "table_contexts",
    "type_document",
    "type_table",
]
