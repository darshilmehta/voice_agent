# ruff: noqa: RUF002  (documents print en dashes, minus, times and divide signs: used on purpose)
"""Typed datasets: a parsed table read as data (docs/DESIGN.md §12.1, workstream 1).

``document_tables`` keeps each table's cells as printed; a ``TypedDataset`` says what they mean: which rows are the
header, which column holds the row labels, what every column and row is (a period such as FY24, a category, a
section heading, a total), the unit of every value (₹ crore, %, x, count) and every value as a number, each tied to
the cell it was read from (``row``/``col`` in the table grid). It also records how the table can be charted
(``Chartability``). Keys (``revenue_from_operations``, ``fy24``) are stable, readable ids within the dataset: the
visual spec refers to rows and columns by them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .canvas import Unit

ColumnType = Literal[
    "label",  # the row labels
    "period",  # values of one period (header FY24, 31 Mar 2024)
    "currency",
    "percent",
    "ratio",
    "count",
    "number",
    "change",  # the document's own change column (+13.6%, +118 bps): shown, never charted against periods
    "year",  # a column of years ("Commissioned": 1996, 2004)
    "date",  # dates (the Hindi notice's तिथि)
    "text",
    "note",  # note references ("Note": 22, 23)
    "empty",
]
RowType = Literal["data", "total", "section"]
PeriodKind = Literal["fy", "quarter", "half", "nine_months", "year", "date", "month"]
ChartKind = Literal["time_series", "categorical", "composition", "kpi", "timeline", "none"]
NUMERIC_COLUMN_TYPES: frozenset[str] = frozenset({"period", "currency", "percent", "ratio", "count", "number"})


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class PeriodInfo(_Model):
    label: str  # canonical: FY24, Q3 FY24, H1 FY24, 2024, 31 Mar 2024
    kind: PeriodKind
    end: date | None
    months: int  # 0 = a point in time (a balance-sheet date)

    @property
    def granularity(self) -> str:
        return {"fy": "year", "year": "year"}.get(self.kind, self.kind)

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.end.toordinal(), -self.months) if self.end is not None else (0, 0)


class DatasetColumn(_Model):
    key: str
    index: int  # column in the table grid
    label: str  # header text, multi-row headers joined ("Revenue FY24")
    header: list[str] = []  # the header rows' texts, top to bottom (spanning cells repeated)
    type: ColumnType
    unit: Unit | None = None  # what the header states, or what most of its cells state
    period: PeriodInfo | None = None
    measure: str | None = None  # the header without its period: "Revenue" for "Revenue FY24"


class DatasetRow(_Model):
    key: str
    index: int  # row in the table grid
    label: str  # without footnote markers
    type: RowType
    section: str | None = None  # the section heading above it ("Current liabilities")
    period: PeriodInfo | None = None  # the label is a period ("Q1 FY24", "FY24 (recommended)")
    unit: Unit | None = None  # what the label states ("(₹ crore)", "Number of …") or most of its cells
    parts: list[str] = []  # total rows: the rows it is the sum of (checked arithmetically)
    footnote: str | None = None


class DatasetValue(_Model):
    """One cell read as a number (``value`` None: printed as "–" / "n.a."). ``row``/``col`` are dataset keys;
    ``cell_row``/``cell_col`` the cell's position in the table grid (its top-left corner when it spans)."""

    row: str
    col: str
    value: float | None
    text: str
    unit: Unit | None
    cell_row: int
    cell_col: int
    decimals: int = 0
    signed: bool = False
    footnote: str | None = None


class DatasetText(_Model):
    """A text cell of a label, text, date or year column ("20 सितंबर 2024", "Hotel limits raised by 10 per cent"):
    what timelines and labels cite."""

    row: str
    col: str
    text: str
    cell_row: int
    cell_col: int


class ChartOption(_Model):
    kind: ChartKind
    confidence: float = Field(ge=0, le=1)


class Chartability(_Model):
    """How the dataset can be charted, best first. ``period_axis``: where its periods are (columns: "FY24 | FY23"
    headers; rows: "Q1 FY24" labels); ``period_order``: the period labels on that axis in chronological order (one
    granularity: quarters without the "Full year" row)."""

    kind: ChartKind
    confidence: float = Field(ge=0, le=1)
    options: list[ChartOption] = []
    period_axis: Literal["rows", "columns"] | None = None
    period_order: list[str] = []
    granularity: str | None = None
    reasons: list[str] = []


class TableContext(_Model):
    """Text next to the table in the document: the paragraphs just before it (where "All amounts are in ₹ crore"
    usually is) and the notes just after it (footnotes)."""

    before: list[str] = []
    after: list[str] = []


class TypedDataset(_Model):
    id: str
    table_id: str
    document_id: str
    version: int
    table_index: int
    page_start: int | None
    page_end: int | None
    title: str
    unit: Unit | None  # the table's unit (caption, a header cell such as "₹ crore", the note above it)
    header_rows: int
    label_column: int | None
    columns: list[DatasetColumn]
    rows: list[DatasetRow]
    values: list[DatasetValue]
    texts: list[DatasetText] = []
    chartability: Chartability
    context: TableContext = TableContext()
    chunk_id: str | None = None  # the table's retrieval chunk, when known (citations)
    warnings: list[str] = []
    typer_version: str
    created_at: datetime | None = None

    _value_index: dict[tuple[str, str], DatasetValue] | None = PrivateAttr(default=None)

    # -------------------------------------------------------------- lookups

    def column(self, key: str) -> DatasetColumn | None:
        return next((c for c in self.columns if c.key == key), None)

    def row(self, key: str) -> DatasetRow | None:
        return next((r for r in self.rows if r.key == key), None)

    def value(self, row: str, col: str) -> DatasetValue | None:
        return self._index().get((row, col))

    def _index(self) -> dict[tuple[str, str], DatasetValue]:
        if self._value_index is None:
            self._value_index = {(v.row, v.col): v for v in self.values}
        return self._value_index

    def text(self, row: str, col: str) -> DatasetText | None:
        return next((t for t in self.texts if t.row == row and t.col == col), None)

    @property
    def value_columns(self) -> list[DatasetColumn]:
        return [c for c in self.columns if c.type in NUMERIC_COLUMN_TYPES]

    @property
    def data_rows(self) -> list[DatasetRow]:
        return [r for r in self.rows if r.type != "section"]

    @property
    def page(self) -> int | None:
        return self.page_start


class DatasetSummary(BaseModel):
    """A dataset as listed for a project (debug panel, tests): enough to write a VisualSpec by hand."""

    id: str
    document_id: str
    filename: str
    table_id: str
    table_index: int
    page: int | None
    title: str
    unit: Unit | None
    chartability: Chartability
    columns: list[DatasetColumn]
    rows: list[DatasetRow]
    warnings: list[str] = []
