"""The live visual canvas as the frontend sees it: contract v1 (docs/DESIGN.md §12.1).

A ``Visual`` is built by code from a validated ``VisualSpec`` (``services.canvas``): the model only picks what to show.
Every number in a Visual is either the value of a table cell (``CellRef``) or the result of a deterministic
``Calculation`` whose inputs are cells. Values are numbers in their unit's scale (4210 with ``scale: "crore"`` is
₹4,210 crore); the frontend formats them.

Conventions inside the contract (additive, matching the renderer):

- ``rows`` are in display order: chronological for periods, document order for categories.
- Percentages are numbers of percent: 18.4 with ``unit.kind: "percent"`` is 18.4%, never 0.184.
- Series in different units carry their own ``unit`` (``line`` and ``bar`` may mix units: drawn as small multiples on a
  shared x axis; the visual's ``unit`` is then null). ``series[].better`` ("up", "down" or null) says which way is good
  when the metric's name makes it clear (revenue up, costs and debt down).
- ``waterfall``: three series ``total`` (bars from zero), ``increase`` and ``decrease`` (floating steps); each row has
  a value in exactly one of them and ``rows[].kind`` "total" (the first, the last and any subtotal) or "delta". A
  step's value is the magnitude of its cell (a cell printed "(712)" contributes 712 to ``decrease``); its CellRef
  carries the printed text.
- ``donut``: one series; rows are the slices; one ``share`` Calculation per slice (the slice named in its label) unless
  the slices are already printed percentages.
- ``kpi``: ``tiles`` (``x`` null, no rows); ``series`` lists one entry per tile with the tile's label (unit, better).
- ``comparison``: ``x`` null; ``series`` are the two sides (two periods, or two documents), ``rows`` the metrics
  (``rows[].x`` is the metric) and each metric's change is the tile with the same label as the row's ``x``.
- ``timeline``: ``events`` only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .projects import Citation

VisualKind = Literal[
    "kpi", "line", "bar", "grouped_bar", "stacked_bar", "waterfall", "donut", "table", "comparison", "timeline"
]
VISUAL_KINDS: tuple[VisualKind, ...] = get_args(VisualKind)
UnitKind = Literal["currency", "percent", "count", "ratio", "duration", "none"]
Scale = Literal["crore", "lakh", "million", "billion", "thousand"]
CalcOp = Literal["growth", "cagr", "diff", "ratio", "share", "sum"]
CALC_OPS: tuple[CalcOp, ...] = get_args(CalcOp)
XType = Literal["period", "category", "date"]
DeltaKind = Literal["abs", "pct", "pp"]
VisualPhase = Literal["preparing", "ready", "failed"]
OverviewStatus = Literal["ready", "building", "none"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Unit(_Frozen):
    """What a number measures. ``label`` is ready to show next to a value ("₹ crore", "%", "x", "days")."""

    kind: UnitKind
    currency: str | None = None  # ISO code: INR, USD, EUR, GBP
    scale: Scale | None = None
    label: str = ""


class CellRef(_Frozen):
    """The exact table cell a value came from. ``row``/``col`` are the cell's 0-based position in the table as
    parsed (``document_tables.cells``); ``text`` is the cell as printed; ``page`` the table's page (None for formats
    without pages)."""

    source_id: str
    document_id: str
    table_id: str
    page: int | None
    row: int
    col: int
    text: str


class Calculation(_Frozen):
    """A derived number, computed by code from cells (labelled "calculated" in the UI)."""

    label: str
    op: CalcOp
    value: float
    unit: Unit | None
    inputs: list[CellRef]
    formula_text: str


class XAxis(_Frozen):
    key: str
    label: str
    type: XType


class Series(_Frozen):
    key: str
    label: str
    unit: Unit | None
    calculated: bool = False
    better: Literal["up", "down"] | None = None  # additive: which way is good, when the metric makes it clear


class Row(_Frozen):
    x: str
    values: dict[str, float | None]
    cells: dict[str, CellRef | None]
    kind: Literal["total", "delta"] | None = None  # additive, waterfall only: a total bar or a step


class Delta(_Frozen):
    value: float
    kind: DeltaKind
    calculation: Calculation


class Tile(_Frozen):
    label: str
    value: float
    unit: Unit | None
    cell: CellRef | None
    delta: Delta | None = None


class TimelineEvent(_Frozen):
    date: str  # "YYYY-MM-DD" when the cell holds a full date, else the label as printed
    label: str
    detail: str | None = None
    cell: CellRef | None = None


class Highlight(_Frozen):
    x: list[str]
    series: list[str]
    note: str | None = None


class Visual(_Frozen):
    id: str
    chat_id: str | None  # None: the project overview
    project_id: str
    created_at: datetime
    updated_at: datetime
    kind: VisualKind
    title: str
    subtitle: str | None
    language: Literal["en", "hi"]
    summary: str  # text alternative: screen readers, transcript, the spoken one-liner
    unit: Unit | None
    x: XAxis | None
    series: list[Series]
    rows: list[Row]
    tiles: list[Tile]
    events: list[TimelineEvent]
    highlight: Highlight | None
    calculations: list[Calculation]
    sources: list[Citation]
    pinned: bool = False
    position: int = 0


class CanvasPanels(BaseModel):
    """A chat's canvas: its panels in position order."""

    panels: list[Visual]


class ProjectOverview(BaseModel):
    """The project's overview dashboard, built after ingestion. ``building`` while documents are being ingested or
    the overview is being rebuilt; ``none`` when there is nothing chartable (yet)."""

    panels: list[Visual]
    status: OverviewStatus


class CanvasOp(BaseModel):
    """An edit of a chat's canvas. ``move`` needs ``position`` (0-based, clamped to the canvas). ``add`` (additive)
    copies another visual of the same project (an overview panel, or another chat's) onto this canvas as a new panel,
    at ``position`` or at the end."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["remove", "pin", "unpin", "move", "add"]
    visual_id: str
    position: int | None = Field(default=None, ge=0)


# ------------------------------------------------------------------ events (both transports)


class VisualEvent(BaseModel):
    """A visual's progress: ``preparing`` (show a skeleton), ``ready`` (with the visual), ``failed`` (with a detail;
    the conversation is unaffected). Sent as SSE ``event: visual`` or WebSocket ``{"type": "visual", …}``."""

    name: ClassVar[str] = "visual"

    phase: VisualPhase
    visual_id: str
    visual: Visual | None = None
    detail: str | None = None
    # Additive (§12.1, "A visual already on the canvas"): a ready visual that took the place of a panel showing the
    # same data (its id), and how: "same" or "covered" (nothing new on screen), "extends" (more than before).
    replaces: str | None = None
    reuse: Literal["same", "covered", "extends"] | None = None

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def ws_message(self) -> dict[str, Any]:
        return {"type": self.name, **self.payload()}


class CanvasEvent(BaseModel):
    """The chat's whole canvas after a change (added, removed, pinned, moved)."""

    name: ClassVar[str] = "canvas"

    panels: list[Visual]

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def ws_message(self) -> dict[str, Any]:
        return {"type": self.name, **self.payload()}
