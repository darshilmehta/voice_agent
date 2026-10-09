"""``VisualSpec``: what to show, chosen from real datasets; ``resolve``: the validator that checks a spec against the
datasets and turns it into concrete series and x values (docs/DESIGN.md §12.1, workstream 3).

A spec names rows and columns by reference, ``"<dataset id>:<key>"`` (keys from ``TypedDataset``): it holds no
numbers. Series are rows or columns of a dataset; the x axis is the other one (a row series runs across the columns,
FY23 → FY24; a column series runs down the rows, Q1 → Q4 or segment by segment). ``x`` may name it explicitly:
``"<dataset>:period"`` / ``"<dataset>:columns"`` (the column headers) or ``"<dataset>:rows"`` / the label column's
key. ``periods`` keeps only those periods on the x axis; ``categories`` picks x items by reference (and orders them,
which a waterfall needs). Anything that doesn't resolve to the datasets is rejected with every problem listed: an
invented column, a period the table doesn't have, mixed units on one axis, too many series.

The planner fills the same schema, narrowed per request to enums of the candidate datasets' real references
(``planner.planner_schema``), so the model can only choose what exists; ``resolve`` still checks everything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...domain.canvas import CalcOp, Unit, VisualKind, XType
from ...domain.datasets import NUMERIC_COLUMN_TYPES, DatasetRow, DatasetValue, PeriodInfo, TypedDataset
from ...settings import Language
from .chartability import date_column

REF: Final = ":"
MAX_DATASETS: Final = 3
MAX_SERIES: Final = 8
MAX_X_FILTER: Final = 30
MAX_HIGHLIGHT: Final = 3
MAX_CALCULATIONS: Final = 4
MAX_TITLE: Final = 80
# Per kind: (series min, series max, x min, x max)
LIMITS: Final[dict[str, tuple[int, int, int, int]]] = {
    "line": (1, 4, 2, 24),
    "bar": (1, 4, 1, 12),
    "grouped_bar": (2, 4, 1, 12),
    "stacked_bar": (2, 6, 1, 12),
    "waterfall": (1, 1, 2, 12),
    "donut": (1, 1, 2, 8),
    "table": (1, 8, 1, 30),
    "kpi": (1, 6, 1, 12),
    "comparison": (1, 3, 1, 2),
    "timeline": (0, 1, 1, 20),
}
MAX_TILES: Final = 6
# Kinds that put every series on one value axis. Lines and bars may mix units: the renderer draws small multiples on
# a shared x axis (never a second y axis), each series with its own unit.
SINGLE_UNIT_KINDS: Final = frozenset({"grouped_bar", "stacked_bar", "waterfall", "donut"})


class SpecError(ValueError):
    """The spec doesn't resolve against the datasets. ``problems`` lists every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class CalcRequest(BaseModel):
    """A calculation to show: ``growth``/``cagr``/``diff`` of ``series`` from one x to another (default: first to
    last); ``ratio`` of ``series`` to ``other`` at ``to`` (default: the last x); ``share`` of ``series`` at ``to`` in
    the total of the series (the table's total row when it has one); ``sum`` of ``series`` over the x values."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: CalcOp
    series: str
    other: str | None = None
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None


class VisualSpec(BaseModel):
    """What to show. References are ``"<dataset id>:<key>"``; periods are canonical labels ("FY24", "Q3 FY24")."""

    model_config = ConfigDict(extra="forbid")

    kind: VisualKind
    datasets: list[str] = Field(min_length=1, max_length=MAX_DATASETS)
    x: str | None = None
    series: list[str] = Field(default_factory=list, max_length=MAX_SERIES)
    periods: list[str] = Field(default_factory=list, max_length=MAX_X_FILTER)
    categories: list[str] = Field(default_factory=list, max_length=MAX_X_FILTER)
    highlight: list[str] = Field(default_factory=list, max_length=MAX_HIGHLIGHT)
    calculations: list[CalcRequest] = Field(default_factory=list, max_length=MAX_CALCULATIONS)
    title: str | None = Field(default=None, max_length=MAX_TITLE)
    language: Language = "en"


# ------------------------------------------------------------------ resolved form

Axis = Literal["row", "col"]


@dataclass(slots=True)
class XItem:
    label: str  # the x value shown (Row.x)
    period: PeriodInfo | None
    order: tuple[int, ...]
    refs: set[str] = field(default_factory=set)  # "<dataset>:<key>" of what it stands for


@dataclass(slots=True)
class Point:
    value: DatasetValue
    dataset: TypedDataset


@dataclass(slots=True)
class ResolvedSeries:
    key: str
    label: str
    unit: Unit | None
    refs: list[str]  # "<dataset>:<key>" it was built from (several when merged across datasets)
    points: dict[str, Point]  # x label → point
    dataset_ids: list[str] = field(default_factory=list)
    total: Point | None = None  # donut / share: the table's total for this series, when it has one
    metric: str = ""  # the label before it was told apart from a same-named series ("Revenue from operations")
    source: str = ""  # what told it apart: the document's name or the table's title


@dataclass(slots=True)
class ResolvedCalc:
    op: CalcOp
    series: ResolvedSeries
    other: ResolvedSeries | None
    from_x: str | None
    to_x: str | None


@dataclass(slots=True)
class TimelineRow:
    dataset: TypedDataset
    row: DatasetRow
    date_col: str
    detail_col: str | None


@dataclass(slots=True)
class Resolved:
    spec: VisualSpec
    kind: VisualKind
    datasets: dict[str, TypedDataset]
    x_type: XType | None
    x_key: str
    x_label: str
    x_items: list[XItem]
    series: list[ResolvedSeries]
    highlight_x: list[str]
    highlight_series: list[str]
    calcs: list[ResolvedCalc]
    timeline: list[TimelineRow] = field(default_factory=list)

    @property
    def dataset_list(self) -> list[TypedDataset]:
        return list(self.datasets.values())


# ------------------------------------------------------------------ helpers


def split_ref(ref: str) -> tuple[str, str]:
    dataset, sep, key = ref.partition(REF)
    if not sep or not dataset or not key:
        raise ValueError(f"'{ref}' is not a '<dataset>{REF}<key>' reference")
    return dataset, key


def ref(dataset_id: str, key: str) -> str:
    return f"{dataset_id}{REF}{key}"


def same_unit(a: Unit | None, b: Unit | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (a.kind, a.currency, a.scale, a.label) == (b.kind, b.currency, b.scale, b.label)


def unit_name(unit: Unit | None) -> str:
    return (unit.label or unit.kind) if unit is not None else "no unit"


def series_axis(ds: TypedDataset, key: str) -> Axis | None:
    """Whether ``key`` names a row or a measured column of ``ds`` as a series; a key that is both is read the way the
    table runs (series across the period columns, or down the period rows)."""
    row = ds.row(key)
    col = ds.column(key)
    is_row = row is not None and row.type != "section"
    is_col = col is not None and col.type in NUMERIC_COLUMN_TYPES
    if is_row and is_col:
        return "col" if ds.chartability.period_axis == "rows" else "row"
    return "row" if is_row else "col" if is_col else None


def row_display(ds: TypedDataset, row: DatasetRow) -> str:
    """A row's label, with its section when another row has the same label ("Borrowings (Current liabilities)")."""
    if row.section and sum(1 for r in ds.rows if r.label == row.label) > 1:
        return f"{row.label} ({row.section})"
    return row.label or row.key


def _x_axis_from_spec(ds: TypedDataset, x: str | None) -> Axis | None:
    """The series axis implied by ``spec.x`` for this dataset (None: not about this dataset / not given)."""
    if x is None:
        return None
    try:
        dataset, key = split_ref(x)
    except ValueError:
        return None
    if dataset != ds.id:
        return None
    label_key = ds.columns[ds.label_column].key if ds.label_column is not None else None
    if key in ("columns", "column_headers"):
        return "row"  # x runs across the columns: series are rows
    if key in ("rows", "row_labels") or key == label_key:
        return "col"
    if key in ("period", "periods"):
        return "row" if ds.chartability.period_axis != "rows" else "col"
    return None


@dataclass(slots=True)
class _Candidate:
    label: str
    key: str  # row or column key on the x axis
    period: PeriodInfo | None
    order: int
    is_total: bool = False


def _x_candidates(
    ds: TypedDataset, axis: Axis, *, include_totals: bool, chosen: set[str] | None
) -> tuple[list[_Candidate], str | None]:
    """The x items a series of ``ds`` on ``axis`` runs over, and the measure they were narrowed to (a row series of
    a "Revenue FY24 | Revenue FY23 | EBITDA FY24 …" table runs over one measure's columns)."""
    measure: str | None = None
    if axis == "row":  # x = the measured columns
        cols = [c for c in ds.columns if c.type in NUMERIC_COLUMN_TYPES]
        if chosen:
            cols = [c for c in cols if c.key in chosen]
        else:
            measures = list(dict.fromkeys(c.measure for c in cols if c.measure))
            if len(measures) > 1:
                measure = measures[0]
                cols = [c for c in cols if c.measure == measure]
        labels = [c.period.label if c.period is not None else c.label for c in cols]
        if len(set(labels)) < len(labels):
            labels = [c.label for c in cols]
        return [_Candidate(label, c.key, c.period, c.index) for label, c in zip(labels, cols, strict=True)], measure
    rows = [
        r
        for r in ds.rows
        if r.type == "data" or (r.type == "total" and (include_totals or (chosen and r.key in chosen)))
    ]
    if chosen:
        rows = [r for r in rows if r.key in chosen]
    out = []
    for r in rows:
        label = r.period.label if r.period is not None and r.type == "data" else row_display(ds, r)
        out.append(_Candidate(label, r.key, r.period if r.type == "data" else None, r.index, r.type == "total"))
    labels = [c.label for c in out]
    if len(set(labels)) < len(labels):  # "FY24 (recommended)" and a "Full year FY24" row: keep the printed labels
        for c in out:
            c.label = row_display(ds, ds.row(c.key))  # type: ignore[arg-type]
    return out, measure


def _common_unit(points: Sequence[Point]) -> tuple[Unit | None, bool]:
    units: list[Unit | None] = []
    for p in points:
        if p.value.value is None:
            continue
        if not any(same_unit(p.value.unit, u) for u in units):
            units.append(p.value.unit)
    if not units:
        return None, True
    return units[0], len(units) == 1


# ------------------------------------------------------------------ resolve


def document_name(filename: str) -> str:
    """ "zephyra_investor_deck_q4fy24.pptx" → "Zephyra investor deck q4fy24": to tell two documents' series apart."""
    stem = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    return (stem[:1].upper() + stem[1:])[:40] if stem else ""


def resolve(
    spec: VisualSpec, datasets: Mapping[str, TypedDataset], *, filenames: Mapping[str, str] | None = None
) -> Resolved:
    """Check ``spec`` against ``datasets`` (by id) and resolve it. Raises SpecError listing every problem.
    ``filenames`` (document id → name) label same-named series from different documents."""
    problems: list[str] = []
    chosen: dict[str, TypedDataset] = {}
    for dataset_id in dict.fromkeys(spec.datasets):
        ds = datasets.get(dataset_id)
        if ds is None:
            problems.append(f"unknown dataset '{dataset_id}'")
        else:
            chosen[dataset_id] = ds
    if problems:
        raise SpecError(problems)
    if spec.x is not None:
        try:
            x_dataset, _ = split_ref(spec.x)
        except ValueError as e:
            raise SpecError([f"x: {e}"]) from None
        if x_dataset not in chosen or _x_axis_from_spec(chosen[x_dataset], spec.x) is None:
            raise SpecError(
                [
                    f"x '{spec.x}' is not an axis of the spec's datasets (use <dataset>:period, :rows, "
                    ":columns or the label column's key)"
                ]
            )
    if spec.kind == "timeline":
        return _resolve_timeline(spec, chosen)

    lo_s, hi_s, lo_x, hi_x = LIMITS[spec.kind]
    if not spec.series:
        raise SpecError([f"a {spec.kind} needs at least one series"])
    if len(spec.series) > hi_s and spec.kind not in ("kpi",):
        problems.append(f"a {spec.kind} shows at most {hi_s} series ({len(spec.series)} given)")

    # categories by dataset (x items picked by reference)
    picked: dict[str, list[str]] = {}
    for c in spec.categories:
        try:
            dataset_id, key = split_ref(c)
        except ValueError as e:
            problems.append(str(e))
            continue
        if dataset_id not in chosen:
            problems.append(f"category '{c}': dataset '{dataset_id}' is not in the spec's datasets")
            continue
        picked.setdefault(dataset_id, []).append(key)

    # series
    raw_series: list[tuple[str, ResolvedSeries, Axis, list[_Candidate]]] = []
    axes: dict[str, Axis] = {}
    for s in dict.fromkeys(spec.series):
        try:
            dataset_id, key = split_ref(s)
        except ValueError as e:
            problems.append(str(e))
            continue
        ds = chosen.get(dataset_id)
        if ds is None:
            problems.append(f"series '{s}': dataset '{dataset_id}' is not in the spec's datasets")
            continue
        wanted = _x_axis_from_spec(ds, spec.x)
        axis: Axis | None
        if wanted is not None:
            row_ok = (r := ds.row(key)) is not None and r.type != "section"
            col_ok = (c := ds.column(key)) is not None and c.type in NUMERIC_COLUMN_TYPES
            if not (row_ok if wanted == "row" else col_ok):
                what = "row" if wanted == "row" else "measured column"
                problems.append(f"series '{s}': '{key}' is not a {what} of '{ds.title}' (x is '{spec.x}')")
                continue
            axis = wanted
        else:
            axis = series_axis(ds, key)
        if axis is None:
            problems.append(f"series '{s}': '{key}' is not a row or a measured column of '{ds.title}'")
            continue
        if dataset_id in axes and axes[dataset_id] != axis:
            problems.append(f"series of '{ds.title}' mix rows and columns")
            continue
        axes[dataset_id] = axis
        chosen_x = set(picked.get(dataset_id, [])) or None
        if chosen_x is not None:
            on_axis = {c.key for c in ds.columns} if axis == "row" else {r.key for r in ds.rows}
            bad = sorted(k for k in chosen_x if k not in on_axis)
            if bad:
                problems.append(
                    f"categories {', '.join(ref(dataset_id, k) for k in bad)} are not on the x axis of '{ds.title}'"
                )
                continue
        candidates, measure = _x_candidates(ds, axis, include_totals=spec.kind == "table", chosen=chosen_x)
        base = ds.row(key) if axis == "row" else ds.column(key)
        label = (row_display(ds, base) if axis == "row" else base.label) or key  # type: ignore[arg-type]
        if measure:
            label = f"{label} · {measure}"
        points: dict[str, Point] = {}
        for cand in candidates:
            value = ds.value(key, cand.key) if axis == "row" else ds.value(cand.key, key)
            if value is not None:
                points[cand.label] = Point(value, ds)
        total = None
        if axis == "col":
            totals = [r for r in ds.rows if r.type == "total" and r.parts and not r.period]
            if len(totals) == 1 and (tv := ds.value(totals[0].key, key)) is not None and tv.value is not None:
                parts = set(totals[0].parts)
                if {c.key for c in candidates if not c.is_total} <= parts:
                    total = Point(tv, ds)
        rs = ResolvedSeries(
            key=key,
            label=label,
            unit=None,
            refs=[s],
            points=points,
            dataset_ids=[dataset_id],
            total=total,
            metric=label,
            source=ds.title,
        )
        raw_series.append((s, rs, axis, candidates))
    if problems:
        raise SpecError(problems)

    # x items: union over series
    x_items: dict[str, XItem] = {}
    for _, rs, _, candidates in raw_series:
        ds_id = rs.dataset_ids[0]
        for cand in candidates:
            item = x_items.get(cand.label)
            order = (
                (*cand.period.sort_key, 0) if cand.period is not None else (0, 0, len(x_items) if item is None else 0)
            )
            if item is None:
                item = x_items[cand.label] = XItem(cand.label, cand.period, order)
            item.refs.add(ref(ds_id, cand.key))
    periodic = bool(x_items) and all(i.period is not None for i in x_items.values())
    # explicit order for picked categories (waterfalls run in the order asked)
    if spec.categories and not periodic:
        rank = {c: n for n, c in enumerate(spec.categories)}
        for item in x_items.values():
            item.order = (min((rank.get(r, 10**6) for r in item.refs), default=10**6),)
    elif not periodic:
        for n, item in enumerate(x_items.values()):
            item.order = (n,)

    # period filter
    available_periods = {
        p.label
        for ds in chosen.values()
        for p in [*(c.period for c in ds.columns if c.period), *(r.period for r in ds.rows if r.period)]
        if p is not None
    }
    for p in spec.periods:
        if p not in available_periods:
            problems.append(f"period '{p}' is not in the datasets (they have: {', '.join(sorted(available_periods))})")
    if problems:
        raise SpecError(problems)
    items = list(x_items.values())
    if spec.periods and periodic:
        items = [i for i in items if i.period is not None and i.period.label in spec.periods]
    elif periodic and not spec.categories and spec.kind not in ("table",):
        # the dominant granularity only (quarters, not the "Full year" row; years, not a stray quarter)
        grans: dict[str, int] = {}
        for i in items:
            grans[i.period.granularity] = grans.get(i.period.granularity, 0) + 1  # type: ignore[union-attr]
        main = max(grans.items(), key=lambda kv: kv[1])[0]
        items = [i for i in items if i.period.granularity == main]  # type: ignore[union-attr]
    items.sort(key=lambda i: i.order)
    if not items:
        raise SpecError(["no x values left after the filters"])
    explicit = bool(spec.periods or spec.categories)
    if len(items) > hi_x:
        if explicit:
            problems.append(f"a {spec.kind} shows at most {hi_x} x values ({len(items)} chosen)")
        else:
            items = items[-hi_x:] if periodic else items[:hi_x]
    if len(items) < lo_x:
        problems.append(f"a {spec.kind} needs at least {lo_x} x values ({len(items)} available)")
    labels = {i.label for i in items}

    # series: keep chosen x, units, merge same-named series across datasets
    series: list[ResolvedSeries] = []
    for _, rs, _, _ in raw_series:
        rs.points = {x: p for x, p in rs.points.items() if x in labels}
        unit, single = _common_unit(list(rs.points.values()))
        if not single and spec.kind != "table":
            problems.append(f"series '{rs.refs[0]}' mixes units")
        rs.unit = unit
        merged = next(
            (
                s
                for s in series
                if s.label == rs.label
                and same_unit(s.unit, rs.unit)
                and not set(s.points) & set(rs.points)
                and s.dataset_ids[0] != rs.dataset_ids[0]
            ),
            None,
        )
        if merged is not None:
            merged.points.update(rs.points)
            merged.refs += rs.refs
            merged.dataset_ids += rs.dataset_ids
            continue
        series.append(rs)
    if not any(p.value.value is not None for s in series for p in s.points.values()):
        problems.append("the chosen series have no numbers at the chosen x values")
    # unique keys and labels
    used: set[str] = set()
    labels_seen: dict[str, int] = {}
    for s in series:
        labels_seen[s.label] = labels_seen.get(s.label, 0) + 1
    documents_of: dict[str, set[str]] = {}
    for s in series:
        documents_of.setdefault(s.label, set()).add(chosen[s.dataset_ids[0]].document_id)
    for s in series:
        key = s.key
        n = 2
        while key in used:
            key, n = f"{s.key}_{n}", n + 1
        used.add(key)
        s.key = key
        ds = chosen[s.dataset_ids[0]]
        name = document_name((filenames or {}).get(ds.document_id, "")) if len(documents_of[s.label]) > 1 else ""
        s.source = name or ds.title
        if labels_seen[s.label] > 1:  # the same metric from two tables: name the document (or the table)
            s.label = f"{s.label} ({s.source})"
    if len(series) < lo_s:
        problems.append(f"a {spec.kind} needs at least {lo_s} series ({len(series)} given)")
    if spec.kind in SINGLE_UNIT_KINDS and len({unit_name(s.unit) + str(s.unit) for s in series}) > 1:
        names = ", ".join(sorted({unit_name(s.unit) for s in series}))
        problems.append(f"a {spec.kind} plots one unit; these series are in {names}")
    if spec.kind in ("stacked_bar", "donut"):
        negative = [
            s.label for s in series for p in s.points.values() if p.value.value is not None and p.value.value < 0
        ]
        if negative:
            problems.append(f"a {spec.kind} can't show negative values ({negative[0]})")
        if any(s.unit is not None and s.unit.kind in ("ratio",) for s in series):
            problems.append(f"a {spec.kind} adds values up; ratios don't add")
    if spec.kind == "donut" and periodic:
        problems.append("a donut shows parts of a whole (categories), not periods")
    if spec.kind == "line" and not periodic:
        problems.append("a line chart needs periods on the x axis")
    if spec.kind == "comparison" and len(items) == 1 and len(series) < 2:
        problems.append("a comparison needs two x values or two series")
    if spec.kind == "kpi" and len(spec.series) > MAX_TILES:
        problems.append(f"at most {MAX_TILES} KPI tiles ({len(spec.series)} given)")

    # highlight
    highlight_x: list[str] = []
    highlight_series: list[str] = []
    by_ref = {r: s for s in series for r in s.refs}
    for h in spec.highlight:
        if h in labels:
            highlight_x.append(h)
            continue
        item = next((i for i in items if h in i.refs), None)
        if item is not None:
            highlight_x.append(item.label)
            continue
        if h in by_ref:
            highlight_series.append(by_ref[h].key)
            continue
        problems.append(f"highlight '{h}' is not an x value or a series of this visual")

    # calculations
    calcs: list[ResolvedCalc] = []
    for c in spec.calculations:
        s = by_ref.get(c.series)
        if s is None:
            problems.append(f"calculation {c.op}: '{c.series}' is not one of the visual's series")
            continue
        other = by_ref.get(c.other) if c.other else None
        if c.other and other is None:
            problems.append(f"calculation {c.op}: '{c.other}' is not one of the visual's series")
            continue
        if c.op == "ratio" and other is None:
            problems.append("calculation ratio: needs 'other', the series to divide by")
            continue
        xs = []
        for x in (c.from_, c.to):
            if x is None:
                xs.append(None)
                continue
            if x in labels:
                xs.append(x)
                continue
            item = next((i for i in items if x in i.refs), None)
            if item is None:
                problems.append(f"calculation {c.op}: '{x}' is not an x value of this visual")
            xs.append(item.label if item else None)
        calcs.append(ResolvedCalc(c.op, s, other, xs[0], xs[1]))
    if problems:
        raise SpecError(problems)

    first = chosen[series[0].dataset_ids[0]]
    x_type: XType = "category"
    if periodic:
        x_type = "date" if all(i.period.kind == "date" for i in items) else "period"  # type: ignore[union-attr]
    axis = raw_series[0][2]
    if axis == "row":
        x_key, x_label = ("period", "Period") if periodic else ("column", "")
    else:
        label_col = first.columns[first.label_column] if first.label_column is not None else None
        x_key = label_col.key if label_col else "row"
        x_label = label_col.label if label_col else ""
    if periodic and x_label in ("", "Particulars", "Metric", "Item"):
        x_key, x_label = "period", "Period"
    return Resolved(
        spec=spec,
        kind=spec.kind,
        datasets=chosen,
        x_type=x_type,
        x_key=x_key,
        x_label=x_label,
        x_items=items,
        series=series,
        highlight_x=highlight_x,
        highlight_series=highlight_series,
        calcs=calcs,
    )


def _resolve_timeline(spec: VisualSpec, chosen: dict[str, TypedDataset]) -> Resolved:
    problems: list[str] = []
    ds = next(iter(chosen.values()))
    date_col = None
    if spec.series:
        try:
            dataset_id, key = split_ref(spec.series[0])
            col = chosen[dataset_id].column(key) if dataset_id in chosen else None
            if col is not None and col.type == "date":
                ds, date_col = chosen[dataset_id], col
            else:
                problems.append(f"timeline: '{spec.series[0]}' is not a column of dates")
        except ValueError as e:
            problems.append(str(e))
    else:
        found = date_column(ds)
        if found is None:
            problems.append(f"timeline: '{ds.title}' has no column of dates")
        date_col = found
    if problems or date_col is None:
        raise SpecError(problems or ["timeline: no dates"])
    picked = [split_ref(c)[1] for c in spec.categories if c.startswith(ds.id + REF)]
    rows = [r for r in ds.rows if r.type == "data" and (not picked or r.key in picked)]
    detail = next((c.key for c in ds.columns if c.type == "text" and c.key != date_col.key), None)
    rows = [r for r in rows if ds.text(r.key, date_col.key) is not None]
    if not rows:
        raise SpecError(["timeline: no dated rows"])
    if len(rows) > LIMITS["timeline"][3]:
        if picked:
            raise SpecError([f"a timeline shows at most {LIMITS['timeline'][3]} events"])
        rows = rows[: LIMITS["timeline"][3]]
    return Resolved(
        spec=spec,
        kind="timeline",
        datasets={ds.id: ds},
        x_type=None,
        x_key="date",
        x_label=date_col.label,
        x_items=[],
        series=[],
        highlight_x=[],
        highlight_series=[],
        calcs=[],
        timeline=[TimelineRow(ds, r, date_col.key, detail) for r in rows],
    )
