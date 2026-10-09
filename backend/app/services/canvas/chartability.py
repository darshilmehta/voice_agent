# ruff: noqa: RUF002  (documents print en dashes, minus, times and divide signs: used on purpose)
"""How a typed dataset can be charted (docs/DESIGN.md §12.1, workstream 2). Pure.

- **time series**: two or more periods of one granularity on an axis (FY23 | FY24 headers, Q1–Q4 FY24 rows) and a
  series with numbers for at least two of them. Confidence grows with the number of periods (two periods are as much
  a comparison as a trend).
- **kpi**: a short list of headline metrics in different units (₹ crore, %, x, count), typically with the two years
  side by side: tiles with their change.
- **composition**: parts of a whole: a total row the parts add up to (checked when typing), or percentage shares
  that sum to 100.
- **categorical**: categories (segments, plants, districts) compared on one measure in one unit.
- **none**: text tables (glossaries, contacts, risks), or fewer than two numbers.

``period_order`` lists the period labels on the period axis, chronologically (the table may print FY24 before FY23).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ...domain.datasets import (
    NUMERIC_COLUMN_TYPES,
    Chartability,
    ChartKind,
    ChartOption,
    DatasetColumn,
    DatasetRow,
    PeriodInfo,
    TypedDataset,
)

PREFERENCE: tuple[ChartKind, ...] = ("time_series", "kpi", "composition", "categorical", "timeline")
KPI_MAX_ROWS = 12


def series_columns(ds: TypedDataset) -> list[DatasetColumn]:
    """Columns that hold measured values (not the document's change column, years or notes)."""
    return [c for c in ds.columns if c.type in NUMERIC_COLUMN_TYPES]


def _ordered_periods(periods: Iterable[PeriodInfo]) -> tuple[str | None, list[PeriodInfo]]:
    """The largest same-granularity group, chronological, one per label."""
    groups: dict[str, dict[str, PeriodInfo]] = {}
    for p in periods:
        groups.setdefault(p.granularity, {}).setdefault(p.label, p)
    if not groups:
        return None, []
    granularity, members = max(groups.items(), key=lambda kv: (len(kv[1]), kv[0] != "date"))
    return granularity, sorted(members.values(), key=lambda p: p.sort_key)


def period_axis(ds: TypedDataset) -> tuple[str | None, str | None, list[PeriodInfo]]:
    """(axis, granularity, chronological periods): "columns" when the headers are periods, "rows" when the row
    labels are (aggregate rows such as "Full year FY24" left out)."""
    col_gran, col_periods = _ordered_periods(c.period for c in series_columns(ds) if c.period is not None)
    row_gran, row_periods = _ordered_periods(r.period for r in ds.rows if r.period is not None and r.type == "data")
    if len(col_periods) >= 2 and len(col_periods) >= len(row_periods):
        return "columns", col_gran, col_periods
    if len(row_periods) >= 2:
        return "rows", row_gran, row_periods
    return None, None, []


def _numbers(ds: TypedDataset, *, row: str | None = None, col: str | None = None) -> int:
    return sum(
        1 for v in ds.values if v.value is not None and (row is None or v.row == row) and (col is None or v.col == col)
    )


def _row_unit_kinds(ds: TypedDataset, rows: Iterable[DatasetRow]) -> list[str]:
    """Each row's unit kind (currency, percent, ratio, …) when all its measured values share one."""
    keys = {c.key for c in series_columns(ds)}
    out = []
    for r in rows:
        kinds = {
            v.unit.kind if v.unit else "none"
            for v in ds.values
            if v.row == r.key and v.col in keys and v.value is not None
        }
        if len(kinds) == 1:
            out.append(next(iter(kinds)))
    return out


def date_column(ds: TypedDataset) -> DatasetColumn | None:
    """A column of dates with a text column beside it: a timeline (the notice's important dates, a revision
    history)."""
    dates = [c for c in ds.columns if c.type == "date"]
    texts = [c for c in ds.columns if c.type in ("text", "label")]
    if dates and texts and len([r for r in ds.rows if r.type == "data"]) >= 2:
        return dates[0]
    return None


def classify(ds: TypedDataset) -> Chartability:
    numbers = [v for v in ds.values if v.value is not None]
    dates = date_column(ds)
    if len(numbers) < 2:
        if dates is not None:
            return Chartability(
                kind="timeline",
                confidence=0.8,
                options=[ChartOption(kind="timeline", confidence=0.8)],
                reasons=[f"dates in '{dates.label}' with descriptions"],
            )
        return Chartability(kind="none", confidence=0.95, reasons=["fewer than two numbers"])
    options: list[ChartOption] = []
    reasons: list[str] = []
    axis, granularity, periods = period_axis(ds)
    labels = {p.label for p in periods}
    data_rows = [r for r in ds.rows if r.type != "section"]
    cols = series_columns(ds)

    # time series
    if axis is not None:
        if axis == "columns":
            period_cols = [c for c in cols if c.period is not None and c.period.label in labels]
            series_ok = any(
                sum(1 for c in period_cols if (v := ds.value(r.key, c.key)) is not None and v.value is not None) >= 2
                for r in data_rows
            )
        else:
            period_rows = [r for r in data_rows if r.period is not None and r.period.label in labels]
            series_ok = any(
                sum(1 for r in period_rows if (v := ds.value(r.key, c.key)) is not None and v.value is not None) >= 2
                for c in cols
            )
        if series_ok:
            n = len(periods)
            confidence = 0.5 if n == 2 else 0.75 if n == 3 else min(0.95, 0.85 + 0.02 * (n - 4))
            options.append(ChartOption(kind="time_series", confidence=confidence))
            reasons.append(f"{n} {granularity} periods in the {axis}: {', '.join(p.label for p in periods)}")

    # kpi: headline metrics in different units
    if axis != "rows" and 2 <= len(data_rows) <= KPI_MAX_ROWS:
        units = _row_unit_kinds(ds, data_rows)
        distinct = len(set(units))
        if distinct >= 2 and len(units) >= 2:
            confidence = 0.8 + (0.05 if axis == "columns" else 0.0)
            options.append(ChartOption(kind="kpi", confidence=confidence))
            reasons.append(f"{len(units)} metrics in {distinct} different units")

    # composition: a checked total, or shares summing to 100%
    totals = [r for r in ds.rows if r.type == "total" and r.parts]
    category_totals = [r for r in totals if all((p := ds.row(k)) is not None and p.period is None for k in r.parts)]
    if category_totals:
        confidence = max(0.65, 0.88 - 0.05 * (len(category_totals) - 1))
        options.append(ChartOption(kind="composition", confidence=confidence))
        reasons.append(f"{len(category_totals)} total row(s) equal to the sum of their parts")
    else:
        for c in cols:
            if c.unit is None or c.unit.kind != "percent":
                continue
            shares = [
                v.value
                for r in data_rows
                if r.type == "data"
                and r.period is None
                and (v := ds.value(r.key, c.key)) is not None
                and v.value is not None
            ]
            if len(shares) >= 2 and abs(sum(shares) - 100) <= 0.06 * len(shares) + 1e-9:
                options.append(ChartOption(kind="composition", confidence=0.85))
                reasons.append(f"shares in '{c.label}' add up to 100%")
                break

    # categorical: categories compared on one measure
    categories = [r for r in data_rows if r.type == "data" and r.period is None]
    if len(categories) >= 2:
        for c in cols:
            vals = [v for r in categories if (v := ds.value(r.key, c.key)) is not None and v.value is not None]
            units = Counter(v.unit for v in vals)
            if len(vals) >= 2 and units and units.most_common(1)[0][1] >= 2:
                confidence = min(0.8, 0.6 + 0.05 * (len(categories) - 2))
                options.append(ChartOption(kind="categorical", confidence=confidence))
                reasons.append(f"{len(categories)} categories compared on '{c.label}'")
                break

    if dates is not None:
        options.append(ChartOption(kind="timeline", confidence=0.6))
        reasons.append(f"dates in '{dates.label}'")

    if not options:
        text_share = 1 - len(numbers) / max(1, len(ds.rows) * max(1, len(ds.columns) - 1))
        return Chartability(
            kind="none", confidence=0.9 if text_share > 0.5 else 0.7, reasons=["no series, periods or parts found"]
        )
    options.sort(key=lambda o: (-o.confidence, PREFERENCE.index(o.kind)))
    best = options[0]
    return Chartability(
        kind=best.kind,
        confidence=round(best.confidence, 2),
        options=[o.model_copy(update={"confidence": round(o.confidence, 2)}) for o in options],
        period_axis=axis,  # type: ignore[arg-type]
        period_order=[p.label for p in periods],
        granularity=granularity,
        reasons=reasons,
    )
