"""The project overview: a few panels built from the most chartable tables as soon as documents are typed
(docs/DESIGN.md §12.1, workstream 9). Pure: datasets in, specs out (built and checked like any other visual).

1. **KPI tiles**: the best KPI table (headline metrics in mixed units, two periods side by side): its first few
   metrics at the latest period, each with its change from the period before.
2. **Trend**: the time series with the most periods, its first one or two amounts as lines. Tables that continue
   each other (the FY23 and FY24 quarterly tables) become one series over every period.
3. **Composition**: parts of a whole: segment revenue at the latest period (or printed shares) as a donut.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ...domain.datasets import ChartKind, TypedDataset
from .chartability import series_columns
from .spec import MAX_DATASETS, SpecError, VisualSpec, ref

KPI_TILES = 4


def _option(ds: TypedDataset, kind: ChartKind) -> float:
    return next((o.confidence for o in ds.chartability.options if o.kind == kind), 0.0)


def _doc_order(ds: TypedDataset) -> tuple[int, int]:
    return (ds.page_start or 0, ds.table_index)


def kpi_spec(ds: TypedDataset, language: str, tiles: int = KPI_TILES) -> VisualSpec | None:
    periods = ds.chartability.period_order
    if ds.chartability.period_axis != "columns" or not periods:
        return None
    cols = [c for c in series_columns(ds) if c.period is not None and c.period.label == periods[-1]]
    if not cols:
        return None
    rows = [
        r
        for r in ds.rows
        if r.type == "data" and (v := ds.value(r.key, cols[0].key)) is not None and v.value is not None
    ][:tiles]
    if not rows:
        return None
    return VisualSpec(
        kind="kpi",
        datasets=[ds.id],
        series=[ref(ds.id, r.key) for r in rows],
        periods=periods[-2:],
        language=language,  # type: ignore[arg-type]
    )


def continuations(ds: TypedDataset, pool: Sequence[TypedDataset]) -> list[TypedDataset]:
    """``ds`` and the tables of the same document that continue it: same columns, periods in rows of the same
    granularity, no period in common (the FY23 and FY24 quarterly tables). Chronological."""
    ch = ds.chartability
    keys = {c.key for c in series_columns(ds)}

    def continues(o: TypedDataset) -> bool:
        oc = o.chartability
        return (
            o.document_id == ds.document_id
            and oc.period_axis == "rows" == ch.period_axis
            and oc.granularity == ch.granularity
            and {c.key for c in series_columns(o)} == keys
            and not set(oc.period_order) & set(ch.period_order)
        )

    group = [ds, *(o for o in pool if o.id != ds.id and continues(o))]
    return sorted(group, key=_first_period_key)[:MAX_DATASETS]


def _first_period_key(d: TypedDataset) -> tuple[int, int]:
    rows = [r for r in d.rows if r.period is not None and r.period.label in d.chartability.period_order]
    return min((r.period.sort_key for r in rows if r.period), default=(0, 0))


def trend_spec(ds: TypedDataset, pool: Sequence[TypedDataset], language: str) -> VisualSpec | None:
    ch = ds.chartability
    if ch.period_axis == "rows":
        group = continuations(ds, pool)
    elif ch.period_axis == "columns":
        group = [ds]
    else:
        return None
    if sum(len(d.chartability.period_order) for d in group) < 3:
        return None
    if ch.period_axis == "rows":
        cols = [c for c in series_columns(ds) if c.unit is not None and c.unit.kind == "currency"] or series_columns(ds)
        if not cols:
            return None
        unit = cols[0].unit
        keys = [c.key for c in cols if c.unit == unit][:2]
        series = [ref(d.id, k) for k in keys for d in group]
    elif ch.period_axis == "columns":
        rows = [r for r in ds.rows if r.type == "data" and r.unit is not None and r.unit.kind == "currency"]
        rows = rows or [r for r in ds.rows if r.type == "data"]
        if not rows:
            return None
        unit = rows[0].unit
        series = [ref(ds.id, r.key) for r in rows if r.unit == unit][:2]
    else:
        return None
    return VisualSpec(kind="line", datasets=[d.id for d in group], series=series, language=language)  # type: ignore[arg-type]


def composition_spec(ds: TypedDataset, language: str) -> VisualSpec | None:
    totals = [r for r in ds.rows if r.type == "total" and r.parts and r.period is None]
    cols = series_columns(ds)
    if totals:
        total = totals[-1]
        parts = [p for p in total.parts if (row := ds.row(p)) is not None and row.period is None]
        if not 2 <= len(parts) <= 8:
            return None
        latest = ds.chartability.period_order[-1] if ds.chartability.period_order else None
        additive = [c for c in cols if c.unit is None or c.unit.kind in ("currency", "count", "none")]
        candidates = [c for c in additive if latest is None or c.period is None or c.period.label == latest]
        if not candidates:
            return None
        col = candidates[0]
        return VisualSpec(
            kind="donut",
            datasets=[ds.id],
            series=[ref(ds.id, col.key)],
            categories=[ref(ds.id, p) for p in parts],
            language=language,  # type: ignore[arg-type]
        )
    shares = [c for c in cols if c.unit is not None and c.unit.kind == "percent"]
    if shares:
        return VisualSpec(kind="donut", datasets=[ds.id], series=[ref(ds.id, shares[0].key)], language=language)  # type: ignore[arg-type]
    return None


def overview_specs(
    datasets: Sequence[TypedDataset],
    *,
    language: str = "en",
    max_panels: int = 3,
    check: Callable[[VisualSpec], bool] | None = None,
) -> list[VisualSpec]:
    """Up to ``max_panels`` specs: KPI tiles, a trend, a composition (each from a different table). ``check`` tells
    whether a spec builds (the caller resolves and builds it); a spec that doesn't is skipped for the next best."""
    ok = check or (lambda _spec: True)
    used: set[str] = set()
    specs: list[VisualSpec] = []

    def take(candidates: list[TypedDataset], make: Callable[[TypedDataset], VisualSpec | None]) -> None:
        for ds in candidates:
            if ds.id in used:
                continue
            try:
                spec = make(ds)
            except (SpecError, ValueError):
                spec = None
            if spec is not None and ok(spec):
                specs.append(spec)
                used.update(spec.datasets)
                return

    kpis = sorted((d for d in datasets if _option(d, "kpi") >= 0.7), key=lambda d: (-_option(d, "kpi"), _doc_order(d)))
    take(kpis, lambda d: kpi_spec(d, language))
    trends = sorted(
        (d for d in datasets if _option(d, "time_series") >= 0.5),
        key=lambda d: (
            -sum(len(o.chartability.period_order) for o in continuations(d, datasets))
            if d.chartability.period_axis == "rows"
            else -len(d.chartability.period_order),
            -_option(d, "time_series"),
            _doc_order(d),
        ),
    )
    take(trends, lambda d: trend_spec(d, datasets, language))

    def composition_rank(d: TypedDataset) -> tuple[float, int, tuple[int, int]]:
        totals = [r for r in d.rows if r.type == "total" and r.parts]
        named_total = any(r.label.strip().lower() in ("total", "कुल", "grand total") for r in totals)
        return (
            -(_option(d, "composition") + (0.1 if named_total else 0) - 0.05 * max(0, len(totals) - 1)),
            0,
            _doc_order(d),
        )

    compositions = sorted((d for d in datasets if _option(d, "composition") >= 0.6), key=composition_rank)
    take(compositions, lambda d: composition_spec(d, language))
    return specs[:max_panels]
