# ruff: noqa: RUF001  (documents print en dashes, minus, times and divide signs: used on purpose)
"""From a resolved spec to a ``Visual`` (contract v1), and the grounding check (docs/DESIGN.md §12.1, workstream 5).

**Grounding guarantee.** Every number in a Visual is either the value of a table cell, carried with a ``CellRef``
whose printed ``text`` parses back to that number, or the value of a ``Calculation`` listed in ``calculations``,
computed here by ``calculator`` from CellRef inputs. That covers row values, tiles, deltas, the summary, the title and
the subtitle (their numbers are cell values, calculation values or numbers printed in the labels, such as the "2024"
of "31 Mar 2024"). ``check_grounding`` verifies a finished Visual against exactly this and is run on every visual
before it is stored; the tests run it over every spec the corpus datasets allow.

Sources: each dataset used becomes one ``Citation`` (the table's chunk). A dataset whose chunk is among the turn's
sources keeps that source's id (the spoken answer's [S2] and the chart's cells point to the same S2); others get the
next free ids.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

from ...domain.canvas import (
    Calculation,
    CellRef,
    Delta,
    Highlight,
    Row,
    Series,
    Tile,
    TimelineEvent,
    Unit,
    Visual,
    XAxis,
)
from ...domain.datasets import DatasetValue, TypedDataset
from ...domain.projects import Citation
from ..sources import snippet
from . import calculator as calc
from .calculator import CalculationError, Operand, format_value
from .parsing import Quantity, localized_label, parse_period, parse_quantity
from .spec import Point, Resolved, ResolvedSeries, SpecError, same_unit

_SOURCE_ID = re.compile(r"^S(\d+)$")


class SourceBook:
    """Citation numbering for one visual: reuses the turn's ids for the same table chunk, then continues after them."""

    def __init__(self, filenames: Mapping[str, str], existing: Sequence[Citation] = ()) -> None:
        self.filenames = filenames
        self.existing = list(existing)
        self._ids: dict[str, str] = {}  # dataset id → source id
        self._citations: dict[str, Citation] = {}
        numbers = [int(m.group(1)) for c in self.existing if (m := _SOURCE_ID.match(c.source_id))]
        self._next = max(numbers, default=0) + 1

    def source_id(self, ds: TypedDataset) -> str:
        if ds.id in self._ids:
            return self._ids[ds.id]
        chunk_id = ds.chunk_id or f"{ds.document_id}:v{ds.version}:table{ds.table_index}"
        reused = next((c for c in self.existing if ds.chunk_id and c.chunk_id == ds.chunk_id), None)
        if reused is not None:
            source_id = reused.source_id
        else:
            source_id = f"S{self._next}"
            self._next += 1
        self._ids[ds.id] = source_id
        self._citations[source_id] = Citation(
            source_id=source_id,
            document_id=ds.document_id,
            filename=self.filenames.get(ds.document_id, ""),
            page_start=ds.page_start,
            page_end=ds.page_end,
            chunk_id=chunk_id,
            snippet=snippet(_table_text(ds)),
        )
        return source_id

    def citations(self) -> list[Citation]:
        return sorted(self._citations.values(), key=lambda c: int(c.source_id[1:]) if c.source_id[1:].isdigit() else 0)


def _table_text(ds: TypedDataset) -> str:
    parts = [ds.title]
    for r in ds.rows[:6]:
        cells = [v.text for v in ds.values if v.row == r.key][:4]
        parts.append(f"{r.label}: {' | '.join(cells)}" if cells else r.label)
    return " · ".join(p for p in parts if p)


def cell_ref(book: SourceBook, ds: TypedDataset, value: DatasetValue) -> CellRef:
    return CellRef(
        source_id=book.source_id(ds),
        document_id=ds.document_id,
        table_id=ds.table_id,
        page=ds.page_start,
        row=value.cell_row,
        col=value.cell_col,
        text=value.text,
    )


# ------------------------------------------------------------------ text (EN / HI)

_TEXT = {
    "en": {
        "trend": "{series} went from {a} in {xa} to {b} in {xb} ({growth}).",
        "trend_flat": "{series}: {a} in {xa} and {b} in {xb}.",
        "more_series": " Also shown: {names}.",
        "bars": "{series}: highest {top} ({top_v}), lowest {low} ({low_v}).",
        "donut": "{top} is the largest part of {series}: {top_v} ({share}).",
        "donut_no_share": "{top} is the largest part of {series}: {top_v}.",
        "kpi": "{items}.",
        "comparison": "{items}.",
        "table": "Figures from {title}.",
        "waterfall": "From {start} ({start_v}) to {end} ({end_v}).",
        "timeline": "Key dates from {first} to {last}.",
        "highest": "Highest",
        "lowest": "Lowest",
        "total": "Total",
        "increase": "Increase",
        "decrease": "Decrease",
        "period": "Period",
        "page": "p.",
        "growth_word": "{v}",
        "vs": "{a} vs {b}",
        "gap": "{b} minus {a} ({x})",
    },
    "hi": {
        "trend": "{series} {xa} में {a} से {xb} में {b} रहा ({growth})।",
        "trend_flat": "{series}: {xa} में {a} और {xb} में {b}।",
        "more_series": " साथ में: {names}।",
        "bars": "{series}: सबसे अधिक {top} ({top_v}), सबसे कम {low} ({low_v})।",
        "donut": "{series} में सबसे बड़ा हिस्सा {top} का है: {top_v} ({share})।",
        "donut_no_share": "{series} में सबसे बड़ा हिस्सा {top} का है: {top_v}।",
        "kpi": "{items}।",
        "comparison": "{items}।",
        "table": "{title} के आँकड़े।",
        "waterfall": "{start} ({start_v}) से {end} ({end_v}) तक।",
        "timeline": "{first} से {last} तक की मुख्य तिथियाँ।",
        "highest": "सबसे अधिक",
        "lowest": "सबसे कम",
        "total": "कुल",
        "increase": "बढ़त",
        "decrease": "कमी",
        "period": "अवधि",
        "page": "पृ.",
        "growth_word": "{v}",
        "vs": "{a} बनाम {b}",
        "gap": "{b} और {a} का अंतर ({x})",
    },
}


def _t(language: str, key: str, **kw: object) -> str:
    return _TEXT["hi" if language == "hi" else "en"][key].format(**kw)


def _show(value: float, unit: Unit | None, decimals: int, language: str) -> str:
    return format_value(value, unit, decimals, language)


def _show_point(p: Point, language: str) -> str:
    v = p.value
    return _show(v.value or 0.0, v.unit, v.decimals, language)


def _show_calc(c: Calculation, language: str) -> str:
    decimals = (
        1
        if c.unit is not None and c.unit.kind == "percent"
        else 2
        if c.unit is not None and c.unit.kind == "ratio"
        else 0
    )
    if c.op in ("diff", "sum"):
        decimals = max((_decimals(i.text) for i in c.inputs), default=0)
    text = _show(c.value, c.unit, decimals, language)
    return ("+" + text) if c.op in ("growth", "diff", "cagr") and c.value > 0 else text


def _decimals(text: str) -> int:
    q = parse_quantity(text)
    return q.decimals if isinstance(q, Quantity) else 0


# ------------------------------------------------------------------ operands and calculations


def _operand(book: SourceBook, p: Point, label: str) -> Operand:
    v = p.value
    period = None
    if v.value is None:
        raise CalculationError(f"no number at {label}")
    row = p.dataset.row(v.row)
    col = p.dataset.column(v.col)
    period = (row.period if row is not None and row.period else None) or (col.period if col is not None else None)
    return Operand(
        value=v.value,
        unit=v.unit,
        cell=cell_ref(book, p.dataset, v),
        decimals=v.decimals,
        label=label,
        period=period,
    )


def _delta_calc(
    book: SourceBook, s: ResolvedSeries, old_x: str, new_x: str, language: str
) -> tuple[Calculation, str] | None:
    """The change from old to new: growth for amounts and counts, percentage points for percentages, a plain
    difference for ratios. None when it can't be computed (a zero or negative base)."""
    a, b = s.points.get(old_x), s.points.get(new_x)
    if a is None or b is None or a.value.value is None or b.value.value is None:
        return None
    try:
        oa, ob = _operand(book, a, old_x), _operand(book, b, new_x)
        if (s.unit is not None and s.unit.kind in ("percent", "ratio")) or (s.unit and s.unit.label in ("bps", "pts")):
            c = calc.diff(oa, ob, series=s.label, language=language)
            return c, "pp" if s.unit is not None and s.unit.kind == "percent" else "abs"
        c = calc.growth(oa, ob, series=s.label, language=language)
        return c, "pct"
    except CalculationError:
        return None


def _requested(book: SourceBook, r: Resolved, language: str) -> list[Calculation]:
    xs = [i.label for i in r.x_items]
    out: list[Calculation] = []
    for rc in r.calcs:
        s = rc.series
        present = [x for x in xs if x in s.points and s.points[x].value.value is not None]
        if not present:
            raise SpecError([f"calculation {rc.op}: '{s.label}' has no numbers"])
        try:
            if rc.op in ("growth", "cagr", "diff"):
                a, b = rc.from_x or present[0], rc.to_x or present[-1]
                if a == b:
                    raise CalculationError(f"{rc.op} needs two different x values")
                fn = {"growth": calc.growth, "cagr": calc.cagr, "diff": calc.diff}[rc.op]
                pa, pb = s.points.get(a), s.points.get(b)
                if pa is None or pb is None:
                    raise CalculationError(f"{s.label} has no number at {a if pa is None else b}")
                out.append(fn(_operand(book, pa, a), _operand(book, pb, b), series=s.label, language=language))
            elif rc.op == "ratio":
                at = rc.to_x or present[-1]
                other = rc.other
                assert other is not None
                pa, pb = s.points.get(at), other.points.get(at)
                if pa is None or pb is None:
                    raise CalculationError(f"no numbers for both series at {at}")
                out.append(
                    calc.ratio(
                        _operand(book, pa, at),
                        _operand(book, pb, at),
                        series=s.label,
                        other=other.label,
                        at=at,
                        language=language,
                    )
                )
            elif rc.op == "share":
                at = rc.to_x or present[-1]
                part = s.points.get(at)
                if part is None:
                    raise CalculationError(f"{s.label} has no number at {at}")
                out.append(_share(book, s, at, part, language))
            elif rc.op == "sum":
                chosen = [
                    x
                    for x in present
                    if (rc.from_x is None or xs.index(x) >= xs.index(rc.from_x))
                    and (rc.to_x is None or xs.index(x) <= xs.index(rc.to_x))
                ]
                out.append(
                    calc.total([_operand(book, s.points[x], x) for x in chosen], series=s.label, language=language)
                )
        except CalculationError as e:
            raise SpecError([f"calculation {rc.op}: {e}"]) from None
    return out


def _share(book: SourceBook, s: ResolvedSeries, at: str, part: Point, language: str) -> Calculation:
    if s.total is not None:
        whole: Operand | list[Operand] = _operand(book, s.total, s.total.dataset.row(s.total.value.row).label)  # type: ignore[union-attr]
    else:
        whole = [_operand(book, p, x) for x, p in s.points.items() if p.value.value is not None]
    return calc.share(_operand(book, part, at), whole, total_label=s.label, language=language)


# ------------------------------------------------------------------ build


def build_visual(
    r: Resolved,
    *,
    visual_id: str,
    project_id: str,
    chat_id: str | None,
    filenames: Mapping[str, str],
    now: datetime,
    existing_sources: Sequence[Citation] = (),
) -> Visual:
    """The Visual for a resolved spec. Raises SpecError when a requested calculation or a waterfall's arithmetic
    doesn't hold on the data."""
    language = r.spec.language
    book = SourceBook(filenames, existing_sources)
    calculations = _requested(book, r, language)
    rows: list[Row] = []
    tiles: list[Tile] = []
    events: list[TimelineEvent] = []
    series: list[Series] = [Series(key=s.key, label=s.label, unit=s.unit) for s in r.series]
    x_axis: XAxis | None = None
    xs = [i.label for i in r.x_items]
    summary = ""

    if r.kind in ("line", "bar", "grouped_bar", "stacked_bar", "table", "donut"):
        x_axis = XAxis(key=r.x_key, label=_axis_label(r, language), type=r.x_type or "category")
        for x in xs:
            values: dict[str, float | None] = {}
            cells: dict[str, CellRef | None] = {}
            for s in r.series:
                p = s.points.get(x)
                if p is not None and p.value.value is not None:
                    values[s.key], cells[s.key] = p.value.value, cell_ref(book, p.dataset, p.value)
                else:
                    values[s.key], cells[s.key] = None, None
            if r.kind == "donut" and values[r.series[0].key] is None:
                continue
            rows.append(Row(x=x, values=values, cells=cells))
        if r.kind == "donut":
            summary, shares = _donut(book, r, language)
            calculations += shares
        elif r.kind == "table":
            summary = _t(language, "table", title=r.dataset_list[0].title)
        elif r.x_type in ("period", "date"):
            summary, implied = _trend_summary(book, r, language)
            calculations += implied
        else:
            summary = _bars_summary(r, language)
    elif r.kind == "waterfall":
        x_axis = XAxis(key=r.x_key, label=_axis_label(r, language), type=r.x_type or "category")
        rows, series, bridge, summary = _waterfall(book, r, language)
        calculations.append(bridge)
    elif r.kind == "kpi":
        tiles, deltas, summary = _kpi_tiles(book, r, language)
        calculations += deltas
        series = []
    elif r.kind == "comparison":
        tiles, deltas, summary = _comparison_tiles(book, r, language)
        calculations += deltas
        series = []
    elif r.kind == "timeline":
        events = _events(book, r)
        summary = _t(language, "timeline", first=events[0].date, last=events[-1].date)

    if calculations and r.kind not in ("kpi", "comparison", "donut", "waterfall") and r.calcs:
        summary += " " + "; ".join(f"{c.label}: {_show_calc(c, language)}" for c in calculations[: len(r.calcs)]) + "."
    calculations = _dedupe(calculations)
    highlight = _highlight(r, language) if rows else None  # tiles and events have nothing to highlight on an axis
    unit = _visual_unit(r)
    title = _title(r, language)
    visual = Visual(
        id=visual_id,
        chat_id=chat_id,
        project_id=project_id,
        created_at=now,
        updated_at=now,
        kind=r.kind,
        title=title,
        subtitle=_subtitle(r, filenames, unit, language),
        language=language,
        summary=summary.strip(),
        unit=unit,
        x=x_axis,
        series=series,
        rows=rows,
        tiles=tiles,
        events=events,
        highlight=highlight,
        calculations=calculations,
        sources=book.citations(),
    )
    problems = check_grounding(visual)
    if problems:  # a bug, not bad input: never show an ungrounded number
        raise AssertionError("ungrounded visual: " + "; ".join(problems[:5]))
    return visual


def _dedupe(calcs: list[Calculation]) -> list[Calculation]:
    seen: set[tuple[object, ...]] = set()
    out = []
    for c in calcs:
        key = (c.op, c.value, tuple((i.table_id, i.row, i.col) for i in c.inputs))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _axis_label(r: Resolved, language: str) -> str:
    if r.x_key == "period":
        return _t(language, "period")
    return r.x_label


def _visual_unit(r: Resolved) -> Unit | None:
    units = [s.unit for s in r.series]
    if units and all(same_unit(units[0], u) for u in units):
        return units[0]
    return None


def _subtitle(r: Resolved, filenames: Mapping[str, str], unit: Unit | None, language: str) -> str | None:
    parts: list[str] = []
    for ds in r.dataset_list:
        name = filenames.get(ds.document_id, "")
        page = ""
        if ds.page_start is not None:
            pages = str(ds.page_start) if ds.page_end in (None, ds.page_start) else f"{ds.page_start}–{ds.page_end}"
            page = f"{_t(language, 'page')} {pages}"
        parts.append(" ".join(p for p in (name, page) if p))
    text = "; ".join(dict.fromkeys(p for p in parts if p))
    label = localized_label(unit, language)
    if label and unit is not None and unit.kind not in ("percent", "ratio"):
        text = f"{text} · {label}" if text else label
    return text or None


# A number as written in text: whole ("0.93" of "0.93x", never its "0"); not inside a word ("FY24", "Q3").
_NUMBER = re.compile(r"(?<![\w.])[−\-+]?\d[\d,]*(?:\.\d+)?(?!\d|[.,]\d)")


def _title(r: Resolved, language: str) -> str:
    """The model's title when every number in it is printed in the visual's labels; otherwise one made from them."""
    allowed = _label_numbers(r)
    if r.spec.title:
        title = r.spec.title.strip()
        if title and all(_num(t) in allowed for t in _NUMBER.findall(title)):
            return title
    table_title = r.dataset_list[0].title
    if r.kind in ("timeline", "table", "kpi"):
        return table_title
    names = [s.label for s in r.series]
    if len(names) > 2 or not names:
        base = table_title
    elif len(names) == 1 and parse_period(names[0]) is not None:  # a column of one period ("FY24")
        base = f"{table_title} ({names[0]})"
    else:
        base = " · ".join(names)
    xs = [i.label for i in r.x_items]
    if r.kind in ("line", "bar", "grouped_bar", "stacked_bar") and r.x_type in ("period", "date") and len(xs) >= 2:
        return f"{base}, {xs[0]}–{xs[-1]}"
    if r.kind == "comparison" and len(names) == 1 and len(xs) == 2:
        return f"{base}: {xs[0]} vs {xs[1]}" if language != "hi" else f"{base}: {xs[0]} बनाम {xs[1]}"
    return base


def _num(token: str) -> str:
    return token.replace(",", "").replace("−", "-").lstrip("+")


def _label_numbers(r: Resolved) -> set[str]:
    texts = [i.label for i in r.x_items] + [s.label for s in r.series] + [ds.title for ds in r.dataset_list]
    return {_num(t) for text in texts for t in _NUMBER.findall(text)}


def _highlight(r: Resolved, language: str) -> Highlight | None:
    if not r.highlight_x and not r.highlight_series:
        return None
    note = None
    if len(r.highlight_x) == 1 and r.series:
        s = next((s for s in r.series if s.key in r.highlight_series), r.series[0])
        vals = {x: p.value.value for x, p in s.points.items() if p.value.value is not None}
        if len(vals) >= 3:
            x = r.highlight_x[0]
            if x in vals and vals[x] == max(vals.values()):
                note = _t(language, "highest")
            elif x in vals and vals[x] == min(vals.values()):
                note = _t(language, "lowest")
    return Highlight(x=r.highlight_x, series=r.highlight_series, note=note)


def _trend_summary(book: SourceBook, r: Resolved, language: str) -> tuple[str, list[Calculation]]:
    s = r.series[0]
    present = [i.label for i in r.x_items if i.label in s.points and s.points[i.label].value.value is not None]
    if len(present) < 2:
        return _bars_summary(r, language), []
    first, last = present[0], present[-1]
    a, b = _show_point(s.points[first], language), _show_point(s.points[last], language)
    found = _delta_calc(book, s, first, last, language)
    others = [x.label for x in r.series[1:]]
    tail = _t(language, "more_series", names=", ".join(others)) if others else ""
    if found is None:
        return _t(language, "trend_flat", series=s.label, a=a, xa=first, b=b, xb=last) + tail, []
    c, _ = found
    text = _t(language, "trend", series=s.label, a=a, xa=first, b=b, xb=last, growth=_show_calc(c, language))
    return text + tail, [c]


def _bars_summary(r: Resolved, language: str) -> str:
    s = r.series[0]
    pts = [
        (x.label, s.points[x.label])
        for x in r.x_items
        if x.label in s.points and s.points[x.label].value.value is not None
    ]
    if not pts:
        return s.label
    top = max(pts, key=lambda xp: xp[1].value.value)  # type: ignore[arg-type, return-value]
    low = min(pts, key=lambda xp: xp[1].value.value)  # type: ignore[arg-type, return-value]
    return _t(
        language,
        "bars",
        series=s.label,
        top=top[0],
        top_v=_show_point(top[1], language),
        low=low[0],
        low_v=_show_point(low[1], language),
    )


def _donut(book: SourceBook, r: Resolved, language: str) -> tuple[str, list[Calculation]]:
    s = r.series[0]
    pts = [
        (x.label, s.points[x.label])
        for x in r.x_items
        if x.label in s.points and s.points[x.label].value.value is not None
    ]
    top = max(pts, key=lambda xp: xp[1].value.value)  # type: ignore[arg-type, return-value]
    shares: list[Calculation] = []
    if s.unit is None or s.unit.kind != "percent":
        try:
            shares = [_share(book, s, x, p, language) for x, p in pts]
        except CalculationError:
            shares = []
    top_share = next((c for c, (x, _) in zip(shares, pts, strict=False) if x == top[0]), None)
    if top_share is not None:
        text = _t(
            language,
            "donut",
            top=top[0],
            series=s.label,
            top_v=_show_point(top[1], language),
            share=_show_calc(top_share, language),
        )
    else:
        text = _t(language, "donut_no_share", top=top[0], series=s.label, top_v=_show_point(top[1], language))
    return text, shares


def _waterfall(book: SourceBook, r: Resolved, language: str) -> tuple[list[Row], list[Series], Calculation, str]:
    s = r.series[0]
    pts = [(x.label, s.points.get(x.label)) for x in r.x_items]
    if any(p is None or p.value.value is None for _, p in pts) or len(pts) < 3:
        raise SpecError(["a waterfall needs a start, at least one step and an end, all with numbers"])
    values = [p.value.value for _, p in pts]  # type: ignore[union-attr]
    start, steps, end = values[0], values[1:-1], values[-1]
    decimals = max(p.value.decimals for _, p in pts)  # type: ignore[union-attr]
    tolerance = 0.5 * 10**-decimals * len(values) + 1e-9
    signed = abs(start + sum(steps) - end) <= tolerance  # type: ignore[operator]
    subtract = not signed and all(v >= 0 for v in steps) and abs(start - sum(steps) - end) <= tolerance  # type: ignore[operator]
    if not (signed or subtract):
        raise SpecError(
            [f"the waterfall's steps don't take {pts[0][0]} to {pts[-1][0]}: pick the rows that add up to the end"]
        )
    keys = {"total": _t(language, "total"), "increase": _t(language, "increase"), "decrease": _t(language, "decrease")}
    series = [Series(key=k, label=v, unit=s.unit) for k, v in keys.items()]
    rows: list[Row] = []
    operands: list[Operand] = []
    for i, (x, p) in enumerate(pts):
        assert p is not None and p.value.value is not None
        ref = cell_ref(book, p.dataset, p.value)
        v = p.value.value
        if i in (0, len(pts) - 1):
            role = "total"
        elif subtract or v < 0:
            role = "decrease"
        else:
            role = "increase"
        shown = abs(v) if role != "total" else v
        rows.append(
            Row(
                x=x,
                values={k: (shown if k == role else None) for k in keys},
                cells={k: (ref if k == role else None) for k in keys},
            )
        )
        if i < len(pts) - 1:
            op = _operand(book, p, x)
            if subtract and i > 0:
                op = Operand(value=-v, unit=op.unit, cell=op.cell, decimals=op.decimals, label=x, period=op.period)
            operands.append(op)
    bridge = calc.total(operands, series=s.label, at=pts[-1][0], language=language)
    summary = _t(
        language,
        "waterfall",
        start=pts[0][0],
        start_v=_show_point(pts[0][1], language),  # type: ignore[arg-type]
        end=pts[-1][0],
        end_v=_show_point(pts[-1][1], language),  # type: ignore[arg-type]
    )
    return rows, series, bridge, summary


def _kpi_tiles(book: SourceBook, r: Resolved, language: str) -> tuple[list[Tile], list[Calculation], str]:
    tiles: list[Tile] = []
    deltas: list[Calculation] = []
    xs = [i.label for i in r.x_items]
    periodic = r.x_type in ("period", "date")
    for s in r.series:
        present = [x for x in xs if x in s.points and s.points[x].value.value is not None]
        if not present:
            continue
        targets = [present[-1]] if periodic else present
        for x in targets:
            p = s.points[x]
            delta = None
            if periodic and len(present) >= 2:
                found = _delta_calc(book, s, present[-2], x, language)
                if found is not None:
                    c, kind = found
                    delta = Delta(value=c.value, kind=kind, calculation=c)  # type: ignore[arg-type]
                    deltas.append(c)
            label = f"{s.label} ({x})" if periodic or len(r.series) > 1 else x
            tiles.append(
                Tile(
                    label=label,
                    value=p.value.value,
                    unit=p.value.unit,
                    cell=cell_ref(book, p.dataset, p.value),
                    delta=delta,
                )  # type: ignore[arg-type]
            )
            if len(tiles) >= 6:
                break
        if len(tiles) >= 6:
            break
    if not tiles:
        raise SpecError(["no numbers for the KPI tiles"])
    items = []
    for t in tiles[:3]:
        text = f"{t.label}: {_show(t.value, t.unit, _decimals(t.cell.text) if t.cell else 0, language)}"
        if t.delta is not None:
            text += f" ({_show_calc(t.delta.calculation, language)})"
        items.append(text)
    return tiles, deltas, _t(language, "kpi", items="; ".join(items))


def _comparison_tiles(book: SourceBook, r: Resolved, language: str) -> tuple[list[Tile], list[Calculation], str]:
    tiles: list[Tile] = []
    deltas: list[Calculation] = []
    xs = [i.label for i in r.x_items]
    items: list[str] = []
    if len(xs) == 2:
        a, b = xs
        for s in r.series:
            pa, pb = s.points.get(a), s.points.get(b)
            if pa is None or pb is None or pa.value.value is None or pb.value.value is None:
                continue
            found = _delta_calc(book, s, a, b, language)
            delta = None
            if found is not None:
                c, kind = found
                delta = Delta(value=c.value, kind=kind, calculation=c)  # type: ignore[arg-type]
                deltas.append(c)
            tiles.append(
                Tile(
                    label=f"{s.label} ({a})",
                    value=pa.value.value,
                    unit=pa.value.unit,
                    cell=cell_ref(book, pa.dataset, pa.value),
                )
            )
            tiles.append(
                Tile(
                    label=f"{s.label} ({b})",
                    value=pb.value.value,
                    unit=pb.value.unit,
                    cell=cell_ref(book, pb.dataset, pb.value),
                    delta=delta,
                )
            )
            text = _t(language, "vs", a=f"{a} {_show_point(pa, language)}", b=f"{b} {_show_point(pb, language)}")
            if delta is not None:
                text += f" ({_show_calc(delta.calculation, language)})"
            items.append(f"{s.label}: {text}")
    else:
        x = xs[0]
        base: tuple[ResolvedSeries, Point] | None = None
        for s in r.series:
            p = s.points.get(x)
            if p is None or p.value.value is None:
                continue
            delta = None
            if base is not None and same_unit(base[0].unit, s.unit):
                try:
                    c = calc.diff(
                        _operand(book, base[1], base[0].label),
                        _operand(book, p, s.label),
                        series=s.label,
                        language=language,
                        label=_t(language, "gap", a=base[0].label, b=s.label, x=x),
                    )
                    delta = Delta(
                        value=c.value,
                        kind="pp" if s.unit is not None and s.unit.kind == "percent" else "abs",
                        calculation=c,
                    )
                    deltas.append(c)
                except CalculationError:
                    delta = None
            tiles.append(
                Tile(
                    label=f"{s.label} ({x})",
                    value=p.value.value,
                    unit=p.value.unit,
                    cell=cell_ref(book, p.dataset, p.value),
                    delta=delta,
                )
            )
            items.append(f"{s.label}: {_show_point(p, language)}")
            if base is None:
                base = (s, p)
    if len(tiles) < 2:
        raise SpecError(["a comparison needs two numbers"])
    return tiles, deltas, _t(language, "comparison", items="; ".join(items))


def _events(book: SourceBook, r: Resolved) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for t in r.timeline:
        ds = t.dataset
        date_cell = ds.text(t.row.key, t.date_col)
        if date_cell is None:
            continue
        period = parse_period(date_cell.text)
        date = period.end.isoformat() if period is not None and period.kind == "date" and period.end else date_cell.text
        label = t.row.label
        label_col = ds.columns[ds.label_column] if ds.label_column is not None else None
        if label_col is not None and re.fullmatch(r"[\d.]+", label or ""):
            label = f"{label_col.label} {label}"
        detail_cell = ds.text(t.row.key, t.detail_col) if t.detail_col else None
        ref = CellRef(
            source_id=book.source_id(ds),
            document_id=ds.document_id,
            table_id=ds.table_id,
            page=ds.page_start,
            row=date_cell.cell_row,
            col=date_cell.cell_col,
            text=date_cell.text,
        )
        events.append(TimelineEvent(date=date, label=label, detail=detail_cell.text if detail_cell else None, cell=ref))
    if not events:
        raise SpecError(["timeline: no dated rows"])
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", e.date) for e in events):
        events.sort(key=lambda e: e.date)  # a revision history printed newest first reads oldest first on a timeline
    return events


# ------------------------------------------------------------------ grounding


def _cell_number(ref: CellRef) -> float | None:
    q = parse_quantity(ref.text)
    return q.value if isinstance(q, Quantity) else None


def check_grounding(v: Visual) -> list[str]:
    """Every number in ``v`` is a cell value (its CellRef's printed text parses to it) or a listed Calculation
    (recomputed from its inputs' printed text). Returns the violations (empty: grounded)."""
    problems: list[str] = []
    source_ids = {c.source_id for c in v.sources}
    calc_values = [c.value for c in v.calculations]

    def known_ref(ref: CellRef, where: str) -> None:
        if ref.source_id not in source_ids:
            problems.append(f"{where}: source {ref.source_id} is not in sources")

    for c in v.calculations:
        inputs = []
        for ref in c.inputs:
            known_ref(ref, f"calculation '{c.label}'")
            n = _cell_number(ref)
            if n is None:
                problems.append(f"calculation '{c.label}': input '{ref.text}' is not a number")
            inputs.append(n)
        if None not in inputs and inputs:
            try:
                expected = calc.recompute(c, inputs)  # type: ignore[arg-type]
            except (ZeroDivisionError, ValueError, IndexError) as e:
                problems.append(f"calculation '{c.label}': {e}")
                continue
            if c.op in ("diff", "sum"):
                decimals = max(_decimals(i.text) for i in c.inputs)
                ok = calc.close(expected, c.value, decimals)
            else:
                ok = calc.close(expected, c.value, 2)
            if not ok:
                problems.append(f"calculation '{c.label}': {c.value} != {expected}")

    keys = {s.key for s in v.series}
    calculated = {s.key for s in v.series if s.calculated}
    for row in v.rows:
        for key, value in row.values.items():
            if key not in keys:
                problems.append(f"row {row.x}: unknown series {key}")
            if value is None:
                continue
            ref = row.cells.get(key)
            if key in calculated:
                if not any(calc.close(value, cv, 2) for cv in calc_values):
                    problems.append(f"row {row.x}, {key}: calculated value {value} is not a calculation")
                continue
            if ref is None:
                problems.append(f"row {row.x}, {key}: {value} has no cell")
                continue
            known_ref(ref, f"row {row.x}")
            n = _cell_number(ref)
            magnitude_ok = (
                v.kind == "waterfall" and key in ("increase", "decrease") and n is not None and abs(n) == value
            )
            if n is None or (n != value and not magnitude_ok):
                problems.append(f"row {row.x}, {key}: {value} is not the cell's '{ref.text}'")
    calcs_by_id = {id(c) for c in v.calculations}
    for t in v.tiles:
        if t.cell is not None:
            known_ref(t.cell, f"tile {t.label}")
            if _cell_number(t.cell) != t.value:
                problems.append(f"tile {t.label}: {t.value} is not the cell's '{t.cell.text}'")
        elif not any(calc.close(t.value, cv, 2) for cv in calc_values):
            problems.append(f"tile {t.label}: {t.value} has no cell or calculation")
        if t.delta is not None:
            if t.delta.value != t.delta.calculation.value:
                problems.append(f"tile {t.label}: delta {t.delta.value} differs from its calculation")
            if id(t.delta.calculation) not in calcs_by_id and t.delta.calculation not in v.calculations:
                problems.append(f"tile {t.label}: delta calculation is not listed in calculations")
    for e in v.events:
        if e.cell is not None:
            known_ref(e.cell, f"event {e.label}")
    if v.highlight is not None:
        xs = {r.x for r in v.rows}
        for x in v.highlight.x:
            if x not in xs:
                problems.append(f"highlight x '{x}' is not a row")
        for s in v.highlight.series:
            if s not in keys:
                problems.append(f"highlight series '{s}' is not a series")

    # numbers in text: the summary, title, subtitle and highlight note
    allowed_values = (
        [val for r in v.rows for val in r.values.values() if val is not None] + [t.value for t in v.tiles] + calc_values
    )
    # Labels are printed text (period labels, row labels, dates), so their numbers ("2024" of "31 Mar 2024") may be
    # repeated; any other number in the text must be a value shown (at the precision it is written with).
    label_texts = (
        [r.x for r in v.rows]
        + [s.label for s in v.series]
        + [t.label for t in v.tiles]
        + [text for e in v.events for text in (e.date, e.label, e.detail or "")]
        + [c.label for c in v.calculations]
        + [ref.text for r in v.rows for ref in r.cells.values() if ref is not None]
        + [e.cell.text for e in v.events if e.cell is not None]
    )
    label_numbers = {_num(t) for text in label_texts for t in _NUMBER.findall(text)}
    pages = {str(p) for c in v.sources for p in (c.page_start, c.page_end) if p is not None}
    printed = label_numbers | {_num(t) for c in v.sources for t in _NUMBER.findall(c.snippet)}  # the tables' own text
    checks = (
        ("summary", v.summary, label_numbers | pages),
        ("subtitle", v.subtitle or "", label_numbers | pages),
        ("note", (v.highlight.note if v.highlight else "") or "", label_numbers),
        ("title", v.title, printed),
    )
    for field, text, free in checks:
        for token in _NUMBER.findall(text):
            raw = _num(token)
            if raw in free:
                continue
            try:
                number = float(raw)
            except ValueError:
                problems.append(f"{field}: '{token}' is not a number")
                continue
            decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
            if not any(calc.close(abs(number), abs(a), decimals) for a in allowed_values):
                problems.append(f"{field}: {token} is neither a cell value, a calculation nor printed in a label")
    return problems
