"""The project overview: a few panels built from the most chartable tables as soon as documents are typed
(docs/DESIGN.md §12.1, workstream 9). Pure: datasets in, specs out (built and checked like any other visual).

1. **KPI tiles**: the best KPI table (headline metrics in mixed units, two periods side by side): its first few
   metrics at the latest period, each with its change from the period before.
2. **Trend**: the time series with the most periods, its first one or two amounts as lines. Tables that continue
   each other (the FY23 and FY24 quarterly tables) become one series over every period.
3. **Composition**: parts of a whole: segment revenue at the latest period (or printed shares) as a donut.

A project holds several documents (companies, languages), and a panel mixes nothing: each is built from the tables of
one document, and the panels are ordered by document, the document with the most chartable tables first, each
document giving one panel in turn (its KPI tiles, then its trend, then its composition) until there are enough
(``overview_specs``). Their titles say which document they are from and are written in the project's main language
(``main_language``: that of most of its documents, else the app's), while the labels of the data stay as printed
("Suryodaya yojana soochna: seats by course", the series "सीटें"; ``overview_title``).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from ...domain.canvas import Visual
from ...domain.datasets import ChartKind, TypedDataset
from .chartability import series_columns
from .spec import MAX_DATASETS, SpecError, VisualSpec, document_name, ref

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


def _chartable(datasets: Sequence[TypedDataset]) -> dict[str, list[TypedDataset]]:
    """The chartable datasets by document, the document with the most of them first (then in the order met)."""
    by_doc: dict[str, list[TypedDataset]] = {}
    for d in datasets:
        if d.chartability.kind != "none":
            by_doc.setdefault(d.document_id, []).append(d)
    return dict(sorted(by_doc.items(), key=lambda kv: -len(kv[1])))  # (a stable sort: ties keep the order met)


def overview_specs(
    datasets: Sequence[TypedDataset],
    *,
    language: str = "en",
    max_panels: int = 3,
    check: Callable[[VisualSpec], bool] | None = None,
) -> list[VisualSpec]:
    """Up to ``max_panels`` specs: per document KPI tiles, a trend, a composition (each from a different table of that
    document), the documents in turn so that a small overview covers several of them; the panels of a document stay
    together and the document with the most chartable tables comes first. ``check`` tells whether a spec builds (the
    caller resolves and builds it); a spec that doesn't is skipped for the next best."""
    ok = check or (lambda _spec: True)
    by_doc = _chartable(datasets)
    makers: list[Callable[[list[TypedDataset], set[str]], VisualSpec | None]] = [
        lambda docs, used: _take_kpi(docs, used, language, ok),
        lambda docs, used: _take_trend(docs, used, language, ok),
        lambda docs, used: _take_composition(docs, used, language, ok),
    ]
    queues = {doc: list(makers) for doc in by_doc}
    used: dict[str, set[str]] = {doc: set() for doc in by_doc}
    found: dict[str, list[VisualSpec]] = {doc: [] for doc in by_doc}
    total = 0
    while total < max_panels and any(queues.values()):
        for doc, docs in by_doc.items():  # one panel from each document in turn
            while queues[doc] and total < max_panels:
                spec = queues[doc].pop(0)(docs, used[doc])
                if spec is not None:
                    found[doc].append(spec)
                    used[doc].update(spec.datasets)
                    total += 1
                    break
    return [spec for doc in by_doc for spec in found[doc]]


def _take(
    candidates: list[TypedDataset],
    used: set[str],
    make: Callable[[TypedDataset], VisualSpec | None],
    ok: Callable[[VisualSpec], bool],
) -> VisualSpec | None:
    for ds in candidates:
        if ds.id in used:
            continue
        try:
            spec = make(ds)
        except (SpecError, ValueError):
            spec = None
        if spec is not None and ok(spec):
            return spec
    return None


def _take_kpi(
    docs: list[TypedDataset], used: set[str], language: str, ok: Callable[[VisualSpec], bool]
) -> VisualSpec | None:
    kpis = sorted((d for d in docs if _option(d, "kpi") >= 0.7), key=lambda d: (-_option(d, "kpi"), _doc_order(d)))
    return _take(kpis, used, lambda d: kpi_spec(d, language), ok)


def _take_trend(
    docs: list[TypedDataset], used: set[str], language: str, ok: Callable[[VisualSpec], bool]
) -> VisualSpec | None:
    trends = sorted(
        (d for d in docs if _option(d, "time_series") >= 0.5),
        key=lambda d: (
            -sum(len(o.chartability.period_order) for o in continuations(d, docs))
            if d.chartability.period_axis == "rows"
            else -len(d.chartability.period_order),
            -_option(d, "time_series"),
            _doc_order(d),
        ),
    )
    return _take(trends, used, lambda d: trend_spec(d, docs, language), ok)


def _composition_rank(d: TypedDataset) -> tuple[float, int, tuple[int, int]]:
    totals = [r for r in d.rows if r.type == "total" and r.parts]
    named_total = any(r.label.strip().lower() in ("total", "कुल", "grand total") for r in totals)
    return (
        -(_option(d, "composition") + (0.1 if named_total else 0) - 0.05 * max(0, len(totals) - 1)),
        0,
        _doc_order(d),
    )


def _take_composition(
    docs: list[TypedDataset], used: set[str], language: str, ok: Callable[[VisualSpec], bool]
) -> VisualSpec | None:
    compositions = sorted((d for d in docs if _option(d, "composition") >= 0.6), key=_composition_rank)
    return _take(compositions, used, lambda d: composition_spec(d, language), ok)


# ------------------------------------------------------------------ language and titles

_DEVANAGARI = re.compile("[ऀ-ॿ]")
_LATIN = re.compile("[A-Za-z]")


def script_of(text: str) -> str | None:
    """ "hi" when the text is mostly Devanagari, "en" when mostly Latin, None for neither (digits)."""
    hi, en = len(_DEVANAGARI.findall(text)), len(_LATIN.findall(text))
    return None if hi == en == 0 else "hi" if hi > en else "en"


def main_language(datasets: Sequence[TypedDataset], default: str = "en") -> str:
    """The language of the project: that of most of its documents (a document is in the script most of its tables'
    titles, row labels and column headings are in); ``default`` (the app's) when they are split evenly or none can
    tell."""
    texts: dict[str, list[str]] = {}
    for d in datasets:
        texts.setdefault(d.document_id, []).extend([d.title, *(r.label for r in d.rows), *(c.label for c in d.columns)])
    votes = Counter(s for t in texts.values() if (s := script_of(" ".join(t))) is not None)
    if votes["hi"] == votes["en"]:
        return default if default in ("en", "hi") else "en"
    return "hi" if votes["hi"] > votes["en"] else "en"


_FISCAL = re.compile(r"\b(?:q[1-4])?fy\d{2,4}\b", re.IGNORECASE)


def document_title(filename: str) -> str:
    """ "valmora_annual_report_fy24.pdf" → "Valmora annual report FY24": a document's name in a panel's title."""
    return _FISCAL.sub(lambda m: m.group(0).upper(), document_name(filename))


# Words of table labels, for a title in the project's language. A label is translated only when every word of it is
# known; any other stays as printed (a data label may).
_HI_EN: Mapping[str, str] = {
    "सीट": "seats", "सीटें": "seats", "सीटों": "seats", "पाठ्यक्रम": "course", "जिला": "district", "ज़िला": "district",
    "जिले": "districts", "बजट": "budget", "आवंटित": "allocated", "लक्ष्य": "target", "युवा": "youth",
    "केंद्र": "centre", "प्रशिक्षण": "training", "अवधि": "duration", "सप्ताह": "weeks", "राजस्व": "revenue",
    "आय": "income", "मुनाफा": "profit", "मुनाफ़ा": "profit", "लाभ": "profit", "कर्मचारी": "employees",
    "खंड": "segment", "क्षेत्र": "region", "योग्यता": "qualification", "न्यूनतम": "minimum", "कुल": "total",
    "आवेदन": "applications", "राशि": "amount", "संख्या": "number", "और": "and",
}  # fmt: skip
_EN_HI: Mapping[str, str] = {
    "seats": "सीटें", "seat": "सीट", "course": "पाठ्यक्रम", "courses": "पाठ्यक्रम", "district": "जिला",
    "districts": "जिले", "budget": "बजट", "allocated": "आवंटित", "target": "लक्ष्य", "centre": "केंद्र",
    "centres": "केंद्र", "training": "प्रशिक्षण", "duration": "अवधि", "weeks": "सप्ताह", "revenue": "राजस्व",
    "income": "आय", "profit": "मुनाफ़ा", "employees": "कर्मचारी", "segment": "खंड", "segments": "खंड",
    "region": "क्षेत्र", "qualification": "योग्यता", "minimum": "न्यूनतम", "total": "कुल",
    "applications": "आवेदन", "amount": "राशि", "number": "संख्या", "and": "और",
}  # fmt: skip
_GENERIC_X = frozenset({"particulars", "metric", "item", "row", "category", "विवरण"})
_LABEL_PUNCT = re.compile(r"[()\[\]{},.:;/]")


def in_language(label: str, language: str) -> str:
    """``label`` in ``language`` when it is in the other one and every word of it is known; else as printed."""
    if script_of(label) in (None, language):
        return label
    table = _HI_EN if language == "en" else _EN_HI
    words = _LABEL_PUNCT.sub(" ", label).split()
    found = [table.get(w) or table.get(w.casefold()) for w in words]
    if not words or any(w is None for w in found):
        return label
    return " ".join(w for w in found if w)


_PHRASES = {
    "en": {"kpi": "key figures", "by": "{measure} by {category}"},
    "hi": {"kpi": "मुख्य आँकड़े", "by": "{category} के अनुसार {measure}"},
}


def _lower(text: str) -> str:
    return text.lower() if script_of(text) == "en" and text.istitle() else text


def overview_title(visual: Visual, label: str | None, language: str) -> str:
    """A panel's title: the document it is from (``label``, when the overview has several), then what it shows, in the
    project's ``language``: the table's heading, a trend's metrics and periods, "seats by course", as the builder titled
    them, except where those are printed in the other language (a Hindi table in an English project), where the kind
    of panel and the translatable words of its labels say it. The labels of the data stay as printed."""
    phrases = _PHRASES["hi" if language == "hi" else "en"]
    title = visual.title
    if visual.kind == "kpi" and script_of(title) not in (None, language):
        title = phrases["kpi"]
    elif visual.kind == "donut" and visual.series:
        measure = in_language(visual.series[0].label, language)
        category = in_language(visual.x.label, language) if visual.x is not None else ""
        if category and category.casefold() not in _GENERIC_X and script_of(category) in (None, language):
            title = phrases["by"].format(measure=_lower(measure), category=_lower(category))
        elif script_of(measure) in (None, language):
            title = measure
    title = title.replace(": ", ", ")
    first = label.split()[0].casefold() if label else ""
    if not label or title.casefold().startswith(first):  # ("Zephyra at a glance" says whose it is)
        return title
    return f"{label}: {title}"
