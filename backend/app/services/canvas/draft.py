"""The instant draft of a turn's visual (docs/DESIGN.md §12.1, "Instant draft, then refine"): a chart chosen and built
by code, without the model, in a few milliseconds, as soon as the turn's retrieval returns. Pure.

    question (and its English query) → ``Cues``: the kind asked for ("bar chart", "pie", "timeline", "गोल चार्ट"), the
        shapes wanted (a trend, a breakdown, a comparison, headline numbers, dates, a ranking), the periods named
        ("FY24", "Q3 FY24") and the words that may name a series ("EBITDA margin", "बजट")
    candidates: ``planner.score_candidates`` (word overlap with the tables' titles and labels, the shape, the turn's
        retrieved tables, only the named company's documents, or when none is named the document of the turn's top
        source, unless the question compares documents or companies: ``TurnContext.scope``)
    the best candidate → its default chartable shape: periods → a line (two periods: bars), parts of a whole → a donut
        (two periods: stacked or grouped bars), categories → bars, headline metrics → KPI tiles, dates → a timeline;
        its series and periods narrowed to the question's words
    → ``planner.first_valid`` with ``planner.builds``: the same validator and builder as the planner's choice, so every
        number is a cell
    → ``Draft(spec, confident, reasons)``

**Confident** means the planner can't be expected to do better: one clear table (it leads the runner-up by
``CLEAR_MARGIN``, it is the only one with the rows the question names, or the only timeline), the kind drawn is the kind
asked for (when one was), and every series was named by the question (or the table has only one, or the question asks
for the whole table: its headline numbers, its dates). A confident draft is the turn's visual; otherwise
``CanvasService.prepare_visual`` has the planner refine it once the answer's text is complete, and replaces it in place
when the model chooses differently.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from ...domain.canvas import VisualKind
from ...domain.datasets import DatasetColumn, DatasetRow, TypedDataset
from ...domain.projects import Citation
from ...providers.ingestion import document_label
from ..language import wordset
from ..subjects import compares_documents, documents_by_name, named_documents
from .chartability import series_columns
from .overview import composition_spec, continuations
from .parsing import find_period
from .planner import _PERIOD_TOKEN, _STOP, _WORD, _stem, builds, first_valid, score_candidates
from .spec import Resolved, SpecError, VisualSpec, ref, resolve

CLEAR_MARGIN = 2.0  # how far the best candidate must lead the next one to be "one clear table"
MAX_DRAFT_SERIES = 4

Shape = Literal["timeline", "kpi", "composition", "comparison", "trend", "ranking"]

_LETTER = "A-Za-z0-9ऀ-ॣ०-ॿ"


def _pattern(alternatives: str) -> re.Pattern[str]:
    """Whole words of casefolded text (\\b fails after Hindi vowel signs)."""
    return re.compile(f"(?<![{_LETTER}])(?:{alternatives})(?![{_LETTER}])", re.I)


# The kind a question asks for by name. Bare "bar" or "line" are not enough ("business line"); "table" is ("show it as
# a table", "exact figures").
_KINDS_ASKED: tuple[tuple[re.Pattern[str], VisualKind], ...] = tuple(
    (_pattern(p), k)
    for p, k in (
        (r"waterfall|bridge|वॉटरफॉल", "waterfall"),
        (r"pie(?:\s+chart)?|donut|doughnut|पाई(?:\s+चार्ट)?|गोल\s+चार्ट", "donut"),
        (r"stacked", "stacked_bar"),
        (r"grouped\s+bars?|side[\s-]+by[\s-]+side", "grouped_bar"),
        (r"bar\s+(?:chart|graph)s?|bars|column\s+chart|बार\s+(?:चार्ट|ग्राफ़|ग्राफ)", "bar"),
        (r"line\s+(?:chart|graph)s?|trend\s+line|लाइन\s+(?:चार्ट|ग्राफ़|ग्राफ)|रेखा\s+(?:चार्ट|ग्राफ़|ग्राफ)", "line"),
        (r"timeline|टाइमलाइन", "timeline"),
        (r"tables?|tabular|exact\s+(?:\S+\s+){0,2}?(?:figures|numbers)|टेबल|तालिका|सारणी", "table"),
        (r"kpis?|tiles", "kpi"),
    )
)

# What a question wants to see, in English, Hindi and Hinglish (a table's shape decides whether a cue applies).
_SHAPES: tuple[tuple[Shape, re.Pattern[str]], ...] = tuple(
    (s, _pattern(p))
    for s, p in (
        (
            "timeline",
            r"timeline|dates?|deadlines?|when|schedule|came\s+into\s+(?:force|effect)|effective|tareekh|tarikh|"
            r"तारीख|तारीखें|तारीख़ें|तिथि|तिथियाँ|तिथियां|कब",
        ),
        (
            "kpi",
            r"headlines?|highlights?|key\s+(?:numbers|figures|metrics|financials|indicators)|at\s+a\s+glance|overall|"
            r"snapshot|scorecard|ek\s+nazar|मुख्य|झलक|एक\s+नज़र|एक\s+नजर",
        ),
        (
            "composition",
            r"break\s*down|breakdown|split|(?<!per\s)shares?|mix|composition|contribut\w*|consists?|made\s+up|"
            r"makes?\s+up|who\s+owns|ownership|pattern|segment-?wise|by\s+segments?|each\s+segment|hissa|hisse|"
            r"kis\s+kis|हिस्सा|हिस्से|हिस्सेदारी|बंटवारा|बँटवारा|किस\s+किस",
        ),
        (
            "comparison",
            r"compare\w*|comparison|versus|vs|against|than|stack\s+up|last\s+year|previous\s+year|prior\s+year|"
            r"year\s+ago|a\s+year\s+earlier|pichle\s+saal|tulna|better|worse|"
            r"तुलना|मुक़ाबले|मुकाबले|पिछले\s+साल",
        ),
        (
            "trend",
            r"trends?|trending|over\s+(?:the\s+)?(?:years|quarters|time|period)|quarter\s+(?:by|on|after|to)\s+quarter|"
            r"each\s+quarter|every\s+quarter|across\s+(?:the\s+)?quarters|through(?:out)?|har\s+quarter|"
            r"har\s+timahi|move[ds]?|movement|rujhan|रुझान|हर\s+तिमाही|तिमाही\s+दर\s+तिमाही",
        ),
        (
            "ranking",
            r"most|highest|lowest|largest|biggest|smallest|fastest|slowest|top|sabse|सबसे",
        ),
    )
)

# Words that never name a series: chart words, filler, the shapes' own words.
_NOT_SERIES = wordset(
    """chart charts graph graphs plot draw display screen visual visualise visualize pie donut bar bars line lines
    table tables timeline kpi tiles waterfall bridge stacked grouped side want need see look like put up give
    quarter quarterly year annual fy each every all both two three four last previous current next this these those
    trend compare comparison versus against than how much many more less much did do does was were has have had is are
    be been will would could should can company company's group business businesses about across through over under
    up down move moved change changed into turn turns walk me us our their its it's tell show shown showing please
    exact figures numbers number key headline headlines highlight highlights overall performance glance breakdown
    split mix composition consist consists made makes which who what where when why
    mujhe dikhao dikhaiye dikha batao bataiye karo kaisa kaisi kaise raha rahi kitna kitne kitni har ek nazar
    bas chahiye     का की के को और है में से पर दिखाओ दिखाइए बताओ क्या कैसा कैसी कितना हर एक चार्ट ग्राफ़ ग्राफ टेबल"""
)

# English words for the labels of a Hindi document, and common synonyms (both sides of a match are normalised).
_CANON = {
    "sales": "revenue",
    "turnover": "revenue",
    "topline": "revenue",
    "राजस्व": "revenue",
    "आय": "revenue",
    "बिक्री": "revenue",
    "मुनाफ़ा": "profit",
    "मुनाफा": "profit",
    "लाभ": "profit",
    "munafa": "profit",
    "debt": "borrowing",
    "debts": "borrowing",
    "loans": "loan",
    "karza": "borrowing",
    "karz": "borrowing",
    "कर्ज़": "borrowing",
    "कर्ज": "borrowing",
    "ऋण": "borrowing",
    "workforce": "employee",
    "headcount": "employee",
    "staff": "employee",
    "कर्मचारी": "employee",
    "मार्जिन": "margin",
    "budget": "बजट",
    "target": "लक्ष्य",
    "targets": "लक्ष्य",
    "seat": "सीट",
    "seats": "सीट",
    "सीटें": "सीट",
    "सीटों": "सीट",
    "district": "जिला",
    "districts": "जिला",
    "ज़िला": "जिला",
    "ज़िले": "जिला",
    "जिले": "जिला",
    "ज़िलों": "जिला",
    "जिलों": "जिला",
    "course": "पाठ्यक्रम",
    "courses": "पाठ्यक्रम",
    "centre": "केंद्र",
    "centres": "केंद्र",
    "center": "केंद्र",
    "centers": "केंद्र",
    "duration": "अवधि",
}


def _canon(word: str) -> str:
    w = word.casefold()
    w = _CANON.get(w, w)
    return _CANON.get(_stem(w), _stem(w))


def _content_words(text: str | None, *, question: bool = True) -> set[str]:
    """Normalised words that may name a series: a question's without chart words and filler, a label's all of them
    but stop words and periods."""
    if not text:
        return set()
    out = set()
    for w in _WORD.findall(text):
        lw = w.casefold()
        if len(lw) < 2 or lw in _STOP or (question and lw in _NOT_SERIES) or _PERIOD_TOKEN.fullmatch(w):
            continue
        out.add(_canon(lw))
    return out


def _same(a: str, b: str) -> bool:
    """One word for another: equal, or one the start of the other ("employe" / "employees", "share" /
    "shareholding")."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 5 and long_.startswith(short)


def _hits(words: set[str], label: str) -> tuple[frozenset[str], int]:
    """The question's words found in a label, and how many of the label's words weren't asked for."""
    label_words = _content_words(label, question=False)
    hit = frozenset(w for w in words if any(_same(w, lw) for lw in label_words))
    found = {lw for lw in label_words if any(_same(w, lw) for w in hit)}
    return hit, len(label_words - found)


# ------------------------------------------------------------------ cues


@dataclass(frozen=True, slots=True)
class Cues:
    kind: VisualKind | None  # the kind asked for by name
    shapes: frozenset[Shape]
    periods: tuple[str, ...]  # canonical labels named ("FY24", "Q3 FY24"), in order of mention
    words: frozenset[str]  # words that may name a series


def cues(question: str, query_en: str | None = None, *, names: Collection[str] = ()) -> Cues:
    """What the question (and its English query) says about the visual. ``names``: company names to leave out of the
    series words ("Valmora")."""
    texts = [t for t in (question, query_en) if t]
    kind: VisualKind | None = None
    for text in texts:
        found = sorted(
            ((m.start(), k) for pattern, k in _KINDS_ASKED for m in pattern.finditer(text.casefold())),
            key=lambda t: t[0],
        )
        if found:
            kind = found[0][1]
            break
    shapes = frozenset(s for s, pattern in _SHAPES for t in texts if pattern.search(t.casefold()))
    periods: list[str] = []
    for text in texts:
        for m in _PERIOD_TOKEN.finditer(text):
            found_period = find_period(m.group(0))
            if found_period is not None and found_period[0].label not in periods:
                periods.append(found_period[0].label)
    skip = {_canon(n) for n in names}
    words = frozenset(w for t in texts for w in _content_words(t) if w not in skip)
    return Cues(kind, shapes, tuple(periods), words)


def names_table(ds: TypedDataset, c: Cues) -> bool:
    """Every word of the question is in the table's title and no row or column holds them all: it names the table,
    not a row of it ("the balance sheet assets", "Zephyra's Q4 results"; but "borrowings at the end of FY24" is the
    "Total borrowings" row of "Note 14: Borrowings")."""
    title = _content_words(ds.title, question=False)
    if not c.words or not all(any(_same(w, t) for t in title) for w in c.words):
        return False
    return _strength(ds, c.words) < len(c.words)


def _strength(ds: TypedDataset, words: frozenset[str]) -> int:
    """How many of the question's words the table's best row or column label holds."""
    labels = [r.label for r in ds.rows] + [col.label for col in series_columns(ds)]
    return max((len(_hits(set(words), label)[0]) for label in labels), default=0)


# ------------------------------------------------------------------ the draft


@dataclass(slots=True)
class Draft:
    """The draft and why it is (or isn't) confident. ``candidates``: the ranked tables (the planner gets the same)."""

    spec: VisualSpec | None
    confident: bool
    reasons: list[str] = field(default_factory=list)
    candidates: list[TypedDataset] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def dataset(self) -> TypedDataset | None:
        return self.candidates[0] if self.candidates and self.spec is not None else None


@dataclass(slots=True)
class _Plan:
    spec: VisualSpec
    series_named: bool  # every series was named by the question, or the only one the table has
    whole_table: bool = False  # the question asks for all of it (headline numbers, dates, the one breakdown)


Layout = Literal["timeline", "period_rows", "measure_columns", "period_columns", "categories"]


def layout(ds: TypedDataset) -> Layout:
    ch = ds.chartability
    if ch.kind == "timeline":
        return "timeline"
    if ch.period_axis == "rows":
        return "period_rows"
    if ch.period_axis == "columns":
        measures = {c.measure for c in series_columns(ds) if c.period is not None and c.measure}
        return "measure_columns" if measures else "period_columns"
    return "categories"


def _best(items: Sequence[tuple[str, str]], words: frozenset[str], limit: int = MAX_DRAFT_SERIES) -> list[str]:
    """The keys of the items (key, label) the question names: every label with a word of the question, except those
    whose words are all in a better match ("EBITDA" when "EBITDA margin" was asked for), best first."""
    scored = []
    for n, (key, label) in enumerate(items):
        hit, extra = _hits(set(words), label)
        if hit:
            scored.append((hit, extra, n, key))
    kept: list[tuple[frozenset[str], int, int, str]] = []
    for item in sorted(scored, key=lambda t: (-len(t[0]), t[1], t[2])):
        hit, extra = item[0], item[1]
        if any(hit < k[0] or (hit == k[0] and extra > k[1]) for k in kept):
            continue
        if any(hit == k[0] and extra == k[1] for k in kept):  # same words, same fit: the first (document order)
            continue
        kept.append(item)
    kept.sort(key=lambda t: t[2])  # in the table's order
    return [k[3] for k in kept[:limit]]


def _additive(unit_kind: str | None) -> bool:
    return unit_kind in (None, "currency", "count", "none")


def _fy_of(label: str) -> str | None:
    m = re.search(r"FY\d{2}$", label)
    return m.group(0) if m else None


def _plans(ds: TypedDataset, pool: Sequence[TypedDataset], c: Cues, language: str) -> list[_Plan]:
    """Specs for ``ds``, best first (the first that builds is the draft)."""
    lang = language if language in ("en", "hi") else "en"
    shape = layout(ds)
    asked = c.kind
    out: list[_Plan] = []
    if names_table(ds, c):  # "Show the balance sheet assets": every word is the table's own, it wants all of it
        c = replace(c, words=frozenset())

    def spec(**kw: object) -> VisualSpec:
        return VisualSpec(language=lang, **kw)  # type: ignore[arg-type]

    if shape == "timeline" or asked == "timeline":
        out.append(_Plan(spec(kind="timeline", datasets=[ds.id]), True, whole_table=True))
        if shape == "timeline":
            return out

    if shape == "period_rows":  # quarters down the rows, measures across: series are columns
        group = continuations(ds, pool)
        fys = [p for p in c.periods if re.fullmatch(r"FY\d{2}", p)]
        if fys:
            group = [d for d in group if any(_fy_of(p) in fys for p in d.chartability.period_order)] or [ds]
        cols = series_columns(ds)
        named = _best([(col.key, col.label) for col in cols], c.words)
        keys = named or [k.key for k in _default_columns(cols)]
        series = [ref(d.id, k) for k in keys for d in group]
        n_periods = sum(len(d.chartability.period_order) for d in group)
        many = len(keys) > 1
        kind: VisualKind = asked or ("line" if n_periods >= 3 else "grouped_bar" if many else "bar")
        if asked is None and n_periods < 3 and len(named) >= 3:
            kind = "table"  # several figures of two years: exact numbers
        out.append(_Plan(spec(kind=kind, datasets=[d.id for d in group], series=series), bool(named) or len(cols) == 1))
        return out

    if shape == "measure_columns":  # segments down the rows, "Revenue FY24 | Revenue FY23 | EBITDA FY24 …" across
        cols = [col for col in series_columns(ds) if col.period is not None]
        measures = list(dict.fromkeys(col.measure for col in cols if col.measure))
        named = _best([(m, m) for m in measures], c.words, limit=2)
        measure = named[0] if named else measures[0]
        mcols = sorted((col for col in cols if col.measure == measure), key=lambda col: col.period.sort_key)  # type: ignore[union-attr]
        if c.periods:
            mcols = [col for col in mcols if col.period.label in c.periods] or mcols  # type: ignore[union-attr]
        latest = mcols[-1]
        unit = latest.unit.kind if latest.unit else None
        parts = _parts(ds)
        rows = _best([(r.key, r.label) for r in parts], c.words)  # "Freight and Digital"
        cat_keys = rows if len(rows) >= 2 else [r.key for r in parts]
        categories = _categories(ds, cat_keys)
        both = len(mcols) >= 2 and ({"comparison", "trend"} & c.shapes or len(c.periods) >= 2)
        if asked is not None:
            kind = asked
        elif both:
            kind = "grouped_bar"
        elif "ranking" in c.shapes or not _additive(unit):
            kind = "bar"
        else:
            kind = "donut"
        shown = mcols if kind in ("grouped_bar", "stacked_bar", "table") else [latest]
        series = [ref(ds.id, col.key) for col in shown]
        if kind == "stacked_bar":  # parts of a whole over periods: the segments are the series
            series = [ref(ds.id, k) for k in cat_keys]
            categories = []
        out.append(
            _Plan(
                spec(kind=kind, datasets=[ds.id], series=series, categories=categories),
                bool(named) or len(measures) == 1,
                whole_table="composition" in c.shapes and len(measures) == 1,
            )
        )
        return out

    if shape == "period_columns":  # metrics down the rows, FY24 | FY23 across: series are rows
        rows = [r for r in ds.rows if r.type != "section"]
        named = _best([(r.key, r.label) for r in rows], c.words)
        periods = [p for p in c.periods if p in ds.chartability.period_order]
        is_kpi = any(o.kind == "kpi" for o in ds.chartability.options)
        if asked == "waterfall" or (asked is None and "composition" in c.shapes and _has_composition(ds)):
            breakdown = _breakdown(ds, named, periods, lang, asked or "donut")
            if breakdown is not None:
                out.append(_Plan(breakdown, bool(named), whole_table=not named))
        if named:
            units = {_row_unit(ds, ds.row(k)) for k in named}  # type: ignore[arg-type]
            if asked is not None:
                kind = asked
            elif "kpi" in c.shapes and is_kpi:
                kind = "kpi"
            elif len(named) == 1:
                kind = "bar"
            elif len(units) == 1:
                kind = "grouped_bar"
            else:
                kind = "kpi" if is_kpi else "bar"
            out.append(
                _Plan(
                    spec(
                        kind=kind,
                        datasets=[ds.id],
                        series=[ref(ds.id, k) for k in named],
                        periods=periods if kind != "kpi" else [],
                    ),
                    True,
                )
            )
        elif is_kpi and asked in (None, "kpi", "comparison"):
            data = [r for r in ds.rows if r.type == "data"][:MAX_DRAFT_SERIES]
            out.append(
                _Plan(
                    spec(kind="kpi", datasets=[ds.id], series=[ref(ds.id, r.key) for r in data]),
                    False,
                    whole_table="kpi" in c.shapes,
                )
            )
        else:
            cols = [col for col in series_columns(ds) if col.period is not None]
            cols.sort(key=lambda col: col.period.sort_key)  # type: ignore[union-attr]
            if periods:
                cols = [col for col in cols if col.period.label in periods] or cols  # type: ignore[union-attr]
            out.append(
                _Plan(
                    spec(kind=asked or "table", datasets=[ds.id], series=[ref(ds.id, col.key) for col in cols]),
                    False,
                    whole_table=asked in (None, "table"),
                )
            )
        return out

    # categories: rows are things (segments, plants, districts), numeric columns are measures
    cols = series_columns(ds)
    named = _best([(col.key, col.label) for col in cols], c.words)
    parts = _parts(ds)
    picked_rows = _best([(r.key, r.label) for r in parts], c.words)
    categories = _categories(ds, picked_rows if len(picked_rows) >= 2 else [r.key for r in parts])
    keys = named or [col.key for col in _default_columns(cols)]
    same_unit = _same_unit_columns(cols, keys[0]) if keys else []
    composition = _has_composition(ds) and bool(keys) and _additive_column(ds, keys[0])
    if asked is not None:
        kind = asked
    elif composition and "composition" in c.shapes and len(keys) == 1:
        kind = "donut"
    elif not named and len(same_unit) >= 2 and ("comparison" in c.shapes or "ranking" not in c.shapes):
        keys, kind = [col.key for col in same_unit][:MAX_DRAFT_SERIES], "grouped_bar"
    elif len(keys) >= 2:
        kind = "grouped_bar"
    elif "ranking" in c.shapes or "comparison" in c.shapes:
        kind = "bar"
    elif composition:
        kind = "donut"
    else:
        kind = "bar"
    if kind == "donut":
        keys = keys[:1]
    out.append(
        _Plan(
            spec(kind=kind, datasets=[ds.id], series=[ref(ds.id, k) for k in keys], categories=categories),
            bool(named) or len(cols) == 1 or len(same_unit) == len(keys) == len(cols),
        )
    )
    return out


def _categories(ds: TypedDataset, keys: Sequence[str]) -> list[str]:
    """The x items to pick, as references; none when they are the table's data rows anyway (the default)."""
    if list(keys) == [r.key for r in ds.rows if r.type == "data"]:
        return []
    return [ref(ds.id, k) for k in keys]


def _default_columns(cols: Sequence[DatasetColumn]) -> list[DatasetColumn]:
    """A table's main measure when the question names none: its first amount (revenue), else its first column."""
    money = [c for c in cols if c.unit is not None and c.unit.kind == "currency"]
    return (money or list(cols))[:1]


def _same_unit_columns(cols: Sequence[DatasetColumn], key: str) -> list[DatasetColumn]:
    first = next((c for c in cols if c.key == key), None)
    if first is None:
        return []
    return [c for c in cols if c.unit == first.unit and c.type == first.type]


def _parts(ds: TypedDataset) -> list[DatasetRow]:
    """The categories of a table: the parts of its total when it has one, else its data rows."""
    totals = [r for r in ds.rows if r.type == "total" and r.parts and r.period is None]
    if totals:
        parts = [ds.row(k) for k in totals[-1].parts]
        rows = [r for r in parts if r is not None]
        if len(rows) >= 2:
            return rows
    return [r for r in ds.rows if r.type == "data"]


def _has_composition(ds: TypedDataset) -> bool:
    return any(o.kind == "composition" for o in ds.chartability.options)


def _additive_column(ds: TypedDataset, key: str) -> bool:
    col = ds.column(key)
    if col is None:
        return False
    return _additive(col.unit.kind if col.unit else None) or (col.unit is not None and col.unit.kind == "percent")


def _row_unit(ds: TypedDataset, row: DatasetRow) -> str:
    units = {v.unit.label if v.unit else "" for v in ds.values if v.row == row.key and v.value is not None}
    return next(iter(units)) if len(units) == 1 else "mixed"


def _breakdown(
    ds: TypedDataset, named: Sequence[str], periods: Sequence[str], language: str, kind: VisualKind
) -> VisualSpec | None:
    """Parts of a whole of a statement (borrowings, cash flows): a named total's parts, else the table's own total,
    at the period asked for (or the latest), as a donut or a waterfall."""
    totals = [r for r in ds.rows if r.type == "total" and len(r.parts) >= 2 and r.period is None]
    if not totals:
        return composition_spec(ds, language) if kind == "donut" else None
    total = next((t for t in totals if t.key in named), None) or next(
        (t for t in reversed(totals) if set(t.parts) & set(named)), None
    )
    total = total or totals[-1]
    cols = [c for c in series_columns(ds) if c.period is not None]
    if periods:
        cols = [c for c in cols if c.period.label in periods] or cols  # type: ignore[union-attr]
    if not cols:
        return None
    col = max(cols, key=lambda c: c.period.sort_key)  # type: ignore[union-attr]
    keys = [*total.parts, total.key] if kind == "waterfall" else list(total.parts)
    return VisualSpec(
        kind=kind,
        datasets=[ds.id],
        series=[ref(ds.id, col.key)],
        categories=[ref(ds.id, k) for k in keys],
        language=language,  # type: ignore[arg-type]
    )


def _two_companies(
    scored: Sequence[tuple[float, TypedDataset]],
    companies: Mapping[str, Collection[str]],
    c: Cues,
    language: str,
) -> tuple[VisualSpec, list[TypedDataset]] | None:
    """A question naming two companies ("Valmora or Zephyra: who runs the better EBITDA margin?"): the same metric
    from each company's best table, side by side (grouped bars over the years, or one period's comparison)."""
    picks: list[tuple[TypedDataset, str]] = []
    for docs in list(companies.values())[:2]:
        for _, ds in scored:
            if ds.document_id not in docs or layout(ds) != "period_columns":
                continue
            rows = [r for r in ds.rows if r.type != "section"]
            named = _best([(r.key, r.label) for r in rows], c.words, limit=1)
            if named:
                picks.append((ds, named[0]))
                break
    if len(picks) < 2:
        return None
    periods = [p for p in c.periods if all(p in ds.chartability.period_order for ds, _ in picks)]
    kind: VisualKind = c.kind or ("comparison" if len(periods) == 1 else "grouped_bar")
    spec = VisualSpec(
        kind=kind,
        datasets=[ds.id for ds, _ in picks],
        series=[ref(ds.id, key) for ds, key in picks],
        periods=periods,
        language=language if language in ("en", "hi") else "en",  # type: ignore[arg-type]
    )
    return spec, [ds for ds, _ in picks]


def draft_visual(
    question: str,
    language: str,
    datasets: Sequence[TypedDataset],
    *,
    query_en: str | None = None,
    source_chunks: Collection[str] = (),
    documents: Collection[str] | None = None,
    source_documents: Collection[str] = (),
    names: Collection[str] = (),
    companies: Mapping[str, Collection[str]] | None = None,
    filenames: Mapping[str, str] | None = None,
    limit: int = 4,
) -> Draft:
    """The draft visual for ``question`` from ``datasets`` (the chat's), and whether it is confident. ``documents``:
    the documents the visual may draw from (``TurnContext.scope``: the company the question names, else the document
    of the turn's top source), ``names`` the names that matched them and
    ``companies`` the documents of each name (``subjects.documents_by_name``); ``source_chunks`` /
    ``source_documents``: the turn's retrieved passages and their documents. Never raises."""
    started = time.perf_counter()
    c = cues(question, query_en, names=names)
    scored = score_candidates(
        question,
        datasets,
        query_en=query_en,
        source_chunks=source_chunks,
        documents=documents,
        source_documents=source_documents,
        names=names,
    )
    candidates = [ds for _, ds in scored[:limit]]
    draft = Draft(None, False, candidates=candidates, scores=[s for s, _ in scored[:limit]])
    pool = {d.id: d for d in datasets}
    check = builds(pool, filenames)
    try:
        if not scored:
            draft.reasons.append("no chartable table")
        elif companies and len(companies) >= 2 and (pair := _two_companies(scored, companies, c, language)):
            spec, tables = pair
            valid = first_valid(spec, pool, check)
            if valid is not None:
                draft.spec = valid
                draft.candidates = [*tables, *(d for d in candidates if d not in tables)][:limit]
                draft.confident = c.kind is None or valid.kind == c.kind
                draft.reasons = [f"two companies: {', '.join(t.title for t in tables)}", f"kind: {valid.kind}"]
        if draft.spec is None and scored:
            _single(draft, scored, list(datasets), pool, check, c, language)
    except Exception as e:  # a bug in the draft must not cost the turn its visual: the planner still runs
        draft.spec, draft.confident = None, False
        draft.reasons.append(f"draft failed: {type(e).__name__}: {e}")
    draft.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return draft


def _single(
    draft: Draft,
    scored: Sequence[tuple[float, TypedDataset]],
    datasets: Sequence[TypedDataset],
    pool: Mapping[str, TypedDataset],
    check: Callable[[VisualSpec], bool],
    c: Cues,
    language: str,
) -> None:
    """The draft from the best candidate's table (and the tables that continue it)."""
    best = scored[0][1]
    title_covers = names_table(best, c)
    for plan in _plans(best, datasets, c, language):
        valid = first_valid(plan.spec, pool, check)
        if valid is None:
            continue
        draft.spec = valid
        # the runner-up: the best table that isn't drawn, nor continues the drawn one (FY23's quarters when FY24's
        # were asked for: the period named chose between them)
        group = {d.id for d in continuations(best, datasets)} | set(valid.datasets)
        others = [(s, d) for s, d in scored if d.id not in group]
        margin = scored[0][0] - others[0][0] if others else float("inf")
        only_timeline = valid.kind == "timeline" and not any(d.chartability.kind == "timeline" for _, d in others)
        # the series it names are in this table only (or here better than anywhere else)
        unique = (
            plan.series_named
            and not title_covers
            and (strength := _strength(best, c.words)) > 0
            and all(_strength(d, c.words) < strength for _, d in others[:5])
        )
        clear = margin >= CLEAR_MARGIN - 1e-6 or only_timeline or bool(unique)
        kind_ok = c.kind is None or valid.kind == c.kind
        same_series = valid.series == plan.spec.series or valid.kind == "timeline"
        whole = plan.whole_table or (title_covers and not plan.series_named)
        series_ok = (plan.series_named or whole) and same_series
        why = (
            "clear"
            if margin >= CLEAR_MARGIN - 1e-6
            else "the only one with those rows"
            if unique
            else "the only timeline"
            if only_timeline
            else f"leads by {margin:.1f}"
        )
        draft.reasons = [
            f"table: {best.title!r} ({why})",
            f"kind: {valid.kind}" + (f" (asked: {c.kind})" if c.kind else ""),
            "series: " + ("named" if plan.series_named else "the whole table" if whole else "a guess"),
        ]
        draft.confident = clear and kind_ok and series_ok
        return
    draft.reasons.append(f"nothing of {best.title!r} builds")


# ------------------------------------------------------------------ the turn's context, and comparing choices


@dataclass(frozen=True, slots=True)
class TurnContext:
    """What a turn knows about where its visual should come from: its retrieved passages and their documents, and
    the documents (companies) its question names.

    A visual draws from **one document's** tables unless the question compares documents or companies (``compare``:
    it names two companies, or says "both companies" / "दोनों कंपनियों"). ``scope`` is that document: the named
    company's, else the one the turn's top retrieved source is from (``top_document``: the passage the answer is
    drawn from), else none yet (the best table's own document then decides). A question that says just "the company's
    quarterly revenue" in a chat of two companies' documents once drew the deck's Q4 table beside the other company's
    FY23 table."""

    source_chunks: tuple[str, ...] = ()
    source_documents: frozenset[str] = frozenset()
    documents: frozenset[str] | None = None
    names: tuple[str, ...] = ()
    companies: dict[str, frozenset[str]] | None = None
    top_document: str | None = None
    compare: bool = False

    @property
    def scope(self) -> frozenset[str] | None:
        """The documents the visual's tables come from (``rank_candidates``'s ``documents``): the named company's,
        else the top source's document, else (a comparison, or no source) unrestricted."""
        if self.documents:
            return self.documents
        if self.compare or self.top_document is None:
            return None
        return frozenset({self.top_document})


def turn_context(
    question: str, query_en: str | None, sources: Sequence[Citation], labels: Mapping[str, str]
) -> TurnContext:
    """``labels``: document id → document label (file name and title), as retrieval labels its chunks. ``sources``:
    the turn's citations, best first."""
    named = named_documents((question, query_en), labels)
    companies = documents_by_name(named, labels) if named else None
    return TurnContext(
        source_chunks=tuple(c.chunk_id for c in sources if c.chunk_id),
        source_documents=frozenset(c.document_id for c in sources if c.document_id),
        documents=named.document_ids if named else None,
        names=named.names if named else (),
        companies=companies,
        top_document=next((c.document_id for c in sources if c.document_id), None),
        compare=len(set((companies or {}).values())) >= 2 or compares_documents((question, query_en)),
    )


def filename_labels(filenames: Mapping[str, str]) -> dict[str, str]:
    """Document labels from file names alone (when retrieval's, with titles, aren't at hand)."""
    return {d: document_label(f, None) for d, f in filenames.items()}


TABLE_SLACK = CLEAR_MARGIN  # the planner's table may trail the draft's by this much and still be "the same match"


def refinement_loses(
    planned: VisualSpec,
    draft: Draft,
    datasets: Mapping[str, TypedDataset],
    c: Cues,
    ctx: TurnContext,
    filenames: Mapping[str, str] | None = None,
) -> str | None:
    """Why the planner's chart must not replace the draft on screen, or None when it covers the question at least as
    well. The planner is a 4B model reading a catalogue; the draft was chosen by code from the question's own words, the
    turn's retrieval and the company it names, and is the better judge of tables (30 of 32 against 25 of 32 on the blind
    set) where the planner is the better judge of kinds. So the planner's chart replaces it only when it is:

    - from the company's documents when the question names one and the draft is, and from the answer's own documents
      when the draft is (a misheard company name had the planner draw another company's table);
    - from one document's tables, unless the question compares documents or companies (``TurnContext.compare``);
    - from a table that matches the question as well (``Draft.scores``, within ``TABLE_SLACK``: an unsure draft is one
      whose runner-up is that close);
    - showing no fewer of the periods the question names, nor, when it names none, fewer points of a time series (a
      quarterly question must not end in a two-point bar chart of a yearly table);
    - showing no fewer of the series and categories the question names."""
    if draft.spec is None:
        return None
    try:
        before, after = (
            resolve(draft.spec, datasets, filenames=filenames),
            resolve(planned, datasets, filenames=filenames),
        )
    except SpecError:
        return None  # one of them doesn't resolve: nothing to compare (the planner's was built before)
    by_id = {d.id: d for d in datasets.values()}
    drawn = [by_id[i] for i in draft.spec.datasets if i in by_id]
    chosen = [by_id[i] for i in planned.datasets if i in by_id]
    if (
        ctx.documents
        and all(d.document_id in ctx.documents for d in drawn)
        and any(d.document_id not in ctx.documents for d in chosen)
    ):
        return "the planner's table is not from the company's documents"
    if not ctx.compare and len({d.document_id for d in chosen}) > 1:
        return "the planner's chart mixes tables of different documents"
    if (
        ctx.source_documents
        and all(d.document_id in ctx.source_documents for d in drawn)
        and any(d.document_id not in ctx.source_documents for d in chosen)
    ):
        return "the planner's table is not from the documents the answer came from"
    score = dict(zip((d.id for d in draft.candidates), draft.scores, strict=False))
    have = max((score[d.id] for d in drawn if d.id in score), default=None)
    got = max((score[d.id] for d in chosen if d.id in score), default=None)
    if have is not None and got is not None and got < have - TABLE_SLACK + 1e-6:
        return f"the draft's table matches the question better ({have:.1f} against {got:.1f})"
    if c.periods and _periods_shown(c, after) < _periods_shown(c, before):
        return "the planner's chart shows fewer of the periods asked for"
    periodic = {"period", "date"}
    if (
        not c.periods
        and before.x_type in periodic
        and after.x_type in periodic
        and len(after.x_items) < len(before.x_items)
    ):
        return f"the planner's chart has fewer periods ({len(after.x_items)} against {len(before.x_items)})"
    if _names_shown(c, after) < _names_shown(c, before):
        return "the planner's chart shows fewer of the series the question names"
    return None


def _periods_shown(c: Cues, r: Resolved) -> int:
    """How many of the periods the question names are on the chart: the period itself, or a quarter or half of the
    fiscal year named ("FY24" on a chart of Q1 FY24 to Q4 FY24)."""
    labels = [i.label for i in r.x_items]
    return sum(1 for p in c.periods if any(x == p or x.endswith(f" {p}") for x in labels))


def _names_shown(c: Cues, r: Resolved) -> int:
    """How many of the chart's series and x items (a donut's slices) carry a word of the question."""
    labels = {s.metric or s.label for s in r.series} | {i.label for i in r.x_items}
    return sum(1 for label in labels if _hits(set(c.words), label)[0])


def same_choice(a: VisualSpec, b: VisualSpec, datasets: Mapping[str, TypedDataset] | None = None) -> bool:
    """The same chart: kind, tables, series and what is on the x axis (titles, highlights and calculations aside).
    With ``datasets`` both are resolved, so "all the segments" and the segments listed one by one are the same."""
    if a.kind != b.kind or set(a.datasets) != set(b.datasets) or set(a.series) != set(b.series):
        return False
    if datasets is not None:
        try:
            ra, rb = resolve(a, datasets), resolve(b, datasets)
        except SpecError:
            pass
        else:
            return [i.label for i in ra.x_items] == [i.label for i in rb.x_items] and [
                e.row.key for e in ra.timeline
            ] == [e.row.key for e in rb.timeline]
    return set(a.periods) == set(b.periods) and set(a.categories) == set(b.categories)
