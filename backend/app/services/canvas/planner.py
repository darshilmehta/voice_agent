"""The visual planner (docs/DESIGN.md §12.1, workstream 5, the model's part): should this question get a visual,
and which one.

    question → visual_intent (keywords, no model): "requested" ("show me", "chart", "dikhao", "दिखाओ"),
               "suggested" (trends, comparisons, breakdowns, two or more periods, an answer with 3+ numbers) or "none"
             → candidate datasets ranked by overlap with the question (and the turn's retrieved table chunks)
             → one JSON call to the router model (qwen3:4b-instruct) with a schema whose enums are the candidates'
               real rows, columns, periods and categories (labels such as "D2 · Revenue"), bounded output, timeout
             → VisualSpec (labels mapped back to dataset ids and keys) → resolve; on failure a simpler variant of
               the same choice (no highlight or calculations, no filters, a compatible kind)
             → for a "requested" visual the model can't place (timeout, invalid): a default chart of the best
               candidate (``source: "heuristic"``)

A turn draws a draft first (``draft.py``: code, the same candidates) and calls ``VisualPlanner.plan_detailed`` only when
the draft isn't confident, once its answer's text is complete (``CanvasService.prepare_visual`` through
``conversation.TurnVisual``), with the draft's candidates; a spoken edit the rules can't read asks it again with the
visual as context. Candidates are company-aware (``rank_candidates``): a question that names one company's documents
is offered only their tables.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from ...domain.canvas import VISUAL_KINDS
from ...domain.datasets import TypedDataset
from ...providers.llm import LLMClient, LLMError, LLMMessage
from .builder import build_visual
from .chartability import series_columns
from .overview import composition_spec, kpi_spec, trend_spec
from .parsing import localized_label
from .spec import MAX_CALCULATIONS, CalcRequest, SpecError, VisualSpec, ref, resolve, row_display, split_ref

log = logging.getLogger(__name__)

Intent = Literal["requested", "suggested", "none"]

_GATE_LETTER = "A-Za-z0-9ऀ-ॣ०-ॿ"


def _gate(alternatives: str) -> re.Pattern[str]:
    """Whole words or phrases, in English, Hinglish or Hindi (\\b fails after Hindi vowel signs)."""
    return re.compile(f"(?<![{_GATE_LETTER}])(?:{alternatives})(?![{_GATE_LETTER}])", re.I)


# The user asks to see something.
_REQUESTED = _gate(
    r"show|chart|charts|graph|graphs|plot|visuali[sz]e|draw|display|dashboard|diagram|pie|bar\s+chart|line\s+chart|"
    r"put\s+(?:it|that|this|them)\s+on\s+(?:the\s+)?screen|on\s+(?:the\s+)?screen|"
    r"put\s+(?:\S+\s+){1,6}?up\s+(?:against|next\s+to|beside|alongside|side\s+by\s+side)|"
    r"let\s+me\s+see|can\s+i\s+see|i(?:\s+want|\s+would\s+like|['\u2019]d\s+like)\s+to\s+see|"
    r"dikhao|dikhaao|dikhaiye|dikhayiye|dikhaye|dikha\s+do|dikhana|chart\s+banao|graph\s+banao|"
    r"दिखाओ|दिखाइए|दिखाइये|दिखाएं|दिखाएँ|दिखा\s+दो|दिखाना|चार्ट|ग्राफ़|ग्राफ|आरेख"
)
# ... unless "show" is what a document does ("What does the report show about debt?").
_SHOWS = re.compile(r"\b(?:does|did|do|will|would)\s+(?:[^\s?.!।]+\s+){0,5}?show\b|\bshows\b", re.I)
_VERBS_OF_CHANGE = (
    r"move[ds]?|moving|change[ds]?|changing|grow|grows|grew|grown|growing|evolve[ds]?|trend\w*|perform\w*|fare[ds]?|"
    r"stack\s+up|compare[ds]?|do|did|go|went|look"
)
_CATEGORY_NOUNS = (
    r"segments?|business(?:es)?|divisions?|districts?|regions?|grades?|courses?|categor(?:y|ies)|plants?|"
    r"facilit(?:y|ies)|units?"
)
_SUPERLATIVE = r"most|highest|lowest|largest|biggest|smallest|least|fastest|slowest|best|worst|top|sabse|सबसे"
# The words suggest a visual helps: a trend, a comparison, a breakdown, a ranking of categories, a bridge, headline
# numbers, dates. Plain facts ("What share of revenue came from exports?", "dividend per share", "each day") don't.
_SUGGESTED = _gate(
    # trends
    r"trends?|trending|over\s+(?:the\s+)?(?:years|quarters|months|time|period)|year\s+(?:on|over)\s+year|yoy|"
    r"quarter\s+(?:by|on|after|to|over)\s+quarter|quarterly|(?:each|every)\s+quarter|"
    r"across\s+(?:the\s+|all\s+(?:the\s+)?)?(?:quarters|years|segments|businesses)|"
    rf"how\s+(?:has|have|did|does|do|is|are)\s+(?:\S+\s+){{0,6}}?(?:{_VERBS_OF_CHANGE})|"
    # comparisons
    r"compare[ds]?|comparing|comparison|versus|vs\.?|stack\s+up|stacks\s+up|relative\s+to|"
    r"against\s+(?:the\s+)?(?:previous|last|prior|same|a\s+year)|"
    r"(?:better|worse|higher|lower|bigger|smaller|stronger|weaker)\s+than|"
    r"(?:which|who)\s+(?:\S+\s+){0,5}?(?:better|worse|stronger|weaker)|"
    r"(?:from|since|over|against|versus|than|with)\s+(?:the\s+)?(?:last|previous|prior)\s+year|"
    r"(?:a|one)\s+year\s+(?:ago|earlier|before)|"
    r"tulna|mukable|mukabale|muqable|pichle\s+saal|तुलना|मुक़ाबले|मुकाबले|पिछले\s+साल|"
    # breakdowns
    r"break\s*down|breakdown|break\s+(?:\S+\s+){1,4}?down|split\s+(?:by|across|between|of|into)|composition|"
    r"distribution|(?:revenue|sales|business|product|segment|portfolio)\s+mix|made\s+up\s+of|consists?\s+of|"
    rf"comprises?|by\s+(?:{_CATEGORY_NOUNS})|segment-?wise|district-?wise|each\s+(?:{_CATEGORY_NOUNS})|"
    r"who\s+owns|ownership|shareholding|contribut\w*\s+(?:the\s+)?most|kis\s+kis|"
    r"बंटवारा|बँटवारा|किस[\s-]+किस|जिलेवार|ज़िलेवार|खंडवार|"
    # rankings of categories
    rf"(?:which|what|kaun\s*s[aei]|kaunsa|kaunsi|कौन\s*स[ाीे]|किस)\s+(?:\S+\s+){{0,6}}?(?:{_SUPERLATIVE})|"
    rf"(?:{_SUPERLATIVE})\s+(?:\S+\s+){{0,6}}?(?:which|kaun|किस|कौन)|rank\w*|top\s+(?:\d+|three|five|ten)|"
    # bridges and headline numbers
    r"walk\s+(?:me\s+|us\s+)?through|bridge|waterfall|turns?\s+into|from\s+(?:revenue|sales|ebitda)\s+(?:\S+\s+){0,3}?to|"
    r"headline\s+(?:numbers|figures|metrics)|key\s+(?:numbers|figures|metrics|financials|highlights|indicators)|"
    r"highlights|at\s+a\s+glance|snapshot|scorecard|ek\s+nazar|एक\s+नज़र|एक\s+नजर|मुख्य\s+(?:आंकड़े|आँकड़े|वित्तीय)|"
    # dates
    r"timeline|milestones|(?:key|important)\s+dates|schedule|"
    r"rujhan|रुझान|हर\s+तिमाही|तिमाही\s+दर\s+तिमाही|har\s+(?:quarter|timahi|saal)"
)
# "Q3 FY24" is one period (a single quarter's figure is a fact, not a comparison); "Q3 and Q4", "FY23 vs FY24" are two.
_PERIOD_TOKEN = re.compile(
    r"\b(?:(?:Q[1-4]|H[12])\s*FY\s?'?\d{2,4}|FY\s?'?\d{2,4}|Q[1-4]|H[12]|(?:19|20)\d{2})\b", re.I
)
# A figure, not digits inside an identifier: "L24119GJ1994PLC023871" (a CIN), "GHI/2024/00418377" (a policy number) or
# the grade "L5" are not three numbers worth a chart.
_NUMBER = re.compile(r"(?<![A-Za-z0-9/_])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9/_])")


def visual_intent(question: str, answer: str | None = None) -> Intent:
    """Is a visual worth it? "requested": the user asked to see something ("show", "chart", "put … up against",
    "dikhao", "दिखाओ"); "suggested": a trend, a comparison, a breakdown, a ranking of categories ("which segment grew
    the fastest"), a bridge, headline numbers, dates, two or more periods, or an answer with three or more numbers;
    "none": a plain fact or a chat turn (EN, HI, Hinglish)."""
    text = question.casefold()
    if _REQUESTED.search(text) and not (_SHOWS.search(text) and not _REQUESTED.search(_SHOWS.sub(" ", text))):
        return "requested"
    if (
        _SUGGESTED.search(text)
        or len({m.group(0).upper().replace(" ", "") for m in _PERIOD_TOKEN.finditer(question)}) >= 2
    ):
        return "suggested"
    if answer is not None and len(_NUMBER.findall(_PERIOD_TOKEN.sub(" ", answer))) >= 3:  # figures, not "FY24"
        return "suggested"
    return "none"


# ------------------------------------------------------------------ candidates

_WORD = re.compile(r"[0-9A-Za-zÀ-ɏऀ-ॣॱ-ॿ]+")
_STOP = frozenset(
    re.split(
        r"\s+",
        "the a an of in on for to and or is was were what how show me please chart graph plot with by from at as it "
        "this that which compare between vs over did has have do does give tell can you their its our ka ki ke ko "
        "aur hai dikhao dikhaiye karo end against last year years at its whole का की के को और है दिखाओ",
    )
)
# Words that say which shape of table the question wants (English, Hindi, Hinglish).
_WANTS = (
    (re.compile(r"quarter|तिमाही|timahi|q[1-4]\b", re.I), lambda ds: ds.chartability.granularity == "quarter"),
    (
        re.compile(r"trend|over\s+the\s+years|year\s+on\s+year|grow|growth|rujhan|रुझान|वृद्धि", re.I),
        lambda ds: any(o.kind == "time_series" for o in ds.chartability.options),
    ),
    (
        re.compile(r"segment|breakdown|split|share|mix|composition|hissa|हिस्सा|खंड|बंटवारा", re.I),
        lambda ds: any(o.kind == "composition" for o in ds.chartability.options),
    ),
    (
        re.compile(r"highlight|headline|key\s+(numbers|metrics|figures)|kpi|overall|at\s+a\s+glance|performance", re.I),
        lambda ds: ds.chartability.kind == "kpi",
    ),
    (re.compile(r"date|deadline|timeline|when|तिथि|कब|tareekh", re.I), lambda ds: ds.chartability.kind == "timeline"),
)


def _stem(word: str) -> str:
    """ "quarterly" → "quarter", "segments" → "segment": enough to match a question to table labels."""
    if not word.isascii() or len(word) <= 4:
        return word
    for suffix, keep in (("ies", "y"), ("ly", ""), ("ing", ""), ("es", ""), ("ed", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + keep
    return word


def _words(text: str) -> set[str]:
    return {_stem(w.lower()) for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 1}


def _dataset_words(ds: TypedDataset) -> set[str]:
    texts = [ds.title, *(r.label for r in ds.rows), *(c.label for c in ds.columns), *ds.chartability.period_order]
    return set().union(*(_words(t) for t in texts))


RETRIEVED_TABLE_BOOST = 3  # the turn's retrieval found this table
SOURCE_DOCUMENT_BOOST = 6  # the table is in a document the turn's sources come from (strongly preferred)


def score_candidates(
    question: str,
    datasets: Sequence[TypedDataset],
    *,
    query_en: str | None = None,
    source_chunks: Collection[str] = (),
    documents: Collection[str] | None = None,
    source_documents: Collection[str] = (),
    names: Collection[str] = (),
) -> list[tuple[float, TypedDataset]]:
    """Every chartable dataset with its score, best first (see ``rank_candidates``). ``names``: the company names the
    question was matched to its documents by; they choose the documents, so they don't count as words ("Zephyra" in
    the title "Zephyra at a glance" says nothing about which of Zephyra's tables is meant)."""
    words = _words(question) | (_words(query_en) if query_en else set())
    words -= {_stem(n.casefold()) for n in names}
    text = f"{question} {query_en or ''}"
    chartable = [ds for ds in datasets if ds.chartability.kind != "none"]
    if documents:  # the question names a company: only its documents' tables (when it has chartable ones)
        chartable = [ds for ds in chartable if ds.document_id in documents] or chartable
    scored = []
    for ds in chartable:
        overlap = len(words & _dataset_words(ds))
        title_words = words & _words(ds.title)
        # a period in the title ("Quarterly performance: FY24") tells the year apart, its words say what the table is
        title_hit = sum(0.5 if _PERIOD_TOKEN.fullmatch(w) else 1 for w in title_words)
        shape = sum(1.5 for pattern, fits in _WANTS if pattern.search(text) and fits(ds))
        boost = RETRIEVED_TABLE_BOOST if ds.chunk_id and ds.chunk_id in source_chunks else 0
        boost += SOURCE_DOCUMENT_BOOST if ds.document_id in source_documents else 0
        score = overlap + 2 * title_hit + shape + boost + ds.chartability.confidence  # the title says most
        scored.append((score, -(ds.page_start or 0), ds))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [(score, ds) for score, _, ds in scored]


def rank_candidates(
    question: str,
    datasets: Sequence[TypedDataset],
    *,
    query_en: str | None = None,
    source_chunks: Collection[str] = (),
    documents: Collection[str] | None = None,
    source_documents: Collection[str] = (),
    names: Collection[str] = (),
    limit: int = 4,
) -> list[TypedDataset]:
    """Chartable datasets most related to the question: word overlap with the title, labels and periods, the shape
    the question asks for (quarterly, a trend, a breakdown, headline numbers, dates), a boost for the tables the
    turn's retrieval found and for the documents its sources come from; the chartability confidence breaks ties.

    Company-aware (§12.1): ``documents`` are the documents the question names (``subjects.named_documents``: "Valmora's
    segments" names the Valmora documents); when they have chartable tables, only those are candidates, so neither
    the draft nor the planner can pick the other company's lookalike table, however many words it shares with the
    question. ``source_documents``: the documents of the turn's retrieved sources (which since #37 keep to a named
    company's passages), strongly preferred. ``names``: the names that matched ``documents``."""
    scored = score_candidates(
        question,
        datasets,
        query_en=query_en,
        source_chunks=source_chunks,
        documents=documents,
        source_documents=source_documents,
        names=names,
    )
    return [ds for _, ds in scored[:limit]]


# ------------------------------------------------------------------ the model-facing schema


@dataclass(slots=True)
class Catalog:
    """The candidates as the model sees them: aliases (D1, D2), labels per option, and the way back to refs."""

    aliases: dict[str, TypedDataset] = field(default_factory=dict)
    series: dict[str, str] = field(default_factory=dict)  # "D1 · Revenue" → "<dataset id>:<key>"
    categories: dict[str, str] = field(default_factory=dict)
    periods: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # the prompt's description of the tables

    def alias_of(self, dataset_id: str) -> str:
        return next(a for a, ds in self.aliases.items() if ds.id == dataset_id)


def _unique_label(table: dict[str, str], label: str, value: str) -> str:
    out, n = label, 2
    while out in table and table[out] != value:
        out, n = f"{label} ({n})", n + 1
    table[out] = value
    return out


def build_catalog(candidates: Sequence[TypedDataset], filenames: Mapping[str, str], language: str) -> Catalog:
    cat = Catalog()
    periods: dict[str, tuple[int, int]] = {}
    for n, ds in enumerate(candidates, start=1):
        alias = f"D{n}"
        cat.aliases[alias] = ds
        ch = ds.chartability
        page = f" p.{ds.page_start}" if ds.page_start else ""
        head = f'{alias} "{ds.title}" ({filenames.get(ds.document_id, "")}{page}) — {ch.kind.replace("_", " ")}'
        if ch.period_axis:
            head += f"; periods in {ch.period_axis}: {', '.join(ch.period_order)}"
        cat.lines.append(head)
        rows = [r for r in ds.rows if r.type != "section"]
        cols = series_columns(ds)

        def unit_of(u: Any) -> str:
            text = localized_label(u, language)
            return f" [{text}]" if text else ""

        row_items = [f"{alias} · {row_display(ds, r)}" for r in rows]
        col_items = [f"{alias} · {c.label}" for c in cols]
        row_labels = [
            _unique_label(cat.series, label, ref(ds.id, r.key)) for label, r in zip(row_items, rows, strict=True)
        ]
        col_labels = [
            _unique_label(cat.series, label, ref(ds.id, c.key)) for label, c in zip(col_items, cols, strict=True)
        ]
        for label, r in zip(row_items, rows, strict=True):
            _unique_label(cat.categories, label, ref(ds.id, r.key))
        for label, c in zip(col_items, cols, strict=True):
            _unique_label(cat.categories, label, ref(ds.id, c.key))
        cat.lines.append(
            "  rows: "
            + "; ".join(
                f"{lab.split(' · ', 1)[1]}{unit_of(r.unit)}{' (total)' if r.type == 'total' else ''}"
                for lab, r in zip(row_labels, rows, strict=True)
            )
        )
        cat.lines.append(
            "  columns: "
            + "; ".join(f"{lab.split(' · ', 1)[1]}{unit_of(c.unit)}" for lab, c in zip(col_labels, cols, strict=True))
        )
        for p in [*(c.period for c in ds.columns if c.period), *(r.period for r in ds.rows if r.period)]:
            if p is not None:
                periods.setdefault(p.label, p.sort_key)
    cat.periods = sorted(periods, key=lambda label: periods[label])
    return cat


def _literal(values: Sequence[str]) -> Any:
    return Literal[tuple(values)] if values else str  # type: ignore[valid-type]


def planner_schema(cat: Catalog) -> type[BaseModel]:
    """The JSON schema the model fills, with this request's enums."""
    series = _literal(list(cat.series))
    x_values = _literal(list(dict.fromkeys([*cat.periods, *cat.categories])))
    calc = create_model(
        "PlannedCalculation",
        __config__=ConfigDict(extra="forbid"),
        op=(Literal["growth", "cagr", "diff", "share"], ...),
        series=(series, ...),
    )
    return create_model(
        "PlannedVisual",
        __config__=ConfigDict(extra="forbid"),
        kind=(Literal[(*VISUAL_KINDS, "none")], ...),  # type: ignore[valid-type]
        datasets=(list[_literal(list(cat.aliases))], Field(min_length=1, max_length=3)),  # type: ignore[valid-type]
        series=(list[series], Field(min_length=0, max_length=6)),  # type: ignore[valid-type]
        periods=(list[_literal(cat.periods)], Field(default_factory=list, max_length=12)),  # type: ignore[valid-type]
        categories=(list[_literal(list(cat.categories))], Field(default_factory=list, max_length=8)),  # type: ignore[valid-type]
        highlight=(list[x_values], Field(default_factory=list, max_length=2)),  # type: ignore[valid-type]
        calculations=(list[calc], Field(default_factory=list, max_length=2)),  # type: ignore[valid-type]
        title=(str, Field(default="", max_length=80)),
    )


SYSTEM_PROMPT = """You pick ONE chart for the user's question from the tables listed. You never write numbers: code \
fills them in from the tables. Answer with JSON only, on one line, without line breaks or indentation; leave out \
empty lists.

Kinds:
- line: a trend over 3 or more periods
- bar: values compared across 2 periods or across categories (one or a few series)
- grouped_bar: several series side by side per period or category
- stacked_bar: parts of a whole over periods (e.g. segment revenue FY23 vs FY24)
- donut: parts of a whole at one period (e.g. revenue by segment in FY24)
- waterfall: a bridge from a start value through steps to an end value (cash at the start, cash flows, cash at the end)
- kpi: a few headline numbers with their change (e.g. "key highlights", "how did the company do")
- comparison: one or two values side by side (two years, or two companies)
- table: exact figures, many rows
- timeline: dates and events
- none: no table can show what was asked

Rules:
- "datasets": the tables you use (usually one).
- "series": what is plotted. grouped_bar, stacked_bar, donut and waterfall need series in one unit.
- For periods running across a table's columns, the series are its rows; for periods down its rows, its columns.
- donut and waterfall plot ONE series (a column such as "Revenue FY24"); the slices or steps are "categories".
- To compare two companies or documents, take the same row from each table (e.g. "D1 · Revenue" and "D3 · Revenue").
- "periods": only the periods the user asked about (empty = all).
- "categories": only the rows or columns to show (empty = all).
- "highlight": the period or category the question is about, if any.
- "calculations": growth, cagr, diff or share, only when asked for (e.g. "by how much did it grow").
- "title": short, in the user's language, no numbers."""


def planner_messages(question: str, language: str, cat: Catalog, answer: str | None = None) -> list[LLMMessage]:
    lang = "Hindi" if language == "hi" else "English"
    user = [f"Question: {question}", f"Language for the title: {lang}"]
    if answer:
        user.append(f"Spoken answer: {answer[:400]}")
    user.append("Tables:")
    user += cat.lines
    return [LLMMessage("system", SYSTEM_PROMPT), LLMMessage("user", "\n".join(user))]


# ------------------------------------------------------------------ planning


@dataclass(slots=True)
class PlanResult:
    spec: VisualSpec | None
    intent: Intent
    source: Literal["model", "heuristic", "none"]
    reason: str = ""
    latency_ms: int = 0
    candidates: list[str] = field(default_factory=list)  # dataset ids offered
    raw: dict[str, Any] | None = None  # the model's output


def to_spec(choice: Mapping[str, Any], cat: Catalog, language: str) -> VisualSpec | None:
    """The model's choice in VisualSpec form (labels → refs). None for kind "none". Periods the chosen tables don't
    have are dropped (the model fills the filter in for a table without periods); a timeline takes its dates from the
    table's date column, so rows chosen as its series are let go."""
    if choice.get("kind") == "none":
        return None
    series = [cat.series[s] for s in dict.fromkeys(choice.get("series") or []) if s in cat.series]
    datasets = [cat.aliases[a].id for a in choice.get("datasets") or [] if a in cat.aliases]
    for s in series:
        dataset_id = s.split(":", 1)[0]
        if dataset_id not in datasets:
            datasets.append(dataset_id)
    if not datasets:
        return None
    by_id = {ds.id: ds for ds in cat.aliases.values()}
    available = {
        p.label
        for d in datasets
        if d in by_id
        for p in [*(c.period for c in by_id[d].columns), *(r.period for r in by_id[d].rows)]
        if p is not None
    }
    categories = [cat.categories[c] for c in choice.get("categories") or [] if c in cat.categories]
    categories = [c for c in categories if c.split(":", 1)[0] in datasets]
    highlight = []
    for h in choice.get("highlight") or []:
        if h in cat.categories:
            highlight.append(cat.categories[h])
        elif h in cat.periods:
            highlight.append(h)
    calcs = [
        CalcRequest(op=c["op"], series=cat.series[c["series"]])
        for c in choice.get("calculations") or []
        if c.get("series") in cat.series and cat.series[c["series"]] in series
    ][:MAX_CALCULATIONS]
    kind = choice["kind"]
    return VisualSpec(
        kind=kind,
        datasets=datasets[:3],
        series=[] if kind == "timeline" else series,
        periods=[p for p in dict.fromkeys(choice.get("periods") or []) if p in available],
        categories=categories,
        highlight=highlight[:3],
        calculations=calcs,
        title=(choice.get("title") or None),
        language=language,  # type: ignore[arg-type]
    )


_KIND_FALLBACK = {
    "line": "bar",
    "donut": "bar",
    "stacked_bar": "grouped_bar",
    "grouped_bar": "bar",
    "waterfall": "bar",
    "comparison": "bar",
}
# Kinds that show one series over categories: rows chosen as their "series" are the categories.
_ONE_SERIES = frozenset({"donut", "waterfall"})


def _rows_of_one_dataset(
    spec: VisualSpec, datasets: Mapping[str, TypedDataset]
) -> tuple[TypedDataset, list[str]] | None:
    refs = [split_ref(s) for s in spec.series]
    if len(refs) < 2 or len({d for d, _ in refs}) != 1:
        return None
    ds = datasets.get(refs[0][0])
    if ds is None:
        return None
    keys = [k for _, k in refs if (r := ds.row(k)) is not None and r.type != "section"]
    return (ds, keys) if len(keys) == len(refs) else None


def _measure_column(ds: TypedDataset, periods: Sequence[str]) -> str | None:
    """The column a transposed spec runs down: the asked period's (or the latest) column of the first measure."""
    cols = series_columns(ds)
    if periods:
        cols = [c for c in cols if c.period is not None and c.period.label in periods] or cols
    measures = list(dict.fromkeys(c.measure for c in cols if c.measure))
    if measures:
        cols = [c for c in cols if c.measure == measures[0]]
    if not cols:
        return None
    return max(cols, key=lambda c: c.period.sort_key if c.period else (0, 0)).key


def transposed(spec: VisualSpec, datasets: Mapping[str, TypedDataset]) -> VisualSpec | None:
    """Several rows of one table chosen as "series" (segments for a donut, cash flows for a waterfall) read as the
    categories of one measured column: the period asked for, else the latest."""
    found = _rows_of_one_dataset(spec, datasets)
    if found is None:
        return None
    ds, rows = found
    col = _measure_column(ds, spec.periods)
    if col is None:
        return None
    return spec.model_copy(
        update={
            "datasets": [ds.id],
            "series": [ref(ds.id, col)],
            "categories": [ref(ds.id, k) for k in rows],
            "periods": [],
            "highlight": [],
            "calculations": [],
        }
    )


def bridges(spec: VisualSpec, datasets: Mapping[str, TypedDataset]) -> list[VisualSpec]:
    """Waterfalls that add up: each checked total of the table with its parts before it (the cash flows before
    "Net increase in cash"), those whose parts the model chose first."""
    if spec.kind != "waterfall" or not spec.datasets or spec.datasets[0] not in datasets:
        return []
    ds = datasets[spec.datasets[0]]
    chosen = {split_ref(r)[1] for r in [*spec.series, *spec.categories] if r.startswith(ds.id + ":")}
    col = next(
        (split_ref(s)[1] for s in spec.series if (c := ds.column(split_ref(s)[1])) is not None and c.period), None
    )
    col = col or _measure_column(ds, spec.periods)
    if col is None:
        return []
    totals = [r for r in ds.rows if r.type == "total" and len(r.parts) >= 2 and r.period is None]
    totals.sort(key=lambda r: -len(chosen & {*r.parts, r.key}))
    return [
        VisualSpec(
            kind="waterfall",
            datasets=[ds.id],
            series=[ref(ds.id, col)],
            categories=[ref(ds.id, k) for k in [*t.parts, t.key]],
            title=spec.title,
            language=spec.language,
        )
        for t in totals
    ]


def variants(spec: VisualSpec, datasets: Mapping[str, TypedDataset] | None = None) -> list[VisualSpec]:
    """The spec, then forms of the same choice the model may have meant, tried in order when it doesn't build: its
    rows as categories (``transposed``), waterfalls that add up (``bridges``), no highlight or calculations, no
    filters, a compatible kind, its first series alone."""
    datasets = datasets or {}
    out = [spec]
    flipped = transposed(spec, datasets)
    if flipped is not None and spec.kind in _ONE_SERIES:
        out.append(flipped)
    out += bridges(flipped or spec, datasets)
    plain = spec.model_copy(update={"highlight": [], "calculations": []})
    out.append(plain)
    out.append(plain.model_copy(update={"periods": [], "categories": []}))
    if flipped is not None and spec.kind not in _ONE_SERIES:
        out.append(flipped)
    kind = spec.kind
    while kind in _KIND_FALLBACK:
        kind = _KIND_FALLBACK[kind]  # type: ignore[assignment]
        out.append(plain.model_copy(update={"kind": kind}))
        out.append(plain.model_copy(update={"kind": kind, "periods": [], "categories": []}))
        if flipped is not None:
            out.append(flipped.model_copy(update={"kind": kind}))
    if len(spec.series) > 1:
        out.append(plain.model_copy(update={"series": spec.series[:1]}))
    return out


def _single_unit_categories(spec: VisualSpec, datasets: Mapping[str, TypedDataset]) -> VisualSpec | None:
    """A column mixing units ("tpa" for plants, "seats" for offices) charted over the rows in its main unit."""
    if len(spec.series) != 1:
        return None
    dataset_id, key = split_ref(spec.series[0])
    ds = datasets.get(dataset_id)
    if ds is None or ds.column(key) is None:
        return None
    values = [
        v for v in ds.values if v.col == key and v.value is not None and (r := ds.row(v.row)) and r.type == "data"
    ]
    units = Counter(v.unit for v in values)
    if len(units) < 2:
        return None
    main = units.most_common(1)[0][0]
    rows = [v.row for v in values if v.unit == main]
    return (
        spec.model_copy(update={"categories": [ref(ds.id, r) for r in rows], "periods": []}) if len(rows) >= 2 else None
    )


def first_valid(
    spec: VisualSpec,
    datasets: Mapping[str, TypedDataset],
    check: Callable[[VisualSpec], bool] | None = None,
) -> VisualSpec | None:
    """The first of ``variants`` that resolves (and passes ``check``: the caller builds it, which also checks a
    waterfall's arithmetic)."""
    tried: list[VisualSpec] = []
    for candidate in variants(spec, datasets):
        if candidate in tried:
            continue
        tried.append(candidate)
        try:
            resolve(candidate, datasets)
        except SpecError as e:
            if any("mixes units" in p for p in e.problems):
                narrowed = _single_unit_categories(candidate, datasets)
                if narrowed is not None and narrowed not in tried:
                    tried.append(narrowed)
                    try:
                        resolve(narrowed, datasets)
                    except SpecError:
                        continue
                    if check is None or check(narrowed):
                        return narrowed
            continue
        if check is None or check(candidate):
            return candidate
    return None


def builds(
    datasets: Mapping[str, TypedDataset], filenames: Mapping[str, str] | None = None
) -> Callable[[VisualSpec], bool]:
    """A check for ``first_valid``: the spec resolves and its visual builds (a waterfall's steps add up, a requested
    calculation holds)."""

    def check(spec: VisualSpec) -> bool:
        try:
            r = resolve(spec, datasets, filenames=filenames)
            build_visual(r, visual_id="vis_plan", project_id="", chat_id=None, filenames=filenames or {}, now=_NOW)
        except (SpecError, AssertionError):
            return False
        return True

    return check


_NOW = datetime(2000, 1, 1, tzinfo=UTC)


def default_spec(ds: TypedDataset, pool: Sequence[TypedDataset], language: str) -> VisualSpec | None:
    """A sensible chart of one dataset without the model: its trend, KPI tiles or composition."""
    by_id = {d.id: d for d in pool}
    check = builds(by_id)
    makers = {
        "time_series": lambda: trend_spec(ds, pool, language),
        "kpi": lambda: kpi_spec(ds, language),
        "composition": lambda: composition_spec(ds, language),
    }
    for kind in [o.kind for o in ds.chartability.options]:
        maker = makers.get(kind)
        spec = maker() if maker else None
        if spec is not None and (valid := first_valid(spec, by_id, check)) is not None:
            return valid
    cols = series_columns(ds)
    if ds.chartability.period_axis == "columns":
        rows = [r for r in ds.rows if r.type == "data"][:1]
        series = [ref(ds.id, r.key) for r in rows]
    else:
        series = [ref(ds.id, c.key) for c in cols[:1]]
    if not series:
        return None
    return first_valid(VisualSpec(kind="bar", datasets=[ds.id], series=series, language=language), by_id, check)  # type: ignore[arg-type]


NO_VISUAL = "the model chose no visual"  # PlanResult.reason when the model answered kind "none"


class VisualPlanner:
    """Chooses a visual for a question with the router model. Long-lived; stateless between calls."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str | None = None,
        timeout_s: float = 6.0,
        max_tokens: int = 200,
        max_candidates: int = 4,
    ) -> None:
        self.llm = llm
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.max_candidates = max_candidates

    async def plan(
        self,
        question: str,
        language: str,
        candidates: Sequence[TypedDataset],
        *,
        answer: str | None = None,
        filenames: Mapping[str, str] | None = None,
        query_en: str | None = None,
        source_chunks: Collection[str] = (),
        documents: Collection[str] | None = None,
        source_documents: Collection[str] = (),
        names: Collection[str] = (),
        force: bool = False,
    ) -> VisualSpec | None:
        """The visual for ``question`` (None: not worth one, or nothing fits). ``candidates``: the datasets the
        question may draw on (the chat's documents); ``force``: plan even when the question doesn't call for a
        visual; ``documents`` / ``source_documents`` / ``names``: the company-aware ranking (``rank_candidates``)."""
        return (
            await self.plan_detailed(
                question,
                language,
                candidates,
                answer=answer,
                filenames=filenames,
                query_en=query_en,
                source_chunks=source_chunks,
                documents=documents,
                source_documents=source_documents,
                names=names,
                force=force,
            )
        ).spec

    async def plan_detailed(
        self,
        question: str,
        language: str,
        candidates: Sequence[TypedDataset],
        *,
        answer: str | None = None,
        filenames: Mapping[str, str] | None = None,
        query_en: str | None = None,
        source_chunks: Collection[str] = (),
        documents: Collection[str] | None = None,
        source_documents: Collection[str] = (),
        names: Collection[str] = (),
        ranked: Sequence[TypedDataset] | None = None,
        force: bool = False,
        fallback: bool = True,
    ) -> PlanResult:
        """``ranked``: the candidates to offer, already ranked (the draft's: the same tables, in the same order);
        otherwise ``rank_candidates``. ``fallback``: a requested visual the model can't place gets the best
        candidate's default chart (off when a draft is already on screen: it stays instead)."""
        started = time.perf_counter()
        intent = visual_intent(question, answer)
        if intent == "none" and not force:
            return PlanResult(None, intent, "none", "the question doesn't call for a visual")
        if ranked is None:
            ranked = rank_candidates(
                question,
                candidates,
                query_en=query_en,
                source_chunks=source_chunks,
                documents=documents,
                source_documents=source_documents,
                names=names,
                limit=self.max_candidates,
            )
        ranked = list(ranked)[: self.max_candidates]
        if not ranked:
            return PlanResult(None, intent, "none", "no chartable table")
        pool = {d.id: d for d in candidates}
        cat = build_catalog(ranked, filenames or {}, language)
        schema = planner_schema(cat)
        result = PlanResult(None, intent, "none", candidates=[d.id for d in ranked])
        try:
            choice = await asyncio.wait_for(
                self.llm.generate_json(
                    planner_messages(question, language, cat, answer),
                    schema,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                ),
                self.timeout_s,
            )
            raw = choice.model_dump()
            result.raw = raw
            spec = to_spec(raw, cat, language)
            if spec is None:
                result.reason = NO_VISUAL
            else:
                valid = first_valid(spec, pool, builds(pool, filenames))
                if valid is not None:
                    result.spec, result.source = valid, "model"
                else:
                    result.reason = "the model's choice doesn't resolve"
        except TimeoutError:
            result.reason = f"planner timed out after {self.timeout_s:.1f}s"
        except (LLMError, ValueError) as e:
            result.reason = f"planner failed: {e}"
        if result.spec is None and result.reason != NO_VISUAL and (intent == "requested" or force) and fallback:
            # asked to see something and the model couldn't place it: the best candidate's default chart
            default = default_spec(ranked[0], list(candidates), language)
            if default is not None:
                result.spec, result.source = default, "heuristic"
        result.latency_ms = round((time.perf_counter() - started) * 1000)
        log.info(
            "visual plan: intent=%s source=%s kind=%s %sms%s",
            intent,
            result.source,
            result.spec.kind if result.spec else None,
            result.latency_ms,
            f" ({result.reason})" if result.reason else "",
        )
        return result
