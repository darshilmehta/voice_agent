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

Not wired into the router or the turns yet: ``VisualPlanner.plan`` is the entry point the integration calls (through
``CanvasService.prepare_visual``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from ...domain.canvas import VISUAL_KINDS
from ...domain.datasets import TypedDataset
from ...providers.llm import LLMClient, LLMError, LLMMessage
from .chartability import series_columns
from .overview import composition_spec, kpi_spec, trend_spec
from .parsing import localized_label
from .spec import MAX_CALCULATIONS, CalcRequest, SpecError, VisualSpec, ref, resolve, row_display

log = logging.getLogger(__name__)

Intent = Literal["requested", "suggested", "none"]

_REQUESTED = re.compile(
    r"\b(show|chart|graph|plot|visuali[sz]e|draw|display|dashboard|diagram|pie|bar\s+chart|line\s+chart|"
    r"put\s+(it|that|this)\s+on\s+(the\s+)?screen|on\s+screen|"
    r"dikhao|dikhaao|dikhaiye|dikhaye|dikha\s+do|dikhana|chart\s+banao|graph\s+banao)\b"
    r"|दिखाओ|दिखाइए|दिखाएं|दिखाएँ|दिखा\s+दो|दिखाना|चार्ट|ग्राफ़|ग्राफ|आरेख",
    re.I,
)
_SUGGESTED = re.compile(
    r"\b(compare|comparison|versus|vs\.?|trend|trends|over\s+(the\s+)?(years|quarters|time|period)|"
    r"year\s+on\s+year|yoy|quarter\s+(by|on)\s+quarter|quarterly|breakdown|break\s+down|split|share|mix|"
    r"composition|growth|grew|grown|increase|decrease|declined?|rise|rose|fell|fall|moved?|changed?|"
    r"how\s+has|how\s+did|bridge|segment(s|-wise)?|by\s+segment|by\s+district|each|all\s+the|"
    r"tulna|badhat|badha|ghata|hissa|rujhan)\b"
    r"|तुलना|रुझान|बढ़त|वृद्धि|बढ़ा|घटा|हिस्सा|बंटवारा|हर\s+तिमाही|तिमाही|सालाना|खंड",
    re.I,
)
_PERIOD_TOKEN = re.compile(r"\b(?:FY\s?'?\d{2,4}|Q[1-4]|H[12]|(?:19|20)\d{2})\b", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def visual_intent(question: str, answer: str | None = None) -> Intent:
    """Is a visual worth it? "requested": the user asked to see something; "suggested": a trend, comparison,
    breakdown, several periods, or an answer with three or more numbers; "none": a plain fact or a chat turn."""
    if _REQUESTED.search(question):
        return "requested"
    if (
        _SUGGESTED.search(question)
        or len({m.group(0).upper().replace(" ", "") for m in _PERIOD_TOKEN.finditer(question)}) >= 2
    ):
        return "suggested"
    if answer is not None and len(_NUMBER.findall(answer)) >= 3:
        return "suggested"
    return "none"


# ------------------------------------------------------------------ candidates

_WORD = re.compile(r"[0-9A-Za-zÀ-ɏऀ-ॣॱ-ॿ]+")
_STOP = frozenset(
    re.split(
        r"\s+",
        "the a an of in on for to and or is was were what how show me please chart graph plot with by from at as it "
        "this that which compare between vs over did has have do does give tell can you their its our ka ki ke ko "
        "aur hai dikhao dikhaiye का की के को और है दिखाओ",
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


def rank_candidates(
    question: str,
    datasets: Sequence[TypedDataset],
    *,
    query_en: str | None = None,
    source_chunks: Sequence[str] = (),
    limit: int = 4,
) -> list[TypedDataset]:
    """Chartable datasets most related to the question: word overlap with the title, labels and periods, the shape
    the question asks for (quarterly, a trend, a breakdown, headline numbers, dates), a boost for the tables the
    turn's retrieval found; the chartability confidence breaks ties."""
    words = _words(question) | (_words(query_en) if query_en else set())
    text = f"{question} {query_en or ''}"
    scored = []
    for ds in datasets:
        if ds.chartability.kind == "none":
            continue
        overlap = len(words & _dataset_words(ds))
        title_hit = len(words & _words(ds.title))
        shape = sum(1.5 for pattern, fits in _WANTS if pattern.search(text) and fits(ds))
        boost = 3 if ds.chunk_id and ds.chunk_id in source_chunks else 0
        score = overlap + title_hit + shape + boost + ds.chartability.confidence
        scored.append((score, -(ds.page_start or 0), ds))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [ds for _, _, ds in scored[:limit]]


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
fills them in from the tables. Answer with JSON only.

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
- "series": what is plotted. Charts other than kpi and table need series in one unit.
- For periods running across a table's columns, the series are its rows; for periods down its rows, its columns.
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
    """The model's choice in VisualSpec form (labels → refs). None for kind "none"."""
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
    return VisualSpec(
        kind=choice["kind"],
        datasets=datasets[:3],
        series=series,
        periods=list(dict.fromkeys(choice.get("periods") or [])),
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


def variants(spec: VisualSpec) -> list[VisualSpec]:
    """Simpler forms of the same choice, tried in order when the spec doesn't resolve."""
    out = [spec]
    plain = spec.model_copy(update={"highlight": [], "calculations": []})
    out.append(plain)
    out.append(plain.model_copy(update={"periods": [], "categories": []}))
    kind = spec.kind
    while kind in _KIND_FALLBACK:
        kind = _KIND_FALLBACK[kind]  # type: ignore[assignment]
        out.append(plain.model_copy(update={"kind": kind}))
        out.append(plain.model_copy(update={"kind": kind, "periods": [], "categories": []}))
    if len(spec.series) > 1:
        out.append(plain.model_copy(update={"series": spec.series[:1]}))
    return out


def first_valid(spec: VisualSpec, datasets: Mapping[str, TypedDataset]) -> VisualSpec | None:
    for candidate in variants(spec):
        try:
            resolve(candidate, datasets)
        except SpecError:
            continue
        return candidate
    return None


def default_spec(ds: TypedDataset, pool: Sequence[TypedDataset], language: str) -> VisualSpec | None:
    """A sensible chart of one dataset without the model: its trend, KPI tiles or composition."""
    makers = {
        "time_series": lambda: trend_spec(ds, pool, language),
        "kpi": lambda: kpi_spec(ds, language),
        "composition": lambda: composition_spec(ds, language),
    }
    order = [o.kind for o in ds.chartability.options]
    for kind in order:
        maker = makers.get(kind)
        spec = maker() if maker else None
        if spec is not None and first_valid(spec, {d.id: d for d in pool}) is not None:
            return spec
    cols = series_columns(ds)
    if ds.chartability.period_axis == "columns":
        rows = [r for r in ds.rows if r.type == "data"][:1]
        series = [ref(ds.id, r.key) for r in rows]
    else:
        series = [ref(ds.id, c.key) for c in cols[:1]]
    if not series:
        return None
    return first_valid(VisualSpec(kind="bar", datasets=[ds.id], series=series, language=language), {ds.id: ds})  # type: ignore[arg-type]


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
        source_chunks: Sequence[str] = (),
        force: bool = False,
    ) -> VisualSpec | None:
        """The visual for ``question`` (None: not worth one, or nothing fits). ``candidates``: the datasets the
        question may draw on (the chat's documents); ``force``: plan even when the question doesn't call for a
        visual."""
        return (
            await self.plan_detailed(
                question,
                language,
                candidates,
                answer=answer,
                filenames=filenames,
                query_en=query_en,
                source_chunks=source_chunks,
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
        source_chunks: Sequence[str] = (),
        force: bool = False,
    ) -> PlanResult:
        started = time.perf_counter()
        intent = visual_intent(question, answer)
        if intent == "none" and not force:
            return PlanResult(None, intent, "none", "the question doesn't call for a visual")
        ranked = rank_candidates(
            question, candidates, query_en=query_en, source_chunks=source_chunks, limit=self.max_candidates
        )
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
                result.reason = "the model chose no visual"
            else:
                valid = first_valid(spec, pool)
                if valid is not None:
                    result.spec, result.source = valid, "model"
                else:
                    result.reason = "the model's choice doesn't resolve"
        except TimeoutError:
            result.reason = f"planner timed out after {self.timeout_s:.1f}s"
        except (LLMError, ValueError) as e:
            result.reason = f"planner failed: {e}"
        if result.spec is None and result.reason != "the model chose no visual" and (intent == "requested" or force):
            # asked to see something and the model couldn't place it: the best candidate's default chart
            fallback = default_spec(ranked[0], list(candidates), language)
            if fallback is not None:
                result.spec, result.source = fallback, "heuristic"
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
