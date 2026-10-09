"""Turning per-query records into the numbers of the report (pure: no models, no I/O).

``QueryRecord`` is what the harness measures for one (question, query variant); ``summarize`` aggregates a list of
them into one JSON-friendly dict: ranking quality before and after the reranker (page-level and fact-level),
citation accuracy, the abstention confusion matrix at a threshold, a threshold sweep, score distributions per
language, AUC of the gate signals and latency per stage. See ``metrics`` for the definitions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import metrics as m
from .metrics import GateSample, Passage, Target

VARIANTS = ("raw", "routed", "english", "clean")
STAGES = ("candidate", "rerank")


@dataclass
class AnswerRecord:
    """The answer step for one query (``--answers`` mode)."""

    text: str
    abstained: bool  # the gate refused: the fixed abstention text, no LLM call
    cited: list[Passage] = field(default_factory=list)  # passages of the sources the answer cites
    contains_ok: bool | None = None  # answer_contains found (answerable questions only)
    llm_ms: float | None = None


@dataclass
class QueryRecord:
    """Everything measured for one question under one query variant."""

    qid: str
    variant: str  # raw | routed | clean
    category: str
    language: str
    should_abstain: bool
    question: str
    query: str  # the query text passed to retrieval
    query_en: str | None
    rerank_query: str
    targets: list[Target]
    candidates: list[Passage]  # fused hybrid-search candidates, before the reranker, best first
    ranked: list[Passage]  # after the reranker (rerank_top_n), best first, with rerank scores
    top_score: float | None  # best rerank score; None when nothing was retrieved
    gap: float | None
    dense: float | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    sources: list[Passage] = field(default_factory=list)  # context handed to the answer step (after budget/dedupe)
    answer: AnswerRecord | None = None
    veto: bool = False  # the gate refuses it whatever the score (Confidence.missing_periods, missing_subjects)
    subtype: str | None = None  # unanswerable questions: near_miss_year, other_company, ...

    def gate(self) -> GateSample:
        evidence = False
        if self.targets and not self.should_abstain:
            evidence = m.score_ranking(self.ranked, self.targets).covered_all == len(self.targets)
        return GateSample(
            top_score=self.top_score,
            gap=self.gap,
            dense=self.dense,
            should_answer=not self.should_abstain,
            language=self.language,
            evidence_retrieved=evidence,
            veto=self.veto,
            subtype=self.subtype,
        )

    def answered(self, threshold: float) -> bool:
        """Did the pipeline's gate let the question through at ``threshold``."""
        return m.answers(self.gate(), threshold)


# ------------------------------------------------------------------ per query


def _first(passages: Sequence[Passage], targets: Sequence[Target], hit: m.Hit) -> int | None:
    return m.score_ranking(passages, targets, hit=hit).first_rank


def _brief(p: Passage) -> dict[str, Any]:
    pages = (
        None
        if p.page_start is None
        else (p.page_start if p.page_end in (None, p.page_start) else f"{p.page_start}-{p.page_end}")
    )
    return {
        "document": p.document,
        "pages": pages,
        "score": None if p.score is None else round(p.score, 4),
        "chunk": p.chunk_id,
    }


def query_row(r: QueryRecord, threshold: float) -> dict[str, Any]:
    """A compact JSON row for one query (the rows of results.json; used for the miss and gate-error listings)."""
    row: dict[str, Any] = {
        "id": r.qid,
        "variant": r.variant,
        "category": r.category,
        "language": r.language,
        "abstain": r.should_abstain,
        "question": r.question,
        "query": r.query,
        "query_en": r.query_en,
        "top_score": r.top_score,
        "gap": r.gap,
        "dense": r.dense,
        "answered": r.answered(threshold),
        "veto": r.veto,
        "top": [_brief(p) for p in r.ranked[:3]],
        "timings_ms": r.timings_ms,
    }
    if r.subtype:
        row["subtype"] = r.subtype
    if r.targets:
        row["expected"] = [{"document": t.document, "pages": list(t.pages)} for t in r.targets]
        row["candidate_first_rank"] = _first(r.candidates, r.targets, m.page_hit)
        row["rerank_first_rank"] = _first(r.ranked, r.targets, m.page_hit)
        row["rerank_all_covered"] = m.score_ranking(r.ranked, r.targets).covered_all == len(r.targets)
        row["fact_first_rank"] = _first(r.ranked, r.targets, m.fact_hit)
    if r.answer is not None:
        row["answer"] = {
            "text": r.answer.text,
            "abstained": r.answer.abstained,
            "contains_ok": r.answer.contains_ok,
            "cited": [_brief(p) for p in r.answer.cited],
        }
    return row


# ------------------------------------------------------------------ aggregation


def _ranking_block(records: Sequence[QueryRecord], stage: str, hit: m.Hit) -> dict[str, Any]:
    scores = [
        m.score_ranking(r.candidates if stage == "candidate" else r.ranked, r.targets, hit=hit)
        for r in records
        if r.targets and not r.should_abstain
    ]
    return m.summarize_ranking(scores)


def _group(records: Sequence[QueryRecord], key: Callable[[QueryRecord], str]) -> dict[str, list[QueryRecord]]:
    groups: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return dict(sorted(groups.items()))


def _rate(flags: Iterable[bool]) -> float | None:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def _slice(records: Sequence[QueryRecord], threshold: float) -> dict[str, Any]:
    """The compact block used per category and per language."""
    answerable = [r for r in records if not r.should_abstain]
    unanswerable = [r for r in records if r.should_abstain]
    gate_ok_answerable = _rate(r.answered(threshold) for r in answerable)
    gate_ok_unanswerable = _rate(not r.answered(threshold) for r in unanswerable)
    out: dict[str, Any] = {"n": len(records), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}
    if answerable:
        out["candidate"] = _ranking_block(records, "candidate", m.page_hit)
        out["rerank"] = _ranking_block(records, "rerank", m.page_hit)
    out["answered_rate_answerable"] = gate_ok_answerable
    out["abstained_rate_unanswerable"] = gate_ok_unanswerable
    return out


def _distribution(samples: Sequence[GateSample], attr: str = "top_score") -> dict[str, Any]:
    def values(answerable: bool) -> list[float]:
        return [getattr(s, attr) or 0.0 for s in samples if s.should_answer == answerable]

    ans, unans = values(True), values(False)
    return {
        "answerable": m.describe(ans),
        "unanswerable": m.describe(unans),
        "hist_answerable": m.histogram(ans),
        "hist_unanswerable": m.histogram(unans),
        "auc": m.auc(ans, unans),
    }


def summarize(records: Sequence[QueryRecord], *, threshold: float) -> dict[str, Any]:
    """All aggregate numbers for a set of query records (one per question)."""
    answerable = [r for r in records if not r.should_abstain]
    samples = [r.gate() for r in records]
    languages = sorted({r.language for r in records})
    out: dict[str, Any] = {
        "n": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(records) - len(answerable),
        "threshold": threshold,
    }
    if answerable:
        out["page_level"] = {s: _ranking_block(records, s, m.page_hit) for s in STAGES}
        out["fact_level"] = {s: _ranking_block(records, s, m.fact_hit) for s in STAGES}
        top1 = [m.top1_citation(r.ranked, r.targets) for r in answerable]
        out["citation_top1"] = {
            "lenient": _rate(c["lenient"] for c in top1),
            "exact": _rate(c["exact"] for c in top1),
            "mean_page_span": _mean_span(r.ranked[0] for r in answerable if r.ranked),
        }
        if any(r.sources for r in answerable):  # the harness fills sources when the gate lets a question through
            out["context"] = {
                "all_targets_in_context": _rate(
                    bool(r.sources) and m.score_ranking(r.sources, r.targets).covered_all == len(r.targets)
                    for r in answerable
                ),
                "mean_sources": _mean([float(len(r.sources)) for r in answerable if r.sources]),
            }
    out["by_category"] = {k: _slice(v, threshold) for k, v in _group(records, lambda r: r.category).items()}
    out["by_language"] = {k: _slice(v, threshold) for k, v in _group(records, lambda r: r.language).items()}

    out["abstention"] = {
        "threshold": threshold,
        "overall": m.confusion(samples, threshold),
        "by_language": {lang: m.confusion([s for s in samples if s.language == lang], threshold) for lang in languages},
        "by_subtype": m.hallucination_by_subtype(samples, threshold),
        "score_gate": m.confusion(samples, threshold, gate=False),  # without vetoes, for comparison
        "vetoed": sum(1 for s in samples if s.veto),
    }
    out["sweep"] = {
        "overall": m.sweep(samples),
        "by_language": {lang: m.sweep([s for s in samples if s.language == lang]) for lang in languages},
    }
    out["best_thresholds"] = {
        "overall": m.best_thresholds(samples),
        "by_language": {lang: m.best_thresholds([s for s in samples if s.language == lang]) for lang in languages},
    }
    out["score_distribution"] = {
        "overall": _distribution(samples),
        "by_language": {lang: _distribution([s for s in samples if s.language == lang]) for lang in languages},
    }
    out["signal_auc"] = {
        attr: {
            "overall": m.auc(
                [getattr(s, attr) or 0.0 for s in samples if s.should_answer],
                [getattr(s, attr) or 0.0 for s in samples if not s.should_answer],
            ),
            "by_language": {
                lang: m.auc(
                    [getattr(s, attr) or 0.0 for s in samples if s.should_answer and s.language == lang],
                    [getattr(s, attr) or 0.0 for s in samples if not s.should_answer and s.language == lang],
                )
                for lang in languages
            },
        }
        for attr in ("top_score", "gap", "dense")
    }
    stages: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for stage, ms in r.timings_ms.items():
            stages[stage].append(ms)
    out["latency_ms"] = {stage: m.latency_stats(v) for stage, v in stages.items()}

    answered = [r for r in records if r.answer is not None]
    if answered:
        out["answers"] = _answer_block(answered)
    return out


def _mean_span(passages: Iterable[Passage]) -> float | None:
    spans = [(p.page_end or p.page_start) - p.page_start + 1 for p in passages if p.page_start is not None]
    return sum(spans) / len(spans) if spans else None


def _answer_block(records: Sequence[QueryRecord]) -> dict[str, Any]:
    """Answer-step results: accuracy of ``answer_contains``, citation quality and refusals."""
    answerable = [r for r in records if not r.should_abstain and r.answer is not None]
    unanswerable = [r for r in records if r.should_abstain and r.answer is not None]
    cites = [m.citation_scores(r.answer.cited, r.targets) for r in answerable if r.answer and not r.answer.abstained]
    return {
        "n": len(records),
        "answerable": {
            "n": len(answerable),
            "contains_ok": _rate(bool(r.answer and r.answer.contains_ok) for r in answerable),
            "abstained": _rate(bool(r.answer and r.answer.abstained) for r in answerable),
            "cites_a_correct_page": _rate(bool(c["any_correct"]) for c in cites),
            "citation_precision": _mean([c["precision"] for c in cites if c["precision"] is not None]),
            "citation_recall": _mean([c["recall"] for c in cites if c["recall"] is not None]),
            "no_citation": _rate(c["cited"] == 0 for c in cites),
            "by_category": {
                k: _rate(bool(r.answer and r.answer.contains_ok) for r in v)
                for k, v in _group(answerable, lambda r: r.category).items()
            },
            "by_language": {
                k: _rate(bool(r.answer and r.answer.contains_ok) for r in v)
                for k, v in _group(answerable, lambda r: r.language).items()
            },
        },
        "unanswerable": {
            "n": len(unanswerable),
            "abstained": _rate(bool(r.answer and r.answer.abstained) for r in unanswerable),
            "answered_anyway": [r.qid for r in unanswerable if r.answer and not r.answer.abstained],
        },
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ------------------------------------------------------------------ variants


def pipeline_records(raw: Mapping[str, QueryRecord], routed: Mapping[str, QueryRecord]) -> list[QueryRecord]:
    """What the chat pipeline will do once the router exists (§3.4): the routed query (with the router's English
    query) where there is one, the raw question otherwise."""
    return [routed.get(qid, rec) for qid, rec in raw.items()]


def summarize_variants(by_variant: Mapping[str, Mapping[str, QueryRecord]], *, threshold: float) -> dict[str, Any]:
    """``by_variant[variant][question id]``. Produces the headline ``pipeline`` summary and the comparisons:
    raw vs routed vs English-only on the questions that have an English query (Hindi/Hinglish), noisy vs clean on ASR
    questions."""
    raw = by_variant.get("raw", {})
    routed = by_variant.get("routed", {})
    english = by_variant.get("english", {})
    clean = by_variant.get("clean", {})
    out: dict[str, Any] = {}
    if raw:
        out["pipeline"] = summarize(pipeline_records(raw, routed), threshold=threshold)
        out["raw"] = summarize(list(raw.values()), threshold=threshold)
    if routed:
        out["routed"] = summarize(list(routed.values()), threshold=threshold)
        out["raw_on_routed"] = summarize([raw[q] for q in routed if q in raw], threshold=threshold)
    if english:
        out["english"] = summarize(list(english.values()), threshold=threshold)
    if clean:
        out["clean"] = summarize(list(clean.values()), threshold=threshold)
        out["raw_on_clean"] = summarize([raw[q] for q in clean if q in raw], threshold=threshold)
    return out
