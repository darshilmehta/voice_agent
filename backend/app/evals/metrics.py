"""Retrieval metrics: pure functions over ranked passages and expected targets (no models, no I/O).

Vocabulary
    passage   one retrieved chunk in rank order: document, page range, text.
    target    one expected location from the question set: a document, the pages that state the fact (any of them
              counts) and the evidence strings of the manifest facts it comes from.
    hit       a passage satisfies a target. ``page_hit`` (the headline): same document and the passage's page range
              contains one of the target's pages; a page-less target (DOCX) is matched on evidence text instead.
              ``fact_hit``: same document and the passage text itself contains all evidence strings of one of the
              target's facts (stricter: the right page but the wrong chunk is a miss).

Per question: ``recall@k`` is the share of targets covered by the top k passages (a two-hop question with one of its
two pages in the top 3 scores 0.5); ``success@k`` is 1 when all targets are covered; the reciprocal rank is
1/rank of the first passage that hits any target. Averages are macro (every question counts once). Questions that
should be abstained are excluded from all of these; they feed the abstention metrics.

The answer-or-abstain decision follows the pipeline's gate: answer when the best reranker score is at least the
threshold and nothing vetoes the question (a fiscal year it names that the best passage doesn't state); nothing
retrieved always abstains. ``gate=False`` gives the score gate alone, for comparison.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .textnorm import contains_all, normalize

KS: tuple[int, ...] = (1, 3, 5)
_DIGIT_COMMA = re.compile(r"(\d),(\d)")
_SPACE_PERCENT = re.compile(r"\s+%")
SCORE_BUCKETS: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0000001)
SWEEP_GRID: tuple[float, ...] = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


@dataclass(frozen=True, slots=True)
class Passage:
    """One retrieved chunk."""

    document: str  # manifest file name
    page_start: int | None
    page_end: int | None
    text: str
    chunk_id: str = ""
    score: float | None = None  # reranker score (post-rerank) or fused score (candidates)


@dataclass(frozen=True, slots=True)
class Target:
    """One expected location."""

    document: str
    pages: tuple[int, ...] = ()  # empty: page-less document
    evidence: tuple[tuple[str, ...], ...] = ()  # alternatives; each is a set of strings that must all occur
    approximate: bool = False  # OCR text: match evidence ignoring spaces and punctuation


# ------------------------------------------------------------------ hits


def _span(p: Passage) -> tuple[int, int] | None:
    if p.page_start is None:
        return None
    return p.page_start, p.page_end if p.page_end is not None else p.page_start


def fact_hit(p: Passage, t: Target) -> bool:
    """The passage's own text contains the evidence of one of the target's facts."""
    if p.document != t.document:
        return False
    return any(contains_all(p.text, alt, approximate=t.approximate) for alt in t.evidence)


def page_hit(p: Passage, t: Target) -> bool:
    """The passage is in the target's document and its page range includes one of the target's pages."""
    if p.document != t.document:
        return False
    if not t.pages:
        return fact_hit(p, t) if t.evidence else True
    span = _span(p)
    return span is not None and any(span[0] <= page <= span[1] for page in t.pages)


def exact_page_hit(p: Passage, t: Target) -> bool:
    """Like ``page_hit`` but the passage must not reach beyond the target's pages (a chunk that merges the right
    page with a neighbouring one would cite a page range wider than the evidence)."""
    if p.document != t.document:
        return False
    if not t.pages:
        return fact_hit(p, t) if t.evidence else True
    span = _span(p)
    return span is not None and all(page in t.pages for page in range(span[0], span[1] + 1))


Hit = Callable[[Passage, Target], bool]


# ------------------------------------------------------------------ ranking quality


@dataclass(frozen=True, slots=True)
class RankingScore:
    n_targets: int
    n_passages: int
    covered: dict[int, int]  # k → targets covered by the top k passages
    covered_all: int  # targets covered by any passage in the list
    first_rank: int | None  # 1-based rank of the first passage that hits any target
    all_rank: int | None  # rank by which every target is covered (None if some never is)

    def recall(self, k: int) -> float:
        return self.covered[k] / self.n_targets

    def success(self, k: int) -> bool:
        return self.covered[k] == self.n_targets

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_rank if self.first_rank else 0.0


def score_ranking(
    passages: Sequence[Passage], targets: Sequence[Target], *, ks: Sequence[int] = KS, hit: Hit = page_hit
) -> RankingScore:
    """Coverage of ``targets`` by a ranked list of passages."""
    if not targets:
        raise ValueError("score_ranking needs at least one target (unanswerable questions are not ranked)")
    firsts: list[int | None] = []
    for t in targets:
        firsts.append(next((i for i, p in enumerate(passages, start=1) if hit(p, t)), None))
    found = [r for r in firsts if r is not None]
    return RankingScore(
        n_targets=len(targets),
        n_passages=len(passages),
        covered={k: sum(1 for r in found if r <= k) for k in ks},
        covered_all=len(found),
        first_rank=min(found) if found else None,
        all_rank=max(found) if len(found) == len(targets) else None,
    )


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_ranking(scores: Sequence[RankingScore], *, ks: Sequence[int] = KS) -> dict[str, float | int | None]:
    """Macro-averaged recall@k, success@k, recall@all (every passage in the list) and MRR."""
    out: dict[str, float | int | None] = {"n": len(scores)}
    for k in ks:
        out[f"recall@{k}"] = _mean(s.recall(k) for s in scores if k in s.covered)
        out[f"success@{k}"] = _mean(float(s.success(k)) for s in scores if k in s.covered)
    out["recall@all"] = _mean(s.covered_all / s.n_targets for s in scores)
    out["success@all"] = _mean(float(s.covered_all == s.n_targets) for s in scores)
    out["mrr"] = _mean(s.reciprocal_rank for s in scores)
    return out


def top1_citation(passages: Sequence[Passage], targets: Sequence[Target]) -> dict[str, bool]:
    """Would citing the best passage cite a right page? ``lenient``: its page range includes an expected page;
    ``exact``: the range stays within the expected pages."""
    if not passages:
        return {"lenient": False, "exact": False}
    top = passages[0]
    return {
        "lenient": any(page_hit(top, t) for t in targets),
        "exact": any(exact_page_hit(top, t) for t in targets),
    }


def citation_scores(cited: Sequence[Passage], targets: Sequence[Target]) -> dict[str, float | bool | None]:
    """Quality of the passages an answer actually cites: ``any_correct`` (at least one cited passage is a hit),
    ``precision`` (share of cited passages that are hits), ``recall`` (share of targets covered by the citations)."""
    hits = [any(page_hit(p, t) for t in targets) for p in cited]
    covered = sum(1 for t in targets if any(page_hit(p, t) for p in cited))
    return {
        "cited": len(cited),
        "any_correct": any(hits),
        "precision": (sum(hits) / len(hits)) if hits else None,
        "recall": covered / len(targets) if targets else None,
    }


# ------------------------------------------------------------------ distributions, latency


def percentile(values: Sequence[float], q: float) -> float:
    """The q-quantile (0..1) by linear interpolation; ``values`` must not be empty."""
    if not values:
        raise ValueError("percentile of nothing")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    """n, min, p10, p25, median, p75, p90, max, mean of a sample (all None but n when empty)."""
    if not values:
        return {
            "n": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def histogram(values: Sequence[float], edges: Sequence[float] = SCORE_BUCKETS) -> list[int]:
    """Counts per bucket [edges[i], edges[i+1]); values outside the edges are clamped into the first/last bucket."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(counts)):
            if v < edges[i + 1] or i == len(counts) - 1:
                counts[i] += 1
                break
    return counts


def latency_stats(values_ms: Sequence[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values_ms),
        "mean": sum(values_ms) / len(values_ms),
        "p50": percentile(values_ms, 0.50),
        "p95": percentile(values_ms, 0.95),
        "max": max(values_ms),
    }


# ------------------------------------------------------------------ abstention gate


@dataclass(frozen=True, slots=True)
class GateSample:
    """One question as the confidence gate sees it."""

    top_score: float | None  # best reranker score; None when nothing was retrieved
    gap: float | None  # best minus runner-up
    dense: float | None  # cosine(query, best passage)
    should_answer: bool  # the documents contain the answer
    language: str
    evidence_retrieved: bool = False  # answerable questions: the expected evidence is in the post-rerank top N
    veto: bool = False  # the gate refuses it whatever the score (Confidence.missing_periods)
    subtype: str | None = None  # unanswerable questions: near_miss_year, other_company, ...


def answers_at(sample: GateSample, threshold: float) -> bool:
    """The score gate: answer when the best score reaches the threshold."""
    return sample.top_score is not None and sample.top_score >= threshold


def answers(sample: GateSample, threshold: float, *, gate: bool = True) -> bool:
    """The pipeline's gate at ``threshold``: the score gate, unless vetoed; ``gate=False``: the score gate alone."""
    return answers_at(sample, threshold) and not (gate and sample.veto)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def confusion(samples: Sequence[GateSample], threshold: float, *, gate: bool = True) -> dict[str, float | int | None]:
    """The answer-or-abstain confusion matrix at ``threshold`` (``gate=False``: the score gate alone, see
    ``answers``). "Positive" = the documents hold the answer."""

    def ok(s: GateSample) -> bool:
        return answers(s, threshold, gate=gate)

    aa = sum(1 for s in samples if s.should_answer and ok(s))  # answered, answerable
    ba = sum(1 for s in samples if s.should_answer and not ok(s))  # abstained, answerable (miss)
    au = sum(1 for s in samples if not s.should_answer and ok(s))  # answered, unanswerable (bad)
    bu = sum(1 for s in samples if not s.should_answer and not ok(s))  # abstained, unanswerable
    precision = _ratio(aa, aa + au)  # of the questions answered, how many had an answer
    recall = _ratio(aa, aa + ba)  # of the answerable questions, how many were answered
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    return {
        "threshold": threshold,
        "answered_answerable": aa,
        "abstained_answerable": ba,
        "answered_unanswerable": au,
        "abstained_unanswerable": bu,
        "answerable": aa + ba,
        "unanswerable": au + bu,
        "answer_precision": precision,
        "answer_recall": recall,
        "answer_f1": f1,
        "false_abstain_rate": _ratio(ba, aa + ba),
        "hallucination_risk": _ratio(au, au + bu),
        "abstain_precision": _ratio(bu, ba + bu),
        "abstain_recall": _ratio(bu, au + bu),
        "accuracy": _ratio(aa + bu, len(samples)),
        "answered_with_evidence": sum(1 for s in samples if s.should_answer and ok(s) and s.evidence_retrieved),
    }


def hallucination_by_subtype(
    samples: Sequence[GateSample], threshold: float, *, gate: bool = True
) -> dict[str, dict[str, float | int | None]]:
    """Unanswerable questions answered anyway, per subtype (near misses vs off-topic need different signals)."""
    out: dict[str, dict[str, float | int | None]] = {}
    for s in samples:
        if s.should_answer:
            continue
        row = out.setdefault(s.subtype or "-", {"n": 0, "answered": 0, "hallucination_risk": None})
        row["n"] = int(row["n"] or 0) + 1
        row["answered"] = int(row["answered"] or 0) + answers(s, threshold, gate=gate)
    for row in out.values():
        row["hallucination_risk"] = _ratio(int(row["answered"] or 0), int(row["n"] or 0))
    return dict(sorted(out.items()))


def sweep(
    samples: Sequence[GateSample], thresholds: Sequence[float] = SWEEP_GRID, *, gate: bool = True
) -> list[dict[str, float | int | None]]:
    """``confusion`` at each threshold."""
    return [confusion(samples, t, gate=gate) for t in thresholds]


def auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Probability that a random answerable question scores above a random unanswerable one (ties count half):
    1.0 = the signal separates them perfectly, 0.5 = no information."""
    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _floor(value: float, digits: int = 5) -> float:
    factor = 10**digits
    return math.floor(value * factor) / factor


def candidate_thresholds(samples: Sequence[GateSample]) -> list[float]:
    """Every observed best score, floored to 5 decimals: the only thresholds where the decision can change."""
    return sorted({_floor(s.top_score) for s in samples if s.top_score is not None})


def best_thresholds(
    samples: Sequence[GateSample], *, recall_target: float = 0.95
) -> dict[str, dict[str, float | int | None]]:
    """Two recommendations over every observed score: ``max_f1`` (best F1 of answering; ties go to the higher
    threshold, the safer one) and ``recall_target`` (the highest threshold that still answers at least
    ``recall_target`` of the answerable questions). Empty when either class is missing."""
    if not any(s.should_answer for s in samples) or all(s.should_answer for s in samples):
        return {}
    points = [confusion(samples, t) for t in [0.0, *candidate_thresholds(samples)]]
    best = max(points, key=lambda p: ((p["answer_f1"] or 0.0), float(p["threshold"] or 0.0)))
    out: dict[str, dict[str, float | int | None]] = {"max_f1": best}
    keep = [p for p in points if (p["answer_recall"] or 0.0) >= recall_target]
    if keep:
        out["recall_target"] = max(keep, key=lambda p: float(p["threshold"] or 0.0))
    return out


# ------------------------------------------------------------------ answers


def normalize_answer(text: str) -> str:
    """Comparison form for answer checks: the usual normalisation plus no thousands separators and no space before
    a percent sign ("4,210" == "4210", "21.0 %" == "21.0%")."""
    text = normalize(text)
    prev = None
    while prev != text:
        prev = text
        text = _DIGIT_COMMA.sub(r"\1\2", text)
    return _SPACE_PERCENT.sub("%", text)


def answer_contains_all(answer: str, needles: Sequence[str]) -> bool:
    """Every needle is in the answer; a needle ``"a|b"`` is satisfied by either alternative."""
    haystack = normalize_answer(answer)
    for needle in needles:
        alternatives = [normalize_answer(a) for a in needle.split("|")]
        if not any(a and a in haystack for a in alternatives):
            return False
    return bool(needles)
