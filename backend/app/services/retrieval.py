"""Retrieval (docs/DESIGN.md §3.2): hybrid search → rerank → top N with a confidence signal.

    query (+ English query_en from the router, §3.4) → BGE-M3 dense + sparse
      → Qdrant: top prefetch_k dense + top prefetch_k sparse, filtered to the project (and chat's documents), RRF
      → bge-reranker-v2-m3 scores (with query_en when given, except Hindi passages: the user's own words;
        the reranker is calibrated best within one language, §9.2)
      → top rerank_top_n + confidence (top score, gap to the runner-up, dense similarity)

``search`` and ``rerank`` are public so the voice loop can start retrieval speculatively on the raw utterance and
rerank once the router's English query arrives (§3.3 d, §9.5). Deciding to answer or abstain is the caller's job,
with ``Confidence.above_threshold`` as the gate: the best reranker score reaches ``min_rerank_score``, the best
passage states every fiscal year the question names (tuned on the phase-2 eval set, docs/DESIGN.md §9.2) and, in a
chat of several documents, it is about the company or entity the question names (``subjects.py``: a question about
Valmora is not answered from Zephyra's deck).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..providers.ingestion import Chunk, compose_embed_text
from ..providers.registry import Container
from ..providers.retrieval import Embedder, Reranker, RetrievalFilters, SearchHit, VectorStore
from ..settings import RetrievalSection
from .subjects import NamedDocuments, named_documents

log = logging.getLogger(__name__)

RRF_K = 60  # the usual reciprocal-rank-fusion constant (also Qdrant's)
LABEL_TTL_S = 300.0  # how long a document's label is remembered (it changes only when the document is re-indexed)


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: Chunk
    rerank_score: float
    fused_score: float  # from hybrid search (comparable within one search only)
    dense_score: float | None  # cosine(query, chunk), best over the queries searched
    search_rank: int  # 1-based position before reranking


@dataclass(frozen=True, slots=True)
class Confidence:
    top_score: float  # reranker score of the best passage
    gap: float  # top score minus runner-up's (the top score itself when there is a single candidate)
    dense_similarity: float | None  # cosine(query, best passage)
    # The gate: answer only when this is True. top_score ≥ retrieval.min_rerank_score, the best passage states
    # every fiscal period the question names (``missing_periods`` empty) and is about the document the question names
    # (``missing_subjects`` empty).
    above_threshold: bool
    missing_periods: tuple[int, ...] = ()  # fiscal years (FY24 → 24) asked about but absent from the best passage
    # Names (lower case, "valmora") the question is about that none of the passages found are about: the question
    # names one company's documents, and only another's came back (see ``prefer_named_documents``).
    missing_subjects: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    rerank_query: str
    chunks: list[RankedChunk]  # best first, at most rerank_top_n
    candidate_count: int  # passages the reranker scored
    confidence: Confidence | None  # None when nothing matched
    timings_ms: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ pure helpers


def fuse_hit_lists(lists: Sequence[Sequence[SearchHit]], *, limit: int, k: int = RRF_K) -> list[SearchHit]:
    """Merge ranked hit lists (one per query) by reciprocal rank fusion, keeping each chunk once with its best
    dense similarity. A single list is returned as is (already fused by the store)."""
    if len(lists) == 1:
        return list(lists[0][:limit])
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            prev = best.get(cid)
            if prev is None:
                best[cid] = hit
            elif hit.dense_score is not None and (prev.dense_score is None or hit.dense_score > prev.dense_score):
                best[cid] = SearchHit(prev.chunk, prev.score, hit.dense_score)
    order = sorted(scores, key=lambda cid: -scores[cid])  # stable: ties keep first-seen order
    return [SearchHit(best[c].chunk, scores[c], best[c].dense_score) for c in order[:limit]]


def rank_by_scores(hits: Sequence[SearchHit], scores: Sequence[float]) -> list[RankedChunk]:
    """Hits ordered by reranker score, best first; ties keep the search order."""
    if len(hits) != len(scores):
        raise ValueError(f"{len(scores)} scores for {len(hits)} hits")
    ranked = [
        RankedChunk(h.chunk, float(s), h.score, h.dense_score, i)
        for i, (h, s) in enumerate(zip(hits, scores, strict=True), start=1)
    ]
    return sorted(ranked, key=lambda r: (-r.rerank_score, r.search_rank))


def confidence_of(
    ranked: Sequence[RankedChunk],
    min_score: float,
    *,
    queries: Sequence[str | None] = (),
    missing_subjects: tuple[str, ...] = (),
) -> Confidence | None:
    """The confidence signals of a reranked list and the gate's decision. ``queries``: the question as asked and its
    English query; the fiscal periods they name must be stated by the best passage (see ``missing_periods``).
    ``missing_subjects``: from ``prefer_named_documents``; a question that names a document nothing found is about
    is never answered."""
    if not ranked:
        return None
    top = ranked[0]
    runner_up = ranked[1].rerank_score if len(ranked) > 1 else 0.0
    missing = period_veto(queries, top.chunk)
    return Confidence(
        top_score=top.rerank_score,
        gap=top.rerank_score - runner_up,
        dense_similarity=top.dense_score,
        above_threshold=top.rerank_score >= min_score and not missing and not missing_subjects,
        missing_periods=missing,
        missing_subjects=missing_subjects,
    )


# ------------------------------------------------------------------ the documents a question names


def prefer_named_documents(
    ranked: Sequence[RankedChunk], named: NamedDocuments | None
) -> tuple[list[RankedChunk], tuple[str, ...]]:
    """Reranked passages restricted to what the question names: (passages, missing subjects).

    The reranker scores relevance, not whose figure it is: "How many electric vehicles does Valmora run?" scored
    Zephyra's fleet 0.61 (phase-2 eval) and the answer step quoted it as Valmora's. When the question names documents
    (``subjects.named_documents``) and the best passage is neither from one of them nor mentions the name, only the
    passages that are (from a named document, or saying the name) are kept, best first. When none is, nothing found is
    about what was asked: the list is returned as it was, with the names as the missing subjects, which the gate
    treats as not covered. A best passage that is about the question's subject, a question naming no document and a
    single-document chat change nothing."""
    if named is None or not ranked:
        return list(ranked), ()

    def about(r: RankedChunk) -> bool:
        return r.chunk.document_id in named.document_ids or named.mentioned_in(r.chunk.embed_text)

    if about(ranked[0]):
        return list(ranked), ()
    kept = [r for r in ranked if about(r)]
    return (kept, ()) if kept else (list(ranked), named.names)


# ------------------------------------------------------------------ fiscal periods (near-miss abstention)
#
# The reranker scores relevance, not answerability: "Valmora's EBITDA margin for FY25" scores 0.9+ against the FY24
# figures (phase-2 eval: every near-miss-year question passed a score gate). A question that names a fiscal year is
# answered only when the passage it would be answered from states that year.

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# FY24, FY 24, FY'24, FY2024, FY 2024-25, FY24-25, Q4FY24 (the period is named by the year it ends in)
_FISCAL_YEAR = re.compile(r"(?<![A-Za-z])FY\s?'?(\d{4}|\d{2})(?:\s?[-\u2013/]\s?(\d{4}|\d{2}))?\b", re.IGNORECASE)
# 2023-24, 2024-2025, 2023/24 (Indian financial years, also inside Hindi text: वित्त वर्ष 2024-25)
_YEAR_SPAN = re.compile(r"\b(?:19|20)\d{2}\s?[-\u2013/]\s?((?:19|20)?\d{2})\b(?![-/.]\d)")  # not "2024-03-31"
_YEAR = re.compile(r"\b(?:19|20)(\d{2})\b")
# A month before a year places a date in one financial year (April to March): "31 March 2024" is FY24, "June 2024" FY25.
_JAN_MAR = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|जनवरी|फ़रवरी|फरवरी|मार्च"
_APR_DEC = (
    r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r"|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर"
)
_DATED_YEAR = re.compile(rf"(?:\b|(?<=\s))(?:({_JAN_MAR})|({_APR_DEC}))\.?,?\s+(?:19|20)(\d{{2}})\b", re.IGNORECASE)
# The end of an Indian fiscal year (April to March): "31 March 2024", "March 31, 2024", "31st March, 2024", "March
# 2024", "31.03.2024", "31/03/24", "2024-03-31", "31 मार्च 2024" all mean the end of FY24 (the final real run: "net debt
# on 31 March 2024" was answered with FY23's ₹1,188 crore, from a passage about 31 March 2023, 3 of 3 times).
_YEAR_END = re.compile(
    r"(?<![A-Za-z])(?:31(?:st)?\s+)?(?:mar(?:ch)?|मार्च)\.?,?\s+(?:31(?:st)?,?\s+)?((?:19|20)\d{2})(?!\d)"
    r"|(?<![\d.])31\s*([./-])\s*0?3\s*\2\s*((?:19|20)?\d{2})(?![\d.]\d)"
    r"|(?<![\d.])((?:19|20)\d{2})\s*([./-])\s*03\s*\5\s*31(?!\d)",
    re.IGNORECASE,
)


def fiscal_year_ends(text: str) -> list[int]:
    """The fiscal years whose last day ``text`` names (31 March, or March of a year), as the two-digit year they end
    in, in order: "net debt on 31 March 2024" → [24]."""
    out: list[int] = []
    for m in _YEAR_END.finditer(text.translate(_DEVANAGARI_DIGITS)):
        year = int((m.group(1) or m.group(3) or m.group(4))[-2:])
        if year not in out:
            out.append(year)
    return out


def with_fiscal_years(query: str) -> str:
    """``query`` with the fiscal year of each year-end date it names and doesn't name as a fiscal year already: "What
    was net debt on 31 March 2024?" → "What was net debt on 31 March 2024? (FY24)". Searched and reranked like that,
    a passage about FY24 meets the question in its own words."""
    named = asked_periods(query, dates=False)
    extra = [y for y in fiscal_year_ends(query) if y not in named]
    return f"{query} ({', '.join(f'FY{y:02d}' for y in extra)})" if extra else query


def asked_periods(text: str, *, dates: bool = True) -> set[int]:
    """Fiscal years a question names explicitly, as the two-digit year they end in (FY24, FY 2023-24, 2023-24 → 24),
    and with ``dates`` the fiscal years whose end it names ("31 March 2024" → 24). A bare calendar year ("in 2024") is
    not one: it is ambiguous and often part of a date."""
    text = text.translate(_DEVANAGARI_DIGITS)
    out = {int((end or start)[-2:]) for start, end in _FISCAL_YEAR.findall(text)}
    out.update(int(end[-2:]) for end in _YEAR_SPAN.findall(text))
    if dates:
        out.update(fiscal_year_ends(text))
    return out


def stated_periods(text: str) -> set[int]:
    """Fiscal years a passage covers: the explicit ones, the year of each dated month ("31 March 2024" → FY24,
    "June 2024" → FY25) and, for any other year, both fiscal years it can fall in: generous, so that only a clear
    mismatch abstains."""
    text = text.translate(_DEVANAGARI_DIGITS)
    out = asked_periods(text)
    for early, _late, yy in _DATED_YEAR.findall(text):
        out.add(int(yy) if early else (int(yy) + 1) % 100)
    undated = _DATED_YEAR.sub(" ", text)
    for yy in _YEAR.findall(undated):
        out.update((int(yy), (int(yy) + 1) % 100))
    return out


def missing_periods(queries: Sequence[str | None], passage: str) -> tuple[int, ...]:
    """The fiscal years the queries name that ``passage`` doesn't state, sorted (empty: nothing missing). A passage
    that states no year at all can't contradict the question ("EBITDA margin 18.2%" on a slide of an FY24 deck):
    nothing is missing then."""
    asked = _asked(queries)
    if not asked:
        return ()
    stated = stated_periods(passage)
    return tuple(sorted(asked - stated)) if stated else ()


def _asked(queries: Sequence[str | None]) -> set[int]:
    asked: set[int] = set()
    for q in queries:
        if q:
            asked |= asked_periods(q)
    return asked


def passage_text(chunk: Chunk) -> str:
    """What a passage itself says: its headings, overlap and text, without its document's label (every passage of
    "valmora_annual_report_fy24.pdf" states FY24 through its label)."""
    return compose_embed_text(chunk.heading_path, chunk.overlap_text, chunk.text)


def dated_to_another_year(queries: Sequence[str | None], chunk: Chunk) -> tuple[int, ...]:
    """The fiscal years the queries name, when the passage dates its figures to the end of another fiscal year only:
    its own text names a year-end ("at 31 March 2023") and neither that nor any fiscal year it names explicitly is one
    asked for (empty otherwise). The final real run answered "net debt on 31 March 2024" with "Recap of FY23 … Net
    debt stood at ₹ 1,188 crore at 31 March 2023", 3 of 3: through its document's label ("… fy24", "2023-24") that
    passage "stated" FY24. A passage that only compares with another year ("₹ 481 crore, up 45.8% on FY23") dates
    nothing and is left to ``missing_periods``, as before."""
    asked = _asked(queries)
    if not asked:
        return ()
    own = passage_text(chunk)
    ends = set(fiscal_year_ends(own))
    if not ends or ends & asked or asked_periods(own) & asked:
        return ()
    return tuple(sorted(asked))


def period_veto(queries: Sequence[str | None], chunk: Chunk) -> tuple[int, ...]:
    """Why the gate can't answer from this passage: the asked fiscal years it misses (``missing_periods``, on the
    passage with its label) or, when it dates its figures to another year-end, all of them; empty when none."""
    return missing_periods(queries, chunk.embed_text) or dated_to_another_year(queries, chunk)


PERIOD_PROMOTION = 0.5  # a passage of the asked year goes first if it scores at least this share of the best one


def prefer_stated_periods(
    ranked: Sequence[RankedChunk], queries: Sequence[str | None], *, min_score: float = 0.0
) -> list[RankedChunk]:
    """Reranked passages with those the period check accepts first, when the best one is dated to another year-end
    (``dated_to_another_year``) and they are about as relevant (at least ``PERIOD_PROMOTION`` of its score, and
    ``min_score``), each group in its order: the gate reads the best passage, and the answer the first ones ("net debt
    on 31 March 2024": the FY24 passages before the recap of FY23). Otherwise unchanged: a best passage that misses the
    asked year (an FY25 question over FY24 documents) is vetoed as before, without looking further."""
    if not ranked or not dated_to_another_year(queries, ranked[0].chunk):
        return list(ranked)
    floor = max(min_score, ranked[0].rerank_score * PERIOD_PROMOTION)
    fine = [r for r in ranked if r.rerank_score >= floor and not period_veto(queries, r.chunk)]
    if not fine:
        return list(ranked)
    return [*fine, *(r for r in ranked if r not in fine)]


# ------------------------------------------------------------------ service


class RetrievalService:
    def __init__(self, embedder: Embedder, reranker: Reranker, store: VectorStore, config: RetrievalSection) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.store = store
        self.config = config
        self._labels: dict[str, tuple[float, str]] = {}  # document id → (when fetched, its label)

    @classmethod
    def from_container(cls, container: Container) -> RetrievalService:
        embedder, reranker, store = container["embeddings"], container["reranker"], container["vector_store"]
        for provider, kind in ((embedder, Embedder), (reranker, Reranker), (store, VectorStore)):
            if not isinstance(provider, kind):
                raise TypeError(f"expected a {kind.__name__} provider, got {type(provider).__name__}")
        return cls(embedder, reranker, store, container.settings.retrieval)  # type: ignore[arg-type]

    async def search(
        self,
        query: str,
        *,
        project_id: str,
        document_ids: Sequence[str] | None = None,
        query_en: str | None = None,
    ) -> list[SearchHit]:
        """Hybrid candidates for ``query`` (and ``query_en`` when it differs, fused by RRF), at most prefetch_k."""
        hits, _ = await self._search(query, project_id=project_id, document_ids=document_ids, query_en=query_en)
        return hits

    async def _search(
        self, query: str, *, project_id: str, document_ids: Sequence[str] | None, query_en: str | None
    ) -> tuple[list[SearchHit], dict[str, float]]:
        filters = scope_of(project_id, document_ids)
        t0 = time.perf_counter()
        vectors = await self.embedder.embed(_queries(query, query_en))  # one forward pass for both queries
        t1 = time.perf_counter()
        # The documents' labels are fetched beside the search (and remembered), so ranking finds them ready.
        lists, _ = await asyncio.gather(
            asyncio.gather(*(self.store.hybrid_search(v, filters) for v in vectors)), self.document_labels(filters)
        )
        t2 = time.perf_counter()
        hits = fuse_hit_lists(lists, limit=self.config.prefetch_k)
        return hits, {"embed": _ms(t1 - t0), "search": _ms(t2 - t1)}

    async def document_labels(self, filters: RetrievalFilters) -> dict[str, str]:
        """document id → label (file name and title) of the documents in ``filters``, for ``subjects.named_documents``.
        Only for a scope of several named documents (a chat's); remembered for ``LABEL_TTL_S``. A store that can't
        tell, or one that fails, gives no labels: retrieval then simply doesn't check whose passage it found."""
        ids = filters.document_ids
        if ids is None or len(ids) < 2:
            return {}
        now = time.monotonic()
        stale = [d for d in ids if d not in self._labels or now - self._labels[d][0] > LABEL_TTL_S]
        if stale:
            try:
                fetched = await self.store.document_labels(RetrievalFilters(filters.project_id, tuple(stale)))
            except Exception as e:  # a missing label only turns the subject check off
                log.warning("document labels unavailable (%s): %s", type(e).__name__, e)
                fetched = {}
            for d, label in fetched.items():
                self._labels[d] = (now, label)
        return {d: self._labels[d][1] for d in ids if d in self._labels and self._labels[d][1]}

    async def rerank(
        self, query: str, hits: Sequence[SearchHit], *, native_query: str | None = None
    ) -> list[RankedChunk]:
        """Every hit scored against ``query`` with the cross-encoder, best first.

        ``native_query``: the question as the user asked it, when ``query`` is its English version. Hindi passages
        are then scored against the user's own words and the others against the English query: the cross-encoder
        is calibrated best within one language (§9.2). Same number of passages scored, in two calls."""
        if not hits:
            return []
        texts = [h.chunk.embed_text for h in hits]
        native = [i for i, h in enumerate(hits) if h.chunk.language == "hi"]
        if not native_query or not native or same_query(native_query, query):
            return rank_by_scores(hits, await self.reranker.score(query, texts))
        native_set = set(native)
        english = [i for i in range(len(hits)) if i not in native_set]
        scores = [0.0] * len(hits)
        for idx, q in ((english, query), (native, native_query)):
            if idx:
                for i, score in zip(idx, await self.reranker.score(q, [texts[i] for i in idx]), strict=True):
                    scores[i] = score
        return rank_by_scores(hits, scores)

    async def retrieve(
        self,
        query: str,
        *,
        project_id: str,
        document_ids: Sequence[str] | None = None,
        query_en: str | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        hits, timings = await self._search(query, project_id=project_id, document_ids=document_ids, query_en=query_en)
        scope = scope_of(project_id, document_ids)
        return await self.rank(query, hits, query_en=query_en, timings=timings, started=t0, scope=scope)

    async def search_timed(
        self,
        query: str,
        *,
        project_id: str,
        document_ids: Sequence[str] | None = None,
        query_en: str | None = None,
    ) -> tuple[list[SearchHit], dict[str, float]]:
        """``search`` with its embed and search timings (the first half of ``retrieve``)."""
        return await self._search(query, project_id=project_id, document_ids=document_ids, query_en=query_en)

    async def rank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        query_en: str | None = None,
        timings: dict[str, float] | None = None,
        started: float | None = None,
        scope: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Rerank search hits and build the result (the second half of ``retrieve``): the reranker scores
        ``query_en`` when given. ``started`` (a ``perf_counter`` value) is when the search began, for the total.
        ``scope``: where the hits were searched; with several documents in it, a question that names one company's
        documents is answered from those (``prefer_named_documents``). Without it passages are taken as they come."""
        t1 = time.perf_counter()
        rerank_query = (query_en or "").strip() or query
        # a year-end date is searched and reranked with its fiscal year: "… on 31 March 2024? (FY24)"
        ranked = await self.rerank(
            with_fiscal_years(rerank_query),
            hits,
            native_query=with_fiscal_years(query) if rerank_query != query else None,
        )
        named = named_documents((query, query_en), await self.document_labels(scope)) if scope is not None else None
        ranked, missing_subjects = prefer_named_documents(ranked, named)
        ranked = prefer_stated_periods(ranked, (query, query_en), min_score=self.config.min_rerank_score)
        t2 = time.perf_counter()
        return RetrievalResult(
            query=query,
            rerank_query=rerank_query,
            chunks=ranked[: self.config.rerank_top_n],
            candidate_count=len(hits),
            confidence=confidence_of(
                ranked, self.config.min_rerank_score, queries=(query, query_en), missing_subjects=missing_subjects
            ),
            timings_ms={
                **(timings or {}),
                "rerank": _ms(t2 - t1),
                "total": _ms(t2 - (t1 if started is None else started)),
            },
        )


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def scope_of(project_id: str, document_ids: Sequence[str] | None) -> RetrievalFilters:
    return RetrievalFilters(project_id, tuple(document_ids) if document_ids is not None else None)


# ------------------------------------------------------------------ speculative retrieval

SpeculationOutcome = Literal[
    "used",  # the route asked for the same query: the speculative result is the turn's retrieval
    "reused_search",  # same query plus an English one: its search hits were kept, the English query added
    "discarded",  # the route asked for another query, or for no retrieval
    "none",  # no speculation (no documents, retrieval off, nothing to search)
]


def same_query(a: str | None, b: str | None) -> bool:
    """Equal up to case, spacing and trailing punctuation."""

    def norm(q: str | None) -> str:
        return " ".join((q or "").casefold().split()).rstrip("?.!।").strip()

    return norm(a) == norm(b)


class SpeculativeRetrieval:
    """Retrieval of the raw utterance, started while the router decides (§3.3 d, §9.5), so a document turn whose
    route keeps the query doesn't wait for retrieval after the router.

    ``rerank=False`` only searches (cheap: embedding + vector search) and leaves the reranker, which shares the GPU
    with the router model, for the final query. ``result_for`` hands the work over or discards it (see
    ``SpeculationOutcome``); ``discard`` cancels whatever is still running. Errors surface only when the result is
    used (a discarded speculation's failure is ignored).
    """

    def __init__(
        self,
        service: RetrievalService,
        query: str,
        *,
        project_id: str,
        document_ids: Sequence[str] | None,
        rerank: bool = True,
    ) -> None:
        self.service = service
        self.query = query
        self.project_id = project_id
        self.document_ids = document_ids
        self.started = time.perf_counter()
        self._search: asyncio.Task[tuple[list[SearchHit], dict[str, float]]] = asyncio.ensure_future(
            service.search_timed(query, project_id=project_id, document_ids=document_ids)
        )
        self._ranked: asyncio.Task[RetrievalResult] | None = asyncio.ensure_future(self._rank()) if rerank else None
        for task in (self._search, self._ranked):
            if task is not None:
                task.add_done_callback(_ignore_failure)

    @property
    def scope(self) -> RetrievalFilters:
        return scope_of(self.project_id, self.document_ids)

    async def _rank(self) -> RetrievalResult:
        hits, timings = await asyncio.shield(self._search)
        return await self.service.rank(self.query, hits, timings=timings, started=self.started, scope=self.scope)

    @property
    def reranks(self) -> bool:
        return self._ranked is not None

    async def wait_ranked(self) -> RetrievalResult | None:
        """The reranked result once it is ready (None without reranking or when it failed)."""
        if self._ranked is None:
            return None
        await asyncio.wait([self._ranked])
        return self.peek()

    def peek(self) -> RetrievalResult | None:
        """The reranked result if it is ready, without waiting."""
        task = self._ranked
        if task is None or not task.done() or task.cancelled() or task.exception() is not None:
            return None
        return task.result()

    async def result_for(self, query: str, query_en: str | None) -> tuple[RetrievalResult, SpeculationOutcome]:
        """The turn's retrieval for the route's ``query`` (and English query)."""
        en = query_en if query_en and not same_query(query_en, query) else None
        if not same_query(query, self.query):
            self.discard()
            result = await self.service.retrieve(
                query, project_id=self.project_id, document_ids=self.document_ids, query_en=en
            )
            return result, "discarded"
        if en is None:
            if self._ranked is not None:
                return await self._ranked, "used"
            hits, timings = await self._search
            ranked = await self.service.rank(query, hits, timings=timings, started=self.started, scope=self.scope)
            return ranked, "used"
        if self._ranked is not None:
            self._ranked.cancel()  # scored the wrong query; the reranker should score the English one
        hits, timings = await self._search
        t0 = time.perf_counter()
        en_hits = await self.service.search(en, project_id=self.project_id, document_ids=self.document_ids)
        fused = fuse_hit_lists([hits, en_hits], limit=self.service.config.prefetch_k)
        timings = {**timings, "search_en": _ms(time.perf_counter() - t0)}
        result = await self.service.rank(
            query, fused, query_en=en, timings=timings, started=self.started, scope=self.scope
        )
        return result, "reused_search"

    def discard(self) -> None:
        for task in (self._ranked, self._search):
            if task is not None and not task.done():
                task.cancel()


def _ignore_failure(task: asyncio.Task[Any]) -> None:
    """Mark a speculative task's failure as seen (its error matters only if the result is used, and then it is raised
    again by awaiting the task)."""
    if not task.cancelled():
        task.exception()


def _queries(query: str, query_en: str | None) -> list[str]:
    """The texts searched: the query and its English form, each with the fiscal year of a year-end date it names."""
    q = query.strip()
    if not q:
        raise ValueError("empty query")
    en = (query_en or "").strip()
    texts = [q, en] if en and en.casefold() != q.casefold() else [q]
    return [with_fiscal_years(t) for t in texts]
