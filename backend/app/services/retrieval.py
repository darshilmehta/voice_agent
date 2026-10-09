"""Retrieval (docs/DESIGN.md §3.2): hybrid search → rerank → top N with a confidence signal.

    query (+ English query_en from the router, §3.4) → BGE-M3 dense + sparse
      → Qdrant: top prefetch_k dense + top prefetch_k sparse, filtered to the project (and chat's documents), RRF
      → bge-reranker-v2-m3 scores (query_en when given: the reranker is calibrated best EN-EN, §9.2)
      → top rerank_top_n + confidence (top score, gap to the runner-up, dense similarity)

``search`` and ``rerank`` are public so the voice loop can start retrieval speculatively on the raw utterance and
rerank once the router's English query arrives (§3.3 d, §9.5). Deciding to answer or abstain is the caller's job:
``Confidence.above_threshold`` applies the configured ``min_rerank_score`` only; multi-signal thresholds are tuned
on the eval set in phase 2.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..providers.ingestion import Chunk
from ..providers.registry import Container
from ..providers.retrieval import Embedder, Reranker, RetrievalFilters, SearchHit, VectorStore
from ..settings import RetrievalSection

RRF_K = 60  # the usual reciprocal-rank-fusion constant (also Qdrant's)


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
    above_threshold: bool  # top_score ≥ retrieval.min_rerank_score


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


def confidence_of(ranked: Sequence[RankedChunk], min_score: float) -> Confidence | None:
    if not ranked:
        return None
    top = ranked[0]
    runner_up = ranked[1].rerank_score if len(ranked) > 1 else 0.0
    return Confidence(
        top_score=top.rerank_score,
        gap=top.rerank_score - runner_up,
        dense_similarity=top.dense_score,
        above_threshold=top.rerank_score >= min_score,
    )


# ------------------------------------------------------------------ service


class RetrievalService:
    def __init__(self, embedder: Embedder, reranker: Reranker, store: VectorStore, config: RetrievalSection) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.store = store
        self.config = config

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
        filters = RetrievalFilters(project_id, tuple(document_ids) if document_ids is not None else None)
        t0 = time.perf_counter()
        vectors = await self.embedder.embed(_queries(query, query_en))  # one forward pass for both queries
        t1 = time.perf_counter()
        lists = await asyncio.gather(*(self.store.hybrid_search(v, filters) for v in vectors))
        t2 = time.perf_counter()
        hits = fuse_hit_lists(lists, limit=self.config.prefetch_k)
        return hits, {"embed": _ms(t1 - t0), "search": _ms(t2 - t1)}

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[RankedChunk]:
        """Every hit scored against ``query`` with the cross-encoder, best first."""
        if not hits:
            return []
        scores = await self.reranker.score(query, [h.chunk.embed_text for h in hits])
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
        return await self.rank(query, hits, query_en=query_en, timings=timings, started=t0)

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
    ) -> RetrievalResult:
        """Rerank search hits and build the result (the second half of ``retrieve``): the reranker scores
        ``query_en`` when given. ``started`` (a ``perf_counter`` value) is when the search began, for the total."""
        t1 = time.perf_counter()
        rerank_query = (query_en or "").strip() or query
        ranked = await self.rerank(rerank_query, hits)
        t2 = time.perf_counter()
        return RetrievalResult(
            query=query,
            rerank_query=rerank_query,
            chunks=ranked[: self.config.rerank_top_n],
            candidate_count=len(hits),
            confidence=confidence_of(ranked, self.config.min_rerank_score),
            timings_ms={
                **(timings or {}),
                "rerank": _ms(t2 - t1),
                "total": _ms(t2 - (t1 if started is None else started)),
            },
        )


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


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

    async def _rank(self) -> RetrievalResult:
        hits, timings = await asyncio.shield(self._search)
        return await self.service.rank(self.query, hits, timings=timings, started=self.started)

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
            return await self.service.rank(query, hits, timings=timings, started=self.started), "used"
        if self._ranked is not None:
            self._ranked.cancel()  # scored the wrong query; the reranker should score the English one
        hits, timings = await self._search
        t0 = time.perf_counter()
        en_hits = await self.service.search(en, project_id=self.project_id, document_ids=self.document_ids)
        fused = fuse_hit_lists([hits, en_hits], limit=self.service.config.prefetch_k)
        timings = {**timings, "search_en": _ms(time.perf_counter() - t0)}
        result = await self.service.rank(query, fused, query_en=en, timings=timings, started=self.started)
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
    q = query.strip()
    if not q:
        raise ValueError("empty query")
    en = (query_en or "").strip()
    return [q, en] if en and en.casefold() != q.casefold() else [q]
