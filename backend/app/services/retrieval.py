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
            timings_ms={**timings, "rerank": _ms(t2 - t1), "total": _ms(t2 - t0)},
        )


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _queries(query: str, query_en: str | None) -> list[str]:
    q = query.strip()
    if not q:
        raise ValueError("empty query")
    en = (query_en or "").strip()
    return [q, en] if en and en.casefold() != q.casefold() else [q]
