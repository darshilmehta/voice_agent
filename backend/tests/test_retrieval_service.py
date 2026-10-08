"""Retrieval service with fake providers: fusion across queries, reranking, confidence."""

from __future__ import annotations

import asyncio

import pytest

from app.providers.registry import build_container
from app.providers.retrieval import RetrievalFilters
from app.services.retrieval import (
    RankedChunk,
    RetrievalService,
    confidence_of,
    fuse_hit_lists,
    rank_by_scores,
)
from app.settings import RetrievalSection

from .fakes import FakeEmbedder, FakeReranker, FakeStore, hit, make_chunk, vector_for

CONFIG = RetrievalSection(prefetch_k=20, fusion="rrf", rerank_top_n=2, min_rerank_score=0.3, context_token_budget=3000)


def chunks(n: int):
    return [make_chunk(i, text=f"passage {i}") for i in range(n)]


# ------------------------------------------------------------------ pure helpers


def test_single_list_passes_through_with_limit():
    c = chunks(3)
    hits = [hit(x, score=1 / (i + 1)) for i, x in enumerate(c)]
    assert fuse_hit_lists([hits], limit=2) == hits[:2]


def test_rrf_merges_lists_and_keeps_best_dense_score():
    a, b, c = chunks(3)
    first = [hit(a, dense=0.5), hit(b, dense=0.4)]
    second = [hit(b, dense=0.9), hit(c, dense=0.7), hit(a, dense=None)]
    fused = fuse_hit_lists([first, second], limit=10, k=60)
    # b: 1/62 + 1/61 > a: 1/61 + 1/63 > c: 1/62
    assert [h.chunk.chunk_id for h in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert [h.dense_score for h in fused] == [0.9, 0.5, 0.7]
    assert len(fuse_hit_lists([first, second], limit=2)) == 2


def test_rank_by_scores_orders_and_keeps_search_rank():
    c = chunks(3)
    ranked = rank_by_scores([hit(x) for x in c], [0.2, 0.9, 0.2])
    assert [(r.chunk.chunk_index, r.search_rank) for r in ranked] == [(1, 2), (0, 1), (2, 3)]  # ties: search order
    with pytest.raises(ValueError):
        rank_by_scores([hit(c[0])], [0.1, 0.2])


def test_confidence_signals():
    a, b = chunks(2)
    ranked = [RankedChunk(a, 0.92, 0.03, 0.71, 2), RankedChunk(b, 0.40, 0.03, 0.66, 1)]
    conf = confidence_of(ranked, 0.3)
    assert conf.top_score == 0.92 and conf.gap == pytest.approx(0.52)
    assert conf.dense_similarity == 0.71 and conf.above_threshold is True
    single = confidence_of(ranked[1:], 0.5)
    assert single.gap == 0.40 and single.above_threshold is False
    assert confidence_of([], 0.3) is None


# ------------------------------------------------------------------ service


def service(store: FakeStore, scorer=lambda q, p: 0.0) -> tuple[RetrievalService, FakeEmbedder, FakeReranker]:
    embedder, reranker = FakeEmbedder(), FakeReranker(scorer)
    return RetrievalService(embedder, reranker, store, CONFIG), embedder, reranker


def test_retrieve_reranks_with_the_english_query():
    c = chunks(4)
    lists = [[hit(c[0]), hit(c[1]), hit(c[2])], [hit(c[3]), hit(c[1])]]
    store = FakeStore(results=lists)
    svc, embedder, reranker = service(store, scorer=lambda q, p: 0.95 if "passage 3" in p else 0.1)
    hi = "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?"
    en = "What was the EBITDA margin in FY24?"
    res = asyncio.run(svc.retrieve(hi, project_id="proj1", document_ids=["doc1"], query_en=en))

    assert embedder.calls == [[hi, en]]  # one forward pass for both queries
    assert [s[0] for s in store.searches] == [vector_for(hi), vector_for(en)]
    assert all(s[1] == RetrievalFilters("proj1", ("doc1",)) for s in store.searches)
    assert reranker.calls[0][0] == en  # the reranker scores the English query (§9.2)
    assert reranker.calls[0][1] == [h.chunk.embed_text for h in fuse_hit_lists(lists, limit=20)]
    assert res.rerank_query == en and res.query == hi
    assert res.candidate_count == 4
    assert [r.chunk.chunk_index for r in res.chunks][:1] == [3] and len(res.chunks) == 2  # rerank_top_n
    assert res.confidence.top_score == 0.95 and res.confidence.gap == pytest.approx(0.85)
    assert set(res.timings_ms) == {"embed", "search", "rerank", "total"}


def test_retrieve_without_english_query_searches_once_and_reranks_the_query():
    c = chunks(2)
    store = FakeStore(results=[[hit(c[0]), hit(c[1])]])
    svc, embedder, reranker = service(store, scorer=lambda q, p: 0.5)
    res = asyncio.run(svc.retrieve("What was the EBITDA margin?", project_id="proj1"))
    assert embedder.calls == [["What was the EBITDA margin?"]]
    assert store.searches[0][1] == RetrievalFilters("proj1")
    assert reranker.calls[0][0] == "What was the EBITDA margin?"
    assert res.confidence.above_threshold is True


def test_identical_english_query_is_not_searched_twice():
    svc, embedder, _ = service(FakeStore(results=[[]]))
    asyncio.run(svc.retrieve("EBITDA margin FY24", project_id="p", query_en="ebitda margin fy24 "))
    assert embedder.calls == [["EBITDA margin FY24"]]


def test_nothing_found_means_no_confidence_and_no_rerank():
    svc, _, reranker = service(FakeStore(results=[[]]))
    res = asyncio.run(svc.retrieve("anything", project_id="p"))
    assert res.chunks == [] and res.confidence is None and res.candidate_count == 0
    assert reranker.calls == []


def test_empty_query_is_rejected():
    svc, _, _ = service(FakeStore())
    with pytest.raises(ValueError, match="empty query"):
        asyncio.run(svc.retrieve("  ", project_id="p"))


def test_service_builds_from_the_container(load_local):
    container = build_container(load_local())
    svc = RetrievalService.from_container(container)
    assert svc.config == container.settings.retrieval
    assert svc.embedder is container["embeddings"] and svc.store is container["vector_store"]
