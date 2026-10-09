"""QdrantStore against a recording fake client: filters, hybrid query, points, collection setup, deletes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from qdrant_client import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from app.providers.base import ProviderContext
from app.providers.ingestion import point_id
from app.providers.retrieval import (
    DENSE,
    SPARSE,
    DenseSparse,
    IndexedChunk,
    QdrantStore,
    RetrievalFilters,
    SparseVector,
    VectorStoreError,
    build_filter,
    build_hybrid_query,
    cosine,
    rerank_batch_size,
    to_hit,
    to_point,
)

from .fakes import make_chunk

QUERY = DenseSparse(dense=[1.0, 0.0, 0.0], sparse=SparseVector([7, 42], [0.3, 0.1]))


def not_found() -> UnexpectedResponse:
    return UnexpectedResponse(404, "Not Found", b'{"status":{"error":"Not found: Collection"}}', httpx.Headers())


class FakeClient:
    """Records every call; answers like Qdrant would."""

    def __init__(self, *, exists: bool = False, dense_size: int = 3, points: list[Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.exists = exists
        self.dense_size = dense_size
        self.points = points or []
        self.raise_on_query: Exception | None = None

    def _record(self, op: str, /, **kw: Any) -> None:
        self.calls.append((op, kw))

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def last(self, name: str) -> dict[str, Any]:
        return next(kw for n, kw in reversed(self.calls) if n == name)

    async def collection_exists(self, name: str) -> bool:
        self._record("collection_exists", name=name)
        return self.exists

    async def get_collection(self, name: str) -> Any:
        self._record("get_collection", name=name)
        params = SimpleNamespace(
            vectors={DENSE: qm.VectorParams(size=self.dense_size, distance=qm.Distance.COSINE)},
            sparse_vectors={SPARSE: qm.SparseVectorParams()},
        )
        return SimpleNamespace(config=SimpleNamespace(params=params))

    async def create_collection(self, name: str, **kw: Any) -> None:
        self._record("create_collection", name=name, **kw)
        self.exists = True

    async def create_payload_index(self, name: str, **kw: Any) -> None:
        self._record("create_payload_index", name=name, **kw)

    async def upsert(self, name: str, **kw: Any) -> None:
        self._record("upsert", name=name, **kw)

    async def query_points(self, name: str, **kw: Any) -> Any:
        self._record("query_points", name=name, **kw)
        if self.raise_on_query:
            raise self.raise_on_query
        return SimpleNamespace(points=self.points)

    async def delete(self, name: str, **kw: Any) -> None:
        self._record("delete", name=name, **kw)

    async def count(self, name: str, **kw: Any) -> Any:
        self._record("count", name=name, **kw)
        return SimpleNamespace(count=7)

    async def close(self) -> None:
        self._record("close")


@pytest.fixture
def store(load_local) -> QdrantStore:
    settings = load_local(VECTOR_STORE__COLLECTION="test_chunks", RETRIEVAL__PREFETCH_K="20")
    return QdrantStore(settings.vector_store, ProviderContext(settings=settings, http=httpx.AsyncClient()))


def use(store: QdrantStore, client: FakeClient) -> FakeClient:
    store._client = client  # type: ignore[assignment]
    return client


def indexed(i: int, **over: Any) -> IndexedChunk:
    return IndexedChunk(make_chunk(i, **over), DenseSparse([0.0, 1.0, 0.0], SparseVector([1, 5], [0.4, 0.2])))


# ------------------------------------------------------------------ pure builders


def test_filter_scopes_to_the_project():
    flt = build_filter(RetrievalFilters("proj1"))
    assert flt.must == [qm.FieldCondition(key="project_id", match=qm.MatchValue(value="proj1"))]


def test_filter_narrows_to_documents():
    flt = build_filter(RetrievalFilters("proj1", ("d1", "d2")))
    assert flt.must[1] == qm.FieldCondition(key="document_id", match=qm.MatchAny(any=["d1", "d2"]))


def test_filters_require_a_project_and_freeze_document_ids():
    with pytest.raises(ValueError, match="project_id"):
        RetrievalFilters("")
    assert RetrievalFilters("p", ["a"]).document_ids == ("a",)  # type: ignore[arg-type]


@pytest.mark.parametrize(("fusion", "expected"), [("rrf", qm.Fusion.RRF), ("dbsf", qm.Fusion.DBSF)])
def test_hybrid_query_prefetches_both_vectors_with_the_filter(fusion, expected):
    flt = build_filter(RetrievalFilters("proj1"))
    q = build_hybrid_query(QUERY, flt, prefetch_k=20, fusion=fusion, limit=15)
    dense, sparse = q["prefetch"]
    assert (dense.using, dense.limit, dense.filter, dense.query) == (DENSE, 20, flt, [1.0, 0.0, 0.0])
    assert (sparse.using, sparse.limit, sparse.filter) == (SPARSE, 20, flt)
    assert sparse.query == qm.SparseVector(indices=[7, 42], values=[0.3, 0.1])
    assert q["query"] == qm.FusionQuery(fusion=expected)
    assert q["query_filter"] == flt and q["limit"] == 15
    assert q["with_payload"] is True and q["with_vectors"] == [DENSE]


def test_points_round_trip_the_chunk():
    item = indexed(3, heading_path=["A", "B"], overlap_text="tail of the previous chunk")
    p = to_point(item)
    assert p.id == item.chunk.point_id
    assert p.vector[DENSE] == [0.0, 1.0, 0.0]
    assert p.vector[SPARSE] == qm.SparseVector(indices=[1, 5], values=[0.4, 0.2])
    assert p.payload["project_id"] == "proj1" and p.payload["document_id"] == "doc1"
    point = SimpleNamespace(payload=p.payload, score=0.03, vector={DENSE: [1.0, 1.0, 0.0]})
    h = to_hit(point, [1.0, 0.0, 0.0])
    assert h.chunk == item.chunk and h.score == 0.03
    assert h.dense_score == pytest.approx(cosine([1, 0, 0], [1, 1, 0])) == pytest.approx(0.7071, abs=1e-4)
    assert to_hit(SimpleNamespace(payload=p.payload, score=0.1, vector=None), [1.0]).dense_score is None


# ------------------------------------------------------------------ store


def test_first_upsert_creates_the_collection_once(store):
    client = use(store, FakeClient(exists=False))
    asyncio.run(store.upsert([indexed(0), indexed(1)]))
    asyncio.run(store.upsert([indexed(2)]))
    create = client.last("create_collection")
    assert create["name"] == "test_chunks"
    assert create["vectors_config"] == {DENSE: qm.VectorParams(size=3, distance=qm.Distance.COSINE)}
    assert create["sparse_vectors_config"] == {SPARSE: qm.SparseVectorParams()}
    indexes = [kw["field_name"] for n, kw in client.calls if n == "create_payload_index"]
    assert indexes == ["project_id", "document_id"]
    assert client.names().count("create_collection") == 1 and client.names().count("upsert") == 2
    assert [p.id for p in client.last("upsert")["points"]] == [make_chunk(2).point_id]


def test_existing_collection_with_another_dimension_is_refused(store):
    use(store, FakeClient(exists=True, dense_size=768))
    with pytest.raises(VectorStoreError, match="768-d"):
        asyncio.run(store.upsert([indexed(0)]))


def test_existing_collection_with_the_right_schema_is_reused(store):
    client = use(store, FakeClient(exists=True, dense_size=3))
    asyncio.run(store.upsert([indexed(0)]))
    assert "create_collection" not in client.names()


def test_hybrid_search_queries_with_prefetch_k_and_returns_hits(store):
    chunk = make_chunk(0)
    point = SimpleNamespace(payload=chunk.model_dump(mode="json"), score=0.5, vector={DENSE: [1.0, 0.0, 0.0]})
    client = use(store, FakeClient(points=[point]))
    hits = asyncio.run(store.hybrid_search(QUERY, RetrievalFilters("proj1", ("doc1",))))
    q = client.last("query_points")
    k = store.retrieval.prefetch_k  # also the number of candidates the reranker scores
    assert q["name"] == "test_chunks" and q["limit"] == k
    assert [p.limit for p in q["prefetch"]] == [k, k]
    assert q["query_filter"] == build_filter(RetrievalFilters("proj1", ("doc1",)))
    assert [(h.chunk, h.score, h.dense_score) for h in hits] == [(chunk, 0.5, pytest.approx(1.0))]


def test_search_before_any_ingest_finds_nothing(store):
    client = use(store, FakeClient())
    client.raise_on_query = not_found()
    assert asyncio.run(store.hybrid_search(QUERY, RetrievalFilters("proj1"))) == []


def test_empty_document_scope_finds_nothing_without_a_query(store):
    client = use(store, FakeClient())
    assert asyncio.run(store.hybrid_search(QUERY, RetrievalFilters("proj1", ()))) == []
    assert asyncio.run(store.count(RetrievalFilters("proj1", ()))) == 0
    assert client.calls == []


def test_delete_document_keeps_the_given_chunks(store):
    client = use(store, FakeClient())
    asyncio.run(store.delete_document("doc1", keep=["doc1:v2:0000", "doc1:v2:0001"]))
    flt = client.last("delete")["points_selector"].filter
    assert flt.must == [qm.FieldCondition(key="document_id", match=qm.MatchValue(value="doc1"))]
    assert flt.must_not[0].has_id == [point_id("doc1:v2:0000"), point_id("doc1:v2:0001")]


def test_delete_document_and_project(store):
    client = use(store, FakeClient())
    asyncio.run(store.delete_document("doc1"))
    assert client.last("delete")["points_selector"].filter.must_not is None
    asyncio.run(store.delete_project("proj1"))
    flt = client.last("delete")["points_selector"].filter
    assert flt.must == [qm.FieldCondition(key="project_id", match=qm.MatchValue(value="proj1"))]


def test_count_uses_the_filter(store):
    client = use(store, FakeClient())
    assert asyncio.run(store.count(RetrievalFilters("proj1", ("doc1",)))) == 7
    assert client.last("count")["count_filter"] == build_filter(RetrievalFilters("proj1", ("doc1",)))
    assert client.last("count")["exact"] is True


def test_client_is_created_lazily_without_network(store):
    assert store._client is None
    client = store.client()
    assert store.client() is client  # constructed once, no compatibility request
    asyncio.run(store.close())
    assert store._client is None


def test_the_reranker_uses_small_length_sorted_batches_on_mps():
    """Less padding: ~45% faster on the M4 (see RERANK_BATCH_SIZE_MPS); other devices keep large batches."""
    assert rerank_batch_size("mps") == 2
    assert rerank_batch_size("cpu") == rerank_batch_size("cuda") == 32
