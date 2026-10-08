"""Embeddings, reranker and vector store providers (docs/DESIGN.md §3.2, §6).

Models load lazily from their local snapshot on first use and run off the event loop (``asyncio.to_thread``);
torch, FlagEmbedding and sentence-transformers come from the optional ``ml`` dependency group.
``qdrant-client`` is a runtime dependency but is also imported lazily to keep startup light.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from .base import HealthStatus, Provider, ProviderContext, ProviderHealth
from .ingestion import Chunk, point_id
from .models import LocalModelProvider, free_torch_memory, quiet_ml_env, require_modules
from .registry import register

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client import models as qm

    from ..settings import EmbeddingsSection, RerankerSection, RetrievalSection, VectorStoreSection

DENSE = "dense"
SPARSE = "sparse"

# Longest text the embedder reads. Chunks are ≤ ~target_tokens, but whole tables can be longer; BGE-M3 handles 8192.
EMBED_MAX_TOKENS = 2048
# Query + passage tokens the cross-encoder reads (as in smoke test 04). Chunks of target_tokens (500) fit almost
# whole; long table chunks are cut, which bounds their cost. Reranking is compute-bound: on an M4 (MPS, fp16),
# 20 candidates take ~0.2 s at ~30 tokens each but ~2.1 s at ~425 tokens each, so candidate count and chunk size
# drive voice latency (tune prefetch_k / chunking in phase 2).
RERANK_MAX_TOKENS = 512
RERANK_BATCH_SIZE = 32
UPSERT_BATCH = 256


class VectorStoreError(RuntimeError):
    """The vector store is unusable as configured (e.g. the collection was built for another embedding size)."""


# ------------------------------------------------------------------ data types


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: list[int]  # token ids
    values: list[float]  # learned lexical weights


@dataclass(frozen=True, slots=True)
class DenseSparse:
    """BGE-M3's two outputs for one text."""

    dense: list[float]
    sparse: SparseVector


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    chunk: Chunk
    vectors: DenseSparse


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Scope of a search: always one project (documents never leak across projects, §3.9), optionally narrowed to
    some of its documents. ``document_ids=None`` means every document; an empty sequence means none."""

    project_id: str
    document_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.project_id:
            raise ValueError("project_id is required: retrieval is always scoped to one project")
        if self.document_ids is not None:
            object.__setattr__(self, "document_ids", tuple(self.document_ids))


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float  # fused score (RRF/DBSF): comparable within one result list only
    dense_score: float | None  # cosine(query, chunk): a calibrated-ish similarity for confidence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ------------------------------------------------------------------ embeddings


class Embedder(Provider):
    capability = "embeddings"
    dim: ClassVar[int] = 0  # dense vector size

    async def embed(self, texts: Sequence[str]) -> list[DenseSparse]:
        """Dense + sparse vectors for each text, in order."""
        raise NotImplementedError(f"{type(self).__name__}.embed")


class _LazyModel(LocalModelProvider):
    """A model loaded on first use, used by one thread at a time, and freed on close."""

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._lock = threading.Lock()
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self, snapshot: Path) -> Any:  # pragma: no cover - needs the ml group and model files
        raise NotImplementedError

    def _model_locked(self) -> Any:
        """The model, loading it from the local snapshot if needed. Call with ``self._lock`` held."""
        if self._model is None:
            require_modules(f"{self.capability} provider {self.name!r}", self.required_modules)
            snapshot = self.snapshot_path()
            quiet_ml_env()
            self._model = self._load(snapshot)
        return self._model

    def unload(self) -> None:
        with self._lock:
            model, self._model = self._model, None
        if model is not None:
            del model
            free_torch_memory()

    async def close(self) -> None:
        await asyncio.to_thread(self.unload)


def _use_fp16(device: str, fp16: bool) -> bool:
    return fp16 and device != "cpu"  # half precision on CPU is slow or unsupported


@register
class BgeM3Embedder(Embedder, _LazyModel):
    name = "bge_m3"
    dim = 1024
    required_modules = ("FlagEmbedding", "torch")

    @property
    def cfg(self) -> EmbeddingsSection:
        return self.config  # type: ignore[return-value]

    async def embed(self, texts: Sequence[str]) -> list[DenseSparse]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, list(texts))

    def _load(self, snapshot: Path) -> Any:
        from FlagEmbedding import BGEM3FlagModel

        cfg = self.cfg
        return BGEM3FlagModel(str(snapshot), use_fp16=_use_fp16(cfg.device, cfg.fp16), devices=cfg.device)

    def _embed_sync(self, texts: list[str]) -> list[DenseSparse]:
        with self._lock:
            out = self._model_locked().encode(
                texts,
                batch_size=self.cfg.batch_size,
                max_length=EMBED_MAX_TOKENS,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        dense = out["dense_vecs"].tolist()
        return [
            DenseSparse(dense=dense[i], sparse=_sparse(weights)) for i, weights in enumerate(out["lexical_weights"])
        ]


def _sparse(weights: dict[Any, Any]) -> SparseVector:
    pairs = sorted((int(k), float(v)) for k, v in weights.items())
    return SparseVector(indices=[k for k, _ in pairs], values=[v for _, v in pairs])


# ------------------------------------------------------------------ reranker


class Reranker(Provider):
    capability = "reranker"

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to the query, in order (0..1 for cross-encoders with a sigmoid head)."""
        raise NotImplementedError(f"{type(self).__name__}.score")


@register
class BgeReranker(Reranker, _LazyModel):
    name = "bge_reranker"
    required_modules = ("sentence_transformers", "torch")

    @property
    def cfg(self) -> RerankerSection:
        return self.config  # type: ignore[return-value]

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._score_sync, query, list(passages))

    def _load(self, snapshot: Path) -> Any:
        import torch
        from sentence_transformers import CrossEncoder

        cfg = self.cfg
        dtype = torch.float16 if _use_fp16(cfg.device, cfg.fp16) else torch.float32
        return CrossEncoder(
            str(snapshot),
            device=cfg.device,
            max_length=RERANK_MAX_TOKENS,
            local_files_only=True,
            model_kwargs={"dtype": dtype},
        )

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            scores = self._model_locked().predict(
                [(query, p) for p in passages], batch_size=RERANK_BATCH_SIZE, show_progress_bar=False
            )
        return [float(s) for s in scores]


# ------------------------------------------------------------------ vector store


class VectorStore(Provider):
    capability = "vector_store"

    async def upsert(self, chunks: Sequence[IndexedChunk]) -> None:
        """Insert or overwrite chunks (ids are deterministic per chunk id)."""
        raise NotImplementedError(f"{type(self).__name__}.upsert")

    async def hybrid_search(
        self, query: DenseSparse, filters: RetrievalFilters, *, limit: int | None = None
    ) -> list[SearchHit]:
        """Dense + sparse candidates fused into one ranked list, best first."""
        raise NotImplementedError(f"{type(self).__name__}.hybrid_search")

    async def delete_document(self, document_id: str, *, keep: Iterable[str] = ()) -> None:
        """Delete a document's chunks, except the chunk ids in ``keep`` (used to replace a document's index)."""
        raise NotImplementedError(f"{type(self).__name__}.delete_document")

    async def delete_project(self, project_id: str) -> None:
        raise NotImplementedError(f"{type(self).__name__}.delete_project")

    async def count(self, filters: RetrievalFilters) -> int:
        raise NotImplementedError(f"{type(self).__name__}.count")


def build_filter(filters: RetrievalFilters) -> qm.Filter:
    """Qdrant filter for a project, optionally narrowed to documents."""
    from qdrant_client import models as qm

    must: list[Any] = [qm.FieldCondition(key="project_id", match=qm.MatchValue(value=filters.project_id))]
    if filters.document_ids:
        must.append(qm.FieldCondition(key="document_id", match=qm.MatchAny(any=list(filters.document_ids))))
    return qm.Filter(must=must)


def build_hybrid_query(
    query: DenseSparse, flt: qm.Filter, *, prefetch_k: int, fusion: str, limit: int
) -> dict[str, Any]:
    """Arguments for ``query_points``: top ``prefetch_k`` dense and sparse candidates (both filtered), fused."""
    from qdrant_client import models as qm

    sparse = qm.SparseVector(indices=list(query.sparse.indices), values=list(query.sparse.values))
    return {
        "prefetch": [
            qm.Prefetch(query=list(query.dense), using=DENSE, limit=prefetch_k, filter=flt),
            qm.Prefetch(query=sparse, using=SPARSE, limit=prefetch_k, filter=flt),
        ],
        "query": qm.FusionQuery(fusion=qm.Fusion.DBSF if fusion == "dbsf" else qm.Fusion.RRF),
        "query_filter": flt,
        "limit": limit,
        "with_payload": True,
        "with_vectors": [DENSE],
    }


def to_point(item: IndexedChunk) -> qm.PointStruct:
    from qdrant_client import models as qm

    v = item.vectors
    return qm.PointStruct(
        id=item.chunk.point_id,
        vector={
            DENSE: list(v.dense),
            SPARSE: qm.SparseVector(indices=list(v.sparse.indices), values=list(v.sparse.values)),
        },
        payload=item.chunk.model_dump(mode="json"),
    )


def to_hit(point: Any, query_dense: Sequence[float]) -> SearchHit:
    vectors = point.vector if isinstance(point.vector, dict) else {}
    dense = vectors.get(DENSE)
    return SearchHit(
        chunk=Chunk.model_validate(point.payload),
        score=float(point.score),
        dense_score=cosine(query_dense, dense) if dense else None,
    )


def _not_found(e: Exception) -> bool:
    return getattr(e, "status_code", None) == 404


@register
class QdrantStore(VectorStore):
    """Named vectors ``dense`` (cosine) + ``sparse`` in one collection; keyword indexes on project and document."""

    name = "qdrant"

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._client: AsyncQdrantClient | None = None
        self._ready = False
        self._ensure_lock = asyncio.Lock()

    @property
    def cfg(self) -> VectorStoreSection:
        return self.config  # type: ignore[return-value]

    @property
    def retrieval(self) -> RetrievalSection:
        return self.ctx.settings.retrieval

    @property
    def collection(self) -> str:
        return self.cfg.collection

    async def health(self) -> ProviderHealth:
        base = self.cfg.url.rstrip("/")
        key = self.cfg.api_key
        headers = {"api-key": key} if key else {}
        r, ms, err = await self._probe(f"{base}/readyz", headers=headers)
        if r is None:
            return self._health(HealthStatus.DOWN, f"Qdrant unreachable at {base} ({err})", ms)
        if r.status_code != 200:
            return self._health(HealthStatus.DOWN, f"Qdrant not ready: HTTP {r.status_code}", ms)
        c, _, _ = await self._probe(f"{base}/collections/{self.collection}/exists", headers=headers)
        exists = c is not None and c.status_code == 200 and c.json().get("result", {}).get("exists", False)
        state = "exists" if exists else "will be created on first ingest"
        return self._health(HealthStatus.OK, f"ready; collection {self.collection!r} {state}", ms)

    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            cfg = self.cfg
            self._client = AsyncQdrantClient(
                url=cfg.url,
                api_key=cfg.api_key,
                prefer_grpc=cfg.prefer_grpc,
                timeout=cfg.timeout_s,
                check_compatibility=False,  # no request at construction; health reports reachability
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._ready = False

    async def ensure_collection(self, dense_dim: int) -> None:
        """Create the collection and payload indexes if missing; refuse one built for another embedding size."""
        if self._ready:
            return
        from qdrant_client import models as qm
        from qdrant_client.http.exceptions import UnexpectedResponse

        async with self._ensure_lock:
            if self._ready:
                return
            client = self.client()
            if await client.collection_exists(self.collection):
                info = await client.get_collection(self.collection)
                _check_schema(self.collection, info, dense_dim)
            else:
                try:
                    await client.create_collection(
                        self.collection,
                        vectors_config={DENSE: qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
                        sparse_vectors_config={SPARSE: qm.SparseVectorParams()},
                    )
                except UnexpectedResponse as e:  # created concurrently by another worker
                    if e.status_code != 409:
                        raise
            for key in ("project_id", "document_id"):
                await client.create_payload_index(
                    self.collection, field_name=key, field_schema=qm.PayloadSchemaType.KEYWORD, wait=True
                )
            self._ready = True

    async def upsert(self, chunks: Sequence[IndexedChunk]) -> None:
        if not chunks:
            return
        await self.ensure_collection(len(chunks[0].vectors.dense))
        client = self.client()
        points = [to_point(c) for c in chunks]
        for start in range(0, len(points), UPSERT_BATCH):
            await client.upsert(self.collection, points=points[start : start + UPSERT_BATCH], wait=True)

    async def hybrid_search(
        self, query: DenseSparse, filters: RetrievalFilters, *, limit: int | None = None
    ) -> list[SearchHit]:
        if filters.document_ids is not None and not filters.document_ids:
            return []
        k = self.retrieval.prefetch_k
        kwargs = build_hybrid_query(
            query, build_filter(filters), prefetch_k=k, fusion=self.retrieval.fusion, limit=limit or k
        )
        try:
            res = await self.client().query_points(self.collection, **kwargs)
        except Exception as e:
            if _not_found(e):  # nothing ingested yet
                return []
            raise
        return [to_hit(p, query.dense) for p in res.points]

    async def delete_document(self, document_id: str, *, keep: Iterable[str] = ()) -> None:
        from qdrant_client import models as qm

        keep_ids = [point_id(c) for c in keep]
        flt = qm.Filter(
            must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))],
            must_not=[qm.HasIdCondition(has_id=keep_ids)] if keep_ids else None,
        )
        await self._delete(flt)

    async def delete_project(self, project_id: str) -> None:
        from qdrant_client import models as qm

        await self._delete(qm.Filter(must=[qm.FieldCondition(key="project_id", match=qm.MatchValue(value=project_id))]))

    async def _delete(self, flt: qm.Filter) -> None:
        from qdrant_client import models as qm

        try:
            await self.client().delete(self.collection, points_selector=qm.FilterSelector(filter=flt), wait=True)
        except Exception as e:
            if not _not_found(e):
                raise

    async def count(self, filters: RetrievalFilters) -> int:
        if filters.document_ids is not None and not filters.document_ids:
            return 0
        try:
            res = await self.client().count(self.collection, count_filter=build_filter(filters), exact=True)
        except Exception as e:
            if _not_found(e):
                return 0
            raise
        return res.count

    async def drop_collection(self) -> None:
        """Delete the whole collection (tests, re-index). Recreated on the next upsert."""
        await self.client().delete_collection(self.collection)
        self._ready = False


def _check_schema(name: str, info: Any, dense_dim: int) -> None:
    params = info.config.params
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    sparse = params.sparse_vectors or {}
    dense = vectors.get(DENSE)
    if dense is None or SPARSE not in sparse:
        raise VectorStoreError(f"collection {name!r} lacks the named vectors {DENSE!r} + {SPARSE!r}; re-index needed")
    if dense.size != dense_dim:
        raise VectorStoreError(
            f"collection {name!r} holds {dense.size}-d vectors but the embedder makes {dense_dim}-d; re-index needed"
        )
