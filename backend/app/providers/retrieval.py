"""Embeddings, reranker and vector store providers (docs/DESIGN.md §3.2, §6).

Models load lazily from their local snapshot on first use (or at startup, ``preload``) and run off the event loop;
torch, FlagEmbedding and sentence-transformers come from the optional ``ml`` dependency group.
``qdrant-client`` is a runtime dependency but is also imported lazily to keep startup light.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from .base import HealthStatus, Provider, ProviderContext, ProviderHealth
from .ingestion import Chunk, point_id
from .models import LazyModelProvider
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
# whole; long table chunks are cut, which bounds their cost (none of the phase-2 eval corpus's: its longest chunk is
# ~430 tokens). Reranking is compute-bound and linear in the number of candidates (retrieval.prefetch_k).
RERANK_MAX_TOKENS = 512
# Pairs per forward pass. sentence-transformers sorts pairs by length, so small batches pad each pair only to its
# neighbours' length instead of the longest candidate's: on the M4 (MPS, fp16, phase-2 eval candidates of ~50-430
# tokens, p50) 8 candidates take ~340 ms in batches of 2 against ~615 ms in one batch of 32, 12 take ~490 against
# ~930 ms, 16 take ~640 against ~1,240 ms (batches of 1 and 4 are within ~40 ms of 2). Other devices keep large
# batches (unmeasured; CUDA prefers them).
RERANK_BATCH_SIZE = 32
RERANK_BATCH_SIZE_MPS = 2
UPSERT_BATCH = 256
# Chunks of a document read to find its label (``VectorStore.document_labels``): the longest wins, and only the chunks
# under the document's title lack the title in theirs.
LABEL_SAMPLE = 16
TEXT_SAMPLE = 64  # chunks read per document for the speech vocabulary's words


def rerank_batch_size(device: str) -> int:
    return RERANK_BATCH_SIZE_MPS if device == "mps" else RERANK_BATCH_SIZE


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


def _use_fp16(device: str, fp16: bool) -> bool:
    return fp16 and device != "cpu"  # half precision on CPU is slow or unsupported


@register
class BgeM3Embedder(Embedder, LazyModelProvider):
    name = "bge_m3"
    dim = 1024
    required_modules = ("FlagEmbedding", "torch")

    @property
    def cfg(self) -> EmbeddingsSection:
        return self.config  # type: ignore[return-value]

    async def embed(self, texts: Sequence[str]) -> list[DenseSparse]:
        if not texts:
            return []
        return await self._run(self._embed_sync, list(texts))

    def _load(self, snapshot: Path) -> Any:
        from FlagEmbedding import BGEM3FlagModel

        cfg = self.cfg
        return BGEM3FlagModel(str(snapshot), use_fp16=_use_fp16(cfg.device, cfg.fp16), devices=cfg.device)

    def _warm(self, model: Any) -> None:
        with self._torch_use():
            model.encode(["warm up"], batch_size=1, max_length=32, return_dense=True, return_sparse=True)

    def _embed_sync(self, texts: list[str]) -> list[DenseSparse]:
        """One batch at a time, each holding the model and the GPU only for that batch: a question's query embedding
        waits for at most one batch of a document being ingested, not for the whole document."""
        size = self.cfg.batch_size
        vectors: list[DenseSparse] = []
        for start in range(0, len(texts), size):
            with self._lock:
                model = self._model_locked()
                with self._torch_use():
                    out = model.encode(
                        texts[start : start + size],
                        batch_size=size,
                        max_length=EMBED_MAX_TOKENS,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    )
            dense = out["dense_vecs"].tolist()
            vectors += [DenseSparse(dense=dense[i], sparse=_sparse(w)) for i, w in enumerate(out["lexical_weights"])]
        return vectors


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
class BgeReranker(Reranker, LazyModelProvider):
    name = "bge_reranker"
    required_modules = ("sentence_transformers", "torch")

    @property
    def cfg(self) -> RerankerSection:
        return self.config  # type: ignore[return-value]

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return await self._run(self._score_sync, query, list(passages))

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

    def _warm(self, model: Any) -> None:
        with self._torch_use():
            model.predict([("warm up", "warm up")], show_progress_bar=False)

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            model = self._model_locked()
            with self._torch_use():
                scores = model.predict(
                    [(query, p) for p in passages],
                    batch_size=rerank_batch_size(self.cfg.device),
                    show_progress_bar=False,
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

    async def outdated(self, document_ids: Sequence[str], chunking_version: str) -> list[str]:
        """The documents among ``document_ids`` whose index must be rebuilt: no chunks indexed, or chunks built by
        another ``chunking_version`` (what is embedded changed). Order of ``document_ids`` kept."""
        raise NotImplementedError(f"{type(self).__name__}.outdated")

    async def document_labels(self, filters: RetrievalFilters) -> dict[str, str]:
        """document id → its label (``Chunk.document_label``: file name and title) for the documents in
        ``filters.document_ids``; documents without chunks, or labels, are left out. The label is the longest one among
        a sample of the document's chunks: chunks under the title omit it from theirs. A store that can't say returns
        nothing, and retrieval then doesn't check which document's passage it found (``services/subjects.py``)."""
        return {}

    async def document_texts(self, filters: RetrievalFilters, *, limit: int = 0) -> dict[str, list[str]]:
        """document id → the headings and text of a sample of its chunks (up to ``limit`` each, ``TEXT_SAMPLE`` by
        default), for the speech vocabulary's words of a Hindi document (§9.3). A store that can't say returns
        nothing."""
        return {}


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

    async def outdated(self, document_ids: Sequence[str], chunking_version: str) -> list[str]:
        from qdrant_client import models as qm

        if not document_ids:
            return []
        client = self.client()
        try:
            exists = await client.collection_exists(self.collection)
        except Exception as e:
            raise VectorStoreError(f"cannot check the index of {len(document_ids)} document(s): {e}") from e
        if not exists:
            return list(document_ids)
        out = []
        for doc_id in document_ids:
            of_doc = qm.FieldCondition(key="document_id", match=qm.MatchValue(value=doc_id))
            current = qm.FieldCondition(key="chunking_version", match=qm.MatchValue(value=chunking_version))
            total = await client.count(self.collection, count_filter=qm.Filter(must=[of_doc]), exact=True)
            other = await client.count(
                self.collection, count_filter=qm.Filter(must=[of_doc], must_not=[current]), exact=True
            )
            if total.count == 0 or other.count > 0:
                out.append(doc_id)
        return out

    async def document_labels(self, filters: RetrievalFilters) -> dict[str, str]:
        ids = filters.document_ids
        if not ids:
            return {}
        client = self.client()

        async def label_of(document_id: str) -> tuple[str, str]:
            flt = build_filter(RetrievalFilters(filters.project_id, (document_id,)))
            points, _ = await client.scroll(
                self.collection,
                scroll_filter=flt,
                limit=LABEL_SAMPLE,
                with_payload=["document_label"],
                with_vectors=False,
            )
            labels = [str((p.payload or {}).get("document_label") or "") for p in points]
            return document_id, max(labels, key=len, default="")

        try:
            pairs = await asyncio.gather(*(label_of(d) for d in ids))
        except Exception as e:
            if _not_found(e):  # nothing ingested yet
                return {}
            raise
        return {d: label for d, label in pairs if label}

    async def document_texts(self, filters: RetrievalFilters, *, limit: int = TEXT_SAMPLE) -> dict[str, list[str]]:
        ids = filters.document_ids
        if not ids:
            return {}
        client = self.client()

        async def texts_of(document_id: str) -> tuple[str, list[str]]:
            flt = build_filter(RetrievalFilters(filters.project_id, (document_id,)))
            points, _ = await client.scroll(
                self.collection,
                scroll_filter=flt,
                limit=limit,
                with_payload=["text", "heading_path"],
                with_vectors=False,
            )
            out = []
            for p in points:
                payload = p.payload or {}
                out.append("\n".join([*map(str, payload.get("heading_path") or []), str(payload.get("text") or "")]))
            return document_id, out

        try:
            pairs = await asyncio.gather(*(texts_of(d) for d in ids))
        except Exception as e:
            if _not_found(e):
                return {}
            raise
        return {d: texts for d, texts in pairs if texts}

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
