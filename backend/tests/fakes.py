"""In-memory stand-ins for the ingestion/retrieval providers: no ML libraries, models or Qdrant needed."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from app.providers.ingestion import Chunk, DocumentParser, ParsedDocument
from app.providers.retrieval import (
    DenseSparse,
    Embedder,
    IndexedChunk,
    Reranker,
    RetrievalFilters,
    SearchHit,
    SparseVector,
    VectorStore,
)


def make_chunk(index: int = 0, *, document_id: str = "doc1", project_id: str = "proj1", **over: Any) -> Chunk:
    fields: dict[str, Any] = {
        "chunk_id": f"{document_id}:v1:{index:04d}",
        "project_id": project_id,
        "document_id": document_id,
        "version": 1,
        "chunk_index": index,
        "chunking_version": "v1",
        "page_start": 1,
        "page_end": 1,
        "heading_path": ["Section"],
        "content_type": "paragraph",
        "language": "en",
        "text": f"text of chunk {index}",
        "token_count": 5,
    }
    fields.update(over)
    return Chunk(**fields)


def make_parsed(**over: Any) -> ParsedDocument:
    fields: dict[str, Any] = {
        "source_name": "report.pdf",
        "format": "pdf",
        "page_count": 3,
        "pages": [],
        "items": [],
        "tables": [],
        "markdown": "",
        "language": "en",
        "parse_seconds": 0.5,
    }
    fields.update(over)
    return ParsedDocument(**fields)


def hit(chunk: Chunk, score: float = 0.5, dense: float | None = 0.8) -> SearchHit:
    return SearchHit(chunk=chunk, score=score, dense_score=dense)


def vector_for(text: str) -> DenseSparse:
    digest = hashlib.sha256(text.encode()).digest()
    return DenseSparse(
        dense=[b / 255 for b in digest[:8]], sparse=SparseVector([digest[0], 300 + digest[1]], [0.5, 0.2])
    )


class FakeEmbedder(Embedder):
    name = "fake"
    dim = 8

    def __init__(self) -> None:  # no config/context needed
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[DenseSparse]:
        self.calls.append(list(texts))
        return [vector_for(t) for t in texts]


class FakeReranker(Reranker):
    name = "fake"

    def __init__(self, scorer: Callable[[str, str], float]) -> None:
        self.scorer = scorer
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [self.scorer(query, p) for p in passages]


class FakeStore(VectorStore):
    name = "fake"

    def __init__(self, results: Sequence[Sequence[SearchHit]] = ()) -> None:
        self.results = list(results)  # one list per hybrid_search call, in order
        self.searches: list[tuple[DenseSparse, RetrievalFilters, int | None]] = []
        self.points: dict[str, IndexedChunk] = {}
        self.deletes: list[tuple[str, list[str]]] = []
        self.count_override: int | None = None

    async def upsert(self, chunks: Sequence[IndexedChunk]) -> None:
        for c in chunks:
            self.points[c.chunk.chunk_id] = c

    async def hybrid_search(
        self, query: DenseSparse, filters: RetrievalFilters, *, limit: int | None = None
    ) -> list[SearchHit]:
        self.searches.append((query, filters, limit))
        return list(self.results.pop(0)) if self.results else []

    async def delete_document(self, document_id: str, *, keep: Iterable[str] = ()) -> None:
        kept = list(keep)
        self.deletes.append((document_id, kept))
        for cid in [c for c, p in self.points.items() if p.chunk.document_id == document_id and c not in kept]:
            del self.points[cid]

    async def delete_project(self, project_id: str) -> None:
        for cid in [c for c, p in self.points.items() if p.chunk.project_id == project_id]:
            del self.points[cid]

    async def count(self, filters: RetrievalFilters) -> int:
        if self.count_override is not None:
            return self.count_override
        return sum(
            1
            for p in self.points.values()
            if p.chunk.project_id == filters.project_id
            and (filters.document_ids is None or p.chunk.document_id in filters.document_ids)
        )


class FakeParser(DocumentParser):
    name = "fake"

    def __init__(self, parsed: ParsedDocument, chunks: list[Chunk]) -> None:
        self.parsed = parsed
        self.chunks = chunks
        self.parsed_paths: list[Any] = []
        self.chunk_calls: list[dict[str, Any]] = []

    async def parse(self, path: Any) -> ParsedDocument:
        self.parsed_paths.append(path)
        self.parsed._native = object()
        return self.parsed

    async def chunk(self, document: ParsedDocument, *, project_id: str, document_id: str, version: int) -> list[Chunk]:
        self.chunk_calls.append({"project_id": project_id, "document_id": document_id, "version": version})
        return list(self.chunks)
