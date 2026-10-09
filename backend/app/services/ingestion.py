"""Ingestion pipeline (docs/DESIGN.md §3.1): parse → chunk → embed → upsert.

Library layer only: no database writes. The caller (job queue / API, built by another workstream) owns the
document record and status; it persists what ``IngestionResult`` returns (counts, tables with cells, timings) and
marks the document FAILED when ``ingest_file`` raises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..providers.ingestion import Chunk, DocumentParser, IngestionError, ParsedDocument, ParsedTable
from ..providers.registry import Container
from ..providers.retrieval import Embedder, IndexedChunk, RetrievalFilters, VectorStore


@dataclass(frozen=True, slots=True)
class IngestionTimings:
    parse_s: float
    chunk_s: float
    embed_s: float
    index_s: float  # upsert + removal of stale chunks + count check
    total_s: float


@dataclass(frozen=True, slots=True)
class IngestionResult:
    project_id: str
    document_id: str
    version: int
    page_count: int | None  # None for formats without pages (DOCX, MD, TXT)
    chunk_count: int
    table_count: int
    language: str | None
    document: ParsedDocument  # pages, items, tables (markdown + cells) and markdown export, for persistence
    chunks: list[Chunk]
    timings: IngestionTimings
    warnings: list[str] = field(default_factory=list)

    @property
    def tables(self) -> list[ParsedTable]:
        return self.document.tables

    @property
    def parse_s_per_page(self) -> float | None:
        return self.timings.parse_s / self.page_count if self.page_count else None


class IngestionService:
    def __init__(self, parser: DocumentParser, embedder: Embedder, store: VectorStore) -> None:
        self.parser = parser
        self.embedder = embedder
        self.store = store

    @classmethod
    def from_container(cls, container: Container) -> IngestionService:
        return cls(
            _expect(container["ingestion"], DocumentParser),
            _expect(container["embeddings"], Embedder),
            _expect(container["vector_store"], VectorStore),
        )

    async def ingest_file(
        self, path: Path | str, project_id: str, document_id: str, version: int, *, filename: str | None = None
    ) -> IngestionResult:
        """Index one document version. Afterwards the index holds exactly this version's chunks for
        ``document_id``: older versions and stale chunks are removed only once the new ones are stored, so the
        document stays searchable throughout. Raises IngestionError (or ModelUnavailableError) on failure.

        ``filename``: the name the user knows the document by, when ``path`` is a temporary copy with another name;
        chunks are labelled with it (``Chunk.document_label``)."""
        t0 = time.perf_counter()
        parsed = await self.parser.parse(Path(path))
        if filename and filename != parsed.source_name:
            parsed = parsed.model_copy(update={"source_name": filename})
        t1 = time.perf_counter()
        try:
            chunks = await self.parser.chunk(parsed, project_id=project_id, document_id=document_id, version=version)
        finally:
            parsed.drop_native()
        t2 = time.perf_counter()
        if not chunks:
            raise IngestionError(f"{parsed.source_name}: no text could be extracted (scanned without OCR? empty?)")

        vectors = await self.embedder.embed([c.embed_text for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
        t3 = time.perf_counter()

        await self.store.upsert([IndexedChunk(c, v) for c, v in zip(chunks, vectors, strict=True)])
        await self.store.delete_document(document_id, keep=[c.chunk_id for c in chunks])
        indexed = await self.store.count(RetrievalFilters(project_id, (document_id,)))
        if indexed != len(chunks):
            raise IngestionError(f"{parsed.source_name}: index holds {indexed} chunks, expected {len(chunks)}")
        t4 = time.perf_counter()

        return IngestionResult(
            project_id=project_id,
            document_id=document_id,
            version=version,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            table_count=len(parsed.tables),
            language=parsed.language,
            document=parsed,
            chunks=chunks,
            timings=IngestionTimings(
                parse_s=round(t1 - t0, 3),
                chunk_s=round(t2 - t1, 3),
                embed_s=round(t3 - t2, 3),
                index_s=round(t4 - t3, 3),
                total_s=round(t4 - t0, 3),
            ),
            warnings=list(parsed.warnings),
        )


def _expect[T](provider: object, kind: type[T]) -> T:
    if not isinstance(provider, kind):
        raise TypeError(f"expected a {kind.__name__} provider, got {type(provider).__name__}")
    return provider
