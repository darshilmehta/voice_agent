"""Ingestion service with fake providers: parse → chunk → embed → upsert → replace the document's index."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.providers.ingestion import IngestionError
from app.providers.registry import build_container
from app.services.ingestion import IngestionService

from .fakes import FakeEmbedder, FakeParser, FakeStore, make_chunk, make_parsed, vector_for


def make_service(chunks, store: FakeStore | None = None, **parsed):
    parser = FakeParser(make_parsed(**parsed), chunks)
    embedder = FakeEmbedder()
    store = store or FakeStore()
    return IngestionService(parser, embedder, store), parser, embedder, store


def test_ingest_file_indexes_every_chunk():
    chunks = [make_chunk(0, overlap_text="tail"), make_chunk(1, content_type="table", table_index=0)]
    svc, parser, embedder, store = make_service(chunks, page_count=3)
    res = asyncio.run(svc.ingest_file("/tmp/report.pdf", "proj1", "doc1", 1))

    assert parser.parsed_paths == [Path("/tmp/report.pdf")]
    assert parser.chunk_calls == [{"project_id": "proj1", "document_id": "doc1", "version": 1}]
    assert embedder.calls == [[c.embed_text for c in chunks]]  # headings + overlap + text
    assert {cid: p.vectors for cid, p in store.points.items()} == {c.chunk_id: vector_for(c.embed_text) for c in chunks}
    assert store.deletes == [("doc1", [c.chunk_id for c in chunks])]  # stale chunks/versions removed after upsert
    assert (res.chunk_count, res.page_count, res.table_count) == (2, 3, 0)
    assert res.chunks == chunks and res.document is parser.parsed
    assert res.parse_s_per_page == pytest.approx(res.timings.parse_s / 3)
    assert res.timings.total_s >= res.timings.parse_s >= 0
    assert parser.parsed._native is None  # Docling tree released after chunking


def test_reingest_replaces_the_previous_version():
    store = FakeStore()
    old = [make_chunk(i, chunk_id=f"doc1:v1:{i:04d}", version=1) for i in range(3)]
    svc, *_ = make_service(old, store=store)
    asyncio.run(svc.ingest_file("a.pdf", "proj1", "doc1", 1))
    new = [make_chunk(0, chunk_id="doc1:v2:0000", version=2)]
    svc, *_ = make_service(new, store=store)
    res = asyncio.run(svc.ingest_file("a.pdf", "proj1", "doc1", 2))
    assert list(store.points) == ["doc1:v2:0000"] and res.chunk_count == 1


def test_document_without_text_fails_before_touching_the_index():
    svc, _, embedder, store = make_service([])
    with pytest.raises(IngestionError, match="no text"):
        asyncio.run(svc.ingest_file("scan.pdf", "proj1", "doc1", 1))
    assert embedder.calls == [] and store.points == {} and store.deletes == []


def test_index_count_mismatch_is_an_error():
    store = FakeStore()
    store.count_override = 1
    svc, *_ = make_service([make_chunk(0), make_chunk(1)], store=store)
    with pytest.raises(IngestionError, match="expected 2"):
        asyncio.run(svc.ingest_file("a.pdf", "proj1", "doc1", 1))


def test_service_builds_from_the_container(load_local):
    container = build_container(load_local())
    svc = IngestionService.from_container(container)
    assert svc.parser is container["ingestion"] and svc.store is container["vector_store"]
