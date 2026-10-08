"""End to end on the smoke-test annual report: Docling → chunks → BGE-M3 → Qdrant → hybrid search → reranker.

annual_report.pdf (scripts/smoke/05_docling.py) has 3 pages; page 2 holds "Key Metrics" with a FY23/FY24 table
(EBITDA margin 16.9% → 18.2%) followed by a sentence restating the margin.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from app.providers.ingestion import DoclingParser
from app.providers.registry import Container
from app.providers.retrieval import RetrievalFilters
from app.services.ingestion import IngestionResult, IngestionService
from app.services.retrieval import RetrievalResult, RetrievalService

from .conftest import METRICS

pytestmark = pytest.mark.integration

PROJECT = "it-project"
DOCUMENT = "annual-report"
EN = "What was the EBITDA margin in FY24?"
HI = "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?"


@pytest.fixture(scope="module")
def ingested(container: Container, smoke_docs: Path, loop: asyncio.AbstractEventLoop) -> IngestionResult:
    svc = IngestionService.from_container(container)
    pdf = smoke_docs / "annual_report.pdf"
    res = loop.run_until_complete(svc.ingest_file(pdf, PROJECT, DOCUMENT, 1))
    t = res.timings
    METRICS["ingest (cold: includes Docling, tokenizer and BGE-M3 load)"] = (
        f"parse {t.parse_s:.2f}s ({res.parse_s_per_page:.2f} s/page), chunk {t.chunk_s:.2f}s, "
        f"embed {t.embed_s:.2f}s, index {t.index_s:.2f}s, total {t.total_s:.2f}s"
    )
    parser = container["ingestion"]
    assert isinstance(parser, DoclingParser)
    t0 = time.perf_counter()
    warm = loop.run_until_complete(parser.parse(pdf))
    METRICS["parse warm"] = f"{(time.perf_counter() - t0) / (warm.page_count or 1):.2f} s/page"
    t0 = time.perf_counter()
    loop.run_until_complete(container["embeddings"].embed([c.embed_text for c in res.chunks]))  # type: ignore[attr-defined]
    METRICS["embed warm"] = f"{len(res.chunks)} chunks in {time.perf_counter() - t0:.2f}s"
    parser.release()  # Docling is for ingestion only (§8): free it before the reranker loads
    METRICS["chunks"] = ", ".join(
        f"#{c.chunk_index} p{c.page_start} {c.content_type} {c.heading_path} {c.token_count} tok" for c in res.chunks
    )
    return res


def retrieve(container: Container, loop: asyncio.AbstractEventLoop, query: str, **kw) -> RetrievalResult:
    svc = RetrievalService.from_container(container)
    return loop.run_until_complete(svc.retrieve(query, project_id=PROJECT, **kw))


def ranks(res: RetrievalResult) -> str:
    return " | ".join(
        f"{i}. p{r.chunk.page_start} {r.chunk.content_type} rerank={r.rerank_score:.3f} "
        f"dense={r.dense_score:.3f} (search #{r.search_rank})"
        for i, r in enumerate(res.chunks, start=1)
    )


def is_fy24_margin(text: str) -> bool:
    return "18.2%" in text and "EBITDA" in text


def test_ingestion_result(ingested: IngestionResult, container: Container, loop: asyncio.AbstractEventLoop):
    assert (ingested.page_count, ingested.table_count, ingested.language) == (3, 1, "en")
    assert ingested.chunk_count == len(ingested.chunks) >= 4
    for c in ingested.chunks:
        assert (c.project_id, c.document_id, c.version, c.chunking_version) == (PROJECT, DOCUMENT, 1, "v1")
        assert c.page_start is not None and 1 <= c.page_start <= (c.page_end or 0) <= 3
        assert c.heading_path and c.language == "en"
        assert c.token_count <= 500 or c.content_type == "table"

    (table,) = ingested.tables
    assert table.page_start == 2 and table.bbox is not None
    assert ["EBITDA margin", "16.9%", "18.2%"] in table.grid()
    assert ["Revenue from operations (INR crore)", "3,142", "4,210"] in table.grid()
    assert any(c.column_header for c in table.cells)

    (tchunk,) = [c for c in ingested.chunks if c.content_type == "table"]  # the table is one chunk, whole
    assert tchunk.table_index == 0 and tchunk.page_start == 2
    assert all(row in tchunk.text for row in ("| EBITDA margin", "18.2%", "4,210", "441"))

    store = container["vector_store"]
    count = loop.run_until_complete(store.count(RetrievalFilters(PROJECT, (DOCUMENT,))))  # type: ignore[attr-defined]
    assert count == ingested.chunk_count


def test_english_query_ranks_the_fy24_margin_first(ingested, container, loop):
    retrieve(container, loop, EN)  # loads the reranker
    res = retrieve(container, loop, EN)
    METRICS["EN ranks"] = ranks(res)
    METRICS["EN timings (warm, ms)"] = res.timings_ms
    METRICS["EN confidence"] = res.confidence
    top = res.chunks[0].chunk
    assert top.page_start == 2 and is_fy24_margin(top.text)
    assert res.confidence is not None and res.confidence.above_threshold


def test_hindi_query_with_router_english_query_ranks_the_fy24_margin_first(ingested, container, loop):
    res = retrieve(container, loop, HI, query_en=EN)
    METRICS["HI + query_en ranks"] = ranks(res)
    METRICS["HI + query_en confidence"] = res.confidence
    top = res.chunks[0].chunk
    assert top.page_start == 2 and is_fy24_margin(top.text)
    assert res.confidence is not None and res.confidence.above_threshold


def test_hindi_query_alone_ranks_the_fy24_margin_first(ingested, container, loop):
    res = retrieve(container, loop, HI)
    METRICS["HI alone ranks"] = ranks(res)
    METRICS["HI alone confidence"] = res.confidence
    top = res.chunks[0].chunk
    assert top.page_start == 2 and is_fy24_margin(top.text)


def test_retrieval_never_leaves_the_project_or_document_scope(ingested, container, loop):
    svc = RetrievalService.from_container(container)
    other_project = loop.run_until_complete(svc.retrieve(EN, project_id="another-project"))
    assert other_project.chunks == [] and other_project.confidence is None
    other_doc = loop.run_until_complete(svc.retrieve(EN, project_id=PROJECT, document_ids=["another-doc"]))
    assert other_doc.chunks == []
    scoped = loop.run_until_complete(svc.retrieve(EN, project_id=PROJECT, document_ids=[DOCUMENT]))
    assert scoped.chunks and {r.chunk.document_id for r in scoped.chunks} == {DOCUMENT}


def test_delete_document_then_project(ingested, container, loop):
    store = container["vector_store"]
    scope = RetrievalFilters(PROJECT)
    loop.run_until_complete(store.delete_document(DOCUMENT))  # type: ignore[attr-defined]
    assert loop.run_until_complete(store.count(scope)) == 0  # type: ignore[attr-defined]
    loop.run_until_complete(store.delete_project(PROJECT))  # type: ignore[attr-defined]  (no-op, must not fail)
