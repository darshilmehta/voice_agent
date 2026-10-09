"""Retrieval evaluation harness (docs/DESIGN.md §10 phases 2 and 9).

Ingests the eval corpus (``scripts/eval/build_corpus.py``) into a throwaway Qdrant collection with the real
``IngestionService`` (Docling, chunking, BGE-M3), then asks every question of ``evals/retrieval/questions.jsonl`` the
way the chat pipeline does, with the real ``RetrievalService`` (hybrid search, RRF, reranker, confidence), and
reports ranking quality before and after the reranker, citation accuracy, the abstention gate and latency.

    cd backend
    uv run --group ml python -m app.evals.retrieval_eval            # the default config, every query variant
    uv run --group ml python -m app.evals.retrieval_eval --prefetch-k 8,12,20 --rerank-top-n 5 --threshold 0.05
    uv run --group ml python -m app.evals.retrieval_eval --answers  # also run the answer LLM (Ollama must be up)
    uv run python -m app.evals.retrieval_eval --render ../data/eval/results/<run>/results.json   # re-render summary.md

Why a module of the backend (not a script): it is built from the app's own services, providers and settings, so it
runs in the backend's environment (``--group ml``), and its pure parts (``manifest``, ``questions``, ``metrics``,
``analysis``, ``report``) are unit-tested in CI with fakes.

Query variants (per question): ``raw`` = ``retrieve(question)``, what the phase-1 pipeline does; ``routed`` =
``retrieve(question, query_en=...)`` for Hindi/Hinglish questions, what the phase-3 router will do (both queries are
searched and fused, the reranker scores the English one); ``english`` = ``retrieve(query_en)`` alone; ``clean`` = the
correctly transcribed question for ASR-noise questions. The headline ("pipeline") uses ``routed`` where a question
has an English query and ``raw`` otherwise.
Candidates (before the reranker) come from the public ``RetrievalService.search``; everything after comes from
``RetrievalService.retrieve``, untouched. Nothing in the retrieval code is changed or reimplemented here.

Environment (defaults suit the main checkout; the same variables as the integration tests):
    MODELS_ROOT   directory with huggingface/ (HF_HOME) and docling/ (artifacts)    default <repo>/data/models
    QDRANT_URL    a local Qdrant; a uniquely named collection is created and dropped default http://127.0.0.1:6333
    EVAL_DOCS     the generated corpus folder                                       default <repo>/data/eval/docs
    EVAL_RESULTS  where run folders go                                              default <repo>/data/eval/results
    plus the config overrides the app understands (EMBEDDINGS__DEVICE=cpu, LLM__BASE_URL=..., APP_CONFIG_FILE=...)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..providers.base import HealthStatus, ProviderContext
from ..providers.ingestion import Chunk, DocumentParser
from ..providers.llm import LLMClient, LLMMessage
from ..providers.registry import Container, build_container, provider_class
from ..providers.retrieval import SearchHit, VectorStore
from ..services.ingestion import IngestionResult, IngestionService
from ..services.language import choose_language
from ..services.prompts import ANSWER_LENGTHS, abstention, answer_system_prompt, answer_user_prompt
from ..services.retrieval import RankedChunk, RetrievalResult, RetrievalService
from ..services.sources import Source, build_sources, finalize_answer
from ..settings import PROJECT_ROOT, ConfigError, RetrievalSection, Settings, load_settings
from .analysis import AnswerRecord, QueryRecord, query_row, summarize_variants
from .manifest import Manifest, ManifestError, load_manifest
from .metrics import Passage, Target, answer_contains_all
from .questions import Question, QuestionFileError, load_questions, summarize_questions, validate_questions
from .textnorm import contains_all

PROJECT_ID = "eval-project"
EVAL_ROOT = PROJECT_ROOT / "evals/retrieval"
RESULTS_FORMAT = 1


class EvalSetupError(RuntimeError):
    """Something the evaluation needs is missing (corpus, models, Qdrant, LLM); the message says what to do."""


# ------------------------------------------------------------------ options


@dataclass(frozen=True)
class EvalOptions:
    """What to vary. Empty lists mean "the configured value"; several values make a grid (every combination is run
    on the same ingested collection; chunk sizes re-ingest)."""

    chunk_sizes: tuple[int, ...] = ()
    overlap: int | None = None
    prefetch_ks: tuple[int, ...] = ()
    rerank_top_ns: tuple[int, ...] = ()
    fusions: tuple[str, ...] = ()
    threshold: float | None = None  # min_rerank_score for the confusion matrix (default: the configured one)
    variants: tuple[str, ...] = ("raw", "routed", "english", "clean")
    answers: bool = False
    answer_length: str = "short"
    categories: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()
    limit: int | None = None
    keep_collection: bool = False
    reuse_collection: str | None = None
    progress_every: int = 25


# ------------------------------------------------------------------ services (one set per configuration)


@dataclass
class EvalServices:
    """The services of one configuration. The real factory builds them from the app's providers; tests use fakes."""

    ingestion: IngestionService
    retrieval: RetrievalService
    store: VectorStore
    llm: LLMClient | None = None
    close: Callable[[], Awaitable[None]] | None = None


ServiceFactory = Callable[[Settings], EvalServices]


def container_factory(container: Container) -> ServiceFactory:
    """Services for any settings, sharing the container's models: the embedder, reranker and LLM are built once (they
    load lazily and are expensive); the document parser and the Qdrant store are built per configuration because they
    read the chunking and retrieval settings."""
    embedder, reranker, llm = container["embeddings"], container["reranker"], container["llm"]
    http = container.http

    def make(settings: Settings) -> EvalServices:
        ctx = ProviderContext(settings=settings, http=http)
        parser = provider_class("ingestion", settings.ingestion.provider)(settings.ingestion, ctx)
        store = provider_class("vector_store", settings.vector_store.provider)(settings.vector_store, ctx)
        if not isinstance(parser, DocumentParser) or not isinstance(store, VectorStore):
            raise TypeError("the configured ingestion/vector_store providers don't implement the interfaces")
        ingestion = IngestionService(parser, embedder, store)  # type: ignore[arg-type]
        retrieval = RetrievalService(embedder, reranker, store, settings.retrieval)  # type: ignore[arg-type]
        return EvalServices(ingestion, retrieval, store, llm if isinstance(llm, LLMClient) else None, close=store.close)

    return make


# ------------------------------------------------------------------ settings


def with_params(
    settings: Settings,
    *,
    collection: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    prefetch_k: int | None = None,
    rerank_top_n: int | None = None,
    fusion: str | None = None,
    threshold: float | None = None,
) -> Settings:
    """``settings`` with some knobs changed (Settings sections are frozen: copies, validated by construction)."""
    out = settings
    if collection is not None:
        out = out.model_copy(update={"vector_store": out.vector_store.model_copy(update={"collection": collection})})
    if chunk_size is not None or overlap is not None:
        chunking = out.ingestion.chunking
        size = chunk_size if chunk_size is not None else chunking.target_tokens
        ov = overlap if overlap is not None else chunking.overlap_tokens
        if size <= 0 or not 0 <= ov < size:
            raise EvalSetupError(f"invalid chunking: target {size} tokens with overlap {ov}")
        chunking = chunking.model_copy(
            update={"target_tokens": size, "overlap_tokens": ov, "version": f"eval-{size}-{ov}"}
        )
        out = out.model_copy(update={"ingestion": out.ingestion.model_copy(update={"chunking": chunking})})
    updates: dict[str, Any] = {}
    if prefetch_k is not None:
        updates["prefetch_k"] = prefetch_k
    if rerank_top_n is not None:
        updates["rerank_top_n"] = rerank_top_n
    if fusion is not None:
        updates["fusion"] = fusion
    if threshold is not None:
        updates["min_rerank_score"] = threshold
    if updates:
        try:
            retrieval = RetrievalSection.model_validate({**out.retrieval.model_dump(), **updates})
        except ValidationError as e:
            first = e.errors()[0]
            raise EvalSetupError(f"invalid retrieval parameter {first['loc'][0]}: {first['msg']}") from e
        out = out.model_copy(update={"retrieval": retrieval})
    return out


def settings_snapshot(s: Settings) -> dict[str, Any]:
    """The parameters that matter for reading a result."""
    return {
        "embeddings": {"model": s.embeddings.model, "device": s.embeddings.device, "fp16": s.embeddings.fp16},
        "reranker": {"model": s.reranker.model, "device": s.reranker.device, "fp16": s.reranker.fp16},
        "llm": {"chat_model": s.llm.chat_model, "temperature_answer": s.llm.temperature.answer},
        "chunking": s.ingestion.chunking.model_dump(),
        "ocr": {"enabled": s.ingestion.ocr, "engine": s.ingestion.ocr_engine},
        "retrieval": s.retrieval.model_dump(),
    }


# ------------------------------------------------------------------ corpus and questions


def document_id(name: str) -> str:
    return "eval-" + re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")


def targets_for(question: Question, manifest: Manifest) -> list[Target]:
    """Expected locations as metric targets: the pages, and the evidence strings of the facts they come from."""
    out = []
    for e in question.expected:
        facts = [manifest.fact(i) for i in e.facts]
        out.append(
            Target(
                document=e.document,
                pages=tuple(e.pages),
                evidence=tuple(tuple(f.evidence) for f in facts),
                approximate=any(f.ocr for f in facts),
            )
        )
    return out


def select_questions(questions: Sequence[Question], options: EvalOptions) -> list[Question]:
    out = [
        q
        for q in questions
        if (not options.categories or q.category in options.categories)
        and (not options.languages or q.language in options.languages)
        and (not options.ids or q.id in options.ids)
    ]
    return out[: options.limit] if options.limit else out


def passage(chunk: Chunk, names: dict[str, str], score: float | None) -> Passage:
    """A retrieved chunk as a metric passage. Text = heading path + chunk text (what is indexed and cited)."""
    return Passage(
        document=names.get(chunk.document_id, chunk.document_id),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text="\n".join([*chunk.heading_path, chunk.text]),
        chunk_id=chunk.chunk_id,
        score=score,
    )


def candidate_passage(hit: SearchHit, names: dict[str, str]) -> Passage:
    return passage(hit.chunk, names, hit.score)


def ranked_passage(r: RankedChunk, names: dict[str, str]) -> Passage:
    return passage(r.chunk, names, r.rerank_score)


# ------------------------------------------------------------------ ingestion


@dataclass
class IngestedCorpus:
    collection: str
    documents: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    ids: dict[str, str]  # file name -> document id


def fact_coverage(manifest: Manifest, name: str, chunks: Sequence[Chunk]) -> dict[str, Any]:
    """Which planted facts survived parsing and chunking: the evidence is in some chunk (and that chunk's page range
    includes a page that states it). Separates ingestion failures (OCR, tables, chunk splits) from retrieval ones."""
    facts = manifest.facts_of(name)
    in_chunk: list[str] = []
    right_page: list[str] = []
    missing: list[str] = []
    for f in facts:
        found = page_ok = False
        for c in chunks:
            if not contains_all("\n".join([*c.heading_path, c.text]), f.evidence, approximate=f.ocr):
                continue
            found = True
            if (
                not f.pages
                or c.page_start is None
                or any(c.page_start <= p <= (c.page_end or c.page_start) for p in f.pages)
            ):
                page_ok = True
        (in_chunk if found else missing).append(f.id)
        if page_ok:
            right_page.append(f.id)
    return {
        "facts": len(facts),
        "in_a_chunk": len(in_chunk),
        "in_a_chunk_on_a_stating_page": len(right_page),
        "missing": missing,
        "approximate_matching": any(f.ocr for f in facts),
    }


def _chunk_stats(chunks: Sequence[Chunk]) -> dict[str, Any]:
    tokens = [c.token_count for c in chunks]
    spans = [(c.page_end or c.page_start) - c.page_start + 1 for c in chunks if c.page_start is not None]
    return {
        "content_types": dict(Counter(c.content_type for c in chunks)),
        "tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "tokens_max": max(tokens) if tokens else None,
        "multi_page_chunks": sum(1 for s in spans if s > 1),
    }


async def drop_collection(store: VectorStore) -> None:
    drop = getattr(store, "drop_collection", None)
    if drop is not None:
        await drop()
    else:  # fakes and other stores without a collection concept
        await store.delete_project(PROJECT_ID)


async def ingest_corpus(
    services: EvalServices, manifest: Manifest, docs_dir: Path, collection: str, log: Callable[[str], None]
) -> IngestedCorpus:
    ids: dict[str, str] = {}
    documents: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for doc in manifest.documents:
        path = docs_dir / doc.name
        if not path.is_file():
            raise EvalSetupError(f"{path} not found: build the corpus first (uv run scripts/eval/build_corpus.py)")
        ids[doc.name] = document_id(doc.name)
    for doc in manifest.documents:
        log(f"  ingesting {doc.name} ...")
        result: IngestionResult = await services.ingestion.ingest_file(
            docs_dir / doc.name, PROJECT_ID, ids[doc.name], 1
        )
        t = result.timings
        documents.append(
            {
                "document": doc.name,
                "document_id": ids[doc.name],
                "format": doc.format,
                "pages_planned": doc.pages,
                "pages_parsed": result.page_count,
                "chunks": result.chunk_count,
                "tables": result.table_count,
                "language": result.language,
                "seconds": {
                    "parse": t.parse_s,
                    "chunk": t.chunk_s,
                    "embed": t.embed_s,
                    "index": t.index_s,
                    "total": t.total_s,
                },
                "warnings": result.warnings[:5],
                **_chunk_stats(result.chunks),
                "fact_coverage": fact_coverage(manifest, doc.name, result.chunks),
            }
        )
        for c in result.chunks:
            chunk_rows.append(
                {
                    "collection": collection,
                    "document": doc.name,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "pages": [c.page_start, c.page_end],
                    "content_type": c.content_type,
                    "heading_path": c.heading_path,
                    "tokens": c.token_count,
                    "text": c.text,
                }
            )
        cov = documents[-1]["fact_coverage"]
        log(
            f"    {result.chunk_count} chunks ({result.page_count or '-'} pages) in {t.total_s:.1f}s; "
            f"{cov['in_a_chunk']}/{cov['facts']} planted facts found in a chunk"
        )
    release = getattr(services.ingestion.parser, "release", None)
    if release is not None:
        release()  # Docling is for ingestion only (§8): free it before the reranker loads
    return IngestedCorpus(collection, documents, chunk_rows, ids)


# ------------------------------------------------------------------ one configuration


def query_variants(q: Question, variants: Sequence[str]) -> list[tuple[str, str, str | None]]:
    """(variant, query text, English query) for each variant that applies to the question."""
    out: list[tuple[str, str, str | None]] = []
    if "raw" in variants:
        out.append(("raw", q.question, None))
    if "routed" in variants and q.query_en:
        out.append(("routed", q.question, q.query_en))
    if "english" in variants and q.query_en:
        out.append(("english", q.query_en, None))
    if "clean" in variants and q.clean_question:
        out.append(("clean", q.clean_question, None))
    return out


async def answer_one(
    services: EvalServices,
    settings: Settings,
    q: Question,
    result: RetrievalResult,
    sources: list[Source],
    names: dict[str, str],
    length: str,
) -> AnswerRecord:
    """The answer step exactly as ChatTurnService builds it: gate, numbered sources, grounded prompt, streamed answer,
    citations validated. Single-turn (no history)."""
    language = choose_language(q.question, None, "en")
    confidence = result.confidence
    if confidence is None or not confidence.above_threshold:
        return AnswerRecord(abstention(language, "not_covered"), abstained=True)
    assert services.llm is not None
    prompt = [
        LLMMessage("system", answer_system_prompt(language, length)),  # type: ignore[arg-type]
        LLMMessage("user", answer_user_prompt(q.question, sources, language)),
    ]
    t0 = time.perf_counter()
    text = await services.llm.generate(prompt, max_tokens=ANSWER_LENGTHS[length].max_tokens)  # type: ignore[index]
    llm_ms = round((time.perf_counter() - t0) * 1000, 1)
    answer, citations = finalize_answer(text, sources)
    by_chunk = {s.chunk.chunk_id: s for s in sources}
    cited = [
        passage(by_chunk[c.chunk_id].chunk, names, by_chunk[c.chunk_id].rerank_score)
        for c in citations
        if c.chunk_id in by_chunk
    ]
    return AnswerRecord(
        answer,
        abstained=False,
        cited=cited,
        contains_ok=None if q.should_abstain else answer_contains_all(answer, q.answer_contains),
        llm_ms=llm_ms,
    )


async def run_queries(
    services: EvalServices,
    settings: Settings,
    manifest: Manifest,
    questions: Sequence[Question],
    options: EvalOptions,
    ingested: IngestedCorpus,
    log: Callable[[str], None],
) -> dict[str, dict[str, QueryRecord]]:
    """Run every question under every applicable variant; returns records[variant][question id]."""
    names = {v: k for k, v in ingested.ids.items()}  # document id -> file name
    doc_ids = list(ingested.ids.values())
    records: dict[str, dict[str, QueryRecord]] = {}
    if options.answers and services.llm is None:
        raise EvalSetupError("--answers needs an LLM provider")
    # Warm-up: loads the reranker (and the embedder after a restart) so the first question isn't timed with them.
    await services.retrieval.retrieve("warm-up query", project_id=PROJECT_ID, document_ids=doc_ids)
    todo = [(q, v) for q in questions for v in query_variants(q, options.variants)]
    for n, (q, (variant, text, query_en)) in enumerate(todo, start=1):
        candidates = await services.retrieval.search(
            text, project_id=PROJECT_ID, document_ids=doc_ids, query_en=query_en
        )
        result = await services.retrieval.retrieve(text, project_id=PROJECT_ID, document_ids=doc_ids, query_en=query_en)
        conf = result.confidence
        sources: list[Source] = []
        if conf is not None and conf.above_threshold:
            sources = build_sources(result.chunks, names, budget_tokens=settings.retrieval.context_token_budget)
        rec = QueryRecord(
            qid=q.id,
            variant=variant,
            category=q.category,
            language=q.language,
            should_abstain=q.should_abstain,
            question=q.question,
            query=text,
            query_en=query_en,
            rerank_query=result.rerank_query,
            targets=targets_for(q, manifest),
            candidates=[candidate_passage(h, names) for h in candidates],
            ranked=[ranked_passage(r, names) for r in result.chunks],
            top_score=conf.top_score if conf else None,
            gap=conf.gap if conf else None,
            dense=conf.dense_similarity if conf else None,
            timings_ms=dict(result.timings_ms),
            sources=[passage(s.chunk, names, s.rerank_score) for s in sources],
            veto=conf is not None and bool(conf.missing_periods or conf.missing_subjects),
            subtype=q.subtype,
        )
        is_pipeline_variant = variant == "routed" or (
            variant == "raw" and not (q.query_en and "routed" in options.variants)
        )
        if options.answers and is_pipeline_variant:
            rec.answer = await answer_one(services, settings, q, result, sources, names, options.answer_length)
        records.setdefault(variant, {})[q.id] = rec
        if n % options.progress_every == 0 or n == len(todo):
            log(f"  {n}/{len(todo)} queries")
    return records


# ------------------------------------------------------------------ orchestration


def run_label(chunk_size: int, overlap: int, prefetch_k: int, top_n: int, fusion: str) -> str:
    return f"chunk{chunk_size}+{overlap}, prefetch_k={prefetch_k}, rerank_top_n={top_n}, fusion={fusion}"


async def run_eval(
    settings: Settings,
    factory: ServiceFactory,
    manifest: Manifest,
    questions: Sequence[Question],
    docs_dir: Path,
    options: EvalOptions,
    *,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ingest, query and aggregate. Returns (results dict, chunk rows). One ingestion per chunk size; every
    (prefetch_k, rerank_top_n, fusion) combination runs on it."""
    problems = validate_questions(questions, manifest)
    if problems:
        raise EvalSetupError("the question set is inconsistent with the manifest:\n  - " + "\n  - ".join(problems[:20]))
    if "raw" not in options.variants:
        raise EvalSetupError("the raw variant is required: it is the baseline and the pipeline view's fallback")
    selected = select_questions(questions, options)
    if not selected:
        raise EvalSetupError("no question matches the filters")
    base = settings.retrieval
    chunk_sizes = options.chunk_sizes or (settings.ingestion.chunking.target_tokens,)
    overlap = options.overlap if options.overlap is not None else settings.ingestion.chunking.overlap_tokens
    prefetch_ks = options.prefetch_ks or (base.prefetch_k,)
    top_ns = options.rerank_top_ns or (base.rerank_top_n,)
    fusions = options.fusions or (base.fusion,)
    threshold = options.threshold if options.threshold is not None else base.min_rerank_score
    if options.reuse_collection and len(chunk_sizes) != 1:
        raise EvalSetupError("--reuse-collection works with a single --chunk-size (the one it was ingested with)")

    run_id = uuid.uuid4().hex[:8]
    runs: list[dict[str, Any]] = []
    ingestions: dict[str, Any] = {}
    chunk_rows: list[dict[str, Any]] = []
    for size in chunk_sizes:
        collection = options.reuse_collection or f"eval_{run_id}_c{size}"
        ingest_settings = with_params(
            settings, collection=collection, chunk_size=size, overlap=overlap, threshold=threshold
        )
        ingest_services = factory(ingest_settings)
        log(f"chunk size {size} (+{overlap} overlap) -> collection {collection}")
        try:
            if options.reuse_collection:
                ids = {d.name: document_id(d.name) for d in manifest.documents}
                ingested = IngestedCorpus(collection, [], [], ids)
                log("  reusing the existing collection, no ingestion")
            else:
                ingested = await ingest_corpus(ingest_services, manifest, docs_dir, collection, log)
            ingestions[str(size)] = {"collection": collection, "overlap": overlap, "documents": ingested.documents}
            chunk_rows += ingested.chunks
            for k in prefetch_ks:
                for top_n in top_ns:
                    for fusion in fusions:
                        label = run_label(size, overlap, k, top_n, fusion)
                        log(f"run: {label}")
                        run_settings = with_params(
                            ingest_settings, prefetch_k=k, rerank_top_n=top_n, fusion=fusion, threshold=threshold
                        )
                        services = factory(run_settings)
                        try:
                            records = await run_queries(
                                services, run_settings, manifest, selected, options, ingested, log
                            )
                        finally:
                            if services.close is not None:
                                await services.close()
                        summary = summarize_variants(records, threshold=threshold)
                        rows = [query_row(r, threshold) for by_q in records.values() for r in by_q.values()]
                        runs.append(
                            {
                                "label": label,
                                "params": {
                                    "chunk_size": size,
                                    "overlap": overlap,
                                    "prefetch_k": k,
                                    "rerank_top_n": top_n,
                                    "fusion": fusion,
                                    "threshold": threshold,
                                },
                                "settings": settings_snapshot(run_settings),
                                "ingestion": str(size),
                                "variants": summary,
                                "rows": rows,
                            }
                        )
        finally:
            if not options.keep_collection and not options.reuse_collection:
                try:
                    await drop_collection(ingest_services.store)
                except Exception as e:  # best effort: a leftover throwaway collection is harmless
                    log(f"  could not drop collection {collection}: {type(e).__name__}: {e}")
            if ingest_services.close is not None:
                await ingest_services.close()
    results = {
        "format": RESULTS_FORMAT,
        "meta": run_meta(settings, options, threshold),
        "corpus": {
            "docs_dir": str(docs_dir),
            "documents": [d.model_dump() for d in manifest.documents],
            "facts": len(manifest.facts),
        },
        "questions": {"n": len(selected), **summarize_questions(selected)},
        "ingestion": ingestions,
        "runs": runs,
    }
    return results, chunk_rows


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, cwd=PROJECT_ROOT
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, cwd=PROJECT_ROOT
        )
        sha = out.stdout.strip() or None
        return f"{sha}{'+dirty' if sha and dirty.stdout.strip() else ''}" if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_meta(settings: Settings, options: EvalOptions, threshold: float) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "command": " ".join(sys.argv),
        "config_file": str(settings.source) if settings.source else None,
        "threshold": threshold,
        "variants": list(options.variants),
        "answers": options.answers,
        "filters": {
            "categories": list(options.categories),
            "languages": list(options.languages),
            "ids": list(options.ids),
            "limit": options.limit,
        },
    }


# ------------------------------------------------------------------ outputs


def write_outputs(out_dir: Path, results: dict[str, Any], chunk_rows: Sequence[dict[str, Any]]) -> Path:
    from .report import render_markdown

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(results), encoding="utf-8")
    if chunk_rows:
        with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for row in chunk_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_dir


# ------------------------------------------------------------------ CLI


def _ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in value.split(",") if v.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from e


def _words(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.evals.retrieval_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("what to measure")
    g.add_argument("--questions", type=Path, default=EVAL_ROOT / "questions.jsonl", help="question set (JSONL)")
    g.add_argument("--manifest", type=Path, default=EVAL_ROOT / "manifest.json", help="corpus manifest")
    g.add_argument(
        "--docs", type=Path, default=None, help="generated corpus folder (env EVAL_DOCS, default data/eval/docs)"
    )
    g.add_argument(
        "--variants",
        type=_words,
        default=("raw", "routed", "english", "clean"),
        help="query variants: raw,routed,english,clean",
    )
    g.add_argument("--categories", type=_words, default=(), help="only these categories")
    g.add_argument("--languages", type=_words, default=(), help="only these languages (en,hi,hinglish)")
    g.add_argument("--ids", type=_words, default=(), help="only these question ids")
    g.add_argument("--limit", type=int, default=None, help="only the first N questions (after filters)")
    t = p.add_argument_group("parameters to compare (comma-separated lists run every combination)")
    t.add_argument(
        "--prefetch-k",
        type=_ints,
        default=(),
        help="candidates per search leg before fusion (config: retrieval.prefetch_k)",
    )
    t.add_argument(
        "--rerank-top-n", type=_ints, default=(), help="passages kept after the reranker (retrieval.rerank_top_n)"
    )
    t.add_argument(
        "--chunk-size",
        type=_ints,
        default=(),
        help="chunk target tokens; each size re-ingests (ingestion.chunking.target_tokens)",
    )
    t.add_argument("--overlap", type=int, default=None, help="chunk overlap tokens (ingestion.chunking.overlap_tokens)")
    t.add_argument("--fusion", type=_words, default=(), help="rrf and/or dbsf (retrieval.fusion)")
    t.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="min_rerank_score for the confusion matrix (config value by default)",
    )
    a = p.add_argument_group("answers")
    a.add_argument(
        "--answers",
        action="store_true",
        help="also run the answer LLM and check answer_contains and citations (slow; needs Ollama)",
    )
    a.add_argument("--answer-length", choices=sorted(ANSWER_LENGTHS), default="short")
    o = p.add_argument_group("environment and output")
    o.add_argument(
        "--config", type=Path, default=None, help="config file (env APP_CONFIG_FILE, default config/local.config.json)"
    )
    o.add_argument("--out", type=Path, default=None, help="results folder (default data/eval/results/<timestamp>)")
    o.add_argument(
        "--keep-collection",
        action="store_true",
        help="don't drop the throwaway Qdrant collection (its name is printed)",
    )
    o.add_argument(
        "--reuse-collection",
        default=None,
        metavar="NAME",
        help="skip ingestion and query this kept collection (same --chunk-size)",
    )
    o.add_argument(
        "--render",
        type=Path,
        default=None,
        metavar="RESULTS_JSON",
        help="only re-render summary.md from a results.json (no models)",
    )
    return p


def make_settings(config: Path | None) -> Settings:
    """The app's settings with the model cache, Docling artifacts and Qdrant taken from the environment like the
    integration tests do (MODELS_ROOT, QDRANT_URL)."""
    models_root = Path(os.environ.get("MODELS_ROOT") or PROJECT_ROOT / "data/models")
    env = {**os.environ}
    env.setdefault("MODEL_CACHE__HF_HOME", str(models_root / "huggingface"))
    env.setdefault("INGESTION__ARTIFACTS_PATH", str(models_root / "docling"))
    if os.environ.get("QDRANT_URL"):
        env.setdefault("VECTOR_STORE__URL", os.environ["QDRANT_URL"])
    settings = load_settings(config, env)
    from ..offline import apply_runtime_env

    apply_runtime_env(settings)  # HF_HOME and, under strict_offline, HF_HUB_OFFLINE=1 as the app does at startup
    return settings


async def preflight(container: Container, options: EvalOptions) -> None:
    """Fail early, with the reason, when the models, Qdrant or the LLM aren't usable."""
    needed = {"embeddings", "reranker", "vector_store", "ingestion"} | ({"llm"} if options.answers else set())
    problems = []
    for h in await container.health():
        if h.capability in needed and h.status in (HealthStatus.DOWN, HealthStatus.DEGRADED):
            problems.append(f"{h.capability} ({h.provider}): {h.status.value}: {h.detail}")
    if problems:
        raise EvalSetupError("not ready:\n  - " + "\n  - ".join(problems))


async def amain(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.render:
        results = json.loads(args.render.read_text(encoding="utf-8"))
        from .report import render_markdown

        target = args.render.with_name("summary.md")
        target.write_text(render_markdown(results), encoding="utf-8")
        print(f"wrote {target}")
        return 0

    options = EvalOptions(
        chunk_sizes=args.chunk_size,
        overlap=args.overlap,
        prefetch_ks=args.prefetch_k,
        rerank_top_ns=args.rerank_top_n,
        fusions=args.fusion,
        threshold=args.threshold,
        variants=args.variants,
        answers=args.answers,
        answer_length=args.answer_length,
        categories=args.categories,
        languages=args.languages,
        ids=args.ids,
        limit=args.limit,
        keep_collection=args.keep_collection,
        reuse_collection=args.reuse_collection,
    )
    docs_dir = args.docs or Path(os.environ.get("EVAL_DOCS") or PROJECT_ROOT / "data/eval/docs")
    results_root = Path(os.environ.get("EVAL_RESULTS") or PROJECT_ROOT / "data/eval/results")
    out_dir = args.out or results_root / datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        manifest = load_manifest(args.manifest)
        questions = load_questions(args.questions)
        settings = make_settings(args.config)
        container = build_container(settings)
    except (ConfigError, ManifestError, QuestionFileError, OSError) as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 2
    try:
        await preflight(container, options)
        results, chunk_rows = await run_eval(
            settings, container_factory(container), manifest, questions, docs_dir, options
        )
    except EvalSetupError as e:
        print(f"setup problem: {e}", file=sys.stderr)
        return 2
    finally:
        await container.close()
    write_outputs(out_dir, results, chunk_rows)
    print(f"\nresults: {out_dir}/summary.md  ({out_dir}/results.json)")
    return 0


def main() -> None:
    code = asyncio.run(amain())
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("EVAL_FAST_EXIT", "1") == "1":
        # onnxruntime threads abort during interpreter teardown on macOS (docs/DESIGN.md §9.6): skip it.
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
