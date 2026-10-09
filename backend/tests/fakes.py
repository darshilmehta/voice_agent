"""In-memory stand-ins for the ingestion/retrieval/LLM/speech providers: no ML libraries, models, Qdrant or Ollama
needed."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from app.providers.ingestion import (
    Chunk,
    DocumentParser,
    IngestionError,
    ParsedDocument,
    ParsedTable,
    TableCell,
    chunk_id,
    detect_language,
)
from app.providers.llm import LLMClient, LLMError, LLMMessage
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
from app.providers.speech import SpeechRecognizer, SpeechSynthesizer, Transcript, VoiceActivityDetector
from app.providers.web_search import SearchResult, WebSearch, site_of
from app.services.language import message_language


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

    def __init__(self, results: Sequence[Sequence[SearchHit]] = (), *, search_points: bool = False) -> None:
        self.results = list(results)  # one list per hybrid_search call, in order
        self.search_points = search_points  # no preset results: return the stored points matching the filters
        self.searches: list[tuple[DenseSparse, RetrievalFilters, int | None]] = []
        self.points: dict[str, IndexedChunk] = {}
        self.deletes: list[tuple[str, list[str]]] = []
        self.project_deletes: list[str] = []
        self.count_override: int | None = None
        self.fail_with: Exception | None = None  # raised by search and delete calls

    async def upsert(self, chunks: Sequence[IndexedChunk]) -> None:
        for c in chunks:
            self.points[c.chunk.chunk_id] = c

    async def hybrid_search(
        self, query: DenseSparse, filters: RetrievalFilters, *, limit: int | None = None
    ) -> list[SearchHit]:
        self.searches.append((query, filters, limit))
        if self.fail_with is not None:
            raise self.fail_with
        if self.results:
            return list(self.results.pop(0))
        if not self.search_points:
            return []
        return [
            SearchHit(p.chunk, 1.0 / (i + 1), 0.5)
            for i, p in enumerate(self._matching(filters))
            if filters.document_ids is None or filters.document_ids
        ]

    def _matching(self, filters: RetrievalFilters) -> list[IndexedChunk]:
        return [
            p
            for p in self.points.values()
            if p.chunk.project_id == filters.project_id
            and (filters.document_ids is None or p.chunk.document_id in filters.document_ids)
        ]

    async def delete_document(self, document_id: str, *, keep: Iterable[str] = ()) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        kept = list(keep)
        self.deletes.append((document_id, kept))
        for cid in [c for c, p in self.points.items() if p.chunk.document_id == document_id and c not in kept]:
            del self.points[cid]

    async def delete_project(self, project_id: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.project_deletes.append(project_id)
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


class TextParser(DocumentParser):
    """Parses text uploads for real-ish pipeline tests: pages are separated by form feeds, chunks are paragraphs
    (blank-line separated) with their page, a paragraph of markdown table rows ("| a | b |") becomes a table.
    ``fail_with`` makes parse raise; ``gate`` (an Event) makes parse wait, to observe PROCESSING."""

    name = "fake_text"

    def __init__(self) -> None:
        self.fail_with: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.parsed_paths: list[Path] = []
        self.release_calls = 0
        self._pages: dict[str, list[list[str]]] = {}

    async def parse(self, path: Any) -> ParsedDocument:
        path = Path(path)
        self.parsed_paths.append(path)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_with is not None:
            raise self.fail_with
        if not path.is_file():
            raise IngestionError(f"{path.name}: file not found")
        text = path.read_text(encoding="utf-8")
        pages = [[b.strip() for b in page.split("\n\n") if b.strip()] for page in text.split("\f")]
        tables = []
        for number, blocks in enumerate(pages, start=1):
            for block in blocks:
                if block.startswith("|"):
                    rows = [[c.strip() for c in line.strip("|").split("|")] for line in block.splitlines()]
                    cells = [
                        TableCell(row=r, col=c, text=v, column_header=r == 0)
                        for r, row in enumerate(rows)
                        for c, v in enumerate(row)
                    ]
                    tables.append(
                        ParsedTable(
                            index=len(tables),
                            ref=f"#/tables/{len(tables)}",
                            page_start=number,
                            page_end=number,
                            bbox=(10.0, 20.0, 300.0, 120.0),
                            heading_path=["Tables"],
                            caption=None,
                            num_rows=len(rows),
                            num_cols=max(len(r) for r in rows),
                            markdown=block,
                            cells=cells,
                        )
                    )
        doc = ParsedDocument(
            source_name=path.name,
            format=path.suffix.lstrip("."),
            page_count=len(pages),
            pages=[],
            items=[],
            tables=tables,
            markdown=text,
            language=detect_language(text),
            parse_seconds=0.01,
        )
        self._pages[path.name] = pages
        doc._native = path.name
        return doc

    async def chunk(self, document: ParsedDocument, *, project_id: str, document_id: str, version: int) -> list[Chunk]:
        chunks = []
        for number, blocks in enumerate(self._pages.pop(document._native), start=1):
            for block in blocks:
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(document_id, version, index),
                        project_id=project_id,
                        document_id=document_id,
                        version=version,
                        chunk_index=index,
                        chunking_version="v1",
                        page_start=number,
                        page_end=number,
                        heading_path=["Body"],
                        content_type="table" if block.startswith("|") else "paragraph",
                        language=detect_language(block),
                        text=block,
                        token_count=len(block.split()),
                    )
                )
        return chunks

    def release(self) -> None:
        self.release_calls += 1


def keyword_scorer(query: str, passage: str) -> float:
    """Reranker stand-in: the share of the query's longer words found in the passage (0..1)."""
    words = {w.strip("?.,!").casefold() for w in query.split() if len(w.strip("?.,!")) > 3}
    if not words:
        return 0.0
    text = passage.casefold()
    return sum(1 for w in words if w in text) / len(words)


RouteScript = dict[str, Any] | str | Exception | None


class FakeLLM(LLMClient):
    """Streams a scripted reply in small pieces and records every call. ``reply`` may be a function of the messages;
    ``fail_with`` raises before (or, with ``fail_after`` pieces, during) the stream.

    JSON output (the router): ``route`` is the proposal to return — a dict, raw JSON text, an exception to raise, or
    a function of the messages giving one of those; None (the default) raises LLMError, so the turn falls back to a
    document question as in phase 1. ``json_delay`` makes the call slow (timeouts)."""

    name = "fake"

    def __init__(self, reply: str | Callable[[list[LLMMessage]], str] = "The answer [S1].") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None
        self.fail_after: int | None = None
        self.piece_chars = 6
        self.delay = 0.0  # seconds before each piece
        self.sent = 0  # pieces yielded by the last stream
        self.closed = False  # the last stream was closed before it finished (generation cancelled)
        self.hold_after: int | None = None  # streams started now stop after this many pieces until released
        self.released = False
        self.route: RouteScript | Callable[[list[LLMMessage]], RouteScript] = None
        self.json_calls: list[dict[str, Any]] = []
        self.json_delay = 0.0
        self.json_cancelled = 0  # router calls cancelled before they answered

    async def generate_json[M: BaseModel](
        self,
        messages: Sequence[LLMMessage],
        schema: type[M],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> M:
        self.json_calls.append({"messages": list(messages), "schema": schema, "model": model, "max_tokens": max_tokens})
        try:
            if self.json_delay:
                await asyncio.sleep(self.json_delay)
        except asyncio.CancelledError:
            self.json_cancelled += 1
            raise
        script = self.route(list(messages)) if callable(self.route) else self.route
        if script is None:
            raise LLMError("no route scripted")
        if isinstance(script, Exception):
            raise script
        content = script if isinstance(script, str) else json.dumps(script)
        try:
            return schema.model_validate_json(content)
        except ValidationError as e:
            raise LLMError(f"model output doesn't match {schema.__name__}: {e.errors()[:1]}") from e

    async def stream(  # type: ignore[override]
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self.calls.append(
            {"messages": list(messages), "model": model, "temperature": temperature, "max_tokens": max_tokens}
        )
        if self.fail_with is not None and self.fail_after is None:
            raise self.fail_with
        text = self.reply(list(messages)) if callable(self.reply) else self.reply
        self.sent, self.closed = 0, False
        hold = self.hold_after
        finished = False
        try:
            for i in range(0, len(text), self.piece_chars):
                if self.fail_after is not None and i // self.piece_chars >= self.fail_after:
                    raise self.fail_with or LLMError("stream broke")
                while hold is not None and i // self.piece_chars >= hold and not self.released:
                    await asyncio.sleep(0.005)  # the model is "still thinking" (until cancelled)
                if self.delay:
                    await asyncio.sleep(self.delay)
                self.sent += 1
                yield text[i : i + self.piece_chars]
            finished = True
        finally:
            self.closed = not finished


# ------------------------------------------------------------------ speech
#
# Test audio encodes what was "said" in its loudness: speech(ms, tone=30) is a constant 0.30 signal, silence is zeros.
# FakeVAD hears speech in any frame above 0.02; FakeSTT maps the utterance's peak level (in hundredths) to a script.

IN_RATE = 16_000
OUT_RATE = 24_000


def speech(ms: float, tone: int = 30) -> bytes:
    """PCM16 16 kHz "speech" at level tone/100 (FakeSTT's key for what was said)."""
    return np.full(int(IN_RATE * ms / 1000), round(tone / 100 * 32767), dtype="<i2").tobytes()


def silence(ms: float) -> bytes:
    return bytes(int(IN_RATE * ms / 1000) * 2)


class FakeVAD(VoiceActivityDetector):
    name = "fake"

    def __init__(self) -> None:
        self.fail_with: Exception | None = None
        self.streams = 0

    def open_stream(self) -> _FakeVADStream:
        self.streams += 1
        return _FakeVADStream(self)


class _FakeVADStream:
    def __init__(self, vad: FakeVAD) -> None:
        self.vad = vad

    async def __call__(self, frames: np.ndarray) -> list[float]:
        if self.vad.fail_with is not None:
            raise self.vad.fail_with
        return [0.9 if float(np.abs(f).max()) > 0.02 else 0.0 for f in frames]

    def reset(self) -> None:
        pass


class FakeSTT(SpeechRecognizer):
    """``scripts`` maps a tone to the text heard; the language is the only allowed one, else read from the script."""

    name = "fake"

    def __init__(self) -> None:
        self.scripts: dict[int, str | Callable[[float], str]] = {}  # a callable gets the audio's length in ms
        self.calls: list[dict[str, Any]] = []
        self.cancelled = 0  # transcriptions cancelled while running (stale speculative jobs)
        self.fail_with: Exception | None = None
        self.delay = 0.0

    async def transcribe(self, pcm16k: np.ndarray, languages: Sequence[str]) -> Transcript:  # type: ignore[override]
        tone = round(float(np.abs(pcm16k).max()) * 100) if pcm16k.size else 0
        ms = len(pcm16k) * 1000 / IN_RATE
        self.calls.append({"tone": tone, "ms": ms, "languages": list(languages)})
        if self.delay:
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if self.fail_with is not None:
            raise self.fail_with
        script = self.scripts.get(tone, "")
        text = script(ms) if callable(script) else script
        language = languages[0] if len(languages) == 1 else (message_language(text) or "en")
        return Transcript(text, language)  # type: ignore[arg-type]


class FakeTTS(SpeechSynthesizer):
    """100 ms of audio per word, so what was heard at a given played_ms is exact."""

    name = "fake"
    MS_PER_WORD = 100

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_with: Exception | None = None
        self.fail_from = 0  # with fail_with: the first call that fails (0 = every call)
        self.delay = 0.0

    @property
    def sample_rate(self) -> int:
        return OUT_RATE

    async def synthesize(self, text: str, language: str) -> bytes:  # type: ignore[override]
        self.calls.append((text, language))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_with is not None and len(self.calls) > self.fail_from:
            raise self.fail_with
        samples = OUT_RATE * self.MS_PER_WORD // 1000 * len(text.split())
        return np.full(samples, 1000, dtype="<i2").tobytes()


# ------------------------------------------------------------------ live data


def web_result(n: int, *, site: str = "news.example.com", snippet: str | None = None, **over: Any) -> SearchResult:
    url = over.pop("url", f"https://{site}/story-{n}")
    fields: dict[str, Any] = {
        "url": url,
        "title": f"Story {n}",
        "snippet": snippet if snippet is not None else f"live fact {n}",
        "site": site_of(url),
        "engine": "fake",
        "published": "2026-10-09T08:00:00",
    }
    fields.update(over)
    return SearchResult(**fields)


class FakeWebSearch(WebSearch):
    """A scripted live-data search. ``script``: (seconds after the previous one, result), yielded in order; then
    ``fail_with`` is raised, or the stream stays open while ``hang`` (until cancelled or the deadline). ``reason``:
    what ``unavailable_reason`` says. Records every query, and how many streams were cancelled or are still open."""

    name = "fake"

    def __init__(self, script: Sequence[tuple[float, SearchResult]] = ()) -> None:  # no config/context needed
        self.script = list(script)
        self.queries: list[str] = []
        self.started_at: list[float] = []  # loop time of each stream's start
        self.fail_with: Exception | None = None
        self.hang = False
        self.reason: str | None = None
        self.cancelled = 0
        self.open = 0
        self.pages: dict[str, str] = {}
        self.page_delay = 0.0
        self.fetched: list[str] = []

    def unavailable_reason(self) -> str | None:
        return self.reason

    async def stream(  # type: ignore[override]
        self, query: str, max_results: int | None = None
    ) -> AsyncIterator[SearchResult]:
        self.queries.append(query)
        self.started_at.append(asyncio.get_running_loop().time())
        self.open += 1
        try:
            for delay, result in self.script[: max_results or None]:
                await asyncio.sleep(delay)
                yield result
            if self.fail_with is not None:
                raise self.fail_with
            if self.hang:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.open -= 1

    async def fetch_page(self, url: str) -> str | None:
        self.fetched.append(url)
        await asyncio.sleep(self.page_delay)
        return self.pages.get(url)
