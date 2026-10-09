"""Numbered sources for a grounded answer and validation of the answer's citations (docs/DESIGN.md §3.2, §3.7).

reranked chunks → only READY documents of the chat → dedupe → token budget (best first)
  → grouped by document section (reading order inside a group) → [S1] … [Sn]
live web search results (services/web_search.py) → [W1] … [Wn], in arrival order
answer text → [S#] / [W#] markers checked against the sources → unknown ids removed, "[S1, W2]" → "[S1][W2]"
  → citations = the sources actually cited, in order of first mention
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..domain.projects import SNIPPET_CHARS, Citation
from ..providers.ingestion import Chunk
from .retrieval import RankedChunk

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_SPACE = re.compile(r"\s+")
# [S1], [s2], [S1, S3], [S1; W2], [W1] — with the whitespace before it, so a removed marker leaves no gap.
_MARKER = re.compile(r"(\s*)\[\s*([SW]\d+(?:\s*[,;]\s*[SW]\d+)*)\s*\]", re.IGNORECASE)
_ID = re.compile(r"([SW])(\d+)", re.IGNORECASE)


class Citable(Protocol):
    """A numbered source an answer may cite: a document passage (``Source``) or a web result (``WebSource``)."""

    @property
    def source_id(self) -> str: ...

    def citation(self) -> Citation: ...


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str  # "S1"
    chunk: Chunk
    filename: str
    rerank_score: float

    @property
    def pages(self) -> str:
        start, end = self.chunk.page_start, self.chunk.page_end
        if start is None:
            return ""
        return f"page {start}" if end in (None, start) else f"pages {start}-{end}"

    def header(self) -> str:
        parts = [f"[{self.source_id}] {self.filename}", self.pages, " > ".join(self.chunk.heading_path)]
        return " · ".join(p for p in parts if p)

    def citation(self) -> Citation:
        return Citation(
            source_id=self.source_id,
            document_id=self.chunk.document_id,
            filename=self.filename,
            page_start=self.chunk.page_start,
            page_end=self.chunk.page_end,
            chunk_id=self.chunk.chunk_id,
            snippet=snippet(self.chunk.text),
        )


def estimate_tokens(text: str) -> int:
    """LLM tokens, estimated without a tokenizer: ~4 characters per token for Latin script, ~0.9 tokens per
    Devanagari character (measured for Qwen3, §9.1). Errs on the high side."""
    devanagari = len(_DEVANAGARI.findall(text))
    return math.ceil((len(text) - devanagari) / 3.5 + devanagari * 0.9)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Up to ``limit`` characters of ``text`` on one line, cut at a word boundary with an ellipsis."""
    flat = _SPACE.sub(" ", text).strip()
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:") + "…"


def _truncate_to_tokens(text: str, budget: int) -> str:
    if estimate_tokens(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # longest prefix within the budget
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + " …"


def build_sources(ranked: Sequence[RankedChunk], filenames: Mapping[str, str], *, budget_tokens: int) -> list[Source]:
    """Sources for the prompt from reranked chunks (best first).

    Chunks of documents not in ``filenames`` (not READY, deleted) are dropped, as are repeats (same chunk, or the
    same text). Chunks are taken best first while their headers and text fit ``budget_tokens``; the best one is
    always kept (cut to the budget if it alone is too long). The kept chunks are then grouped by document section,
    groups ordered by their best chunk and chunks in reading order inside a group, and numbered S1, S2, …
    """
    kept: list[tuple[int, RankedChunk, Chunk]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    used = 0
    for rank, r in enumerate(ranked):
        chunk = r.chunk
        if chunk.document_id not in filenames or chunk.chunk_id in seen_ids:
            continue
        key = _SPACE.sub(" ", chunk.text).strip().casefold()
        if key in seen_texts:
            continue
        cost = estimate_tokens(chunk.text) + 24  # header line
        if used + cost > budget_tokens:
            if kept:
                continue  # a smaller, lower-ranked chunk may still fit
            chunk = chunk.model_copy(update={"text": _truncate_to_tokens(chunk.text, max(budget_tokens - 24, 1))})
            cost = budget_tokens
        seen_ids.add(r.chunk.chunk_id)
        seen_texts.add(key)
        kept.append((rank, r, chunk))
        used += cost

    groups: dict[tuple[str, tuple[str, ...]], list[tuple[int, RankedChunk, Chunk]]] = {}
    for item in kept:
        chunk = item[2]
        groups.setdefault((chunk.document_id, tuple(chunk.heading_path)), []).append(item)
    ordered = sorted(groups.values(), key=lambda g: min(rank for rank, _, _ in g))
    sources: list[Source] = []
    for group in ordered:
        for _, r, chunk in sorted(group, key=lambda item: (item[2].chunk_index, item[0])):
            sources.append(Source(f"S{len(sources) + 1}", chunk, filenames[chunk.document_id], r.rerank_score))
    return sources


def format_sources(sources: Sequence[Source]) -> str:
    return "\n\n".join(f"{s.header()}\n{s.chunk.text.strip()}" for s in sources)


_OPEN_MARKER = re.compile(r"\s*\[[^\]]{0,24}$")


def trim_open_marker(text: str) -> str:
    """A stopped answer without a citation marker cut in half at its end ("… 18.2% [S")."""
    return _OPEN_MARKER.sub("", text)


def strip_markers(text: str) -> str:
    """Text without [S#] / [W#] markers (earlier answers in the prompt history: their numbers meant other
    sources)."""
    return _MARKER.sub("", text).strip()


def finalize_answer(text: str, sources: Sequence[Citable]) -> tuple[str, list[Citation]]:
    """The answer as it is saved, and the citations it makes.

    Markers naming a source that doesn't exist are removed (with the space before them); lists are normalized to
    one marker per source ("[S1, W2]" → "[S1][W2]"), which is the form the UI turns into chips. Citations are the
    existing sources the text cites (documents and web results alike), in order of first mention.
    """
    by_id = {s.source_id: s for s in sources}
    cited: list[str] = []

    def replace(m: re.Match[str]) -> str:
        ids = [f"{kind.upper()}{int(n)}" for kind, n in _ID.findall(m.group(2))]
        valid = [i for i in dict.fromkeys(ids) if i in by_id]
        for i in valid:
            if i not in cited:
                cited.append(i)
        return m.group(1) + "".join(f"[{i}]" for i in valid) if valid else ""

    clean = _MARKER.sub(replace, text).strip()
    return clean, [by_id[i].citation() for i in cited]
