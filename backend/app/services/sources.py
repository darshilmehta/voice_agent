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

from ..domain.projects import SECTION_SEPARATOR, SNIPPET_CHARS, Citation
from ..providers.ingestion import Chunk
from .retrieval import RankedChunk

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_SPACE = re.compile(r"\s+")
# [S1], [s2], [S1, S3], [S1; W2], [W1] — with the whitespace before it, so a removed marker leaves no gap.
_MARKER = re.compile(r"(\s*)\[\s*([SW]\d+(?:\s*[,;]\s*[SW]\d+)*)\s*\]", re.IGNORECASE)
_ID = re.compile(r"([SW])(\d+)", re.IGNORECASE)


def section_of(heading_path: Sequence[str]) -> str | None:
    """A chunk's heading path as one line ("4. Travel > 4.2 Domestic > 4.2.1 Hotels"), None without headings. Where a
    passage has no page (DOCX, MD, TXT) this is how a citation says where in the document it is."""
    parts = [_SPACE.sub(" ", h).strip() for h in heading_path]
    return SECTION_SEPARATOR.join(p for p in parts if p) or None


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
            section=section_of(self.chunk.heading_path),
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


def build_sources(
    ranked: Sequence[RankedChunk],
    filenames: Mapping[str, str],
    *,
    budget_tokens: int,
    max_sources: int | None = None,
    best_budget_tokens: int | None = None,
) -> list[Source]:
    """Sources for the prompt from reranked chunks (best first).

    Chunks of documents not in ``filenames`` (not READY, deleted) are dropped, as are repeats (same chunk, or the
    same text). Chunks are taken best first while their headers and text fit ``budget_tokens`` (and, with
    ``max_sources``, until there are that many); the best one is always kept, whole up to ``best_budget_tokens``
    (default: the budget) and cut to that beyond it: a short answer's budget is for the passages besides the best
    one, never a reason to cut the table that holds the answer. The kept chunks are then grouped by document
    section, groups ordered by their best chunk and chunks in reading order inside a group, and numbered S1, S2, …
    """
    kept: list[tuple[int, RankedChunk, Chunk]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    used = 0
    for rank, r in enumerate(ranked):
        if max_sources is not None and len(kept) >= max_sources:
            break
        chunk = r.chunk
        if chunk.document_id not in filenames or chunk.chunk_id in seen_ids:
            continue
        key = _SPACE.sub(" ", chunk.text).strip().casefold()
        if key in seen_texts:
            continue
        cost = estimate_tokens(chunk.text) + 24  # header line
        limit = budget_tokens if kept else max(budget_tokens, best_budget_tokens or 0)
        if used + cost > limit:
            if kept:
                continue  # a smaller, lower-ranked chunk may still fit
            chunk = chunk.model_copy(update={"text": _truncate_to_tokens(chunk.text, max(limit - 24, 1))})
            cost = limit
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
    return "\n\n".join(f"{s.header()}\n{compact_tables(s.chunk.text.strip())}" for s in sources)


_TABLE_RULE = re.compile(r":?-{3,}:?")


def compact_tables(text: str) -> str:
    """Markdown table rows without their padding ("| Revenue      | 7,365  |" → "| Revenue | 7,365 |", rules →
    "|---|"): the same table in ~12% fewer prompt tokens (measured on the eval corpus' 25 tables), and every token of
    evidence costs ~3 ms of the voice answer's first word (§9.5). Other lines are left as they are."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            if all(_TABLE_RULE.fullmatch(c) for c in cells if c) and any(cells):
                line = "|" + "|".join("---" for _ in cells) + "|"
            else:
                line = "| " + " | ".join(cells) + " |"
        lines.append(line)
    return "\n".join(lines)


# An answer that says the documents don't have it: "The documents do not specify Valmora's EBITDA margin for FY25.",
# "The report doesn't provide…", "The policy does not cover hotel provisions", "[S2] … but does not give the FY23
# breakdown", "not mentioned in the sources", "are not available in the provided sources", "is not covered by the
# policy", "इस बारे में जानकारी नहीं दी गई है".
_DOCUMENT_WORDS = (
    r"documents?|sources?|report|deck|minutes|filing|file|policy|policies|manual|handbook|contract|agreement|"
    r"presentation|slides?|statements?|text|passages?|tables?|provided\s+(?:sources|information|text|documents)"
)
# Between the document and its "does not …": nothing that makes it report a fact ("the report shows revenue does not
# include other income" says what the report says).
_REPORTING = r"(?!\b(?:shows?|says?|states?|notes?|reports?|explains?|confirms?|clarifies|that)\b)"
_HI_NO = "नही[\u0901\u0902]?"  # "नहीं", "नही" (written without the dot) and "नहीँ"
_NOT_COVERED = re.compile(
    rf"(?:\b(?:{_DOCUMENT_WORDS})\b|\[\s*[SW]\d+\s*\])"
    rf"(?:{_REPORTING}[^.!?।]){{0,80}}?\b(?:do(?:es)?\s*n[o'\u2019]?t|did\s*n[o'\u2019]?t|cannot|can't)\s+(?:\w+\s+)?(?:cover|mention|"
    r"provide|specify|include|contain|state|say|give|list|disclose|show|have|offer|address|detail|break)\w*\b"
    r"|\bnot\s+(?:explicitly\s+|specifically\s+|directly\s+)?(?:mentioned|specified|provided|available|covered|"
    r"stated|given|disclosed|found|listed|included|addressed|reported|shown|detailed)\s+(?:anywhere\s+)?"
    r"(?:in|by)\s+(?:the|these|this|any|those)\b"
    r"|\bno\s+(?:specific\s+|such\s+|further\s+)?(?:information|mention|data|details|figures?|provisions?|breakdown)"
    r"\s+(?:about|on|of|regarding|for|in\s+the\s+(?:documents?|sources?|report|policy))\b"
    r"|\b(?:i\s+)?(?:couldn['\u2019]t|could\s+not|can['\u2019]t|cannot)\s+find\b"
    rf"|(?:जानकारी|उल्लेख|उपलब्ध|मौजूद|दर्ज|पता|विवरण|ब्योरा|ब्यौरा|ज\u093c?िक्र)\s+{_HI_NO}"
    rf"|{_HI_NO}\s+(?:दी|दिया|दिए|दिये)\s+(?:गई|गयी|गया|गए|गये)|{_HI_NO}\s+(?:बताया|मिल)"
    r"|\b(?:jaa?nkari|ullekh|uplabdh|maujood|pata)\s+nah(?:i|in|ee)\b",  # romanized Hindi ("uplabdh nahi hai")
    re.IGNORECASE,
)


def says_not_covered(text: str) -> bool:
    """The text says the documents don't have the answer (a model's own non-answer, or a summary point about one)."""
    return bool(_NOT_COVERED.search(text))


def not_covered_at(text: str) -> int | None:
    """Where the text says the documents don't have something: the last character of the claim (its verb, "…doesn't
    give"), so the clause it is in is the denying one even when the claim begins in an earlier clause ("[S2], but
    the deck doesn't give FY23"), or None."""
    m = _NOT_COVERED.search(text)
    return m.end() - 1 if m else None


# Sentence ends: punctuation followed by whitespace, or a line break. "Rs. 5,000", "U.S." and "18.2%" end none.
_SENTENCE_CUT = re.compile("(?<=[.!?।॥])[\"'\u201d\u2019)\\]]*\\s+|\\n+")
_ABBREVIATED = re.compile(
    r"(?:^|\b)(?:rs|mr|mrs|ms|dr|vs|e\.g|i\.e|etc|approx|inc|ltd|co|st|fig|no)\.$|(?:[a-z]\.){2,}$|^\(?\d{1,2}\.$",
    re.IGNORECASE,
)
# Where a clause starts inside a sentence ("…, but the report doesn't give FY23").
_CLAUSE = re.compile(
    r",\s+|;\s+|\s+[\u2014\u2013-]\s+|\s+(?=(?:but|however|although|though|while|whereas)\b)"
    r"|\s+(?=(?:लेकिन|परंतु|परन्तु|किंतु|किन्तु|मगर|जबकि|हालांकि|हालाँकि)\s)",  # Hindi (no \b: matras aren't \w)
    re.I,
)


def split_sentences(text: str) -> list[str]:
    """The text's sentences (and lines), in order, without the whitespace between them."""
    out: list[str] = []
    start = 0
    for m in _SENTENCE_CUT.finditer(text):
        head = text[start : m.start()]
        words = head.split()
        if "\n" not in m.group(0) and words and _ABBREVIATED.search(words[-1]):
            continue  # "Rs. 4,210", "U.S. sales", a list item's "1."
        if head.strip():
            out.append(head.strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def clauses(sentence: str) -> list[str]:
    """A sentence cut at its clause boundaries (commas, semicolons, dashes, "but", "however"…)."""
    return [c.strip() for c in _CLAUSE.split(sentence) if c.strip()]


def answer_declines(answer: str, questions: Sequence[str | None]) -> bool:
    """The answer says the documents don't cover what was asked (B9, quality round): an abstention, recorded as such,
    however many sources it cites for what it says around it.

    - A question that names fiscal periods ("revenue in FY25"): every period it names is only in clauses that say the
      documents don't have it ("FY24 revenue was ₹7,365 crore [S1], but the documents don't give FY25" declines a
      question about FY25; for a question about FY24 and FY25 it is a partial answer, no abstention).
    - Otherwise: the answer leads with it (its first clause says the documents don't cover it)."""
    from .retrieval import asked_periods  # (retrieval imports this module)

    sentences = [s for s in split_sentences(strip_markers(answer)) if len(s.split()) >= 2]
    if not any(says_not_covered(s) for s in sentences):
        return False
    asked: set[int] = set()
    for q in questions:
        if q:
            asked |= asked_periods(q)
    parts = [c for s in sentences for c in denial_clauses(s)]
    if asked:
        denied: set[int] = set().union(*(asked_periods(c) for c, no in parts if no))
        answered: set[int] = set().union(*(asked_periods(c) for c, no in parts if not no))
        return asked <= denied and not (asked & answered)
    return parts[0][1]


def denial_clauses(sentence: str) -> list[tuple[str, bool]]:
    """The sentence's clauses, each with whether it says the documents don't have something. A claim that spans
    clauses ("the report, as filed, does not…") marks the clause it starts in."""
    out = [(c, says_not_covered(c)) for c in clauses(sentence)]
    at = not_covered_at(sentence)
    if at is not None and not any(no for _, no in out):
        pos = 0
        for i, (c, _) in enumerate(out):
            pos = sentence.find(c, pos)
            if pos <= at < pos + len(c) or i == len(out) - 1:
                out[i] = (c, True)
                break
            pos += len(c)
    return out


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
