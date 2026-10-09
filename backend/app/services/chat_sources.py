"""The sources cited anywhere in a chat, numbered chat-wide (used by summaries and transcript export, §3.9).

Every agent answer numbers its own sources ``[S1]``, ``[S2]`` … and the numbers restart in the next answer, so a
chat-wide view needs its own: one number per distinct (document, page range), in order of first citation. Answers'
markers are rewritten to those numbers (``[S2]`` → ``[3]``), which are then what the summary prompt, the summary's
citations and the exported transcript's sources list all refer to.

Live web results an answer cited (``[W1]``, ``kind: "web"``, §3.7) are numbered the same way, one number per URL,
and keep their URL and title; as a ``SourceRef`` they are the site with no pages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from ..domain.projects import Citation
from ..domain.summaries import SourceRef
from ..settings import Language

# [S1], [s2], [S1, S3], [S1; W2]: the forms answers use (sources.finalize_answer writes "[S1][W2]").
_MARKER = re.compile(r"(\s*)\[\s*([SW]\d+(?:\s*[,;]\s*[SW]\d+)*)\s*\]", re.IGNORECASE)
_ID = re.compile(r"([SW])(\d+)", re.IGNORECASE)
EN_DASH = "\u2013"
EM_DASH = "\u2014"
_REPEAT = re.compile(r"(\[(\d+)\])(?:\[\2\])+")  # [3][3] → [3]


@dataclass(slots=True)
class SourceEntry:
    number: int
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    snippet: str  # of the first chunk cited for this page range
    cited_by: list[int] = field(default_factory=list)  # seq of the messages that cite it
    kind: Literal["document", "web"] = "document"
    url: str | None = None  # web results
    title: str | None = None  # web results

    def ref(self) -> SourceRef:
        web = {"kind": "web", "url": self.url, "title": self.title} if self.kind == "web" else {}
        return SourceRef(
            document_id=self.document_id,
            filename=self.filename,
            page_start=self.page_start,
            page_end=self.page_end,
            **web,
        )


RefKey = tuple[str, str, int | None, int | None, str | None]


def ref_key(ref: SourceRef) -> RefKey:
    """Documents by document and pages, web results by URL (two results from one site are two sources)."""
    return ref.document_id, ref.filename, ref.page_start, ref.page_end, ref.url if ref.kind == "web" else None


class ChatSources:
    """Chat-wide numbering of cited sources: ``register`` each agent message's citations in transcript order."""

    def __init__(self) -> None:
        self.entries: list[SourceEntry] = []
        self._by_key: dict[RefKey, SourceEntry] = {}
        self._by_url: dict[str, SourceEntry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, ref: SourceRef, *, snippet: str = "", cited_by: int | None = None) -> SourceEntry:
        entry = self._by_key.get(ref_key(ref))
        if entry is None and ref.kind == "web" and ref.url:
            entry = self._by_url.get(ref.url)
        if entry is None:
            entry = SourceEntry(
                len(self.entries) + 1,
                ref.document_id,
                ref.filename,
                ref.page_start,
                ref.page_end,
                snippet,
                kind=ref.kind,
                url=ref.url,
                title=ref.title,
            )
            self.entries.append(entry)
            self._by_key[ref_key(ref)] = entry
            if ref.kind == "web" and ref.url:
                self._by_url[ref.url] = entry
        if cited_by is not None and cited_by not in entry.cited_by:
            entry.cited_by.append(cited_by)
        return entry

    def register(self, seq: int, citations: Iterable[Citation]) -> dict[str, int]:
        """Number one message's citations; returns {"S1": chat-wide number, …} for rewriting its text."""
        mapping: dict[str, int] = {}
        for c in citations:
            if c.kind == "web":
                mapping[c.source_id.upper()] = self.add_web(c, cited_by=seq).number
                continue
            ref = SourceRef(
                document_id=c.document_id, filename=c.filename, page_start=c.page_start, page_end=c.page_end
            )
            mapping[c.source_id.upper()] = self.add(ref, snippet=c.snippet, cited_by=seq).number
        return mapping

    def add_web(self, c: Citation, *, cited_by: int | None = None) -> SourceEntry:
        """A web result, one entry per URL."""
        key = c.url or c.filename
        entry = self._by_url.get(key)
        if entry is None:
            site = c.site or c.filename
            entry = SourceEntry(
                len(self.entries) + 1, "", site, None, None, c.snippet, kind="web", url=c.url, title=c.title
            )
            self.entries.append(entry)
            self._by_url[key] = entry
            self._by_key[ref_key(entry.ref())] = entry  # a summary names it by its URL
        if cited_by is not None and cited_by not in entry.cited_by:
            entry.cited_by.append(cited_by)
        return entry

    def find(self, ref: SourceRef) -> SourceEntry | None:
        return self._by_key.get(ref_key(ref))

    def get(self, number: int) -> SourceEntry | None:
        return self.entries[number - 1] if 1 <= number <= len(self.entries) else None


def number_markers(text: str, mapping: Mapping[str, int]) -> str:
    """``text`` with its ``[S#]`` markers replaced by chat-wide numbers (``[S1][S2]`` → ``[3][4]``). Markers of
    sources the message doesn't list are dropped, as is a repeated number ("[3][3]" → "[3]")."""

    def replace(m: re.Match[str]) -> str:
        numbers = [mapping.get(f"{kind.upper()}{int(n)}") for kind, n in _ID.findall(m.group(2))]
        marks = "".join(f"[{n}]" for n in dict.fromkeys(n for n in numbers if n is not None))
        return m.group(1) + marks if marks else ""

    return _REPEAT.sub(r"\1", _MARKER.sub(replace, text)).strip()


def pages_label(start: int | None, end: int | None, language: Language = "en", *, short: bool = False) -> str:
    """ "page 2", "pages 2-3" ("p. 2", "pp. 2-3" when short), with an en dash; Hindi "पृष्ठ 2". Empty without a page."""
    if start is None:
        return ""
    span = f"{start}" if end in (None, start) else f"{start}{EN_DASH}{end}"
    if language == "hi":
        return f"पृ. {span}" if short else f"पृष्ठ {span}"
    plural = end not in (None, start)
    if short:
        return f"{'pp.' if plural else 'p.'} {span}"
    return f"{'pages' if plural else 'page'} {span}"


def ref_label(ref: SourceRef, language: Language = "en", *, short: bool = True) -> str:
    """ "annual_report.pdf, p. 2"; a web result "web: livemint.com — Infosys share price"."""
    if ref.kind == "web":
        title = " ".join((ref.title or "").split())
        return f"web: {ref.filename}" + (f" {EM_DASH} {title}" if title else "")
    pages = pages_label(ref.page_start, ref.page_end, language, short=short)
    name = ref.filename or "(unknown document)"
    return f"{name}, {pages}" if pages else name
