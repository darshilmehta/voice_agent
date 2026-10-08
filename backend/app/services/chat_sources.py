"""The sources cited anywhere in a chat, numbered chat-wide (used by summaries and transcript export, §3.9).

Every agent answer numbers its own sources ``[S1]``, ``[S2]`` … and the numbers restart in the next answer, so a
chat-wide view needs its own: one number per distinct (document, page range), in order of first citation. Answers'
markers are rewritten to those numbers (``[S2]`` → ``[3]``), which are then what the summary prompt, the summary's
citations and the exported transcript's sources list all refer to.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..domain.projects import Citation
from ..domain.summaries import SourceRef
from ..settings import Language

# [S1], [s2], [S1, S3], [S1; S2]: the forms answers use (sources.finalize_answer writes "[S1][S2]").
_MARKER = re.compile(r"(\s*)\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_ID = re.compile(r"S(\d+)", re.IGNORECASE)
EN_DASH = "\u2013"
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

    def ref(self) -> SourceRef:
        return SourceRef(
            document_id=self.document_id, filename=self.filename, page_start=self.page_start, page_end=self.page_end
        )


def ref_key(ref: SourceRef) -> tuple[str, str, int | None, int | None]:
    return ref.document_id, ref.filename, ref.page_start, ref.page_end


class ChatSources:
    """Chat-wide numbering of cited sources: ``register`` each agent message's citations in transcript order."""

    def __init__(self) -> None:
        self.entries: list[SourceEntry] = []
        self._by_key: dict[tuple[str, str, int | None, int | None], SourceEntry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, ref: SourceRef, *, snippet: str = "", cited_by: int | None = None) -> SourceEntry:
        entry = self._by_key.get(ref_key(ref))
        if entry is None:
            entry = SourceEntry(
                len(self.entries) + 1, ref.document_id, ref.filename, ref.page_start, ref.page_end, snippet
            )
            self.entries.append(entry)
            self._by_key[ref_key(ref)] = entry
        if cited_by is not None and cited_by not in entry.cited_by:
            entry.cited_by.append(cited_by)
        return entry

    def register(self, seq: int, citations: Iterable[Citation]) -> dict[str, int]:
        """Number one message's citations; returns {"S1": chat-wide number, …} for rewriting its text."""
        mapping: dict[str, int] = {}
        for c in citations:
            ref = SourceRef(
                document_id=c.document_id, filename=c.filename, page_start=c.page_start, page_end=c.page_end
            )
            mapping[c.source_id.upper()] = self.add(ref, snippet=c.snippet, cited_by=seq).number
        return mapping

    def find(self, ref: SourceRef) -> SourceEntry | None:
        return self._by_key.get(ref_key(ref))

    def get(self, number: int) -> SourceEntry | None:
        return self.entries[number - 1] if 1 <= number <= len(self.entries) else None


def number_markers(text: str, mapping: Mapping[str, int]) -> str:
    """``text`` with its ``[S#]`` markers replaced by chat-wide numbers (``[S1][S2]`` → ``[3][4]``). Markers of
    sources the message doesn't list are dropped, as is a repeated number ("[3][3]" → "[3]")."""

    def replace(m: re.Match[str]) -> str:
        numbers = [mapping.get(f"S{int(n)}") for n in _ID.findall(m.group(2))]
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
    """ "annual_report.pdf, p. 2"."""
    pages = pages_label(ref.page_start, ref.page_end, language, short=short)
    name = ref.filename or "(unknown document)"
    return f"{name}, {pages}" if pages else name
