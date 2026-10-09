"""Planting facts: every fact is registered at the moment its text is put on a page, with an assertion that the
evidence strings really are in that text. ``build_corpus.py`` later checks the rendered files too (pypdf,
python-pptx, python-docx), so a wrong page can't survive either way."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from app.evals.manifest import DocumentEntry, Fact, Manifest
from app.evals.textnorm import contains_all

_TAGS = re.compile(r"<[^>]+>")


def plain(markup: str) -> str:
    """Text of a reportlab-style markup string (tags dropped, entities decoded)."""
    return html.unescape(_TAGS.sub("", markup))


@dataclass(frozen=True)
class FactSpec:
    """A fact to plant: ``evidence`` strings must all appear in the text it is attached to."""

    id: str
    evidence: tuple[str, ...]
    statement: str
    kind: str = "number"
    ocr: bool = False


def F(id: str, evidence: list[str], statement: str, kind: str = "number", *, ocr: bool = False) -> FactSpec:
    return FactSpec(id, tuple(evidence), statement, kind, ocr)


@dataclass
class _Planted:
    spec: FactSpec
    document: str
    pages: list[int] = field(default_factory=list)
    locator: str | None = None


class Registry:
    """All planted facts of all documents."""

    def __init__(self) -> None:
        self._facts: dict[str, _Planted] = {}
        self.documents: list[DocumentEntry] = []

    def add_document(self, entry: DocumentEntry) -> None:
        self.documents.append(entry)

    def plant(
        self, spec: FactSpec, *, document: str, text: str, page: int | None = None, locator: str | None = None
    ) -> None:
        """Register ``spec`` as stated in ``text`` (plain text of the element) on ``page`` / under ``locator``."""
        if not contains_all(text, spec.evidence, approximate=spec.ocr):
            raise ValueError(
                f"fact {spec.id}: evidence {list(spec.evidence)} is not in the text it is planted in: {text!r}"
            )
        planted = self._facts.get(spec.id)
        if planted is None:
            planted = self._facts[spec.id] = _Planted(spec, document, locator=locator)
        elif planted.spec.evidence != spec.evidence or planted.document != document:
            raise ValueError(f"fact {spec.id} planted twice with different evidence or document")
        if page is not None and page not in planted.pages:
            planted.pages.append(page)
        if locator and planted.locator is None:
            planted.locator = locator

    def manifest(self) -> Manifest:
        """The planted manifest: planted pages only, to be finalized against the rendered files."""
        facts = [
            Fact(
                id=p.spec.id,
                document=p.document,
                pages=[],
                planted_pages=sorted(p.pages),
                locator=p.locator,
                evidence=list(p.spec.evidence),
                statement=p.spec.statement,
                kind=p.spec.kind,  # type: ignore[arg-type]
                ocr=p.spec.ocr,
            )
            for p in sorted(self._facts.values(), key=lambda p: (p.document, p.pages, p.spec.id))
        ]
        return Manifest(documents=self.documents, facts=facts)
