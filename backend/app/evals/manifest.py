"""The eval corpus manifest: every document and every planted fact with its real page (or slide).

``scripts/eval/build_corpus.py`` generates the documents deterministically, plants facts while it lays pages out,
then renders text back out of the finished files (pypdf for PDFs, python-pptx for slides, python-docx for DOCX)
and calls ``finalize_manifest``: a fact's pages are the pages whose rendered text actually contains its evidence,
and the build fails loudly when a planted page doesn't, when evidence is nowhere, or when a page count is off.
The result is committed as ``evals/retrieval/manifest.json`` (text, not a binary); the question set refers to
facts by id and CI checks every question against it.

Page-less formats (DOCX: Docling reads no pages from it) locate facts by section heading instead
(``locator``); the harness then matches retrieved chunks on their text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .textnorm import contains_all, normalize

MANIFEST_VERSION = 1

DocFormat = Literal["pdf", "pptx", "docx"]
DocLanguage = Literal["en", "hi"]
FactKind = Literal["number", "identifier", "table_cell", "definition", "clause", "date", "text", "footnote"]


class ManifestError(RuntimeError):
    """The manifest and the rendered documents disagree (or the manifest is malformed). ``problems`` lists all."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        shown = "\n  - ".join(self.problems[:60])
        more = f"\n  ... and {len(self.problems) - 60} more" if len(self.problems) > 60 else ""
        super().__init__(f"{len(self.problems)} problem(s):\n  - {shown}{more}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentEntry(_Model):
    name: str  # file name inside the corpus directory
    title: str
    format: DocFormat
    language: DocLanguage
    pages: int | None = None  # pages (PDF) or slides (PPTX); None for DOCX
    scanned: bool = False  # image-only pages: only OCR can read them
    company: str | None = None  # the (fictional) organisation the document is about


class Fact(_Model):
    """A planted fact. ``evidence`` are verbatim strings that must all appear together on a page (or in a section)."""

    id: str
    document: str
    pages: list[int] = Field(default_factory=list)  # every page/slide whose text contains the evidence
    planted_pages: list[int] = Field(default_factory=list)  # where the generator put it (always within ``pages``)
    locator: str | None = None  # section heading, for page-less formats
    evidence: list[str]
    statement: str  # the fact in plain words
    kind: FactKind
    ocr: bool = False  # only readable through OCR: fact-level matching of chunk text is approximate


class Manifest(_Model):
    version: int = MANIFEST_VERSION
    documents: list[DocumentEntry]
    facts: list[Fact]

    def document(self, name: str) -> DocumentEntry:
        for d in self.documents:
            if d.name == name:
                return d
        raise KeyError(name)

    def fact(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(fact_id)

    def facts_of(self, document: str) -> list[Fact]:
        return [f for f in self.facts if f.document == document]

    def problems(self, *, finalized: bool = True) -> list[str]:
        """Structural problems (does not look at any rendered file). ``finalized=False`` accepts facts that don't
        have their real pages yet (the planted manifest, before ``finalize_manifest``)."""
        out: list[str] = []
        names = [d.name for d in self.documents]
        out += [f"duplicate document {n!r}" for n in sorted({n for n in names if names.count(n) > 1})]
        ids = [f.id for f in self.facts]
        out += [f"duplicate fact id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
        docs = {d.name: d for d in self.documents}
        for f in self.facts:
            doc = docs.get(f.document)
            if doc is None:
                out.append(f"fact {f.id}: unknown document {f.document!r}")
                continue
            if not f.evidence or any(not normalize(e) for e in f.evidence):
                out.append(f"fact {f.id}: needs non-empty evidence strings")
            if doc.pages is None:
                if f.pages or f.planted_pages:
                    out.append(f"fact {f.id}: {doc.name} has no pages but the fact lists pages {f.pages}")
                if not f.locator:
                    out.append(f"fact {f.id}: a fact in the page-less {doc.name} needs a locator (section heading)")
            else:
                bad = [p for p in [*f.pages, *f.planted_pages] if not 1 <= p <= doc.pages]
                if bad:
                    out.append(f"fact {f.id}: pages {bad} are outside 1..{doc.pages} of {doc.name}")
                if finalized and not set(f.planted_pages) <= set(f.pages):
                    out.append(f"fact {f.id}: planted pages {f.planted_pages} are not all in pages {f.pages}")
                if finalized and not f.pages:
                    out.append(f"fact {f.id}: no pages recorded (manifest not finalized?)")
        return out

    def validate(self) -> Manifest:
        problems = self.problems()
        if problems:
            raise ManifestError(problems)
        return self

    def dump_json(self) -> str:
        """JSON with one document or fact per line, so a change to a fact shows as one changed line in a diff."""
        data = self.model_dump(mode="json")

        def lines(items: list[dict]) -> str:
            return ",\n".join("  " + json.dumps(item, ensure_ascii=False) for item in items)

        return (
            f'{{\n"version": {data["version"]},\n"documents": [\n{lines(data["documents"])}\n],\n'
            f'"facts": [\n{lines(data["facts"])}\n]\n}}\n'
        )


def load_manifest(path: Path | str) -> Manifest:
    """Read and validate a manifest file."""
    manifest = Manifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    return manifest.validate()


# ------------------------------------------------------------------ page verification


@dataclass(frozen=True, slots=True)
class RenderedDocument:
    """What was read back out of a finished file: ``pages[i]`` is the text of page/slide i + 1; page-less formats
    give ``sections`` (heading → the text under it) instead."""

    pages: list[str] | None = None
    sections: dict[str, str] | None = None


def _pages_with(evidence: Sequence[str], pages: Sequence[str], *, approximate: bool) -> list[int]:
    return [i for i, text in enumerate(pages, start=1) if contains_all(text, evidence, approximate=approximate)]


def finalize_manifest(planted: Manifest, rendered: Mapping[str, RenderedDocument]) -> Manifest:
    """Check the planted facts against the rendered documents and return the manifest with real pages.

    Raises ``ManifestError`` listing every problem: a document missing, a page count different from the planned
    one, evidence not found anywhere, a planted page whose text doesn't contain the evidence, a locator that is not
    a section of the document. Facts stated on more pages than planted keep all of them (any such page is a valid
    answer), so the manifest never marks a correct retrieval as wrong.
    """
    problems = planted.problems(finalized=False)
    facts: list[Fact] = []
    for doc in planted.documents:
        r = rendered.get(doc.name)
        if r is None:
            problems.append(f"{doc.name}: nothing rendered to check against")
            continue
        if doc.pages is not None:
            if r.pages is None:
                problems.append(f"{doc.name}: expected {doc.pages} rendered pages, got none")
            elif len(r.pages) != doc.pages:
                problems.append(f"{doc.name}: planned {doc.pages} pages but the file has {len(r.pages)}")
        elif r.sections is None:
            problems.append(f"{doc.name}: expected rendered sections, got none")

    docs = {d.name: d for d in planted.documents}
    for f in planted.facts:
        doc = docs.get(f.document)
        r = rendered.get(f.document)
        if doc is None or r is None:
            continue
        if doc.pages is not None:
            if r.pages is None:
                continue
            found = _pages_with(f.evidence, r.pages, approximate=f.ocr)
            if not found:
                problems.append(f"fact {f.id}: evidence {f.evidence} not found on any page of {doc.name}")
                continue
            missing = [p for p in f.planted_pages if p not in found]
            if missing:
                problems.append(
                    f"fact {f.id}: planted on page(s) {missing} of {doc.name} but its evidence {f.evidence} is "
                    f"not on them (found on {found})"
                )
            if not f.planted_pages:
                problems.append(f"fact {f.id}: no planted page")
            facts.append(f.model_copy(update={"pages": found}))
        else:
            if r.sections is None:
                continue
            text = r.sections.get(f.locator or "")
            if text is None:
                problems.append(f"fact {f.id}: locator {f.locator!r} is not a section of {doc.name}")
            elif not contains_all(text, f.evidence):
                problems.append(f"fact {f.id}: evidence {f.evidence} is not under {f.locator!r} in {doc.name}")
            facts.append(f)
    if problems:
        raise ManifestError(problems)
    return Manifest(version=planted.version, documents=planted.documents, facts=facts)


def check_against_rendered(manifest: Manifest, rendered: Mapping[str, RenderedDocument]) -> list[str]:
    """Differences between a (committed) manifest and freshly rendered documents; empty when they agree."""
    try:
        again = finalize_manifest(manifest, rendered)
    except ManifestError as e:
        return e.problems
    diffs: list[str] = []
    old = {f.id: f for f in manifest.facts}
    for f in again.facts:
        if old[f.id].pages != f.pages:
            diffs.append(f"fact {f.id}: manifest says pages {old[f.id].pages}, rendered files give {f.pages}")
    return diffs
