"""The retrieval question set (``evals/retrieval/questions.jsonl``): schema, loading and validation.

One JSON object per line. ``expected`` lists the places retrieval must find, one entry per piece of evidence the
answer needs (several entries = multi-hop or cross-document: all are required). An entry names the document and the
pages that state the fact (any of them counts; empty for page-less DOCX) and the manifest facts it comes from
(``facts``; the harness takes the evidence strings from them, and validation checks pages against them).
``query_en`` is the English query the router (§3.4) would produce for a Hindi or Hinglish turn.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .manifest import Manifest

Language = Literal["en", "hi", "hinglish"]
LANGUAGES: tuple[str, ...] = ("en", "hi", "hinglish")

Category = Literal[
    "exact_fact",
    "paraphrase",
    "identifier",
    "wrong_year_trap",
    "multi_hop",
    "cross_doc",
    "table_cell",
    "glossary",
    "hindi_doc",
    "xlingual",
    "asr_noise",
    "unanswerable",
]
CATEGORIES: tuple[str, ...] = (
    "exact_fact",
    "paraphrase",
    "identifier",
    "wrong_year_trap",
    "multi_hop",
    "cross_doc",
    "table_cell",
    "glossary",
    "hindi_doc",
    "xlingual",
    "asr_noise",
    "unanswerable",
)
CATEGORY_DESCRIPTIONS = {
    "exact_fact": "exact facts and numbers stated in prose",
    "paraphrase": "paraphrased questions with no keyword overlap with the passage",
    "identifier": "CIN, ISIN, policy numbers, clause numbers, notice numbers",
    "wrong_year_trap": "near-duplicate facts that differ by year or segment (FY23 vs FY24, segment A vs B)",
    "multi_hop": "two pieces of evidence on different pages of one document",
    "cross_doc": "the same metric in two companies' documents: the right document must be told apart",
    "table_cell": "a single cell of a financial or policy table",
    "glossary": "abbreviations and defined terms",
    "hindi_doc": "Hindi questions on the Hindi document",
    "xlingual": "Hindi and Hinglish questions on the English documents",
    "asr_noise": "speech-recognition errors in the question (Whisper-style misspellings)",
    "unanswerable": "plausible questions the documents do not answer (near misses included)",
}
UNANSWERABLE_SUBTYPES = ("near_miss_year", "near_miss_segment", "other_company", "missing_detail", "off_topic")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Expected(_Model):
    document: str
    pages: list[int] = Field(default_factory=list)  # empty for page-less documents (DOCX)
    facts: list[str] = Field(default_factory=list)  # manifest fact ids this evidence comes from


class Question(_Model):
    id: str
    question: str
    language: Language
    category: Category
    expected: list[Expected] = Field(default_factory=list)
    answer_contains: list[str] = Field(default_factory=list)  # each entry must appear in the answer; "a|b" = either
    should_abstain: bool = False
    query_en: str | None = None  # the router's English query: required for hi/hinglish, absent for en
    clean_question: str | None = None  # asr_noise: what the user actually said, without the recognition errors
    subtype: str | None = None  # unanswerable: why (see UNANSWERABLE_SUBTYPES)
    notes: str | None = None


class QuestionFileError(ValueError):
    """The question file is malformed or inconsistent with the manifest. ``problems`` lists all."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        shown = "\n  - ".join(self.problems[:60])
        more = f"\n  ... and {len(self.problems) - 60} more" if len(self.problems) > 60 else ""
        super().__init__(f"{len(self.problems)} problem(s) in the question set:\n  - {shown}{more}")


def parse_questions(lines: Iterable[str]) -> list[Question]:
    """Parse JSONL lines (blank lines and ``#`` comments skipped); raises ``QuestionFileError`` on bad lines."""
    questions: list[Question] = []
    problems: list[str] = []
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            questions.append(Question.model_validate(json.loads(line)))
        except json.JSONDecodeError as e:
            problems.append(f"line {number}: not valid JSON ({e})")
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"])
                problems.append(f"line {number}: {loc}: {err['msg']}")
    if problems:
        raise QuestionFileError(problems)
    return questions


def load_questions(path: Path | str) -> list[Question]:
    return parse_questions(Path(path).read_text(encoding="utf-8").splitlines())


def validate_questions(questions: Sequence[Question], manifest: Manifest) -> list[str]:
    """Problems of the question set against the manifest; empty when consistent."""
    problems: list[str] = []
    ids = Counter(q.id for q in questions)
    problems += [f"duplicate question id {i!r}" for i, n in ids.items() if n > 1]
    docs = {d.name: d for d in manifest.documents}
    facts = {f.id: f for f in manifest.facts}
    for q in questions:
        where = f"{q.id}"
        if not q.question.strip():
            problems.append(f"{where}: empty question")
        if q.should_abstain != (q.category == "unanswerable"):
            problems.append(f"{where}: should_abstain must be true exactly for the unanswerable category")
        if q.should_abstain:
            if q.expected or q.answer_contains:
                problems.append(f"{where}: an unanswerable question has no expected pages or answer strings")
            if q.subtype is not None and q.subtype not in UNANSWERABLE_SUBTYPES:
                problems.append(f"{where}: unknown subtype {q.subtype!r} (one of {', '.join(UNANSWERABLE_SUBTYPES)})")
        else:
            if not q.expected:
                problems.append(f"{where}: an answerable question needs expected evidence")
            if not q.answer_contains:
                problems.append(f"{where}: an answerable question needs answer_contains strings")
            if q.subtype is not None:
                problems.append(f"{where}: subtype is for unanswerable questions only")
        if any(not s.strip() for s in q.answer_contains):
            problems.append(f"{where}: empty answer_contains string")
        if q.language == "en" and q.query_en is not None:
            problems.append(f"{where}: English questions have no query_en (the router adds none)")
        if q.language != "en" and not (q.query_en and q.query_en.strip()):
            problems.append(f"{where}: {q.language} questions need the router's query_en")
        if q.category == "asr_noise" and not q.clean_question:
            problems.append(f"{where}: asr_noise questions need the clean_question they were derived from")
        if q.category != "asr_noise" and q.clean_question is not None:
            problems.append(f"{where}: clean_question is for asr_noise questions only")
        for e in q.expected:
            doc = docs.get(e.document)
            if doc is None:
                problems.append(f"{where}: expected document {e.document!r} is not in the manifest")
                continue
            if not e.facts:
                problems.append(f"{where}: expected entry for {e.document} lists no manifest facts")
            union: set[int] = set()
            for fid in e.facts:
                fact = facts.get(fid)
                if fact is None:
                    problems.append(f"{where}: unknown fact {fid!r}")
                elif fact.document != e.document:
                    problems.append(f"{where}: fact {fid} is in {fact.document}, not {e.document}")
                else:
                    union.update(fact.pages)
            if doc.pages is None:
                if e.pages:
                    problems.append(f"{where}: {e.document} has no pages, but expected lists {e.pages}")
            else:
                if any(not 1 <= p <= doc.pages for p in e.pages):
                    problems.append(f"{where}: pages {e.pages} are outside 1..{doc.pages} of {e.document}")
                elif sorted(e.pages) != sorted(union):
                    problems.append(
                        f"{where}: pages {e.pages} differ from the pages of facts {e.facts}: {sorted(union)}"
                    )
    return problems


def summarize_questions(questions: Sequence[Question]) -> dict[str, dict[str, int]]:
    """Counts by category, language and category x language (for the build report and the README)."""
    return {
        "category": dict(Counter(q.category for q in questions)),
        "language": dict(Counter(q.language for q in questions)),
        "abstain": dict(Counter("abstain" if q.should_abstain else "answer" for q in questions)),
    }
