# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.10", "pypdf>=5.0", "python-pptx>=1.0", "python-docx>=1.1"]
# ///
# ruff: noqa: E501  (document text and evidence strings are data)
"""Write evals/retrieval/questions.jsonl from the authored bank (scripts/eval/question_bank.py).

    uv run scripts/eval/build_questions.py            # regenerate and validate
    uv run scripts/eval/build_questions.py --check    # fail if the committed file differs from a fresh build
    uv run scripts/eval/build_questions.py --verify-docs   # also read the generated documents: every answer string
                                                           # must be in the text of the pages the question expects

Fact ids in the bank are resolved through the committed manifest (evals/retrieval/manifest.json), so the expected
documents and pages in the question file can't be mistyped. The result is validated with the same code CI uses
(app.evals.questions.validate_questions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from question_bank import QUESTIONS  # noqa: E402

from app.evals.manifest import Manifest, load_manifest  # noqa: E402
from app.evals.metrics import normalize_answer  # noqa: E402
from app.evals.questions import (  # noqa: E402
    CATEGORIES,
    Question,
    parse_questions,
    summarize_questions,
    validate_questions,
)

MANIFEST = ROOT / "evals/retrieval/manifest.json"
OUT = ROOT / "evals/retrieval/questions.jsonl"


def resolve(entry: dict, manifest: Manifest, number: int) -> dict:
    expected = []
    for group in entry["facts"]:
        ids = [group] if isinstance(group, str) else list(group)
        facts = [manifest.fact(i) for i in ids]
        docs = {f.document for f in facts}
        if len(docs) != 1:
            raise SystemExit(
                f"question {number} ({entry['question']!r}): facts {ids} are in different documents {docs}"
            )
        pages = sorted({p for f in facts for p in f.pages})
        expected.append({"document": facts[0].document, "pages": pages, "facts": ids})
    row = {
        "id": f"q{number:03d}",
        "question": entry["question"],
        "language": entry["language"],
        "category": entry["category"],
        "expected": expected,
        "answer_contains": entry["answers"],
        "should_abstain": entry["abstain"],
    }
    if entry.get("en"):
        row["query_en"] = entry["en"]
    if entry.get("clean"):
        row["clean_question"] = entry["clean"]
    if entry.get("subtype"):
        row["subtype"] = entry["subtype"]
    if entry.get("note"):
        row["notes"] = entry["note"]
    return row


def answer_warnings(rows: list[dict], manifest: Manifest) -> list[str]:
    """answer_contains strings that appear in no cited fact's statement or evidence (probably a typo; review)."""
    out = []
    for r in rows:
        if not r["expected"]:
            continue
        haystack = normalize_answer(
            " ".join(
                " ".join([manifest.fact(i).statement, *manifest.fact(i).evidence])
                for e in r["expected"]
                for i in e["facts"]
            )
        )
        for needle in r["answer_contains"]:
            if not any(normalize_answer(alt) in haystack for alt in needle.split("|")):
                out.append(
                    f"{r['id']}: {needle!r} is not in the statement or evidence of {[i for e in r['expected'] for i in e['facts']]}"
                )
    return out


def verify_against_docs(rows: list[dict], manifest: Manifest, docs_dir: Path) -> list[str]:
    """Independent check on the corpus files: each answer string of an answerable question occurs in the text of the
    pages (or sections) its expected evidence points at. OCR documents have no text layer and are skipped."""
    from corpus.render import render_docx, render_pdf, render_pptx

    from app.evals.metrics import normalize_answer

    rendered: dict[str, object] = {}
    problems = []
    for r in rows:
        if not r["expected"]:
            continue
        texts = []
        skip = False
        for e in r["expected"]:
            doc = manifest.document(e["document"])
            if doc.scanned:
                skip = True
                break
            if e["document"] not in rendered:
                path = docs_dir / doc.name
                rendered[doc.name] = {"pdf": render_pdf, "pptx": render_pptx, "docx": render_docx}[doc.format](path)
            rd = rendered[doc.name]
            if doc.pages is not None:
                texts += [rd.pages[p - 1] for p in e["pages"]]
            else:
                texts += [rd.sections[manifest.fact(i).locator or ""] for i in e["facts"]]
        if skip:
            continue
        haystack = normalize_answer(" ".join(texts))
        for needle in r["answer_contains"]:
            if not any(normalize_answer(alt) in haystack for alt in needle.split("|")):
                problems.append(f"{r['id']}: {needle!r} is not in the text of its expected pages ({r['question']})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing")
    ap.add_argument(
        "--verify-docs",
        nargs="?",
        const=str(ROOT / "data/eval/docs"),
        default=None,
        metavar="DIR",
        help="verify answer strings against the generated documents",
    )
    args = ap.parse_args()

    manifest = load_manifest(MANIFEST)
    rows = [resolve(e, manifest, n) for n, e in enumerate(QUESTIONS, start=1)]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    questions: list[Question] = parse_questions(text.splitlines())
    problems = validate_questions(questions, manifest)
    if problems:
        print("question set is inconsistent:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    for w in answer_warnings(rows, manifest):
        print("warning:", w)
    if args.verify_docs:
        bad = verify_against_docs(rows, manifest, Path(args.verify_docs))
        if bad:
            print("answer strings missing from the expected pages:\n  - " + "\n  - ".join(bad), file=sys.stderr)
            return 1
        print(
            f"verified the answer strings of {sum(1 for r in rows if r['expected'])} answerable questions against {args.verify_docs}"
        )
    counts = summarize_questions(questions)
    print(f"{len(questions)} questions; by language {counts['language']}; {counts['abstain']}")
    for c in CATEGORIES:
        print(f"  {c:16} {counts['category'].get(c, 0)}")
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.relative_to(ROOT)} differs from a fresh build; run build_questions.py", file=sys.stderr)
            return 1
        print("committed questions.jsonl is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
