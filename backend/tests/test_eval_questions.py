"""The retrieval question set: schema validation, consistency with the manifest, and the committed file's shape."""

from __future__ import annotations

import json

import pytest

from app.evals.manifest import DocumentEntry, Fact, Manifest
from app.evals.questions import (
    CATEGORIES,
    LANGUAGES,
    UNANSWERABLE_SUBTYPES,
    QuestionFileError,
    parse_questions,
    summarize_questions,
    validate_questions,
)


def mini_manifest() -> Manifest:
    def f(fid: str, doc: str, pages: list[int], locator: str | None = None) -> Fact:
        return Fact(
            id=fid,
            document=doc,
            pages=pages,
            planted_pages=pages,
            locator=locator,
            evidence=["e"],
            statement=fid,
            kind="number",
        )

    return Manifest(
        documents=[
            DocumentEntry(name="r.pdf", title="R", format="pdf", language="en", pages=5),
            DocumentEntry(name="n.docx", title="N", format="docx", language="hi"),
        ],
        facts=[f("r.a", "r.pdf", [2, 4]), f("r.b", "r.pdf", [3]), f("n.a", "n.docx", [], "1. शीर्षक")],
    )


def q(**over) -> dict:
    base = {
        "id": "q1",
        "question": "What was it?",
        "language": "en",
        "category": "exact_fact",
        "expected": [{"document": "r.pdf", "pages": [2, 4], "facts": ["r.a"]}],
        "answer_contains": ["x"],
        "should_abstain": False,
    }
    base.update(over)
    return base


def check(*rows: dict) -> list[str]:
    qs = parse_questions(json.dumps(r) for r in rows)
    return validate_questions(qs, mini_manifest())


# ------------------------------------------------------------------ schema


def test_a_valid_question_passes():
    assert check(q()) == []


def test_parse_reports_every_bad_line_with_its_number():
    lines = [
        json.dumps(q()),
        "{not json",
        json.dumps(q(id="q2", category="trivia")),
        json.dumps({**q(id="q3"), "extra": 1}),
    ]
    with pytest.raises(QuestionFileError) as e:
        parse_questions(lines)
    text = " ".join(e.value.problems)
    assert "line 2" in text and "not valid JSON" in text
    assert "line 3" in text and "category" in text
    assert "line 4" in text and "extra" in text.lower()


def test_blank_lines_and_comments_are_skipped():
    assert len(parse_questions(["", "# comment", json.dumps(q())])) == 1


def test_language_and_category_vocabularies_are_closed():
    with pytest.raises(QuestionFileError):
        parse_questions([json.dumps(q(language="fr"))])
    assert set(LANGUAGES) == {"en", "hi", "hinglish"}
    assert "unanswerable" in CATEGORIES and len(CATEGORIES) == len(set(CATEGORIES)) == 12


# ------------------------------------------------------------------ consistency with the manifest


def test_expected_document_must_exist_in_the_manifest():
    problems = check(q(expected=[{"document": "ghost.pdf", "pages": [1], "facts": ["r.a"]}]))
    assert any("ghost.pdf" in p and "not in the manifest" in p for p in problems)


def test_expected_pages_must_equal_the_pages_of_the_cited_facts():
    problems = check(q(expected=[{"document": "r.pdf", "pages": [2], "facts": ["r.a"]}]))
    assert any("differ from the pages of facts" in p for p in problems)
    ok = check(
        q(expected=[{"document": "r.pdf", "pages": [2, 3, 4], "facts": ["r.a", "r.b"]}])
    )  # alternatives: the union
    assert ok == []


def test_expected_pages_must_exist_in_the_document():
    problems = check(q(expected=[{"document": "r.pdf", "pages": [9], "facts": ["r.a"]}]))
    assert any("outside 1..5" in p for p in problems)


def test_facts_must_exist_and_belong_to_the_expected_document():
    assert any("unknown fact" in p for p in check(q(expected=[{"document": "r.pdf", "pages": [2], "facts": ["nope"]}])))
    assert any(
        "is in n.docx, not r.pdf" in p
        for p in check(q(expected=[{"document": "r.pdf", "pages": [], "facts": ["n.a"]}]))
    )
    assert any("lists no manifest facts" in p for p in check(q(expected=[{"document": "r.pdf", "pages": [2]}])))


def test_page_less_documents_take_no_pages():
    ok = check(q(language="hi", query_en="what", expected=[{"document": "n.docx", "pages": [], "facts": ["n.a"]}]))
    assert ok == []
    bad = check(q(language="hi", query_en="what", expected=[{"document": "n.docx", "pages": [1], "facts": ["n.a"]}]))
    assert any("has no pages" in p for p in bad)


def test_multi_hop_lists_one_entry_per_piece_of_evidence():
    two = q(
        expected=[
            {"document": "r.pdf", "pages": [2, 4], "facts": ["r.a"]},
            {"document": "r.pdf", "pages": [3], "facts": ["r.b"]},
        ]
    )
    assert check(two) == []


# ------------------------------------------------------------------ answerable and unanswerable


def test_unanswerable_questions_have_no_evidence_and_abstain():
    ok = q(
        id="u", category="unanswerable", should_abstain=True, expected=[], answer_contains=[], subtype="near_miss_year"
    )
    assert check(ok) == []
    assert any("no expected pages" in p for p in check({**ok, "expected": q()["expected"]}))
    assert any("exactly for the unanswerable" in p for p in check({**ok, "should_abstain": False}))
    assert any("exactly for the unanswerable" in p for p in check(q(should_abstain=True)))
    assert any("unknown subtype" in p for p in check({**ok, "subtype": "weather"}))
    assert set(UNANSWERABLE_SUBTYPES) >= {"near_miss_year", "other_company", "off_topic"}


def test_answerable_questions_need_evidence_and_answer_strings():
    assert any("needs expected evidence" in p for p in check(q(expected=[])))
    assert any("needs answer_contains" in p for p in check(q(answer_contains=[])))
    assert any("empty answer_contains" in p for p in check(q(answer_contains=["ok", " "])))
    assert any("subtype is for unanswerable" in p for p in check(q(subtype="off_topic")))


def test_router_query_is_required_exactly_for_non_english_questions():
    assert any("need the router's query_en" in p for p in check(q(language="hi")))
    assert any("need the router's query_en" in p for p in check(q(language="hinglish", query_en="  ")))
    assert any("no query_en" in p for p in check(q(query_en="what")))
    assert check(q(language="hi", query_en="what was it?")) == []


def test_asr_questions_carry_the_clean_transcription():
    assert any("clean_question" in p for p in check(q(category="asr_noise")))
    assert check(q(category="asr_noise", clean_question="What was it?")) == []
    assert any("clean_question is for asr_noise" in p for p in check(q(clean_question="x")))


def test_ids_must_be_unique():
    assert any("duplicate question id" in p for p in check(q(), q()))


def test_summary_counts():
    qs = parse_questions(
        [
            json.dumps(q()),
            json.dumps(q(id="q2", category="unanswerable", should_abstain=True, expected=[], answer_contains=[])),
        ]
    )
    s = summarize_questions(qs)
    assert s["category"] == {"exact_fact": 1, "unanswerable": 1}
    assert s["abstain"] == {"answer": 1, "abstain": 1}
