"""The committed eval files (evals/retrieval/manifest.json and questions.jsonl): well formed, consistent with each
other, and shaped as the eval is meant to be. The documents themselves are generated (data/eval/docs, git-ignored) by
scripts/eval/build_corpus.py, which verifies the manifest's pages against the rendered files."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from app.evals.manifest import load_manifest
from app.evals.questions import CATEGORIES, LANGUAGES, UNANSWERABLE_SUBTYPES, load_questions, validate_questions
from app.settings import PROJECT_ROOT

COMMITTED = PROJECT_ROOT / "evals/retrieval/manifest.json"
QUESTIONS = PROJECT_ROOT / "evals/retrieval/questions.jsonl"
MANIFEST = COMMITTED


def test_committed_manifest_is_valid_and_describes_the_planned_corpus():
    m = load_manifest(COMMITTED)
    docs = {d.name: d for d in m.documents}
    assert len(docs) == 5
    assert {d.format for d in docs.values()} == {"pdf", "pptx", "docx"}
    report = next(d for d in m.documents if d.name.endswith("annual_report_fy24.pdf"))
    assert 25 <= (report.pages or 0) <= 35
    deck = next(d for d in m.documents if d.format == "pptx")
    assert deck.pages == 15
    scans = [d for d in m.documents if d.scanned]
    assert len(scans) == 1 and (scans[0].pages or 0) >= 2
    assert {d.language for d in m.documents} == {"en", "hi"}
    # real pages and slides; the two DOCX files add about 14 nominal pages that no tool can verify
    assert sum(d.pages or 0 for d in m.documents) >= 45
    for d in m.documents:
        assert m.facts_of(d.name), f"{d.name} has no planted facts"
    assert len(m.facts) >= 300


def test_committed_manifest_pages_are_the_rendered_pages_it_verified():
    m = load_manifest(COMMITTED)
    for f in m.facts:
        doc = m.document(f.document)
        if doc.pages is None:
            assert f.pages == [] and f.locator
        else:
            assert f.planted_pages and set(f.planted_pages) <= set(f.pages)
            assert all(1 <= p <= doc.pages for p in f.pages)


def test_committed_manifest_has_the_planted_traps():
    m = load_manifest(COMMITTED)
    by_id = {f.id: f for f in m.facts}
    # near-duplicate facts differ by year on the same table and by page in the prose
    assert by_id["vmr.margin.fy24"].evidence != by_id["vmr.margin.fy23"].evidence
    assert set(by_id["vmr.q.fy24.q3"].pages).isdisjoint(by_id["vmr.q.fy23.q3"].pages)
    # the same metric in two companies' documents
    assert by_id["vmr.margin.fy24"].document != by_id["zph.margin.fy24"].document
    # identifiers
    assert by_id["vmr.cin"].kind == "identifier" and by_id["trv.policy_no"].kind == "identifier"
    # an OCR document and a Hindi one
    assert any(f.ocr for f in m.facts) and any(f.id.startswith("hin.") for f in m.facts)


@pytest.fixture(scope="module")
def committed():
    from app.evals.manifest import load_manifest

    return load_questions(QUESTIONS), load_manifest(MANIFEST)


def test_committed_questions_are_consistent_with_the_committed_manifest(committed):
    questions, manifest = committed
    assert validate_questions(questions, manifest) == []


def test_committed_set_has_the_planned_size_and_balance(committed):
    questions, _ = committed
    assert 150 <= len(questions) <= 200
    counts = Counter(q.category for q in questions)
    assert set(counts) == set(CATEGORIES)
    assert all(n >= 8 for n in counts.values()), counts
    abstain = sum(q.should_abstain for q in questions) / len(questions)
    assert 0.15 <= abstain <= 0.25
    languages = Counter(q.language for q in questions)
    assert set(languages) == set(LANGUAGES)
    assert (languages["hi"] + languages["hinglish"]) / len(questions) >= 0.25
    assert len({q.id for q in questions}) == len(questions)


def test_committed_questions_cover_what_the_eval_is_meant_to_stress(committed):
    questions, manifest = committed
    hindi_docs = {d.name for d in manifest.documents if d.language == "hi"}
    for q in questions:
        docs = {e.document for e in q.expected}
        if q.category == "hindi_doc":
            assert docs <= hindi_docs and q.language == "hi"
        if q.category == "xlingual":
            assert q.language in ("hi", "hinglish") and docs and not docs & hindi_docs
        if q.category == "cross_doc":
            assert len(docs) == 1
        if q.category == "multi_hop":
            assert len(q.expected) == 2 and len(docs) == 1
        if q.language != "en":
            assert q.query_en
    subtypes = Counter(q.subtype for q in questions if q.should_abstain)
    assert set(subtypes) == set(UNANSWERABLE_SUBTYPES)
    assert sum(1 for q in questions if q.category == "asr_noise" and q.language == "hi") >= 3
    assert any("मुनापा" in q.question for q in questions)  # the Whisper error the task names
    scanned = {d.name for d in manifest.documents if d.scanned}
    assert any(e.document in scanned for q in questions for e in q.expected)  # OCR is exercised
    wrong_year = [q for q in questions if q.category == "wrong_year_trap"]
    assert any("FY23" in q.question for q in wrong_year) and any("FY24" in q.question for q in wrong_year)


def test_committed_file_is_one_json_object_per_line_in_utf8():
    text = QUESTIONS.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert text.endswith("\n") and all(json.loads(line) for line in lines)
    assert any("मुनापा" in line for line in lines)  # Devanagari kept as text, not escaped
