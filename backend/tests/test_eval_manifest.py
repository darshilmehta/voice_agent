"""The eval corpus manifest: verifying planted facts against rendered pages, and the committed manifest itself."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.manifest import (
    DocumentEntry,
    Fact,
    Manifest,
    ManifestError,
    RenderedDocument,
    check_against_rendered,
    finalize_manifest,
    load_manifest,
)


def fact(
    fid: str,
    doc: str,
    *,
    planted: list[int] | None = None,
    locator: str | None = None,
    evidence: list[str],
    ocr: bool = False,
) -> Fact:
    return Fact(
        id=fid,
        document=doc,
        planted_pages=planted or [],
        locator=locator,
        evidence=evidence,
        statement=fid,
        kind="number",
        ocr=ocr,
    )


def planted() -> Manifest:
    return Manifest(
        documents=[
            DocumentEntry(name="report.pdf", title="Report", format="pdf", language="en", pages=3),
            DocumentEntry(name="policy.docx", title="Policy", format="docx", language="en", pages=None),
        ],
        facts=[
            fact("r.margin", "report.pdf", planted=[1], evidence=["EBITDA margin", "21.0%"]),
            fact("r.cin", "report.pdf", planted=[2], evidence=["L24119GJ1994PLC023871"]),
            fact("p.clause", "policy.docx", locator="4.2 Air travel", evidence=["4.2.1", "economy"]),
        ],
    )


def rendered() -> dict[str, RenderedDocument]:
    return {
        "report.pdf": RenderedDocument(
            pages=[
                "Highlights\nEBITDA margin   21.0%  (FY24)",
                "Corporate information\nCIN: L24119GJ1994PLC023871",
                "Segments\nGroup EBITDA margin 21.0% again",
            ]
        ),
        "policy.docx": RenderedDocument(
            sections={"4.2 Air travel": "4.2.1 Employees travel economy class.", "5. Hotels": "5.1 limits"}
        ),
    }


def test_finalize_records_every_page_that_states_the_fact():
    m = finalize_manifest(planted(), rendered())
    margin = m.fact("r.margin")
    assert margin.planted_pages == [1] and margin.pages == [1, 3]  # also stated on page 3: both are right answers
    assert m.fact("r.cin").pages == [2]
    clause = m.fact("p.clause")
    assert clause.pages == [] and clause.locator == "4.2 Air travel"
    m.validate()


def test_a_planted_page_that_does_not_contain_the_evidence_fails_loudly():
    r = rendered()
    r["report.pdf"] = RenderedDocument(pages=["EBITDA margin 21.0%", "no identifier here", "x"])
    with pytest.raises(ManifestError) as e:
        finalize_manifest(planted(), r)
    assert any("r.cin" in p and "not found" in p for p in e.value.problems)


def test_a_fact_planted_on_one_page_but_rendered_on_another_fails():
    r = rendered()
    r["report.pdf"] = RenderedDocument(pages=["Highlights", "EBITDA margin 21.0%", "CIN L24119GJ1994PLC023871"])
    with pytest.raises(ManifestError) as e:
        finalize_manifest(planted(), r)
    problems = " ".join(e.value.problems)
    assert "r.margin" in problems and "planted on page(s) [1]" in problems and "found on [2]" in problems
    assert "r.cin" in problems and "planted on page(s) [2]" in problems


def test_a_page_count_different_from_the_plan_fails():
    r = rendered()
    r["report.pdf"] = RenderedDocument(pages=["EBITDA margin 21.0%", "CIN L24119GJ1994PLC023871", "x", "overflow page"])
    with pytest.raises(ManifestError) as e:
        finalize_manifest(planted(), r)
    assert any("planned 3 pages but the file has 4" in p for p in e.value.problems)


def test_evidence_found_nowhere_names_the_document():
    r = rendered()
    r["report.pdf"] = RenderedDocument(pages=["a", "CIN L24119GJ1994PLC023871", "b"])
    with pytest.raises(ManifestError) as e:
        finalize_manifest(planted(), r)
    assert any("r.margin" in p and "any page of report.pdf" in p for p in e.value.problems)


def test_page_less_facts_are_checked_under_their_heading():
    r = rendered()
    r["policy.docx"] = RenderedDocument(
        sections={"4.2 Air travel": "4.2.2 Business class", "5. Hotels": "4.2.1 economy"}
    )
    with pytest.raises(ManifestError, match=r"not under '4\.2 Air travel'"):
        finalize_manifest(planted(), r)
    r["policy.docx"] = RenderedDocument(sections={"Other": "4.2.1 economy"})
    with pytest.raises(ManifestError, match=r"not a section of policy\.docx"):
        finalize_manifest(planted(), r)


def test_all_problems_are_reported_together_not_just_the_first():
    r = {"report.pdf": RenderedDocument(pages=["x", "y"])}  # one page short, no docx, no evidence anywhere
    with pytest.raises(ManifestError) as e:
        finalize_manifest(planted(), r)
    assert len(e.value.problems) >= 4
    assert "problem(s)" in str(e.value)


def test_ocr_facts_are_matched_without_punctuation():
    m = Manifest(
        documents=[DocumentEntry(name="scan.pdf", title="S", format="pdf", language="en", pages=1, scanned=True)],
        facts=[fact("s.policy", "scan.pdf", planted=[1], evidence=["GHI/2024/00418377"], ocr=True)],
    )
    out = finalize_manifest(m, {"scan.pdf": RenderedDocument(pages=["Policy number GHI 2024 00418377"])})
    assert out.fact("s.policy").pages == [1]
    strict = m.model_copy(update={"facts": [fact("s.policy", "scan.pdf", planted=[1], evidence=["GHI/2024/00418377"])]})
    with pytest.raises(ManifestError):
        finalize_manifest(strict, {"scan.pdf": RenderedDocument(pages=["Policy number GHI 2024 00418377"])})


def test_structural_problems_are_found_without_any_rendered_file():
    bad = Manifest(
        documents=[
            DocumentEntry(name="a.pdf", title="A", format="pdf", language="en", pages=2),
            DocumentEntry(name="a.pdf", title="A again", format="pdf", language="en", pages=2),
            DocumentEntry(name="b.docx", title="B", format="docx", language="en"),
        ],
        facts=[
            fact("x", "a.pdf", planted=[5], evidence=["e"]),
            fact("x", "a.pdf", planted=[1], evidence=["e"]),
            fact("y", "ghost.pdf", planted=[1], evidence=["e"]),
            fact("z", "b.docx", evidence=["e"]),  # page-less document without a locator
            fact("w", "a.pdf", planted=[1], evidence=[]),
        ],
    )
    problems = " | ".join(bad.problems(finalized=False))
    assert "duplicate document" in problems and "duplicate fact id" in problems
    assert "outside 1..2" in problems and "unknown document" in problems
    assert "needs a locator" in problems and "non-empty evidence" in problems


def test_a_manifest_must_be_finalized_before_questions_can_use_it():
    with pytest.raises(ManifestError, match="no pages recorded"):
        planted().validate()


def test_check_against_rendered_reports_drift_in_a_committed_manifest():
    committed = finalize_manifest(planted(), rendered())
    assert check_against_rendered(committed, rendered()) == []
    drifted = rendered()
    drifted["report.pdf"] = RenderedDocument(pages=["EBITDA margin 21.0%", "CIN L24119GJ1994PLC023871", "nothing"])
    diffs = check_against_rendered(committed, drifted)
    assert any("r.margin" in d and "[1, 3]" in d and "[1]" in d for d in diffs)


def test_manifest_round_trips_through_json(tmp_path: Path):
    m = finalize_manifest(planted(), rendered())
    path = tmp_path / "manifest.json"
    path.write_text(m.dump_json(), encoding="utf-8")
    assert load_manifest(path) == m
