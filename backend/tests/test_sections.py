"""Citations of documents without pages (DOCX) say where they are by section: the chunk's heading path."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.projects import Citation, Message
from app.domain.summaries import KeyPoint, SourceRef, SummaryData
from app.services.chat_sources import EN_DASH, ChatSources, pages_label, ref_label, section_label
from app.services.chat_summary import _legend, inline_citation, render_summary_markdown
from app.services.export import ExportSource, _render_source
from app.services.memory import transcript
from app.services.retrieval import RankedChunk
from app.services.sources import build_sources, section_of

from .fakes import make_chunk

PATH = ["4. Travel", "4.2 Domestic", "4.2.1 Hotels"]
FULL = "4. Travel > 4.2 Domestic > 4.2.1 Hotels"
FILES = {"doc1": "travel_policy.docx"}


def docx_chunk(index: int, path: list[str], **over):
    return make_chunk(index, heading_path=path, page_start=None, page_end=None, text=f"limits {index}", **over)


def cite(source_id: str, section: str | None, *, page: int | None = None, document_id: str = "doc1") -> Citation:
    return Citation(
        source_id=source_id,
        document_id=document_id,
        filename=FILES.get(document_id, "report.pdf"),
        page_start=page,
        page_end=page,
        chunk_id=f"{document_id}:{source_id}",
        snippet="s",
        section=section,
    )


# ------------------------------------------------------------------ the citation


def test_section_is_the_heading_path_on_one_line():
    assert section_of(PATH) == FULL
    assert section_of(["  4.  Travel ", "", "4.2\nDomestic"]) == "4. Travel > 4.2 Domestic"
    assert section_of([]) is None and section_of([" "]) is None


def test_a_source_cites_its_chunks_section_with_or_without_a_page():
    docx = docx_chunk(0, PATH)
    pdf = make_chunk(1, heading_path=["Results"], page_start=2, page_end=2)
    plain = make_chunk(2, heading_path=[], page_start=None, page_end=None)
    ranked = [RankedChunk(c, 0.9 - i / 10, 0.03, 0.7, i + 1) for i, c in enumerate((docx, pdf, plain))]
    by_chunk = {s.chunk.chunk_index: s.citation() for s in build_sources(ranked, FILES, budget_tokens=3000)}
    assert (by_chunk[0].page_start, by_chunk[0].section) == (None, FULL)
    assert (by_chunk[1].page_start, by_chunk[1].section) == (2, "Results")  # the page wins in labels; the path stays
    assert (by_chunk[2].page_start, by_chunk[2].section) == (None, None)


def test_the_citation_json_has_a_section_for_documents_and_none_for_web_results():
    assert cite("S1", FULL).model_dump()["section"] == FULL
    assert cite("S1", None).model_dump()["section"] is None
    web = Citation(
        source_id="W1", document_id="", filename="livemint.com", page_start=None, page_end=None,
        chunk_id="", snippet="s", kind="web", url="https://livemint.com/a", title="t", site="livemint.com",
    )  # fmt: skip
    assert "section" not in web.model_dump()


def test_older_rows_without_a_section_are_read_with_none():
    old = Citation.coerce({"document_id": "d", "page": 3, "chunk_id": "c"}, 1)
    assert old is not None and old.section is None and old.page_start == 3
    new = Citation.coerce({"document_id": "d", "section": FULL, "chunk_id": "c"}, 1)
    assert new is not None and new.section == FULL and new.page_start is None
    web = Citation.coerce({"kind": "web", "url": "https://a.example", "section": "ignored"}, 1)
    assert web is not None and web.section is None


# ------------------------------------------------------------------ numbering and labels


def test_sections_of_a_document_without_pages_are_separate_sources():
    sources = ChatSources()
    mapping = sources.register(
        2, [cite("S1", FULL), cite("S2", "5. Expenses > 5.1 Meals"), cite("S3", FULL + "") ]
    )  # fmt: skip
    assert mapping == {"S1": 1, "S2": 2, "S3": 1}  # the same section is one source
    assert [e.section for e in sources.entries] == [FULL, "5. Expenses > 5.1 Meals"]
    assert [e.cited_by for e in sources.entries] == [[2], [2]]


def test_where_there_is_a_page_the_page_decides_and_the_section_is_ignored():
    sources = ChatSources()
    mapping = sources.register(2, [cite("S1", "Results", page=2), cite("S2", "Results > Margins", page=2)])
    assert mapping == {"S1": 1, "S2": 1}
    assert sources.entries[0].section is None and sources.entries[0].ref().section is None
    assert sources.register(4, [cite("S1", None, page=2)]) == {"S1": 1}


def test_a_source_is_labelled_by_its_page_else_its_last_heading():
    pdf = SourceRef(document_id="d", filename="report.pdf", page_start=2, page_end=3, section="Results > Margins")
    docx = SourceRef(document_id="d", filename="travel_policy.docx", page_start=None, page_end=None, section=FULL)
    bare = SourceRef(document_id="d", filename="notes.txt", page_start=None, page_end=None)
    assert ref_label(pdf) == f"report.pdf, pp. 2{EN_DASH}3"
    assert ref_label(docx) == "travel_policy.docx, § 4.2.1 Hotels"
    assert ref_label(docx, short=False) == f"travel_policy.docx, § {FULL}"
    assert ref_label(bare) == "notes.txt"
    assert section_label(FULL) == "§ 4.2.1 Hotels" and section_label(FULL, short=False) == f"§ {FULL}"
    assert section_label(None) == "" and section_label(" > ") == ""
    assert pages_label(None, None) == ""  # unchanged: no page, no label


def test_headings_are_escaped_where_they_end_up_in_markdown():
    ref = SourceRef(document_id="d", filename="p.docx", page_start=None, page_end=None, section="A > *Bold_* [x]")
    assert inline_citation([ref]) == r"(p.docx, § \*Bold\_\* \[x\])"
    assert ref_label(ref) == "p.docx, § *Bold_* [x]"  # the prompt legend is plain text


def test_the_summary_prompt_legend_and_citations_use_the_section():
    sources = ChatSources()
    sources.register(2, [cite("S1", FULL), cite("S2", "Results", page=4, document_id="doc2")])
    assert _legend(sources, {1, 2}) == ["[1] travel_policy.docx, § 4.2.1 Hotels", "[2] report.pdf, p. 4"]
    data = SummaryData(
        language="en",
        overview="",
        key_points=[KeyPoint(text="Hotels are capped.", sources=[sources.entries[0].ref()])],
        unanswered_questions=[],
        follow_ups=[],
        windows=1,
        prompt="t",
    )
    assert "- Hotels are capped. (travel_policy.docx, § 4.2.1 Hotels)" in render_summary_markdown(data)
    assert data.key_points[0].sources[0].model_dump()["section"] == FULL


# ------------------------------------------------------------------ export and memory


def source(**over) -> ExportSource:
    fields = dict(
        ref=1, document_id="doc1", filename="travel_policy.docx", page_start=None, page_end=None,
        snippet="Hotels: 6,500", cited_by=[2], section=FULL,
    )  # fmt: skip
    return ExportSource(**(fields | over))


def test_the_export_lists_a_source_without_a_page_by_its_whole_section():
    assert _render_source(source()) == (
        f"- [1] **travel_policy.docx**, § {FULL} — “Hotels: 6,500” _(cited in message #2)_"
    )
    assert _render_source(source(page_start=4, page_end=4)).startswith("- [1] **travel_policy.docx**, page 4 —")
    assert _render_source(source(section=None)).startswith("- [1] **travel_policy.docx** —")
    assert source().model_dump()["section"] == FULL and source(section=None).model_dump()["section"] is None


def test_a_web_source_has_no_section_in_the_export():
    web = ExportSource(
        ref=2, document_id="", filename="livemint.com", page_start=None, page_end=None, snippet="s", cited_by=[2],
        kind="web", url="https://livemint.com/a", title="t",
    )  # fmt: skip
    assert "section" not in web.model_dump()


def message(citations: list[Citation]) -> Message:
    return Message(
        id="m1", chat_id="c1", seq=2, role="agent", modality="text", text="Hotels are capped [S1].", heard_text=None,
        language="en", citations=citations, route=None, latency=None, created_at=datetime(2026, 10, 9, tzinfo=UTC),
    )  # fmt: skip


def test_the_memory_prompt_names_a_page_or_else_a_section():
    line = transcript([message([cite("S1", FULL), cite("S2", "Results", page=4, document_id="doc2")])])
    assert "(sources: report.pdf p.4, travel_policy.docx § 4.2.1 Hotels)" in line
    assert "(sources:" not in transcript([message([cite("S1", None)])])  # no page, no section: nothing to say
