"""Chunk assembly: provenance, tables kept whole, merging, overlap. Pure Python, no Docling needed."""

from __future__ import annotations

import uuid

import pytest

from app.providers.ingestion import (
    ChunkPiece,
    ParsedDocument,
    ParsedItem,
    ParsedTable,
    TableCell,
    assemble_chunks,
    content_budget,
    detect_language,
    document_label,
    document_title,
    point_id,
    tail_text,
)
from app.settings import ChunkingSection


def words(text: str) -> int:
    """Whitespace 'tokenizer' for tests."""
    return len(text.split())


CHUNKING = ChunkingSection(version="v2", target_tokens=50, overlap_tokens=10)  # content budget 40


def piece(text: str, *headings: str, pages: tuple[int, ...] = (1,), labels: tuple[str, ...] = ("text",)) -> ChunkPiece:
    return ChunkPiece(text=text, headings=headings, labels=labels, pages=pages)


def table(ref: str = "#/tables/0", *, index: int = 0, markdown: str | None = None, caption: str | None = None):
    md = markdown or "| Metric | FY23 | FY24 |\n|---|---|---|\n| EBITDA margin | 16.9% | 18.2% |"
    return ParsedTable(
        index=index,
        ref=ref,
        page_start=2,
        page_end=2,
        bbox=(10.0, 20.0, 300.0, 120.0),
        heading_path=["Key Metrics"],
        caption=caption,
        num_rows=2,
        num_cols=3,
        markdown=md,
        cells=[],
    )


def assemble(pieces, tables=(), **kw):
    args = {"project_id": "p1", "document_id": "d1", "version": 3, "chunking": CHUNKING, "count_tokens": words}
    return assemble_chunks(pieces, tables, **(args | kw))


def sentence(n: int, word: str = "alpha") -> str:
    return " ".join([word] * (n - 1)) + " end."


def test_every_chunk_carries_provenance():
    chunks = assemble(
        [
            piece("The company grew strongly this year.", "Overview", pages=(1,)),
            piece("Revenue rose and margins improved.", "Results", pages=(2, 3)),
        ]
    )
    assert [c.chunk_id for c in chunks] == ["d1:v3:0000", "d1:v3:0001"]
    first, second = chunks
    assert (first.project_id, first.document_id, first.version, first.chunk_index) == ("p1", "d1", 3, 0)
    assert first.chunking_version == "v2"
    assert first.heading_path == ["Overview"] and second.heading_path == ["Results"]
    assert (second.page_start, second.page_end) == (2, 3)
    assert first.content_type == "paragraph" and first.language == "en"
    assert first.table_index is None
    assert first.token_count == words(first.embed_text)
    assert first.embed_text == "Overview\nThe company grew strongly this year."


def test_point_ids_are_deterministic_uuids():
    a, b = assemble([piece("one two three.", "A"), piece("four five six.", "B")])
    assert a.point_id == point_id("d1:v3:0000") == str(uuid.UUID(a.point_id))
    assert a.point_id != b.point_id
    again = assemble([piece("changed text.", "A")])[0]
    assert again.point_id == a.point_id  # same chunk id → same point: re-ingest overwrites


def test_a_split_table_becomes_one_whole_chunk():
    big = "| Metric | FY23 | FY24 |\n|---|---|---|\n" + "\n".join(f"| row {i} | {i} | {i + 1} |" for i in range(40))
    t = table(markdown=big, caption="Table 1: Key metrics")
    pieces = [
        piece("Intro paragraph.", "Key Metrics", pages=(2,)),
        ChunkPiece("rows 0-19 as triplets", ("Key Metrics",), ("table",), (2,), t.ref),  # chunker split the table
        ChunkPiece("rows 20-39 as triplets", ("Key Metrics",), ("table",), (2,), t.ref),
        piece("Outro paragraph.", "Key Metrics", pages=(2,)),
    ]
    chunks = assemble(pieces, [t])
    assert [c.content_type for c in chunks] == ["paragraph", "table", "paragraph"]  # never merged with neighbours
    tc = chunks[1]
    assert tc.text == f"Table 1: Key metrics\n{big}"  # the whole markdown, caption first, past the budget
    assert tc.token_count > content_budget(CHUNKING)
    assert tc.table_index == 0 and (tc.page_start, tc.page_end) == (2, 2)
    assert tc.overlap_text == "" and chunks[2].overlap_text == ""  # tables neither give nor take overlap


def test_small_pieces_of_a_section_merge_up_to_the_budget():
    pieces = [piece(sentence(15), "S", pages=(1,)), piece(sentence(15), "S", pages=(2,)), piece(sentence(15), "S")]
    chunks = assemble(pieces)
    # "S" + 15 + 15 = 31 ≤ 40 merge; adding another 15 would exceed the budget
    assert len(chunks) == 2
    assert chunks[0].text == f"{sentence(15)}\n{sentence(15)}"
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_pieces_of_different_sections_never_merge():
    chunks = assemble([piece("short a.", "A"), piece("short b.", "B"), piece("short c.", "A")])
    assert [c.heading_path for c in chunks] == [["A"], ["B"], ["A"]]


def test_overlap_continues_a_split_section():
    first = f"{sentence(20, 'one')} {sentence(6, 'two')} {sentence(3, 'three')}"  # 29 words, 3 sentences
    chunks = assemble([piece(first, "S"), piece(sentence(30, "four"), "S")])
    assert len(chunks) == 2
    # whole trailing sentences within 10 tokens: "two…end." (6) + "three…end." (3)
    assert chunks[1].overlap_text == f"{sentence(6, 'two')} {sentence(3, 'three')}"
    assert words(chunks[1].overlap_text) <= CHUNKING.overlap_tokens
    assert chunks[1].embed_text == f"S\n{chunks[1].overlap_text}\n{sentence(30, 'four')}"
    assert chunks[1].token_count <= CHUNKING.target_tokens
    assert chunks[1].text == sentence(30, "four")  # the cited text excludes the overlap


def test_no_overlap_across_sections_or_when_disabled():
    pieces = [piece(sentence(30), "A"), piece(sentence(30), "B")]
    assert assemble(pieces)[1].overlap_text == ""
    same = [piece(sentence(30), "A"), piece(sentence(30), "A")]
    no_overlap = ChunkingSection(version="v1", target_tokens=50, overlap_tokens=0)
    assert assemble(same, chunking=no_overlap)[1].overlap_text == ""


def test_list_pieces_are_typed_as_lists_and_empty_pieces_are_dropped():
    chunks = assemble(
        [
            piece("Revenue up 34%", "Highlights", labels=("list_item",)),
            piece("   ", "Highlights"),
            piece("EBITDA margin 18.2%", "Highlights", labels=("list_item",)),
        ]
    )
    assert len(chunks) == 1 and chunks[0].content_type == "list"
    assert chunks[0].text == "Revenue up 34%\nEBITDA margin 18.2%"


def test_chunk_language_is_detected_with_the_document_language_as_fallback():
    chunks = assemble(
        [piece("कंपनी ने EBITDA मार्जिन में सुधार किया।", "A"), piece("12,345 67.8%", "B")], default_language="hi"
    )
    assert [c.language for c in chunks] == ["hi", "hi"]


def test_chunks_are_labelled_with_their_document():
    """Tables and slides rarely name their company: the label (file name and title) is embedded and reranked with
    every chunk, so "Valmora's EBITDA" can't match another company's table as well as Valmora's."""
    chunks = assemble(
        [
            piece("| Metric | FY24 |\n| EBITDA margin | 21.0% |", "Financial highlights"),
            piece("Introduction to the plan.", "Valmora Industries: Annual Report"),
        ],
        source_name="valmora_annual-report.FY24.pdf",
        title="Valmora Industries: Annual Report",
    )
    table_chunk, titled = chunks
    assert table_chunk.document_label == "valmora annual report FY24: Valmora Industries: Annual Report"
    assert table_chunk.embed_text.splitlines()[:2] == [table_chunk.document_label, "Financial highlights"]
    assert table_chunk.text.startswith("| Metric")  # shown and cited as before
    assert table_chunk.token_count == words(table_chunk.embed_text)
    assert titled.document_label == "valmora annual report FY24"  # its heading path already starts with the title
    (plain,) = assemble([piece("Text.", "H")])
    assert plain.document_label == "" and plain.embed_text == "H\nText."  # no name given: unlabelled, as before


@pytest.mark.parametrize(
    ("name", "title", "label"),
    [
        (
            "zephyra_investor_deck_q4fy24.pptx",
            "Zephyra Logistics Limited",
            "zephyra investor deck q4fy24: Zephyra Logistics Limited",
        ),
        ("valmora_annual_report.pdf", "Valmora annual report", "valmora annual report"),  # title adds nothing
        ("scan0042.pdf", None, "scan0042"),
        ("", "Group Health Insurance Policy", "Group Health Insurance Policy"),
    ],
)
def test_document_label(name, title, label):
    assert document_label(name, title) == label


def test_document_title_is_the_title_item_or_the_first_heading_of_the_first_page():
    def item(label: str, text: str, page: int | None, level: int | None = None) -> ParsedItem:
        return ParsedItem(ref="#", label=label, text=text, page=page, heading_path=[], level=level)

    def doc(*items: ParsedItem) -> ParsedDocument:
        return ParsedDocument(
            source_name="a.pdf",
            format="pdf",
            page_count=2,
            pages=[],
            items=list(items),
            tables=[],
            markdown="",
            language="en",
            parse_seconds=0.1,
        )

    cover = item("section_header", "VALMORA INDUSTRIES  LIMITED", 1, level=1)
    body = item("text", "Annual Report 2023-24", 1)
    later = item("section_header", "Corporate information", 2, level=1)
    assert document_title(doc(body, cover, later)) == "VALMORA INDUSTRIES LIMITED"
    assert document_title(doc(cover, item("title", "The Real Title", 2, level=0))) == "The Real Title"
    assert document_title(doc(body, later)) is None  # no heading on the first page
    assert document_title(doc(item("title", "x" * 200, 1, level=0))) is None  # not a title
    assert document_title(doc()) is None


def test_pages_may_be_unknown():
    (chunk,) = assemble([piece("A docx paragraph.", "H", pages=())])
    assert chunk.page_start is None and chunk.page_end is None


def test_content_budget_leaves_room_for_overlap():
    assert content_budget(ChunkingSection(version="v1", target_tokens=500, overlap_tokens=80)) == 420
    assert content_budget(ChunkingSection(version="v1", target_tokens=100, overlap_tokens=90)) == 50  # floor: half


@pytest.mark.parametrize(
    ("text", "budget", "expected"),
    [
        ("One two three. Four five. Six.", 3, "Four five. Six."),
        ("One two three. Four five. Six.", 1, "Six."),
        ("a b c d e f g h", 3, "f g h"),  # one long sentence: trailing words
        ("पहला वाक्य। दूसरा वाक्य।", 2, "दूसरा वाक्य।"),  # Hindi danda ends sentences
        ("anything", 0, ""),
        ("", 5, ""),
    ],
)
def test_tail_text(text, budget, expected):
    assert tail_text(text, budget, words) == expected


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("EBITDA margin improved to 18.2%", "en"),
        ("वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?", "hi"),
        ("12,345 | 67.8%", None),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_table_grid_expands_spans():
    t = ParsedTable(
        index=0,
        ref="#/tables/0",
        page_start=None,
        page_end=None,
        bbox=None,
        heading_path=[],
        caption=None,
        num_rows=2,
        num_cols=3,
        markdown="",
        cells=[
            TableCell(row=0, col=0, text="Metric", column_header=True),
            TableCell(row=0, col=1, col_span=2, text="Fiscal year", column_header=True),
            TableCell(row=1, col=0, text="EBITDA margin", row_header=True),
            TableCell(row=1, col=1, text="16.9%"),
            TableCell(row=1, col=2, text="18.2%"),
        ],
    )
    assert t.grid() == [["Metric", "Fiscal year", "Fiscal year"], ["EBITDA margin", "16.9%", "18.2%"]]
