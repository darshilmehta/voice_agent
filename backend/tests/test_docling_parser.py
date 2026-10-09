"""DoclingParser on real Docling, for formats that need no models (MD, TXT, DOCX). Needs the ``ml`` group.

The PDF path (layout, TableFormer, OCR models) is covered by the integration test.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("docling")
pytest.importorskip("semchunk")
docx = pytest.importorskip("docx")  # python-docx, a docling dependency

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer  # noqa: E402

from app.providers.ingestion import DoclingParser, IngestionError  # noqa: E402
from app.providers.registry import build_container  # noqa: E402


class WordTokenizer(BaseTokenizer):
    """Whitespace tokens: lets chunking run without the BGE-M3 tokenizer files."""

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return lambda text: len(text.split())  # semchunk accepts a token counter


ROWS = "\n".join(f"| Segment {i} | {100 + i} | {120 + i} |" for i in range(30))
MARKDOWN = f"""# Annual Report

## Financial Results

Revenue from operations grew 34% year on year. EBITDA margin improved to 18.2% from 16.9%.

| Metric | FY23 | FY24 |
|---|---|---|
| EBITDA margin | 16.9% | 18.2% |
{ROWS}

## Risks

- Single supplier for SKU-48213
- Currency exposure
"""


@pytest.fixture
def parser(load_local, monkeypatch) -> DoclingParser:
    settings = load_local(INGESTION__CHUNKING__TARGET_TOKENS="40", INGESTION__CHUNKING__OVERLAP_TOKENS="8")
    p = build_container(settings)["ingestion"]
    assert isinstance(p, DoclingParser)
    monkeypatch.setattr(p, "_chunk_tokenizer", lambda max_tokens: WordTokenizer(max_tokens=max_tokens))
    return p


def test_markdown_document_model(parser, tmp_path):
    f = tmp_path / "report.md"
    f.write_text(MARKDOWN)
    doc = asyncio.run(parser.parse(f))
    assert (doc.source_name, doc.format, doc.page_count, doc.language) == ("report.md", "md", None, "en")
    headings = [(it.text, it.heading_path) for it in doc.items if it.level is not None]
    assert ("Financial Results", ["Annual Report", "Financial Results"]) in headings
    (table,) = doc.tables
    assert table.heading_path == ["Annual Report", "Financial Results"]
    assert (table.num_rows, table.num_cols) == (32, 3)
    assert table.grid()[:2] == [["Metric", "FY23", "FY24"], ["EBITDA margin", "16.9%", "18.2%"]]
    assert all(c.column_header for c in table.cells if c.row == 0)
    assert "| EBITDA margin" in table.markdown and "18.2%" in doc.markdown
    table_item = next(it for it in doc.items if it.table_index == 0)
    assert table_item.label == "table"


def test_chunks_keep_the_table_whole_and_carry_provenance(parser, tmp_path):
    f = tmp_path / "report.md"
    f.write_text(MARKDOWN)
    doc = asyncio.run(parser.parse(f))
    chunks = asyncio.run(parser.chunk(doc, project_id="p1", document_id="d1", version=2))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) == 1  # 33 rows ≫ the 32-token budget, still one chunk
    assert tables[0].text == doc.tables[0].markdown and tables[0].table_index == 0
    assert tables[0].heading_path == ["Annual Report", "Financial Results"]
    lists = [c for c in chunks if c.content_type == "list"]
    assert lists and lists[0].heading_path == ["Annual Report", "Risks"]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id.startswith("d1:v2:") and c.chunking_version == parser.cfg.chunking.version for c in chunks)
    assert {c.document_label for c in chunks} == {"report"}  # the title "Annual Report" leads every heading path
    assert all(c.language == "en" and c.page_start is None for c in chunks)
    assert all(c.token_count <= 40 for c in chunks if c.content_type != "table")


def test_plain_text_paragraphs(parser, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("First paragraph\nwrapped over lines.\n\nSecond paragraph.\n", encoding="utf-8")
    doc = asyncio.run(parser.parse(f))
    assert doc.format == "txt"
    assert [it.text for it in doc.items] == ["First paragraph wrapped over lines.", "Second paragraph."]
    chunks = asyncio.run(parser.chunk(doc, project_id="p", document_id="d", version=1))
    assert chunks[0].text == "First paragraph wrapped over lines.\nSecond paragraph."


def test_hindi_docx(parser, tmp_path):
    d = docx.Document()
    d.add_heading("वार्षिक रिपोर्ट सारांश", level=1)
    d.add_paragraph("कंपनी ने वित्त वर्ष 2024 में अपने कार्बन उत्सर्जन में 15 प्रतिशत की कमी की।")
    t = d.add_table(rows=2, cols=3)
    for r, row in enumerate([["मापदंड", "FY23", "FY24"], ["EBITDA मार्जिन", "16.9%", "18.2%"]]):
        for c, v in enumerate(row):
            t.cell(r, c).text = v
    f = tmp_path / "hindi.docx"
    d.save(f)
    doc = asyncio.run(parser.parse(f))
    assert doc.language == "hi" and doc.page_count is None
    assert doc.tables[0].grid()[1] == ["EBITDA मार्जिन", "16.9%", "18.2%"]
    chunks = asyncio.run(parser.chunk(doc, project_id="p", document_id="d", version=1))
    assert {c.language for c in chunks} == {"hi"}


def test_chunking_needs_the_docling_tree(parser, tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# A\n\nText.")
    doc = asyncio.run(parser.parse(f))
    doc.drop_native()
    with pytest.raises(IngestionError, match="parse the file again"):
        asyncio.run(parser.chunk(doc, project_id="p", document_id="d", version=1))
