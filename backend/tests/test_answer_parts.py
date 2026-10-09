"""Pure pieces of the chat pipeline: language choice, numbered sources (dedupe, budget, sections), citations."""

from __future__ import annotations

import pytest

from app.domain.projects import Citation, coerce_citations
from app.services.language import choose_language, message_language
from app.services.retrieval import RankedChunk
from app.services.sources import build_sources, estimate_tokens, finalize_answer, snippet, strip_markers

from .fakes import make_chunk

FILES = {"doc1": "annual_report.pdf", "doc2": "deck.pptx"}


def ranked(*chunks, scores=None) -> list[RankedChunk]:
    scores = scores or [1.0 - i * 0.1 for i in range(len(chunks))]
    return [RankedChunk(c, s, 0.1, 0.7, i + 1) for i, (c, s) in enumerate(zip(chunks, scores, strict=True))]


# ------------------------------------------------------------------ language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What was the EBITDA margin in FY24?", "en"),
        ("वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?", "hi"),
        ("FY24 में EBITDA margin क्या था?", "hi"),  # tie → Hindi: the grammar is Hindi
        ("What is the मार्जिन in FY24?", "en"),
        ("EBITDA margin kya tha?", "hi"),  # romanized Hindi (Hinglish) is Hindi
        ("What was the revenue yaar?", "en"),  # one stray Hindi word doesn't make it Hindi
        ("18.2%?", None),
        ("", None),
    ],
)
def test_message_language(text, expected):
    assert message_language(text) == expected


def test_choose_language_prefers_the_request_then_the_script_then_the_fallback():
    assert choose_language("What was it?", "hi", "en") == "hi"
    assert choose_language("यह क्या है?", None, "en") == "hi"
    assert choose_language("123", None, "hi") == "hi"
    assert choose_language("123", None, "en") == "en"


# ------------------------------------------------------------------ sources


def test_sources_are_numbered_grouped_by_section_and_cite_their_chunk():
    a0 = make_chunk(0, heading_path=["Results"], page_start=2, page_end=2, text="table: EBITDA margin 18.2%")
    a1 = make_chunk(1, heading_path=["Results"], page_start=2, page_end=3, text="Margin improved to 18.2%.")
    b0 = make_chunk(0, document_id="doc2", heading_path=["Highlights"], page_start=None, page_end=None, text="Deck")
    a5 = make_chunk(5, heading_path=["Risks"], page_start=3, page_end=3, text="Single supplier risk.")
    sources = build_sources(ranked(a1, b0, a0, a5), FILES, budget_tokens=3000)
    # groups by best rank: Results (a1 best) → Highlights → Risks; reading order inside Results (a0 before a1)
    assert [(s.source_id, s.chunk.chunk_id) for s in sources] == [
        ("S1", a0.chunk_id),
        ("S2", a1.chunk_id),
        ("S3", b0.chunk_id),
        ("S4", a5.chunk_id),
    ]
    assert sources[1].header() == "[S2] annual_report.pdf · pages 2-3 · Results"
    assert sources[2].header() == "[S3] deck.pptx · Highlights"
    c = sources[0].citation()
    assert c == Citation(
        source_id="S1",
        document_id="doc1",
        filename="annual_report.pdf",
        page_start=2,
        page_end=2,
        chunk_id=a0.chunk_id,
        snippet="table: EBITDA margin 18.2%",
        section="Results",
    )


def test_sources_skip_unready_documents_and_duplicates():
    a = make_chunk(0, text="EBITDA margin 18.2%")
    same_text = make_chunk(3, text="  EBITDA   margin 18.2% ")
    gone = make_chunk(0, document_id="doc_deleted", text="orphan vector")
    sources = build_sources(ranked(a, a, same_text, gone), FILES, budget_tokens=3000)
    assert [s.chunk.chunk_id for s in sources] == [a.chunk_id]


def test_budget_keeps_the_best_chunks_and_cuts_an_oversized_first_one():
    big = make_chunk(0, text="word " * 2000)
    small = make_chunk(1, text="EBITDA margin 18.2%")
    sources = build_sources(ranked(big, small), FILES, budget_tokens=300)
    assert [s.chunk.chunk_index for s in sources] == [0]  # the best chunk, cut to fit; nothing else fits
    assert estimate_tokens(sources[0].chunk.text) <= 300 and sources[0].chunk.text.endswith(" …")


def test_budget_skips_a_chunk_that_does_not_fit_but_keeps_smaller_later_ones():
    first = make_chunk(0, text="a " * 400)
    huge = make_chunk(1, text="b " * 4000)
    small = make_chunk(2, text="EBITDA margin 18.2%")
    sources = build_sources(ranked(first, huge, small), FILES, budget_tokens=500)
    assert {s.chunk.chunk_index for s in sources} == {0, 2}


def test_estimate_tokens_counts_devanagari_heavier():
    assert estimate_tokens("a" * 35) == 10
    assert estimate_tokens("क" * 10) == 9


def test_snippet():
    assert snippet("  short\n text ") == "short text"
    long = snippet("word " * 200)
    assert len(long) <= 300 and long.endswith("…") and not long.endswith(" …")


# ------------------------------------------------------------------ citations


def _sources(n: int):
    return build_sources(ranked(*(make_chunk(i, text=f"passage {i}") for i in range(n))), FILES, budget_tokens=3000)


def test_finalize_keeps_valid_markers_in_order_of_mention():
    text, cites = finalize_answer("Revenue grew 34% [S2]. Margin was 18.2% [S1][S2].", _sources(3))
    assert text == "Revenue grew 34% [S2]. Margin was 18.2% [S1][S2]."
    assert [c.source_id for c in cites] == ["S2", "S1"]


def test_finalize_normalizes_lists_and_case_and_drops_unknown_ids():
    text, cites = finalize_answer("A [S1, S3] b [s2;S9] c [S7]. d [S 1]", _sources(3))
    assert text == "A [S1][S3] b [S2] c. d [S 1]"  # "[S 1]" isn't a marker: left as text
    assert [c.source_id for c in cites] == ["S1", "S3", "S2"]


def test_short_answers_get_the_best_few_sources():
    """§9.5: a voice-length answer gets at most ``max_sources`` passages (best first, then grouped as usual)."""
    chunks = [make_chunk(i, text=f"passage {i}", heading_path=[f"H{i}"]) for i in range(5)]
    assert [s.chunk.text for s in build_sources(ranked(*chunks), FILES, budget_tokens=3000)] == [
        f"passage {i}" for i in range(5)
    ]
    short = build_sources(ranked(*chunks), FILES, budget_tokens=3000, max_sources=3)
    assert [(s.source_id, s.chunk.text) for s in short] == [
        ("S1", "passage 0"),
        ("S2", "passage 1"),
        ("S3", "passage 2"),
    ]


def test_a_short_answers_budget_never_cuts_its_best_passage():
    table = make_chunk(0, text="| row | value |\n" * 120)  # ~550 estimated tokens: over the voice budget
    small = [make_chunk(i, text=f"short passage {i}") for i in (1, 2)]
    cut = build_sources(ranked(table, *small), FILES, budget_tokens=300)
    assert len(cut) == 1 and cut[0].chunk.text.endswith(" …")  # without a separate allowance: cut to fit
    whole = build_sources(ranked(table, *small), FILES, budget_tokens=300, max_sources=3, best_budget_tokens=3000)
    assert whole[0].chunk.text == table.text  # kept whole; the voice budget (now spent) is for the others
    assert len(whole) == 1


def test_tables_are_compacted_in_the_prompt():
    from app.services.sources import Source, compact_tables, format_sources

    table = (
        "| Metric                            | FY24   | FY23   |\n"
        "|-----------------------------------|--------|--------|\n"
        "| Revenue from operations (₹ crore) | 7,365  | 6,482  |\n"
        "Note: figures are consolidated.   "
    )
    assert compact_tables(table) == (
        "| Metric | FY24 | FY23 |\n|---|---|---|\n| Revenue from operations (₹ crore) | 7,365 | 6,482 |\n"
        "Note: figures are consolidated.   "  # not a table row: left alone
    )
    assert compact_tables("a | b, not a table") == "a | b, not a table"
    source = Source("S1", make_chunk(0, text=table), "annual_report.pdf", 0.9)
    assert "| 7,365 | 6,482 |" in format_sources([source]) and "-----" not in format_sources([source])


def test_finalize_with_no_sources_removes_every_marker():
    assert finalize_answer("Nothing [S1].", []) == ("Nothing.", [])


def test_strip_markers_for_history():
    assert strip_markers("It was 18.2% [S1][S2], up from 16.9% [S3].") == "It was 18.2%, up from 16.9%."


def test_coerce_citations_reads_old_rows():
    old = [
        {"document_id": "doc_1", "page": 46, "chunk_id": "c9"},
        {"document_id": "doc_2", "page_start": "3", "page_end": 4, "text": "x" * 500},
        "not a citation",
        None,
    ]
    first, second = coerce_citations(old)
    assert (first.source_id, first.page_start, first.page_end, first.filename) == ("S1", 46, 46, "")
    assert (second.source_id, second.page_start, second.page_end, len(second.snippet)) == ("S2", 3, 4, 300)
    assert coerce_citations({"not": "a list"}) == []
