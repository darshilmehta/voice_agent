"""Web sources in chat summaries and transcript exports (docs/DESIGN.md §3.7, §3.9): two results from one site stay
two sources, summaries name them "web: site — title", and their titles and links are escaped in Markdown."""

from __future__ import annotations

import re

from app.domain.projects import Citation
from app.domain.summaries import SourceRef
from app.services.chat_sources import ChatSources
from app.services.chat_summary import DraftPoint, _key_point, _legend
from app.services.export import ExportSource, _render_source


def _web(sid: str, url: str, title: str) -> Citation:
    return Citation(
        source_id=sid,
        document_id="",
        filename="livemint.com",
        page_start=None,
        page_end=None,
        chunk_id="",
        snippet="s",
        kind="web",
        url=url,
        title=title,
        site="livemint.com",
    )


def test_two_web_results_from_one_site_stay_two_sources_in_summaries():
    """F16: the summary legend names web results "web: site — title", and a key point citing the second result
    resolves to its URL (they used to collapse into the first one)."""
    sources = ChatSources()
    sources.register(
        4, [_web("W1", "https://livemint.com/a", "Stock up 2%"), _web("W2", "https://livemint.com/b", "Stock [3] down")]
    )
    assert _legend(sources, {1, 2}) == [
        "[1] web: livemint.com — Stock up 2%",
        "[2] web: livemint.com — Stock (3) down",  # a title can't plant a number of its own
    ]
    point = _key_point(DraftPoint(text="The stock is down", sources=[2]), sources)
    (ref,) = point.sources
    assert (ref.kind, ref.url) == ("web", "https://livemint.com/b")
    assert sources.find(ref).number == 2
    assert ref.model_dump()["url"] == "https://livemint.com/b"
    doc = SourceRef(document_id="d", filename="report.pdf", page_start=2, page_end=2)
    assert doc.model_dump() == {
        "document_id": "d",
        "filename": "report.pdf",
        "page_start": 2,
        "page_end": 2,
        "section": None,
    }


def test_web_titles_and_links_are_escaped_in_the_markdown_export():
    """F18: a web page's title and URL are text from the web: no Markdown of their own."""
    line = _render_source(
        ExportSource(
            ref=1,
            document_id="",
            filename="evil.example.com",
            page_start=None,
            page_end=None,
            snippet="click [here](http://x.example) **now**",
            cited_by=[2],
            kind="web",
            url="https://evil.example.com/a b<script>",
            title="**Win** [prize](javascript:alert(1)) <img src=x>",
        )
    )
    for markup in (r"\[prize", r"<img", r"\*\*Win", r"\[here"):  # none unescaped
        assert re.search(rf"(?<!\\){markup}", line) is None, markup
    assert "\\*\\*Win\\*\\* \\[prize\\](javascript:alert(1)) \\<img src=x\\>" in line
    assert "<https://evil.example.com/a%20b%3Cscript%3E>" in line
    js = _render_source(
        ExportSource(
            ref=2, document_id="", filename="x", page_start=None, page_end=None, snippet="", cited_by=[],
            kind="web", url="javascript:alert(1)", title="t",
        )
    )  # fmt: skip
    assert "javascript:" not in js
