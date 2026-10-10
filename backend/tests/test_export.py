"""Transcript export: Markdown and JSON for voice and text turns, interrupted answers, citations, Hindi, summaries."""

from __future__ import annotations

import json
import re
from urllib.parse import unquote

import pytest

from app.services.chat_summary import ChatSummarizer
from app.services.chats import ChatService
from app.services.export import ExportService, content_disposition, slugify
from app.services.messages import MessageService
from app.services.projects import ProjectService

from .revisit_helpers import ABSTAINED, ANSWERED, ScriptedLLM, Seeder, cite, make_fakes
from .test_chat_summary import reader

EN = "What was the EBITDA margin in FY24?"
P2 = cite("S1", "annual_report.pdf", 2)
P3 = cite("S2", "annual_report.pdf", 3, snippet="Margins by segment.")
DECK7 = cite("S1", "investor_deck.pdf", 7, document_id="doc_deck", snippet="Revenue +34%.")
HEARD = "The EBITDA margin in FY24 was 18.2% [S1]."
FULL = "The EBITDA margin in FY24 was 18.2% [S1][S2]. It improved from 16.9% [S2]."


@pytest.fixture
async def story(db, clock, load_local):
    """A chat with a voice turn whose answer was interrupted, a Hindi text turn and an abstained question, in a project
    called "Annual report FY24"; every timestamp comes from the deterministic clock."""

    class Story:
        projects = ProjectService(db, clock=clock)
        chats = ChatService(db, clock=clock)
        messages = MessageService(db, clock=clock)
        export = ExportService(db, clock=clock)

        async def build(self, title: str | None = "FY24 margins"):
            project = await self.projects.create("Annual report FY24")
            chat = await self.chats.create(project.id, title=title)
            m = self.messages
            await m.append(chat.id, role="user", text=EN, modality="voice", language="en")
            await m.append(
                chat.id,
                role="agent",
                text=FULL,
                heard_text=HEARD,
                modality="voice",
                language="en",
                citations=[P2, P3],
                route=ANSWERED,
                latency={"total_ms": 1234.5},
            )
            await m.append(chat.id, role="user", text="और राजस्व?", language="hi")
            await m.append(chat.id, role="agent", text="राजस्व 34% बढ़ा [S1]।", language="hi", citations=[DECK7])
            await m.append(chat.id, role="user", text="# CEO salary?\n```\nx", language="en")
            await m.append(chat.id, role="agent", text="I couldn't find that.", language="en", route=ABSTAINED)
            return project, chat

    s = Story()
    s.db, s.clock, s.settings = db, clock, load_local()  # type: ignore[attr-defined]
    return s


def stamp(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def untimed(text: str) -> str:
    """The text with every "2026-10-08 09:00:03 UTC" replaced by "<time>" (the clock ticks per service call; what the
    tests check is the layout, not how many calls the services make)."""
    return re.sub(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC", "<time>", text)


# ------------------------------------------------------------------ Markdown


async def test_markdown_transcript(story):
    _, chat = await story.build()
    file = await story.export.export(chat.id, "md")
    assert (file.filename, file.media_type) == ("fy24-margins-2026-10-08.md", "text/markdown; charset=utf-8")
    assert untimed(file.body.decode()) == (
        "# FY24 margins\n"
        "\n"
        "- **Project:** Annual report FY24\n"
        "- **Created:** <time>\n"
        "- **Languages:** English, Hindi\n"
        "- **Messages:** 6 (2 voice, 4 text)\n"
        "- **Exported:** <time> from Docent\n"
        "\n"
        "## Transcript\n"
        "\n"
        "### You · <time> · voice\n"
        "\n"
        "What was the EBITDA margin in FY24?\n"
        "\n"
        "### Docent · <time> · voice · interrupted\n"
        "\n"
        "> **Heard:** The EBITDA margin in FY24 was 18.2% [1].\n"
        "\n"
        "**Full answer:** The EBITDA margin in FY24 was 18.2% [1][2]. It improved from 16.9% [2].\n"
        "\n"
        "### You · <time> · text\n"
        "\n"
        "और राजस्व?\n"
        "\n"
        "### Docent · <time> · text\n"
        "\n"
        "राजस्व 34% बढ़ा [3]।\n"
        "\n"
        "### You · <time> · text\n"
        "\n"
        "\\# CEO salary?\n"
        "\\```\n"
        "x\n"
        "\n"
        "### Docent · <time> · text\n"
        "\n"
        "I couldn't find that.\n"
        "\n"
        "## Sources\n"
        "\n"
        "- [1] **annual_report.pdf**, page 2 — “EBITDA margin improved to 18.2% from 16.9%.” _(cited in message #2)_\n"
        "- [2] **annual_report.pdf**, page 3 — “Margins by segment.” _(cited in message #2)_\n"
        "- [3] **investor_deck.pdf**, page 7 — “Revenue +34%.” _(cited in message #4)_\n"
    )


async def test_heard_text_is_what_was_played_and_the_full_answer_is_kept(story):
    _, chat = await story.build()
    md = (await story.export.export(chat.id, "md")).body.decode()
    heard, full = md.index("**Heard:**"), md.index("**Full answer:**")
    assert heard < full and "16.9%" not in md[heard:full]  # the unheard part is only in the full answer
    assert md.count("interrupted") == 1  # only that one turn


async def test_a_page_cited_twice_is_one_numbered_source(story):
    project = await story.projects.create("P")
    chat = await story.chats.create(project.id, title="Same page")
    m = story.messages
    await m.append(chat.id, role="user", text="Q1")
    await m.append(chat.id, role="agent", text="A1 [S1]", citations=[P2])
    await m.append(chat.id, role="user", text="Q2")
    await m.append(chat.id, role="agent", text="A2 [S1]", citations=[cite("S1", page=2, chunk_id="another-chunk")])
    file = await story.export.export(chat.id, "md")
    md = file.body.decode()
    assert "A1 [1]" in md and "A2 [1]" in md and "[2]" not in md
    assert "_(cited in messages #2, #4)_" in md
    doc = json.loads((await story.export.export(chat.id, "json")).body)
    assert [(s["ref"], s["cited_by"]) for s in doc["sources"]] == [(1, [2, 4])]


async def test_markdown_with_a_summary_numbers_its_citations_like_the_transcript(story):
    _, chat = await story.build()
    llm = ScriptedLLM(json_reply=reader)
    await ChatSummarizer(story.db, llm=llm, settings=story.settings, clock=story.clock).generate(chat.id)
    md = (await story.export.export(chat.id, "md")).body.decode()

    summary = md[md.index("## Summary") : md.index("## Transcript")]
    assert summary == (
        "## Summary\n"
        "\n"
        "### Overview\n"
        "\n"
        "A conversation about the report.\n"
        "\n"
        "### Key points\n"
        "\n"
        "- राजस्व 34% बढ़ा। [3]\n"  # the interrupted answer is no source of facts (B7); numbers as in the export
        "\n"
        "### Questions the documents couldn't answer\n"
        "\n"
        "- \\# CEO salary? ``` x\n"
        "\n"
        "### Open follow-ups\n"
        "\n"
        "- Compare with FY23\n"
        "\n"
    )
    assert "Out of date" not in md  # the summary covers the whole chat


async def test_a_stale_summary_is_marked_in_the_export(story):
    _, chat = await story.build()
    llm = ScriptedLLM(json_reply=reader)
    await ChatSummarizer(story.db, llm=llm, settings=story.settings, clock=story.clock).generate(chat.id)
    await story.messages.append(chat.id, role="user", text="One more thing")
    md = (await story.export.export(chat.id, "md")).body.decode()
    assert "_Out of date: this summary covers messages 1\u20136 of 7._" in md
    doc = json.loads((await story.export.export(chat.id, "json")).body)
    assert (doc["summary"]["stale"], doc["summary"]["covers_seq"], doc["summary"]["message_count"]) == (True, 6, 7)


async def test_an_empty_chat_exports_its_title_and_facts(story):
    project = await story.projects.create("Annual report FY24")
    chat = await story.chats.create(project.id, title="Nothing yet")
    md = (await story.export.export(chat.id, "md")).body.decode()
    assert untimed(md) == (
        "# Nothing yet\n"
        "\n"
        "- **Project:** Annual report FY24\n"
        "- **Created:** <time>\n"
        "- **Languages:** —\n"
        "- **Messages:** 0 (0 voice, 0 text)\n"
        "- **Exported:** <time> from Docent\n"
        "\n"
        "## Transcript\n"
        "\n"
        "_No messages yet._\n"
    )
    doc = json.loads((await story.export.export(chat.id, "json")).body)
    assert (doc["messages"], doc["sources"], doc["summary"], doc["chat"]["languages"]) == ([], [], None, [])


async def test_a_chat_in_an_archived_project_exports(story):
    project, chat = await story.build()
    await story.projects.update(project.id, archived=True)
    md = (await story.export.export(chat.id, "md")).body.decode()
    assert "- **Project:** Annual report FY24 (archived)" in md
    doc = json.loads((await story.export.export(chat.id, "json")).body)
    assert doc["chat"]["project"] == {"id": project.id, "name": "Annual report FY24", "archived": True}


async def test_the_language_of_an_empty_chat_comes_from_its_settings(story):
    project = await story.projects.create("P")
    chat = await story.chats.create(project.id, title="Hindi chat", language="hi")
    assert "- **Languages:** Hindi" in (await story.export.export(chat.id, "md")).body.decode()


async def test_events_are_listed_in_italics(story):
    project = await story.projects.create("P")
    chat = await story.chats.create(project.id, title="Events")
    await story.messages.append(chat.id, role="event", text="Language switched to Hindi")
    md = (await story.export.export(chat.id, "md")).body.decode()
    assert "### Event · <time> · text\n\n_Language switched to Hindi_\n" in untimed(md)


# ------------------------------------------------------------------ JSON


async def test_json_schema(story):
    project, chat = await story.build()
    current = await story.chats.get(chat.id)
    file = await story.export.export(chat.id, "json")
    assert (file.filename, file.media_type) == ("fy24-margins-2026-10-08.json", "application/json")
    text = file.body.decode()
    assert "राजस्व 34% बढ़ा" in text  # readable, not \u-escaped
    doc = json.loads(text)

    assert list(doc) == ["schema_version", "exported_at", "chat", "messages", "sources", "summary"]
    assert doc["schema_version"] == 1 and doc["summary"] is None
    assert re.fullmatch(r"2026-10-08T09:\d\d:\d\dZ", doc["exported_at"])  # ISO 8601, UTC
    assert doc["chat"] == {
        "id": chat.id,
        "title": "FY24 margins",
        "title_is_auto": False,
        "project": {"id": project.id, "name": "Annual report FY24", "archived": False},
        "language": None,
        "languages": ["en", "hi"],
        "document_scope": None,
        "pinned": False,
        "archived": False,
        "message_count": 6,
        "created_at": stamp(chat.created_at),
        "updated_at": stamp(current.updated_at),
        "last_message_at": stamp(current.last_message_at),
    }

    user, agent, hindi_user, hindi_agent, _, abstained = doc["messages"]
    assert list(agent) == [
        "seq", "id", "role", "modality", "language", "created_at", "text", "heard_text", "interrupted",
        "citations", "route", "latency",
    ]  # fmt: skip
    assert (user["seq"], user["role"], user["modality"], user["language"], user["text"]) == (
        1,
        "user",
        "voice",
        "en",
        EN,
    )
    assert (user["heard_text"], user["interrupted"], user["citations"], user["route"]) == (None, False, [], None)
    assert (agent["role"], agent["modality"], agent["interrupted"]) == ("agent", "voice", True)
    assert (agent["text"], agent["heard_text"]) == (FULL, HEARD)  # raw text, markers as saved
    assert agent["route"]["abstained"] is False and agent["latency"] == {"total_ms": 1234.5}
    assert agent["citations"][0] == {
        "source_id": "S1",
        "ref": 1,
        "document_id": "doc_report",
        "filename": "annual_report.pdf",
        "page_start": 2,
        "page_end": 2,
        "chunk_id": "doc_report:v1:S1:2",
        "snippet": "EBITDA margin improved to 18.2% from 16.9%.",
        "section": None,
    }
    assert [c["ref"] for c in agent["citations"]] == [1, 2]
    assert (hindi_user["text"], hindi_user["language"]) == ("और राजस्व?", "hi")
    assert hindi_agent["text"] == "राजस्व 34% बढ़ा [S1]।" and hindi_agent["citations"][0]["source_id"] == "S1"
    assert hindi_agent["citations"][0]["ref"] == 3  # S1 of this answer is the third source of the chat
    assert abstained["route"]["abstained"] is True and abstained["citations"] == []

    assert doc["sources"] == [
        {
            "ref": 1,
            "document_id": "doc_report",
            "filename": "annual_report.pdf",
            "page_start": 2,
            "page_end": 2,
            "snippet": "EBITDA margin improved to 18.2% from 16.9%.",
            "cited_by": [2],
            "section": None,
        },
        {
            "ref": 2,
            "document_id": "doc_report",
            "filename": "annual_report.pdf",
            "page_start": 3,
            "page_end": 3,
            "snippet": "Margins by segment.",
            "cited_by": [2],
            "section": None,
        },
        {
            "ref": 3,
            "document_id": "doc_deck",
            "filename": "investor_deck.pdf",
            "page_start": 7,
            "page_end": 7,
            "snippet": "Revenue +34%.",
            "cited_by": [4],
            "section": None,
        },
    ]


async def test_json_includes_the_summary(story):
    _, chat = await story.build()
    llm = ScriptedLLM(json_reply=reader)
    stored = await ChatSummarizer(story.db, llm=llm, settings=story.settings, clock=story.clock).generate(chat.id)
    doc = json.loads((await story.export.export(chat.id, "json")).body)
    s = doc["summary"]
    assert list(s) == [
        "language", "overview", "key_points", "unanswered_questions", "follow_ups", "content", "covers_seq",
        "message_count", "stale", "model", "created_at",
    ]  # fmt: skip
    assert (s["language"], s["covers_seq"], s["message_count"], s["stale"]) == ("en", 6, 6, False)
    assert s["content"] == stored.content and s["unanswered_questions"] == [
        {"question": "# CEO salary?\n```\nx".replace("\n", " "), "message_seq": 5}
    ]
    # the interrupted answer is no source of facts (B7): the one key point is the Hindi answer's, numbered [3] as in
    # the transcript's sources list
    assert [p["sources"][0]["ref"] for p in s["key_points"]] == [3]


# ------------------------------------------------------------------ file names and headers


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("FY24 margins", "fy24-margins"),
        ("  Q3: Revenue & Costs (draft)!  ", "q3-revenue-costs-draft"),
        ("Résumé of the Café report", "résumé-of-the-café-report"),
        ("वित्त वर्ष 2024 का EBITDA मार्जिन", "वित्त-वर्ष-2024-का-ebitda-मार्जिन"),  # combining marks kept
        ("a/b\\c..d", "a-b-c-d"),
        ("!!!", ""),
        ("x" * 100, "x" * 60),
        ("word " * 20, "word-" * 11 + "word"),  # cut at a word boundary
    ],
)
def test_slugify(title, slug):
    assert slugify(title) == slug


def test_content_disposition_for_ascii_and_unicode_names():
    assert content_disposition("fy24-margins-2026-10-08.md", "fy24-margins-2026-10-08.md") == (
        'attachment; filename="fy24-margins-2026-10-08.md"'
    )
    header = content_disposition("वित्त-वर्ष-2026-10-08.md", "chat-2026-10-08.md")
    assert header.startswith("attachment; filename=\"chat-2026-10-08.md\"; filename*=UTF-8''")
    assert unquote(header.split("UTF-8''")[1]) == "वित्त-वर्ष-2026-10-08.md"
    assert header.isascii()


async def test_filenames_use_the_title_and_the_creation_date(story):
    project = await story.projects.create("P")
    hindi = await story.chats.create(project.id, title="वित्त वर्ष 2024")
    file = await story.export.export(hindi.id, "md")
    assert file.filename == "वित्त-वर्ष-2024-2026-10-08.md" and file.ascii_filename == "2024-2026-10-08.md"
    assert file.content_disposition.isascii() and "filename*=UTF-8''" in file.content_disposition

    symbols = await story.chats.create(project.id, title="???")
    assert (await story.export.export(symbols.id, "json")).filename == "chat-2026-10-08.json"

    mixed = await story.chats.create(project.id, title="Café margins")
    file = await story.export.export(mixed.id, "md")
    assert (file.filename, file.ascii_filename) == ("café-margins-2026-10-08.md", "cafe-margins-2026-10-08.md")


# ------------------------------------------------------------------ the endpoint


@pytest.fixture
def api_chat(make_app):
    with make_app(make_fakes(ScriptedLLM(json_reply=reader))) as api:
        seed = Seeder(api)
        project = seed.project()
        chat = seed.chat(project, title="FY24 margins")
        seed.ask(chat.id, EN, FULL, citations=[P2, P3], route=ANSWERED, heard_text=HEARD, modality="voice")
        seed.ask(chat.id, "और राजस्व?", "राजस्व 34% बढ़ा [S1]।", citations=[DECK7], route=ANSWERED, language="hi")
        yield api, seed, project, chat


def test_export_endpoint_serves_markdown_as_a_download(api_chat):
    api, _, _, chat = api_chat
    r = api.get(f"/api/chats/{chat.id}/export")  # md is the default
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/markdown; charset=utf-8"
    day = chat.created_at.strftime("%Y-%m-%d")
    assert r.headers["content-disposition"] == f'attachment; filename="fy24-margins-{day}.md"'
    assert r.headers["cache-control"] == "no-store"
    assert r.text.startswith("# FY24 margins\n") and "**Heard:** The EBITDA margin in FY24 was 18.2% [1]" in r.text
    assert api.get(f"/api/chats/{chat.id}/export", params={"format": "md"}).content == r.content


def test_export_endpoint_serves_json_as_a_download(api_chat):
    api, _, project, chat = api_chat
    r = api.get(f"/api/chats/{chat.id}/export", params={"format": "json"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    day = chat.created_at.strftime("%Y-%m-%d")
    assert r.headers["content-disposition"] == f'attachment; filename="fy24-margins-{day}.json"'
    doc = r.json()
    assert doc["schema_version"] == 1 and doc["chat"]["id"] == chat.id and doc["chat"]["project"]["id"] == project
    assert [m["modality"] for m in doc["messages"]] == ["voice", "voice", "text", "text"]
    assert doc["messages"][1]["interrupted"] is True and doc["messages"][3]["language"] == "hi"


def test_export_endpoint_with_a_hindi_title(api_chat):
    api, seed, project, _ = api_chat
    chat = seed.chat(project, title="राजस्व वृद्धि")
    r = api.get(f"/api/chats/{chat.id}/export")
    header = r.headers["content-disposition"]
    day = chat.created_at.strftime("%Y-%m-%d")
    assert header.startswith(f"attachment; filename=\"chat-{day}.md\"; filename*=UTF-8''")
    assert unquote(header.split("UTF-8''")[1]) == f"राजस्व-वृद्धि-{day}.md"


def test_export_endpoint_errors(api_chat):
    api, _, _, chat = api_chat
    assert api.get("/api/chats/cht_missing/export").status_code == 404
    assert api.get("/api/chats/cht_missing/export", params={"format": "json"}).status_code == 404
    assert api.get(f"/api/chats/{chat.id}/export", params={"format": "pdf"}).status_code == 422
    assert api.get(f"/api/chats/{chat.id}/export", params={"format": ""}).status_code == 422


def test_export_endpoint_for_an_empty_chat_and_an_archived_project(api_chat):
    api, seed, _, _ = api_chat
    project = seed.project("Old project", archived=True)
    chat = seed.chat(project, title="Empty")
    r = api.get(f"/api/chats/{chat.id}/export")
    assert r.status_code == 200 and "_No messages yet._" in r.text and "Old project (archived)" in r.text
    assert api.get(f"/api/chats/{chat.id}/export", params={"format": "json"}).json()["messages"] == []


def test_export_includes_the_summary_when_there_is_one(api_chat):
    api, _, _, chat = api_chat
    assert api.get(f"/api/chats/{chat.id}/export", params={"format": "json"}).json()["summary"] is None
    assert api.post(f"/api/chats/{chat.id}/summary").status_code == 200
    r = api.get(f"/api/chats/{chat.id}/export")
    assert "## Summary" in r.text and "A conversation about the report." in r.text
    assert api.get(f"/api/chats/{chat.id}/export", params={"format": "json"}).json()["summary"]["covers_seq"] == 4


def test_nothing_audio_is_in_the_export(api_chat):
    api, _, _, chat = api_chat
    body = api.get(f"/api/chats/{chat.id}/export", params={"format": "json"}).text.lower()
    assert "audio" not in body


def test_export_file_name_is_readable_from_the_frontend_origin(api_chat):
    """The frontend runs on another origin, so CORS must expose Content-Disposition or the browser hides the name."""
    api, _, _, chat = api_chat
    r = api.get(f"/api/chats/{chat.id}/export", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "content-disposition" in r.headers["access-control-expose-headers"].lower()
