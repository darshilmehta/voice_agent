"""User-facing chat summaries: generation, idempotence, staleness, grounding, unanswered questions, map-reduce."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from pydantic import BaseModel

from app.domain.projects import Message
from app.providers.llm import LLMError, LLMMessage, LLMUnavailableError
from app.services.chat_sources import ChatSources, number_markers, pages_label
from app.services.chat_summary import (
    dominant_language,
    ground,
    pack,
    prepare,
    render_summary_markdown,
    window_budget,
)
from app.services.messages import MessageService
from app.services.revisit_prompts import MAX_KEY_POINTS, NO_ANSWER_LINE, DraftPoint, SummaryDraft
from app.services.sources import estimate_tokens
from app.services.summaries import SummaryService

from .revisit_helpers import (
    ABSTAINED,
    ANSWERED,
    REDUCE_MARK,
    SUMMARY_MARK,
    ScriptedLLM,
    Seeder,
    cite,
    draft,
    make_fakes,
)

EN = "What was the EBITDA margin in FY24?"
HI = "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?"
EBITDA = "The EBITDA margin in FY24 was 18.2% [S1][S2]."
REVENUE = "Revenue grew 34% [S1]."
P2 = cite("S1", "annual_report.pdf", 2, document_id="doc_report")
P3 = cite("S2", "annual_report.pdf", 3, document_id="doc_report", snippet="Margins by segment.")
DECK7 = cite("S1", "investor_deck.pdf", 7, document_id="doc_deck", snippet="Revenue +34%.")

_MARKS = re.compile(r"((?:\[\d+\])+)(\W*)$")


def split_marks(text: str) -> tuple[str, list[int]]:
    """ "Margin 18.2% [1][2]" → ("Margin 18.2%", [1, 2])."""
    m = _MARKS.search(text)
    numbers = [int(n) for n in re.findall(r"\[(\d+)\]", m.group(1))] if m else []
    return (text[: m.start()].rstrip() + m.group(2) if m else text).strip(), numbers


def reader(messages: list[LLMMessage], schema: type[BaseModel]) -> SummaryDraft:
    """A stand-in summariser that "reads" its prompt: one key point per cited assistant answer, citing what the answer
    cites; given partial summaries it keeps their key points. Deterministic, so tests can check what comes out."""
    user = messages[-1].content
    points: list[tuple[str, list[int]]] = []
    if user.startswith("Partial summaries"):
        section = ""
        for line in user.splitlines():
            if line.startswith("Key points"):
                section = "points"
            elif line.startswith(("Follow-ups", "Part ", "Overview", "Sources")):
                section = ""
            elif section == "points" and line.startswith("- "):
                points.append(split_marks(line[2:]))
    else:
        for line in user.splitlines():
            m = re.match(r"#\d+ Assistant(?: \([^)]*\))?: (.*)", line)
            if m and NO_ANSWER_LINE not in m.group(1):
                points.append(split_marks(m.group(1)))
    return draft("A conversation about the report.", points, ["Compare with FY23"])


@pytest.fixture
def env(make_app):
    llm = ScriptedLLM(json_reply=reader)
    with make_app(make_fakes(llm)) as api:
        seed = Seeder(api)
        project = seed.project()
        yield api, llm, seed, project


def url(chat_id: str) -> str:
    return f"/api/chats/{chat_id}/summary"


def two_answers(seed: Seeder, project: str, **chat_kw: Any) -> str:
    chat = seed.chat(project, **chat_kw).id
    seed.ask(chat, EN, EBITDA, citations=[P2, P3], route=ANSWERED, modality="voice")
    seed.ask(chat, "And revenue?", REVENUE, citations=[DECK7], route=ANSWERED)
    return chat


# ------------------------------------------------------------------ generation


def test_summary_has_overview_key_points_with_pages_and_follow_ups(env):
    api, _, seed, project = env
    chat = two_answers(seed, project)

    r = api.post(url(chat))
    assert r.status_code == 200, r.text
    s = r.json()
    assert set(s) == {
        "id", "chat_id", "language", "overview", "key_points", "unanswered_questions", "follow_ups",
        "content", "covers_seq", "message_count", "stale", "model", "created_at",
    }  # fmt: skip
    assert (s["chat_id"], s["language"], s["stale"]) == (chat, "en", False)
    assert (s["covers_seq"], s["message_count"], s["model"]) == (4, 4, "qwen3:4b-instruct")
    assert s["overview"] == "A conversation about the report." and s["follow_ups"] == ["Compare with FY23"]
    assert s["unanswered_questions"] == []

    first, second = s["key_points"]
    assert first["text"] == "The EBITDA margin in FY24 was 18.2%."
    assert first["sources"] == [  # the answer's [S1][S2] as documents and pages, not markers
        {"document_id": "doc_report", "filename": "annual_report.pdf", "page_start": 2, "page_end": 2},
        {"document_id": "doc_report", "filename": "annual_report.pdf", "page_start": 3, "page_end": 3},
    ]
    assert second["sources"] == [
        {"document_id": "doc_deck", "filename": "investor_deck.pdf", "page_start": 7, "page_end": 7}
    ]

    assert s["content"] == (
        "## Overview\n\nA conversation about the report.\n\n"
        "## Key points\n\n"
        "- The EBITDA margin in FY24 was 18.2%. (annual_report.pdf, p. 2; annual_report.pdf, p. 3)\n"
        "- Revenue grew 34%. (investor_deck.pdf, p. 7)\n\n"
        "## Open follow-ups\n\n- Compare with FY23"
    )
    assert api.get(url(chat)).json() == s  # stored


def test_the_prompt_numbers_sources_across_the_whole_chat(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    api.post(url(chat))

    (call,) = llm.json_calls
    system, user = call
    assert "Write everything in English" in system.content and SUMMARY_MARK in system.content
    assert user.content == (
        "Conversation:\n"
        f"#1 User: {EN}\n"
        "#2 Assistant: The EBITDA margin in FY24 was 18.2% [1][2].\n"  # [S1][S2] of answer 2 → chat-wide numbers
        "#3 User: And revenue?\n"
        "#4 Assistant: Revenue grew 34% [3].\n"  # this answer's own [S1] is another document: number 3
        "\nSources:\n"
        "[1] annual_report.pdf, p. 2\n"
        "[2] annual_report.pdf, p. 3\n"
        "[3] investor_deck.pdf, p. 7"
    )


def test_a_page_cited_by_several_answers_is_one_source(env):
    api, llm, seed, project = env
    chat = seed.chat(project).id
    seed.ask(chat, EN, "18.2% [S1].", citations=[P2], route=ANSWERED)
    seed.ask(
        chat, "Again?", "Yes, 18.2% [S1].", citations=[cite("S1", page=2, chunk_id="another-chunk")], route=ANSWERED
    )
    api.post(url(chat))
    user = llm.json_calls[0][1].content
    assert "#2 Assistant: 18.2% [1]." in user and "#4 Assistant: Yes, 18.2% [1]." in user
    assert user.count("[1] annual_report.pdf, p. 2") == 1 and "[2]" not in user


def test_b7_an_interrupted_answer_is_no_source_of_facts(env):
    """Found in the real run: a heard "Product…" fragment became "no product depends on a single supplier"."""
    api, llm, seed, project = env
    chat = seed.chat(project).id
    seed.ask(
        chat, EN, "The margin was 18.2% [S1]. It rose from 16.9% [S2].", citations=[P2, P3], route=ANSWERED,
        heard_text="The margin was 18.2% [S1].", modality="voice",
    )  # fmt: skip
    api.post(url(chat))
    user = llm.json_calls[0][1].content
    assert "#2 Assistant: (cut off by the user before it finished: incomplete, not an answer)" in user
    assert "18.2%" not in user and "16.9%" not in user and "[1]" not in user


def test_events_are_not_part_of_the_summary(env):
    api, llm, seed, project = env
    chat = seed.chat(project).id
    seed.say(chat, "event", "user interrupted the answer")
    seed.ask(chat, EN, EBITDA, citations=[P2, P3], route=ANSWERED)
    api.post(url(chat))
    assert "interrupted" not in llm.json_calls[0][1].content


def test_unanswered_questions_are_the_abstained_turns(env):
    api, llm, seed, project = env
    chat = seed.chat(project).id
    seed.ask(chat, EN, EBITDA, citations=[P2, P3], route=ANSWERED)
    seed.ask(chat, "What is the CEO's salary?", "I couldn't find that in this chat's documents.", route=ABSTAINED)
    seed.ask(chat, "What is the CEO's  salary?", "I couldn't find that in this chat's documents.", route=ABSTAINED)
    seed.ask(chat, "Who audits the company?", "I couldn't find that in this chat's documents.", route=ABSTAINED)

    s = api.post(url(chat)).json()
    assert s["unanswered_questions"] == [
        {"question": "What is the CEO's salary?", "message_seq": 3},  # the repeat is listed once
        {"question": "Who audits the company?", "message_seq": 7},
    ]
    assert [p["text"] for p in s["key_points"]] == ["The EBITDA margin in FY24 was 18.2%."]
    assert (
        "## Questions the documents couldn't answer\n\n- What is the CEO's salary?\n- Who audits the company?"
        in s["content"]
    )
    user = llm.json_calls[0][1].content
    assert f"#4 Assistant: {NO_ANSWER_LINE}" in user and "couldn't find that" not in user  # the apology isn't a fact


# ------------------------------------------------------------------ idempotence and staleness


def test_asking_again_for_an_unchanged_chat_returns_the_stored_summary(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    first = api.post(url(chat)).json()
    again = api.post(url(chat)).json()
    assert again == first and again["id"] == first["id"]
    assert len(llm.json_calls) == 1  # the model ran once


def test_new_messages_make_the_summary_out_of_date_and_posting_refreshes_it(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    first = api.post(url(chat)).json()
    assert (first["covers_seq"], first["message_count"], first["stale"]) == (4, 4, False)

    seed.ask(chat, "Any debt?", "Net debt fell [S1].", citations=[cite("S1", page=5)], route=ANSWERED)
    old = api.get(url(chat)).json()
    assert (old["covers_seq"], old["message_count"], old["stale"]) == (4, 6, True)  # UI: "summary is out of date"
    assert old["id"] == first["id"] and len(llm.json_calls) == 1  # reading never regenerates

    fresh = api.post(url(chat)).json()
    assert (fresh["covers_seq"], fresh["message_count"], fresh["stale"]) == (6, 6, False)
    assert len(llm.json_calls) == 2 and "Net debt fell" in fresh["content"]
    assert api.get(url(chat)).json() == fresh


def test_a_message_arriving_during_generation_leaves_the_summary_stale(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    llm.before_json = lambda: MessageService(seed.db).append(chat, role="user", text="a late question")

    s = api.post(url(chat)).json()
    assert (s["covers_seq"], s["message_count"], s["stale"]) == (4, 5, True)  # covers what the model saw, no more


def test_concurrent_requests_run_the_model_once(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    llm.delay = 0.0
    summarizer = api.app.state.summarizer

    async def both():
        return await asyncio.gather(summarizer.generate(chat), summarizer.generate(chat))

    a, b = seed.run(both)
    assert a.id == b.id and len(llm.json_calls) == 1


def test_a_memory_summary_is_not_the_user_summary(env):
    api, _, seed, project = env
    chat = two_answers(seed, project)
    seed.run(SummaryService(seed.db).save, chat, kind="memory", content="internal context")
    assert api.get(url(chat)).status_code == 404
    s = api.post(url(chat)).json()
    assert "internal context" not in s["content"]
    assert seed.run(SummaryService(seed.db).get, chat, "memory").content == "internal context"  # untouched


# ------------------------------------------------------------------ grounding


def test_citations_that_point_to_nothing_are_dropped(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)  # sources 1, 2, 3 exist
    llm.json_reply = lambda m, s: draft(
        "Overview [9].",
        [
            ("Margin was 18.2%", [1, 99, 2]),  # 99: no such source
            ("Revenue grew 34% [3] and more [8]", []),  # markers the model wrote inside the text
            ("Nothing cited", [0, -1, 42]),
        ],
        ["Check [5]"],
    )
    s = api.post(url(chat)).json()
    sources = [[(r["filename"], r["page_start"]) for r in p["sources"]] for p in s["key_points"]]
    assert sources == [
        [("annual_report.pdf", 2), ("annual_report.pdf", 3)],
        [("investor_deck.pdf", 7)],  # [3] moved from the text to the sources, [8] dropped
        [],
    ]
    assert [p["text"] for p in s["key_points"]] == ["Margin was 18.2%", "Revenue grew 34% and more", "Nothing cited"]
    assert s["overview"] == "Overview."
    assert s["follow_ups"] == ["Check"]
    assert "99" not in s["content"] and "[" not in s["content"]


def test_a_number_that_was_not_in_the_text_the_model_saw_is_dropped(env):
    """Map-reduce: every window's model also cites source 1, which only the first window contained."""
    api, llm, seed, project = env
    chat = seed.chat(project).id
    for i in range(6):
        seed.ask(
            chat,
            f"Question {i} " + "padding " * 60,
            f"Answer {i} [S1].",
            citations=[cite("S1", page=i + 1)],
            route=ANSWERED,
        )  # source number i + 1 is page i + 1
    api.app.state.summarizer.window_tokens = 300

    def cheating(messages: list[LLMMessage], schema: type[BaseModel]) -> SummaryDraft:
        if REDUCE_MARK in messages[0].content:
            return reader(messages, schema)
        user = messages[-1].content
        first = re.search(r"^#(\d+) ", user, re.MULTILINE).group(1)  # type: ignore[union-attr]
        legend = [int(n) for n in re.findall(r"^\[(\d+)\] ", user, re.MULTILINE)]
        return draft("Part.", [(f"Window from #{first}", [*legend, 1])])

    llm.json_reply = cheating
    s = api.post(url(chat)).json()
    assert len(map_calls(llm)) >= 3
    pages = [[r["page_start"] for r in p["sources"]] for p in s["key_points"]]
    assert pages[0][0] == 1  # the window that contained source 1 may cite it
    assert all(1 not in p for p in pages[1:]) and all(p for p in pages)  # the others lost the stranger, kept their own


# ------------------------------------------------------------------ long chats


def long_chat(seed: Seeder, project: str, turns: int) -> str:
    chat = seed.chat(project).id
    for i in range(turns):
        seed.ask(
            chat,
            f"Question {i}: " + "tell me more about this topic " * 4,
            f"Finding {i}: " + "something notable happened here " * 4 + "[S1].",
            citations=[cite("S1", "annual_report.pdf", i % 5 + 1, document_id="doc_report")],
            route=ANSWERED,
        )
    return chat


def map_calls(llm: ScriptedLLM) -> list[list[LLMMessage]]:
    return [c for c in llm.json_calls if SUMMARY_MARK in c[0].content]


def reduce_calls(llm: ScriptedLLM) -> list[list[LLMMessage]]:
    return [c for c in llm.json_calls if REDUCE_MARK in c[0].content]


def test_a_long_chat_is_summarised_in_windows_and_combined(env):
    api, llm, seed, project = env
    chat = long_chat(seed, project, turns=30)  # 60 messages ≈ 4k tokens
    summarizer = api.app.state.summarizer
    summarizer.window_tokens = 1200

    s = api.post(url(chat)).json()
    maps, reduces = map_calls(llm), reduce_calls(llm)
    assert len(maps) >= 3 and len(reduces) == 1
    assert llm.json_calls[-1] == reduces[0]  # map first, then reduce

    # every message is in exactly one window, each window fits the budget
    seen: list[int] = []
    for _, user in maps:
        lines = [ln for ln in user.content.split("\n\nSources:")[0].splitlines()[1:]]
        assert sum(estimate_tokens(ln) + 4 for ln in lines) <= summarizer.window_tokens
        seen += [int(re.match(r"#(\d+) ", ln).group(1)) for ln in lines]  # type: ignore[union-attr]
    assert seen == list(range(1, 61))

    # the reduce call saw all partial summaries, and the result is grounded in real pages
    assert reduces[0][1].content.count("Part ") == len(maps)
    assert s["covers_seq"] == s["message_count"] == 60 and not s["stale"]
    assert 1 <= len(s["key_points"]) <= MAX_KEY_POINTS
    assert all(
        r["filename"] == "annual_report.pdf" and 1 <= r["page_start"] <= 5
        for p in s["key_points"]
        for r in p["sources"]
    )
    stored = Seeder(api).run(SummaryService(seed.db).get, chat, "user")
    assert stored is not None and stored.data["windows"] == len(maps)  # type: ignore[index]


def test_many_windows_are_combined_in_rounds(env):
    api, llm, seed, project = env
    chat = long_chat(seed, project, turns=40)
    api.app.state.summarizer.window_tokens = 330  # tiny: even the partial summaries need several rounds

    s = api.post(url(chat)).json()
    maps, reduces = map_calls(llm), reduce_calls(llm)
    assert len(maps) >= 8 and len(reduces) >= 3  # groups, then groups of groups, then the final one
    assert llm.json_calls[-1] == reduces[-1]
    assert s["key_points"] and s["covers_seq"] == 80


def test_a_chat_that_fits_one_window_needs_no_reduce(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    api.post(url(chat))
    assert len(map_calls(llm)) == 1 and reduce_calls(llm) == []


def test_window_budget_follows_the_context_size():
    assert window_budget(8192) == int((8192 - 3000) * 0.9)
    assert window_budget(2048) == 800  # never below a usable minimum


def test_pack_groups_consecutive_items_within_budget():
    sizes = [3, 3, 3, 10, 1, 1]
    assert pack(sizes, lambda s: s, 7) == [[3, 3], [3], [10], [1, 1]]
    assert pack(sizes, lambda s: s, 7, min_items=2) == [[3, 3], [3, 10], [1, 1]]  # an oversize item is never alone
    assert pack([], lambda s: s, 7) == []


# ------------------------------------------------------------------ language


def test_language_is_the_dominant_one_or_requested(env):
    api, llm, seed, project = env
    chat = seed.chat(project).id
    seed.ask(chat, HI, "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था [S1]।", citations=[P2], route=ANSWERED, language="hi")
    seed.ask(chat, "And revenue?", "Revenue grew 34% [S1].", citations=[DECK7], route=ANSWERED, language="en")
    seed.ask(chat, "और लागत?", "लागत 12% बढ़ी [S1]।", citations=[P3], route=ANSWERED, language="hi")
    llm.json_reply = lambda m, s: draft("सारांश।", [("EBITDA मार्जिन 18.2% था", [1])], ["FY23 से तुलना करें"])

    s = api.post(url(chat)).json()
    assert s["language"] == "hi"  # four Hindi messages against two English
    system = llm.json_calls[0][0].content
    assert "Write everything in Hindi, in Devanagari script" in system and "EBITDA or FY24 exactly as written" in system
    assert s["content"].startswith(
        "## सारांश\n\nसारांश।\n\n## मुख्य बिंदु\n\n- EBITDA मार्जिन 18.2% था (annual_report.pdf, पृ. 2)"
    )
    assert s["content"].endswith("## आगे के प्रश्न\n\n- FY23 से तुलना करें")

    assert len(llm.json_calls) == 1
    assert api.post(url(chat), params={"language": "hi"}).json() == s  # already in Hindi: stored one
    assert len(llm.json_calls) == 1

    llm.json_reply = lambda m, sc: draft("Overview.", [("Margin was 18.2%", [1])])
    english = api.post(url(chat), params={"language": "en"}).json()  # another language: regenerated
    assert english["language"] == "en" and "Write everything in English" in llm.json_calls[1][0].content
    assert english["content"].startswith("## Overview")
    assert api.get(url(chat)).json() == english


def test_dominant_language_ties_go_to_the_chat_language():
    def msg(language: str | None) -> Message:
        return Message.model_construct(language=language)  # type: ignore[arg-type]

    assert dominant_language([msg("hi"), msg("en")], "hi", "en") == "hi"
    assert dominant_language([msg("hi"), msg("en")], None, "en") == "en"
    assert dominant_language([msg("hi"), msg("hi"), msg("en")], "en", "en") == "hi"
    assert dominant_language([msg(None)], None, "hi") == "hi"
    assert dominant_language([], "fr", "en") == "en"


# ------------------------------------------------------------------ errors


def test_unknown_chat_and_missing_summary_are_404(env):
    api, _, seed, project = env
    assert api.get(url("cht_missing")).status_code == 404
    assert api.post(url("cht_missing")).status_code == 404
    chat = seed.chat(project).id
    r = api.get(url(chat))
    assert r.status_code == 404 and chat in r.json()["detail"]


def test_an_empty_chat_has_nothing_to_summarise(env):
    api, llm, seed, project = env
    chat = seed.chat(project).id
    r = api.post(url(chat))
    assert r.status_code == 422 and "no messages" in r.json()["detail"]
    seed.say(chat, "event", "something happened")
    assert api.post(url(chat)).status_code == 422
    assert llm.json_calls == []


def test_unsupported_language_is_422(env):
    api, _, seed, project = env
    chat = two_answers(seed, project)
    assert api.post(url(chat), params={"language": "fr"}).status_code == 422


def test_model_down_is_503_and_nothing_is_stored(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    llm.json_fail_with, llm.json_fail_times = LLMUnavailableError("Ollama unreachable"), -1
    r = api.post(url(chat))
    assert r.status_code == 503 and "Ollama unreachable" in r.json()["detail"]
    assert len(llm.json_calls) == 1  # no retry against a server that is down
    assert api.get(url(chat)).status_code == 404

    llm.json_fail_with = None
    assert api.post(url(chat)).status_code == 200  # works once it is back


def test_invalid_model_output_is_retried_once(env):
    api, llm, seed, project = env
    chat = two_answers(seed, project)
    llm.json_fail_with, llm.json_fail_times = LLMError("model output doesn't match SummaryDraft"), 1
    assert api.post(url(chat)).status_code == 200 and len(llm.json_calls) == 2

    chat2 = two_answers(seed, project)
    llm.json_fail_times = 2
    r = api.post(url(chat2))
    assert r.status_code == 503 and "doesn't match SummaryDraft" in r.json()["detail"]
    assert api.get(url(chat2)).status_code == 404


# ------------------------------------------------------------------ building blocks


def test_markers_are_renumbered_chat_wide():
    sources = ChatSources()
    sources.register(2, [cite("S1", page=2), cite("S2", page=3)])
    mapping = sources.register(4, [cite("S1", "deck.pdf", 7, document_id="d2"), cite("S2", page=2, chunk_id="other")])
    assert mapping == {"S1": 3, "S2": 1}  # S2 of the second answer is page 2 of the report again
    assert number_markers("A [S1][S2] B [S1, S2] C [S3] D [S2][S2]", mapping) == "A [3][1] B [3][1] C D [1]"


def test_pages_label():
    assert pages_label(2, 2) == "page 2" and pages_label(2, None, short=True) == "p. 2"
    assert pages_label(2, 4) == "pages 2\u20134" and pages_label(2, 4, short=True) == "pp. 2\u20134"
    assert pages_label(None, None) == "" and pages_label(3, 3, "hi") == "पृष्ठ 3"


def test_ground_clamps_and_cleans_a_draft():
    long_text = "word " * 200
    out = ground(
        SummaryDraft(
            overview=long_text,
            key_points=[DraftPoint(text=f"Point {i}", sources=[1, 2]) for i in range(12)]
            + [DraftPoint(text="point 3", sources=[1]), DraftPoint(text="  ", sources=[1])],
            follow_ups=["a", "a", "b", "c", "d", ""],
        ),
        allowed={2},
    )
    assert len(out.key_points) == MAX_KEY_POINTS and all(p.sources == [2] for p in out.key_points)
    assert len(out.overview) <= 700 and out.overview.endswith("…")
    assert out.follow_ups == ["a", "b", "c"]


def test_draft_point_accepts_the_ways_a_model_writes_numbers():
    assert DraftPoint.model_validate({"text": "x", "sources": [1, "2", "[3]", "S4", "x", None, 5.0]}).sources == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert DraftPoint.model_validate({"text": "x", "sources": "3"}).sources == []


def test_prepare_reports_abstentions_and_skips_events():
    def m(seq: int, role: str, text: str, **kw: Any) -> Message:
        base: dict[str, Any] = {
            "id": f"m{seq}",
            "chat_id": "c",
            "seq": seq,
            "role": role,
            "modality": "text",
            "text": text,
            "heard_text": None,
            "language": "en",
            "citations": [],
            "route": None,
            "latency": None,
            "created_at": "2026-10-08T09:00:00Z",
        }
        return Message.model_validate({**base, **kw})

    prepared = prepare(
        [
            m(1, "event", "joined"),
            m(2, "user", "Q1"),
            m(3, "agent", "No.", route={"abstained": True}),
            m(4, "agent", "unprompted abstention", route={"abstained": True}),  # same question: listed once
        ]
    )
    assert [ln.text for ln in prepared.lines] == [
        "#2 User: Q1",
        f"#3 Assistant: {NO_ANSWER_LINE}",
        f"#4 Assistant: {NO_ANSWER_LINE}",
    ]
    assert [(q.question, q.message_seq) for q in prepared.unanswered] == [("Q1", 2)]


def test_markdown_rendering_of_a_summary_without_sections():
    from app.domain.summaries import SummaryData

    data = SummaryData(
        language="en", overview="Short.", key_points=[], unanswered_questions=[], follow_ups=[], windows=1, prompt="x"
    )
    assert render_summary_markdown(data) == "## Overview\n\nShort."
    assert render_summary_markdown(data, level=3) == "### Overview\n\nShort."


def test_prepare_lists_questions_the_documents_did_not_cover():
    """A mixed question the documents didn't cover is answered from general knowledge (``general_note``): the
    answer stays in the transcript, marked as not from the documents, and its question is listed as unanswered."""

    def m(seq: int, role: str, text: str, **kw: Any) -> Message:
        base: dict[str, Any] = {
            "id": f"m{seq}",
            "chat_id": "c",
            "seq": seq,
            "role": role,
            "modality": "text",
            "text": text,
            "heard_text": None,
            "language": "en",
            "citations": [],
            "route": None,
            "latency": None,
            "created_at": "2026-10-08T09:00:00Z",
        }
        return Message.model_validate({**base, **kw})

    prepared = prepare(
        [
            m(1, "user", "How does our margin compare with the industry?"),
            m(
                2,
                "agent",
                "Industry margins are usually 12-15%.",
                route={"answer": "general", "general_note": "not_covered", "abstained": False},
            ),
            m(3, "user", "What is the capital of France?"),
            m(4, "agent", "Paris.", route={"answer": "general", "abstained": False}),
        ]
    )
    assert [ln.text for ln in prepared.lines] == [
        "#1 User: How does our margin compare with the industry?",
        "#2 Assistant (general knowledge, not from the documents): Industry margins are usually 12-15%.",
        "#3 User: What is the capital of France?",
        "#4 Assistant (general knowledge, not from the documents): Paris.",
    ]
    # Only the document question that went uncovered is "unanswered"; a general question answered as such isn't.
    assert [(q.question, q.message_seq) for q in prepared.unanswered] == [
        ("How does our margin compare with the industry?", 1)
    ]


# ------------------------------------------------------------------ quality round: B8, B9


def test_b8_repeated_points_merge_and_non_answers_lose_their_points_and_chips():
    raw = SummaryDraft(
        overview="About the report.",
        key_points=[
            DraftPoint(text="The EBITDA margin in FY24 was 18.2%.", sources=[1]),
            DraftPoint(text="In FY24 the EBITDA margin was 18.2%", sources=[2]),  # the same, in other words
            DraftPoint(text="The report does not provide the CEO's salary.", sources=[3]),  # a non-answer, with a chip
            DraftPoint(text="रिपोर्ट में CEO के वेतन की जानकारी नहीं दी गई है।", sources=[3]),
            DraftPoint(text="Revenue grew 34% in FY24.", sources=[2]),
        ],
        follow_ups=[],
    )
    grounded = ground(raw, {1, 2, 3})
    assert [(p.text, p.sources) for p in grounded.key_points] == [
        ("The EBITDA margin in FY24 was 18.2%.", [1, 2]),
        ("Revenue grew 34% in FY24.", [2]),
    ]


def test_b8_a_hindi_summary_written_in_english_is_asked_for_once_more(env):
    api, llm, seed, project = env
    chat = seed.chat(project, language="hi").id
    seed.ask(chat, HI, "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था [S1]।", citations=[P2], route=ANSWERED, language="hi")
    replies = [
        draft("A conversation about the margin.", [("The EBITDA margin was 18.2%", [1])]),
        draft("मार्जिन के बारे में बातचीत।", [("EBITDA मार्जिन 18.2% था", [1])]),
    ]
    llm.json_reply = lambda m, s: replies.pop(0)
    s = api.post(url(chat), params={"language": "hi"}).json()
    assert s["overview"] == "मार्जिन के बारे में बातचीत।" and s["key_points"][0]["text"] == "EBITDA मार्जिन 18.2% था"
    first, second = llm.json_calls
    assert "(Write everything in Hindi" in first[-1].content  # asked last, in the conversation's message too
    assert second[-1].content.endswith("हिंदी (देवनागरी) में लिखें।")  # then insisting
    llm.json_reply = lambda m, sc: draft("Still English.", [("Margin 18.2%", [1])])
    seed.ask(chat, "और?", "बस इतना ही [S1]।", citations=[P2], route=ANSWERED, language="hi")
    s = api.post(url(chat), params={"language": "hi"}).json()  # still English after the second try: it stands
    assert s["overview"] == "Still English." and len(llm.json_calls) == 4


def test_b9_an_answer_that_says_the_documents_dont_cover_it_is_unanswered(env):
    """A grounded answer that passed the gate but says the documents don't cover the question is an abstention: it
    is listed among the unanswered questions, and makes no key point."""
    api, _, seed, project = env
    chat = seed.chat(project).id
    seed.ask(chat, EN, EBITDA, citations=[P2, P3], route=ANSWERED)
    soft = {**ABSTAINED, "abstained_by": "answer"}
    seed.ask(chat, "What is the FY25 margin?", "The documents do not specify the FY25 margin.", route=soft)
    s = api.post(url(chat)).json()
    assert [q["question"] for q in s["unanswered_questions"]] == ["What is the FY25 margin?"]
    assert [p["text"] for p in s["key_points"]] == ["The EBITDA margin in FY24 was 18.2%."]
