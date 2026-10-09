"""POST /api/chats/{id}/messages: the SSE event stream of a text turn, with fake retrieval and LLM providers."""

from __future__ import annotations

import json
from functools import partial

import pytest
from fastapi.testclient import TestClient

from app.providers.llm import LLMUnavailableError
from app.providers.retrieval import RetrievalFilters
from app.services.messages import MessageService
from app.services.prompts import ABSTENTIONS, ANSWER_LENGTHS

REPORT = (
    "Annual Report 2024\n\nRevenue from operations grew 34% year on year.\f"
    "Key Metrics\n\n| Metric | FY23 | FY24 |\n| EBITDA margin | 16.9% | 18.2% |\n\n"
    "EBITDA margin improved to 18.2% from 16.9%."
)
EN = "What was the EBITDA margin in FY24?"
HI = "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?"


def _run(api: TestClient, fn, *args, **kwargs):
    return api.portal.call(partial(fn, *args, **kwargs))  # type: ignore[union-attr]


def drain(api: TestClient) -> None:
    _run(api, api.app.state.container["job_queue"].join)  # type: ignore[attr-defined]


def project_with_report(api: TestClient, name: str = "Annual report FY24", text: str = REPORT) -> tuple[str, str]:
    p = api.post("/api/projects", json={"name": name}).json()["id"]
    r = api.post(f"/api/projects/{p}/documents", files={"file": ("annual_report.txt", text.encode(), "text/plain")})
    assert r.status_code == 202, r.text
    drain(api)
    return p, r.json()["id"]


def new_chat(api: TestClient, project_id: str, **body) -> str:
    return api.post(f"/api/projects/{project_id}/chats", json=body).json()["id"]


def ask(api: TestClient, chat_id: str, text: str, language: str | None = None) -> list[tuple[str, dict]]:
    r = api.post(f"/api/chats/{chat_id}/messages", json={"text": text, "language": language})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert r.headers["cache-control"] == "no-cache"
    return parse_sse(r.text)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Strict reader for the contract: every event is exactly ``event:`` + one ``data:`` JSON line + blank line."""
    assert body.endswith("\n\n")
    events = []
    for block in body[:-2].split("\n\n"):
        name, data = block.split("\n")
        assert name.startswith("event: ") and data.startswith("data: ")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def names(events) -> list[str]:
    return [e for e, _ in events]


def payload(events, name: str) -> dict:
    (data,) = [d for e, d in events if e == name]
    return data


def answer_text(events) -> str:
    return "".join(d["text"] for e, d in events if e == "delta")


def source_with(events, needle: str) -> dict:
    return next(s for s in payload(events, "sources")["sources"] if needle in s["snippet"])


# ------------------------------------------------------------------ answers


def test_answer_streams_in_contract_order_and_cites_only_what_it_uses(app, fakes):
    p, doc_id = project_with_report(app)
    chat = new_chat(app, p)
    fakes.llm.reply = lambda messages: (
        "The EBITDA margin in FY24 was 18.2% "
        + "".join(f"[{s}]" for s in _ids_containing(messages[-1].content, "18.2%"))
        + " [S9], up from the year before as revenue grew faster than costs."
    )
    events = ask(app, chat, EN)

    assert names(events)[:2] == ["user_message", "sources"] and names(events)[-1] == "agent_message"
    assert set(names(events)[2:-1]) == {"delta"} and len(events) > 4
    user, agent = payload(events, "user_message"), payload(events, "agent_message")
    transcript = app.get(f"/api/chats/{chat}/messages").json()["items"]
    assert transcript == [user, agent]  # the events carry the saved messages, same shape as the transcript
    assert (user["role"], user["text"], user["language"], user["citations"]) == ("user", EN, "en", [])

    sources = payload(events, "sources")
    assert sources["abstained"] is False
    assert set(sources["confidence"]) == {"top_score", "gap", "dense_similarity", "above_threshold"}
    assert sources["confidence"]["above_threshold"] is True
    assert [s["source_id"] for s in sources["sources"]] == [f"S{i}" for i in range(1, len(sources["sources"]) + 1)]
    for s in sources["sources"]:
        assert set(s) == {
            "source_id",
            "document_id",
            "filename",
            "page_start",
            "page_end",
            "chunk_id",
            "snippet",
            "section",
        }
        assert s["document_id"] == doc_id and s["filename"] == "annual_report.txt" and len(s["snippet"]) <= 300

    table, sentence = source_with(events, "| EBITDA margin"), source_with(events, "improved to 18.2%")
    cited = f"[{table['source_id']}][{sentence['source_id']}]"
    tail = ", up from the year before as revenue grew faster than costs."
    assert answer_text(events).endswith(f"18.2% {cited} [S9]{tail}")  # raw stream, as generated
    assert agent["text"] == f"The EBITDA margin in FY24 was 18.2% {cited}{tail}"  # unknown [S9] removed
    assert agent["citations"] == [table, sentence]  # only cited sources, in order of mention
    assert table["page_start"] == table["page_end"] == 2
    assert (agent["role"], agent["language"], agent["modality"]) == ("agent", "en", "text")
    assert agent["route"]["abstained"] is False and agent["route"]["intent"] == "document_qa"
    assert agent["route"]["stopped"] is False and agent["route"]["length"] == "short"
    assert user["modality"] == agent["modality"] == "text"
    assert agent["route"]["prompt"] == "answer-v2" and agent["route"]["model"] == "qwen3:4b-instruct"
    assert {"retrieval_ms", "rerank_ms", "first_delta_ms", "llm_ms", "total_ms"} <= set(agent["latency"])

    (call,) = fakes.llm.calls
    system, prompt = call["messages"][0], call["messages"][-1]
    assert system.role == "system" and "[S1]" in system.content
    assert prompt.role == "user" and f"Question: {EN}" in prompt.content and "Answer in English" in prompt.content
    assert "[S1] annual_report.txt · page " in prompt.content and "| EBITDA margin | 16.9% | 18.2% |" in prompt.content
    assert call["max_tokens"] == ANSWER_LENGTHS["short"].max_tokens  # the text endpoint asks for short answers
    assert ANSWER_LENGTHS["short"].instruction in system.content


def _ids_containing(prompt: str, needle: str) -> list[str]:
    blocks = prompt.split("\n\n[")
    return [b.split("]")[0].lstrip("[") for b in blocks if needle in b and b.lstrip("[").startswith("S")]


def test_citation_lists_are_normalized_and_bogus_ids_dropped(app, fakes):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    fakes.llm.reply = "Margin 18.2% [S1, s2] and growth 34% [S1]. Unknown [S42]"
    agent = payload(ask(app, chat, EN), "agent_message")
    assert agent["text"] == "Margin 18.2% [S1][S2] and growth 34% [S1]. Unknown"
    assert [c["source_id"] for c in agent["citations"]] == ["S1", "S2"]


def test_answer_without_citations_is_saved_without_citations(app, fakes):
    p, _ = project_with_report(app)
    fakes.llm.reply = "It was 18.2%."
    agent = payload(ask(app, new_chat(app, p), EN), "agent_message")
    assert agent["text"] == "It was 18.2%." and agent["citations"] == []


def test_recent_messages_are_sent_as_context_without_old_markers(app, fakes):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    fakes.llm.reply = "It was 18.2% [S1]."
    ask(app, chat, EN)
    ask(app, chat, "And in FY23?")
    history = [(m.role, m.content) for m in fakes.llm.calls[1]["messages"][1:-1]]
    assert history == [("user", EN), ("assistant", "It was 18.2%.")]


# ------------------------------------------------------------------ abstention


def test_abstains_when_the_documents_do_not_cover_it(app, fakes):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    events = ask(app, chat, "What is the CEO's salary?")
    assert names(events) == ["user_message", "sources", "delta", "agent_message"]
    sources = payload(events, "sources")
    assert sources["abstained"] is True and sources["sources"] == []
    assert sources["confidence"]["above_threshold"] is False and sources["confidence"]["top_score"] < 0.3
    agent = payload(events, "agent_message")
    assert agent["text"] == answer_text(events) == ABSTENTIONS["not_covered"]["en"]
    assert agent["citations"] == [] and agent["route"]["abstained"] is True
    assert agent["route"]["abstain_reason"] == "not_covered" and agent["route"]["model"] is None
    assert agent["route"]["stopped"] is False
    assert fakes.llm.calls == []  # no model, so no invented facts


def test_hindi_abstention_is_in_hindi(app, fakes):
    p, _ = project_with_report(app)
    fakes.reranker.scorer = lambda q, passage: 0.01
    agent = payload(ask(app, new_chat(app, p), "कंपनी के सीईओ का वेतन कितना है?"), "agent_message")
    assert agent["text"] == ABSTENTIONS["not_covered"]["hi"] and agent["language"] == "hi"


def test_chat_without_ready_documents_abstains_without_searching(app, fakes):
    p = app.post("/api/projects", json={"name": "Empty"}).json()["id"]
    events = ask(app, new_chat(app, p), EN)
    assert payload(events, "sources") == {
        "sources": [],
        "confidence": {"top_score": 0.0, "gap": 0.0, "dense_similarity": 0.0, "above_threshold": False},
        "abstained": True,
    }
    assert payload(events, "agent_message")["text"] == ABSTENTIONS["no_documents"]["en"]
    assert payload(events, "agent_message")["route"]["abstain_reason"] == "no_documents"
    assert fakes.embedder.calls == [] and fakes.store.searches == []


# ------------------------------------------------------------------ language


def test_hindi_question_gets_a_hindi_answer(app, fakes):
    p, _ = project_with_report(app)
    fakes.reranker.scorer = lambda q, passage: 0.9 if "18.2%" in passage else 0.05
    fakes.llm.reply = "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था [S1]।"
    events = ask(app, new_chat(app, p), HI)
    assert payload(events, "user_message")["language"] == "hi"
    agent = payload(events, "agent_message")
    assert agent["language"] == "hi" and agent["citations"][0]["source_id"] == "S1"
    system, question = fakes.llm.calls[0]["messages"][0].content, fakes.llm.calls[0]["messages"][-1].content
    assert "Answer in Hindi, in Devanagari script" in question and "EBITDA or FY24 exactly as written" in system
    assert fakes.reranker.calls[-1][0] == HI  # phase 1: no router, so no English query yet


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        (EN, None, "en"),
        (HI, None, "hi"),
        ("FY24 में EBITDA margin क्या था?", None, "hi"),  # Hindi grammar, English terms
        ("What is the मार्जिन in FY24?", None, "en"),
        (EN, "hi", "hi"),  # asked for explicitly
        (HI, "en", "en"),
    ],
)
def test_answer_language(app, fakes, text, language, expected):
    p, _ = project_with_report(app)
    fakes.reranker.scorer = lambda q, passage: 0.9
    payload(ask(app, new_chat(app, p), text, language), "agent_message")
    name = {"en": "English", "hi": "Hindi"}[expected]
    assert f"Answer in {name}" in fakes.llm.calls[0]["messages"][-1].content


def test_text_without_letters_uses_the_chat_language(app, fakes):
    p, _ = project_with_report(app)
    chat = new_chat(app, p, language="hi")
    fakes.reranker.scorer = lambda q, passage: 0.9
    fakes.llm.reply = "EBITDA मार्जिन 18.2% था [S1]।"
    agent = payload(ask(app, chat, "18.2%?"), "agent_message")
    assert agent["language"] == "hi" and agent["route"]["language"] == "hi"
    assert "Answer in Hindi" in fakes.llm.calls[-1]["messages"][-1].content


# ------------------------------------------------------------------ scope


def test_retrieval_stays_within_the_project_and_the_chats_ready_documents(app, fakes):
    p, report = project_with_report(app)
    other_project, other_doc = project_with_report(app, "Other", "Other company EBITDA margin was 99% in FY24.")
    deck = app.post(f"/api/projects/{p}/documents", files={"file": ("deck.md", b"# Deck\n\nEBITDA margin 18.2%")})
    drain(app)
    fakes.parser.fail_with = RuntimeError("broken file")
    failed = app.post(f"/api/projects/{p}/documents", files={"file": ("bad.txt", b"EBITDA margin FY24 broken")})
    drain(app)
    fakes.parser.fail_with = None
    deck_id, failed_id = deck.json()["id"], failed.json()["id"]

    events = ask(app, new_chat(app, p), EN)
    assert fakes.store.searches[-1][1] == RetrievalFilters(p, (report, deck_id))  # READY only, this project only
    cited_docs = {s["document_id"] for s in payload(events, "sources")["sources"]}
    assert cited_docs <= {report, deck_id} and other_doc not in cited_docs and failed_id not in cited_docs

    scoped = new_chat(app, p, document_scope=[deck_id, failed_id])
    events = ask(app, scoped, EN)
    assert fakes.store.searches[-1][1] == RetrievalFilters(p, (deck_id,))
    assert {s["document_id"] for s in payload(events, "sources")["sources"]} == {deck_id}

    events = ask(app, new_chat(app, other_project), EN)
    assert {s["document_id"] for s in payload(events, "sources")["sources"]} == {other_doc}


# ------------------------------------------------------------------ failures


def test_llm_down_is_an_error_event_and_the_user_message_stays(app, fakes):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    fakes.llm.fail_with = LLMUnavailableError("Ollama unreachable at http://127.0.0.1:11434 (ConnectError)")
    events = ask(app, chat, EN)
    assert names(events) == ["user_message", "sources", "error"]
    assert payload(events, "error") == {
        "detail": "answer generation failed: LLMUnavailableError: Ollama unreachable at http://127.0.0.1:11434 "
        "(ConnectError)",
        "stage": "llm",
    }
    assert [m["role"] for m in app.get(f"/api/chats/{chat}/messages").json()["items"]] == ["user"]


def test_llm_failing_mid_answer_ends_the_stream_without_saving_a_partial_answer(app, fakes):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    fakes.llm.reply = "The margin was 18.2% according to [S1]."
    fakes.llm.fail_after = 2
    events = ask(app, chat, EN)
    # the answer's first words are held while they are checked (its passages are strong: services/answer_guard.py)
    assert names(events) == ["user_message", "sources", "error"]
    assert payload(events, "error")["stage"] == "llm"
    assert app.get(f"/api/chats/{chat}/messages").json()["total"] == 1


def test_empty_answer_is_an_error(app, fakes):
    p, _ = project_with_report(app)
    fakes.llm.reply = "  [S99] "
    events = ask(app, new_chat(app, p), EN)
    assert names(events)[-1] == "error" and payload(events, "error")["stage"] == "llm"


def test_retrieval_failure_is_an_error_event(app, fakes):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    fakes.store.fail_with = ConnectionError("Qdrant unreachable")
    events = ask(app, chat, EN)
    assert names(events) == ["user_message", "error"]
    assert payload(events, "error") == {
        "detail": "document search failed: ConnectionError: Qdrant unreachable",
        "stage": "retrieval",
    }
    assert app.get(f"/api/chats/{chat}/messages").json()["total"] == 1


def test_saving_the_answer_failing_is_a_storage_error(app, fakes, monkeypatch):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    real_append = MessageService.append

    async def append(self, chat_id, *, role, **kw):
        if role == "agent":
            raise RuntimeError("database is locked")
        return await real_append(self, chat_id, role=role, **kw)

    monkeypatch.setattr(MessageService, "append", append)
    events = ask(app, chat, EN)
    assert names(events)[-1] == "error"
    assert payload(events, "error") == {
        "detail": "could not save the answer: RuntimeError: database is locked",
        "stage": "storage",
    }


# ------------------------------------------------------------------ request validation


def test_request_validation_happens_before_the_stream(app):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    url = f"/api/chats/{chat}/messages"
    for body in (
        {"text": ""},
        {"text": "   "},
        {"text": "x" * 4001},
        {"text": "hi", "language": "fr"},
        {"text": "hi", "mode": "voice"},
        {},
    ):
        r = app.post(url, json=body)
        assert r.status_code == 422, body
        assert r.headers["content-type"] == "application/json"
    assert app.post("/api/chats/cht_nope/messages", json={"text": "hi"}).status_code == 404
    assert app.post(url, json={"text": "x" * 4000}).status_code == 200
    assert app.get(f"/api/chats/{chat}/messages").json()["total"] == 2  # only the valid one (+ its answer)


def test_legacy_free_form_citations_are_read_tolerantly(app):
    p, _ = project_with_report(app)
    chat = new_chat(app, p)
    db = app.app.state.container["metadata_db"]  # type: ignore[attr-defined]
    legacy = [{"document_id": "doc_1", "page": 46, "chunk_id": "c9"}, "garbage", {"source_id": "S3", "page_start": 2}]
    _run(app, MessageService(db).append, chat, role="agent", text="old answer", citations=legacy)
    (item,) = app.get(f"/api/chats/{chat}/messages").json()["items"]
    assert item["citations"] == [
        {
            "source_id": "S1",
            "document_id": "doc_1",
            "filename": "",
            "page_start": 46,
            "page_end": 46,
            "chunk_id": "c9",
            "snippet": "",
            "section": None,
        },
        {
            "source_id": "S3",
            "document_id": "",
            "filename": "",
            "page_start": 2,
            "page_end": 2,
            "chunk_id": "",
            "snippet": "",
            "section": None,
        },
    ]
