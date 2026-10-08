"""Service layer (repositories) on a real, freshly migrated SQLite database under tmp_path."""

from __future__ import annotations

import asyncio

import pytest

from app.db import models as orm
from app.services.base import InvalidInput, NotFound
from app.services.chats import DEFAULT_TITLE, ChatService
from app.services.documents import DocumentService
from app.services.messages import MessageService
from app.services.pins import PinService
from app.services.projects import ProjectService
from app.services.summaries import SummaryService

from .conftest import add_document, count_rows


@pytest.fixture
def svc(db, clock):
    class Services:
        projects = ProjectService(db, clock=clock)
        chats = ChatService(db, clock=clock)
        messages = MessageService(db, clock=clock)
        summaries = SummaryService(db, clock=clock)
        pins = PinService(db, clock=clock)
        documents = DocumentService(db, clock=clock)

    return Services


# ------------------------------------------------------------------ projects


async def test_create_and_get_project(svc):
    p = await svc.projects.create("  Annual report FY24 ", "  ")
    assert p.id.startswith("prj_")
    assert p.name == "Annual report FY24"
    assert p.description is None
    assert not p.pinned and not p.archived
    assert p.created_at == p.updated_at
    assert await svc.projects.get(p.id) == p


async def test_project_name_must_not_be_empty(svc):
    with pytest.raises(InvalidInput, match="name"):
        await svc.projects.create("   ")


async def test_projects_list_pinned_first_then_recent_activity(svc):
    a = await svc.projects.create("A")
    b = await svc.projects.create("B")
    c = await svc.projects.create("C")
    d = await svc.projects.create("D")
    await svc.projects.update(b.id, pinned=True)
    await svc.projects.update(d.id, pinned=True)  # pinned last → shown first
    await svc.chats.create(a.id)  # activity moves A above C
    assert [p.name for p in await svc.projects.list()] == ["D", "B", "A", "C"]
    pins = await svc.pins.list()
    assert [p.name for p in pins.projects] == ["D", "B"]
    assert c.id not in {p.id for p in pins.projects}


async def test_pin_and_archive_are_idempotent_and_not_activity(svc):
    p = await svc.projects.create("A")
    pinned = await svc.projects.update(p.id, pinned=True)
    again = await svc.projects.update(p.id, pinned=True)
    assert again.pinned_at == pinned.pinned_at  # re-pinning keeps the original position
    assert again.updated_at == p.updated_at
    unpinned = await svc.projects.update(p.id, pinned=False)
    assert unpinned.pinned_at is None


async def test_rename_and_describe_project(svc):
    p = await svc.projects.create("Draft", "old")
    renamed = await svc.projects.update(p.id, name="Vendor contracts")
    assert renamed.name == "Vendor contracts" and renamed.description == "old"
    assert renamed.updated_at > p.updated_at
    cleared = await svc.projects.update(p.id, description=None)
    assert cleared.description is None and cleared.name == "Vendor contracts"


async def test_archived_projects_are_hidden_by_default(svc):
    keep = await svc.projects.create("Keep")
    old = await svc.projects.create("Old")
    await svc.projects.update(old.id, archived=True, pinned=True)
    assert [p.id for p in await svc.projects.list()] == [keep.id]
    assert {p.id for p in await svc.projects.list(include_archived=True)} == {keep.id, old.id}
    assert (await svc.pins.list()).projects == []
    restored = await svc.projects.update(old.id, archived=False)
    assert not restored.archived and restored.pinned


async def test_project_counts(svc, db):
    p = await svc.projects.create("A")
    await svc.chats.create(p.id)
    await svc.chats.create(p.id)
    await add_document(db, p.id)
    got = await svc.projects.get(p.id)
    assert (got.chat_count, got.document_count) == (2, 1)


async def test_unknown_project_is_not_found(svc):
    for call in (
        svc.projects.get("prj_nope"),
        svc.projects.update("prj_nope", name="x"),
        svc.projects.delete("prj_nope"),
        svc.chats.create("prj_nope"),
        svc.chats.list("prj_nope"),
        svc.documents.list("prj_nope"),
    ):
        with pytest.raises(NotFound, match="prj_nope"):
            await call


async def test_deleting_a_project_deletes_everything_inside_it(svc, db):
    doomed = await svc.projects.create("Doomed")
    kept = await svc.projects.create("Kept")
    doc = await add_document(db, doomed.id)
    await add_document(db, kept.id)
    for project in (doomed, kept):
        chat = await svc.chats.create(project.id)
        await svc.messages.append(chat.id, role="user", text="hi")
        await svc.messages.append(chat.id, role="agent", text="hello")
        await svc.summaries.save(chat.id, kind="user", content="greetings")
        await svc.summaries.save(chat.id, kind="memory", content="said hi")

    assert await svc.projects.delete(doomed.id) == [doc]

    expected = {
        orm.Project: 1,
        orm.Document: 1,
        orm.DocumentVersion: 1,
        orm.IngestionJob: 1,
        orm.Chat: 1,
        orm.Message: 2,
        orm.ChatSummary: 2,
    }
    assert {m: await count_rows(db, m) for m in expected} == expected
    assert [p.id for p in await svc.projects.list()] == [kept.id]


# ------------------------------------------------------------------ chats


async def test_create_chat_defaults(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    assert chat.id.startswith("cht_")
    assert (chat.title, chat.title_is_auto) == (DEFAULT_TITLE, True)
    assert chat.document_scope is None and chat.message_count == 0 and chat.last_message_at is None
    named = await svc.chats.create(p.id, title=" FY24 margins ", language="hi")
    assert (named.title, named.title_is_auto, named.language) == ("FY24 margins", False, "hi")
    assert (await svc.projects.get(p.id)).updated_at == named.created_at  # creating a chat is project activity


async def test_rename_chat_makes_the_title_the_users(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    renamed = await svc.chats.update(chat.id, title="Risk section")
    assert (renamed.title, renamed.title_is_auto) == ("Risk section", False)
    assert await svc.chats.get(chat.id) == renamed


async def test_chats_list_newest_activity_first(svc):
    p = await svc.projects.create("A")
    first = await svc.chats.create(p.id, title="first")
    second = await svc.chats.create(p.id, title="second")
    third = await svc.chats.create(p.id, title="third")
    assert [c.title for c in await svc.chats.list(p.id)] == ["third", "second", "first"]
    await svc.messages.append(first.id, role="user", text="back to this one")
    assert [c.title for c in await svc.chats.list(p.id)] == ["first", "third", "second"]
    await svc.chats.update(third.id, archived=True)
    assert [c.id for c in await svc.chats.list(p.id)] == [first.id, second.id]
    assert len(await svc.chats.list(p.id, include_archived=True)) == 3


async def test_document_scope(svc, db):
    p = await svc.projects.create("A")
    other = await svc.projects.create("B")
    d1 = await add_document(db, p.id, "a.pdf")
    d2 = await add_document(db, p.id, "b.pdf")
    foreign = await add_document(db, other.id, "c.pdf")

    chat = await svc.chats.create(p.id, document_scope=[d2, d1, d2])
    assert chat.document_scope == [d2, d1]  # de-duplicated, order kept
    assert (await svc.chats.update(chat.id, document_scope=None)).document_scope is None  # back to all
    with pytest.raises(InvalidInput, match=foreign):  # documents never leak across projects
        await svc.chats.update(chat.id, document_scope=[d1, foreign])
    with pytest.raises(InvalidInput, match="at least one"):
        await svc.chats.update(chat.id, document_scope=[])
    assert (await svc.chats.get(chat.id)).document_scope is None  # failed updates changed nothing


async def test_deleting_a_chat_deletes_its_messages_and_summaries_only(svc, db):
    p = await svc.projects.create("A")
    doomed = await svc.chats.create(p.id)
    kept = await svc.chats.create(p.id)
    for chat in (doomed, kept):
        await svc.messages.append(chat.id, role="user", text="q")
        await svc.summaries.save(chat.id, kind="user", content="s")
    await svc.chats.delete(doomed.id)
    assert [c.id for c in await svc.chats.list(p.id)] == [kept.id]
    assert await count_rows(db, orm.Message) == 1
    assert await count_rows(db, orm.ChatSummary) == 1
    with pytest.raises(NotFound):
        await svc.chats.get(doomed.id)
    with pytest.raises(NotFound):
        await svc.chats.delete(doomed.id)


async def test_pinned_chats_for_the_sidebar(svc):
    a = await svc.projects.create("Annual report FY24")
    b = await svc.projects.create("Vendor contracts")
    margins = await svc.chats.create(a.id, title="FY24 margins")
    risks = await svc.chats.create(a.id, title="Risks")
    clauses = await svc.chats.create(b.id, title="Clauses")
    await svc.chats.create(a.id, title="not pinned")
    for chat in (risks, margins, clauses):
        await svc.chats.update(chat.id, pinned=True)
    pins = await svc.pins.list()
    assert [(c.title, c.project_name) for c in pins.chats] == [
        ("Clauses", "Vendor contracts"),
        ("FY24 margins", "Annual report FY24"),
        ("Risks", "Annual report FY24"),
    ]
    await svc.chats.update(risks.id, archived=True)
    await svc.projects.update(b.id, archived=True)  # hides its chats from the pins too
    assert [c.title for c in (await svc.pins.list()).chats] == ["FY24 margins"]
    await svc.chats.update(margins.id, pinned=False)
    assert (await svc.pins.list()).chats == []


# ------------------------------------------------------------------ messages


async def test_append_numbers_messages_and_records_activity(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    m1 = await svc.messages.append(chat.id, role="user", text="What was FY24 revenue?", modality="voice", language="en")
    m2 = await svc.messages.append(
        chat.id,
        role="agent",
        text="FY24 revenue was ₹4,210 crore, up 12% on FY23.",
        heard_text="FY24 revenue was ₹4,210 crore,",
        language="en",
        citations=[{"document_id": "doc_1", "page": 46, "chunk_id": "c9"}],
        route={"intent": "document_qa", "needs_retrieval": True},
        latency={"stt_ms": 469, "router_ms": 1490},
    )
    assert (m1.seq, m2.seq) == (1, 2)
    assert m1.id.startswith("msg_") and m1.citations == [] and not m1.interrupted
    assert m2.interrupted and (m2.citations[0].page_start, m2.citations[0].page_end) == (46, 46)  # legacy shape
    chat = await svc.chats.get(chat.id)
    assert chat.message_count == 2 and chat.last_message_at == m2.created_at
    assert (await svc.projects.get(p.id)).updated_at == m2.created_at
    page = await svc.messages.list(chat.id)
    assert page.items == [m1, m2]
    assert (page.total, page.has_more, page.next_cursor) == (2, False, None)


async def test_append_validates_input(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    with pytest.raises(InvalidInput, match="role"):
        await svc.messages.append(chat.id, role="robot", text="beep")  # type: ignore[arg-type]
    with pytest.raises(InvalidInput, match="modality"):
        await svc.messages.append(chat.id, role="user", text="hi", modality="smoke")  # type: ignore[arg-type]
    with pytest.raises(NotFound):
        await svc.messages.append("cht_nope", role="user", text="hi")
    assert (await svc.chats.get(chat.id)).message_count == 0


async def test_transcript_pages_forward_and_backward(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    for i in range(1, 8):
        await svc.messages.append(chat.id, role="user" if i % 2 else "agent", text=f"m{i}")

    def texts(page):
        return [m.text for m in page.items]

    forward = await svc.messages.list(chat.id, limit=3)
    assert texts(forward) == ["m1", "m2", "m3"] and forward.has_more and forward.next_cursor == 3
    forward = await svc.messages.list(chat.id, after=forward.next_cursor, limit=3)
    assert texts(forward) == ["m4", "m5", "m6"] and forward.next_cursor == 6
    forward = await svc.messages.list(chat.id, after=6, limit=3)
    assert texts(forward) == ["m7"] and not forward.has_more and forward.next_cursor is None

    backward = await svc.messages.list(chat.id, before=8, limit=3)  # open at the latest
    assert texts(backward) == ["m5", "m6", "m7"] and backward.has_more and backward.next_cursor == 5
    backward = await svc.messages.list(chat.id, before=backward.next_cursor, limit=3)
    assert texts(backward) == ["m2", "m3", "m4"] and backward.next_cursor == 2
    backward = await svc.messages.list(chat.id, before=2, limit=3)
    assert texts(backward) == ["m1"] and not backward.has_more and backward.next_cursor is None
    assert backward.total == 7

    exact = await svc.messages.list(chat.id, limit=7)
    assert len(exact.items) == 7 and not exact.has_more


async def test_transcript_rejects_bad_paging(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    with pytest.raises(InvalidInput, match="either"):
        await svc.messages.list(chat.id, after=1, before=5)
    with pytest.raises(InvalidInput, match="limit"):
        await svc.messages.list(chat.id, limit=0)
    with pytest.raises(NotFound):
        await svc.messages.list("cht_nope")
    empty = await svc.messages.list(chat.id)
    assert (empty.items, empty.total, empty.has_more) == ([], 0, False)


async def test_concurrent_appends_get_unique_gap_free_numbers(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    await asyncio.gather(*(svc.messages.append(chat.id, role="user", text=f"m{i}") for i in range(25)))
    page = await svc.messages.list(chat.id, limit=100)
    assert [m.seq for m in page.items] == list(range(1, 26))
    assert sorted(m.text for m in page.items) == sorted(f"m{i}" for i in range(25))
    assert page.total == 25


async def test_messages_of_different_chats_are_numbered_separately(svc):
    p = await svc.projects.create("A")
    a = await svc.chats.create(p.id)
    b = await svc.chats.create(p.id)
    await svc.messages.append(a.id, role="user", text="a1")
    assert (await svc.messages.append(b.id, role="user", text="b1")).seq == 1
    assert (await svc.messages.append(a.id, role="user", text="a2")).seq == 2


# ------------------------------------------------------------------ summaries


async def test_summary_covers_latest_message_and_goes_stale(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    assert await svc.summaries.get(chat.id) is None
    await svc.messages.append(chat.id, role="user", text="q1")
    last = await svc.messages.append(chat.id, role="agent", text="a1")
    saved = await svc.summaries.save(
        chat.id, kind="user", content="## Overview\nRevenue.", data={"topics": ["revenue"]}, model="qwen3:4b-instruct"
    )
    assert saved.id.startswith("sum_") and saved.covers_message_id == last.id and not saved.stale
    assert await svc.summaries.get(chat.id) == saved
    await svc.messages.append(chat.id, role="user", text="q2")
    assert (await svc.summaries.get(chat.id)).stale  # "out of date — update"


async def test_saving_a_summary_replaces_the_previous_one_of_that_kind(svc, db):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    first = await svc.messages.append(chat.id, role="user", text="q1")
    await svc.messages.append(chat.id, role="agent", text="a1")
    await svc.summaries.save(chat.id, kind="memory", content="old memory", covers_message_id=first.id)
    assert (await svc.summaries.get(chat.id, "memory")).stale
    await svc.summaries.save(chat.id, kind="memory", content="new memory")
    await svc.summaries.save(chat.id, kind="user", content="digest")
    assert (await svc.summaries.get(chat.id, "memory")).content == "new memory"
    assert (await svc.summaries.get(chat.id, "user")).content == "digest"
    assert await count_rows(db, orm.ChatSummary) == 2


async def test_summary_must_cover_a_message_of_its_chat(svc):
    p = await svc.projects.create("A")
    chat = await svc.chats.create(p.id)
    other = await svc.chats.create(p.id)
    foreign = await svc.messages.append(other.id, role="user", text="elsewhere")
    with pytest.raises(InvalidInput, match="covers_message_id"):
        await svc.summaries.save(chat.id, kind="user", content="x", covers_message_id=foreign.id)
    with pytest.raises(InvalidInput, match="kind"):
        await svc.summaries.save(chat.id, kind="weekly", content="x")  # type: ignore[arg-type]
    with pytest.raises(NotFound):
        await svc.summaries.get("cht_nope")
    empty = await svc.summaries.save(chat.id, kind="user", content="nothing yet")
    assert empty.covers_message_id is None and not empty.stale


# ------------------------------------------------------------------ documents


async def test_documents_list_newest_first(svc, db, clock):
    p = await svc.projects.create("A")
    older = await add_document(db, p.id, "old.pdf", created_at=clock(), updated_at=clock())
    newer = await add_document(db, p.id, "new.pdf", created_at=clock(), updated_at=clock(), status="READY")
    docs = await svc.documents.list(p.id)
    assert [d.id for d in docs] == [newer, older]
    assert docs[0].status == "READY" and docs[1].status == "PENDING"
    assert docs[0].filename == "new.pdf" and docs[0].version == 1
