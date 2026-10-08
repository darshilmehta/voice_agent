"""Automatic chat titles: when they are generated, what they look like, and that a user's title is never replaced."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from app.providers.llm import LLMError, LLMMessage, LLMUnavailableError
from app.providers.runtime import Job, JobQueue
from app.services import titles as titles_module
from app.services.base import InvalidInput, Unavailable
from app.services.chats import DEFAULT_TITLE, ChatService
from app.services.messages import MessageService, add_agent_message_hook
from app.services.projects import ProjectService
from app.services.revisit_prompts import TITLE_MAX_WORDS, clean_title, fallback_title
from app.services.titles import TITLE_MAX_TOKENS, TitleLocked, TitleService

from .revisit_helpers import ABSTAINED, ScriptedLLM, Seeder, is_title, make_fakes

EN = "What was the EBITDA margin in FY24?"
HI = "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?"


# ------------------------------------------------------------------ cleaning (pure)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FY24 EBITDA margin", "FY24 EBITDA margin"),
        ('"FY24 EBITDA margin"', "FY24 EBITDA margin"),
        ("“FY24 EBITDA margin.”", "FY24 EBITDA margin"),
        ("Title: FY24 EBITDA margin", "FY24 EBITDA margin"),
        ("**Title:** FY24 EBITDA margin", "FY24 EBITDA margin"),
        ("# FY24 EBITDA margin!", "FY24 EBITDA margin"),
        ("  FY24 EBITDA margin [S1].\nThe model went on explaining", "FY24 EBITDA margin"),
        ("\n\nFY24   EBITDA\tmargin?", "FY24 EBITDA margin"),
        ("one two three four five six seven eight", "one two three four five six"),  # at most six words
        ("वित्त वर्ष 2024 का EBITDA मार्जिन।", "वित्त वर्ष 2024 का EBITDA मार्जिन"),  # Hindi full stop
        ("शीर्षक: EBITDA मार्जिन FY24", "EBITDA मार्जिन FY24"),
        ("«Marge EBITDA»", "Marge EBITDA"),
        ("New chat", None),  # the placeholder is not a title
        ("   ", None),
        ('""', None),
        ("...", None),
    ],
)
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_clean_title_limits_length():
    title = clean_title("x" * 30 + " " + "y" * 30 + " " + "z" * 30)
    assert title is not None and len(title) <= 80 and len(title.split()) <= TITLE_MAX_WORDS


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (EN, "What was the EBITDA margin in FY24"),
        ("  what   is\nthe revenue?? ", "what is the revenue"),
        (HI, "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था"),
        ("Summarise the risks [S1]", "Summarise the risks"),
        (
            "Can you walk me through every single one of the covenants listed in the credit agreement and explain them",
            "Can you walk me through every single one of the covenants…",
        ),
        ("?", None),
        ("", None),
    ],
)
def test_fallback_title(question, expected):
    title = fallback_title(question)
    assert title == expected
    if title:
        assert len(title) <= 60


# ------------------------------------------------------------------ service-level tests


class InlineQueue(JobQueue):
    """A job queue that keeps jobs until the test runs them, so "the answer doesn't wait for the title" is visible."""

    name = "inline"

    def __init__(self) -> None:
        self.jobs: list[tuple[str, Job]] = []
        self.fail_with: Exception | None = None

    async def submit(self, name: str, job: Job) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.jobs.append((name, job))

    async def run_all(self) -> None:
        while self.jobs:
            _, job = self.jobs.pop(0)
            await job()


@dataclass
class Env:
    projects: ProjectService
    chats: ChatService
    messages: MessageService
    titles: TitleService
    llm: ScriptedLLM
    queue: InlineQueue

    async def new_chat(self, **kw: Any) -> str:
        project = await self.projects.create("Annual report FY24")
        return (await self.chats.create(project.id, **kw)).id

    async def turn(self, chat_id: str, question: str = EN, answer: str = "It was 18.2% [S1].", **kw: Any) -> None:
        modality = kw.pop("modality", "text")
        language = kw.pop("language", "en")
        await self.messages.append(chat_id, role="user", text=question, modality=modality, language=language)
        await self.messages.append(chat_id, role="agent", text=answer, modality=modality, language=language, **kw)

    async def title(self, chat_id: str) -> tuple[str, bool]:
        chat = await self.chats.get(chat_id)
        return chat.title, chat.title_is_auto


@pytest.fixture
def env(db, clock, load_local) -> Env:
    llm = ScriptedLLM("FY24 EBITDA margin")
    queue = InlineQueue()
    titles = TitleService(db, llm=llm, settings=load_local(), queue=queue, clock=clock)
    add_agent_message_hook(db, titles.on_agent_message)
    return Env(
        ProjectService(db, clock=clock),
        ChatService(db, clock=clock),
        MessageService(db, clock=clock),
        titles,
        llm,
        queue,
    )


async def test_title_is_generated_in_the_background_after_the_first_answer(env):
    chat = await env.new_chat()
    assert await env.title(chat) == (DEFAULT_TITLE, True)
    before = await env.chats.get(chat)

    await env.turn(chat)
    assert [name for name, _ in env.queue.jobs] == ["chat-title"]  # queued, not run: the answer didn't wait for it
    assert env.llm.calls == [] and (await env.title(chat)) == (DEFAULT_TITLE, True)

    await env.queue.run_all()
    assert await env.title(chat) == ("FY24 EBITDA margin", True)  # still automatic: the user can rename later
    after = await env.chats.get(chat)
    assert after.updated_at > before.updated_at

    (call,) = env.llm.title_calls
    system, prompt = call["messages"]
    assert call["max_tokens"] == TITLE_MAX_TOKENS and call["temperature"] is not None
    assert "At most 6 words" in system.content and "Write the title in English" in system.content
    assert EN in prompt.content and "It was 18.2% [S1]." in prompt.content


async def test_a_voice_turn_gets_a_title_too(env):
    chat = await env.new_chat()
    await env.turn(chat, modality="voice")
    await env.queue.run_all()
    assert await env.title(chat) == ("FY24 EBITDA margin", True)


async def test_the_title_is_not_regenerated_by_later_messages(env):
    chat = await env.new_chat()
    await env.turn(chat)
    await env.queue.run_all()
    env.llm.reply = "Something else entirely"
    await env.turn(chat, "And in FY23?", "It was 16.9% [S1].")
    await env.turn(chat, "And revenue?", "Up 34% [S1].")
    assert env.queue.jobs == []
    assert len(env.llm.title_calls) == 1
    assert await env.title(chat) == ("FY24 EBITDA margin", True)


async def test_user_messages_and_chats_with_a_user_title_start_no_job(env):
    untitled = await env.new_chat()
    await env.messages.append(untitled, role="user", text=EN)
    assert env.queue.jobs == []  # only an agent message triggers it

    named = await env.new_chat(title="Board pack")
    await env.turn(named)
    assert env.queue.jobs == [] and env.llm.calls == []
    assert await env.title(named) == ("Board pack", False)


async def test_a_title_the_user_sets_before_the_job_runs_is_kept(env):
    chat = await env.new_chat()
    await env.turn(chat)
    assert len(env.queue.jobs) == 1
    await env.chats.update(chat, title="My own name")
    await env.queue.run_all()
    assert await env.title(chat) == ("My own name", False)
    assert env.llm.calls == []  # not even asked


class GatedLLM(ScriptedLLM):
    """The model takes as long as the test wants: ``gate`` is opened by the test."""

    def __init__(self) -> None:
        super().__init__("FY24 EBITDA margin")
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, messages, **kw):  # type: ignore[override]
        self.entered.set()
        await self.release.wait()
        async for piece in super().stream(messages, **kw):
            yield piece


async def test_renaming_while_the_title_is_generated_wins(db, clock, load_local):
    llm = GatedLLM()
    queue = InlineQueue()
    titles = TitleService(db, llm=llm, settings=load_local(), queue=queue, clock=clock)
    add_agent_message_hook(db, titles.on_agent_message)
    chats, messages = ChatService(db, clock=clock), MessageService(db, clock=clock)
    project = await ProjectService(db, clock=clock).create("P")
    chat = (await chats.create(project.id)).id
    await messages.append(chat, role="user", text=EN)
    await messages.append(chat, role="agent", text="It was 18.2%.")

    job = asyncio.create_task(queue.run_all())
    await asyncio.wait_for(llm.entered.wait(), 2)  # the model is thinking …
    await chats.update(chat, title="Renamed by the user")  # … and the user renames the chat
    llm.release.set()
    await asyncio.wait_for(job, 2)

    renamed = await chats.get(chat)
    assert (renamed.title, renamed.title_is_auto) == ("Renamed by the user", False)
    assert len(llm.calls) == 1  # the model did run; its title was discarded


async def test_failing_model_falls_back_to_the_first_question(env):
    env.llm.fail_with = LLMUnavailableError("Ollama unreachable")
    chat = await env.new_chat()
    await env.turn(chat)
    await env.queue.run_all()
    assert len(env.llm.title_calls) == 2  # one try, one retry, then the fallback
    assert await env.title(chat) == ("What was the EBITDA margin in FY24", True)


async def test_one_retry_is_enough_when_the_first_attempt_fails(env):
    replies = iter([LLMError("boom"), "FY24 EBITDA margin"])

    def flaky(messages: list[LLMMessage]) -> str:
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    env.llm.reply = flaky
    chat = await env.new_chat()
    await env.turn(chat)
    await env.queue.run_all()
    assert len(env.llm.title_calls) == 2
    assert await env.title(chat) == ("FY24 EBITDA margin", True)


@pytest.mark.parametrize("reply", ["", "   ", '""', "New chat", "..."])
async def test_unusable_replies_fall_back_after_one_retry(env, reply):
    env.llm.reply = reply
    chat = await env.new_chat()
    await env.turn(chat)
    await env.queue.run_all()
    assert len(env.llm.title_calls) == 2
    assert await env.title(chat) == ("What was the EBITDA margin in FY24", True)


async def test_a_slow_model_times_out_and_falls_back(env, monkeypatch):
    monkeypatch.setattr(titles_module, "TITLE_TIMEOUT_S", 0.05)
    env.llm.delay = 1.0
    chat = await env.new_chat()
    await env.turn(chat)
    await asyncio.wait_for(env.queue.run_all(), 2)
    assert len(env.llm.title_calls) == 2
    assert await env.title(chat) == ("What was the EBITDA margin in FY24", True)


async def test_hindi_title(env):
    env.llm.reply = "“वित्त वर्ष 2024 का EBITDA मार्जिन।”"
    chat = await env.new_chat()
    await env.turn(chat, HI, "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था [S1]।", language="hi")
    await env.queue.run_all()
    assert await env.title(chat) == ("वित्त वर्ष 2024 का EBITDA मार्जिन", True)
    system = env.llm.title_calls[0]["messages"][0].content
    assert "Write the title in Hindi, in Devanagari script" in system and "EBITDA or FY24 exactly as written" in system


async def test_hindi_fallback_title(env):
    env.llm.fail_with = LLMError("down")
    chat = await env.new_chat()
    await env.turn(chat, HI, "…", language="hi")
    await env.queue.run_all()
    assert await env.title(chat) == ("वित्त वर्ष 2024 में EBITDA मार्जिन कितना था", True)


async def test_an_abstained_answer_is_not_shown_to_the_model(env):
    chat = await env.new_chat()
    await env.turn(chat, "What is the CEO's salary?", "I couldn't find that.", route=ABSTAINED)
    await env.queue.run_all()
    prompt = env.llm.title_calls[0]["messages"][1].content
    assert "CEO's salary" in prompt and "Answer:" not in prompt


async def test_only_one_job_per_chat_while_one_is_pending(env):
    chat = await env.new_chat()
    await env.turn(chat)
    await env.messages.append(chat, role="user", text="And FY23?")
    await env.messages.append(chat, role="agent", text="16.9%")
    assert len(env.queue.jobs) == 1
    await env.queue.run_all()
    assert len(env.llm.title_calls) == 1


async def test_a_dropped_job_is_retried_by_the_next_answer(env):
    chat = await env.new_chat()
    await env.turn(chat)
    env.queue.jobs.clear()  # shutdown dropped it; the chat is still "New chat"
    env.titles._pending.clear()  # (a restart forgets in-flight jobs)
    await env.turn(chat, "And FY23?", "16.9%")
    assert len(env.queue.jobs) == 1
    await env.queue.run_all()
    assert (await env.title(chat))[0] == "FY24 EBITDA margin"


async def test_a_queue_that_refuses_jobs_never_fails_the_answer(env):
    env.queue.fail_with = RuntimeError("job queue is shut down")
    chat = await env.new_chat()
    await env.turn(chat)  # the agent message is saved
    assert (await env.messages.list(chat)).total == 2
    env.queue.fail_with = None
    await env.turn(chat, "Again?", "Yes.")  # nothing stuck in "pending"
    assert len(env.queue.jobs) == 1


async def test_a_job_for_a_deleted_chat_ends_quietly(env):
    chat = await env.new_chat()
    await env.turn(chat)
    await env.chats.delete(chat)
    await env.queue.run_all()
    assert env.llm.calls == []  # the chat is gone before the model is asked


async def test_an_agent_greeting_before_any_question_waits_for_the_first_question(env):
    chat = await env.new_chat()
    await env.messages.append(chat, role="agent", text="Hi! Ask me about your documents.")
    await env.queue.run_all()
    assert env.llm.calls == [] and await env.title(chat) == (DEFAULT_TITLE, True)
    await env.turn(chat)
    await env.queue.run_all()
    assert await env.title(chat) == ("FY24 EBITDA margin", True)


async def test_a_failing_hook_never_fails_the_save(db, clock):
    async def broken(message) -> None:
        raise RuntimeError("hook bug")

    remove = add_agent_message_hook(db, broken)
    messages = MessageService(db, clock=clock)
    project = await ProjectService(db, clock=clock).create("P")
    chat = (await ChatService(db, clock=clock).create(project.id)).id
    saved = await messages.append(chat, role="agent", text="hello")
    assert saved.seq == 1
    remove()
    remove()  # removing twice is harmless


async def test_generate_modes(env):
    chat = await env.new_chat()
    with pytest.raises(InvalidInput, match="no message to title"):
        await env.titles.generate(chat, replace="auto")
    await env.turn(chat)
    await env.queue.run_all()

    env.llm.reply = "Second title"
    assert (await env.titles.generate(chat)).title == "FY24 EBITDA margin"  # placeholder mode: nothing to do
    assert (await env.titles.generate(chat, replace="auto")).title == "Second title"

    await env.chats.update(chat, title="Mine")
    with pytest.raises(TitleLocked):
        await env.titles.generate(chat, replace="auto")
    forced = await env.titles.generate(chat, replace="any")
    assert (forced.title, forced.title_is_auto) == ("Second title", True)

    env.llm.fail_with = LLMError("down")
    with pytest.raises(Unavailable):
        await env.titles.generate(chat, replace="auto", fallback=False)
    assert (await env.title(chat)) == ("Second title", True)  # unchanged


# ------------------------------------------------------------------ through the app


@pytest.fixture
def titled_app(make_app):
    llm = ScriptedLLM(lambda messages: "FY24 EBITDA margin" if is_title(messages) else "It was 18.2% [S1].")
    with make_app(make_fakes(llm), auto_titles=True) as client:
        yield client, llm, Seeder(client)


def test_the_sidebar_shows_the_title_after_the_first_answer(titled_app):
    api, llm, seed = titled_app
    project = seed.project()
    chat = seed.chat(project)
    assert chat.title == DEFAULT_TITLE and chat.title_is_auto
    before = seed.chat_row(chat.id)

    seed.ask(chat.id, EN, "It was 18.2% [S1].", modality="voice")
    seed.drain()

    after = seed.chat_row(chat.id)
    assert (after["title"], after["title_is_auto"]) == ("FY24 EBITDA margin", True)
    assert after["updated_at"] > before["updated_at"]
    listed = api.get(f"/api/projects/{project}/chats").json()["items"]
    assert [c["title"] for c in listed] == ["FY24 EBITDA margin"]

    seed.ask(chat.id, "And FY23?", "16.9%")
    seed.drain()
    assert len(llm.title_calls) == 1


def test_a_text_turn_through_the_chat_endpoint_gets_a_title(titled_app):
    api, _, seed = titled_app
    project = api.post("/api/projects", json={"name": "Annual report FY24"}).json()["id"]
    upload = api.post(
        f"/api/projects/{project}/documents",
        files={"file": ("annual_report.txt", b"Key Metrics\n\nEBITDA margin improved to 18.2% in FY24.", "text/plain")},
    )
    assert upload.status_code == 202
    seed.drain()
    chat = api.post(f"/api/projects/{project}/chats", json={}).json()["id"]
    r = api.post(f"/api/chats/{chat}/messages", json={"text": EN})
    assert r.status_code == 200 and "event: agent_message" in r.text
    seed.drain()
    assert seed.chat_row(chat)["title"] == "FY24 EBITDA margin"


def test_regenerate_endpoint(titled_app):
    api, llm, seed = titled_app
    chat = seed.chat(seed.project())
    url = f"/api/chats/{chat.id}/title:regenerate"

    assert api.post(url).status_code == 422  # nothing to title yet
    seed.ask(chat.id, EN, "It was 18.2% [S1].")
    seed.drain()

    llm.reply = lambda messages: "Margins in FY24"
    r = api.post(url)
    assert r.status_code == 200
    assert (r.json()["title"], r.json()["title_is_auto"]) == ("Margins in FY24", True)

    api.patch(f"/api/chats/{chat.id}", json={"title": "My name"})
    r = api.post(url)
    assert r.status_code == 409 and "force=true" in r.json()["detail"]
    assert seed.chat_row(chat.id)["title"] == "My name"

    llm.reply = lambda messages: "Forced title"
    r = api.post(url, params={"force": "true"})
    assert (r.status_code, r.json()["title"], r.json()["title_is_auto"]) == (200, "Forced title", True)

    llm.fail_with = LLMUnavailableError("Ollama unreachable")
    r = api.post(url)
    assert r.status_code == 503
    assert seed.chat_row(chat.id)["title"] == "Forced title"

    assert api.post("/api/chats/cht_missing/title:regenerate").status_code == 404


def test_titles_are_off_unless_enabled(make_app):
    llm = ScriptedLLM("Should not be asked")
    with make_app(make_fakes(llm)) as api:  # the suite's default: auto_titles=False
        seed = Seeder(api)
        chat = seed.chat(seed.project())
        seed.ask(chat.id, EN, "It was 18.2%.")
        seed.drain()
        assert seed.chat_row(chat.id)["title"] == DEFAULT_TITLE and llm.calls == []
