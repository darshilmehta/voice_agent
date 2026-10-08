"""Shared pieces for the title, summary and export tests: a scripted LLM with JSON output, and a way to seed chats."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from typing import Any

from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.domain.projects import Chat, Citation, Message, Modality, Role
from app.providers.llm import LLMError, LLMMessage
from app.providers.runtime import InProcessJobQueue
from app.services.chats import ChatService
from app.services.messages import MessageService
from app.services.projects import ProjectService

from .conftest import Fakes
from .fakes import FakeEmbedder, FakeLLM, FakeReranker, FakeStore, TextParser, keyword_scorer

TITLE_MARK = "write titles for conversations"  # in the title system prompt


def is_title(messages: Sequence[LLMMessage]) -> bool:
    return TITLE_MARK in messages[0].content


class ScriptedLLM(FakeLLM):
    """FakeLLM that also answers ``generate_json``: ``json_reply`` is called with the messages and the schema and
    returns a model instance, a dict (validated like the real client does) or raises. Title calls (``generate``) use
    ``reply`` like any FakeLLM call. ``json_calls`` records every structured call."""

    def __init__(
        self,
        reply: str | Callable[[list[LLMMessage]], str] = "FY24 EBITDA margin",
        json_reply: Callable[[list[LLMMessage], type[BaseModel]], Any] | None = None,
    ) -> None:
        super().__init__(reply)
        self.json_reply = json_reply
        self.json_calls: list[list[LLMMessage]] = []
        self.json_fail_with: Exception | None = None
        self.json_fail_times = 0  # raise json_fail_with for this many calls (-1: always), then answer normally
        self.before_json: Callable[[], Awaitable[Any]] | None = None  # awaited at the start of every structured call

    @property
    def title_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if is_title(c["messages"])]

    async def generate_json(  # type: ignore[override]
        self,
        messages: Sequence[LLMMessage],
        schema: type[BaseModel],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Any:
        self.json_calls.append(list(messages))
        if self.before_json is not None:
            await self.before_json()
        if self.json_fail_with is not None and self.json_fail_times != 0:
            self.json_fail_times = max(self.json_fail_times - 1, -1)
            raise self.json_fail_with
        if self.json_reply is None:
            raise LLMError("no scripted JSON reply")
        out = self.json_reply(list(messages), schema)
        if isinstance(out, BaseModel):
            return out
        return schema.model_validate(out)


def make_fakes(llm: FakeLLM) -> Fakes:
    return Fakes(TextParser(), FakeEmbedder(), FakeReranker(keyword_scorer), FakeStore(search_points=True), llm)


def cite(
    source_id: str = "S1",
    filename: str = "annual_report.pdf",
    page: int | None = 2,
    *,
    page_end: int | None = None,
    document_id: str = "doc_report",
    snippet: str = "EBITDA margin improved to 18.2% from 16.9%.",
    chunk_id: str | None = None,
) -> Citation:
    return Citation(
        source_id=source_id,
        document_id=document_id,
        filename=filename,
        page_start=page,
        page_end=page if page_end is None else page_end,
        chunk_id=chunk_id or f"{document_id}:v1:{source_id}:{page}",
        snippet=snippet,
    )


class Seeder:
    """Runs service calls on the app's event loop, so sync tests can build chats through the same database the API
    uses (and trigger the same hooks)."""

    def __init__(self, api: TestClient) -> None:
        self.api = api
        self.db = api.app.state.container["metadata_db"]  # type: ignore[attr-defined]

    def run[T](self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        return self.api.portal.call(partial(fn, *args, **kwargs))  # type: ignore[union-attr]

    def project(self, name: str = "Annual report FY24", *, archived: bool = False) -> str:
        projects = ProjectService(self.db)
        p = self.run(projects.create, name)
        if archived:
            self.run(projects.update, p.id, archived=True)
        return p.id

    def chat(self, project_id: str, *, title: str | None = None, language: str | None = None) -> Chat:
        return self.run(ChatService(self.db).create, project_id, title=title, language=language)

    def say(
        self,
        chat_id: str,
        role: Role,
        text: str,
        *,
        modality: Modality = "text",
        language: str | None = "en",
        citations: Sequence[Citation] = (),
        route: dict[str, Any] | None = None,
        heard_text: str | None = None,
        latency: dict[str, Any] | None = None,
    ) -> Message:
        return self.run(
            MessageService(self.db).append,
            chat_id,
            role=role,
            text=text,
            modality=modality,
            language=language,
            citations=list(citations),
            route=route,
            heard_text=heard_text,
            latency=latency,
        )

    def ask(self, chat_id: str, question: str, answer: str, **kw: Any) -> tuple[Message, Message]:
        """A user message and the agent's answer after it."""
        modality = kw.pop("modality", "text")
        language = kw.pop("language", "en")
        user = self.say(chat_id, "user", question, modality=modality, language=language)
        agent = self.say(chat_id, "agent", answer, modality=modality, language=language, **kw)
        return user, agent

    def drain(self) -> None:
        """Wait for background jobs (the title job)."""
        queue = self.api.app.state.container["job_queue"]  # type: ignore[attr-defined]
        assert isinstance(queue, InProcessJobQueue)
        self.run(queue.join)

    def chat_row(self, chat_id: str) -> dict[str, Any]:
        r = self.api.get(f"/api/chats/{chat_id}")
        assert r.status_code == 200, r.text
        return r.json()


ABSTAINED = {"abstained": True, "abstain_reason": "not_covered", "stopped": False, "intent": "document_qa"}
ANSWERED = {"abstained": False, "abstain_reason": None, "stopped": False, "intent": "document_qa"}
