"""The user-facing chat summary as structured data (docs/DESIGN.md §3.9), and how pages are named.

``SummaryData`` is what ``chat_summaries.data`` holds for kind ``user`` (and ``content`` holds it rendered as Markdown);
``UserSummary`` is the API's view of the stored summary. Sources are real document and page references, never the
``[S#]`` markers of single answers: those restart in every answer and mean nothing across a whole chat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from ..settings import Language

SUMMARY_SCHEMA_VERSION = 1


class SourceRefJSON(TypedDict):
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    kind: NotRequired[Literal["document", "web"]]
    url: NotRequired[str | None]
    title: NotRequired[str | None]


class SourceRef(BaseModel):
    """A document and page range cited somewhere in a chat. ``page_start``/``page_end`` are None when the source has
    no page (plain text). A live web result an answer cited (docs/DESIGN.md §3.7) is ``kind: "web"`` with its
    ``url`` and ``title`` (``filename`` is its site, no pages); two results from one site stay two sources. Document
    references serialize as before (no ``kind``)."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    kind: Literal["document", "web"] = "document"
    url: str | None = None
    title: str | None = None

    @model_serializer(mode="wrap")
    def _without_web_fields(self, handler: SerializerFunctionWrapHandler) -> SourceRefJSON:
        data = handler(self)
        if self.kind == "document":
            for name in ("kind", "url", "title"):
                data.pop(name, None)
        return data


class KeyPoint(BaseModel):
    text: str
    sources: list[SourceRef]  # only sources that answers in this chat actually cited


class UnansweredQuestion(BaseModel):
    """A question the documents couldn't answer (the agent abstained): the user's words, and which message they were."""

    question: str
    message_seq: int


class SummaryData(BaseModel):
    """The structured summary, as stored in ``chat_summaries.data``."""

    schema_version: Literal[1] = SUMMARY_SCHEMA_VERSION
    language: Language
    overview: str
    key_points: list[KeyPoint]
    unanswered_questions: list[UnansweredQuestion]
    follow_ups: list[str]
    windows: int  # message windows summarised separately (1 = the chat fitted in one prompt; more = map-reduce)
    prompt: str  # prompt version (revisit_prompts.SUMMARY_PROMPT_VERSION)


class UserSummary(BaseModel):
    """The chat's summary, as returned by ``GET``/``POST /api/chats/{id}/summary``.

    ``covers_seq`` is the last message it includes and ``message_count`` how many messages the chat has now;
    ``stale`` (covers_seq < message_count) tells the UI to show "summary is out of date".
    """

    id: str
    chat_id: str
    language: Language
    overview: str
    key_points: list[KeyPoint]
    unanswered_questions: list[UnansweredQuestion]
    follow_ups: list[str]
    content: str  # the same summary as Markdown (citations as "(file.pdf, p. 2)")
    covers_seq: int
    message_count: int
    stale: bool
    model: str | None
    created_at: datetime
