"""The user-facing chat summary as structured data (docs/DESIGN.md §3.9), and how pages are named.

``SummaryData`` is what ``chat_summaries.data`` holds for kind ``user`` (and ``content`` holds it rendered as Markdown);
``UserSummary`` is the API's view of the stored summary. Sources are real document and page references, never the
``[S#]`` markers of single answers: those restart in every answer and mean nothing across a whole chat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..settings import Language

SUMMARY_SCHEMA_VERSION = 1


class SourceRef(BaseModel):
    """A document and page range cited somewhere in a chat. ``page_start``/``page_end`` are None when the source has
    no page (plain text)."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None


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
