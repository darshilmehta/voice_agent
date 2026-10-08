"""Revisiting a chat: regenerated titles and the user summary (docs/DESIGN.md §3.9).

The services behind them are ``services/titles.py`` and ``chat_summary.py``; this module only maps them to HTTP.
Errors are the app's usual ones: unknown chat → 404, bad input → 422, the model failing → 503.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..domain.projects import Chat
from ..domain.summaries import UserSummary
from ..services.base import NotFound
from ..services.chat_summary import ChatSummarizer
from ..services.titles import TitleLocked, TitleService
from ..settings import Language

router = APIRouter(tags=["revisit"])


def summarizer(request: Request) -> ChatSummarizer:
    """The app's long-lived summarizer (it serialises concurrent requests for one chat)."""
    return request.app.state.summarizer


def titles(request: Request) -> TitleService:
    return request.app.state.titles


Summarizer = Annotated[ChatSummarizer, Depends(summarizer)]
Titles = Annotated[TitleService, Depends(titles)]


@router.get("/api/chats/{chat_id}/summary")
async def get_summary(chat_id: str, summaries: Summarizer) -> UserSummary:
    """The chat's current summary. 404 if the chat doesn't exist or has no summary yet. ``stale`` is true when
    messages were added after the last one the summary covers (``covers_seq`` < ``message_count``)."""
    summary = await summaries.get(chat_id)
    if summary is None:
        raise NotFound("summary of chat", chat_id)
    return summary


@router.post("/api/chats/{chat_id}/summary")
async def create_summary(chat_id: str, summaries: Summarizer, language: Language | None = None) -> UserSummary:
    """Generate the summary, or refresh it when messages were added; for an unchanged chat the stored summary is
    returned without calling the model. ``language`` (en | hi) overrides the chat's dominant language; a stored
    summary in another language is regenerated. Empty chat → 422; model unavailable → 503 (nothing stored). Long chats
    take longer (several model calls)."""
    return await summaries.generate(chat_id, language=language)


@router.post("/api/chats/{chat_id}/title:regenerate")
async def regenerate_title(chat_id: str, titles: Titles, force: bool = False) -> Chat:
    """Ask the model for a new title from the chat's first question and answer, and return the updated chat. A title
    the user set is kept (409) unless ``force=true``, which replaces it and makes the title automatic again. Needs a
    user message (422); the model failing → 503 and the title stays as it is."""
    try:
        return await titles.generate(chat_id, replace="any" if force else "auto", fallback=False)
    except TitleLocked as e:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{e}; pass force=true to replace it") from e
