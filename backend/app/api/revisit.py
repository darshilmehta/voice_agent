"""Revisiting a chat: regenerating a title (docs/DESIGN.md §3.9); the service is ``services/titles.py``. Errors are the
app's usual ones: unknown chat → 404, bad input → 422, the model failing → 503.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..domain.projects import Chat
from ..services.titles import TitleLocked, TitleService

router = APIRouter(tags=["revisit"])


def titles(request: Request) -> TitleService:
    return request.app.state.titles


Titles = Annotated[TitleService, Depends(titles)]


@router.post("/api/chats/{chat_id}/title:regenerate")
async def regenerate_title(chat_id: str, titles: Titles, force: bool = False) -> Chat:
    """Ask the model for a new title from the chat's first question and answer, and return the updated chat. A title
    the user set is kept (409) unless ``force=true``, which replaces it and makes the title automatic again. Needs a
    user message (422); the model failing → 503 and the title stays as it is."""
    try:
        return await titles.generate(chat_id, replace="any" if force else "auto", fallback=False)
    except TitleLocked as e:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{e}; pass force=true to replace it") from e
