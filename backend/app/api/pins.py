"""GET /api/pins: pinned projects and chats for the sidebar (docs/DESIGN.md §3.9)."""

from __future__ import annotations

from fastapi import APIRouter

from ..domain.projects import PinnedItems
from .deps import Pins

router = APIRouter(prefix="/api/pins", tags=["pins"])


@router.get("")
async def list_pins(pins: Pins) -> PinnedItems:
    """Pinned, non-archived projects and chats, most recently pinned first. Chats carry their project's name."""
    return await pins.list()
