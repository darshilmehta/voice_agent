"""The live visual canvas (docs/DESIGN.md §12.1, contract v1): a chat's canvas and its edits, adding a visual from a
spec (tests and the debug panel; the conversation adds visuals itself later), the project overview, and the project's
typed datasets (what a spec can refer to).

Errors are the app's usual ones: unknown chat, project or visual → 404; a spec that doesn't resolve against the
datasets → 422 with every problem listed.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from ..domain.canvas import CanvasOp, CanvasPanels, ProjectOverview, Visual
from ..domain.datasets import DatasetSummary
from ..services.canvas.service import CanvasService
from ..services.canvas.spec import VisualSpec

router = APIRouter(tags=["canvas"])


def canvas_service(request: Request) -> CanvasService:
    """The app's long-lived canvas service (it tracks overview builds and owns the planner)."""
    return request.app.state.canvas


Canvas = Annotated[CanvasService, Depends(canvas_service)]


class DatasetList(BaseModel):
    items: list[DatasetSummary]


@router.get("/api/chats/{chat_id}/canvas")
async def get_canvas(chat_id: str, canvas: Canvas) -> CanvasPanels:
    """The chat's panels in position order."""
    return await canvas.canvas(chat_id)


@router.post("/api/chats/{chat_id}/canvas/ops")
async def canvas_op(chat_id: str, body: CanvasOp, canvas: Canvas) -> CanvasPanels:
    """``remove``, ``pin``, ``unpin`` or ``move`` (to ``position``, 0-based, clamped) a panel; returns the canvas."""
    return await canvas.apply(chat_id, body)


@router.post("/api/chats/{chat_id}/visuals", status_code=status.HTTP_201_CREATED)
async def add_visual(chat_id: str, body: VisualSpec, canvas: Canvas) -> Visual:
    """Validate the spec against the chat's datasets (its project's READY documents within the chat's scope), build
    the visual from the cells and add it at the end of the canvas. Past ``canvas.max_panels`` the oldest unpinned
    panel is removed."""
    return await canvas.add_visual(chat_id, body)


@router.get("/api/projects/{project_id}/overview")
async def get_overview(project_id: str, canvas: Canvas) -> ProjectOverview:
    """The overview built from the project's documents: ``building`` while documents ingest (or it is being
    rebuilt), ``ready`` with panels, ``none`` when nothing is chartable."""
    return await canvas.overview(project_id)


@router.get("/api/projects/{project_id}/datasets")
async def list_datasets(project_id: str, canvas: Canvas) -> DatasetList:
    """The typed datasets of the project's READY documents: ids, columns and rows (keys for a VisualSpec), units,
    chartability."""
    return DatasetList(items=await canvas.dataset_summaries(project_id))
