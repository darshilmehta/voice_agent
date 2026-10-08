"""Projects (docs/DESIGN.md §3.9) and the listing of their documents; upload and deletion are in ``documents``."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from ..domain.projects import Document, Project
from .deps import Documents, Pipeline, Projects
from .schemas import Body, LongText, Name, Patch

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(Body):
    name: Name
    description: LongText | None = None


class ProjectUpdate(Patch):
    NOT_NULL = ("name", "pinned", "archived")

    name: Name | None = None
    description: LongText | None = None
    pinned: bool | None = None
    archived: bool | None = None


class ProjectList(BaseModel):
    items: list[Project]


class DocumentList(BaseModel):
    items: list[Document]


@router.get("")
async def list_projects(projects: Projects, include_archived: bool = False) -> ProjectList:
    """Pinned projects first (most recently pinned on top), then by most recent activity. Archived ones only
    with ``include_archived=true``."""
    return ProjectList(items=await projects.list(include_archived=include_archived))


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(body: ProjectCreate, projects: Projects) -> Project:
    return await projects.create(body.name, body.description)


@router.get("/{project_id}")
async def get_project(project_id: str, projects: Projects) -> Project:
    return await projects.get(project_id)


@router.patch("/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate, projects: Projects) -> Project:
    """Rename, describe, pin/unpin, archive/unarchive."""
    return await projects.update(project_id, **body.changes())


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, pipeline: Pipeline) -> Response:
    """Delete the project with its documents (their vectors and stored files too), chats, messages and summaries."""
    await pipeline.delete_project(project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{project_id}/documents")
async def list_documents(project_id: str, documents: Documents) -> DocumentList:
    return DocumentList(items=await documents.list(project_id))
