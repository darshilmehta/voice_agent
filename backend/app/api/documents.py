"""Document upload, status and deletion (docs/DESIGN.md §3.1, §3.9). The project's document list is in ``projects``."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException

from ..domain.projects import Document
from ..services.base import InvalidInput
from .deps import Documents, Pipeline

router = APIRouter(tags=["documents"])

# Multipart framing around the file (boundaries, part headers) on top of the upload limit.
_MULTIPART_SLACK = 64 * 1024

_UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


@router.post(
    "/api/projects/{project_id}/documents",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Document,
    responses={
        200: {"model": Document, "description": "Identical content already in the project: that document"},
        202: {"model": Document, "description": "New document, PENDING; ingestion runs in the background"},
        422: {"description": "Not allowed: extension, size, or content that doesn't match the extension"},
    },
    openapi_extra=_UPLOAD_BODY,
)
async def upload_document(project_id: str, request: Request, pipeline: Pipeline) -> Response:
    """Upload one file (``multipart/form-data``, field ``file``). Poll the document (or the project's list) until its
    status is READY or FAILED."""
    limit = pipeline.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + _MULTIPART_SLACK:
        raise InvalidInput(f"file is larger than the {pipeline.settings.server.max_upload_mb} MB upload limit")
    try:
        form = await request.form(max_files=1, max_fields=8)
    except MultiPartException as e:
        raise InvalidInput(f"expected multipart/form-data with one file in the field 'file': {e.message}") from None
    except AssertionError:  # starlette: python-multipart missing or not a form body
        raise InvalidInput("expected multipart/form-data with one file in the field 'file'") from None
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise InvalidInput("expected multipart/form-data with one file in the field 'file'")
        document, created = await pipeline.upload(project_id, upload.filename, upload.file)
    finally:
        await form.close()
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK,
        content=document.model_dump(mode="json"),
    )


@router.get("/api/documents/{document_id}")
async def get_document(document_id: str, documents: Documents) -> Document:
    return await documents.get(document_id)


@router.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, pipeline: Pipeline) -> Response:
    """Delete the document: its vectors, stored files and records; chats scoped to it no longer list it."""
    await pipeline.delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
