"""FastAPI dependencies: services built on the container's providers, and service errors as HTTP responses."""

from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from ..providers.storage import MetadataDB
from ..services.base import InvalidInput, NotFound, Unavailable
from ..services.chat_turns import ChatTurnService
from ..services.chats import ChatService
from ..services.document_pipeline import DocumentPipeline
from ..services.documents import DocumentService
from ..services.messages import MessageService
from ..services.pins import PinService
from ..services.projects import ProjectService


def metadata_db(request: Request) -> MetadataDB:
    db = request.app.state.container["metadata_db"]
    if not isinstance(db, MetadataDB):
        raise TypeError(f"metadata_db provider is a {type(db).__name__}, not a MetadataDB")
    return db


DB = Annotated[MetadataDB, Depends(metadata_db)]


def project_service(db: DB) -> ProjectService:
    return ProjectService(db)


def chat_service(db: DB) -> ChatService:
    return ChatService(db)


def message_service(db: DB) -> MessageService:
    return MessageService(db)


def document_service(db: DB) -> DocumentService:
    return DocumentService(db)


def pin_service(db: DB) -> PinService:
    return PinService(db)


def document_pipeline(request: Request) -> DocumentPipeline:
    """The app's long-lived pipeline (created at startup: it owns the upload lock and the queue hook)."""
    return request.app.state.document_pipeline


def chat_turns(request: Request) -> ChatTurnService:
    """Text turns, with the app's canvas: an answer's visual travels in its SSE stream (§12.1)."""
    return ChatTurnService.from_container(
        request.app.state.container, canvas=getattr(request.app.state, "canvas", None)
    )


Projects = Annotated[ProjectService, Depends(project_service)]
Chats = Annotated[ChatService, Depends(chat_service)]
Messages = Annotated[MessageService, Depends(message_service)]
Documents = Annotated[DocumentService, Depends(document_service)]
Pins = Annotated[PinService, Depends(pin_service)]
Pipeline = Annotated[DocumentPipeline, Depends(document_pipeline)]
ChatTurns = Annotated[ChatTurnService, Depends(chat_turns)]


def install_error_handlers(app: FastAPI) -> None:
    """NotFound → 404, InvalidInput → 422 and Unavailable → 503, with FastAPI's usual ``{"detail": …}`` body."""

    async def not_found(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    async def invalid(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    async def unavailable(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.add_exception_handler(NotFound, not_found)
    app.add_exception_handler(InvalidInput, invalid)
    app.add_exception_handler(Unavailable, unavailable)
