"""Upload → background ingestion → READY/FAILED, and deletion with everything stored elsewhere (docs/DESIGN.md §3.1).

    upload: validate (extension, size, content sniffing) → SHA-256 → dedupe within the project
            → object store (generated key, never the uploaded filename) → document PENDING + job QUEUED → job queue
    job:    temp copy from the object store → IngestionService.ingest_file (parse, chunk, embed, index)
            → tables saved as datasets (§12.1) + document READY with counts, or FAILED with the error
            → when the queue drains, the parser releases its models (§8: Docling is loaded only for ingestion)
    delete: vectors first (a document must never stay searchable without its record), then rows (chat scopes
            pruned), then stored files

Only the provider interfaces are used (``ObjectStore``, ``JobQueue``, ``VectorStore`` …), so S3 or Redis are drop-ins.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import unicodedata
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import BinaryIO

from ..db.types import new_id
from ..domain.projects import Document
from ..providers.ingestion import IngestionError
from ..providers.models import ModelUnavailableError
from ..providers.registry import Container
from ..providers.retrieval import VectorStore
from ..providers.runtime import JobQueue
from ..providers.storage import MetadataDB, ObjectNotFound, ObjectStore
from ..settings import Settings
from .base import InvalidInput, Unavailable
from .documents import FAILED, DocumentService
from .ingestion import IngestionService
from .projects import ProjectService

log = logging.getLogger(__name__)

# The content types the app accepts, by extension. The client's declared type is ignored: the server sniffs.
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
}
_OOXML_MAIN_PART = {".docx": "word/document.xml", ".pptx": "ppt/presentation.xml"}
# Signatures of binary formats that must not pass as text.
_BINARY_MAGIC = (b"%PDF-", b"PK\x03\x04", b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"\xd0\xcf\x11\xe0", b"\x7fELF", b"MZ")
FILENAME_MAX = 255
_HASH_CHUNK = 1024 * 1024
_SNIFF_BYTES = 8192


# ------------------------------------------------------------------ validation (pure)


@dataclass(frozen=True, slots=True)
class UploadCheck:
    filename: str  # display name, cleaned
    extension: str  # lower-case, with the dot
    mime: str


def clean_filename(raw: str | None) -> str:
    """The uploaded file's name for display: last path component only, Unicode NFC, without control or bidi-format
    characters, at most 255 characters (extension kept). It is never used as a path."""
    name = unicodedata.normalize("NFC", raw or "")
    name = re.split(r"[\\/]", name)[-1]  # some browsers send a full client path
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    if not name or name in (".", ".."):
        raise InvalidInput("the uploaded file has no name")
    if len(name) > FILENAME_MAX:
        stem, ext = os.path.splitext(name)
        ext = ext if len(ext) <= 16 else ""
        name = stem[: FILENAME_MAX - len(ext)].rstrip() + ext
    return name


def check_name(raw: str | None, allowed_extensions: list[str]) -> UploadCheck:
    filename = clean_filename(raw)
    ext = os.path.splitext(filename)[1].lower()
    allowed = [e.lower() for e in allowed_extensions if e.lower() in MIME_TYPES]
    if ext not in allowed:
        raise InvalidInput(f"{filename}: file type {ext or '(none)'} is not allowed (allowed: {', '.join(allowed)})")
    return UploadCheck(filename, ext, MIME_TYPES[ext])


def measure(f: BinaryIO, max_bytes: int) -> tuple[int, str]:
    """Size and SHA-256 of the file from the start, refusing it as soon as it exceeds ``max_bytes``."""
    f.seek(0)
    digest, size = hashlib.sha256(), 0
    while chunk := f.read(_HASH_CHUNK):
        size += len(chunk)
        if size > max_bytes:
            raise InvalidInput(f"file is larger than the {max_bytes // (1024 * 1024)} MB upload limit")
        digest.update(chunk)
    if size == 0:
        raise InvalidInput("file is empty")
    return size, digest.hexdigest()


def sniff(check: UploadCheck, f: BinaryIO) -> None:
    """Check that the content matches the extension (magic bytes; OOXML package parts; text without binary data)."""
    f.seek(0)
    head = f.read(_SNIFF_BYTES)
    ext, name = check.extension, check.filename
    if ext == ".pdf":
        if b"%PDF-" not in head[:1024]:
            raise InvalidInput(f"{name}: content is not a PDF")
    elif ext in _OOXML_MAIN_PART:
        if not head.startswith(b"PK\x03\x04"):
            raise InvalidInput(f"{name}: content is not a {ext[1:].upper()} file")
        f.seek(0)
        try:
            with zipfile.ZipFile(f) as z:
                parts = set(z.namelist())
        except (zipfile.BadZipFile, OSError) as e:
            raise InvalidInput(f"{name}: damaged {ext[1:].upper()} file ({e})") from None
        if "[Content_Types].xml" not in parts or _OOXML_MAIN_PART[ext] not in parts:
            raise InvalidInput(f"{name}: content is not a {ext[1:].upper()} file")
    elif ext in (".txt", ".md"):
        if b"\x00" in head or head.startswith(_BINARY_MAGIC):
            raise InvalidInput(f"{name}: content is binary, not text (save it as UTF-8 text)")
    else:  # pragma: no cover - check_name only lets MIME_TYPES extensions through
        raise InvalidInput(f"{name}: no content check for {ext}")
    f.seek(0)


def storage_key(project_id: str, document_id: str, version: int, extension: str) -> str:
    """Object key of an uploaded version: generated from ids, so it never contains the uploaded filename."""
    return f"{project_id}/{document_id}/v{version}{extension}"


def describe_failure(e: BaseException) -> str:
    """A readable reason for a FAILED document."""
    if isinstance(e, IngestionError | ModelUnavailableError):
        return str(e)
    if isinstance(e, ObjectNotFound):
        return "the uploaded file is missing from storage"
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


# ------------------------------------------------------------------ pipeline


class DocumentPipeline:
    """Long-lived (one per app): owns the upload lock and registers the queue's idle hook."""

    def __init__(
        self,
        db: MetadataDB,
        *,
        settings: Settings,
        store: ObjectStore,
        queue: JobQueue,
        ingestion: IngestionService,
        vectors: VectorStore,
    ) -> None:
        self.documents = DocumentService(db)
        self.projects = ProjectService(db)
        self.settings = settings
        self.store = store
        self.queue = queue
        self.ingestion = ingestion
        self.vectors = vectors
        self._upload_lock = asyncio.Lock()  # dedupe check + insert are atomic within this process

    @classmethod
    def from_container(cls, container: Container) -> DocumentPipeline:
        parts = {
            "db": (container["metadata_db"], MetadataDB),
            "store": (container["object_store"], ObjectStore),
            "queue": (container["job_queue"], JobQueue),
            "vectors": (container["vector_store"], VectorStore),
        }
        for capability, (provider, kind) in parts.items():
            if not isinstance(provider, kind):
                raise TypeError(f"{capability}: expected a {kind.__name__} provider, got {type(provider).__name__}")
        return cls(
            container["metadata_db"],  # type: ignore[arg-type]
            settings=container.settings,
            store=container["object_store"],  # type: ignore[arg-type]
            queue=container["job_queue"],  # type: ignore[arg-type]
            ingestion=IngestionService.from_container(container),
            vectors=container["vector_store"],  # type: ignore[arg-type]
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.settings.server.max_upload_mb * 1024 * 1024

    async def start(self) -> None:
        """Hook the queue's idle signal and resume ingestions a restart interrupted."""
        try:
            self.queue.on_idle(self._release_parser)
            jobs = await self.documents.recover_interrupted()
        except NotImplementedError as e:  # placeholder providers (cloud template in tests): nothing to run
            log.warning("document pipeline not started: %s", e)
            return
        for job_id in jobs:
            await self._submit(job_id)
        if jobs:
            log.info("re-queued %d interrupted ingestion job(s)", len(jobs))

    # -------------------------------------------------------------- upload

    async def upload(self, project_id: str, filename: str | None, file: BinaryIO) -> tuple[Document, bool]:
        """Validate and store an upload, then queue its ingestion. Returns (document, created): ``created`` is False
        when the project already has a document with identical content (that document is returned unchanged).
        A duplicate of a FAILED document is ingested again instead (``created`` True, same document id)."""
        await self.projects.get(project_id)  # 404 before reading anything
        check = check_name(filename, self.settings.ingestion.allowed_extensions)
        size, sha256 = await asyncio.to_thread(measure, file, self.max_upload_bytes)
        await asyncio.to_thread(sniff, check, file)

        async with self._upload_lock:
            existing = await self.documents.find_duplicate(project_id, sha256)
            if existing is not None and existing.status != FAILED:
                return existing, False
            if existing is not None:
                doc, job_id = await self.documents.requeue(existing.id)
            else:
                document_id = new_id("doc")
                key = storage_key(project_id, document_id, 1, check.extension)
                file.seek(0)
                await self.store.put(key, file)
                try:
                    doc, job_id = await self.documents.create(
                        project_id,
                        document_id=document_id,
                        filename=check.filename,
                        mime=check.mime,
                        size_bytes=size,
                        sha256=sha256,
                        storage_key=key,
                    )
                except BaseException:
                    await self.store.delete(key)
                    raise
        await self._submit(job_id)
        log.info("document %s (%s, %d bytes) queued for ingestion", doc.id, doc.filename, doc.size_bytes)
        return doc, True

    async def _submit(self, job_id: str) -> None:
        await self.queue.submit(f"ingest {job_id}", partial(self.run_job, job_id))

    # -------------------------------------------------------------- ingestion job

    async def run_job(self, job_id: str) -> None:
        """One ingestion attempt. Never raises for ingestion problems: the outcome is the document's status."""
        target = await self.documents.start_job(job_id)
        if target is None:
            return
        doc_id = target.document_id
        ext = os.path.splitext(target.storage_key)[1]
        temp_name = f"{doc_id}{ext}"  # parser reads the extension; messages name the file
        stage = "fetch"
        try:
            async with self.store.open_temp_copy(target.storage_key, filename=temp_name) as path:
                stage = "ingest"
                await self.documents.set_stage(job_id, stage)
                result = await self.ingestion.ingest_file(path, target.project_id, doc_id, target.version)
            stage = "persist"
            saved = await self.documents.finish_job(
                job_id, page_count=result.page_count, chunk_count=result.chunk_count, tables=result.tables
            )
        except Exception as e:
            error = describe_failure(e).replace(temp_name, target.filename)
            log.warning("ingestion of %s (%s) failed at %s: %s", doc_id, target.filename, stage, error)
            await self._purge_vectors(doc_id)  # a document that isn't READY must not be searchable
            await self.documents.fail_job(job_id, error, stage=stage)
            return
        if not saved:  # deleted while ingesting: drop what was just indexed
            await self._purge_vectors(doc_id)
            return
        t = result.timings
        log.info(
            "document %s READY: %s pages, %d chunks, %d tables in %.1fs (parse %.1fs, embed %.1fs)",
            doc_id,
            result.page_count,
            result.chunk_count,
            result.table_count,
            t.total_s,
            t.parse_s,
            t.embed_s,
        )

    async def _purge_vectors(self, document_id: str) -> None:
        try:
            await self.vectors.delete_document(document_id)
        except Exception as e:
            log.warning("could not remove vectors of %s: %s", document_id, e)

    async def _release_parser(self) -> None:
        await asyncio.to_thread(self.ingestion.parser.release)

    # -------------------------------------------------------------- deletion

    async def delete_document(self, document_id: str) -> None:
        """Delete a document: its vectors, rows (versions, jobs, tables; removed from chat scopes) and stored files."""
        files = await self.documents.files(document_id)
        await self._delete_vectors(partial(self.vectors.delete_document, document_id), f"document {document_id}")
        await self.documents.delete(document_id)
        await self._delete_files(f"{files.project_id}/{document_id}", files.storage_keys)

    async def delete_project(self, project_id: str) -> None:
        """Delete a project: its vectors, rows (documents, chats, messages, summaries cascade) and stored files."""
        await self.projects.get(project_id)
        await self._delete_vectors(partial(self.vectors.delete_project, project_id), f"project {project_id}")
        document_ids = await self.projects.delete(project_id)
        await self._delete_files(project_id, [])
        log.info("project %s deleted with %d document(s)", project_id, len(document_ids))

    async def _delete_vectors(self, delete: Callable[[], Awaitable[None]], what: str) -> None:
        try:
            await delete()
        except Exception as e:
            raise Unavailable(
                f"could not remove the vectors of {what} ({type(e).__name__}: {e}); nothing was deleted, try again"
            ) from e

    async def _delete_files(self, prefix: str, keys: list[str]) -> None:
        """Best effort: the records are gone either way; a leftover file is logged, never served."""
        try:
            for key in keys:
                await self.store.delete(key)
            await self.store.delete_prefix(prefix)
        except Exception as e:
            log.warning("could not delete stored files under %s: %s", prefix, e)
