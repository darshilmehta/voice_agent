"""Documents of a project: the database side of upload, ingestion jobs and deletion (docs/DESIGN.md §3.1, §3.9).

Orchestration (object store, job queue, vectors) lives in ``services.document_pipeline``; this module only reads and
writes rows. Status lifecycles:

- document: PENDING → PROCESSING → READY (page and chunk counts set) | FAILED (``error`` set)
- ingestion job (one per attempt): QUEUED → RUNNING → SUCCEEDED | FAILED | CANCELLED (interrupted by a restart)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models as orm
from ..domain.projects import Document, DocumentTable
from ..providers.ingestion import ParsedTable
from .base import NotFound, Service, get_or_404
from .canvas.store import delete_document_visuals

# Document statuses
PENDING, PROCESSING, READY, FAILED = "PENDING", "PROCESSING", "READY", "FAILED"
# Ingestion job statuses
QUEUED, RUNNING, SUCCEEDED, CANCELLED = "QUEUED", "RUNNING", "SUCCEEDED", "CANCELLED"

ERROR_MAX = 2000


@dataclass(frozen=True, slots=True)
class JobTarget:
    """What an ingestion job works on, read when the job starts."""

    job_id: str
    document_id: str
    project_id: str
    version: int
    filename: str
    storage_key: str


@dataclass(frozen=True, slots=True)
class DocumentFiles:
    """Where a document's data lives outside the database, for deletion."""

    document_id: str
    project_id: str
    storage_keys: list[str]


class DocumentService(Service):
    async def list(self, project_id: str) -> list[Document]:
        """The project's documents, newest first."""
        stmt = (
            select(orm.Document)
            .where(orm.Document.project_id == project_id)
            .order_by(orm.Document.created_at.desc(), orm.Document.id.desc())
        )
        async with self.db.session() as s:
            await get_or_404(s, orm.Project, project_id, "project")
            return [Document.model_validate(d) for d in (await s.scalars(stmt)).all()]

    async def get(self, document_id: str) -> Document:
        async with self.db.session() as s:
            return Document.model_validate(await get_or_404(s, orm.Document, document_id, "document"))

    async def find_duplicate(self, project_id: str, sha256: str) -> Document | None:
        """The project's document whose current version has this content, if any (dedupe, §3.1)."""
        stmt = (
            select(orm.Document)
            .where(orm.Document.project_id == project_id, orm.Document.sha256 == sha256)
            .order_by(orm.Document.created_at)
            .limit(1)
        )
        async with self.db.session() as s:
            await get_or_404(s, orm.Project, project_id, "project")
            row = (await s.scalars(stmt)).first()
            return Document.model_validate(row) if row is not None else None

    async def create(
        self,
        project_id: str,
        *,
        document_id: str,
        filename: str,
        mime: str,
        size_bytes: int,
        sha256: str,
        storage_key: str,
    ) -> tuple[Document, str]:
        """A new PENDING document with its first version and a QUEUED ingestion job. Returns (document, job id)."""
        async with self.db.session() as s:
            project = await get_or_404(s, orm.Project, project_id, "project")
            now = self.now()
            doc = orm.Document(
                id=document_id,
                project_id=project_id,
                filename=filename,
                mime=mime,
                size_bytes=size_bytes,
                sha256=sha256,
                version=1,
                status=PENDING,
                created_at=now,
                updated_at=now,
            )
            s.add(doc)
            await s.flush()  # no ORM relationships: the parent must exist before its children are inserted
            s.add(
                orm.DocumentVersion(
                    document_id=document_id,
                    version=1,
                    filename=filename,
                    mime=mime,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    storage_key=storage_key,
                    created_at=now,
                )
            )
            job = orm.IngestionJob(document_id=document_id, version=1, status=QUEUED, attempts=0, created_at=now)
            s.add(job)
            project.updated_at = now
            await s.flush()
            return Document.model_validate(doc), job.id

    async def requeue(self, document_id: str) -> tuple[Document, str]:
        """Ingest the current version again: the document goes back to PENDING (error cleared) with a new QUEUED job."""
        async with self.db.session() as s:
            doc = await get_or_404(s, orm.Document, document_id, "document")
            now = self.now()
            doc.status, doc.error, doc.page_count, doc.chunk_count, doc.updated_at = PENDING, None, None, None, now
            job = orm.IngestionJob(
                document_id=document_id, version=doc.version, status=QUEUED, attempts=0, created_at=now
            )
            s.add(job)
            await s.flush()
            return Document.model_validate(doc), job.id

    async def indexed_documents(self) -> list[str]:
        """READY documents of every project with no ingestion job queued or running: the ones whose index a newer
        chunking version may have made stale. Oldest first."""
        active = select(orm.IngestionJob.document_id).where(orm.IngestionJob.status.in_((QUEUED, RUNNING)))
        stmt = (
            select(orm.Document.id)
            .where(orm.Document.status == READY, orm.Document.id.not_in(active))
            .order_by(orm.Document.created_at, orm.Document.id)
        )
        async with self.db.session() as s:
            return list((await s.scalars(stmt)).all())

    async def queue_reindex(self, document_id: str) -> str | None:
        """A QUEUED job that ingests a READY document's current version again (its index is stale). The document
        stays READY, and searchable on its old index, until the job starts (PROCESSING, then READY again).
        None when the document is gone, not READY, or already has a job queued or running."""
        async with self.db.session() as s:
            doc = await s.get(orm.Document, document_id)
            if doc is None or doc.status != READY:
                return None
            active = await s.scalar(
                select(orm.IngestionJob.id).where(
                    orm.IngestionJob.document_id == document_id, orm.IngestionJob.status.in_((QUEUED, RUNNING))
                )
            )
            if active is not None:
                return None
            job = orm.IngestionJob(
                document_id=document_id, version=doc.version, status=QUEUED, attempts=0, created_at=self.now()
            )
            s.add(job)
            await s.flush()
            return job.id

    async def start_job(self, job_id: str) -> JobTarget | None:
        """Mark a QUEUED job RUNNING and its document PROCESSING. None when there is nothing to do (the document was
        deleted, or the job already ran)."""
        async with self.db.session() as s:
            job = await s.get(orm.IngestionJob, job_id)
            if job is None or job.status != QUEUED:
                return None
            doc = await s.get(orm.Document, job.document_id)
            version = await s.scalar(
                select(orm.DocumentVersion).where(
                    orm.DocumentVersion.document_id == job.document_id, orm.DocumentVersion.version == job.version
                )
            )
            if doc is None or version is None:
                return None
            now = self.now()
            job.status, job.stage, job.attempts, job.started_at, job.error = (
                RUNNING,
                "fetch",
                job.attempts + 1,
                now,
                None,
            )
            doc.status, doc.error, doc.updated_at = PROCESSING, None, now
            return JobTarget(job.id, doc.id, doc.project_id, job.version, version.filename, version.storage_key)

    async def set_stage(self, job_id: str, stage: str) -> None:
        async with self.db.session() as s:
            await s.execute(update(orm.IngestionJob).where(orm.IngestionJob.id == job_id).values(stage=stage))

    async def finish_job(
        self, job_id: str, *, page_count: int | None, chunk_count: int, tables: Sequence[ParsedTable]
    ) -> bool:
        """Record a successful ingestion in one transaction: the version's tables (replacing earlier ones), document
        READY with its counts, job SUCCEEDED. False when the document no longer exists (deleted meanwhile)."""
        async with self.db.session() as s:
            job = await s.get(orm.IngestionJob, job_id)
            doc = await s.get(orm.Document, job.document_id) if job is not None else None
            if job is None or doc is None:
                return False
            now = self.now()
            await _replace_tables(s, doc.id, job.version, tables, now)
            doc.status, doc.page_count, doc.chunk_count, doc.error, doc.updated_at = (
                READY,
                page_count,
                chunk_count,
                None,
                now,
            )
            job.status, job.stage, job.finished_at = SUCCEEDED, None, now
            return True

    async def fail_job(self, job_id: str, error: str, *, stage: str | None) -> None:
        """Record a failed ingestion: job FAILED at ``stage``, document FAILED with the error (if both still exist)."""
        error = _clip(error)
        async with self.db.session() as s:
            job = await s.get(orm.IngestionJob, job_id)
            if job is None:
                return
            now = self.now()
            job.status, job.stage, job.error, job.finished_at = FAILED, stage, error, now
            doc = await s.get(orm.Document, job.document_id)
            if doc is not None:
                doc.status, doc.error, doc.updated_at = FAILED, error, now

    async def recover_interrupted(self) -> list[str]:
        """After a restart nothing is running: jobs left RUNNING were interrupted (marked CANCELLED, their documents
        get a new QUEUED job) and jobs left QUEUED were never started. Returns the job ids to submit, oldest first."""
        async with self.db.session() as s:
            now = self.now()
            running = (await s.scalars(select(orm.IngestionJob).where(orm.IngestionJob.status == RUNNING))).all()
            for job in running:
                job.status, job.error, job.finished_at = CANCELLED, "interrupted: the server stopped", now
                s.add(orm.IngestionJob(document_id=job.document_id, version=job.version, status=QUEUED, created_at=now))
            await s.execute(
                update(orm.Document).where(orm.Document.status == PROCESSING).values(status=PENDING, updated_at=now)
            )
            await s.flush()
            queued = select(orm.IngestionJob.id).where(orm.IngestionJob.status == QUEUED)
            return list((await s.scalars(queued.order_by(orm.IngestionJob.created_at, orm.IngestionJob.id))).all())

    async def files(self, document_id: str) -> DocumentFiles:
        """The document's project and stored files (every version). Raises NotFound."""
        async with self.db.session() as s:
            doc = await get_or_404(s, orm.Document, document_id, "document")
            keys = select(orm.DocumentVersion.storage_key).where(orm.DocumentVersion.document_id == document_id)
            return DocumentFiles(doc.id, doc.project_id, list((await s.scalars(keys)).all()))

    async def delete(self, document_id: str) -> None:
        """Delete the document's rows (versions, jobs, tables and their datasets cascade), the canvas visuals built
        from it, and remove it from every chat's ``document_scope``; a scope left empty becomes null (all documents),
        the only valid empty value."""
        async with self.db.session() as s:
            doc = await get_or_404(s, orm.Document, document_id, "document")
            chats = (
                await s.scalars(
                    select(orm.Chat).where(orm.Chat.project_id == doc.project_id, orm.Chat.document_scope.is_not(None))
                )
            ).all()
            now = self.now()
            for chat in chats:
                scope = chat.document_scope or []
                if document_id in scope:
                    chat.document_scope = [d for d in scope if d != document_id] or None
                    chat.updated_at = now
            await delete_document_visuals(s, doc.project_id, document_id)  # canvas panels built from its tables
            result = await s.execute(delete(orm.Document).where(orm.Document.id == document_id))
            if result.rowcount == 0:  # type: ignore[attr-defined]
                raise NotFound("document", document_id)

    async def ready_documents(self, project_id: str, scope: Sequence[str] | None = None) -> dict[str, str]:
        """READY documents of the project (narrowed to ``scope`` when given): id → filename, oldest first. Retrieval
        is limited to these, so documents still ingesting, failed, or deleted are never cited."""
        stmt = (
            select(orm.Document.id, orm.Document.filename)
            .where(orm.Document.project_id == project_id, orm.Document.status == READY)
            .order_by(orm.Document.created_at, orm.Document.id)
        )
        if scope is not None:
            stmt = stmt.where(orm.Document.id.in_(list(scope)))
        async with self.db.session() as s:
            return {doc_id: filename for doc_id, filename in (await s.execute(stmt)).all()}

    async def tables(self, document_id: str) -> list[DocumentTable]:
        """The parsed tables of the document's current version, in document order."""
        async with self.db.session() as s:
            doc = await get_or_404(s, orm.Document, document_id, "document")
            stmt = (
                select(orm.DocumentTable)
                .where(orm.DocumentTable.document_id == document_id, orm.DocumentTable.version == doc.version)
                .order_by(orm.DocumentTable.table_index)
            )
            return [DocumentTable.model_validate(t) for t in (await s.scalars(stmt)).all()]


async def _replace_tables(
    s: AsyncSession, document_id: str, version: int, tables: Sequence[ParsedTable], now: datetime
) -> None:
    await s.execute(
        delete(orm.DocumentTable).where(
            orm.DocumentTable.document_id == document_id, orm.DocumentTable.version == version
        )
    )
    for t in tables:
        s.add(
            orm.DocumentTable(
                document_id=document_id,
                version=version,
                table_index=t.index,
                page_start=t.page_start,
                page_end=t.page_end,
                bbox=list(t.bbox) if t.bbox is not None else None,
                heading_path=list(t.heading_path),
                caption=t.caption,
                num_rows=t.num_rows,
                num_cols=t.num_cols,
                markdown=t.markdown,
                cells=[c.model_dump(mode="json") for c in t.cells],
                created_at=now,
            )
        )


def _clip(text: str) -> str:
    text = text.strip() or "unknown error"
    return text if len(text) <= ERROR_MAX else text[: ERROR_MAX - 1] + "…"
