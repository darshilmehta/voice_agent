"""The canvas's rows: typed datasets, chat and overview visuals, overview build markers (docs/DESIGN.md §12.1).

Database only; orchestration (typing after ingestion, the overview, the planner) is ``service.CanvasService``.
Deletes cascade in the database: a document's tables take their datasets with them, a chat its panels, a project its
overview and panels. A visual whose cells came from a deleted document is removed with it (``delete_document_visuals``,
called inside ``DocumentService.delete``'s transaction).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import models as orm
from ...domain.canvas import CanvasOp, Visual
from ...domain.datasets import DatasetSummary, TypedDataset
from ..base import NotFound, Service, get_or_404

# Document statuses (as in ``services.documents``, which imports this module for its cascade)
PENDING, PROCESSING, READY = "PENDING", "PROCESSING", "READY"


def _visual(row: orm.CanvasVisual) -> Visual:
    return Visual.model_validate(
        {**row.visual, "position": row.position, "pinned": row.pinned, "updated_at": row.updated_at}
    )


def _dataset(row: orm.TableDataset) -> TypedDataset:
    return TypedDataset.model_validate({**row.data, "id": row.id, "created_at": row.created_at})


async def delete_document_visuals(s: AsyncSession, project_id: str, document_id: str) -> int:
    """Delete the project's visuals (chat panels and overview) built from ``document_id``. Returns how many."""
    rows = (await s.scalars(select(orm.CanvasVisual).where(orm.CanvasVisual.project_id == project_id))).all()
    doomed = [r.id for r in rows if document_id in (r.document_ids or [])]
    if doomed:
        await s.execute(delete(orm.CanvasVisual).where(orm.CanvasVisual.id.in_(doomed)))
    return len(doomed)


class CanvasStore(Service):
    # -------------------------------------------------------------- datasets

    async def replace_datasets(self, document_id: str, version: int, datasets: Sequence[TypedDataset]) -> bool:
        """Store a document version's datasets, replacing any earlier ones. False when the document (or its tables)
        no longer exist."""
        async with self.db.session() as s:
            doc = await s.get(orm.Document, document_id)
            if doc is None or doc.version != version:
                return False
            table_ids = set(
                (
                    await s.scalars(
                        select(orm.DocumentTable.id).where(
                            orm.DocumentTable.document_id == document_id, orm.DocumentTable.version == version
                        )
                    )
                ).all()
            )
            if any(d.table_id not in table_ids for d in datasets):
                return False
            await s.execute(delete(orm.TableDataset).where(orm.TableDataset.document_id == document_id))
            now = self.now()
            for d in datasets:
                data = d.model_dump(mode="json", exclude={"id", "created_at"})
                s.add(
                    orm.TableDataset(
                        id=d.id,
                        table_id=d.table_id,
                        document_id=document_id,
                        version=version,
                        typer_version=d.typer_version,
                        chart_kind=d.chartability.kind,
                        chart_confidence=d.chartability.confidence,
                        data=data,
                        created_at=now,
                    )
                )
            return True

    async def document_datasets(self, document_id: str) -> list[TypedDataset]:
        stmt = (
            select(orm.TableDataset)
            .join(orm.Document, orm.Document.id == orm.TableDataset.document_id)
            .where(orm.TableDataset.document_id == document_id, orm.TableDataset.version == orm.Document.version)
        )
        async with self.db.session() as s:
            rows = (await s.scalars(stmt)).all()
        return sorted((_dataset(r) for r in rows), key=lambda d: d.table_index)

    async def project_datasets(self, project_id: str, document_ids: Sequence[str] | None = None) -> list[TypedDataset]:
        """Datasets of the project's READY documents (current versions; narrowed to ``document_ids``), in document
        then table order."""
        stmt = (
            select(orm.TableDataset, orm.Document.created_at)
            .join(orm.Document, orm.Document.id == orm.TableDataset.document_id)
            .where(
                orm.Document.project_id == project_id,
                orm.Document.status == READY,
                orm.TableDataset.version == orm.Document.version,
            )
        )
        if document_ids is not None:
            stmt = stmt.where(orm.Document.id.in_(list(document_ids)))
        async with self.db.session() as s:
            rows = (await s.execute(stmt)).all()
        datasets = [(created, _dataset(r)) for r, created in rows]
        datasets.sort(key=lambda cd: (cd[0], cd[1].document_id, cd[1].table_index))
        return [d for _, d in datasets]

    async def summaries(self, project_id: str) -> list[DatasetSummary]:
        filenames = await self.filenames(project_id)
        return [
            DatasetSummary(
                id=d.id,
                document_id=d.document_id,
                filename=filenames.get(d.document_id, ""),
                table_id=d.table_id,
                table_index=d.table_index,
                page=d.page_start,
                title=d.title,
                unit=d.unit,
                chartability=d.chartability,
                columns=d.columns,
                rows=d.rows,
                warnings=d.warnings,
            )
            for d in await self.project_datasets(project_id)
        ]

    async def documents_to_type(self, typer_version: str) -> list[tuple[str, str]]:
        """(document id, project id) of READY documents whose current tables have no datasets, or older ones."""
        tables = (
            select(orm.DocumentTable.document_id, func.count(orm.DocumentTable.id).label("n"))
            .join(orm.Document, orm.Document.id == orm.DocumentTable.document_id)
            .where(orm.Document.status == READY, orm.DocumentTable.version == orm.Document.version)
            .group_by(orm.DocumentTable.document_id)
        )
        typed = (
            select(orm.TableDataset.document_id, func.count(orm.TableDataset.id).label("n"))
            .join(orm.Document, orm.Document.id == orm.TableDataset.document_id)
            .where(orm.TableDataset.version == orm.Document.version, orm.TableDataset.typer_version == typer_version)
            .group_by(orm.TableDataset.document_id)
        )
        async with self.db.session() as s:
            have = dict((await s.execute(tables)).all())
            done = dict((await s.execute(typed)).all())
            todo = [doc for doc, n in have.items() if done.get(doc, 0) != n]
            if not todo:
                return []
            projects = dict(
                (
                    await s.execute(select(orm.Document.id, orm.Document.project_id).where(orm.Document.id.in_(todo)))
                ).all()
            )
        return [(doc, projects[doc]) for doc in todo if doc in projects]

    async def filenames(self, project_id: str) -> dict[str, str]:
        async with self.db.session() as s:
            rows = (
                await s.execute(
                    select(orm.Document.id, orm.Document.filename).where(orm.Document.project_id == project_id)
                )
            ).all()
        return dict(rows)  # type: ignore[arg-type]

    async def document_counts(self, project_id: str) -> dict[str, int]:
        async with self.db.session() as s:
            rows = (
                await s.execute(
                    select(orm.Document.status, func.count(orm.Document.id))
                    .where(orm.Document.project_id == project_id)
                    .group_by(orm.Document.status)
                )
            ).all()
        return dict(rows)  # type: ignore[arg-type]

    async def projects_with_documents(self) -> list[str]:
        async with self.db.session() as s:
            return list(
                (await s.scalars(select(orm.Document.project_id).where(orm.Document.status == READY).distinct())).all()
            )

    async def overview_inputs(self) -> dict[str, tuple[list[str], str | None]]:
        """Per project with READY documents: its datasets ("<id>:<typer version>") and its overview fingerprint."""
        datasets = (
            select(orm.Document.project_id, orm.TableDataset.id, orm.TableDataset.typer_version)
            .join(orm.Document, orm.Document.id == orm.TableDataset.document_id)
            .where(orm.Document.status == READY, orm.TableDataset.version == orm.Document.version)
        )
        out: dict[str, tuple[list[str], str | None]] = {p: ([], None) for p in await self.projects_with_documents()}
        async with self.db.session() as s:
            for project_id, dataset_id, typer in (await s.execute(datasets)).all():
                out.setdefault(project_id, ([], None))[0].append(f"{dataset_id}:{typer}")
            for marker in (await s.scalars(select(orm.ProjectOverview))).all():
                if marker.project_id in out:
                    out[marker.project_id] = (out[marker.project_id][0], marker.fingerprint)
        return out

    # -------------------------------------------------------------- chat canvas

    async def chat_context(self, chat_id: str) -> tuple[str, list[str] | None, str | None]:
        """(project id, document scope, language) of a chat. Raises NotFound."""
        async with self.db.session() as s:
            chat = await get_or_404(s, orm.Chat, chat_id, "chat")
            return chat.project_id, chat.document_scope, chat.language

    async def panels(self, chat_id: str) -> list[Visual]:
        async with self.db.session() as s:
            await get_or_404(s, orm.Chat, chat_id, "chat")
            return [_visual(r) for r in await self._chat_rows(s, chat_id)]

    async def _chat_rows(self, s: AsyncSession, chat_id: str) -> list[orm.CanvasVisual]:
        stmt = (
            select(orm.CanvasVisual)
            .where(orm.CanvasVisual.chat_id == chat_id)
            .order_by(orm.CanvasVisual.position, orm.CanvasVisual.created_at)
        )
        return list((await s.scalars(stmt)).all())

    @staticmethod
    def _renumber(rows: Sequence[orm.CanvasVisual]) -> None:
        for n, row in enumerate(rows):
            if row.position != n:
                row.position = n

    async def add_panel(
        self, chat_id: str, visual: Visual, spec: dict[str, Any], document_ids: Sequence[str], *, max_panels: int
    ) -> Visual:
        """Append a visual to the chat's canvas. Past ``max_panels`` the oldest unpinned panels are removed."""
        async with self.db.session() as s:
            chat = await get_or_404(s, orm.Chat, chat_id, "chat")
            rows = await self._chat_rows(s, chat_id)
            now = self.now()
            row = orm.CanvasVisual(
                id=visual.id,
                project_id=chat.project_id,
                chat_id=chat_id,
                position=len(rows),
                pinned=False,
                kind=visual.kind,
                document_ids=list(document_ids),
                spec=spec,
                visual=visual.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
            )
            s.add(row)
            rows.append(row)
            unpinned = sorted((r for r in rows if not r.pinned and r is not row), key=lambda r: r.created_at)
            while len(rows) > max_panels and unpinned:
                old = unpinned.pop(0)
                rows.remove(old)
                await s.delete(old)
            self._renumber(rows)
            await s.flush()
            return _visual(row)

    async def apply(self, chat_id: str, op: CanvasOp) -> list[Visual]:
        async with self.db.session() as s:
            await get_or_404(s, orm.Chat, chat_id, "chat")
            rows = await self._chat_rows(s, chat_id)
            target = next((r for r in rows if r.id == op.visual_id), None)
            if target is None:
                raise NotFound("visual", op.visual_id)
            now = self.now()
            if op.op == "remove":
                rows.remove(target)
                await s.delete(target)
            elif op.op in ("pin", "unpin"):
                target.pinned = op.op == "pin"
                target.updated_at = now
            elif op.op == "move":
                position = min(op.position if op.position is not None else len(rows) - 1, len(rows) - 1)
                rows.remove(target)
                rows.insert(position, target)
                target.updated_at = now
            self._renumber(rows)
            await s.flush()
            return [_visual(r) for r in rows]

    # -------------------------------------------------------------- overview

    async def overview_panels(self, project_id: str) -> list[Visual]:
        stmt = (
            select(orm.CanvasVisual)
            .where(orm.CanvasVisual.project_id == project_id, orm.CanvasVisual.chat_id.is_(None))
            .order_by(orm.CanvasVisual.position)
        )
        async with self.db.session() as s:
            await get_or_404(s, orm.Project, project_id, "project")
            return [_visual(r) for r in (await s.scalars(stmt)).all()]

    async def overview_fingerprint(self, project_id: str) -> str | None:
        async with self.db.session() as s:
            row = await s.get(orm.ProjectOverview, project_id)
            return row.fingerprint if row is not None else None

    async def replace_overview(
        self, project_id: str, built: Sequence[tuple[Visual, dict[str, Any], list[str]]], fingerprint: str
    ) -> list[Visual]:
        async with self.db.session() as s:
            if await s.get(orm.Project, project_id) is None:
                return []
            await s.execute(
                delete(orm.CanvasVisual).where(
                    orm.CanvasVisual.project_id == project_id, orm.CanvasVisual.chat_id.is_(None)
                )
            )
            now = self.now()
            rows = []
            for n, (visual, spec, document_ids) in enumerate(built):
                row = orm.CanvasVisual(
                    id=visual.id,
                    project_id=project_id,
                    chat_id=None,
                    position=n,
                    pinned=False,
                    kind=visual.kind,
                    document_ids=document_ids,
                    spec=spec,
                    visual=visual.model_dump(mode="json"),
                    created_at=now,
                    updated_at=now,
                )
                s.add(row)
                rows.append(row)
            marker = await s.get(orm.ProjectOverview, project_id)
            if marker is None:
                s.add(orm.ProjectOverview(project_id=project_id, fingerprint=fingerprint, built_at=now))
            else:
                marker.fingerprint, marker.built_at = fingerprint, now
            await s.flush()
            return [_visual(r) for r in rows]

    async def is_busy(self, project_id: str) -> bool:
        """Documents of the project still ingesting."""
        counts = await self.document_counts(project_id)
        return bool(counts.get(PENDING, 0) or counts.get(PROCESSING, 0))
