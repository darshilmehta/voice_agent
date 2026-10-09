"""The live visual canvas (docs/DESIGN.md §12.1): typed datasets after ingestion, the project overview, chat
canvases and the entry point the conversation will use. Long-lived (one per app, ``app.state.canvas``).

    ingestion READY ──document_ready──► type the version's tables (with the text around each, and its retrieval
                                        chunk) → table_datasets → rebuild the project's overview
    document deleted ─document_deleted─► rebuild the overview (its visuals went with the document's rows)
    startup ─────────backfill job (long lane, after the model preload)─► type READY documents that have tables but no
                                        (current) datasets, then rebuild overviews that are out of date
    API ─────────────canvas, ops, POST visuals (spec → validate → build → store), overview, datasets
    conversation ────prepare_visual(chat, question, …) → VisualEvent preparing / ready | failed, CanvasEvent
                     (a turn runs it after its answer's text is complete: ``conversation.TurnVisual``)
                     edit(chat, CanvasEdit, …) → the canvas events of a spoken edit, then its EditResult
                     visual_tables(chat, visual) → the tables behind a visual (questions about what is on screen)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import datetime

from ...db.types import new_id, utcnow
from ...domain.canvas import CanvasEvent, CanvasOp, CanvasPanels, ProjectOverview, Visual, VisualEvent
from ...domain.datasets import DatasetSummary, TableContext, TypedDataset
from ...domain.projects import Citation, DocumentTable
from ...providers.ingestion import Chunk, ParsedDocument
from ...providers.llm import LLMClient
from ...providers.registry import Container
from ...providers.runtime import LANE_LONG, JobQueue
from ...providers.storage import MetadataDB
from ...settings import Settings
from ..base import InvalidInput, NotFound
from ..documents import DocumentService
from .builder import build_visual
from .conversation import CanvasEdit, EditOutcome, EditResult, describe, resolve_target
from .datasets import TYPER_VERSION, table_contexts, type_document
from .overview import overview_specs
from .planner import VisualPlanner, builds, first_valid, visual_intent
from .spec import SpecError, VisualSpec, resolve, split_ref
from .store import CanvasStore

log = logging.getLogger(__name__)

OVERVIEW_VERSION = "o1"  # bump when the overview's choices change, so stored overviews are rebuilt


class CanvasService:
    def __init__(
        self,
        db: MetadataDB,
        *,
        settings: Settings,
        queue: JobQueue | None = None,
        llm: LLMClient | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.db = db
        self.store = CanvasStore(db, clock=clock)
        self.documents = DocumentService(db, clock=clock)
        self.settings = settings
        self.cfg = settings.canvas
        self.queue = queue
        self.now = clock
        self.planner = (
            VisualPlanner(
                llm,
                model=settings.llm.router_model,
                timeout_s=self.cfg.planner_timeout_ms / 1000,
                max_tokens=self.cfg.planner_max_tokens,
                max_candidates=self.cfg.planner_candidates,
            )
            if llm is not None
            else None
        )
        # Awaited before the backfill starts (the app passes the startup model preload, as for ingestion).
        self.wait_before_backfill: Callable[[], Awaitable[None]] | None = None
        self._building: dict[str, int] = {}  # project id → overview builds in progress
        self._locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def from_container(cls, container: Container) -> CanvasService:
        queue = container.providers.get("job_queue")
        llm = container.providers.get("llm")
        return cls(
            container["metadata_db"],  # type: ignore[arg-type]
            settings=container.settings,
            queue=queue if isinstance(queue, JobQueue) else None,
            llm=llm if isinstance(llm, LLMClient) else None,
        )

    @property
    def language(self) -> str:
        return self.settings.client.default_language

    def _lock(self, project_id: str) -> asyncio.Lock:
        return self._locks.setdefault(project_id, asyncio.Lock())

    # -------------------------------------------------------------- datasets (ingestion hook, backfill)

    async def document_ready(
        self,
        project_id: str,
        document_id: str,
        version: int,
        *,
        parsed: ParsedDocument | None = None,
        chunks: Sequence[Chunk] = (),
    ) -> list[TypedDataset]:
        """After a document becomes READY: type its tables and rebuild the project's overview. Never raises (the
        document is READY either way; the backfill retries what failed)."""
        self._building[project_id] = self._building.get(project_id, 0) + 1
        try:
            datasets = await self.type_document(document_id, version=version, parsed=parsed, chunks=chunks)
            await self.rebuild_overview(project_id)
            return datasets
        except Exception:
            log.exception("canvas: typing the tables of %s failed", document_id)
            return []
        finally:
            self._done_building(project_id)

    async def document_deleted(self, project_id: str, document_id: str) -> None:
        try:
            await self.rebuild_overview(project_id)
        except Exception:
            log.exception("canvas: rebuilding the overview of %s after deleting %s failed", project_id, document_id)

    def _done_building(self, project_id: str) -> None:
        n = self._building.get(project_id, 0) - 1
        if n > 0:
            self._building[project_id] = n
        else:
            self._building.pop(project_id, None)

    async def type_document(
        self,
        document_id: str,
        *,
        version: int | None = None,
        parsed: ParsedDocument | None = None,
        chunks: Sequence[Chunk] = (),
    ) -> list[TypedDataset]:
        """Type the current version's stored tables. The text around each table comes from ``parsed`` (at
        ingestion) or from the datasets typed before (a re-typing keeps it); the table's chunk from ``chunks``."""
        doc = await self.documents.get(document_id)
        if version is not None and doc.version != version:
            return []
        tables = await self.documents.tables(document_id)
        if not tables:
            return []
        previous = {d.table_index: d for d in await self.store.document_datasets(document_id)}
        contexts: dict[int, TableContext] = (
            table_contexts(parsed) if parsed is not None else {i: d.context for i, d in previous.items()}
        )
        chunk_ids = {c.table_index: c.chunk_id for c in chunks if c.table_index is not None} or {
            i: d.chunk_id for i, d in previous.items() if d.chunk_id
        }
        datasets = await asyncio.to_thread(
            type_document, tables, contexts, new_id=lambda: new_id("ds"), chunk_ids=chunk_ids
        )
        if not await self.store.replace_datasets(document_id, doc.version, datasets):
            return []
        kinds = ", ".join(f"{d.chartability.kind}" for d in datasets)
        log.info("canvas: typed %d table(s) of %s (%s)", len(datasets), document_id, kinds)
        return datasets

    async def backfill(self) -> int:
        """Type READY documents whose tables have no datasets (ingested before the canvas) or older ones, and
        rebuild out-of-date overviews. Returns the number of documents typed."""
        typed = 0
        for document_id, _ in await self.store.documents_to_type(TYPER_VERSION):
            try:
                if await self.type_document(document_id):
                    typed += 1
            except Exception:
                log.exception("canvas backfill: typing %s failed", document_id)
        for project_id in await self.stale_overviews():
            try:
                await self.rebuild_overview(project_id)
            except Exception:
                log.exception("canvas backfill: overview of %s failed", project_id)
        if typed:
            log.info("canvas backfill: typed the tables of %d document(s)", typed)
        return typed

    async def stale_overviews(self) -> list[str]:
        """Projects with READY documents whose overview was built from other datasets (or never)."""
        inputs = await self.store.overview_inputs()
        return [p for p, (keys, stored) in inputs.items() if stored != self._fingerprint_of(keys)]

    async def schedule_backfill(self) -> None:
        """Queue the backfill in the long lane (behind any ingestion queued before it, after the model preload),
        when there is something to do."""
        if self.queue is None:
            return
        try:
            needed = bool(await self.store.documents_to_type(TYPER_VERSION) or await self.stale_overviews())
        except NotImplementedError:  # placeholder metadata DB (cloud template in tests)
            return
        if not needed:
            return

        async def job() -> None:
            if self.wait_before_backfill is not None:
                await self.wait_before_backfill()
            await self.backfill()

        try:
            await self.queue.submit("canvas backfill", job, lane=LANE_LONG)
        except (NotImplementedError, RuntimeError) as e:
            log.warning("canvas backfill not queued: %s", e)

    async def dataset_summaries(self, project_id: str) -> list[DatasetSummary]:
        await self._project(project_id)
        return await self.store.summaries(project_id)

    async def _project(self, project_id: str) -> None:
        from ..projects import ProjectService

        await ProjectService(self.db).get(project_id)

    # -------------------------------------------------------------- overview

    def _fingerprint(self, datasets: Sequence[TypedDataset]) -> str:
        return self._fingerprint_of([f"{d.id}:{d.typer_version}" for d in datasets])

    def _fingerprint_of(self, dataset_keys: Sequence[str]) -> str:
        parts = [OVERVIEW_VERSION, str(self.cfg.overview_panels), self.language, *sorted(dataset_keys)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    async def rebuild_overview(self, project_id: str, *, force: bool = False) -> list[Visual]:
        """Build the overview from the project's datasets, unless they haven't changed since the last build."""
        async with self._lock(project_id):
            self._building[project_id] = self._building.get(project_id, 0) + 1
            try:
                datasets = await self.store.project_datasets(project_id)
                fingerprint = self._fingerprint(datasets)
                if not force and fingerprint == await self.store.overview_fingerprint(project_id):
                    return await self.store.overview_panels(project_id)
                filenames = await self.store.filenames(project_id)
                by_id = {d.id: d for d in datasets}
                built: list[tuple[Visual, dict[str, object], list[str]]] = []
                # a panel rebuilt from the same spec keeps its id (a "show in chat" from a page loaded earlier works)
                previous = await self.store.overview_ids_by_spec(project_id)

                def check(spec: VisualSpec) -> bool:
                    key = _spec_key(spec.model_dump(mode="json", by_alias=True))
                    try:
                        visual = self._build(
                            spec, by_id, filenames, project_id=project_id, chat_id=None, visual_id=previous.get(key)
                        )
                    except (SpecError, AssertionError) as e:
                        log.info("canvas overview: skipped a %s (%s)", spec.kind, e)
                        return False
                    built.append((visual, spec.model_dump(mode="json", by_alias=True), _documents(spec, by_id)))
                    return True

                if self.cfg.overview_panels:
                    overview_specs(datasets, language=self.language, max_panels=self.cfg.overview_panels, check=check)
                panels = await self.store.replace_overview(project_id, built[: self.cfg.overview_panels], fingerprint)
                log.info("canvas: overview of %s rebuilt with %d panel(s)", project_id, len(panels))
                return panels
            finally:
                self._done_building(project_id)

    async def overview(self, project_id: str) -> ProjectOverview:
        panels = await self.store.overview_panels(project_id)  # 404 for an unknown project
        building = project_id in self._building or await self.store.is_busy(project_id)
        if building:
            return ProjectOverview(panels=panels, status="building")
        return ProjectOverview(panels=panels, status="ready" if panels else "none")

    # -------------------------------------------------------------- chat canvas

    async def canvas(self, chat_id: str) -> CanvasPanels:
        return CanvasPanels(panels=await self.store.panels(chat_id))

    async def apply(self, chat_id: str, op: CanvasOp) -> CanvasPanels:
        if op.op == "move" and op.position is None:
            raise InvalidInput("move needs a position")
        return CanvasPanels(panels=await self.store.apply(chat_id, op, max_panels=self.cfg.max_panels))

    def _build(
        self,
        spec: VisualSpec,
        datasets: dict[str, TypedDataset],
        filenames: dict[str, str],
        *,
        project_id: str,
        chat_id: str | None,
        visual_id: str | None = None,
        sources: Sequence[Citation] = (),
    ) -> Visual:
        resolved = resolve(spec, datasets, filenames=filenames)
        return build_visual(
            resolved,
            visual_id=visual_id or new_id("vis"),
            project_id=project_id,
            chat_id=chat_id,
            filenames=filenames,
            now=self.now(),
            existing_sources=sources,
        )

    async def _chat_datasets(self, chat_id: str) -> tuple[str, dict[str, TypedDataset], dict[str, str], str | None]:
        project_id, scope, language = await self.store.chat_context(chat_id)
        datasets = await self.store.project_datasets(project_id, scope)
        return project_id, {d.id: d for d in datasets}, await self.store.filenames(project_id), language

    async def add_visual(self, chat_id: str, spec: VisualSpec, *, sources: Sequence[Citation] = ()) -> Visual:
        """Validate ``spec`` against the chat's datasets (its project's READY documents, within its scope), build
        the visual and add it to the chat's canvas. Raises InvalidInput listing what doesn't resolve."""
        project_id, datasets, filenames, _ = await self._chat_datasets(chat_id)
        try:
            visual = self._build(spec, datasets, filenames, project_id=project_id, chat_id=chat_id, sources=sources)
        except SpecError as e:
            raise InvalidInput("; ".join(e.problems)) from None
        return await self.store.add_panel(
            chat_id,
            visual,
            spec.model_dump(mode="json", by_alias=True),
            _documents(spec, datasets),
            max_panels=self.cfg.max_panels,
        )

    async def prepare_visual(
        self,
        chat_id: str,
        question: str,
        *,
        language: str,
        answer: str | None = None,
        sources: Sequence[Citation] = (),
        query_en: str | None = None,
        force: bool = False,
        visual_id: str | None = None,
    ) -> AsyncIterator[VisualEvent | CanvasEvent]:
        """The conversation's entry point (run beside the spoken answer, never before it): decide whether the
        question deserves a visual, plan it with the model, build it from the cells, add it to the chat's canvas.

        Yields nothing when the question doesn't call for a visual (``visual_intent`` "none", unless ``force``);
        otherwise ``visual{phase: preparing}`` first (show a skeleton), then ``visual{phase: ready, visual}`` and
        ``canvas{panels}``, or ``visual{phase: failed, detail}``. ``sources``: the turn's citations, so the visual's
        cells reuse their [S#] ids. ``visual_id``: the id to give it (the turn announces it before planning starts).
        Never raises for planning or building problems."""
        if self.planner is None:
            return
        if visual_intent(question, answer) == "none" and not force:
            return
        visual_id = visual_id or new_id("vis")
        yield VisualEvent(phase="preparing", visual_id=visual_id)
        try:
            project_id, datasets, filenames, _ = await self._chat_datasets(chat_id)
            spec = await self.planner.plan(
                question,
                language,
                list(datasets.values()),
                answer=answer,
                filenames=filenames,
                query_en=query_en,
                source_chunks=[c.chunk_id for c in sources],
                force=force,
            )
            if spec is None:
                yield VisualEvent(phase="failed", visual_id=visual_id, detail="no table fits this question")
                return
            visual = self._build(
                spec, datasets, filenames, project_id=project_id, chat_id=chat_id, visual_id=visual_id, sources=sources
            )
            stored = await self.store.add_panel(
                chat_id,
                visual,
                spec.model_dump(mode="json", by_alias=True),
                _documents(spec, datasets),
                max_panels=self.cfg.max_panels,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # the conversation is unaffected by a visual that fails
            log.warning("canvas: visual for %r failed: %s", question[:80], e)
            yield VisualEvent(phase="failed", visual_id=visual_id, detail=str(e)[:300])
            return
        yield VisualEvent(phase="ready", visual_id=visual_id, visual=stored)
        with suppress(Exception):
            yield CanvasEvent(panels=await self.store.panels(chat_id))

    # -------------------------------------------------------------- the conversation's edits and questions

    async def edit(
        self,
        chat_id: str,
        edit: CanvasEdit,
        *,
        utterance: str,
        language: str,
        query_en: str | None = None,
    ) -> AsyncIterator[VisualEvent | CanvasEvent | EditResult]:
        """An edit said in the conversation (``conversation.parse_edit``), applied to the visual it means
        (``resolve_target``). Yields the canvas events (a rebuilt visual: ``visual{ready}`` with its id kept, a model
        rebuild ``visual{preparing}`` first; then the ``canvas`` snapshot) and, last, the ``EditResult``. Rebuilds go
        through the spec, the validator and the builder: every number is still a cell. Never raises for an edit that
        can't be done (the result says why)."""
        panels = await self.store.panels(chat_id)
        if not panels:
            yield EditResult("nothing", edit.op)
            return
        if edit.op == "clear":
            yield CanvasEvent(panels=await self.store.remove_unpinned(chat_id))
            yield EditResult("done", edit.op)
            return
        target = resolve_target(utterance, panels, kind_hint=edit.target_kind)
        assert target is not None
        if edit.op in ("remove", "pin", "unpin"):
            canvas = await self.apply(chat_id, CanvasOp(op=edit.op, visual_id=target.id))
            yield CanvasEvent(panels=canvas.panels)
            yield EditResult("done", edit.op, target.id)
            return
        project_id, datasets, filenames, _ = await self._chat_datasets(chat_id)
        try:
            spec = VisualSpec.model_validate(await self.store.panel_spec(chat_id, target.id))
        except (NotFound, ValueError):
            yield EditResult("failed", edit.op, target.id, detail="the visual's spec can't be read")
            return
        changed: VisualSpec | None = None
        outcome: EditOutcome = "done"
        if edit.op == "kind" and edit.kind is not None:
            if edit.kind == target.kind:
                yield EditResult("same", edit.op, target.id)
                return
            changed = first_valid(spec.model_copy(update={"kind": edit.kind}), datasets, builds(datasets, filenames))
            outcome = "done" if changed is not None else "cannot"
        elif edit.op in ("periods", "only"):
            available = _periods_of(spec, datasets)
            asked = [p for p in edit.periods if p in available]
            if asked and edit.op == "only":
                changed = spec.model_copy(update={"periods": asked, "highlight": [], "calculations": []})
            elif asked:
                shown = list(spec.periods) or list(available)
                if all(p in shown for p in asked):
                    yield EditResult("already", edit.op, target.id)
                    return
                order = {p: n for n, p in enumerate(available)}
                periods = sorted(dict.fromkeys([*shown, *asked]), key=lambda p: order.get(p, 0))
                changed = spec.model_copy(update={"periods": periods, "calculations": []})
            if changed is not None:
                changed = first_valid(changed, datasets, builds(datasets, filenames))
                outcome = "done" if changed is not None else "cannot"
        if changed is None and outcome == "done":  # the rules can't say: the planner, with the visual as context
            async for event in self._edit_with_model(
                chat_id, target, edit, utterance=utterance, language=language, query_en=query_en
            ):
                yield event
            return
        if changed is None:
            yield EditResult(outcome, edit.op, target.id)
            return
        visual = self._build(
            changed,
            datasets,
            filenames,
            project_id=project_id,
            chat_id=chat_id,
            visual_id=target.id,
            sources=target.sources,
        )
        stored = await self.store.replace_panel(
            chat_id, target.id, visual, changed.model_dump(mode="json", by_alias=True), _documents(changed, datasets)
        )
        yield VisualEvent(phase="ready", visual_id=target.id, visual=stored)
        yield CanvasEvent(panels=await self.store.panels(chat_id))
        yield EditResult("done", edit.op, target.id)

    async def _edit_with_model(
        self,
        chat_id: str,
        target: Visual,
        edit: CanvasEdit,
        *,
        utterance: str,
        language: str,
        query_en: str | None,
    ) -> AsyncIterator[VisualEvent | CanvasEvent | EditResult]:
        """An edit the rules can't read ("add profit to that chart", FY23 from another table): the planner chooses
        again for the request, told what is on screen, with the visual's tables first among the candidates."""
        if self.planner is None:
            yield EditResult("failed", edit.op, target.id, source="model", detail="no planner")
            return
        yield VisualEvent(phase="preparing", visual_id=target.id)  # "Updating…" on the panel
        try:
            project_id, datasets, filenames, _ = await self._chat_datasets(chat_id)
            spec = VisualSpec.model_validate(await self.store.panel_spec(chat_id, target.id))
            chunks = [d.chunk_id for d in (datasets.get(i) for i in spec.datasets) if d is not None and d.chunk_id]
            question = f"Change the chart on screen ({describe(target)}): {utterance}"
            english = f"Change the chart on screen ({describe(target)}): {query_en}" if query_en else None
            changed = await self.planner.plan(
                question,
                language,
                list(datasets.values()),
                filenames=filenames,
                query_en=english,
                source_chunks=chunks,
                force=True,
            )
            if changed is None:
                yield VisualEvent(phase="failed", visual_id=target.id, detail="no table fits that change")
                yield EditResult("failed", edit.op, target.id, source="model", detail="no table fits that change")
                return
            visual = self._build(
                changed,
                datasets,
                filenames,
                project_id=project_id,
                chat_id=chat_id,
                visual_id=target.id,
                sources=target.sources,
            )
            stored = await self.store.replace_panel(
                chat_id,
                target.id,
                visual,
                changed.model_dump(mode="json", by_alias=True),
                _documents(changed, datasets),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("canvas: editing %s with the model failed: %s", target.id, e)
            yield VisualEvent(phase="failed", visual_id=target.id, detail=str(e)[:300])
            yield EditResult("failed", edit.op, target.id, source="model", detail=str(e)[:300])
            return
        yield VisualEvent(phase="ready", visual_id=target.id, visual=stored)
        yield CanvasEvent(panels=await self.store.panels(chat_id))
        yield EditResult("done", edit.op, target.id, source="model")

    async def visual_tables(self, chat_id: str, visual: Visual) -> list[tuple[TypedDataset, DocumentTable]]:
        """The tables a chat's visual was built from (a question about what is on screen is answered from them)."""
        try:
            spec = VisualSpec.model_validate(await self.store.panel_spec(chat_id, visual.id))
        except (NotFound, ValueError):
            return []
        _, datasets, _, _ = await self._chat_datasets(chat_id)
        out = []
        for dataset_id in spec.datasets:
            ds = datasets.get(dataset_id)
            if ds is None:
                continue
            table = next((t for t in await self.documents.tables(ds.document_id) if t.id == ds.table_id), None)
            if table is not None:
                out.append((ds, table))
        return out


def _periods_of(spec: VisualSpec, datasets: dict[str, TypedDataset]) -> list[str]:
    """The periods of the spec's datasets, in time order."""
    found: dict[str, tuple[int, int]] = {}
    for dataset_id in spec.datasets:
        ds = datasets.get(dataset_id)
        if ds is None:
            continue
        for p in [*(c.period for c in ds.columns), *(r.period for r in ds.rows)]:
            if p is not None:
                found.setdefault(p.label, p.sort_key)
    return sorted(found, key=lambda label: found[label])


def _spec_key(spec: dict[str, object]) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False)


def _documents(spec: VisualSpec, datasets: dict[str, TypedDataset]) -> list[str]:
    ids = []
    for ref in [*spec.series, *spec.categories]:
        with suppress(ValueError):
            dataset_id, _ = split_ref(ref)
            if dataset_id in datasets:
                ids.append(datasets[dataset_id].document_id)
    ids += [datasets[d].document_id for d in spec.datasets if d in datasets]
    return list(dict.fromkeys(ids))
