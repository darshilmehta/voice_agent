"""A turn's web search (docs/DESIGN.md §3.7): started as soon as the route asks for it, running beside document
retrieval, its results handed to the answer in batches.

    WebSearchRun(provider, query)            the provider's stream() consumed by a background task
      results numbered W1, W2, … in arrival order (stable: what the answer cites never changes number)
      the top ``fetch_pages`` results' pages fetched in the background; their text is used once it is there
      first_batch()   with stream_partial_results: the first results, plus what other engines add within
                      partial_wait_ms of the first one (or the deadline); without it: everything, once the search ends
      next_batch()    after the answer: results (and page texts) that arrived since, once the search has ended
      close()/aclose() cancel the search and page fetches (the turn was stopped, or is done with it)

Every wait is bounded by ``timeout_s`` from the start: the turn never waits longer for the web than that.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ..domain.projects import Citation
from ..providers.web_search import SearchResult, WebSearch, WebSearchRefused, WebSearchTimeout, describe
from ..settings import WebSearchSection
from .sources import snippet

SearchStatus = Literal["running", "done", "timeout", "failed", "cancelled"]
DEADLINE_GRACE_S = 0.25  # the provider enforces timeout_s itself; this only guards against one that doesn't
WEB_CONTENT_CHARS = 1_200  # page text given to the model per result
WEB_SNIPPET_CHARS = 300  # a result's snippet in the prompt (each prompt token costs ~3 ms before the first word)
TASK_PREFIX = "web-search:"  # names of a run's tasks (what is still running is easy to find)


@dataclass(slots=True)
class WebSource:
    """A web result as an answer source: ``[W1]``…"""

    source_id: str
    result: SearchResult
    content: str | None = None  # the page's text, once fetched
    content_given: bool = False  # the model has seen the page text

    def citation(self) -> Citation:
        r = self.result
        return Citation(
            source_id=self.source_id,
            document_id="",
            filename=r.site,
            page_start=None,
            page_end=None,
            chunk_id="",
            snippet=snippet(r.snippet or r.title),
            kind="web",
            url=r.url,
            title=r.title,
            site=r.site,
            published=r.published,
        )

    def header(self) -> str:
        r = self.result
        parts = [f"[{self.source_id}] {r.title}", r.site, (r.published or "")[:10]]
        return " · ".join(p for p in parts if p)

    def prompt_text(self, *, with_content: bool) -> str:
        lines = [self.header()]
        if self.result.snippet:
            lines.append(snippet(self.result.snippet, WEB_SNIPPET_CHARS))
        if with_content and self.content:
            lines.append(f"Page text: {self.content[:WEB_CONTENT_CHARS]}")
        return "\n".join(lines)


def format_web_sources(sources: Sequence[WebSource], *, with_content: bool = True) -> str:
    return "\n\n".join(s.prompt_text(with_content=with_content) for s in sources)


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """The live-data tool's progress for the UI's "searching the web" badge (SSE ``tool``, WebSocket ``tool``):
    ``start`` (with the query that leaves the machine), ``results`` (web sources taken into the answer, numbered
    W1…), then one of ``done`` / ``timeout`` / ``failed``."""

    phase: Literal["start", "results", "done", "timeout", "failed"]
    query: str
    sources: tuple[Citation, ...] = ()
    count: int | None = None  # results the search returned (terminal phases)
    detail: str | None = None  # failed: why
    elapsed_ms: float | None = None
    tool: str = "web_search"
    name: ClassVar[str] = "tool"

    def payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.tool, "phase": self.phase, "query": self.query}
        if self.phase == "results":
            data["sources"] = [c.model_dump(mode="json") for c in self.sources]
        if self.count is not None:
            data["count"] = self.count
        if self.detail is not None:
            data["detail"] = self.detail
        if self.elapsed_ms is not None:
            data["elapsed_ms"] = self.elapsed_ms
        return data


@dataclass(eq=False)
class WebSearchRun:
    provider: WebSearch
    query: str
    config: WebSearchSection
    results: list[WebSource] = field(default_factory=list)
    status: SearchStatus = "running"
    error: str | None = None
    taken: int = 0  # results handed to the answer
    first_result_ms: float | None = None
    finished_ms: float | None = None
    reported: bool = False  # the terminal ToolEvent was produced
    started: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        self.deadline = loop.time() + self.config.timeout_s
        self._changed = asyncio.Event()
        self._fetches: set[asyncio.Task[None]] = set()
        self._task: asyncio.Task[None] = asyncio.ensure_future(self._run())
        self._task.set_name(TASK_PREFIX + "search")

    # -------------------------------------------------------------- the search

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 1)

    async def _run(self) -> None:
        try:
            async with asyncio.timeout_at(self.deadline + DEADLINE_GRACE_S):
                stream = self.provider.stream(self.query, self.config.max_results)
                async with contextlib.aclosing(stream):  # type: ignore[type-var]
                    async for result in stream:
                        self._add(result)
            self._finish("done")
        except (WebSearchTimeout, TimeoutError):
            self._finish("timeout")
        except asyncio.CancelledError:
            self._finish("cancelled")
            raise
        except WebSearchRefused as e:
            self._finish("failed", str(e))
        except Exception as e:  # a provider bug must not break the turn: it answers without live data
            self._finish("failed", describe(e))

    def _add(self, result: SearchResult) -> None:
        source = WebSource(f"W{len(self.results) + 1}", result)
        self.results.append(source)
        if self.first_result_ms is None:
            self.first_result_ms = self.ms()
        if len(self.results) <= self.config.fetch_pages:
            task = asyncio.ensure_future(self._fetch(source))
            task.set_name(TASK_PREFIX + "page")
            self._fetches.add(task)
            task.add_done_callback(self._fetches.discard)
        self._changed.set()

    async def _fetch(self, source: WebSource) -> None:
        try:
            source.content = await self.provider.fetch_page(source.result.url)
        except asyncio.CancelledError:
            raise
        except Exception:  # a page that can't be read is just not used
            return
        if source.content:
            self._changed.set()

    def _finish(self, status: SearchStatus, error: str | None = None) -> None:
        if self.status == "running":
            self.status, self.error, self.finished_ms = status, error, self.ms()
        self._changed.set()

    async def _wait(self, ready: Callable[[], bool], until: float | None = None) -> None:
        """Until ``ready()`` or ``until`` (loop time), never past the deadline."""
        end = self.deadline + DEADLINE_GRACE_S * 2 if until is None else min(until, self.deadline)
        while not ready():
            self._changed.clear()
            left = end - self._loop.time()
            if left <= 0:
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), left)

    # -------------------------------------------------------------- batches

    def _take(self) -> list[WebSource]:
        batch = self.results[self.taken :]
        self.taken = len(self.results)
        return batch

    async def first_batch(self) -> list[WebSource]:
        """The results the answer starts with (module docstring)."""
        if not self.config.stream_partial_results:
            await self._wait(lambda: self.finished)
            return self._take()
        await self._wait(lambda: bool(self.results) or self.finished)
        if self.results and not self.finished:
            settle = self._loop.time() + self.config.partial_wait_ms / 1000
            await self._wait(lambda: self.finished or len(self.results) >= self.config.max_results, settle)
        return self._take()

    async def next_batch(self, *, last: bool = True) -> list[WebSource]:
        """Results that arrived after the answer started. ``last`` (the answer's last chance to continue): once the
        search and the page fetches have ended; otherwise as soon as new results (or page texts) have settled for
        partial_wait_ms. Bounded by the deadline either way."""
        if last:
            await self._wait(lambda: self.finished and not self.pending_pages)
            return self._take()
        await self._wait(lambda: self.finished or len(self.results) > self.taken or bool(self.fresh_pages()))
        if not self.finished:
            settle = self._loop.time() + self.config.partial_wait_ms / 1000
            await self._wait(lambda: self.finished, settle)
        return self._take()

    @property
    def pending_pages(self) -> bool:
        return any(not t.done() for t in self._fetches)

    def fresh_pages(self) -> list[WebSource]:
        """Sources already given to the model whose page text has arrived since."""
        return [s for s in self.results[: self.taken] if s.content and not s.content_given]

    # -------------------------------------------------------------- events and clean-up

    def results_event(self, batch: Sequence[WebSource]) -> ToolEvent:
        return ToolEvent("results", self.query, tuple(s.citation() for s in batch), elapsed_ms=self.ms())

    def terminal_event(self) -> ToolEvent | None:
        """``done`` / ``timeout`` / ``failed`` once the search has ended (once)."""
        if not self.finished or self.reported or self.status == "cancelled":
            return None
        self.reported = True
        phase: Literal["done", "timeout", "failed"] = self.status  # type: ignore[assignment]
        return ToolEvent(phase, self.query, count=len(self.results), detail=self.error, elapsed_ms=self.finished_ms)

    def stop(self) -> ToolEvent | None:
        """End the search now (the answer needs nothing more): ``done`` with what arrived, unless it ended."""
        if not self.finished:
            self._finish("done")
        self.close()
        return self.terminal_event()

    def close(self) -> None:
        """Cancel the search and page fetches (synchronous: safe in any clean-up)."""
        if not self._task.done():
            self._task.cancel()
        for task in list(self._fetches):
            task.cancel()

    async def aclose(self) -> None:
        """``close`` and wait until the tasks have ended. If the caller is cancelled meanwhile, its cancellation
        propagates and the tasks (already cancelled) end on their own."""
        self.close()
        tasks = [t for t in (self._task, *self._fetches) if not t.done()]
        if tasks:
            await asyncio.wait(tasks)

    def record(self) -> dict[str, Any]:
        """For the message's ``route.web_search``."""
        return {
            "query": self.query,
            "provider": self.provider.name,
            "status": self.status,
            "error": self.error,
            "results": len(self.results),
            "used": self.taken,
            "pages": sum(1 for s in self.results if s.content_given),
        }

    def latency(self) -> dict[str, Any]:
        return {"web_first_result_ms": self.first_result_ms, "web_search_ms": self.finished_ms}
