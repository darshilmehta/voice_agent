"""When should a turn's visual planner call go to Ollama? Measured on the real model (docs/DESIGN.md §12.1, opt-in).

    CANVAS_PARSE_CACHE=<folder> RUN_INTEGRATION=1 uv run pytest tests/integration/test_canvas_scheduling.py -s

Needs Ollama (qwen3:4b-instruct) and the eval corpus's datasets (Docling, or a parse cache). Nothing else: the
answer's prompt is the real voice answer prompt over a real table of the corpus, the planner the real planner over
every dataset of the corpus.

Ollama serves one request at a time here (the planner and the answer share the model), so the question is only what
waits for what. Per question, four schedules, in rotating order:

    alone           the answer by itself (its first token: the baseline)
    planner_first   the planner's request sent just before the answer's (what starting the visual with the turn would
                    do): the answer's first token waits
    on_first_delta  the planner's request sent when the answer's first piece arrives (queued behind the answer)
    after_answer    the planner's request sent when the answer has finished

Reported: the answer's first token per schedule, when the visual's plan is ready (from the answer's request), the
planner's latency on an otherwise idle model (``after_answer``: nothing else is running then), its prompt size, and
the cost of ranking the candidates in code (what "speculative preparation" could save).
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

import httpx
import pytest

from app.providers.base import ProviderContext
from app.providers.ingestion import Chunk
from app.providers.llm import LLMMessage, OllamaLLM
from app.services.canvas.planner import VisualPlanner, build_catalog, rank_candidates
from app.services.prompts import answer_system_prompt, answer_user_prompt
from app.services.sources import Source
from app.settings import load_settings

from .canvas_corpus import CorpusDocument, corpus_fixture  # noqa: F401  (the "corpus" fixture)
from .conftest import LOCAL_CONFIG, METRICS
from .test_answer_latency import OllamaStats

pytestmark = pytest.mark.integration

AR = "valmora_annual_report_fy24.pdf"
# (question, title of the table the answer reads)
QUESTIONS = [
    ("How did revenue move across the quarters of FY24?", "Quarterly performance: FY24"),
    ("Show me how each segment's revenue changed from FY23 to FY24.", "Segment results"),
    ("Compare EBITDA margin by segment.", "Segment results"),
    ("How did profit after tax grow through FY24, quarter by quarter?", "Quarterly performance: FY24"),
    ("Chart the key financial highlights for FY24.", "Financial highlights"),
    ("How has the workforce changed between FY23 and FY24?", "People and culture"),
]
SCHEDULES = ("alone", "planner_first", "on_first_delta", "after_answer")


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def test_when_the_planner_runs(corpus: dict[str, CorpusDocument], tmp_path):
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    datasets = [d for doc in corpus.values() for d in doc.datasets]
    filenames = {doc.document_id: doc.filename for doc in corpus.values()}
    report = corpus[AR]
    stats = OllamaStats()
    async with httpx.AsyncClient(transport=stats) as http:
        try:
            tags = (await http.get(f"{settings.llm.base_url}/api/tags", timeout=2)).json()
        except httpx.HTTPError as e:
            pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
        if settings.llm.chat_model not in {m["name"] for m in tags.get("models", [])}:
            pytest.skip(f"{settings.llm.chat_model} not pulled")
        llm = OllamaLLM(settings.llm, ProviderContext(settings=settings, http=http))
        planner = VisualPlanner(
            llm,
            model=settings.llm.router_model,
            timeout_s=60,
            max_tokens=settings.canvas.planner_max_tokens,
            max_candidates=settings.canvas.planner_candidates,
        )
        system = answer_system_prompt("en", "short")

        def answer_prompt(question: str, title: str, tag: str) -> list[LLMMessage]:
            index = next(d.table_index for d in report.datasets if d.title == title)
            table = next(t for t in report.parsed.tables if t.index == index)
            chunk = Chunk(
                chunk_id=f"c{tag}",
                project_id="p",
                document_id=report.document_id,
                version=1,
                chunk_index=0,
                chunking_version="v1",
                page_start=table.page_start,
                page_end=table.page_end,
                heading_path=[title, tag],  # a tag: no earlier run's evidence is in Ollama's cache
                content_type="table",
                language="en",
                text=table.markdown,
                token_count=len(table.markdown) // 4,
            )
            sources = [Source("S1", chunk, AR, 0.9)]
            return [LLMMessage("system", system), LLMMessage("user", answer_user_prompt(question, sources, "en"))]

        async def plan(question: str) -> float:
            t0 = time.perf_counter()
            await planner.plan(question, "en", datasets, filenames=filenames, force=True)
            return time.perf_counter() - t0

        # warm: both system prompts read once, as after the app's startup
        await llm.generate(answer_prompt(*QUESTIONS[0], "warm"), max_tokens=1)
        await plan("Show me quarterly revenue")

        rows: list[dict[str, Any]] = []
        for n, (question, title) in enumerate(QUESTIONS):
            order = SCHEDULES[n % 4 :] + SCHEDULES[: n % 4]
            for schedule in order:
                tag = f"{time.time_ns() % 10**9}-{schedule}"
                prompt = answer_prompt(question, title, tag)
                planning: asyncio.Task[float] | None = None
                t0 = time.perf_counter()
                if schedule == "planner_first":
                    planning = asyncio.ensure_future(plan(f"{question} ({tag})"))
                    await asyncio.sleep(0.03)  # the planner's request reaches Ollama first
                first = None
                async for _piece in llm.stream(prompt, max_tokens=180):
                    if first is None:
                        first = time.perf_counter() - t0
                        if schedule == "on_first_delta":
                            planning = asyncio.ensure_future(plan(f"{question} ({tag})"))
                end = time.perf_counter() - t0
                if schedule == "after_answer":
                    planning = asyncio.ensure_future(plan(f"{question} ({tag})"))
                planned = None
                if planning is not None:
                    alone = await planning
                    planned = time.perf_counter() - t0
                    if schedule == "after_answer":
                        rows.append({"schedule": "planner_idle", "planner": alone})
                rows.append({"schedule": schedule, "first": first, "end": end, "planned": planned})

    def by(schedule: str, key: str) -> list[float]:
        return [r[key] for r in rows if r["schedule"] == schedule and r.get(key) is not None]

    for schedule in SCHEDULES:
        firsts = by(schedule, "first")
        line = f"first token p50 {statistics.median(firsts):.2f} s (max {max(firsts):.2f})"
        line += f", answer done p50 {statistics.median(by(schedule, 'end')):.2f} s"
        if planned := by(schedule, "planned"):
            line += f", plan ready p50 {statistics.median(planned):.2f} s (max {max(planned):.2f}) after the request"
        METRICS[f"schedule {schedule}"] = line
    idle = by("planner_idle", "planner")
    METRICS["planner alone on an idle model p50 / p95 / max"] = (
        f"{statistics.median(idle):.2f} / {_pct(idle, 0.95):.2f} / {max(idle):.2f} s"
    )
    json_calls = [s for s in stats.done if s["kind"] == "json"]
    if json_calls:
        METRICS["planner prompt tokens p50 / output tokens p50"] = (
            f"{statistics.median(s.get('prompt_eval_count', 0) for s in json_calls):.0f} / "
            f"{statistics.median(s.get('eval_count', 0) for s in json_calls):.0f}"
        )
    t0 = time.perf_counter()
    for question, _ in QUESTIONS * 10:
        build_catalog(rank_candidates(question, datasets, limit=4), filenames, "en")
    METRICS["candidate ranking + catalog in code"] = f"{(time.perf_counter() - t0) / 60 * 1000:.1f} ms per question"
