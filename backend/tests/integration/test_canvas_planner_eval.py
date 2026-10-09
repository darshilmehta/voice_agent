"""The visual planner on the real model (qwen3:4b-instruct via Ollama) over 30 labelled EN / HI / Hinglish questions
about the eval corpus (opt-in).

    CANVAS_PARSE_CACHE=<folder> RUN_INTEGRATION=1 uv run pytest tests/integration/test_canvas_planner_eval.py -s

Needs Ollama with the router model, the Docling models (or a parse cache) and the corpus. Every document's datasets
are candidates, as in a project holding all four documents; the planner ranks them and offers the best few to the
model. Reports per question whether the kind, the table(s) and the series match the labels, the planner's source
(model / heuristic fallback), and that every planned visual builds and is grounded.

Latency is reported twice: the wall time of a call, and Ollama's own time for it (prompt evaluation + generation, from
the response), which doesn't include waiting behind other clients of the same server. Accuracy is measured with a
generous timeout (``CANVAS_PLANNER_EVAL_TIMEOUT_S``, default 30 s) so a busy server doesn't turn into misses; how many
calls fit the configured ``canvas.planner_timeout_ms`` is reported from the model time.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.domain.datasets import TypedDataset
from app.providers.base import ProviderContext
from app.providers.llm import OllamaLLM
from app.services.canvas.builder import build_visual, check_grounding
from app.services.canvas.planner import VisualPlanner
from app.services.canvas.spec import resolve, split_ref
from app.settings import load_settings

from .canvas_corpus import CorpusDocument, corpus_fixture  # noqa: F401  (the "corpus" fixture)
from .conftest import LOCAL_CONFIG, METRICS

pytestmark = pytest.mark.integration

CASES = json.loads((Path(__file__).parent / "canvas_planner_cases.json").read_text(encoding="utf-8"))
MIN_DATASET_ACCURACY = 0.7  # a floor against regressions; the measured numbers are printed
EVAL_TIMEOUT_S = float(os.environ.get("CANVAS_PLANNER_EVAL_TIMEOUT_S", "30"))


def _label(ds: TypedDataset, key: str) -> str:
    row, col = ds.row(key), ds.column(key)
    return (row.label if row is not None else col.label if col is not None else key).casefold()


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


async def test_planner_on_the_real_model(corpus: dict[str, CorpusDocument], tmp_path):
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    datasets = [d for doc in corpus.values() for d in doc.datasets]
    by_id = {d.id: d for d in datasets}
    filenames = {doc.document_id: doc.filename for doc in corpus.values()}
    names = CASES["documents"]
    timings: list[dict] = []

    async def record(response: httpx.Response) -> None:
        if response.request.url.path == "/api/chat" and response.status_code == 200:
            await response.aread()
            body = response.json()
            timings.append(
                {
                    k: body.get(k, 0)
                    for k in ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")
                }
            )

    async with httpx.AsyncClient(event_hooks={"response": [record]}) as http:
        try:
            tags = (await http.get(f"{settings.llm.base_url}/api/tags", timeout=2)).json()
        except httpx.HTTPError as e:
            pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
        if settings.llm.router_model not in {m["name"] for m in tags.get("models", [])}:
            pytest.skip(f"{settings.llm.router_model} not pulled")
        llm = OllamaLLM(settings.llm, ProviderContext(settings=settings, http=http))
        planner = VisualPlanner(
            llm,
            model=settings.llm.router_model,
            timeout_s=EVAL_TIMEOUT_S,
            max_tokens=settings.canvas.planner_max_tokens,
            max_candidates=settings.canvas.planner_candidates,
        )
        await planner.plan("Show me quarterly revenue", "en", datasets[:4], filenames=filenames)  # warm up
        results = []
        for case in CASES["cases"]:
            timings.clear()
            started = time.perf_counter()
            plan = await planner.plan_detailed(
                case["question"], case["language"], datasets, filenames=filenames, query_en=case.get("query_en")
            )
            wall = time.perf_counter() - started
            spec = plan.spec
            kind_ok = spec is not None and spec.kind in case["kinds"]
            accepted = [(names[d], title.casefold()) for d, title in case["datasets"]]
            dataset_ok = spec is not None and all(
                any(filenames[by_id[d].document_id] == doc and t in by_id[d].title.casefold() for doc, t in accepted)
                for d in spec.datasets
            )
            series_ok = bool(spec) or not case["series"]
            grounded = None
            if spec is not None:
                r = resolve(spec, by_id, filenames=filenames)
                v = build_visual(
                    r, visual_id="vis_eval", project_id="p", chat_id="c", filenames=filenames, now=datetime.now(UTC)
                )
                grounded = check_grounding(v) == []
                if case["series"]:
                    # what is plotted: the spec's rows or columns, as the visual names them (a segment row of a
                    # "Revenue FY24 | Revenue FY23" table is "Freight Services · Revenue"), and its tiles and rows
                    labels = [_label(by_id[split_ref(s)[0]], split_ref(s)[1]) for s in spec.series]
                    labels += [s.label.casefold() for s in r.series] + [t.label.casefold() for t in v.tiles]
                    series_ok = any(w.casefold() in label for w in case["series"] for label in labels)
            t = timings[-1] if timings else {}
            results.append(
                {
                    "id": case["id"],
                    "language": case["language"],
                    "kind": spec.kind if spec else None,
                    "titles": [by_id[d].title for d in spec.datasets] if spec else [],
                    "source": plan.source,
                    "ok": {"kind": kind_ok, "table": dataset_ok, "series": series_ok},
                    "grounded": grounded,
                    "wall": wall,
                    "model": (t.get("prompt_eval_duration", 0) + t.get("eval_duration", 0)) / 1e9 if t else None,
                    "prompt_tokens": t.get("prompt_eval_count"),
                    "output_tokens": t.get("eval_count"),
                    "reason": plan.reason,
                }
            )

    n = len(results)

    def share(key: str) -> str:
        ok = sum(1 for r in results if r["ok"][key])
        return f"{ok}/{n} = {ok / n:.0%}"

    def correct(r: dict) -> bool:
        return all(r["ok"].values())

    METRICS["planner questions"] = f"{n} ({dict(Counter(r['language'] for r in results))})"
    METRICS["planner kind / table / series accuracy"] = f"{share('kind')} / {share('table')} / {share('series')}"
    METRICS["planner all correct"] = f"{sum(map(correct, results))}/{n} = {sum(map(correct, results)) / n:.0%}"
    for language in ("en", "hi"):
        rows = [r for r in results if r["language"] == language]
        METRICS[f"planner all correct ({language})"] = f"{sum(map(correct, rows))}/{len(rows)}"
    METRICS["planner sources"] = str(dict(Counter(r["source"] for r in results)))
    walls = [r["wall"] for r in results]
    METRICS["planner wall p50 / p95 / max"] = (
        f"{statistics.median(walls):.2f} / {_pct(walls, 0.95):.2f} / {max(walls):.2f} s (includes queueing in Ollama)"
    )
    model = [r["model"] for r in results if r["model"] is not None]
    if model:
        budget = settings.canvas.planner_timeout_ms / 1000
        METRICS["planner model time p50 / p95 / max"] = (
            f"{statistics.median(model):.2f} / {_pct(model, 0.95):.2f} / {max(model):.2f} s "
            f"({sum(1 for m in model if m <= budget)}/{len(model)} within canvas.planner_timeout_ms)"
        )
        tokens = [r["output_tokens"] for r in results if r["output_tokens"]]
        prompts = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
        METRICS["planner tokens: prompt p50 / output p50, max"] = (
            f"{statistics.median(prompts):.0f} / {statistics.median(tokens):.0f}, {max(tokens)}"
        )
    misses = [
        f"{r['id']}: {r['kind']} on {r['titles']} ({', '.join(k for k, v in r['ok'].items() if not v)} wrong; "
        f"{r['source']}{'; ' + r['reason'] if r['reason'] else ''})"
        for r in results
        if not correct(r)
    ]
    if misses:
        METRICS["planner misses"] = "\n  " + "\n  ".join(misses)
    assert all(r["grounded"] in (True, None) for r in results)
    assert sum(1 for r in results if r["ok"]["table"]) / n >= MIN_DATASET_ACCURACY
