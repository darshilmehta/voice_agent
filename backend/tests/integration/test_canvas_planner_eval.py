"""The turn's visual on the real model (qwen3:4b-instruct via Ollama) over labelled EN / HI / Hinglish questions about
the eval corpus (opt-in), three paths compared (docs/DESIGN.md §12.1, "Instant draft, then refine"):

- ``old``: the planner alone, as #38 shipped it (its candidate ranking frozen below: words only, no company), forced;
- ``draft``: the instant draft alone (code, no model): ``draft.draft_visual`` with the company the question names;
- ``draft+planner``: what a turn does now: the draft, and when it isn't confident the planner (forced, with the
  draft's candidates), whose choice replaces it; a planner that finds no table withdraws it, one that fails keeps it.

Three sets:

- ``tuning`` (``canvas_planner_cases.json``, 30 questions): the set the planner's prompt (and the draft) was tuned on;
- ``holdout`` (``canvas_planner_holdout.json``, 27 questions): written for #38 without seeing the planner; its questions
  and misses were visible while the draft was written (not tuned on, but no longer blind);
- ``holdout_v2`` (``canvas_planner_holdout_v2.json``, 36 questions, #39): written blind; run once, at the end. Four of
  its questions ask for something no table holds: right when no visual is drawn.

    CANVAS_PARSE_CACHE=<folder> RUN_INTEGRATION=1 uv run pytest tests/integration/test_canvas_planner_eval.py -s
    CANVAS_EVAL_SETS=holdout_v2 …  (one set; default: tuning and holdout)

Needs Ollama with the router model, the Docling models (or a parse cache) and the corpus. Every document's datasets
are candidates, as in a project holding all four documents. No retrieval: the question (and its English query) is all
the draft and the planner get (in a conversation the turn's retrieved tables and their documents rank first too).

Per path and question: whether the kind, the table(s) and the series match the labels (lenient: any expected series
among what is plotted; v2 also strict: all of them), that every visual builds and is grounded; for draft+planner how
often the planner was skipped (a confident draft) and what it did; the planner's latency (wall and Ollama's own time).
Accuracy is measured with a generous timeout (``CANVAS_PLANNER_EVAL_TIMEOUT_S``, default 30 s).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.domain.datasets import TypedDataset
from app.providers.base import ProviderContext
from app.providers.ingestion import document_label
from app.providers.llm import OllamaLLM
from app.services.canvas.builder import build_visual, check_grounding
from app.services.canvas.conversation import visual_want
from app.services.canvas.draft import draft_visual, same_choice
from app.services.canvas.planner import _WANTS, NO_VISUAL, VisualPlanner, _dataset_words, _words
from app.services.canvas.spec import VisualSpec, resolve, split_ref
from app.services.subjects import documents_by_name, named_documents
from app.settings import load_settings

from .canvas_corpus import CorpusDocument, corpus_fixture  # noqa: F401  (the "corpus" fixture)
from .conftest import LOCAL_CONFIG, METRICS

pytestmark = pytest.mark.integration

SETS = {
    "tuning": "canvas_planner_cases.json",
    "holdout": "canvas_planner_holdout.json",
    "holdout_v2": "canvas_planner_holdout_v2.json",
}
WHICH = [s for s in os.environ.get("CANVAS_EVAL_SETS", "tuning,holdout").split(",") if s]
PATHS = ("old", "draft", "draft+planner")
MIN_DATASET_ACCURACY = 0.7  # a floor against regressions on the tuning set; the measured numbers are printed
EVAL_TIMEOUT_S = float(os.environ.get("CANVAS_PLANNER_EVAL_TIMEOUT_S", "30"))


def legacy_rank(
    question: str, datasets: Sequence[TypedDataset], query_en: str | None, limit: int
) -> list[TypedDataset]:
    """``rank_candidates`` as #38 shipped it (words and shapes only; no company, no sources): the old path."""
    words = _words(question) | (_words(query_en) if query_en else set())
    text = f"{question} {query_en or ''}"
    scored = []
    for ds in datasets:
        if ds.chartability.kind == "none":
            continue
        overlap = len(words & _dataset_words(ds))
        title_hit = len(words & _words(ds.title))
        shape = sum(1.5 for pattern, fits in _WANTS if pattern.search(text) and fits(ds))
        score = overlap + 2 * title_hit + shape + ds.chartability.confidence
        scored.append((score, -(ds.page_start or 0), ds))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [ds for _, _, ds in scored[:limit]]


def _label(ds: TypedDataset, key: str) -> str:
    row, col = ds.row(key), ds.column(key)
    return (row.label if row is not None else col.label if col is not None else key).casefold()


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def table_matches(ds: TypedDataset, filename: str, accepted: list[tuple[str, str]]) -> bool:
    """The dataset is one of the accepted tables: the same document, and the label's title words in its title (v2's
    labels are the documents' headings, not a Docling parse: punctuation and spacing aside)."""
    title = _norm(ds.title)
    return any(filename == doc and _norm(t) in title for doc, t in accepted)


def score(spec: VisualSpec | None, case: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Kind, table and series right (and, for a "none" case, no visual), and the visual grounded."""
    by_id, filenames, names = ctx["by_id"], ctx["filenames"], ctx["names"]
    if not case["kinds"]:  # v2: nothing holds what is asked: right when nothing is drawn
        ok = spec is None
        return {"ok": {"kind": ok, "table": ok, "series": ok}, "strict": ok, "series_38": ok, "grounded": None}
    kind_ok = spec is not None and spec.kind in case["kinds"]
    accepted = [(names[d], t) for d, t in case["datasets"]]
    table_ok = spec is not None and all(
        table_matches(by_id[d], filenames[by_id[d].document_id], accepted) for d in spec.datasets
    )
    series_ok = bool(spec) or not case["series"]
    strict = series_38 = series_ok
    grounded = None
    if spec is not None:
        r = resolve(spec, by_id, filenames=filenames)
        v = build_visual(
            r, visual_id="vis_eval", project_id="p", chat_id="c", filenames=filenames, now=datetime.now(UTC)
        )
        grounded = check_grounding(v) == []
        if case["series"]:
            # what is plotted: the spec's rows or columns, as the visual names them, its tiles, and its x items (the
            # slices of a donut, the steps of a waterfall: v2's definition; #38 counted only the first three)
            labels = [_label(by_id[split_ref(s)[0]], split_ref(s)[1]) for s in spec.series]
            labels += [s.label.casefold() for s in r.series] + [t.label.casefold() for t in v.tiles]
            series_38 = any(w.casefold() in label for w in case["series"] for label in labels)
            labels += [i.label.casefold() for i in r.x_items]
            series_ok = any(w.casefold() in label for w in case["series"] for label in labels)
            expected = (case.get("expected") or {}).get("series") or []
            strict = all(any(w.casefold() in label for label in labels) for w in expected) if expected else series_ok
    return {
        "ok": {"kind": kind_ok, "table": table_ok, "series": series_ok},
        "strict": strict,
        "series_38": series_38,
        "grounded": grounded,
    }


@pytest.mark.parametrize("which", WHICH)
async def test_the_turns_visual_on_the_real_model(corpus: dict[str, CorpusDocument], tmp_path, which: str):
    cases = json.loads((Path(__file__).parent / SETS[which]).read_text(encoding="utf-8"))
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    datasets = [d for doc in corpus.values() for d in doc.datasets]
    by_id = {d.id: d for d in datasets}
    filenames = {doc.document_id: doc.filename for doc in corpus.values()}
    labels = {d: document_label(name, None) for d, name in filenames.items()}
    docs = cases["documents"]
    names = docs if isinstance(docs, dict) else dict(enumerate(docs))  # v1: {"AR": …}; v2: a list
    ctx = {"by_id": by_id, "filenames": filenames, "names": names}
    limit = settings.canvas.planner_candidates
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

    def model_time() -> float | None:
        t = timings[-1] if timings else None
        return (t["prompt_eval_duration"] + t["eval_duration"]) / 1e9 if t else None

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
            max_candidates=limit,
        )
        await planner.plan("Show me quarterly revenue", "en", datasets[:4], filenames=filenames)  # warm up
        results = []
        for case in cases["cases"]:
            question, query_en = case["question"], case.get("query_en")
            language = "en" if case["language"] == "en" else "hi"  # Hinglish is answered in Hindi
            row: dict[str, Any] = {"id": case["id"], "language": case["language"], "none": not case["kinds"]}
            row["want"] = visual_want(question, query_en)

            # old: the planner alone, #38's candidates
            timings.clear()
            started = time.perf_counter()
            old = await planner.plan_detailed(
                question,
                language,
                datasets,
                filenames=filenames,
                query_en=query_en,
                ranked=legacy_rank(question, datasets, query_en, limit),
                force=True,
            )
            row["old"] = {
                "spec": old.spec,
                "source": old.source,
                "wall": time.perf_counter() - started,
                "model": model_time(),
            } | score(old.spec, case, ctx)

            # draft: code alone, company-aware
            named = named_documents((question, query_en), labels)
            draft = draft_visual(
                question,
                language,
                datasets,
                query_en=query_en,
                documents=named.document_ids if named else None,
                names=named.names if named else (),
                companies=documents_by_name(named, labels) if named else None,
                filenames=filenames,
                limit=limit,
            )
            row["draft"] = {
                "spec": draft.spec,
                "confident": draft.confident,
                "ms": draft.latency_ms,
                "reasons": draft.reasons,
            } | score(draft.spec, case, ctx)

            # draft + planner: what a turn does
            final, outcome, wall, model = draft.spec, "skipped", None, None
            if draft.spec is None or not draft.confident:
                timings.clear()
                started = time.perf_counter()
                refined = await planner.plan_detailed(
                    question,
                    language,
                    datasets,
                    filenames=filenames,
                    query_en=query_en,
                    ranked=draft.candidates or None,
                    force=True,
                    fallback=draft.spec is None,
                )
                wall, model = time.perf_counter() - started, model_time()
                if refined.spec is not None:
                    outcome = "same" if draft.spec and same_choice(refined.spec, draft.spec, by_id) else "changed"
                    final = refined.spec
                elif refined.reason == NO_VISUAL:
                    outcome, final = "none", None
                else:
                    outcome = "failed"
                if draft.spec is None:
                    outcome = f"no draft, {outcome}"
            row["draft+planner"] = {"spec": final, "outcome": outcome, "wall": wall, "model": model} | score(
                final, case, ctx
            )
            results.append(row)

    out: dict[str, str] = {}
    visual = [r for r in results if not r["none"]]
    nones = [r for r in results if r["none"]]
    n = len(visual)

    def correct(r: dict, path: str) -> bool:
        return all(r[path]["ok"].values())

    out["questions"] = f"{len(results)} ({dict(Counter(r['language'] for r in results))}), {len(nones)} expect none"
    for path in PATHS:
        share = {k: sum(1 for r in visual if r[path]["ok"][k]) for k in ("kind", "table", "series")}
        allok = sum(1 for r in visual if correct(r, path))
        out[f"{path}: kind / table / series / all correct"] = (
            f"{share['kind']}/{n} / {share['table']}/{n} / {share['series']}/{n} / {allok}/{n} = {allok / n:.0%}"
        )
        allok_38 = sum(1 for r in visual if r[path]["ok"]["kind"] and r[path]["ok"]["table"] and r[path]["series_38"])
        out[f"{path}: series / all correct, #38's series rule (x items not counted)"] = (
            f"{sum(1 for r in visual if r[path]['series_38'])}/{n} / {allok_38}/{n}"
        )
        by_language = {
            lang: f"{sum(1 for r in visual if r['language'] == lang and correct(r, path))}/"
            f"{sum(1 for r in visual if r['language'] == lang)}"
            for lang in sorted({r["language"] for r in visual})
        }
        out[f"{path}: all correct by language"] = str(by_language)
        if which == "holdout_v2":
            strict = sum(1 for r in visual if r[path]["ok"]["kind"] and r[path]["ok"]["table"] and r[path]["strict"])
            out[f"{path}: all correct, series strict (all expected series)"] = f"{strict}/{n}"
        if nones:
            right = sum(1 for r in nones if correct(r, path))
            out[f"{path}: 'none' questions without a visual"] = f"{right}/{len(nones)}"
        grounded = [r[path]["grounded"] for r in results if r[path]["grounded"] is not None]
        out[f"{path}: visuals grounded"] = f"{sum(grounded)}/{len(grounded)}"
    confident = [r for r in visual if r["draft"]["confident"]]
    out["draft: confident (planner skipped) / right when confident"] = (
        f"{len(confident)}/{n} / {sum(1 for r in confident if correct(r, 'draft'))}/{len(confident)}"
    )
    out["draft+planner: planner outcomes"] = str(dict(Counter(r["draft+planner"]["outcome"] for r in results)))
    out["draft: build time p50 / max"] = (
        f"{statistics.median(r['draft']['ms'] for r in results):.1f} / {max(r['draft']['ms'] for r in results):.1f} ms"
    )
    for path in ("old", "draft+planner"):
        walls = [r[path]["wall"] for r in results if r[path]["wall"] is not None]
        models = [r[path]["model"] for r in results if r[path]["model"] is not None]
        if walls:
            out[f"{path}: planner calls, wall p50 / p95, model p50"] = (
                f"{len(walls)}, {statistics.median(walls):.2f} / {_pct(walls, 0.95):.2f} s, "
                f"{statistics.median(models):.2f} s"
                if models
                else f"{len(walls)}, {statistics.median(walls):.2f} s"
            )
    wanted = [r for r in visual if r["want"] != "none"]
    out["gate: a visual wanted by the words (route.visual)"] = (
        f"{len(wanted)}/{n} ({dict(Counter(r['want'] for r in visual))})"
        + (f"; 'none' questions: {dict(Counter(r['want'] for r in nones))}" if nones else "")
    )
    misses = [r["id"] for r in visual if r["want"] == "none"]
    if misses:
        out["gate misses"] = ", ".join(misses)
    for path in PATHS:
        wrong = [
            f"{r['id']}: {r[path]['spec'].kind if r[path]['spec'] else None} on "
            f"{[by_id[d].title[:40] for d in r[path]['spec'].datasets] if r[path]['spec'] else []} "
            f"({', '.join(k for k, v in r[path]['ok'].items() if not v)} wrong"
            + (f"; {r[path].get('outcome')}" if path == "draft+planner" else "")
            + (f"; {'confident' if r['draft']['confident'] else 'unsure'}" if path == "draft" else "")
            + ")"
            for r in results
            if not correct(r, path)
        ]
        if wrong:
            out[f"{path} misses"] = "\n  " + "\n  ".join(wrong)
    METRICS.update({f"[{which}] {k}": v for k, v in out.items()})
    if dump := os.environ.get("CANVAS_EVAL_DUMP"):  # every question's choices, to look at or score again
        rows = [
            {
                "id": r["id"],
                "language": r["language"],
                "want": r["want"],
                **{
                    path: {
                        k: (v.model_dump(mode="json") if isinstance(v, VisualSpec) else v) for k, v in r[path].items()
                    }
                    for path in PATHS
                },
            }
            for r in results
        ]
        Path(dump).mkdir(parents=True, exist_ok=True)
        (Path(dump) / f"{which}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for path in PATHS:
        assert all(r[path]["grounded"] in (True, None) for r in results), path
    if which == "tuning":
        assert sum(1 for r in visual if r["draft+planner"]["ok"]["table"]) / n >= MIN_DATASET_ACCURACY
