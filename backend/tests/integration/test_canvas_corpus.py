"""Typed datasets, chartability, the overview and the grounding invariant on the eval corpus parsed by the real
Docling (opt-in).

    CANVAS_PARSE_CACHE=<folder> RUN_INTEGRATION=1 uv run pytest tests/integration/test_canvas_corpus.py -s

Needs the Docling models (MODELS_ROOT) and the generated corpus (EVAL_DOCS); no Qdrant, embedder or LLM. Reports:

- **fact typing accuracy**: the manifest's planted facts that live in tables (their row label is printed in a table
  cell): is the row found, and is every number of the fact a typed value of that row in the right unit kind (a
  number kept inside a text cell, such as "7.85% p.a., repayable on …", counts as kept as text)?
- **table accuracy**: every table of the corpus against a hand-labelled expectation (chartability kind, period order,
  units of named rows).
- the overview the annual report gets, and the grounding check over every spec the corpus datasets allow.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.domain.canvas import VISUAL_KINDS, Visual
from app.domain.datasets import TypedDataset
from app.services.canvas.builder import build_visual, check_grounding
from app.services.canvas.chartability import series_columns
from app.services.canvas.overview import overview_specs
from app.services.canvas.parsing import Quantity, parse_quantity
from app.services.canvas.spec import SpecError, VisualSpec, ref, resolve

from .canvas_corpus import MANIFEST, CorpusDocument, corpus_fixture  # noqa: F401  (the "corpus" fixture)
from .conftest import METRICS

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 9, tzinfo=UTC)

# table index → (acceptable chartability kinds, chronological periods or None, {row key prefix: unit label})
GOLD: dict[str, dict[int, tuple[set[str], list[str] | None, dict[str, str]]]] = {
    "valmora_annual_report_fy24.pdf": {
        0: ({"none"}, None, {}),
        1: (
            {"kpi"},
            ["FY23", "FY24"],
            {
                "revenue_from_operations": "₹ crore",
                "ebitda_margin": "%",
                "basic_earnings_per_share": "₹",
                "net_debt_to_ebitda": "x",
                "number_of_employees": "",
                "dividend_per_share": "₹",
            },
        ),
        2: ({"composition"}, None, {"specialty_chemicals": "%"}),
        3: ({"categorical"}, None, {"dahej_complex": "tpa", "bengaluru_centre": "seats"}),
        4: ({"kpi"}, ["FY23", "FY24"], {"women_as_a_share": "%", "average_age": "years"}),
        5: ({"none"}, None, {}),
        6: ({"categorical"}, None, {}),
        7: ({"composition"}, ["FY23", "FY24"], {"specialty_chemicals": ""}),  # mixed: ₹ crore and % columns
        8: ({"time_series"}, ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"], {"q1_fy24": ""}),
        9: ({"time_series"}, ["Q1 FY23", "Q2 FY23", "Q3 FY23", "Q4 FY23"], {}),
        10: (
            {"composition", "categorical"},
            ["FY23", "FY24"],
            {"revenue_from_operations": "₹ crore", "other_income": "₹ crore"},
        ),
        11: ({"composition", "categorical"}, ["FY23", "FY24"], {"finance_costs": "₹ crore", "basic_and_diluted": "₹"}),
        12: ({"composition", "categorical"}, ["31 Mar 2023", "31 Mar 2024"], {"inventories": "₹ crore"}),
        13: ({"composition", "categorical"}, ["31 Mar 2023", "31 Mar 2024"], {"trade_payables": "₹ crore"}),
        14: ({"composition", "categorical"}, ["FY23", "FY24"], {"net_cash_used_in_investing": "₹ crore"}),
        15: ({"composition"}, ["31 Mar 2023", "31 Mar 2024"], {"secured_term_loans": "₹ crore"}),
        16: ({"categorical"}, ["31 Mar 2023", "31 Mar 2024"], {"disputed_income_tax": "₹ crore"}),
        17: ({"time_series"}, ["FY23", "FY24"], {}),
        18: ({"composition"}, None, {"promoters": "%"}),
        19: ({"none"}, None, {}),
    },
    "zephyra_investor_deck_q4fy24.pptx": {
        0: (
            {"kpi"},
            ["FY23", "FY24"],
            {"revenue_from_operations": "₹ crore", "ebitda_margin": "%", "net_debt_to_ebitda": "x"},
        ),
        1: ({"composition"}, ["FY23", "FY24"], {}),
        2: ({"kpi", "time_series"}, ["Q4 FY23", "Q3 FY24", "Q4 FY24"], {"revenue_from_operations": "₹ crore"}),
        3: ({"categorical", "composition", "time_series"}, ["FY23", "FY24"], {"revenue_from_operations": "₹ crore"}),
        4: ({"kpi"}, ["FY23", "FY24"], {"on_time_delivery_rate": "%"}),
        5: ({"none"}, None, {}),
    },
    "valmora_travel_expense_policy.docx": {
        0: ({"none"}, None, {}),
        1: ({"none"}, None, {}),
        2: ({"categorical"}, None, {"l1_to_l2": "₹"}),
        3: ({"categorical"}, None, {}),
        4: ({"timeline"}, None, {}),
    },
    "suryodaya_yojana_soochna.docx": {
        0: ({"timeline"}, None, {}),
        1: ({"composition"}, None, {}),
        2: ({"composition"}, None, {}),
    },
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%.,()₹ ]+", " ", text.casefold())).strip()


def _unit_ok(value_unit, q: Quantity) -> bool:
    if q.kind == "percent":
        return value_unit is not None and value_unit.kind == "percent"
    if q.kind == "ratio":
        return value_unit is not None and value_unit.kind == "ratio"
    if q.kind == "currency":
        return value_unit is not None and value_unit.kind == "currency"
    return True


def _fact_result(fact: dict, doc: CorpusDocument) -> tuple[bool, bool] | None:
    """(found: the row and every number of the fact, each typed or kept inside a text cell; typed: every number a
    typed value of the row in the right unit kind); None for facts that aren't table facts (the label and one of the
    numbers must be printed in the same table)."""
    label, *rest = fact["evidence"]
    numbers = [(t, q) for t in rest if isinstance(q := parse_quantity(t), Quantity)]
    if not numbers:
        return None
    tables = [
        t
        for t in doc.parsed.tables
        if any(_norm(label) in _norm(c.text) for c in t.cells)
        and any(_norm(n) in _norm(c.text) for n, _ in numbers for c in t.cells)
    ]
    if not tables:
        return None
    best = (False, False)
    for ds in doc.datasets:
        if ds.table_index not in {t.index for t in tables}:
            continue
        for row in ds.rows:
            texts = [t.text for t in ds.texts if t.row == row.key]
            in_label = _norm(label) in _norm(row.label) or (len(row.label) > 3 and _norm(row.label) in _norm(label))
            in_text = any(_norm(label) in _norm(x) for x in texts)  # a fact stated inside a text cell
            if not (in_label or in_text):
                continue
            values = [v for v in ds.values if v.row == row.key]
            typed = [any(v.value == q.value and _unit_ok(v.unit, q) for v in values) for _, q in numbers]
            kept = [ok or any(t in x for x in texts) for ok, (t, _) in zip(typed, numbers, strict=True)]
            result = (all(kept), all(typed))
            if result > best:
                best = result
    return best


def test_fact_typing_accuracy(corpus):
    facts = json.loads(MANIFEST.read_text(encoding="utf-8"))["facts"]
    totals: Counter[str] = Counter()
    misses = []
    for fact in facts:
        doc = corpus.get(fact["document"])
        if doc is None:
            continue
        result = _fact_result(fact, doc)
        if result is None:
            continue
        found, typed = result
        totals["table facts"] += 1
        totals["recovered"] += found
        totals["typed as numbers"] += typed
        if not found:
            misses.append(f"{fact['id']}: {fact['evidence']}")
    n = totals["table facts"]
    METRICS["canvas fact typing accuracy"] = (
        f"{totals['recovered']}/{n} = {totals['recovered'] / n:.1%} recovered "
        f"({totals['typed as numbers']} with every number typed)"
    )
    if misses:
        METRICS["canvas fact typing misses"] = "\n  " + "\n  ".join(misses)
    assert n >= 60
    assert totals["recovered"] / n >= 0.95


def test_table_accuracy_against_the_hand_labels(corpus):
    ok = 0
    total = 0
    failures = []
    for name, gold in GOLD.items():
        doc = corpus[name]
        assert len(doc.datasets) == len(gold), f"{name}: {len(doc.datasets)} tables typed, {len(gold)} labelled"
        for ds in doc.datasets:
            kinds, periods, units = gold[ds.table_index]
            problems = []
            if ds.chartability.kind not in kinds:
                problems.append(f"kind {ds.chartability.kind} not in {sorted(kinds)}")
            if periods is not None and ds.chartability.period_order != periods:
                problems.append(f"periods {ds.chartability.period_order} != {periods}")
            for prefix, label in units.items():
                row = next((r for r in ds.rows if r.key.startswith(prefix)), None)
                if row is None:
                    problems.append(f"no row {prefix}*")
                    continue
                if label and (row.unit is None or row.unit.label != label):
                    problems.append(f"{row.key}: unit {row.unit.label if row.unit else None!r} != {label!r}")
            total += 1
            if problems:
                failures.append(f"{name} T{ds.table_index} '{ds.title}': {'; '.join(problems)}")
            else:
                ok += 1
    METRICS["canvas table accuracy (kind, periods, units)"] = f"{ok}/{total} = {ok / total:.0%}"
    kinds = Counter(d.chartability.kind for doc in corpus.values() for d in doc.datasets)
    METRICS["canvas chartability"] = ", ".join(f"{k} {n}" for k, n in kinds.most_common())
    if failures:
        METRICS["canvas table misses"] = "\n  " + "\n  ".join(failures)
    assert ok / total >= 0.9


def _build(
    spec: VisualSpec, by_id: dict[str, TypedDataset], filenames: dict[str, str], project: str = "prj_eval"
) -> Visual:
    r = resolve(spec, by_id, filenames=filenames)
    return build_visual(r, visual_id="vis_eval", project_id=project, chat_id=None, filenames=filenames, now=NOW)


def test_overview_of_the_annual_report(corpus):
    doc = corpus["valmora_annual_report_fy24.pdf"]
    by_id = {d.id: d for d in doc.datasets}
    filenames = {doc.document_id: doc.filename}
    built: list[Visual] = []

    def check(spec: VisualSpec) -> bool:
        try:
            built.append(_build(spec, by_id, filenames))
        except SpecError:
            return False
        return True

    started = time.perf_counter()
    overview_specs(doc.datasets, max_panels=3, check=check)
    elapsed = time.perf_counter() - started
    assert [v.kind for v in built] == ["kpi", "line", "donut"]
    kpi, line, donut = built
    assert [t.label for t in kpi.tiles][:2] == ["Revenue from operations (FY24)", "EBITDA (FY24)"]
    assert [r.x for r in line.rows] == [f"Q{q} FY{y}" for y in (23, 24) for q in (1, 2, 3, 4)]  # both tables, merged
    assert [r.x for r in donut.rows] == ["Specialty Chemicals", "Engineered Plastics", "Digital Services"]
    for v in built:
        assert check_grounding(v) == []
    out = Path(os.environ.get("CANVAS_PARSE_CACHE") or Path.cwd()) / "canvas_overview_sample.json"
    if os.environ.get("CANVAS_PARSE_CACHE"):
        out.write_text(
            json.dumps([v.model_dump(mode="json") for v in built], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        METRICS["canvas overview sample"] = str(out)
    METRICS["canvas overview (annual report)"] = (
        f"{', '.join(f'{v.kind}: {v.title}' for v in built)} in {elapsed * 1000:.0f} ms"
    )
    METRICS["canvas overview summaries"] = "\n  " + "\n  ".join(v.summary for v in built)


def test_every_buildable_spec_on_the_corpus_is_grounded(corpus):
    datasets = [d for doc in corpus.values() for d in doc.datasets]
    by_id = {d.id: d for d in datasets}
    filenames = {doc.document_id: doc.filename for doc in corpus.values()}
    built: Counter[str] = Counter()
    tried = 0
    started = time.perf_counter()
    for ds in datasets:
        rows = [r.key for r in ds.rows if r.type != "section"]
        cols = [c.key for c in series_columns(ds)]
        periods = list(dict.fromkeys(ds.chartability.period_order))
        options = [[k] for k in rows + cols] + [rows[:2], cols[:2], rows[:4], cols[:3]]
        for kind, series in itertools.product(VISUAL_KINDS, options):
            for extra in (
                {},
                {"periods": periods[-2:]},
                {"calculations": [{"op": "growth", "series": ref(ds.id, series[0])}]} if series else {},
            ):
                for language in ("en", "hi"):
                    tried += 1
                    try:
                        spec = VisualSpec(
                            kind=kind,
                            datasets=[ds.id],
                            series=[ref(ds.id, k) for k in series],
                            language=language,
                            **extra,
                        )  # type: ignore[arg-type]
                        v = _build(spec, by_id, filenames)
                    except (SpecError, ValueError):
                        continue
                    assert check_grounding(v) == []
                    built[v.kind] += 1
    METRICS["canvas grounding on the corpus"] = (
        f"{sum(built.values())} visuals built from {tried} specs, all grounded "
        f"({', '.join(f'{k} {n}' for k, n in built.most_common())}) in {time.perf_counter() - started:.1f}s"
    )
    assert sum(built.values()) > 1000


def test_typing_cost(corpus):
    doc = corpus["valmora_annual_report_fy24.pdf"]
    from app.services.canvas.datasets import table_contexts, type_document

    from .canvas_corpus import stored_tables

    tables = stored_tables(doc.parsed, doc.document_id)
    contexts = table_contexts(doc.parsed)
    started = time.perf_counter()
    for _ in range(5):
        type_document(tables, contexts, new_id=lambda: "ds_x")
    per_doc = (time.perf_counter() - started) / 5
    METRICS["canvas typing time (29-page report, 20 tables)"] = f"{per_doc * 1000:.0f} ms"
    seconds = [d.parse_seconds for d in corpus.values() if d.parse_seconds is not None]
    if seconds:
        METRICS["canvas corpus parse (Docling, CPU)"] = f"{sum(seconds):.1f}s (not cached)"
