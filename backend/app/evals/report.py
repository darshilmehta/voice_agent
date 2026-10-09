# ruff: noqa: E501  (report prose and f-string tables)
"""Markdown summary of a results dict (``results.json`` of the retrieval harness). Pure text formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import SCORE_BUCKETS
from .questions import CATEGORY_DESCRIPTIONS

SWEEP_VIEW = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
LANG_SWEEP_VIEW = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
MAX_MISSES = 30
MAX_GATE_ERRORS = 15


# ------------------------------------------------------------------ formatting helpers


def pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def ms(x: float | None) -> str:
    return "-" if x is None else f"{x:,.0f}"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]], *, align_right_from: int = 1) -> str:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(("---" if i < align_right_from else "---:") for i in range(len(header))) + "|")
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _g(d: Mapping[str, Any] | None, *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _rank_cells(block: Mapping[str, Any] | None) -> list[str]:
    return [
        pct(_g(block, "recall@1")),
        pct(_g(block, "recall@3")),
        pct(_g(block, "recall@5")),
        pct(_g(block, "recall@all")),
        pct(_g(block, "success@5")),
        num(_g(block, "mrr")),
    ]


# ------------------------------------------------------------------ sections


def _headline(summary: Mapping[str, Any]) -> str:
    out = []
    n_a, n_u = summary["n_answerable"], summary["n_unanswerable"]
    out.append(f"{summary['n']} questions: {n_a} answerable, {n_u} unanswerable (abstention expected).\n")
    if "page_level" in summary:
        header = ["stage", "Recall@1", "Recall@3", "Recall@5", "Recall@all", "Success@5", "MRR"]
        rows = [
            ["candidates (before rerank)", *_rank_cells(_g(summary, "page_level", "candidate"))],
            ["after rerank", *_rank_cells(_g(summary, "page_level", "rerank"))],
            ["candidates, fact-level", *_rank_cells(_g(summary, "fact_level", "candidate"))],
            ["after rerank, fact-level", *_rank_cells(_g(summary, "fact_level", "rerank"))],
        ]
        out.append(table(header, rows))
        out.append(
            "\nPage-level: a passage hits when it is in the right document and its page range includes an expected "
            "page (evidence text for page-less DOCX). Fact-level: the passage text itself contains the fact. "
            "Recall@k is the share of expected evidence in the top k (a two-hop question with one page in the top 3 "
            "scores 0.5); Success@5 needs all of it; Recall@all uses every passage in the list (the candidate "
            "ceiling is what the reranker can reach).\n"
        )
        c = summary.get("citation_top1", {})
        out.append(
            f"- **Page-citation accuracy** (citing the best passage): {pct(c.get('lenient'))} lenient (an expected page is "
            f"within the cited range), {pct(c.get('exact'))} exact (the cited range stays within the expected pages); "
            f"mean cited range {num(c.get('mean_page_span'), 2)} pages."
        )
        ctx = summary.get("context")
        if ctx:
            out.append(
                f"- **Context recall**: the sources handed to the answer step contain all expected evidence for "
                f"{pct(ctx.get('all_targets_in_context'))} of answerable questions "
                f"(mean {num(ctx.get('mean_sources'), 1)} sources when the gate passes)."
            )
    return "\n".join(out)


def _confusion_table(conf: Mapping[str, Any]) -> str:
    rows = [
        [
            "answerable (documents hold the answer)",
            conf["answered_answerable"],
            conf["abstained_answerable"],
            conf["answerable"],
        ],
        [
            "unanswerable (abstention expected)",
            conf["answered_unanswerable"],
            conf["abstained_unanswerable"],
            conf["unanswerable"],
        ],
    ]
    return table(["", "answered", "abstained", "total"], rows)


def _abstention(summary: Mapping[str, Any]) -> str:
    ab = summary["abstention"]
    t = ab["threshold"]
    overall = ab["overall"]
    gate = (
        f"Gate (`Confidence.above_threshold`): answer when the best reranker score >= `min_rerank_score` = **{t}** and "
        "the best passage states every fiscal year the question names (nothing retrieved always abstains)."
    )
    score_gate = ab.get("score_gate")
    if score_gate:
        gate += (
            f" {ab.get('vetoed', 0)} questions name a year their best passage doesn't state; without that check the "
            f"score alone would answer {score_gate['answered_answerable']} answerable and "
            f"{score_gate['answered_unanswerable']} unanswerable questions."
        )
    out = [gate + "\n"]
    out.append(_confusion_table(overall))
    out.append(
        f"\nOf the questions answered, {pct(overall['answer_precision'])} had an answer in the documents (answer precision); "
        f"{pct(overall['answer_recall'])} of the answerable questions were answered (answer recall). "
        f"**Hallucination risk** (unanswerable questions answered): {pct(overall['hallucination_risk'])}. "
        f"**False-abstain rate**: {pct(overall['false_abstain_rate'])}. "
        f"Answered with the expected evidence among the retrieved passages: {overall['answered_with_evidence']} of {overall['answered_answerable']}.\n"
    )
    langs = ab["by_language"]
    rows = []
    for lang, c in langs.items():
        rows.append(
            [
                lang,
                c["answerable"],
                pct(c["answer_recall"]),
                pct(c["false_abstain_rate"]),
                c["unanswerable"],
                pct(c["abstain_recall"]),
                pct(c["hallucination_risk"]),
            ]
        )
    out.append("**By language**\n")
    out.append(
        table(
            [
                "language",
                "answerable",
                "answered",
                "false-abstain",
                "unanswerable",
                "correctly abstained",
                "hallucination risk",
            ],
            rows,
        )
    )
    subtypes = ab.get("by_subtype")
    if subtypes:
        out.append("\n**Unanswerable questions by subtype** (answered = the gate let it through)\n")
        out.append(
            table(
                ["subtype", "n", "answered", "hallucination risk"],
                [[k, v["n"], v["answered"], pct(v["hallucination_risk"])] for k, v in subtypes.items()],
            )
        )
    return "\n".join(out)


def _sweep(summary: Mapping[str, Any]) -> str:
    sw = summary["sweep"]["overall"]
    rows = []
    for p in sw:
        if p["threshold"] in SWEEP_VIEW:
            rows.append(
                [
                    p["threshold"],
                    p["answered_answerable"],
                    p["abstained_answerable"],
                    p["answered_unanswerable"],
                    p["abstained_unanswerable"],
                    pct(p["answer_precision"]),
                    pct(p["answer_recall"]),
                    num(p["answer_f1"]),
                    pct(p["hallucination_risk"]),
                ]
            )
    out = [
        "**Threshold sweep, all questions** (positive = the documents hold the answer; answered = the gate at that "
        "`min_rerank_score`)\n"
    ]
    out.append(
        table(
            [
                "threshold",
                "answered, answerable",
                "abstained, answerable",
                "answered, unanswerable",
                "abstained, unanswerable",
                "precision",
                "recall",
                "F1",
                "hallucination risk",
            ],
            rows,
        )
    )
    langs = summary["sweep"]["by_language"]
    out.append("\n**By language** (answer recall / hallucination risk)\n")
    header = ["threshold", *[f"{lang}" for lang in langs]]
    lrows = []
    for t in LANG_SWEEP_VIEW:
        row: list[Any] = [t]
        for lang in langs:
            point = next((p for p in langs[lang] if p["threshold"] == t), None)
            row.append(
                "-" if point is None else f"{pct(point['answer_recall'], 0)} / {pct(point['hallucination_risk'], 0)}"
            )
        lrows.append(row)
    out.append(table(header, lrows))
    best = summary["best_thresholds"]
    out.append("\n**Thresholds the data suggests** (searched over every observed score)\n")
    brows = []
    for scope, found in (
        ("all", best.get("overall", {})),
        *((lang, best["by_language"].get(lang, {})) for lang in best.get("by_language", {})),
    ):
        for name, label in (("max_f1", "max F1"), ("recall_target", "highest threshold with answer recall >= 95%")):
            p = found.get(name)
            if p:
                brows.append(
                    [
                        scope,
                        label,
                        num(p["threshold"], 5),
                        num(p["answer_f1"]),
                        pct(p["answer_recall"]),
                        pct(p["hallucination_risk"]),
                    ]
                )
    out.append(
        table(
            ["scope", "criterion", "threshold", "F1", "answer recall", "hallucination risk"], brows, align_right_from=2
        )
        if brows
        else "(not enough answerable and unanswerable questions)"
    )
    auc = summary["signal_auc"]
    langs_list = list(auc["top_score"]["by_language"])
    out.append(
        "\n**How well each signal separates answerable from unanswerable** (AUC: 1.0 = perfectly, 0.5 = not at all)\n"
    )
    out.append(
        table(
            ["signal", "all", *langs_list],
            [
                [name, num(v["overall"]), *[num(v["by_language"].get(lang)) for lang in langs_list]]
                for name, v in auc.items()
            ],
        )
    )
    return "\n".join(out)


def _distribution(summary: Mapping[str, Any]) -> str:
    dist = summary["score_distribution"]
    out = ["**Best reranker score, answerable vs unanswerable**\n"]
    rows = []
    for scope, d in (("all", dist["overall"]), *dist["by_language"].items()):
        for kind in ("answerable", "unanswerable"):
            s = d[kind]
            rows.append(
                [
                    scope,
                    kind,
                    s["n"],
                    num(s["min"], 4),
                    num(s["p25"], 4),
                    num(s["median"], 4),
                    num(s["p75"], 4),
                    num(s["max"], 4),
                ]
            )
    out.append(table(["language", "kind", "n", "min", "p25", "median", "p75", "max"], rows, align_right_from=2))
    edges = SCORE_BUCKETS
    labels = [f"{edges[i]:g} - {min(edges[i + 1], 1.0):g}" for i in range(len(edges) - 1)]
    out.append("\n**Histogram** (count of questions per score bucket)\n")
    langs = list(dist["by_language"])
    header = ["score", *[f"{lang} ans" for lang in langs], *[f"{lang} unans" for lang in langs]]
    hrows = []
    for i, label in enumerate(labels):
        hrows.append(
            [
                label,
                *[dist["by_language"][lang]["hist_answerable"][i] for lang in langs],
                *[dist["by_language"][lang]["hist_unanswerable"][i] for lang in langs],
            ]
        )
    out.append(table(header, hrows))
    return "\n".join(out)


def _by_group(summary: Mapping[str, Any], key: str, descriptions: Mapping[str, str] | None = None) -> str:
    rows = []
    for name, g in summary[key].items():
        rr, cand = g.get("rerank"), g.get("candidate")
        rows.append(
            [
                name,
                g["n"],
                g["n_answerable"],
                pct(_g(rr, "recall@1")),
                pct(_g(rr, "recall@3")),
                pct(_g(rr, "recall@5")),
                num(_g(rr, "mrr")),
                pct(_g(cand, "recall@5")),
                pct(_g(cand, "recall@all")),
                pct(g["answered_rate_answerable"]),
                pct(g["abstained_rate_unanswerable"]),
            ]
        )
    header = ["group", "n", "answerable", "R@1", "R@3", "R@5", "MRR", "cand R@5", "cand R@all", "answered", "abstained"]
    out = [table(header, rows)]
    out.append(
        "\nR@k and MRR are after the reranker; `cand` = before it; `answered` = answerable questions that pass the gate; `abstained` = unanswerable ones that the gate refuses."
    )
    if descriptions:
        out.append("\n" + "; ".join(f"`{k}`: {v}" for k, v in descriptions.items() if k in summary[key]))
    return "\n".join(out)


def _compare(title: str, *entries: tuple[str, Mapping[str, Any] | None]) -> str:
    rows = []
    for label, s in entries:
        if not s:
            continue
        rows.append(
            [
                label,
                s["n"],
                pct(_g(s, "page_level", "rerank", "recall@1")),
                pct(_g(s, "page_level", "rerank", "recall@3")),
                pct(_g(s, "page_level", "rerank", "recall@5")),
                num(_g(s, "page_level", "rerank", "mrr")),
                pct(_g(s, "page_level", "candidate", "recall@5")),
                pct(_g(s, "page_level", "candidate", "recall@all")),
                num(_g(s, "score_distribution", "overall", "answerable", "median"), 4),
                pct(_g(s, "abstention", "overall", "answer_recall")),
            ]
        )
    return f"**{title}**\n\n" + table(
        ["query", "n", "R@1", "R@3", "R@5", "MRR", "cand R@5", "cand R@all", "median best score", "answered"], rows
    )


def _compare_by_category(variants: Mapping[str, Any]) -> str:
    """Raw vs routed vs English-only per category: does the router's English query help on the English documents
    and hurt on the Hindi one?"""
    sources = [
        ("raw", variants.get("raw_on_routed")),
        ("routed", variants.get("routed")),
        ("english only", variants.get("english")),
    ]
    sources = [(label, s) for label, s in sources if s]
    cats = sorted({c for _, s in sources for c in s["by_category"]})
    rows = []
    for cat in cats:
        row: list[Any] = [cat, next((s["by_category"][cat]["n"] for _, s in sources if cat in s["by_category"]), 0)]
        for _, s in sources:
            g = s["by_category"].get(cat, {}).get("rerank")
            row += [pct(_g(g, "recall@1")), pct(_g(g, "recall@5")), num(_g(g, "mrr"))]
        rows.append(row)
    header = ["category", "n"]
    for label, _ in sources:
        header += [f"{label} R@1", f"{label} R@5", f"{label} MRR"]
    return "**By category**\n\n" + table(header, rows)


def _latency(summary: Mapping[str, Any]) -> str:
    rows = []
    for stage in ("embed", "search", "rerank", "total"):
        s = _g(summary, "latency_ms", stage)
        if s:
            rows.append([stage, s["n"], ms(s["mean"]), ms(s["p50"]), ms(s["p95"]), ms(s["max"])])
    return (
        table(["stage", "n", "mean ms", "p50 ms", "p95 ms", "max ms"], rows)
        + "\n\nWarm: one throwaway query loads the models before timing. `total` = embed + search + rerank of `retrieve`."
    )


def _ingestion(results: Mapping[str, Any], key: str) -> str:
    ing = results["ingestion"].get(key)
    if not ing or not ing["documents"]:
        return "(no ingestion in this run: a kept collection was reused)"
    rows = []
    for d in ing["documents"]:
        cov = d["fact_coverage"]
        s = d["seconds"]
        miss = ", ".join(cov["missing"][:6]) + (" ..." if len(cov["missing"]) > 6 else "")
        rows.append(
            [
                d["document"],
                d["pages_parsed"] if d["pages_parsed"] is not None else "-",
                d["chunks"],
                d["tables"],
                f"{d['tokens_mean']} / {d['tokens_max']}",
                d["multi_page_chunks"],
                f"{s['parse']:.1f} / {s['embed']:.1f} / {s['total']:.1f}",
                f"{cov['in_a_chunk']}/{cov['facts']}" + ("*" if cov["approximate_matching"] else ""),
                f"{cov['in_a_chunk_on_a_stating_page']}/{cov['facts']}",
                miss or "-",
            ]
        )
    header = [
        "document",
        "pages",
        "chunks",
        "tables",
        "tokens mean/max",
        "multi-page chunks",
        "parse/embed/total s",
        "facts in a chunk",
        "…on a stating page",
        "facts lost in ingestion",
    ]
    return (
        table(header, rows)
        + "\n\n`facts in a chunk` counts planted facts whose evidence text is inside some chunk after parsing and chunking (`*` = OCR document, matched ignoring spaces and punctuation): anything missing here is an ingestion failure, not a retrieval one."
    )


def _misses(rows: Sequence[Mapping[str, Any]], pipeline_variants: set[tuple[str, str]]) -> str:
    miss = [
        r
        for r in rows
        if not r["abstain"]
        and (r["id"], r["variant"]) in pipeline_variants
        and (r.get("rerank_first_rank") is None or r["rerank_first_rank"] > 5)
    ]
    lost = [
        r
        for r in rows
        if not r["abstain"]
        and (r["id"], r["variant"]) in pipeline_variants
        and r.get("candidate_first_rank") is not None
        and r["candidate_first_rank"] <= 5
        and (r.get("rerank_first_rank") is None or r["rerank_first_rank"] > r["candidate_first_rank"])
    ]
    out = [
        f"{len(miss)} answerable questions have no expected page in the top 5 after reranking; the reranker pushed the right page down on {len(lost)} questions that the candidates had in their top 5.\n"
    ]
    if not miss:
        return out[0]
    rows_out = []
    for r in sorted(miss, key=lambda r: (r["category"], r["id"]))[:MAX_MISSES]:
        exp = "; ".join(
            f"{e['document'].split('_')[0]} p{','.join(map(str, e['pages'])) or '-'}" for e in r["expected"]
        )
        top = "; ".join(f"{t['document'].split('_')[0]} p{t['pages']} ({num(t['score'], 3)})" for t in r["top"])
        rows_out.append(
            [r["id"], r["category"], r["language"], r["question"], exp, r["candidate_first_rank"] or "-", top or "-"]
        )
    out.append(
        table(
            ["id", "category", "lang", "question", "expected", "cand rank", "top 3 retrieved (rerank score)"],
            rows_out,
            align_right_from=9,
        )
    )
    if len(miss) > MAX_MISSES:
        out.append(f"\n... and {len(miss) - MAX_MISSES} more in results.json (`rows`).")
    return "\n".join(out)


def _gate_errors(rows: Sequence[Mapping[str, Any]], pipeline_variants: set[tuple[str, str]], threshold: float) -> str:
    sel = [r for r in rows if (r["id"], r["variant"]) in pipeline_variants]
    false_abstain = sorted(
        (r for r in sel if not r["abstain"] and not r["answered"]), key=lambda r: -(r["top_score"] or 0)
    )
    false_answer = sorted((r for r in sel if r["abstain"] and r["answered"]), key=lambda r: -(r["top_score"] or 0))
    out = [f"**Answerable questions the gate refuses** ({len(false_abstain)}; `min_rerank_score` {threshold})\n"]
    out.append(
        table(
            ["id", "lang", "question", "best score", "rank of expected page"],
            [
                [r["id"], r["language"], r["question"], num(r["top_score"], 4), r.get("rerank_first_rank") or "-"]
                for r in false_abstain[:MAX_GATE_ERRORS]
            ],
            align_right_from=3,
        )
        if false_abstain
        else "(none)"
    )
    out.append(f"\n**Unanswerable questions the gate lets through** ({len(false_answer)})\n")
    out.append(
        table(
            ["id", "lang", "subtype", "question", "best score", "top passage"],
            [
                [
                    r["id"],
                    r["language"],
                    r.get("subtype") or "-",
                    r["question"],
                    num(r["top_score"], 4),
                    (f"{r['top'][0]['document'].split('_')[0]} p{r['top'][0]['pages']}" if r["top"] else "-"),
                ]
                for r in false_answer[:MAX_GATE_ERRORS]
            ],
            align_right_from=4,
        )
        if false_answer
        else "(none)"
    )
    return "\n".join(out)


def _answers(summary: Mapping[str, Any]) -> str:
    a = summary.get("answers")
    if not a:
        return ""
    ans, un = a["answerable"], a["unanswerable"]
    out = ["### Answers (answer LLM)\n"]
    out.append(
        f"- Answerable ({ans['n']}): `answer_contains` satisfied for **{pct(ans['contains_ok'])}**; abstained: {pct(ans['abstained'])}; "
        f"cites at least one correct page: {pct(ans['cites_a_correct_page'])} (citation precision {pct(ans['citation_precision'])}, recall {pct(ans['citation_recall'])}); "
        f"no citation at all: {pct(ans['no_citation'])}."
    )
    out.append(
        f"- Unanswerable ({un['n']}): refused by the gate {pct(un['abstained'])}; answered anyway: {', '.join(un['answered_anyway']) or 'none'}."
    )
    out.append(
        "\n" + table(["group", "answer_contains satisfied"], [[k, pct(v)] for k, v in ans["by_category"].items()])
    )
    out.append(
        "\n" + table(["language", "answer_contains satisfied"], [[k, pct(v)] for k, v in ans["by_language"].items()])
    )
    return "\n".join(out)


def _run_section(results: Mapping[str, Any], run: Mapping[str, Any], index: int) -> str:
    v = run["variants"]
    pipeline = v.get("pipeline")
    p = run["params"]
    out = [f"## Run {index}: {run['label']}\n"]
    s = run["settings"]
    out.append(
        f"Embeddings `{s['embeddings']['model']}` ({s['embeddings']['device']}), reranker `{s['reranker']['model']}` ({s['reranker']['device']}); "
        f"chunks {s['chunking']['target_tokens']} tokens + {s['chunking']['overlap_tokens']} overlap; retrieval {s['retrieval']}.\n"
    )
    if not pipeline:
        return "\n".join([*out, "(no `raw` variant in this run)"])
    out.append("### Headline (pipeline: routed query where the question has an English query, raw otherwise)\n")
    out.append(_headline(pipeline))
    out.append("\n### By category\n")
    out.append(_by_group(pipeline, "by_category", CATEGORY_DESCRIPTIONS))
    out.append("\n### By language\n")
    out.append(_by_group(pipeline, "by_language"))
    if "routed" in v:
        out.append("\n### Raw vs router-rewritten query (Hindi and Hinglish questions)\n")
        out.append(
            _compare(
                "Questions with an English query",
                ("raw (question only)", v.get("raw_on_routed")),
                ("routed (question + English query)", v["routed"]),
                ("English query only", v.get("english")),
            )
        )
        out.append("\n" + _compare_by_category(v))
        out.append(
            "\nThe routed run searches both the question and its English query (fused by RRF) and reranks with the English one, as `RetrievalService.retrieve(query, query_en=...)` does; `English query only` searches and reranks with the English query alone."
        )
    if "clean" in v:
        out.append("\n### Speech-recognition noise\n")
        out.append(
            _compare(
                "ASR-noise questions",
                ("as transcribed (noisy)", v.get("raw_on_clean")),
                ("as spoken (clean)", v["clean"]),
            )
        )
    out.append("\n### Abstention\n")
    out.append(_abstention(pipeline))
    out.append("\n### Threshold sweep and score distributions\n")
    out.append(_sweep(pipeline))
    out.append("\n" + _distribution(pipeline))
    out.append("\n### Latency per stage (ms, pipeline queries)\n")
    out.append(_latency(pipeline))
    out.append("\n### Misses\n")
    pipeline_pairs = {
        (r["id"], r["variant"])
        for r in run["rows"]
        if r["variant"] == "routed" or (r["variant"] == "raw" and not r["query_en"])
    }
    out.append(_misses(run["rows"], pipeline_pairs))
    out.append("\n### Abstention errors\n")
    out.append(_gate_errors(run["rows"], pipeline_pairs, p["threshold"]))
    answers = _answers(pipeline)
    if answers:
        out.append("\n" + answers)
    out.append("\n### Ingestion\n")
    out.append(_ingestion(results, run["ingestion"]))
    return "\n".join(out)


def _grid(runs: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for r in runs:
        s = r["variants"].get("pipeline")
        if not s:
            continue
        conf = s["abstention"]["overall"]
        rows.append(
            [
                r["label"],
                pct(_g(s, "page_level", "rerank", "recall@1")),
                pct(_g(s, "page_level", "rerank", "recall@3")),
                pct(_g(s, "page_level", "rerank", "recall@5")),
                num(_g(s, "page_level", "rerank", "mrr")),
                pct(_g(s, "page_level", "candidate", "recall@5")),
                pct(_g(s, "page_level", "candidate", "recall@all")),
                ms(_g(s, "latency_ms", "rerank", "p50")),
                ms(_g(s, "latency_ms", "total", "p50")),
                num(conf["answer_f1"]),
                pct(conf["hallucination_risk"]),
                pct(conf["false_abstain_rate"]),
            ]
        )
    header = [
        "run",
        "R@1",
        "R@3",
        "R@5",
        "MRR",
        "cand R@5",
        "cand R@all",
        "rerank p50 ms",
        "total p50 ms",
        "gate F1",
        "halluc. risk",
        "false abstain",
    ]
    return table(header, rows)


def render_markdown(results: Mapping[str, Any]) -> str:
    """The full Markdown report for a results dict."""
    meta = results["meta"]
    q = results["questions"]
    out = ["# Retrieval evaluation\n"]
    out.append(
        f"- Run: {meta['timestamp']} (commit {meta.get('git_commit') or 'unknown'}), {meta['platform']}, Python {meta['python']}\n"
        f"- Command: `{meta['command']}`\n"
        f"- Corpus: {len(results['corpus']['documents'])} documents, {results['corpus']['facts']} planted facts "
        f"({', '.join(d['name'] for d in results['corpus']['documents'])})\n"
        f"- Questions: {q['n']} (answerable {q['abstain'].get('answer', 0)}, unanswerable {q['abstain'].get('abstain', 0)}); "
        f"by language {q['language']}\n"
        f"- Gate threshold (`min_rerank_score`): {meta['threshold']}; variants {', '.join(meta['variants'])}"
        f"{'; answers ON' if meta['answers'] else ''}\n"
    )
    runs = results["runs"]
    if len(runs) > 1:
        out.append("## Comparison of runs\n")
        out.append(_grid(runs))
        out.append("")
    for i, run in enumerate(runs, start=1):
        out.append(_run_section(results, run, i))
        out.append("")
    return "\n".join(out).rstrip() + "\n"
