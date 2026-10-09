"""The retrieval-eval harness wired end to end with fake providers: ingestion, both query variants, candidates and
reranked passages, the abstention gate, the answer step, the grid, the outputs. No models, Qdrant or Ollama."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.evals.manifest import DocumentEntry, Fact, Manifest
from app.evals.questions import Expected, Question
from app.evals.report import render_markdown
from app.evals.retrieval_eval import (
    PROJECT_ID,
    EvalOptions,
    EvalServices,
    EvalSetupError,
    amain,
    build_parser,
    container_factory,
    document_id,
    fact_coverage,
    make_settings,
    preflight,
    query_variants,
    run_eval,
    select_questions,
    targets_for,
    with_params,
    write_outputs,
)
from app.providers.llm import LLMMessage
from app.providers.registry import build_container
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.settings import Settings

from .conftest import mock_http
from .fakes import FakeEmbedder, FakeLLM, FakeReranker, FakeStore, TextParser, keyword_scorer

ALPHA = "alpha_report.pdf"
BETA = "beta_deck.pptx"
NOTES = "notes.docx"

CORPUS = {
    # pages are separated by form feeds, paragraphs by blank lines (the fake parser's format)
    ALPHA: "Alpha Corp overview.\n\nEBITDA margin was 21.0% in FY24 for Alpha.\f"
    "Dividend per share was 15.00 rupees.\n\nThe record date is 16 August 2024.",
    BETA: "Beta Ltd EBITDA margin was 12.7% in FY24.",
    NOTES: "4.2.1 Employees in grades L1 to L5 travel economy class on domestic flights.\n\n"
    "5.1 Hotel limits are 7,500 per night.",
}


def fact(fid: str, doc: str, evidence: list[str], pages: list[int] | None = None, locator: str | None = None) -> Fact:
    return Fact(
        id=fid,
        document=doc,
        pages=pages or [],
        planted_pages=pages or [],
        locator=locator,
        evidence=evidence,
        statement=fid,
        kind="number",
    )


def make_manifest() -> Manifest:
    return Manifest(
        documents=[
            DocumentEntry(name=ALPHA, title="Alpha", format="pdf", language="en", pages=2),
            DocumentEntry(name=BETA, title="Beta", format="pptx", language="en", pages=1),
            DocumentEntry(name=NOTES, title="Notes", format="docx", language="en", pages=None),
        ],
        facts=[
            fact("a.margin", ALPHA, ["EBITDA margin", "21.0%"], [1]),
            fact("a.dividend", ALPHA, ["Dividend per share", "15.00"], [2]),
            fact("b.margin", BETA, ["EBITDA margin", "12.7%"], [1]),
            fact("n.clause", NOTES, ["4.2.1", "economy class"], locator="4.2 Air travel"),
            fact("a.lost", ALPHA, ["a sentence the parser dropped"], [1]),
        ],
    )


def Q(
    qid: str, question: str, category: str, facts_: list[tuple[str, str, list[int]]], answers: list[str], **kw
) -> Question:
    return Question(
        id=qid,
        question=question,
        language=kw.pop("language", "en"),
        category=category,  # type: ignore[arg-type]
        expected=[Expected(document=d, pages=p, facts=[f]) for f, d, p in facts_],
        answer_contains=answers,
        should_abstain=not facts_,
        **kw,
    )


def make_questions() -> list[Question]:
    return [
        Q("q1", "What was the EBITDA margin in FY24 for Alpha?", "exact_fact", [("a.margin", ALPHA, [1])], ["21.0%"]),
        Q("q2", "What dividend per share was declared?", "paraphrase", [("a.dividend", ALPHA, [2])], ["15.00"]),
        Q(
            "q3",
            "बीटा का EBITDA मार्जिन कितना था?",
            "xlingual",
            [("b.margin", BETA, [1])],
            ["12.7%"],
            language="hi",
            query_en="What was Beta EBITDA margin in FY24?",
        ),
        Q(
            "q4",
            "What does clause 4.2.1 say about domestic flights?",
            "identifier",
            [("n.clause", NOTES, [])],
            ["economy"],
        ),
        Q("q5", "What is the weather in Mumbai today?", "unanswerable", [], [], subtype="off_topic"),
        Q("q6", "What is Alpha's EBITDA margin for FY25?", "unanswerable", [], [], subtype="near_miss_year"),
        Q(
            "q7",
            "What was the ebit da margin in FY24 for Alpha?",
            "asr_noise",
            [("a.margin", ALPHA, [1])],
            ["21.0%"],
            clean_question="What was the EBITDA margin in FY24 for Alpha?",
        ),
    ]


class Harness:
    """Fake providers shared by every configuration the factory is asked for (one in-memory store)."""

    def __init__(self, settings: Settings, llm_reply=None) -> None:
        self.settings = settings
        self.embedder = FakeEmbedder()
        self.reranker = FakeReranker(keyword_scorer)
        self.store = FakeStore(search_points=True)
        self.parser = TextParser()
        self.llm = FakeLLM(llm_reply or cite_everything)
        self.factory_calls: list[Settings] = []

    def factory(self, settings: Settings) -> EvalServices:
        self.factory_calls.append(settings)
        return EvalServices(
            ingestion=IngestionService(self.parser, self.embedder, self.store),
            retrieval=RetrievalService(self.embedder, self.reranker, self.store, settings.retrieval),
            store=self.store,
            llm=self.llm,
        )


def cite_everything(messages: list[LLMMessage]) -> str:
    """A scripted 'LLM': repeats every numbered source's text and cites it ([S1], [S2], ...)."""
    user = messages[-1].content
    sources = user.split("Sources:\n\n", 1)[1].split("\n\nQuestion:", 1)[0]
    parts = []
    for block in sources.split("\n\n"):
        found = re.match(r"\[(S\d+)\] [^\n]*\n(.*)", block, re.S)
        if found:
            parts.append(f"{found.group(2).strip()} [{found.group(1)}]")
    return " ".join(parts)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in CORPUS.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


@pytest.fixture
def harness(load_local) -> Harness:
    # The fake scores below are built around a 0.3 gate; pin it so a change to the shipped config doesn't move them.
    return Harness(with_params(load_local(), threshold=0.3))


async def run(h: Harness, docs: Path, options: EvalOptions | None = None, questions: list[Question] | None = None):
    return await run_eval(
        h.settings,
        h.factory,
        make_manifest(),
        questions or make_questions(),
        docs,
        options or EvalOptions(prefetch_ks=(20,)),
        log=lambda _msg: None,
    )


# ------------------------------------------------------------------ end to end


async def test_ingests_the_corpus_and_reports_per_document_coverage(harness, docs):
    results, chunks = await run(harness, docs)
    (ing,) = results["ingestion"].values()
    by_doc = {d["document"]: d for d in ing["documents"]}
    assert set(by_doc) == {ALPHA, BETA, NOTES}
    assert by_doc[ALPHA]["chunks"] == 4 and by_doc[ALPHA]["pages_parsed"] == 2
    cov = by_doc[ALPHA]["fact_coverage"]
    assert (
        cov["facts"] == 3 and cov["in_a_chunk"] == 2 and cov["missing"] == ["a.lost"]
    )  # the dropped sentence is caught
    assert by_doc[NOTES]["fact_coverage"]["in_a_chunk"] == 1  # page-less fact found by its text
    assert {c["document"] for c in chunks} == {ALPHA, BETA, NOTES}
    assert all(c["text"] for c in chunks)
    assert harness.parser.release_calls >= 1  # Docling is released after ingestion (§8)


async def test_the_throwaway_collection_is_dropped_afterwards(harness, docs):
    await run(harness, docs)
    assert harness.store.points == {}  # delete_project ran (the fake store has no collections to drop)


async def test_keep_collection_leaves_the_index_alone(harness, docs):
    await run(harness, docs, EvalOptions(prefetch_ks=(20,), keep_collection=True))
    assert len(harness.store.points) == 7  # 4 + 1 + 2 chunks


async def test_headline_ranking_metrics_are_computed_on_pages(harness, docs):
    results, _ = await run(harness, docs)
    pipe = results["runs"][0]["variants"]["pipeline"]
    assert (pipe["n"], pipe["n_answerable"], pipe["n_unanswerable"]) == (7, 5, 2)
    rerank = pipe["page_level"]["rerank"]
    assert rerank["recall@1"] == 1.0 and rerank["mrr"] == 1.0 and rerank["success@5"] == 1.0
    cand = pipe["page_level"]["candidate"]
    assert cand["recall@all"] == 1.0
    assert (
        cand["recall@1"] < rerank["recall@1"]
    )  # the fake store returns chunks in insertion order, the reranker fixes it
    assert pipe["fact_level"]["rerank"]["recall@1"] == 1.0
    assert pipe["citation_top1"]["lenient"] == 1.0
    assert set(pipe["by_category"]) == {
        "exact_fact",
        "paraphrase",
        "xlingual",
        "identifier",
        "unanswerable",
        "asr_noise",
    }
    assert set(pipe["by_language"]) == {"en", "hi"}


async def test_the_router_query_fixes_a_hindi_question_that_the_raw_one_misses(harness, docs):
    results, _ = await run(harness, docs)
    v = results["runs"][0]["variants"]
    raw, routed = v["raw_on_routed"], v["routed"]
    assert (raw["n"], routed["n"]) == (1, 1)
    # raw: the Hindi words match no English passage well enough; routed: the English query is searched and reranks
    assert raw["page_level"]["rerank"]["recall@1"] == 0.0
    assert routed["page_level"]["rerank"]["recall@1"] == 1.0
    assert raw["abstention"]["overall"]["false_abstain_rate"] == 1.0  # the raw score is below the gate
    assert routed["abstention"]["overall"]["false_abstain_rate"] == 0.0
    english = v["english"]  # the English query alone: no Hindi words to confuse the search
    assert english["n"] == 1 and english["page_level"]["rerank"]["recall@1"] == 1.0
    # the reranker was given the English query for the routed run
    assert "What was Beta EBITDA margin in FY24?" in {q for q, _ in harness.reranker.calls}


async def test_retrieval_is_called_with_the_project_scope_and_every_document(harness, docs):
    await run(harness, docs)
    ids = tuple(document_id(name) for name in CORPUS)  # what the chat pipeline passes: every READY document
    assert harness.store.searches
    assert all(f.project_id == PROJECT_ID and f.document_ids == ids for _, f, _ in harness.store.searches)


async def test_the_clean_variant_is_run_for_asr_questions_only(harness, docs):
    results, _ = await run(harness, docs)
    v = results["runs"][0]["variants"]
    assert v["clean"]["n"] == 1 and v["raw_on_clean"]["n"] == 1
    rows = [r for r in results["runs"][0]["rows"] if r["variant"] == "clean"]
    assert [r["id"] for r in rows] == ["q7"] and rows[0]["query"].startswith("What was the EBITDA margin")


async def test_the_confusion_matrix_exposes_a_near_miss_answered_by_the_gate(harness, docs):
    results, _ = await run(harness, docs)
    conf = results["runs"][0]["variants"]["pipeline"]["abstention"]["overall"]
    assert conf["threshold"] == pytest.approx(0.3)  # min_rerank_score as the fixture pins it
    assert conf["answered_answerable"] == 5 and conf["abstained_answerable"] == 0
    # the weather question scores 0 and is refused; "Alpha's FY25 margin" shares words with the FY24 page
    # and slips through
    assert conf["abstained_unanswerable"] == 1 and conf["answered_unanswerable"] == 1
    assert conf["hallucination_risk"] == 0.5
    rows = {r["id"]: r for r in results["runs"][0]["rows"] if r["variant"] == "raw"}
    assert rows["q6"]["answered"] and not rows["q5"]["answered"]


async def test_the_threshold_option_changes_the_gate_not_the_ranking(harness, docs):
    results, _ = await run(harness, docs, EvalOptions(prefetch_ks=(20,), threshold=0.7))
    pipe = results["runs"][0]["variants"]["pipeline"]
    assert pipe["abstention"]["threshold"] == 0.7
    assert pipe["abstention"]["overall"]["answered_unanswerable"] == 0  # 0.6 no longer passes
    assert pipe["abstention"]["overall"]["abstained_answerable"] >= 1
    assert pipe["page_level"]["rerank"]["recall@1"] == 1.0


async def test_sweep_and_distributions_are_reported_per_language(harness, docs):
    results, _ = await run(harness, docs)
    pipe = results["runs"][0]["variants"]["pipeline"]
    thresholds = [p["threshold"] for p in pipe["sweep"]["overall"]]
    assert thresholds == sorted(thresholds) and 0.0 in thresholds and 0.9 in thresholds
    assert set(pipe["sweep"]["by_language"]) == {"en", "hi"}
    dist = pipe["score_distribution"]["by_language"]["en"]
    assert dist["answerable"]["n"] == 4 and dist["unanswerable"]["n"] == 2
    assert dist["answerable"]["median"] > dist["unanswerable"]["median"]
    assert pipe["signal_auc"]["top_score"]["overall"] is not None
    assert set(pipe["latency_ms"]) >= {"embed", "search", "rerank", "total"}


async def test_grid_runs_every_combination_on_one_ingestion(harness, docs):
    opts = EvalOptions(prefetch_ks=(2, 20), rerank_top_ns=(3, 5))
    results, _ = await run(harness, docs, opts)
    runs = results["runs"]
    assert len(runs) == 4 and len(results["ingestion"]) == 1
    assert [(r["params"]["prefetch_k"], r["params"]["rerank_top_n"]) for r in runs] == [
        (2, 3),
        (2, 5),
        (20, 3),
        (20, 5),
    ]
    small = runs[0]["variants"]["pipeline"]["page_level"]["candidate"]["recall@all"]
    large = runs[2]["variants"]["pipeline"]["page_level"]["candidate"]["recall@all"]
    assert small < large == 1.0  # a tiny prefetch_k cannot reach every page
    md = render_markdown(results)
    assert "Comparison of runs" in md and "prefetch_k=2" in md and "prefetch_k=20" in md
    assert runs[0]["settings"]["retrieval"]["prefetch_k"] == 2  # the run's settings are recorded


async def test_chunk_sizes_reingest_into_their_own_collections(harness, docs):
    results, _ = await run(harness, docs, EvalOptions(prefetch_ks=(20,), chunk_sizes=(300, 700), overlap=40))
    assert sorted(results["ingestion"]) == ["300", "700"]
    assert [r["params"]["chunk_size"] for r in results["runs"]] == [300, 700]
    sizes = {s.ingestion.chunking.target_tokens for s in harness.factory_calls}
    assert sizes == {300, 700}
    assert all(s.ingestion.chunking.overlap_tokens == 40 for s in harness.factory_calls)
    assert len({s.vector_store.collection for s in harness.factory_calls}) == 2  # separate collections per size


async def test_filters_select_questions(harness, docs):
    results, _ = await run(harness, docs, EvalOptions(prefetch_ks=(20,), categories=("exact_fact", "unanswerable")))
    assert results["questions"]["n"] == 3
    assert results["runs"][0]["variants"]["pipeline"]["n"] == 3


async def test_an_inconsistent_question_set_is_refused_before_anything_runs(harness, docs):
    bad = [Q("q1", "?", "exact_fact", [("ghost", ALPHA, [1])], ["x"])]
    with pytest.raises(EvalSetupError, match="inconsistent"):
        await run(harness, docs, questions=bad)
    assert harness.parser.parsed_paths == []


async def test_a_missing_document_says_how_to_build_the_corpus(harness, docs):
    (docs / BETA).unlink()
    with pytest.raises(EvalSetupError, match="build the corpus"):
        await run(harness, docs)


# ------------------------------------------------------------------ the answer step


async def test_answers_mode_checks_answer_strings_and_cited_pages(harness, docs):
    results, _ = await run(harness, docs, EvalOptions(prefetch_ks=(20,), answers=True))
    pipe = results["runs"][0]["variants"]["pipeline"]
    a = pipe["answers"]
    assert a["answerable"]["n"] == 5 and a["answerable"]["contains_ok"] == 1.0
    assert a["answerable"]["cites_a_correct_page"] == 1.0 and a["answerable"]["no_citation"] == 0.0
    assert a["unanswerable"]["abstained"] == 0.5 and a["unanswerable"]["answered_anyway"] == ["q6"]
    assert (
        len(harness.llm.calls) == 6
    )  # five answerable + the near miss the gate lets through; the weather question never reaches the LLM
    prompt = harness.llm.calls[0]["messages"]
    assert prompt[0].role == "system" and "numbered sources" in prompt[0].content
    assert "Question: What was the EBITDA margin in FY24 for Alpha?" in prompt[1].content
    rows = {r["id"]: r for r in results["runs"][0]["rows"] if r["variant"] == "raw"}
    assert rows["q5"]["answer"]["abstained"] and "couldn't find" in rows["q5"]["answer"]["text"]


async def test_answers_are_off_by_default(harness, docs):
    results, _ = await run(harness, docs)
    assert harness.llm.calls == [] and "answers" not in results["runs"][0]["variants"]["pipeline"]


async def test_a_wrong_answer_and_a_missing_citation_are_counted(load_local, docs):
    h = Harness(load_local(), llm_reply="Nothing useful, no citation.")
    results, _ = await run(h, docs, EvalOptions(prefetch_ks=(20,), answers=True))
    a = results["runs"][0]["variants"]["pipeline"]["answers"]["answerable"]
    assert a["contains_ok"] == 0.0 and a["no_citation"] == 1.0 and a["cites_a_correct_page"] == 0.0


async def test_answers_without_an_llm_are_refused(load_local, docs):
    h = Harness(load_local())

    def factory(settings: Settings) -> EvalServices:
        s = h.factory(settings)
        s.llm = None
        return s

    with pytest.raises(EvalSetupError, match="needs an LLM"):
        await run_eval(
            h.settings,
            factory,
            make_manifest(),
            make_questions(),
            docs,
            EvalOptions(prefetch_ks=(20,), answers=True),
            log=lambda _m: None,
        )


# ------------------------------------------------------------------ outputs


async def test_outputs_are_a_json_file_a_markdown_summary_and_the_chunks(harness, docs, tmp_path):
    results, chunks = await run(harness, docs)
    out = write_outputs(tmp_path / "results" / "run1", results, chunks)
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["format"] == 1 and data["runs"][0]["variants"]["pipeline"]["n"] == 7
    assert (out / "chunks.jsonl").read_text(encoding="utf-8").count("\n") == len(chunks)
    md = (out / "summary.md").read_text(encoding="utf-8")
    for heading in (
        "# Retrieval evaluation",
        "### Headline",
        "### By category",
        "### By language",
        "### Raw vs router-rewritten query",
        "### Speech-recognition noise",
        "### Abstention",
        "### Threshold sweep and score distributions",
        "### Latency per stage",
        "### Misses",
        "### Abstention errors",
        "### Ingestion",
    ):
        assert heading in md, heading
    assert "Recall@5" in md and "hallucination risk" in md.lower() and "a.lost" in md  # the dropped fact is listed
    assert "q6" in md  # the near miss the gate answers is named in the abstention errors


async def test_summary_can_be_rendered_again_from_results_json(harness, docs, tmp_path):
    results, chunks = await run(harness, docs)
    out = write_outputs(tmp_path / "run", results, chunks)
    (out / "summary.md").write_text("stale", encoding="utf-8")
    assert await amain(["--render", str(out / "results.json")]) == 0
    assert (out / "summary.md").read_text(encoding="utf-8") == render_markdown(results)


# ------------------------------------------------------------------ helpers


def test_document_ids_are_stable_and_safe():
    assert document_id("valmora_annual_report_fy24.pdf") == "eval-valmora-annual-report-fy24"
    assert document_id("Zephyra Deck (Q4).pptx") == "eval-zephyra-deck-q4"


def test_targets_carry_pages_and_the_evidence_of_their_facts():
    (t,) = targets_for(make_questions()[0], make_manifest())
    assert t.document == ALPHA and t.pages == (1,) and t.evidence == (("EBITDA margin", "21.0%"),) and not t.approximate
    (page_less,) = targets_for(make_questions()[3], make_manifest())
    assert page_less.pages == () and page_less.evidence


def test_variants_apply_to_the_questions_that_have_them():
    qs = {q.id: q for q in make_questions()}
    assert [v for v, _, _ in query_variants(qs["q1"], ("raw", "routed", "clean"))] == ["raw"]
    assert [(v, en) for v, _, en in query_variants(qs["q3"], ("raw", "routed", "clean"))] == [
        ("raw", None),
        ("routed", "What was Beta EBITDA margin in FY24?"),
    ]
    assert [(v, t) for v, t, _ in query_variants(qs["q7"], ("raw", "routed", "clean"))][1] == (
        "clean",
        "What was the EBITDA margin in FY24 for Alpha?",
    )
    assert [v for v, _, _ in query_variants(qs["q3"], ("raw",))] == ["raw"]
    assert [(v, t, en) for v, t, en in query_variants(qs["q3"], ("english",))] == [
        ("english", "What was Beta EBITDA margin in FY24?", None)
    ]
    assert [v for v, _, _ in query_variants(qs["q1"], ("english",))] == []  # English questions have no router query


def test_select_questions_filters_and_limits():
    qs = make_questions()
    assert [q.id for q in select_questions(qs, EvalOptions(languages=("hi",)))] == ["q3"]
    assert [q.id for q in select_questions(qs, EvalOptions(ids=("q2", "q5")))] == ["q2", "q5"]
    assert len(select_questions(qs, EvalOptions(limit=2))) == 2


def test_with_params_changes_only_what_is_asked(load_local):
    s = load_local()
    t = with_params(
        s, collection="c", chunk_size=300, overlap=20, prefetch_k=12, rerank_top_n=7, fusion="dbsf", threshold=0.05
    )
    assert t.vector_store.collection == "c" and t.vector_store.url == s.vector_store.url
    assert (t.ingestion.chunking.target_tokens, t.ingestion.chunking.overlap_tokens) == (300, 20)
    assert (t.retrieval.prefetch_k, t.retrieval.rerank_top_n, t.retrieval.fusion, t.retrieval.min_rerank_score) == (
        12,
        7,
        "dbsf",
        0.05,
    )
    assert t.retrieval.context_token_budget == s.retrieval.context_token_budget
    assert s.retrieval.prefetch_k == 8  # the original is untouched
    with pytest.raises(EvalSetupError, match="invalid chunking"):
        with_params(s, chunk_size=100, overlap=100)
    with pytest.raises(EvalSetupError, match="min_rerank_score"):
        with_params(s, threshold=5)
    with pytest.raises(EvalSetupError, match="prefetch_k"):
        with_params(s, prefetch_k=0)
    with pytest.raises(EvalSetupError, match="fusion"):
        with_params(s, fusion="bm25")


def test_fact_coverage_matches_headings_text_and_pages(load_local):
    from app.providers.ingestion import chunk_id

    from .fakes import make_chunk

    chunks = [make_chunk(0, text="EBITDA margin was 21.0%", page_start=2, page_end=2, heading_path=["Highlights"])]
    m = Manifest(
        documents=[DocumentEntry(name="d.pdf", title="D", format="pdf", language="en", pages=3)],
        facts=[
            fact("on_page", "d.pdf", ["EBITDA margin", "21.0%"], [2]),
            fact("wrong_page", "d.pdf", ["EBITDA margin", "21.0%"], [3]),
            fact("in_heading", "d.pdf", ["Highlights"], [2]),
            fact("absent", "d.pdf", ["nothing like this"], [2]),
        ],
    )
    cov = fact_coverage(m, "d.pdf", chunks)
    assert cov["facts"] == 4 and cov["in_a_chunk"] == 3 and cov["in_a_chunk_on_a_stating_page"] == 2
    assert cov["missing"] == ["absent"]
    assert chunk_id("d", 1, 0) == "d:v1:0000"


def test_cli_parses_lists_and_flags():
    a = build_parser().parse_args(
        [
            "--prefetch-k",
            "8,12,20",
            "--rerank-top-n",
            "5",
            "--chunk-size",
            "300,500",
            "--threshold",
            "0.05",
            "--variants",
            "raw",
            "--answers",
            "--categories",
            "xlingual,asr_noise",
        ]
    )
    assert a.prefetch_k == (8, 12, 20) and a.rerank_top_n == (5,) and a.chunk_size == (300, 500)
    assert a.threshold == 0.05 and a.variants == ("raw",) and a.answers and a.categories == ("xlingual", "asr_noise")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--prefetch-k", "eight"])


# ------------------------------------------------------------------ wiring to the app's real providers (no models)


def test_the_real_factory_builds_services_from_the_apps_providers(load_local):
    settings = load_local()
    container = build_container(settings, http=mock_http())  # providers are lazy: nothing is loaded
    make = container_factory(container)
    a = make(with_params(settings, collection="eval_a", chunk_size=300, overlap=40, prefetch_k=12, rerank_top_n=7))
    b = make(with_params(settings, collection="eval_b"))
    assert a.store.collection == "eval_a" and b.store.collection == "eval_b"  # type: ignore[attr-defined]
    assert a.ingestion.parser.cfg.chunking.target_tokens == 300  # type: ignore[attr-defined]
    assert b.ingestion.parser.cfg.chunking.target_tokens == settings.ingestion.chunking.target_tokens  # type: ignore[attr-defined]
    assert a.store.retrieval.prefetch_k == 12  # type: ignore[attr-defined]  (the store reads it from its own context)
    assert (a.retrieval.config.prefetch_k, a.retrieval.config.rerank_top_n) == (12, 7)
    # the expensive models are shared by every configuration
    assert a.retrieval.embedder is b.retrieval.embedder is container["embeddings"]
    assert a.retrieval.reranker is b.retrieval.reranker is container["reranker"]
    assert a.ingestion.embedder is container["embeddings"] and a.llm is container["llm"]


async def test_preflight_names_what_is_missing(load_local):
    container = build_container(load_local(), http=mock_http())  # no model files under tmp_path
    with pytest.raises(EvalSetupError, match="embeddings") as e:
        await preflight(container, EvalOptions())
    assert "reranker" in str(e.value) and "ingestion" in str(e.value)


def test_settings_take_models_and_qdrant_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6444")
    s = make_settings(None)
    assert s.path(s.model_cache.hf_home) == tmp_path / "models" / "huggingface"
    assert s.path(s.ingestion.artifacts_path) == tmp_path / "models" / "docling"
    assert s.vector_store.url == "http://127.0.0.1:6444"


async def test_the_raw_variant_is_required(harness, docs):
    with pytest.raises(EvalSetupError, match="raw variant"):
        await run(harness, docs, EvalOptions(variants=("routed",)))
