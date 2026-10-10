"""Retrieval service with fake providers: fusion across queries, reranking, confidence."""

from __future__ import annotations

import asyncio

import pytest

from app.providers.registry import build_container
from app.providers.retrieval import RetrievalFilters
from app.services.retrieval import (
    RankedChunk,
    RetrievalService,
    asked_periods,
    confidence_of,
    fiscal_year_ends,
    fuse_hit_lists,
    missing_periods,
    prefer_stated_periods,
    rank_by_scores,
    stated_periods,
    with_fiscal_years,
)
from app.settings import RetrievalSection

from .fakes import FakeEmbedder, FakeReranker, FakeStore, hit, make_chunk, vector_for

CONFIG = RetrievalSection(prefetch_k=20, fusion="rrf", rerank_top_n=2, min_rerank_score=0.3, context_token_budget=3000)


def chunks(n: int):
    return [make_chunk(i, text=f"passage {i}") for i in range(n)]


# ------------------------------------------------------------------ pure helpers


def test_single_list_passes_through_with_limit():
    c = chunks(3)
    hits = [hit(x, score=1 / (i + 1)) for i, x in enumerate(c)]
    assert fuse_hit_lists([hits], limit=2) == hits[:2]


def test_rrf_merges_lists_and_keeps_best_dense_score():
    a, b, c = chunks(3)
    first = [hit(a, dense=0.5), hit(b, dense=0.4)]
    second = [hit(b, dense=0.9), hit(c, dense=0.7), hit(a, dense=None)]
    fused = fuse_hit_lists([first, second], limit=10, k=60)
    # b: 1/62 + 1/61 > a: 1/61 + 1/63 > c: 1/62
    assert [h.chunk.chunk_id for h in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert [h.dense_score for h in fused] == [0.9, 0.5, 0.7]
    assert len(fuse_hit_lists([first, second], limit=2)) == 2


def test_rank_by_scores_orders_and_keeps_search_rank():
    c = chunks(3)
    ranked = rank_by_scores([hit(x) for x in c], [0.2, 0.9, 0.2])
    assert [(r.chunk.chunk_index, r.search_rank) for r in ranked] == [(1, 2), (0, 1), (2, 3)]  # ties: search order
    with pytest.raises(ValueError):
        rank_by_scores([hit(c[0])], [0.1, 0.2])


def test_confidence_signals():
    a, b = chunks(2)
    ranked = [RankedChunk(a, 0.92, 0.03, 0.71, 2), RankedChunk(b, 0.40, 0.03, 0.66, 1)]
    conf = confidence_of(ranked, 0.3)
    assert conf.top_score == 0.92 and conf.gap == pytest.approx(0.52)
    assert conf.dense_similarity == 0.71 and conf.above_threshold is True
    single = confidence_of(ranked[1:], 0.5)
    assert single.gap == 0.40 and single.above_threshold is False
    assert confidence_of([], 0.3) is None


@pytest.mark.parametrize(
    ("text", "periods"),
    [
        ("What is Valmora's EBITDA margin for FY25?", {25}),
        ("revenue in fy24 vs FY 23", {23, 24}),
        ("FY2025 guidance", {25}),
        ("FY 2024-25 capex, FY24-25 and 2023-24", {24, 25}),
        ("वाल्मोरा का FY25 का लाभांश", {25}),
        ("वित्त वर्ष २०२४-२५ में", {25}),  # Devanagari digits
        ("Q1 FY25 revenue", {25}),
        ("zephyra investor deck q4fy24", {24}),
        ("What happened in 2024?", set()),  # a bare calendar year is not a fiscal period
        ("How many employees does Valmora have?", set()),
    ],
)
def test_periods_a_question_names(text, periods):
    assert asked_periods(text) == periods


@pytest.mark.parametrize(
    ("text", "periods"),
    [
        ("EBITDA margin was 21.0% in FY24, up from 19.8% in FY23", {23, 24}),
        ("The market capitalisation on 31 March 2024 was ₹ 22,640 crore", {24}),
        ("paid on or after 4 September 2024", {25}),
        ("31 मार्च 2024 को", {24}),
        ("founded in 1998", {98, 99}),  # an undated year: either fiscal year it can fall in
    ],
)
def test_periods_a_passage_states(text, periods):
    assert stated_periods(text) == periods


def test_gate_refuses_a_question_about_a_period_the_best_passage_does_not_state():
    """Near misses ("FY25" when the documents stop at FY24) score high with the reranker: the gate checks periods."""
    fy24 = make_chunk(0, text="The EBITDA margin was 21.0% in FY24, up from 19.8% in FY23.")
    outlook = make_chunk(1, text="For FY25 the Board approved capital expenditure of ₹ 1,100 crore.")
    ranked = [RankedChunk(fy24, 0.95, 0.03, 0.7, 1), RankedChunk(outlook, 0.6, 0.02, 0.6, 2)]

    asked_fy25 = confidence_of(ranked, 0.05, queries=["What is the EBITDA margin for FY25?", None])
    assert asked_fy25.top_score == 0.95 and asked_fy25.missing_periods == (25,)
    assert asked_fy25.above_threshold is False

    hindi = confidence_of(ranked, 0.05, queries=["FY25 का EBITDA मार्जिन?", "What is the EBITDA margin for FY25?"])
    assert hindi.above_threshold is False and hindi.missing_periods == (25,)
    for question in ("What was the EBITDA margin in FY24?", "What was the EBITDA margin?", "FY23 vs FY24 margin"):
        conf = confidence_of(ranked, 0.05, queries=[question])
        assert conf.above_threshold is True and conf.missing_periods == ()
    assert missing_periods(["FY24 and FY26?"], fy24.embed_text) == (26,)
    # a passage that names no year can't contradict the question
    assert missing_periods(["EBITDA margin in FY25?"], "EBITDA margin 18.2%") == ()


def test_the_document_label_counts_as_stated():
    """A chunk of "annual report fy24" covers FY24 even when its own text names no year."""
    chunk = make_chunk(0, text="Net debt declined to ₹ 831 crore.", document_label="valmora annual report fy24")
    conf = confidence_of([RankedChunk(chunk, 0.9, 0.03, 0.7, 1)], 0.05, queries=["Net debt in FY24?"])
    assert conf.above_threshold is True


# ------------------------------------------------------------------ the end of a fiscal year (last round, item 3)
#
# The final real run: "What was Valmora's net debt on 31 March 2024?" was answered ₹1,188 crore (FY23's, from p.16
# "Recap of FY23 … at 31 March 2023") 3 of 3 times; the answer is ₹831 crore (p.17). The date named no fiscal year, and
# every passage of "valmora annual report fy24" states FY24 through its document label.

LABEL = "valmora annual report fy24: Valmora Industries Limited - Annual Report 2023-24"
RECAP = make_chunk(
    0,
    heading_path=["Management discussion and analysis", "Recap of FY23, the comparative year"],
    text="In FY23, revenue was ₹ 6,482 crore. Net debt stood at ₹ 1,188 crore at 31 March 2023, or 0.93x EBITDA.",
    document_label=LABEL,
)
MDA = make_chunk(
    1,
    heading_path=["Management discussion and analysis: FY24 financial performance"],
    text="Net debt declined to ₹ 831 crore from ₹ 1,188 crore, and ROCE rose to 19.6% from 17.9%.",
    document_label=LABEL,
)
NET_DEBT = "What was Valmora's net debt on 31 March 2024?"


@pytest.mark.parametrize(
    "text",
    [
        "What was Valmora's net debt on 31 March 2024?",
        "What was Valmora's net debt on March 31, 2024?",
        "net debt as at 31st March, 2024",
        "net debt as at 31.03.2024",
        "net debt on 31/03/24",
        "net debt on 2024-03-31",
        "net debt at the end of March 2024",
        "वाल्मोरा का शुद्ध ऋण 31 मार्च 2024 को कितना था?",
        "३१ मार्च २०२४ को शुद्ध ऋण",
    ],
)
def test_the_last_day_of_march_names_the_fiscal_year_that_ends_on_it(text):
    assert fiscal_year_ends(text) == [24]
    assert asked_periods(text) == {24}
    assert with_fiscal_years(text) == f"{text} (FY24)"


def test_other_dates_and_years_name_no_fiscal_year_end():
    for text in ("What happened in 2024?", "net debt on 30 June 2024", "summary 2024", "Q4 FY24 results"):
        assert fiscal_year_ends(text) == []
    assert with_fiscal_years("net debt on 31 March 2024 (FY24)") == "net debt on 31 March 2024 (FY24)"
    assert asked_periods("2024-03-31") == {24}  # an ISO date is not the span "2024-03"


def test_a_passage_about_another_year_is_vetoed_whatever_its_documents_label_says():
    ranked = [RankedChunk(RECAP, 0.93, 0.03, 0.7, 1)]
    conf = confidence_of(ranked, 0.02, queries=[NET_DEBT, None])
    assert conf.missing_periods == (24,) and conf.above_threshold is False  # abstains rather than answer FY23's
    # with the label, it "stated" FY24: that is what let the recap of FY23 answer
    assert missing_periods([NET_DEBT], RECAP.embed_text) == ()


def test_the_passage_that_states_the_year_asked_for_goes_first():
    ranked = [RankedChunk(RECAP, 0.93, 0.03, 0.7, 1), RankedChunk(MDA, 0.81, 0.03, 0.7, 2)]
    ordered = prefer_stated_periods(ranked, [NET_DEBT, None], min_score=0.02)
    assert [r.chunk.chunk_index for r in ordered] == [1, 0]
    conf = confidence_of(ordered, 0.02, queries=[NET_DEBT, None])
    assert conf.above_threshold is True and conf.top_score == 0.81
    # one that is much less relevant is not promoted: the gate abstains instead
    weak = [RankedChunk(RECAP, 0.93, 0.03, 0.7, 1), RankedChunk(MDA, 0.2, 0.03, 0.7, 2)]
    assert prefer_stated_periods(weak, [NET_DEBT, None], min_score=0.02) == weak
    # a question that names no period, or whose best passage states it, keeps the reranker's order
    assert prefer_stated_periods(ranked, ["What was Valmora's net debt?"]) == ranked
    assert prefer_stated_periods(ranked, ["Net debt at 31 March 2023?"]) == ranked


def test_a_year_end_date_is_searched_and_reranked_with_its_fiscal_year():
    store = FakeStore(results=[[hit(RECAP), hit(MDA)], [hit(MDA)]])
    svc, embedder, reranker = service(store, scorer=lambda q, p: 0.93 if "Recap" in p else 0.81)
    hi = "वाल्मोरा का शुद्ध ऋण 31 मार्च 2024 को कितना था?"
    res = asyncio.run(svc.retrieve(hi, project_id="proj1", document_ids=["doc1"], query_en=NET_DEBT))
    assert embedder.calls == [[f"{hi} (FY24)", f"{NET_DEBT} (FY24)"]]
    assert reranker.calls[0][0] == f"{NET_DEBT} (FY24)"
    assert res.query == hi and res.rerank_query == NET_DEBT  # recorded as asked
    assert [r.chunk.chunk_index for r in res.chunks] == [1, 0]  # the FY24 paragraph first
    assert res.confidence.above_threshold is True


# ------------------------------------------------------------------ service


def service(store: FakeStore, scorer=lambda q, p: 0.0) -> tuple[RetrievalService, FakeEmbedder, FakeReranker]:
    embedder, reranker = FakeEmbedder(), FakeReranker(scorer)
    return RetrievalService(embedder, reranker, store, CONFIG), embedder, reranker


def test_retrieve_reranks_with_the_english_query():
    c = chunks(4)
    lists = [[hit(c[0]), hit(c[1]), hit(c[2])], [hit(c[3]), hit(c[1])]]
    store = FakeStore(results=lists)
    svc, embedder, reranker = service(store, scorer=lambda q, p: 0.95 if "passage 3" in p else 0.1)
    hi = "वित्त वर्ष 2024 में EBITDA मार्जिन कितना था?"
    en = "What was the EBITDA margin in FY24?"
    res = asyncio.run(svc.retrieve(hi, project_id="proj1", document_ids=["doc1"], query_en=en))

    assert embedder.calls == [[hi, en]]  # one forward pass for both queries
    assert [s[0] for s in store.searches] == [vector_for(hi), vector_for(en)]
    assert all(s[1] == RetrievalFilters("proj1", ("doc1",)) for s in store.searches)
    assert reranker.calls[0][0] == en  # the reranker scores the English query (§9.2)
    assert reranker.calls[0][1] == [h.chunk.embed_text for h in fuse_hit_lists(lists, limit=20)]
    assert res.rerank_query == en and res.query == hi
    assert res.candidate_count == 4
    assert [r.chunk.chunk_index for r in res.chunks][:1] == [3] and len(res.chunks) == 2  # rerank_top_n
    assert res.confidence.top_score == 0.95 and res.confidence.gap == pytest.approx(0.85)
    assert set(res.timings_ms) == {"embed", "search", "rerank", "total"}


def test_hindi_passages_are_reranked_with_the_users_own_words():
    """With an English query from the router, English passages are scored against it and Hindi passages against
    the question as asked: the cross-encoder compares best within one language (§9.2)."""
    en_chunk = make_chunk(0, text="EBITDA margin was 21.0% in FY24.")
    hi_chunk = make_chunk(1, text="योजना में प्रशिक्षण निःशुल्क है।", language="hi")
    store = FakeStore(results=[[hit(en_chunk), hit(hi_chunk)], [hit(hi_chunk)]])
    svc, _, reranker = service(store, scorer=lambda q, p: 0.9 if "योजना" in q and "योजना" in p else 0.2)
    hi = "क्या योजना में प्रशिक्षण मुफ़्त है?"
    en = "Is the training under the scheme free?"
    res = asyncio.run(svc.retrieve(hi, project_id="proj1", query_en=en))
    assert reranker.calls == [(en, [en_chunk.embed_text]), (hi, [hi_chunk.embed_text])]
    assert [r.chunk.chunk_index for r in res.chunks] == [1, 0] and res.confidence.top_score == 0.9
    assert res.rerank_query == en

    store = FakeStore(results=[[hit(hi_chunk), hit(en_chunk)]])  # no English query: one call, the question
    svc, _, reranker = service(store)
    asyncio.run(svc.retrieve(hi, project_id="proj1"))
    assert reranker.calls == [(hi, [hi_chunk.embed_text, en_chunk.embed_text])]


def test_retrieve_without_english_query_searches_once_and_reranks_the_query():
    c = chunks(2)
    store = FakeStore(results=[[hit(c[0]), hit(c[1])]])
    svc, embedder, reranker = service(store, scorer=lambda q, p: 0.5)
    res = asyncio.run(svc.retrieve("What was the EBITDA margin?", project_id="proj1"))
    assert embedder.calls == [["What was the EBITDA margin?"]]
    assert store.searches[0][1] == RetrievalFilters("proj1")
    assert reranker.calls[0][0] == "What was the EBITDA margin?"
    assert res.confidence.above_threshold is True


def test_identical_english_query_is_not_searched_twice():
    svc, embedder, _ = service(FakeStore(results=[[]]))
    asyncio.run(svc.retrieve("EBITDA margin FY24", project_id="p", query_en="ebitda margin fy24 "))
    assert embedder.calls == [["EBITDA margin FY24"]]


def test_nothing_found_means_no_confidence_and_no_rerank():
    svc, _, reranker = service(FakeStore(results=[[]]))
    res = asyncio.run(svc.retrieve("anything", project_id="p"))
    assert res.chunks == [] and res.confidence is None and res.candidate_count == 0
    assert reranker.calls == []


def test_empty_query_is_rejected():
    svc, _, _ = service(FakeStore())
    with pytest.raises(ValueError, match="empty query"):
        asyncio.run(svc.retrieve("  ", project_id="p"))


def test_service_builds_from_the_container(load_local):
    container = build_container(load_local())
    svc = RetrievalService.from_container(container)
    assert svc.config == container.settings.retrieval
    assert svc.embedder is container["embeddings"] and svc.store is container["vector_store"]
