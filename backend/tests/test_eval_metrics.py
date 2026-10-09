"""Retrieval-eval metrics: hits, recall@k, MRR, citation accuracy, the abstention matrix and the threshold sweep."""

from __future__ import annotations

import pytest

from app.evals import metrics as m
from app.evals.metrics import GateSample, Passage, Target
from app.evals.textnorm import contains_all, normalize, squash


def P(doc: str, start: int | None, end: int | None = None, text: str = "", score: float | None = None) -> Passage:
    return Passage(doc, start, start if end is None else end, text, f"{doc}:{start}", score)


def T(doc: str, *pages: int, evidence: tuple[tuple[str, ...], ...] = (), approximate: bool = False) -> Target:
    return Target(doc, tuple(pages), evidence, approximate)


# ------------------------------------------------------------------ text matching


def test_normalize_folds_layout_not_numbers():
    assert normalize("EBITDA  margin\n 21.0%") == "ebitda margin 21.0%"
    assert normalize(f"₹ 4,210 crore {chr(0x2013)} FY24") == "₹ 4,210 crore - fy24"  # en dash folds to a hyphen
    assert normalize("मार्जिन १८.२ प्रतिशत") == "मार्जिन 18.2 प्रतिशत"  # Devanagari digits read as digits
    assert contains_all("| EBITDA margin |   21.0% |", ["EBITDA margin", "21.0%"])
    assert not contains_all("EBITDA margin 21.5%", ["21.0%"])


def test_contains_all_needs_something_to_look_for():
    assert not contains_all("anything", [])
    assert not contains_all("anything", [""])
    assert not contains_all("anything", ["  "])


def test_approximate_matching_ignores_spaces_and_punctuation_for_ocr():
    ocr = "Policy number GHI 2024 00418377"
    assert not contains_all(ocr, ["GHI/2024/00418377"])
    assert contains_all(ocr, ["GHI/2024/00418377"], approximate=True)
    assert squash("A-b c.") == "abc"


# ------------------------------------------------------------------ hits


def test_page_hit_uses_the_passage_page_range():
    t = T("a.pdf", 4)
    assert m.page_hit(P("a.pdf", 4), t)
    assert m.page_hit(P("a.pdf", 3, 5), t)  # a chunk spanning the page counts
    assert not m.page_hit(P("a.pdf", 5, 6), t)
    assert not m.page_hit(P("b.pdf", 4), t)  # right page, wrong document
    assert not m.page_hit(P("a.pdf", None), t)  # a page-less passage can't satisfy a page target


def test_a_target_lists_alternative_pages():
    t = T("a.pdf", 3, 17, 20)
    assert m.page_hit(P("a.pdf", 17), t)
    assert m.page_hit(P("a.pdf", 19, 20), t)
    assert not m.page_hit(P("a.pdf", 18), t)


def test_exact_page_hit_rejects_chunks_that_reach_beyond_the_expected_pages():
    t = T("a.pdf", 4)
    assert m.exact_page_hit(P("a.pdf", 4), t)
    assert not m.exact_page_hit(P("a.pdf", 3, 4), t)
    assert m.exact_page_hit(P("a.pdf", 3, 4), T("a.pdf", 3, 4))


def test_a_page_less_target_is_matched_on_evidence_text():
    t = T("policy.docx", evidence=(("4.2.1", "economy class"),))
    assert m.page_hit(P("policy.docx", None, text="4.2.1 Employees travel economy class."), t)
    assert not m.page_hit(P("policy.docx", None, text="4.2.2 Business class on long flights."), t)
    assert not m.page_hit(P("other.docx", None, text="4.2.1 economy class"), t)


def test_fact_hit_needs_the_text_not_just_the_page():
    t = T("a.pdf", 4, evidence=(("EBITDA margin", "21.0%"),))
    assert m.fact_hit(P("a.pdf", 4, text="EBITDA margin 21.0% in FY24"), t)
    assert not m.fact_hit(P("a.pdf", 4, text="Revenue grew 13.6%"), t)  # the right page, the wrong chunk
    assert m.page_hit(P("a.pdf", 4, text="Revenue grew 13.6%"), t)


def test_fact_hit_takes_any_alternative_and_ocr_is_loose():
    t = T("s.pdf", 1, evidence=(("GHI/2024/00418377",), ("policy number",)), approximate=True)
    assert m.fact_hit(P("s.pdf", 1, text="Policy no GHI 2024 00418377"), t)
    assert not m.fact_hit(P("s.pdf", 1, text="nothing relevant"), t)


# ------------------------------------------------------------------ ranking


def test_recall_and_reciprocal_rank_of_a_single_target():
    passages = [P("a.pdf", 1), P("a.pdf", 2), P("a.pdf", 3)]
    s = m.score_ranking(passages, [T("a.pdf", 2)])
    assert s.first_rank == 2 and s.reciprocal_rank == 0.5
    assert [s.recall(k) for k in (1, 3, 5)] == [0.0, 1.0, 1.0]
    assert [s.success(k) for k in (1, 3, 5)] == [False, True, True]
    assert s.covered_all == 1 and s.all_rank == 2


def test_a_miss_scores_zero_everywhere():
    s = m.score_ranking([P("a.pdf", 1), P("b.pdf", 2)], [T("a.pdf", 9)])
    assert s.first_rank is None and s.reciprocal_rank == 0.0 and s.all_rank is None
    assert s.recall(5) == 0.0 and s.covered_all == 0


def test_multi_hop_recall_is_partial_until_every_page_is_covered():
    passages = [P("a.pdf", 14), P("a.pdf", 1), P("a.pdf", 3), P("a.pdf", 15)]
    s = m.score_ranking(passages, [T("a.pdf", 14), T("a.pdf", 15)])
    assert s.recall(1) == 0.5 and not s.success(3)
    assert s.recall(3) == 0.5
    assert s.recall(5) == 1.0 and s.success(5)
    assert s.first_rank == 1 and s.all_rank == 4  # MRR uses the first hit, all_rank the last needed


def test_cross_document_questions_need_each_document():
    passages = [P("valmora.pdf", 3), P("zephyra.pptx", 3)]
    s = m.score_ranking(passages, [T("valmora.pdf", 3), T("zephyra.pptx", 3)])
    assert s.recall(1) == 0.5 and s.recall(3) == 1.0


def test_score_ranking_refuses_questions_without_targets():
    with pytest.raises(ValueError, match="at least one target"):
        m.score_ranking([P("a.pdf", 1)], [])


def test_summaries_are_macro_averages():
    hit1 = m.score_ranking([P("a.pdf", 1), P("a.pdf", 2)], [T("a.pdf", 1)])  # rank 1
    hit2 = m.score_ranking([P("a.pdf", 1), P("a.pdf", 2)], [T("a.pdf", 2)])  # rank 2
    miss = m.score_ranking([P("a.pdf", 1), P("a.pdf", 2)], [T("a.pdf", 8)])
    s = m.summarize_ranking([hit1, hit2, miss])
    assert s["n"] == 3
    assert s["recall@1"] == pytest.approx(1 / 3)
    assert s["recall@3"] == pytest.approx(2 / 3)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert s["success@all"] == pytest.approx(2 / 3)


def test_summary_of_nothing_is_none_not_zero():
    s = m.summarize_ranking([])
    assert s["n"] == 0 and s["recall@1"] is None and s["mrr"] is None


def test_top1_citation_lenient_and_exact():
    t = [T("a.pdf", 4)]
    assert m.top1_citation([P("a.pdf", 4)], t) == {"lenient": True, "exact": True}
    assert m.top1_citation([P("a.pdf", 3, 4)], t) == {"lenient": True, "exact": False}
    assert m.top1_citation([P("a.pdf", 7)], t) == {"lenient": False, "exact": False}
    assert m.top1_citation([], t) == {"lenient": False, "exact": False}


def test_citation_scores_for_an_answers_cited_passages():
    targets = [T("a.pdf", 1), T("a.pdf", 9)]
    s = m.citation_scores([P("a.pdf", 1), P("a.pdf", 5)], targets)
    assert s["any_correct"] and s["precision"] == 0.5 and s["recall"] == 0.5 and s["cited"] == 2
    none = m.citation_scores([], targets)
    assert not none["any_correct"] and none["precision"] is None and none["recall"] == 0.0


# ------------------------------------------------------------------ distributions


def test_percentile_interpolates():
    assert m.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert m.percentile([5], 0.9) == 5
    assert m.percentile([0, 10], 0.25) == 2.5
    with pytest.raises(ValueError):
        m.percentile([], 0.5)


def test_describe_and_latency_stats():
    d = m.describe([0.0, 0.1, 0.2, 0.3, 1.0])
    assert d["n"] == 5 and d["min"] == 0.0 and d["max"] == 1.0 and d["median"] == 0.2
    assert d["mean"] == pytest.approx(0.32)
    assert m.describe([])["median"] is None
    lat = m.latency_stats([10.0, 20.0, 30.0, 40.0])
    assert lat["p50"] == 25.0 and lat["max"] == 40.0 and lat["mean"] == 25.0
    assert m.latency_stats([])["p95"] is None


def test_histogram_buckets_and_clamping():
    edges = (0.0, 0.1, 0.5, 1.0)
    assert m.histogram([0.0, 0.05, 0.1, 0.49, 0.5, 0.99, 1.0, 7.0], edges) == [2, 2, 4]  # 1.0 and 7.0 clamp to the last


# ------------------------------------------------------------------ abstention


def G(score: float | None, answerable: bool, lang: str = "en", evidence: bool = False) -> GateSample:
    return GateSample(score, None if score is None else score / 2, 0.5, answerable, lang, evidence)


SAMPLES = [
    G(0.9, True, evidence=True),
    G(0.6, True, evidence=True),
    G(0.2, True),  # an answerable question the gate would refuse at 0.3
    G(0.4, False),  # an unanswerable question the gate would answer at 0.3
    G(0.05, False),
    G(None, False),  # nothing retrieved
]


def test_a_vetoed_question_is_refused_at_any_threshold():
    """The gate's period check refuses a near miss whatever its score; ``gate=False`` shows the score gate alone."""
    near_miss = GateSample(0.95, 0.5, 0.6, False, "en", veto=True, subtype="near_miss_year")
    answerable = GateSample(0.9, 0.4, 0.7, True, "en")
    assert not m.answers(near_miss, 0.05) and m.answers(near_miss, 0.05, gate=False)
    c = m.confusion([near_miss, answerable], 0.05)
    assert (c["answered_unanswerable"], c["answered_answerable"]) == (0, 1)
    assert m.confusion([near_miss, answerable], 0.05, gate=False)["answered_unanswerable"] == 1
    assert all(p["answered_unanswerable"] == 0 for p in m.sweep([near_miss, answerable]))
    assert m.hallucination_by_subtype([near_miss, answerable], 0.05) == {
        "near_miss_year": {"n": 1, "answered": 0, "hallucination_risk": 0.0}
    }
    assert m.hallucination_by_subtype([near_miss], 0.05, gate=False)["near_miss_year"]["hallucination_risk"] == 1.0


def test_confusion_matrix_at_a_threshold():
    c = m.confusion(SAMPLES, 0.3)
    assert (c["answered_answerable"], c["abstained_answerable"]) == (2, 1)
    assert (c["answered_unanswerable"], c["abstained_unanswerable"]) == (1, 2)
    assert c["answer_precision"] == pytest.approx(2 / 3)
    assert c["answer_recall"] == pytest.approx(2 / 3)
    assert c["answer_f1"] == pytest.approx(2 / 3)
    assert c["hallucination_risk"] == pytest.approx(1 / 3)
    assert c["false_abstain_rate"] == pytest.approx(1 / 3)
    assert c["abstain_recall"] == pytest.approx(2 / 3)
    assert c["accuracy"] == pytest.approx(4 / 6)
    assert c["answered_with_evidence"] == 2


def test_nothing_retrieved_abstains_even_at_threshold_zero():
    c = m.confusion([G(None, False), G(0.0, True)], 0.0)
    assert c["abstained_unanswerable"] == 1 and c["answered_answerable"] == 1


def test_gate_answers_at_exactly_the_threshold():
    assert m.answers_at(G(0.3, True), 0.3)
    assert not m.answers_at(G(0.2999, True), 0.3)


def test_sweep_recall_never_rises_with_the_threshold():
    points = m.sweep(SAMPLES, (0.0, 0.1, 0.3, 0.5, 0.95))
    recalls = [p["answer_recall"] for p in points]
    assert recalls == sorted(recalls, reverse=True)
    assert points[0]["answered_unanswerable"] == 2  # at 0 everything with a score is answered (not the None one)
    assert points[-1]["answered_answerable"] == 0
    assert points[-1]["answer_precision"] is None  # nothing answered: precision undefined, not 0


def test_best_thresholds_search_the_observed_scores():
    best = m.best_thresholds(SAMPLES)
    assert set(best) == {"max_f1", "recall_target"}
    # answering from 0.2 up: 3 of 3 answerable, 1 of 3 unanswerable (0.4) -> precision 3/4, recall 1, F1 6/7
    assert best["max_f1"]["threshold"] == pytest.approx(0.2)
    assert best["max_f1"]["answer_f1"] == pytest.approx(6 / 7)
    assert best["max_f1"]["hallucination_risk"] == pytest.approx(1 / 3)
    assert best["recall_target"]["threshold"] == pytest.approx(0.2)  # the last cut that still answers every answerable


def test_the_recall_target_can_pick_a_lower_threshold_than_max_f1():
    samples = [G(0.9, True), G(0.8, True), G(0.7, True), G(0.05, True), G(0.6, False), G(0.04, False)]
    best = m.best_thresholds(samples, recall_target=1.0)
    # keeping all four answerable questions means answering from their lowest score, 0.05, which also lets
    # the unanswerable 0.6 through; the best-F1 cut is higher
    assert best["recall_target"]["threshold"] == pytest.approx(0.05)
    assert best["recall_target"]["answer_recall"] == 1.0
    assert best["max_f1"]["threshold"] >= best["recall_target"]["threshold"]


def test_best_thresholds_need_both_classes():
    assert m.best_thresholds([G(0.5, True), G(0.7, True)]) == {}
    assert m.best_thresholds([G(0.5, False)]) == {}


def test_auc_measures_separation():
    assert m.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert m.auc([0.1], [0.9]) == 0.0
    assert m.auc([0.5, 0.5], [0.5]) == 0.5
    assert m.auc([0.9, 0.2], [0.5]) == 0.5
    assert m.auc([], [0.5]) is None


# ------------------------------------------------------------------ answers


def test_answer_contains_normalises_numbers_and_accepts_alternatives():
    assert m.answer_contains_all("Revenue was ₹ 4210 crore.", ["4,210"])  # thousands separators don't matter
    assert m.answer_contains_all("EBITDA margin was 21.0 %", ["21.0%"])
    assert m.answer_contains_all("The CIN is l24119gj1994plc023871.", ["L24119GJ1994PLC023871"])
    assert m.answer_contains_all("कंपनी का मार्जिन १८.२% था", ["18.2%"])
    assert m.answer_contains_all("It is 1,18,426.", ["1,18,426|118,426"])
    assert m.answer_contains_all("It is 118426.", ["1,18,426|118,426"])
    assert not m.answer_contains_all("EBITDA margin was 21.5%", ["21.0%"])
    assert not m.answer_contains_all("anything", [])
    assert not m.answer_contains_all("21.0%", ["21.0%", "FY23"])  # every needle is required
