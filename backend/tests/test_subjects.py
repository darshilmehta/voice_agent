"""Which documents a question names, and what retrieval does when only another document's passages come back."""

from __future__ import annotations

import asyncio

import pytest

from app.providers.ingestion import document_label, split_document_label
from app.providers.retrieval import RetrievalFilters
from app.services.retrieval import (
    RankedChunk,
    RetrievalService,
    SpeculativeRetrieval,
    confidence_of,
    prefer_named_documents,
)
from app.services.subjects import asked_names, documents_by_name, label_words, mentions, named_documents
from app.settings import RetrievalSection

from .fakes import FakeEmbedder, FakeReranker, FakeStore, hit, make_chunk

# The labels the phase-2 eval corpus gets (file name, then the title when it adds words).
LABELS = {
    "ar": "valmora annual report fy24: VALMORA INDUSTRIES LIMITED",
    "deck": "zephyra investor deck q4fy24: Zephyra Logistics Limited",
    "hindi": "suryodaya yojana soochna",
    "travel": "valmora travel expense policy",
    "health": "valmora group health policy scan: Group Health Insurance Policy: Schedule",
}
VALMORA = {"ar", "travel", "health"}


def named(*queries: str | None, labels=LABELS):
    n = named_documents(queries, labels)
    return None if n is None else (set(n.document_ids), n.names)


# ------------------------------------------------------------------ labels


def test_a_label_splits_into_file_name_and_title():
    label = document_label("valmora_annual_report_fy24.pdf", "Valmora Industries Limited - Annual Report")
    assert split_document_label(label) == ("valmora annual report fy24", "Valmora Industries Limited - Annual Report")
    assert split_document_label("valmora annual report fy24") == ("valmora annual report fy24", "")
    assert split_document_label("") == ("", "")


def test_label_words_are_latin_words_of_three_letters_or_more():
    assert label_words("zephyra investor deck q4fy24: Zephyra Logistics Limited") == {
        "zephyra",
        "investor",
        "deck",
        "logistics",
        "limited",
    }
    assert label_words("सूर्योदय योजना FY24 Q4") == frozenset()


def test_names_are_the_capitalised_words_of_the_questions():
    assert asked_names(["What is Valmora's order book in Contract Logistics?", None]) == {
        "valmora",
        "contract",
        "logistics",
    }
    assert asked_names(["how many warehouses does valmora run?"]) == frozenset()  # lower case: no name
    assert asked_names(["What was FY24 EBITDA for Q4FY24?"]) == {"ebitda"}  # no words with digits, no "FY"
    assert asked_names(["Who is the head of Human Resources at Zephyra Limited?"]) == {"human", "resources", "zephyra"}
    assert asked_names(["वाल्मोरा का FY25 का शुद्ध मुनाफा?", "What will Valmora's net profit be in FY25?"]) == {"valmora"}


# ------------------------------------------------------------------ naming documents


def test_a_company_name_names_its_documents():
    assert named("How many electric vehicles does Valmora run in last-mile delivery?") == (VALMORA, ("valmora",))
    assert named("What is the ESOP exercise price at Zephyra?") == ({"deck"}, ("zephyra",))
    assert named("Valmora ka fleet size kitna hai?", "How big is Valmora's fleet?") == (VALMORA, ("valmora",))
    assert named("ज़ेफ़िरा का कितना रासायनिक उत्पादन है?", "How much chemical production does Zephyra have?") == (
        {"deck"},
        ("zephyra",),
    )


def test_a_question_naming_both_companies_names_both_documents_sets():
    got = named("Compare Valmora's EBITDA margin with Zephyra's in FY24")
    assert got == (VALMORA | {"deck"}, ("valmora", "zephyra"))


def test_file_name_words_decide_before_title_words():
    """ "Logistics" is in Zephyra's title only: Valmora's file name word wins, so the question is about Valmora."""
    assert named("What is Valmora's order book in Contract Logistics?") == (VALMORA, ("valmora",))
    # no file name word matches: the title does
    assert named("What does Logistics report?") == ({"deck"}, ("logistics",))


def test_a_title_tells_apart_documents_whose_file_names_say_nothing():
    labels = {"a": "scan 0012: Zephyra Logistics Limited - Investor presentation", "b": "scan 0013: Valmora Industries"}
    assert named("What is Zephyra's dividend?", labels=labels) == ({"a"}, ("zephyra",))
    assert named("What is Valmora's dividend?", labels=labels) == ({"b"}, ("valmora",))


@pytest.mark.parametrize(
    "question",
    [
        "What was the EBITDA margin in FY24?",  # no name
        "what is valmora's revenue?",  # lower case: not recognised as a name
        "What was Wall Mora's net debt at the end of March 2024?",  # a name no document has
        "What is the capital of France?",
        "What is the FY24 revenue?",  # "FY24" is a year, not a company
        "Tell me about the Company's risks",  # corporate words are no names
    ],
)
def test_questions_that_name_no_document_name_none(question):
    assert named(question) is None


def test_a_name_in_every_label_names_nothing():
    labels = {"a": "valmora annual report", "b": "valmora travel policy"}
    assert named("What is Valmora's revenue?", labels=labels) is None
    # a capitalised word that is in some labels only does name them: the user is naming the document
    assert named("What does the Policy say?", labels=labels) == ({"b"}, ("policy",))


def test_one_document_or_none_has_nothing_to_tell_apart():
    assert named("What is Valmora's revenue?", labels={"a": "valmora annual report"}) is None
    assert named("What is Valmora's revenue?", labels={}) is None
    # documents indexed before labels existed say nothing
    assert named("What is Valmora's revenue?", labels={"a": "", "b": "valmora report"}) is None


def test_mentions_matches_whole_words_case_insensitively():
    assert mentions("Valmora's EBITDA margin", ["valmora"]) and mentions("VALMORA Industries", ["valmora"])
    assert not mentions("Valmoras and others", ["valmora"]) and not mentions("anything", [])
    assert mentions("peer Zephyra and Valmora", ["zephyra", "valmora"])


# ------------------------------------------------------------------ preferring the named documents


def rc(document: str, score: float, text: str = "fleet", index: int = 0, label: str = "") -> RankedChunk:
    chunk = make_chunk(index, document_id=document, text=text, document_label=label or LABELS.get(document, ""))
    return RankedChunk(chunk, score, 0.03, 0.7, index + 1)


VALMORA_NAMED = named_documents(["Valmora's EVs?"], LABELS)
assert VALMORA_NAMED is not None and set(VALMORA_NAMED.document_ids) == VALMORA


def test_the_best_passage_of_a_named_document_changes_nothing():
    ranked = [rc("ar", 0.9, index=0), rc("deck", 0.5, index=1)]
    assert prefer_named_documents(ranked, VALMORA_NAMED) == (ranked, ())
    assert prefer_named_documents(ranked, None) == (ranked, ())
    assert prefer_named_documents([], VALMORA_NAMED) == ([], ())


def test_another_documents_best_passage_gives_way_to_the_named_documents_passages():
    ranked = [
        rc("deck", 0.61, "1,150 electric vehicles", 0),
        rc("ar", 0.14, "fleet highlights", 1),
        rc("deck", 0.10, "more fleet", 2),
        rc("travel", 0.06, "hotel limits", 3),
    ]
    kept, missing = prefer_named_documents(ranked, VALMORA_NAMED)
    assert [(r.chunk.document_id, r.rerank_score) for r in kept] == [("ar", 0.14), ("travel", 0.06)]
    assert missing == ()


def test_a_passage_that_names_the_company_counts_as_about_it():
    ranked = [rc("deck", 0.6, "Valmora runs 1,150 EVs, peer data", 0), rc("deck", 0.5, "x", 1)]
    assert prefer_named_documents(ranked, VALMORA_NAMED) == (ranked, ())  # best passage says "Valmora"
    ranked = [rc("deck", 0.6, "nothing", 0), rc("deck", 0.5, "A Valmora fleet comparison", 1)]
    kept, _ = prefer_named_documents(ranked, VALMORA_NAMED)
    assert [r.chunk.chunk_index for r in kept] == [1]


def test_nothing_about_the_named_company_means_not_covered():
    ranked = [rc("deck", 0.61, "1,150 EVs", 0), rc("deck", 0.2, "more", 1), rc("hindi", 0.1, "योजना", 2)]
    kept, missing = prefer_named_documents(ranked, VALMORA_NAMED)
    assert kept == ranked and missing == ("valmora",)
    conf = confidence_of(ranked, 0.02, missing_subjects=missing)
    assert conf.top_score == 0.61 and conf.above_threshold is False and conf.missing_subjects == ("valmora",)
    assert confidence_of(ranked, 0.02).above_threshold is True  # the field is additive: absent, nothing changes


# ------------------------------------------------------------------ the service

CONFIG = RetrievalSection(prefetch_k=20, fusion="rrf", rerank_top_n=3, min_rerank_score=0.02, context_token_budget=3000)
PROJECT = "proj1"


def fleet(document: str, index: int, text: str = "fleet of vehicles"):
    return make_chunk(index, document_id=document, text=text, document_label=LABELS[document])


def build(hits_lists, scores: dict[str, float], *, labels=LABELS):
    store = FakeStore(results=hits_lists)
    store.labels = dict(labels)
    reranker = FakeReranker(lambda q, p: next((v for k, v in scores.items() if k in p), 0.0))
    return RetrievalService(FakeEmbedder(), reranker, store, CONFIG), store


def retrieve(svc, question, *, ids=None, query_en=None):
    ids = list(LABELS) if ids is None else ids
    return asyncio.run(svc.retrieve(question, project_id=PROJECT, document_ids=ids, query_en=query_en))


def test_a_question_about_valmora_is_answered_from_valmoras_documents():
    """The phase-2 eval's q173: "How many electric vehicles does Valmora run?" found Zephyra's fleet at 0.61."""
    candidates = [hit(fleet("deck", 0, "1,150 electric vehicles")), hit(fleet("ar", 1, "fleet highlights"))]
    svc, _ = build([candidates], {"1,150": 0.61, "highlights": 0.14})
    res = retrieve(svc, "How many electric vehicles does Valmora run in last-mile delivery?")
    assert [r.chunk.document_id for r in res.chunks] == ["ar"]  # the other company's figure is not in the context
    assert res.confidence.top_score == 0.14 and res.confidence.above_threshold is True
    assert res.confidence.missing_subjects == ()


def test_nothing_from_valmora_found_means_not_covered_whatever_the_score():
    candidates = [hit(fleet("deck", 0, "1,150 electric vehicles")), hit(fleet("deck", 1, "fleet plan"))]
    svc, _ = build([candidates], {"1,150": 0.61, "plan": 0.3})
    res = retrieve(svc, "How many electric vehicles does Valmora run in last-mile delivery?")
    assert res.confidence.top_score == 0.61 and res.confidence.above_threshold is False
    assert res.confidence.missing_subjects == ("valmora",)
    assert [r.chunk.document_id for r in res.chunks] == ["deck", "deck"]  # as ranked: for a look at what came back


def test_the_english_query_names_the_company_of_a_hindi_question():
    candidates = [hit(fleet("deck", 0, "1,150 electric vehicles"))]
    svc, _ = build([candidates, candidates], {"1,150": 0.6})
    res = retrieve(svc, "वाल्मोरा के पास कितनी इलेक्ट्रिक गाड़ियाँ हैं?", query_en="How many EVs does Valmora have?")
    assert res.confidence.above_threshold is False and res.confidence.missing_subjects == ("valmora",)


def test_questions_that_name_nobody_and_single_document_chats_are_untouched():
    candidates = [hit(fleet("deck", 0, "1,150 electric vehicles")), hit(fleet("ar", 1, "fleet highlights"))]
    svc, store = build([candidates], {"1,150": 0.61, "highlights": 0.14})
    res = retrieve(svc, "How many electric vehicles are there in last-mile delivery?")
    assert [r.chunk.document_id for r in res.chunks] == ["deck", "ar"] and res.confidence.above_threshold is True

    svc, store = build([candidates], {"1,150": 0.61, "highlights": 0.14})
    res = retrieve(svc, "How many electric vehicles does Valmora run?", ids=["deck"])  # one document: nothing to choose
    assert [r.chunk.document_id for r in res.chunks] == ["deck", "ar"] and res.confidence.above_threshold is True
    assert store.label_calls == []

    svc, store = build([candidates], {"1,150": 0.61, "highlights": 0.14})
    res = asyncio.run(svc.retrieve("How many EVs does Valmora run?", project_id=PROJECT))  # every document: no ids
    assert res.confidence.above_threshold is True and store.label_calls == []


def test_labels_are_fetched_beside_the_search_and_remembered():
    candidates = [hit(fleet("ar", 0))]
    svc, store = build([candidates, candidates, candidates], {"fleet": 0.5})
    retrieve(svc, "How many vehicles does Valmora run?")
    retrieve(svc, "How many vehicles does Valmora run?")
    assert len(store.label_calls) == 1 and set(store.label_calls[0]) == set(LABELS)
    svc._labels = {d: (when - 10_000, label) for d, (when, label) in svc._labels.items()}  # past their time
    retrieve(svc, "How many vehicles does Valmora run?")
    assert len(store.label_calls) == 2


def test_a_store_without_labels_or_that_fails_turns_the_check_off():
    candidates = [hit(fleet("deck", 0, "1,150 electric vehicles"))]
    svc, store = build([candidates], {"1,150": 0.61})
    store.fail_labels = RuntimeError("qdrant is down")
    res = retrieve(svc, "How many electric vehicles does Valmora run?")
    assert res.confidence.above_threshold is True and res.confidence.missing_subjects == ()  # as before this check

    svc, store = build([candidates], {"1,150": 0.61}, labels={})
    res = retrieve(svc, "How many electric vehicles does Valmora run?")  # chunks carry labels, the store says none
    assert res.confidence.above_threshold is True


def test_a_document_still_being_indexed_is_left_out_and_asked_again():
    candidates = [hit(fleet("ar", 0))]
    labels = {d: label for d, label in LABELS.items() if d != "deck"}
    svc, store = build([candidates, candidates], {"fleet": 0.5}, labels=labels)
    retrieve(svc, "How many vehicles does Zephyra run?")  # no deck label yet: Zephyra names nothing
    store.labels["deck"] = LABELS["deck"]
    res = retrieve(svc, "How many vehicles does Zephyra run?")
    assert res.confidence.above_threshold is False and res.confidence.missing_subjects == ("zephyra",)


def test_speculative_retrieval_checks_the_subject_too():
    candidates = [hit(fleet("deck", 0, "1,150 electric vehicles"))]
    svc, _ = build([candidates, candidates], {"1,150": 0.61})
    question = "How many electric vehicles does Valmora run?"

    async def run():
        spec = SpeculativeRetrieval(svc, question, project_id=PROJECT, document_ids=list(LABELS))
        used, outcome = await spec.result_for(question, None)
        return used, outcome

    result, outcome = asyncio.run(run())
    assert outcome == "used" and result.confidence.above_threshold is False
    assert RetrievalFilters(PROJECT, tuple(LABELS)) == svc.store.searches[0][1]


def test_the_named_documents_by_company():
    n = named_documents(("Compare Valmora and Zephyra revenue",), LABELS)
    assert n is not None
    assert documents_by_name(n, LABELS) == {"valmora": frozenset(VALMORA), "zephyra": frozenset({"deck"})}
    one = named_documents(("Show Zephyra's segments",), LABELS)
    assert one is not None and documents_by_name(one, LABELS) == {"zephyra": frozenset({"deck"})}
