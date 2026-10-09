"""The chat's vocabulary for the speech recognizer (services/voice/vocabulary.py, DESIGN §9.3): names and terms from
the documents, the prompts, the near-miss correction, the Devanagari names, and the per-chat cache."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.providers.llm import LLMError
from app.services.voice import vocabulary as vocab
from app.services.voice.vocabulary import (
    PROMPT_TOKENS,
    DocumentWords,
    SpeechHints,
    SpeechVocabulary,
    Vocabulary,
    build_vocabulary,
    document_spelling,
    edit_distance,
    estimate_tokens,
    file_words,
    title_phrases,
    transliterate,
)

# Whisper's tokenizer, for the words these tests use: a known word is one token, a name several.
TOKENS = {
    "valmora": 3, "zephyra": 4, "suryodaya": 4, "taloja": 3, "vamora": 3, "varmora": 3, "vimora": 3, "talaja": 3,
    "specialty": 2, "chemicals": 2, "engineered": 2, "plastics": 2, "contract": 1, "logistics": 2, "freight": 2,
    "services": 1, "digital": 1, "works": 1, "north": 1, "america": 1, "karthik": 3, "nambiar": 3, "remains": 1,
    "industries": 1, "hosur": 2, "polymers": 2, "yojana": 3, "fills": 1, "zephyr": 2,
}  # fmt: skip


def count(text: str) -> int:
    return sum(TOKENS.get(w.strip(".,").casefold(), 2) for w in text.split())


def table(first_column: list[str], *, heading: str = "", header: str = "Segment", more: list[str] = ()) -> dict:
    cells: list[dict[str, Any]] = [{"row": 0, "col": 0, "text": header, "column_header": True}]
    cells += [{"row": i + 1, "col": 0, "text": t, "column_header": False} for i, t in enumerate(first_column)]
    cells += [{"row": i + 1, "col": 1, "text": t, "column_header": False} for i, t in enumerate(more)]
    return {"heading_path": [heading] if heading else [], "cells": cells}


VALMORA = DocumentWords(
    "valmora_annual_report_fy24.pdf",
    "valmora annual report fy24: VALMORA INDUSTRIES LIMITED",
    [
        table(["Specialty Chemicals", "Engineered Plastics", "Digital Services 1", "Total"], more=["EBITDA margin"]),
        table(["Specialty Chemicals", "Engineered Plastics"], header="Segment", more=["EBITDA", "EBITDA"]),
        table(["Taloja Works", "Hosur Polymers", "North America"], header="Facility", more=["EBITDA"]),
        table(["Mr. Karthik Nambiar"], header="Director", more=["CIN"]),
    ],
)
ZEPHYRA = DocumentWords(
    "zephyra_investor_deck_q4fy24.pptx",
    "zephyra investor deck q4fy24: Zephyra Logistics Limited - Investor presentation, Q4 and FY24",
    [table(["Freight Services", "Contract Logistics", "Digital Services"], more=["EBITDA"])],
)
SURYODAYA = DocumentWords(
    "suryodaya_yojana_soochna.docx",
    "",
    [
        table(["कमलपुर", "हरिपुर", "कुल"], heading="सूर्योदय ग्रामीण कौशल एवं रोज़गार योजना 2024-25", header="जिला"),
        table(["सोलर पैनल तकनीशियन"], heading="सूर्योदय ग्रामीण कौशल एवं रोज़गार योजना 2024-25", header="पाठ्यक्रम"),
    ],
)
POLICY = DocumentWords("valmora_travel_expense_policy.docx", "valmora travel expense policy")


# ------------------------------------------------------------------ extraction


def test_title_phrases_drop_legal_suffixes_and_read_capitals_as_title_case():
    assert title_phrases("Valmora Industries Limited - Annual Report 2023-24") == [
        "Valmora Industries",
        "Annual Report",
    ]
    assert title_phrases("VALMORA INDUSTRIES LIMITED") == ["Valmora Industries"]
    assert title_phrases("Zephyra Logistics Limited - Investor presentation, Q4 and FY24") == [
        "Zephyra Logistics",
        "Investor",
    ]


def test_file_words_keep_what_the_document_is_about():
    assert file_words("valmora_annual_report_fy24.pdf") == ["valmora"]
    assert file_words("suryodaya_yojana_soochna.docx") == ["suryodaya", "yojana"]
    assert file_words("annual_report.txt") == []


def test_build_vocabulary_names_terms_acronyms_and_hindi():
    v = build_vocabulary([VALMORA, ZEPHYRA, SURYODAYA, POLICY], count)
    # names: the title phrases with a file name word, else the file name; "Valmora" (policy) is in "Valmora Industries"
    assert v.names == ("Valmora Industries", "Zephyra Logistics", "Suryodaya Yojana")
    assert v.keywords == ("Valmora", "Zephyra", "Suryodaya")
    # in more tables first; the documents take turns; phrases of known words only ("North America") last
    assert v.terms[:3] == ("Specialty Chemicals", "Freight Services", "Engineered Plastics")
    assert v.terms.index("Contract Logistics") < v.terms.index("Taloja Works")
    assert v.terms[-2:] == ("Digital Services", "North America")  # in two tables, then one
    assert "Total" not in v.terms and "Karthik Nambiar" in v.terms  # honorific dropped
    assert v.acronyms == ("EBITDA",)  # "CIN" is in one table only
    assert v.terms_hi[0] == "सूर्योदय ग्रामीण कौशल एवं रोज़गार योजना"  # in both tables' headings, numbers dropped
    assert "कुल" not in v.terms_hi and "कमलपुर" in v.terms_hi


def test_prompts_fit_the_budget_and_put_names_first():
    v = build_vocabulary([VALMORA, ZEPHYRA, SURYODAYA], count)
    prompts = v.prompts(count=count)
    assert prompts["en"].startswith("Valmora Industries, Zephyra Logistics, Suryodaya Yojana, EBITDA, ")
    assert prompts["en"].endswith(".")
    assert prompts["hi"].startswith("Valmora, Zephyra, Suryodaya, EBITDA के बारे में। सूर्योदय ग्रामीण")
    for prompt in prompts.values():
        assert estimate_tokens(prompt, count) <= PROMPT_TOKENS + 2


def test_hindi_prompt_puts_the_devanagari_names_first():
    v = Vocabulary(keywords=("Valmora",), keywords_hi=("वाल्मोरा",), acronyms=("EBITDA",))
    assert v.prompts(["hi"]) == {"hi": "वाल्मोरा, Valmora, EBITDA के बारे में।"}


def test_no_documents_no_prompt():
    v = build_vocabulary([DocumentWords("annual_report.txt")], count)
    assert v.empty and v.prompts() == {}


def test_a_long_prompt_item_is_skipped_for_shorter_ones():
    v = Vocabulary(names=("A" * 600, "Valmora"))
    assert v.prompts(["en"]) == {"en": "Valmora."}


def test_estimate_without_a_tokenizer():
    assert estimate_tokens("Valmora") == 3
    assert estimate_tokens("सूर्योदय") == 11


# ------------------------------------------------------------------ correcting near misses


def hints() -> SpeechHints:
    v = build_vocabulary([VALMORA, ZEPHYRA, SURYODAYA], count)
    return SpeechHints(v.prompts(count=count), v.lexicon(count), count)


def test_lexicon_holds_the_unfamiliar_words():
    lexicon = hints().lexicon
    assert {"Valmora", "Zephyra", "Suryodaya", "Taloja", "Karthik", "Nambiar"} <= set(lexicon)
    assert not {"Services", "Logistics", "Contract", "North"} & set(lexicon)


@pytest.mark.parametrize(
    ("heard", "fixed"),
    [
        ("What was Vamora's revenue in FY24?", "What was Valmora's revenue in FY24?"),
        ("What was Varmora's revenue?", "What was Valmora's revenue?"),
        ("Compare Vimora and Zephyra on revenue growth.", "Compare Valmora and Zephyra on revenue growth."),
        ("Tell me more about the Talaja Works plant.", "Tell me more about the Taloja Works plant."),
        ("How did Zephyr do?", "How did Zephyra do?"),
    ],
)
def test_correct_restores_near_misses(heard, fixed):
    assert hints().correct(heard) == fixed


@pytest.mark.parametrize(
    "text",
    [
        "What remains of the revenue?",  # a common word: one token, never changed
        "it fills the warehouses",  # lower case: Whisper didn't take it for a name
        "Valmora and Zephyra",  # already right
        "वाल्मुरा का रेवेन्यू",  # Devanagari is left alone
        "",
    ],
)
def test_correct_leaves_other_words_alone(text):
    assert hints().correct(text) == text


def test_correct_skips_a_tie():
    h = SpeechHints({}, ("Valmora", "Volmora"), count)
    assert h.correct("Vulmora") == "Vulmora"


def test_edit_distance_counts_transpositions_once():
    assert edit_distance("valmora", "vamora") == 1
    assert edit_distance("valmora", "vlamora") == 1
    assert edit_distance("taloja", "tallaja") == 2


# ------------------------------------------------------------------ names in Devanagari


class JsonLLM:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[list[str]] = []

    async def generate_json(self, messages, schema, *, max_tokens=None, **_: Any):
        self.calls.append(messages[-1].content.split("\n"))
        if isinstance(self.answer, Exception):
            raise self.answer
        return schema.model_validate(self.answer)


@pytest.fixture(autouse=True)
def _fresh_transliterations(monkeypatch):
    monkeypatch.setattr(vocab, "_TRANSLITERATIONS", {})


async def test_transliterate_keeps_devanagari_answers_and_remembers_them():
    llm = JsonLLM({"names": ["वाल्मोरा", "Zephyra", "सूर्योदय"]})
    assert await transliterate(llm, ["Valmora", "Zephyra", "Suryodaya"]) == {  # type: ignore[arg-type]
        "Valmora": "वाल्मोरा",
        "Suryodaya": "सूर्योदय",
    }
    await transliterate(llm, ["Valmora"])  # type: ignore[arg-type]
    assert llm.calls == [["Valmora", "Zephyra", "Suryodaya"]]


async def test_transliterate_failure_gives_nothing():
    assert await transliterate(JsonLLM(LLMError("down")), ["Valmora"]) == {}  # type: ignore[arg-type]


def test_document_spelling_prefers_the_documents_own():
    terms = ["सूर्योदय ग्रामीण कौशल एवं रोज़गार योजना", "कमलपुर"]
    assert document_spelling("सुर्योदय", terms) == "सूर्योदय"
    assert document_spelling("वाल्मोरा", terms) == "वाल्मोरा"


# ------------------------------------------------------------------ per chat, cached


class Docs:
    """The two DocumentService calls the vocabulary makes."""

    def __init__(self) -> None:
        self.ready = {"doc_v": "valmora_annual_report_fy24.pdf"}
        self.tables_of = {"doc_v": VALMORA.tables, "doc_z": ZEPHYRA.tables}
        self.queries = 0
        self.table_calls: list[str] = []

    async def ready_documents(self, project_id, scope=None):
        self.queries += 1
        return dict(self.ready)

    async def tables(self, document_id):
        self.table_calls.append(document_id)

        class T:
            def __init__(self, t):
                self.t = t

            def model_dump(self, include):
                return {k: self.t[k] for k in include}

        return [T(t) for t in self.tables_of.get(document_id, [])]


class Chat:
    id, project_id, document_scope = "cht_1", "prj_1", None


async def test_hints_are_cached_and_rebuilt_when_the_documents_change(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(vocab.time, "monotonic", lambda: now[0])
    docs = Docs()
    sv = SpeechVocabulary(docs, None, count, JsonLLM({"names": ["वाल्मोरा"]}))  # type: ignore[arg-type]
    first = await sv.hints(Chat())  # type: ignore[arg-type]
    assert first.prompts["en"].startswith("Valmora, EBITDA, Specialty Chemicals")
    assert first.prompts["hi"].startswith("वाल्मोरा, Valmora, EBITDA के बारे में।")
    assert sv.cached("cht_1") is first
    assert await sv.hints(Chat()) is first and docs.queries == 1  # type: ignore[arg-type]
    now[0] += vocab.REFRESH_S + 1
    assert await sv.hints(Chat()) is first and docs.queries == 2  # type: ignore[arg-type]  # same documents
    docs.ready["doc_z"] = "zephyra_investor_deck_q4fy24.pptx"
    now[0] += vocab.REFRESH_S + 1
    second = await sv.hints(Chat())  # type: ignore[arg-type]
    assert "Zephyra" in second.prompts["en"] and docs.table_calls == ["doc_v", "doc_v", "doc_z"]


async def test_concurrent_requests_build_once():
    docs = Docs()
    sv = SpeechVocabulary(docs, None, count)  # type: ignore[arg-type]
    a, b = await asyncio.gather(sv.hints(Chat()), sv.hints(Chat()))  # type: ignore[arg-type]
    assert a is b and docs.table_calls == ["doc_v"]
