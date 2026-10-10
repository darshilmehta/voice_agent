"""What the agent says about the conversation and its documents (quality round): the resume line names the topic as
a person would (UX1), and an answer whose evidence comes from more than one document, or from one that isn't the
obvious one, says which (UX5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.domain.conversation import ConversationState
from app.services.chat_turns import ChatTurnService
from app.services.prompts import (
    answer_user_prompt,
    documents_note,
    resume_text,
    short_document_name,
    spoken_topic,
)
from app.services.sources import Source

from .fakes import make_chunk

# ------------------------------------------------------------------ UX1


@pytest.mark.parametrize(
    ("topic", "language", "said"),
    [
        ("net profit fiscal", "en", "the net profit"),
        ("ebitda margin", "en", "the EBITDA margin"),
        ("many employees company", "en", "the many employees"),
        ("fiscal year", "en", None),
        ("net profit fiscal", "hi", "शुद्ध लाभ"),
        ("ebitda margin", "hi", "EBITDA मार्जिन"),
        ("revenue growth", "hi", "राजस्व वृद्धि"),
        ("मुनाफा", "hi", "मुनाफा"),
        ("", "en", None),
    ],
)
def test_ux1_a_topic_label_as_it_is_said(topic, language, said):
    assert spoken_topic(topic, language) == said


def test_ux1_the_resume_line_never_speaks_a_raw_label():
    report = ["valmora_annual_report_fy24.pdf"]
    assert resume_text("en", "net profit fiscal", report) == (
        "Sure, back to the Valmora annual report. We were talking about the net profit. What would you like to know?"
    )
    assert resume_text("hi", "net profit fiscal", report) == (
        "ज़रूर, Valmora annual report पर वापस चलते हैं। हम शुद्ध लाभ के बारे में बात कर रहे थे। आप क्या जानना चाहेंगे?"
    )
    assert resume_text("en", "fiscal year", ["a.pdf", "b.pdf"]) == (
        "Sure, back to your documents. What would you like to know?"  # nothing worth saying is left of the label
    )


READY = {
    "d_ar": "valmora_annual_report_fy24.pdf",
    "d_deck": "zephyra_investor_deck_q4fy24.pptx",
    "d_travel": "valmora_travel_expense_policy.docx",
    "d_notice": "suryodaya_yojana_soochna.docx",
    "d_health": "valmora_group_health_policy_scan.pdf",
}


@pytest.mark.parametrize(
    ("said", "named"),
    [
        ("Let's go back to the annual report.", "d_ar"),
        ("back to the investor deck", "d_deck"),
        ("let's go back to the travel policy", "d_travel"),
        ("Let's go back to the report.", None),  # "report" alone names no document
        ("back to the policy", None),  # two policies
        ("let's go back to the document", None),
    ],
)
def test_final_check_the_document_a_resume_names(said, named):
    from app.services.prompts import named_document

    assert named_document(said, READY) == named


def test_final_check_back_to_the_annual_report_after_another_document_names_that_report():
    """Final real-model run, the demo's drift row: after Hindi questions about the Suryodaya notice and a general one,
    "Let's go back to the annual report" got "Sure, back to the Suryodaya Yojana Soochna. We were talking about the
    seats." (the last document topic), not the annual report the user named."""
    state = ConversationState(chat_id="c", document_topic="seats total", active_document_ids=["d_notice"])
    p = SimpleNamespace(state=state, ready=READY)
    plan = SimpleNamespace(mode="resume", language="en", ack=None)
    assert ChatTurnService._fixed_reply(plan, p, "Let's go back to the annual report.") == (
        "Sure, back to the Valmora annual report. What would you like to know?"
    )
    # Naming the document of the topic, or no document: the topic is kept, as before
    assert ChatTurnService._fixed_reply(plan, p, "back to the Suryodaya notice") == ChatTurnService._fixed_reply(
        plan, p, "Let's go back to the report."
    )
    assert "seats" in ChatTurnService._fixed_reply(plan, p, "Let's go back to the report.")


# ------------------------------------------------------------------ UX5


def source(i: int, document_id: str, filename: str) -> Source:
    return Source(f"S{i}", make_chunk(i, document_id=document_id, text=f"Revenue {i}"), filename, 0.9)


VALMORA = "valmora_annual_report_fy24.pdf"
ZEPHYRA = "zephyra_investor_deck_q4fy24.pptx"


def test_ux5_short_document_names():
    assert short_document_name(VALMORA) == "the Valmora annual report"
    assert short_document_name(ZEPHYRA) == "the Zephyra investor deck"
    assert short_document_name("annual_report.pdf") == "the annual report"


def test_ux5_the_prompt_asks_to_name_the_documents():
    both = [source(1, "d1", VALMORA), source(2, "d2", ZEPHYRA)]
    note = documents_note(both)
    assert "2 documents: the Valmora annual report and the Zephyra investor deck" in note
    assert note in answer_user_prompt("What was revenue in FY24?", both, "en", name_documents=True)
    assert "documents:" not in answer_user_prompt("What was revenue in FY24?", both, "en")
    one = documents_note([source(1, "d2", ZEPHYRA)])
    assert one.startswith("(These sources are from the Zephyra investor deck")


@pytest.mark.parametrize(
    ("documents", "ready", "active", "named"),
    [
        (["d1", "d2"], {"d1": VALMORA, "d2": ZEPHYRA}, ["d1"], True),  # evidence from two documents
        (["d2"], {"d1": VALMORA, "d2": ZEPHYRA}, ["d1"], True),  # not the document we were talking about
        (["d2"], {"d1": VALMORA, "d2": ZEPHYRA}, [], True),  # several documents, none obvious yet
        (["d1"], {"d1": VALMORA, "d2": ZEPHYRA}, ["d1"], False),  # the one we were talking about
        (["d1"], {"d1": VALMORA}, [], False),  # the chat's only document
    ],
)
def test_ux5_when_the_answer_names_its_document(documents, ready, active, named):
    p = SimpleNamespace(
        sources=[source(i, d, ready[d]) for i, d in enumerate(documents, start=1)],
        state=ConversationState(chat_id="c", active_document_ids=active),
        ready=ready,
    )
    assert ChatTurnService._name_documents(p) is named  # type: ignore[arg-type]
