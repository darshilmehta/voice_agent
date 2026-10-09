"""The last round's backend fixes in whole turns (``ChatTurnService`` on fakes, scripted model), from the post-fix
real-model verification: a year-end date is the end of a fiscal year (item 3), no invented figures after a misheard
question (item 1), a rewrite that names another document's subject (item 4), Hindi short replies cut mid-sentence
(item 5) and the say-again line (item 6)."""

from __future__ import annotations

from typing import Any

import pytest

from app.providers.retrieval import IndexedChunk
from app.services.chat_turns import AgentMessageEvent, ChatTurnService, DeltaEvent, wait_for_background
from app.services.chats import ChatService
from app.services.projects import ProjectService
from app.services.prompts import ACK_TEXTS, abstention
from app.services.retrieval import RetrievalService

from .conftest import add_document
from .fakes import make_chunk, vector_for

LABEL = "valmora annual report fy24: Valmora Industries Limited - Annual Report 2023-24"
REPORT = [
    (
        ["Management discussion and analysis", "Recap of FY23, the comparative year"],
        "In FY23, Valmora's revenue from operations was ₹ 6,482 crore. Net debt stood at ₹ 1,188 crore at 31 March "
        "2023, or 0.93x EBITDA.",
    ),
    (
        ["Management discussion and analysis: FY24 financial performance"],
        "Net debt declined to ₹ 831 crore from ₹ 1,188 crore, and ROCE rose to 19.6% from 17.9%.",
    ),
    (["Highlights"], "| Metric | FY24 | FY23 |\n|---|---|---|\n| Revenue from operations (₹ crore) | 7,365 | 6,482 |"),
]


@pytest.fixture
async def world(db, load_local, fakes):
    """Valmora's annual report (three passages) in the fake index, a chat, the turn service on fakes."""
    settings = load_local()
    project = await ProjectService(db).create("Valmora")
    doc = await add_document(db, project.id, "valmora_annual_report_fy24.pdf", status="READY", page_count=30)
    for i, (heading, text) in enumerate(REPORT):
        chunk = make_chunk(
            i,
            project_id=project.id,
            document_id=doc,
            text=text,
            heading_path=heading,
            document_label=LABEL,
            page_start=16 + i,
            page_end=16 + i,
        )
        await fakes.store.upsert([IndexedChunk(chunk, vector_for(text))])
    chat = await ChatService(db).create(project.id)
    retrieval = RetrievalService(fakes.embedder, fakes.reranker, fakes.store, settings.retrieval)
    service = ChatTurnService(db, retrieval=retrieval, llm=fakes.llm, settings=settings)
    return service, chat.id


async def turn(service: ChatTurnService, chat_id: str, text: str, **kw: Any) -> list[Any]:
    t = await service.begin(chat_id, text, **kw)
    events = [e async for e in service.run(t)]
    await wait_for_background()
    return events


def saved(events: list[Any]):
    return next(e.message for e in events if isinstance(e, AgentMessageEvent))


def spoken(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, DeltaEvent))


def answer_prompts(llm) -> list[str]:
    return [c["messages"][-1].content for c in llm.calls if "Question:" in c["messages"][-1].content]


# ------------------------------------------------------------------ item 3: the end of a fiscal year

NET_DEBT = "What was Valmora's net debt on 31 March 2024?"


def recap_first(q: str, p: str) -> float:
    """What the real reranker did: the recap of FY23 ("… at 31 March 2023") just above the FY24 paragraph."""
    return 0.93 if "Recap of FY23" in p else 0.81 if "831" in p else 0.0


async def test_a_year_end_date_is_answered_from_that_fiscal_years_passage(world, fakes):
    service, chat_id = world
    fakes.reranker.scorer = recap_first
    fakes.llm.route = {"intent": "document_qa", "query": None}
    fakes.llm.reply = "Net debt was ₹ 831 crore at 31 March 2024 [S1]."
    events = await turn(service, chat_id, NET_DEBT, modality="voice")
    prompt = answer_prompts(fakes.llm)[-1]
    first = prompt.split("[S2]")[0]
    assert "831" in first and "Recap of FY23" not in first  # the FY24 paragraph is S1
    assert "31 March 2024 is the end of FY24" in prompt
    assert saved(events).text == "Net debt was ₹ 831 crore at 31 March 2024 [S1]."
    assert saved(events).route["abstained"] is False


async def test_without_the_fiscal_years_passage_it_abstains_rather_than_give_another_years_figure(world, fakes):
    service, chat_id = world
    fakes.reranker.scorer = lambda q, p: 0.93 if "Recap of FY23" in p else 0.0
    fakes.llm.route = {"intent": "document_qa", "query": None}
    fakes.llm.reply = "Net debt was ₹ 1,188 crore [S1]."
    events = await turn(service, chat_id, NET_DEBT, modality="voice")
    message = saved(events)
    assert message.route["abstained"] is True and "1,188" not in message.text
    assert answer_prompts(fakes.llm) == []  # the gate abstained: no answer model call


@pytest.mark.parametrize("asked", ["How much net debt was there as at March 31, 2024?", "Net debt on 31.03.2024?"])
async def test_a_general_proposal_for_a_year_end_question_is_checked_against_the_documents(world, fakes, asked):
    """The B1 check: a year-end date is the documents' subject, like "FY24" (a match below the strong 0.6 makes a
    question that isn't about the documents' subject a mixed one)."""
    service, chat_id = world
    fakes.reranker.scorer = lambda q, p: 0.45 if "Recap of FY23" in p else 0.4 if "831" in p else 0.0
    fakes.llm.route = {"intent": "general_qa", "query": None}
    fakes.llm.reply = "Net debt was ₹ 831 crore at 31 March 2024 [S1]."
    events = await turn(service, chat_id, asked, modality="voice")
    assert saved(events).route["intent"] == "document_qa"
    assert "31 March 2024 is the end of FY24" in answer_prompts(fakes.llm)[-1]


# ------------------------------------------------------------------ item 1: no invented figures after a misheard word
#
# The last real run, in a chat with the Hindi scheme notice: "टूलकिट के लिए कितनी सहायता मिलती है?" heard as "तूलकेच…",
# routed general_qa, spoken "यह आपके दस्तावेज़ों से नहीं है, लेकिन तूलकेच के लिए सहायता ₹ 1,500 प्रति माह मिलती है।" (the
# notice says ₹ 10,000); "बैंक ऋण पर कितना अनुदान…" heard as "बेख रिन…" gave the disclaimer twice and "₹ 5,000 प्रति माह"
# (the notice says 35% up to ₹ 1,75,000).

NOTICE = [
    (["3. पात्रता"], "आवेदक की आयु 18 वर्ष से 35 वर्ष के बीच होनी चाहिए।"),
    (["4. योजना के लाभ"], "प्रशिक्षण सफलतापूर्वक पूरा करने पर ₹ 10,000 की टूलकिट सहायता मिलेगी।"),
    (["4. योजना के लाभ"], "स्वरोज़गार शुरू करने के लिए बैंक ऋण पर 35 प्रतिशत अनुदान दिया जाएगा, जो अधिकतम ₹ 1,75,000 होगा।"),
]
TOOLKIT_HEARD = "तूलकेच के लिए कितनी सहायता मिलती है?"
LOAN_HEARD = "बेख रिन पर कितना अनुदान मिलता है?"
INVENTED = "यह आपके दस्तावेज़ों से नहीं है, लेकिन तूलकेच के लिए सहायता ₹ 1,500 प्रति माह मिलती है।"


@pytest.fixture
async def notice(db, load_local, fakes):
    """The Hindi scheme notice (three passages) in the fake index, a chat, the turn service on fakes."""
    settings = load_local()
    project = await ProjectService(db).create("Suryodaya")
    doc = await add_document(db, project.id, "suryodaya_yojana_soochna.docx", status="READY")
    for i, (heading, text) in enumerate(NOTICE):
        chunk = make_chunk(
            i, project_id=project.id, document_id=doc, text=text, heading_path=heading, language="hi",
            document_label="suryodaya yojana soochna", page_start=None, page_end=None,
        )  # fmt: skip
        await fakes.store.upsert([IndexedChunk(chunk, vector_for(text))])
    chat = await ChatService(db).create(project.id)
    retrieval = RetrievalService(fakes.embedder, fakes.reranker, fakes.store, settings.retrieval)
    service = ChatTurnService(db, retrieval=retrieval, llm=fakes.llm, settings=settings)
    return service, chat.id


def no_answer_call(llm) -> bool:
    """The answer model was never asked (only the router's JSON calls, which ``llm.calls`` doesn't hold)."""
    return llm.calls == []


async def test_a_general_proposal_for_a_figure_is_answered_from_the_documents(notice, fakes):
    """The B1 check sends a question for an amount to the documents, whatever the router said: the notice's toolkit
    passage answers it."""
    service, chat_id = notice
    fakes.llm.route = {"intent": "general_qa", "query": "How much assistance is given for the toolkit?"}
    fakes.reranker.scorer = lambda q, p: 0.3 if "टूलकिट" in p else 0.0  # weak, but over the gate (not "strong")
    fakes.llm.reply = "प्रशिक्षण पूरा करने पर ₹ 10,000 की टूलकिट सहायता मिलती है [S1]।"
    events = await turn(service, chat_id, TOOLKIT_HEARD, modality="voice", language="hi")
    message = saved(events)
    assert message.route["intent"] == "document_qa" and message.route["abstained"] is False
    assert any(o.startswith("general_qa→document_qa") for o in message.route["router"]["overrides"])
    assert "10,000" in message.text and "1,500" not in message.text


async def test_a_misheard_figure_question_the_documents_dont_answer_is_asked_again(notice, fakes):
    service, chat_id = notice
    fakes.llm.route = {"intent": "general_qa", "query": "How much assistance is given for Tulkech?"}
    fakes.reranker.scorer = lambda q, p: 0.0  # nothing over the gate
    fakes.llm.reply = INVENTED
    events = await turn(service, chat_id, TOOLKIT_HEARD, modality="voice", language="hi")
    message = saved(events)
    assert message.text == ACK_TEXTS["repeat"]["hi"]  # "माफ़ कीजिए, मैं ठीक से सुन नहीं पाया। …"
    assert message.route["abstained"] is True and message.route["say_again"]["misheard"] == {"तूलकेच": "टूलकिट"}
    assert no_answer_call(fakes.llm) and "1,500" not in spoken(events)
    # typed, the same question is declined: nothing was misheard
    events = await turn(service, chat_id, TOOLKIT_HEARD, language="hi")
    assert saved(events).text == abstention("hi", "not_covered") and no_answer_call(fakes.llm)


async def test_speech_recognition_unsure_of_a_question_the_documents_dont_answer_asks_again(notice, fakes):
    service, chat_id = notice
    fakes.llm.route = {"intent": "general_qa", "query": "How much grant on a bank loan?"}
    fakes.reranker.scorer = lambda q, p: 0.0
    fakes.llm.reply = "यह आपके दस्तावेज़ों से नहीं है, लेकिन यह आपके दस्तावेज़ों से नहीं है, लेकिन ₹ 5,000 प्रति माह।"
    events = await turn(service, chat_id, "कितना अनुदान मिलता है उस पर?", modality="voice", language="hi", unsure=True)
    message = saved(events)
    assert message.text == ACK_TEXTS["repeat"]["hi"] and message.route["say_again"]["unsure"] is True
    assert "5,000" not in spoken(events) and no_answer_call(fakes.llm)
    events = await turn(service, chat_id, LOAN_HEARD, modality="voice", language="hi")  # "बेख रिन" ≈ "बैंक ऋण"
    assert saved(events).route["say_again"]["misheard"] == {"बेख": "बैंक", "रिन": "ऋण"}


async def test_a_misheard_hindi_word_is_read_as_the_passages_spell_it(notice, fakes):
    """The documents do answer it ("अनुदान" finds the passage): the answer model reads "बैंक ऋण", not "बेख रिन" (the
    real model, given "बेख रिन…", said the documents have nothing on it)."""
    service, chat_id = notice
    fakes.llm.route = {"intent": "document_qa", "query": "How much subsidy is given on beer?"}
    fakes.reranker.scorer = lambda q, p: 0.4 if "अनुदान" in p else 0.0
    fakes.llm.reply = "बेख रिन पर 35 प्रतिशत अनुदान मिलता है, अधिकतम ₹ 1,75,000 [S1]।"
    events = await turn(service, chat_id, LOAN_HEARD, modality="voice", language="hi")
    prompt = answer_prompts(fakes.llm)[-1]
    assert "Question: बैंक ऋण पर कितना अनुदान मिलता है?" in prompt and '"बेख" is बैंक' in prompt
    assert saved(events).text == "बैंक ऋण पर 35 प्रतिशत अनुदान मिलता है, अधिकतम ₹ 1,75,000 [S1]।"  # echoed: respelled


async def test_a_figure_question_that_stays_general_is_declined_not_guessed(notice, fakes):
    """When the documents weren't searched for it (here: two words, too short for the B1 check), a general answer
    would still have to guess the figure: declined, with no model call."""
    service, chat_id = notice
    fakes.llm.route = {"intent": "general_qa", "query": "How much is the stipend?"}
    fakes.llm.reply = "वजीफ़ा ₹ 2,000 प्रति माह है।"
    events = await turn(service, chat_id, "वजीफ़ा कितना?", language="hi")
    message = saved(events)
    assert message.text == abstention("hi", "not_covered") and message.route["abstained"] is True
    assert no_answer_call(fakes.llm)


async def test_a_mixed_question_for_a_figure_the_documents_dont_cover_is_declined(world, fakes):
    service, chat_id = world
    fakes.llm.route = {"intent": "mixed", "query": None}
    fakes.reranker.scorer = lambda q, p: 0.0
    fakes.llm.reply = "Valmora's market share is about 12%."
    events = await turn(service, chat_id, "How much market share does Valmora have compared to its peers?")
    message = saved(events)
    assert message.route["abstained"] is True and "12%" not in message.text
    assert no_answer_call(fakes.llm)


async def test_a_general_answer_in_a_documents_chat_states_no_figure_and_no_second_disclaimer(world, fakes):
    service, chat_id = world
    fakes.llm.route = {"intent": "general_qa", "query": None}
    fakes.reranker.scorer = lambda q, p: 0.0
    fakes.llm.reply = "Not from your documents, but Jupiter is the largest planet. About 1,300 Earths could fit in it."
    events = await turn(service, chat_id, "Which planet is the largest one?", modality="voice")
    message = saved(events)
    assert message.text == (
        "Not from your documents, but Jupiter is the largest planet. I don't have a reliable figure for that."
    )
    assert [c["check"] for c in message.route["checks"]] == ["disclaimer", "general_figure"]
    # typed: no prefix, the plain line
    fakes.llm.reply = "Jupiter is the largest planet. About 1,300 Earths could fit in it."
    events = await turn(service, chat_id, "Which planet is the largest one?")
    assert saved(events).text == "Jupiter is the largest planet. I couldn't find that in your documents."


async def test_a_hindi_general_answer_says_its_disclaimer_once(world, fakes):
    service, chat_id = world
    fakes.llm.route = {"intent": "general_qa", "query": "Which planet is the largest?"}
    fakes.reranker.scorer = lambda q, p: 0.0
    fakes.llm.reply = "यह आपके दस्तावेज़ों से नहीं है, लेकिन बृहस्पति सबसे बड़ा ग्रह है।"
    events = await turn(service, chat_id, "सबसे बड़ा ग्रह कौन सा है?", modality="voice", language="hi")
    assert saved(events).text == "यह आपके दस्तावेज़ों से नहीं है, लेकिन बृहस्पति सबसे बड़ा ग्रह है।"
