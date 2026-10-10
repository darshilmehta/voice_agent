"""Polish round, item 4: a spoken Hindi question heard wrong in noise is asked again, not answered (``ChatTurnService``
on fakes, scripted router and model). In café noise "सूर्योदय योजना में आवेदन की अंतिम तिथि क्या है?" came out as
"…आवेदन की जाति ख्या है" and "…आबेदन की अंटिम सिथी क्या है?", which Whisper was sure of and which read as valid
Devanagari: the router model called them unclear, the B1 check sent them to the documents, and the answers made no
sense or denied the deadline. A clear question is never asked again for this."""

from __future__ import annotations

import pytest

from app.providers.retrieval import IndexedChunk
from app.services.chat_turns import ChatTurnService
from app.services.chats import ChatService
from app.services.projects import ProjectService
from app.services.prompts import ACK_TEXTS
from app.services.retrieval import RetrievalService
from app.services.subjects import misheard_words

from .conftest import add_document
from .fakes import make_chunk, vector_for
from .test_last_round import answer_prompts, saved, spoken, turn

NOTICE = [
    (["5. आवेदन प्रक्रिया"], "आवेदन की अंतिम तिथि 30 नवंबर 2024 है। आवेदन के लिए कोई शुल्क नहीं है।"),
    (["6. सीटें"], "| पाठ्यक्रम | सीटों की संख्या |\n|---|---|\n| कुल | 18,000 |"),
    (["4. योजना के लाभ"], "प्रशिक्षण सफलतापूर्वक पूरा करने पर ₹ 10,000 की टूलकिट सहायता मिलेगी।"),
]
SAY_AGAIN = ACK_TEXTS["repeat"]["hi"]
CASTE = "सूर्योदय योजना में आवेदन की जाति ख्या है"  # the reported transcript
DEADLINE_HEARD = "सूर्योदय योजना में आबेदन की अंटिम सिथी क्या है?"


@pytest.fixture
async def notice(db, load_local, fakes):
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
    return ChatTurnService(db, retrieval=retrieval, llm=fakes.llm, settings=settings), chat.id


async def test_an_unclear_question_with_a_misheard_question_word_is_asked_again(notice, fakes):
    """The reported case: the notice matches "सूर्योदय योजना में आवेदन" strongly (0.82 on the real reranker), the
    real model answered by repeating the question."""
    service, chat_id = notice
    fakes.llm.route = {"intent": "clarification", "query": None}
    fakes.reranker.scorer = lambda q, p: 0.82 if "आवेदन" in p else 0.01
    fakes.llm.reply = f"{CASTE} [S1]।"
    events = await turn(service, chat_id, CASTE, modality="voice", language="hi")
    message = saved(events)
    assert message.text == SAY_AGAIN and spoken(events) == SAY_AGAIN
    say_again = message.route["say_again"]
    assert say_again["heard_wrong"] == 0.82 and list(say_again["misheard"]) == ["ख्या"]  # a question word, misheard
    # ("जाति" sounds like "जाता" but is spelled with the same consonants: a word, not a mishearing)
    assert answer_prompts(fakes.llm) == []  # the model never answered it


async def test_an_unclear_question_the_documents_match_weakly_is_asked_again(notice, fakes):
    service, chat_id = notice
    fakes.llm.route = {"intent": "clarification", "query": None}
    fakes.reranker.scorer = lambda q, p: 0.03 if "आवेदन" in p else 0.0  # over the gate, barely
    fakes.llm.reply = "दस्तावेज़ में आवेदन की अंतिम तिथि का जिक्र नहीं है।"
    events = await turn(service, chat_id, DEADLINE_HEARD, modality="voice", language="hi")
    message = saved(events)
    assert message.text == SAY_AGAIN and message.route["say_again"]["heard_wrong"] == 0.03
    assert {"आबेदन": "आवेदन", "अंटिम": "अंतिम"}.items() <= message.route["say_again"]["misheard"].items()


async def test_a_router_timeout_with_misheard_words_and_a_weak_match_is_asked_again(notice, fakes):
    service, chat_id = notice
    fakes.llm.route = TimeoutError("router timed out")
    fakes.reranker.scorer = lambda q, p: 0.05 if "आवेदन" in p else 0.0
    events = await turn(service, chat_id, DEADLINE_HEARD, modality="voice", language="hi")
    assert saved(events).text == SAY_AGAIN
    # strongly matched, the same transcript is answered (the misheard words are respelled for the model)
    fakes.reranker.scorer = lambda q, p: 0.9 if "आवेदन" in p else 0.0
    fakes.llm.reply = "आवेदन की अंतिम तिथि 30 नवंबर 2024 है [S1]।"
    events = await turn(service, chat_id, DEADLINE_HEARD, modality="voice", language="hi")
    assert saved(events).text == "आवेदन की अंतिम तिथि 30 नवंबर 2024 है [S1]।"


@pytest.mark.parametrize(
    ("question", "route", "score"),
    [
        ("सूर्योदय योजना में आवेदन की अंतिम तिथि क्या है?", "document_qa", 0.99),
        ("कुल कितनी सीटें हैं?", "general_qa", 0.28),  # weak, and "सीटें" is "सीटों" inflected, not misheard
        ("तूलकेच के लिए कितनी सहायता मिलती है?", "general_qa", 0.13),  # misheard, found: answered as before
        (DEADLINE_HEARD, "document_qa", 0.03),  # the router read it as a question: answered (respelled)
        # called unclear, but matched strongly and only a word of its subject misheard: answered (real run: 0.975)
        ("सूर्योदय योजना में आवेदन की अन्तिम तिखी क्या है?", "clarification", 0.97),
        ("तूल्कित के लिए कितनी सहायता मिलती है।", "clarification", 0.5),  # clean speech, misheard: real run 0.50
    ],
)
async def test_questions_the_router_reads_as_questions_are_answered(notice, fakes, question, route, score):
    service, chat_id = notice
    fakes.llm.route = {"intent": route, "query": None}
    fakes.reranker.scorer = lambda q, p: score if any(w in p for w in ("आवेदन", "सीटों", "टूलकिट")) else 0.0
    fakes.llm.reply = "उत्तर [S1]।"
    events = await turn(service, chat_id, question, modality="voice", language="hi")
    message = saved(events)
    assert message.text == "उत्तर [S1]।" and "say_again" not in message.route


async def test_typed_or_english_it_is_answered(notice, fakes):
    service, chat_id = notice
    fakes.llm.route = {"intent": "clarification", "query": None}
    fakes.reranker.scorer = lambda q, p: 0.82 if "आवेदन" in p else 0.01
    fakes.llm.reply = "उत्तर [S1]।"
    events = await turn(service, chat_id, CASTE, language="hi")  # typed: what was typed is what was meant
    assert saved(events).text == "उत्तर [S1]।"


def test_misheard_question_words_count_only_when_asked_for():
    assert misheard_words(CASTE, [NOTICE[0][1]]) == {}
    assert "ख्या" in misheard_words(CASTE, [NOTICE[0][1]], common=True)
