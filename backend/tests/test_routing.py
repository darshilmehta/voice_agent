"""The router's rules without a model or a database: languages (incl. Hinglish), the keyword fast path, validation of
the model's proposals, the router prompt and the conversation state's transitions (docs/DESIGN.md §3.4, §3.5)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.domain.conversation import ConversationState, TurnRoute
from app.domain.projects import Citation, Message
from app.providers.llm import LLMError
from app.services.conversation import TurnOutcome, advance
from app.services.language import asked_language, decide_language, is_hinglish, message_language
from app.services.planning import interrupted_answer
from app.services.router import (
    LLMTurnRouter,
    RouteRequest,
    RouterProposal,
    acknowledgement,
    fast_route,
    heuristic_topic,
    is_correction,
    mentions_documents,
    normalize,
    refers_back,
    router_messages,
    standalone_question,
    validate,
)

from .fakes import FakeLLM

DOCS = ["annual_report.pdf"]


def msg(role: str, text: str, *, heard: str | None = None, seq: int = 1, **route: Any) -> Message:
    return Message(
        id=f"msg_{seq}",
        chat_id="cht_1",
        seq=seq,
        role=role,  # type: ignore[arg-type]
        modality="voice",
        text=text,
        heard_text=heard,
        language=None,
        citations=[],
        route=route or None,
        latency=None,
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
    )


def history(*pairs: tuple[str, str] | tuple[str, str, str]) -> list[Message]:
    return [msg(p[0], p[1], heard=p[2] if len(p) > 2 else None, seq=i) for i, p in enumerate(pairs, start=1)]


MARGIN = history(("user", "What was the EBITDA margin in FY24?"), ("agent", "It was 18.2% [S1]."))


def request(
    utterance: str,
    hist: list[Message] | None = None,
    *,
    language: str = "en",
    state: ConversationState | None = None,
) -> RouteRequest:
    hist = hist or []
    return RouteRequest(
        utterance,
        language,  # type: ignore[arg-type]
        hist,
        state or ConversationState(chat_id="cht_1"),
        DOCS,
        interrupted_answer(hist),
    )


def proposal(intent: str, query: str | None = None) -> RouterProposal:
    return RouterProposal(intent=intent, query=query)  # type: ignore[arg-type]


# ------------------------------------------------------------------ languages


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FY24 mein revenue kitna tha?", "hi"),
        ("achha, aur debt ke baare mein batao", "hi"),
        ("EBITDA margin kya tha FY23 mein?", "hi"),
        ("theek hai", "hi"),
        ("batao", "hi"),
        ("What was the revenue yaar", "en"),  # one stray Hindi word
        ("Can you explain the main risks?", "en"),  # "main" is English too
        ("Hi there, to the point please", "en"),  # "hi", "to" never count as Hindi
        ("FY24 में EBITDA margin क्या था?", "hi"),
        ("What is the मार्जिन in FY24?", "en"),
        ("18.2%?", None),
    ],
)
def test_message_language_handles_code_mixing(text, expected):
    assert message_language(text) == expected


def test_hinglish_needs_more_than_one_stray_word():
    assert is_hinglish("revenue kitna tha") and not is_hinglish("what was the revenue in FY24 yaar")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("answer in Hindi", "hi"),
        ("Can you reply in English please?", "en"),
        ("Hindi mein batao", "hi"),
        ("english me bolo", "en"),
        ("हिंदी में बताइए", "hi"),
        ("अंग्रेज़ी में बताओ", "en"),
        ("in Hindi please", "hi"),
        ("Switch to English", "en"),
        ("speak Hindi", "hi"),
        ("don't answer in English, answer in Hindi", "hi"),  # the later request wins
        ("what is revenue in Hindi?", None),  # a question about a word, not a request
        ("Tell me the Hindi word for revenue", None),
        ("What was revenue in FY24?", None),
    ],
)
def test_asked_language(text, expected):
    assert asked_language(text) == expected


def test_language_rule_order():
    # asked beats everything, and is recorded as the new preference
    d = decide_language("answer in Hindi", requested="en", preferred="en", fallback="en")
    assert (d.language, d.reason, d.asked) == ("hi", "asked", "hi")
    # a forced language beats the preference
    assert decide_language("What was revenue?", requested="hi", preferred="en", fallback="en").reason == "requested"
    # the preference beats the utterance's own language
    d = decide_language("What was revenue?", preferred="hi", fallback="en")
    assert (d.language, d.reason, d.input_language) == ("hi", "preferred", "en")
    # otherwise the utterance (Hinglish → Hindi), then the fallback
    assert decide_language("revenue kitna tha?", fallback="en").language == "hi"
    assert decide_language("18.2%?", fallback="hi").reason == "fallback"


def test_voice_passing_the_spoken_language_does_not_override_a_preference():
    # the voice session passes language=<spoken>, input_language=<spoken>: that forces nothing
    d = decide_language("What was revenue?", requested="en", spoken="en", preferred="hi", fallback="en")
    assert (d.language, d.reason) == ("hi", "preferred")
    d = decide_language("what was revenue", requested="hi", spoken="hi", fallback="en")  # STT heard Hindi
    assert (d.language, d.reason) == ("hi", "utterance")


# ------------------------------------------------------------------ fast path


@pytest.mark.parametrize("text", ["Stop.", "stop talking", "bas karo", "रुको", "That's enough!"])
def test_stop_phrases_need_no_model(text):
    decision = fast_route(request(text, MARGIN))
    assert decision is not None and decision.route.intent == "stop" and decision.source == "heuristic"


@pytest.mark.parametrize("text", ["Okay.", "mm-hmm", "Got it", "theek hai", "achha", "ठीक है", "yes"])
def test_acknowledgements_while_idle_are_backchannels(text):
    decision = fast_route(request(text, MARGIN))
    assert decision is not None and decision.route.intent == "backchannel"
    assert not decision.route.needs_retrieval


def test_yes_to_an_offer_is_not_a_backchannel():
    offer = history(("user", "How much did revenue grow?"), ("agent", "34%. Want the breakdown by segment?"))
    assert fast_route(request("yes", offer)) is None  # the router decides: an answer to the question


def test_a_backchannel_after_a_barge_in_goes_to_the_router():
    cut = history(("user", "What was the margin?"), ("agent", "The margin was 18.2% and", "The margin"))
    assert fast_route(request("okay", cut)) is None


def test_a_language_request_alone_repeats_the_previous_question_or_is_acknowledged():
    answered = [
        MARGIN[0],
        MARGIN[1].model_copy(update={"route": {"answer": "grounded", "topic": "ebitda margin", "query_en": None}}),
    ]
    decision = fast_route(request("हिंदी में बताइए", answered, language="hi"))
    assert decision is not None and decision.route.intent == "correction"
    assert decision.route.rewritten_query == "What was the EBITDA margin in FY24?" and decision.route.needs_retrieval
    assert decision.route.topic == "ebitda margin" and decision.route.response_language == "hi"
    decision = fast_route(request("please answer in Hindi", language="hi"))  # nothing to repeat yet
    assert decision is not None and (decision.route.intent, decision.reply) == ("conversation", "language")
    assert fast_route(request("answer in Hindi: what was the revenue?", language="hi")) is None  # asks more


def test_b5_asked_for_english_the_previous_hinglish_question_is_asked_again_in_english():
    hinglish = [
        msg("user", "FY24 mein revenue kitna tha?", seq=1),
        msg("agent", "FY24 में राजस्व ₹ 7,365 करोड़ था।", seq=2, answer="grounded", query_en="What was revenue in FY24?"),
    ]
    decision = fast_route(request("answer in English please", hinglish, language="en"))
    assert decision is not None and decision.language_request
    assert (decision.route.rewritten_query, decision.route.query_en) == ("What was revenue in FY24?", None)
    # asked for Hindi, the question keeps its own words and its English search query
    decision = fast_route(request("हिंदी में बताइए", hinglish, language="hi"))
    assert decision is not None and decision.language_request
    assert (decision.route.rewritten_query, decision.route.query_en) == (
        "FY24 mein revenue kitna tha?",
        "What was revenue in FY24?",
    )


@pytest.mark.parametrize(
    ("expected", "pieces", "verdict"),
    [
        ("en", ["The EBITDA ", "margin was 18.2%"], True),
        ("en", ["FY24 में ", "राजस्व"], False),  # asked for English, written in Hindi
        ("en", ["₹ 7,365 ", "crore [S1]."], None),  # too few letters yet
        ("hi", ["FY24 में ", "EBITDA"], True),
        ("hi", ["EBITDA margin FY24 ", "Valmora revenue grew"], False),  # 24+ Latin letters, no Devanagari
        ("hi", ["EBITDA मार्जिन"], True),
    ],
)
def test_b5_script_check(expected, pieces, verdict):
    from app.services.language import ScriptCheck

    check = ScriptCheck(expected)
    for piece in pieces:
        check.feed(piece)
    assert check.verdict == verdict


def test_b5_script_check_decides_short_answers_at_the_end():
    from app.services.language import ScriptCheck, script_language

    short = ScriptCheck("hi")
    short.feed("Yes, 18.2%.")
    assert short.verdict is None and short.finish() is None  # "Yes" alone: can't tell
    english = ScriptCheck("hi")
    english.feed("It was about 18.2% then.")
    assert english.finish() is False
    assert script_language("FY24 में EBITDA 21.0% था") == "hi" and script_language("₹ 7,365 [S1]") is None
    assert script_language("Revenue was 7,365 crore") == "en"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Thanks!", "thanks"),
        ("shukriya", "thanks"),
        ("theek hai, shukriya", "thanks"),
        ("okay thank you so much", "thanks"),
        ("Hello", "greeting"),
    ],
)
def test_thanks_and_greetings_get_fixed_replies(text, kind):
    decision = fast_route(request(text, MARGIN))
    assert decision is not None and decision.route.intent == "conversation" and decision.reply == kind


def test_an_english_standalone_question_naming_the_documents_is_a_document_question():
    decision = fast_route(request("What does the annual report say about debt?"))
    assert decision is not None and decision.route.intent == "document_qa" and decision.route.needs_retrieval
    assert decision.route.topic == "debt"


@pytest.mark.parametrize(
    "text",
    [
        "What does it say in the report?",  # refers back
        "And the debt in the report?",  # follow-up opener
        "No, I meant the debt in the report",  # correction
        "रिपोर्ट में मुनाफा कितना था?",  # Hindi: the router writes the English query
        "What was the EBITDA margin in FY24?",  # names no document: router (or a confident retrieval)
        "Let's go back to the report",  # resume
    ],
)
def test_the_fast_path_is_conservative(text):
    assert fast_route(request(text, MARGIN)) is None


def test_hindi_punctuation_and_word_ends():
    # Hindi transcripts end with a danda; words ending in a vowel sign or anusvara still end there
    assert normalize("बस।") == "बस" and fast_route(request("बस।", MARGIN)).route.intent == "stop"
    assert acknowledgement("ठीक है।") == "ack"
    assert is_correction("नहीं, FY23 वाला") and refers_back("तो FY23 में?") and refers_back("और FY23?")
    assert not is_correction("नहींचाहिए")  # not a word of its own


@pytest.mark.parametrize(
    ("text", "files"),
    [
        ("How do I write a research paper?", DOCS),  # a generic noun, not the user's documents
        ("What makes a good contract clause?", DOCS),
        ("Explain how planets form in a disk", ["plan.pdf"]),  # a file named "plan" isn't "planets"
    ],
)
def test_naming_documents_needs_more_than_a_generic_word(text, files):
    assert not mentions_documents(text, files)
    assert mentions_documents("What does the plan say about hiring?", ["plan.pdf"])
    assert mentions_documents("What does the uploaded contract say about termination?", files)


def test_standalone_question():
    assert standalone_question(request("What was the EBITDA margin in FY24?"))
    assert not standalone_question(request("And in FY23?", MARGIN))
    assert not standalone_question(request("Why did it fall?", MARGIN))
    assert not standalone_question(request("Margin?"))  # too short to stand alone


def test_heuristic_topics_ignore_years():
    assert heuristic_topic("What was the EBITDA margin in FY24?") == "ebitda margin"
    assert heuristic_topic("What was the EBITDA margin in FY23?") == "ebitda margin"
    assert heuristic_topic("What's the capital of France?") == "capital france"


# ------------------------------------------------------------------ validation of the model's proposal


def test_resume_phrases_always_resume():
    # §9.5: "let's go back to the annual report" was routed as a backchannel and the answer denied document access
    state = ConversationState(chat_id="c", active_topic="capital france", document_topic="ebitda margin")
    req = request("let's go back to the annual report", MARGIN, state=state)
    decision = validate(proposal("backchannel"), req)
    route = decision.route
    assert route.intent == "resume_document" and not route.needs_retrieval
    assert route.topic == "ebitda margin" and route.is_topic_shift
    assert decision.overrides == ("backchannel→resume_document: resume phrase",)
    assert route.confidence < 0.8  # overridden


def test_a_resume_with_a_question_searches():
    req = request("Anyway, back to the report. What about debt?", MARGIN)
    route = validate(proposal("resume_document", "What does the report say about debt?"), req).route
    assert route.rewritten_query == "What does the report say about debt?" and route.needs_retrieval
    route = validate(proposal("resume_document"), req).route  # no rewrite: the utterance still asks
    assert route.rewritten_query == req.utterance and route.needs_retrieval


def test_a_stop_or_backchannel_that_asks_something_is_a_question():
    assert (
        validate(proposal("backchannel"), request("okay so what was the debt?", MARGIN)).route.intent == "document_qa"
    )
    assert validate(proposal("stop"), request("I'd like to hear about debt", MARGIN)).route.intent == "conversation"
    assert validate(proposal("backchannel"), request("bas karo", MARGIN)).route.intent == "stop"


def test_corrections_rewrite_against_the_heard_answer():
    cut = history(
        ("user", "What was the EBITDA margin in FY24?"),
        ("agent", "The EBITDA margin in FY24 was 18.2%, up from…", "The EBITDA margin in FY24 was"),
    )
    req = request("no, I meant FY23", cut)
    assert req.interrupted is not None and req.interrupted.heard == "The EBITDA margin in FY24 was"
    route = validate(proposal("correction", "What was the EBITDA margin in FY23?"), req).route
    assert route.intent == "correction" and route.rewritten_query == "What was the EBITDA margin in FY23?"
    assert route.needs_retrieval and not route.is_topic_shift
    # the model didn't rewrite: the corrected question is kept with the correction
    route = validate(proposal("correction"), req).route
    assert route.rewritten_query == "What was the EBITDA margin in FY24? — no, I meant FY23"
    # nothing to correct: a plain question
    assert validate(proposal("correction"), request("no, I meant FY23")).route.intent == "document_qa"


def test_queries_by_language():
    # English standalone: a repeated query is dropped
    route = validate(
        proposal("document_qa", "What was the EBITDA margin in FY24?"), request("What was the EBITDA margin in FY24?")
    ).route
    assert (route.rewritten_query, route.query_en) == (None, None)
    # Hindi standalone question: English search query, its own words kept
    hindi = "वित्त वर्ष 2024 में कंपनी का मुनापा कितना था?"  # misspelled transcript (मुनाफा)
    route = validate(
        proposal("document_qa", "What was the company's profit in FY2024?"), request(hindi, language="hi")
    ).route
    assert (route.rewritten_query, route.query_en) == (None, "What was the company's profit in FY2024?")
    assert route.response_language == "hi"
    # Hinglish follow-up: the English standalone question is both
    route = validate(
        proposal("document_qa", "What was the EBITDA margin in FY23?"), request("aur FY23 mein?", MARGIN)
    ).route
    assert route.rewritten_query == route.query_en == "What was the EBITDA margin in FY23?"
    # a Devanagari "English" query is not English
    route = validate(proposal("document_qa", "FY23 में मार्जिन क्या था?"), request("aur FY23 mein?", MARGIN)).route
    assert route.query_en is None and route.rewritten_query == "FY23 में मार्जिन क्या था?"
    # general turns don't search, so no English query
    route = validate(proposal("general_qa", "What is the capital of India?"), request("भारत की राजधानी क्या है?")).route
    assert route.query_en is None and not route.needs_retrieval


def test_yes_to_an_offer_searches_for_what_was_offered():
    offer = history(("user", "How much did revenue grow?"), ("agent", "34% [S1]. Want the breakdown by segment?"))
    route = validate(proposal("document_qa"), request("yes please", offer)).route
    assert route.rewritten_query == "Want the breakdown by segment?" and route.needs_retrieval


def test_conversation_and_clarification_carry_no_query_or_topic():
    for intent in ("conversation", "clarification", "backchannel"):
        route = validate(proposal(intent, "something"), request("hmm what about that", MARGIN)).route
        assert route.rewritten_query is None and route.topic == "" and not route.needs_retrieval


def test_topic_shift_follows_the_turn():
    state = ConversationState(chat_id="c", active_topic="ebitda margin", last_intent="document_qa")
    general = validate(proposal("general_qa"), request("What's the capital of France?", MARGIN, state=state)).route
    assert general.topic == "capital france" and general.is_topic_shift
    follow = validate(
        proposal("document_qa", "What was the EBITDA margin in FY23?"), request("and FY23?", MARGIN, state=state)
    ).route
    assert follow.topic == "ebitda margin" and not follow.is_topic_shift
    thanks = validate(proposal("conversation"), request("thanks a lot, really", MARGIN, state=state)).route
    assert not thanks.is_topic_shift


async def test_the_router_model_call_and_invalid_json():
    llm = FakeLLM()
    router = LLMTurnRouter(llm)
    llm.route = {"intent": "general_qa", "query": None}
    assert await router.propose(request("What's the capital of France?")) == proposal("general_qa")
    call = llm.json_calls[-1]
    assert call["schema"] is RouterProposal and call["max_tokens"] == router.max_tokens
    llm.route = '{"intent": "chit_chat"}'
    with pytest.raises(LLMError):
        await router.propose(request("hi"))
    llm.route = "not json"
    with pytest.raises(LLMError):
        await router.propose(request("hi"))


def test_the_router_prompt_carries_context_and_what_was_heard():
    cut = history(
        ("user", "What was revenue in FY24?"),
        ("agent", "Revenue in FY24 was ₹4,210 crore [S1] and grew 34%.", "Revenue in FY24 was ₹4,210 crore and"),
    )
    state = ConversationState(chat_id="c", active_topic="revenue")
    system, user = router_messages(request("No wait, I meant the EBITDA margin", cut, state=state))
    assert system.role == "system" and "correction" in system.content and len(system.content) < 2600
    lines = user.content.splitlines()
    assert lines[0] == "Documents: annual_report.pdf" and "Topic: revenue" in lines
    assert "Assistant: Revenue in FY24 was ₹4,210 crore and" in lines  # what was heard, markers stripped
    assert lines[-2] == '(The user cut that answer off after hearing: "Revenue in FY24 was ₹4,210 crore and")'
    assert lines[-1] == "Utterance: No wait, I meant the EBITDA margin"


# ------------------------------------------------------------------ conversation state transitions


def route(
    intent: str, *, topic: str = "", shift: bool = False, language: str = "en", query: str | None = None
) -> TurnRoute:
    return TurnRoute(
        intent=intent,  # type: ignore[arg-type]
        needs_retrieval=intent in ("document_qa", "mixed", "correction"),
        rewritten_query=query,
        topic=topic,
        is_topic_shift=shift,
        response_language=language,  # type: ignore[arg-type]
        confidence=0.8,
    )


def cite(doc: str) -> Citation:
    return Citation(
        source_id="S1", document_id=doc, filename="r.pdf", page_start=1, page_end=1, chunk_id="c", snippet=""
    )


def test_state_follows_document_general_hindi_and_back():
    s = ConversationState(chat_id="c")
    s = advance(
        s,
        TurnOutcome(
            route("document_qa", topic="ebitda margin"),
            query="What was the EBITDA margin in FY24?",
            input_language="en",
            searched=True,
            cited_document_ids=["doc_1"],
            message_id="m2",
        ),
    )
    assert (s.active_topic, s.document_topic, s.document_query) == (
        "ebitda margin",
        "ebitda margin",
        "What was the EBITDA margin in FY24?",
    )
    assert s.active_document_ids == ["doc_1"] and s.last_intent == "document_qa"

    s = advance(s, TurnOutcome(route("general_qa", topic="capital france", shift=True), input_language="en"))
    assert (s.active_topic, s.previous_topic, s.document_topic) == ("capital france", "ebitda margin", "ebitda margin")

    s = advance(s, TurnOutcome(route("conversation", language="hi"), input_language="hi"))  # "धन्यवाद"
    assert s.active_topic == "capital france" and (s.input_language, s.response_language) == ("hi", "hi")

    s = advance(s, TurnOutcome(route("resume_document", topic="ebitda margin", shift=True), input_language="en"))
    assert (s.active_topic, s.previous_topic) == ("ebitda margin", "capital france")
    assert s.active_document_ids == ["doc_1"]  # kept: "the report" is still the one we talked about


def test_state_interruptions_and_language_preference():
    s = ConversationState(chat_id="c")
    s = advance(s, TurnOutcome(route("document_qa", topic="margin"), completed=False, message_id="m2"))
    assert s.last_interrupted_message_id == "m2"
    s = advance(s, TurnOutcome(route("stop"), interrupted_message_id="m2"))  # "stop" keeps it
    assert s.last_interrupted_message_id == "m2" and s.last_intent == "document_qa"
    s = advance(
        s, TurnOutcome(route("correction", topic="margin", language="hi"), asked_language="hi", message_id="m6")
    )
    assert s.last_interrupted_message_id is None  # the correction was answered
    assert s.preferred_language == "hi" and s.response_language == "hi"
    s = advance(s, TurnOutcome(route("document_qa", topic="debt", shift=True, language="hi"), input_language="en"))
    assert s.preferred_language == "hi" and s.input_language == "en"  # the preference sticks
