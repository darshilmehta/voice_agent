"""The answer's checks while it streams (services/answer_guard.py, quality round): denials the evidence contradicts,
live figures from nowhere, mistyped codes, misheard names, and the length of short answers. Pure: the model's stream
is a string fed in small pieces."""

from __future__ import annotations

import pytest

from app.services.answer_guard import (
    LAG_WORDS,
    AnswerGuard,
    Coverage,
    Released,
    drop_denial,
    evidence_units,
    figures_in,
    has_denial_cue,
    hindi,
    hindi_hold,
    identifiers_in,
    nearest_identifier,
    number_facts,
    sentence_end,
    terms,
    unsupported_figures,
)
from app.services.sources import answer_declines, clauses, says_not_covered

QUARTERS_FY24 = """| Quarter | Revenue | EBITDA | EBITDA margin |
|---|---|---|---|
| Q1 FY24 | 1,742 | 352 | 20.2% |
| Q4 FY24 | 1,933 | 422 | 21.8% |"""
QUARTERS_FY23 = """| Quarter | Revenue | EBITDA | EBITDA margin |
|---|---|---|---|
| Q1 FY23 | 1,512 | 285 | 18.8% |
| Q4 FY23 | 1,711 | 355 | 20.7% |"""
HOTELS = """| Grade | Tier-1 city (₹ per night) | Tier-2 city (₹ per night) |
|---|---|---|
| L1 to L2 | 5,500 | 3,800 |
| L3 to L4 | 7,500 | 5,200 |"""
OUTLOOK = (
    "FY24 was a strong year. Revenue grew by more than 13 per cent. Looking ahead, the Board has approved capital "
    "expenditure of ₹ 1,100 crore over FY25 and FY26."
)
CIN = "L24119GJ1994PLC023871"


def quarterly_coverage() -> Coverage:
    units = [
        *evidence_units(QUARTERS_FY24, "Quarterly performance: FY24"),
        *evidence_units(QUARTERS_FY23, "Quarterly performance: FY23"),
    ]
    return Coverage(
        units,
        terms("Show me Valmora's quarterly revenue and EBITDA for FY23 and FY24", ignore=["valmora"]),
        correction="The chart on screen shows quarterly revenue and EBITDA, from the Valmora annual report [S4][S5].",
        retry_note="[S4] and [S5] are the chart's tables",
        names=frozenset({"valmora", "zephyra"}),  # the chat's documents' label words, as in a turn
        visual=True,
    )


def hotel_coverage(question: str = "What are the hotel provisions for L3 employees in tier 2 cities?") -> Coverage:
    return Coverage(
        evidence_units(HOTELS, "5. Accommodation 5.1 Hotel limits per night"),
        terms(question),
        correction="The Valmora travel expense policy covers this in 5.1 Hotel limits per night [S1].",
        retry_note="[S1] (5.1 Hotel limits per night)",
    )


def stream(guard: AnswerGuard, text: str, size: int = 6) -> tuple[str, list[str], Released]:
    """The text fed in pieces: what was released, the verdict that ended it (if any) and the last result."""
    out: list[str] = []
    verdicts: list[str] = []
    last = Released()
    for i in range(0, len(text), size):
        last = guard.feed(text[i : i + size])
        out.append(last.text)
        if last.verdict is not None:
            verdicts.append(last.verdict)
            return "".join(out), verdicts, last
    last = guard.finish()
    out.append(last.text)
    if last.verdict is not None:
        verdicts.append(last.verdict)
    return "".join(out), verdicts, last


# ------------------------------------------------------------------ coverage (item 1)


@pytest.mark.parametrize(
    "denial",
    [
        "The report does not provide quarterly revenue and EBITDA figures for FY24.",
        "For FY23, the quarterly revenue and EBITDA data are not available in the provided sources.",
        "The documents don't cover this.",  # names nothing: it is about the question
        "The Zephyra investor deck does not break down EBITDA by quarter for FY23.",
    ],
)
def test_a_denial_of_what_the_chart_shows_is_contradicted(denial):
    assert quarterly_coverage().contradicted_by(denial)


@pytest.mark.parametrize(
    "sentence",
    [
        "The documents do not give Valmora's revenue for FY25.",  # no FY25 row
        "The report does not give the quarterly order book for FY24.",  # no such column
        "Revenue in Q4 FY24 was ₹1,933 crore [S4].",  # no denial
        "The report shows revenue does not include other income.",  # a fact, not a denial
    ],
)
def test_what_the_evidence_doesnt_state_can_be_denied(sentence):
    assert not quarterly_coverage().contradicted_by(sentence)


def test_the_hotel_limits_contradict_a_denial_of_hotel_provisions():
    coverage = hotel_coverage()
    assert coverage.contradicted_by(
        "The documents do not cover the hotel provisions for L3 employees in tier 2 cities. [S1][S2][S3]"
    )
    assert coverage.contradicted_by("The policy does not cover hotel provisions.")
    # a grade the table doesn't name ("level 5", not "L5") and a sentence that only mentions a period don't count
    assert not coverage.contradicted_by("The policy does not specify hotel limits for level 5 employees.")


def test_a_period_and_a_word_in_different_sentences_are_no_evidence():
    """The chairperson mentions FY25 (capex) and revenue (FY24), in different sentences: "no FY25 revenue" stands."""
    coverage = Coverage(evidence_units(OUTLOOK, "Message"), terms("What was revenue in FY25?"), "", "")
    assert not coverage.contradicted_by("The annual report does not mention revenue for FY25.")
    assert coverage.contradicted_by("The annual report does not mention capital expenditure for FY25.")


def test_a_first_sentence_that_denies_the_evidence_asks_the_model_again():
    guard = AnswerGuard(coverage=quarterly_coverage())
    released, verdicts, _ = stream(guard, "The report does not provide quarterly figures for FY24 [S1]. Q4 was best.")
    assert verdicts == ["retry"] and released == ""  # nothing was said yet
    assert guard.checks[0]["check"] == "coverage" and guard.checks[0]["action"] == "retry"
    guard.restart()
    released, verdicts, _ = stream(
        guard, "The chart shows revenue rising every quarter of FY24, to ₹1,933 crore in Q4 [S5]."
    )
    assert released == "The chart shows revenue rising every quarter of FY24, to ₹1,933 crore in Q4 [S5]."
    assert verdicts == []


def test_a_second_denial_is_replaced_by_the_correction_and_ends_the_answer():
    guard = AnswerGuard(coverage=quarterly_coverage())
    guard.restart()  # the model was asked again
    released, verdicts, _ = stream(guard, "The documents don't cover this. Sorry about that. More text here.")
    assert released == quarterly_coverage().correction and verdicts == ["stop"]
    assert guard.checks[-1]["action"] == "corrected"


def test_a_later_denial_is_dropped_or_loses_its_denying_clause():
    guard = AnswerGuard(coverage=quarterly_coverage())
    text = (
        "Valmora's revenue rose every quarter of FY24, from ₹1,742 crore to ₹1,933 crore [S5]. "
        "Q4 FY23 revenue was ₹1,711 crore [S4], but the report does not give quarterly EBITDA for FY23. "
        "For FY23, the quarterly revenue and EBITDA data are not available in the provided sources. "
        "EBITDA margin peaked at 21.8% [S5]."
    )
    released, verdicts, _ = stream(guard, text)
    assert verdicts == []
    assert released == (
        "Valmora's revenue rose every quarter of FY24, from ₹1,742 crore to ₹1,933 crore [S5]. "
        "Q4 FY23 revenue was ₹1,711 crore [S4]. "
        "EBITDA margin peaked at 21.8% [S5]."
    )
    assert [c["action"] for c in guard.checks] == ["clause_dropped", "dropped"]


def test_the_first_sentence_goes_out_three_words_behind_until_it_may_deny():
    guard = AnswerGuard(coverage=quarterly_coverage())
    first = guard.feed("The chart shows Valmora's revenue rising in every quarter of FY24, from ₹1,742 ")
    assert first.text == "The chart shows Valmora's revenue rising in every quarter of"  # all but the last 3 words
    # the sentence ends: the rest of it, checked; the next one is held whole
    rest = guard.feed("crore to ₹1,933 crore [S5]. EBITDA rose from ").text
    assert rest == " FY24, from ₹1,742 crore to ₹1,933 crore [S5]. "  # with its space: speech knows it ended
    guard = AnswerGuard(coverage=quarterly_coverage())
    held = guard.feed("The report does not provide the quarterly figures for FY24 or FY23 in any of ")
    assert held.text == ""  # "not": the whole sentence is checked first
    guard = AnswerGuard(coverage=quarterly_coverage())
    guard.feed("Revenue grew in every quarter of FY24, from ₹1,742 crore to ₹1,933 crore [S5]. ")
    assert guard.feed("EBITDA margin rose from 20.2% to 21.8% in the fourth quarter ").text == ""  # later: whole


def test_a_document_as_the_subject_is_held_until_its_verb_says_what_it_does():
    guard = AnswerGuard(coverage=quarterly_coverage())
    assert guard.feed("The Valmora annual report ").text == ""  # it may go on "does not mention…"
    assert guard.feed("does not mention quarterly revenue for ").text == ""  # it did: checked whole
    guard = AnswerGuard(coverage=quarterly_coverage())
    assert guard.feed("The Valmora annual report says its CIN is ").text == "The Valmora annual report says"
    guard = AnswerGuard(coverage=quarterly_coverage())
    assert (
        guard.feed("According to the report, revenue rose in every quarter ").text
        == "According to the report, revenue rose"
    )
    # Hindi puts its verb last: a sentence that names a document is held until a clause's final auxiliary has come
    guard = AnswerGuard(coverage=quarterly_coverage())
    assert guard.feed("रिपोर्ट में FY24 की चौथी तिमाही का राजस्व ").text == ""  # it may go on "… उपलब्ध नहीं है"
    assert guard.feed("₹1,933 करोड़ बताया गया है, और ").text == (
        "रिपोर्ट में FY24 की चौथी तिमाही का राजस्व ₹1,933 करोड़ बताया गया है,"  # it did not: out, to its verb
    )


# ------------------------------------------------------------------ Hindi: its negation comes last

HINDI_FACT = "वालमोरा का FY24 का राजस्व ₹7,365 करोड़ रहा, जो पिछले साल से 13.6% अधिक है [S1]। EBITDA भी बढ़ा।"
HINDI_DENIAL = "रिपोर्ट में FY24 की तिमाही राजस्व और EBITDA की जानकारी उपलब्ध नहीं है [S1]। बाकी बातें।"


def test_a_plain_hindi_sentence_goes_out_three_words_behind_the_model():
    guard = AnswerGuard(coverage=quarterly_coverage())
    first = guard.feed("वालमोरा का FY24 का राजस्व ₹7,365 करोड़ रहा, जो पिछले साल ")
    assert first.text == "वालमोरा का FY24 का राजस्व ₹7,365 करोड़ रहा,"  # all but the last three words
    rest = guard.feed("से 13.6% अधिक है [S1]। EBITDA भी ").text
    assert rest == " जो पिछले साल से 13.6% अधिक है [S1]। "  # the sentence's end, checked; the next one is held whole
    assert guard.checks == []


def test_a_plain_hindi_answer_is_released_progressively_and_whole():
    """Fed in small pieces (a token or so), the first sentence comes out long before it ends, never out of order."""
    guard = AnswerGuard(coverage=quarterly_coverage())
    pieces = [HINDI_FACT[i : i + 3] for i in range(0, len(HINDI_FACT), 3)]
    out, first_at, fed = "", None, ""
    for i, piece in enumerate(pieces):
        released = guard.feed(piece)
        out += released.text
        fed += piece
        if released.text.strip() and first_at is None:
            first_at = i
        assert HINDI_FACT.startswith(out)  # always the answer's own beginning
        if first_at is not None and "।" not in fed:  # then at most the last three words and the one being written wait
            assert len(fed[len(out) :].split()) <= LAG_WORDS + 1
    out += guard.finish().text
    assert out == HINDI_FACT and guard.checks == []
    sentence_ends_at = HINDI_FACT.index("।") // 3
    assert first_at is not None and first_at < sentence_ends_at / 2  # held a few words, not the whole sentence


def test_a_hindi_sentence_that_names_no_document_but_attributes_one_is_not_held_for_it():
    guard = AnswerGuard(coverage=quarterly_coverage())
    out = guard.feed("रिपोर्ट के अनुसार FY24 की चौथी तिमाही में राजस्व ₹1,933 करोड़ ").text
    assert out == "रिपोर्ट के अनुसार FY24 की चौथी तिमाही में"  # "according to the report": not its subject


def test_a_hindi_denial_is_held_whole_and_asks_the_model_again():
    guard = AnswerGuard(coverage=quarterly_coverage())
    released, verdicts, _ = stream(guard, HINDI_DENIAL, size=3)
    assert verdicts == ["retry"] and released == ""  # nothing of it was said
    assert [(c["check"], c["action"]) for c in guard.checks] == [("coverage", "retry")]
    guard.restart()
    released, verdicts, _ = stream(guard, "FY24 की चौथी तिमाही में राजस्व ₹1,933 करोड़ रहा [S5]।", size=3)
    assert released == "FY24 की चौथी तिमाही में राजस्व ₹1,933 करोड़ रहा [S5]।" and verdicts == []
    guard.restart()  # denied again: the fixed sentence, and the answer ends
    released, verdicts, _ = stream(guard, HINDI_DENIAL, size=3)
    assert released == quarterly_coverage().correction and verdicts == ["stop"]
    assert guard.checks[-1]["action"] == "corrected"


@pytest.mark.parametrize(
    "denial",
    [
        "दस्तावेज़ों में FY24 की तिमाही राजस्व की जानकारी उपलब्ध नहीं है [S1]।",
        "इस बारे में FY24 की तिमाही EBITDA की जानकारी नही दी गई है।",  # "नही", written without the dot
        "मुझे FY24 की तिमाही राजस्व और EBITDA का पता नहीं है।",
        "माफ़ कीजिए, FY24 की तिमाही EBITDA की जानकारी उपलब्ध नहीं है।",
        "FY24 की तिमाही EBITDA का आंकड़ा रिपोर्ट में उपलब्ध नहीं है।",  # what it is about comes late: a few words are out
    ],
)
def test_hindi_denials_of_what_the_chart_shows_are_never_said_whole(denial):
    guard = AnswerGuard(coverage=quarterly_coverage())
    released, verdicts, _ = stream(guard, denial + " अगला वाक्य।", size=4)
    assert verdicts and verdicts[-1] in ("retry", "stop")
    assert "नहीं" not in released and "नही" not in released and "उपलब्ध" not in released  # the denial isn't heard
    assert guard.checks[-1]["check"] == "coverage"


@pytest.mark.parametrize(
    "denial",
    [  # the shapes qwen3:4b-instruct wrote for FY25 questions: the documents come first, the negation last
        "दस्तावेज़ वाल्मोरा के FY24 के तिमाही EBITDA के बारे में जानकारी नहीं देते हैं।",
        "दस्तावेज़ में वाल्मोरा के FY24 के तिमाही राजस्व की कोई जानकारी नहीं है।",
        "इस बारे में FY24 के तिमाही राजस्व की जानकारी उपलब्ध नहीं है।",
    ],
)
def test_a_hindi_denial_that_names_the_documents_first_is_asked_again_before_a_word_is_said(denial):
    guard = AnswerGuard(coverage=quarterly_coverage())
    released, verdicts, _ = stream(guard, denial, size=1)  # a character at a time
    assert released == "" and verdicts == ["retry"]


def test_a_hindi_sentence_whose_verb_comes_last_is_held_only_until_its_verb():
    guard = AnswerGuard(coverage=quarterly_coverage())
    assert guard.feed("दस्तावेज़ में FY24 की चौथी तिमाही का राजस्व ₹1,933 करोड़ ").text == ""  # verb not seen yet
    assert guard.feed("दर्ज है, ").text == "दस्तावेज़ में FY24 की चौथी तिमाही का राजस्व ₹1,933 करोड़ दर्ज है,"  # seen: out
    assert guard.feed("जो पिछली तिमाही से ").text == ""  # the next clause waits for its own verb
    assert guard.feed("अधिक है। अगला ").text == " जो पिछली तिमाही से अधिक है। "
    assert guard.checks == []


def test_a_hindi_denial_in_a_later_clause_loses_only_that_clause():
    guard = AnswerGuard(coverage=quarterly_coverage())
    text = "FY23 की चौथी तिमाही का राजस्व ₹1,711 करोड़ था [S4], लेकिन FY23 के EBITDA की जानकारी उपलब्ध नहीं है। अगला।"
    released, verdicts, _ = stream(guard, text, size=3)
    assert verdicts == [] and released.startswith("FY23 की चौथी तिमाही का राजस्व ₹1,711 करोड़ था [S4]।")
    assert "लेकिन" not in released and "नहीं" not in released
    assert guard.checks[0]["action"] == "clause_dropped"


def test_a_hindi_denial_with_its_opening_already_out_is_ended_there_and_corrected():
    guard = AnswerGuard(coverage=quarterly_coverage())
    text = "तिमाही राजस्व और EBITDA के आंकड़े FY24 के लिए उपलब्ध नहीं हैं [S1]। अगला।"
    released, verdicts, _ = stream(guard, text, size=3)
    assert released.startswith("तिमाही राजस्व") and "नहीं" not in released
    assert released.endswith("… " + quarterly_coverage().correction) and verdicts == ["stop"]
    assert guard.checks[-1]["action"] == "cut"


def test_hindi_hold_reads_what_a_sentence_is_about_and_where_its_clauses_end():
    assert hindi_hold("रिपोर्ट में FY24 की राशि ") == (True, 0)  # a document, no verb yet
    assert hindi_hold("रिपोर्ट में FY24 की राशि दर्ज है, जो ") == (True, len("रिपोर्ट में FY24 की राशि दर्ज है,"))
    assert hindi_hold("रिपोर्ट में FY24 की राशि दर्ज ह") == (True, 0)  # "है" or "हैं" still being written
    assert hindi_hold("वालमोरा का राजस्व ₹7,365 करोड़ था ") == (False, len("वालमोरा का राजस्व ₹7,365 करोड़ था"))
    assert hindi_hold("रिपोर्ट के अनुसार राजस्व ") == (False, 0)  # an attribution
    assert hindi_hold("The Valmora annual report में राजस्व ") == (True, 0)  # English nouns count too
    assert hindi_hold("राजस्व बढ़ा, लेकिन ") == (True, 0)  # a contrast: what the documents lack comes next
    # nuktas and chandrabindus are spelled both ways
    assert hindi_hold("दस्तावेज\u093c ")[0] and hindi_hold("दस्तावेज़ों में राशि हूँ ")[1] > 0
    assert hindi("ज\u093c") == "ज" and hindi("हूँ") == "हूं"


@pytest.mark.parametrize(
    "text",
    [
        "FY25 की जानकारी उपलब्ध नहीं है",
        "FY25 की जानकारी उपलब्ध नही है",
        "यह जानकारी रिपोर्ट में न दी गई है",
        "यह जानकारी अनुपलब्ध है",
        "इसका उल्लेख नहीं किया गया",
        "मुझे पता नहीं है",
        "मुझे खेद है कि यह जानकारी मेरे पास नहीं है",
        "क्षमा करें",
        "माफ़ कीजिए",
        "FY25 ki jankari uplabdh nahi hai",
        "yeh ullekh nahin hai",
        "केवल Q1 का आंकड़ा दिया गया है",
    ],
)
def test_hindi_negations_and_apologies_are_cues(text):
    assert has_denial_cue(text)


@pytest.mark.parametrize(
    "text",
    [
        "वालमोरा का राजस्व ₹7,365 करोड़ था",
        "कंपनी की क्षमता बढ़ी है और नई नीति आई",  # "क्षमता" is not "क्षमा"; "न" inside a word isn't a negation
        "राजस्व की जानकारी उपलब्ध है",  # available: a positive statement (about, but no cue)
    ],
)
def test_plain_hindi_has_no_denial_cue(text):
    assert not has_denial_cue(text)


@pytest.mark.parametrize(
    ("text", "denies"),
    [
        ("FY25 के राजस्व की जानकारी दस्तावेज़ों में नही दी गई है।", True),  # "नही"
        ("यह जानकारी रिपोर्ट में मौजूद नहीं है।", True),
        ("मुझे इसका पता नहीं है।", True),
        ("FY25 ki jankari uplabdh nahi hai.", True),  # romanized
        ("FY25 के राजस्व की जानकारी उपलब्ध है।", False),
        ("कंपनी का राजस्व नहीं बढ़ा।", False),  # a fact about revenue, not about what the documents hold
    ],
)
def test_says_not_covered_in_hindi(text, denies):
    assert says_not_covered(text) is denies


def test_a_hindi_contrast_starts_a_clause():
    assert clauses("Q4 का राजस्व ₹1,711 करोड़ था लेकिन FY23 की जानकारी उपलब्ध नहीं है।") == [
        "Q4 का राजस्व ₹1,711 करोड़ था",
        "लेकिन FY23 की जानकारी उपलब्ध नहीं है।",
    ]
    assert drop_denial("Q4 का राजस्व ₹1,711 करोड़ था [S4], लेकिन FY23 की जानकारी उपलब्ध नहीं है।") == (
        "Q4 का राजस्व ₹1,711 करोड़ था [S4]।"
    )


def test_a_first_sentence_whose_opening_is_out_is_ended_when_it_turns_into_a_denial():
    guard = AnswerGuard(coverage=quarterly_coverage())
    text = "Quarterly revenue and EBITDA figures for the company in FY24 are not provided in the report. More."
    released, verdicts, _ = stream(guard, text)
    assert released.startswith("Quarterly revenue and") and "not provided" not in released
    assert released.endswith("… " + quarterly_coverage().correction) and verdicts == ["stop"]
    assert guard.checks[-1]["action"] == "cut"


def test_drop_denial_keeps_the_answer_before_the_denying_clause():
    assert drop_denial("Q4 revenue was ₹1,933 crore [S2], but the deck doesn't give FY23.") == (
        "Q4 revenue was ₹1,933 crore [S2]."
    )
    assert drop_denial("For FY23, the data are not available in the provided sources.") is None  # too little left


# ------------------------------------------------------------------ live figures (item 2)


def test_a_live_figure_from_nowhere_is_replaced_by_the_honest_line():
    allowed = figures_in(["Revenue from operations was ₹ 7,365 crore in FY24.", "USD to INR today"])
    guard = AnswerGuard(figures=allowed, live_line="I can't look up live data, so I won't guess a figure.")
    released, _, _ = stream(
        guard, "Valmora's FY24 revenue was ₹7,365 crore [S1]. The rupee is at about 83.50 to the dollar today."
    )
    assert (
        released
        == "Valmora's FY24 revenue was ₹7,365 crore [S1]. I can't look up live data, so I won't guess a figure."
    )
    assert guard.checks[0]["check"] == "live_figure"
    assert unsupported_figures("In FY24 (2023-24) the 3 segments grew 13.6% [S1].", {"13.6"}) == []


# ------------------------------------------------------------------ identifiers (item 3)


def test_a_mistyped_code_is_copied_from_the_source():
    known = identifiers_in([f"CIN: {CIN}", "ISIN INE413T01011, BSE 544019, PAN AAACV1234F"])
    assert CIN in known and "INE413T01011" in known and "544019" not in known
    assert nearest_identifier("L24119GJ1994PLC023971", known) == CIN  # one digit off
    assert nearest_identifier("L24119GJ1994PLC023871", known) is None  # right already
    assert nearest_identifier("U72900GJ2012PTC070918", known) is None  # far from all: not a slip
    guard = AnswerGuard(identifiers=known)
    released, _, _ = stream(guard, "Valmora's CIN is L24119GJ1994PLC023971 [S1].", size=5)
    assert released == f"Valmora's CIN is {CIN} [S1]."
    assert guard.checks == [{"check": "identifier", "action": "corrected", "text": "L24119GJ1994PLC023971", "to": CIN}]


# ------------------------------------------------------------------ names (item 10)


def test_a_misheard_name_is_written_as_the_documents_spell_it():
    guard = AnswerGuard(renames={"Wall Mora": "Valmora", "Mora": "Valmora"})
    released, _, _ = stream(guard, "Wall Mora's revenue was ₹7,365 crore [S1]; Mora grew 13.6%.", size=4)
    assert released == "Valmora's revenue was ₹7,365 crore [S1]; Valmora grew 13.6%."


# ------------------------------------------------------------------ length (item 7)


def test_a_short_answer_stops_at_the_sentence_that_reaches_the_word_limit():
    guard = AnswerGuard(max_words=20, max_sentences=3)
    text = (
        "Revenue rose 13.6% to ₹7,365 crore in FY24 [S1]. EBITDA grew faster, by 20.4% to ₹1,545 crore, lifting the "
        "margin to 21.0% [S1]. Profit after tax rose 31.2% [S1]. Net debt fell."
    )
    released, verdicts, _ = stream(guard, text)
    assert verdicts == ["stop"]
    assert released.endswith("lifting the margin to 21.0% [S1].")  # the sentence that reached 20 words, whole
    guard = AnswerGuard(max_words=45, max_sentences=3)
    released, _, _ = stream(
        guard, "Revenue was ₹7,365 crore [S1]. EBITDA was ₹1,545 crore [S1]. PAT was ₹871 crore [S1]. More."
    )
    assert released == "Revenue was ₹7,365 crore [S1]. EBITDA was ₹1,545 crore [S1]. PAT was ₹871 crore [S1]."


def test_without_checks_text_passes_through_as_it_comes():
    guard = AnswerGuard()
    assert guard.feed("Margin 18.").text == "Margin 18."
    assert guard.feed("2% [S1]").text == "2% [S1]"


def test_sentence_ends():
    assert sentence_end("Revenue was Rs. 4,210 crore. Next") == len("Revenue was Rs. 4,210 crore.")
    assert sentence_end("It was 18.2% [S1]. More") == len("It was 18.2% [S1].")
    assert sentence_end("Revenue was 18.2") is None
    assert sentence_end("1. Revenue grew") is None
    assert sentence_end("राजस्व बढ़ा। अब") == len("राजस्व बढ़ा।")


# ------------------------------------------------------------------ short replies cut by the token cap (last round, 5)


def fed(guard: AnswerGuard, text: str, size: int = 4) -> str:
    return "".join(guard.feed(text[i : i + size]).text for i in range(0, len(text), size))


def test_a_short_reply_is_released_in_whole_words():
    guard = AnswerGuard(whole_sentences=True)
    assert fed(guard, "क्या आप किसी विशिष्ट विषय") == "क्या आप किसी विशिष्ट "  # "विषय" may still grow
    assert guard.active


def test_a_reply_cut_by_the_cap_in_its_first_sentence_ends_at_its_last_whole_word():
    """The last real run spoke "क्या आप किसी विशिष्ट विषय" and "या इसक": the cap cut them mid-sentence, the second
    mid-word."""
    guard = AnswerGuard(whole_sentences=True)
    released = fed(guard, "क्या आप किसी विशिष्ट विषय के बारे में या इसक")
    end = guard.finish(truncated=True)
    assert released + end.text == "क्या आप किसी विशिष्ट विषय के बारे में या …"
    assert guard.checks[-1] == {"check": "length", "action": "cut_at_word", "text": released.strip() + " इसक"}


def test_a_reply_cut_by_the_cap_after_a_whole_sentence_ends_there():
    guard = AnswerGuard(whole_sentences=True)
    released = fed(guard, "क्या आप योजना की पात्रता के बारे में पूछ रहे हैं? या इसके लाभ के बारे में जानना")
    assert released == "क्या आप योजना की पात्रता के बारे में पूछ रहे हैं? "  # the second sentence is held whole
    assert guard.finish(truncated=True).text == ""  # and dropped: it was cut
    assert guard.checks[-1]["action"] == "cut_sentence_dropped"


def test_a_reply_that_ends_by_itself_keeps_its_last_words():
    guard = AnswerGuard(whole_sentences=True)
    released = fed(guard, "Do you mean the eligibility rules? Or the benefits")
    assert released + guard.finish().text == "Do you mean the eligibility rules? Or the benefits"


def test_the_first_word_after_a_fixed_prefix_is_lower_cased_when_common():
    guard = AnswerGuard(lower_first=True)
    assert guard.feed("The capital of France is Paris.").text == "the capital of France is Paris."
    guard = AnswerGuard(lower_first=True)
    assert guard.feed("EBITDA stands for earnings").text == "EBITDA stands for earnings"


# ------------------------------------------------------------------ "not covered" answers (item 5)


@pytest.mark.parametrize(
    ("text", "denies"),
    [
        ("The policy does not cover hotel provisions.", True),  # "policy": missed before
        ("[S2] provides Q4 FY23 revenue, but does not give the quarterly breakdown for FY23.", True),
        ("For FY23, the quarterly revenue and EBITDA data are not available in the provided sources.", True),
        ("This is not covered by the policy.", True),
        ("The deck doesn't break down EBITDA by quarter.", True),
        ("FY25 के राजस्व की जानकारी दस्तावेज़ों में नहीं दी गई है।", True),
        ("Revenue did not change much in FY24 [S1].", False),
        ("The report shows revenue does not include other income.", False),  # what the report says
    ],
)
def test_says_not_covered(text, denies):
    assert says_not_covered(text) is denies


@pytest.mark.parametrize(
    ("answer", "question", "declines"),
    [
        ("The Valmora annual report does not mention Valmora's revenue for FY25. [S1]", "Revenue in FY25?", True),
        ("FY24 revenue was ₹7,365 crore [S1], but the documents don't give FY25.", "Revenue in FY25?", True),
        ("FY24 revenue was ₹7,365 crore [S1], but the documents don't give FY25.", "Revenue in FY24 and FY25?", False),
        ("The documents do not specify an EBITDA margin target.", "What is the margin target?", True),
        ("Hotels are capped at ₹7,500 [S1]. The policy does not cover serviced apartments.", "Hotel limits?", False),
        ("Revenue was ₹7,365 crore in FY24 [S1].", "Revenue in FY24?", False),
    ],
)
def test_an_answer_declines_when_it_denies_what_was_asked(answer, question, declines):
    assert answer_declines(answer, [question]) is declines


# ------------------------------------------------------------------ malformed and sign-flipped numbers (polish round)

SOURCES = "| Metric | FY24 | FY23 |\n| EBITDA margin | 21.0% | 19.8% |\n| EBITDA growth | 20.4% | 9.1% |"
HISTORY = "Valmora's EBITDA margin in FY24 was 21.0% [S1]."


@pytest.mark.parametrize(
    ("said", "fixed"),
    [
        ("EBITDA rose 20.-4% [S1].", "EBITDA rose 20.4% [S1]."),  # the last real run
        ("EBITDA rose 20-.4% in FY24.", "EBITDA rose 20.4% in FY24."),
        ("Revenue was ₹ 7,-365 crore.", "Revenue was ₹ 7,365 crore."),
        ("Revenue was ₹ 7,,365 crore.", "Revenue was ₹ 7,365 crore."),
        ("EBITDA grew 20..4%.", "EBITDA grew 20.4%."),
    ],
)
def test_a_sign_or_separator_wedged_into_a_number_is_taken_out(said, fixed):
    for size in (1, 3, 6, 50):  # however the model's pieces cut the number
        guard = AnswerGuard(numbers=number_facts([SOURCES]))
        out, verdicts, _ = stream(guard, said, size)
        assert out == fixed and verdicts == []
        assert [c["check"] for c in guard.checks] == ["number"] and guard.checks[0]["action"] == "repaired"


def test_a_minus_on_a_figure_the_input_states_unsigned_is_dropped():
    """The last real run: a conversation reply reused the history's 21.0% as "-21.0%"."""
    for size in (1, 4, 50):
        guard = AnswerGuard(numbers=number_facts([HISTORY, "So is that a good margin?"]), whole_sentences=True)
        out, _, _ = stream(guard, "Yes, a margin of -21.0% is healthy for the sector.", size)
        assert out == "Yes, a margin of 21.0% is healthy for the sector."
        assert guard.checks == [{"check": "number", "action": "sign_dropped", "text": "-21.0", "to": "21.0"}]


@pytest.mark.parametrize(
    ("sources", "said"),
    [
        ("Net cash flow was (21.0) crore.", "Net cash flow was -21.0 crore."),  # accounting brackets: negative
        ("Margin fell 21.0% in FY24.", "The margin change was -21.0%."),  # stated as a fall
        ("Other income was -3.2 crore.", "Other income was -3.2 crore."),  # signed in the source
        ("Revenue was 7,365 crore.", "Profit was -412 crore."),  # a figure the input doesn't give: not judged
        ("Revenue grew 21.0%.", "Margins declined by -21.0%."),  # the answer itself speaks of a fall
        ("Revenue grew 21.0%.", "Tier-1 cities in 2023-24, and Q4-21.0 aside."),  # hyphens inside words and ranges
    ],
)
def test_signs_that_may_be_right_are_kept(sources, said):
    guard = AnswerGuard(numbers=number_facts([sources]))
    out, _, _ = stream(guard, said, 3)
    assert out == said and guard.checks == []


def test_number_facts_reads_signs_brackets_and_falls():
    facts = number_facts(["Margin 21.0% [S1].", "A loss of 5.2 crore.", "(3.1)", "-4.0", "Revenue 7,365; Tier-1"])
    assert {"21", "7365", "1"} <= facts.unsigned
    assert {"5.2", "3.1", "4"} <= facts.signed


def test_numbers_hold_only_the_token_being_written():
    guard = AnswerGuard(numbers=number_facts([SOURCES]))
    assert guard.active
    assert guard.feed("EBITDA rose ").text == "EBITDA rose "
    assert guard.feed("20.").text == ""  # may still grow: "20.-4"
    assert guard.feed("-4% in").text == "20.4% in"  # whitespace ended it; "in" is no number
    assert guard.finish().text == ""
