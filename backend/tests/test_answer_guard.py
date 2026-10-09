"""The answer's checks while it streams (services/answer_guard.py, quality round): denials the evidence contradicts,
live figures from nowhere, mistyped codes, misheard names, and the length of short answers. Pure: the model's stream
is a string fed in small pieces."""

from __future__ import annotations

import pytest

from app.services.answer_guard import (
    AnswerGuard,
    Coverage,
    Released,
    drop_denial,
    evidence_units,
    figures_in,
    identifiers_in,
    nearest_identifier,
    sentence_end,
    terms,
    unsupported_figures,
)
from app.services.sources import answer_declines, says_not_covered

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
    guard = AnswerGuard(coverage=quarterly_coverage())
    assert guard.feed("वालमोरा का राजस्व हर तिमाही में बढ़ा और चौथी तिमाही में ").text == ""  # Hindi: "नहीं" comes last


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
