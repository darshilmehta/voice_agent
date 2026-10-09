"""A long spoken sentence is synthesized in clauses (voice/session.py ``synthesis_pieces``, DESIGN §9.5): its first
clause is heard while the rest is synthesized, so a long second sentence leaves no silence after the first chunk."""

from __future__ import annotations

import pytest

from app.services.voice.session import SPLIT_MIN_PART, synthesis_pieces

from .test_voice_api import QUESTION, VoiceClient, of, voice  # noqa: F401  (voice is a fixture)

LONG = "Revenue was Rs 7,365 crore, up 13.6% from Rs 6,482 crore in FY23, driven mainly by Specialty Chemicals."


def test_a_long_sentence_is_cut_at_the_clause_nearest_its_middle_and_again_while_long():
    assert synthesis_pieces(LONG) == [
        "Revenue was Rs 7,365 crore,",
        "up 13.6% from Rs 6,482 crore in FY23,",  # the first part (12 words) cut again
        "driven mainly by Specialty Chemicals.",
    ]


def test_hindi_clauses_and_long_parts_are_cut_again():
    text = "Valmora की आय 7,365 करोड़ रुपये रही, जो पिछले साल से 13.6% अधिक है, और इसका मुख्य कारण स्पेशियलिटी केमिकल्स है।"
    assert synthesis_pieces(text) == [
        "Valmora की आय 7,365 करोड़ रुपये रही,",
        "जो पिछले साल से 13.6% अधिक है,",
        "और इसका मुख्य कारण स्पेशियलिटी केमिकल्स है।",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The EBITDA margin was 18.2%, up from 17.1%.",  # short: one piece
        "Revenue grew from Rs 6,482 crore to Rs 7,365 crore over the year to March 2024.",  # no clause
        "Yes, revenue grew from Rs 6,482 crore to Rs 7,365 crore over the year ending March 2024.",  # too short a part
    ],
)
def test_sentences_without_a_good_clause_stay_whole(text):
    assert synthesis_pieces(text) == [text]


def test_pieces_are_never_too_short():
    for piece in synthesis_pieces(LONG + " " + LONG):
        assert len(piece.split()) >= SPLIT_MIN_PART


def test_a_spoken_long_sentence_comes_as_two_chunks(voice):  # noqa: F811
    voice.fakes.llm.reply = LONG + " [S1]"
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start(language="en")
        c.say(QUESTION)
        got = c.until("agent_message")
        c.send("end")
    chunks = [m["text"] for m in of(got, "audio_chunk")]
    assert chunks[0] == "Revenue was Rs 7,365 crore,"  # the first chunk (the chunker's first clause)
    assert " ".join(chunks) == LONG
    assert len(chunks) == 3  # the rest of the sentence, in two clauses
