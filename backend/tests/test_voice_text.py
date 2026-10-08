"""Speakable chunks, heard text and backchannels (app/services/voice/speech_text.py)."""

from __future__ import annotations

import pytest

from app.services.voice.speech_text import (
    SpeechChunker,
    SpokenChunk,
    heard_text,
    is_backchannel,
    is_filler,
    spoken_text,
)


def chunk_stream(text: str, piece: int = 3, **kw) -> list[str]:
    """Feed ``text`` in pieces of ``piece`` characters, as the LLM streams it."""
    chunker = SpeechChunker(**kw)
    out: list[str] = []
    for i in range(0, len(text), piece):
        out += chunker.feed(text[i : i + piece])
    return out + chunker.flush()


# ------------------------------------------------------------------ chunking


def test_first_chunk_is_the_first_clause_then_whole_sentences():
    text = "Yes, the EBITDA margin was 18.2% [S1]. Revenue grew 34% in FY24 [S2]. Margins improved."
    assert chunk_stream(text) == [
        "Yes, the EBITDA margin was 18.2%.",  # "Yes," alone is too short to be the first chunk
        "Revenue grew 34% in FY24.",
        "Margins improved.",
    ]


def test_first_chunk_is_cut_after_eight_words_without_a_boundary():
    text = "The company reported an EBITDA margin of eighteen point two percent in FY24."
    chunks = chunk_stream(text)
    assert chunks[0] == "The company reported an EBITDA margin of eighteen"
    assert chunks[1:] == ["point two percent in FY24."]


def test_clause_boundary_within_eight_words_ends_the_first_chunk():
    assert chunk_stream("In FY24, revenue grew 34% to 4,210 crore. Done.") == [
        "In FY24,",
        "revenue grew 34% to 4,210 crore.",
        "Done.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Margin was 18.2% and revenue 4,210 crore in FY24 overall.",  # decimals and thousands never split
        "It was Rs. 4,210 crore in total for the year.",  # abbreviation
    ],
)
def test_numbers_and_abbreviations_do_not_split(text):
    chunks = chunk_stream(text, piece=1)
    assert " ".join(chunks) == text
    assert all(not c.endswith(("18.", "4,", "Rs.")) for c in chunks)


def test_markers_are_never_spoken_even_when_split_across_deltas():
    text = "Margin 18.2% [S1][S2] and growth 34% [S3, S4]. Next [S"
    for piece in (1, 2, 5):
        chunks = chunk_stream(text, piece=piece)
        assert "".join(chunks).count("[") == 0, chunks
        assert chunks[-1] == "Next"


def test_hindi_sentences_end_at_the_danda():
    text = "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था [S1]। राजस्व 34% बढ़ा।"
    assert chunk_stream(text) == ["वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था।", "राजस्व 34% बढ़ा।"]


def test_markdown_is_not_spoken():
    assert spoken_text("**18.2%** is the `margin` # note") == "18.2% is the margin note"
    assert spoken_text(" [S1]. ") == ""


def test_long_sentences_are_cut_at_a_clause():
    first = "Sure."
    long = " ".join(f"word{i}," if i == 20 else f"word{i}" for i in range(40)) + "."
    chunks = chunk_stream(f"{first} {long}")
    assert chunks[0] == "Sure."
    assert chunks[1].endswith("word20,") and len(chunks[1].split()) == 21
    assert " ".join(chunks) == f"{first} {long}"


def test_only_the_first_sentences_are_spoken():
    chunks = chunk_stream("One is here. Two is here. Three is here. Four is here.", max_sentences=2)
    assert chunks == ["One is here.", "Two is here."]


def test_text_without_punctuation_is_flushed_at_the_end():
    assert chunk_stream("eighteen point two") == ["eighteen point two"]


# ------------------------------------------------------------------ heard text


CHUNKS = [
    SpokenChunk(0, "The EBITDA margin was 18.2%.", 0, 500),
    SpokenChunk(1, "Revenue grew 34% in FY24.", 500, 500),
]


@pytest.mark.parametrize(
    ("played_ms", "heard"),
    [
        (0, ""),
        (99, ""),  # less than one word's share
        (100, "The"),
        (499, "The EBITDA margin was"),
        (500, "The EBITDA margin was 18.2%."),
        (700, "The EBITDA margin was 18.2%. Revenue grew"),  # 40% of 5 words
        (1000, "The EBITDA margin was 18.2%. Revenue grew 34% in FY24."),
        (5000, "The EBITDA margin was 18.2%. Revenue grew 34% in FY24."),
    ],
)
def test_heard_text_is_played_chunks_plus_a_proportional_share(played_ms, heard):
    assert heard_text(CHUNKS, played_ms) == heard


# ------------------------------------------------------------------ backchannels


@pytest.mark.parametrize(
    "text",
    ["Mm-hmm.", "Hmmmm...", "Okay.", "OK, okay", "Yeah", "uh-huh", "Haan ji", "achha", "ठीक है।", "हाँ", "", "you"],
)
def test_backchannels(text):
    assert is_backchannel(text, 2)


@pytest.mark.parametrize(
    "text",
    ["Stop.", "Wait", "No wait, I meant the revenue", "okay okay okay", "What about FY23?", "रुको", "okay stop"],
)
def test_not_backchannels(text):
    assert not is_backchannel(text, 2)


def test_fillers_are_only_non_lexical_sounds():
    assert is_filler("Hmm.") and is_filler("Mm-hmm") and is_filler("uh, um")
    assert not is_filler("Okay.") and not is_filler("") and not is_filler("hmm what")
