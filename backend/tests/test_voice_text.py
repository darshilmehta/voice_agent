"""Speakable chunks, heard text and backchannels (app/services/voice/speech_text.py)."""

from __future__ import annotations

import pytest

from app.services.voice.speech_text import (
    SpeechChunker,
    SpokenChunk,
    heard_text,
    is_acknowledgement,
    is_backchannel,
    is_filler,
    looks_garbled,
    real_words,
    spoken_text,
    transcript_garbled,
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
    text = "Yes, margin was 18.2% [S1]. Revenue grew 34% in FY24 [S2]. Margins improved."
    assert chunk_stream(text) == [
        "Yes, margin was 18.2%.",  # "Yes," alone is too short to be the first chunk
        "Revenue grew 34% in FY24.",
        "Margins improved.",
    ]


def test_first_chunk_is_cut_after_five_words_without_an_earlier_boundary():
    text = "The company reported an EBITDA margin of 18.2% in FY24 [S1]. Next."
    assert chunk_stream(text) == ["The company reported an EBITDA", "margin of 18.2% in FY24.", "Next."]


def test_clause_boundary_within_five_words_ends_the_first_chunk():
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
    assert chunk_stream(text) == ["वित्त वर्ष 2024 में EBITDA", "मार्जिन 18.2% था।", "राजस्व 34% बढ़ा।"]


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


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("It was worth about Rs. 5,000 crore.", "It was worth about"),  # the 5-word cut backs off before "Rs."
        ("Sales in the U.S. grew 12% in FY24.", "Sales in the U.S. grew"),  # initialisms don't end a sentence
        ("For example, e.g. margins rose.", "For example,"),
    ],
)
def test_never_cut_after_an_abbreviation(text, first):
    for piece in (1, 3, 7):
        chunks = chunk_stream(text, piece=piece)
        assert chunks[0] == first, chunks
        assert not any(c.endswith(("Rs.", "U.S.", "e.g.")) for c in chunks), chunks
        assert " ".join(chunks) == text


def test_list_numbers_are_not_sentences_and_short_chunks_do_not_use_up_the_budget():
    text = "Yes. Two points: 1. Revenue grew 34%. 2. Margins improved to 18.2%. Third one here."
    chunks = chunk_stream(text, max_sentences=2)
    assert "1." not in chunks and "2." not in chunks  # "1." never stands alone as a chunk
    assert chunks[0] == "Yes."  # one word: not counted as one of the two spoken sentences
    assert chunks[-1].endswith("Margins improved to 18.2%.") and "Third" not in " ".join(chunks)


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


# What Whisper writes for a short "mm-hmm" and other hums (seen in the end-to-end run: "M M", then "MM").
HUMS = ["M M", "MM", "m-m", "Mm-hmm.", "Mhmm.", "hmm", "Hmmm?", "mhm", "uh huh", "Um.", "उम्म", "हम्म", "हूँ", "हूं"]
HUMS += ["MMHUM", "mhum", "Mmhum."]  # Whisper's spelling of a voiced "mm-hmm"
HUMS += ["MAMMA.", "Mama.", "Mmm-ma", "M-ma", "Mamma"]  # Kokoro's "Mm-hmm." as Whisper wrote it (last round, item 6)


@pytest.mark.parametrize("text", HUMS)
def test_hums_are_backchannels_and_fillers_with_no_real_words(text):
    assert is_backchannel(text, 2) and is_filler(text) and real_words(text) == 0


@pytest.mark.parametrize("text", ["Mamata", "mammal", "ma'am", "MAMMA, stop.", "Mama ji kahan hain?"])
def test_words_like_the_mm_hmm_spellings_stay_words(text):
    assert not is_filler(text) and real_words(text) >= 1 and not is_backchannel(text, 2)


@pytest.mark.parametrize(
    "text",
    [
        "अच्छा, ठीक है",  # three words, two acknowledgements: only non-acknowledgement words count
        "achha theek hai",
        "okay okay okay",
        "Yeah, yeah, right, sure.",
        "Mm-hmm, okay.",
        "yes please",  # an acknowledgement plus one other word (within backchannel_max_words)
    ],
)
def test_acknowledgements_count_once_against_the_word_limit(text):
    assert is_backchannel(text, 2)


@pytest.mark.parametrize(
    "text",
    [
        "Stop.",
        "Wait",
        "No wait, I meant the revenue",
        "What about FY23?",
        "रुको",
        "okay stop",  # an interruption cue is never a backchannel, however short
        "haan, kya?",
        "FY23",  # no acknowledgement at all
        "okay so tell me more",  # three other words: over the limit
    ],
)
def test_not_backchannels(text):
    assert not is_backchannel(text, 2)


@pytest.mark.parametrize("text", ["Yeah, right.", "Okay.", "achha theek hai", "Mm-hmm, okay.", "हाँ जी", "MMHUM"])
def test_acknowledgements_alone_are_not_questions(text):
    assert is_acknowledgement(text)


@pytest.mark.parametrize("text", ["yes please", "Yeah, but what about FY23?", "Stop.", "", "you"])
def test_anything_more_than_acknowledgements_is_not_one(text):
    assert not is_acknowledgement(text)


def test_fillers_are_only_non_lexical_sounds():
    assert is_filler("Hmm.") and is_filler("Mm-hmm") and is_filler("uh, um")
    assert not is_filler("Okay.") and not is_filler("") and not is_filler("hmm what")
    assert real_words("No wait") == 2 and real_words("M M") == 0 and real_words("you") == 0


# ------------------------------------------------------------------ citation markers as words (quality round, item 6)


@pytest.mark.parametrize(
    ("raw", "said"),
    [
        ("[W1] mentions that the rupee rose 0.3% today.", "A web source mentions that the rupee rose 0.3% today."),
        ("However, [W1] mentions that the rupee rose.", "However, a web source mentions that the rupee rose."),
        ("According to [W1], the rupee rose.", "According to a web source, the rupee rose."),
        ("The rupee rose, as reported by [W2].", "The rupee rose, as reported by a web source."),
        ("[S1][S2] show that revenue grew.", "The sources show that revenue grew."),
        ("[W1] and [W2] say the market fell.", "Web sources say the market fell."),
        (
            "The report says margins rose [S1], while [W1] notes the stock fell.",
            "The report says margins rose, while a web source notes the stock fell.",
        ),
        ("Revenue was 18.2% [S1]. EBITDA rose [S1][S2].", "Revenue was 18.2%. EBITDA rose."),  # citations: dropped
        ("Revenue rose 12% [S1] and profit fell 5% [S2].", "Revenue rose 12% and profit fell 5%."),
        ("[W1] के अनुसार रुपया मज़बूत हुआ।", "एक स्रोत के अनुसार रुपया मज़बूत हुआ।"),
        ("[W1] बताता है कि रुपया मज़बूत हुआ।", "एक स्रोत बताता है कि रुपया मज़बूत हुआ।"),
        ("राजस्व 7,365 करोड़ रुपये था [S1]।", "राजस्व 7,365 करोड़ रुपये था।"),
    ],
)
def test_a_citation_used_as_a_word_is_said_and_any_other_dropped(raw, said):
    assert spoken_text(raw) == said


def test_a_marker_opening_a_chunk_mid_sentence_is_said_in_lower_case():
    chunks = chunk_stream("However, [W1] mentions that the rupee rose 0.3% today. [W2] adds that it closed higher.")
    assert chunks == [
        "However, a web source mentions that the rupee",
        "rose 0.3% today.",
        "A web source adds that it closed higher.",
    ]


# ------------------------------------------------------------------ garbled transcripts (quality round, item 8)


@pytest.mark.parametrize(
    "text",
    [  # from the real run, with a film playing
        "आब आब आब आब आब आब आब आब आब आब आब आब",
        "It just takes 1 profit from only 1 army... Which most likely means 1 profit from only 1 army... "
        "Just 1 employee sign here... Just 1 employee sign here... Just 1 employee sign here...",
        "अप बवबवववववववववववववववव",
        "आश़््गें। आश़््गें।",
        "यह सब पड़़ा। शावन बढ़ लेरा है। उतो चण दिखे तुछु। तुछु। तुछु। तुछु। तुछु।",
        "Let's start with thank you and what I want to wish always away my understandgradeelle걱",
        "Is used by runsTAIC traffic",
        "Our highlight by recording these songs for its меня dollars series",
    ],
)
def test_garbled_transcripts(text):
    assert looks_garbled(text)


@pytest.mark.parametrize(
    "text",
    [
        "What was the EBITDA margin in FY24?",
        "No no no, I meant FY23",
        "okay okay okay",
        "Mm-hmm.",
        "FY24 में EBITDA margin क्या था?",
        "वालमोरा की FY24 की तिमाही आय का चार्ट दिखाओ",
        "कृपया बताएं कि पिछले साल कंपनी का शुद्ध लाभ कितना था?",
        "What is the hotel limit for our L3 employee in a tier 1 city?",
        "What's the café's revenue?",
        "इसे टेबल में दिखाओ",
        "What is the profit that Wilmura made in FY24?",  # misheard, but speech: answered (item 10)
    ],
)
def test_ordinary_speech_is_not_garbled(text):
    assert not looks_garbled(text)


class _Transcript:
    def __init__(self, text: str, **fields: float) -> None:
        self.text = text
        self.__dict__.update(fields)


def test_the_recognizers_confidence_counts_when_it_reports_one():
    question = "What was the EBITDA margin in FY24?"
    assert not transcript_garbled(_Transcript(question))  # no confidence fields: the text decides
    assert not transcript_garbled(_Transcript(question, avg_logprob=-0.4, no_speech_prob=0.1))
    assert transcript_garbled(_Transcript(question, avg_logprob=-1.4))
    assert transcript_garbled(_Transcript(question, avg_logprob=-1.1, no_speech_prob=0.8))  # Whisper's "silence"
    assert not transcript_garbled(_Transcript(question, avg_logprob=-1.1, no_speech_prob=0.2))
    assert transcript_garbled(_Transcript(question, compression_ratio=3.1))  # a repetition loop
    assert transcript_garbled(_Transcript(question, confidence=0.2))
    assert transcript_garbled(_Transcript("आब आब आब आब आब आब", avg_logprob=-0.2))
