"""Text on its way to and from the speaker (docs/DESIGN.md §3.3 b-c): speakable chunks, what was heard, backchannels.

    answer deltas → SpeechChunker → chunks for TTS: the first at the first clause boundary (or after 5 words, so audio
                    starts early, §9.5), then whole sentences; [S#] markers and markdown removed; at most
                    voice.max_spoken_sentences sentences are spoken (the rest stays on screen)
    chunks + played_ms → heard_text: fully played chunks + the share (by words) of the chunk playing at played_ms
    utterance text → is_backchannel: "mm-hmm", "M M", "okay", "haan", "achha theek hai" … (only words that aren't
                    acknowledgements count against backchannel_max_words; interruption cues never pass)

English and Hindi: sentence ends include the danda (।, ॥); words are whitespace-separated in both scripts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..sources import strip_markers, trim_open_marker

FIRST_CHUNK_MAX_WORDS = 5  # a short first chunk starts audio early: 5 words ≈ 0.5 s of Kokoro on CPU, 8 ≈ 0.63 s (§9.5)
FIRST_CHUNK_MIN_WORDS = 2  # ... but a clause boundary after one word ("So,") doesn't end it
CHUNK_MAX_WORDS = 30  # a longer sentence is cut at its last clause boundary (or here) so speech keeps flowing

SENTENCE_END = ".!?।॥"
CLAUSE_END = ",;:"
_CLOSERS = "\"'\u201d\u2019)]\u00bb"
# Words whose trailing period isn't a sentence end ("Rs. 4,210 crore"); initialisms ("U.S.", "e.g.") count too.
_ABBREVIATIONS = {"rs", "mr", "mrs", "ms", "dr", "vs", "e.g", "i.e", "etc", "approx", "inc", "ltd", "co", "st", "fig"}
_INITIALISM = re.compile(r"(?:[A-Za-z]\.){2,}")
_LIST_NUMBER = re.compile(r"\(?\d{1,2}[.)]")  # "1." "2)" opening a list item, not ending a sentence
_MARKDOWN = re.compile(r"[*`#]+")
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\S+")
_BRACKETED = re.compile(r"\[[^\]]*\]")


def spoken_text(raw: str) -> str:
    """What TTS should say for a piece of answer text: no [S#] markers, no markdown, single spaces. Empty if nothing
    speakable (only punctuation or markers) remains."""
    text = _SPACE.sub(" ", _MARKDOWN.sub("", strip_markers(raw))).strip()
    return text if any(ch.isalnum() for ch in text) else ""


def _mask_brackets(text: str) -> str:
    """``text`` with every [...] blanked out (same length): citation markers are neither words nor boundaries, so
    "[S3, S4]" can't end a chunk at its comma."""
    return _BRACKETED.sub(lambda m: " " * len(m.group(0)), text)


def _words(text: str) -> list[re.Match[str]]:
    """Word tokens (a "।" or "—" standing alone, e.g. after a blanked marker, is not a word)."""
    return [m for m in _WORD.finditer(text) if any(ch.isalnum() for ch in m.group(0))]


def _is_abbreviation(token: str) -> bool:
    """A token whose final period doesn't end a sentence: "Rs.", "e.g.", "U.S."."""
    word = token.strip(_CLOSERS + "(\"'")
    return word.endswith(".") and (word.lower().rstrip(".") in _ABBREVIATIONS or bool(_INITIALISM.fullmatch(word)))


@dataclass(frozen=True, slots=True)
class _Boundary:
    end: int  # cut position (after the punctuation and any closing quote)
    sentence: bool


class SpeechChunker:
    """Cuts streamed answer text into chunks to synthesize, as soon as each is complete.

    ``feed(delta)`` returns the chunks completed by that delta, ``flush()`` the rest at the end of the answer. A
    boundary is punctuation followed by whitespace (so "18.2%" and "4,210" never split), or the end of the answer.
    """

    def __init__(self, *, max_sentences: int | None = None) -> None:
        self._buffer = ""
        self._first = True
        self._sentences = 0
        self.max_sentences = max_sentences
        self.spoken: list[str] = []  # every chunk returned, in order

    @property
    def exhausted(self) -> bool:
        """The spoken-sentence limit is reached: later text is shown, not spoken."""
        return self.max_sentences is not None and self._sentences >= self.max_sentences

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while (cut := self._next_cut()) is not None:
            raw, self._buffer = self._buffer[: cut.end], self._buffer[cut.end :]
            self._emit(raw, cut.sentence, out)
        if final:
            raw, self._buffer = trim_open_marker(self._buffer), ""
            self._emit(raw, True, out)
        return out

    def _emit(self, raw: str, sentence: bool, out: list[str]) -> None:
        if self.exhausted:
            return
        text = spoken_text(raw)
        if not text:
            return
        self._first = False
        if sentence and len(text.split()) >= 2:  # "1." or "Yes." alone doesn't use up the spoken-sentence budget
            self._sentences += 1
        out.append(text)
        self.spoken.append(text)

    def _visible(self) -> str:
        """The buffer up to an unfinished citation marker at its end ("… 18.2% [S"), with markers blanked out:
        positions match the buffer, so cuts apply to it directly."""
        buf = self._buffer
        open_at = buf.rfind("[")
        return _mask_brackets(buf[:open_at] if open_at > buf.rfind("]") else buf)

    def _boundaries(self, text: str) -> list[_Boundary]:
        found = []
        for i, ch in enumerate(text):
            if ch not in SENTENCE_END and ch not in CLAUSE_END:
                continue
            j = i + 1
            while j < len(text) and text[j] in _CLOSERS:
                j += 1
            if j >= len(text) or not text[j].isspace():
                continue  # "18.2%", "4,210", or not known yet what follows
            sentence = ch in SENTENCE_END
            if ch == ".":
                prev = text[: i + 1].split()
                if prev and (_is_abbreviation(prev[-1]) or _LIST_NUMBER.fullmatch(prev[-1])):
                    continue  # "Rs. 5,000", "U.S. sales", a list item opening with "1."
            found.append(_Boundary(j, sentence))
        return found

    @staticmethod
    def _word_cut(words: list[re.Match[str]], limit: int) -> _Boundary:
        """Cut after at most ``limit`` words, but never right after an abbreviation ("… about Rs." | "5,000")."""
        n = limit
        while n > 1 and _is_abbreviation(words[n - 1].group(0)):
            n -= 1
        return _Boundary(words[n - 1].end(), False)

    def _next_cut(self) -> _Boundary | None:
        text = self._visible()
        words = _words(text)
        if not words:
            return None
        if self._first:
            for b in self._boundaries(text):
                n = len(_words(text[: b.end]))
                if n > FIRST_CHUNK_MAX_WORDS:
                    break
                if n >= (1 if b.sentence else FIRST_CHUNK_MIN_WORDS):
                    return b
            if len(words) > FIRST_CHUNK_MAX_WORDS:  # the last word is complete once the next has started
                return self._word_cut(words, FIRST_CHUNK_MAX_WORDS)
            return None
        boundaries = self._boundaries(text)
        for b in boundaries:
            if b.sentence:
                if len(_words(text[: b.end])) <= CHUNK_MAX_WORDS:
                    return b
                break
        if len(words) > CHUNK_MAX_WORDS:
            clauses = [b for b in boundaries if not b.sentence and len(_words(text[: b.end])) <= CHUNK_MAX_WORDS]
            return clauses[-1] if clauses else self._word_cut(words, CHUNK_MAX_WORDS)
        return None


# ------------------------------------------------------------------ what was heard


@dataclass(frozen=True, slots=True)
class SpokenChunk:
    """One synthesized chunk of an agent turn, as sent to the client."""

    index: int
    text: str
    start_ms: float  # position in the turn's audio (sum of the previous chunks' durations)
    duration_ms: float

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


def heard_text(chunks: Sequence[SpokenChunk], played_ms: float) -> str:
    """What the user heard of a turn whose audio played for ``played_ms``: every fully played chunk, plus the share of
    the chunk playing at ``played_ms`` in proportion to its duration, counted in whole words (rounded down)."""
    words: list[str] = []
    for chunk in chunks:
        chunk_words = chunk.text.split()
        if played_ms >= chunk.end_ms:
            words += chunk_words
            continue
        if played_ms > chunk.start_ms and chunk.duration_ms > 0:
            share = (played_ms - chunk.start_ms) / chunk.duration_ms
            words += chunk_words[: math.floor(len(chunk_words) * share)]
        break
    return " ".join(words)


# ------------------------------------------------------------------ backchannels

# Acknowledgements that mean "go on", in English, romanized Hindi and Devanagari. "stop", "wait", "no" are not here.
BACKCHANNELS = {
    "mm", "mhm", "mhmm", "mmhm", "mm-hm", "mm-hmm", "mmhmm", "hmm", "hm", "hmm-hmm", "uh-huh", "uh huh", "uhhuh",
    "um", "uh", "ah", "oh", "aha",
    "ok", "okay", "yeah", "yes", "yep", "yup", "right", "sure", "alright", "all right",
    "i see", "got it", "cool", "nice", "great", "fine", "wow", "really", "thanks", "thank you", "go on",
    "haan", "han", "haa", "ha", "haanji", "haan ji", "ji", "ji haan", "achha", "acha", "accha", "achcha", "theek",
    "thik", "theek hai", "thik hai", "sahi", "sahi hai", "bilkul",
    "हाँ", "हां", "हा", "जी", "जी हाँ", "जी हां", "हाँ जी", "हां जी", "अच्छा", "अच्छा जी", "ठीक", "ठीक है", "सही",
    "सही है", "बिल्कुल", "ओके",
}  # fmt: skip
# Words that make an utterance more than an acknowledgement, however short ("okay stop", "yes but…", "haan, kya?").
INTERRUPTION_CUES = {
    "stop", "wait", "no", "nope", "not", "but", "actually", "sorry", "hold", "pause", "excuse", "repeat", "again",
    "what", "why", "how", "when", "where", "which", "who",
    "nahi", "nahin", "ruko", "ruk", "ruka", "bas", "kya", "kyon", "kyun", "kaise", "lekin",
    "नहीं", "नही", "रुको", "रुकिए", "रुक", "बस", "क्या", "क्यों", "कैसे", "लेकिन",
}  # fmt: skip
# What Whisper tends to write for noise or breath rather than speech: treated as no words at all.
NOISE_TRANSCRIPTS = {"you", "thanks for watching", "thank you for watching", "subtitles by the amara.org community"}
# Non-lexical sounds as Whisper writes them, one token at a time: "M M", "MM", "Mhmm", "hmm", "MMHUM" (a voiced
# "mm-hmm"), "uh-huh", "um", "ah".
_HUM = re.compile(r"[mh]*m[mh]*|m+h+u+m+|u+[hm]+|h+u+h+|a+h+|o+h+|e+r+m*")
_HUM_HI = {"हम", "हम्म", "हम्मम", "ह्म", "ह्म्म", "हूँ", "हूं", "हुं", "हुँ", "उम", "उम्म", "उं", "उँ", "अं", "अँ", "ऊं", "ऊँ"}
_PUNCT = re.compile(r"[^\w\s'\-ऀ-ॿ]|[।॥]")
_REPEAT = re.compile(r"(.)\1{2,}")


def normalize_utterance(text: str) -> str:
    """Lower case, no punctuation, stretched letters squeezed ("Hmmmm." → "hmm"), Whisper's noise phrases → ""."""
    words = [_REPEAT.sub(r"\1\1", w.strip("-'")) for w in _PUNCT.sub(" ", text.casefold()).split()]
    phrase = " ".join(w for w in words if w)
    return "" if phrase in NOISE_TRANSCRIPTS else phrase


def _is_hum(token: str) -> bool:
    parts = [p for p in token.split("-") if p]
    return bool(parts) and all(_HUM.fullmatch(p) or p in _HUM_HI for p in parts)


def real_words(text: str) -> int:
    """Words in a transcript that are neither noise nor hums ("M M" → 0, "No wait" → 2)."""
    return sum(not _is_hum(w) for w in normalize_utterance(text).split())


def is_filler(text: str) -> bool:
    """Only non-lexical sounds ("hmm", "mm-hmm", "M M", "उम्म"): not worth an answer even when the agent is silent.
    Lexical acknowledgements ("okay", "yes", "haan") are left to the answer pipeline."""
    words = normalize_utterance(text).split()
    return bool(words) and all(_is_hum(w) for w in words)


def _acknowledgement_cover(words: list[str]) -> tuple[int, list[str]]:
    """(acknowledgements found, the words that aren't part of one). Phrases ("theek hai", "ठीक है") count once."""
    found, others, i = 0, [], 0
    while i < len(words):
        for size in (3, 2, 1):
            if " ".join(words[i : i + size]) in BACKCHANNELS or (size == 1 and _is_hum(words[i])):
                found, i = found + 1, i + size
                break
        else:
            others.append(words[i])
            i += 1
    return found, others


def is_acknowledgement(text: str) -> bool:
    """Only acknowledgements and hums ("Yeah, right.", "okay", "achha theek hai", "mm-hmm"), nothing else: not a
    question, so never answered as one (the answer would be "I couldn't find that in the documents")."""
    words = normalize_utterance(text).split()
    if not words:
        return False
    _, others = _acknowledgement_cover(words)
    return not others


def is_backchannel(text: str, max_words: int) -> bool:
    """True for an utterance that only acknowledges, however many times ("mm-hmm", "M M", "okay okay", "achha theek
    hai", "अच्छा, ठीक है"), or has no words at all. Only words that aren't acknowledgements count against ``max_words``:
    an acknowledgement with up to that many other words is still one ("yes please"), unless one of them is an
    interruption cue ("okay stop", "haan, kya?"). Without any acknowledgement it isn't one ("Stop.", "FY23?")."""
    words = normalize_utterance(text).split()
    if not words:
        return True
    found, others = _acknowledgement_cover(words)
    if not others:
        return True
    if not found or any(w in INTERRUPTION_CUES for w in others):
        return False
    return len(others) <= max_words
