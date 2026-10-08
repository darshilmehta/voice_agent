"""Text on its way to and from the speaker (docs/DESIGN.md §3.3 b-c): speakable chunks, what was heard, backchannels.

    answer deltas → SpeechChunker → chunks for TTS: the first at the first clause boundary (or after 8 words, so audio
                    starts early, §9.5), then whole sentences; [S#] markers and markdown removed; at most
                    voice.max_spoken_sentences sentences are spoken (the rest stays on screen)
    chunks + played_ms → heard_text: fully played chunks + the share (by words) of the chunk playing at played_ms
    utterance text → is_backchannel: "mm-hmm", "okay", "haan", "achha" … (at most backchannel_max_words words)

English and Hindi: sentence ends include the danda (।, ॥); words are whitespace-separated in both scripts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..sources import strip_markers, trim_open_marker

FIRST_CHUNK_MAX_WORDS = 8  # the first chunk is short so the first audio comes early (§9.5)
FIRST_CHUNK_MIN_WORDS = 2  # ... but a clause boundary after one word ("So,") doesn't end it
CHUNK_MAX_WORDS = 30  # a longer sentence is cut at its last clause boundary (or here) so speech keeps flowing

SENTENCE_END = ".!?।॥"
CLAUSE_END = ",;:"
_CLOSERS = "\"'\u201d\u2019)]\u00bb"
# Words whose trailing period isn't a sentence end ("Rs. 4,210 crore").
_ABBREVIATIONS = {"rs", "mr", "mrs", "ms", "dr", "vs", "e.g", "i.e", "etc", "approx", "inc", "ltd", "co", "st", "fig"}
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
        if sentence:
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
                prev = text[:i].split()
                if prev and prev[-1].lower().rstrip(".") in _ABBREVIATIONS:
                    continue
            found.append(_Boundary(j, sentence))
        return found

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
            if len(words) > FIRST_CHUNK_MAX_WORDS:  # the 8th word is complete once a 9th has started
                return _Boundary(words[FIRST_CHUNK_MAX_WORDS - 1].end(), False)
            return None
        boundaries = self._boundaries(text)
        for b in boundaries:
            if b.sentence:
                if len(_words(text[: b.end])) <= CHUNK_MAX_WORDS:
                    return b
                break
        if len(words) > CHUNK_MAX_WORDS:
            clauses = [b for b in boundaries if not b.sentence and len(_words(text[: b.end])) <= CHUNK_MAX_WORDS]
            return clauses[-1] if clauses else _Boundary(words[CHUNK_MAX_WORDS - 1].end(), False)
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
    "mm", "mhm", "mm-hm", "mm-hmm", "mmhmm", "hmm", "hm", "uh-huh", "uh huh", "uhhuh", "um", "uh", "ah", "oh", "aha",
    "ok", "okay", "ok ok", "okay okay", "yeah", "yes", "yep", "yup", "right", "sure", "alright", "all right",
    "i see", "got it", "cool", "nice", "great", "fine", "wow", "really", "thanks", "thank you", "go on",
    "haan", "han", "haa", "ha", "haanji", "haan ji", "ji", "ji haan", "achha", "acha", "accha", "achcha", "theek",
    "thik", "theek hai", "thik hai", "sahi", "sahi hai", "bilkul",
    "हाँ", "हां", "हा", "जी", "जी हाँ", "जी हां", "हाँ जी", "हां जी", "अच्छा", "अच्छा जी", "ठीक", "ठीक है", "सही",
    "सही है", "बिल्कुल", "हम्म", "हम", "ओके",
}  # fmt: skip
# What Whisper tends to write for noise or breath rather than speech: treated as no words at all.
NOISE_TRANSCRIPTS = {"you", "thanks for watching", "thank you for watching", "subtitles by the amara.org community"}
_PUNCT = re.compile(r"[^\w\s'\-ऀ-ॿ]|[।॥]")
_REPEAT = re.compile(r"(.)\1{2,}")


def normalize_utterance(text: str) -> str:
    """Lower case, no punctuation, stretched letters squeezed ("Hmmmm." → "hmm"), Whisper's noise phrases → ""."""
    words = [_REPEAT.sub(r"\1\1", w.strip("-'")) for w in _PUNCT.sub(" ", text.casefold()).split()]
    phrase = " ".join(w for w in words if w)
    return "" if phrase in NOISE_TRANSCRIPTS else phrase


FILLERS = {"mm", "mhm", "mm-hm", "mm-hmm", "mmhmm", "hmm", "hm", "uh-huh", "uhhuh", "um", "uh", "ah", "oh", "हम्म", "हम"}


def is_filler(text: str) -> bool:
    """Only non-lexical sounds ("hmm", "mm-hmm", "uh"): not worth an answer even when the agent is silent. Lexical
    acknowledgements ("okay", "yes", "haan") are left to the answer pipeline."""
    words = normalize_utterance(text).split()
    return bool(words) and all(w in FILLERS for w in words)


def is_backchannel(text: str, max_words: int) -> bool:
    """True for an utterance that only acknowledges ("mm-hmm", "okay", "haan ji") or has no words at all; False when
    it has more than ``max_words`` words or any word that isn't an acknowledgement ("wait", "stop", a question)."""
    phrase = normalize_utterance(text)
    words = phrase.split()
    if not words:
        return True
    if len(words) > max_words:
        return False
    return phrase in BACKCHANNELS or all(w in BACKCHANNELS for w in words)
