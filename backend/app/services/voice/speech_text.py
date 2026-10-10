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


def spoken_text(raw: str, *, sentence_start: bool = True) -> str:
    """What TTS should say for a piece of answer text: [S#] / [W#] markers dropped, or said as "a web source" / "one
    source" where the sentence uses them as a word (``speakable_markers``), no markdown, single spaces. Empty if
    nothing speakable (only punctuation or markers) remains. ``sentence_start``: the piece starts a sentence (a marker
    that opens it is said with a capital letter)."""
    text = speakable_markers(raw, sentence_start=sentence_start)
    text = _SPACE.sub(" ", _MARKDOWN.sub("", strip_markers(text))).strip()
    return text if any(ch.isalnum() for ch in text) else ""


# Citation markers as words (quality round, item 6): "[W1] mentions that the rupee rose" was spoken "However,
# mentions that…". A marker group ("[W1]", "[S1][S2]", "[W1] and [W2]") that is the subject of its clause (it opens
# the clause and a word follows) or the object of a preposition ("according to [W1]", "as reported by [S2]", Hindi
# "[W1] के अनुसार") is said as what it is; anywhere else it is dropped, with the space before it.
_MARKER_GROUP = re.compile(
    r"(\s*)\[\s*[SW]\d+(?:\s*[,;]\s*[SW]\d+)*\s*\](?:(?:\s*(?:,|and|और)?\s*)\[\s*[SW]\d+(?:\s*[,;]\s*[SW]\d+)*\s*\])*",
    re.IGNORECASE,
)
_MARKER_ID = re.compile(r"([SW])\d+", re.IGNORECASE)
_CLAUSE_OPENERS = {"and", "but", "however", "while", "whereas", "also", "though", "although", "so", "then", "yet"}
_PREPOSITIONS = {
    "to",
    "per",
    "by",
    "from",
    "in",
    "on",
    "see",
    "cites",
    "cite",
    "citing",
    "as",
    "than",
    "at",
    "under",
    "with",
}
_POSTPOSITIONS = {"के", "की", "का", "में", "ने", "से", "पर", "द्वारा", "अनुसार", "को"}
_SPOKEN_SOURCE = {
    # (language, web, several) → words
    ("en", True, False): "a web source",
    ("en", True, True): "web sources",
    ("en", False, False): "one source",
    ("en", False, True): "the sources",
    ("hi", True, False): "एक स्रोत",
    ("hi", True, True): "कुछ स्रोत",
    ("hi", False, False): "एक स्रोत",
    ("hi", False, True): "कुछ स्रोत",
}


def speakable_markers(text: str, *, sentence_start: bool = True) -> str:
    """``text`` with each citation marker group that is used as a word replaced by words ("a web source", "one source",
    "एक स्रोत"), the others left for ``strip_markers``."""
    hindi = any("ऀ" <= ch <= "ॿ" for ch in text)

    def replace(m: re.Match[str]) -> str:
        before = text[: m.start()].rstrip()
        after = text[m.end() :].lstrip()
        prev = re.sub(r"[^\wऀ-ॿ]", "", before.split()[-1]).casefold() if before.split() else ""
        nxt = re.sub(r"[^\wऀ-ॿ]", "", after.split()[0]) if after.split() else ""
        opens = (not before and (sentence_start or after)) or before[-1:] in ".!?।,;:—" or prev in _CLAUSE_OPENERS
        subject = opens and bool(nxt) and nxt[:1].isalpha() and nxt.casefold() not in {"and", "or"}
        objected = (prev in _PREPOSITIONS and before[-1:].isalnum()) or nxt in _POSTPOSITIONS
        if not (subject or objected):
            return m.group(0)  # a citation after a fact: dropped by strip_markers
        kinds = {k.upper() for k in _MARKER_ID.findall(m.group(0))}
        several = len(_MARKER_ID.findall(m.group(0))) > 1
        words = _SPOKEN_SOURCE[("hi" if hindi else "en", kinds == {"W"}, several)]
        at_start = (not before and sentence_start) or before[-1:] in ".!?।"
        if at_start and not hindi:
            words = words[:1].upper() + words[1:]
        return m.group(1) + words

    return _MARKER_GROUP.sub(replace, text)


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
        self._sentence_start = True  # the next chunk starts a sentence
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
        text = spoken_text(raw, sentence_start=self._sentence_start)
        self._sentence_start = sentence
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
    filler: bool = False  # "Let me look that up." while the web is searched (§3.7): not part of the answer

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


def heard_text(chunks: Sequence[SpokenChunk], played_ms: float) -> str:
    """What the user heard of a turn's answer whose audio played for ``played_ms``: every fully played chunk, plus
    the share of the chunk playing at ``played_ms`` in proportion to its duration, counted in whole words (rounded
    down). A filler is played but isn't the answer: its words don't count (its time does)."""
    words: list[str] = []
    for chunk in chunks:
        if chunk.filler:
            if played_ms < chunk.end_ms:
                break
            continue
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
# Kokoro's "Mm-hmm." as Whisper writes it: "MAMMA.", "Mama.", "Mmm-ma", "M-ma" (last round, item 6: it stopped the
# answer as an interruption). "mamma" and "mama" are hums only when the utterance is nothing but hums ("मामा", uncle,
# is a word: "Mama ji kahan hain?" stays a question); "ma" only as part of "mm-ma". "Mamata", "mammal", "ma'am" never.
_MAMMA = re.compile(r"m+a?m+a+h*")
_MA = re.compile(r"m+a+h*")
_PUNCT = re.compile(r"[^\w\s'\-ऀ-ॿ]|[।॥]")
_REPEAT = re.compile(r"(.)\1{2,}")


def normalize_utterance(text: str) -> str:
    """Lower case, no punctuation, stretched letters squeezed ("Hmmmm." → "hmm"), Whisper's noise phrases → "", and an
    utterance of hums only with "mamma" or "mama" among them read as "mm-hmm" ("MAMMA." → "mm-hmm")."""
    words = [_REPEAT.sub(r"\1\1", w.strip("-'")) for w in _PUNCT.sub(" ", text.casefold()).split()]
    words = [w for w in words if w]
    if any(_MAMMA.fullmatch(w) for w in words) and all(_MAMMA.fullmatch(w) or _is_hum(w) for w in words):
        words = ["mm-hmm" if _MAMMA.fullmatch(w) else w for w in words]
    phrase = " ".join(words)
    return "" if phrase in NOISE_TRANSCRIPTS else phrase


def _is_hum(token: str) -> bool:
    parts = [p for p in token.split("-") if p]

    def hum(p: str) -> bool:
        return bool(_HUM.fullmatch(p) or p in _HUM_HI)

    if len(parts) > 1 and any(hum(p) for p in parts):  # "mm-ma", "m-ma"
        return all(hum(p) or _MA.fullmatch(p) or _MAMMA.fullmatch(p) for p in parts)
    return bool(parts) and all(hum(p) for p in parts)


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


# ------------------------------------------------------------------ garbled transcripts (quality round, item 8)
#
# What Whisper writes for noise, a TV, or speech it couldn't make out (the final real run, with a film playing):
# "आब आब आब आब …" (60 times), "Just 1 employee sign here... Just 1 employee sign here...", "अप बवबवववववव…",
# "आश़््गें।", "understandgradeelle걱", "…for its меня dollars series…", "Is used by runsTAIC traffic". Answering them
# gave "I couldn't find that in this chat's documents" or "Could you please clarify your question?"; the user should
# be asked to say it again instead.

# Letters of neither script the app speaks (Latin with accents, Devanagari): Cyrillic, Hangul, CJK, …; and U+FFFD.
_FOREIGN = re.compile("[^\\W\\d_A-Za-zÀ-ɏऀ-ॿ]|�")
_NUKTA_BASES = "कखगजडढफयनरळ"
_BAD_DEVANAGARI = re.compile(
    "\u094d\u094d"  # two viramas
    "|\u093c\u093c"  # two nuktas
    f"|[^{_NUKTA_BASES}\u093c]\u093c"  # a nukta on a letter that takes none ("श़")
    "|[\u093e-\u094c\u0962\u0963][\u093e-\u094c\u0962\u0963]"  # two vowel signs in a row
    "|[\u0904-\u0914][\u093e-\u094d]"  # a vowel sign or virama on a vowel letter
    "|(?:^|\\s)[\u093e-\u094d\u0901-\u0903]"  # a word that starts with a sign
)
_RUN = re.compile(r"(\w)\1{4,}")  # the same letter five times ("ववववव")
_CAMEL = re.compile(r"[a-z]{2,}[A-Z]{2,}")  # "runsTAIC"
_LONG_WORD = re.compile(r"[^\W\d_]{25,}")


def _repeats(words: list[str]) -> bool:
    """A loop: one word four times in a row, a phrase of two to five words three times in a row, or one word making up
    two fifths of eight or more."""
    n = len(words)
    for size in range(1, 6):
        need = 4 if size == 1 else 3
        for i in range(0, n - size * need + 1):
            chunk = words[i : i + size]
            if all(words[i + k * size : i + (k + 1) * size] == chunk for k in range(need)):
                return True
    if n >= 8:
        top = max(words.count(w) for w in set(words))
        return top * 5 >= n * 2
    return False


def looks_garbled(text: str) -> bool:
    """The transcript reads like noise rather than speech (above). Acknowledgements repeated ("okay okay okay") and
    hums are not garbled: they have their own handling."""
    if not text.strip() or is_acknowledgement(text) or is_filler(text):
        return False
    if _FOREIGN.search(text) or _BAD_DEVANAGARI.search(text) or _CAMEL.search(text) or _LONG_WORD.search(text):
        return True
    if _RUN.search(text.casefold()):
        return True
    return _repeats(normalize_utterance(text).split())


# Speech recognition's own confidence, when the recognizer reports it (Whisper's thresholds: a decode whose average
# log probability is below -1 failed, one that compresses better than 2.4 is repetitive, and one whose no-speech
# probability is above 0.6 with a log probability below -1 is silence). Read only if present: ``Transcript`` may not
# carry them.
AVG_LOGPROB_MIN = -1.2
COMPRESSION_MAX = 2.4
NO_SPEECH_MAX = 0.6
CONFIDENCE_MIN = 0.35  # a 0-1 confidence, if that is what the recognizer gives instead


def transcript_garbled(transcript: object) -> bool:
    """Should this transcript be asked again ("Sorry, I didn't catch that")? Its text looks garbled, or the
    recognizer's confidence (``avg_logprob``, ``no_speech_prob``, ``compression_ratio`` or ``confidence``, whichever
    it has) says so."""

    def number(name: str) -> float | None:
        value = getattr(transcript, name, None)
        return float(value) if isinstance(value, int | float) else None

    logprob, no_speech = number("avg_logprob"), number("no_speech_prob")
    compression, confidence = number("compression_ratio"), number("confidence")
    if logprob is not None and (
        logprob < AVG_LOGPROB_MIN or (no_speech is not None and no_speech > NO_SPEECH_MAX and logprob < -1.0)
    ):
        return True
    if compression is not None and compression > COMPRESSION_MAX:
        return True
    if confidence is not None and 0.0 <= confidence <= 1.0 and confidence < CONFIDENCE_MIN:
        return True
    return looks_garbled(str(getattr(transcript, "text", "") or ""))


# Below this average log probability speech recognition isn't sure of the words, though they may read as a question
# (measured on the synthetic clips, §9.3: clear questions -0.05 to -0.5, Hindi -0.2 to -0.6, misheard or wrong-language
# transcripts -0.7 to -1.0): a question the documents don't answer is then asked again rather than declined.
AVG_LOGPROB_UNSURE = -0.7


def transcript_unsure(transcript: object) -> bool:
    """Speech recognition wasn't sure of this transcript (``avg_logprob`` below ``AVG_LOGPROB_UNSURE``, or a 0-1
    ``confidence`` below 0.6), when it reports it."""
    logprob, confidence = getattr(transcript, "avg_logprob", None), getattr(transcript, "confidence", None)
    if isinstance(logprob, int | float) and logprob < AVG_LOGPROB_UNSURE:
        return True
    return isinstance(confidence, int | float) and 0.0 <= confidence < 0.6
