"""The chat's own words for the speech recognizer (docs/DESIGN.md §9.3): a vocabulary prompt per language.

Whisper small spells the names it doesn't know by sound: "Valmora" came out as "Mora's", "Vimora" or "Vamora",
"Zephyra" as "Zephyr" or "जफर", and the wrong name then spread into the answer, the chat title, the chart titles and
web queries. Given the names as an ``initial_prompt`` (text Whisper treats as what was said just before), it spells
them as the documents do. The prompt is built from the chat's READY documents:

1. names: the capitalised phrases of a document's title that contain a word of its file name ("Valmora Industries",
   "Zephyra Logistics"), else the distinctive words of the file name ("Suryodaya Yojana");
2. terms: short capitalised phrases in the documents' tables (segments such as "Specialty Chemicals", facilities,
   programmes, people) and acronyms used in several tables ("EBITDA"). Words the model already knows are one token of
   its tokenizer ("Digital", "Services", "North America"); phrases made only of those are left for last, phrases
   used in more tables come first, and the documents take turns so every document's words get in;
3. for Hindi: the Devanagari headings and first-column entries of Hindi documents (a scheme's name, districts).

Each prompt is kept short (Whisper reads at most 223 prompt tokens, and every token is decoding time): about 110
tokens. The English prompt is a plain list. The Hindi one starts with the names in Latin script in a short Devanagari
frame, then the Devanagari terms (measured: a Latin-only prompt makes Whisper write Hindi in Latin script, and a
Devanagari-only one doesn't help the names).

``SpeechVocabulary`` caches the prompts per chat and rebuilds them when the chat's READY documents change. Building is
cheap (a query per document's tables, the labels from the vector store) and never on a transcription's path: the
voice session refreshes it in the background and transcribes with whatever it has.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ...domain.projects import Chat
from ...providers.ingestion import split_document_label
from ...providers.llm import LLMClient, LLMMessage
from ...providers.registry import Container
from ...providers.retrieval import RetrievalFilters, VectorStore
from ...providers.speech import SpeechRecognizer
from ...providers.storage import MetadataDB
from ..documents import DocumentService
from ..language import wordset

log = logging.getLogger(__name__)

PROMPT_TOKENS = 110  # per language; Whisper's own limit is 223
# The Hindi prompt also carries the Hindi documents' own frequent words (last round, item 4: "आयु" was heard as "आयो" in
# 3 of 3 clips, and the question answered as the income limit): up to this many tokens in all, still well under 223.
HINDI_PROMPT_TOKENS = 170
MAX_HINDI_CONTENT_WORDS = 12
MAX_ACRONYMS = 4
MIN_ACRONYM_TABLES = 2  # an acronym in one table only is a label ("CIN", "ISIN"), not the documents' vocabulary
MAX_PHRASE_WORDS = 4
MAX_PHRASE_CHARS = 40
MAX_HINDI_WORDS = 7
CACHE_CHATS = 64
REFRESH_S = 30.0  # a chat's READY documents are checked again at most this often

TokenCounter = Callable[[str], int | None]

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019&.\-]*")
_ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z&]{2,7}(?![A-Za-z0-9])")
_FILE_SEPARATORS = re.compile(r"[\s_\-.+]+")
_TITLE_SEPARATORS = re.compile(r"\s+[-\u2013\u2014|]\s+|[:,;()\[\]]")
_NUMBERING = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[A-Za-z][.)])\s+")
_FOOTNOTE = re.compile(r"\s+\d{1,2}$")  # "Digital Services 1": a footnote mark
_HONORIFIC = re.compile(r"^(?:Mr|Mrs|Ms|Dr|Prof|Shri|Smt)\.?\s+")
_MARKUP = re.compile(r"[*_`#]+")

_CONNECTORS = frozenset({"of", "and", "&", "for", "the", "in", "on", "de", "da", "von"})
_LEGAL = frozenset({
    "limited", "ltd", "pvt", "private", "inc", "incorporated", "corp", "corporation", "llc", "plc", "llp", "co",
    "company",
})  # fmt: skip
# File name words that describe the document, not what it is about.
_GENERIC_FILE_WORDS = frozenset({
    "annual", "report", "reports", "policy", "policies", "deck", "investor", "investors", "presentation", "slides",
    "travel", "expense", "expenses", "group", "health", "insurance", "scan", "scanned", "copy", "final", "draft",
    "document", "doc", "docs", "file", "notice", "soochna", "suchna", "minutes", "statement", "statements",
    "financial", "financials", "results", "result", "quarterly", "summary", "overview", "brochure", "handbook",
    "manual", "guide", "guidelines", "agreement", "contract", "terms", "schedule", "letter", "memo", "note", "notes",
    "update", "new", "old", "version", "rev", "revised", "signed", "the", "and", "of", "for", "with", "english",
    "hindi",
})  # fmt: skip
_GENERIC_PHRASES = frozenset({"grand total", "sub total", "net total", "not applicable", "key metrics"})
_COMMON_ACRONYMS = frozenset({"USD", "EUR", "INR", "GBP", "THE", "AND", "FOR", "YES", "TOTAL", "NOTE", "PDF"})
_GENERIC_HINDI = frozenset({"कुल", "योग", "अन्य", "विवरण", "क्रम", "संख्या", "टिप्पणी"})


# ------------------------------------------------------------------ the vocabulary


@dataclass(frozen=True, slots=True)
class DocumentWords:
    """What the vocabulary is built from, for one document."""

    filename: str
    label: str = ""  # file name words and title (``providers.ingestion.document_label``), "" when unknown
    tables: Sequence[Mapping[str, Any]] = ()  # dicts with ``heading_path`` and ``cells`` (as ``DocumentTable``)
    texts: Sequence[str] = ()  # a sample of its chunks' headings and text (``VectorStore.document_texts``)


@dataclass(frozen=True, slots=True)
class Vocabulary:
    names: tuple[str, ...] = ()  # "Valmora Industries", "Zephyra Logistics"
    keywords: tuple[str, ...] = ()  # the names' distinctive words: "Valmora", "Zephyra"
    terms: tuple[str, ...] = ()  # "Specialty Chemicals", "Contract Logistics", … (best first)
    acronyms: tuple[str, ...] = ()  # "EBITDA"
    terms_hi: tuple[str, ...] = ()  # Devanagari: "सूर्योदय ग्रामीण कौशल एवं रोज़गार योजना", "कमलपुर", …
    keywords_hi: tuple[str, ...] = ()  # the keywords in Devanagari ("वाल्मोरा"), when known
    words_hi: tuple[str, ...] = ()  # Hindi documents' concept words ("पात्रता", "आयु", "आय"), for the prompt
    lexicon_hi: frozenset[str] = frozenset()  # every Devanagari word of the Hindi documents' texts

    @property
    def empty(self) -> bool:
        return not (self.names or self.terms or self.acronyms or self.terms_hi or self.words_hi)

    def prompts(self, languages: Iterable[str] = ("en", "hi"), count: TokenCounter | None = None) -> dict[str, str]:
        """language → prompt; a language with nothing to say is left out. ``count``: the recognizer's tokenizer."""
        out = {}
        for language in languages:
            prompt = hindi_prompt(self, count) if language == "hi" else english_prompt(self, count)
            if prompt:
                out[language] = prompt
        return out

    def lexicon(self, count: TokenCounter | None = None) -> tuple[str, ...]:
        """The unfamiliar words of every name and term (not only those the prompts had room for): what
        ``SpeechHints.correct`` restores a near miss to ("Vamora" → "Valmora")."""
        words: dict[str, str] = {}
        for phrase in (*self.keywords, *self.names, *self.terms):
            for word in phrase.split():
                name_like = len(word) >= LEXICON_MIN_LETTERS and word.isalpha() and word.istitle()
                if name_like and _unfamiliar(word, count, LEXICON_MIN_TOKENS):
                    words.setdefault(word.casefold(), word)
        return tuple(words.values())


@dataclass(frozen=True, slots=True)
class SpeechHints:
    """What a chat's transcriptions get: the vocabulary prompts and the lexicon of its unfamiliar words."""

    prompts: Mapping[str, str] = field(default_factory=dict)
    lexicon: tuple[str, ...] = ()
    count: TokenCounter | None = None
    lexicon_hi: frozenset[str] = frozenset()  # the Hindi documents' words (``hindi_lexicon``)

    def correct(self, text: str) -> str:
        """``text`` with near misses of the lexicon's words spelled as the documents do ("What was Vamora's revenue"
        → "What was Valmora's revenue"; measured: Whisper with the prompt still wrote "Vamora", "Varmora", "Vimora",
        "Talaja"). Only a capitalised Latin word (Whisper capitalises what it takes for a name) that the tokenizer finds
        unfamiliar (a common word is one token and never changes), within one edit of a lexicon word (two for a word
        of seven letters or more), starting with the same or a confusable letter, and closest to exactly one of
        them."""
        if not self.lexicon or not text:
            return text
        lexicon = {w.casefold(): w for w in self.lexicon}

        def fix(m: re.Match[str]) -> str:
            word, suffix = m.group(1), m.group(2) or ""
            folded = word.casefold()
            if folded in lexicon or len(word) < LEXICON_MIN_LETTERS - 1 or not word.istitle():
                return m.group(0)
            if not _unfamiliar(word, self.count, 2):
                return m.group(0)
            best: list[tuple[int, str]] = []
            for key, target in lexicon.items():
                limit = 2 if len(key) >= 7 else 1
                if abs(len(key) - len(folded)) > limit or not _same_onset(key, folded):
                    continue
                d = edit_distance(folded, key)
                if d <= limit:
                    best.append((d, target))
            best.sort()
            if not best or (len(best) > 1 and best[0][0] == best[1][0]):
                return m.group(0)
            return best[0][1] + suffix

        return _LATIN_TOKEN.sub(fix, text)

    def correct_hindi(self, text: str) -> str:
        """``text`` with Devanagari words that differ from exactly one word of the Hindi documents only in a vowel sign
        written as the documents write it ("आयो" → "आयु": Whisper wrote "आवेदक की आयो कितनी होनी चाहिए?" for every
        synthetic clip of the age question, last round item 4). A word the documents have, a function word, or one
        whose shape (its letters and where vowel signs fall) matches several of their words is left as it is."""
        if not self.lexicon_hi or not text:
            return text
        by_shape: dict[tuple[tuple[str, bool], ...], set[str]] = {}
        for w in self.lexicon_hi:
            by_shape.setdefault(_hindi_shape(w), set()).add(w)

        def fix(m: re.Match[str]) -> str:
            word = m.group(0)
            plain = word.replace("\u093c", "")
            if plain in self.lexicon_hi or word in self.lexicon_hi or plain in _HINDI_STOP or len(plain) < 3:
                return word
            near = by_shape.get(_hindi_shape(plain), set())
            return next(iter(near)) if len(near) == 1 else word

        return _DEVANAGARI_RUN.sub(fix, text)


LEXICON_MIN_LETTERS = 5
LEXICON_MIN_TOKENS = 3  # " Valmora" is 3 tokens, " Taloja" 3, " Zephyra" 4; " Logistics" 2, " Services" 1
_LATIN_TOKEN = re.compile(r"(?<![A-Za-z])([A-Za-z]+)(['\u2019]s)?(?![A-Za-z])")
_ONSETS = ("vwb", "zsjx", "ckq", "fp", "td", "ae", "iey", "ou")


def _unfamiliar(word: str, count: TokenCounter | None, min_tokens: int) -> bool:
    """The recognizer's tokenizer needs ``min_tokens`` or more for the word, lower-cased (without a tokenizer: eight
    letters or more)."""
    n = count(" " + word.casefold()) if count is not None else None
    return n >= min_tokens if n is not None else len(word) >= 8


def _same_onset(a: str, b: str) -> bool:
    return a[0] == b[0] or any(a[0] in group and b[0] in group for group in _ONSETS)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance with adjacent transpositions (optimal string alignment)."""
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[len(b)]


def estimate_tokens(text: str, count: TokenCounter | None = None) -> int:
    """Whisper tokens of ``text``: the tokenizer's count, else about 3 Latin characters per token and 1.3 tokens per
    Devanagari character (measured on Whisper's multilingual tokenizer: " Valmora" 3, " सूर्योदय" 10)."""
    if count is not None and (n := count(" " + text)) is not None:
        return n
    deva = len(_DEVANAGARI.findall(text))
    return int((len(text) - deva) / 3 + deva * 1.3) + 1


def _fill(items: Iterable[str], budget: int, count: TokenCounter | None) -> list[str]:
    """The items, in order, that fit in ``budget`` tokens (one too long is skipped; a shorter one later may fit)."""
    out: list[str] = []
    used = 0
    for item in dict.fromkeys(items):
        cost = estimate_tokens(item, count) + 1  # and its separator
        if used + cost <= budget:
            out.append(item)
            used += cost
    return out


def english_prompt(v: Vocabulary, count: TokenCounter | None = None) -> str:
    items = _fill([*v.names, *v.acronyms, *v.terms], PROMPT_TOKENS, count)
    return ", ".join(items) + "." if items else ""


def hindi_prompt(v: Vocabulary, count: TokenCounter | None = None) -> str:
    """The names (in Devanagari when known, and as the documents spell them) and acronyms in a short Devanagari
    frame, then the Devanagari terms: Hindi stays in Devanagari, and a name is written one of the two ways the prompt
    gives (measured on Hindi clips: names 2/20 with the Latin names only, 9/20 with "वाल्मोरा", "जेफिरा" added)."""
    latin = _fill([*v.keywords_hi, *v.keywords, *v.acronyms], PROMPT_TOKENS * 2 // 3, count)
    frame = f"{', '.join(latin)} के बारे में।" if latin else ""
    hindi = _fill(v.terms_hi, PROMPT_TOKENS - estimate_tokens(frame, count), count)
    parts = [*([frame] if frame else []), *([", ".join(hindi) + "।"] if hindi else [])]
    said = set(" ".join(parts).split())
    left = HINDI_PROMPT_TOKENS - estimate_tokens(" ".join(parts), count) - 1
    words = _fill([w for w in v.words_hi if w not in said], left, count) if left > 0 else []
    return " ".join([*parts, *([", ".join(words) + "।"] if words else [])])


# ------------------------------------------------------------------ extraction (pure)


def _clean(text: str) -> str:
    text = _NUMBERING.sub("", _MARKUP.sub("", text or "").strip())
    return " ".join(_FOOTNOTE.sub("", text).split())


def _is_name_word(word: str) -> bool:
    return word[:1].isupper() or word.casefold() in _CONNECTORS


def title_phrases(title: str) -> list[str]:
    """Maximal runs of capitalised Latin words in each part of a title, legal suffixes dropped: "Valmora Industries
    Limited - Annual Report 2023-24" → ["Valmora Industries", "Annual Report"]. A part written in capitals
    ("VALMORA INDUSTRIES LIMITED") is read in title case."""
    out = []
    for part in _TITLE_SEPARATORS.split(title):
        if part.isupper():
            part = part.title()
        run: list[str] = []
        for token in [*part.split(), ""]:
            word = token.strip("'\u2019\".")
            if token and _LATIN_WORD.fullmatch(word) and _is_name_word(word) and not word.isupper():
                run.append(word)
                continue
            while run and run[-1].casefold().rstrip(".") in _LEGAL | _CONNECTORS:
                run.pop()
            while run and run[0].casefold() in _CONNECTORS:
                run.pop(0)
            if run:
                out.append(" ".join(run))
            run = []
    return out


def file_words(filename: str) -> list[str]:
    """The distinctive words of a file name: "valmora_annual_report_fy24.pdf" → ["valmora"]."""
    words = [w.casefold() for w in _FILE_SEPARATORS.split(Path(filename).stem) if w]
    return [w for w in words if w.isalpha() and len(w) >= 3 and w not in _GENERIC_FILE_WORDS]


def _phrase_ok(text: str) -> bool:
    """A short capitalised Latin phrase of two to four words: a segment, a facility, a programme, a person."""
    if len(text) > MAX_PHRASE_CHARS or any(ch.isdigit() for ch in text) or _DEVANAGARI.search(text):
        return False
    words = text.split()
    if not 2 <= len(words) <= MAX_PHRASE_WORDS or text.casefold() in _GENERIC_PHRASES:
        return False
    return words[0][:1].isupper() and all(_LATIN_WORD.fullmatch(w) and _is_name_word(w) for w in words)


def _hindi_ok(text: str) -> str:
    """The Devanagari phrase without numbers, if it is a short one (a name, a scheme, a heading), else ""."""
    words = [w for w in text.split() if not any(ch.isdigit() for ch in w)]
    if not words or len(words) > MAX_HINDI_WORDS or not _DEVANAGARI.search(text):
        return ""
    phrase = " ".join(words).strip(" :-\u2013।")
    return "" if phrase in _GENERIC_HINDI else phrase


def rarity(phrase: str, count: TokenCounter | None) -> int:
    """How unfamiliar a phrase is to the recognizer: its words' tokens beyond one each (0: every word is a single
    token, i.e. one the model knows). Without a tokenizer: words of eight letters or more count one each."""
    total = 0
    for word in phrase.split():
        n = count(" " + word) if count is not None else None
        total += (n - 1) if n is not None else int(len(word) >= 8)
    return total


@dataclass(slots=True)
class _Seen:
    """A candidate term: in how many tables, which document saw it first, and in what order."""

    tables: int = 0
    document: int = 0
    order: int = 0


def build_vocabulary(documents: Sequence[DocumentWords], count: TokenCounter | None = None) -> Vocabulary:
    names: list[str] = []
    keywords: Counter[str] = Counter()
    terms: dict[str, _Seen] = {}
    acronyms: Counter[str] = Counter()
    hindi: dict[str, _Seen] = {}
    for d, doc in enumerate(documents):
        _names_of(doc, names, keywords)
        for table in doc.tables:
            found, found_hi = _table_words(table)
            for text in found:
                if text.isupper() and " " not in text:
                    acronyms[text] += 1
                else:
                    seen = terms.setdefault(text, _Seen(document=d, order=len(terms)))
                    seen.tables += 1
            for text in found_hi:
                seen = hindi.setdefault(text, _Seen(document=d, order=len(hindi)))
                seen.tables += 1

    names = _distinct_names(names, keywords)
    known = {n.casefold() for n in names} | set(keywords)
    candidates = {t: s for t, s in terms.items() if _without_legal(t).casefold() not in known}
    rare = {t: rarity(t, count) for t in candidates}
    # Per document: unfamiliar words first, then used in more tables, then rarer, then in document order …
    by_document: dict[int, list[str]] = {}
    for t in sorted(candidates, key=lambda t: (rare[t] == 0, -candidates[t].tables, -rare[t], candidates[t].order)):
        by_document.setdefault(candidates[t].document, []).append(t)
    # … and the documents take turns, the familiar-only phrases of all of them last.
    columns = [by_document[d] for d in sorted(by_document)]
    ranked = [t for row in zip_longest(*columns) for t in row if t is not None]
    ranked = [t for t in ranked if rare[t]] + [t for t in ranked if not rare[t]]
    # Acronyms used across the documents' tables ("EBITDA"), not one table's labels ("CIN", "ISIN").
    floor = max(MIN_ACRONYM_TABLES, max(acronyms.values(), default=0) // 3)
    top_acronyms = [a for a, n in acronyms.most_common() if n >= floor and a.casefold() not in keywords]
    top_acronyms = top_acronyms[:MAX_ACRONYMS]
    ranked_hindi = sorted(hindi, key=lambda t: (-hindi[t].tables, len(t.split()), hindi[t].order))
    hindi_texts = [t for doc in documents for t in doc.texts]
    words_hi = hindi_content_words(hindi_texts)
    return Vocabulary(
        names=tuple(names),
        keywords=tuple(k.capitalize() for k, _ in keywords.most_common()),
        terms=tuple(ranked),
        acronyms=tuple(top_acronyms),
        terms_hi=tuple(ranked_hindi),
        words_hi=words_hi,
        lexicon_hi=hindi_lexicon(hindi_texts),
    )


# Words of a Hindi document that say nothing about its subject: postpositions, pronouns, auxiliaries, common verbs.
_HINDI_STOP = wordset(
    """
    का की के को में से पर तक और या भी है हैं था थी थे हो होगा होगी होंगे होता होती होते होने होना हुआ हुई हुए जो जिस
    जिन जिसे जिससे जिसका जिसकी जिसके यह वह ये वे इस उस इन उन इसका इसकी इसके उसका उसकी उसके इसमें उसमें इसे उसे लिए
    द्वारा साथ बाद पहले अंदर बीच तथा एवं अथवा किसी कोई कुछ सभी सब हर अन्य अधिक कम नहीं न ही तो कि क्या कब कहाँ कैसे
    कितना कितनी कितने करना करने करता करती करते करें किया किए की गई गए जाता जाती जाते जाएगा जाएगी जाएंगे दिया दी
    दिए दिया जा सकता सकती सकते रहे रहा रही रहेगा वाला वाली वाले प्रति अपने अपनी अपना एक दो तीन चार पाँच
    """
)
# The concepts a question about a scheme or policy asks for ("who is eligible", "the age limit", "income", "which
# documents", "districts", "the benefits"): first among the document's words when it has them.
_HINDI_CONCEPTS = [
    "पात्रता",
    "आयु",
    "आय",
    "दस्तावेज़",
    "दस्तावेज",
    "जिला",
    "जिले",
    "लाभ",
    "सहायता",
    "अनुदान",
    "ऋण",
    "वजीफ़ा",
    "वजीफा",
    "शुल्क",
    "तिथि",
    "सीमा",
    "आरक्षण",
    "प्रमाणपत्र",
    "उम्र",
]
_DEVANAGARI_RUN = re.compile(r"[ऀ-ॣॱ-ॿ]+")


def hindi_content_words(texts: Sequence[str], limit: int = MAX_HINDI_CONTENT_WORDS) -> tuple[str, ...]:
    """A Hindi document's own words for the speech recognizer's prompt: the concepts its questions ask for that its
    headings and text have ("पात्रता", "आयु", "आय", "दस्तावेज़", "जिला", "लाभ", …). Measured on synthetic clips, long
    frequent words ("प्रशिक्षण") bled into others ("वार्षिक आय" → "वार्षिक्षण"), so only these."""
    counts: Counter[str] = Counter()
    for text in texts:
        lines = text.splitlines()
        for n, line in enumerate(lines):
            weight = 3 if n < len(lines) - 1 else 1  # the last line is the chunk's text, the others its headings
            for word in _DEVANAGARI_RUN.findall(line):
                if len(word) >= 2 and word not in _HINDI_STOP:
                    counts[word] += weight
    concepts = [w for w in _HINDI_CONCEPTS if w in counts]
    return tuple(concepts[:limit])


_VOWEL_SIGNS = frozenset(chr(c) for c in range(0x093E, 0x094D)) | {"\u0962", "\u0963"}
_MARKS = frozenset("\u0901\u0902\u0903\u093c")  # chandrabindu, anusvara, visarga, nukta


def _hindi_shape(word: str) -> tuple[tuple[str, bool], ...]:
    """A Devanagari word's letters (a consonant with its virama, or an independent vowel) and whether each carries a
    vowel sign: "आयो" and "आयु" are (("आ", False), ("य", True)); "आय" is (("आ", False), ("य", False))."""
    out: list[list[object]] = []
    for ch in unicodedata.normalize("NFD", word):
        if ch in _VOWEL_SIGNS and out:
            out[-1][1] = True
        elif ch == "\u094d" and out:
            out[-1][0] = str(out[-1][0]) + ch
        elif ch not in _MARKS:
            out.append([ch, False])
    return tuple((str(b), bool(v)) for b, v in out)


def hindi_lexicon(texts: Sequence[str]) -> frozenset[str]:
    """The Devanagari words of the Hindi documents' texts (nuktas dropped), for ``SpeechHints.correct_hindi``."""
    return frozenset(w.replace("\u093c", "") for t in texts for w in _DEVANAGARI_RUN.findall(t) if len(w) >= 2)


def _without_legal(phrase: str) -> str:
    words = phrase.split()
    while len(words) > 1 and words[-1].casefold().rstrip(".") in _LEGAL:
        words.pop()
    return " ".join(words)


def _distinct_names(names: list[str], keywords: Counter[str]) -> list[str]:
    """Each name once, a name inside a longer one dropped ("Valmora" of "Valmora Industries"), the names of the
    subjects most documents are about first."""
    unique = list(dict.fromkeys(names))
    words = [{w.casefold() for w in n.split()} for n in unique]
    kept = [n for n, w in zip(unique, words, strict=True) if not any(w < other for other in words)]
    return sorted(kept, key=lambda n: -max((keywords[w.casefold()] for w in n.split()), default=0))


def _names_of(doc: DocumentWords, names: list[str], keywords: Counter[str]) -> None:
    _, title = split_document_label(doc.label) if doc.label else ("", "")
    stems = file_words(doc.filename)
    matched = False
    for phrase in title_phrases(title):
        words = {w.casefold() for w in phrase.split()}
        hits = [s for s in stems if s in words]
        if hits:
            names.append(phrase)
            keywords.update(hits)
            matched = True
    if not matched and stems:
        names.append(" ".join(s.capitalize() for s in stems))
        keywords[stems[0]] += 1  # the first distinctive word names the subject ("suryodaya" of "suryodaya yojana")


def _table_words(table: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(Latin phrases and acronyms, Devanagari phrases) of one table, each once, in the table's order: its headings,
    its first column, and any cell that is a short capitalised phrase."""
    latin: dict[str, None] = {}
    hindi: dict[str, None] = {}
    for heading in table.get("heading_path") or []:
        if phrase := _hindi_ok(_clean(str(heading))):
            hindi[phrase] = None
    for cell in table.get("cells") or []:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("text"), str):
            continue
        text = _HONORIFIC.sub("", _clean(cell["text"]))
        if not text:
            continue
        latin.update(dict.fromkeys(a for a in _ACRONYM.findall(text) if a not in _COMMON_ACRONYMS))
        if _DEVANAGARI.search(text):
            if cell.get("col") == 0 and not cell.get("column_header") and (phrase := _hindi_ok(text)):
                hindi[phrase] = None
        elif _phrase_ok(text):
            latin[text] = None
    return list(latin), list(hindi)


# ------------------------------------------------------------------ names in Devanagari


class _Devanagari(BaseModel):
    names: list[str]


TRANSLITERATE_TIMEOUT_S = 10.0
MAX_TRANSLITERATED = 6
_TRANSLITERATIONS: dict[str, str] = {}  # Latin name → Devanagari, for the life of the process


def transliteration_messages(names: Sequence[str]) -> list[LLMMessage]:
    return [
        LLMMessage(
            "system",
            "You write names in Devanagari script, as a Hindi speaker would write them. Answer with JSON: "
            '{"names": [...]}, one entry per name, in the same order.',
        ),
        LLMMessage("user", "\n".join(names)),
    ]


async def transliterate(llm: LLMClient, names: Sequence[str]) -> dict[str, str]:
    """Latin name → its Devanagari spelling, asked of the LLM once per name (remembered). A name the model doesn't
    give back in Devanagari is left out; a failure gives what is already known."""
    wanted = [n for n in dict.fromkeys(names) if n not in _TRANSLITERATIONS][:MAX_TRANSLITERATED]
    if wanted:
        try:
            answer = await asyncio.wait_for(
                llm.generate_json(transliteration_messages(wanted), _Devanagari, max_tokens=24 * len(wanted)),
                TRANSLITERATE_TIMEOUT_S,
            )
        except Exception as e:  # optional: the Hindi prompt works without it
            log.info("speech vocabulary: names not transliterated (%s: %s)", type(e).__name__, e)
        else:
            for name, spelled in zip(wanted, answer.names, strict=False):
                deva = " ".join(spelled.split())
                if deva and len(deva.split()) <= 2 and all(_DEVANAGARI.match(ch) or ch == " " for ch in deva):
                    _TRANSLITERATIONS[name] = deva
    return {n: _TRANSLITERATIONS[n] for n in names if n in _TRANSLITERATIONS}


def document_spelling(spelled: str, terms_hi: Sequence[str]) -> str:
    """The documents' own Devanagari spelling of a transliterated name when they have a near one ("सुर्योदय" from the
    model, "सूर्योदय" in the scheme's title), else the model's."""
    if " " in spelled:
        return spelled
    limit = 1 if len(spelled) < 6 else 2
    words = dict.fromkeys(w for t in terms_hi for w in t.split())
    near = sorted((edit_distance(spelled, w), w) for w in words if abs(len(w) - len(spelled)) <= limit)
    return near[0][1] if near and near[0][0] <= limit else spelled


def speech_vocabulary(container: Container) -> SpeechVocabulary | None:
    """The app's speech vocabulary (one per process; None when the container lacks a database or a recognizer)."""
    db, stt = container.providers.get("metadata_db"), container.providers.get("stt")
    if not isinstance(db, MetadataDB) or not isinstance(stt, SpeechRecognizer):
        return None
    store, llm = container.providers.get("vector_store"), container.providers.get("llm")
    return SpeechVocabulary(
        DocumentService(db),
        store if isinstance(store, VectorStore) else None,
        stt.count_tokens,
        llm if isinstance(llm, LLMClient) else None,
        tuple(container.settings.stt.languages),
    )


# ------------------------------------------------------------------ per chat, cached


@dataclass(slots=True)
class _Entry:
    key: tuple[Any, ...]
    hints: SpeechHints
    checked_at: float


@dataclass(slots=True)
class SpeechVocabulary:
    """The speech hints of each chat, rebuilt when its READY documents change (checked at most every ``REFRESH_S``).
    ``count`` is the recognizer's tokenizer; ``llm`` (optional) spells the names in Devanagari for the Hindi prompt.
    A failure gives no hints: transcription then works as it does without them."""

    documents: DocumentService
    store: VectorStore | None = None
    count: TokenCounter | None = None
    llm: LLMClient | None = None
    languages: Sequence[str] = ("en", "hi")
    _cache: dict[str, _Entry] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def cached(self, chat_id: str) -> SpeechHints | None:
        entry = self._cache.get(chat_id)
        return entry.hints if entry is not None else None

    async def hints(self, chat: Chat) -> SpeechHints:
        entry = self._cache.get(chat.id)
        if entry is not None and time.monotonic() - entry.checked_at < REFRESH_S:
            return entry.hints
        async with self._locks.setdefault(chat.id, asyncio.Lock()):
            ready = await self.documents.ready_documents(chat.project_id, chat.document_scope)
            key = (chat.project_id, tuple(sorted(ready.items())))
            entry = self._cache.get(chat.id)
            if entry is not None and entry.key == key:
                entry.checked_at = time.monotonic()
                return entry.hints
            t0 = time.perf_counter()
            vocabulary = build_vocabulary(await self._words(chat.project_id, ready), self.count)
            if self.llm is not None and "hi" in self.languages and vocabulary.keywords:
                spelled = await transliterate(self.llm, vocabulary.keywords)
                keywords_hi = tuple(
                    document_spelling(spelled[k], vocabulary.terms_hi) for k in vocabulary.keywords if k in spelled
                )
                vocabulary = replace(vocabulary, keywords_hi=keywords_hi)
            hints = SpeechHints(
                vocabulary.prompts(self.languages, self.count),
                vocabulary.lexicon(self.count),
                self.count,
                vocabulary.lexicon_hi,
            )
            self._remember(chat.id, _Entry(key, hints, time.monotonic()))
            log.info(
                "speech vocabulary of chat %s (%d documents, %.0f ms): prompts %s; lexicon %s",
                chat.id,
                len(ready),
                (time.perf_counter() - t0) * 1000,
                dict(hints.prompts),
                hints.lexicon,
            )
            return hints

    def _remember(self, chat_id: str, entry: _Entry) -> None:
        self._cache.pop(chat_id, None)
        self._cache[chat_id] = entry
        while len(self._cache) > CACHE_CHATS:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            self._locks.pop(oldest, None)

    async def _words(self, project_id: str, ready: Mapping[str, str]) -> list[DocumentWords]:
        labels: dict[str, str] = {}
        texts: dict[str, list[str]] = {}
        if self.store is not None and ready:
            with contextlib.suppress(Exception):  # no titles: the file names still give the names
                labels = await self.store.document_labels(RetrievalFilters(project_id, tuple(ready)))
            with contextlib.suppress(Exception):  # no texts: the tables still give the terms
                texts = await self.store.document_texts(RetrievalFilters(project_id, tuple(ready)))
        out = []
        for doc_id, filename in ready.items():
            tables: list[dict[str, Any]] = []
            try:
                tables = [t.model_dump(include={"heading_path", "cells"}) for t in await self.documents.tables(doc_id)]
            except Exception as e:  # deleted meanwhile, a database hiccup: its name still counts
                log.info("speech vocabulary: the tables of %s are unavailable: %s", doc_id, e)
            hindi = [t for t in texts.get(doc_id, []) if len(_DEVANAGARI.findall(t)) * 2 > len(t.replace(" ", ""))]
            out.append(DocumentWords(filename, labels.get(doc_id, ""), tables, hindi))
        return out
