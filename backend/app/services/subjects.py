"""Which of a chat's documents a question names (docs/DESIGN.md §3.2).

Every chunk is labelled with its document (file name and title, ``providers.ingestion.document_label``), which lets
"Valmora's EBITDA" find Valmora's tables, which don't name their company. The other side of that is a question about
one company that retrieval answers from the other company's document ("How many electric vehicles does Valmora run?"
found Zephyra's fleet): a passage the reranker finds relevant is not evidence about the company asked about.

``named_documents`` finds the documents a question is about by the names in it: capitalised words of the question
(or of its English query) that appear in the labels of some of the chat's documents but not of all (in a chat of two
Valmora documents "Valmora" tells nothing). File name words decide first; title words only when no file name word of
the question matches (a file called ``scan_0012.pdf`` is still told apart by its title). Words with digits ("FY24",
"Q4FY24") and words under three letters are never names, and neither are a few question words and company suffixes.

Pure and cheap, so the retrieval service can run it on every question. ``RetrievalService`` (``retrieval.py``) does the
rest: it keeps the passages that are about the named documents, or declines when there are none.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..providers.ingestion import split_document_label
from .language import wordset

# Latin words of three letters or more: "Valmora's" is "Valmora"; "FY24" and "Q4FY24" leave nothing. Devanagari and
# other scripts never match: a Hindi question is matched through its English query.
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}")

# Capitalised words that start a question or follow a company's name, not name anything.
_NOT_NAME_WORDS = (
    "the and for with from what how who whom whose which when where why does did are was were can could would "
    "should will has have had tell give show list explain describe please about this that these those there their "
    "our your his her its limited ltd inc corp corporation company plc llc pvt private group holdings"
)
_NOT_NAMES = frozenset(_NOT_NAME_WORDS.split())


@dataclass(frozen=True, slots=True)
class NamedDocuments:
    """The documents a question names, and the names (lower case) that matched them."""

    document_ids: frozenset[str]
    names: tuple[str, ...]

    def mentioned_in(self, text: str) -> bool:
        """The text says one of the names (a passage of another document that talks about the company anyway)."""
        return mentions(text, self.names)


def label_words(text: str) -> frozenset[str]:
    """The words of a document label that could be a name: Latin, three letters or more, lower case."""
    return frozenset(w.casefold() for w in _WORD.findall(text))


def asked_names(queries: Sequence[str | None]) -> frozenset[str]:
    """Capitalised words of the questions (lower case): the candidates for a company or entity name."""
    out: set[str] = set()
    for q in queries:
        if q:
            out.update(w.casefold() for w in _WORD.findall(q) if w[0].isupper())
    return frozenset(out - _NOT_NAMES)


def named_documents(queries: Sequence[str | None], labels: Mapping[str, str]) -> NamedDocuments | None:
    """The documents among ``labels`` (document id → document label) that ``queries`` name, or None when they name none
    (no capitalised word of the question is in some labels and not in others), or when fewer than two documents are
    labelled (nothing to tell apart)."""
    docs = {d: split_document_label(label) for d, label in labels.items() if label.strip()}
    if len(docs) < 2:
        return None
    asked = asked_names(queries)
    if not asked:
        return None
    by_name = {d: label_words(name) for d, (name, _) in docs.items()}
    by_label = {d: by_name[d] | label_words(title) for d, (_, title) in docs.items()}
    asked -= frozenset.intersection(*by_label.values())  # in every document's label: tells none apart
    if not asked:
        return None
    for words in (by_name, by_label):
        named = {d for d, w in words.items() if w & asked}
        if named:
            matched = sorted(asked & frozenset().union(*(words[d] for d in named)))
            return NamedDocuments(frozenset(named), tuple(matched))
    return None


def documents_by_name(named: NamedDocuments, labels: Mapping[str, str]) -> dict[str, frozenset[str]]:
    """The named documents grouped by the name that matched them ("valmora" → the Valmora documents, "zephyra" → the
    Zephyra deck): a question naming two companies compares them (the live canvas puts one table of each side by
    side). File name words first, as in ``named_documents``."""
    out: dict[str, frozenset[str]] = {}
    docs = {d: split_document_label(labels[d]) for d in named.document_ids if d in labels}
    for name in named.names:
        for part in (0, 1):
            found = frozenset(d for d, label in docs.items() if name in label_words(label[part]))
            if found:
                out[name] = found
                break
    return out


# "both companies", "the two reports", "each company", "compare the documents", Hindi "दोनों कंपनियों", Hinglish "dono
# companies": a question about the documents side by side, whether or not it names them. "Compare FY23 and FY24" and
# "the company's revenue" are not (periods, one company).
_ENTITY = r"compan(?:y|ies)|firms?|businesses|organi[sz]ations?|documents?|reports?|decks?|presentations?|files?"
_COMPARES = re.compile(
    rf"(?<![a-z])(?:(?:both|two|each|every|all)\s+(?:of\s+)?(?:the\s+)?(?:{_ENTITY})"
    r"|(?:compar\w*|versus|vs\.?|between|across)\s+(?:the\s+)?(?:two\s+|both\s+)?(?:companies|documents|reports|decks|files)"
    rf"|dono\s+(?:companies|company|companiyon|documents|reports))(?![a-z])"
    r"|(?<![ऀ-ॿ])(?:दोनों|सभी|हर)\s+(?:कं?पनि|कम्पनि|दस्तावे|रिपोर्ट)[ऀ-ॿ]*",
    re.IGNORECASE,
)


def compares_documents(texts: Sequence[str | None]) -> bool:
    """The question (or its English query) explicitly puts the documents or companies side by side without naming them
    ("show both companies' revenue", "दोनों कंपनियों का राजस्व"). A question that names two companies compares them too
    (``documents_by_name`` finds two), which is for the caller to see."""
    return any(_COMPARES.search(t) for t in texts if t)


def mentions(text: str, names: Sequence[str]) -> bool:
    """Does ``text`` contain one of ``names`` as a word (case-insensitive; "Valmora's" contains "valmora")?"""
    if not names:
        return False
    pattern = "|".join(re.escape(n) for n in names)
    return re.search(rf"(?<![^\W\d_])(?:{pattern})(?![^\W\d_])", text, re.IGNORECASE) is not None


# ------------------------------------------------------------------ misheard names (quality round, item 10)
#
# Speech recognition writes a company's name as words that sound like it ("Wall Mora", "well Mora", "Wilmura" for
# Valmora). Retrieval still finds the company (its label is embedded with every chunk), but the answer model echoes
# the name it was given ("Mora's revenue"). ``misheard_names`` maps such words of a question to the documents' own
# spelling, so the answer uses it.

# File name and title words that name no one (they describe the document).
_GENERIC_LABEL_WORDS = wordset(
    """
    annual report reports document documents file files final draft copy scan scanned notes minutes deck slides
    presentation investor investors policy policies travel expense expenses group health insurance statement
    statements financial financials results quarterly quarter summary overview brochure manual handbook contract
    agreement memo letter board meeting plan budget data sheet table tables appendix version updated english hindi
    soochna yojana industries industry limited private company companies holdings logistics services global india
    indian international corporation
    """
)
_LATIN_WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
# Words that never start a two-word name ("is Zefira" is "is" + a name).
_FUNCTION_WORDS = wordset(
    "is a an of in on at to by as it its and or for from with that this about me my our us show tell give did does "
    "do how much many was were be"
)


def sound_key(word: str) -> str:
    """The consonants of a word as they sound, in order, repeats merged: "Valmora", "Wall Mora", "Wilmura" and
    "wellmore" are all "vlmr"; "Zephyra" and "Zefira" are "sfr"."""
    w = word.casefold()
    for a, b in (("ph", "f"), ("ck", "k"), ("qu", "k"), ("q", "k"), ("w", "v"), ("x", "ks"), ("z", "s"), ("c", "k")):
        w = w.replace(a, b)
    consonants = [ch for ch in w if ch.isalpha() and ch not in "aeiouyh"]
    return "".join(ch for i, ch in enumerate(consonants) if i == 0 or ch != consonants[i - 1])


def _label_names(labels: Mapping[str, str]) -> dict[str, str]:
    """The names in the documents' labels: lower case → as written (a title's capitals, else capitalised)."""
    out: dict[str, str] = {}
    skip = _GENERIC_LABEL_WORDS | _NOT_NAMES
    for label in labels.values():
        name, title = split_document_label(label)
        for word in _WORD.findall(title):
            if len(word) >= 5 and word.casefold() not in skip and word[0].isupper():
                out.setdefault(word.casefold(), word)
        for word in _WORD.findall(name):
            if len(word) >= 5 and word.casefold() not in skip:
                out.setdefault(word.casefold(), word[:1].upper() + word[1:])
    return out


def _sounds_like(said: str, names: Mapping[str, str]) -> str | None:
    """The one document name ``said`` sounds like (the same consonants, or the end or start of it: "Mora" for
    "Valmora"), or None (none, or more than one)."""
    key = sound_key(said)
    low = said.casefold()
    found = {
        n
        for n in names
        if (len(key) >= 3 and key == sound_key(n))
        or (len(low) >= 4 and len(low) * 2 >= len(n) and (n.endswith(low) or n.startswith(low)))
    }
    return names[found.pop()] if len(found) == 1 else None


_HINDI_POSSESSIVES = frozenset({"ka", "ki", "ke"})  # "Valmora ka FY24" may come out as "Valmuraka, FY24"


def misheard_names(texts: Sequence[str | None], labels: Mapping[str, str]) -> dict[str, str]:
    """Words of the question (one capitalised word, or two adjacent words one of which is capitalised) that sound like
    a name in the documents' labels without being spelled like it: misheard words → the documents' spelling
    ({"Wall Mora": "Valmora"}). Empty when the question spells every name right or names none."""
    names = _label_names(labels)
    if not names:
        return {}
    out: dict[str, str] = {}
    for text in texts:
        if not text:
            continue
        tokens = list(_LATIN_WORD.finditer(text))
        i = 0
        while i < len(tokens):
            for span in (2, 1):
                group = tokens[i : i + span]
                if len(group) < span or (span == 2 and text[group[0].end() : group[1].start()] != " "):
                    continue
                words = [re.sub(r"['\u2019]s$", "", g.group(0)) for g in group]
                if not any(w[:1].isupper() for w in words) or any(w.casefold() in _NOT_NAMES for w in words):
                    continue
                if span == 2 and words[0].casefold() in _FUNCTION_WORDS:
                    continue
                said = "".join(words)
                if said.casefold() in names or any(w.casefold() in names for w in words):
                    continue  # spelled as the documents spell it
                name = _sounds_like(said, names)
                if name is None and len(said) >= 7 and said[-2:].casefold() in _HINDI_POSSESSIVES:
                    name = _sounds_like(said[:-2], names)  # "Valmuraka": "Valmora ka" heard as one word
                if name is not None:
                    out[" ".join(words)] = name
                    i += span - 1
                    break
            i += 1
    return out


# ------------------------------------------------------------------ misheard words (last round, item 1)

# Devanagari consonants by how they sound to a recognizer that confuses aspiration, voicing and dental / retroflex
# ("टूलकिट" heard as "तूलकेच", "बैंक ऋण" as "बेख रिन"); vowel signs, the anusvara and the virama say little.
_HINDI_SOUNDS = {
    **dict.fromkeys("कख", "k"),
    **dict.fromkeys("चछजझ", "c"),
    **dict.fromkeys("टठडढतथदध", "t"),
    **dict.fromkeys("णनञङ", "n"),
    **dict.fromkeys("पफबभव", "p"),
    "म": "m",
    "य": "y",
    "र": "r",
    "ऋ": "r",
    "ृ": "r",
    **dict.fromkeys("लळ", "l"),
    **dict.fromkeys("शषस", "s"),
    "ह": "h",
}
_DEVANAGARI_WORD = re.compile("[ऀ-ॣॱ-ॿ]+")
# Words of a question that say nothing about its subject (question words, postpositions, auxiliaries, common verbs).
_HINDI_FUNCTION_WORDS = wordset(
    """
    क्या कितना कितनी कितने कौन कौनसा कौनसी कब कहाँ कहां क्यों कैसे का की के को में से पर तक और या भी है हैं था थी थे
    हो होगा होगी होता होती होते मिलता मिलती मिलते मिलेगा मिलेगी मिलेंगे दिया दी दिए दिया जाता जाती जाते जाएगा जाएगी
    करना करता करती करते करें कर सकता सकती सकते चाहिए लिए लिये वाला वाली वाले यह वह ये वो इस उस इसका उसका इसकी उसकी
    बताओ बताइए बताइये बताएं बताएँ मुझे हमें आप आपको आपके आपकी अपने अपनी कुछ सब सभी बारे बार एक दो
    """
)


HINDI_FUNCTION_WORDS = _HINDI_FUNCTION_WORDS  # (question words, postpositions, auxiliaries, common verbs)
DEVANAGARI_WORD = _DEVANAGARI_WORD


def hindi_sound_key(word: str) -> str:
    """A Devanagari word's consonants as they sound, repeats merged: "टूलकिट" → "tlkt", "बैंक" and "बेख" → "pk", "ऋण"
    and "रिन" → "rn"."""
    keys = [_HINDI_SOUNDS.get(ch, "") for ch in unicodedata.normalize("NFD", word)]
    keys = [k for k in keys if k]
    return "".join(k for i, k in enumerate(keys) if i == 0 or k != keys[i - 1])


def misheard_words(question: str, passages: Sequence[str], *, common: bool = False) -> dict[str, str]:
    """Words of a (spoken) question that the passages have in another spelling that sounds the same: likely misheard
    ({"बेख": "बैंक", "रिन": "ऋण"}). Devanagari words of three letters or more that aren't function words and aren't in
    any passage as written; a sound key of two consonants or more, equal to a passage word's (one that is spelled
    differently, or of four consonants or more and one consonant off: "तूलकेच" for "टूलकिट"). Latin words are left
    to ``misheard_names``. ``common``: Hindi's question words and postpositions count as written too ("ख्या" is
    "क्या" misheard, polish round, item 4)."""
    norm = lambda w: unicodedata.normalize("NFC", w).replace("़", "")  # noqa: E731  (nukta: "दस्तावेज़" = "दस्तावेज")
    written = {norm(w) for p in passages for w in _DEVANAGARI_WORD.findall(p)}
    if common:
        written |= {norm(w) for w in _HINDI_FUNCTION_WORDS}
    by_key: dict[str, str] = {}
    for w in sorted(written, key=len, reverse=True):
        key = hindi_sound_key(w)
        if len(key) >= 2:
            by_key.setdefault(key, w)
    out: dict[str, str] = {}
    for word in _DEVANAGARI_WORD.findall(question):
        w = norm(word)
        if len(w) < 3 or w in written or w in _HINDI_FUNCTION_WORDS:
            continue
        key = hindi_sound_key(w)
        if len(key) >= 2 and key in by_key:
            out[word] = by_key[key]
        elif len(key) >= 4:
            near = [v for k, v in by_key.items() if k[0] == key[0] and _one_off(k, key)]
            if len(near) == 1:
                out[word] = near[0]
    return out


def _one_off(a: str, b: str) -> bool:
    """One substitution, insertion or deletion apart."""
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


# ------------------------------------------------------------------ the subject a question names (last round, item 4)

# Devanagari consonants as the Latin letters ``sound_key`` keeps: "वालमोरा" → "vlmr" (Valmora), "ज़ेफायरा" → "sfr"
# (Zephyra), "सूर्योदय" → "srd" (Suryodaya). A nukta changes a few (ज़ → z → s, फ़ → f, ड़ → r).
_LATIN_SOUNDS = {
    **dict.fromkeys("कख", "k"), **dict.fromkeys("गघ", "g"), **dict.fromkeys("चछ", "k"), **dict.fromkeys("जझ", "j"),
    **dict.fromkeys("टठतथ", "t"), **dict.fromkeys("डढदध", "d"), **dict.fromkeys("णनञङंँ", "n"), "प": "p", "फ": "f",
    **dict.fromkeys("बभ", "b"), "म": "m", "र": "r", **dict.fromkeys("लळ", "l"), "व": "v", **dict.fromkeys("शषस", "s"),
}  # fmt: skip
_NUKTA_SOUNDS = {"ज": "s", "फ": "f", "ड": "r", "ढ": "r", "क": "k", "ख": "k", "ग": "g"}


def devanagari_sound_key(word: str) -> str:
    """A Devanagari word's consonants as ``sound_key`` writes a Latin name's, repeats merged."""
    chars = unicodedata.normalize("NFD", word)
    keys = []
    for i, ch in enumerate(chars):
        nukta = i + 1 < len(chars) and chars[i + 1] == "़"
        key = _NUKTA_SOUNDS.get(ch) if nukta else None
        key = key or _LATIN_SOUNDS.get(ch, "")
        if key:
            keys.append(key)
    return "".join(k for i, k in enumerate(keys) if i == 0 or k != keys[i - 1])


def subject_names(text: str | None, labels: Mapping[str, str]) -> frozenset[str]:
    """The documents' names (lower case, from their labels: "valmora", "suryodaya") that ``text`` names: spelled as
    they are, misheard (``misheard_names``: "Valmuraka" is Valmora), or in Devanagari ("वालमोरा")."""
    names = _label_names(labels)
    if not text or not names:
        return frozenset()
    said = {re.sub(r"['\u2019]s$", "", w).casefold() for w in _LATIN_WORD.findall(text)}
    found = {n for n in names if n in said}
    found |= {name.casefold() for name in misheard_names([text], labels).values()}
    for word in _DEVANAGARI_WORD.findall(text):
        key = devanagari_sound_key(word)
        if len(key) >= 3:
            found |= {n for n in names if sound_key(n) == key}
    return frozenset(found)


def respell(text: str, renames: Mapping[str, str]) -> str:
    """``text`` with each misheard name (or Hindi word) replaced by the documents' spelling (whole words: "Mora's" →
    "Valmora's", "बेख" → "बैंक")."""
    for said, name in sorted(renames.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![A-Za-z\u0900-\u097f]){re.escape(said)}(?![A-Za-z\u0900-\u097f])", name, text, flags=re.I)
    return text
