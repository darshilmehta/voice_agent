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
                if name is not None:
                    out[" ".join(words)] = name
                    i += span - 1
                    break
            i += 1
    return out


def respell(text: str, renames: Mapping[str, str]) -> str:
    """``text`` with each misheard name replaced by the documents' spelling (whole words: "Mora's" → "Valmora's")."""
    for said, name in sorted(renames.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(said)}(?![A-Za-z])", name, text, flags=re.IGNORECASE)
    return text
