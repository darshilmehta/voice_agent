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


def mentions(text: str, names: Sequence[str]) -> bool:
    """Does ``text`` contain one of ``names`` as a word (case-insensitive; "Valmora's" contains "valmora")?"""
    if not names:
        return False
    pattern = "|".join(re.escape(n) for n in names)
    return re.search(rf"(?<![^\W\d_])(?:{pattern})(?![^\W\d_])", text, re.IGNORECASE) is not None
