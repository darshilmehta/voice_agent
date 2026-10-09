# ruff: noqa: RUF001  (the folding table below is made of look-alike characters on purpose)
"""Text normalisation shared by the corpus verification and the retrieval metrics.

Evidence strings (the verbatim text a planted fact consists of) are compared with page texts and chunk texts after
the same normalisation: Unicode NFC, dashes/quotes/no-break spaces folded to ASCII, Devanagari digits to ASCII
digits, case folded, all whitespace collapsed to single spaces. That makes the match robust to line wrapping,
table markdown padding and typographic variants, but never to wrong numbers.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE = re.compile(r"\s+")
_FOLD = str.maketrans(
    {
        " ": " ",
        " ": " ",
        " ": " ",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "­": None,  # soft hyphen
        "​": None,  # zero-width space
        "‍": None,  # zero-width joiner
        "‌": None,  # zero-width non-joiner
    }
)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NOT_ALNUM = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """The comparison form of ``text`` (see the module docstring)."""
    text = unicodedata.normalize("NFC", text).translate(_FOLD).translate(_DEVANAGARI_DIGITS).casefold()
    return _WHITESPACE.sub(" ", text).strip()


def squash(text: str) -> str:
    """Letters and digits only, normalised: for approximate matching of OCR output (spaces and punctuation lost)."""
    return _NOT_ALNUM.sub("", normalize(text))


def contains_all(text: str, needles: Iterable[str], *, approximate: bool = False) -> bool:
    """True when every needle occurs in ``text`` (normalised; ``approximate`` ignores spaces and punctuation).

    No needles (or an empty one) is never a match: a fact without evidence can't be checked."""
    needles = list(needles)
    if not needles:
        return False
    fold = squash if approximate else normalize
    haystack = fold(text)
    for needle in needles:
        folded = fold(needle)
        if not folded or folded not in haystack:
            return False
    return True
