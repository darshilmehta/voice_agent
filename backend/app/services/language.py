"""Which language to answer in (docs/DESIGN.md §1, §3.4): the language of the user's latest message unless the
request asks for one. The app speaks English and Hindi only."""

from __future__ import annotations

import re

from ..settings import Language

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")


def message_language(text: str) -> Language | None:
    """'hi' when at least as many words are written in Devanagari as in Latin script, 'en' when Latin words are the
    majority, None without words in either script.

    Counted by words, not characters: Hindi questions routinely carry Latin terms ("FY24 में EBITDA margin क्या था?"
    is Hindi), while an English question with one Hindi word ("What is the मार्जिन?") is English. Romanized Hindi
    (Hinglish) reads as English: telling them apart needs a model (the phase-3 router).
    """
    devanagari = latin = 0
    for word in text.split():
        if _DEVANAGARI.search(word):
            devanagari += 1
        elif _LATIN.search(word):
            latin += 1
    if devanagari == 0 and latin == 0:
        return None
    return "hi" if devanagari >= latin else "en"


def choose_language(text: str, requested: Language | None, fallback: Language) -> Language:
    """The answer language: ``requested`` when given, else the message's script, else ``fallback`` (the chat's
    language or the app default)."""
    if requested is not None:
        return requested
    return message_language(text) or fallback
