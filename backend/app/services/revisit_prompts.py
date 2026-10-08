"""Prompts and text cleaning for the "revisit" features: automatic chat titles (§3.9).

Separate from ``prompts.py``, which holds the document-answer prompts. Everything here is for a 4B model with a small
context: short instructions, one job per call, structured JSON for summaries, and strict post-processing so a sloppy
reply (quotes, a "Title:" prefix, a trailing full stop) never reaches the sidebar.
"""

from __future__ import annotations

import re

from ..providers.llm import LLMMessage
from ..settings import Language
from .chats import DEFAULT_TITLE
from .prompts import LANGUAGE_NAMES

TITLE_PROMPT_VERSION = "title-v1"

# ------------------------------------------------------------------ titles

TITLE_MAX_WORDS = 6
TITLE_MAX_CHARS = 80
FALLBACK_TITLE_CHARS = 60
TITLE_QUESTION_CHARS = 400  # of the first question sent to the model
TITLE_ANSWER_CHARS = 300  # of the first answer

_KEEP_HI = " Keep figures and terms such as EBITDA or FY24 exactly as written."


def title_messages(language: Language, question: str, answer: str | None) -> list[LLMMessage]:
    """The prompt for a chat title. ``answer`` is the agent's first answer (None when the documents didn't cover the
    question: its fixed apology says nothing about the topic)."""
    system = (
        "You write titles for conversations about the user's documents.\n"
        "Rules:\n"
        f"1. At most {TITLE_MAX_WORDS} words, naming the subject (for example: FY24 EBITDA margin).\n"
        f"2. Write the title in {LANGUAGE_NAMES[language]}.{_KEEP_HI if language == 'hi' else ''}\n"
        "3. Plain words only: no quotes, no markdown, no trailing punctuation, no prefix such as Title.\n"
        "4. Reply with the title and nothing else."
    )
    parts = [f"First question: {_clip(question, TITLE_QUESTION_CHARS)}"]
    if answer:
        parts.append(f"Answer: {_clip(answer, TITLE_ANSWER_CHARS)}")
    return [LLMMessage("system", system), LLMMessage("user", "\n".join(parts))]


_MARKER = re.compile(r"\s*\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_PREFIX = re.compile(r"^(?:chat\s+)?(?:title|शीर्षक)\s*[:\uff1a\-\u2013\u2014]\s*", re.IGNORECASE)
# straight and curly quotes, guillemets, CJK corner brackets
_QUOTES = "\"'`\u201c\u201d\u2018\u2019\u201e\u00ab\u00bb\u2039\u203a\u300c\u300d\u300e\u300f"
_EDGE_JUNK = _QUOTES + "*_#> \t"
_TRAILING = ".,;:!?\u2026\u0964\u0965-\u2013\u2014"  # … । ॥ and dashes


def clean_title(raw: str) -> str | None:
    """A model reply as a sidebar title: first line, no [S#] markers, "Title:" prefix, markdown or quotes, no
    trailing punctuation, at most ``TITLE_MAX_WORDS`` words and ``TITLE_MAX_CHARS`` characters. None if nothing is
    left (or it is the placeholder)."""
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    line = line.replace("**", "").replace("__", "")  # bold
    line = _PREFIX.sub("", _MARKER.sub("", line).strip().lstrip("#> ").strip())
    previous = None
    while previous != line:  # quotes, markdown marks and punctuation can be nested: "“Margin.”"
        previous = line
        line = line.strip(_EDGE_JUNK).rstrip(_TRAILING + _EDGE_JUNK).strip()
    words = line.split()
    if len(words) > TITLE_MAX_WORDS:
        line = " ".join(words[:TITLE_MAX_WORDS]).rstrip(_TRAILING + _EDGE_JUNK)
    line = _SPACE.sub(" ", line)
    if len(line) > TITLE_MAX_CHARS:
        line = _cut(line, TITLE_MAX_CHARS).rstrip("…")
    if not line or line.casefold() == DEFAULT_TITLE.casefold():
        return None
    return line


def fallback_title(question: str) -> str | None:
    """The first user question as a title when the model gave none: markers and extra spaces removed, trailing
    question mark or full stop dropped, cut at a word boundary to ``FALLBACK_TITLE_CHARS`` characters with an
    ellipsis. None for an empty question."""
    text = _SPACE.sub(" ", _MARKER.sub("", question)).strip().strip(_EDGE_JUNK)
    text = text.rstrip(_TRAILING + _EDGE_JUNK)
    if len(text) > FALLBACK_TITLE_CHARS:
        text = _cut(text, FALLBACK_TITLE_CHARS)
    return text or None


def _cut(text: str, limit: int) -> str:
    """At most ``limit`` characters, ending at a word boundary (when one is near) with an ellipsis."""
    if len(text) <= limit:
        return text
    head = text[: limit - 1]
    space = head.rfind(" ")
    if space > limit // 2:
        head = head[:space]
    return head.rstrip(" ,;:।") + "…"


def _clip(text: str, limit: int) -> str:
    return _SPACE.sub(" ", text).strip()[:limit]
