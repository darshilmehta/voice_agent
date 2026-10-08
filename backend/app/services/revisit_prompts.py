"""Prompts and text cleaning for the "revisit" features: automatic chat titles and user-facing summaries (§3.9).

Separate from ``prompts.py``, which holds the document-answer prompts. Everything here is for a 4B model with a small
context: short instructions, one job per call, structured JSON for summaries, and strict post-processing so a sloppy
reply (quotes, a "Title:" prefix, a trailing full stop) never reaches the sidebar.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, field_validator

from ..providers.llm import LLMMessage
from ..settings import Language
from .chats import DEFAULT_TITLE
from .prompts import LANGUAGE_NAMES

TITLE_PROMPT_VERSION = "title-v1"
SUMMARY_PROMPT_VERSION = "summary-v1"

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


# ------------------------------------------------------------------ summaries

MAX_KEY_POINTS = 8
MAX_FOLLOW_UPS = 3
OVERVIEW_CHARS = 700
POINT_CHARS = 320
FOLLOW_UP_CHARS = 200
QUESTION_CHARS = 300  # an unanswered question as listed in the summary
MESSAGE_CHARS = 1000  # one message as shown to the summarizer

NO_ANSWER_LINE = "(no answer: the documents did not cover this)"


_SOURCE_NUMBER = re.compile(r"\[?\s*S?\s*(\d{1,6})\s*\]?", re.IGNORECASE)


class DraftPoint(BaseModel):
    """One key point as the model writes it: ``sources`` are chat-wide citation numbers ([3] → 3)."""

    text: str
    sources: list[int]

    @field_validator("sources", mode="before")
    @classmethod
    def _numbers(cls, value: Any) -> list[int]:
        """Models write [3], "3", "[3]", "S3" or 3.0 for the same thing; keep the positive numbers, drop the rest."""
        if not isinstance(value, list):
            return []
        numbers: list[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            if isinstance(item, float) and item.is_integer():
                item = int(item)
            if isinstance(item, int):
                number = item
            elif isinstance(item, str) and (m := _SOURCE_NUMBER.fullmatch(item)):
                number = int(m.group(1))
            else:
                continue
            if number > 0:
                numbers.append(number)
        return numbers


class SummaryDraft(BaseModel):
    """What the LLM returns for a summary (one window of the chat, or a combination of partial summaries). All fields
    are required so the constrained decoder always writes them; the unanswered questions are not asked of the model
    (they come from the abstained turns themselves, which cannot be mis-copied)."""

    overview: str
    key_points: list[DraftPoint]
    follow_ups: list[str]


_SUMMARY_LANGUAGE_RULE = {
    "en": "Write everything in English.",
    "hi": (
        "Write everything in Hindi, in Devanagari script. Keep figures, citation numbers and terms such as EBITDA "
        "or FY24 exactly as written."
    ),
}

_SUMMARY_FORMAT = (
    "Reply with JSON only:\n"
    '- "overview": one to three sentences on what this was about.\n'
    f'- "key_points": at most {MAX_KEY_POINTS} short statements of what was established from the documents. Each has '
    '"text" and "sources": the citation numbers (like 3 for [3]) of the answers it comes from, or an empty list '
    "when they cite none. Copy figures, names and dates exactly.\n"
    f'- "follow_ups": at most {MAX_FOLLOW_UPS} things the user may want to ask or check next, grounded in the '
    "conversation; an empty list if there are none."
)


def summary_system_prompt(language: Language) -> str:
    return (
        "You summarise one conversation between a user and a document assistant, so the user can revisit it later.\n"
        'The conversation is a list of lines "#<n> User:" and "#<n> Assistant:". Assistant lines cite documents as '
        "[1], [2] …; the list of sources after the conversation names the document and page of each number.\n"
        f"{_SUMMARY_FORMAT}\n"
        "Rules:\n"
        "1. Use only what the conversation says. Never add facts, figures or citation numbers that are not in it.\n"
        f"2. Lines saying {NO_ANSWER_LINE} are questions the documents could not answer: make no key point from "
        "them.\n"
        f"3. {_SUMMARY_LANGUAGE_RULE[language]}"
    )


def summary_user_prompt(lines: Sequence[str], legend: Sequence[str]) -> str:
    sources = "\n".join(legend) if legend else "(none: no answer in this part cites a document)"
    return "Conversation:\n" + "\n".join(lines) + "\n\nSources:\n" + sources


def reduce_system_prompt(language: Language) -> str:
    return (
        "You combine partial summaries of consecutive parts of one conversation between a user and a document "
        "assistant into a single summary.\n"
        f"{_SUMMARY_FORMAT}\n"
        "Rules:\n"
        "1. Use only what the partial summaries say. Merge duplicates; keep the most important points.\n"
        "2. Keep each key point's citation numbers exactly as given; never add numbers that are not in the partial "
        "summaries.\n"
        f"3. {_SUMMARY_LANGUAGE_RULE[language]}"
    )


def reduce_user_prompt(parts: Sequence[str], legend: Sequence[str]) -> str:
    sources = "\n".join(legend) if legend else "(none)"
    return "Partial summaries:\n\n" + "\n\n".join(parts) + "\n\nSources:\n" + sources


def render_partial(index: int, draft: SummaryDraft) -> str:
    """A partial summary as the reduce prompt shows it."""
    lines = [f"Part {index}:", f"Overview: {draft.overview}"]
    if draft.key_points:
        lines.append("Key points:")
        lines.extend(f"- {p.text}{''.join(f'[{n}]' for n in p.sources)}" for p in draft.key_points)
    if draft.follow_ups:
        lines.append("Follow-ups:")
        lines.extend(f"- {f}" for f in draft.follow_ups)
    return "\n".join(lines)


# Headings of the rendered summary, by the summary's language.
HEADINGS: dict[Language, dict[str, str]] = {
    "en": {
        "overview": "Overview",
        "key_points": "Key points",
        "unanswered": "Questions the documents couldn't answer",
        "follow_ups": "Open follow-ups",
    },
    "hi": {
        "overview": "सारांश",
        "key_points": "मुख्य बिंदु",
        "unanswered": "वे प्रश्न जिनका उत्तर दस्तावेज़ों में नहीं मिला",
        "follow_ups": "आगे के प्रश्न",
    },
}
