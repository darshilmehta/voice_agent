"""Prompts and fixed texts of the document answer (docs/DESIGN.md §3.3 a, §3.6).

The answer prompt grounds the model in numbered sources: document facts only from them, every fact cited as [S#],
numbers copied exactly, "the documents don't cover it" instead of guessing, and never a claim that it can't access
documents (§9.5). Abstentions and errors are fixed texts, so they can never invent facts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..settings import Language
from .sources import Source, format_sources

PROMPT_VERSION = "answer-v1"

LANGUAGE_NAMES: dict[Language, str] = {"en": "English", "hi": "Hindi, in Devanagari script"}

AbstainReason = Literal["not_covered", "no_documents"]
AnswerLength = Literal["short", "full"]


@dataclass(frozen=True, slots=True)
class LengthStyle:
    instruction: str  # rule 5 of the system prompt
    max_tokens: int  # generation cap (Hindi needs ~0.75-0.9 tokens per character, §9.1)


ANSWER_LENGTHS: dict[AnswerLength, LengthStyle] = {
    # Voice (§3.3 a): a short spoken reply; tables and detail stay on screen as citations.
    "short": LengthStyle(
        "Be brief: one to three sentences that read well aloud. Leave further detail to the cited sources unless "
        "the user asks for it. No preamble.",
        384,
    ),
    # Text: a fuller written answer.
    "full": LengthStyle(
        "Answer completely but concisely: a short paragraph, or a few bullet points when listing several figures. "
        "No preamble.",
        768,
    ),
}

ABSTENTIONS: dict[AbstainReason, dict[Language, str]] = {
    "not_covered": {
        "en": "I couldn't find that in this chat's documents, so I can't answer it from them.",
        "hi": "इस चैट के दस्तावेज़ों में इस बारे में जानकारी नहीं मिली, इसलिए इसका उत्तर उनसे नहीं दिया जा सकता।",
    },
    "no_documents": {
        "en": (
            "This chat has no ready documents yet. Upload a document to the project, or wait until it has "
            "finished processing, and ask again."
        ),
        "hi": (
            "इस चैट में अभी कोई तैयार दस्तावेज़ नहीं है। प्रोजेक्ट में दस्तावेज़ अपलोड करें, या उसकी प्रोसेसिंग पूरी होने तक रुकें, और फिर से पूछें।"
        ),
    },
}


def abstention(language: Language, reason: AbstainReason) -> str:
    return ABSTENTIONS[reason][language]


def answer_system_prompt(language: Language, length: AnswerLength = "short") -> str:
    name = LANGUAGE_NAMES[language]
    keep = " Keep figures, source ids and terms such as EBITDA or FY24 exactly as written." if language == "hi" else ""
    return (
        "You answer questions about the user's documents in a voice and text assistant.\n"
        "Rules:\n"
        "1. For any fact about the documents use only the numbered sources in the user's message. Never invent or "
        "guess figures, names or dates, and don't use outside knowledge for document facts.\n"
        "2. Right after each fact, cite its source id in square brackets, like [S1] or [S1][S2]. Cite only ids that "
        "appear in the sources.\n"
        "3. Copy numbers exactly as the sources write them, with their units, percent signs and years. Never round, "
        "convert or recompute them.\n"
        "4. If the sources don't contain the answer, say briefly that the documents don't cover it. Don't guess.\n"
        f"5. {ANSWER_LENGTHS[length].instruction}\n"
        f"6. Answer in {name}.{keep}\n"
        "7. The sources are the user's documents: never say that you can't access documents or files."
    )


def answer_user_prompt(question: str, sources: Sequence[Source], language: Language) -> str:
    return (
        f"Sources:\n\n{format_sources(sources)}\n\n"
        f"Question: {question.strip()}\n\n"
        f"(Answer in {LANGUAGE_NAMES[language]}, citing the sources like [S1].)"
    )
