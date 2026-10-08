"""Prompts and fixed texts of the answer (docs/DESIGN.md §3.3 a, §3.4, §3.6).

The answer prompt grounds the model in numbered sources: document facts only from them, every fact cited as [S#],
numbers copied exactly, "the documents don't cover it" instead of guessing, and never a claim that it can't access
documents (§9.5). Abstentions and errors are fixed texts, so they can never invent facts.

Turns the router sends elsewhere get their own prompt (``AnswerMode``): general questions are answered from general
knowledge with no citations and a prompt that says so; mixed questions cite document facts and mark general knowledge
as such; conversation gets a one-line reply; an unclear request a short question back. Fixed texts, no model and no
retrieval (so never an abstention): returning to the documents without a question ("back to the report"), an
acknowledgement said while the agent is idle ("okay", "theek hai" → "Anything else?", and nothing at all if that was
just asked), thanks and greetings. "Stop" gets nothing. The model is never called without either sources or an
explicit non-document prompt (§9.5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..settings import Language
from .sources import Source, format_sources

PROMPT_VERSION = "answer-v1"

AnswerMode = Literal[
    "grounded",  # document question: numbered sources, citations, abstention gate
    "mixed",  # document facts with citations + general knowledge, marked as such
    "general",  # general knowledge, no sources, no citations
    "conversation",  # a short social reply
    "clarification",  # one short question back
    "resume",  # fixed text: back to the documents, what we were talking about
    "ack",  # fixed short reply: "Anything else?" after an acknowledgement, "You're welcome." after thanks
    "silent",  # stop (or a second acknowledgement in a row): nothing is said
]
# Prompt ids recorded in a message's route (None: fixed text, no model).
PROMPT_IDS: dict[AnswerMode, str | None] = {
    "grounded": PROMPT_VERSION,
    "mixed": "mixed-v1",
    "general": "general-v1",
    "conversation": "conversation-v1",
    "clarification": "clarification-v1",
    "resume": None,
    "ack": None,
    "silent": None,
}
# Why a turn that wanted the documents is answered from general knowledge instead.
GeneralNote = Literal["not_covered", "no_documents", "retrieval_off"]

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


def answer_system_prompt(language: Language, length: AnswerLength = "short", *, mixed: bool = False) -> str:
    """The grounded prompt; ``mixed`` adds that general knowledge may put the document facts in context."""
    name = LANGUAGE_NAMES[language]
    keep = " Keep figures, source ids and terms such as EBITDA or FY24 exactly as written." if language == "hi" else ""
    general = (
        "8. The question also needs general knowledge or judgement. You may add it after the document facts, but say "
        'plainly that it is general knowledge, not from the documents (for example "In general, …"), and never '
        "cite a source for it.\n"
        if mixed
        else ""
    )
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
        "7. The sources are the user's documents: never say that you can't access documents or files.\n"
        f"{general}"
    ).rstrip("\n")


_GENERAL_NOTES: dict[GeneralNote, str] = {
    "not_covered": (
        "The user's documents were searched and don't cover this. Say so in a few words first, then answer from "
        "general knowledge."
    ),
    "no_documents": (
        "This chat has no ready documents yet. If the user expects an answer from their documents, say in a few "
        "words that there are none to check yet; then answer from general knowledge."
    ),
    "retrieval_off": (
        "Document search is turned off for this chat. If the user expects an answer from their documents, say in a "
        "few words that you are answering from general knowledge instead."
    ),
}


def general_system_prompt(language: Language, length: AnswerLength = "short", note: GeneralNote | None = None) -> str:
    """A question that isn't about the user's documents (or that they don't cover): general knowledge, said
    honestly, no citations."""
    situation = _GENERAL_NOTES[note] if note else "This question is not about the user's documents."
    return (
        "You are a voice and text assistant that talks with the user about their uploaded documents and also answers "
        "general questions.\n"
        f"{situation}\n"
        "Rules:\n"
        "1. Answer from general knowledge. Don't cite sources and never write source markers like [S1].\n"
        "2. Never claim that anything you say comes from the user's documents.\n"
        "3. If you don't know, or the answer needs live or current data (prices, news, weather), say so briefly "
        "instead of guessing.\n"
        f"4. {ANSWER_LENGTHS[length].instruction}\n"
        f"5. Answer in {LANGUAGE_NAMES[language]}.\n"
        "6. Never say that you can't access the user's documents or files."
    )


def conversation_system_prompt(language: Language) -> str:
    """Greetings, thanks, small talk, questions about the assistant."""
    return (
        "You are a friendly voice assistant that helps the user with questions about their uploaded documents.\n"
        "Reply to the user's last remark naturally, in one short sentence (two at most). Don't state any facts about "
        "the documents and never write source markers like [S1]. Never say that you can't access documents.\n"
        f"Reply in {LANGUAGE_NAMES[language]}."
    )


def clarification_system_prompt(language: Language) -> str:
    """An unclear request: ask what the user means instead of guessing."""
    return (
        "You are a voice assistant that answers questions about the user's uploaded documents.\n"
        "The user's last request is unclear. Ask exactly one short question to find out what they mean (for example "
        "which year, figure, document or topic). Don't answer it and don't guess. Never write source markers like "
        f"[S1].\nAsk in {LANGUAGE_NAMES[language]}."
    )


def general_user_prompt(question: str, language: Language) -> str:
    return f"{question.strip()}\n\n(Answer in {LANGUAGE_NAMES[language]}.)"


def with_memory(system: str, memory: str | None) -> str:
    """The system prompt with the chat's memory summary (§3.5): what was said before the recent messages."""
    memory = (memory or "").strip()
    if not memory:
        return system
    return f"{system}\n\nEarlier in this conversation (summary; the recent messages follow):\n{memory}"


# ------------------------------------------------------------------ fixed texts

RESUME_TEXTS: dict[Language, tuple[str, str]] = {  # (with a topic, without)
    "en": (
        "Sure, back to {documents}. We were talking about {topic}. What would you like to know?",
        "Sure, back to {documents}. What would you like to know?",
    ),
    "hi": (
        "ज़रूर, {documents} पर वापस चलते हैं। हम {topic} के बारे में बात कर रहे थे। आप क्या जानना चाहेंगे?",
        "ज़रूर, {documents} पर वापस चलते हैं। आप क्या जानना चाहेंगे?",
    ),
}
YOUR_DOCUMENTS: dict[Language, str] = {"en": "your documents", "hi": "आपके दस्तावेज़ों"}


def document_name(filename: str) -> str:
    """A filename as it reads aloud: "annual_report.pdf" → "annual report"."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return " ".join(stem.replace("_", " ").replace("-", " ").split()) or filename


def resume_text(language: Language, topic: str | None, filenames: Sequence[str]) -> str:
    """Back to the documents without a new question: no model, nothing invented."""
    if len(filenames) == 1:
        documents = f"the {document_name(filenames[0])}" if language == "en" else document_name(filenames[0])
    else:
        documents = YOUR_DOCUMENTS[language]
    with_topic, without = RESUME_TEXTS[language]
    return with_topic.format(documents=documents, topic=topic) if topic else without.format(documents=documents)


AckKind = Literal["ack", "thanks", "greeting", "language"]
ACK_TEXTS: dict[AckKind, dict[Language, str]] = {
    # After "okay" / "got it" / "theek hai" while the agent is idle: what ChatGPT's voice mode does, keep the floor
    # open in two words. (A backchannel *during* an answer never gets here: the voice session resumes playback.)
    "ack": {"en": "Anything else?", "hi": "और कुछ जानना है?"},
    "thanks": {"en": "You're welcome.", "hi": "आपका स्वागत है।"},
    "greeting": {"en": "Hi! What would you like to know?", "hi": "नमस्ते! आप क्या जानना चाहेंगे?"},
    "language": {"en": "Sure, I'll answer in English from now on.", "hi": "ज़रूर, अब से मैं हिंदी में जवाब दूँगा।"},
}


def ack_text(kind: AckKind, language: Language) -> str:
    return ACK_TEXTS[kind][language]


# Saved (as an ``event`` message, nothing spoken) for turns that get no answer.
SILENT_NOTICES = {"stop": "Stopped", "backchannel": "Acknowledged"}


# ------------------------------------------------------------------ memory summary

MEMORY_PROMPT_VERSION = "memory-v1"


def memory_system_prompt() -> str:
    return (
        "You keep the memory of a conversation between a user and an assistant that answers questions about the "
        "user's documents. Update the memory with the new messages and output the whole updated memory.\n"
        "Write at most 10 short bullet points in English:\n"
        "- what the user asked about and what was found, keeping figures, years and page numbers exactly;\n"
        "- questions the documents could not answer;\n"
        "- the user's preferences, such as the language they want answers in.\n"
        "Drop greetings and small talk. Output only the bullet points."
    )


def memory_user_prompt(previous: str | None, transcript: str) -> str:
    return f"Current memory:\n{(previous or '').strip() or '(empty)'}\n\nNew messages:\n{transcript.strip()}"


def answer_user_prompt(question: str, sources: Sequence[Source], language: Language) -> str:
    return (
        f"Sources:\n\n{format_sources(sources)}\n\n"
        f"Question: {question.strip()}\n\n"
        f"(Answer in {LANGUAGE_NAMES[language]}, citing the sources like [S1].)"
    )
