"""Turn routes and conversation state (docs/DESIGN.md §3.4, §3.5).

A ``TurnRoute`` is the validated decision for one user message: what kind of turn it is, whether the documents are
searched and with which query, the topic and the answer language. The model only proposes a route; application code
validates it and owns the ``ConversationState``, which it updates from validated routes after each turn.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from ..settings import Language

Intent = Literal[
    "document_qa",  # about the documents' content, including follow-ups
    "general_qa",  # general knowledge, unrelated to the documents
    "mixed",  # document facts + general knowledge or judgement
    "conversation",  # greetings, thanks, small talk, questions about the assistant
    "resume_document",  # back to the documents after a digression ("back to the report")
    "correction",  # revises the previous or interrupted question ("no, I meant FY23")
    "stop",  # be quiet
    "backchannel",  # "okay", "mm-hmm", "haan": nothing to answer
    "clarification",  # too ambiguous to answer: ask a short question back
    "canvas_edit",  # changes a visual on the chat's canvas ("make it a bar chart", "हटा दो"), §12.1
]
INTENTS: tuple[Intent, ...] = get_args(Intent)
SILENT_INTENTS: frozenset[Intent] = frozenset({"stop", "backchannel"})  # no answer, nothing spoken
# Turns that keep the conversation where it is: no new topic.
TOPIC_NEUTRAL_INTENTS: frozenset[Intent] = frozenset(
    {"conversation", "clarification", "stop", "backchannel", "canvas_edit"}
)
# Does the answer deserve a visual on the canvas (§12.1)? "requested": the user asked to see something; "suggest": a
# trend, comparison or breakdown that a chart shows better; "none". Set by application code, never by the model.
VisualWant = Literal["none", "suggest", "requested"]

RouteSource = Literal[
    "heuristic",  # keyword fast path (stop, backchannel, an explicit question about the documents)
    "retrieval",  # a standalone question the documents clearly answer: the router call was cancelled
    "llm",  # the router model's proposal, validated
    "fallback",  # the router failed or timed out: phase-1 behaviour (a document question)
]


class TurnRoute(BaseModel):
    """§3.4. ``rewritten_query``: the question made standalone (follow-ups, corrections), None when the utterance
    already stands alone; ``query_en``: an English search query for non-English turns (the reranker scores EN-EN
    best); ``tools``: the live-data tools to run (§3.7); ``visual``: whether the answer gets a visual on the canvas
    (§12.1, ``services/canvas/conversation.visual_want``). The router model proposes only the intent and the
    standalone English question; ``needs_retrieval`` (the retrieval policy), ``topic`` (content words of the
    question), ``is_topic_shift``, ``response_language`` (services/language.py) and ``confidence`` (by how the route
    was decided) are set by application code."""

    model_config = ConfigDict(frozen=True)

    intent: Intent
    needs_retrieval: bool
    rewritten_query: str | None = None
    query_en: str | None = None
    tools: list[Literal["web_search"]] = Field(default_factory=list)
    topic: str = ""
    is_topic_shift: bool = False
    response_language: Language
    confidence: float = Field(ge=0, le=1)
    visual: VisualWant = "none"  # additive (§12.1): from the question's words, set by application code


class ConversationState(BaseModel):
    """§3.5, persisted per chat (``chat_states``) and read by the next turn. Defaults are a new chat's state."""

    model_config = ConfigDict(from_attributes=True)

    chat_id: str
    active_topic: str | None = None
    previous_topic: str | None = None
    document_topic: str | None = None  # topic of the last document turn: what "back to the report" returns to
    document_query: str | None = None  # the last standalone document question
    active_document_ids: list[str] = Field(default_factory=list)  # documents the last document answer cited
    input_language: Language | None = None  # language of the user's last message
    response_language: Language | None = None  # language of the last answer
    preferred_language: Language | None = None  # the language the user asked for ("answer in Hindi"); sticks
    last_intent: Intent | None = None  # of the last turn that wasn't stop or backchannel
    last_interrupted_message_id: str | None = None  # the answer the user last cut off, until a later one completes
    retrieval_enabled: bool = True  # False: document questions are answered from general knowledge, saying so
    updated_at: datetime | None = None
