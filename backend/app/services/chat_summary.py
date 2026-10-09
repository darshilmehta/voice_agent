"""User-facing chat summaries: "what did we talk about" (docs/DESIGN.md §3.9, §3.5).

    transcript ──► chat-wide source numbers ([S2] of answer 4 → [3]) ──► message windows that fit the context
        one window:   LLM → SummaryDraft (overview, key points with source numbers, follow-ups)
        more windows: LLM per window → partial drafts → LLM combines (grouped again while they don't fit) → draft
    draft ──► grounding: a key point keeps only source numbers that were in the text the model saw
          ──► numbers → document and page references (SourceRef) ──► SummaryData + Markdown ──► ``chat_summaries``

The unanswered questions are not asked of the model: they are the user's questions of the turns where the agent
abstained (``route.abstained``) or answered from general knowledge because the documents didn't cover the question
(``route.general_note == "not_covered"``), copied as said. This is a different artifact from the internal memory summary
(§3.5), which keeps prompts short; this one is for the user to read.

A summary is kept with the last message it covers. Asking again for an unchanged chat (and the same language) returns
the stored one without calling the model; messages added since mark it ``stale``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from ..db.types import utcnow
from ..domain.projects import ChatSummary, Message
from ..domain.summaries import (
    SUMMARY_SCHEMA_VERSION,
    KeyPoint,
    SourceRef,
    SummaryData,
    UnansweredQuestion,
    UserSummary,
)
from ..providers.llm import LLMClient, LLMError, LLMMessage, LLMUnavailableError
from ..providers.registry import Container
from ..providers.storage import MetadataDB
from ..settings import Language, Settings
from .base import InvalidInput, Service, Unavailable
from .chat_sources import ChatSources, number_markers, ref_label
from .chats import ChatService
from .markdown_text import escape_block_markers, escape_inline
from .messages import MessageService
from .revisit_prompts import (
    FOLLOW_UP_CHARS,
    HEADINGS,
    MAX_FOLLOW_UPS,
    MAX_KEY_POINTS,
    MESSAGE_CHARS,
    NO_ANSWER_LINE,
    OVERVIEW_CHARS,
    POINT_CHARS,
    QUESTION_CHARS,
    SUMMARY_PROMPT_VERSION,
    DraftPoint,
    SummaryDraft,
    reduce_system_prompt,
    reduce_user_prompt,
    render_partial,
    summary_system_prompt,
    summary_user_prompt,
)
from .sources import estimate_tokens
from .summaries import SummaryService

log = logging.getLogger(__name__)

# Context budget (tokens), for a model with ``llm.num_ctx`` (8192 for qwen3:4b-instruct): the prompt's instructions
# (~500), the sources legend (~500) and the JSON answer (up to ~2000, Hindi counting ~0.9 tokens per character, §9.1)
# come off the top; the rest is the conversation text of one window.
PROMPT_RESERVE_TOKENS = 3000
MIN_WINDOW_TOKENS = 800
SLACK = 0.9  # estimate_tokens errs high, but not by 10% everywhere
LLM_ATTEMPTS = 2  # one try, one retry when the model's JSON doesn't validate

_MARK = re.compile(r"\[(\d+)\]")
_INLINE_MARK = re.compile(r"\s*\[\d+\]")  # with the space before it: "x [3]." → "x."
_SPACE = re.compile(r"\s+")


def window_budget(num_ctx: int) -> int:
    """Tokens of conversation (or of partial summaries) per model call."""
    return max(MIN_WINDOW_TOKENS, int((num_ctx - PROMPT_RESERVE_TOKENS) * SLACK))


# ------------------------------------------------------------------ preparing the transcript


@dataclass(frozen=True, slots=True)
class Line:
    seq: int
    text: str  # "#3 User: …"
    tokens: int
    numbers: frozenset[int]  # chat-wide source numbers the line cites


@dataclass(frozen=True, slots=True)
class Prepared:
    lines: list[Line]
    sources: ChatSources
    unanswered: list[UnansweredQuestion]


def _flat(text: str, limit: int) -> str:
    text = _SPACE.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def prepare(messages: Sequence[Message]) -> Prepared:
    """The conversation as lines for the prompt, with chat-wide source numbers, plus the questions the agent
    abstained on. Agent lines show what the user heard (``heard_text``) when an answer was interrupted; an abstained
    answer's fixed apology is replaced by a marker so it can't be summarised as a fact."""
    sources = ChatSources()
    lines: list[Line] = []
    unanswered: list[UnansweredQuestion] = []
    asked: set[str] = set()
    last_user: Message | None = None
    for m in messages:
        if m.role == "user":
            last_user = m
            lines.append(_line(m.seq, f"#{m.seq} User: {_flat(m.text, MESSAGE_CHARS)}"))
        elif m.role == "agent":
            mapping = sources.register(m.seq, m.citations)
            route = m.route or {}
            uncovered = route.get("abstained") is True or route.get("general_note") == "not_covered"
            if uncovered and last_user is not None:
                question = _flat(last_user.text, QUESTION_CHARS)
                if question.casefold() not in asked:
                    asked.add(question.casefold())
                    unanswered.append(UnansweredQuestion(question=question, message_seq=last_user.seq))
            if route.get("abstained") is True:
                lines.append(_line(m.seq, f"#{m.seq} Assistant: {NO_ANSWER_LINE}"))
                continue
            heard = m.heard_text is not None
            body = _flat(number_markers(m.heard_text if heard else m.text, mapping), MESSAGE_CHARS)
            label = "Assistant (interrupted: the user heard only this)" if heard else "Assistant"
            if route.get("answer") == "general" or route.get("general_note") == "not_covered":
                label += " (general knowledge, not from the documents)"
            if route.get("web_sources"):  # live data (§3.7): its web sources are numbered like the documents
                label += " (includes live web results)"
            lines.append(_line(m.seq, f"#{m.seq} {label}: {body}"))
    return Prepared(lines, sources, unanswered)


def _line(seq: int, text: str) -> Line:
    numbers = frozenset(int(n) for n in _MARK.findall(text))
    return Line(seq, text, estimate_tokens(text) + 4, numbers)


def pack[T](items: Sequence[T], tokens: Callable[[T], int], budget: int, *, min_items: int = 1) -> list[list[T]]:
    """Consecutive groups of items whose tokens fit ``budget`` (a group has at least ``min_items``, so an item
    larger than the budget travels alone, or in a pair)."""
    groups: list[list[T]] = []
    current: list[T] = []
    used = 0
    for item in items:
        cost = tokens(item)
        if current and used + cost > budget and len(current) >= min_items:
            groups.append(current)
            current, used = [], 0
        current.append(item)
        used += cost
    if current:
        groups.append(current)
    return groups


def dominant_language(messages: Sequence[Message], chat_language: str | None, default: Language) -> Language:
    """The language most messages are in; a tie (or no language recorded) goes to the chat's language, then the
    app default."""
    counts = Counter(m.language for m in messages if m.language in ("en", "hi"))
    fallback: Language = chat_language if chat_language in ("en", "hi") else default  # type: ignore[assignment]
    if not counts:
        return fallback
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return fallback
    return top[0][0]  # type: ignore[return-value]


# ------------------------------------------------------------------ cleaning and grounding drafts


def _clip(text: str, limit: int) -> str:
    text = _SPACE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip(" ,;:।") + "…"


def ground(draft: SummaryDraft, allowed: set[int]) -> SummaryDraft:
    """The draft cleaned up and grounded: texts trimmed and limited in number and length, ``[n]`` markers the model
    wrote inside a point's text moved to its sources, and every source number that is not in ``allowed`` (the numbers
    in the text the model was shown) dropped."""
    points: list[DraftPoint] = []
    seen: set[str] = set()
    dropped = 0
    for p in draft.key_points:
        inline = [int(n) for n in _MARK.findall(p.text)]
        text = _clip(_INLINE_MARK.sub("", p.text), POINT_CHARS)
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        numbers = list(dict.fromkeys([*p.sources, *inline]))
        kept = [n for n in numbers if n in allowed]
        dropped += len(numbers) - len(kept)
        points.append(DraftPoint(text=text, sources=kept))
        if len(points) == MAX_KEY_POINTS:
            break
    follow_ups = list(
        dict.fromkeys(f for f in (_clip(_INLINE_MARK.sub("", f), FOLLOW_UP_CHARS) for f in draft.follow_ups) if f)
    )[:MAX_FOLLOW_UPS]
    if dropped:
        log.info("summary: dropped %d citation(s) that point to no source in the text the model was shown", dropped)
    return SummaryDraft(
        overview=_clip(_INLINE_MARK.sub("", draft.overview), OVERVIEW_CHARS), key_points=points, follow_ups=follow_ups
    )


# ------------------------------------------------------------------ rendering


def inline_citation(refs: Sequence[SourceRef], language: Language = "en") -> str:
    """ "(annual_report.pdf, p. 2; investor_deck.pdf, p. 7)"; a web result's title is text from the web, escaped."""
    return (
        "("
        + "; ".join(escape_inline(ref_label(r, language)) if r.kind == "web" else ref_label(r, language) for r in refs)
        + ")"
    )


def render_summary_markdown(
    data: SummaryData, cite: Callable[[Sequence[SourceRef]], str] | None = None, *, level: int = 2
) -> str:
    """The summary as Markdown with headings in its language. ``cite`` turns a key point's sources into the text that
    follows it (default: "(file.pdf, p. 2)"; the transcript export numbers them like its sources list)."""
    cite = cite or (lambda refs: inline_citation(refs, data.language))
    heading = "#" * level
    names = HEADINGS[data.language]
    safe = escape_block_markers  # the user's questions and the model's sentences can't become headings or quotes
    out: list[str] = []
    if data.overview:
        out += [f"{heading} {names['overview']}", "", safe(data.overview), ""]
    if data.key_points:
        out += [f"{heading} {names['key_points']}", ""]
        out += [f"- {safe(p.text)}{' ' + cite(p.sources) if p.sources else ''}" for p in data.key_points]
        out.append("")
    if data.unanswered_questions:
        out += [f"{heading} {names['unanswered']}", ""]
        out += [f"- {safe(q.question)}" for q in data.unanswered_questions]
        out.append("")
    if data.follow_ups:
        out += [f"{heading} {names['follow_ups']}", ""]
        out += [f"- {safe(f)}" for f in data.follow_ups]
        out.append("")
    return "\n".join(out).strip()


def summary_view(summary: ChatSummary) -> UserSummary:
    """The stored summary as the API returns it. A row whose ``data`` isn't ours (or was damaged) still shows its
    Markdown, as the overview."""
    try:
        data = SummaryData.model_validate(summary.data)
    except ValidationError:
        log.warning("chat %s: stored summary has unreadable data; showing its text", summary.chat_id)
        data = SummaryData(
            language="en",
            overview=summary.content,
            key_points=[],
            unanswered_questions=[],
            follow_ups=[],
            windows=1,
            prompt="unknown",
        )
    return UserSummary(
        id=summary.id,
        chat_id=summary.chat_id,
        language=data.language,
        overview=data.overview,
        key_points=data.key_points,
        unanswered_questions=data.unanswered_questions,
        follow_ups=data.follow_ups,
        content=summary.content,
        covers_seq=summary.covers_seq,
        message_count=summary.message_count,
        stale=summary.stale,
        model=summary.model,
        created_at=summary.created_at,
    )


# ------------------------------------------------------------------ service


class ChatSummarizer(Service):
    """``generate(chat_id)`` creates or refreshes the chat's user summary, ``get(chat_id)`` reads it. Create one per
    app: it serialises concurrent requests for the same chat so the model runs once."""

    def __init__(
        self,
        db: MetadataDB,
        *,
        llm: LLMClient,
        settings: Settings,
        clock: Callable[[], datetime] = utcnow,
        window_tokens: int | None = None,
    ) -> None:
        super().__init__(db, clock=clock)
        self.llm = llm
        self.settings = settings
        self.window_tokens = window_tokens or window_budget(settings.llm.num_ctx)
        self.chats = ChatService(db, clock=clock)
        self.messages = MessageService(db, clock=clock)
        self.summaries = SummaryService(db, clock=clock)
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    @classmethod
    def from_container(cls, container: Container) -> ChatSummarizer:
        db, llm = container["metadata_db"], container["llm"]
        if not isinstance(db, MetadataDB) or not isinstance(llm, LLMClient):
            raise TypeError(
                f"expected MetadataDB and LLMClient providers, got {type(db).__name__}, {type(llm).__name__}"
            )
        return cls(db, llm=llm, settings=container.settings)

    async def get(self, chat_id: str) -> UserSummary | None:
        """The chat's stored summary (None if it has none yet). Unknown chat → NotFound."""
        summary = await self.summaries.get(chat_id, "user")
        return summary_view(summary) if summary else None

    async def generate(self, chat_id: str, *, language: Language | None = None) -> UserSummary:
        """Create or refresh the summary and return it.

        Idempotent for an unchanged chat: if the stored summary covers the latest message (and is in the requested
        language, when one is requested) it is returned as it is. An empty chat → InvalidInput; the model failing →
        ``Unavailable`` (nothing is stored; retry later).
        """
        async with self._exclusive(chat_id):
            chat = await self.chats.get(chat_id)
            stored = await self.summaries.get(chat_id, "user")
            if stored is not None and not stored.stale:
                view = summary_view(stored)
                if language is None or view.language == language:
                    return view
            messages = await self.messages.all(chat_id)
            if not messages:
                raise InvalidInput("nothing to summarise: the chat has no messages yet")
            lang = language or dominant_language(messages, chat.language, self.settings.client.default_language)
            data = await self._summarise(messages, lang)
            saved = await self.summaries.save(
                chat_id,
                kind="user",
                content=render_summary_markdown(data),
                data=data.model_dump(mode="json"),
                covers_message_id=messages[-1].id,  # not "the latest": messages may arrive while the model works
                model=self.settings.llm.chat_model,
            )
            return summary_view(saved)

    @asynccontextmanager
    async def _exclusive(self, chat_id: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        self._users[chat_id] = self._users.get(chat_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._users[chat_id] -= 1
            if not self._users[chat_id]:
                del self._users[chat_id]
                self._locks.pop(chat_id, None)

    # -------------------------------------------------------------- map-reduce

    async def _summarise(self, messages: Sequence[Message], language: Language) -> SummaryData:
        prepared = prepare(messages)
        windows = pack(prepared.lines, lambda line: line.tokens, self.window_tokens)
        if not windows:  # only events: nothing was said
            raise InvalidInput("nothing to summarise: the chat has no user or agent messages yet")
        sources = prepared.sources
        partials: list[SummaryDraft] = []
        for window in windows:
            numbers = {n for line in window for n in line.numbers}
            prompt = [
                LLMMessage("system", summary_system_prompt(language)),
                LLMMessage("user", summary_user_prompt([line.text for line in window], _legend(sources, numbers))),
            ]
            partials.append(await self._ask(prompt, numbers))
        draft = partials[0] if len(partials) == 1 else await self._reduce(partials, sources, language)
        log.info(
            "chat summary: %d messages in %d window(s), %d key points",
            len(messages),
            len(windows),
            len(draft.key_points),
        )
        return SummaryData(
            schema_version=SUMMARY_SCHEMA_VERSION,
            language=language,
            overview=draft.overview,
            key_points=[_key_point(p, sources) for p in draft.key_points],
            unanswered_questions=prepared.unanswered,
            follow_ups=draft.follow_ups,
            windows=len(windows),
            prompt=SUMMARY_PROMPT_VERSION,
        )

    async def _reduce(self, partials: list[SummaryDraft], sources: ChatSources, language: Language) -> SummaryDraft:
        """Combine partial summaries into one. When their text doesn't fit one prompt they are combined in groups
        first, and those results again, until one call can take them all."""
        level = partials
        while True:
            texts = [render_partial(i, p) for i, p in enumerate(level, start=1)]
            sizes = [estimate_tokens(t) + 4 for t in texts]
            if len(level) == 1 or sum(sizes) <= self.window_tokens:
                return await self._combine(level, texts, sources, language)
            indexes = list(range(len(level)))
            groups = pack(indexes, sizes.__getitem__, self.window_tokens, min_items=2)
            if len(groups) >= len(level):  # nothing would shrink: combine in pairs
                groups = [indexes[i : i + 2] for i in range(0, len(indexes), 2)]
            level = [
                await self._combine([level[i] for i in g], [texts[i] for i in g], sources, language) for g in groups
            ]

    async def _combine(
        self, drafts: list[SummaryDraft], texts: list[str], sources: ChatSources, language: Language
    ) -> SummaryDraft:
        numbers = {n for d in drafts for p in d.key_points for n in p.sources}
        if len(drafts) == 1:
            return drafts[0]
        prompt = [
            LLMMessage("system", reduce_system_prompt(language)),
            LLMMessage("user", reduce_user_prompt(texts, _legend(sources, numbers))),
        ]
        return await self._ask(prompt, numbers)

    async def _ask(self, prompt: list[LLMMessage], allowed: set[int]) -> SummaryDraft:
        """One structured call: the model's JSON validated as a ``SummaryDraft`` (one retry when it doesn't validate),
        cleaned and grounded to ``allowed`` source numbers."""
        error: Exception | None = None
        for _ in range(LLM_ATTEMPTS):
            try:
                draft = await self.llm.generate_json(prompt, SummaryDraft, model=self.settings.llm.chat_model)
            except LLMUnavailableError as e:
                raise Unavailable(f"summary generation failed: {e}") from e
            except LLMError as e:
                error = e
                log.warning("summary: model output rejected: %s", e)
                continue
            except Exception as e:  # e.g. a placeholder provider
                raise Unavailable(f"summary generation failed: {type(e).__name__}: {e}") from e
            return ground(draft, allowed)
        raise Unavailable(f"summary generation failed: {error}")


def _legend(sources: ChatSources, numbers: set[int]) -> list[str]:
    """ "[3] annual_report.pdf, p. 2", "[4] web: livemint.com — Infosys share price" (a web title can't plant a
    "[n]" of its own: brackets become parentheses)."""
    return [
        f"[{n}] {ref_label(e.ref()).replace('[', '(').replace(']', ')')}"
        for n in sorted(numbers)
        if (e := sources.get(n)) is not None
    ]


def _key_point(point: DraftPoint, sources: ChatSources) -> KeyPoint:
    """A key point with its numbers turned into document and page references (a number that isn't a source of this
    chat is dropped: the last grounding check)."""
    refs: list[SourceRef] = []
    for n in point.sources:
        entry = sources.get(n)
        if entry is not None and entry.ref() not in refs:
            refs.append(entry.ref())
    return KeyPoint(text=point.text, sources=refs)
