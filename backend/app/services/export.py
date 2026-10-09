"""Transcript export: a chat as a Markdown or JSON file to download (docs/DESIGN.md §3.9). No audio: it isn't stored.

Both formats are built from the same ``TranscriptExport`` model, so they always say the same thing:

- Markdown reads like a document: title and facts, the user summary (if one exists), every message with speaker, time
  and modality, what was actually heard of an interrupted answer, inline citations as ``[1]`` and a sources list.
- JSON is the stable machine form. ``schema_version`` changes only when a field is removed or changes meaning; new
  fields may be added without a new version. ``messages[].text`` keeps the answer's own ``[S#]`` markers exactly as
  saved; ``messages[].citations[]`` maps each ``source_id`` to ``ref``, the number in the top-level ``sources`` list
  (and in the Markdown's ``[n]``).

Source numbers are chat-wide: one per distinct document and page range, in order of first citation (``chat_sources``).
Live web results (``[W#]``, §3.7) are sources too, one per URL: their citations and sources carry ``kind: "web"``,
``url`` and ``title`` (document entries are unchanged: no ``kind`` means a document), and Markdown lists them with
their title and link.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from ..domain.projects import Chat, Message
from ..domain.summaries import SourceRef, SummaryData, UnansweredQuestion, UserSummary
from .base import Service
from .chat_sources import EN_DASH, ChatSources, number_markers, pages_label
from .chat_summary import render_summary_markdown, summary_view
from .chats import ChatService
from .markdown_text import escape_block_markers, escape_inline, safe_link
from .messages import MessageService
from .projects import ProjectService
from .summaries import SummaryService

EXPORT_SCHEMA_VERSION = 1
ExportFormat = Literal["md", "json"]

MEDIA_TYPES: dict[ExportFormat, str] = {"md": "text/markdown; charset=utf-8", "json": "application/json"}
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi"}
SPEAKERS = {"user": "You", "agent": "Assistant", "event": "Event"}
SLUG_MAX = 60


# ------------------------------------------------------------------ the JSON schema (also the Markdown's source)


class ExportProject(BaseModel):
    id: str
    name: str
    archived: bool


class ExportChat(BaseModel):
    id: str
    title: str
    title_is_auto: bool
    project: ExportProject
    language: str | None  # the chat's configured language (None: decided per message)
    languages: list[str]  # languages actually used, in order of first appearance
    document_scope: list[str] | None  # None = all of the project's documents
    pinned: bool
    archived: bool
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None


class _WebFields(BaseModel):
    """``kind``, ``url`` and ``title`` appear on web sources only: document entries stay as they were."""

    @model_serializer(mode="wrap")
    def _without_web_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if data.get("kind") == "document":
            for name in ("kind", "url", "title"):
                data.pop(name, None)
        return data


class ExportCitation(_WebFields):
    source_id: str  # the marker in the message text: "S1" for [S1], "W1" for a web result [W1]
    ref: int  # the source's number in the top-level ``sources`` list
    document_id: str
    filename: str  # web results: the site
    page_start: int | None
    page_end: int | None
    chunk_id: str
    snippet: str
    kind: Literal["document", "web"] = "document"
    url: str | None = None
    title: str | None = None


class ExportMessage(BaseModel):
    seq: int
    id: str
    role: Literal["user", "agent", "event"]
    modality: Literal["voice", "text"]
    language: str | None
    created_at: datetime
    text: str  # the full text; an agent answer's [S#] markers as saved
    heard_text: str | None  # interrupted answers: what was actually played before the barge-in
    interrupted: bool
    citations: list[ExportCitation]
    route: dict[str, Any] | None  # router decision (intent, abstained, confidence, model, …)
    latency: dict[str, Any] | None  # stage timings in ms


class ExportSource(_WebFields):
    ref: int
    document_id: str
    filename: str  # web results: the site
    page_start: int | None
    page_end: int | None
    snippet: str
    cited_by: list[int]  # seq of the messages that cite it
    kind: Literal["document", "web"] = "document"
    url: str | None = None
    title: str | None = None


class ExportSummarySource(BaseModel):
    ref: int  # the number in ``sources``
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None


class ExportKeyPoint(BaseModel):
    text: str
    sources: list[ExportSummarySource]


class ExportSummary(BaseModel):
    language: str
    overview: str
    key_points: list[ExportKeyPoint]
    unanswered_questions: list[UnansweredQuestion]
    follow_ups: list[str]
    content: str  # the same summary as Markdown
    covers_seq: int
    message_count: int
    stale: bool  # messages were added after the last one it covers
    model: str | None
    created_at: datetime


class TranscriptExport(BaseModel):
    schema_version: Literal[1] = EXPORT_SCHEMA_VERSION
    exported_at: datetime
    chat: ExportChat
    messages: list[ExportMessage]
    sources: list[ExportSource]
    summary: ExportSummary | None


@dataclass(frozen=True, slots=True)
class ExportFile:
    filename: str  # may contain non-ASCII letters (a Hindi title)
    ascii_filename: str
    media_type: str
    body: bytes

    @property
    def content_disposition(self) -> str:
        return content_disposition(self.filename, self.ascii_filename)


# ------------------------------------------------------------------ service


class ExportService(Service):
    async def export(self, chat_id: str, fmt: ExportFormat = "md") -> ExportFile:
        """The chat as a file. Unknown chat → NotFound. A chat without messages exports fine (title and facts only)."""
        chat = await ChatService(self.db, clock=self.now).get(chat_id)
        project = await ProjectService(self.db, clock=self.now).get(chat.project_id)
        messages = await MessageService(self.db, clock=self.now).all(chat_id)
        stored = await SummaryService(self.db, clock=self.now).get(chat_id, "user")
        summary = summary_view(stored) if stored is not None else None

        sources = ChatSources()
        exported = [_export_message(m, sources) for m in messages]
        summary_md: str | None = None
        export_summary: ExportSummary | None = None
        if summary is not None:
            export_summary, summary_md = _export_summary(summary, sources)
        doc = TranscriptExport(
            exported_at=self.now(),
            chat=_export_chat(chat, project.name, project.archived, messages),
            messages=exported,
            sources=[
                ExportSource(
                    ref=e.number,
                    document_id=e.document_id,
                    filename=e.filename,
                    page_start=e.page_start,
                    page_end=e.page_end,
                    snippet=e.snippet,
                    cited_by=e.cited_by,
                    kind=e.kind,
                    url=e.url,
                    title=e.title,
                )
                for e in sources.entries
            ],
            summary=export_summary,
        )
        body = render_json(doc) if fmt == "json" else render_markdown(doc, summary_md)
        filename, ascii_filename = export_filenames(chat, fmt)
        return ExportFile(filename, ascii_filename, MEDIA_TYPES[fmt], body.encode("utf-8"))


def _export_chat(chat: Chat, project_name: str, project_archived: bool, messages: list[Message]) -> ExportChat:
    languages = list(dict.fromkeys(m.language for m in messages if m.language in LANGUAGE_NAMES))
    return ExportChat(
        id=chat.id,
        title=chat.title,
        title_is_auto=chat.title_is_auto,
        project=ExportProject(id=chat.project_id, name=project_name, archived=project_archived),
        language=chat.language,
        languages=[str(lang) for lang in languages] or ([chat.language] if chat.language else []),
        document_scope=chat.document_scope,
        pinned=chat.pinned,
        archived=chat.archived,
        message_count=chat.message_count,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        last_message_at=chat.last_message_at,
    )


def _export_message(m: Message, sources: ChatSources) -> ExportMessage:
    mapping = sources.register(m.seq, m.citations)
    return ExportMessage(
        seq=m.seq,
        id=m.id,
        role=m.role,
        modality=m.modality,
        language=m.language,
        created_at=m.created_at,
        text=m.text,
        heard_text=m.heard_text,
        interrupted=m.heard_text is not None,
        citations=[
            ExportCitation(
                source_id=c.source_id,
                ref=mapping[c.source_id.upper()],
                document_id=c.document_id,
                filename=c.filename,
                page_start=c.page_start,
                page_end=c.page_end,
                chunk_id=c.chunk_id,
                snippet=c.snippet,
                kind=c.kind,
                url=c.url,
                title=c.title,
            )
            for c in m.citations
        ],
        route=m.route,
        latency=m.latency,
    )


def _export_summary(summary: UserSummary, sources: ChatSources) -> tuple[ExportSummary, str]:
    """The summary for the export, its sources as numbers of the export's sources list, and its Markdown with
    those numbers as citations."""

    def ref_number(ref: SourceRef) -> int:
        return (sources.find(ref) or sources.add(ref)).number  # a source no message cites can't happen; don't lose it

    key_points = [
        ExportKeyPoint(
            text=p.text,
            sources=[
                ExportSummarySource(
                    ref=ref_number(r),
                    document_id=r.document_id,
                    filename=r.filename,
                    page_start=r.page_start,
                    page_end=r.page_end,
                )
                for r in p.sources
            ],
        )
        for p in summary.key_points
    ]
    data = SummaryData(
        language=summary.language,
        overview=summary.overview,
        key_points=summary.key_points,
        unanswered_questions=summary.unanswered_questions,
        follow_ups=summary.follow_ups,
        windows=1,
        prompt="export",
    )
    markdown = render_summary_markdown(data, lambda refs: "".join(f"[{ref_number(r)}]" for r in refs), level=3)
    exported = ExportSummary(
        language=summary.language,
        overview=summary.overview,
        key_points=key_points,
        unanswered_questions=summary.unanswered_questions,
        follow_ups=summary.follow_ups,
        content=summary.content,
        covers_seq=summary.covers_seq,
        message_count=summary.message_count,
        stale=summary.stale,
        model=summary.model,
        created_at=summary.created_at,
    )
    return exported, markdown


# ------------------------------------------------------------------ rendering


def render_json(doc: TranscriptExport) -> str:
    return json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


_SPACE = re.compile(r"\s+")


def _utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _text(text: str) -> str:
    """Message text as Markdown that can't break the document's structure: no heading or code fence of its own."""
    return escape_block_markers(text.replace("\r\n", "\n").strip())


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.split("\n"))


def render_markdown(doc: TranscriptExport, summary_markdown: str | None = None) -> str:
    chat = doc.chat
    languages = ", ".join(LANGUAGE_NAMES.get(lang, lang) for lang in chat.languages) or "—"
    voice = sum(1 for m in doc.messages if m.modality == "voice")
    facts = [
        ("Project", chat.project.name + (" (archived)" if chat.project.archived else "")),
        ("Created", _utc(chat.created_at)),
        ("Languages", languages),
        ("Messages", f"{len(doc.messages)} ({voice} voice, {len(doc.messages) - voice} text)"),
        ("Exported", _utc(doc.exported_at)),
    ]
    out = [f"# {_SPACE.sub(' ', chat.title).strip()}", ""]
    out += [f"- **{name}:** {value}" for name, value in facts]

    if doc.summary is not None and summary_markdown:
        s = doc.summary
        out += ["", "## Summary", ""]
        if s.stale:
            out += [f"_Out of date: this summary covers messages 1{EN_DASH}{s.covers_seq} of {s.message_count}._", ""]
        out.append(summary_markdown)

    out += ["", "## Transcript", ""]
    if not doc.messages:
        out.append("_No messages yet._")
    for m in doc.messages:
        out += _render_message(m)

    if doc.sources:
        out += ["## Sources", ""]
        for s in doc.sources:
            out.append(_render_source(s))
    return "\n".join(out).rstrip() + "\n"


def _render_message(m: ExportMessage) -> list[str]:
    mapping = {c.source_id.upper(): c.ref for c in m.citations}
    head = [SPEAKERS[m.role], _utc(m.created_at), m.modality]
    if m.interrupted:
        head.append("interrupted")
    out = [f"### {' · '.join(head)}", ""]
    if m.role == "event":
        out += [f"_{_text(m.text)}_", ""]
    elif m.interrupted:
        out += [
            _quote(f"**Heard:** {_text(number_markers(m.heard_text or '', mapping)) or '(nothing)'}"),
            "",
            f"**Full answer:** {_text(number_markers(m.text, mapping))}",
            "",
        ]
    else:
        out += [_text(number_markers(m.text, mapping)), ""]
    return out


def _render_source(s: ExportSource) -> str:
    pages = pages_label(s.page_start, s.page_end)
    where = f"**{s.filename or '(unknown document)'}**" + (f", {pages}" if pages else "")
    snippet_text = _SPACE.sub(" ", s.snippet).strip()
    if s.kind == "web":  # a live web result: its title, the site and the link, all text from the web: escaped
        title = escape_inline(s.title or s.filename or "web result")
        link = safe_link(s.url)
        where = f"**{title}** (web: {escape_inline(s.filename)})" + (f" <{link}>" if link else "")
        snippet_text = escape_inline(snippet_text)
    snippet = f" — “{snippet_text}”" if snippet_text else ""
    cited = ""
    if s.cited_by:
        cited = f" _(cited in message{'s' if len(s.cited_by) > 1 else ''} {', '.join(f'#{n}' for n in s.cited_by)})_"
    return f"- [{s.ref}] {where}{snippet}{cited}"


# ------------------------------------------------------------------ file names


def slugify(text: str, limit: int = SLUG_MAX) -> str:
    """Lower-case words joined by "-", keeping letters, digits and the combining marks of scripts such as Devanagari
    (which ``str.isalnum`` would split on); "" when nothing is left."""
    text = unicodedata.normalize("NFKC", text).casefold()
    words: list[str] = []
    word: list[str] = []
    for ch in text:
        if unicodedata.category(ch)[0] in "LNM":
            word.append(ch)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    slug = "-".join(words)
    if len(slug) > limit:
        slug = slug[:limit].rsplit("-", 1)[0] if "-" in slug[:limit] else slug[:limit]
    return slug


def export_filenames(chat: Chat, fmt: ExportFormat) -> tuple[str, str]:
    """The download's name and its ASCII form: "fy24-margins-2026-10-08.md", the slugified title and the chat's
    creation date (UTC), "chat" when the title leaves no slug (the ASCII form of a Hindi title)."""
    day = chat.created_at.astimezone(UTC).strftime("%Y-%m-%d")
    ascii_title = unicodedata.normalize("NFKD", chat.title).encode("ascii", "ignore").decode()
    return f"{slugify(chat.title) or 'chat'}-{day}.{fmt}", f"{slugify(ascii_title) or 'chat'}-{day}.{fmt}"


def content_disposition(filename: str, ascii_filename: str) -> str:
    """``attachment`` header: the ASCII name and, when the real name differs (a Hindi title), the UTF-8 one too
    (RFC 6266 / 5987), which browsers prefer."""
    header = f'attachment; filename="{ascii_filename}"'
    if filename != ascii_filename:
        header += f"; filename*=UTF-8''{quote(filename, safe='')}"
    return header
