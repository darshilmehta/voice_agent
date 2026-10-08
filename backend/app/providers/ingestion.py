"""Document parsing and chunking (docs/DESIGN.md §3.1).

``DoclingParser`` turns PDF/DOCX/PPTX/MD/TXT files into a ``ParsedDocument`` (pages, items with heading paths,
tables as markdown plus structured cells for typed datasets, §12.1) and splits it into ``Chunk``s with
provenance. Docling, transformers and the OCR engine are optional (``ml`` group) and imported lazily.

Chunking: Docling's ``HybridChunker`` (BGE-M3 tokenizer from the local snapshot) splits the document along its
structure; ``assemble_chunks`` then keeps every table whole as one chunk, merges small neighbouring pieces of the
same section up to the token budget and adds overlap between consecutive chunks of a section.
"""

from __future__ import annotations

import asyncio
import gc
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from . import models
from .base import HealthStatus, Provider, ProviderContext, ProviderHealth, local_snapshot
from .models import ModelUnavailableError, quiet_ml_env, require_modules
from .registry import register

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument

    from ..settings import ChunkingSection, IngestionSection

# Directories `docling-tools models download` creates under artifacts_path.
DOCLING_LAYOUT = "docling-project--docling-layout-heron"
DOCLING_TABLES = "docling-project--docling-models"
OCR_ARTIFACTS = {"rapidocr": "RapidOcr"}

# File types Docling converts (TXT is read directly: Docling has no plain-text backend).
DOCLING_FORMATS = frozenset({".pdf", ".docx", ".pptx", ".md"})

ContentType = Literal["paragraph", "list", "table"]
_TABLE_LABELS = frozenset({"table", "document_index"})
_CHUNK_NAMESPACE = uuid.UUID("5b0f6a4e-2f0c-4c1a-9a43-3f8e0f3c2d71")


class IngestionError(RuntimeError):
    """The document can't be ingested (type not allowed, too many pages, conversion failed, no text)."""


# ------------------------------------------------------------------ domain model


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class ParsedPage(_Model):
    number: int  # 1-based
    width: float
    height: float


class TableCell(_Model):
    """One cell of a table. ``row``/``col`` are 0-based; spanning cells appear once with their span."""

    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str
    column_header: bool = False
    row_header: bool = False


class ParsedTable(_Model):
    """A table kept whole: markdown for reading and retrieval, cells for typed datasets (§12.1)."""

    index: int  # 0-based position among the document's tables
    ref: str  # Docling reference (#/tables/N)
    page_start: int | None
    page_end: int | None
    bbox: tuple[float, float, float, float] | None  # (left, top, right, bottom) on page_start, top-left origin, pt
    heading_path: list[str]
    caption: str | None
    num_rows: int
    num_cols: int
    markdown: str
    cells: list[TableCell]

    def grid(self) -> list[list[str]]:
        """Rows of cell texts; a spanning cell's text fills every position it covers."""
        rows = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for c in self.cells:
            for r in range(c.row, min(c.row + c.row_span, self.num_rows)):
                for k in range(c.col, min(c.col + c.col_span, self.num_cols)):
                    rows[r][k] = c.text
        return rows


class ParsedItem(_Model):
    """One body element in reading order (heading, paragraph, list item, caption, table, …)."""

    ref: str
    label: str  # Docling label: title, section_header, text, paragraph, list_item, caption, table, …
    text: str  # tables: their markdown
    page: int | None
    heading_path: list[str]  # headings in scope; a heading's own path ends with itself
    level: int | None = None  # headings only (title = 0)
    table_index: int | None = None


class ParsedDocument(_Model):
    source_name: str
    format: str  # pdf, docx, pptx, md, txt
    page_count: int | None  # None for formats without pages (DOCX, MD, TXT)
    pages: list[ParsedPage]
    items: list[ParsedItem]
    tables: list[ParsedTable]
    markdown: str
    language: str | None
    parse_seconds: float
    warnings: list[str] = []

    _native: Any = PrivateAttr(default=None)  # the DoclingDocument, kept until chunking is done

    def drop_native(self) -> None:
        """Release the Docling tree once chunks exist; the rest of the model stays usable."""
        self._native = None


class Chunk(_Model):
    """A retrievable unit with provenance. ``text`` is shown and cited; ``embed_text`` is embedded and reranked."""

    chunk_id: str
    project_id: str
    document_id: str
    version: int
    chunk_index: int
    chunking_version: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    content_type: ContentType
    language: str | None
    text: str
    overlap_text: str = ""  # tail of the previous chunk of the same section (context for embedding)
    table_index: int | None = None
    token_count: int

    @property
    def embed_text(self) -> str:
        return compose_embed_text(self.heading_path, self.overlap_text, self.text)

    @property
    def point_id(self) -> str:
        return point_id(self.chunk_id)


def compose_embed_text(heading_path: Sequence[str], overlap_text: str, text: str) -> str:
    """Headings, overlap and text joined like Docling's ``contextualize`` (newline-delimited)."""
    return "\n".join([*heading_path, *([overlap_text] if overlap_text else []), text])


def chunk_id(document_id: str, version: int, index: int) -> str:
    return f"{document_id}:v{version}:{index:04d}"


def point_id(chunk_id: str) -> str:
    """Deterministic vector-store id for a chunk id, so re-ingesting overwrites instead of duplicating."""
    return str(uuid.uuid5(_CHUNK_NAMESPACE, chunk_id))


# ------------------------------------------------------------------ language

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")  # Devanagari block
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str | None:
    """'hi' when Devanagari is ≥30% of the letters, 'en' for Latin script, None without letters.

    The app speaks English and Hindi only (§1); Hindi text routinely carries Latin acronyms (EBITDA, FY24), hence
    the low threshold. Romanized Hindi (Hinglish) reads as 'en': telling them apart needs a model.
    """
    dev = len(_DEVANAGARI.findall(text))
    lat = len(_LATIN.findall(text))
    if dev + lat == 0:
        return None
    return "hi" if dev / (dev + lat) >= 0.3 else "en"


# ------------------------------------------------------------------ chunk assembly (pure, no Docling)


@dataclass(frozen=True, slots=True)
class ChunkPiece:
    """One HybridChunker output, reduced to what assembly needs."""

    text: str
    headings: tuple[str, ...]
    labels: tuple[str, ...]  # Docling labels of the piece's items
    pages: tuple[int, ...]
    table_ref: str | None = None  # set when the piece is (part of) a table


@dataclass(slots=True)
class _Block:
    headings: tuple[str, ...]
    texts: list[str]
    labels: set[str]
    pages: set[int]
    table: ParsedTable | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


def content_budget(chunking: ChunkingSection) -> int:
    """Tokens available to a chunk's headings + text, leaving room for the overlap within ``target_tokens``."""
    return max(chunking.target_tokens - chunking.overlap_tokens, chunking.target_tokens // 2, 1)


def table_text(table: ParsedTable) -> str:
    return f"{table.caption}\n{table.markdown}" if table.caption else table.markdown


_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def tail_text(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> str:
    """The end of ``text`` within ``max_tokens``: whole trailing sentences, else trailing words of the last one."""
    text = " ".join(text.split())
    if max_tokens <= 0 or not text:
        return ""
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    kept: list[str] = []
    for s in reversed(sentences):
        if count_tokens(" ".join([s, *kept])) > max_tokens:
            break
        kept.insert(0, s)
    if kept:
        return " ".join(kept)
    words = sentences[-1].split()
    lo, hi = 1, len(words)  # smallest start index whose suffix fits (suffixes shrink as start grows)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(" ".join(words[mid:])) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    suffix = " ".join(words[lo:])
    return suffix if suffix and count_tokens(suffix) <= max_tokens else ""


def assemble_chunks(
    pieces: Iterable[ChunkPiece],
    tables: Sequence[ParsedTable],
    *,
    project_id: str,
    document_id: str,
    version: int,
    chunking: ChunkingSection,
    count_tokens: Callable[[str], int],
    default_language: str | None = None,
) -> list[Chunk]:
    """Turn chunker pieces into chunks with provenance.

    - A table becomes exactly one chunk holding its whole markdown, even past the budget (the chunker may have
      split it into several pieces; they collapse back into one).
    - Consecutive text pieces with the same heading path merge while headings + text fit ``content_budget``.
    - A chunk that continues the section of the previous text chunk starts with that chunk's tail
      (``overlap_tokens``) in ``overlap_text``; tables neither give nor receive overlap.
    """
    by_ref = {t.ref: t for t in tables}
    budget = content_budget(chunking)
    blocks: list[_Block] = []
    seen_tables: set[str] = set()
    for p in pieces:
        table = by_ref.get(p.table_ref) if p.table_ref else None
        if table is not None:
            if table.ref not in seen_tables:
                seen_tables.add(table.ref)
                pages = {n for n in (table.page_start, table.page_end) if n is not None} or set(p.pages)
                blocks.append(_Block(p.headings, [table_text(table)], {"table"}, pages, table))
            continue
        if not p.text.strip():
            continue
        prev = blocks[-1] if blocks else None
        if (
            prev is not None
            and prev.table is None
            and prev.headings == p.headings
            and count_tokens("\n".join([*p.headings, *prev.texts, p.text])) <= budget
        ):
            prev.texts.append(p.text)
            prev.labels.update(p.labels)
            prev.pages.update(p.pages)
            continue
        blocks.append(_Block(p.headings, [p.text], set(p.labels), set(p.pages)))

    chunks: list[Chunk] = []
    for i, b in enumerate(blocks):
        prev = blocks[i - 1] if i else None
        overlap = ""
        if (
            chunking.overlap_tokens > 0
            and b.table is None
            and prev is not None
            and prev.table is None
            and prev.headings == b.headings
        ):
            overlap = tail_text(prev.text, chunking.overlap_tokens, count_tokens)
        text = b.text
        content: ContentType = (
            "table" if b.table is not None else "list" if b.labels and b.labels <= {"list_item"} else "paragraph"
        )
        cid = chunk_id(document_id, version, i)
        heading_path = list(b.headings)
        chunks.append(
            Chunk(
                chunk_id=cid,
                project_id=project_id,
                document_id=document_id,
                version=version,
                chunk_index=i,
                chunking_version=chunking.version,
                page_start=min(b.pages) if b.pages else None,
                page_end=max(b.pages) if b.pages else None,
                heading_path=heading_path,
                content_type=content,
                language=detect_language(text) or default_language,
                text=text,
                overlap_text=overlap,
                table_index=b.table.index if b.table is not None else None,
                token_count=count_tokens(compose_embed_text(heading_path, overlap, text)),
            )
        )
    return chunks


# ------------------------------------------------------------------ providers


class DocumentParser(Provider):
    capability = "ingestion"

    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a file into a ``ParsedDocument``. Raises IngestionError for files that can't be ingested."""
        raise NotImplementedError(f"{type(self).__name__}.parse")

    async def chunk(self, document: ParsedDocument, *, project_id: str, document_id: str, version: int) -> list[Chunk]:
        """Split a document returned by ``parse`` into chunks with provenance."""
        raise NotImplementedError(f"{type(self).__name__}.chunk")

    def release(self) -> None:
        """Free parsing models; they load again on the next parse (§8: Docling is loaded only for ingestion)."""


@register
class DoclingParser(DocumentParser):
    name = "docling"
    required_modules = ("docling", "docling_core", "transformers", "semchunk")

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._lock = threading.Lock()  # one conversion at a time; also guards (un)loading
        self._converter: DocumentConverter | None = None
        self._tokenizer: Any = None  # transformers tokenizer of the embedding model

    @property
    def cfg(self) -> IngestionSection:
        return self.config  # type: ignore[return-value]

    @property
    def artifacts_path(self) -> Path:
        return self.ctx.settings.path(self.cfg.artifacts_path)

    @property
    def loaded(self) -> bool:
        return self._converter is not None

    async def health(self) -> ProviderHealth:
        cfg = self.cfg
        path = self.artifacts_path
        if not path.is_dir():
            return self._health(HealthStatus.DOWN, f"{path} missing (run scripts/setup/download_models.sh docling)")
        needed = [DOCLING_LAYOUT, DOCLING_TABLES]
        if cfg.ocr and cfg.ocr_engine in OCR_ARTIFACTS:
            needed.append(OCR_ARTIFACTS[cfg.ocr_engine])
        missing = [d for d in needed if not (path / d).is_dir()]
        if missing:
            return self._health(HealthStatus.DOWN, f"missing model folders: {', '.join(missing)}")
        libs = models.missing_modules(self.required_modules)
        if libs:
            return self._health(HealthStatus.DEGRADED, f"{', '.join(libs)} not installed ({models.ML_INSTALL_HINT})")
        ocr = f"OCR {cfg.ocr_engine}" if cfg.ocr else "OCR off"
        return self._health(HealthStatus.OK, f"layout + table models present, {ocr}")

    async def parse(self, path: Path) -> ParsedDocument:
        return await asyncio.to_thread(self._parse_sync, Path(path))

    async def chunk(self, document: ParsedDocument, *, project_id: str, document_id: str, version: int) -> list[Chunk]:
        return await asyncio.to_thread(
            self._chunk_sync, document, project_id=project_id, document_id=document_id, version=version
        )

    async def close(self) -> None:
        await asyncio.to_thread(self.release)

    def release(self) -> None:
        """Drop the converter and its pipelines (layout, TableFormer, OCR sessions) so their memory and
        onnxruntime threads are freed now, not during interpreter shutdown (which aborts on macOS, §9.6)."""
        with self._lock:
            conv, self._converter = self._converter, None
            if conv is not None:
                conv.initialized_pipelines.clear()
                del conv
        gc.collect()
        models.free_torch_memory()

    # -------------------------------------------------------------- parsing

    def _parse_sync(self, path: Path) -> ParsedDocument:
        cfg = self.cfg
        suffix = path.suffix.lower()
        allowed = [e.lower() for e in cfg.allowed_extensions]
        if suffix not in allowed:
            raise IngestionError(f"{path.name}: file type {suffix or '(none)'} is not allowed ({', '.join(allowed)})")
        if suffix != ".txt" and suffix not in DOCLING_FORMATS:
            raise IngestionError(f"{path.name}: the docling parser can't read {suffix} files")
        if not path.is_file():
            raise IngestionError(f"{path}: file not found")
        require_modules("document parsing", self.required_modules)
        quiet_ml_env()

        t0 = time.perf_counter()
        warnings: list[str] = []
        if suffix == ".txt":
            doc = _text_document(path)
        else:
            doc, warnings = self._convert(path, suffix)
        parsed = to_parsed_document(
            doc,
            source_name=path.name,
            fmt=suffix.lstrip("."),
            parse_seconds=time.perf_counter() - t0,
            warnings=warnings,
        )
        parsed._native = doc
        return parsed

    def _convert(self, path: Path, suffix: str) -> tuple[DoclingDocument, list[str]]:
        from docling.datamodel.base_models import ConversionStatus

        if suffix == ".pdf" and not self.artifacts_path.is_dir():
            raise ModelUnavailableError(
                f"Docling models missing at {self.artifacts_path} (run scripts/setup/download_models.sh docling)"
            )
        with self._lock:
            conv = self._converter = self._converter or self._build_converter()
            try:
                res = conv.convert(path, raises_on_error=False, max_num_pages=self.cfg.max_pages)
            except Exception as e:  # docling raises for unreadable/corrupt input before producing a result
                raise IngestionError(f"{path.name}: conversion failed: {type(e).__name__}: {e}") from e
        errors = [e.error_message for e in res.errors]
        if res.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            detail = "; ".join(errors) or res.status.value
            raise IngestionError(f"{path.name}: conversion {res.status.value}: {detail}")
        return res.document, errors

    def _build_converter(self) -> DocumentConverter:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        cfg = self.cfg
        opts = PdfPipelineOptions(artifacts_path=str(self.artifacts_path))
        opts.do_table_structure = True
        opts.do_ocr = cfg.ocr
        if cfg.ocr:
            opts.ocr_options = _ocr_options(cfg.ocr_engine)
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.MD],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
        )

    # -------------------------------------------------------------- chunking

    def _chunk_sync(self, document: ParsedDocument, *, project_id: str, document_id: str, version: int) -> list[Chunk]:
        doc = document._native
        if doc is None:
            raise IngestionError(f"{document.source_name}: the Docling tree was released; parse the file again")
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        chunking = self.cfg.chunking
        tok = self._chunk_tokenizer(content_budget(chunking))
        chunker = HybridChunker(tokenizer=tok, merge_peers=False)  # merging is ours, so tables stay apart
        pieces = [to_piece(c) for c in chunker.chunk(dl_doc=doc)]
        return assemble_chunks(
            pieces,
            document.tables,
            project_id=project_id,
            document_id=document_id,
            version=version,
            chunking=chunking,
            count_tokens=tok.count_tokens,
            default_language=document.language,
        )

    def _chunk_tokenizer(self, max_tokens: int) -> Any:
        """Docling tokenizer wrapper. ``max_tokens`` must be explicit: left unset, Docling looks it up on the Hub."""
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        return HuggingFaceTokenizer(tokenizer=self._get_tokenizer(), max_tokens=max_tokens)

    def _get_tokenizer(self) -> Any:
        """The embedding model's tokenizer from its local snapshot, so chunk sizes are in embedder tokens."""
        with self._lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                repo = self.ctx.settings.embeddings.model
                snap = local_snapshot(self.ctx.hf_home, repo)
                if snap is None:
                    raise ModelUnavailableError(
                        f"tokenizer for {repo} not downloaded under {self.ctx.hf_home} "
                        "(run scripts/setup/download_models.sh)"
                    )
                self._tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
            return self._tokenizer


def _ocr_options(engine: str) -> Any:
    """Explicit OCR engine options: docling's auto mode silently runs without OCR when it finds none (§9.4)."""
    from docling.datamodel import pipeline_options as po

    options = {
        "rapidocr": po.RapidOcrOptions,
        "easyocr": po.EasyOcrOptions,
        "tesseract": po.TesseractCliOcrOptions,
        "ocrmac": po.OcrMacOptions,
    }
    return options[engine]()


def _text_document(path: Path) -> DoclingDocument:
    """A plain-text file as a Docling document: one paragraph per blank-line-separated block."""
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 decodes almost anything
        text = raw.decode("latin-1")
    doc = DoclingDocument(name=path.stem)
    for block in re.split(r"\n\s*\n", text):
        para = " ".join(block.split())
        if para:
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=para)
    return doc


# ------------------------------------------------------------------ Docling → domain model


def to_parsed_document(
    doc: DoclingDocument, *, source_name: str, fmt: str, parse_seconds: float, warnings: Sequence[str] = ()
) -> ParsedDocument:
    from docling_core.types.doc import DocItem, PictureItem, SectionHeaderItem, TableItem, TextItem, TitleItem

    pages = [
        ParsedPage(number=n, width=round(p.size.width, 2), height=round(p.size.height, 2))
        for n, p in sorted(doc.pages.items())
    ]
    headings: dict[int, str] = {}
    items: list[ParsedItem] = []
    tables: list[ParsedTable] = []
    for node, _ in doc.iterate_items():
        if not isinstance(node, DocItem):
            continue
        page = node.prov[0].page_no if node.prov else None
        if isinstance(node, TitleItem | SectionHeaderItem):
            # Same scoping as Docling's HierarchicalChunker, so items and chunks agree on heading paths.
            level = node.level if isinstance(node, SectionHeaderItem) else 0
            for k in [k for k in headings if k >= level]:
                del headings[k]
            headings[level] = node.text
            path = [headings[k] for k in sorted(headings)]
            items.append(
                ParsedItem(
                    ref=node.self_ref, label=node.label.value, text=node.text, page=page, heading_path=path, level=level
                )
            )
            continue
        path = [headings[k] for k in sorted(headings)]
        if isinstance(node, TableItem):
            table = _to_table(node, doc, index=len(tables), heading_path=path)
            tables.append(table)
            items.append(
                ParsedItem(
                    ref=node.self_ref,
                    label=node.label.value,
                    text=table.markdown,
                    page=page,
                    heading_path=path,
                    table_index=table.index,
                )
            )
        elif isinstance(node, TextItem):
            if node.text.strip():
                items.append(
                    ParsedItem(ref=node.self_ref, label=node.label.value, text=node.text, page=page, heading_path=path)
                )
        elif isinstance(node, PictureItem):
            caption = node.caption_text(doc)
            if caption:
                items.append(
                    ParsedItem(ref=node.self_ref, label=node.label.value, text=caption, page=page, heading_path=path)
                )
    sample = "\n".join(it.text for it in items if it.table_index is None)[:20_000]
    return ParsedDocument(
        source_name=source_name,
        format=fmt,
        page_count=len(pages) or None,
        pages=pages,
        items=items,
        tables=tables,
        markdown=doc.export_to_markdown(),
        language=detect_language(sample),
        parse_seconds=round(parse_seconds, 3),
        warnings=list(warnings),
    )


def _to_table(node: Any, doc: DoclingDocument, *, index: int, heading_path: list[str]) -> ParsedTable:
    data = node.data
    cells = sorted(
        (
            TableCell(
                row=c.start_row_offset_idx,
                col=c.start_col_offset_idx,
                row_span=max(c.row_span, 1),
                col_span=max(c.col_span, 1),
                text=c.text,
                column_header=c.column_header,
                row_header=c.row_header,
            )
            for c in data.table_cells
        ),
        key=lambda c: (c.row, c.col),
    )
    page_nos = sorted({p.page_no for p in node.prov})
    bbox = None
    if node.prov:
        prov = node.prov[0]
        page = doc.pages.get(prov.page_no)
        b = prov.bbox.to_top_left_origin(page.size.height) if page else prov.bbox
        bbox = (round(b.l, 2), round(b.t, 2), round(b.r, 2), round(b.b, 2))
    return ParsedTable(
        index=index,
        ref=node.self_ref,
        page_start=page_nos[0] if page_nos else None,
        page_end=page_nos[-1] if page_nos else None,
        bbox=bbox,
        heading_path=heading_path,
        caption=node.caption_text(doc) or None,
        num_rows=data.num_rows,
        num_cols=data.num_cols,
        markdown=node.export_to_markdown(doc=doc),
        cells=cells,
    )


def to_piece(chunk: Any) -> ChunkPiece:
    """A Docling DocChunk as a ChunkPiece. Its items may be re-created as plain DocItems by the chunker, so tables
    are recognised by label, not by class."""
    items = chunk.meta.doc_items
    labels = tuple(str(getattr(it.label, "value", it.label)) for it in items)
    table_ref = next((it.self_ref for it, lab in zip(items, labels, strict=True) if lab in _TABLE_LABELS), None)
    pages = tuple(sorted({p.page_no for it in items for p in (it.prov or [])}))
    return ChunkPiece(
        text=chunk.text,
        headings=tuple(chunk.meta.headings or ()),
        labels=labels,
        pages=pages,
        table_ref=table_ref,
    )
