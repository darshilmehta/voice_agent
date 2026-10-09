"""The eval corpus (scripts/eval/build_corpus.py) parsed with the real Docling, once, and typed into datasets: shared by
the canvas integration tests.

Parsing a 29-page PDF on the CPU takes about a minute, so parsed documents are cached as JSON in ``CANVAS_PARSE_CACHE``
(a folder; default: a session temp folder, i.e. parsed once per test session). Point it at a folder you keep to parse
each document only once across runs:

    CANVAS_PARSE_CACHE=/tmp/canvas_cache RUN_INTEGRATION=1 uv run pytest tests/integration/test_canvas_corpus.py -s
"""

from __future__ import annotations

import asyncio
import itertools
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.domain.datasets import TypedDataset
from app.domain.projects import DocumentTable
from app.offline import apply_runtime_env
from app.providers.base import ProviderContext
from app.providers.ingestion import DoclingParser, ParsedDocument
from app.services.canvas.datasets import table_contexts, type_document
from app.settings import PROJECT_ROOT, load_settings

from .conftest import LOCAL_CONFIG

CORPUS = (
    "valmora_annual_report_fy24.pdf",
    "zephyra_investor_deck_q4fy24.pptx",
    "valmora_travel_expense_policy.docx",
    "suryodaya_yojana_soochna.docx",
)
MANIFEST = PROJECT_ROOT / "evals/retrieval/manifest.json"


@dataclass
class CorpusDocument:
    filename: str
    document_id: str
    parsed: ParsedDocument
    datasets: list[TypedDataset]
    parse_seconds: float | None  # None: read from the cache


def eval_docs() -> Path:
    docs = Path(os.environ.get("EVAL_DOCS") or PROJECT_ROOT / "data/eval/docs")
    if not (docs / CORPUS[0]).is_file():
        pytest.skip(f"eval corpus not found in {docs} (run: uv run scripts/eval/build_corpus.py, or set EVAL_DOCS)")
    return docs


async def _parse(path: Path, models_root: Path, cache: Path) -> tuple[ParsedDocument, float | None]:
    cached = cache / f"{path.name}.parsed.json"
    if cached.is_file():
        return ParsedDocument.model_validate_json(cached.read_text(encoding="utf-8")), None
    env = {
        **os.environ,
        "APP_ROOT_DIR": str(cache),
        "MODEL_CACHE__HF_HOME": str(models_root / "huggingface"),
        "INGESTION__ARTIFACTS_PATH": str(models_root / "docling"),
    }
    settings = load_settings(LOCAL_CONFIG, env)
    apply_runtime_env(settings)
    async with httpx.AsyncClient() as http:
        parser = DoclingParser(settings.ingestion, ProviderContext(settings=settings, http=http))
        try:
            parsed = await parser.parse(path)
        finally:
            parser.release()
    parsed.drop_native()
    cached.write_text(parsed.model_dump_json(), encoding="utf-8")
    return parsed, parsed.parse_seconds


def stored_tables(parsed: ParsedDocument, document_id: str) -> list[DocumentTable]:
    """The tables as ``documents.finish_job`` stores them."""
    now = datetime.now(UTC)
    return [
        DocumentTable(
            id=f"tbl_{document_id}_{t.index}",
            document_id=document_id,
            version=1,
            table_index=t.index,
            page_start=t.page_start,
            page_end=t.page_end,
            bbox=list(t.bbox) if t.bbox else None,
            heading_path=t.heading_path,
            caption=t.caption,
            num_rows=t.num_rows,
            num_cols=t.num_cols,
            markdown=t.markdown,
            cells=[c.model_dump(mode="json") for c in t.cells],
            created_at=now,
        )
        for t in parsed.tables
    ]


def load_corpus(models_root: Path, cache: Path) -> dict[str, CorpusDocument]:
    docs = eval_docs()
    cache.mkdir(parents=True, exist_ok=True)
    ids = itertools.count(1)
    out: dict[str, CorpusDocument] = {}
    for n, name in enumerate(CORPUS, start=1):
        parsed, seconds = asyncio.run(_parse(docs / name, models_root, cache))
        document_id = f"doc_eval{n}"
        datasets = type_document(
            stored_tables(parsed, document_id),
            table_contexts(parsed),
            new_id=lambda: f"ds_eval{next(ids):03d}",
        )
        out[name] = CorpusDocument(name, document_id, parsed, datasets, seconds)
    return out


@pytest.fixture(scope="session", name="corpus")
def corpus_fixture(models_root: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, CorpusDocument]:
    cache = Path(os.environ.get("CANVAS_PARSE_CACHE") or tmp_path_factory.mktemp("canvas_parse_cache"))
    return load_corpus(models_root, cache)
