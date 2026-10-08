"""Upload → background ingestion → READY/FAILED, dedupe, deletion with vectors and files (fake ML providers)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import time
import zipfile
from functools import partial

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import models as orm
from app.providers.models import ModelUnavailableError
from app.providers.runtime import InProcessJobQueue
from app.providers.storage import ObjectNotFound
from app.services.document_pipeline import check_name, clean_filename, storage_key
from app.services.documents import DocumentService

REPORT = (
    b"Annual Report 2024\n\nRevenue from operations grew 34% year on year.\f"
    b"Key Metrics\n\n| Metric | FY23 | FY24 |\n| EBITDA margin | 16.9% | 18.2% |\n\n"
    b"EBITDA margin improved to 18.2% from 16.9%."
)


def _run(api: TestClient, fn, *args, **kwargs):
    return api.portal.call(partial(fn, *args, **kwargs))  # type: ignore[union-attr]


def _db(api: TestClient):
    return api.app.state.container["metadata_db"]  # type: ignore[attr-defined]


def _project(api: TestClient, name: str = "Annual report FY24") -> str:
    return api.post("/api/projects", json={"name": name}).json()["id"]


def upload(api: TestClient, project_id: str, name: str, content: bytes, ctype: str = "application/octet-stream"):
    return api.post(f"/api/projects/{project_id}/documents", files={"file": (name, content, ctype)})


def drain(api: TestClient) -> None:
    """Wait for the background ingestion jobs (and the idle hook after them)."""
    _run(api, api.app.state.container["job_queue"].join)  # type: ignore[attr-defined]


def wait_for_status(api: TestClient, doc_id: str, status: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        doc = api.get(f"/api/documents/{doc_id}").json()
        if doc["status"] == status or time.monotonic() > deadline:
            return doc
        time.sleep(0.01)


def uploads_dir(tmp_path):
    return tmp_path / "data/uploads"


def docx_bytes(main_part: str = "word/document.xml") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr(main_part, "<document/>")
    return buf.getvalue()


# ------------------------------------------------------------------ lifecycle


def test_upload_is_ingested_in_the_background(app, fakes, tmp_path):
    p = _project(app)
    r = upload(app, p, "annual_report.txt", REPORT, "text/plain")
    assert r.status_code == 202, r.text
    doc = r.json()
    assert doc["id"].startswith("doc_") and doc["project_id"] == p
    assert (doc["filename"], doc["mime"], doc["size_bytes"], doc["version"]) == (
        "annual_report.txt",
        "text/plain",
        len(REPORT),
        1,
    )
    assert doc["sha256"] == hashlib.sha256(REPORT).hexdigest()
    assert doc["status"] == "PENDING" and doc["page_count"] is None and doc["chunk_count"] is None
    assert set(doc) == {
        "id",
        "project_id",
        "filename",
        "mime",
        "size_bytes",
        "sha256",
        "version",
        "status",
        "page_count",
        "chunk_count",
        "error",
        "created_at",
        "updated_at",
    }

    drain(app)
    ready = app.get(f"/api/documents/{doc['id']}").json()
    assert (ready["status"], ready["page_count"], ready["chunk_count"], ready["error"]) == ("READY", 2, 5, None)
    assert [d["status"] for d in app.get(f"/api/projects/{p}/documents").json()["items"]] == ["READY"]
    assert app.get(f"/api/projects/{p}").json()["document_count"] == 1

    # stored under a generated key, never the uploaded name; the parser read a temporary copy named by id
    stored = uploads_dir(tmp_path) / storage_key(p, doc["id"], 1, ".txt")
    assert stored.read_bytes() == REPORT
    assert [f.name for f in uploads_dir(tmp_path).rglob("*") if f.is_file()] == ["v1.txt"]
    (parsed_path,) = fakes.parser.parsed_paths
    assert parsed_path.name == f"{doc['id']}.txt" and not parsed_path.exists()

    # vectors indexed for this project/document; tables saved as datasets; parser released once the queue drained
    assert {(c.chunk.project_id, c.chunk.document_id) for c in fakes.store.points.values()} == {(p, doc["id"])}
    (table,) = _run(app, DocumentService(_db(app)).tables, doc["id"])
    assert (table.table_index, table.page_start, table.num_rows, table.num_cols) == (0, 2, 2, 3)
    assert table.bbox == [10.0, 20.0, 300.0, 120.0] and table.heading_path == ["Tables"]
    assert {"row": 1, "col": 2, "text": "18.2%"}.items() <= table.cells[5].items()
    assert fakes.parser.release_calls == 1


def test_processing_is_visible_while_the_job_runs(app, fakes):
    p = _project(app)
    gate = asyncio.Event()
    fakes.parser.gate = gate
    doc_id = upload(app, p, "a.md", b"# Notes\n\nHello.").json()["id"]
    assert wait_for_status(app, doc_id, "PROCESSING")["status"] == "PROCESSING"
    _run(app, gate.set)
    drain(app)
    assert app.get(f"/api/documents/{doc_id}").json()["status"] == "READY"


def test_failed_ingestion_is_recorded_with_a_readable_error(app, fakes):
    p = _project(app)
    r = upload(app, p, "blank.txt", b"  \n\n \n")  # text, but nothing to index
    drain(app)
    doc = app.get(f"/api/documents/{r.json()['id']}").json()
    assert doc["status"] == "FAILED"
    assert doc["error"].startswith("blank.txt: no text could be extracted")  # the temp name is not shown
    assert doc["page_count"] is None and doc["chunk_count"] is None
    assert (doc["id"], []) in fakes.store.deletes  # nothing of it stays searchable
    jobs = _run(app, _jobs, _db(app), doc["id"])
    assert [(j.status, j.stage, j.attempts) for j in jobs] == [("FAILED", "ingest", 1)]
    assert jobs[0].error == doc["error"] and jobs[0].finished_at is not None


def test_missing_ml_libraries_fail_the_document_not_the_app(app, fakes):
    p = _project(app)
    fakes.parser.fail_with = ModelUnavailableError("document parsing needs docling: install the ML dependency group")
    doc_id = upload(app, p, "a.txt", b"hello").json()["id"]
    drain(app)
    doc = app.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "FAILED" and "install the ML dependency group" in doc["error"]
    assert app.get("/health").status_code == 200


async def _jobs(db, doc_id):
    async with db.session() as s:
        stmt = select(orm.IngestionJob).where(orm.IngestionJob.document_id == doc_id).order_by(orm.IngestionJob.id)
        return list((await s.scalars(stmt)).all())


# ------------------------------------------------------------------ dedupe


def test_identical_content_returns_the_existing_document(app, fakes):
    p, other = _project(app), _project(app, "Other")
    first = upload(app, p, "annual_report.txt", REPORT)
    drain(app)
    again = upload(app, p, "renamed copy.txt", REPORT)
    assert again.status_code == 200 and again.json()["id"] == first.json()["id"]
    assert again.json()["filename"] == "annual_report.txt" and again.json()["status"] == "READY"
    assert len(app.get(f"/api/projects/{p}/documents").json()["items"]) == 1
    assert len(fakes.parser.parsed_paths) == 1  # not ingested twice
    elsewhere = upload(app, other, "annual_report.txt", REPORT)  # dedupe is per project
    assert elsewhere.status_code == 202 and elsewhere.json()["id"] != first.json()["id"]


def test_uploading_a_failed_document_again_retries_it(app, fakes):
    p = _project(app)
    fakes.parser.fail_with = RuntimeError("transient")
    doc_id = upload(app, p, "a.txt", b"hello world").json()["id"]
    drain(app)
    assert app.get(f"/api/documents/{doc_id}").json()["error"] == "RuntimeError: transient"
    fakes.parser.fail_with = None
    retry = upload(app, p, "a.txt", b"hello world")
    assert retry.status_code == 202 and retry.json()["id"] == doc_id
    assert retry.json()["status"] == "PENDING" and retry.json()["error"] is None
    drain(app)
    assert app.get(f"/api/documents/{doc_id}").json()["status"] == "READY"
    assert [j.status for j in _run(app, _jobs, _db(app), doc_id)] == ["FAILED", "SUCCEEDED"]


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("name", "content", "detail"),
    [
        ("virus.exe", b"MZ\x90\x00", "file type .exe is not allowed"),
        ("noext", b"hello", "file type (none) is not allowed"),
        ("report.pdf", b"<html>not a pdf</html>", "content is not a PDF"),
        ("report.docx", b"%PDF-1.7 renamed", "content is not a DOCX file"),
        ("deck.pptx", docx_bytes(), "content is not a PPTX file"),  # a Word package renamed to .pptx
        ("broken.docx", b"PK\x03\x04garbage", "damaged DOCX file"),
        ("notes.txt", b"text\x00with NUL", "binary, not text"),
        ("notes.md", b"%PDF-1.4 a pdf renamed", "binary, not text"),
        ("empty.txt", b"", "file is empty"),
    ],
)
def test_rejected_uploads_are_422_and_leave_nothing(app, tmp_path, name, content, detail):
    p = _project(app)
    r = upload(app, p, name, content)
    assert r.status_code == 422, r.text
    assert detail in r.json()["detail"]
    assert app.get(f"/api/projects/{p}/documents").json() == {"items": []}
    assert not any(f.is_file() for f in uploads_dir(tmp_path).rglob("*"))


def test_accepted_formats_are_sniffed_not_trusted(app):
    p = _project(app)
    for name, content in (
        ("report.PDF", b"%PDF-1.7\n..."),
        ("notes.docx", docx_bytes()),
        ("deck.pptx", docx_bytes("ppt/presentation.xml")),
        ("readme.md", "# शीर्षक\n\nनमस्ते".encode()),
    ):
        r = upload(app, p, name, content, "application/x-whatever")
        assert r.status_code == 202, (name, r.text)
    mimes = {d["filename"]: d["mime"] for d in app.get(f"/api/projects/{p}/documents").json()["items"]}
    assert mimes["report.PDF"] == "application/pdf"
    assert mimes["readme.md"] == "text/markdown"
    assert mimes["deck.pptx"].endswith("presentationml.presentation")
    drain(app)


def test_size_limit(make_app):
    with make_app(SERVER__MAX_UPLOAD_MB="1") as api:
        p = _project(api)
        r = upload(api, p, "big.txt", b"a" * (1024 * 1024 + 1))
        assert r.status_code == 422 and "1 MB upload limit" in r.json()["detail"]
        too_big = api.post(
            f"/api/projects/{p}/documents",
            content=b"x" * 10,
            headers={"Content-Length": str(5 * 1024 * 1024), "Content-Type": "multipart/form-data; boundary=x"},
        )
        assert too_big.status_code == 422  # refused from the header, before reading the body
        assert upload(api, p, "ok.txt", b"a" * (1024 * 1024)).status_code == 202
        drain(api)


def test_upload_needs_a_file_field_and_an_existing_project(app):
    p = _project(app)
    assert app.post(f"/api/projects/{p}/documents", json={"file": "x"}).status_code == 422
    r = app.post(f"/api/projects/{p}/documents", files={"other": ("a.txt", b"x")})
    assert r.status_code == 422 and "field 'file'" in r.json()["detail"]
    assert upload(app, "prj_nope", "a.txt", b"x").status_code == 404
    assert app.get("/api/documents/doc_nope").status_code == 404
    assert app.delete("/api/documents/doc_nope").status_code == 404


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("C:\\fakepath\\Annual Report.pdf", "Annual Report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("evil\u202eftp.txt", "evilftp.txt"),  # bidi override removed
        ("  spaced.md ", "spaced.md"),
        ("x" * 300 + ".pdf", "x" * 251 + ".pdf"),
    ],
)
def test_filenames_are_cleaned_for_display(raw, clean):
    assert clean_filename(raw) == clean


def test_filename_checks():
    assert check_name("A.Pdf", [".pdf"]).extension == ".pdf"
    with pytest.raises(Exception, match="no name"):
        clean_filename("///")


# ------------------------------------------------------------------ deletion


def test_delete_document_removes_vectors_files_rows_and_scope(app, fakes, tmp_path):
    p = _project(app)
    d1 = upload(app, p, "a.txt", b"alpha report").json()["id"]
    d2 = upload(app, p, "b.txt", b"beta report").json()["id"]
    drain(app)
    both = app.post(f"/api/projects/{p}/chats", json={"document_scope": [d1, d2]}).json()
    only = app.post(f"/api/projects/{p}/chats", json={"document_scope": [d1]}).json()

    assert app.delete(f"/api/documents/{d1}").status_code == 204
    assert app.get(f"/api/documents/{d1}").status_code == 404
    assert {c.chunk.document_id for c in fakes.store.points.values()} == {d2}
    assert not (uploads_dir(tmp_path) / p / d1).exists()
    assert (uploads_dir(tmp_path) / storage_key(p, d2, 1, ".txt")).is_file()
    assert app.get(f"/api/chats/{both['id']}").json()["document_scope"] == [d2]
    assert app.get(f"/api/chats/{only['id']}").json()["document_scope"] is None  # empty scope → all documents
    assert _run(app, _jobs, _db(app), d1) == []
    with pytest.raises(Exception, match="not found"):
        _run(app, DocumentService(_db(app)).tables, d1)
    assert app.delete(f"/api/documents/{d1}").status_code == 404


def test_delete_document_refuses_while_the_vector_store_is_down(app, fakes, tmp_path):
    p = _project(app)
    doc_id = upload(app, p, "a.txt", b"alpha").json()["id"]
    drain(app)
    fakes.store.fail_with = ConnectionError("qdrant down")
    r = app.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 503 and "qdrant down" in r.json()["detail"] and "nothing was deleted" in r.json()["detail"]
    assert app.get(f"/api/documents/{doc_id}").json()["status"] == "READY"
    assert (uploads_dir(tmp_path) / storage_key(p, doc_id, 1, ".txt")).is_file()


def test_delete_project_removes_its_vectors_and_files_only(app, fakes, tmp_path):
    doomed, kept = _project(app, "Doomed"), _project(app, "Kept")
    upload(app, doomed, "a.txt", b"alpha")
    kept_doc = upload(app, kept, "b.txt", b"beta").json()["id"]
    drain(app)
    assert app.delete(f"/api/projects/{doomed}").status_code == 204
    assert fakes.store.project_deletes == [doomed]
    assert {c.chunk.project_id for c in fakes.store.points.values()} == {kept}
    assert not (uploads_dir(tmp_path) / doomed).exists()
    assert (uploads_dir(tmp_path) / storage_key(kept, kept_doc, 1, ".txt")).is_file()
    assert app.get(f"/api/projects/{doomed}").status_code == 404
    fakes.store.fail_with = ConnectionError("qdrant down")
    assert app.delete(f"/api/projects/{kept}").status_code == 503
    assert app.get(f"/api/projects/{kept}").status_code == 200


def test_document_deleted_while_ingesting_leaves_no_vectors(app, fakes):
    p = _project(app)
    gate = asyncio.Event()
    fakes.parser.gate = gate
    doc_id = upload(app, p, "a.txt", b"alpha beta").json()["id"]
    wait_for_status(app, doc_id, "PROCESSING")
    assert app.delete(f"/api/documents/{doc_id}").status_code == 204
    _run(app, gate.set)
    drain(app)
    assert fakes.store.points == {}
    assert app.get(f"/api/documents/{doc_id}").status_code == 404


def test_restart_resumes_interrupted_ingestion(make_app, fakes, monkeypatch):
    from .conftest import Fakes
    from .fakes import FakeEmbedder, FakeLLM, FakeReranker, FakeStore, TextParser, keyword_scorer

    monkeypatch.setattr(InProcessJobQueue, "SHUTDOWN_GRACE_S", 0.05)
    fakes.parser.gate = asyncio.Event()  # never opened: the server stops mid-ingestion
    with make_app() as api:
        p = _project(api)
        doc_id = upload(api, p, "a.txt", b"alpha beta").json()["id"]
        assert wait_for_status(api, doc_id, "PROCESSING")["status"] == "PROCESSING"
    fresh = Fakes(TextParser(), FakeEmbedder(), FakeReranker(keyword_scorer), FakeStore(search_points=True), FakeLLM())
    with make_app(fresh) as api:
        drain(api)
        assert api.get(f"/api/documents/{doc_id}").json()["status"] == "READY"
        jobs = _run(api, _jobs, _db(api), doc_id)
        assert [(j.status, j.error) for j in jobs] == [
            ("CANCELLED", "interrupted: the server stopped"),
            ("SUCCEEDED", None),
        ]


async def test_storage_failure_after_put_removes_the_file(db, load_local, tmp_path, monkeypatch):
    """If the rows can't be written, the stored file is removed again."""
    from app.providers.registry import build_container
    from app.services.document_pipeline import DocumentPipeline

    from .conftest import mock_http

    settings = load_local()
    container = build_container(settings, http=mock_http())
    store = container["object_store"]
    await store.start()
    pipeline = DocumentPipeline(
        db,
        settings=settings,
        store=store,  # type: ignore[arg-type]
        queue=container["job_queue"],  # type: ignore[arg-type]
        ingestion=None,  # type: ignore[arg-type]
        vectors=container["vector_store"],  # type: ignore[arg-type]
    )
    project = await pipeline.projects.create("P")

    async def broken(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline.documents, "create", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        await pipeline.upload(project.id, "a.txt", io.BytesIO(b"hello"))
    assert not any(f.is_file() for f in uploads_dir(tmp_path).rglob("*"))
    with pytest.raises(ObjectNotFound):
        await store.get(storage_key(project.id, "doc_x", 1, ".txt"))
