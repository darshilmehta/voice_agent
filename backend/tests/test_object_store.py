"""FilesystemStore: put/get/temp copy/delete under object_store.root, with keys that can't escape it."""

from __future__ import annotations

import io

import pytest

from app.providers.base import ProviderContext
from app.providers.storage import FilesystemStore, ObjectNotFound, check_key

from .conftest import mock_http


@pytest.fixture
async def store(load_local, tmp_path):
    settings = load_local()
    async with mock_http() as http:
        s = FilesystemStore(settings.object_store, ProviderContext(settings=settings, http=http))
        await s.start()
        yield s


async def test_put_get_and_exists(store, tmp_path):
    assert await store.put("prj_1/doc_1/v1.pdf", b"%PDF-1.7 hello") == 14
    assert await store.get("prj_1/doc_1/v1.pdf") == b"%PDF-1.7 hello"
    assert await store.exists("prj_1/doc_1/v1.pdf")
    assert (tmp_path / "data/uploads/prj_1/doc_1/v1.pdf").read_bytes() == b"%PDF-1.7 hello"
    assert not await store.exists("prj_1/doc_1/v2.pdf")
    with pytest.raises(ObjectNotFound):
        await store.get("prj_1/doc_1/v2.pdf")


async def test_put_streams_file_objects_and_replaces_atomically(store, tmp_path):
    big = io.BytesIO(b"x" * (3 * 1024 * 1024 + 5))
    assert await store.put("a/b.txt", big) == 3 * 1024 * 1024 + 5
    await store.put("a/b.txt", b"second")
    assert await store.get("a/b.txt") == b"second"
    assert [p.name for p in (tmp_path / "data/uploads/a").iterdir()] == ["b.txt"]  # no .part leftovers


async def test_temp_copy_is_private_and_removed(store):
    await store.put("prj_1/doc_1/v1.md", b"# Notes")
    async with store.open_temp_copy("prj_1/doc_1/v1.md", filename="doc_1.md") as path:
        assert path.name == "doc_1.md" and path.read_bytes() == b"# Notes"
        assert "uploads" not in str(path)
        path.write_bytes(b"changed")  # consumers can't alter the stored original
        copy_dir = path.parent
    assert not copy_dir.exists()
    assert await store.get("prj_1/doc_1/v1.md") == b"# Notes"
    with pytest.raises(ObjectNotFound):
        async with store.open_temp_copy("prj_1/doc_1/missing.md"):
            pass
    with pytest.raises(ValueError):
        async with store.open_temp_copy("prj_1/doc_1/v1.md", filename="../evil.md"):
            pass


async def test_delete_and_delete_prefix_prune_empty_folders(store, tmp_path):
    for key in ("prj_1/doc_1/v1.pdf", "prj_1/doc_1/v2.pdf", "prj_1/doc_2/v1.pdf", "prj_2/doc_3/v1.pdf"):
        await store.put(key, b"%PDF-")
    assert await store.delete("prj_1/doc_1/v1.pdf") is True
    assert await store.delete("prj_1/doc_1/v1.pdf") is False  # already gone: not an error
    assert await store.delete_prefix("prj_1/doc_1") == 1
    assert not (tmp_path / "data/uploads/prj_1/doc_1").exists()
    assert await store.delete_prefix("prj_1/") == 1
    assert not (tmp_path / "data/uploads/prj_1").exists()
    assert await store.delete_prefix("prj_9") == 0
    assert await store.get("prj_2/doc_3/v1.pdf") == b"%PDF-"
    assert (tmp_path / "data/uploads").is_dir()  # the root itself stays


@pytest.mark.parametrize(
    "key",
    ["", "/etc/passwd", "../outside", "a/../../b", "a//b", ".hidden", "a/.git/config", "a/b c", "a\\b", "x" * 200],
)
def test_unsafe_keys_are_rejected(key):
    with pytest.raises(ValueError, match="invalid object key"):
        check_key(key)


async def test_store_refuses_unsafe_keys_and_symlink_escapes(store, tmp_path):
    with pytest.raises(ValueError):
        await store.put("../escape.txt", b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "data/uploads/link").symlink_to(outside)
    with pytest.raises(ValueError, match="outside the store"):
        await store.put("link/file.txt", b"x")
    assert not (outside / "file.txt").exists()
    with pytest.raises(ValueError):
        await store.delete_prefix("")
