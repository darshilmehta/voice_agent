"""HTTP API for projects, chats, transcripts and pins (docs/DESIGN.md §3.9), on a fresh SQLite DB under tmp_path."""

from __future__ import annotations

from functools import partial

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.registry import build_container
from app.services.messages import MessageService

from .conftest import add_document, mock_http


def _project(api: TestClient, name: str = "Annual report FY24", **body) -> dict:
    r = api.post("/api/projects", json={"name": name, **body})
    assert r.status_code == 201, r.text
    return r.json()


def _chat(api: TestClient, project_id: str, **body) -> dict:
    r = api.post(f"/api/projects/{project_id}/chats", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _run(api: TestClient, fn, *args, **kwargs):
    """Call an async function on the app's event loop (where its database engine lives)."""
    return api.portal.call(partial(fn, *args, **kwargs))  # type: ignore[union-attr]


def _db(api: TestClient):
    return api.app.state.container["metadata_db"]  # type: ignore[attr-defined]


def _append(api: TestClient, chat_id: str, text: str, role: str = "user"):
    return _run(api, MessageService(_db(api)).append, chat_id, role=role, text=text)


# ------------------------------------------------------------------ projects


def test_create_get_and_list_projects(api):
    created = _project(api, description="Board pack")
    assert created["id"].startswith("prj_")
    assert created["name"] == "Annual report FY24" and created["description"] == "Board pack"
    assert created["pinned"] is False and created["archived"] is False
    assert (created["chat_count"], created["document_count"]) == (0, 0)
    assert api.get(f"/api/projects/{created['id']}").json() == created
    assert api.get("/api/projects").json() == {"items": [created]}


def test_projects_list_pinned_first(api):
    a, b, c = (_project(api, n) for n in ("A", "B", "C"))
    r = api.patch(f"/api/projects/{a['id']}", json={"pinned": True})
    assert r.status_code == 200 and r.json()["pinned"] is True and r.json()["pinned_at"]
    names = [p["name"] for p in api.get("/api/projects").json()["items"]]
    assert names[0] == "A" and set(names) == {"A", "B", "C"}
    api.patch(f"/api/projects/{c['id']}", json={"archived": True})
    assert "C" not in [p["name"] for p in api.get("/api/projects").json()["items"]]
    archived = api.get("/api/projects", params={"include_archived": "true"}).json()["items"]
    assert {p["id"] for p in archived} == {a["id"], b["id"], c["id"]}


def test_patch_project_changes_only_what_is_sent(api):
    p = _project(api, description="keep me")
    r = api.patch(f"/api/projects/{p['id']}", json={"name": "  Vendor contracts  "})
    assert r.status_code == 200
    assert r.json()["name"] == "Vendor contracts" and r.json()["description"] == "keep me"
    r = api.patch(f"/api/projects/{p['id']}", json={"description": None})
    assert r.json()["description"] is None and r.json()["name"] == "Vendor contracts"


def test_project_validation_errors_are_422(api):
    assert api.post("/api/projects", json={}).status_code == 422
    assert api.post("/api/projects", json={"name": "   "}).status_code == 422
    assert api.post("/api/projects", json={"name": "x" * 201}).status_code == 422
    assert api.post("/api/projects", json={"name": "ok", "colour": "red"}).status_code == 422  # unknown field
    p = _project(api)
    r = api.patch(f"/api/projects/{p['id']}", json={"name": None})
    assert r.status_code == 422 and "cannot be null: name" in r.text
    assert api.patch(f"/api/projects/{p['id']}", json={"pinned": "maybe"}).status_code == 422


def test_unknown_ids_are_404(api):
    for method, path in (
        ("GET", "/api/projects/prj_nope"),
        ("PATCH", "/api/projects/prj_nope"),
        ("DELETE", "/api/projects/prj_nope"),
        ("GET", "/api/projects/prj_nope/documents"),
        ("GET", "/api/projects/prj_nope/chats"),
        ("POST", "/api/projects/prj_nope/chats"),
        ("GET", "/api/chats/cht_nope"),
        ("PATCH", "/api/chats/cht_nope"),
        ("DELETE", "/api/chats/cht_nope"),
        ("GET", "/api/chats/cht_nope/messages"),
    ):
        r = api.request(method, path, json={} if method in ("PATCH", "POST") else None)
        assert r.status_code == 404, (method, path, r.text)
        assert "not found" in r.json()["detail"]


def test_delete_project_cascades(api):
    p = _project(api)
    other = _project(api, "Other")
    chat = _chat(api, p["id"])
    _append(api, chat["id"], "hello")
    _run(api, add_document, _db(api), p["id"])
    assert api.delete(f"/api/projects/{p['id']}").status_code == 204
    assert api.get(f"/api/projects/{p['id']}").status_code == 404
    assert api.get(f"/api/chats/{chat['id']}").status_code == 404
    assert api.get(f"/api/chats/{chat['id']}/messages").status_code == 404
    assert [x["id"] for x in api.get("/api/projects").json()["items"]] == [other["id"]]


def test_list_project_documents(api):
    p = _project(api)
    empty = _project(api, "Empty")
    doc_id = _run(api, add_document, _db(api), p["id"], "annual_report.pdf", status="READY", page_count=212)
    body = api.get(f"/api/projects/{p['id']}/documents").json()
    assert [d["id"] for d in body["items"]] == [doc_id]
    doc = body["items"][0]
    assert (doc["filename"], doc["status"], doc["page_count"], doc["version"]) == ("annual_report.pdf", "READY", 212, 1)
    assert api.get(f"/api/projects/{empty['id']}/documents").json() == {"items": []}
    assert api.get(f"/api/projects/{p['id']}").json()["document_count"] == 1


# ------------------------------------------------------------------ chats


def test_create_list_and_update_chats(api):
    p = _project(api)
    auto = _chat(api, p["id"])
    assert auto["id"].startswith("cht_") and auto["project_id"] == p["id"]
    assert (auto["title"], auto["title_is_auto"], auto["document_scope"]) == ("New chat", True, None)
    named = _chat(api, p["id"], title="FY24 margins", language="hi")
    assert (named["title_is_auto"], named["language"]) == (False, "hi")

    listed = api.get(f"/api/projects/{p['id']}/chats").json()["items"]
    assert [c["id"] for c in listed] == [named["id"], auto["id"]]

    _append(api, auto["id"], "first question")  # activity moves it to the top
    listed = api.get(f"/api/projects/{p['id']}/chats").json()["items"]
    assert [c["id"] for c in listed] == [auto["id"], named["id"]]
    assert listed[0]["message_count"] == 1 and listed[0]["last_message_at"]

    r = api.patch(f"/api/chats/{auto['id']}", json={"title": "Risk section", "pinned": True})
    assert r.status_code == 200
    assert (r.json()["title"], r.json()["title_is_auto"], r.json()["pinned"]) == ("Risk section", False, True)
    assert api.get(f"/api/chats/{auto['id']}").json() == r.json()
    assert api.get(f"/api/projects/{p['id']}").json()["chat_count"] == 2


def test_chat_validation_errors_are_422(api):
    p = _project(api)
    chat = _chat(api, p["id"])
    assert api.post(f"/api/projects/{p['id']}/chats", json={"language": "fr"}).status_code == 422
    assert api.post(f"/api/projects/{p['id']}/chats", json={"title": ""}).status_code == 422
    assert api.patch(f"/api/chats/{chat['id']}", json={"title": None}).status_code == 422
    assert api.patch(f"/api/chats/{chat['id']}", json={"pinned": None}).status_code == 422
    assert api.patch(f"/api/chats/{chat['id']}", json={"titel": "typo"}).status_code == 422


def test_chat_document_scope(api):
    p = _project(api)
    other = _project(api, "Other")
    db = _db(api)
    mine = _run(api, add_document, db, p["id"], "a.pdf")
    foreign = _run(api, add_document, db, other["id"], "b.pdf")
    chat = _chat(api, p["id"], document_scope=[mine])
    assert chat["document_scope"] == [mine]

    r = api.patch(f"/api/chats/{chat['id']}", json={"document_scope": [mine, foreign]})
    assert r.status_code == 422 and foreign in r.json()["detail"]
    assert api.patch(f"/api/chats/{chat['id']}", json={"document_scope": []}).status_code == 422
    assert api.post(f"/api/projects/{p['id']}/chats", json={"document_scope": ["doc_nope"]}).status_code == 422

    r = api.patch(f"/api/chats/{chat['id']}", json={"document_scope": None})
    assert r.status_code == 200 and r.json()["document_scope"] is None  # back to all documents


def test_delete_chat(api):
    p = _project(api)
    chat = _chat(api, p["id"])
    kept = _chat(api, p["id"])
    _append(api, chat["id"], "bye")
    assert api.delete(f"/api/chats/{chat['id']}").status_code == 204
    assert api.get(f"/api/chats/{chat['id']}").status_code == 404
    assert [c["id"] for c in api.get(f"/api/projects/{p['id']}/chats").json()["items"]] == [kept["id"]]


# ------------------------------------------------------------------ messages


def test_messages_paginate_in_order(api):
    p = _project(api)
    chat = _chat(api, p["id"])
    for i in range(1, 6):
        _append(api, chat["id"], f"m{i}", role="user" if i % 2 else "agent")
    url = f"/api/chats/{chat['id']}/messages"

    page = api.get(url, params={"limit": 2}).json()
    assert [m["text"] for m in page["items"]] == ["m1", "m2"]
    assert (page["total"], page["has_more"], page["next_cursor"]) == (5, True, 2)
    first = page["items"][0]
    assert first["id"].startswith("msg_") and first["seq"] == 1 and first["role"] == "user"
    assert first["modality"] == "text" and first["citations"] == [] and first["interrupted"] is False

    page = api.get(url, params={"limit": 2, "after": page["next_cursor"]}).json()
    assert [m["text"] for m in page["items"]] == ["m3", "m4"]

    latest = api.get(url, params={"limit": 2, "before": 6}).json()
    assert [m["text"] for m in latest["items"]] == ["m4", "m5"] and latest["next_cursor"] == 4
    older = api.get(url, params={"limit": 10, "before": latest["next_cursor"]}).json()
    assert [m["text"] for m in older["items"]] == ["m1", "m2", "m3"] and older["has_more"] is False

    assert api.get(url).json()["total"] == 5


def test_message_paging_validation(api):
    p = _project(api)
    url = f"/api/chats/{_chat(api, p['id'])['id']}/messages"
    assert api.get(url, params={"after": 1, "before": 3}).status_code == 422
    assert api.get(url, params={"limit": 0}).status_code == 422
    assert api.get(url, params={"limit": 201}).status_code == 422
    assert api.get(url, params={"after": -1}).status_code == 422
    assert api.get(url).json() == {"items": [], "total": 0, "has_more": False, "next_cursor": None}


# ------------------------------------------------------------------ pins, persistence, CORS


def test_pins_for_the_sidebar(api):
    a = _project(api, "Annual report FY24")
    b = _project(api, "Vendor contracts")
    margins = _chat(api, a["id"], title="FY24 margins")
    _chat(api, a["id"], title="unpinned")
    api.patch(f"/api/projects/{b['id']}", json={"pinned": True})
    api.patch(f"/api/chats/{margins['id']}", json={"pinned": True})
    pins = api.get("/api/pins").json()
    assert [p["name"] for p in pins["projects"]] == ["Vendor contracts"]
    assert [(c["title"], c["project_name"], c["pinned"]) for c in pins["chats"]] == [
        ("FY24 margins", "Annual report FY24", True)
    ]
    api.patch(f"/api/chats/{margins['id']}", json={"pinned": False})
    assert api.get("/api/pins").json()["chats"] == []


def test_projects_chats_and_pins_survive_a_restart(load_local):
    settings = load_local()

    def app() -> TestClient:
        return TestClient(create_app(settings, build_container(settings, http=mock_http()), preload_models=False))

    with app() as api:
        p = _project(api)
        chat = _chat(api, p["id"], title="FY24 margins")
        api.patch(f"/api/chats/{chat['id']}", json={"pinned": True})
        _append(api, chat["id"], "What was FY24 revenue?")
    with app() as api:
        assert [x["id"] for x in api.get("/api/projects").json()["items"]] == [p["id"]]
        assert [c["title"] for c in api.get("/api/pins").json()["chats"]] == ["FY24 margins"]
        messages = api.get(f"/api/chats/{chat['id']}/messages").json()["items"]
        assert [m["text"] for m in messages] == ["What was FY24 revenue?"]


def test_cors_preflight_allows_patch(api):
    r = api.options(
        "/api/projects/prj_x",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "PATCH"},
    )
    assert r.status_code == 200
    assert "PATCH" in r.headers["access-control-allow-methods"]


def test_openapi_lists_the_endpoints(api):
    paths = api.get("/openapi.json").json()["paths"]
    assert set(paths["/api/projects"]) == {"get", "post"}
    assert set(paths["/api/projects/{project_id}"]) == {"get", "patch", "delete"}
    assert set(paths["/api/projects/{project_id}/chats"]) == {"get", "post"}
    assert set(paths["/api/chats/{chat_id}"]) == {"get", "patch", "delete"}
    assert set(paths["/api/chats/{chat_id}/messages"]) == {"get", "post"}
    assert set(paths["/api/projects/{project_id}/documents"]) == {"get", "post"}
    assert set(paths["/api/documents/{document_id}"]) == {"get", "delete"}
    assert set(paths["/api/pins"]) == {"get"}
