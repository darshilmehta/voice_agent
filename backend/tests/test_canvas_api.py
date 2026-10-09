"""The canvas end to end with fake ML providers: datasets typed after ingestion, the overview, a chat's canvas and its
edits, cascades, the backfill job, and the shapes of contract v1 (services/canvas/service.py, api/canvas.py)."""

from __future__ import annotations

import asyncio
from functools import partial

from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.db import models as orm
from app.domain.canvas import CanvasEvent, Visual, VisualEvent
from app.services.canvas.builder import check_grounding

REPORT = """Valmora annual report FY24

Revenue grew in every quarter.\f| Metric | FY24 | FY23 |
| Revenue from operations (₹ crore) | 7,365 | 6,482 |
| EBITDA (₹ crore) | 1,545 | 1,283 |
| EBITDA margin | 21.0% | 19.8% |
| Number of employees | 9,842 | 9,310 |\f| Quarter | Revenue (₹ crore) | EBITDA (₹ crore) |
| Q1 FY24 | 1,742 | 352 |
| Q2 FY24 | 1,801 | 372 |
| Q3 FY24 | 1,889 | 399 |
| Q4 FY24 | 1,933 | 422 |\f| Segment | Revenue FY24 (₹ crore) | Revenue FY23 (₹ crore) |
| Specialty Chemicals | 3,568 | 3,214 |
| Engineered Plastics | 2,303 | 2,126 |
| Digital Services | 1,494 | 1,142 |
| Total | 7,365 | 6,482 |
""".encode()
POLICY = b"Travel policy\n\nHotel limits apply per night.\f| Grade | Tier-1 city (\xe2\x82\xb9 per night) |\n| L1 to L2 | 5,500 |\n| L3 to L4 | 7,500 |"


def run(api: TestClient, fn, *args, **kwargs):
    return api.portal.call(partial(fn, *args, **kwargs))  # type: ignore[union-attr]


def drain(api: TestClient) -> None:
    run(api, api.app.state.container["job_queue"].join)  # type: ignore[attr-defined]


def db(api: TestClient):
    return api.app.state.container["metadata_db"]  # type: ignore[attr-defined]


def canvas(api: TestClient):
    return api.app.state.canvas  # type: ignore[attr-defined]


async def count(database, model, *where) -> int:
    async with database.session() as s:
        return await s.scalar(select(func.count()).select_from(model).where(*where)) or 0


def project(api: TestClient, name: str = "Annual report FY24") -> str:
    return api.post("/api/projects", json={"name": name}).json()["id"]


def upload(api: TestClient, project_id: str, name: str = "report.txt", content: bytes = REPORT) -> str:
    r = api.post(f"/api/projects/{project_id}/documents", files={"file": (name, content, "text/plain")})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def chat(api: TestClient, project_id: str, **body) -> str:
    return api.post(f"/api/projects/{project_id}/chats", json=body).json()["id"]


def datasets(api: TestClient, project_id: str) -> dict[str, dict]:
    items = api.get(f"/api/projects/{project_id}/datasets").json()["items"]
    return {d["title"] if d["title"] != "Tables" else f"p{d['page']}": d for d in items}


def ready_report(api: TestClient) -> tuple[str, str, dict[str, dict]]:
    p = project(api)
    doc = upload(api, p)
    drain(api)
    return p, doc, {f"p{d['page']}": d for d in api.get(f"/api/projects/{p}/datasets").json()["items"]}


# ------------------------------------------------------------------ datasets and the overview


def test_tables_are_typed_after_ingestion(app):
    p, doc, ds = ready_report(app)
    assert list(ds) == ["p2", "p3", "p4"]
    assert {k: d["chartability"]["kind"] for k, d in ds.items()} == {"p2": "kpi", "p3": "time_series", "p4": "composition"}
    assert ds["p2"]["document_id"] == doc and ds["p2"]["filename"] == "report.txt"
    rows = {r["key"]: r for r in ds["p2"]["rows"]}
    assert rows["revenue_from_operations"]["unit"]["label"] == "₹ crore"
    assert [c["key"] for c in ds["p3"]["columns"]] == ["quarter", "revenue", "ebitda"]
    assert ds["p3"]["chartability"]["period_order"] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    assert app.get("/api/projects/prj_nope/datasets").status_code == 404


def test_overview_is_built_from_the_most_chartable_tables(app):
    p, _, ds = ready_report(app)
    body = app.get(f"/api/projects/{p}/overview").json()
    assert body["status"] == "ready"
    kinds = [v["kind"] for v in body["panels"]]
    assert kinds == ["kpi", "line", "donut"]
    kpi, line, donut = (Visual.model_validate(v) for v in body["panels"])
    for v in (kpi, line, donut):
        assert v.chat_id is None and v.project_id == p and check_grounding(v) == []
    assert [t.label for t in kpi.tiles] == [
        "Revenue from operations (FY24)",
        "EBITDA (FY24)",
        "EBITDA margin (FY24)",
        "Number of employees (FY24)",
    ]
    assert kpi.tiles[0].delta is not None and kpi.tiles[0].delta.value == 13.6
    assert [r.x for r in line.rows] == ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    assert [s.key for s in line.series] == ["revenue", "ebitda"]
    assert [r.x for r in donut.rows] == ["Specialty Chemicals", "Engineered Plastics", "Digital Services"]
    assert [p_["position"] for p_ in body["panels"]] == [0, 1, 2]


def test_overview_status_follows_ingestion(app, fakes):
    p = project(app)
    assert app.get(f"/api/projects/{p}/overview").json() == {"panels": [], "status": "none"}
    fakes.parser.gate = asyncio.Event()
    upload(app, p)
    assert app.get(f"/api/projects/{p}/overview").json()["status"] == "building"
    app.portal.call(fakes.parser.gate.set)  # type: ignore[union-attr]
    drain(app)
    assert app.get(f"/api/projects/{p}/overview").json()["status"] == "ready"
    assert app.get("/api/projects/prj_nope/overview").status_code == 404


def test_a_project_without_tables_has_no_overview(app):
    p = project(app)
    upload(app, p, "notes.txt", b"Just prose.\n\nNothing to chart.")
    drain(app)
    assert app.get(f"/api/projects/{p}/overview").json() == {"panels": [], "status": "none"}


def test_overview_size_is_configurable(make_app):
    with make_app(CANVAS__OVERVIEW_PANELS="1") as api:
        p, _, _ = ready_report(api)
        assert [v["kind"] for v in api.get(f"/api/projects/{p}/overview").json()["panels"]] == ["kpi"]


# ------------------------------------------------------------------ a chat's canvas


def add(api: TestClient, chat_id: str, **spec):
    return api.post(f"/api/chats/{chat_id}/visuals", json=spec)


def test_add_visuals_and_edit_the_canvas(app):
    p, _, ds = ready_report(app)
    c = chat(app, p)
    q = ds["p3"]["id"]
    r = add(app, c, kind="line", datasets=[q], series=[f"{q}:revenue"], highlight=["Q4 FY24"], title="Revenue by quarter")
    assert r.status_code == 201, r.text
    first = r.json()
    assert first["id"].startswith("vis_") and first["chat_id"] == c and first["position"] == 0 and not first["pinned"]
    assert first["title"] == "Revenue by quarter" and first["highlight"] == {"x": ["Q4 FY24"], "series": [], "note": "Highest"}
    h = ds["p2"]["id"]
    second = add(app, c, kind="kpi", datasets=[h], series=[f"{h}:revenue_from_operations", f"{h}:ebitda_margin"]).json()
    third = add(app, c, kind="donut", datasets=[ds["p4"]["id"]], series=[f"{ds['p4']['id']}:revenue_fy24"]).json()
    panels = app.get(f"/api/chats/{c}/canvas").json()["panels"]
    assert [v["id"] for v in panels] == [first["id"], second["id"], third["id"]]

    def op(**body):
        r = app.post(f"/api/chats/{c}/canvas/ops", json=body)
        assert r.status_code == 200, r.text
        return [(v["id"], v["position"], v["pinned"]) for v in r.json()["panels"]]

    assert op(op="pin", visual_id=second["id"]) == [(first["id"], 0, False), (second["id"], 1, True), (third["id"], 2, False)]
    assert op(op="move", visual_id=third["id"], position=0) == [
        (third["id"], 0, False),
        (first["id"], 1, False),
        (second["id"], 2, True),
    ]
    assert op(op="move", visual_id=third["id"], position=99)[-1] == (third["id"], 2, False)  # clamped
    assert op(op="unpin", visual_id=second["id"])[1] == (second["id"], 1, False)
    assert op(op="remove", visual_id=first["id"]) == [(second["id"], 0, False), (third["id"], 1, False)]
    assert app.get(f"/api/chats/{c}/canvas").json()["panels"][0]["id"] == second["id"]

    assert app.post(f"/api/chats/{c}/canvas/ops", json={"op": "pin", "visual_id": "vis_nope"}).status_code == 404
    assert app.post(f"/api/chats/{c}/canvas/ops", json={"op": "move", "visual_id": third["id"]}).status_code == 422
    assert app.post(f"/api/chats/{c}/canvas/ops", json={"op": "explode", "visual_id": third["id"]}).status_code == 422
    assert app.get("/api/chats/cht_nope/canvas").status_code == 404


def test_invalid_specs_are_rejected_with_every_problem(app):
    p, _, ds = ready_report(app)
    c = chat(app, p)
    h = ds["p2"]["id"]
    r = add(app, c, kind="line", datasets=[h], series=[f"{h}:market_share", f"{h}:ebitda"], periods=["FY21"])
    assert r.status_code == 422
    assert "'market_share' is not a row or a measured column" in r.json()["detail"]
    r = add(app, c, kind="bar", datasets=["ds_invented"], series=["ds_invented:revenue"])
    assert r.status_code == 422 and "unknown dataset" in r.json()["detail"]
    r = add(app, c, kind="bar", datasets=[h], series=[f"{h}:ebitda"], colour="red")
    assert r.status_code == 422  # unknown fields are refused
    r = add(app, c, kind="pie", datasets=[h], series=[f"{h}:ebitda"])
    assert r.status_code == 422
    assert app.get(f"/api/chats/{c}/canvas").json()["panels"] == []
    assert add(app, "cht_nope", kind="bar", datasets=[h], series=[f"{h}:ebitda"]).status_code == 404


def test_a_chat_only_sees_the_documents_in_its_scope(app):
    p, doc, ds = ready_report(app)
    other = upload(app, p, "policy.txt", POLICY)
    drain(app)
    scoped = chat(app, p, document_scope=[other])
    h = ds["p2"]["id"]
    r = add(app, scoped, kind="bar", datasets=[h], series=[f"{h}:ebitda"])
    assert r.status_code == 422 and "unknown dataset" in r.json()["detail"]
    assert add(app, chat(app, p), kind="bar", datasets=[h], series=[f"{h}:ebitda"]).status_code == 201


def test_oldest_unpinned_panels_make_room(make_app):
    with make_app(CANVAS__MAX_PANELS="2") as api:
        p, _, ds = ready_report(api)
        c = chat(api, p)
        h = ds["p2"]["id"]
        ids = [add(api, c, kind="bar", datasets=[h], series=[f"{h}:{k}"]).json()["id"] for k in ("ebitda", "revenue_from_operations")]
        api.post(f"/api/chats/{c}/canvas/ops", json={"op": "pin", "visual_id": ids[0]})
        third = add(api, c, kind="bar", datasets=[h], series=[f"{h}:ebitda_margin"]).json()["id"]
        panels = api.get(f"/api/chats/{c}/canvas").json()["panels"]
        assert [(v["id"], v["position"]) for v in panels] == [(ids[0], 0), (third, 1)]


# ------------------------------------------------------------------ cascades


def test_deleting_a_chat_deletes_its_canvas(app):
    p, _, ds = ready_report(app)
    c = chat(app, p)
    h = ds["p2"]["id"]
    add(app, c, kind="bar", datasets=[h], series=[f"{h}:ebitda"])
    assert run(app, count, db(app), orm.CanvasVisual, orm.CanvasVisual.chat_id == c) == 1
    assert app.delete(f"/api/chats/{c}").status_code == 204
    assert run(app, count, db(app), orm.CanvasVisual, orm.CanvasVisual.chat_id == c) == 0
    assert len(app.get(f"/api/projects/{p}/overview").json()["panels"]) == 3  # the overview stays


def test_deleting_a_document_deletes_its_datasets_and_visuals(app):
    p, doc, ds = ready_report(app)
    other = upload(app, p, "policy.txt", POLICY)
    drain(app)
    c = chat(app, p)
    h = ds["p2"]["id"]
    from_report = add(app, c, kind="bar", datasets=[h], series=[f"{h}:ebitda"]).json()["id"]
    policy_ds = next(d for d in app.get(f"/api/projects/{p}/datasets").json()["items"] if d["document_id"] == other)
    from_policy = add(app, c, kind="bar", datasets=[policy_ds["id"]], series=[f"{policy_ds['id']}:tier_1_city"]).json()["id"]
    assert run(app, count, db(app), orm.TableDataset, orm.TableDataset.document_id == doc) == 3

    assert app.delete(f"/api/documents/{doc}").status_code == 204
    assert run(app, count, db(app), orm.TableDataset, orm.TableDataset.document_id == doc) == 0
    assert [v["id"] for v in app.get(f"/api/chats/{c}/canvas").json()["panels"]] == [from_policy]
    assert from_report not in [v["id"] for v in app.get(f"/api/projects/{p}/overview").json()["panels"]]
    # rebuilt from what is left: the policy's hotel limits are categorical, nothing for the overview
    assert app.get(f"/api/projects/{p}/overview").json() == {"panels": [], "status": "none"}


def test_deleting_a_project_deletes_everything_canvas(app):
    p, _, ds = ready_report(app)
    c = chat(app, p)
    h = ds["p2"]["id"]
    add(app, c, kind="bar", datasets=[h], series=[f"{h}:ebitda"])
    assert app.delete(f"/api/projects/{p}").status_code == 204
    for model in (orm.CanvasVisual, orm.TableDataset, orm.ProjectOverview):
        assert run(app, count, db(app), model) == 0


# ------------------------------------------------------------------ backfill


async def forget_canvas(database) -> None:
    """The state of a database from before the canvas: tables, but no datasets or overviews."""
    async with database.session() as s:
        for model in (orm.TableDataset, orm.CanvasVisual, orm.ProjectOverview):
            await s.execute(delete(model))


def test_backfill_types_documents_ingested_before_the_canvas(app):
    p, doc, _ = ready_report(app)
    run(app, forget_canvas, db(app))
    assert app.get(f"/api/projects/{p}/datasets").json()["items"] == []
    assert run(app, canvas(app).backfill) == 1
    items = app.get(f"/api/projects/{p}/datasets").json()["items"]
    assert [d["chartability"]["kind"] for d in items] == ["kpi", "time_series", "composition"]
    assert [v["kind"] for v in app.get(f"/api/projects/{p}/overview").json()["panels"]] == ["kpi", "line", "donut"]
    assert run(app, canvas(app).backfill) == 0  # nothing left to do


def test_backfill_runs_as_a_long_lane_job_at_startup(make_app, fakes):
    with make_app() as api:
        p, doc, _ = ready_report(api)
        run(api, forget_canvas, db(api))
    with make_app() as api:  # restart: the backfill is queued at startup
        drain(api)
        assert len(api.get(f"/api/projects/{p}/datasets").json()["items"]) == 3
        assert api.get(f"/api/projects/{p}/overview").json()["status"] == "ready"
        released = fakes.parser.release_calls
    with make_app() as api:  # nothing stale: no job, so the long lane stays idle
        drain(api)
        assert fakes.parser.release_calls == released


def test_typing_failure_leaves_the_document_ready(app, monkeypatch):
    import app.services.canvas.service as service

    def broken(*_a, **_k):
        raise RuntimeError("typing exploded")

    monkeypatch.setattr(service, "type_document", broken)
    p = project(app)
    doc = upload(app, p)
    drain(app)
    assert app.get(f"/api/documents/{doc}").json()["status"] == "READY"
    assert app.get(f"/api/projects/{p}/datasets").json()["items"] == []


# ------------------------------------------------------------------ the conversation's entry point (events only)


def test_prepare_visual_yields_contract_events(app, fakes):
    p, _, ds = ready_report(app)
    c = chat(app, p)
    labels = {}

    def route(messages):
        user = messages[-1].content
        labels["prompt"] = user
        return {"kind": "line", "datasets": ["D1"], "series": ["D1 · Revenue"], "title": "Quarterly revenue"}

    fakes.llm.route = route

    async def collect():
        return [e async for e in canvas(app).prepare_visual(c, "Show me revenue by quarter", language="en")]

    events = run(app, collect)
    assert [type(e).__name__ for e in events] == ["VisualEvent", "VisualEvent", "CanvasEvent"]
    preparing, ready, board = events
    assert isinstance(preparing, VisualEvent) and preparing.phase == "preparing"
    assert isinstance(ready, VisualEvent) and ready.phase == "ready" and ready.visual is not None
    assert ready.visual.id == preparing.visual_id and ready.visual.kind == "line"
    assert isinstance(board, CanvasEvent) and [v.id for v in board.panels] == [ready.visual.id]
    assert ready.ws_message()["type"] == "visual" and ready.payload()["phase"] == "ready"
    assert set(board.ws_message()) == {"type", "panels"}
    assert '"Quarterly performance' in labels["prompt"] or "D1" in labels["prompt"]


def test_prepare_visual_stays_quiet_for_plain_questions(app, fakes):
    p, _, _ = ready_report(app)
    c = chat(app, p)

    async def collect(question):
        return [e async for e in canvas(app).prepare_visual(c, question, language="en")]

    assert run(app, collect, "Who is the company secretary?") == []
    assert fakes.llm.json_calls == []
    fakes.llm.route = {"kind": "none", "datasets": ["D1"], "series": []}
    events = run(app, collect, "Compare EBITDA across segments")
    assert [(e.phase, e.detail) for e in events] == [("preparing", None), ("failed", "no table fits this question")]  # type: ignore[union-attr]
