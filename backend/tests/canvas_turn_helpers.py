"""Helpers for the canvas-in-conversation tests: a scripted router and visual planner on the one fake model, and a
project whose report has chartable tables (the canvas API tests' miniature annual report)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from app.providers.llm import LLMMessage
from app.services.canvas.planner import SYSTEM_PROMPT as PLANNER_PROMPT

from .test_canvas_api import REPORT, chat, drain, project, upload

Choice = dict[str, Any] | Exception | None
PlannerScript = Choice | Callable[[list[LLMMessage]], Choice]


def is_planner(messages: list[LLMMessage]) -> bool:
    return messages[0].content == PLANNER_PROMPT


def utterance_of(messages: list[LLMMessage]) -> str:
    return messages[-1].content.rsplit("Utterance: ", 1)[-1]


def choose(kind: str, *words: str, **extra: Any) -> Callable[[list[LLMMessage]], Choice]:
    """A planner that picks the first offered table with a row or column named like each of ``words`` (as the
    model would: labels such as "D1 · Revenue" from the request's own enums)."""

    def pick(messages: list[LLMMessage]) -> Choice:
        lines = messages[-1].content.split("Tables:\n", 1)[1].splitlines()
        for word in words or ("",):
            alias = None
            for line in lines:
                if not line.startswith("  "):
                    alias = line.split(" ", 1)[0]
                    continue
                items = line.strip().split(": ", 1)[1].split("; ")
                labels = [re.sub(r"( \[[^\]]*\])?( \(total\))?$", "", item) for item in items]
                found = [label for label in labels if word.casefold() in label.casefold()]
                if found and alias:
                    series = [f"{alias} · {label}" for label in found[:1]]
                    for other in words[1:]:
                        series += [f"{alias} · {lab}" for lab in labels if other.casefold() in lab.casefold()][:1]
                    return {"kind": kind, "datasets": [alias], "series": list(dict.fromkeys(series)), **extra}
        return {"kind": "none", "datasets": ["D1"], "series": []}

    return pick


class Script:
    """The fake model's JSON calls: router proposals by utterance, the planner's choice, and when each call came."""

    def __init__(self, router: dict[str, dict[str, Any]] | None = None, planner: PlannerScript = None) -> None:
        self.router = router or {}
        self.planner = planner
        self.planner_delay = 0.0
        self.router_delay = 0.0
        self.planner_at: list[float] = []  # perf_counter of each planner call

    def route(self, messages: list[LLMMessage]) -> Choice:
        if is_planner(messages):
            return self.planner(messages) if callable(self.planner) else self.planner
        return self.router.get(utterance_of(messages))

    def delay(self, messages: list[LLMMessage]) -> float:
        if is_planner(messages):
            self.planner_at.append(time.perf_counter())
            return self.planner_delay
        return self.router_delay

    def install(self, llm: Any) -> Script:
        llm.route = self.route
        llm.json_delay = self.delay
        return self


def planner_calls(llm: Any) -> list[dict[str, Any]]:
    return [c for c in llm.json_calls if is_planner(c["messages"])]


def report_chat(api: TestClient) -> tuple[str, str, dict[str, dict[str, Any]]]:
    """A project with the report (KPI table p2, quarterly time series p3, segment composition p4) and a chat."""
    p = project(api)
    upload(api, p)
    drain(api)
    ds = {f"p{d['page']}": d for d in api.get(f"/api/projects/{p}/datasets").json()["items"]}
    return p, chat(api, p), ds


def add_visual(api: TestClient, chat_id: str, **spec: Any) -> dict[str, Any]:
    r = api.post(f"/api/chats/{chat_id}/visuals", json=spec)
    assert r.status_code == 201, r.text
    return r.json()


def line_of_revenue(api: TestClient, chat_id: str, ds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    q = ds["p3"]["id"]
    return add_visual(api, chat_id, kind="line", datasets=[q], series=[f"{q}:revenue"], title="Revenue by quarter")


def donut_of_segments(api: TestClient, chat_id: str, ds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s = ds["p4"]["id"]
    return add_visual(api, chat_id, kind="donut", datasets=[s], series=[f"{s}:revenue_fy24"], title="Segment revenue")


__all__ = ["REPORT", "Script", "add_visual", "choose", "donut_of_segments", "line_of_revenue", "report_chat"]
