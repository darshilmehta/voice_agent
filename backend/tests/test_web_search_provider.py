"""The SearXNG provider (docs/DESIGN.md §3.7) against a mocked SearXNG: one request per engine, first engine first,
de-duplication, per-request and overall timeouts, the offline guard, health, and page fetching (text only, size cap,
public addresses only)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.providers.base import HealthStatus, ProviderContext
from app.providers.registry import build_container
from app.providers.web_search import (
    PAGE_MAX_BYTES,
    SearXNGSearch,
    WebSearchError,
    WebSearchRefused,
    WebSearchTimeout,
    clean_query,
    page_text,
    url_key,
)
from app.settings import Settings, load_settings

from .conftest import LOCAL_CONFIG

ENGINES = '["slow", "fast", "broken"]'


@pytest.fixture
def web_settings(write_config, tmp_path) -> Callable[..., Settings]:
    """Local config with web search allowed (strict_offline_exceptions) and turned on, plus env overrides."""
    allowed = write_config(LOCAL_CONFIG, strict_offline_exceptions=["web_search"])

    def load(**env: str) -> Settings:
        base = {
            "APP_ROOT_DIR": str(tmp_path),
            "TOOLS__WEB_SEARCH__ENABLED": "true",
            "TOOLS__WEB_SEARCH__ENGINES": ENGINES,
            "TOOLS__WEB_SEARCH__TIMEOUT_S": "2",
            "TOOLS__WEB_SEARCH__REQUEST_TIMEOUT_S": "1.0",
        }
        return load_settings(allowed, {**base, **env})

    return load


def item(url: str, title: str = "A title", content: str = "some text", **extra: Any) -> dict[str, Any]:
    return {"url": url, "title": title, "content": content, **extra}


class SearXNG:
    """A mocked SearXNG: per engine, a delay and a result list (or an exception). Honours the request's read
    timeout like a real server would (``ReadTimeout``), unless the engine ``ignores_timeout``."""

    def __init__(self) -> None:
        self.engines: dict[str, tuple[float, list[dict[str, Any]] | Exception]] = {}
        self.ignores_timeout: set[str] = set()
        self.requests: list[httpx.Request] = []
        self.config_engines: list[str] | None = None  # /config: None → every engine named in ``engines``
        self.healthy = True

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/healthz":
            if not self.healthy:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, text="OK")
        if request.url.path == "/config":
            names = self.config_engines if self.config_engines is not None else list(self.engines)
            return httpx.Response(200, json={"engines": [{"name": n, "enabled": True} for n in names]})
        engine = request.url.params.get("engines", "")
        delay, outcome = self.engines.get(engine, (0.0, []))
        timeout = request.extensions.get("timeout", {}).get("read")
        if timeout is not None and delay > timeout and engine not in self.ignores_timeout:
            await asyncio.sleep(timeout)
            raise httpx.ReadTimeout("timed out", request=request)
        await asyncio.sleep(delay)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(200, json={"query": request.url.params["q"], "results": outcome})

    def searches(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == "/search"]


@pytest.fixture
def searxng() -> SearXNG:
    return SearXNG()


def provider(settings: Settings, server: SearXNG) -> SearXNGSearch:
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return SearXNGSearch(settings.tools.web_search, ProviderContext(settings=settings, http=http))


async def collect(p: SearXNGSearch, query: str = "infosys share price today", max_results: int | None = None):
    """(results, arrival times in ms, the error the stream ended with)."""
    out, times = [], []
    t0 = time.perf_counter()
    error: Exception | None = None
    try:
        async for r in p.stream(query, max_results):
            out.append(r)
            times.append((time.perf_counter() - t0) * 1000)
    except WebSearchError as e:
        error = e
    return out, times, error


# ------------------------------------------------------------------ fan-out


async def test_one_json_request_per_engine_and_the_first_engine_to_answer_comes_first(web_settings, searxng):
    searxng.engines = {
        "slow": (0.3, [item("https://slow.example.com/a", "Slow A")]),
        "fast": (0.02, [item("https://fast.example.com/a", "Fast A"), item("https://fast.example.com/b", "Fast B")]),
        "broken": (0.0, httpx.ConnectError("engine down")),
    }
    p = provider(web_settings(), searxng)
    results, times, error = await collect(p)
    assert error is None
    assert [r.title for r in results] == ["Fast A", "Fast B", "Slow A"]  # whichever engine answers first
    assert times[0] < 200 and times[2] >= 250  # the fast engine wasn't held up by the slow one
    assert (results[0].engine, results[0].site) == ("fast", "fast.example.com")
    searches = searxng.searches()
    assert sorted(r.url.params["engines"] for r in searches) == ["broken", "fast", "slow"]
    for r in searches:  # the query, and nothing else of the conversation, leaves the machine
        assert dict(r.url.params) == {
            "q": "infosys share price today",
            "format": "json",
            "safesearch": "1",
            "engines": r.url.params["engines"],
        }
        assert r.method == "GET" and r.content == b""


async def test_results_are_deduplicated_by_url_and_capped(web_settings, searxng):
    searxng.engines = {
        "fast": (0.0, [item("https://www.news.example.com/story/?utm_source=x", "First"), item("ftp://x/y", "no")]),
        "slow": (0.05, [item("https://news.example.com/story", "Same story"), item("https://b.example.com/", "B")]),
        "broken": (0.1, [item("https://c.example.com/1", "C1"), item("https://c.example.com/2", "C2")]),
    }
    results, _, _ = await collect(provider(web_settings(), searxng), max_results=3)
    assert [r.title for r in results] == ["First", "B", "C1"]  # duplicates and non-web links dropped, 3 at most
    assert url_key("https://www.News.example.com/story/?utm_source=x#top") == url_key("https://news.example.com/story")


async def test_each_engine_contributes_at_most_its_share(web_settings, searxng):
    many = [item(f"https://fast.example.com/{i}", f"F{i}") for i in range(10)]
    searxng.engines = {
        "fast": (0.0, many),
        "slow": (0.05, [item("https://slow.example.com/1", "S1")]),
        "broken": (0.1, []),
    }
    results, _, _ = await collect(provider(web_settings(), searxng), max_results=5)
    assert [r.title for r in results] == ["F0", "F1", "S1"]  # 5 results over 3 engines: 2 each at most


async def test_a_slow_engine_hits_its_request_timeout_and_the_others_still_answer(web_settings, searxng):
    searxng.engines = {
        "slow": (5.0, [item("https://slow.example.com/a", "never")]),
        "fast": (0.0, [item("https://fast.example.com/a", "Fast")]),
    }
    t0 = time.perf_counter()
    results, _, error = await collect(provider(web_settings(TOOLS__WEB_SEARCH__REQUEST_TIMEOUT_S="0.2"), searxng))
    assert [r.title for r in results] == ["Fast"] and error is None
    assert time.perf_counter() - t0 < 1.0  # the slow engine's 5 s didn't hold the stream


async def test_the_overall_deadline_ends_the_stream_after_what_arrived(web_settings, searxng):
    searxng.engines = {
        "slow": (30.0, [item("https://slow.example.com/a", "never")]),
        "fast": (0.0, [item("https://fast.example.com/a", "Fast")]),
    }
    searxng.ignores_timeout = {"slow"}  # an engine that hangs past its own timeout
    p = provider(web_settings(TOOLS__WEB_SEARCH__TIMEOUT_S="1", TOOLS__WEB_SEARCH__REQUEST_TIMEOUT_S="20"), searxng)
    t0 = time.perf_counter()
    results, _, error = await collect(p)
    assert [r.title for r in results] == ["Fast"]
    assert isinstance(error, WebSearchTimeout) and 0.9 < time.perf_counter() - t0 < 1.5
    await asyncio.sleep(0)
    assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and "_ask" in repr(t)]


async def test_closing_the_stream_cancels_the_requests_still_running(web_settings, searxng):
    searxng.engines = {"fast": (0.0, [item("https://fast.example.com/a", "Fast")]), "slow": (10.0, [])}
    searxng.ignores_timeout = {"slow"}
    p = provider(web_settings(), searxng)
    stream = p.stream("x")
    assert (await anext(stream)).title == "Fast"
    await stream.aclose()  # the turn was stopped
    await asyncio.sleep(0)
    assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and "_ask" in repr(t)]


async def test_every_engine_failing_is_an_error_and_searxng_down_marks_the_tool_unavailable(web_settings, searxng):
    searxng.engines = {
        name: (0.0, httpx.ConnectError("refused")) for name in ("slow", "fast", "broken")
    }  # SearXNG itself unreachable
    p = provider(web_settings(), searxng)
    _, _, error = await collect(p)
    assert isinstance(error, WebSearchError) and "every engine failed" in str(error)
    assert p.unavailable_reason() is not None and "unreachable" in p.unavailable_reason()
    assert (await p.health()).status == HealthStatus.OK  # reachable again: available at once
    assert p.unavailable_reason() is None


async def test_blocked_engines_are_failures_and_no_results_is_not_an_error(web_settings, searxng):
    searxng.engines = {
        "fast": (0.0, []),
        "slow": (0.0, []),
        "broken": (0.0, httpx.ConnectError("x")),
    }
    results, _, error = await collect(provider(web_settings(), searxng))
    assert results == [] and error is None  # engines answered, with nothing


async def test_engines_searxng_doesnt_have_are_skipped(web_settings, searxng):
    """SearXNG answers a request for an unknown engine with its default engines (a slow, full search)."""
    searxng.engines = {"fast": (0.0, [item("https://fast.example.com/a", "Fast")])}
    searxng.config_engines = ["fast"]
    p = provider(web_settings(), searxng)
    results, _, _ = await collect(p)
    assert [r.title for r in results] == ["Fast"]
    assert [r.url.params["engines"] for r in searxng.searches()] == ["fast"]
    health = await p.health()
    assert health.status == HealthStatus.DEGRADED and "broken, slow" in health.detail


def test_queries_are_one_short_line():
    assert clean_query("  what's   the\nprice\ttoday ") == "what's the price today"
    assert len(clean_query("word " * 100)) <= 200


# ------------------------------------------------------------------ the offline guard


@pytest.mark.parametrize("stage", ["stream", "fetch"])
async def test_refused_unless_enabled(load_local, searxng, stage):
    p = provider(load_local(), searxng)  # tools.web_search.enabled: false
    with pytest.raises(WebSearchRefused, match="turned off"):
        if stage == "stream":
            await anext(p.stream("x"))
        else:
            await p.fetch_page("https://example.com/")
    assert searxng.requests == []  # nothing left the machine
    assert p.unavailable_reason() == "web search is turned off (tools.web_search.enabled)"
    assert (await p.health()).status == HealthStatus.DISABLED


async def test_refused_under_strict_offline_without_the_exception(load_local, searxng):
    p = provider(load_local(TOOLS__WEB_SEARCH__ENABLED="true"), searxng)  # strict_offline, no exception
    with pytest.raises(WebSearchRefused, match="strict_offline_exceptions"):
        await anext(p.stream("x"))
    assert searxng.requests == []
    with pytest.raises(Exception, match="web_search"):
        build_container(load_local(TOOLS__WEB_SEARCH__ENABLED="true"))  # and the app wouldn't start


async def test_allowed_with_the_exception(web_settings, searxng):
    searxng.engines = {"fast": (0.0, [item("https://fast.example.com/a", "Fast")])}
    s = web_settings()
    build_container(s)  # starts
    results, _, _ = await collect(provider(s, searxng))
    assert len(results) == 1


async def test_health(web_settings, searxng):
    searxng.engines = {name: (0.0, []) for name in ("slow", "fast", "broken")}
    p = provider(web_settings(), searxng)
    assert (await p.health()).status == HealthStatus.OK
    searxng.healthy = False
    down = await p.health()
    assert down.status == HealthStatus.DOWN and "unreachable" in down.detail
    assert p.unavailable_reason() is not None  # turns answer without it at once


# ------------------------------------------------------------------ pages


class Pages:
    def __init__(self) -> None:
        self.responses: dict[str, httpx.Response] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.get(str(request.url), httpx.Response(404))


@pytest.fixture
def pages(web_settings, searxng) -> tuple[SearXNGSearch, Pages]:
    p = provider(web_settings(), searxng)
    server = Pages()
    p.page_client = httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=False)
    addresses = {"news.example.com": ["93.184.215.14"], "intranet.example.com": ["10.0.0.7"]}

    async def resolve(host: str) -> list[str]:
        return addresses.get(host, ["93.184.215.15"])

    p.resolve = resolve  # type: ignore[method-assign]
    return p, server


HTML = """<html><head><title>t</title><style>.x{color:red}</style><script>var secret = 1;</script></head>
<body><nav>Home | Markets | Login to your account now please</nav>
<p>Infosys shares closed at Rs 1,380.45 on Thursday, up 0.43 percent on the day.</p>
<p>Trading volume was 2.18 lakh shares on the BSE, below the monthly average.</p>
<footer>Copyright notice and other long footer text that is not useful at all</footer></body></html>"""


async def test_a_page_is_reduced_to_its_readable_text(pages):
    p, server = pages
    server.responses["https://news.example.com/a"] = httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8", "set-cookie": "id=1"}, text=HTML
    )
    text = await p.fetch_page("https://news.example.com/a")
    assert text == (
        "Infosys shares closed at Rs 1,380.45 on Thursday, up 0.43 percent on the day.\n"
        "Trading volume was 2.18 lakh shares on the BSE, below the monthly average."
    )
    assert "secret" not in text and "Login" not in text
    assert not p.page_client.cookies  # nothing a page sets is kept


async def test_pages_are_capped_and_text_only(pages):
    p, server = pages
    huge = "<p>" + "word " * (PAGE_MAX_BYTES // 2) + "</p>"
    server.responses["https://news.example.com/huge"] = httpx.Response(
        200, headers={"content-type": "text/html"}, text=huge
    )
    server.responses["https://news.example.com/file.pdf"] = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7"
    )
    text = await p.fetch_page("https://news.example.com/huge")
    assert text is not None and len(text) <= 1_510 and text.endswith("…")
    assert await p.fetch_page("https://news.example.com/file.pdf") is None


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/api/projects",
        "http://localhost:6333/collections",
        "http://[::1]/",
        "http://192.168.1.1/admin",
        "https://intranet.example.com/",  # resolves to a private address
        "file:///etc/passwd",
        "http://user:pw@news.example.com/",
        "http://printer.local/",
    ],
)
async def test_pages_on_this_machine_or_private_networks_are_never_fetched(pages, url):
    p, server = pages
    assert await p.fetch_page(url) is None
    assert server.requests == []


async def test_redirects_are_checked_too(pages):
    p, server = pages
    server.responses["https://news.example.com/r"] = httpx.Response(302, headers={"location": "http://127.0.0.1:8000/"})
    server.responses["https://news.example.com/ok"] = httpx.Response(301, headers={"location": "/final"})
    server.responses["https://news.example.com/final"] = httpx.Response(
        200, headers={"content-type": "text/plain"}, text="A plain text page with enough words in it to be kept."
    )
    assert await p.fetch_page("https://news.example.com/r") is None
    assert [str(r.url) for r in server.requests] == ["https://news.example.com/r"]
    assert await p.fetch_page("https://news.example.com/ok") == "A plain text page with enough words in it to be kept."


def test_page_text_prefers_paragraphs_and_falls_back_to_other_blocks():
    table = "<table><tr><td>Revenue for the year was up sharply on strong demand</td></tr></table>"
    assert page_text(table, "text/html") == "Revenue for the year was up sharply on strong demand"
    assert json.dumps(page_text("<p>short</p>", "text/html")) == '""'
