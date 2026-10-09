"""Live-data web search providers (docs/DESIGN.md §3.7).

    WebSearch.stream(query, max_results) → SearchResult one at a time, as they arrive
      searxng      self-hosted SearXNG (127.0.0.1:8888, Compose profile "websearch"). SearXNG answers only once all
                   of a request's engines are done, so the provider fans out one JSON request per engine
                   (engines=<name>, format=json) and yields whichever engine answers first; results are
                   de-duplicated by URL, at most a fair share per engine, with a per-request timeout
                   (request_timeout_s) and an overall deadline (timeout_s → WebSearchTimeout after what arrived)
      search_api   a production search API (Brave, Tavily, …): placeholder
    WebSearch.fetch_page(url) → the page as plain text: http(s) only, public addresses only (no loopback or private
      networks, redirects checked too), HTML or plain text only, at most PAGE_MAX_BYTES read, PAGE_TEXT_CHARS kept

Privacy: the query string is the only thing sent to SearXNG (which forwards it to public engines). Callers pass the
rewritten English search question, never document text or the transcript (services/live_data.py builds it). Page
fetches GET a result's own URL with no cookies kept and nothing else attached.

Guard: ``stream`` and ``fetch_page`` refuse (``WebSearchRefused``) unless ``tools.web_search.enabled`` and, under
``strict_offline``, ``"web_search"`` is in ``strict_offline_exceptions``; the registry also refuses to start
otherwise. After a search fails because SearXNG itself can't be reached, the provider reports itself unavailable
for UNHEALTHY_COOLDOWN_S (or until a health check succeeds), so turns answer from the documents at once instead of
waiting for timeouts.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import math
import re
import socket
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from .base import HealthStatus, PlaceholderProvider, Provider, ProviderHealth
from .registry import register

QUERY_MAX_CHARS = 200  # a search query, not a document
SNIPPET_MAX_CHARS = 400
PAGE_MAX_BYTES = 400_000  # read at most this much of a page
PAGE_TEXT_CHARS = 1_500  # keep at most this much of its text
PAGE_MAX_REDIRECTS = 3
UNHEALTHY_COOLDOWN_S = 30.0
USER_AGENT = "Mozilla/5.0 (compatible; gibberlink-local/0.1; +https://github.com/darshilmehta/voice_agent)"
_TRACKING_PARAMS = re.compile(r"^(utm_.*|fbclid|gclid|msclkid|ref|ref_src|igshid|mc_cid|mc_eid)$", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SearchResult:
    url: str
    title: str
    snippet: str  # the engine's text for the result (may be empty)
    site: str  # host without "www." ("livemint.com")
    engine: str  # the engine that returned it first
    published: str | None = None  # as the engine gives it (ISO date-time), when it does


class WebSearchError(RuntimeError):
    """The search failed (SearXNG unreachable, every engine failed, bad response)."""


class WebSearchRefused(WebSearchError):
    """The configuration doesn't allow web search (disabled, or strict_offline without the exception)."""


class WebSearchTimeout(WebSearchError):
    """The overall deadline passed. Results yielded before it stay valid."""


def describe(e: BaseException) -> str:
    text = str(e)
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def clean_query(query: str) -> str:
    """One line, single spaces, at most QUERY_MAX_CHARS (cut at a word)."""
    q = _SPACE.sub(" ", query).strip()
    if len(q) > QUERY_MAX_CHARS:
        cut = q[:QUERY_MAX_CHARS]
        q = cut[: cut.rfind(" ")] if " " in cut else cut
    return q


def site_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def url_key(url: str) -> str:
    """A URL for de-duplication: scheme-less, lower-case host without "www.", no fragment, no trailing slash, no
    tracking parameters."""
    parts = urlsplit(url.strip())
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)))
    path = parts.path.rstrip("/")
    return urlunsplit(("", site_of(url), path, query, ""))


def _text(value: Any, limit: int) -> str:
    text = _SPACE.sub(" ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse_results(data: Any, engine: str) -> list[SearchResult]:
    """SearXNG's JSON ``results`` as SearchResults (http(s) links with a title only), in its order."""
    out: list[SearchResult] = []
    items = data.get("results") if isinstance(data, dict) else None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = _text(item.get("title"), 200)
        if not title or urlsplit(url).scheme not in ("http", "https") or not site_of(url):
            continue
        published = item.get("publishedDate") or item.get("pubdate")
        out.append(
            SearchResult(
                url=url,
                title=title,
                snippet=_text(item.get("content"), SNIPPET_MAX_CHARS),
                site=site_of(url),
                engine=str(item.get("engine") or engine),
                published=str(published) if published else None,
            )
        )
    return out


class WebSearch(Provider):
    """The live-data tool's interface: ``stream`` (results as they arrive), ``fetch_page`` (a result page as text) and
    ``unavailable_reason`` (why it can't run now, None when it can)."""

    capability = "web_search"

    def unavailable_reason(self) -> str | None:
        """Why a search can't run now (turned off, strict offline, recently unreachable), or None."""
        return refusal(self)

    def stream(self, query: str, max_results: int | None = None) -> AsyncIterator[SearchResult]:
        raise NotImplementedError(f"{type(self).__name__}.stream")

    async def fetch_page(self, url: str) -> str | None:
        """The page's readable text (None when it can't be fetched or isn't text)."""
        raise NotImplementedError(f"{type(self).__name__}.fetch_page")


def refusal(provider: Provider) -> str | None:
    """What the configuration says (docs/DESIGN.md §6.3): None when web search may run."""
    settings = provider.ctx.settings
    if not settings.tools.web_search.enabled:
        return "web search is turned off (tools.web_search.enabled)"
    if settings.strict_offline and "web_search" not in settings.strict_offline_exceptions:
        return 'strict_offline is on and "web_search" is not in strict_offline_exceptions'
    return None


# ------------------------------------------------------------------ page text


class _TextExtractor(HTMLParser):
    """Readable text of an HTML page: text blocks outside scripts, styles, navigation and forms; paragraphs first
    (an article's body), other blocks (table cells, list items) only when paragraphs are scarce."""

    SKIP = frozenset({"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "template", "iframe"})
    BLOCKS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "td", "th", "blockquote", "article", "section", "div", "br"})
    PARAGRAPH_ENOUGH = 400  # characters of paragraph text that make the rest unnecessary

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skipping = 0
        self.blocks: list[tuple[bool, str]] = []  # (is a paragraph, text)
        self.current: list[str] = []
        self.in_paragraph = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCKS:
            self._flush()
            self.in_paragraph = tag == "p"

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self.skipping = max(0, self.skipping - 1)
        elif tag in self.BLOCKS:
            self._flush()
            self.in_paragraph = False

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.current.append(data)

    def _flush(self) -> None:
        text = _SPACE.sub(" ", "".join(self.current)).strip()
        self.current = []
        if len(text) >= 40:  # menus, buttons and captions are short; sentences aren't
            self.blocks.append((self.in_paragraph, text))

    def text(self) -> str:
        self._flush()
        paragraphs = [t for is_p, t in self.blocks if is_p]
        chosen = paragraphs if sum(map(len, paragraphs)) >= self.PARAGRAPH_ENOUGH else [t for _, t in self.blocks]
        return "\n".join(dict.fromkeys(chosen))


def page_text(body: str, content_type: str, limit: int = PAGE_TEXT_CHARS) -> str:
    """Plain text of an HTML or text page, at most ``limit`` characters (cut at a word)."""
    if "html" in content_type:
        parser = _TextExtractor()
        with contextlib.suppress(Exception):  # malformed markup: keep what was parsed
            parser.feed(body)
            parser.close()
        text = parser.text()
    else:
        text = "\n".join(line.strip() for line in body.splitlines() if line.strip())
    if len(text) > limit:
        cut = text[:limit]
        text = cut[: cut.rfind(" ")] if " " in cut else cut
        text = text.rstrip() + " …"
    return text


def public_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    return ip.is_global and not ip.is_multicast


# ------------------------------------------------------------------ SearXNG


@register
class SearXNGSearch(WebSearch):
    """Self-hosted SearXNG. The container is local, but it forwards queries to public engines, so it is remote."""

    name = "searxng"
    is_remote = True

    def __init__(self, config: Any, ctx: Any) -> None:
        super().__init__(config, ctx)
        self._down_until = 0.0
        self._down_reason = ""
        self._known: frozenset[str] | None = None  # the engines SearXNG has (from /config), once asked
        self.page_client: httpx.AsyncClient | None = None  # page fetches: no cookies kept (tests may set one)

    @property
    def base_url(self) -> str:
        return (self.config.url or "").rstrip("/")  # type: ignore[attr-defined]

    async def close(self) -> None:
        if self.page_client is not None:
            await self.page_client.aclose()
            self.page_client = None

    def unavailable_reason(self) -> str | None:
        reason = refusal(self)
        if reason is None and time.monotonic() < self._down_until:
            left = self._down_until - time.monotonic()
            reason = f"SearXNG was unreachable ({self._down_reason}); retrying in {left:.0f} s"
        return reason

    def _mark_down(self, reason: str) -> None:
        self._down_until, self._down_reason = time.monotonic() + UNHEALTHY_COOLDOWN_S, reason

    async def health(self) -> ProviderHealth:
        reason = refusal(self)
        if reason is not None:
            return self._health(HealthStatus.DISABLED, reason)
        base = self.base_url
        r, ms, err = await self._probe(f"{base}/healthz")
        if r is None or r.status_code != 200:
            detail = err or f"HTTP {r.status_code}"  # type: ignore[union-attr]
            self._mark_down(detail)
            return self._health(HealthStatus.DOWN, f"SearXNG unreachable at {base} ({detail})", ms)
        self._down_until = 0.0  # reachable again
        engines = await self._engines()
        missing = sorted(set(self.config.engines) - {e for e in engines if e})  # type: ignore[attr-defined]
        names = ", ".join(e for e in engines if e) or "SearXNG's defaults"
        if missing:
            detail = f"engines {', '.join(missing)} are not enabled in SearXNG; searching with {names}"
            return self._health(HealthStatus.DEGRADED, detail, ms)
        return self._health(HealthStatus.OK, f"ready (engines: {names})", ms)

    def _check_allowed(self) -> None:
        reason = refusal(self)
        if reason is not None:
            raise WebSearchRefused(reason)

    async def _engines(self) -> list[str | None]:
        """The configured engines SearXNG actually has (asked once, from /config). SearXNG answers a request for an
        unknown engine with its default engines, which would turn one fast request into a slow full search; None
        stands for those defaults when no engine is configured (or none of them exists)."""
        configured: list[str] = list(dict.fromkeys(self.config.engines))  # type: ignore[attr-defined]
        if configured and self._known is None:
            with contextlib.suppress(httpx.HTTPError, ValueError, TypeError, AttributeError):
                r = await self.ctx.http.get(f"{self.base_url}/config", timeout=httpx.Timeout(2.0, connect=1.0))
                if r.status_code == 200:
                    engines = r.json().get("engines") or []
                    self._known = frozenset(e["name"] for e in engines if isinstance(e, dict) and e.get("enabled"))
        known = [e for e in configured if self._known is None or e in self._known]
        return list(known) or [None]

    async def stream(  # type: ignore[override]
        self, query: str, max_results: int | None = None
    ) -> AsyncIterator[SearchResult]:
        """Results as engines answer: the first engine's first, each engine's best ``per_engine`` (so several engines
        contribute), de-duplicated by URL, until ``max_results`` or every engine answered. Raises WebSearchTimeout
        at the deadline (after yielding what came) and WebSearchError when every engine failed. Closing the stream
        cancels the requests still running."""
        self._check_allowed()
        q = clean_query(query)
        if not q:
            raise WebSearchError("empty query")
        cfg = self.config
        limit = max(1, max_results or cfg.max_results)  # type: ignore[attr-defined]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + cfg.timeout_s  # type: ignore[attr-defined]
        engines = await self._engines()
        per_engine = max(2, math.ceil(limit / len(engines)))
        answers: asyncio.Queue[tuple[str, list[SearchResult] | BaseException]] = asyncio.Queue()
        tasks = [asyncio.ensure_future(self._ask(q, engine, answers, deadline - loop.time())) for engine in engines]
        seen: set[str] = set()
        errors: list[tuple[str, BaseException]] = []
        yielded = 0
        try:
            for _ in tasks:
                remaining = deadline - loop.time()
                try:
                    if remaining <= 0:
                        raise TimeoutError
                    engine, outcome = await asyncio.wait_for(answers.get(), remaining)
                except TimeoutError:
                    raise WebSearchTimeout(
                        f"search not finished after {cfg.timeout_s} s ({yielded} results)"  # type: ignore[attr-defined]
                    ) from None
                if isinstance(outcome, BaseException):
                    errors.append((engine, outcome))
                    continue
                for result in outcome[:per_engine]:
                    key = url_key(result.url)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield result
                    yielded += 1
                    if yielded >= limit:
                        return
            if yielded == 0 and errors and len(errors) == len(tasks):
                if all(isinstance(e, httpx.TransportError) for _, e in errors):
                    self._mark_down(describe(errors[0][1]))
                raise WebSearchError("every engine failed: " + "; ".join(f"{e}: {describe(x)}" for e, x in errors))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _ask(
        self,
        query: str,
        engine: str | None,
        answers: asyncio.Queue[tuple[str, list[SearchResult] | BaseException]],
        budget_s: float,
    ) -> None:
        """One SearXNG request for one engine; its results (or the error) go to ``answers``."""
        name = engine or "searxng"
        params = {"q": query, "format": "json", "safesearch": "1"}
        if engine:
            params["engines"] = engine
        timeout = max(0.1, min(float(self.config.request_timeout_s), budget_s))  # type: ignore[attr-defined]
        try:
            r = await self.ctx.http.get(
                f"{self.base_url}/search",
                params=params,
                headers={"Accept": "application/json"},
                timeout=httpx.Timeout(timeout, connect=min(1.0, timeout)),
            )
            if r.status_code != 200:
                raise WebSearchError(f"SearXNG returned HTTP {r.status_code}")
            data = r.json()
            results = parse_results(data, name)
            unresponsive = data.get("unresponsive_engines") if isinstance(data, dict) else None
            if not results and unresponsive:
                raise WebSearchError(f"engine unresponsive: {unresponsive}")
            answers.put_nowait((name, results))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            answers.put_nowait((name, e))

    # -------------------------------------------------------------- pages

    def _pages(self) -> httpx.AsyncClient:
        if self.page_client is None:
            self.page_client = httpx.AsyncClient(follow_redirects=False, headers={"User-Agent": USER_AGENT})
        return self.page_client

    async def resolve(self, host: str) -> list[str]:
        """Addresses ``host`` resolves to (tests replace this)."""
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return [str(info[4][0]) for info in infos]

    async def _allowed_url(self, url: str) -> bool:
        """http(s) to a public address only: never this machine or a private network (a result can't make the
        backend reach internal services)."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme not in ("http", "https") or not host or parts.username or parts.password:
            return False
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return public_address(host)
        try:
            addresses = await asyncio.wait_for(self.resolve(host), 2.0)
        except (OSError, TimeoutError):
            return False
        return bool(addresses) and all(public_address(a) for a in addresses)

    async def fetch_page(self, url: str) -> str | None:  # type: ignore[override]
        self._check_allowed()
        client = self._pages()
        timeout = httpx.Timeout(float(self.config.request_timeout_s), connect=2.0)  # type: ignore[attr-defined]
        try:
            for _ in range(PAGE_MAX_REDIRECTS + 1):
                if not await self._allowed_url(url):
                    return None
                async with client.stream(
                    "GET", url, timeout=timeout, headers={"Accept": "text/html,text/plain;q=0.9"}
                ) as response:
                    if response.is_redirect and "location" in response.headers:
                        url = urljoin(url, response.headers["location"])
                        continue
                    content_type = response.headers.get("content-type", "").lower()
                    if response.status_code != 200 or not content_type.startswith(("text/html", "text/plain")):
                        return None
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) >= PAGE_MAX_BYTES:
                            break
                    text = bytes(body[:PAGE_MAX_BYTES]).decode(response.encoding or "utf-8", errors="replace")
                    return page_text(text, content_type) or None
            return None
        except (httpx.HTTPError, LookupError, UnicodeError):
            return None
        finally:
            client.cookies.clear()  # nothing a page sets is kept or sent later


@register
class SearchAPI(WebSearch, PlaceholderProvider):
    name = "search_api"
