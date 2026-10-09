"""Small helpers for putting user and model text into generated Markdown (summaries, transcript export)."""

from __future__ import annotations

import re
from urllib.parse import quote, urlsplit

_BLOCK_START = re.compile(r"^(\s{0,3})(#|>|```|~~~)", re.MULTILINE)


_INLINE = re.compile(r"([\\`*_\[\]<>#|~!])")
_LINK_SAFE = ":/?#@!$&'()*+,;=%-._~"


def escape_inline(text: str) -> str:
    """Untrusted text (a web page's title) as literal Markdown on one line: no emphasis, links, images, HTML or
    tables of its own."""
    return _INLINE.sub(r"\\\1", " ".join(text.split()))


def safe_link(url: str | None) -> str | None:
    """An http(s) URL that can sit in a Markdown autolink (``<…>``): characters that could end or break it
    percent-encoded; None for anything else (javascript:, data:, …)."""
    if not url or urlsplit(url.strip()).scheme.lower() not in ("http", "https"):
        return None
    return quote(url.strip(), safe=_LINK_SAFE)


def escape_block_markers(text: str) -> str:
    """``text`` with a backslash before a ``#`` heading, ``>`` quote or code fence that starts a line, so what was said
    can't turn into structure of the document it is placed in ("# CEO salary?" stays a sentence)."""
    return _BLOCK_START.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", text)
