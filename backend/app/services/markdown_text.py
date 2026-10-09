"""Small helpers for putting user and model text into generated Markdown (summaries, transcript export)."""

from __future__ import annotations

import re

_BLOCK_START = re.compile(r"^(\s{0,3})(#|>|```|~~~)", re.MULTILINE)


def escape_block_markers(text: str) -> str:
    """``text`` with a backslash before a ``#`` heading, ``>`` quote or code fence that starts a line, so what was said
    can't turn into structure of the document it is placed in ("# CEO salary?" stays a sentence)."""
    return _BLOCK_START.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", text)
