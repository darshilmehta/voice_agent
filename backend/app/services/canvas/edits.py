"""Edits that rebuild a visual keep what was on screen (docs/DESIGN.md §12.1, "Canvas edits"). Pure.

A visual is rebuilt from its stored spec through the validator and the builder; ``planner.first_valid`` tries lesser
forms of a spec that doesn't build (fewer series, no filters, another kind). Fine for a new chart, wrong for an edit:
"show it as a table" said of a chart of two quarterly tables (FY23's and FY24's) must not come back as one table's
first series.

- ``keeps_data``: the rebuilt spec shows every table, series and x item of the visual it replaces (a table may show
  more: its total rows).
- ``change_kind``: the visual in another kind with all its data, or None ("I can't show that chart that way.").
- ``merge_planned``: what the planner chose for an edit the rules can't read ("add EBITDA to that chart"), plus
  everything that was on screen, unless the user asked to drop something.
- ``edit_language``: the language of a rebuilt visual: the chat's response language, not the edit utterance's.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from ...domain.canvas import VisualKind
from ...domain.datasets import TypedDataset
from ..language import asked_language
from .planner import first_valid
from .spec import MAX_DATASETS, MAX_SERIES, SpecError, VisualSpec, resolve

_LETTER = "A-Za-z0-9ऀ-ॣ०-ॿ"
# The user wants something gone: then what the planner dropped is not a loss.
_DROPS = re.compile(
    f"(?<![{_LETTER}])(?:only|just|instead|replace|replacing|swap|without|except|remove|delete|drop|hide|"
    "sirf|keval|hata\\w*|hatao|bajaye|bajay|की\\s+जगह|ki\\s+jagah|केवल|सिर्फ़|सिर्फ|हटा\\w*|बजाय)"
    f"(?![{_LETTER}])",
    re.IGNORECASE,
)


def _shown(spec: VisualSpec, datasets: Mapping[str, TypedDataset]) -> set[str] | None:
    """The x items a spec shows, without a table's total rows (None: it doesn't resolve)."""
    try:
        resolved = resolve(spec, datasets)
    except SpecError:
        return None
    if resolved.kind == "timeline":
        return {row.row.key for row in resolved.timeline}
    return {i.label for i in resolved.x_items if not i.is_total}


def keeps_data(old: VisualSpec, new: VisualSpec, datasets: Mapping[str, TypedDataset]) -> bool:
    """``new`` shows all of what ``old`` did: its tables, its series, its x items (periods, categories). A table also
    shows total rows ("Full year FY24") that a chart doesn't, so only the other rows count."""
    if not set(old.datasets) <= set(new.datasets) or not set(old.series) <= set(new.series):
        return False
    before, after = _shown(old, datasets), _shown(new, datasets)
    return before is None or (after is not None and before <= after)


def change_kind(
    spec: VisualSpec,
    kind: VisualKind,
    datasets: Mapping[str, TypedDataset],
    check: Callable[[VisualSpec], bool],
) -> VisualSpec | None:
    """``spec`` as a ``kind`` that shows all its data. The forms ``first_valid`` falls back to that give up part of
    the chart (its first series alone, another kind, a narrower selection) are not accepted: None, so the reply says
    it can't."""

    def whole(candidate: VisualSpec) -> bool:
        return candidate.kind == kind and keeps_data(spec, candidate, datasets) and check(candidate)

    return first_valid(spec.model_copy(update={"kind": kind}), datasets, whole)


def merge_planned(
    old: VisualSpec,
    planned: VisualSpec,
    utterance: str,
    datasets: Mapping[str, TypedDataset],
    check: Callable[[VisualSpec], bool],
) -> VisualSpec | None:
    """The planner's choice for an edit the rules can't read, together with what was on screen: a small model asked
    to "show it as a table" may pick one of the two tables behind the chart, or to "add EBITDA" may leave the revenue
    out. Unless the user asked to drop something (only, instead, remove …), the tables, series and selections of the
    visual on screen stay. If that doesn't build, the visual in the planner's kind with all its data; else None."""
    if _DROPS.search(utterance):
        return planned
    if keeps_data(old, planned, datasets):
        return planned
    merged = planned.model_copy(
        update={
            "datasets": _union(old.datasets, planned.datasets)[:MAX_DATASETS],
            "series": _union(old.series, planned.series)[:MAX_SERIES],
            "periods": _selection(old.periods, planned.periods),
            "categories": _selection(old.categories, planned.categories),
            "title": planned.title or old.title,
        }
    )
    found = first_valid(merged, datasets, lambda s: keeps_data(old, s, datasets) and check(s))
    return found or change_kind(old, planned.kind, datasets, check)


def _union(first: list[str], second: list[str]) -> list[str]:
    return list(dict.fromkeys([*first, *second]))


def _selection(old: list[str], new: list[str]) -> list[str]:
    """Periods or categories to keep: none (everything) when the screen showed everything; else both."""
    if not old:
        return []
    return _union(old, new) if new else list(old)


def edit_language(utterance: str, chat_language: str | None, visual_language: str) -> str:
    """The language a rebuilt visual is drawn in: the one the user asked for in the edit ("make it a table, in Hindi");
    else the chat's response language, so a Hindi edit said in an English chat doesn't turn its labels and units into
    Hindi ("₹ करोड़"); else the visual's own."""
    return asked_language(utterance) or chat_language or visual_language
