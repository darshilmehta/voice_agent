"""The canvas in the conversation (docs/DESIGN.md §12.1, workstreams 5 and 8): what is on screen as context for the
router and the answer, edits of the canvas said in words (English, Hindi, Hinglish), and a turn's visual prepared
beside its answer. Pure functions, except ``TurnVisual`` (a task) and the per-chat registry of visuals in progress.

    visual_want(utterance, rewritten)  "requested" ("show me", "chart", "dikhao", "दिखाओ") / "suggest" (trends,
                                       comparisons, breakdowns, two or more periods) / "none": ``route.visual``
    screen_lines(panels, utterance)    the canvas as a few short lines (kind, title, x and series labels, highlight;
                                       no numbers) for the router, plus what an utterance points at ("the second bar":
                                       Engineered Plastics; "the dip": Q2 FY23)
    parse_edit(utterance)              the keyword fast path for canvas edits: a kind change ("make that a bar
                                       chart", "show it as a table", "isko bar chart mein dikhao"), remove ("remove the
                                       pie", "हटा दो"), pin / unpin, periods ("put FY23 next to it", "FY23 भी जोड़ो",
                                       "only FY24"), clear; an edit verb aimed at a chart that the grammar can't read
                                       ("add profit to that chart") is a "model" edit: the planner rebuilds the visual
    resolve_target(utterance, panels)  which visual "that / it / the pie / the second chart / the revenue chart"
                                       means: a named kind, an ordinal, title words, else the one the conversation
                                       touched last (the most recently added or edited)
    TurnVisual                         ``CanvasService.prepare_visual`` as its own task, its events queued for the
                                       turn's transport; cancellable; the outcome written onto the turn's message

"Focused" means most recently added or edited: the frontend doesn't tell the backend which panel the user looks at.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ...domain.canvas import CanvasEvent, Visual, VisualEvent, VisualKind
from ...domain.conversation import VisualWant
from ..language import wordset
from .parsing import find_period
from .planner import visual_intent

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ text helpers

_LETTER = "A-Za-z0-9ऀ-ॣ०-ॿ"
_PUNCT = re.compile(f"[^{_LETTER}\\s'-]|[।॥]")


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.casefold()).split())


def _has(pattern: str, text: str) -> re.Match[str] | None:
    """``pattern`` as whole words of normalized ``text`` (\\b fails after Hindi vowel signs)."""
    return re.search(f"(?<![{_LETTER}])(?:{pattern})(?![{_LETTER}])", text)


def _words(text: str) -> list[str]:
    return _norm(text).split()


# ------------------------------------------------------------------ route.visual


def visual_want(*texts: str | None, answer: str | None = None) -> VisualWant:
    """``route.visual`` from the user's words (the utterance and its standalone form) and, once there is one, the
    answer (three or more figures make a chart worth suggesting)."""
    intents = [visual_intent(t) for t in texts if t]
    if "requested" in intents:
        return "requested"
    if "suggested" in intents or (answer is not None and visual_intent("", answer) == "suggested"):
        return "suggest"
    return "none"


# ------------------------------------------------------------------ kinds, ordinals, references

# (pattern, kind), most specific first. A kind word alone ("bar", "line") only counts inside an edit.
_KIND_WORDS: tuple[tuple[str, VisualKind], ...] = (
    (r"pie(?:\s+chart)?|donut(?:\s+chart)?|doughnut(?:\s+chart)?|पाई(?:\s+चार्ट)?|गोल\s+चार्ट", "donut"),
    (r"stacked(?:\s+bars?)?(?:\s+chart)?", "stacked_bar"),
    (r"grouped\s+bars?(?:\s+chart)?|side\s+by\s+side\s+bars?", "grouped_bar"),
    (r"waterfall(?:\s+chart)?|bridge\s+chart", "waterfall"),
    (r"bar\s+(?:chart|graph)|bars|column\s+chart|bar|बार\s*(?:चार्ट|ग्राफ़|ग्राफ)?", "bar"),
    (r"line\s+(?:chart|graph)|trend\s+line|line|लाइन\s*(?:चार्ट|ग्राफ़|ग्राफ)?|रेखा\s*(?:चार्ट|ग्राफ़|ग्राफ)", "line"),
    (r"table|tabular|टेबल|तालिका|सारणी", "table"),
    (r"timeline|टाइमलाइन", "timeline"),
    (r"kpi|kpis|tiles|cards|टाइल्स?", "kpi"),
)
_KIND_FAMILY: dict[str, frozenset[str]] = {
    "donut": frozenset({"donut"}),
    "bar": frozenset({"bar", "grouped_bar", "stacked_bar", "comparison"}),
    "grouped_bar": frozenset({"grouped_bar", "bar"}),
    "stacked_bar": frozenset({"stacked_bar", "bar"}),
    "waterfall": frozenset({"waterfall"}),
    "line": frozenset({"line"}),
    "table": frozenset({"table"}),
    "timeline": frozenset({"timeline"}),
    "kpi": frozenset({"kpi", "comparison"}),
}
_ORDINALS: tuple[tuple[str, int], ...] = (
    (r"first|1st|पहला|पहली|पहले|pehla|pehli|pehle|pahla|pahli", 0),
    (r"second|2nd|दूसरा|दूसरी|दूसरे|doosra|doosri|dusra|dusri|doosre|dusre", 1),
    (r"third|3rd|तीसरा|तीसरी|तीसरे|teesra|teesri|tisra|tisri", 2),
    (r"fourth|4th|चौथा|चौथी|चौथे|chautha|chauthi", 3),
    (r"fifth|5th|पाँचवाँ|पांचवां|पाँचवीं|panchva|paanchva", 4),
    (r"last|final|आखिरी|आख़िरी|अंतिम|aakhri|akhri|aakhiri", -1),
)
# A whole visual ("the second chart") and a part of one ("the second bar").
_PANEL_NOUNS = r"charts?|graphs?|visuals?|panels?|plots?|figures?|diagrams?|चार्ट|ग्राफ़|ग्राफ|आरेख"
_CHART_NOUNS = f"{_PANEL_NOUNS}|pies?|donuts?|slices?|tiles?|screen|canvas|स्क्रीन"
_POINT_NOUNS = r"bars?|points?|slices?|columns?|tiles?|dots?|steps?|बार|कॉलम|स्लाइस|wala|wali|wale|वाला|वाली|वाले"
_DEIXIS = (
    r"it|that|this|these|those|them|isko|isse|ise|isey|is|ye|yeh|usko|use|usse|wo|woh|vo|"
    r"इसे|इसको|इस|यह|ये|उसे|उसको|उस|वो|वह"
)


def kind_mentions(text: str) -> list[tuple[int, VisualKind]]:
    """The chart kinds named in ``text`` (position in the normalized text, kind), in order."""
    n = _norm(text)
    found: list[tuple[int, int, VisualKind]] = []
    taken: list[tuple[int, int]] = []
    for pattern, kind in _KIND_WORDS:
        for m in re.finditer(f"(?<![{_LETTER}])(?:{pattern})(?![{_LETTER}])", n):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.end(), kind))
    return [(s, k) for s, _, k in sorted(found)]


def ordinal_before(text: str, nouns: str) -> int | None:
    """The 0-based index an ordinal right before one of ``nouns`` says ("the second chart", "दूसरा बार", "the last
    bar": -1), or None."""
    n = _norm(text)
    for pattern, index in _ORDINALS:
        if _has(f"(?:{pattern})\\s+(?:[{_LETTER}]+\\s+)?(?:{nouns})", n):
            return index
    return None


# ------------------------------------------------------------------ the canvas as context


def _x_labels(v: Visual) -> list[str]:
    if v.kind in ("kpi", "comparison") and v.tiles:
        return [t.label for t in v.tiles]
    if v.kind == "timeline":
        return [e.label for e in v.events]
    return [r.x for r in v.rows]


def _short_list(items: Sequence[str], limit: int = 8) -> str:
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:3])}, … {', '.join(items[-2:])} ({len(items)} in all)"


def describe(v: Visual) -> str:
    """One visual in a line, without numbers: its kind and title, the x items, the series and units, the highlight."""
    parts = [f'{v.kind.replace("_", " ")} "{v.title}"']
    xs = _x_labels(v)
    if xs:
        what = "tiles" if v.kind in ("kpi", "comparison") and v.tiles else "events" if v.kind == "timeline" else "x"
        parts.append(f"{what}: {_short_list(xs)}")
    series = [f"{s.label}{f' ({s.unit.label})' if s.unit and s.unit.label else ''}" for s in v.series]
    if series and v.kind not in ("kpi", "timeline"):
        parts.append(f"series: {_short_list(series, 4)}")
    if v.highlight is not None and v.highlight.x:
        note = f" ({v.highlight.note})" if v.highlight.note else ""
        parts.append(f"highlighted: {', '.join(v.highlight.x)}{note}")
    return " — ".join(parts)


def recent(panels: Sequence[Visual]) -> list[Visual]:
    """Panels, the one the conversation touched last first."""
    return sorted(panels, key=lambda v: (v.updated_at, v.created_at), reverse=True)


MAX_SCREEN_PANELS = 3


def screen_lines(panels: Sequence[Visual], utterance: str = "") -> list[str]:
    """What is on screen, for the router (and the answer of a question about it): the latest panels (at most
    MAX_SCREEN_PANELS) in screen order, numbered as the user sees them, the latest marked; then what the utterance
    points at, when code can tell ("the second bar", "the dip")."""
    if not panels:
        return []
    ordered = sorted(panels, key=lambda v: v.position)
    shown = {v.id for v in recent(panels)[:MAX_SCREEN_PANELS]}
    latest = recent(panels)[0].id
    lines = [
        f"{n}. {describe(v)}{' (latest)' if v.id == latest else ''}"
        for n, v in enumerate(ordered, start=1)
        if v.id in shown
    ]
    if utterance and refers_to_screen(utterance):
        target = resolve_target(utterance, panels, infer_kind=True)
        point = point_reference(utterance, target) if target is not None else None
        if point:
            lines.append(f"The user points at {point}.")
    return lines


# Points of a series that code can find: the dip (the largest fall from the previous point, else the lowest), the
# peak / highest, the lowest.
_DIP = r"dip|dipped|drop|dropped|fall|fell|decline|declined|slump|girawat|गिरावट|गिरा"
_PEAK = (
    r"peak|peaked|spike|jump|jumped|highest|biggest|largest|maximum|sabse\s+zyada|sabse\s+jyada|"
    r"सबसे\s+(?:ज़्यादा|ज्यादा|अधिक|बड़ा|ऊँचा)"
)
_LOW = r"lowest|smallest|minimum|sabse\s+kam|सबसे\s+कम"
# A point of a chart named without naming the chart: "that dip", "the spike", "dipped there". ("The highest revenue"
# is a document question.)
_SHAPE = r"dip|drop|fall|peak|spike|jump|गिरावट|girawat"
_SHAPE_REFERENCE = f"(?:that|the|this|wo|woh|वो|वह|यह|इस)\\s+(?:{_SHAPE})|(?:{_DIP}|{_PEAK})\\s+(?:there|wahan|वहाँ|वहां)"


def refers_to_screen(text: str) -> bool:
    """The utterance is about a visual on screen: it names a chart ("the chart", "this graph", "the pie", "स्क्रीन
    पर", "is graph mein"), a part of one by its place ("the second bar", "दूसरा बार") or a shape of one ("that dip",
    "why did it dip there"). A table is a document's table unless it is "on screen"."""
    n = _norm(text)
    if _has(r"on\s+(?:the\s+)?screen|स्क्रीन\s+पर|screen\s+(?:par|pe)", n):
        return True
    if _has(f"(?:the|this|that|these|those|is|iss|us|ye|yeh|इस|उस|यह)\\s+(?:[{_LETTER}]+\\s+)?(?:{_CHART_NOUNS})", n):
        return True
    if ordinal_before(n, _POINT_NOUNS) is not None:
        return True
    return bool(_has(_SHAPE_REFERENCE, n))


def _first_series_points(v: Visual) -> list[tuple[str, float]]:
    if not v.series or not v.rows:
        return []
    key = v.series[0].key
    return [(r.x, r.values[key]) for r in v.rows if r.values.get(key) is not None]  # type: ignore[misc]


def point_reference(text: str, v: Visual) -> str | None:
    """What a question points at on visual ``v``: an ordinal point ("the second bar" → "the second bar of …:
    Engineered Plastics"), the dip, the peak or the lowest point (found by code in the first series), or the
    highlighted point ("there", "that point"). None when it points at nothing code can tell."""
    n = _norm(text)
    title = f'"{v.title}"'
    xs = _x_labels(v)
    index = ordinal_before(n, _POINT_NOUNS)
    if index is not None and xs and -len(xs) <= index < len(xs):
        ordinal = next(p.split("|")[0] for p, i in _ORDINALS if i == index)
        return f"the {ordinal} item of {title}: {xs[index]}"
    points = _first_series_points(v)
    if points:
        if _has(_DIP, n):
            falls = [(points[i - 1][1] - points[i][1], points[i][0]) for i in range(1, len(points))]
            fall, x = max(falls, default=(0.0, ""))
            if fall > 0:
                return f"the dip in {title}: {x} ({v.series[0].label} fell from the point before)"
            return f"the lowest point of {title}: {min(points, key=lambda p: p[1])[0]}"
        if _has(_PEAK, n):
            return f"the highest point of {title}: {max(points, key=lambda p: p[1])[0]}"
        if _has(_LOW, n):
            return f"the lowest point of {title}: {min(points, key=lambda p: p[1])[0]}"
    if v.highlight is not None and v.highlight.x and _has(r"there|that\s+point|that\s+one|wahan|वहाँ|वहां", n):
        return f"the highlighted point of {title}: {', '.join(v.highlight.x)}"
    return None


# ------------------------------------------------------------------ which visual

# Words that never tell visuals apart.
_TARGET_STOP = wordset(
    """
    the a an it that this these those them chart charts graph graphs visual panel plot figure one make turn change
    switch convert show display put remove delete drop hide pin unpin please can you could would as into to in of on
    from with and instead now too also again same screen canvas next it's let's
    """
)


def resolve_target(
    text: str, panels: Sequence[Visual], *, kind_hint: str | None = None, infer_kind: bool = False
) -> Visual | None:
    """The visual ``text`` means. A named kind ("the pie", "the bar chart"; ``kind_hint``, or with ``infer_kind`` the
    first kind named in ``text`` that is on screen: for questions, not for edits, where "a bar chart" is what to
    make), an ordinal ("the second chart": screen order), words of a title or series ("the revenue chart"), else the
    one the conversation touched last ("that", "it", "isko", "इसे"). None without panels."""
    if not panels:
        return None
    ordered = sorted(panels, key=lambda v: v.position)
    latest_first = recent(panels)
    n = _norm(text)
    if kind_hint is None and infer_kind:
        kinds = {v.kind for v in panels}
        kind_hint = next((k for _, k in kind_mentions(n) if _KIND_FAMILY.get(k, frozenset({k})) & kinds), None)
    candidates = latest_first
    if kind_hint is not None:
        family = _KIND_FAMILY.get(kind_hint, frozenset({kind_hint}))
        of_kind = [v for v in latest_first if v.kind in family]
        if of_kind:
            candidates = of_kind
    index = ordinal_before(n, _PANEL_NOUNS)
    if index is not None:
        pool = [v for v in ordered if v in candidates] if kind_hint else ordered
        if -len(pool) <= index < len(pool):
            return pool[index]
    words = {w for w in n.split() if len(w) > 2 and w not in _TARGET_STOP}
    if words and len(candidates) > 1:

        def overlap(v: Visual) -> int:
            names = _norm(" ".join([v.title, *(s.label for s in v.series)])).split()
            return len(words & set(names))

        best = max(candidates, key=overlap)  # max keeps the first (the latest) on ties
        if overlap(best):
            return best
    return candidates[0]


# ------------------------------------------------------------------ edits


EditOp = Literal["kind", "remove", "pin", "unpin", "periods", "only", "clear", "model"]


@dataclass(frozen=True, slots=True)
class CanvasEdit:
    """An edit of the canvas said in words. ``target_kind``: a kind naming the visual to edit ("the pie"); ``kind``:
    the kind to show it as; ``periods``: to add (``periods``) or to keep (``only``)."""

    op: EditOp
    kind: VisualKind | None = None
    target_kind: VisualKind | None = None
    periods: tuple[str, ...] = ()

    def record(self) -> dict[str, object]:
        out: dict[str, object] = {"op": self.op}
        if self.kind:
            out["kind"] = self.kind
        if self.periods:
            out["periods"] = list(self.periods)
        return out


_REMOVE = (
    r"remove|delete|drop|hide|close|discard|get\s+rid\s+of|take\s+(?:it\s+|that\s+|this\s+)?(?:away|off|down)|"
    r"हटा|हटाओ|हटाइए|हटाएं|हटाएँ|हटा\s+दो|हटा\s+दीजिए|मिटा|मिटाओ|मिटा\s+दो|"
    r"hata|hatao|hataiye|hatayein|hata\s+do|hata\s+dijiye|mita|mitao|mita\s+do"
)
_CLEAR = (
    r"clear\s+(?:the\s+)?(?:canvas|screen|board|charts?|everything|all)|"
    r"(?:remove|delete|hide|close)\s+(?:all|every|everything)(?:\s+(?:the\s+)?(?:charts?|graphs?|visuals?))?|"
    r"(?:sab|sabhi|saare|sare|सब|सभी|सारे)\s+(?:\w+\s+)?(?:hata|hatao|hata\s+do|हटा|हटाओ|हटा\s+दो|मिटा\s+दो)"
)
_UNPIN = r"unpin|un-pin|अनपिन|pin\s+(?:hata|hatao|hata\s+do|nikal\s+do)|पिन\s+(?:हटा|हटाओ|हटा\s+दो)"
_PIN = r"pin|पिन|keep\s+(?:it|that|this)\s+(?:on\s+(?:the\s+)?screen|there)"
_ADD = (
    r"add|put|include|show|compare|plot|bring\s+in|"
    r"jodo|jod\s+do|add\s+karo|add\s+kar\s+do|dikhao|daalo|dalo|जोड़ो|जोड़\s+दो|जोड़ें|दिखाओ|डालो"
)
_ALONG = r"next\s+to|alongside|beside|too|also|as\s+well|with|against|bhi|भी|saath|साथ"
_ONLY = r"only|just|sirf|keval|केवल|सिर्फ़|सिर्फ"
# Words an edit may consist of besides kinds, targets and periods: verbs, references, prepositions, politeness.
_EDIT_WORDS = wordset(
    """
    make turn change switch convert redo redraw show display put draw plot give use see view let's lets let us me it
    that this them these those the same chart charts graph graphs visual panel one figure previous last instead
    please now can could would you will as into to in a an ok okay and so rather kindly version form format type
    style again just only like rather than of
    isko ise isey isse is ye yeh usko use usse wo woh vo ko mein me main men ek ki ke ka jagah ab zara thoda
    dikhao dikhaiye dikhaye dikha do dijiye banao banaiye bana badlo badal dalo karo kar kariye kijiye chahiye
    इसे इसको इस यह ये उसे उसको उस वो वह को में एक की के का जगह अब ज़रा थोड़ा
    दिखाओ दिखाइए दिखाएं दिखाएँ दिखा दो दीजिए बनाओ बनाइए बना बदलो बदल करो कर कीजिए करें चाहिए चार्ट ग्राफ़ ग्राफ
    add include compare next alongside beside too also well with against bhi saath jodo jod daalo sirf keval
    भी साथ जोड़ो जोड़ जोड़ें डालो सिर्फ़ सिर्फ केवल
    """
)
# Question words: a question is never an edit ("what caused that drop", spoken without a question mark).
_QUESTION = r"what|why|how|which|who|when|where|kyun|kyon|kaise|kaun|kitna|kitni|kitne|क्यों|कैसे|कौन|कितना|कितनी|कितने"
# Something in a kind change that says "this one, shown another way": a reference, a preposition of form, a verb of
# change, "instead".
_CHANGE = (
    f"{_DEIXIS}|as|into|instead|make|turn|change|switch|convert|redo|redraw|mein|me|में|ki\\s+jagah|की\\s+जगह|"
    r"badlo|badal|बदलो|बदल|banao|बनाओ"
)
# An edit verb aimed at a visual, for edits the grammar can't read ("add profit to that chart", "show EBITDA
# instead on it"): the planner rebuilds the visual.
_EDIT_VERB = r"make|turn|change|switch|convert|add|put|include|replace|swap|show|plot|highlight|sort|only|just"
_AT_VISUAL = f"(?:to|on|in|into|from|of)\\s+(?:{_DEIXIS}|(?:the|that|this)\\s+(?:\\w+\\s+)?(?:{_CHART_NOUNS}))"


def _periods(text: str) -> tuple[tuple[str, ...], str]:
    """The periods named in ``text`` (canonical labels) and the text without them."""
    found: list[str] = []
    rest = text
    while (hit := find_period(rest)) is not None:
        period, rest = hit
        found.append(period.label)
    return tuple(dict.fromkeys(found)), rest


def _only_edit_words(words: Sequence[str]) -> bool:
    return all(w in _EDIT_WORDS for w in words)


def parse_edit(utterance: str) -> CanvasEdit | None:
    """The canvas edit an utterance says, when the keyword grammar can read it (module docstring), else None. Only
    for a chat whose canvas has visuals (the caller checks); a question is never an edit."""
    text = utterance.strip()
    if not text or text.endswith("?"):
        return None
    n = _norm(text)
    ws = n.split()
    if not ws or len(ws) > 14 or _has(_QUESTION, n):
        return None
    if _has(_CLEAR, n):
        return CanvasEdit("clear")
    kinds = kind_mentions(n)
    target_kind = kinds[0][1] if len(kinds) > 1 else None
    if _has(_UNPIN, n):
        return CanvasEdit("unpin", target_kind=kinds[0][1] if kinds else None)
    if _has(_REMOVE, n) and len(ws) <= 9:
        return CanvasEdit("remove", target_kind=kinds[-1][1] if kinds else None)
    if _has(_PIN, n) and len(ws) <= 9:
        return CanvasEdit("pin", target_kind=kinds[-1][1] if kinds else None)
    periods, rest = _periods(text)
    rest_words = _norm(rest).split()
    if periods and _only_edit_words(rest_words):
        if _has(_ONLY, n):
            return CanvasEdit("only", periods=periods)
        if _has(_ADD, n) or _has(_ALONG, n):
            return CanvasEdit("periods", periods=periods)
    if kinds:
        # Everything but the kinds must be edit vocabulary ("show revenue as a bar chart" names a metric: a new
        # visual, not an edit), and something must say "change this one" ("show the table" may mean a document's).
        stripped = n
        for pattern, _ in _KIND_WORDS:
            stripped = re.sub(f"(?<![{_LETTER}])(?:{pattern})(?![{_LETTER}])", " ", stripped)
        rest_ws = stripped.split()
        if _only_edit_words(rest_ws) and (len(kinds) > 1 or _has(_CHANGE, stripped)):
            return CanvasEdit("kind", kind=kinds[-1][1], target_kind=target_kind)
    if _has(f"(?:{_EDIT_VERB})", n) and _has(_AT_VISUAL, n):
        return CanvasEdit("model")
    return None


# ------------------------------------------------------------------ outcomes and fixed replies

EditOutcome = Literal["done", "nothing", "already", "same", "cannot", "failed"]


@dataclass(frozen=True, slots=True)
class EditResult:
    """What an edit did: ``done``; ``nothing`` on screen; ``already`` there (a period); ``same`` kind already;
    ``cannot`` (the visual can't be shown that way); ``failed`` (the planner found no way). ``visual_id``: the visual
    edited (removed, pinned, rebuilt)."""

    outcome: EditOutcome
    op: EditOp
    visual_id: str | None = None
    source: Literal["rules", "model"] = "rules"
    detail: str | None = None

    def record(self, edit: CanvasEdit) -> dict[str, object]:
        out = {**edit.record(), "outcome": self.outcome, "source": self.source}
        if self.visual_id:
            out["visual_id"] = self.visual_id
        if self.detail:
            out["detail"] = self.detail
        return out


EDIT_REPLIES: dict[EditOutcome, dict[str, str]] = {
    "done": {"en": "Done.", "hi": "हो गया।"},
    "nothing": {"en": "There's no chart on screen yet.", "hi": "अभी स्क्रीन पर कोई चार्ट नहीं है।"},
    "already": {"en": "That's already on the chart.", "hi": "यह पहले से चार्ट पर है।"},
    "same": {"en": "It's already shown that way.", "hi": "यह पहले से ऐसे ही दिखाया गया है।"},
    "cannot": {"en": "I can't show that chart that way.", "hi": "यह चार्ट उस तरह नहीं दिखा सकता।"},
    "failed": {"en": "I couldn't change that chart.", "hi": "मैं वह चार्ट नहीं बदल सका।"},
}


def edit_reply(outcome: EditOutcome, language: str) -> str:
    texts = EDIT_REPLIES[outcome]
    return texts.get(language, texts["en"])


# ------------------------------------------------------------------ a turn's visual

VisualStatus = Literal["preparing", "ready", "failed", "cancelled", "none"]
CanvasEvents = VisualEvent | CanvasEvent


@dataclass(eq=False)
class TurnVisual:
    """A turn's visual, prepared beside its answer: ``events`` (``CanvasService.prepare_visual``) consumed by its own
    task, every event queued for the turn's transport. ``announce``: send ``preparing`` (a skeleton) and ``failed``
    (a quiet note); off for a suggested visual, which appears only if it works. Cancelling it (a newer turn that
    needs the model, the stop button, the session ending, a stream that can't wait any longer) ends the queue with
    ``failed {detail}`` when a skeleton was shown, so the client drops it."""

    chat_id: str
    visual_id: str
    want: VisualWant
    announce: bool
    status: VisualStatus = "preparing"
    detail: str | None = None
    queue: asyncio.Queue[CanvasEvents | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    _callbacks: list[Callable[[TurnVisual], Awaitable[None]]] = field(default_factory=list)
    _settled: bool = False
    _cancel_detail: str = "cancelled"
    _ended: bool = False

    def start(self, events: AsyncIterator[CanvasEvents]) -> TurnVisual:
        if self.announce:
            self.queue.put_nowait(VisualEvent(phase="preparing", visual_id=self.visual_id))
        self.task = asyncio.ensure_future(self._run(events))
        self.task.set_name(f"visual-{self.visual_id}")
        self.task.add_done_callback(self._finished)  # also when cancelled before it ever ran
        _registry()[self.chat_id] = self
        return self

    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()

    async def _run(self, events: AsyncIterator[CanvasEvents]) -> None:
        try:
            async with contextlib.aclosing(events):  # type: ignore[type-var]
                async for event in events:
                    if isinstance(event, VisualEvent):
                        if event.phase == "preparing":
                            continue  # announced at the start
                        self.status = event.phase
                        self.detail = event.detail
                        if event.phase == "failed" and not self.announce:
                            continue
                    self.queue.put_nowait(event)
            if self.status == "preparing":  # nothing to show after all (no table, nothing worth it)
                self.status, self.detail = "none", self.detail or "no visual"
                if self.announce:
                    self.queue.put_nowait(VisualEvent(phase="failed", visual_id=self.visual_id, detail="no visual"))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # prepare_visual doesn't raise; a bug must not reach the answer
            log.exception("canvas: the visual of a turn in chat %s failed", self.chat_id)
            self.status, self.detail = "failed", f"{type(e).__name__}: {e}"[:300]
            if self.announce:
                self.queue.put_nowait(VisualEvent(phase="failed", visual_id=self.visual_id, detail="failed"))

    def _finished(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() and self.status == "preparing":
            self.status, self.detail = "cancelled", self._cancel_detail
            if self.announce:
                self.queue.put_nowait(VisualEvent(phase="failed", visual_id=self.visual_id, detail=self._cancel_detail))
        self.queue.put_nowait(None)
        if _registry().get(self.chat_id) is self:
            del _registry()[self.chat_id]
        self._settled = True
        for callback in self._callbacks:
            _spawn(callback(self))

    def when_settled(self, callback: Callable[[TurnVisual], Awaitable[None]]) -> None:
        """Run ``callback(self)`` (as its own task) once the visual is ready, failed or cancelled; at once if it is."""
        if self._settled:
            _spawn(callback(self))
        else:
            self._callbacks.append(callback)

    def cancel(self, detail: str = "cancelled") -> None:
        """Stop preparing it (a no-op once it is done)."""
        if self.task is not None and not self.task.done():
            self._cancel_detail = detail
            self.task.cancel()

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait until it is done (True), at most ``timeout`` seconds (False)."""
        if self.task is None:
            return True
        done, _ = await asyncio.wait({self.task}, timeout=timeout)
        return bool(done)

    def pending(self) -> list[CanvasEvents]:
        """The events queued so far, without waiting."""
        out: list[CanvasEvents] = []
        while not self._ended:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                self._ended = True
                break
            out.append(item)
        return out

    async def events(self, timeout: float | None = None) -> AsyncIterator[CanvasEvents]:
        """The events still to come, until it is done; past ``timeout`` seconds it is cancelled (``failed`` "timed
        out") and what that leaves is yielded."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while not self._ended:
            left = None if deadline is None else max(0.0, deadline - loop.time())
            try:
                item = await asyncio.wait_for(self.queue.get(), left)
            except TimeoutError:
                self.cancel("timed out")
                deadline = None  # the cancellation's own events follow at once
                continue
            if item is None:
                self._ended = True
                return
            yield item

    def record(self) -> dict[str, object]:
        """``route`` fields of the turn's message, for what the user saw: the visual (``visual_id``, ready), or the
        failure note of a requested one (``visual_status`` failed / cancelled / none, ``visual_detail``). Nothing
        while it is being prepared, nor for a suggested one that didn't work out (nothing was shown)."""
        if self.status == "ready":
            return {"visual_status": "ready", "visual_id": self.visual_id}
        if self.status == "preparing" or not self.announce:
            return {}
        return {"visual_status": self.status, "visual_detail": self.detail}


# The visual each chat is preparing (one per chat: a newer one replaces an older one in the registry, which the next
# turn cancels or waits for). Per event loop, as the other per-chat registries.
_VISUALS: dict[int, tuple[asyncio.AbstractEventLoop, dict[str, TurnVisual]]] = {}


def _registry() -> dict[str, TurnVisual]:
    from ..conversation import loop_local

    return loop_local(_VISUALS, dict)


def visual_in_progress(chat_id: str) -> TurnVisual | None:
    visual = _registry().get(chat_id)
    return visual if visual is not None and not visual.done else None


_CALLBACKS: set[asyncio.Task[object]] = set()


def _spawn(coro: Awaitable[object]) -> None:
    async def run() -> None:
        try:
            await coro
        except Exception:
            log.exception("canvas: a callback of a turn's visual failed")

    task = asyncio.ensure_future(run())
    _CALLBACKS.add(task)
    task.add_done_callback(_CALLBACKS.discard)


async def settled_callbacks() -> None:
    """Wait for the callbacks of settled visuals (tests, shutdown)."""
    loop = asyncio.get_running_loop()
    while pending := [t for t in _CALLBACKS if t.get_loop() is loop and not t.done()]:
        await asyncio.wait(pending)
