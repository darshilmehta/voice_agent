# ruff: noqa: RUF001, RUF002  (documents print en dashes, minus, times and divide signs: used on purpose)
"""Numbers, units and periods as documents print them (docs/DESIGN.md §12.1, workstream 1). Pure functions.

- ``parse_quantity("₹ 4,21,000 crore")`` → 421000 (currency INR, scale crore). Indian (4,21,000) and Western (421,000)
  grouping, negatives in parentheses or with a minus/en dash, ``+`` signs, footnote markers ("1,234*", "12.5%¹"),
  currency prefixes (₹, Rs., INR, $, US$, €, £), scale words (crore, lakh, mn, bn, करोड़, लाख), %, x, bps, pts and a
  short unit word ("2,10,000 tpa", "5 meetings"). Devanagari digits are read too. "–", "—", "n.a.", "NA", "nil"
  are null (``NULL``); anything else that isn't one quantity is None (text).
- ``detect_unit("Revenue (₹ in crore)")`` → Unit(currency, INR, crore, "₹ crore") from captions, headers and notes.
- ``find_period("Revenue FY24")`` → (Period FY24, "Revenue"): Indian fiscal years (FY24, FY 2023-24, 2023-24,
  वित्त वर्ष 2023-24), quarters (Q3 FY24, 3QFY24), halves (H1 FY24), nine months (9M FY24), calendar years, dates
  (31 Mar 2024, March 31, 2024, 31.03.2024, 20 सितंबर 2024) and months (Mar 2024). Fiscal years end in March.

A value keeps the scale it is printed in: the contract's values are "numbers in the unit's scale".
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

from ...domain.canvas import Scale, Unit

QuantityKind = Literal["currency", "percent", "ratio", "bps", "pp", "number"]
PeriodKind = Literal["fy", "quarter", "half", "nine_months", "year", "date", "month"]

CURRENCY_SYMBOLS: Final = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
SCALE_HI: Final = {"crore": "करोड़", "lakh": "लाख", "million": "मिलियन", "billion": "बिलियन", "thousand": "हज़ार"}


class _Null:
    """A cell printed as "not available" ("–", "n.a."): a null value, unlike text."""

    _instance: _Null | None = None

    def __new__(cls) -> _Null:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NULL"

    def __bool__(self) -> bool:
        return False


NULL: Final = _Null()


@dataclass(frozen=True, slots=True)
class Quantity:
    value: float
    kind: QuantityKind
    currency: str | None = None
    scale: Scale | None = None
    unit_word: str | None = None  # "tpa", "seats", "days", "meetings"
    footnote: str | None = None
    signed: bool = False  # printed with an explicit + or - (a change, not a level)
    decimals: int = 0  # digits after the decimal point as printed

    def unit(self) -> Unit | None:
        """The unit this cell states on its own (None for a bare number)."""
        if self.kind == "percent":
            return PERCENT
        if self.kind == "ratio":
            return RATIO
        if self.kind == "bps":
            return Unit(kind="none", label="bps")
        if self.kind == "pp":
            return Unit(kind="none", label="pts")
        if self.kind == "currency":
            return currency_unit(self.currency or "INR", self.scale)
        if self.scale is not None:
            return Unit(
                kind="none", scale=self.scale, label=self.scale + (f" {self.unit_word}" if self.unit_word else "")
            )
        if self.unit_word:
            kind = "duration" if _DURATION_WORD.fullmatch(self.unit_word) else "count"
            return Unit(kind=kind, label=self.unit_word)
        return None


PERCENT: Final = Unit(kind="percent", label="%")
RATIO: Final = Unit(kind="ratio", label="x")
COUNT: Final = Unit(kind="count", label="")


def currency_unit(currency: str, scale: Scale | None) -> Unit:
    symbol = CURRENCY_SYMBOLS.get(currency, currency)
    return Unit(kind="currency", currency=currency, scale=scale, label=f"{symbol} {scale}" if scale else symbol)


def localized_label(unit: Unit | None, language: str) -> str:
    """The unit's label in the conversation's language (Hindi: "₹ करोड़")."""
    if unit is None:
        return ""
    if language != "hi":
        return unit.label
    if unit.kind == "currency" and unit.scale:
        return f"{CURRENCY_SYMBOLS.get(unit.currency or '', unit.currency or '')} {SCALE_HI[unit.scale]}".strip()
    return unit.label


# ------------------------------------------------------------------ text clean-up

_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_DIGITS = str.maketrans("०१२३४५६७८९" + _SUPERSCRIPTS, "0123456789" * 2)
_SPACE = re.compile(r"[\s    ​﻿]+")
_FOOTNOTE_TAIL = re.compile(r"(?:\s*(?:[⁰¹²³⁴⁵⁶⁷⁸⁹]+|[*†‡§¶#]+|\[\w{1,2}\]|\^\w{1,2}))+$")
_FOOTNOTE_HEAD = re.compile(r"^(?:[*†‡§¶#]+)\s*")
_NULLS = frozenset(
    {"", "-", "--", "---", "–", "—", "―", "−", "n.a", "na", "n/a", "nil", "nm", "n.m", "none", "...", "…",
     "not applicable", "not available", "lागू नहीं", "लागू नहीं", "शून्य"}
)  # fmt: skip


def normalize_space(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def split_footnote(text: str) -> tuple[str, str | None]:
    """("12.5%¹", None) → ("12.5%", "1"). Superscripts are read before any digit normalisation."""
    text = normalize_space(text)
    footnote = None
    m = _FOOTNOTE_TAIL.search(text)
    if m and m.start() > 0:
        footnote = m.group(0).strip().translate(_DIGITS).strip("[]^")
        text = text[: m.start()].rstrip()
    m = _FOOTNOTE_HEAD.match(text)
    if m and m.end() < len(text):
        footnote = footnote or m.group(0).strip()
        text = text[m.end() :]
    return text, footnote


def is_null(text: str) -> bool:
    core, _ = split_footnote(text)
    return core.casefold().rstrip(".").strip() in _NULLS


# ------------------------------------------------------------------ quantities

_CUR = r"(?:₹|rs\.?|inr|rupees?|रु\.?|रुपये|us\$|usd|\$|eur|€|£|gbp)"
_CURRENCY_CODE = (
    (re.compile(r"^(?:₹|rs\.?|inr|rupees?|रु\.?|रुपये)$", re.I), "INR"),
    (re.compile(r"^(?:us\$|usd|\$)$", re.I), "USD"),
    (re.compile(r"^(?:eur|€)$", re.I), "EUR"),
    (re.compile(r"^(?:£|gbp)$", re.I), "GBP"),
)
_SCALE_WORDS = (
    (re.compile(r"^(?:crores?|cr\.?|करोड़|करोड)$", re.I), "crore"),
    (re.compile(r"^(?:lakhs?|lacs?|लाख)$", re.I), "lakh"),
    (re.compile(r"^(?:millions?|mn|mio|मिलियन)$", re.I), "million"),
    (re.compile(r"^(?:billions?|bn|बिलियन)$", re.I), "billion"),
    (re.compile(r"^(?:thousands?|हज़ार|हजार|'000)$", re.I), "thousand"),
)
_NUMBER_RUN = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
_PLAIN = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_WESTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_INDIAN = re.compile(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?")
_PREFIX = re.compile(rf"^[\s(]*(?P<s1>[+\-–−])?[\s(]*(?P<cur>{_CUR})?[\s(]*(?P<s2>[+\-–−])?\s*$", re.I)
_SUFFIX_TOKENS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("close", re.compile(r"\)")),
    ("pct", re.compile(r"%|per\s*cent\b|percent\b|pct\b|प्रतिशत", re.I)),
    ("pp", re.compile(r"(?:pp|ppts?|pts?|percentage\s+points?|%\s*points?)(?![^\W\d_])", re.I)),
    ("bps", re.compile(r"(?:bps|bp|basis\s+points?)(?![^\W\d_])", re.I)),
    ("ratio", re.compile(r"(?:x|×|times)(?![^\W\d_])", re.I)),
    (
        "scale",
        re.compile(
            r"(?:crores?|cr\b\.?|lakhs?|lacs?|millions?|mn|mio|billions?|bn|thousands?|करोड़|करोड|लाख|हज़ार|हजार)(?![^\W\d_])",
            re.I,
        ),
    ),
    ("cur", re.compile(rf"{_CUR}(?![^\W\d_])", re.I)),
    ("pa", re.compile(r"p\.\s?a\.?|per\s+annum\b", re.I)),
)
# Letters, including Devanagari vowel signs and nasalisation marks (Python's \w leaves those out).
_L = r"(?:[^\W\d_]|[\u0900-\u0963\u0971-\u097F])"
_NOT_AFTER_LETTER = r"(?<![^\W\d_])(?<![\u0900-\u0963\u0971-\u097F])"
_UNIT_WORDS = re.compile(rf"^{_L}+(?:[ /-]{_L}+)?\.?$")
_DURATION_WORD = re.compile(
    r"(?:days?|weeks?|months?|years?|yrs?|hours?|hrs?|minutes?|mins?|दिन|सप्ताह|हफ़्ते|महीने|माह|वर्ष|साल|घंटे)", re.I
)
_SIGNS = "+-–−"


def _scale_of(word: str) -> Scale | None:
    for pattern, scale in _SCALE_WORDS:
        if pattern.match(word.strip()):
            return scale  # type: ignore[return-value]
    return None


def _currency_of(token: str) -> str | None:
    token = token.strip()
    for pattern, code in _CURRENCY_CODE:
        if pattern.match(token):
            return code
    return None


def _number(run: str) -> tuple[float, int] | None:
    if not (_PLAIN.fullmatch(run) or _WESTERN.fullmatch(run) or _INDIAN.fullmatch(run)):
        return None
    digits = run.replace(",", "")
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    return float(digits), decimals


def parse_quantity(text: str) -> Quantity | _Null | None:
    """One quantity printed in a cell; ``NULL`` for "not available" markers; None for anything else (text, labels,
    several numbers, a period such as "FY24" or "2023-24")."""
    if not text or not text.strip():
        return NULL
    core, footnote = split_footnote(text)
    if core.casefold().rstrip(".").strip() in _NULLS:
        return NULL
    core = core.translate(_DIGITS)
    if core.endswith(".") and core[-2:-1].isdigit():
        core = core[:-1]  # "1,234." (sentence punctuation)
    # "21.0% 1": a footnote number Docling printed after a unit marker
    m = re.fullmatch(r"(.*?[%x)a-z])\s+([1-9])", core, re.I)
    if m and _NUMBER_RUN.search(m.group(1)):
        core, footnote = m.group(1), footnote or m.group(2)
    runs = list(_NUMBER_RUN.finditer(core))
    if len(runs) != 1:
        return None
    run = runs[0]
    parsed = _number(run.group(0).rstrip(","))
    if parsed is None:
        return None
    value, decimals = parsed
    prefix, suffix = core[: run.start()], core[run.end() :]
    pm = _PREFIX.match(prefix)
    if pm is None:
        return None
    found: dict[str, str] = {}
    rest = suffix
    while True:
        rest = rest.lstrip()
        if not rest:
            break
        for name, pattern in _SUFFIX_TOKENS:
            sm = pattern.match(rest)
            if sm and sm.group(0):
                if name in found and name != "close":
                    return None  # "12 % %", two scales
                found[name] = sm.group(0)
                rest = rest[sm.end() :]
                break
        else:
            break
    word = rest.strip().rstrip(")").strip() or None
    if word is not None and not _UNIT_WORDS.match(word):
        return None
    opened = "(" in prefix
    closed = "close" in found
    if opened != closed:
        return None
    sign = pm.group("s1") or pm.group("s2")
    negative = opened or (sign is not None and sign != "+")
    currency = _currency_of(pm.group("cur") or found.get("cur", "")) if (pm.group("cur") or "cur" in found) else None
    scale = _scale_of(found["scale"]) if "scale" in found else None
    if word and currency and scale is None and word.lower() in ("m", "b", "k"):
        scale, word = {"m": "million", "b": "billion", "k": "thousand"}[word.lower()], None  # type: ignore[assignment]
    kind: QuantityKind
    if "pct" in found:
        kind = "percent"
    elif "bps" in found:
        kind = "bps"
    elif "pp" in found:
        kind = "pp"
    elif "ratio" in found:
        kind = "ratio"
    elif currency is not None:
        kind = "currency"
    else:
        kind = "number"
    if "pa" in found and word is None:
        word = "p.a."
    return Quantity(
        value=-value if negative else value,
        kind=kind,
        currency=currency,
        scale=scale,
        unit_word=word,
        footnote=footnote,
        signed=sign is not None,
        decimals=decimals,
    )


def is_year_value(q: Quantity, text: str) -> bool:
    """A bare four-digit year ("2018"), as found in "commissioned" or "since" columns."""
    return q.kind == "number" and q.decimals == 0 and 1900 <= q.value <= 2100 and "," not in text and not q.signed


# ------------------------------------------------------------------ units from captions, headers and notes

_UNIT_CURRENCY_SCALE = re.compile(
    rf"{_NOT_AFTER_LETTER}(?P<cur>{_CUR})\s*(?:in\s+)?(?P<scale>crores?|cr\b\.?|lakhs?|lacs?|millions?|mn\b|billions?|bn\b|thousands?|'000|करोड़|करोड|लाख)",
    re.I,
)
_UNIT_SCALE_CURRENCY = re.compile(
    rf"{_NOT_AFTER_LETTER}(?P<scale>crores?|lakhs?|lacs?|millions?|billions?|thousands?|करोड़|करोड|लाख)\s*(?:of\s+)?(?P<cur>{_CUR})(?![^\W\d_])",
    re.I,
)
_UNIT_IN_SCALE = re.compile(
    r"(?:(?P<paren>\()|\bin\s+|\bamounts?\s+in\s+)\s*(?P<scale>crores?|lakhs?|lacs?|millions?|billions?|thousands?)\b"
    r"(?P<more>[^()]{0,24}\))?",
    re.I,
)
_UNIT_CURRENCY_ONLY = re.compile(rf"\(\s*(?:in\s+)?(?P<cur>{_CUR})(?:\s+per\s+{_L}+(?:\s+{_L}+)?)?\s*\)", re.I)
_UNIT_PERCENT = re.compile(r"%|\(\s*per\s*cent\s*\)|\bpercentage\b", re.I)
_UNIT_RATIO = re.compile(r"\(\s*(?:x|times)\s*\)", re.I)
_UNIT_COUNT = re.compile(r"\(\s*(?:nos?\.?|number|count|#)\s*\)|^\s*(?:number|no\.)\s+of\b|\bheadcount\b", re.I)
_UNIT_DURATION = re.compile(rf"\(\s*(?:in\s+)?(?P<word>{_DURATION_WORD.pattern})\s*\)", re.I)


def detect_amount_unit(text: str | None) -> Unit | None:
    """The currency and scale a sentence states for a whole table ("All amounts are in ₹ crore", "(₹ in crore)",
    "Rs. in lakhs", "(US$ million)", "(in millions)"). A stated amount such as "₹ 41 crore" is not a unit statement,
    nor is a percentage in passing: this is what captions, notes and headings are read with."""
    if not text:
        return None
    text = normalize_space(text)
    m = _UNIT_CURRENCY_SCALE.search(text) or _UNIT_SCALE_CURRENCY.search(text)
    if m:
        raw = m.group("scale").rstrip(".")
        scale = "thousand" if raw == "'000" else _scale_of(raw)
        return currency_unit(_currency_of(m.group("cur")) or "INR", scale)
    m = _UNIT_IN_SCALE.search(text)
    if m:
        scale = _scale_of(m.group("scale"))
        if scale in ("crore", "lakh"):  # Indian scales are rupee amounts
            return currency_unit("INR", scale)
        if scale is not None:
            label = scale
            if m.group("paren") and m.group("more"):
                label = normalize_space(f"{m.group('scale')}{m.group('more')[:-1]}")
            return Unit(kind="none", scale=scale, label=label)
    return None


def detect_unit(text: str | None) -> Unit | None:
    """The unit a column header or row label states: an amount unit (``detect_amount_unit``), a currency alone
    ("(₹)", "(₹ per night)"), "%", "(x)", a duration ("(days)", "अवधि (सप्ताह)") or a count ("Number of
    employees", "(nos.)")."""
    if not text:
        return None
    unit = detect_amount_unit(text)
    if unit is not None:
        return unit
    text = normalize_space(text)
    m = _UNIT_CURRENCY_ONLY.search(text)
    if m:
        return currency_unit(_currency_of(m.group("cur")) or "INR", None)
    if _UNIT_PERCENT.search(text):
        return PERCENT
    if _UNIT_RATIO.search(text):
        return RATIO
    m = _UNIT_DURATION.search(text)
    if m:
        return Unit(kind="duration", label=m.group("word"))
    if _UNIT_COUNT.search(text):
        return COUNT
    return None


# ------------------------------------------------------------------ periods

_MONTHS_EN = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11,
    "dec": 12,
}  # fmt: skip
_MONTHS_HI = {
    "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
    "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}  # fmt: skip
_MONTH = (
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
    r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|" + "|".join(_MONTHS_HI) + r")\.?"
)
_FY = r"(?:fy|f\.y\.|fiscal(?:\s+year)?|financial\s+year|वित्त(?:ीय)?\s+वर्ष|वि\.\s?व\.)"
_YEARS = r"(?P<y1>(?:19|20)\d{2}|\d{2})(?:\s*[-–/]\s*(?P<y2>(?:19|20)\d{2}|\d{2}))?"
_B = r"(?<![^\W_])(?<![ऀ-ॿ])"  # not inside a word (Latin or Devanagari letters and digits)
_E = r"(?![^\W_])(?![ऀ-ॿ])"

_PERIOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "quarter_fy",
        re.compile(rf"{_B}(?:q|quarter\s*)(?P<n>[1-4])\s*['’]?\s*(?:of\s+)?{_FY}\s*['’]?\s*{_YEARS}{_E}", re.I),
    ),
    ("quarter_fy", re.compile(rf"{_B}(?P<n>[1-4])q\s*{_FY}\s*['’]?\s*{_YEARS}{_E}", re.I)),
    ("half_fy", re.compile(rf"{_B}(?:h|half\s*)(?P<n>[12])\s*['’]?\s*{_FY}\s*['’]?\s*{_YEARS}{_E}", re.I)),
    ("nine_fy", re.compile(rf"{_B}(?:9m|9\s*months|nine\s+months)\s*{_FY}\s*['’]?\s*{_YEARS}{_E}", re.I)),
    (
        "year_ended",
        re.compile(
            rf"{_B}(?:(?:for\s+the\s+)?year\s+ended|ye)\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH}[\s,]*(?P<y>(?:19|20)\d{{2}}){_E}",
            re.I,
        ),
    ),
    ("fy", re.compile(rf"{_B}{_FY}\s*['’]?\s*{_YEARS}{_E}", re.I)),
    ("quarter_cy", re.compile(rf"{_B}q(?P<n>[1-4])\s*(?:cy\s*)?(?P<y>(?:19|20)\d{{2}}){_E}", re.I)),
    (
        "date_dmy",
        re.compile(
            rf"{_B}(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[\s.\-/]*{_MONTH}[\s.,\-/']*(?P<y>(?:19|20)\d{{2}}){_E}", re.I
        ),
    ),
    ("date_mdy", re.compile(rf"{_B}{_MONTH}\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>(?:19|20)\d{{2}}){_E}", re.I)),
    ("date_iso", re.compile(rf"{_B}(?P<y>(?:19|20)\d{{2}})-(?P<m>\d{{2}})-(?P<d>\d{{2}}){_E}")),
    ("date_num", re.compile(rf"{_B}(?P<d>\d{{1,2}})[./](?P<m>\d{{1,2}})[./](?P<y>(?:19|20)\d{{2}}){_E}")),
    ("fy_range", re.compile(rf"{_B}(?P<y1>(?:19|20)\d{{2}})\s*[-–/]\s*(?P<y2>(?:19|20)?\d{{2}}){_E}")),
    ("month", re.compile(rf"{_B}{_MONTH}[\s\-'’,]*(?P<y>(?:19|20)\d{{2}}|\d{{2}}){_E}", re.I)),
    ("cy", re.compile(rf"{_B}cy\s*['’]?\s*(?P<y>(?:19|20)\d{{2}}|\d{{2}}){_E}", re.I)),
    ("year", re.compile(rf"{_B}(?P<y>(?:19|20)\d{{2}}){_E}")),
    ("quarter", re.compile(rf"{_B}q(?P<n>[1-4]){_E}", re.I)),
    ("half", re.compile(rf"{_B}h(?P<n>[12]){_E}", re.I)),
)


@dataclass(frozen=True, slots=True)
class Period:
    """A reporting period. ``label`` is canonical ("FY24", "Q3 FY24", "31 Mar 2024"), so the same period printed
    differently in two tables matches. ``end`` is the last day (None for an unanchored "Q1"); ``months`` its length
    (0 for a point in time such as a balance-sheet date)."""

    label: str
    kind: PeriodKind
    end: date | None
    months: int
    index: int = 0  # unanchored quarters/halves: their number

    @property
    def granularity(self) -> str:
        return {"fy": "year", "year": "year", "nine_months": "nine_months"}.get(self.kind, self.kind)

    @property
    def sort_key(self) -> tuple[int, int]:
        if self.end is None:
            return (self.index, 0)
        return (self.end.toordinal(), -self.months)

    @property
    def iso(self) -> str | None:
        return self.end.isoformat() if self.end is not None and self.kind == "date" else None


def _year(text: str) -> int:
    y = int(text)
    return y if y >= 100 else (2000 + y if y < 70 else 1900 + y)


def _month_number(text: str) -> int | None:
    text = text.strip(".").lower()
    if text in _MONTHS_HI:
        return _MONTHS_HI[text]
    return _MONTHS_EN.get(text[:3])


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _fy_end(y1: str, y2: str | None) -> int | None:
    """End year of "FY24", "FY 2023-24", "23-24"; None when the two years aren't consecutive."""
    first = _year(y1)
    if y2 is None:
        return first
    second = int(y2) if len(y2) == 4 else (first // 100) * 100 + int(y2)
    if second < first:
        second += 100
    return second if second == first + 1 else None


def _fy_label(end_year: int) -> str:
    return f"FY{end_year % 100:02d}"


def _date_label(d: date) -> str:
    return f"{d.day} {calendar.month_abbr[d.month]} {d.year}"


def _build(kind: str, m: re.Match[str]) -> Period | None:
    g = m.groupdict()
    try:
        if kind in ("quarter_fy", "half_fy", "nine_fy", "fy"):
            end_year = _fy_end(g["y1"], g.get("y2"))
            if end_year is None:
                return None
            if kind == "fy":
                return Period(_fy_label(end_year), "fy", date(end_year, 3, 31), 12)
            n = int(g.get("n") or 0)
            if kind == "quarter_fy":
                month = (3 + 3 * n) % 12 or 12
                year = end_year if n == 4 else end_year - 1
                return Period(f"Q{n} {_fy_label(end_year)}", "quarter", _month_end(year, month), 3)
            if kind == "half_fy":
                end = _month_end(end_year - 1, 9) if n == 1 else date(end_year, 3, 31)
                return Period(f"H{n} {_fy_label(end_year)}", "half", end, 6)
            return Period(f"9M {_fy_label(end_year)}", "nine_months", _month_end(end_year - 1, 12), 9)
        if kind == "fy_range":
            end_year = _fy_end(g["y1"], g["y2"])
            return Period(_fy_label(end_year), "fy", date(end_year, 3, 31), 12) if end_year else None
        if kind == "year_ended":
            month = _month_number(g["month"])
            if month is None:
                return None
            end = date(int(g["y"]), month, int(g["d"]))
            if month == 3:
                return Period(_fy_label(end.year), "fy", end, 12)
            return Period(str(end.year) if month == 12 else f"YE {_date_label(end)}", "year", end, 12)
        if kind == "quarter_cy":
            n, year = int(g["n"]), int(g["y"])
            return Period(f"Q{n} {year}", "quarter", _month_end(year, 3 * n), 3)
        if kind in ("date_dmy", "date_mdy", "date_iso", "date_num"):
            month = int(g["m"]) if g.get("m") else _month_number(g["month"])
            if month is None:
                return None
            d = date(int(g["y"]), month, int(g["d"]))
            return Period(_date_label(d), "date", d, 0)
        if kind == "month":
            month = _month_number(g["month"])
            if month is None:
                return None
            year = _year(g["y"])
            return Period(f"{calendar.month_abbr[month]} {year}", "month", _month_end(year, month), 1)
        if kind in ("cy", "year"):
            year = _year(g["y"])
            return Period(str(year), "year", date(year, 12, 31), 12)
        if kind == "quarter":
            return Period(f"Q{g['n']}", "quarter", None, 3, index=int(g["n"]))
        if kind == "half":
            return Period(f"H{g['n']}", "half", None, 6, index=int(g["n"]))
    except ValueError:  # 31 Feb, month 13
        return None
    return None


def find_period(text: str | None) -> tuple[Period, str] | None:
    """The first period stated in ``text`` and the rest of the text without it ("Revenue FY24" → FY24, "Revenue")."""
    if not text:
        return None
    text, _ = split_footnote(unicodedata.normalize("NFC", text))
    text = text.translate(_DIGITS)
    for kind, pattern in _PERIOD_PATTERNS:
        for m in pattern.finditer(text):
            period = _build(kind, m)
            if period is not None:
                rest = normalize_space(text[: m.start()] + " " + text[m.end() :])
                return period, _tidy_rest(rest)
    return None


_CONNECTORS = re.compile(
    r"^(?:in|for|of|during|the|as\s+at|as\s+on|ended|ending|year|:|-|–|,)\s+|\s+(?:in|for|of|during|the|as\s+at|as\s+on|:|-|–|,)$",
    re.I,
)


def _tidy_rest(rest: str) -> str:
    rest = re.sub(r"\(\s*\)", "", rest)
    previous = None
    while previous != rest:
        previous = rest
        rest = _CONNECTORS.sub("", rest).strip(" ,:;–-")
    return normalize_space(re.sub(r"\s+([,.;:])", r"\1", rest))


_PERIOD_REMARK = re.compile(
    r"^(?:full\s+year|year|total|annual|quarter(?:\s+ended)?|half\s+year|as\s+at|as\s+on|year\s+ended|ended|"
    r"actual|audited|unaudited|reported|recommended|provisional|estimated|est\.?|e|a|p|"
    r"पूर्ण\s+वर्ष|वर्ष|कुल)?$",
    re.I,
)


def parse_period(text: str | None) -> Period | None:
    """``text`` as a period label: the period must be essentially all of it ("FY24", "Q3 FY24", "Full year FY24",
    "FY24 (recommended)", "31 Mar 2024"); "Revenue FY24" or "2009 (expanded 2023)" are not period labels."""
    found = find_period(text)
    if found is None:
        return None
    period, rest = found
    rest = re.sub(r"\([^)]*\)", "", rest).strip(" ,:;–-")  # remarks in brackets: "(recommended)", "(paid in FY24)"
    return period if _PERIOD_REMARK.match(rest) else None


def is_aggregate_period_label(text: str) -> bool:
    """A row like "Full year FY24" or "Total FY24": the sum of the periods above it, not one more of them."""
    return bool(re.search(r"\b(full\s+year|total|annual|पूर्ण\s+वर्ष|कुल)\b", text, re.I))
