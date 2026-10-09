"""When a question needs live data, and the search query that may leave the machine (docs/DESIGN.md §3.7).

**Live-data cues** (``live_data_cue``), English, Hindi (Devanagari) and Hinglish, deliberately conservative. Three
kinds, checked in this order:

- *strong time cues* ask for the current value of something live: "right now", "real-time", "live price", "latest
  news", "current share price", "current repo rate", "price today", "the stock doing today", "today's news",
  "लाइव", "मौजूदा भाव". They count even in a question about the documents ("the report says revenue grew 34%; how is
  the stock doing today?").
- *weak time cues* only say "now": "today", "this week", "this morning", "at the moment", "आज", "अभी", "इस समय",
  "फ़िलहाल", "ताज़ा", "aaj", "abhi", "filhaal". Not when they name a meeting ("today's agenda", "this week's
  deliverables", "आज की बैठक", "aaj ki meeting").
- *topic cues* name data that is live by nature: "news", "headlines", "latest update", "current price", "share
  price", "stock doing", "market cap", "exchange rate", "weather", "Sensex", "ख़बर", "शेयर का भाव", "khabar".

Weak time cues and topic cues don't count when the question points at the documents ("the report", "the minutes",
"according to the…", "रिपोर्ट के अनुसार", a filename) or at a past period or something a document states ("FY24",
"Q3", "2023", "at the end of the year", "on 31 March", "last quarter", "mentioned", "assumed", "used", "reported",
"per unit", "buyback"). Accounting terms carry no cue at all: "current ratio", "current assets", "current rates of
depreciation" (a "current … rate" needs a qualifier such as repo or exchange), "latest annual report", "अभी तक"
(so far), "आजकल" (nowadays); "doing now" needs a stock, shares or market subject. Romanized Hindi cues count only
in Hinglish messages ("Abhi" may be a name).

**The search query** (``web_query``): built only from the router's standalone English question (or an English
utterance that stands alone), and reduced to what asks for the live data:

- only the clauses with a live-data cue; parts and phrases that point at the documents removed ("…, discussed in
  the board minutes", "from the Falcon merger minutes", "according to the report"), filenames and the chat's
  document names removed;
- figures that the user didn't say themselves removed (amounts, percentages, decimals, ₹/Rs/crore/lakh/million: the
  router may fold an earlier answer's figures into its question); years and FY / quarter tags stay;
- emails, URLs, long digit runs (phone, account and card numbers) and PAN-like ids removed; one line, at most
  WEB_QUERY_MAX_CHARS.

Document text and the transcript are never inputs. A turn without an English question gets no search: a Hindi
utterance itself never leaves the machine. By design, an entity the router resolves from the conversation (a
company's name for "the stock") can be part of the query.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from .language import message_language

WEB_QUERY_MAX_CHARS = 160

# Why a question that wants live data gets none although web search is turned on: the tool can't run now
# ("unavailable"), or there was nothing to search for or the search gave nothing in time ("failed").
LiveNote = Literal["unavailable", "failed"]

_W = "(?<![\\w\\u0900-\\u097F])"  # start of a word (Devanagari-aware: \b fails after Hindi vowel signs)
_E = "(?![\\w\\u0900-\\u097F])"  # end of a word

_MEETING_EN = (
    r"(?:meetings?|agenda|minutes|calls?|sessions?|discussions?|presentations?|notes|deliverables|action\s+items?|"
    r"decisions?|reviews?|stand-?ups?|syncs?|board)"
)
_MEETING_HI = "(?:बैठक|मीटिंग|एजेंडा|चर्चा|कॉल|सभा)"

_STRONG_EN = re.compile(
    r"""\b(?:
        right\s+now|as\s+of\s+(?:now|today)|real[\s-]?time|
        live\s+(?:data|price|prices|score|scores|updates?|news|rate|rates|market|stock|feed)|
        latest\s+(?:\w+\s+)?(?:news|price|prices|rate|rates|headlines|quote|score)|
        current\s+(?:market|stock|share|trading)\s+(?:price|prices|valuation|quote)|
        current\s+(?:repo|exchange|dollar|rupee|inflation|policy|gold|petrol|market)\s+rates?|
        currently\s+(?:trading|priced|worth|valued|quoted)|
        (?:price|prices|rate|rates|trading|valued|worth)\s+(?:right\s+)?(?:now|today)|
        (?:stocks?|shares?|markets?|sensex|nifty)\s+(?:\w+\s+)?doing\s+(?:right\s+)?(?:now|today)|
        today'?s\s+(?:stock\s+|share\s+|market\s+|closing\s+|opening\s+)?
            (?:price|prices|rate|rates|news|headlines|close|quote|weather)|
        yesterday'?s?\s+(?:close|closing|price|news|market)|(?:this|last)\s+hour
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_WEAK_EN = re.compile(
    rf"\b(?:today(?:'?s)?|tonight|this\s+(?:week|morning)|at\s+the\s+moment)(?!\w)"
    rf"(?!'?s?\s+(?:\w+\s+)?{_MEETING_EN}\b)",
    re.IGNORECASE,
)
_TOPIC_EN = re.compile(
    r"""\b(?:
        news|headlines?|latest\s+(?:\w+\s+)?(?:updates?|developments)|current\s+prices?|
        (?:stock|share)\s+(?:price|prices|market|quote)|(?:stock|stocks|shares?)\s+(?:doing|trading|performing)|
        trading\s+at|market\s+cap(?:italisation|italization)?|exchange\s+rates?|forex|weather|
        (?:usd|dollar|rupee|inr|euro|eur|gbp|pound|yen)\s+(?:to|vs\.?|versus|against|in)\s+
            (?:usd|dollars?|rupees?|inr|euros?|eur|gbp|pounds?|yen)|
        (?:dollar|rupee|gold|silver|petrol|diesel|crude|bitcoin)\s+(?:rate|price|prices)|
        sensex|nifty|nasdaq|dow\s+jones
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_STRONG_HI = re.compile(f"{_W}(?:लाइव|रियल\\s*टाइम|(?:मौजूदा|वर्तमान)\\s+(?:कीमत|भाव|दर|रेट|प्राइस)){_E}")
_WEAK_HI = re.compile(
    f"{_W}(?:आज(?!\\s+(?:की|के|का)\\s+{_MEETING_HI})|अभी(?!\\s*तक)|इस\\s+समय|फ़िलहाल|फिलहाल|ताज़ा|ताजा|ताज़ी|ताजी){_E}"
)
_TOPIC_HI = re.compile(
    f"{_W}(?:ख़बर|खबर|ख़बरें|खबरें|समाचार|न्यूज़|न्यूज|सुर्खियाँ|सुर्खियां|मौसम|सेंसेक्स|निफ्टी|विनिमय\\s+दर|"
    f"शेयर\\s+(?:की\\s+कीमत|का\\s+भाव|का\\s+दाम|प्राइस|बाज़ार|बाजार)|"
    f"(?:सोने|चांदी|चाँदी|डॉलर|रुपये|पेट्रोल|डीज़ल|डीजल)\\s+(?:का|की)\\s+(?:भाव|दाम|कीमत|रेट|दर))"
    f"{_E}"
)
# Romanized Hindi: counted only in a message that is Hinglish ("aj" or "abhi" may be names in English).
_WEAK_HINGLISH = re.compile(
    r"\b(?:(?:aaj|aj)(?!\s+(?:ki|ke|ka)\s+(?:meeting|baithak|agenda|call))|abhi(?!\s+tak)|filhaal|filhal|taaza|taza|"
    r"taazi|tazi|is\s+samay)\b",
    re.IGNORECASE,
)
_TOPIC_HINGLISH = re.compile(
    r"\b(?:khabar|khabre|khabrein|samachar|mausam|bhav|bhaav|"
    r"(?:share|stock)\s+(?:ka|ki)\s+(?:price|rate|bhav|bhaav|daam|keemat|kimat))\b",
    re.IGNORECASE,
)
_DOC_NOUNS = (
    r"(?:report|reports|document|documents|doc|docs|file|files|pdf|deck|slides|filing|annexure|appendix|minutes|"
    r"memo|presentation|notes|agreement|contract|statements?|transcript|forecast|section|table|offer|policy)"
)
_DOC_POINTER = re.compile(
    rf"\b(?:the|this|that|these|those|my|our|your|uploaded)\s+(?:[a-z0-9'-]+\s+)?{_DOC_NOUNS}\b|"
    r"\baccording\s+to\s+the\b|\b(?:report|document|file|minutes)\s+(?:mein|me|ke\s+anusar)\b|"
    r"(?:रिपोर्ट|दस्तावेज़|दस्तावेज|डॉक्यूमेंट|फ़ाइल|फाइल)\s*(?:में|के\s+अनुसार)",
    re.IGNORECASE,
)
# A past period, or something a document states: the question is about the documents, not live data.
_PAST_OR_STATED = re.compile(
    r"\b(?:FY\s?'?\d{2,4}|Q[1-4]|H[12]|(?:19|20)\d\d|(?:end|close|start)\s+of\s+(?:the\s+)?(?:year|quarter|period|fy)|"
    r"as\s+(?:of|on|at)\s+\d|on\s+\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+|last\s+(?:year|quarter|fiscal)|"
    r"mentioned|assumed|used|stated|reported|disclosed|listed|quoted\s+in|per\s+unit|buyback|"
    r"(?:did|do|does)\s+(?:they|we|you|it|the\s+company)\s+use)\b",
    re.IGNORECASE,
)


def live_data_cue(text: str | None, *, documents: Sequence[str] = ()) -> str | None:
    """The words that make ``text`` ask for live or current data, or None (module docstring). ``documents``: the
    chat's filenames; naming one points at the documents too."""
    if not text or not text.strip():
        return None
    for pattern in (_STRONG_EN, _STRONG_HI):
        if m := pattern.search(text):
            return _cue(m)
    if about_the_documents(text, documents):
        return None
    hinglish = message_language(text) == "hi"
    patterns = [_WEAK_EN, _WEAK_HI, *([_WEAK_HINGLISH] if hinglish else []), _TOPIC_EN, _TOPIC_HI]
    if hinglish:
        patterns.append(_TOPIC_HINGLISH)
    for pattern in patterns:
        if m := pattern.search(text):
            return _cue(m)
    return None


def _cue(m: re.Match[str]) -> str:
    return " ".join(m.group(0).split()).casefold()


def points_at_documents(text: str, documents: Sequence[str] = ()) -> bool:
    """The text refers to the user's documents ("the report", "the minutes", "according to the…", "रिपोर्ट में", a
    filename)."""
    if _DOC_POINTER.search(text):
        return True
    low = text.casefold()
    return any(len(stem) > 3 and stem in low for stem in _stems(documents))


def about_the_documents(text: str, documents: Sequence[str] = ()) -> bool:
    """Points at the documents, at a past period, or at something a document states (module docstring)."""
    return points_at_documents(text, documents) or bool(_PAST_OR_STATED.search(text))


def _stems(documents: Sequence[str]) -> list[str]:
    return [n.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").casefold().strip() for n in documents]


# ------------------------------------------------------------------ the search query

_DASHES = "\N{EM DASH}\N{EN DASH}"
_CLAUSES = re.compile(rf"\n+|(?<=[.;!?])\s+|\s+[{_DASHES}]\s+|\s+-\s+|;\s*")  # lines, sentences, dashes, semicolons
_PARTS = re.compile(r",\s+")  # not "4,512"
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"\+?\d[\d\s-]{7,}\d")  # phone, account and card numbers
_ID_LIKE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")  # PAN-like ids
_FILENAME = re.compile(r"\b[\w.-]+\.(?:pdf|docx?|pptx?|xlsx?|csv|txt|md)\b", re.IGNORECASE)
_POINTER_PHRASE = re.compile(
    r"(?:\b(?:in|from|per|of|on|by|as\s+per|according\s+to|as\s+(?:stated|mentioned|reported|discussed)\s+in|"
    r"(?:discussed|mentioned|stated|reported|listed)\s+in)\s+)?"
    rf"\b(?:the|this|that|these|those|my|our|your|uploaded)\s+(?:[\w'-]+\s+){{0,3}}?{_DOC_NOUNS}\b"
    r"(?:\s+(?:says?|said|shows?|mentions?|states?)(?:\s+that)?)?",
    re.IGNORECASE,
)
_FIGURE = re.compile(
    r"(?:\b(?:of|at|by|to|from|around|about|over|under|nearly|almost|approximately)\s+)?"
    r"(?:(?:₹|\bRs\.?|\bINR|\bUSD|\$|€|£)\s?)?(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\w)"
    r"(?:\s?(?:%|per\s?cent\b|crores?\b|cr\b|lakhs?\b|lacs?\b|million\b|mn\b|billion\b|bn\b|thousand\b|trillion\b))?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?:19|20)\d\d")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DANGLING = re.compile(
    r"\b(?:of|at|to|by|was|is|were|are|from|with|about|around|than|and|for|in|its|their|a|an|the)\s*(?=[?.!,;:)]|$)",
    re.IGNORECASE,
)
_OPENER = re.compile(r"^(?:and|but|so|also|then|now|okay|ok|well|great|thanks)\b[\s,]*", re.IGNORECASE)


def web_query(question: str | None, *, documents: Sequence[str] = (), utterance: str | None = None) -> str | None:
    """The search query for an English standalone question (module docstring), or None when there is nothing
    English to search for. ``utterance``: what the user actually said (figures in it may stay)."""
    if not question or re.search("[ऀ-ॿ]", question):
        return None
    if not " ".join(question.split()) or message_language(question) == "hi":
        return None
    clauses = [c.strip(" ,") for c in _CLAUSES.split(question) if c and c.strip(" ,")]
    live = [c for c in clauses if live_data_cue(c, documents=documents) is not None] or clauses
    kept: list[str] = []
    for clause in live:
        # Comma-separated parts that point at the documents, or restate figures the user didn't say without asking
        # for live data ("Given revenue grew 34% to Rs 4,512 crore, …"), go; if all do, their phrases go (below).
        parts = [p.strip() for p in _PARTS.split(clause) if p.strip()]
        about_live = [
            p
            for p in parts
            if not points_at_documents(p, documents)
            and (live_data_cue(p, documents=documents) is not None or not _unsaid_figure(p, utterance))
        ]
        kept.extend(about_live or parts)
    text = " ".join(" ".join(kept).split())
    text = _FILENAME.sub(" ", text)
    for stem in _stems(documents):  # the chat's document names, e.g. "(falcon merger board minutes)"
        if len(stem) > 3:
            text = re.sub(re.escape(stem).replace("\\ ", r"[\s_-]+"), " ", text, flags=re.IGNORECASE)
    for pattern in (_POINTER_PHRASE, _EMAIL, _URL, _LONG_DIGITS, _ID_LIKE):
        text = pattern.sub(" ", text)
    text = _strip_figures(text, utterance)
    text = _tidy(_OPENER.sub("", _tidy(text)))
    if len(text) > WEB_QUERY_MAX_CHARS:
        cut = text[:WEB_QUERY_MAX_CHARS]
        text = cut[: cut.rfind(" ")] if " " in cut else cut
    return text or None


def _said(figure: str, utterance: str | None) -> bool:
    """A year, or a number the user said themselves."""
    number = _NUMBER.search(figure)
    digits = number.group(0).replace(",", "") if number else ""
    if _YEAR.fullmatch(digits) and not re.search(r"[%₹$€£]|rs|crore|lakh|million|billion", figure, re.IGNORECASE):
        return True
    return digits in {n.replace(",", "") for n in _NUMBER.findall(utterance or "")}


def _unsaid_figure(text: str, utterance: str | None) -> bool:
    return any(not _said(m.group(0), utterance) for m in _FIGURE.finditer(text))


def _strip_figures(text: str, utterance: str | None) -> str:
    """Figures the user didn't say (an earlier answer's, folded into the question by the router) are removed with
    the preposition before them; years stay, FY / quarter tags aren't figures."""
    return _FIGURE.sub(lambda m: m.group(0) if _said(m.group(0), utterance) else " ", text)


def _tidy(text: str) -> str:
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    text = " ".join(text.split())
    previous = None
    while previous != text:
        previous = text
        text = _DANGLING.sub("", text)
        text = re.sub(r"\s+([?.!,;:)])", r"\1", text)
        text = re.sub(r"([,;:])(?=[?.!,;:]|$)", "", text)
        text = " ".join(text.split())
    return text.strip(" ,;:-")
