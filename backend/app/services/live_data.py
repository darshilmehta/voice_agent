"""When a question needs live data, and the search query that may leave the machine (docs/DESIGN.md §3.7).

**Live-data cues** (``live_data_cue``), English, Hindi (Devanagari) and Hinglish, deliberately conservative:

- *time cues* say the answer must be current: "today", "right now", "latest news", "current price", "live score",
  "real-time", "आज", "अभी", "ताज़ा", "मौजूदा भाव", "aaj", "abhi", "filhaal". They count even in a question about
  the documents ("the report says revenue grew 34%; how is the stock doing today?").
- *topic cues* name data that is live by nature: "news", "headlines", "share price", "stock doing", "market cap",
  "exchange rate", "weather", "Sensex", "ख़बर", "शेयर का भाव", "मौसम", "khabar", "bhav". They count only when the
  question doesn't point at the documents ("what does the report say about the share price?" stays a document
  question).
- Accounting and document terms carry no cue: "current ratio", "current assets", "current year figures", "latest
  annual report", "revenue forecast", "अभी तक" (so far), "आजकल" (nowadays).

**The search query** (``web_query``): built only from the router's standalone English question (or an English
utterance that stands alone): the clauses that ask for the live data (parts that point at the documents dropped),
emails, URLs and long digit runs removed, one line, at most WEB_QUERY_MAX_CHARS. Document text and the transcript
are never inputs. A turn without an English question gets no search: a Hindi utterance itself never leaves the
machine.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from .language import message_language

WEB_QUERY_MAX_CHARS = 160

# Why a question that wants live data gets none: the tool can't run ("unavailable"), or there was nothing to search
# for or the search gave nothing in time ("failed"). The answer starts by saying so (prompts.live_notice).
LiveNote = Literal["unavailable", "failed"]

_W = "(?<![\\w\\u0900-\\u097F])"  # start of a word (Devanagari-aware: \b fails after Hindi vowel signs)
_E = "(?![\\w\\u0900-\\u097F])"  # end of a word

_TIME_EN = re.compile(
    r"""\b(?:
        today|today's|todays|tonight|right\s+now|as\s+of\s+(?:now|today)|at\s+the\s+moment|this\s+(?:week|morning)|
        real[\s-]?time|live\s+(?:data|price|prices|score|scores|updates?|news|rate|rates|market|stock|feed)|
        latest\s+(?:\w+\s+)?(?:news|price|prices|rate|rates|updates?|headlines|quote|developments|score)|
        current\s+(?:market\s+|stock\s+|share\s+|trading\s+)?(?:price|prices|valuation|quote)|
        current\s+(?:(?:repo|exchange|dollar|rupee|inflation|policy|gold|petrol|market)\s+)?rates?|
        currently\s+(?:trading|priced|worth|valued|quoted)|
        (?:price|rate|trading|doing|valued|worth)\s+(?:now|today)|
        yesterday'?s?\s+(?:close|closing|price|news|market)|(?:this|last)\s+hour
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_TOPIC_EN = re.compile(
    r"""\b(?:
        news|headlines?|
        (?:stock|share)\s+(?:price|prices|market|quote)|(?:stock|stocks|shares?)\s+(?:doing|trading|performing)|
        trading\s+at|market\s+cap(?:italisation|italization)?|exchange\s+rates?|forex|weather|
        (?:usd|dollar|rupee|inr|euro|eur|gbp|pound|yen)\s+(?:to|vs\.?|versus|against|in)\s+
            (?:usd|dollars?|rupees?|inr|euros?|eur|gbp|pounds?|yen)|
        (?:dollar|rupee|gold|silver|petrol|diesel|crude|bitcoin)\s+(?:rate|price|prices)|
        sensex|nifty|nasdaq|dow\s+jones
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_TIME_HI = re.compile(
    f"{_W}(?:आज|अभी(?!\\s*तक)|इस\\s+समय|फ़िलहाल|फिलहाल|ताज़ा|ताजा|ताज़ी|ताजी|लाइव|रियल\\s*टाइम|"
    f"(?:मौजूदा|वर्तमान)\\s+(?:कीमत|भाव|दर|रेट|प्राइस))"
    f"{_E}"
)
_TOPIC_HI = re.compile(
    f"{_W}(?:ख़बर|खबर|ख़बरें|खबरें|समाचार|न्यूज़|न्यूज|सुर्खियाँ|सुर्खियां|मौसम|सेंसेक्स|निफ्टी|विनिमय\\s+दर|"
    f"शेयर\\s+(?:की\\s+कीमत|का\\s+भाव|का\\s+दाम|प्राइस|बाज़ार|बाजार)|"
    f"(?:सोने|चांदी|चाँदी|डॉलर|रुपये|पेट्रोल|डीज़ल|डीजल)\\s+(?:का|की)\\s+(?:भाव|दाम|कीमत|रेट|दर))"
    f"{_E}"
)
# Romanized Hindi: counted only in a message that is Hinglish ("aj" or "abhi" may be names in English).
_TIME_HINGLISH = re.compile(
    r"\b(?:aaj|aj|abhi(?!\s+tak)|filhaal|filhal|taaza|taza|taazi|tazi|is\s+samay)\b",
    re.IGNORECASE,
)
_TOPIC_HINGLISH = re.compile(
    r"\b(?:khabar|khabre|khabrein|samachar|mausam|bhav|bhaav|"
    r"(?:share|stock)\s+(?:ka|ki)\s+(?:price|rate|bhav|bhaav|daam|keemat|kimat))\b",
    re.IGNORECASE,
)
_DOC_POINTER = re.compile(
    r"\b(?:the|this|that|these|those|my|our|your|uploaded)\s+(?:[a-z0-9'-]+\s+)?(?:report|reports|document|"
    r"documents|doc|docs|file|files|pdf|deck|slides|filing|annexure|appendix)\b|\baccording\s+to\s+the\b|"
    r"\b(?:report|document|file)\s+(?:mein|me|ke\s+anusar)\b|"
    r"(?:रिपोर्ट|दस्तावेज़|दस्तावेज|डॉक्यूमेंट|फ़ाइल|फाइल)\s*(?:में|के\s+अनुसार)",
    re.IGNORECASE,
)


def live_data_cue(text: str | None, *, documents: Sequence[str] = ()) -> str | None:
    """The words that make ``text`` ask for live or current data, or None (module docstring). ``documents``: the
    chat's filenames; naming one points at the documents too."""
    if not text or not text.strip():
        return None
    for pattern in (_TIME_EN, _TIME_HI):
        if m := pattern.search(text):
            return _cue(m)
    hinglish = message_language(text) == "hi"
    if hinglish and (m := _TIME_HINGLISH.search(text)):
        return _cue(m)
    if points_at_documents(text, documents):
        return None
    for pattern in (_TOPIC_EN, _TOPIC_HI):
        if m := pattern.search(text):
            return _cue(m)
    if hinglish and (m := _TOPIC_HINGLISH.search(text)):
        return _cue(m)
    return None


def _cue(m: re.Match[str]) -> str:
    return " ".join(m.group(0).split()).casefold()


def points_at_documents(text: str, documents: Sequence[str] = ()) -> bool:
    """The text refers to the user's documents ("the report", "according to the…", "रिपोर्ट में", a filename)."""
    if _DOC_POINTER.search(text):
        return True
    low = text.casefold()
    for name in documents:
        stem = name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").casefold().strip()
        if len(stem) > 3 and stem in low:
            return True
    return False


# ------------------------------------------------------------------ the search query

_DASHES = "\N{EM DASH}\N{EN DASH}"
_CLAUSES = re.compile(rf"(?<=[.;!?])\s+|\s+[{_DASHES}]\s+|\s+-\s+|;\s*")  # sentences, dashes, semicolons
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"\+?\d[\d\s-]{7,}\d")  # phone and account numbers
_OPENER = re.compile(r"^(?:and|but|so|also|then|now|okay|ok|well|great|thanks)\b[\s,]*", re.IGNORECASE)


def web_query(question: str | None, *, documents: Sequence[str] = ()) -> str | None:
    """The search query for an English standalone question (module docstring), or None when there is nothing
    English to search for."""
    if not question:
        return None
    text = " ".join(question.split())
    if not text or re.search("[ऀ-ॿ]", text) or message_language(text) == "hi":
        return None
    clauses = [c.strip(" ,") for c in _CLAUSES.split(text) if c and c.strip(" ,")]
    live = [c for c in clauses if live_data_cue(c, documents=documents) is not None] or clauses
    kept: list[str] = []
    for clause in live:
        parts = [p.strip() for p in clause.split(",") if p.strip()]
        about_live = [p for p in parts if not points_at_documents(p, documents)] or parts
        kept.append(" ".join(about_live))
    text = _OPENER.sub("", " ".join(kept))
    for pattern in (_EMAIL, _URL, _LONG_DIGITS):
        text = pattern.sub(" ", text)
    text = " ".join(text.split()).strip(" ,;:-")
    if len(text) > WEB_QUERY_MAX_CHARS:
        cut = text[:WEB_QUERY_MAX_CHARS]
        text = cut[: cut.rfind(" ")] if " " in cut else cut
    return text or None
