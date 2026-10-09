"""When a question needs live data, and the search query that may leave the machine (docs/DESIGN.md §3.7).

**Live-data cues** (``live_data_cue``), English, Hindi (Devanagari) and Hinglish, deliberately conservative. Three
kinds, checked in this order:

- *strong time cues* ask for the current value of something live: "right now", "real-time", "live price", "latest
  news", "current share price", "current repo rate", "price today", "the stock doing today", "today's news",
  "लाइव", "मौजूदा भाव". They count even in a question about the documents ("the report says revenue grew 34%; how is
  the stock doing today?"), with two exceptions: the cue's own clause goes on to name the document it asks about
  ("latest price *in the price list*", "real-time monitoring requirements *in the SOP*", "as of today … *per the
  report*"), or it is a technical phrase ("live data feeds", "the live updates section", "real-time monitoring").
  A document named *before* the cue (a premise: "the report says …; how is the stock doing today?") changes nothing.
- *weak time cues* only say "now": "today", "this week", "this morning", "at the moment", "आज", "अभी", "इस समय",
  "फ़िलहाल", "ताज़ा", "aaj", "abhi", "filhaal". Not when they name a meeting, a mail, an invoice or a date ("today's
  agenda", "the agenda for today", "today's town hall", "today's email", "today's date on the invoice", "आज की
  बैठक", "aaj ki presentation").
- *topic cues* name data that is live by nature: "news", "headlines", "latest update", "current price", "share
  price", "stock doing", "market cap", "exchange rate", "weather", "Sensex", "ख़बर", "शेयर का भाव", "khabar".

Weak time cues and topic cues don't count when the question points at the documents ("the report", "the minutes",
"the SOP", "the price list", "according to the…", "रिपोर्ट के अनुसार", a filename) or at a past period or something a
document states ("FY24", "Q3", "2023", "at the end of the year", "on 31 March", "last quarter", "mentioned",
"assumed", "used", "reported", "per unit", "buyback"). Accounting terms carry no cue at all: "current ratio",
"current assets", "current rates of depreciation" (a "current … rate" needs a qualifier such as repo or exchange),
"latest annual report", "अभी तक" (so far), "आजकल" (nowadays); "doing now" needs a stock, shares or market subject.
Romanized Hindi cues count only in Hinglish messages ("Abhi" may be a name). Deliberately left live: "any news on the
dividend?", "latest update on the litigation", "अभी कंपनी का कर्ज़ कितना है?" (nothing says it is about a document).

**The search query** (``web_query``): built only from the router's standalone English question (or an English
utterance that stands alone), and reduced to what asks for the live data:

- only the clauses with a live-data cue; parts and phrases that point at the documents removed ("…, discussed in
  the board minutes", "from the Falcon merger minutes", "according to the report"), filenames and the chat's
  document names removed;
- figures that the user didn't say themselves removed (the router may fold an earlier answer's figures into its
  question). Token by token: a whitespace-separated token with a digit in it ("₹4512cr", "INR4512", "$1.2bn",
  "18.2x", "Rs.4,512", "34pc", "1:2") goes, with the connecting word before it ("of", "to", "from", "Rs", "page")
  and its unit after it ("crore", "million", "per cent"), unless every number in it is in the user's own utterance,
  or it is a year, an FY tag, a quarter or a half ("2024", "FY24", "FY2024-25", "Q3", "Q3FY24", "H1"). Spelled-out
  amounts with a scale word ("four thousand five hundred crore rupees", "paanch hazaar") go too, unless the user said
  them. Punctuation left behind (stray dashes, unbalanced brackets) is cleaned up;
- emails, URLs, long digit runs (phone, account and card numbers) and PAN-like ids removed; one line, at most
  WEB_QUERY_MAX_CHARS;
- a query with nothing left to look up is not sent: fewer than two content words (not counting question and request
  words like "explain", "what", "tell me", nor time words like "today", "now", "as of") and no live topic by itself
  ("news", "weather", "Sensex", "exchange rate" …). "Explain" and "As of today" stay on the machine.

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

# What "today's …" / "…for today" is about when it is not live data: a meeting, a mail, an invoice, a date.
_MEETING_EN = (
    r"(?:meetings?|agenda|minutes|calls?|sessions?|discussions?|presentations?|notes|deliverables|action\s+items?|"
    r"decisions?|reviews?|stand-?ups?|syncs?|board|town[\s-]?halls?|all[\s-]hands|webinars?|e-?mails?|mails?|"
    r"invoices?|newsletters?|date)"
)
_MEETING_HI = "(?:बैठक|मीटिंग|एजेंडा|चर्चा|कॉल|सभा|प्रेजेंटेशन|प्रेज़ेंटेशन|ईमेल|मेल|इनवॉइस|वेबिनार|टाउन\\s*हॉल|तारीख़|तारीख|तारीख़|तिथि)"
_MEETING_HINGLISH = (
    r"(?:meeting|baithak|agenda|call|presentation|email|mail|invoice|webinar|town\s*hall|tareekh|tarikh|date)"
)

# Technical phrases that merely contain a strong cue: "live data feeds", "the live updates section", "real-time
# monitoring requirements".
_TECH_AFTER = (
    r"(?!\s+(?:feeds?|pipelines?|streams?|streaming|sources?|integrations?|connectors?|connections?|apis?|endpoints?|"
    r"sync|synchroni[sz]ation|sections?|features?|modules?|tabs?|pages?|panels?|screens?|widgets?|modes?|"
    r"requirements?|specifications?|specs?|architecture|capabilit(?:y|ies)|functionality|support|systems?|services?|"
    r"engines?|protocols?|latency|processing|monitoring|analytics|dashboards?|alerts?|notifications?|tracking|"
    r"collaboration|communications?|messaging|constraints?|design|mechanisms?|layers?)\b)"
)
# "current exchange rate assumption", "latest price forecast": a figure a document states, not today's value.
_STATED_AFTER = r"(?!\s+(?:assumptions?|projections?|forecasts?|estimates?|scenarios?)\b)"

_STRONG_EN = re.compile(
    rf"""\b(?:
        right\s+now|as\s+of\s+(?:now|today)|
        (?:real[\s-]?time|
            live\s+(?:data|price|prices|score|scores|updates?|news|rate|rates|market|stock|feed)){_TECH_AFTER}|
        latest\s+(?:\w+\s+)?(?:news|price|prices|rate|rates|headlines|quote|score){_STATED_AFTER}|
        current\s+(?:market|stock|share|trading)\s+(?:price|prices|valuation|quote){_STATED_AFTER}|
        current\s+(?:repo|exchange|dollar|rupee|inflation|policy|gold|petrol|market)\s+rates?{_STATED_AFTER}|
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
# "the agenda for today", "minutes from today": the cue's own words say what it is about.
_MEETING_BEFORE = re.compile(
    rf"\b(?:{_MEETING_EN}\s+(?:for|of|from|on|in|at|during|about)\s+(?:the\s+)?|date\s+)$", re.IGNORECASE
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
    rf"\b(?:(?:aaj|aj)(?!\s+(?:ki|ke|ka)\s+{_MEETING_HINGLISH})|abhi(?!\s+tak)|filhaal|filhal|taaza|taza|"
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
    r"memo|presentation|notes|agreement|contract|statements?|transcript|forecast|section|table|offer|policy|"
    r"sops?|manuals?|price\s+lists?|quotations?|invoices?|specs?|specifications?|data\s?sheets?|"
    r"newsletters?|itinerar(?:y|ies)|brochures?|handbooks?|proposals?|spreadsheets?)"
)
_POINTER_NATIVE = (
    r"\b(?:report|document|file|minutes)\s+(?:mein|me|ke\s+anusar)\b|"
    r"(?:रिपोर्ट|दस्तावेज़|दस्तावेज|डॉक्यूमेंट|फ़ाइल|फाइल)\s*(?:में|के\s+अनुसार)"
)
_POINTER_NATIVE_RE = re.compile(_POINTER_NATIVE, re.IGNORECASE)
_DOC_POINTER = re.compile(
    rf"\b(?:the|this|that|these|those|my|our|your|uploaded)\s+(?:[a-z0-9'-]+\s+)?{_DOC_NOUNS}\b|"
    rf"\baccording\s+to\s+the\b|{_POINTER_NATIVE}",
    re.IGNORECASE,
)
# "in the price list", "per the report", "according to the quotation": naming where the answer is to come from.
_SOURCE_PREP = (
    r"(?:in|from|per|within|inside|according\s+to|as\s+per|"
    r"(?:as\s+)?(?:stated|mentioned|listed|given|shown|reported|described|discussed|quoted)\s+in)"
)
_SOURCE_REF = re.compile(
    rf"\b{_SOURCE_PREP}\s+(?:the|this|that|these|those|my|our|your|uploaded)\s+(?:[\w'-]+\s+){{0,3}}?{_DOC_NOUNS}\b",
    re.IGNORECASE,
)
# A past period, or something a document states: the question is about the documents, not live data.
_PAST_OR_STATED = re.compile(
    r"\b(?:FY\s?'?\d{2,4}|Q[1-4]|H[12]|(?:19|20)\d\d|(?:end|close|start)\s+of\s+(?:the\s+)?(?:year|quarter|period|fy)|"
    r"as\s+(?:of|on|at)\s+\d|on\s+\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+|last\s+(?:year|quarter|fiscal)|"
    r"mentioned|assumed|assumptions?|used|stated|reported|disclosed|listed|quoted\s+in|per\s+unit|buyback|"
    r"(?:did|do|does)\s+(?:they|we|you|it|the\s+company)\s+use)\b",
    re.IGNORECASE,
)


def live_data_cue(text: str | None, *, documents: Sequence[str] = ()) -> str | None:
    """The words that make ``text`` ask for live or current data, or None (module docstring). ``documents``: the
    chat's filenames; naming one points at the documents too."""
    if not text or not text.strip():
        return None
    for pattern in (_STRONG_EN, _STRONG_HI):
        for m in pattern.finditer(text):
            if not _names_a_document_after(text, m.end(), documents):
                return _cue(m)
    if about_the_documents(text, documents):
        return None
    if m := _weak_en(text):
        return _cue(m)
    hinglish = message_language(text) == "hi"
    patterns = [_WEAK_HI, *([_WEAK_HINGLISH] if hinglish else []), _TOPIC_EN, _TOPIC_HI]
    if hinglish:
        patterns.append(_TOPIC_HINGLISH)
    for pattern in patterns:
        if m := pattern.search(text):
            return _cue(m)
    return None


def asks_live_figure(text: str | None, *, documents: Sequence[str] = ()) -> bool:
    """A live-data question that asks for a figure or news only the web has right now (a strong cue, or a topic cue:
    "USD to INR today", "the share price right now", "latest news", "आज सोने का भाव"), not merely "today" ("what
    should I cook today?"). Without live data, such a question gets a fixed honest line, never a model's guess
    (quality round, item 2)."""
    if text is None or live_data_cue(text, documents=documents) is None:
        return False
    for strong in (_STRONG_EN, _STRONG_HI):
        if any(not _names_a_document_after(text, m.end(), documents) for m in strong.finditer(text)):
            return True
    patterns = [_TOPIC_EN, _TOPIC_HI, *([_TOPIC_HINGLISH] if message_language(text) == "hi" else [])]
    return any(p.search(text) for p in patterns)


def _cue(m: re.Match[str]) -> str:
    return " ".join(m.group(0).split()).casefold()


def _weak_en(text: str) -> re.Match[str] | None:
    """The first "today" / "this week" … that isn't part of "the agenda for today", "minutes from today"."""
    for m in _WEAK_EN.finditer(text):
        if not _MEETING_BEFORE.search(text[: m.start()]):
            return m
    return None


def _names_a_document_after(text: str, end: int, documents: Sequence[str]) -> bool:
    """The rest of the cue's own clause names the document the question is about ("… in the price list", "per the
    report", a filename): a strong cue then asks about the document, not the web. A document named before the cue is
    only a premise ("the report says …; how is the stock doing today?")."""
    tail = _CLAUSES.split(text[end:], maxsplit=1)[0]
    if _SOURCE_REF.search(tail) or _POINTER_NATIVE_RE.search(tail):
        return True
    return _names_a_file(tail, documents)


def points_at_documents(text: str, documents: Sequence[str] = ()) -> bool:
    """The text refers to the user's documents ("the report", "the minutes", "according to the…", "रिपोर्ट में", a
    filename)."""
    return bool(_DOC_POINTER.search(text)) or _names_a_file(text, documents)


def _names_a_file(text: str, documents: Sequence[str]) -> bool:
    """The text names one of the chat's documents ("annual report", "annual_report.pdf")."""
    low = text.casefold().replace("_", " ").replace("-", " ")
    return any(len(stem) > 3 and stem in low for stem in _stems(documents))


def about_the_documents(text: str, documents: Sequence[str] = ()) -> bool:
    """Points at the documents, at a past period, or at something a document states (module docstring)."""
    return points_at_documents(text, documents) or bool(_PAST_OR_STATED.search(text))


def _stems(documents: Sequence[str]) -> list[str]:
    return [n.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").casefold().strip() for n in documents]


# ------------------------------------------------------------------ the search query

_DASHES = "\N{EM DASH}\N{EN DASH}"
# Lines, sentences, dashes, semicolons. A full stop ends a sentence only before a capital letter, a quote or a
# bracket: "Rs. 4,512 crore" and "the U.S. dollar" stay in one piece.
_CLAUSES = re.compile(rf"\n+|(?<=[;!?])\s+|(?<=\.)\s+(?=[A-Z\"'(\[ऀ-ॿ])|\s+[{_DASHES}]\s+|\s+-\s+|;\s*")
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
_DANGLING = re.compile(
    r"\b(?:of|at|to|by|was|is|were|are|from|with|about|around|than|and|for|in|its|their|a|an|the)\s*(?=[?.!,;:)]|$)",
    re.IGNORECASE,
)
_OPENER = re.compile(r"^(?:and|but|so|also|then|now|okay|ok|well|great|thanks)\b[\s,]*", re.IGNORECASE)
_STRAY = re.compile(r"(?<!\S)[-\u2013\u2014&/+~=*#|\\:;,]+(?!\S)")  # punctuation standing alone between spaces


def web_query(question: str | None, *, documents: Sequence[str] = (), utterance: str | None = None) -> str | None:
    """The search query for an English standalone question (module docstring), or None when there is nothing
    English, or nothing worth looking up, to search for. ``utterance``: what the user actually said (figures in it
    may stay)."""
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
            and (live_data_cue(p, documents=documents) is not None or not _scrub_figures(p, utterance)[1])
        ]
        kept.extend(about_live or parts)
    text = " ".join(" ".join(kept).split())
    text = _FILENAME.sub(" ", text)
    for stem in _stems(documents):  # the chat's document names, e.g. "(falcon merger board minutes)"
        if len(stem) > 3:
            text = re.sub(re.escape(stem).replace("\\ ", r"[\s_-]+"), " ", text, flags=re.IGNORECASE)
    for pattern in (_POINTER_PHRASE, _EMAIL, _URL, _LONG_DIGITS, _ID_LIKE):
        text = pattern.sub(" ", text)
    text = _tidy(_scrub_figures(_tidy(text), utterance)[0])
    text = _tidy(_OPENER.sub("", text))
    if len(text) > WEB_QUERY_MAX_CHARS:
        cut = text[:WEB_QUERY_MAX_CHARS]
        text = cut[: cut.rfind(" ")] if " " in cut else cut
    return text if _worth_searching(text) else None


# ------------------------------------------------------------------ figures the user didn't say


def _wordset(text: str) -> frozenset[str]:
    return frozenset(text.split())


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EDGE_PUNCTUATION = "()[]{}<>\"'\u201c\u201d\u2018\u2019.,;:!?"
_SENTENCE_END = re.compile(r"[?!.]+$")
# A year, an FY tag, a quarter or a half: a period, not a figure ("2024", "2024-25", "FY24", "FY2024-25", "Q3",
# "Q3'24", "3Q24", "H1", "Q3FY24").
_PERIOD = re.compile(
    r"""(?:
        (?:19|20)\d\d(?:[-\u2013/](?:19|20)?\d\d)?
      | (?:Q[1-4]|[1-4]Q|H[12]|[12]H)(?:['\u2019-]?(?:(?:19|20)\d\d|\d\d))?
      | (?:Q[1-4]|H[12])?(?:FY|CY)['\u2019-]?\d{2}(?:\d{2})?(?:[-\u2013/]\d{2}(?:\d{2})?)?
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_YEAR = re.compile(r"(?:19|20)\d\d(?:[-\u2013/](?:19|20)?\d\d)?")
_FY_NUMBER = re.compile(r"'?\d{2}(?:\d{2})?(?:[-\u2013/]\d{2}(?:\d{2})?)?")  # the "24" of "FY 24"
_CURRENCY = frozenset({"\u20b9", "$", "\u20ac", "\u00a3", "rs", "inr", "usd", "eur", "gbp", "us$"})
# Words that lead into a figure, left over when it goes: "revenue of Rs 4,512 crore", "rose from 3,367 to 4,512",
# "(see page 12)".
_BEFORE_FIGURE = _CURRENCY | _wordset(
    "of at by to from around about over under nearly almost approximately approx roughly above below near ~ \u2248 + "
    "page pages pg slide slides"
)
_SCALE_WORDS = _wordset(
    "hundred hundreds thousand thousands lakh lakhs lac lacs crore crores million millions billion billions trillion "
    "trillions percent sau hazaar hazar hajaar karod arab kharab"
) | {"per cent"}
# Units that follow a figure: "4,512 crore", "34 per cent", "1.2 bn", "5 rupees", "3 pm".
_UNITS = _SCALE_WORDS | _wordset(
    "cr mn mln bn bln tn pc pct bps bp rupees rupee dollars dollar euros euro pounds rs inr usd eur gbp am pm"
)
_NUMBER_WORDS = _wordset(
    "a an zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety half "
    # Hindi, romanized ("do", "char" and "das" are left out: ordinary English words)
    "ek teen chaar paanch panch chhe chhah saat aath nau gyarah barah terah chaudah pandrah solah satrah atharah "
    "unnis bees pachees tees chalis pachaas pachas saath sattar assi nabbe dedh dhai saadhe sadhe sawa paune"
)
_JOIN_WORDS = frozenset({"and", "point"})  # "four point five lakh", "two and a half crore"


def _core(token: str) -> str:
    return token.strip(_EDGE_PUNCTUATION)


def _word(token: str) -> str:
    return _core(token).casefold()


def _kind(token: str) -> str | None:
    """ "scale" for crore / million / hundred / percent …, "num" for a spelled-out number word, else None."""
    word = _word(token)
    if word in _SCALE_WORDS:
        return "scale"
    return "num" if all(part in _NUMBER_WORDS for part in word.split("-")) else None


def _spelled_amount_end(tokens: list[str], i: int) -> int:
    """The end of a spelled-out amount that has a scale word and starts at ``tokens[i]`` ("four thousand five hundred
    crore", "two and a half lakh", "a million", "paanch hazaar"), else ``i``."""
    n, j = len(tokens), i
    scale = number = False
    while j < n:
        kind = _kind(tokens[j])
        if kind is None:
            if j > i and _word(tokens[j]) in _JOIN_WORDS and j + 1 < n and _kind(tokens[j + 1]) is not None:
                j += 1
                continue
            break
        scale, number = scale or kind == "scale", number or kind == "num"
        j += 1
        if tokens[j - 1][-1] in ",;:.?!)":  # an amount ends at punctuation
            break
    return j if scale and number and _kind(tokens[i]) is not None else i


def _scrub_figures(text: str, utterance: str | None) -> tuple[str, bool]:
    """``text`` without the figures the user didn't say (an earlier answer's, folded into the question by the
    router), and whether any went. A whitespace-separated token with a digit goes unless every number in it is in the
    utterance, or it is a period (a year, FY tag, quarter or half); spelled-out amounts with a scale word go unless the
    utterance has the same words. With a figure go the words that led into it and its unit ("revenue of Rs 4,512
    crore" → "revenue")."""
    said = {n.replace(",", "") for n in _NUMBER.findall(utterance or "")}
    said_words = " ".join((utterance or "").casefold().replace("-", " ").split())
    tokens = _merge_per_cent(text.split())
    out: list[str] = []
    dropped = False
    i, n = 0, len(tokens)

    def drop(end: int) -> int:
        """Drop the figure (up to ``end``) with the words that led into it and its unit; where the next token is."""
        nonlocal dropped
        dropped = True
        while out and _word(out[-1]) in _BEFORE_FIGURE:
            out.pop()
        last = tokens[end - 1]
        while end < n and _word(tokens[end]) in _UNITS:
            last, end = tokens[end], end + 1
        if (ending := _SENTENCE_END.search(last)) and out and not _SENTENCE_END.search(out[-1]):
            out[-1] += ending.group(0)  # the question mark of "… 4,512 crore?" stays
        return end

    while i < n:
        token = tokens[i]
        if (end := _spelled_amount_end(tokens, i)) > i:
            phrase = " ".join(_word(t).replace("-", " ") for t in tokens[i:end])
            if phrase in said_words:  # the user said it themselves
                out.extend(tokens[i:end])
                i = end
            else:
                i = drop(end)
        elif any(ch.isdigit() for ch in token) and not _figure_stays(tokens, i, out, said):
            i = drop(i + 1)
        else:
            out.append(token)
            i += 1
    return " ".join(out), dropped


def _merge_per_cent(tokens: list[str]) -> list[str]:
    merged: list[str] = []
    for token in tokens:
        if merged and _word(token) in ("cent", "cents") and _word(merged[-1]) == "per":
            merged[-1] = f"{merged[-1]} {token}"
        else:
            merged.append(token)
    return merged


def _figure_stays(tokens: list[str], i: int, out: list[str], said: set[str]) -> bool:
    """A token with a digit that may stay: its numbers are the user's own, or it is a period."""
    core = _core(tokens[i])
    numbers = {n.replace(",", "") for n in _NUMBER.findall(core)}
    if numbers and numbers <= said:
        return True
    previous = _word(out[-1]) if out else ""
    if previous in ("fy", "cy") and _FY_NUMBER.fullmatch(core):  # "FY 24"
        return True
    if not _PERIOD.fullmatch(core):
        return False
    if _YEAR.fullmatch(core):  # a year-looking number is a figure with a currency before it or a unit after it
        following = _word(tokens[i + 1]) if i + 1 < len(tokens) else ""
        return previous not in _CURRENCY and following not in _UNITS
    return True


def _balanced(text: str) -> str:
    """``text`` without brackets that lost their partner: "(see annual report FY24," → "see annual report FY24,"."""
    stack: list[int] = []
    unmatched: set[int] = set()
    for i, ch in enumerate(text):
        if ch in "([":
            stack.append(i)
        elif ch in ")]":
            if stack and text[stack[-1]] == {")": "(", "]": "["}[ch]:
                stack.pop()
            else:
                unmatched.add(i)
    unmatched.update(stack)
    return "".join(ch for i, ch in enumerate(text) if i not in unmatched)


def _tidy(text: str) -> str:
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    text = " ".join(text.split())
    previous = None
    while previous != text:
        previous = text
        text = _balanced(text)
        text = _STRAY.sub(" ", text)
        text = _DANGLING.sub("", text)
        text = re.sub(r"\s+([?.!,;:)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"([,;:])(?=[?.!,;:]|$)", "", text)
        text = " ".join(text.split())
    return text.strip(" ,;:-")


# ------------------------------------------------------------------ nothing left to look up

_WORDS = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")
# Question and request words, time words and cue qualifiers: the frame of a live question, not what it asks about.
_FRAME = _wordset(
    """a an the and or but if so nor of at to by for in on with without about around from into onto over under as than
    is are was were be been being am do does did has have had will would can could should may might shall must
    i me my mine we us our you your it its they them their he she him her his this that these those there here
    what whats which who whom whose when where why how hows whether also just only very really
    explain tell show give describe summarize summarise list find get check look say says said please kindly know
    let lets want need like help any some anything something everything
    today todays tonight now currently current latest recent recently live real time right moment week morning
    yesterday yesterdays tomorrow soon fy cy q h"""
)


def _worth_searching(text: str) -> bool:
    """The query asks about something: at least two content words, or a live topic by itself ("latest news",
    "weather today", "Sensex"). "Explain" and "As of today" are not worth a search."""
    words = (m.casefold().replace("'", "").replace("\u2019", "") for m in _WORDS.findall(text))
    content = [w for w in words if w not in _FRAME]
    return len(content) >= 2 or bool(_TOPIC_EN.search(text))
