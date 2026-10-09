"""Checks on an answer while it streams, before the user hears it (docs/DESIGN.md §3.4, "Answer checks", quality round).

The 4B model sometimes says what the screen and the sources contradict, misspells what it should copy, or runs long.
``AnswerGuard`` sits between the model's stream and the turn's deltas (so between the model and the speaker) and lets
text through once it has been checked:

- **Coverage** (item 1): a sentence saying the documents don't cover something that a grounded visual on screen or a
  strong passage does cover ("the report does not provide quarterly figures for FY24" with the FY24 quarters on the
  chart). The first sentence is asked again once, with the evidence named; a second denial is replaced by a fixed
  sentence pointing at the evidence ("The chart on screen shows …"), a later one is dropped (or its denying clause).
  "Covers" is code, not the model: one row of the visual's table, or one sentence or table row of a strong passage,
  states every fiscal period, every number and every content word the denial names (for a denial that names nothing,
  the question's). A denial of something no such row or sentence states stands: the documents may really not have it
  ("FY25 revenue" against FY24 tables, "an EBITDA margin target" against the margin). Strict on purpose: asking the
  model again over a true denial invites it to make the answer up.
- **Live figures** (item 2): a question that wants live data and gets none (web search off or unavailable): a sentence
  with a figure that is in no source and not in the question is replaced, once, by a fixed honest line.
- **Identifiers** (item 3): a long code (CIN, ISIN, GSTIN, account numbers) that is in no source but within two
  characters of one that is, is replaced by the source's; one near nothing is recorded as unverified.
- **Names** (item 10): a name the user was misheard as ("Wall Mora") is written as the documents spell it.
- **Length** (item 7): a short answer stops at the end of the sentence that reaches ``max_words``, or at
  ``max_sentences``, never inside a sentence.

What is held, and for how long: with a coverage check, the first sentence goes out ``LAG_WORDS`` (3) words behind the
model until it holds a negation ("not", "only", "no", "नहीं", "nahi"…, which every denial needs) or while its subject
is a document whose verb hasn't come yet ("The Valmora annual report …" may go on "does not mention"; "… says"
releases it); from a negation on, it is held to its end and checked. Hindi, whose negation comes last, is released the
same way (3 words behind, whole from its first negation, apology or "केवल"), except that a sentence *about what the
documents contain* ("रिपोर्ट में …", "… की जानकारी …", "इस बारे में …", a contrast "…, लेकिन …") is held to its
clause's final auxiliary (है, हैं, था, थे, होगा…; ``hindi_hold``): a negation always precedes it, so a denial is
seen before any of its clause is out, and a clause with its auxiliary and no negation goes out at once. Later
sentences are held whole: the voice session synthesises whole sentences after its first chunk anyway, so speech pays
only the first chunk's wait (~3 words, or a document subject's verb). A live-figure check holds whole sentences (such
answers are rare). Without either, only a word that may still be an identifier or a misheard name is held.
Every change is recorded (``checks``: ``route.checks`` on the saved message).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .language import wordset
from .retrieval import asked_periods, stated_periods
from .sources import clauses, not_covered_at, says_not_covered, strip_markers
from .subjects import respell

LAG_WORDS = 3  # with a coverage check, text goes out this many words behind the model (until a negation)

Verdict = Literal["retry", "stop"]


@dataclass(frozen=True, slots=True)
class Released:
    """What the guard lets through now, and whether the model should be asked again or stopped."""

    text: str = ""
    verdict: Verdict | None = None


# ------------------------------------------------------------------ words, numbers, periods

_MARKERS = re.compile(r"\[[^\]]*\]")
_DEVANAGARI = re.compile("[\u0900-\u097f]")
_TOKEN = re.compile(r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+(?:[.,]\d+)*|[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
_FY_TOKEN = re.compile(r"^(?:fy\d{2,4}|q[1-4]fy\d{2,4}|(?:19|20)\d\d)$", re.IGNORECASE)
_WORD_STOP = wordset(
    """
    the and are was were been being does did not don doesn didn isn aren wasn weren any this that these those there
    their its for from with about into onto only also full complete specific specifically explicitly separate
    separately detailed details detail information info data figure figures number numbers value values amount
    amounts breakdown break down provide provides provided providing mention mentions mentioned cover covers covered
    include includes included contain contains contained state states stated give gives given list lists listed
    disclose discloses disclosed show shows shown specify specifies specified available unavailable say says said
    provision provisions rule rules guideline guidelines employee employees staff people person but yet
    crore crores lakh lakhs million billion thousand
    percent cent rupee rupees inr usd
    have has had find found report reports reported document documents source sources deck policy policies text
    passage passages table tables file files annual investor presentation statement statements company sorry
    unfortunately however although though while whereas which what how much many can cannot could would should will
    may might they them our your his her its such some all each every other more most less than then there here
    where when who whom whose why either neither nor yes per one two
    """
)


def stem(word: str) -> str:
    """A word as compared: lower case, without "'s" or "n't" ("doesn't" → "doesn"), "-ies" → "-y", "-ly" and a
    plural "-s" dropped ("quarterly" → "quarter", "cities" → "city", "provisions" → "provision")."""
    w = re.split(r"['\u2019]", word.casefold())[0]
    if w.endswith("n") and re.search(r"n['\u2019]t$", word.casefold()):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ly") and len(w) > 5:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def number_key(token: str) -> str | None:
    """A number as compared: "1,933" → "1933", "15.00" → "15", "83.50" → "83.5"; None for a non-number."""
    plain = token.replace(",", "")
    try:
        value = float(plain)
    except ValueError:
        return None
    return f"{value:.6f}".rstrip("0").rstrip(".")


@dataclass(frozen=True, slots=True)
class Terms:
    """What a sentence (or a unit of evidence) names: fiscal periods, numbers and codes ("l3", "q4", "5,500"), and
    content words (stemmed)."""

    periods: frozenset[int]
    numbers: frozenset[str]
    words: frozenset[str]

    @property
    def empty(self) -> bool:
        return not (self.periods or self.numbers or self.words)


def terms(text: str, *, ignore: Iterable[str] = (), stated: bool = False) -> Terms:
    """The terms of ``text`` (citation markers ignored). ``stated``: periods as a passage states them (generous:
    "31 March 2024" is FY24, a bare year either fiscal year it can be in), else as a question names them."""
    text = _MARKERS.sub(" ", text)
    periods = stated_periods(text) if stated else asked_periods(text)
    numbers: set[str] = set()
    words: set[str] = set()
    skip = {stem(w) for w in ignore}
    for m in _TOKEN.finditer(text):
        token = m.group(0)
        if _FY_TOKEN.match(token):
            continue
        if token[0].isdigit():
            key = number_key(token)
            if key is not None:
                numbers.add(key)
        elif any(ch.isdigit() for ch in token):
            numbers.add(token.casefold())
        elif len(token) >= 3:
            w = stem(token)
            if w not in _WORD_STOP and token.casefold() not in _WORD_STOP and w not in skip:
                words.add(w)
    return Terms(frozenset(periods), frozenset(numbers), frozenset(words))


def _word_matches(word: str, words: frozenset[str]) -> bool:
    return word in words or (
        len(word) >= 5 and any(len(w) >= 5 and (w.startswith(word) or word.startswith(w)) for w in words)
    )


# ------------------------------------------------------------------ coverage (item 1)

_DENIAL_CUE = re.compile(
    r"\b(?:not|no|none|nothing|neither|nor|cannot|unable|unfortunately|sorry|only|without|lacks?|missing|unavailable|"
    r"absent|unclear|nah(?:i|in|ee)|nhi|uplabdh|ullekh)\b|n['\u2019]t\b",  # (the last four: romanized Hindi)
    re.IGNORECASE,
)


# A sentence about a document ("The Valmora annual report …", "The policy …") may be about to say it lacks something:
# its subject is held until a verb that reports what it says follows ("says", "shows", "lists"…), or a negation.
_DOCUMENT_NOUNS = wordset(
    """
    report reports document documents deck decks policy policies presentation filing minutes statement statements
    source sources passage passages manual handbook sop contract agreement table tables
    """
)
_REPORTING_VERBS = wordset(
    """
    says said say states stated shows showed show notes noted reports reported mentions mentioned lists listed gives
    gave puts records recorded highlights explains explained describes described provides provided sets confirms
    indicates indicated specifies specified puts shows estimates expects puts according
    """
)


def awaits_its_verb(text: str) -> bool:
    """The sentence so far opens with a document as its subject and hasn't said what the document does yet."""
    words = [re.sub(r"[^a-z]", "", w.casefold()) for w in _mask(text).split()[:9]]
    at = next((i for i, w in enumerate(words) if w in _DOCUMENT_NOUNS), None)
    if at is None or {"according", "per", "in", "from", "as"} & set(words[:at]):  # not its subject: "according to
        return False  # the report, …", "in the deck, …"
    return not any(w in _REPORTING_VERBS for w in words[at + 1 :])


# Hindi puts its verb last: "रिपोर्ट में FY25 की जानकारी उपलब्ध नहीं है।" says what it is about first, and only its last
# words say whether it denies it. English is released a few words behind the model because its negation comes early; a
# Hindi sentence is checked by its structure instead (``hindi_hold``): one about what the documents contain (it names a
# document, or availability: "रिपोर्ट", "जानकारी", "उल्लेख", "उपलब्ध"…) goes out clause by clause, each when its final
# auxiliary (है, हैं, था, थे, होगा…, which a negation always precedes) has come without one; any other sentence goes
# out LAG_WORDS behind the model, as English does, and is held whole from the moment a negation or apology shows up.
_HI_PUNCT = ",.;:!?\u0964\u0965\"'\u201c\u201d\u2018\u2019()[]{}\u2014\u2013-\u2026"


def hindi(text: str) -> str:
    """Devanagari as compared: without nuktas ("दस्तावेज़" = "दस्तावेज") and with the chandrabindu as an anusvara
    ("हूँ" = "हूं")."""
    return unicodedata.normalize("NFD", text).replace("\u093c", "").replace("\u0901", "\u0902")


def _hindi_words(text: str) -> frozenset[str]:
    return frozenset(hindi(w) for w in text.split())


_HI_CUE = re.compile(  # on hindi(text): a negation, an apology or a restriction
    "नही|नहि|अनुपलब्ध|अनुल्लेख|केवल|सिर्फ|खेद|माफ|क्षमा|दुर्भाग्य|अफसोस|अभाव|गायब",
)
_HI_AUXILIARIES = _hindi_words("है हैं हूं था थी थे थीं होगा होगी होंगे होंगी")  # the verb group's last word
_HI_ABOUT_STEMS = (  # starts a word: what a "not covered" sentence names ("रिपोर्ट में", "दस्तावेज़ों", "उपलब्ध नहीं")
    *_hindi_words(
        "रिपोर्ट दस्तावेज डॉक्यूमेंट स्रोत प्रस्तुति प्रेजेंटेशन पॉलिसी नीति फाइल तालिका टेबल अनुच्छेद स्लाइड मैनुअल "
        "हैंडबुक अनुबंध समझौत डेक जानकारी उल्लेख उपलब्ध मौजूद दर्ज जिक्र विवरण ब्योरा ब्यौरा शामिल आंकड डेटा"
    ),
)
# whole words: "इस बारे में", "मुझे पता नहीं", and a contrast, which so often introduces what the documents lack ("Q4 का
# राजस्व ₹1,711 करोड़ था, लेकिन FY23 की जानकारी उपलब्ध नहीं है")
_HI_ABOUT_WORDS = _hindi_words("पता बारे संबंध सम्बन्ध विषय मुझे लेकिन परंतु परन्तु किंतु किन्तु मगर जबकि हालांकि हालाकि")
_HI_ATTRIBUTION = _hindi_words("अनुसार मुताबिक हिसाब आधार")  # "रिपोर्ट के अनुसार …": the report is named, not described


def _hindi_tokens(text: str) -> list[tuple[str, int, bool]]:
    """The words of ``text`` as ``(word as compared, end offset, complete)``: a word is complete once whitespace or
    punctuation follows it (the last one may still be growing: "है" → "हैं")."""
    out = []
    for m in _WORD.finditer(_mask(text)):
        raw = m.group(0)
        out.append((hindi(raw.strip(_HI_PUNCT)), m.end(), m.end() < len(text) or raw[-1] in _HI_PUNCT))
    return out


def hindi_hold(text: str) -> tuple[bool, int]:
    """For a Hindi sentence so far with no denial cue in it: ``(about, safe)``. ``about``: it is about what the
    documents contain, so it may still end "… उपलब्ध नहीं है" (held until a clause's verb has come). ``safe``: how far
    the text can be released then: the end of its last complete final auxiliary (0: none yet)."""
    tokens = _hindi_tokens(text)
    about = False
    safe = 0
    for i, (word, end, complete) in enumerate(tokens):
        if word in _HI_AUXILIARIES and complete:
            safe = end
        elif word in _HI_ABOUT_WORDS or word.startswith(_HI_ABOUT_STEMS) or word.casefold() in _DOCUMENT_NOUNS:
            ahead = tokens[i + 1 : i + 3]
            if not (
                len(ahead) == 2 and ahead[0][0] == "के" and ahead[1][0] in _HI_ATTRIBUTION and ahead[1][2]
            ):  # "रिपोर्ट के अनुसार": an attribution, not the subject
                about = True
    return about, safe


def has_denial_cue(text: str) -> bool:
    """A negation or restriction that a "the documents don't cover it" sentence needs (English, Hindi, romanized
    Hindi)."""
    plain = _MARKERS.sub(" ", text)
    return bool(
        _DENIAL_CUE.search(plain)
        or _HI_CUE.search(hindi(plain))
        or any(word == "न" for word, _, _ in _hindi_tokens(text))
    )


@dataclass(frozen=True, slots=True)
class Unit:
    """One row of a table (with its header and heading) or one sentence of a passage (with its heading)."""

    text: str
    terms: Terms
    figure: bool  # states a number that is not a period


def evidence_units(text: str, heading: str = "") -> list[Unit]:
    """A passage's units: a markdown table's rows (each with the header row, the caption above the table and the
    heading), else its sentences."""
    from .sources import split_sentences

    heading = re.sub(r"(?<![A-Za-z0-9])\d+(?:\.\d+)*\.?", " ", heading)  # section numbers ("5.1") state nothing
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = [line for line in lines if line.startswith("|")]
    texts: list[str] = []
    if len(rows) >= 2:
        header = rows[0]
        caption = " ".join(line for line in lines[: lines.index(header)] if not line.startswith("|"))
        caption = re.sub(r"[\d.,]+", " ", caption)  # its words name the table; its numbers aren't the row's
        body = [r for r in rows[1:] if not re.fullmatch(r"[|:\-\s]+", r)]
        texts += [f"{heading} {caption} {header} {r}" for r in body]
        rest = " ".join(line for line in lines if not line.startswith("|"))
        texts += [f"{heading} {s}" for s in split_sentences(rest)]
    else:
        texts += [f"{heading} {s}" for s in split_sentences(text)]
    out = []
    for t in texts:
        found = terms(t, stated=True)
        out.append(Unit(t, found, bool(found.numbers)))
    return out


@dataclass
class Coverage:
    """What answers the question already: the tables of the visual on screen and the strong passages, as units.

    ``correction`` replaces a first sentence that still denies them after the model was asked again; ``retry_note``
    is what the model is told then."""

    units: list[Unit]
    question: Terms
    correction: str
    retry_note: str
    names: frozenset[str] = frozenset()  # document name words: they say whose document, not what is in it
    visual: bool = False

    def contradicted_by(self, sentence: str) -> bool:
        """``sentence`` says the documents don't have what the evidence states: for each fiscal period it names (or
        once, if none), one unit states that period and every number and content word it names."""
        if not says_not_covered(sentence):
            return False
        said = terms(sentence, ignore=self.names)
        if said.empty:
            said = self.question  # "the documents don't cover this": it is about the question
        words = said.words or self.question.words
        if not words:
            return False  # what is denied can't be told (a Hindi denial of a question without an English query)

        def stated(period: int | None) -> bool:
            for unit in self.units:
                t = unit.terms
                if not unit.figure or (period is not None and period not in t.periods):
                    continue
                if said.numbers <= t.numbers and all(_word_matches(w, t.words) for w in words):
                    return True
            return False

        return all(stated(p) for p in said.periods) if said.periods else stated(None)


def drop_denial(sentence: str, released: str = "") -> str | None:
    """The sentence up to the clause that starts the denial ("Q4 revenue was ₹1,933 crore [S2], but the deck doesn't
    give FY23." → "Q4 revenue was ₹1,933 crore [S2]."), when that keeps three words or more and all of ``released``
    (what was already said of it); else None."""
    at = not_covered_at(sentence)
    if at is None:
        return None
    cut = None
    pos = 0
    for part in clauses(sentence):
        pos = sentence.find(part, pos)
        if pos > at:
            break
        if part and pos > 0:
            cut = pos
        pos += len(part)
    if cut is None or cut < len(released):
        return None
    kept = sentence[:cut].rstrip(" ,;:\u2014\u2013-")
    kept = re.sub(r"\s+(?:but|however|although|though|while|whereas)$", "", kept, flags=re.IGNORECASE)
    if len(strip_markers(kept).split()) < 3:
        return None
    return kept if kept.endswith((".", "!", "?", "।")) else kept + ("।" if re.search("[ऀ-ॿ]", kept) else ".")


# ------------------------------------------------------------------ identifiers (item 3)

_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{8,}(?![A-Za-z0-9])"
    r"|(?<![\d,.])\d{10,}(?![\d,.])"
)


def identifiers_in(texts: Iterable[str]) -> list[str]:
    """Codes in the sources: letters and digits, eight or more ("L24119GJ1994PLC023871", "INE413T01011"), or ten or
    more digits."""
    found: dict[str, None] = {}
    for text in texts:
        for m in _IDENTIFIER.finditer(text):
            found.setdefault(m.group(0), None)
    return list(found)


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def nearest_identifier(token: str, known: Sequence[str]) -> str | None:
    """The one source code ``token`` is a slip of (at most two characters off, one per eight), or None."""
    if token in known:
        return None
    limit = max(1, len(token) // 8)
    scored = [(_distance(token.upper(), k.upper()), k) for k in known if abs(len(k) - len(token)) <= 1 and k != token]
    close = sorted((d, k) for d, k in scored if d <= min(limit, 2))
    if not close or (len(close) > 1 and close[0][0] == close[1][0]):
        return None
    return close[0][1]


# ------------------------------------------------------------------ live figures (item 2)

_FIGURE = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?![A-Za-z0-9])")
_PERIODISH = re.compile(r"(?<![\d,.])(?:19|20)\d\d(?:[-\u2013/](?:19|20)?\d\d)?(?![\d,]|\.\d)")


def figures_in(texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for text in texts:
        for m in _FIGURE.finditer(_MARKERS.sub(" ", text)):
            key = number_key(m.group(0))
            if key is not None:
                out.add(key)
    return out


def unsupported_figures(sentence: str, allowed: set[str]) -> list[str]:
    """Figures of ``sentence`` that are in no source and not in the question (years and small counts aside)."""
    out = []
    text = _PERIODISH.sub(" ", _MARKERS.sub(" ", sentence))  # "2023-24", "2024": periods, not figures
    for m in _FIGURE.finditer(text):
        token = m.group(0)
        if _PERIODISH.fullmatch(token) or text[max(0, m.start() - 2) : m.start()].casefold().endswith(("fy", "q", "h")):
            continue
        key = number_key(token)
        if key is None or key in allowed:
            continue
        if "." not in key and int(key) < 10:
            continue
        out.append(token)
    return out


# ------------------------------------------------------------------ a general answer's own disclaimer (item 1)

# What the model writes when it says the answer isn't from the documents, at its start, after the app has said so
# itself ("Not from your documents, but …" / "यह आपके दस्तावेज़ों से नहीं है, लेकिन …").
_DISCLAIMER = re.compile(
    r"^\s*(?:(?:this|that|it)\s+(?:is\s+not|isn't|is\s+n't|does\s+not\s+come|doesn't\s+come)\s+|not\s+)"
    r"(?:from|in)\s+(?:your|the)\s+(?:uploaded\s+)?documents?\b[\s,.;:\u2014-]*(?:but\b[\s,]*)?"
    r"|^\s*(?:यह\s+)?(?:जानकारी\s+)?(?:आपके|आपकी|इन)\s+(?:दस्तावेज़ों|दस्तावेजों|दस्तावेज़|दस्तावेज|डॉक्यूमेंट्स?)\s+"
    r"(?:से|में)\s+(?:नहीं|नही)\s*(?:है|हैं|मिली|मिलती)?[\s,।.;:\u2014-]*(?:(?:लेकिन|परंतु|पर|मगर)\b[\s,]*)?",
    re.IGNORECASE,
)


def drop_disclaimer(text: str) -> str:
    """``text`` without a "not from your documents" opening (one, at its start); the rest's first letter as it was."""
    m = _DISCLAIMER.match(text)
    if not m or not m.group(0).strip():
        return text
    rest = text[m.end() :]
    return rest if rest.strip() else ""


# ------------------------------------------------------------------ the guard

_SENTENCE_PUNCT = ".!?\u0964\u0965"
_CLOSERS = "\"'\u201d\u2019)\u00bb"
_ABBREVIATIONS = wordset("rs mr mrs ms dr vs e.g i.e etc approx inc ltd co st fig no")
_INITIALISM = re.compile(r"(?:[A-Za-z]\.){2,}")
_LIST_NUMBER = re.compile(r"\(?\d{1,2}[.)]")
_WORD = re.compile(r"\S+")
# Sentence openers lower-cased after a fixed prefix ("Not from your documents, but the capital of France is Paris.").
_COMMON_STARTERS = wordset(
    """
    the a an it in this that these those there generally typically usually most many some yes no according as for on
    at by its their today currently if when while although based historically one
    """
)


def _mask(text: str) -> str:
    """Citation markers blanked out (same length): "[S1, S2]" holds no sentence end."""
    return _MARKERS.sub(lambda m: " " * len(m.group(0)), text)


def sentence_end(text: str) -> int | None:
    """Where the first complete sentence of ``text`` ends (after its punctuation and closing quotes; a line break ends
    one too), or None. Punctuation ends a sentence only before whitespace, and never after "Rs.", "U.S." or "1."."""
    masked = _mask(text)
    for i, ch in enumerate(masked):
        if ch == "\n":
            if masked[:i].strip():
                return i
            continue
        if ch not in _SENTENCE_PUNCT:
            continue
        j = i + 1
        while j < len(masked) and masked[j] in _CLOSERS:
            j += 1
        if j >= len(masked) or not masked[j].isspace():
            continue
        if ch == ".":
            before = masked[: i + 1].split()
            last = before[-1].strip(_CLOSERS + "(") if before else ""
            if (
                last.lower().rstrip(".") in _ABBREVIATIONS
                or _INITIALISM.fullmatch(last)
                or _LIST_NUMBER.fullmatch(last)
            ):
                continue
        return j
    return None


def _words(text: str) -> int:
    return sum(1 for w in strip_markers(text).split() if any(ch.isalnum() for ch in w))


@dataclass
class AnswerGuard:
    """The checks of one answer (module docstring). Feed it the model's pieces; send on what it releases.

    ``feed`` / ``finish`` return the text to send now and, possibly, a verdict: ``"retry"`` (close the stream and ask
    the model again with ``coverage.retry_note``, then call ``restart``) or ``"stop"`` (close the stream: the answer
    is complete)."""

    coverage: Coverage | None = None
    identifiers: Sequence[str] = ()
    figures: set[str] | None = None  # allowed figures (sources and question): live-data and general answers
    live_line: str = ""  # replaces a sentence with a figure from nowhere
    figure_check: str = "live_figure"  # its name in ``checks``: "live_figure", or "general_figure" (item 1)
    # The answer follows the fixed "Not from your documents, but …": the model's own disclaimer at its start is dropped
    # (the last real run spoke it twice, in Hindi).
    after_disclaimer: bool = False
    renames: Mapping[str, str] = field(default_factory=dict)  # misheard name → the documents' spelling
    max_words: int | None = None
    max_sentences: int | None = None
    lower_first: bool = False  # the answer follows a fixed prefix: its first word in lower case when common
    checks: list[dict[str, Any]] = field(default_factory=list)
    attempt: int = 1

    def __post_init__(self) -> None:
        self.sentence_hold = self.coverage is not None or self.figures is not None or self.after_disclaimer
        self.lag_release = self.coverage is not None and self.figures is None
        self.token_hold = bool(self.identifiers) or bool(self.renames)
        self._phrase_words = max((len(k.split()) for k in self.renames), default=1)
        self._reset()

    def _reset(self) -> None:
        self.buffer = ""  # not yet released
        self.current = ""  # released part of the sentence being written
        self.sentences = 0
        self.words = 0
        self.released = False
        self._last = ""  # the last character released
        self.done = False
        self.live_said = False

    def restart(self) -> None:
        """The model is asked again (nothing of its first answer was released)."""
        self.attempt += 1
        self._reset()

    @property
    def active(self) -> bool:
        return self.sentence_hold or self.token_hold or self.max_words is not None or self.max_sentences is not None

    # -------------------------------------------------------------- streaming

    def feed(self, piece: str) -> Released:
        if self.done:
            return Released(verdict="stop")
        self.buffer += piece
        out: list[str] = []
        while (end := sentence_end(self.current + self.buffer)) is not None:
            k = max(0, end - len(self.current))
            unreleased, self.buffer = self.buffer[:k], self.buffer[k:]
            r = self._sentence(unreleased)
            out.append(self._spaced(r.text))
            if r.verdict is not None:
                return Released("".join(out), r.verdict)
            if r.text.strip():  # the space after it goes too: downstream (speech) sees the sentence has ended
                space = self.buffer[: len(self.buffer) - len(self.buffer.lstrip())]
                out.append(self._spaced(space))
                self.buffer = self.buffer[len(space) :]
        out.append(self._spaced(self._partial()))
        return Released("".join(out))

    def finish(self) -> Released:
        """The model's stream ended: the last sentence."""
        if self.done:
            return Released()
        rest, self.buffer = self.buffer, ""
        if not (self.current + rest).strip():
            return Released()
        r = self._sentence(rest)
        return Released(self._spaced(r.text), r.verdict)

    def _spaced(self, text: str) -> str:
        """``text`` as it follows what was released: no second space after a dropped sentence or a sentence's own
        trailing space, none before the answer's first word."""
        if text[:1].isspace() and (not self._last or self._last.isspace()):
            text = text.lstrip()
        if text:
            self._last = text[-1]
        return text

    # -------------------------------------------------------------- one sentence

    def _sentence(self, unreleased: str) -> Released:
        full = self.current + unreleased
        self.current = ""
        if self.after_disclaimer and not self.released and full == unreleased:
            bare = drop_disclaimer(unreleased)
            if bare != unreleased:
                self._record("disclaimer", "dropped", unreleased[: len(unreleased) - len(bare)])
                full = unreleased = bare
                if not bare.strip():  # the disclaimer was the whole sentence
                    return Released()
        out = self._fix_tokens(unreleased)
        replaced = False
        stop = False
        if self.coverage is not None and self.coverage.contradicted_by(full):
            if self.sentences == 0 and not self.released and self.attempt == 1:
                self._record("coverage", "retry", full)
                return Released(verdict="retry")
            out, stop = self._correct_denial(full, unreleased)
            replaced = True
        elif self.figures is not None and unsupported_figures(full, self.figures):
            replaced = True
            if self.live_said:
                self._record(self.figure_check, "dropped", full)
                out = ""
            else:
                self._record(self.figure_check, "replaced", full)
                self.live_said = True
                out = " " + self.live_line
        out = self._first(out)
        if out.strip():
            self.released = True
        n = _words(out) if replaced else _words(full)
        self.words += n
        if n >= 2:
            self.sentences += 1
        if stop or (self.max_words is not None and self.words >= self.max_words and n >= 2):
            self.done = True
        if self.max_sentences is not None and self.sentences >= self.max_sentences:
            self.done = True
        if self.done:
            if not stop and self.max_words is not None and n >= 2:
                self._record("length", "stopped", f"{self.words} words, {self.sentences} sentences")
            return Released(out, "stop")
        return Released(out)

    def _correct_denial(self, full: str, unreleased: str) -> tuple[str, bool]:
        """A denial the evidence contradicts, when the model can't (or can no longer) be asked again: its denying
        clause dropped when what comes before it can stand; else a first sentence is replaced by the fixed correction
        and a later one dropped. When the sentence's opening is already out (rare: its negation came more than
        ``LAG_WORDS`` into its own clause), it is ended there: "…", and the correction for a first sentence; the
        answer stops."""
        assert self.coverage is not None
        released = full[: len(full) - len(unreleased)]
        kept = drop_denial(full, released)
        if kept is not None:
            self._record("coverage", "clause_dropped", full)
            return self._fix_tokens(kept[len(released) :]), False
        if released.strip():
            self._record("coverage", "cut", full)
            return ("\u2026 " + self.coverage.correction if self.sentences == 0 else "\u2026"), True
        if self.sentences > 0:
            self._record("coverage", "dropped", full)
            return "", False
        self._record("coverage", "corrected", full)
        lead = unreleased[: len(unreleased) - len(unreleased.lstrip())]
        return lead + self.coverage.correction, True

    def _first(self, text: str) -> str:
        """The answer's first words after a fixed prefix: a common opener in lower case ("The" → "the")."""
        if not self.lower_first or self.released or not text.strip():
            return text
        m = re.match(r"(\s*)([A-Z][a-z]*)\b", text)
        if m and m.group(2).casefold() in _COMMON_STARTERS:
            text = m.group(1) + m.group(2).lower() + text[m.end() :]
        return text

    # -------------------------------------------------------------- the sentence being written

    def _partial(self) -> str:
        if self.done or not self.buffer:
            return ""
        if self.sentence_hold:
            # A live-figure check holds whole sentences. A coverage check lets the first sentence out LAG_WORDS behind
            # the model while it holds no negation (the voice answer's first chunk waits for it); from its first "not"
            # / "only" / "नहीं" on, the rest of it waits for its end (a denial's own clause is then still unsaid, and
            # can be dropped or the model asked again). Later sentences are held whole: speech waits for whole
            # sentences after the first chunk anyway, and a denial is then dropped without a trace. English holds
            # a sentence whose subject is a document until its verb; Hindi, whose negation comes last, a sentence
            # about what the documents contain until a clause's final auxiliary (``hindi_hold``).
            first = self.sentences == 0 and self.attempt <= 2
            sofar = self.current + self.buffer
            if not (self.lag_release and first) or has_denial_cue(sofar):
                return ""
            cut = None
            if _DEVANAGARI.search(sofar):
                about, safe = hindi_hold(sofar)
                if about:  # its clauses go out as their verbs come; the one being written waits
                    cut = safe - len(self.current)
                    if cut <= 0:
                        return ""
            elif awaits_its_verb(sofar):
                return ""
            if cut is None:
                words = list(_WORD.finditer(_mask(self.buffer)))
                ready = len(words) - LAG_WORDS - (0 if self.buffer[-1].isspace() else 1)
                if ready <= 0:
                    return ""
                cut = words[ready - 1].end()
        elif self.token_hold:
            # the word being written (it may be a code or a name), and for a misheard phrase ("Wall Mora") the words
            # before it that may start one; whole phrases are respelled first
            if self.renames:
                end = max(self.buffer.rfind(" "), self.buffer.rfind("\n")) + 1
                self.buffer = self._respell(self.buffer[:end]) + self.buffer[end:]
            hold = (self._phrase_words - 1 if self.renames else 0) + (0 if self.buffer[-1].isspace() else 1)
            if hold == 0:
                cut = len(self.buffer)
            else:
                words = list(_WORD.finditer(self.buffer))
                if len(words) <= hold:
                    return ""
                cut = words[len(words) - hold - 1].end()
        else:
            cut = len(self.buffer)
        text, self.buffer = self.buffer[:cut], self.buffer[cut:]
        text = self._first(self._fix_tokens(text))
        if text.strip():
            self.released = True
        self.current += text
        return text

    # -------------------------------------------------------------- names and codes

    def _respell(self, text: str) -> str:
        fixed = respell(text, self.renames)
        if fixed != text:
            self._record("name", "respelled", text, to=fixed)
        return fixed

    def _fix_tokens(self, text: str) -> str:
        if self.renames:
            text = self._respell(text)
        if self.identifiers:

            def swap(m: re.Match[str]) -> str:
                token = m.group(0)
                if token in self.identifiers:
                    return token
                near = nearest_identifier(token, self.identifiers)
                if near is None:
                    self._record("identifier", "unverified", token)
                    return token
                self._record("identifier", "corrected", token, to=near)
                return near

            text = _IDENTIFIER.sub(swap, text)
        return text

    def _record(self, check: str, action: str, text: str, **extra: str) -> None:
        self.checks.append({"check": check, "action": action, "text": " ".join(text.split())[:200], **extra})
