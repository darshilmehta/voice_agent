"""Which language a message is in, and which language to answer in (docs/DESIGN.md §1, §3.4). The app speaks English
and Hindi only; Hinglish (Hindi written in Latin script, mixed with English) is Hindi.

The language of a message (``message_language``):

- **Devanagari**: Hindi when at least as many words are in Devanagari as in Latin script. Counted by words, not
  characters: Hindi questions routinely carry Latin terms ("FY24 में EBITDA margin क्या था?" is Hindi), while an
  English question with one Hindi word ("What is the मार्जिन?") is English.
- **Romanized Hindi (Hinglish)**: Latin-script text whose words are largely Hindi function words ("FY24 mein revenue
  kitna tha?", "achha, aur debt ke baare mein batao") is Hindi; English with a stray Hindi word ("what was the
  revenue yaar") stays English. Words that are also common English ("the", "to", "main", "me", "par", "hi") never
  count as Hindi.
- No letters in either script ("18.2%?"): unknown (None).

The answer language (``decide_language``), first rule that applies:

1. **asked**: the utterance asks for a language ("answer in Hindi", "hindi mein batao", "अंग्रेज़ी में बताइए"). The
   request sticks: it becomes the chat's preferred language until the user asks for another one.
2. **requested**: the caller forces a language for this message (the API's ``language``). A voice turn that passes
   the spoken language itself as ``language`` forces nothing beyond rule 4, so an earlier request still holds.
3. **preferred**: the language the user asked for earlier in the chat (rule 1).
4. **utterance**: the language of the latest utterance: the spoken language when speech recognition reports one,
   else the text's (above). Hinglish is answered in Hindi, in Devanagari, which the Hindi voice can speak.
5. **fallback**: the previous answer's language, else the chat's language, else the app default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..settings import Language

_HINDI_LETTERS = "\\u0900-\\u097F"  # the Devanagari block, as a regex range
_DEVANAGARI = re.compile(f"[{_HINDI_LETTERS}]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[a-z]+")


def wordset(text: str) -> frozenset[str]:
    """A word list written as a block of text: its whitespace-separated words."""
    return frozenset(text.split())


# Romanized Hindi words that are not (common) English words. Deliberately left out because they are also English:
# the, to, main, me, par, pe, hi, so, na, ya, mat, bare, hum, ok.
HINGLISH_WORDS = wordset(
    """
    hai hain tha thi thhe ho hoga hogi honge hota hoti hote hua hui hue kya kyaa kyu kyun kyon kaise kaisa kaisi
    kitna kitni kitne kaun kaunsa kaunsi konsa konsi kab kahan kaha kidhar kis kise kisne kiska kiski kiske mein mai
    mujhe mujhko mera meri mere humein hamein humko hamara hamari humara humari aap aapka aapki aapke apna apni apne
    tum tumhara tumhe yeh ye yah woh wo vo voh unka unki unke uska uski uske iska iski iske inka isme usme isse usse
    ka ki ke ko se ne tak aur lekin magar bhi toh nahi nahin nai haan ji acha achha accha theek thik sahi galat
    bilkul batao bataiye bataye bataen bolo boliye suno dekho samjhao samjhaiye samjha samjho karo kariye karna
    karke kar kiya kiye raha rahi rahe gaya gayi gaye diya diye liya liye lekar chahiye chahte chahta chahti sakte
    sakta sakti wala wali wale abhi ab phir fir wapas vapas chalo baat baare matlab yaani yani kuch sab sabse zyada
    jyada kam bahut bohot bahot saal pichle pichhle agle pehle baad dono aaj kal yaar bhai kripya dhanyavaad
    dhanyawad shukriya namaste jaankari jankari rupaye
    """
)


def _counts(text: str) -> tuple[int, int, int]:
    """(Devanagari words, Latin words, Latin words that are romanized Hindi)."""
    devanagari = latin = hinglish = 0
    for word in text.split():
        if _DEVANAGARI.search(word):
            devanagari += 1
        elif _LATIN.search(word):
            latin += 1
            if any(w in HINGLISH_WORDS for w in _WORD.findall(word.casefold())):
                hinglish += 1
    return devanagari, latin, hinglish


def is_hinglish(text: str) -> bool:
    """Latin-script text that is mostly romanized Hindi: at least two Hindi words making up a quarter of the Latin
    words, or one Hindi word in an utterance of at most two words ("batao", "theek hai")."""
    _, latin, hinglish = _counts(text)
    if latin == 0:
        return False
    return (hinglish >= 2 and hinglish / latin >= 0.25) or (hinglish >= 1 and latin <= 2)


def message_language(text: str) -> Language | None:
    """The language a message is written in: 'hi' for Devanagari (by word majority, ties to Hindi) and for
    romanized Hindi, 'en' for other Latin-script text, None without words in either script."""
    devanagari, latin, _ = _counts(text)
    if devanagari == 0 and latin == 0:
        return None
    if devanagari >= latin:
        return "hi"
    return "hi" if is_hinglish(text) else "en"


def script_language(text: str) -> Language | None:
    """The language a text is *written* in, by script alone: 'hi' when at least a quarter of its words are in
    Devanagari, 'en' for Latin script (romanized Hindi included: the English voice reads it, the Hindi one can't),
    None without letters ("₹ 7,365 [S1]"). For what was answered (saved ``language``) and which voice speaks it."""
    devanagari, latin, _ = _counts(_MARKERS.sub(" ", text))
    if devanagari == 0 and latin == 0:
        return None
    return "hi" if devanagari * 3 >= latin else "en"  # Hindi answers keep "FY24", "EBITDA", names in Latin script


_MARKERS = re.compile(r"\[\s*[SW]\d+(?:\s*[,;]\s*[SW]\d+)*\s*\]", re.IGNORECASE)  # [S1] citation markers
_DEVANAGARI_LETTER = re.compile("[\u0900-\u0963\u0966-\u097f]")
_LATIN_LETTER = re.compile("[A-Za-z]")


class ScriptCheck:
    """Is a streamed answer in the expected language? Fed the first pieces; ``verdict`` becomes True or False as soon
    as the letters so far tell (B5: asked for English, the 4B model may still answer a Hinglish question in Hindi):

    - expected English: False at the first Devanagari word (3 letters); True after 8 Latin letters without any (two
      words or so: "FY24 में…" is caught, "The EBITDA…" goes out at once);
    - expected Hindi: True at the first Devanagari word; False after 24 Latin letters without any (Hindi answers often
      start with "FY24", "EBITDA" or a company name, so Latin letters alone prove nothing at first).

    ``finish()`` decides at the end of a short answer (None: too few letters to tell)."""

    def __init__(self, expected: Language) -> None:
        self.expected = expected
        self.devanagari = 0
        self.latin = 0
        self.verdict: bool | None = None

    def feed(self, piece: str) -> bool | None:
        if self.verdict is None:
            self.devanagari += len(_DEVANAGARI_LETTER.findall(piece))
            self.latin += len(_LATIN_LETTER.findall(piece))
            if self.expected == "en":
                if self.devanagari >= 3:
                    self.verdict = False
                elif self.latin >= 8:
                    self.verdict = True
            elif self.devanagari >= 3:
                self.verdict = True
            elif self.latin >= 24:
                self.verdict = False
        return self.verdict

    def finish(self) -> bool | None:
        if self.verdict is None:
            hindi_in_english = self.expected == "en" and self.devanagari >= 3
            english_in_hindi = self.expected == "hi" and self.devanagari == 0 and self.latin >= 8
            if hindi_in_english or english_in_hindi:
                self.verdict = False
        return self.verdict


def is_devanagari(text: str) -> bool:
    """Mostly written in Devanagari (by words)."""
    devanagari, latin, _ = _counts(text)
    return devanagari > 0 and devanagari >= latin


# ------------------------------------------------------------------ "answer in Hindi"

_NAMES: dict[Language, str] = {
    "hi": r"(?:hindi|हिंदी|हिन्दी)",
    "en": r"(?:english|angrezi|angreji|अंग्रेज़ी|अंग्रेजी|इंग्लिश)",
}
# No letter right before / after (Python's \b is unreliable next to Hindi vowel signs).
_LEFT = f"(?<![A-Za-z{_HINDI_LETTERS}])"
_RIGHT = f"(?![A-Za-z{_HINDI_LETTERS}])"
_VERBS = (
    f"{_LEFT}(?:answer|reply|respond|speak|talk|say|tell|explain|write|continue|switch|translate|batao|bataiye|"
    "bolo|boliye|bol|baat|samjhao|samjhaiye|jawab|likho|bataye|bataiye|batayein|बताओ|बताइए|बताइये|बताये|बतायें|बताएं|"
    "बताएँ|बोलो|बोलिए|बोलिये|बात|समझाओ|समझाइए|"
    f"जवाब|लिखो){_RIGHT}"
)


def _request_patterns(name: str) -> list[re.Pattern[str]]:
    return [
        # "answer in Hindi", "can you reply in English please", "switch to English", "translate it into Hindi"
        re.compile(f"{_VERBS}.{{0,24}}?{_LEFT}(?:in|into|to)\\s+{name}{_RIGHT}", re.IGNORECASE),
        # "speak Hindi", "use English"
        re.compile(f"{_LEFT}(?:speak|talk|use)\\s+{name}{_RIGHT}", re.IGNORECASE),
        # "Hindi mein batao", "हिंदी में बताइए", "english me bolo"
        re.compile(f"{_LEFT}{name}\\s*(?:mein|me|main|में)\\s*{_VERBS}", re.IGNORECASE),
    ]


_REQUESTS: dict[Language, list[re.Pattern[str]]] = {lang: _request_patterns(n) for lang, n in _NAMES.items()}
# The language alone: "Hindi", "in Hindi please", "हिंदी में", "English please"
_SHORT_REQUEST: dict[Language, re.Pattern[str]] = {
    lang: re.compile(
        f"^\\W*(?:(?:in|into)\\s+)?{name}\\s*(?:(?:mein|me|में)\\s*)?(?:please|plz|pls)?\\W*$", re.IGNORECASE
    )
    for lang, name in _NAMES.items()
}


def asked_language(text: str) -> Language | None:
    """The language the utterance asks the assistant to use, if it asks for one: a verb of speaking or answering with
    the language ("answer in Hindi", "hindi mein batao", "अंग्रेज़ी में बताइए", "switch to English"), or the language
    alone ("in Hindi please"). Questions *about* a language ("what is revenue in Hindi?", "tell me the Hindi word
    for it") don't count. When both are asked for, the later one wins ("don't answer in English, answer in Hindi")."""
    for lang, pattern in _SHORT_REQUEST.items():
        if pattern.match(text):
            return lang
    found = [(m.end(), lang) for lang, patterns in _REQUESTS.items() for p in patterns for m in p.finditer(text)]
    return max(found)[1] if found else None


# ------------------------------------------------------------------ the answer language

LanguageReason = Literal["asked", "requested", "preferred", "utterance", "fallback"]


@dataclass(frozen=True, slots=True)
class LanguageDecision:
    language: Language
    reason: LanguageReason
    asked: Language | None  # the language the utterance asked for (becomes the chat's preference)
    input_language: Language | None  # the utterance's own language


def decide_language(
    text: str,
    *,
    requested: Language | None = None,
    spoken: Language | None = None,
    preferred: Language | None = None,
    fallback: Language,
) -> LanguageDecision:
    """The answer language by the rules in the module docstring. ``spoken``: the language speech recognition
    detected (voice), which outranks the text's script for the utterance's own language."""
    asked = asked_language(text)
    utterance = spoken or message_language(text)
    if asked is not None:
        return LanguageDecision(asked, "asked", asked, utterance)
    if requested is not None and requested != spoken:
        return LanguageDecision(requested, "requested", None, utterance)
    if preferred is not None:
        return LanguageDecision(preferred, "preferred", None, utterance)
    if utterance is not None:
        return LanguageDecision(utterance, "utterance", None, utterance)
    return LanguageDecision(fallback, "fallback", None, None)


def choose_language(text: str, requested: Language | None, fallback: Language) -> Language:
    """The answer language without chat state: ``requested`` when given, else the message's language, else
    ``fallback`` (the chat's language or the app default)."""
    if requested is not None:
        return requested
    return message_language(text) or fallback
