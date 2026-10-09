"""The turn router (docs/DESIGN.md §3.4): what kind of turn a user message is, as a validated ``TurnRoute``.

Two ways to a route, cheapest first:

- **Keyword fast path** (no model): "stop", "bas", "ruko" → stop; "okay", "got it", "haan", "theek hai" →
  backchannel (not when the agent just asked something, where "yes" is an answer); "thanks", "hello", "namaste" →
  conversation; an English standalone question that names the documents ("what does the report say about debt?")
  → document question. Hindi and Hinglish questions always go to the model, which writes the English search query
  their retrieval needs (misspelled Hindi transcripts score ~0.02 against English passages without it).
- **Router model** (``LLMTurnRouter``): qwen3:4b-instruct with a JSON schema, given the utterance, the last two
  exchanges, the current topic and, after a barge-in, what the user heard of the interrupted answer. Its output is
  three fields (intent, standalone query, English query): prompt prefill (~370 tokens/s) and output length (~34
  tokens/s) are its whole latency (§9.1), so the prompt is short and everything else (topic, shift, language,
  confidence) is derived by application code.

The model only proposes; ``validate`` decides: resume phrases ("back to the report") always resume (the model once
took them for backchannels, §9.5), a "stop"/"backchannel" that asks a question is a question, a correction with
nothing before it is a question, an English turn needs no English query, a query that repeats the utterance is
dropped.

**Live data** (§3.7, ``with_live_tools``): application code, not the model, decides when the web search tool runs.
A turn that asks a question (document, mixed, general, correction, resume with a question) gets
``tools=["web_search"]`` when its utterance or standalone question carries a live-data cue (services/live_data.py:
"today", "latest news", "share price", "आज", "abhi"…) and the tool is available (``RouteRequest.available_tools``);
otherwise the decision still records the cue, so the answer can say live data isn't available. A live question
never takes the document fast path: the router model writes its standalone English question, which is what gets
searched (it resolves "the stock" from the conversation), and for Hindi and Hinglish turns that English query is
kept for every answering intent.

**The canvas** (§12.1): with visuals on screen, the router's input gets them as a few lines (kind, title, x and series
labels, highlight, and what the utterance points at: "the second bar" → Engineered Plastics), and the model may say
``canvas_edit``; the keyword fast path catches the common edits first ("make that a bar chart", "remove the pie",
"pin this", "हटा दो", "isko table mein dikhao"). A question about a chart is a document question (validation turns a
"clarification" about "the second bar" into one); an edit is never a question. Chats without a canvas get exactly the
prompt they had before.

Routing a turn as a whole (speculative retrieval alongside the router, timeout, fallback, retrieval policy) is
``services/planning.py``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from pydantic import BaseModel

from ..domain.conversation import SILENT_INTENTS, TOPIC_NEUTRAL_INTENTS, ConversationState, Intent, TurnRoute
from ..domain.projects import Message
from ..providers.llm import LLMClient, LLMMessage
from ..settings import Language
from .canvas.conversation import CanvasEdit, parse_edit, refers_to_screen
from .canvas.planner import visual_intent
from .language import asked_language, is_devanagari, message_language, wordset
from .live_data import live_data_cue
from .sources import strip_markers

ROUTER_PROMPT_VERSION = "router-v1"
ROUTER_MAX_TOKENS = 96  # the JSON needs ~15-60 tokens; the cap bounds a runaway (it then fails validation)
ROUTER_HISTORY_MESSAGES = 4  # two exchanges
ROUTER_HISTORY_CHARS = 240
QUERY_MAX = 400
TOPIC_MAX = 60

# Route confidence by how it was decided (the model's own confidence isn't calibrated, and costs output tokens).
CONFIDENCE = {"stop": 0.95, "heuristic": 0.9, "llm": 0.8, "llm_overridden": 0.6, "fallback": 0.0}

ReplyKind = Literal["thanks", "greeting", "language"]

# ------------------------------------------------------------------ text helpers

# Keep letters (incl. Devanagari and its vowel signs) and digits; the danda (। ॥) is punctuation.
_PUNCT = re.compile("[^\\w\\s\\u0900-\\u0963\\u0966-\\u097F'-]|[\\u0964\\u0965]")
_END = "(?![A-Za-z0-9\\u0900-\\u0963\\u0966-\\u097F])"  # end of a word (\b fails after Hindi vowel signs)


def normalize(text: str) -> str:
    """Lowercase, punctuation removed, spaces collapsed ("Okay!!" → "okay", "Mm-hmm." → "mm-hmm")."""
    return " ".join(_PUNCT.sub(" ", text.casefold()).split())


def words(text: str) -> list[str]:
    return normalize(text).split()


def _phrases(text: str) -> frozenset[str]:
    return frozenset(p.strip() for p in text.replace("\n", " ").split("|") if p.strip())


STOP_PHRASES = _phrases(
    """
    stop|stop it|stop please|please stop|stop now|stop talking|stop stop|okay stop|ok stop|no stop|wait stop|
    be quiet|quiet|silence|enough|that's enough|thats enough|that is enough|cancel|cancel that|shut up|hold on|
    ruko|ruk jao|ruk jaiye|rukiye|bas|bas karo|bas bas|chup|chup karo|band karo|
    रुको|रुक जाओ|रुकिए|बस|बस करो|बस बस|चुप|चुप करो|बंद करो
    """
)
# Acknowledgements are recognised by their words, so any short mix of them counts ("okay cool", "haan theek hai",
# "theek hai, shukriya", "got it, thanks a lot"): said while the agent is idle, they must never become a document
# question that abstains.
ACK_WORDS = wordset(
    """
    ok okay okk mm mmm mhm mm-hmm uh-huh uh huh hmm hm right alright all i see got it sure yeah yep yes yup cool nice
    great fine understood makes sense that's thats good perfect awesome wonderful then
    haan haa han ha ji achha acha accha theek thik hai sahi samajh gaya gayi bilkul
    हाँ हां हा जी अच्छा ठीक है सही हम्म समझ गया गई बिल्कुल
    """
)
THANKS_WORDS = wordset("thanks thank thankyou shukriya dhanyavaad dhanyawad धन्यवाद शुक्रिया")
_THANKS_FILLERS = wordset("you so much a lot very many bahut helps helped that was it बहुत")
# Words of a request that only changes the answer language ("now answer in English please", "हिंदी में बताइए").
_LANGUAGE_REQUEST_WORDS = wordset(
    """
    please pls plz now from on can could would you will answer reply respond speak talk say tell explain write
    continue switch use repeat in into to me it that this the same again it's language
    hindi english angrezi angreji batao bataiye bataye batayein bolo boliye bol karo kariye baat jawab do dijiye mein
    me main ab se zara thoda
    हिंदी हिन्दी अंग्रेज़ी अंग्रेजी इंग्लिश में बताओ बताइए बताइये बताये बतायें बताएं बताएँ बोलो बोलिए बोलिये कीजिए करो
    करें बात जवाब दो दीजिए अब से ज़रा थोड़ा यह इसे
    """
)
GREETING_PHRASES = _phrases(
    """
    hi|hello|hey|hi there|hello there|hey there|good morning|good afternoon|good evening|namaste|namaskar|
    नमस्ते|नमस्कार
    """
)
_STOP_WORDS = frozenset({"stop", "quiet", "enough", "cancel", "ruko", "bas", "chup", "रुको", "बस", "चुप"})

_QUESTION_START = re.compile(
    r"^(what|what's|whats|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|can|could|will|would|"
    r"should|shall|has|have|had|tell|show|list|give|explain|describe|summarize|summarise|compare|find)\b"
)
_HINDI_QUESTION_WORDS = wordset(
    """
    क्या कितना कितनी कितने कौन कौनसा कौनसी कब कहाँ कहां क्यों कैसे बताओ बताइए बताइये बताएं बताएँ समझाओ समझाइए kya
    kitna kitni kitne kaun kaunsa konsa kab kahan kyun kyon kaise batao bataiye bataye samjhao
    """
)
# Words that point back into the conversation: a question with one of them needs the history to be understood.
_BACK_REFERENCES = wordset(
    """
    it its it's that this these those they them their theirs he she him her his hers same again also too instead
    else previous earlier above former latter then there one ones and or but so यह वह ये वो इसका इसकी इसके उसका उसकी
    उसके इसमें उसमें इसे उसे इन उन इनका उनका और भी फिर वही वहाँ वहां तब yeh ye woh wo vo iska iski iske uska uski
    uske isme usme aur bhi phir fir wahi
    """
)
# Words that point at something said before or shown on screen, leaving out the conjunctions and adverbs of
# ``_BACK_REFERENCES`` ("level 3 and level 4" is no back-reference): what a clarification is really for.
_DEICTIC = wordset(
    """
    it its it's that this these those they them their theirs he she him her his hers same previous earlier above
    former latter one ones यह वह ये वो इसका इसकी इसके उसका उसकी उसके इसमें उसमें इसे उसे इन उन इनका उनका वही वहाँ वहां
    yeh ye woh wo vo iska iski iske uska uski uske isme usme wahi
    """
)
_FOLLOW_UP_START = re.compile(f"^(?:and|or|but|so|also|what about|how about|why|aur|to|toh|और|तो){_END}")
_CORRECTION = re.compile(
    "^[\\s,.!?-]*(?:no|nope|nah|not that|sorry|wait|actually|oops|i mean|i meant|rather|correction|nahi|nahin|"
    f"नहीं|मेरा मतलब|मतलब){_END}|\\bi meant\\b|\\bmera matlab\\b|मेरा मतलब",
    re.IGNORECASE,
)
_DOC_NOUNS = (
    r"report|reports|document|documents|doc|docs|file|files|pdf|deck|slides|contract|agreement|paper|filing|"
    r"page|section|chapter|annexure|appendix|uploaded"
)
_RESUME = re.compile(
    rf"\b(?:(?:go|come|get|going|coming|let'?s go|let us go|lets get)\s+back\s+to|back\s+to\s+(?:the|my|our|your)\s+"
    rf"(?:{_DOC_NOUNS}|topic|question|earlier)|return\s+to\s+(?:the|my)\s+(?:{_DOC_NOUNS})|where\s+were\s+we|"
    rf"(?:{_DOC_NOUNS})\s+(?:par|pe)\s+(?:wapas|vapas)|(?:wapas|vapas)\s+(?:{_DOC_NOUNS}))\b"
    r"|(?:रिपोर्ट|दस्तावेज़|दस्तावेज|डॉक्यूमेंट|फ़ाइल|फाइल)\s*(?:पर|की ओर)\s*(?:वापस|लौट)|वापस\s*(?:रिपोर्ट|दस्तावेज़|दस्तावेज)",
    re.IGNORECASE,
)
# Naming the user's documents, not just using a word like "paper" ("how do I write a research paper?"): a determiner
# or possessive before the noun (an adjective may come between: "the annual report"), or "according to".
_DOC_MENTION = re.compile(
    rf"\b(?:the|this|that|these|those|my|our|your|uploaded)\s+(?:[a-z0-9'-]+\s+)?(?:{_DOC_NOUNS})\b|\baccording to\b",
    re.IGNORECASE,
)

# Words left out of heuristic topic labels.
_TOPIC_STOP = wordset(
    """
    a an the of in on at to for from by with about and or but is are was were be been do does did what what's whats
    which who whom whose when where why how can could would should will shall has have had tell me show list give
    explain describe our their its it this that these those there please i you we they my your company company's
    report reports document documents doc docs file files pdf deck slides contract paper page section uploaded annual
    anyway okay ok back just according say says said much many get got year grow grew increase increased decrease
    decreased fall fell rise rose change changed happen happened mean means क्या कितना कितनी कितने कौन कब कहाँ कहां
    क्यों कैसे में का की के को से है था थी थे और भी यह वह
    """
)


def is_question(text: str) -> bool:
    t = text.strip()
    if t.endswith(("?", "؟")):
        return True
    n = normalize(t)
    return bool(_QUESTION_START.match(n)) or any(w in _HINDI_QUESTION_WORDS for w in n.split())


def refers_back(text: str) -> bool:
    """Needs the conversation to be understood: a back-reference ("it", "that year", "इसका") or a follow-up opener
    ("and…", "what about…")."""
    n = normalize(text)
    return bool(_FOLLOW_UP_START.match(n)) or any(w in _BACK_REFERENCES for w in n.split())


def leans_on_the_conversation(text: str) -> bool:
    """Can't be understood without something said or shown before ("what about that one?", "and in FY23?", "why did
    it grow?"): the one kind of utterance for which asking back is right. "What is the hotel limit per night for level
    3 and level 4 employees in tier 1 cities?" stands on its own."""
    n = normalize(text)
    return bool(_FOLLOW_UP_START.match(n)) or any(w in _DEICTIC for w in n.split())


def is_correction(text: str) -> bool:
    return bool(_CORRECTION.search(text))


def is_resume(text: str) -> bool:
    """Going back to the documents: "let's go back to the report", "back to the document", "रिपोर्ट पर वापस"."""
    return bool(_RESUME.search(text))


def asks_beyond_resume(text: str) -> bool:
    """A return to the documents that also asks something ("back to the report. What about debt?")."""
    rest = _RESUME.sub(" . ", text)
    return any(is_question(part) for part in re.split(r"[.,;!—]+", rest) if part.strip()) or rest.rstrip().endswith("?")


def mentions_documents(text: str, filenames: Sequence[str] = ()) -> bool:
    if _DOC_MENTION.search(text):
        return True
    n = normalize(text)
    for name in filenames:
        stem = normalize(name.rsplit(".", 1)[0].replace("_", " ").replace("-", " "))
        if len(stem) > 3 and re.search(f"(?<!\\w){re.escape(stem)}(?!\\w)", n):
            return True
    return False


# ------------------------------------------------------------------ facts or general knowledge (B1)

# "What is EBITDA?", "What does EBITDA stand for?", "define working capital", "how is EBITDA calculated?", "how do I
# make chai?", "EBITDA क्या होता है?", "EBITDA ka matlab kya hai?": general knowledge, whatever the documents say.
_DEFINITIONAL = re.compile(
    r"^(?:(?:so|and|ok|okay|by the way|btw)[, ]+)?(?:"
    r"(?:what|who)\s*(?:'s|\s+is|\s+are)\s+(?:an?\s+|the\s+(?:term|concept|meaning|definition|full\s+form)\s+(?:of\s+)?)?"
    r"[\w&/ -]{1,40}\??$"
    r"|what\s+(?:does|do)\s+.{1,40}\s+(?:mean|stand\s+for)\b"
    r"|what(?:'s|\s+is)\s+the\s+(?:meaning|definition|full\s+form)\s+of\b"
    r"|(?:define|definition\s+of)\b"
    r"|explain\s+(?:what\s+)?.{1,40}\s+(?:is|are|means)\b"
    r"|how\s+(?:do|can|should|would|could)\s+(?:i|you|we|one|people)\b"
    r"|how\s+to\b"
    r"|how\s+(?:is|are)\s+.{1,40}\s+(?:calculated|computed|measured|defined|pronounced)\b"
    r"|why\s+do\s+(?:companies|people|firms|businesses)\b"
    r")"
    r"|^\S+(?:\s+\S+){0,2}\s+(?:क्या|kya)\s+(?:है|hai)\s*[?।]?$"
    r"|(?:मतलब|अर्थ|परिभाषा|matlab|arth)\s*(?:क्या|kya)"
    r"|(?:क्या|kya)\s+(?:होता|होती|होते|hota|hoti|hote)\s+(?:है|हैं|hai|hain)"
    r"|(?:कैसे|kaise)\s+\S+\s*(?:ते|te)\s+(?:हैं|hain)",
    re.IGNORECASE,
)
# Questions about the assistant itself ("who are you?", "can you speak Hindi?"): conversation, not facts. ("Can you
# tell me the revenue?" is a fact question.)
_TO_THE_ASSISTANT = re.compile(
    r"\b(?:who|what|how)\s+are\s+you\b|^are\s+you\b|\byour\s+(?:name|job|purpose)\b|\bwhat\s+can\s+you\s+do\b|"
    r"\bcan\s+you\s+(?:hear|speak|understand|see)\b|\b(?:aap|tum)\s+(?:kaun|kaise)\b|(?:आप|तुम)\s+(?:कौन|कैसे)",
    re.IGNORECASE,
)
# What the user's documents are about: the company, its board, its figures for a period ("the company", "our revenue",
# "FY24", "at the end of the year", "कंपनी", "वित्त वर्ष").
_DOCUMENT_SUBJECT = re.compile(
    r"\b(?:the|this|our|its|their)\s+(?:[a-z'-]+\s+)?(?:company|company's|firm|group|business|businesses|"
    r"organi[sz]ation|bank|board|management|auditors?|segments?|plants?|subsidiar(?:y|ies)|promoters?|"
    r"shareholders?|directors?|ceo|cfo|chair(?:person|man)?|employees|workforce|staff|headcount|products?|"
    r"suppliers?|customers?|revenue|profits?|margins?|debt|dividends?|results|accounts)\b"
    r"|\b(?:company's|firm's|group's)\b"
    r"|\b(?:q[1-4]\s*)?fy\s?'?\d{2,4}\b"
    r"|\b(?:year[- ]end|end\s+of\s+the\s+(?:fiscal\s+|financial\s+)?(?:year|quarter)|fiscal\s+year|financial\s+year|"
    r"last\s+(?:fiscal\s+)?year|this\s+year|previous\s+year|the\s+year|last\s+quarter|the\s+quarter)\b"
    r"|\b(?:31|30)\s+(?:march|june|september|december)\b"
    r"|कंपनी|कम्पनी|वित्त\s*वर्ष|वित्तीय\s*वर्ष|रिपोर्ट|दस्तावेज़|दस्तावेज|पिछले\s+साल|इस\s+साल|बोर्ड|"
    r"\b(?:kampani|kampni|pichhle\s+saal|pichle\s+saal|is\s+saal)\b",
    re.IGNORECASE,
)
# Filename words that name no one ("annual_report_fy24.pdf" names no company; "valmora_annual_report.pdf" does).
_GENERIC_FILE_WORDS = wordset(
    """
    annual report reports document documents doc docs file files final draft copy scan scanned notes minutes deck
    slides presentation investor investors policy policies travel expense expenses group health insurance statement
    statements financial financials results quarterly quarter summary overview brochure manual handbook contract
    agreement memo letter board meeting plan budget data sheet table tables appendix version updated new old english
    hindi soochna yojana
    """
)
_NAME_WORD = re.compile(r"[a-z]{4,}")


# Asks for a judgement ("is an 18% margin good?", "should we…"): document facts plus general knowledge at most.
_OPINION = re.compile(
    r"^(?:is|are|was|were)\b.*\b(?:good|bad|high|low|healthy|strong|weak|reasonable|normal|typical|enough|"
    r"better|worse|risky|safe)\b|\bshould\b|\bin\s+your\s+(?:opinion|view)\b|\bdo\s+you\s+think\b|"
    r"\b(?:compare|compared|comparison)\b|अच्छा\s+है|achha\s+hai",
    re.IGNORECASE,
)


def asks_for_judgement(text: str) -> bool:
    return bool(_OPINION.search(text))


def is_definitional(text: str) -> bool:
    """Asks what a term means or how something is done in general ("What is EBITDA?", "how do I…?", "X क्या होता
    है?"), not about the documents' subject ("What is the company's EBITDA?" is a fact question)."""
    t = " ".join(text.strip().split())
    return bool(_DEFINITIONAL.search(t)) and not about_the_documents(t)


def about_the_documents(text: str, filenames: Sequence[str] = ()) -> bool:
    """Points at what the documents are about: names them ("the report"), their subject ("the company", "our
    revenue", "its board") or a reporting period ("FY24", "at the end of the year", "वित्त वर्ष"), or a name from a
    document's filename ("Valmora" for valmora_annual_report_fy24.pdf)."""
    if mentions_documents(text, filenames) or _DOCUMENT_SUBJECT.search(text):
        return True
    names = {
        w
        for name in filenames
        for w in _NAME_WORD.findall(name.rsplit(".", 1)[0].casefold())
        if w not in _GENERIC_FILE_WORDS
    }
    return bool(names & {w.removesuffix("'s") for w in words(text)})


def asks_to_see_the_documents(text: str, query: str | None, filenames: Sequence[str] = ()) -> bool:
    """Asks to be shown something ("show me…", "chart", "dikhao", "दिखाओ") about the documents' subject ("FY24",
    "the company", a name from a filename), in its own words or its English form: a document question, whatever the
    router model proposed (§12.1: the visual is drawn from the documents' tables)."""
    asks = visual_intent(text) == "requested" or (query is not None and visual_intent(query) == "requested")
    about = about_the_documents(text, filenames) or (query is not None and about_the_documents(query, filenames))
    return asks and about


def asks_about_facts(text: str) -> bool:
    """A question about facts (a figure, a name, a date, which one…), not a definition or how-to, and not a question
    to the assistant itself: in a project with documents, such a question may well be about them even when the router
    model says "general" (B1)."""
    return (
        is_question(text)
        and len(words(text)) >= 3
        and not is_definitional(text)
        and not (_TO_THE_ASSISTANT.search(text))
    )


def heuristic_topic(text: str) -> str:
    """A short, stable topic label: the question's content words without years or figures ("What was the EBITDA
    margin in FY24?" → "ebitda margin"), so a follow-up about another year stays on the same topic."""
    content = [w.removesuffix("'s") for w in words(text) if w not in _TOPIC_STOP and len(w) > 1]
    label = [w for w in content if not any(c.isdigit() for c in w)] or content
    return " ".join(label[:4])[:TOPIC_MAX]


def same_topic(a: str | None, b: str | None) -> bool:
    """Topics sharing a content word are the same conversation thread ("ebitda margin" ~ "margin trend")."""
    if not a or not b:
        return False
    return a == b or bool(set(a.split()) & set(b.split()))


def same_text(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


# ------------------------------------------------------------------ inputs and outputs


@dataclass(frozen=True, slots=True)
class InterruptedAnswer:
    """The agent's last answer, cut off (barge-in, stop): what the user heard of it and the question it answered."""

    message_id: str
    heard: str
    question: str | None


@dataclass(frozen=True, slots=True)
class RouteRequest:
    utterance: str
    language: Language  # the answer language (decided by application code, services/language.py)
    history: Sequence[Message] = ()  # recent user and agent messages, oldest first
    state: ConversationState | None = None
    documents: Sequence[str] = ()  # filenames of the chat's READY documents
    interrupted: InterruptedAnswer | None = None
    available_tools: frozenset[str] = frozenset()  # tools that can run now ("web_search", §3.7)
    enabled_tools: frozenset[str] = frozenset()  # tools turned on in the config, whether they can run now or not
    # The chat's canvas as a few lines (services/canvas/conversation.screen_lines, §12.1): kinds, titles, x and series
    # labels, what the utterance points at; empty when nothing is on screen.
    screen: Sequence[str] = ()

    def live_cue(self, *texts: str | None) -> str | None:
        """The live-data cue of the utterance, else of the given texts (its standalone question), or None."""
        for text in (self.utterance, *texts):
            cue = live_data_cue(text, documents=self.documents)
            if cue is not None:
                return cue
        return None

    @property
    def previous_question(self) -> str | None:
        """The user's last message before this one (what a correction corrects)."""
        if self.interrupted is not None and self.interrupted.question:
            return self.interrupted.question
        users = [m.text for m in self.history if m.role == "user"]
        return users[-1] if users else None

    @property
    def last_answer(self) -> Message | None:
        agents = [m for m in self.history if m.role == "agent"]
        return agents[-1] if agents else None

    @property
    def agent_asked(self) -> bool:
        """The agent's last message asked the user something (a clarification, "want the breakdown?")."""
        last = self.last_answer
        if last is None:
            return False
        return heard(last).rstrip().endswith(("?", "؟")) and (last.route or {}).get("answer") != "ack"


def heard(m: Message) -> str:
    """What the user heard of a message: all of it, or of an interrupted voice answer only ``heard_text`` ("" when
    nothing of it was played)."""
    return m.heard_text if m.heard_text is not None else m.text


class RouterProposal(BaseModel):
    """What the router model returns: the intent and ``query``, the question as a standalone English question —
    None when the utterance is an English question that already stands alone, or asks nothing. For an English turn
    it is the rewritten query (follow-ups, corrections); for a Hindi or Hinglish turn the English search query too.
    Two short fields keep the router's output, and so its latency, small (§9.1: ~30 ms per output token)."""

    intent: Intent
    query: str | None


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: TurnRoute
    source: Literal["heuristic", "retrieval", "llm", "fallback"]
    proposal: RouterProposal | None = None
    overrides: tuple[str, ...] = ()
    error: str | None = None
    llm_ms: float | None = None
    reply: ReplyKind | None = None  # a fixed reply (heuristic thanks / greeting)
    language_request: bool = False  # "answer in English please": the previous question again, in the asked language
    live: str | None = None  # the live-data cue of a question that wants current data (§3.7), tool or not
    retrieval_wait_ms: float | None = None  # waited for retrieval to check a "general" proposal (B1)

    def record(self) -> dict[str, object]:
        """How the route was decided, for the message's ``route.router`` (transcript and evals)."""
        return {
            "source": self.source,
            "proposed_intent": self.proposal.intent if self.proposal is not None else None,
            "overrides": list(self.overrides),
            "error": self.error,
            "prompt": ROUTER_PROMPT_VERSION if self.source in ("llm", "fallback") else None,
            "live_cue": self.live,
            "retrieval_wait_ms": self.retrieval_wait_ms,
        }


class TurnRouter(Protocol):
    """Proposes a route for a turn (the router model). Raises when it can't; the planner then falls back."""

    async def propose(self, request: RouteRequest) -> RouterProposal: ...


# ------------------------------------------------------------------ routes without the model


def canvas_edit(req: RouteRequest) -> CanvasEdit | None:
    """The canvas edit the utterance says (keyword grammar, §12.1), when there is a canvas to edit."""
    return parse_edit(req.utterance) if req.screen else None


def needs_retrieval_by_default(intent: Intent, has_query: bool) -> bool:
    return intent in ("document_qa", "mixed", "correction") or (intent == "resume_document" and has_query)


def topic_shift(state: ConversationState | None, intent: Intent, topic: str) -> bool:
    """A content turn whose topic shares nothing with the active one (and returning to the documents from elsewhere)."""
    if intent in TOPIC_NEUTRAL_INTENTS or intent == "correction" or state is None or state.active_topic is None:
        return False
    if intent == "resume_document":
        return state.document_topic is not None and state.document_topic != state.active_topic
    return bool(topic) and not same_topic(topic, state.active_topic)


def _route(
    req: RouteRequest,
    intent: Intent,
    *,
    confidence: float,
    query: str | None = None,
    query_en: str | None = None,
    topic: str | None = None,
) -> TurnRoute:
    if topic is None:
        topic = "" if intent in TOPIC_NEUTRAL_INTENTS else heuristic_topic(query_en or query or req.utterance)
    if intent == "resume_document" and req.state is not None and req.state.document_topic:
        topic = req.state.document_topic
    return TurnRoute(
        intent=intent,
        needs_retrieval=needs_retrieval_by_default(intent, has_query=query is not None),
        rewritten_query=query,
        query_en=query_en,
        topic=topic,
        is_topic_shift=topic_shift(req.state, intent, topic),
        response_language=req.language,
        confidence=round(min(max(confidence, 0.0), 1.0), 3),
    )


def acknowledgement(text: str) -> Literal["thanks", "ack"] | None:
    """A short utterance made only of acknowledgement words ("okay", "got it", "haan theek hai") → "ack"; with a
    thanks in it ("okay thanks", "theek hai, shukriya") → "thanks". Not a question, at most six words."""
    ws = words(text)
    if not ws or len(ws) > 6 or text.rstrip().endswith("?"):
        return None
    if any(w in THANKS_WORDS for w in ws) and all(w in ACK_WORDS | THANKS_WORDS | _THANKS_FILLERS for w in ws):
        return "thanks"
    return "ack" if all(w in ACK_WORDS for w in ws) else None


def only_asks_for_a_language(text: str) -> bool:
    """The utterance asks for an answer language and nothing else ("now answer in English please")."""
    return asked_language(text) is not None and all(w in _LANGUAGE_REQUEST_WORDS for w in words(text))


_ANSWERED = frozenset({"grounded", "mixed", "general"})


def fast_route(req: RouteRequest) -> RouteDecision | None:
    """A route without the model, or None. Conservative: only turns whose type is unmistakable."""
    n = normalize(req.utterance)
    if n in STOP_PHRASES:
        return RouteDecision(_route(req, "stop", confidence=CONFIDENCE["stop"]), "heuristic")
    if canvas_edit(req) is not None:  # "make that a bar chart", "हटा दो", "pin this" (§12.1)
        return RouteDecision(_route(req, "canvas_edit", confidence=CONFIDENCE["heuristic"]), "heuristic")
    if only_asks_for_a_language(req.utterance):
        # "हिंदी में बताइए" after an answer: that question again, in the asked language (the turn's language);
        # before any answer: a short acknowledgement. The language itself is decided (and kept) by services/language.
        # Asked for English, the question is asked again in English (its English query): a 4B model answers a
        # Hinglish question in Hindi whatever the prompt says (B5).
        last, previous = req.last_answer, req.previous_question
        if last is not None and previous and (last.route or {}).get("answer") in _ANSWERED:
            r = last.route or {}
            query, query_en = r.get("rewritten_query") or previous, r.get("query_en")
            if req.language == "en" and query_en:
                query, query_en = query_en, None
            route = _route(
                req,
                "correction",
                confidence=CONFIDENCE["heuristic"],
                query=query,
                query_en=query_en,
                topic=r.get("topic") or None,
            )
            overrides = ("language request: the previous question again",)
            return RouteDecision(route, "heuristic", overrides=overrides, language_request=True)
        route = _route(req, "conversation", confidence=CONFIDENCE["heuristic"])
        return RouteDecision(route, "heuristic", reply="language")
    ack = acknowledgement(req.utterance)
    if ack == "thanks" or n in GREETING_PHRASES:
        route = _route(req, "conversation", confidence=CONFIDENCE["heuristic"])
        return RouteDecision(route, "heuristic", reply="thanks" if ack == "thanks" else "greeting")
    if ack == "ack" and req.interrupted is None and not req.agent_asked:
        return RouteDecision(_route(req, "backchannel", confidence=CONFIDENCE["heuristic"]), "heuristic")
    if (
        message_language(req.utterance) == "en"
        and standalone_question(req)
        and mentions_documents(req.utterance, req.documents)
        and not ("web_search" in req.available_tools and req.live_cue() is not None)  # the router writes its query
    ):
        return RouteDecision(_route(req, "document_qa", confidence=CONFIDENCE["heuristic"]), "heuristic")
    return None


def standalone_question(req: RouteRequest) -> bool:
    """A question that makes sense without the conversation: no back-references or follow-up opener, not a
    correction or a return to the documents, at least four words, and not an answer to something the agent asked
    or a reaction to an answer the user just cut off."""
    text = req.utterance
    return (
        is_question(text)
        and len(words(text)) >= 4
        and not refers_back(text)
        and not is_correction(text)
        and not is_resume(text)
        and not req.agent_asked
        and req.interrupted is None
        and not (req.screen and refers_to_screen(text))  # "what's the second bar?": the router reads the screen
    )


def retrieval_route(req: RouteRequest, score: float) -> RouteDecision:
    """A standalone question the documents clearly answer (the speculative retrieval came back confident before the
    router): a document question, no model wait."""
    return RouteDecision(_route(req, "document_qa", confidence=score), "retrieval")


def fallback_route(req: RouteRequest, error: str, llm_ms: float | None = None) -> RouteDecision:
    """The router failed or timed out: phase-1 behaviour, a document question on the raw utterance (retrieval and
    the abstention gate keep it honest)."""
    route = _route(req, "document_qa", confidence=CONFIDENCE["fallback"])
    return RouteDecision(route, "fallback", error=error, llm_ms=llm_ms)


# ------------------------------------------------------------------ validation


def _clean(value: str | None, limit: int) -> str | None:
    value = (value or "").strip().strip("\"'`").strip()
    return value[:limit] or None


def validate(proposal: RouterProposal, req: RouteRequest, *, llm_ms: float | None = None) -> RouteDecision:
    """Turn the model's proposal into a route; application code has the last word (module docstring)."""
    text = req.utterance
    overrides: list[str] = []
    intent: Intent = proposal.intent

    if is_resume(text) and intent not in ("resume_document", "document_qa", "mixed", "correction"):
        overrides.append(f"{intent}→resume_document: resume phrase")
        intent = "resume_document"
    if normalize(text) in STOP_PHRASES and intent != "stop":
        overrides.append(f"{intent}→stop: stop phrase")
        intent = "stop"
    if intent in SILENT_INTENTS:
        n_words = len(words(text))
        if is_question(text) and n_words > 2:
            overrides.append(f"{intent}→document_qa: asks a question")
            intent = "document_qa"
        elif intent == "stop" and n_words > 3 and not (set(words(text)) & _STOP_WORDS):
            overrides.append("stop→conversation: no stop word")
            intent = "conversation"
        elif intent == "backchannel" and n_words > 4:
            overrides.append("backchannel→conversation: too long")
            intent = "conversation"
    previous = req.previous_question
    if intent == "correction" and previous is None:
        overrides.append("correction→document_qa: nothing to correct")
        intent = "document_qa"
    # The canvas (§12.1): a question about a chart on screen is a document question (its tables answer it); an edit
    # is never a question.
    if intent == "canvas_edit" and is_question(text):
        overrides.append("canvas_edit→document_qa: asks a question")
        intent = "document_qa"
    elif intent in ("clarification", "conversation", "general_qa") and req.screen and refers_to_screen(text):
        overrides.append(f"{intent}→document_qa: asks about a chart on screen")
        intent = "document_qa"
    elif intent in ("clarification", "conversation", "general_qa") and asks_to_see_the_documents(
        text, proposal.query, req.documents
    ):  # "वालमोरा की FY24 की तिमाही आय का चार्ट दिखाओ" taken for small talk: answered "I can't show charts"
        overrides.append(f"{intent}→document_qa: asks to see the documents' figures")
        intent = "document_qa"

    # The model's query is the standalone question in English. An English turn: it is the rewritten query. A Hindi or
    # Hinglish turn: it is the English search query, and also the rewritten one when the utterance needs the
    # conversation to be understood (follow-ups, corrections) — a standalone Hindi question keeps its own words.
    proposed = _clean(proposal.query, QUERY_MAX)
    query: str | None = None
    query_en: str | None = None
    if proposed is not None and not same_text(proposed, text) and intent not in _NO_QUERY:
        if message_language(text) == "en" or is_devanagari(proposed):
            query = proposed
        else:
            # The English query searches the documents, and for a live question the web (§3.7), whatever the intent;
            # for a canvas edit the planner reads it (§12.1).
            if (
                intent in _SEARCHING
                or intent == "canvas_edit"
                or (intent in ANSWERING and req.live_cue(proposed) is not None)
            ):
                query_en = proposed
            if refers_back(text) or is_correction(text) or intent in ("correction", "resume_document"):
                query = proposed
    if intent == "correction" and query is None:
        query = f"{previous} — {text}"
        overrides.append("correction: no rewrite, previous question kept")
    if intent in _SEARCHING and query is None and query_en is None and req.agent_asked and not is_question(text):
        offer = _last_question(req)  # "yes please" to "Want the breakdown by segment?"
        if offer:
            query = offer
            overrides.append("answer to the agent's question: its question kept")
    if intent == "resume_document" and query is None and asks_beyond_resume(text):
        query = text

    confidence = CONFIDENCE["llm_overridden" if overrides else "llm"]
    route = _route(req, intent, confidence=confidence, query=query, query_en=query_en)
    return RouteDecision(route, "llm", proposal=proposal, overrides=tuple(overrides), llm_ms=llm_ms)


_SEARCHING: frozenset[Intent] = frozenset({"document_qa", "mixed", "correction", "resume_document"})
_NO_QUERY: frozenset[Intent] = frozenset({"stop", "backchannel", "conversation", "clarification"})
# Intents that answer a question: the ones a live-data tool can serve (resume only with a question).
ANSWERING: frozenset[Intent] = frozenset({"document_qa", "mixed", "general_qa", "correction", "resume_document"})


def with_live_tools(decision: RouteDecision, req: RouteRequest) -> RouteDecision:
    """The decision with ``tools=["web_search"]`` when the question needs live data and the tool is available, and
    the live-data cue recorded either way (module docstring). Applied to every route, however it was decided."""
    route = decision.route
    if route.intent not in ANSWERING or (route.intent == "resume_document" and route.rewritten_query is None):
        return decision
    cue = req.live_cue(route.rewritten_query, route.query_en)
    if cue is None:
        return decision
    tools: list[Literal["web_search"]] = ["web_search"] if "web_search" in req.available_tools else []
    return replace(decision, route=route.model_copy(update={"tools": tools}), live=cue)


_SENTENCE = re.compile(r"[^.!?।]*\?")


def _last_question(req: RouteRequest) -> str | None:
    """The last question the agent asked ("Want the breakdown by segment?")."""
    last = req.last_answer
    if last is None:
        return None
    questions = _SENTENCE.findall(strip_markers(heard(last)))
    return questions[-1].strip() if questions else None


# ------------------------------------------------------------------ the router model

ROUTER_SYSTEM_PROMPT = """\
Classify the user's latest utterance in a voice chat about their uploaded documents (users speak English, Hindi \
or Hinglish). Reply with compact one-line JSON.
intent, one of:
document_qa: asks about the documents' content, including follow-ups on what was just said ("and in FY23?", \
"why did it grow?", a yes to the assistant's offer)
general_qa: general knowledge or how-to, unrelated to the documents ("capital of France?", "chai kaise banate hain?")
mixed: needs document facts plus outside knowledge or an opinion
conversation: greeting, thanks, praise, small talk, questions about the assistant
resume_document: goes back to the documents after a digression ("back to the report")
correction: corrects the previous or interrupted question ("no, I meant FY23")
clarification: refers to something never mentioned, so it can't be understood ("what about that one?")
stop: asks the assistant to stop or be quiet ("stop", "bas karo")
backchannel: an acknowledgement that asks nothing ("okay", "haan")
query: the question as a standalone English question, resolving references from the conversation and applying \
corrections. Use null only when the utterance itself is a complete English question (don't repeat it) or asks \
nothing.
Examples: "What was revenue in FY24?" -> {"intent":"document_qa","query":null}; after it, "and FY23?" -> \
{"intent":"document_qa","query":"What was revenue in FY23?"}; "no, I meant profit" -> \
{"intent":"correction","query":"What was profit in FY24?"}"""


# Told only when something is on screen (in the user's message, so the cached system prompt stays the same).
SCREEN_HINT = (
    '(intent canvas_edit: the utterance changes a chart on screen, e.g. "make it a bar chart", "remove that", '
    '"add FY23 to it", "isko table mein dikhao"; a question about a chart is document_qa, with the query naming what '
    "it points at.)"
)


def _history_lines(req: RouteRequest) -> list[str]:
    lines = []
    for m in list(req.history)[-ROUTER_HISTORY_MESSAGES:]:
        text = " ".join(strip_markers(heard(m)).split())
        if not text:  # an answer nobody heard
            continue
        if len(text) > ROUTER_HISTORY_CHARS:
            text = text[: ROUTER_HISTORY_CHARS - 1] + "…"
        lines.append(f"{'User' if m.role == 'user' else 'Assistant'}: {text}")
    return lines


def router_messages(req: RouteRequest) -> list[LLMMessage]:
    state = req.state
    lines = [f"Documents: {', '.join(req.documents[:6]) if req.documents else 'none'}"]
    topic = state.active_topic if state is not None else None
    if topic:
        lines.append(f"Topic: {topic}")
    history = _history_lines(req)
    if history:
        lines.append("Conversation:")
        lines.extend(history)
    if req.interrupted is not None:
        played = " ".join(req.interrupted.heard.split())[:ROUTER_HISTORY_CHARS]
        lines.append(
            f'(The user cut that answer off after hearing: "{played}")'
            if played
            else "(The user cut the answer off before hearing any of it)"
        )
    if req.screen:  # only chats with a canvas: everyone else's prompt is unchanged (router-v1)
        lines.append("On screen (charts the app drew from the documents' tables):")
        lines.extend(req.screen)
        lines.append(SCREEN_HINT)
    lines.append(f"Utterance: {req.utterance}")
    return [LLMMessage("system", ROUTER_SYSTEM_PROMPT), LLMMessage("user", "\n".join(lines))]


@dataclass
class LLMTurnRouter:
    """The router model: one JSON-schema call with a small token cap (``generate_json``)."""

    llm: LLMClient
    model: str | None = None  # default: llm.router_model
    max_tokens: int = ROUTER_MAX_TOKENS
    calls: int = field(default=0, repr=False)

    async def propose(self, request: RouteRequest) -> RouterProposal:
        self.calls += 1
        return await self.llm.generate_json(
            router_messages(request), RouterProposal, model=self.model, max_tokens=self.max_tokens
        )
