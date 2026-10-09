# poc_gibberlink — Design

> **Status:** Phase −1 **complete** (smoke tests incl. network-off run) · Phase 0 **complete** (skeleton, health, frontend shell, Docker, CI) · Phase 1 **in progress** (2026-10-08).
> **Last updated:** 2026-10-08 (config moved to JSON: local + cloud template)
> **Origin:** derived from [`blueprint.md`](blueprint.md). This document records what we actually decided; where the two disagree, this one wins.

---

## 1. What we are building

A fully local, ChatGPT "voice mode"-style agent that talks with you about your uploaded documents.

**Voice-first (product principle, confirmed by the user 2026-10-09).** The primary way to use the product is a spoken conversation. Text is secondary: transcripts, summaries and search exist to *revisit* conversations, and typing is a fallback when speaking isn't possible. Every screen and default follows from this:

- Opening or starting a chat lands in **voice mode** (presence field §3.8, a large mic control, live captions, the current answer's sources); the transcript is a panel you open, and "Type instead" is a collapsed secondary input.
- Answers are written to be **heard**: 1–3 spoken sentences, details and citations on screen. Text turns use the same pipeline (`modality` parameter) with a somewhat fuller style.
- The answer pipeline is **transport-agnostic**: one service yields typed events (user message, sources, deltas, agent message, error) with cancellation; text is a thin SSE adapter, voice a WebSocket adapter that feeds deltas to TTS.
- Engineering order still builds the text path first (phase 1) because it is the same brain voice needs, and lets answer quality be debugged apart from audio; voice follows immediately after (see §10 execution order).

- You speak; the agent answers **briefly**, in voice, with citations shown on screen.
- You can **interrupt** at any time — change the question, correct it, say "stop", switch topic or language — and the agent adapts immediately. It is a conversation, not a narration.
- It answers from your documents when the question is about them (with page citations), answers from general knowledge when it isn't, and says so when the documents don't contain the answer.
- **Languages:** English and Hindi (including mixing the two).
- **Everything runs on this machine.** No cloud LLM, no cloud storage, no telemetry. The single, explicit exception is the optional **live-data tool** (web search, §3.7), which sends only a short search query off the machine. The code is structured so cloud backends can be added later through configuration (§6), but none are implemented.

### Out of scope (decided)

| Dropped | Reason |
|---|---|
| GibberLink / ggwave agent-to-agent sound mode | No second-agent demo; the product is a single human↔agent voice conversation. The project keeps the name. |
| Gujarati | English + Hindi are enough for the demo. |
| gpt-oss:20b | Doesn't fit this machine (§2). |
| WebRTC (for now) | Everything is on localhost; WebSocket audio streaming is simpler. Kept behind an interface for later. |

---

## 2. Machine and constraints

| | |
|---|---|
| Machine | Apple M4 (4P + 6E cores, 10-core GPU, Metal 4), macOS 26.6 |
| Memory | **16 GB unified** (CPU and GPU share it) |
| GPU working-set cap | **12.7 GB** (Metal `recommendedMaxWorkingSetSize`, measured) |
| Disk | ~238 GB free |
| Tooling | Python 3.12, uv, Node 24, Docker 29.8 (Desktop memory capped at **1.5 GB**), Ollama 0.40.1 (brew service), ffmpeg 8.1, espeak-ng |

Consequences:

1. **Only Qdrant runs in Docker.** Docker on macOS has no Metal GPU access, so Ollama, the backend and all ML models run natively on the host. Compose still defines backend/frontend images under a `full` profile for Linux/cloud later.
2. **gpt-oss:20b is out.** ~14 GB in Ollama exceeds the 12.7 GB GPU cap on its own; it is also a reasoning model (always emits hidden reasoning → voice latency) trained mostly on English.
3. **faster-whisper is CPU-only on Mac** (CTranslate2 has no Metal backend). `mlx-whisper` (Apple GPU) is benchmarked against it behind the same interface.
4. **Memory is the binding constraint** — see the budget in §8.

---

## 3. Architecture

```text
                       Browser (Next.js)
   ┌──────────────────────────────────────────────────────┐
   │ mic ─► Silero VAD (in-browser) ─► PCM 16 kHz ──┐     │
   │                 │ speech onset while agent talks    │
   │                 ▼                                │     │
   │          duck / stop playback ◄──────┐           │     │
   │ speaker ◄─ audio queue ◄─ TTS chunks │           │     │
   │ transcript · citations · state badges│           │     │
   └──────────────────────────────────────┼───────────┼─────┘
                         WebSocket (audio frames + JSON events)
   ┌──────────────────────────────────────┼───────────┼─────┐
   │ FastAPI (host)                       │           ▼     │
   │  VoiceSession ── server VAD (endpointing) ─► STT       │
   │       │                                        │       │
   │       │                  ┌─────────────────────┘       │
   │       │                  ▼                             │
   │       │          Router (Qwen, JSON) ──┐ speculative   │
   │       │                  │             │ retrieval     │
   │       │      ┌───────────┼──────────┐  ▼               │
   │       │      ▼           ▼          ▼  Hybrid search   │
   │       │   doc QA     general     control  (Qdrant)     │
   │       │      │           │      (stop/    + reranker   │
   │       │      ▼           │      correct)               │
   │       │   context builder (citations S1..Sn)           │
   │       │      └─────┬─────┘                             │
   │       │            ▼                                   │
   │       │   Answer (Qwen, streamed tokens)               │
   │       │            ▼                                   │
   │       │   sentence splitter ─► TTS (Kokoro) ─► audio ──┘
   │       ▼                                                │
   │  SessionState (topic, language, active docs, mode)     │
   │  SQLite: projects, documents, jobs, chats, messages    │
   └────────────────────────────────────────────────────────┘
        Ollama (host, :11434)      Qdrant (Docker, :6333)
```

### 3.1 Ingestion (once per document)

```text
upload → validate (ext, size, MIME) → SHA-256 (dedupe / versioning) → document record
→ Docling (layout, tables, OCR) → normalize → structure-aware chunking
→ BGE-M3 dense + sparse → Qdrant upsert → ingestion QA → READY
```

Chunks keep provenance: `document_id`, `version`, `chunk_id`, `page_start/end`, `heading_path`, `content_type` (paragraph/table/list), `language`. A citation says where its passage is by page, and, where there is none (Docling gives DOCX no pages), by `section`, the chunk's heading path ("4. Travel > 4.2 Domestic > 4.2.1 Hotels"); the UI shows "§ 4.2.1 Hotels" (whole path on hover), the summary and export "file.docx, § 4.2.1 Hotels". A page wins where present. Tables are never split and are stored as markdown (+ optional text summary). Tables are **also persisted as typed, cell-level datasets** (§12.1, workstream 1) so later features (visual canvas, calculator) never re-parse text. Chunk sizing (~300–700 tokens, 50–100 overlap) is a starting point to tune with evals.

### 3.2 Retrieval (per document question)

```text
query (rewritten if it's a follow-up; + the router's query_en for Hindi/Hinglish) → BGE-M3 dense + sparse
→ Qdrant prefetch (top `retrieval.prefetch_k` = 8 per leg, both queries, filtered to the project and the chat's
  documents) → RRF fusion
→ bge-reranker-v2-m3 (English passages vs query_en, Hindi passages vs the user's own words; batches of 2 on MPS)
→ top 5 → context builder (dedupe, group by section, budget, [S#] ids)
→ confidence gate → answer or abstain
```

**As tuned in phase 2** (194-question eval, `evals/`): every chunk is embedded and reranked as *document label + heading path + text*, where the label is the file name plus the document's title — tables and slides rarely name their company, so without it "Valmora's EBITDA" matched the other company's table. The gate answers when the best reranker score is at least `retrieval.min_rerank_score` (**0.02**) **and** the best passage states every fiscal year the question names (FY24, FY 2023-24, Q4FY24, Hindi digits); declining relevant passages that don't hold the answer is the answer model's job (on the eval it declines 20 of the 23 unanswerable questions the gate lets through). **Whose passage.** In a chat of several documents the gate also asks whether the passages found are about what the question names (`services/subjects.py`). The capitalised words of the question (or its English query) that are in the labels of some of the chat's documents and not of all name those documents ("Valmora" names the three Valmora documents, "Zephyra" the deck); file name words decide before title words, and words with digits ("FY24"), words under three letters, question words and company suffixes are never names. When the best passage is neither from a named document nor says the name, only the passages that are (from a named document, or saying the name) are kept, best first; when there are none the question is not covered (`Confidence.missing_subjects`, a veto like the fiscal-year one, which a caller can use for an answer that says whose documents it did find). Labels come from the store (`VectorStore.document_labels`: the longest label among 16 chunks of each document, fetched beside the search and remembered for 5 minutes); a chat of one document, a question that names no document and a store without labels are unchanged. On the eval it fixed the two answers that quoted the other company ("How many electric vehicles does Valmora run?" said 1,150, Zephyra's fleet; "the ESOP exercise price at Zephyra" said ₹ 1,150, from Valmora's report; both now decline) and changed no other question's passages. The gate's numbers didn't move: the named company's own passages score above 0.02, and 8 of the 10 other-company questions were never a matter of whose passage it was (their best passage already is the named company's: it is relevant, it just doesn't hold the answer, which the answer model declines). Changing what is embedded bumps `ingestion.chunking.version`; READY documents indexed with an older version are re-ingested in the background at startup (they stay searchable on the old index, show PROCESSING while re-ingesting, and keep their table ids when the tables are unchanged).

**A date at the end of a fiscal year (last round, item 3).** Indian fiscal years run April to March, so "31 March 2024", "March 31, 2024", "31st March, 2024", "March 2024", "31.03.2024", "31/03/24", "2024-03-31" and "31 मार्च 2024" name the end of FY24 (`retrieval.fiscal_year_ends`). The final real run answered "What was Valmora's net debt on 31 March 2024?" with ₹1,188 crore, FY23's figure from p.16 ("Recap of FY23 … Net debt stood at ₹ 1,188 crore at 31 March 2023"), 3 of 3 times; the answer is ₹831 crore (p.17). Three things let it through: the date named no fiscal year, so the gate had no period to check; every passage of "valmora annual report fy24" states FY24 through its document label ("… fy24", "2023-24"), so even "FY24" would have passed the recap of FY23; and the search and the reranker read the date, not the year. Now such a date is a period the question names (`asked_periods`: the gate, the B1 check's "about the documents' subject", the answer checks); the queries searched and reranked carry its fiscal year ("… on 31 March 2024? (FY24)", `with_fiscal_years`; the question is recorded as asked); a passage whose own text (headings and text, without the label) dates its figures to another year-end ("at 31 March 2023") and names none of the asked years is vetoed like one that misses the year (`dated_to_another_year`; a passage that only compares, "₹ 481 crore, up 45.8% on FY23", dates nothing and is left alone); when the best passage is vetoed that way, the passages the check accepts that score at least half as well go first (`prefer_stated_periods`), else the gate abstains; and the answer prompt says "Indian fiscal years run from April to March: 31 March 2024 is the end of FY24. Answer with the figure as at 31 March 2024 (FY24), not another year's" (`fiscal_year_end_note`). Measured with the real BGE-M3 and reranker over the eval corpus (the parse cache cut by heading and page, every question of `questions.jsonl` plus the repro, the old period logic against the new): 4 of 196 questions changed, all for the better. "What was Valmora's net debt on 31 March 2024?" (and q007, "at"): the recap of FY23 was the best passage (0.99) and is now out of the five kept, which are FY24's highlights, borrowings and balance sheet; the Hindi question (q128) gets p.17 ("Net debt declined to ₹ 831 crore") first instead of the recap. Answered 153 of 158 answerable and abstained on 16 of 38 unanswerable before and after; the best passage on an expected page 132 → 134, holding the answer 126 → 129. Those passages answered by the real model (qwen3:4b-instruct, voice, 3 runs each): on main the English question got "The Valmora annual report covers this in Recap of FY23, the comparative year [S1]" (the coverage check pointing at the FY23 recap) 3 of 3; now "The documents do not provide Valmora's net debt as of 31 March 2024" (an abstention: in this cut the p.17 sentence is not among the eight candidates) 3 of 3, also with main's candidates; the Hindi question "₹ 831 करोड़" 3 of 3, before and after.

### 3.3 The conversation loop — what makes it feel like a conversation

**a) Short spoken replies, details on screen.** The voice answer prompt targets at most two short sentences, about 45 words, ending at a natural point (quality round: "one to three sentences" gave three long ones, 36–45 s of speech), and code ends a short answer at the end of the sentence that reaches 45 words, or at its third sentence, never inside one (§3.4 "Answer checks"). Citations, tables and longer detail go to the transcript panel, not into speech. When an answer's evidence comes from more than one document, or from one that isn't the obvious one (the chat searches several and the conversation wasn't about this one), the prompt asks the answer to say which, in a few words ("the Zephyra investor deck says…"; `route.named_documents`): citations alone are on screen, not heard.

**b) Speak while generating.** LLM tokens stream into a sentence splitter; each sentence is synthesized and queued as soon as it's complete. Audio starts after the first sentence. Target: **~1.5–2.5 s** from end of user speech to first agent audio (to be measured, not promised).

**c) Barge-in: duck, then decide.**

```text
agent speaking → browser VAD detects speech → duck agent volume immediately (<200 ms target)
   → server transcribes the interruption
      ├─ backchannel ("mm-hmm", "okay", noise) → restore volume, continue
      └─ real speech → stop playback, cancel LLM stream + queued TTS,
                       truncate the agent's turn in history to what was actually played,
                       process the new utterance
```

Truncating to what was *heard* is what lets "no, I meant FY25" work as a correction of a half-finished answer.

**d) Turn-taking details.**
- End of user turn: ~600 ms of trailing silence (server VAD), tuned in testing.
- Browser `getUserMedia` echo cancellation on, so the agent's own voice doesn't trigger barge-in. Headphones recommended for demos.
- Retrieval starts speculatively in parallel with routing to hide latency; discarded if the router says no retrieval is needed.

### 3.4 Router

Every user turn goes through a router that returns validated JSON (Pydantic), never prose:

```python
class TurnRoute(BaseModel):
    intent: Literal[
        "document_qa", "general_qa", "mixed",      # mixed = document facts + general reasoning
        "conversation", "resume_document",
        "correction",                               # revises the previous / interrupted question
        "stop", "backchannel", "clarification",
        "canvas_edit",                              # changes a visual on screen (§12.1): "make it a bar chart"
    ]
    needs_retrieval: bool
    rewritten_query: str | None   # only for contextual follow-ups / corrections
    query_en: str | None          # English search query for non-English turns (reranker scores EN–EN best)
    tools: list[Literal["web_search"]] = []   # live data needed (§3.7)
    topic: str
    is_topic_shift: bool
    response_language: Literal["en", "hi"]
    confidence: float = Field(ge=0, le=1)
    visual: Literal["none", "suggest", "requested"] = "none"   # does the answer get a visual (§12.1)
```

Router input: the new utterance, the last few turns, the session state, and — if the agent was interrupted — the partial answer that was played. Application code owns state; the model only proposes changes. Default language rule: answer in the language of the latest utterance unless asked otherwise.

**As built (phase 3).** The schema above is the persisted `route`; the model itself proposes only `{intent, query}` (`query` = the standalone English question) and application code validates it and derives the rest (topic, topic shift, languages, confidence):

- **Keyword fast path, no model:** stop phrases ("stop", "bas", "ruko", "रुको"); acknowledgements while the agent is idle ("okay", "yeah right", "haan theek hai"); thanks and greetings; a request that only changes the language; English standalone questions that name the documents; a garbled voice transcript (below). Fixed replies need no LLM at all.
- **Garbled transcripts (quality round, item 8).** With a film playing, the real run's transcripts included "आब आब आब …" (60 times), "Just 1 employee sign here… Just 1 employee sign here…", "अप बवबवववव…", "आश़््गें।", "understandgradeelle걱", "Is used by runsTAIC traffic"; they were answered "I couldn't find that in this chat's documents" or "Could you please clarify your question?". The voice session marks a transcript garbled (`voice/speech_text.transcript_garbled`) when the recognizer's confidence says so, if it reports one (`avg_logprob` below −1.2, or above −1 with `no_speech_prob` over 0.6, `compression_ratio` over 2.4, or a 0–1 `confidence` under 0.35; read only if present: `Transcript` doesn't carry them yet), or when its text does: letters of neither script (Cyrillic, Hangul, U+FFFD), impossible Devanagari (two viramas or nuktas, a nukta on a letter that takes none, two vowel signs in a row, a word starting with a sign), a letter five times in a row, "runsTAIC", a 25-letter word, or a loop (a word four times in a row, a phrase of two to five words three times, one word two fifths of eight or more). Such a turn (`RouteRequest.garbled`) gets "Sorry, I didn't catch that. Could you say it again?" / "माफ़ कीजिए, मैं ठीक से सुन नहीं पाया। क्या आप फिर से कह सकते हैं?" (intent `clarification`, a fixed reply, no router or answer call), and nothing at all if the previous reply was that already; "stop" still stops. On the real run's 53 user transcripts: 10 of the garbled ones caught, none of the real questions; nonsense in valid Devanagari ("भोस काम्त्तर बड़ें …") and the film's English dialogue need the recognizer's confidence.
- **Speculative retrieval** runs in parallel with the router call: reused when the route keeps the query (for Hindi the reranker scores the English query), discarded otherwise. An English standalone question whose retrieval scores ≥ 0.6 before the router answers skips the router.
- **"General" questions about facts are checked against the documents (B1, 2026-10-09).** The 4B router labels document questions general, especially after a digression ("How many employees did the company have at year end?", "Which product depends on a single supplier?", Hindi profit / revenue questions): 4 of 11 such eval cases. In a chat with READY documents, a `general_qa`, `conversation` or `clarification` proposal for a question about facts (a clarification only when the question stands on its own: no "it", "that", "this", no "and…" / "what about…" opener, so "What is the hotel limit per night for level 3 and level 4 employees in tier 1 cities?", answered "which organization do you mean?" in the final end-to-end run, reaches §5.1 of the travel policy; not a definition or how-to such as "What is EBITDA?", "how is EBITDA calculated?", "X क्या होता है?", and not a question about the assistant) waits for the speculative retrieval, at most `FACT_WAIT_S` (1.5 s) after the router (it used to be checked only if retrieval had already finished, which under load it rarely had). Retrieval passes the confidence gate → `document_qa` when the match is strong (≥ 0.6) or the question names the documents' subject ("the company", "our revenue", "FY24", "at the end of the year", "कंपनी", a name from a filename), else `mixed` (and `mixed` for judgement questions); it doesn't, but the question names the documents' subject → `document_qa`, which abstains and is listed as unanswered; otherwise the general answer stands. The retrieval waited for becomes the turn's (`TurnPlan.prefetched`), never run twice; `route.router.retrieval_wait_ms` records the wait. Cost: none when retrieval finishes before the router (idle machine: ~0.4 s vs ~1.3 s), else the rest of retrieval, ≤ 1.5 s, only on such turns. Router eval: 11/11 B1 cases reach the documents or stay general as they should (7/11 with the router alone).
- **Bounds:** `num_predict` 96 and `llm.router_timeout_ms` (2.5 s); on timeout or invalid output the turn falls back to a document question (`router.source: "fallback"`).
- **Validation overrides:** resume phrases always resume; a "stop"/"backchannel" that asks something is a question; a correction with nothing before it is a document question; "yes please" to the agent's offer searches for what was offered.
- **The canvas (phase 10, §12.1).** With visuals on the chat's canvas the router's input (the user message, so the cached system prompt is unchanged, and only then: chats without a canvas get exactly the router-v1 prompt) gets an "On screen" block: the latest three panels as one line each (kind, title, x and series labels, highlight; no numbers, ~40 tokens a panel), what the utterance points at when code can tell ("the second bar" → Engineered Plastics, "the dip" → the largest fall), and a one-line description of `canvas_edit`. Canvas edits take the **keyword fast path** first (only with something on screen; `services/canvas/conversation.parse_edit`, §12.1 "Canvas edits"), else the model may say `canvas_edit` (its English `query` kept for a Hindi edit). Overrides: an edit is never a question (→ `document_qa`); a `clarification` / `conversation` / `general_qa` that is about a chart on screen ("what's the second bar?") → `document_qa`, answered from that chart's table; one that asks to *see* the documents' figures ("show", "chart", "dikhao", "दिखाओ" + the documents' subject, in the utterance or its English query) → `document_qa` (the real run had "वालमोरा की FY24 की तिमाही आय का चार्ट दिखाओ" as small talk, answered "I can't show charts"). A question about the screen is never a standalone question (the router reads the screen, so the speculative-retrieval shortcut doesn't skip it).
- **`route.visual`** (application code, the retrieval policy): "requested" when the user's words ask to see something ("show", "chart", "put … up against", "dikhao", "दिखाओ"; not "what does the report show"), "suggest" for trends, comparisons, breakdowns, rankings of categories ("which segment grew the fastest", "kaunsa … sabse zyada"), bridges ("walk me through"), headline numbers, dates or two or more periods ("Q3 FY24" is one), never for a plain fact ("dividend per share", "what share of revenue came from exports"), from the utterance, its standalone form or its English query (`canvas.planner.visual_intent`); only for `document_qa`, `mixed` and `correction` turns answered from the documents, and never for a question about the chart on screen; else "none". The answer's own figures (three or more) may still make it "suggest" when the answer drew on a table (§12.1).

| Intent | Search | Answer (`route.answer`) | `abstained` |
|---|---|---|---|
| document_qa | yes (rewritten / English query) | grounded, `[S#]` | only if not covered |
| mixed | yes | grounded + marked general knowledge; not covered → general with `general_note: "not_covered"` | false |
| general_qa | no | general (says it isn't from the documents, no citations; spoken in a chat with documents, the answer to a question about facts starts "Not from your documents, but …"); a live figure with no live data: a fixed line (§3.7) | false |
| conversation | no | short LLM reply, or a fixed ack for thanks / greetings / language requests | false |
| clarification | no | one question back | false |
| resume_document | no (yes if it also asks something) | fixed text naming the document and topic as said aloud ("the Valmora annual report", "the net profit"; never the raw topic label), or grounded | false unless grounded and not covered |
| correction | inherits the last answered question's mode | as that mode | as that mode |
| backchannel | no | "Anything else?" / "और कुछ जानना है?"; nothing if that was just said | false |
| stop | no | silent: an `event` message "Stopped", nothing spoken | false |
| canvas_edit | no | the edit applied (§12.1), then fixed text: "Done." / "हो गया।", or why not ("There's no chart on screen yet.", "It's already shown that way.") | false |

**Language rule** (first that applies): a language the user asks for (it then sticks) → the request's forced `language` → the pinned preference → the utterance's language (romanized Hindi counts as Hindi and is answered in Devanagari) → the previous answer's language. Hindi answers keep citations.

**Answering in the asked language (B5, quality round).** A request that only changes the language ("answer in English please") asks the previous question again: asked for English, in its English form (`query_en`), with a line in the question saying the user asked for English (the 4B model answered a Hinglish question in Hindi whatever the system prompt said). Every model answer's first letters are held back until they show its script (two words or so of English; the first Devanagari word of Hindi), then sent as one piece; in the wrong script, with nothing sent yet, the stream is closed and the model asked once more, insisting on the language (`route.language_retry`). If it still answers in the other script, the answer stands: the message's `language` is the script actually written (`route.language` keeps the language asked for), and the voice that speaks it follows the text's script, decided on the answer's first chunk with letters and kept for the whole answer.

**Answer checks (quality round, 2026-10-09).** The final real-model run had the spoken answer contradict the chart on screen, a guessed exchange rate, a CIN off by a digit, last year's dividend for "the dividend", "not covered" answers saved as answers, and long answers. Code now checks an answer while it streams, before anyone hears it (`services/answer_guard.py`, `AnswerGuard` between the model's stream and the turn's deltas), and records what it changed in `route.checks` (`[{check, action, text, to?}]`):

- **Coverage (item 1).** A sentence saying the documents don't cover something is checked against what answers the question already: the tables of the visual's draft on screen (§12.1, now among the answer's sources) and the strong passages (reranker ≥ 0.5). It is contradicted when one table row (with its header, caption and heading) or one passage sentence states every fiscal period, every number or code ("L3", "Q4", "5,200") and every content word the denial names (for a denial that names nothing, the question's; document names, "data", "figures", "provisions" and such don't count). Strict on purpose: asking again over a true denial invites the model to make the answer up, so "FY25 revenue" against FY24 tables and "an EBITDA margin target" against the margin stand. A contradicted first sentence closes the stream and the model is asked once more, the evidence named ("IMPORTANT: the sources do contain this: [S4][S5], the tables of the chart on screen (…)"); denied again, it is replaced by a fixed sentence pointing at the evidence ("The chart on screen shows Valmora's quarterly revenue and EBITDA, from the Valmora annual report [S4][S5]." / "The Valmora travel expense policy covers this in 5.1 Hotel limits per night [S1].") and the answer ends there. A later sentence loses its denying clause ("Q4 FY23 revenue was ₹1,711 crore [S4], but the report does not give …" → "Q4 FY23 revenue was ₹1,711 crore [S4].") or is dropped. To be checked before it is heard, the first sentence goes out 3 words behind the model until it holds a negation ("not", "no", "only", "नहीं"…, which every denial needs) and while its subject is a document whose verb hasn't come yet ("The Valmora annual report …" may go on "does not mention"; "… says" lets it go), then waits for its end. Hindi, whose negation comes last, is released the same way: 3 words behind the model, whole from its first negation ("नहीं", "नही", "न", "nahi"), apology ("माफ़", "खेद") or "केवल". A sentence *about what the documents contain* (it names a document or availability: "रिपोर्ट", "दस्तावेज़", "जानकारी", "उल्लेख", "उपलब्ध", "इस बारे में", or a contrast, "…, लेकिन …", which so often introduces what they lack) is held until its clause's final auxiliary (है, हैं, था, थे, होगा…) has come, because a negation always precedes it: with the auxiliary and no negation that clause goes out at once, the next one waits for its own (`answer_guard.hindi_hold`); "रिपोर्ट के अनुसार …" is an attribution, not that. Later sentences go out whole (speech is synthesised a sentence at a time after its first chunk anyway). A Hindi denial that opens with the missing item and names the documents late ("तिमाही आंकड़े FY24 के लिए इस रिपोर्ट में उपलब्ध नहीं हैं") has its first word or two out when it turns, and is ended there ("…") and corrected as in English; the denials the real model wrote name the documents first ("दस्तावेज़ में … की कोई जानकारी नहीं है"), so they are held from their first word and the model is asked again. A sentence whose opening was already out and whose own clause turns into a denial is ended there ("…") and corrected.
- **Live figures (item 2).** A question for a live figure (a strong or topic cue: "USD to INR today", "the share price right now", "latest news"; `live_data.asks_live_figure`) that gets no live data and no document answer is answered with a fixed line, no model call: "I can't look up live data such as today's rates, prices or news, so I won't guess a figure." (after the §3.7 notice when web search is on: "… So I won't guess today's figure."). In an answer from the documents with such a question, a sentence with a figure that is in no source and not in the question is replaced by that line (once; another is dropped).
- **Codes (item 3).** A code in the answer (letters and digits, eight or more: CIN, ISIN, GSTIN; or ten digits or more) that is in no source but one or two characters from exactly one that is (one per eight characters) is replaced by the source's ("L24119GJ1994PLC023971" → "…023871"); one near no source code is recorded as `unverified`.
- **Latest period (item 4).** A question that names no period (no FY, year, quarter, "last year", "पिछले साल") whose sources' tables give figures for several fiscal years (tables only: a paragraph's years are as often plans, "capex over FY25 and FY26") gets one line after the question: "If the sources give it for more than one period, answer for the latest, FY24, and say that it is for FY24."
- **"Not covered" answers (item 5).** A grounded answer that says the documents don't cover what was asked is an abstention (`abstained_by: "answer"`, listed among the summary's unanswered questions), however many sources it cites around it: for a question that names fiscal periods, when every period it names is only in denying clauses ("FY24 revenue was ₹7,365 crore [S1], but the documents don't give FY25" declines a question about FY25, and answers one about FY24 and FY25 in part); otherwise when the answer leads with the denial. It is saved without source chips (and its markers). The detector also knows "the policy does not cover …", "[S2] … does not give …", "are not available in the provided sources", "is not covered by …", Hindi "… नहीं दिए गए" (also "नही" written without the dot, "मौजूद / दर्ज / पता नहीं", romanized "uplabdh nahi"; a Hindi "लेकिन", "परंतु" or "जबकि" starts a clause), and no longer takes "the report shows revenue does not include other income" for one.
- **Misheard names (item 10).** A capitalised word (or two adjacent words, one capitalised) of the question that sounds like a name in the documents' labels without being spelled like it ("Wall Mora", "well Mora", "Wilmura" → Valmora: the same consonants as they sound, "vlmr"; "Mora", the end of it; `subjects.misheard_names`) is written as the documents spell it in the question the answer model reads, with a line saying so; the answer is respelled as it streams if it echoes the misheard name anyway. The user message keeps what was heard.
- **General answers in a documents chat (item 9).** Spoken (voice) general answers to questions about facts (not a definition, a how-to or small talk), or to a mixed question the documents don't cover, in a chat with READY documents start with "Not from your documents, but …" / "यह आपके दस्तावेज़ों से नहीं है, लेकिन …" (`route.not_from_documents`), the model told to go straight on (its first word lower-cased when it is a common opener). Decided for voice only: on screen the "general knowledge" label says it, but a listener has no label, and the B1 check (§3.4 above) already sends most fact questions to the documents, so the prefix is rare and is said exactly where a general answer could pass for a document fact ("How many employees does it have?"). Not for definitions or small talk, where it would be noise.
- **Length (item 7).** A short answer (voice, and text chats' default) stops at the end of the sentence that reaches 45 words (`SHORT_ANSWER_WORDS`), or its third sentence (web answers: the same, then at most one continuation, §3.7); the prompt asks for at most two short sentences, about 45 words (`answer-v2`, `general-v2`, `mixed-v2`, `live-v2`).

Measured on the real model (qwen3:4b-instruct; `tests/integration/test_answer_quality_llm.py`, and the same scenarios on main before the change; retrieval cached with each scenario's ranking, voice turns): "Show me Valmora's quarterly revenue and EBITDA for FY23 and FY24" with the quarterly tables ranked 4th and 5th: before, a chart of the 8 quarters and the answer "For FY23, the quarterly revenue and EBITDA data are not available in the provided sources", citing the other company's deck (2/2); after, the tables are S4 and S5 of the answer, which cites them and denies nothing (7/7, and 3/3 of a one-document variant). "USD to INR today": "approximately 83.50 INR per USD" (4/4) → the fixed line (8/8). "What is the dividend per share?": "₹ 15.00" → "₹15.00 for FY24" (6/6). FY25 revenue: saved as an answer with an [S1] chip (2/2) → an abstention without chips (5/5). Length over 33 answers: words p50 26 → 22, max 115 → 58; spoken (macOS `say -o`, never played) p50 13.5 → 10.0 s, max 50.1 → 22.2 s. The CIN and "Wall Mora" were right in every run before and after (the checks are a safety net, tested with scripted slips). Cost: the answer's first delta p50 668 → 871 ms in steady-state runs (+~180 ms per question p50, on an Ollama shared with other work).

### 3.5 Conversation state and memory

```json
{
  "project_id": "…",
  "chat_id": "…",
  "active_document_ids": ["doc_01"],
  "active_topic": "financial_performance",
  "previous_topic": "small_talk",
  "input_language": "hi",
  "response_language": "hi",
  "agent_state": "SPEAKING",
  "last_interrupted_message_id": "m41",
  "retrieval_enabled": true
}
```

Agent states: `IDLE → LISTENING → THINKING → SPEAKING → (INTERRUPTED → LISTENING)`. Prompt context = recent messages + the chat's compact memory summary + retrieved evidence + current utterance; never the full transcript. Live state lives in the session store for the duration of a voice session; everything said is persisted as messages (§3.9).

**As built (phase 3).** State persists per chat in `chat_states` (migration 0003): active / previous / document topic, the last document query, active document ids, input / response / preferred language, last intent, `last_interrupted_message_id`, `retrieval_enabled`. Only application code changes it, through a pure `advance(old state, what the turn did)`; writes are chained per chat and readers wait for pending ones, and a completed answer's state is written before its message is saved. The memory summary (`SummaryKind` "memory") is refreshed in the background every `llm.memory_summary_every_turns` (3) exchanges or `llm.memory_summary_token_budget` (1,500) tokens once the chat outgrows the prompt's 6-message window; it starts only when no turn is answering and is cancelled the moment one starts.

### 3.6 Failure handling

Degrade, never fake: LLM down → explicit error; Qdrant down → general chat only, document mode disabled with a message; Docling fails → document marked `FAILED` with the error; STT fails → text input still works; TTS fails → text still shown.

### 3.7 Live-data tools (web search) — in scope

Knowledge comes from the ingested documents. When a question is **partly about the documents but needs current data** ("the report says revenue grew 34% — how is the stock doing today?"), the agent can call a live-data tool.

```text
router → intent mixed/live, tools=["web_search"], query_en
   ├─ immediately: speak a short pre-synthesized filler ("Let me look that up…")      ← no added wait
   ├─ document part: retrieval + rerank as usual (runs in parallel)
   └─ web_search.stream(query) ──► results arrive one by one
          first result(s) → LLM streams a *partial* answer ("Early results say…")    ← spoken as it streams
          more results    → LLM continues / corrects ("…and a second source adds…")
          done or timeout → final sentence; sources shown on screen as [W1], [W2] (documents stay [S1]…)
   user can barge in at any point; the search is cancelled with the turn
```

Rules:

- **Application code decides** when a tool runs (router output validated by Pydantic); the LLM never calls the network itself. A question-answering turn gets `tools=["web_search"]` when its utterance (or its standalone / English question) carries a live-data cue and the tool is available. Cues (`services/live_data.py`, EN / HI / Hinglish) are conservative: *strong* time cues ("right now", "real-time", "latest news", "current share price", "current repo rate", "price today", "the stock doing today", "लाइव") always count, except when the cue's own clause goes on to name the document it asks about ("the latest price *in the price list*", "real-time monitoring requirements *in the SOP*", "as of today … *per the report*", "the current share price *in the valuation section*") or the phrase is technical ("live data feeds", "the live updates section", "real-time monitoring"); a document named *before* the cue is only a premise ("the report says revenue grew 34%; how is the stock doing today?" stays live). *Weak* time cues ("today", "this week", "आज", "अभी", "aaj", "abhi"; not when they name a meeting, mail, invoice or date: "today's agenda", "the agenda for today", "today's town hall / webinar / email / presentation", "today's date on the invoice", "आज की बैठक / प्रेजेंटेशन / तारीख़", "aaj ki meeting") and *topic* cues ("news", "latest update", "current price", "share price", "exchange rate", "weather", "ख़बर", "khabar") don't count when the question points at the documents ("the report", "the minutes", "the SOP", "the manual", "the price list", "the quotation", "the invoice", "the spec / datasheet", "according to the…", a filename) or at a past period / something stated ("FY24", "Q3", "at the end of the year", "on 31 March", "mentioned", "assumed", "an assumption", "used", "per unit", "buyback"). "Current ratio", "current rates of depreciation" (a "current … rate" needs a qualifier such as repo or exchange) and "how is the company doing now" (no stock subject) are document questions. Questions that nothing marks as about a document stay live even when ambiguous ("any news on the dividend?", "the latest update on the litigation", "अभी कंपनी का कर्ज़ कितना है?").
- Only the rewritten **search query** leaves the machine — never document text, never the transcript. It is built from the router's English standalone question: only the clauses asking for live data, phrases and parts pointing at the documents removed, filenames and document names removed, emails, URLs, long digit runs and ID-like tokens removed. **Figures the user didn't say** (an earlier answer's amounts the router folded into its question) are removed token by token: any whitespace-separated token containing a digit ("₹4512cr", "INR4512", "$1.2bn", "18.2x", "Rs.4,512", "34pc", "1:2") goes, with the word that led into it ("of", "to", "from", "Rs", "page") and its unit ("crore", "million", "per cent"), unless every number in it is in the user's own utterance, or it is a year, an FY tag, a quarter or a half ("2024", "FY24", "FY2024-25", "Q3", "Q3FY24", "H1"). Spelled-out amounts with a scale word (crore, lakh, million, billion, thousand, hundred, per cent; romanized Hindi such as "paanch hazaar" as well) go unless the user said them, and leftover punctuation is cleaned up. A query with nothing left to look up is **not sent**: fewer than two content words (not counting question and request words such as "explain", "what", "tell me", nor time words such as "today", "now", "as of") and no live topic by itself ("news", "weather", "Sensex", "exchange rate"): "Explain", "As of today", "How is it doing today?" search nothing; the tool is then dropped and the turn answers as with the tool off: no search, no notice, only the one-line "never guess current figures" hint (such questions are document questions that merely contain a cue). **By design**, an entity the router resolves from the conversation (the company's name for "the stock") can be part of the query. A Hindi turn without an English query searches nothing.
- Web facts and document facts are labelled separately in the answer and in citations; the agent must not present web data as coming from the report. Web text enters prompts without square brackets (no planted `[S1]`), and earlier answers that used the web are marked so in the history.
- Partial answers are streamed: the provider interface yields results incrementally, and answer generation starts on the first useful result (target: first spoken words < 1 s after the filler). While the search runs, the model reads the prompt's start (system, history, document passages) so only the web snippets remain to read. Later results get at most `max_continuations` one-sentence continuations after the answer; **an answer is complete when its own text is**: stopped while only a continuation was still to come, it is saved complete, not interrupted (voice: speech after the answer has played ends the turn, it is not a barge-in). A continuation that is kept is one whole sentence: voice queues it for speech at once (it is not held back for the end of the turn), so the answer can't count as fully heard while a kept continuation is still unspoken.
- **Offline guard:** with `strict_offline: true` the tool is refused unless `"web_search"` is listed in `strict_offline_exceptions`. The UI shows a "searching the web" badge whenever it runs (`tool` events).
- **Without live data:** turned off (the default) → no search and nothing said about it (one prompt line: never guess current prices, rates or news); a question for a live figure that no document answers gets a fixed line instead of the model (the 4B model said "approximately 83.50" for "USD to INR today" whatever the prompt said; §3.4 "Answer checks"). Turned on but unavailable now (SearXNG unreachable) → "I can't look up live data right now." only when the documents don't answer the question. Timeouts (`tools.web_search.timeout_s`) and failures end the search gracefully: "I couldn't get live data just now; from the report, …".
- **Page fetches** (top `fetch_pages` results): http(s) on ports 80/443 to public addresses only, connecting to the very addresses that were checked (no DNS rebinding; if one doesn't connect the next checked address is tried, at most four, sharing the connect timeout, so an IPv6-first host works on an IPv4-only network), no environment proxies, a generic browser User-Agent, no cookies kept, at most 400 KB read and inflated (one gzip/deflate layer inflated by us with a cap; stacked or other encodings refused), text extracted off the event loop. The guard is wired through httpx's pool's private `_network_backend`, so `httpx` and `httpcore` are pinned below their next minor release (`httpx<0.29`, `httpcore<1.1`) and `tests/test_web_search_provider.py` sends real requests through the real page client over a fake socket layer (the real ones trapped), asserting the connection went to the checked IP and a rebound address was refused: an upgrade that stopped honouring the attribute fails that test instead of silently skipping the guard.

**Protocol (both transports).** `tool` events: `{name: "web_search", phase: "start" | "results" | "done" | "timeout" | "failed", query, …}`; `results` adds `sources` (the new web citations, numbered `W1`, `W2`… in arrival order, never renumbered); terminal phases add `count`, `elapsed_ms` and, for `failed`, `detail`. Order within a turn: `tool start` → (`tool results`) → (terminal) → `sources` (document passages `[S#]`, then the first web results `[W#]`) → `delta`s; a `tool done` may arrive between deltas once nothing is left to add, and a continuation (`tool results`, then more `delta`s) may follow the answer several seconds later. **Only `agent_message` ends a turn.** A web citation is a `Citation` with `kind: "web"`: `{source_id: "W1", document_id: "", filename: <site>, page_start: null, page_end: null, chunk_id: "", snippet, kind: "web", url, title, site, published}`; document citations are unchanged (no `kind`). Saved answers carry `route.tools`, `route.web_search {query, provider, status, results, used, pages}`, `route.live_note`, and `route.basis`: a sorted subset of `["documents", "web", "general"]` saying what the answer drew on ("documents" if it cites a `[S#]`, "web" if it cites a `[W#]`, "general" for general-knowledge answers, mixed answers that add uncited general knowledge, and replies without citations outside document modes).

```python
class WebSearch(Protocol):            # searxng (local) | search_api (production, stub)
    def stream(self, query: str, max_results: int) -> AsyncIterator[SearchResult]: ...
```

**Decided (2026-10-08): SearXNG locally, a search API in production.**

| | Local / POC | Production |
|---|---|---|
| Provider | **SearXNG**, self-hosted in Docker on `127.0.0.1:8888` (~150–250 MB) | **Search API** (e.g. Brave Search API for an independent index, or Tavily/Exa which also return extracted page text) |
| Account / cost | none | API key (`${WEB_SEARCH_API_KEY}`), per-query pricing |
| Where the query goes | SearXNG forwards it to public engines (DuckDuckGo, Wikipedia, Brave, Google…): engines see the query + IP, no cookies/account | the vendor only, under its data-processing terms |
| Streaming partials | SearXNG answers once all engines finish, so the backend fans out **one request per engine** (`engines=` param) and yields whichever returns first | stream/page through the API's results; same `stream()` contract |
| Page content | fetch top 1–2 result pages, extract text locally | vendor-extracted content where offered, else the same local extraction |
| Reliability | fine for a demo; public engines rate-limit/block automated traffic, so **not for production** | SLA-backed |

Production additions: fallback chain (search API → SearXNG if deployed → document-only answer), Redis result cache, per-user rate limits, outbound-query audit log, query scrubbing (names/emails/numbers where possible; never document text), vendor terms review before go-live.

Local status (phase 8 built): config has `provider: searxng`, `enabled: false` by default; enabling it needs the Compose profile `websearch`, `enabled: true` and `strict_offline_exceptions: ["web_search"]` (config/README.md). In the fully offline smoke run (test 12) web search is expected to be unavailable and must degrade to a document-only answer.

### 3.8 Voice presence UI (the "breathing" particle field)

**Idea (user, 2026-10-08):** while the conversation happens, an animated presence that *breathes* with the voices of both the human and the agent. Modern and sleek. The direction (v2) combines the particle field of the [antigravity.google](https://antigravity.google) hero (thousands of short strokes that organise into rings and swirl around the cursor, blue → violet → magenta → coral → amber) with a small breathing core. Prototype: [`docs/prototypes/voice-presence.html`](prototypes/voice-presence.html). v1, a solid shader orb, is in git history.

**Concept: one presence, two voices, sound moves.**

| Who is speaking | What the screen does |
|---|---|
| Agent | The active band of rings widens, brightens and shifts **cool** (blue/violet); the core swells; every syllable sends a wave of strokes **outward** (sound leaving the agent) |
| User | The band tightens and shifts **warm** (coral/amber); waves travel **inward** (sound arriving) |
| Both (barge-in) | The core dims and the field fades as the agent ducks, then the inward waves take over, matching the audio ducking in §3.3 |
| Nobody (idle) | Faint dotted rings in the full spectrum, breathing at ~0.2 Hz; ambient specks across the page swirl into small rings around the cursor and settle when it stops |
| Thinking | The field rotates slowly and the strokes tilt into a swirl |

**Driven by the voice's shape, not just volume.** Both streams are analysed in the browser (zero added latency, no backend round trip):

- **Loudness** (RMS envelope) → band radius, width and density. Fast attack (~40 ms), slow release (~250 ms), so it breathes instead of jittering.
- **Brightness** (spectral centroid) → colour intensity within the speaker's palette.
- **Onsets** (sudden energy rise, roughly syllables) → a radial wave (outward for the agent, inward for the user).

Sources: the user's mic `MediaStream` (after browser echo cancellation, so the agent's own voice doesn't drive the user's waves) and the agent's playback graph (the TTS `GainNode` feeds an `AnalyserNode`). Both already exist in the client for VAD and playback.

**Rendering.** One WebGL2 draw call of ~8,000 instanced strokes (ambient field, presence rings, core), each a capsule with an antialiased edge; all motion is computed in the vertex shader from a handful of uniforms per frame (time, levels, brightness, think, duck, pointer, 4 outward + 4 inward wave start times). No libraries. Light theme uses normal blending on near-white; dark theme uses additive blending for a luminous look. Canvas 2D fallback draws the rings only; `prefers-reduced-motion` freezes drift, rotation and waves while the band still reflects who is speaking. The field is sized to stay clear of the captions below it.

**Screen layout (voice mode).**

```text
┌──────────────────────────────────────────────┐
│  annual_report.pdf · contract.pdf      EN · HI │  documents in scope, language
│        ·  ·    ·      ·     ·    ·   ·         │  ambient field (swirls around the cursor)
│                  ⁘⁘⁘⁘⁘⁘⁘                        │
│               ⁘⁘   ◉ core  ⁘⁘                   │  presence rings + core
│                  ⁘⁘⁘⁘⁘⁘⁘                        │
│      "FY24 revenue was ₹4,210 crore, up…"     │  live captions: agent words highlight as spoken,
│                                    p.46 ↗     │  citations appear as chips
│                 Listening…                   │  state label (also announced to screen readers)
│        [ stop ]     ( ● mic )     [ think ]   │
└──────────────────────────────────────────────┘
```

- **Live captions:** the agent's words highlight as they're spoken (Kokoro returns per-token durations); the user's partial transcript appears in a warm colour while they speak.
- **Docking:** when the visual canvas (§12.1) shows a chart, the field shrinks and docks so the visual takes the stage; it keeps breathing as the presence indicator.
- **Later states:** web search (§3.7) sends strokes orbiting outward ("searching"); a language switch briefly tints the caption badge.
- **Accessibility:** state always available as text (`aria-live`), captions on by default, colour never the only signal (wave direction and band shape differ per speaker), reduced motion respected.

**Build plan.** Phase 5 ships the field with agent-driven motion and captions; phase 6 adds the user's warm inward waves, barge-in visuals and docking; phase 9 polishes (palettes per theme, onset tuning, stroke count on low-end GPUs).

### 3.9 Projects, chats and transcripts

**Idea (user, 2026-10-08):** organise everything into projects. A project holds documents and several chats; everything persists locally; a sidebar lists projects and chats with pinning; every chat's transcript is stored and viewable; chats can be summarised on demand.

**Vocabulary** (used in UI, API and code alike):

| Term | Meaning | Replaces (earlier drafts) |
|---|---|---|
| **Project** | A container of documents and chats about one subject ("Annual report FY24", "Vendor contracts") | blueprint's optional *workspace* |
| **Document** | An uploaded file, ingested once, belonging to one project | — |
| **Chat** | One conversation inside a project, by voice and/or text, answered from the project's documents | *session* in §3.5 / §6 |
| **Message** | One utterance in a chat: from the user or the agent | *turn* |
| **Transcript** | A chat's full ordered list of messages, viewable and exportable | — |
| **Summary** | A user-facing digest of a chat, generated on demand | — (distinct from the internal *memory summary*, §3.5) |
| **Voice session** | The live WebSocket audio connection while a chat is in voice mode; not stored | — |

So "session" now only means a live connection; anything persisted is a project, document, chat or message.

**Behaviour.**

- A chat answers from **its project's documents** (Qdrant filter on `project_id`), optionally narrowed to selected documents per chat. Documents never leak across projects.
- New chats get an **automatic title** after the first answer (≤ 6 words, in the chat's language; a background job in the job queue's short lane, off the answer's path and never behind an ingestion; falls back to the first question; never replaces a title the user set).
- **Pinning:** projects and chats can be pinned; pinned items appear at the top of the sidebar.
- **Deleting** a chat removes its messages and summaries; deleting a project removes its documents, vectors, uploads, chats and summaries. Everything is local, so delete really deletes.

**Sidebar and screens.**

```text
┌────────────────────┬────────────────────────────────────────────────┐
│ ＋ New project      │  Annual report FY24  ›  FY24 margins      ⋯     │
│ ⌕ Search            │  annual_report.pdf · investor_deck.pdf  EN·HI   │
│                    │                                                │
│ PINNED             │        (voice mode: particle field §3.8,       │
│  ★ FY24 margins     │         text mode: transcript + composer)      │
│  ★ Vendor contracts │                                                │
│                    │                                                │
│ PROJECTS           │                                                │
│ ▾ Annual report FY24│                                                │
│     FY24 margins  2m│   [ Summary ]  [ Transcript ]  [ 🎙 Voice ]      │
│     Risk section  1d│                                                │
│     ＋ New chat      │                                                │
│ ▸ Vendor contracts  │                                                │
│ ▸ Board decks       │                                                │
│                    │                                                │
│ ● All systems OK    │  ← health from /health, opens the status panel │
└────────────────────┴────────────────────────────────────────────────┘
```

- **Project page:** documents (upload, ingestion status, page/chunk counts), its chats, and later a project digest built from chat summaries.
- **Chat page (voice-first):** opens in **voice mode** — presence field, large mic, live captions, the current answer's source chips. The **transcript** is a panel/drawer to revisit the conversation (always one click away, and the default view when reopening an old chat from history), and **"Type instead"** expands a text composer. Starting a new chat requests the microphone and starts listening.
- **Home:** the primary action is **Start a conversation** (in the current or a chosen project); recent chats are listed for revisiting.

**Transcripts: yes, every chat is stored.** Each message keeps:

- text (the user's from STT or typing; the agent's full generated answer), language, modality (voice/text), timestamps;
- for interrupted agent answers, both the full answer and **what was actually heard** (the barge-in truncation, §3.3), shown as "interrupted after: …";
- citations (document, page, chunk) and, for debugging, the router decision and stage latencies.

Audio itself is **not** stored by default (privacy and disk); a later `chats.store_audio` option can add it. Transcripts can be viewed in the app and **exported** as Markdown or JSON (a local download).

**Summaries ("what did we talk about").** A *Summarize* action on a chat produces, with the local LLM:

- a short overview;
- topics discussed;
- key answers with their page citations;
- questions the documents couldn't answer;
- suggested follow-ups.

Long chats are summarised in chunks and then combined (map-reduce) to fit the model's context. The summary is cached with the message it covers up to; when new messages arrive it shows "out of date — update". This is separate from the internal memory summary that keeps prompts short (§3.5), though both come from the same messages.

What counts (quality round): an answer the user cut off is never a source of facts, in the summary or the memory: the summarizer sees "(cut off by the user before it finished: incomplete, not an answer)" instead of what was heard (a heard "Product…" had become "no product depends on a single supplier"). "Questions the documents couldn't answer" lists abstained turns: the gate's (`route.abstained`), mixed answers the documents didn't cover, and grounded answers that passed the gate but say the documents don't cover what was asked (`route.abstained_by: "answer"`, B9; since the quality round also when they cite what they say around it, saved without source chips, §3.4 "Answer checks"); since B1 also routes misrouted document questions to the documents, they show up there too. Key points that only say something isn't covered are dropped (they came with a page chip), repeats are merged with their sources, and a Hindi summary written in English is asked for once more, insisting on Hindi (the language is also asked for last, in the conversation's message).

**Data model (SQLite now, Postgres later via SQLAlchemy).**

```text
projects        id, name, description, pinned_at, archived_at, created_at, updated_at
documents       id, project_id → projects, filename, mime, size_bytes, sha256, version, status,
                page_count, chunk_count, error, created_at          (+ document_versions, ingestion_jobs)
chats           id, project_id → projects, title, title_is_auto, pinned_at, document_scope (null = all),
                language, created_at, updated_at, last_message_at, archived_at
messages        id, chat_id → chats, role (user | agent | event), modality (voice | text), text,
                heard_text (agent, if interrupted), language, citations (json), route (json),
                latency (json), created_at
chat_summaries  id, chat_id → chats, kind (user | memory), content (json/markdown),
                covers_message_id, model, created_at
```

Every row carries ids that work unchanged in the cloud profile; a nullable `owner_id` is added when authentication (OIDC) is implemented.

**API.**

```text
GET/POST          /api/projects                       list (with pinned first) / create
PATCH/DELETE      /api/projects/{id}                  rename, pin, archive / delete (cascade)
GET/POST          /api/projects/{id}/documents        list / upload
DELETE            /api/documents/{id}
GET/POST          /api/projects/{id}/chats            list / create
PATCH/DELETE      /api/chats/{id}                     rename, pin, document scope / delete
GET               /api/chats/{id}/messages            transcript, paginated
GET               /api/chats/{id}/export?format=md|json   export (attachment; Content-Disposition exposed to the frontend)
GET/POST          /api/chats/{id}/summary             read / generate or refresh (?language=en|hi; unchanged chat → stored copy)
POST              /api/chats/{id}/title:regenerate    new automatic title (?force=true replaces a user title)
WS                /ws/chats/{id}                      live voice or text session for that chat
```

**Phases.** Phase 1 brings projects, documents, chats, messages, the sidebar (with pinning) and the transcript view for text chat. Phase 3 (router, state and memory) adds automatic titles, the memory summary, user-facing summaries and transcript export. Voice phases (4–6) record voice messages into the same chats.

### 3.10 Voice session protocol (WebSocket)

One WebSocket per live voice session on a chat. The backend and the voice-first chat page (§1, §3.8) build against exactly this.

**Endpoint:** `WS /ws/chats/{chat_id}/voice`. Unknown chat → close code `4404`; a project with no READY document still connects (turns abstain, as in text chat). One active voice session per chat: a second connection closes the first with code `4409`. A browser page whose `Origin` is not in `server.cors_allowed_origins` is closed with `4403` (a missing `Origin`, i.e. a non-browser client, is allowed). 4403 and 4404 are final: the client doesn't reconnect. Messages over 64 KiB close the connection.

**Audio frames (binary WebSocket messages).**

| Direction | Format |
|---|---|
| Client → server | Raw **PCM16 little-endian, mono, 16 kHz**, 20–64 ms per frame (512 samples = 1,024 bytes is typical). Captured with `getUserMedia({echoCancellation, noiseSuppression, autoGainControl})` and downsampled in an AudioWorklet. Sent continuously while the mic is on. |
| Server → client | **12-byte header + PCM16 LE mono 24 kHz** (Kokoro's rate). Header: `uint32 turn_id`, `uint32 chunk_index`, `uint32 seq` (all little-endian; `seq` counts frames within the chunk). The client plays only frames whose `turn_id` is the current agent turn and drops the rest (stale audio after an interruption). |

**Control messages (JSON text frames, one object each, field `type`).**

Client → server:

| `type` | Fields | Meaning |
|---|---|---|
| `start` | `language: "en" \| "hi" \| null` | Begin the session after mic permission. `null` = detect per utterance. |
| `barge_in_start` | `turn_id`, `played_ms` | The browser's VAD heard speech while the agent was speaking; the client has already ducked playback locally. `played_ms` = agent audio of that turn actually played so far. |
| `playback` | `turn_id`, `played_ms` | Optional progress report (~every 250 ms while playing), so the server knows what was heard even if the socket drops. |
| `playback_done` | `turn_id` | The last audio chunk of that turn finished playing. |
| `stop` | — | User pressed stop: cancel the current answer (same handling as a confirmed barge-in, without a new user turn). |
| `end` | — | Close the session cleanly. |

Server → client:

| `type` | Fields | Meaning |
|---|---|---|
| `ready` | `session_id`, `input_sample_rate: 16000`, `output_sample_rate: 24000`, `language` | Session accepted; start sending audio. |
| `state` | `state: "listening" \| "thinking" \| "speaking" \| "interrupted"` | Drives the presence field and the state label. |
| `user_speech` | `phase: "start" \| "end"` | Server VAD saw speech start / end of turn (end after `vad.end_of_turn_ms` silence). |
| `transcript_partial` | `text` | Optional live partial transcript of the current user utterance. |
| `user_message` | `message: Message` | Final transcript saved (`modality: "voice"`, detected `language`). Begins a turn; `turn_id` = the agent turn that will answer it. |
| `turn` | `turn_id` | The agent turn id for the answer that follows (sent with or right after `user_message`). |
| `sources` | same payload as the SSE `sources` event | Retrieval result for this turn. |
| `delta` | `turn_id`, `text` | Answer text as generated (captions may show it ahead of audio). |
| `audio_chunk` | `turn_id`, `chunk_index`, `text`, `duration_ms`, (`filler: true`), (`tail: true`) | Announces one synthesized clause/sentence: its exact text and audio length. Its binary frames follow. Used for word-synced captions and to compute what was heard. `filler: true` marks the "Let me look that up." spoken when a turn searches the web (§3.7): not part of the answer, never counted in `heard_text`. `tail: true` marks "It's on screen now." / "स्क्रीन पर दिखा दिया है।", spoken after the answer when its visual became ready while the answer was being heard (§12.1): the turn's last chunk, not part of the answer, never counted in `heard_text`. |
| `visual` | `turn_id`, `phase: "preparing" \| "ready" \| "failed"`, `visual_id`, (`visual`), (`detail`) | The answer's visual (§12.1, the same payload as the SSE `visual` event): a skeleton, the `Visual`, or why not (`detail: "cancelled"` when it was given up: drop the skeleton, no note). Also a canvas edit's rebuilt visual (its id kept). |
| `canvas` | `turn_id`, `panels: [Visual]` | The chat's whole canvas after a change (§12.1): a visual added, edited, removed or pinned. |
| `tool` | `turn_id`, `name: "web_search"`, `phase`, `query`, … | Live-data tool progress (§3.7, same payload as the SSE `tool` event): drives the "searching the web" badge. |
| `agent_message` | `message: Message` | Final saved agent message (full text, typed citations, `route`, `latency`; `heard_text` set only if interrupted). |
| `barge_in` | `turn_id`, `decision: "stop" \| "resume"` | Server's decision after `barge_in_start`: **stop** = cancel generation + TTS, client stops playback and flushes queued audio for that turn; **resume** = it was a backchannel/noise, client restores volume. |
| `error` | `detail`, `stage: "stt" \| "retrieval" \| "llm" \| "tts" \| "storage" \| "audio"` | The turn failed at that stage; the session stays open and keeps listening. |

**Turn-taking rules.**

1. **End of turn**: the server's VAD (Silero, `vad.*` config) on the incoming audio is the source of truth. After `end_of_turn_ms` of silence the utterance is transcribed (STT; language restricted to the configured languages), saved as a user message, and answered with `ChatTurnService` using `modality="voice"`, `length="short"`. Utterances shorter than `vad.min_speech_ms` are ignored.
2. **Speaking**: answer deltas are cut into speakable chunks (first chunk at the first clause boundary or ≤ 5 words; then sentences; never right after an abbreviation such as "Rs." or "U.S."), each synthesized by Kokoro and streamed as `audio_chunk` + binary frames while generation continues. Citation markers aren't spoken; one used as a word is said as what it is (quality round, item 6: "[W1] mentions that the rupee rose" was spoken "However, mentions that…"): as the subject of its clause (it opens the clause and a word follows) "a web source" / "one source" (several: "web sources" / "the sources"; Hindi "एक स्रोत" / "कुछ स्रोत"), after a preposition ("according to [W1]", "as reported by [S2]", "[W1] के अनुसार") the same; anywhere else it is dropped with the space before it. A garbled transcript is answered "Sorry, I didn't catch that…" (§3.4).
3. **Barge-in** (§3.3 duck-then-decide): on `barge_in_start` the server keeps listening and decides:
   - **stop** when the transcript has real words beyond acknowledgements (more than `voice.barge_in.backchannel_max_words` of them, or an interruption cue such as "wait", "no", "रुको", a question word), or when speech is still going at the deadline. It cancels the answer and saves the agent message with **`heard_text`** = the text of fully played chunks plus the proportional share (whole words) of the chunk playing at the current playback position; the new utterance becomes the next user turn.
   - **resume** for acknowledgements and hums ("okay", "mm-hmm", "M M", "haan", "achha theek hai", "हम्म"), noise, or a short burst that has ended. A partial transcript with fewer than 2 real words never decides stop on its own.

   A decision is normally sent within `voice.barge_in.decision_timeout_ms`; when the transcription of the speech is still running at that deadline (under load the snapshot STT of "Yeah, right" often arrives after 700 ms, and "still talking, no transcript yet" used to stop the answer: 3 of 4 trials in the real run), the server waits for it, at most until the acknowledgement cap (2 × the timeout, ~1.4 s); at the cap, speech still going on without a transcript stops the answer as before. The server may also send `barge_in: stop` **without** a `barge_in_start` (the user spoke while the answer was still being written and the client's VAD didn't fire); the client treats it the same way.
   - **Speech after a `resume`** (quality round, B3): "No (pause) wait, I meant…" got `resume` at the deadline (the pause) and the agent talked over the user for 4.6 s, until the sentence ended. Speech that goes on while the agent answers with no decision pending (after a `resume`, or without any `barge_in_start`) is transcribed again every 400 ms of new speech (the first time at `decision_timeout_ms` of speech, or right after the `resume`); real words that aren't an acknowledgement send `barge_in: stop` at once, while the user is still talking. The client re-ducks on its own VAD's next speech onset (a new `barge_in_start`); the protocol has no server-to-client "duck" message.
4. **Stop**: `stop` cancels like a confirmed barge-in; `route.stopped = true` as in text chat. A `stop` during a pending barge-in decision is answered with `barge_in: stop`. A `stop` while the utterance is still being transcribed covers it too: the client gets `state: interrupted`, `user_message`, `turn`, an empty stopped `agent_message`, then `state: listening`, and nothing is answered.
5. **Ignored utterances** (too short, an acknowledgement or hum, noise, a failed transcription): every `user_speech start` is still closed by `user_speech end`, followed by `barge_in: resume` if a decision was pending and the current `state` again (or `error {stage: "stt"}` first), so the client can clear its captions.
6. Everything said is persisted as messages in the chat (§3.9): the transcript view and the text endpoint see the same history. An interrupted answer records why in `route.interrupted`: `"barge_in"`, `"stop"`, or `"disconnect"` (client dropped, replaced by another tab, server shutdown).

**Ordering guarantees** (the client relies on these and still guards against stale messages):

- Within a turn the server sends `user_message`, `turn`, `sources`, then the `delta` / `audio_chunk` + frames stream, then `agent_message`. **`agent_message` comes after the turn's last binary frame**, so the client sends `playback_done` only when the answer has really finished playing. If TTS fails partway, `error {stage: "tts"}` comes before `agent_message`. A chunk's frames come in order and before the next `audio_chunk`, but other messages (`delta`, `user_speech`, `barge_in`) may arrive between them; frames are matched by their header, not by position.
- When an answer that was already saved complete is cut during playback, its `agent_message` is sent again with the same `id`, now carrying `heard_text`; the client updates it in place.
- After `barge_in: stop` (or a client `stop`) the server sends nothing more for that `turn_id`: no `delta`, `audio_chunk` or frames. The interrupted turn's `agent_message` (with `heard_text`) comes before the next turn's `user_message`.
- `turn_id` strictly increases within a session. A reconnect is a new session: the client resets its turn tracking on `ready` and reloads the transcript tail.
- `stop` while thinking (before any speech) cancels the turn and saves the agent message with `route.stopped = true` (empty or partial text), then the server returns to `listening`. Every cut voice turn ends with an agent message, even an empty one.
- After a cut, the state goes straight to `thinking` when the interrupting utterance is already being answered.
- **Live data (§3.7).** A turn that searches the web sends `tool` messages after `turn` (never after a cut). Its filler `audio_chunk` (`filler: true`, chunk 0) and all its frames come right after `tool start` and **before `sources`**: the one audio that may precede `sources` (it carries no answer text and no citation; `sources` still precedes every `delta`). `state: speaking` comes with the answer's first audio, not the filler's. `tool done` may arrive between deltas, and a continuation (`tool results`, more `delta`s and `audio_chunk`s) may follow seconds after the answer's text: only `agent_message` ends the turn. The answer's last sentence is spoken as soon as its text is complete, not after the continuation, and a kept continuation sentence is spoken as soon as it is kept, not at the end of the turn. Speech after the answer has fully played, while only a continuation was still to come, is not a barge-in: no `barge_in` decision, no `interrupted` state; the answer's `agent_message` (complete, no `heard_text`) comes before the next `user_message`.
- **The canvas (§12.1).** An answer's visual starts as a draft built by code when the turn's retrieval returns; its `visual` and `canvas` messages, all carrying the answer's `turn_id`, are **held until the turn's first answer `audio_chunk` and its frames, then sent at once** (so they never come before the first audio, nor `sources`; a turn without any audio sends them before `agent_message`). A draft the planner refines (once the answer's text is complete) is replaced in place: a second `visual ready` with the same `visual_id`, then `canvas`, **before or after `agent_message`**, as the planner finishes (typically 3–5 s after the answer's text); or withdrawn: `visual failed` (a quiet note) and `canvas` without it. `agent_message` doesn't wait for the visual, and the turn still ends with it (`playback_done`). A requested visual sends `visual preparing` first (a skeleton, at once followed by the draft's `ready`) and ends with `visual ready` + `canvas`, or `visual failed`; a suggested one sends only `ready` + `canvas`, or nothing. A turn cut before its first audio sends no visual at all (its draft is withdrawn). **Exceptions** to the rules above, for the visual only: its messages may follow the turn's `agent_message`, and a cut (`barge_in: stop`) doesn't stop them (a visual already being refined isn't cancelled by the cut, only by what comes next: a new turn that needs the model, the client's `stop`, the session closing; the draft on screen then stays, and without one `visual failed {detail: "cancelled"}` follows, after the next turn's `user_message` and `turn`, if a skeleton was shown). When `visual ready` arrives while the answer is still being sent or played (for a draft the planner is checking: once the planner has kept or replaced it), the turn speaks its tail: one more `audio_chunk {tail: true}` + frames after the answer's last chunk, before `agent_message` when the answer's audio was still being sent, else after it (the second exception to "`agent_message` after the last frame": only while the client is still playing the turn's audio, at least 0.3 s before it should end, so the client plays it as part of the turn before `playback_done`). No tail after the answer was heard (playback done), when the turn was cut, while the user is talking, when the visual failed, or when the answer's speech already named the chart ("The chart shows …", quality round); for a draft the planner is checking, the tail waits until the visual is settled (done), not only until the planner has answered. A canvas edit's `visual` / `canvas` come inside its turn, after `sources` and before its `delta` ("Done.").

**Startup.** The backend preloads the embedder, reranker, STT and TTS models and the Ollama model (every Ollama request uses the same `num_ctx` and `keep_alive`, `OllamaLLM.runtime_options`, so nothing reloads it), then has it read the router's prompt, the answers' system prompt and the visual planner's system prompt once (B6: the first routed turns used to read the router prompt cold, 3.0 s instead of 1.6 s, past the 2.5 s router timeout), at startup, in the background (~20–35 s, reported by `/health`), so the first spoken question isn't slowed by model loading (§9 measured ~3.7 s for a cold reranker).

Repeated input errors (odd-length audio frames, VAD failures) are reported at most once per kind every 5 s.

---

## 4. Model stack

| Role | Model | Size | Runs on | Notes |
|---|---|---|---|---|
| LLM (router + answers) | **qwen3:4b-instruct** (Ollama, Q4) | 2.5 GB | GPU | Non-thinking 2507 release; 34 tok/s, first sentence 0.6 s |
| LLM (kept, not default) | **qwen3:8b** (Ollama, Q4) | 5.2 GB | GPU | Better Hindi, 2× slower; candidate for Hindi-only use |
| Embeddings | **BAAI/bge-m3** | 2.3 GB | GPU (MPS, fp16) | Dense (1024-d) + sparse in one pass; multilingual |
| Reranker | **BAAI/bge-reranker-v2-m3** | 2.3 GB | GPU (MPS, fp16) | Cross-encoder; top 20 → top 5 |
| Document parsing | **Docling** (layout, TableFormer, OCR) | ~1 GB | CPU/GPU | Loaded only during ingestion; macOS Vision OCR available |
| VAD | **Silero VAD** | ~2 MB | Browser + server | Bundled with packages |
| STT | **Whisper** — `small` vs `large-v3-turbo`, faster-whisper (CPU) vs mlx-whisper (GPU) | 0.5–1.6 GB | CPU or GPU | Keep one after Phase −1 benchmark |
| TTS | **Kokoro-82M** | 0.33 GB | CPU/GPU | English voices (`af_*`, `am_*`, `bf_*`, `bm_*`) + Hindi (`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`) |
| Vector DB | **Qdrant v1.19.2** | — | Docker | Not a model |

Licenses (verify again before any redistribution): Qwen3 Apache-2.0, BGE-M3 MIT, Kokoro Apache-2.0, Docling MIT (individual models vary), Qdrant Apache-2.0, Whisper MIT.

---

## 5. Where data lives

```text
data/                         (gitignored)
├── uploads/                  original files (object store: filesystem provider)
├── processed/                Docling output (structured JSON / markdown)
├── sqlite/app.db             projects, documents, document_versions, ingestion_jobs, chats, messages, chat_summaries (§3.9)
├── qdrant/                   Qdrant storage (bind-mounted into the container)
└── models/
    ├── huggingface/          HF_HOME — bge-m3, reranker, whisper variants, kokoro
    └── docling/              docling-tools artifacts (artifacts_path)
```

Ollama models live in `~/.ollama/models` (Ollama's own store).

---

## 6. Local vs cloud: configuration and providers

Goal: the same backend and frontend run locally or on a server; switching is a config file, plus implementing whichever placeholder providers the deployment needs.

### 6.1 Config files

| File | Loaded? | Content |
|---|---|---|
| [`config/local.config.json`](../config/local.config.json) | **yes, by default** | Everything local; cloud-only fields are `null` / empty |
| [`config/cloud.config.json`](../config/cloud.config.json) | **no** — template only | Mock server deployment: `*.example.com` hosts, `${VAR}` secrets, CUDA devices |
| [`config/README.md`](../config/README.md) | — | Loading rules, deploy steps, provider status |

Both files have **identical keys** (145) and are validated by one Pydantic schema; a test loads both so the template can't drift.

Top-level sections: `profile`, `strict_offline`, `app`, `server`, `auth`, `client`, `llm`, `embeddings`, `reranker`, `vector_store`, `metadata_db`, `object_store`, `ingestion`, `retrieval`, `stt`, `vad`, `tts`, `voice`, `audio_transport`, `job_queue`, `session_store`, `event_bus`, `model_cache`, `observability`.

### 6.2 Loading

```text
APP_CONFIG_FILE  (default config/local.config.json)
  → parse JSON
  → resolve "${VAR}" strings from env        (missing var → startup error; secrets never in files)
  → env overrides with "__" nesting          (LLM__CHAT_MODEL=qwen3:4b-instruct)
  → Pydantic validation
  → strict_offline guard
  → container.py: registry[(capability, provider)] → instances
```

### 6.3 Rules

- Each provider class declares `is_remote`. With `strict_offline: true`, startup **fails** if any remote provider is configured or any URL is not loopback, and the process sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`.
- Providers are chosen per capability, so mixed setups (local LLM + S3, etc.) need no code change.
- Placeholder providers exist so wiring, validation and tests are real; they raise `NotImplementedError("<Provider> is a cloud placeholder")`. A test boots with `cloud.config.json` and asserts every capability resolves to the intended class.
- `services/` imports only interfaces, never concrete providers.
- Self-hosted providers take `url` / `api_key` / TLS / timeout settings from day one, so moving them to a server is config-only.
- Files are never passed between components as local paths: everything goes through `ObjectStore`; Docling reads from a temp copy. This keeps S3 a drop-in.
- Metadata DB uses SQLAlchemy, so Postgres is mostly a URL change.
- Ollama provider rejects `*-cloud` model tags.
- `strict_offline_exceptions` (default `[]`) is the only way a remote capability runs in the local profile; currently only `web_search` may be listed.
- `config/versions.json` pins model names/revisions, image tags, `embedding_model`, `embedding_dim`, `chunking_version` and prompt versions. Changing the embedding model requires an explicit re-index.

### 6.4 Provider matrix

| Capability | Built in this POC | Placeholder |
|---|---|---|
| llm | `ollama` | `openai_compatible` |
| embeddings / reranker | `bge_m3` / `bge_reranker` (mps, cuda, cpu) | — |
| vector_store | `qdrant` (local or remote) | — |
| metadata_db | `sqlite` | `postgres` |
| object_store | `filesystem` | `s3` |
| ingestion | `docling` | — |
| stt | `mlx_whisper` (macOS), `faster_whisper` (cpu, cuda) | — |
| vad / tts | `silero` / `kokoro` | — |
| audio_transport | `websocket` | `webrtc` |
| auth | `none` | `oidc` |
| job_queue / session_store / event_bus | `in_process` / `in_memory` | `redis` |
| observability | `none` | `prometheus`, `otlp` |
| tools.web_search | `searxng` (local, phase 8) | `search_api` (production) |

### 6.5 Frontend: cloud-ready by construction

- **No build-time environment baking.** The Next.js server reads one variable, `BACKEND_URL`, at runtime, so one image runs anywhere.
- Everything else comes from the backend at runtime: `GET /api/config/public` returns the `client` section plus the non-secret `auth` fields (provider, issuer, client id).
- Auth layer present from the start: `none` locally (no login screen); `oidc` placeholder for deployment.
- The browser talks to the backend directly over HTTPS + WebSocket (`wss://` in cloud); CORS origins come from `server.cors_allowed_origins`.
- The backend is stateless per request apart from the session store, so it can scale horizontally once the `redis` providers exist.

### 6.6 What still isn't config-only

Running multiple backend replicas (needs `redis` providers), real authentication (`oidc`), HTTPS termination (reverse proxy / load balancer), moving existing vectors (re-index or Qdrant snapshot), and internet-grade audio (`webrtc`).

### Interfaces

```python
class LLMClient(Protocol):            # ollama | openai_compatible(stub) | mock
    async def generate(self, messages, **kw) -> str: ...
    def stream(self, messages, **kw) -> AsyncIterator[str]: ...
    async def generate_json(self, messages, schema: type[BaseModel], **kw) -> BaseModel: ...

class Embedder(Protocol):             # bge_m3 | cloud(stub)
    async def embed(self, texts: list[str]) -> list[DenseSparse]: ...

class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...

class VectorStore(Protocol):          # qdrant (local URL or cloud URL)
    async def upsert(self, chunks: list[IndexedChunk]) -> None: ...
    async def hybrid_search(self, q: DenseSparse, filters: RetrievalFilters, top_k: int) -> list[RetrievedChunk]: ...
    async def delete_document(self, document_id: str) -> None: ...

class DocumentParser(Protocol):       # docling
    async def parse(self, path: Path) -> ParsedDocument: ...

class ObjectStore(Protocol):          # filesystem | s3(stub)
class MetadataDB(Protocol):           # sqlite | postgres(stub)
class SpeechRecognizer(Protocol):     # faster_whisper | mlx_whisper | cloud(stub)
    async def transcribe(self, pcm: bytes, language_hint: str | None) -> Transcript: ...
class SpeechSynthesizer(Protocol):    # kokoro | cloud(stub)
    def stream(self, sentences: AsyncIterator[str], language: str) -> AsyncIterator[AudioChunk]: ...
class VoiceActivityDetector(Protocol):# silero
class AudioTransport(Protocol):       # websocket | webrtc(stub)
class JobQueue(Protocol):             # in_process | redis(stub); submit(name, job, lane=) with lanes "long" (ingestion, job_queue.concurrency workers) and "short" (titles)
class SessionStore(Protocol):         # in_memory | redis(stub)
class EventBus(Protocol):             # in_process | redis(stub)
```

Business logic (`services/`) depends only on these interfaces, never on a concrete provider.

---

## 7. Repository layout

```text
poc_gibberlink/
├── README.md  Makefile  .env.example  .gitignore
├── docs/                    blueprint.md (original), DESIGN.md (this)
├── config/                  local.config.json cloud.config.json (template) versions.json README.md
├── infra/
│   ├── docker-compose.yml   qdrant (default); backend + frontend under profile "full"
│   └── docker/              backend.Dockerfile frontend.Dockerfile
├── backend/                 Python 3.12 · uv · FastAPI
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py  settings.py  container.py
│   │   ├── api/             health models documents chat voice ws
│   │   ├── domain/          Document, Chunk, TurnRoute, SessionState, Event (Pydantic)
│   │   ├── providers/       base.py, registry.py + one module per capability group (becomes a package as it grows)
│   │   ├── services/
│   │   │   ├── ingestion/   pipeline normalizer chunker ingestion_qa
│   │   │   ├── retrieval/   hybrid context_builder confidence
│   │   │   ├── agent/       router prompts memory answer citations
│   │   │   └── voice/       session turn_taking barge_in sentence_splitter
│   │   └── observability/   structured logs, per-stage latency timers
│   └── tests/               unit tests use mock providers
├── frontend/                Next.js (App Router) · TypeScript
│   ├── app/  components/    DocumentPanel Transcript Citation AgentState LanguageBadge
│   └── lib/                 runtime-config.ts api.ts ws.ts audio.ts vad.ts (@ricky0123/vad-web) auth.ts
├── evals/                   questions.jsonl retrieval_cases.jsonl conversation_cases.jsonl
├── scripts/
│   ├── setup/               download_models.sh
│   └── smoke/               01_ollama.py … (Phase −1)
└── data/                    see §5
```

---

## 8. Memory budget (16 GB unified, 12.7 GB GPU cap)

| Component | When loaded | Measured (2026-10-10) |
|---|---|---|
| macOS + browser + editor | always | ~5 GB |
| Docker VM (Qdrant) | always | ≤1.5 GB cap (Qdrant itself ~270 MB) |
| Ollama runner, qwen3:4b-instruct, `num_ctx` 8192, q8_0 KV | conversation | weights 2.4 GB + KV 0.61 GB + compute 0.17 GB (`ollama ps`: 3.3 GB) |
| … its prompt cache in RAM (llama-server `--cache-ram`) | grows with use | **up to 8 GiB by default** (measured 8.1 GB: 108 prompts); **≤ 1 GiB with `LLAMA_ARG_CACHE_RAM=1024`** (setup notes, §9) |
| Backend, every conversation model loaded and used | conversation | **5.3 GB** (peak 6.1): BGE-M3 + reranker fp16 on MPS 2.2 GB, mlx-whisper small 0.46 GB + MLX buffer cache ≤ 256 MiB, Kokoro + G2P on the CPU ~0.7–0.9 GB, Python/torch the rest |
| Docling models | ingestion only, released when ingestion drains | 1.7 GB peak |

Measured (§9.5): with Safari/Chrome/Claude app open, the machine was already swapping before our stack loaded. **Demo rule: close browsers other than the app's tab and heavy apps.**

**Where the memory went (final real-model run, 14.7 of 15.4 GB of swap in use).** The backend's footprint was 5.9 GB (peak 6.5) and Ollama's `llama-server` 9.3 GB while `ollama ps` said 3.3 GB. Not `OLLAMA_NUM_PARALLEL` (the runner runs with `-np 1`), not the KV cache (612 MiB at 8192 tokens, q8_0), not mmap accounting (the mapped weights aren't even in the footprint): it is llama-server's host prompt cache. Ollama 0.40 doesn't pass `--cache-ram`, so llama-server keeps the KV state of every distinct prompt it has served (~75 KB per token at q8_0, ~70 MB for a 1,000-token prompt) until 8 GiB; a voice turn makes 3–6 model calls (router, answer, warm-ups, titles, memory, the visual planner), so about 30 turns fill it, and it sits there, idle, pushing everything else into swap. Measured on a private `ollama serve` (same flags): +70 MB of footprint per distinct prompt with the default (3.5 GB after 40); with `LLAMA_ARG_CACHE_RAM=1024` the cache holds 15 prompts (873 MiB) and the footprint plateaus at 2.0–2.5 GB, while reusing a cached prefix is unaffected (the answer's prefix after 8 other prompts: first token 190 ms vs 254 ms). The app can't set this: it is the Ollama service's environment (setup notes). Unloading the model (`ollama stop`) frees it at once (swap 11.5 → 3.4 GB measured).

**`num_ctx` 4096 vs 8192** (same server): KV 306 vs 612 MiB, compute buffers 79 vs 166 MiB, so 0.39 GB saved; first token identical (1,157-token prompt cold 3.26 s, cached prefix 122 ms, in both). But the prompts don't all fit: the largest prompt in a day of logs was 2,685 tokens (p99 1,767), and a typed answer may carry 3,000 tokens of evidence plus history and 768 output tokens; the summaries' map-reduce window (`num_ctx − 3000`, ×0.9) would shrink from 4,672 to 986 tokens, five times the calls. **Kept at 8192**: the prompt cache, not the context, was the memory.

**Backend** (phys_footprint after each model's preload and some inference): mlx-whisper +1.5 GB, of which 751 MiB was MLX's cache of freed Metal buffers, kept without limit; capped at 256 MiB (`MLX_CACHE_LIMIT_BYTES`, +17 ms per transcription p50; 64 MiB cost +31 ms). Kokoro +0.7–0.9 GB on the CPU (weights, spaCy, misaki's lexicons); BGE-M3 +2.1–2.3 GB (1.1 GB on MPS, the rest CPU-side from loading), the reranker +1.3 GB. Docling is lazy and released after ingestion; nothing else can wait (the first spoken question needs VAD, STT, TTS, embedder and reranker). Whole stack: ~17.6 GB before (backend 5.9 + runner 9.3 + mapped weights 2.4) → ~10 GB with the setting (backend 5.3 + runner ≤ 2.5 + weights 2.4).

---

## 9. Phase −1: downloads and smoke tests

Each smoke test is a self-contained script in `scripts/smoke/` (PEP 723 inline dependencies, run with `uv run`).

| # | Component | Checks | Status |
|---|---|---|---|
| 01 | Ollama + qwen3 8b / 4b-instruct | TTFT + tokens/s (thinking off), reasoning leak, Hindi, router JSON (full vs slim schema) on 10 EN/HI/Hinglish turns, memory | ✅ ran — see §9.1 |
| 02 | Qdrant | loopback-only, telemetry off, dense+sparse collection, dense/sparse/RRF ranking, identifier rescue, doc filter, delete-by-doc, persistence, latency | ✅ **15/15** — p50 2.7 ms, p95 4.1 ms, 267 MB |
| 03 | BGE-M3 | offline load, dense+sparse on MPS, EN↔HI cross-lingual, dense/sparse/hybrid retrieval, speed | ✅ 13/14 — see §9.2 |
| 04 | Reranker | offline load, top-1 on EN/HI/Hinglish/identifier, FY24 vs FY23, abstention, latency | ✅ 7/8 — see §9.2 |
| 05 | Docling | PDF (table, headings, pages), scanned PDF (OCR), Hindi DOCX, PPTX, HybridChunker | ✅ 17/17 — see §9.4 |
| 06–07 | faster-whisper vs mlx-whisper | 20 synthetic EN/HI clips (Kokoro + macOS `say`): WER/CER, language ID, latency, memory | ✅ — see §9.3 |
| 08 | Silero VAD | segmentation, mid-sentence pause, noise, streaming start/end delays | ✅ 8/8 (Python) — see §9.4; browser in 10 |
| 09 | Kokoro | offline EN + HI, CPU vs MPS, time to first audio vs chunk length | ✅ 10/10 — see §9.3 |
| 10 | Echo / barge-in | 10a browser VAD (automated) · 10b real mic over speakers (manual, `scripts/smoke/web/mic.html`) | ✅ 10a 5/5 · 10b (user, speakers): **0 false barge-ins** while agent spoke; real speech → duck → full stop ✅ |
| 11 | End-to-end latency | clip → STT → router ∥ retrieval → rerank → first chunk → TTS | ✅ ran — 4.3 s + VAD ≈ 5.0–5.4 s; optimizations §9.5 |
| 12 | Offline | Wi-Fi off, `scripts/smoke/run_offline.sh` (uv cache only, HF offline) → `data/smoke/offline_run.log` | ✅ all 9 pass with network off (05 Docling re-run 22:08 after OpenCV fix: 17/17, see §9.6) |
| 13 | Memory stress | all conversation models loaded together | ✅ 14.2 GB used / 2.9 GB free, no swap growth (browsers closed) |

### 9.1 Smoke test 01 results (Ollama)

| | qwen3:4b-instruct | qwen3:8b |
|---|---|---|
| Cold load | 1.1 s | 4.1 s |
| First token / first sentence (EN) | 111 ms / 0.60 s | 176 ms / 1.30 s |
| First sentence (HI) | 1.03 s | 4.98 s |
| Generation speed | 34 tok/s | 18 tok/s |
| Router, full schema (10 cases) | 9/10 intent · 2.7 s median | 10/10 · 4.1 s |
| Router, slim schema | 9/10 intent · **1.08 s** median | 10/10 · 2.4 s |
| GPU memory (ctx 8192) | 3.2 GB | 6.1 GB |

Findings:

- **`qwen3:4b` is the always-thinking 2507 release** (same weights as `qwen3:4b-thinking`); it leaks reasoning into answers even with `think: false`. Use **`qwen3:4b-instruct`**. `qwen3:8b` is the original hybrid release and turns thinking off cleanly.
- **Router latency is output-length bound** (prompt processing ~0.1–0.3 s; the rest is generating JSON). The slim schema (`intent, retrieve, lang, query`) halves it.
- **As built (phase 3):** with only `{intent, query}` in the output, a routed turn takes p50 ≈ 0.65–0.7 s, p95 ≈ 0.9–1.0 s (~395 prompt tokens prefill in ~110 ms because the static prefix stays cached; ~18 output tokens at ~30 ms each). Cold prompt prefill runs at ~370 tokens/s (a 1.6k-token prompt ≈ 4.3 s) vs ~170 ms with a cached prefix, and a router call between two answers doesn't evict the answer's cached prefix. 43/43 intents on `backend/tests/integration/router_cases.json` (EN/HI/Hinglish; the prompt was tuned on that set, so this is optimistic).
- **Hindi costs ~0.75–0.9 tokens per character**, so Hindi answers start much later on 8b (5 s) — too slow for voice.
- **Both models together don't fit the GPU budget** alongside embeddings, reranker, Whisper and Kokoro (~13.5 GB > 12.7 GB cap). Pick one model for the voice loop.
- 4b-instruct router misses: `mixed` → `document_qa` (harmless — still retrieves) and, on the slim schema, a correction without `retrieve=true`. The second is fixed deterministically: a `correction` inherits the retrieval decision of the turn it corrects (app logic, not the model).

### 9.2 Smoke tests 03–04 results (BGE-M3, reranker)

Corpus: 13 annual-report / contract passages (`scripts/smoke/fixtures/finance_corpus.json`) with FY23/FY24 distractors, a product code and one Hindi passage; 8 queries (EN, HI, Hinglish, identifier, both cross-lingual directions).

| | Result |
|---|---|
| BGE-M3 load (offline, MPS fp16) | 1.9–3.2 s, 1.15 GB |
| Cross-lingual cosine (EN↔HI, EN↔Hinglish) | 0.87–0.93 vs 0.34–0.38 unrelated |
| Dense top-1 / sparse top-1 / hybrid (RRF) top-1 | 8/8 · 5/8 · 5/8 |
| Hybrid top-3 recall | **8/8** |
| Query embed latency / ingestion throughput | 39 ms p50 · ~11 chunks/s (400-token chunks) |
| Reranker top-1 | **8/8** |
| FY24 vs FY23 reranker score | 1.000 vs 0.476 |
| Reranker: unanswerable questions (EN, HI) | top score 0.000–0.001 |
| Reranker: lowest answerable top score | 0.067 (Hindi paraphrase), 0.291 (Hinglish) |
| Rerank 20 candidates | 202 ms p50, 1.1 GB |

Findings → design changes:

- **Libraries phone home unless given local paths.** FlagEmbedding calls `snapshot_download` on the repo id and, offline, rejects our deliberately partial snapshot. Providers must resolve models to a local snapshot directory (`local_snapshot(repo_id)`) and never pass bare repo ids. `strict_offline` caught this.
- **Equal-weight RRF demotes cross-lingual answers to #2**: sparse matching is noise when query and document languages differ. Hybrid is judged on recall (top-3 = 8/8); the **reranker owns final order**. Optional later: down-weight sparse when query language ≠ document language.
- **Reranker scores are not calibrated across languages.** A fixed 0.3 abstention cutoff would reject real Hindi questions (and, measured in phase 4 on the smoke documents, even Whisper's exact transcript of an English revenue question scored 0.282: answerable questions scored 0.074–0.983, unanswerable ones ≤ 0.013, so `retrieval.min_rerank_score` became 0.05 until phase 2). **Phase 2 re-tuned the gate on the 194-question eval:** unanswerable near misses score as high as answers (median 0.45; 12/12 "FY25" questions above 0.9), so no score threshold separates them — a fiscal-year check does (near-miss-year questions let through 12 → 5); Hindi calibration was mostly a cross-lingual-pair problem, and scoring Hindi passages with the Hindi question puts Hindi answerables at ≥ 0.11; top score + gap + dense similarity was tried, but dense similarity separates poorly (AUC 0.65) and was not adopted. `min_rerank_score` is **0.02**: 3/156 false abstains, 23/38 unanswerable questions through the gate, 3/38 answered end to end. A follow-up looked at the 10 "other company" questions (all through the gate): two were answered from the other company's document (§3.2, "Whose passage", fixed), the other eight have the named company's own passage on top (Valmora's Contract Logistics page for "Valmora's order book in Contract Logistics"), so no check of whose passage it is can refuse them; they stay the answer model's to decline, as the near misses do.

### 9.3 Smoke tests 06–07, 09 results (speech)

**STT** — 20 clips (5 EN + 5 HI sentences × Kokoro and macOS `say` Rishi/Lekha voices), warm, greedy decoding, audio passed as 16 kHz PCM arrays:

| Variant | p50 latency | EN WER | HI CER | Lang ID | Memory |
|---|---|---|---|---|---|
| faster-whisper small (CPU int8) | 1.96 s | 0.00 | 0.12 | 20/20 | 1.1 GB |
| faster-whisper large-v3-turbo (CPU int8) | 9.67 s | 0.00 | 0.04 | 20/20 | 1.8 GB |
| **mlx-whisper small (GPU)** | **0.44 s** | 0.00 | 0.12 | 20/20 | 1.2 GB |
| mlx-whisper large-v3-turbo (GPU) | 1.80 s | 0.00 | 0.05 | 20/20 | 2.1 GB |

- **Default: mlx-whisper small.** Latency is ~constant per utterance (Whisper always encodes a 30 s window), so turbo's 1.8 s is a fixed tax on every turn.
- Small's Hindi errors are spelling-level (मुनापा for मुनाफा). Routed through 4b-instruct, 4/5 misspelled transcripts produced the same English search query as the correct text. Turbo remains a config switch.
- faster-whisper is not viable on this Mac (CPU-only); it stays the CUDA/cloud provider.
- **Pass audio as PCM arrays, never files**: faster-whisper's file path breaks on current PyAV (`metadata_errors` kwarg removed). The backend receives PCM over WebSocket anyway.
- Finding for the Hindi eval: 4b-instruct translated राजस्व (revenue) as "tax" even from correct text.

**Names: a vocabulary prompt per chat (2026-10-10).** Whisper small spelled the documents' names by sound ("Valmora" → "Mora's", "Vimora", "Vamora"; "Zephyra" → "Zephyr", "जफर"), and the wrong name spread into answers, titles, charts and web queries. `services/voice/vocabulary.py` builds an `initial_prompt` per language from the chat's READY documents, cached until they change: title phrases that contain a file-name word ("Valmora Industries"), table phrases ranked by how many tables use them (segments, facilities, people; the documents take turns; phrases whose words are each a single token of Whisper's tokenizer, i.e. words it knows, last), acronyms used across tables ("EBITDA"), ~110 tokens; the Hindi prompt puts the names, also in Devanagari (asked of the LLM once per name, the documents' own spelling preferred), in a short Devanagari frame with the Hindi documents' headings and first-column entries (a Latin-only prompt made Whisper write Hindi in Latin script; Devanagari terms alone didn't help the names). A lexicon of the names' unfamiliar words restores near misses Whisper still makes ("Vamora's" → "Valmora's", "Talaja" → "Taloja"; only a capitalised multi-token Latin word within 1–2 edits of exactly one entry). The prompted decode is one greedy pass capped by the audio's length; a repetition loop (Hindi prompts tipped Whisper into "ॐ ॐ ॐ…", and its temperature fallback then took 3.9–5.8 s), an echo of the prompt, a failed decode, or likely no speech ("Thank you." for noise, with a prompt) is decoded again without it. Barge-in checks stay unprompted. The transcript carries Whisper's `avg_logprob`, `no_speech_prob` and `compression_ratio` for the say-again check (`speech_text.transcript_garbled`): clear questions decode at −0.05 to −0.5, Hindi −0.2 to −0.6, garbled or wrong-language transcripts −0.7 to −1.0.

| 64 clips with names (say Samantha/Daniel/Rishi/Lekha, Kokoro), mlx-whisper small | before | after |
|---|---|---|
| names right, all | 25/82 (30%) | **64/82 (78%)** |
| English, say / Kokoro | 44% / 62% | 92% / 100% |
| Hindi (Devanagari), say / Kokoro | 0% / 0% | 50% / 60% |
| Hinglish (romanized Hindi read by Indian voices) | 0% | 40% |
| WER (EN, name sentences), say / Kokoro | 0.142 / 0.067 | 0.040 / 0.000 |
| CER (HI, name sentences), say / Kokoro | 0.53 / 0.50 | 0.33 / 0.26 |
| 20 generic smoke clips: EN WER / HI CER | 0.059 / 0.147 | 0.059 / 0.192 (one clip: "रुको" heard as "रोगगार", pulled by "रोज़गार" in the scheme's title) |
| 7 controls (hums, "okay", silence, noise, breath) with a name | 0 | 0 |
| STT time (interleaved A/B, 60 clips) | — | +17 ms EN, +21 ms HI p50 |

faster-whisper small (CPU int8): names 35% → 79% (English 44% → 95%). The answer side has a safety net for what still gets through (`subjects.misheard_names`); these numbers are the recognizer's alone.

**TTS (Kokoro)** — offline from local files, EN `af_heart`, HI `hf_alpha`:

| | CPU | MPS |
|---|---|---|
| Full sentence, time to first audio | 0.63 s | 0.39 s (first run: 0.92 s — noisy) |
| Real-time factor | 0.16 | 0.10 |
| Short chunk (1–6 words), first audio | — | **120–250 ms** |
| Memory | — | 0.56 GB |

- **Stream at clause level**: split the first sentence at the first comma/clause so audio starts in ~0.15–0.25 s.
- CPU vs MPS ordering flipped between runs; device chosen by the end-to-end test with the LLM sharing the GPU.
- Kokoro deps need `transformers>=4.45` pinned (otherwise uv backtracks to a 2021 release needing Rust) and the spaCy `en_core_web_sm` wheel bundled (misaki downloads it at runtime otherwise).

### 9.4 Smoke tests 05, 08 results (Docling, VAD)

**Docling** (offline, `artifacts_path=data/models/docling`):

| | Result |
|---|---|
| 3-page PDF | headings ✓, 1 table with correct cells (EBITDA margin FY24 = 18.2%) on page 2 ✓, markdown keeps table ✓ |
| Speed | 0.29–0.32 s/page warm (~1 min per 200 pages); first call ~7 s incl. model load |
| Scanned (image-only) page | RapidOCR recovered text + table in 1.1 s |
| Hindi DOCX / PPTX | Devanagari + table preserved / slide text extracted |
| HybridChunker (BGE-M3 tokenizer, 512) | every chunk has heading path + page; table kept whole in one chunk |
| Peak RSS | 1.7 GB (ingestion only) |

- **Set the OCR engine explicitly** (`RapidOcrOptions()`); `OcrAutoOptions` logs "No OCR engine found" and silently produces empty text for scans.
- **Pin `rapidocr>=3.9.1,<3.10`**: 3.10 dropped `PP_OCRV6_LANGS`, which docling requires (though 3.10 is inside docling's declared range).
- Chunker tokenizer loads from the local BGE-M3 snapshot (default tokenizer would be downloaded).
- Not yet covered: OCR of **Hindi** scans (PP-OCR model language coverage unverified) — blueprint phase 2.

**Silero VAD** (streaming, 32 ms chunks, `min_silence 600 ms`):

| Gap noise | Speech-start detected | End-of-turn fired |
|---|---|---|
| digital silence | ≤ 41 ms | 0.64–0.73 s |
| quiet room (0.003) | ≤ 41 ms | 0.76–0.97 s |
| noisier (0.01–0.02) | ≤ 41 ms | 0.74–1.06 s |

- Compute 0.1 ms per 32 ms chunk; noise alone never triggers; a 300 ms mid-sentence pause doesn't split the turn.
- **End-of-turn costs 0.65–1.05 s**, not 0.6 s: Silero's lower "off" threshold (threshold − 0.15) keeps speech "on" through background noise after the last word. Design response: **speculative STT** — start transcribing after ~300 ms of silence, discard if the user resumes; final threshold 0.5–0.6 tuned live.

### 9.5 Smoke tests 10–11, 13 (browser VAD, end-to-end, memory)

**Browser VAD (10a, automated)** — real-time path (MicVAD, Silero v5, AudioWorklet) fed by a virtual microphone: 3/3 turns, speech-start **61–68 ms**, end-of-turn 0.76–0.86 s, **zero non-local requests**. Note: vad-web's `NonRealTimeVAD` only supports the legacy model; use `MicVAD` (v5). Echo/barge-in with a real mic (10b) is a manual check: `scripts/smoke/web/mic.html`.

**End-to-end (11) + memory (13), first run — invalid due to memory pressure:**

- Before loading anything: 12.1 GB used, 5.5 GB already in swap (Safari ~2.7 GB, Claude app ~1 GB, Chrome, Docker VM…). The ~5 GB "OS + browser" budget in §8 was optimistic.
- Loading everything (Kokoro on both CPU and MPS for comparison) pushed **+4.6 GB to swap**; every stage slowed (STT 3.0 s vs 0.44 s alone, TTS 0.8–1.7 s vs 0.12–0.25 s). Median 6.6 s + VAD — not representative.
- Functionally the loop worked: correction ("No wait, I meant the EBITDA margin") → rewritten query → correct 18.2% answer; misspelled Hindi carbon question → correct Hindi answer; unanswerable Hindi profit question → "not in the documents".
- Bug found: "let's go back to the annual report" with no topic history routed as backchannel and the answer claimed "I don't have access to documents" — router needs session topic state (as designed) and the answer prompt must never deny document access.
- Fixes: Kokoro on **CPU only** (config default now `cpu`, frees GPU memory); first spoken chunk = first clause **or 8 words**; rerun with browsers closed.

**End-to-end, valid run (browsers closed, all models loaded, no swap growth):**

| Stage (median of 5 turns) | ms |
|---|---|
| STT (mlx-whisper small) | 469 |
| Router (qwen3:4b-instruct, JSON) ∥ speculative retrieval (65–207) | 1490 |
| Rerank (English query) | 206 |
| Answer → first speakable chunk | 1479 |
| TTS first audio (Kokoro CPU, ≤8 words) | 627 |
| **Total after end-of-turn** | **4321** |
| **User stops → first agent audio** (+ VAD 0.65–1.05 s) | **≈ 5.0–5.4 s** |

Memory: 14.2 GB used / 2.9 GB available with every conversation model resident (MPS 3.6 GB, Ollama 3.2 GB, process RSS 2.1 GB); no swap growth.

Latency plan for the build (estimates from these measurements):

| Change | Saves |
|---|---|
| **Speculative answer**: start generation on the speculative retrieval in parallel with the router; keep if the router agrees (most document turns), else discard (needs `OLLAMA_NUM_PARALLEL≥2`) | ~1.3 s |
| First spoken chunk 3–5 words (fewer tokens to wait for, faster TTS) | ~0.5 s |
| Speculative STT during the end-of-turn silence (§9.4) | ~0.3 s |
| Keyword fast path for stop / backchannel (no LLM) | those turns ≈ instant |
| **Rerank fewer candidates** (8–10 instead of 20) and/or shorter chunks: measured in phase 1 at real chunk sizes, reranking 20 candidates × ~425 tokens takes **≈ 2.1 s** on the M4 GPU (the 202 ms in §9.2 was for one-sentence passages); 6 candidates ≈ 210 ms | up to ~1.8 s |
| Instant pre-synthesized acknowledgement ("Sure,", "Let me check") | perceived wait ≈ 1 s |

Expected after these: **~2.5–3 s** to the first content audio on this Mac. The original 1.5–2.5 s target is likely out of reach with router + answer model on an M4 base; perceived latency is handled with acknowledgements.

- Still-open bug: "let's go back to the annual report" → router `backchannel`, answer claims no document access even with the prompt guard (4b ignores it). Fix in build: rule + session-state for resume phrases; never call the answer model without either sources or an explicit general-knowledge route.
- **Barge-in semantics (from the user's mic test):** ducking alone isn't enough — the agent kept talking and finished its thought. Required behavior: duck on speech onset → **stop and discard the rest of the answer** once speech is confirmed (≥ `minSpeechMs`, then transcript check) → restore volume on misfire. `mic.html` now implements duck → stop → restore.

**Voice loop as built (phases 4–6, measured 2026-10-09).** Opt-in `tests/integration/test_voice_e2e.py` (real models, real-time audio) and a browser run of the voice-first chat page against the real backend (synthetic mic through the real worklet, browser VAD and socket):

| ms after the user stops speaking | Document answer | Correction after barge-in | Abstained (no LLM) |
|---|---|---|---|
| End of turn detected (`end_of_turn_ms` 600) | ~610–660 | ~660 | ~705 |
| `user_message` (speculative STT reused: 112–314 ms after end of turn; up to ~1.3 s on a first turn) | 790–1,980 | ~785–795 | ~950–975 |
| First `delta` (retrieval 260–590, of which rerank 190–340; LLM first token 1.1–2.1 s) | 1,900–4,100 | ~2,160–2,590 | ~1,230–1,250 |
| **First agent audio** (5-word first chunk, Kokoro CPU ~0.5 s) | **2,700–4,900** (browser median 3,200) | **~3,240–3,610** | **~1,620** |

- Before these changes the same test measured 6.8 s; §9.5's estimate was ~5.0–5.4 s. Applied: background preload (including Ollama with the answers' `num_ctx`, which saved a reload worth ~0.8 s on the first answer), speculative STT, 5-word first chunk, 8 reranked candidates.
- Browser side: first audio frame ≈ server + 0.12 s; duck 60–77 ms after speech onset; barge-in `stop` 0.61–0.67 s after speech start; backchannel `resume` ~0.7–0.78 s; local Stop/Esc 1–2 ms.
- What remains is mostly Ollama prompt prefill (~370 tokens/s, ≈ 2.7 ms per prompt token of sources). Next steps (phase 9): fewer or shorter passages for voice answers, a speculative answer started before routing, an instant acknowledgement, Kokoro on MPS under load.
- Measured with other apps (and, for the browser run, another agent's Ollama calls) running: slightly pessimistic.

**Answer latency, quality round (2026-10-09).** The voice answer's time to its first token was 3.3–4.7 s in the combined run. Profiled with `tests/integration/test_answer_latency.py` (Ollama only; the real turn pipeline with cached retrieval over eval-corpus passages: three voice chats of five turns, run-tagged passages so no earlier run's prompts are in the cache). Prompt reading runs at ~350 tokens/s on this machine, and an answer's prompt was ~1,300 tokens, nearly all of it read anew each turn: the evidence (five passages, up to 3,000 tokens), the history (a 6-message window sliding by one message every turn) and, on a mode or language change, the system prompt. What changed:

| Change | Effect |
|---|---|
| Short (voice) answers get the best 3 passages, the best one whole and the others within ~600 tokens (`ANSWER_LENGTHS["short"]`: `max_sources`, `context_tokens`); full answers are unchanged | evidence ~1,000 → ~500 tokens |
| Markdown tables in the prompt without padding or long rules (`compact_tables`) | tables ~12% fewer tokens, lossless |
| History window: at least 4 messages, its start moving 4 messages at a time, so it only grows at its end between moves | the history stays in Ollama's prompt cache |
| After a turn (while its answer is spoken; given up when the next turn starts), the model reads the next answer's prompt up to its evidence: system prompt, memory, history | the next answer reads only its evidence and question |
| One answer system prompt for English and Hindi: the question asks for the language (`ANSWER_LANGUAGE_RULE`); B5's script check catches a wrong-script answer | a language switch no longer re-reads the system prompt and history |
| Startup warm-up of the router and answer system prompts (B6) | first turns after startup |

| LLM first token (model's own time, 14 answers) | p50 | p95 | max | grounding (expected figure in the answer) |
|---|---|---|---|---|
| main before the round | 3,281 ms | 4,488 ms | 4,624 ms | 10/13 |
| this round | **1,355 ms** | 1,937 ms | 2,218 ms | 13/13 |

Wall clock (same runs, Ollama otherwise idle): 3,313 → 1,452 ms p50. Per kind after: English document question 1.35 s, follow-up 1.2 s, routed fact 1.36 s, general 0.8 s, Hindi 1.85 s (Devanagari costs ~0.9 tokens per character), first turn of a chat ~2 s (no history yet, its evidence all new). No model reload in any run. Ollama 0.40.1 keeps at least five prompt prefixes cached at once (measured), so the router's call, titles and summaries don't evict the answer's prefix; another client's traffic on a shared Ollama can.

**Voice latency, STT and memory round (2026-10-10).** The final real-model run measured a median of **5.4 s** from the end of speech to the first audio, with the Mac at 14.7 of 15.4 GB of swap (§8: Ollama's prompt cache): router 1.05–1.7 s, retrieval 1.2–2.1 s (rerank 0.55–0.97 s), LLM first token 2.0 s median, first delta to first chunk 1.0–1.2 s, and multi-second silences mid-answer. Re-measured without the swap (Ollama's prompt cache capped, a private server so no other client queues in front), on the tree with the shorter voice answers and the answer guard (#44): a harness like `test_voice_e2e.py` over the five eval documents, eight spoken questions per run (six English, two Hindi; `say` and Kokoro clips streamed in real time, nothing played), timing every model call:

| ms after the end of speech (median, p90) | main (#44), 16 turns | this round, 24 turns |
|---|---|---|
| `user_message` (VAD 600 ms + speculative STT) | 901 (1,049) | 937 (1,263) |
| router call (cancelled when the speculative retrieval is confident, §3.4) | 1,312 (2,334) | 1,234 (1,961) |
| retrieval (rerank) | 1,455 (2,380) · rerank 638 | 1,240 (2,087) · rerank 640 |
| LLM first token, the answer's own (prompt reading ~390 tokens/s) | 1,789 (3,758) | 1,712 (2,041) |
| first `delta` | 4,071 (7,291) | 4,000 (7,032) |
| first delta → first audio (TTS of the 5-word chunk: 660 → 520 ms) | 946 (1,418) | 774 (1,050) |
| **first audio** | **5,094 (8,996)**; English 4,748, Hindi 8,404 | **4,860 (7,549)**; English 4,759, Hindi 7,350 |
| silences mid-answer | 1 in 16 answers (907 ms) | 0 in 24 |

- The swap was most of the final run's extra latency and all of its mid-answer silences: with it gone, the router and retrieval are ~1.2 s and the first token ~1.7 s. This round adds: long sentences synthesized in clauses (Kokoro on the CPU takes ~2.8 s for 20 words, ~0.5 s for 5), the MLX cache cap, and the vocabulary prompt (+20 ms of STT).
- What remains is prompt reading: an answer reads ~650 new tokens (evidence and question) at ~390 tokens/s on this GPU, ~1.7 s; f16 instead of q8_0 KV reads no faster (388 vs 385 tokens/s, measured), and neither does `num_ctx` 4096. The router and retrieval run in parallel (~1.2 s) before it, so a document answer's first token comes ~2.9 s after the user message.
- **Hindi answers waited for the answer guard** (#44): the first delta came 2.6 s after the model's first token (12 s once, on main), against ~0.2 s in English: the guard held every Hindi sentence whole, and a Hindi first sentence is often one 100-token clause chain (real model: 4.9 s for 128 tokens). That was the Hindi median's ~2.5 s extra. Fixed (§3.4 "Answer checks", `hindi_hold`): Hindi goes out 3 words behind the model like English, and is held longer only while it is about what the documents contain and its clause hasn't reached its final auxiliary. Time from the model's first token to the first released text, a fake model at 30 tokens/s (Devanagari ~0.9 tokens a character, Latin ~3.5 characters a token; `ChatTurnService` in real time, six answers a language, coverage check on): English 0.21 s p50 (max 0.28); Hindi **3.52 s → 0.57 s** p50 (max 4.20 → 0.80), and 5 words out (what the voice's 5-word first chunk needs) 3.52 → 1.36 s p50 (English 0.44 s). Guard alone on a virtual clock: Hindi 2.35 → 0.38 s p50 (5 words 0.92 s). Real `qwen3:4b-instruct` (cached retrieval, 6 Hindi questions; the raw token streams replayed through the old and the new guard): short answers (31–36 tokens, 7-word sentences) 0.92–1.21 s → 0.46–0.49 s to the first word, and the same to 5 words (the sentence ends first: three words behind needs a 9-word sentence for a 5-word chunk, as in English); long first sentences (72, 128, 132 tokens) 2.52 / 4.93 / 4.82 s → 0.52 / 0.78 / 0.50 s to the first word and 1.07 / 1.72 / 0.99 s to 5 words. What is left of Hindi's extra is its token cost, ~0.17 s a word against ~0.05 in English.
- Kokoro on MPS is 2–3× faster alone (5 words 234 ms, 20 words 954 ms) but slower in the loop, sharing the GPU with Ollama and the reranker (first delta → audio 0.91–1.23 s vs 0.85–0.90 s on the CPU, two rounds): it stays on the CPU. Torch threads (4, 6, 10) make no difference.

Not done: an instant spoken acknowledgement ("Sure,") for slow routed turns. The protocol lets only the web search filler precede `sources` (§3.10), the acknowledgement would be spoken before the route is known (wrong for "stop", a backchannel or an abstention), and with the first token at ~1.4 s plus the router's ~1 s and the first chunk's TTS, the remaining wait is short enough to measure in the full voice run first.

### 9.6 Smoke test 12 results (network off)

Run 2026-10-08 21:51 with Wi-Fi off (`network: unreachable ✓`), `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`:

| Test | Result offline |
|---|---|
| 01 Ollama (4b-instruct) | ✅ 18/18 |
| 02 Qdrant | ✅ 15/15 |
| 03 BGE-M3 / 04 reranker | ✅ |
| 05 Docling | ❌ crashed at OCR init → fixed (below); **re-run with network off: ✅ 17/17** |
| 06 Whisper (mlx small/turbo, fw small) | ✅ 20/20 language ID each; mlx-small 0.40 s |
| 08 VAD / 09 Kokoro | ✅ / ✅ 10/10 |
| 11 + 13 end-to-end | ✅ no swap growth; 4.5 s after end-of-turn (≈ 5.2–5.6 s with VAD), same answers as online |

**OpenCV trap (applies to the backend):** docling depends on `opencv-python-headless`, rapidocr on `opencv-python`; both install into the same `cv2/` directory, and removing either one deletes it while the other still looks installed. A pre-flight `uv sync` removed one and broke the environment ("RapidOCR is not installed" is docling's message for *any* import failure; the real error was `No module named 'cv2'`). Fix: depend on `opencv-python-headless` explicitly and override `opencv-python` away (`[tool.uv] override-dependencies = ["opencv-python; sys_platform == 'never'"]`). **The backend `pyproject.toml` must carry the same override.**

Also: onnxruntime threads abort during Python shutdown on macOS (`libc++abi … recursive_mutex`) after work completes; scripts exit with `os._exit()`; the backend should shut down OCR sessions explicitly.

Kokoro device: offline run measured MPS 0.31 s vs CPU 0.50 s full-sentence first audio (2 of 3 runs favour MPS). Config stays `cpu` to keep GPU memory for the LLM; revisited in the voice loop (§9.5, 2026-10-10): MPS is slower there, sharing the GPU with Ollama and the reranker.

### Setup notes

- Ollama runs as a brew service (`brew services start ollama`), bound to `127.0.0.1:11434`.
- **Cap Ollama's prompt cache** (§8): llama-server keeps every distinct prompt's state in RAM up to 8 GiB by default, which pushed the 16 GB Mac into heavy swap. Set `LLAMA_ARG_CACHE_RAM=1024` (MiB) in the service's environment; Ollama passes its environment on to llama-server. Quick, until the next reboot: `launchctl setenv LLAMA_ARG_CACHE_RAM 1024 && brew services restart ollama`. Lasting: add the key to `EnvironmentVariables` in `~/Library/LaunchAgents/sh.brew.ollama.plist` (next to `OLLAMA_FLASH_ATTENTION` and `OLLAMA_KV_CACHE_TYPE`) and reload the service; `brew services` may rewrite that file on an upgrade. Check: the runner's footprint (`footprint $(pgrep -f 'llama-server.*--port')`) stays near 2–2.5 GB, and Ollama's log shows `cache state: … (limits: 1024.000 MiB …)`.
- Docker Desktop memory is capped at 1.5 GB (Settings → Resources); only Qdrant runs in Docker.
- Before going offline: install spaCy `en_core_web_sm` into the backend env (Kokoro's `misaki` otherwise fetches it at first run).

---

## 10. Build phases (after Phase −1 is green)

**Progress**

| Phase | Status | Delivered in |
|---|---|---|
| −1 Downloads + smoke tests | ✅ done | §9; initial commit |
| 0 Skeleton | ✅ done | PRs #1–#3 (hygiene, backend skeleton, frontend shell), #5 (CI), #8 (Docker images + `full` profile, `docker.config.json`, `strict_offline_local_hosts`) |
| 1 Projects + text document chat | ✅ done | #10 persistence (SQLite + Alembic, projects/chats/messages/pins API), #11 ingestion + hybrid retrieval, #12 sidebar, project and chat pages, transcript view, #16 upload + background ingestion + streamed cited chat (transport-agnostic `ChatTurnService`), #14 upload UI + streaming composer + citation popovers. Verified end to end on the real models: FY24 EBITDA 18.2% cited p.2 in EN and HI, out-of-document question abstains, transcript survives reload |
| 4–6 Voice loop | ✅ done | #17 + #19 protocol (§3.10), #21 voice session backend (VAD, speculative STT, Kokoro streaming, barge-in, preload), #20 voice-first chat page (presence field, captions, browser VAD barge-in, transcript panel, "Start a conversation"), #18 citation tables. Independently reviewed (both sides) and verified end to end on the real models in the browser: spoken question → cited spoken answer, barge-in with `heard_text`, backchannels ignored, stop, Hindi, reload, second tab, backend restart; first audio median ~3.2 s in the browser, 2.7–4.9 s across runs (§9.5) |
| 2 Retrieval quality | ✅ done | eval set #22; tuning: R@1 71.8 → 85.6 %, R@5 87.8 → 96.8 %, MRR 0.797 → 0.920, page citations 74.4 → 88.5 %, end-to-end correct answers 79.5 → 92.9 %, unanswerable answered 4/38 → 3/38, retrieve p50 707 → 445 ms (document labels, fiscal-year veto, Hindi-aware reranking, small rerank batches, re-index on a chunking-version change) |
| 3 + 7 Router, state, revisit features | ✅ done | #23 automatic titles, user summaries, transcript export; #24 router + conversation state + memory summary + drift and EN/HI/Hinglish switching: 43/43 intents on the labelled set with `qwen3:4b-instruct` (prompt tuned on that set), router p50 ≈ 0.65 s / p95 ≈ 1 s on routed turns, 0 on fast-path turns; #25 summary/export/title UI and routed-answer labels; #29 title job lane |
| 10 Live visual canvas | ✅ done | **part of the MVP** (user decision 2026-10-09); §12.1, contract v1. #34 backend (typed datasets, chartability, `VisualSpec` + validator + calculator, planner, overview, canvas API), #32 canvas panel, then the conversation: visuals planned after the answer and streamed on both transports, `route.visual`, spoken and typed edits (EN / HI / Hinglish), the canvas as context, the spoken tail. Grounding 100% (every visual of every run). Then the instant draft (§12.1 "Instant draft, then refine"): a chart built by code when the retrieval returns, on screen with the answer's first audio (2–4 ms after it, 9 of 9 visuals in the real voice run), the planner only to refine an unsure draft after the answer (skipped for 61–80% of the labelled questions); company-aware candidates; a wider gate. Draft + planner: tuning 30/30, hold-out v1 24/27, hold-out v2 27/32 (the planner alone 24/32) |

Design-only PRs so far: #4 and #6 (voice presence UI, §3.8), #7 (projects, chats, transcripts, §3.9).

**Execution order (voice-first).** After phase 1 lands, go straight to the voice loop with the voice-first chat page: **4 → 5 → 6**, then **3 + 7** (router, drift, language switching; summaries/titles/export as revisit features), then **2** tuning on the eval set, **8** (web search), **10** (live visual canvas, in the MVP since 2026-10-09: its backend foundation and canvas panel are built in parallel with 2 and 8, the router/voice integration after the quality round), **9** (evals, latency, polish). Phase numbers keep their meaning; only the order changes.

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Skeleton: config loader, provider registry, `/health`, `/api/config/public`, frontend shell with runtime config | `/health` reports every provider; both config files validate; `cloud.config.json` wiring test passes (placeholders raise NotImplemented) |
| 1 | Text-only document chat inside projects: projects, documents, chats, messages persisted; sidebar with pinning; transcript view (§3.9); upload → Docling → chunks → BGE-M3 → Qdrant → Qwen with citations | fact questions cited; out-of-document questions abstain; two docs distinguished; documents never leak across projects; chats and pins survive a restart |
| 2 | Hybrid retrieval + reranker + confidence gate | identifiers, paraphrases, numbers, page citations correct on eval set |
| 3 | Router + conversation state; chat memory summary; automatic chat titles; user-facing chat summaries + transcript export (§3.9) | unrelated questions skip retrieval; follow-ups and "back to the report" work; summary cites pages and flags unanswered questions; export opens as Markdown/JSON |
| 4 | Voice in: browser mic → WebSocket → VAD → Whisper → transcript events | EN/HI transcription, silence handling, end-of-turn detection |
| 5 | Voice out: streamed answer → sentence TTS → browser playback; voice presence field (agent motion) + live captions (§3.8); **chat page opens in voice mode**, transcript as a panel, "Type instead" secondary (§1 voice-first) | first audio after first sentence; text and audio in sync; field follows agent voice; a new chat is usable end to end without typing |
| 6 | Barge-in (duck-then-decide), corrections, stop; user waves + barge-in visuals (§3.8) | interrupt mid-answer; "no, I meant…" handled; backchannels ignored; field ducks with the audio |
| 7 | Topic drift + EN/HI switching + code-mixing | drift script passes: document → general → Hindi → document |
| 8 | Live-data tools: web search with filler + streamed partial answers (§3.7) | mixed doc+live question answered with separate [S]/[W] citations; first words < 1 s after filler; barge-in cancels search; timeout falls back to document-only answer |
| 9 | Evals + observability + UI polish | retrieval Recall@5/MRR, groundedness, latency dashboard; demo script runs end to end |
| 10 | **Live visual canvas** (§12.1, MVP): typed datasets, chartability, `VisualSpec` + validator + calculator, overview dashboard on upload, canvas panel, router/voice integration | every plotted number grounded to a cell (100%), hallucinated values 0; "show me revenue over five years" puts a cited chart on screen ≤ 1 s after the spoken answer starts; voice edits ("make it a bar chart", "remove that") work; overview dashboard after upload. **As built:** grounding ✅ (every visual of every run, tests and real model); edits ✅ (EN / HI / Hinglish, fast path, tested and in the real run); overview ✅; a cited chart for "show me…" ✅ on screen 2–4 ms after the answer's first audio (real voice run, 9 of 9 visuals) (the draft, code; the planner refines an unsure one after the answer: §12.1 "Instant draft, then refine"); to check in the final real-model voice run in the browser: how often the tail is spoken with real playback |

Each phase is green (tests + acceptance criteria) before the next starts.

---

## 11. Open items

- **Reranker cost at real chunk sizes** (≈ 2.1 s for 20 × ~425-token candidates): choose the rerank candidate count with the chat pipeline (being measured now) and revisit chunk size in phase 2.

- Web search provider decided: SearXNG locally, search API in production (§3.7). Add the SearXNG container + per-engine fan-out in phase 8.

- STT decided: mlx-whisper `small` (§9.3); turbo is a config switch if Hindi accuracy matters more than 1.4 s.
- **LLM decided (2026-10-08): `qwen3:4b-instruct` for both router and answers** (fits memory, 2–5× faster, clean EN; see §9.1). `qwen3:8b` and `qwen3:4b` (thinking) stay installed. Revisit **language-based selection** (8b for Hindi) only if the Hindi quality eval shows 4b-instruct is clearly worse — and only if smoke test 13 shows both can stay loaded (otherwise EN↔HI switches cost a 1–4 s model reload).
- `qwen3:4b` (thinking) stays installed for non-voice reasoning tasks; not used by this project.
- Kokoro Hindi voice quality: intelligible to Whisper (CER 0.00–0.11 on turbo); still worth a human listen — samples in `data/smoke/audio/kokoro_hi_*.wav`.
- Tune end-of-turn silence, barge-in thresholds, chunk sizes and abstention thresholds with measurements.

---

## 12. Ideas backlog

Ideas raised for later stages; not scheduled until the core voice loop (phases 0–7) is solid.

- Live-data tools beyond web search (e.g. currency/stock lookups as dedicated tools) — same tool pattern as §3.7.
- **Live visual canvas** — see §12.1.
- _(more ideas from the user to be added here)_

### 12.1 Live visual canvas (generative visuals for data-heavy documents)

**Idea (user, 2026-10-08):** for financial and other data-heavy documents, the agent builds charts, dashboards and other visuals on screen *while the conversation happens*, for things words can't express well (trends, breakdowns, comparisons).

**How it fits the voice design:** the voice stays short and says the *insight*; the screen holds the *evidence*.

> User: "How has revenue moved over the last five years?"
> Agent (voice): "Steady growth, with a dip in FY21 — I've put the trend on screen."
> Screen: line chart FY20–FY24, FY21 dip highlighted, every point cited to its page.

#### Principles

1. **The LLM picks, code builds.** The model never writes chart code or HTML. It outputs a small, validated `VisualSpec` (JSON, Pydantic) choosing from enumerated options — chart type, which dataset, which columns, highlight. Application code fills in the numbers and the frontend renders with a fixed component library. This keeps a 4B model reliable and makes rendering safe.
2. **Every number is grounded.** Values come from structured tables extracted at ingestion, never from model text. Each data point carries provenance (document, page, table, cell); hover shows it, click opens the source page. A validator rejects any spec whose values aren't found in the sources.
3. **Arithmetic is done by code.** Growth %, CAGR, margins, differences and shares are computed by a deterministic calculator tool and labelled "calculated"; the model only asks for them.
4. **Visuals are optional and never block speech.** The spoken answer streams first; the visual arrives in parallel (skeleton placeholder → chart). If it fails, the conversation is unaffected.

#### Enhancements beyond the original idea

- **A canvas, not one-off charts.** Visuals accumulate on a board for the session. Voice can edit it: "make that a bar chart", "put FY23 next to it", "remove the pie", "pin this". Edits are state operations (add / update / remove / pin panel) validated by the server, like the rest of the conversation state.
- **Instant overview dashboard on upload.** During ingestion, chartable tables are detected and a default dashboard is pre-built (KPI tiles + key trends). When the user asks the first question, the canvas already has context.
- **Visuals become conversation context.** The current canvas is part of session state, so "why did it dip there?" or "what's the second bar?" resolve against what's on screen.
- **Prepared while the user is still speaking.** Partial transcripts (re-transcribing the growing utterance every ~1 s) drive speculative retrieval of the relevant datasets, so the visual is ready by the time the answer starts. Discarded if the final question differs.
- **Beyond finance:** contract timelines (dates, notice periods, obligations), party/relationship diagrams, clause-vs-clause comparison tables, document-vs-document comparisons ("do the presentation and the annual report agree?"), and live data from web search (§3.7) plotted against document figures — always visually distinguished ([S] vs [W]).
- **Localized and accessible:** Hindi labels when the conversation is in Hindi, Indian number formatting (lakh/crore) where the source uses it, a text summary for every visual, and a spoken one-line description.

#### Visual types (initial set)

KPI tiles · line (trend) · bar / grouped bar (comparison) · stacked bar (composition over time) · waterfall (bridges, e.g. revenue → profit) · donut (share, sparingly) · table (exact figures) · comparison card (A vs B) · timeline (dates/obligations).

#### What we will build

| # | Workstream | Details |
|---|---|---|
| 1 | **Structured datasets at ingestion** | Persist every Docling table as a typed dataset (columns typed as year / period / currency / percent / count, units and scale detected, e.g. "₹ crore"), with cell-level provenance. Stored in SQLite (metadata + values) and indexed in Qdrant as a table chunk for retrieval. **Hook to add in phase 1 already** — cheap now, avoids re-ingesting later. |
| 2 | **Chartability detection** | Classify each dataset: time series, categorical comparison, composition, single KPIs, not chartable. Drives the overview dashboard and limits the options offered to the LLM. |
| 3 | **`VisualSpec` schema + validator** | Pydantic schema with enums of chart types and *the actual column names of the retrieved datasets*; numeric grounding check; size limits. |
| 4 | **Calculator tool** | Deterministic growth, CAGR, ratio, difference, share; results labelled as calculated with their inputs cited. |
| 5 | **Router + visual planner** | Router gains `visual: none | suggest | requested`; a constrained-JSON planner call picks dataset/columns/chart type. Heuristics trigger it for trends, comparisons, ≥3 numbers, "show me". |
| 6 | **Canvas state + events** | Session canvas (panels, layout, pinned) in the session store; WebSocket events `VISUAL_PREPARING`, `VISUAL_READY`, `CANVAS_UPDATED`, `VISUAL_FAILED`. |
| 7 | **Frontend renderer** | Canvas panel beside the transcript; fixed component library rendering `VisualSpec` (chart library bundled locally, no CDN — ECharts or Vega-Lite, decided at build time); tooltips with citations; click-through to the source page; skeleton while preparing; light/dark. |
| 8 | **Voice ↔ visual coordination** | Spoken answer references the visual ("on screen now"); voice canvas edits; visual state fed back to the router. |
| 9 | **Overview dashboard** | Auto-built on upload from workstreams 1–2. |
| 10 | **Speculative preparation** | Partial-transcript retrieval of datasets while the user speaks. |
| 11 | **Evaluation** | Numeric grounding = 100% required; chart-type appropriateness (human-rated set); render success rate; time to visual after speech starts; hallucinated-value rate = 0. |

#### Constraints and risks

- **Table extraction quality is the ceiling.** Visuals are only as good as Docling's tables; units/scale detection and multi-row headers need tests on real annual reports.
- **4B model limits:** keep choices enumerated and small; complex multi-panel dashboards may need the 8B model run asynchronously (not in the voice path).
- **No extra models needed;** memory impact is the frontend chart library plus SQLite datasets.
- **Latency:** the visual must never delay first audio; target visual on screen ≤ 1 s after the spoken answer starts.

#### Placement

**Part of the MVP (user decision 2026-10-09)** as phase 10 (§10). Workstream 1 landed in phase 1 (every Docling table is stored as a cell-level dataset in `document_tables`). Workstreams 2–4, 6 (storage and API), 7 and 9 were built first and in parallel; 5 and 8 (router and voice integration) after the quality round ("In the conversation" below); 10 (speculative preparation) was looked at and not built (nothing measurable to save).

#### Contract v1 (backend ↔ frontend)

Application code turns a validated `VisualSpec` (what the model picked) into a **`Visual`** (what the frontend renders). The frontend never sees model output and never computes numbers.

```text
Visual {
  id: "vis_…", chat_id | null (null = project overview), project_id, created_at, updated_at,
  kind: "kpi" | "line" | "bar" | "grouped_bar" | "stacked_bar" | "waterfall" | "donut" | "table" | "comparison" | "timeline",
  title, subtitle | null, language: "en" | "hi",
  summary: str                     # text alternative (screen readers, transcript, spoken one-liner)
  unit: Unit | null                # default unit of the values
  x: {key, label, type: "period" | "category" | "date"} | null        # null for kpi / comparison
  series: [{key, label, unit: Unit | null, calculated: bool}]
  rows: [{x: str, values: {series_key: number | null}, cells: {series_key: CellRef | null}}]
  tiles: [{label, value: number, unit: Unit | null, cell: CellRef | null,
           delta: {value: number, kind: "abs" | "pct" | "pp", calculation: Calculation} | null}]    # kpi / comparison only
  events: [{date: "YYYY-MM-DD" | label, label, detail | null, cell: CellRef | null}]                 # timeline only
  highlight: {x: [str], series: [str], note | null} | null
  calculations: [Calculation]      # every derived number shown anywhere in this visual
  sources: [Citation]              # the document citations (source_id S1…) that CellRefs point to
  pinned: bool, position: int      # on the canvas
}
Unit        { kind: "currency" | "percent" | "count" | "ratio" | "duration" | "none", currency: "INR" | "USD" | … | null,
              scale: "crore" | "lakh" | "million" | "billion" | "thousand" | null, label: "₹ crore" }
CellRef     { source_id: "S1", document_id, table_id, page, row, col, text }      # the exact cell the value came from
Calculation { label, op: "growth" | "cagr" | "diff" | "ratio" | "share" | "sum", value: number, unit: Unit | null,
              inputs: [CellRef], formula_text }                                     # labelled "calculated" in the UI
```

- Values are numbers in the unit's scale (4210 with `scale: "crore"` = ₹4,210 crore); the frontend formats them (Indian grouping for INR/lakh/crore sources, Hindi labels when `language: "hi"`).
- **Canvas** (per chat): `GET /api/chats/{id}/canvas` → `{panels: [Visual]}` in position order; `POST /api/chats/{id}/canvas/ops` with `{op: "remove" | "pin" | "unpin" | "move", visual_id, position?}` → the new canvas. Visuals are added by the conversation (and, for testing and the debug panel, `POST /api/chats/{id}/visuals` with a `VisualSpec`).
- **Overview** (per project, built after ingestion): `GET /api/projects/{id}/overview` → `{panels: [Visual], status: "ready" | "building" | "none"}`. Each panel is built from one document's tables, the panels are grouped by document (the one with the most chartable tables first, one panel from each in turn: KPI tiles, trend, composition) and titled in the project's main language (that of most of its documents, else `client.default_language`) with the document's name when there are several ("Suryodaya yojana soochna: seats by course"); the labels of the data stay as printed (`overview.py`).
- **Events** on both transports (SSE and the voice WebSocket, with `turn_id` on the WebSocket): `visual {phase: "preparing" | "ready" | "failed", visual_id, visual?: Visual, detail?}` and `canvas {panels: [Visual]}` (a snapshot after any change). A visual never delays the spoken answer; `preparing` lets the UI show a skeleton (the page shows it only when `ready` hasn't arrived within 150 ms, so the instant draft, whose `preparing` and `ready` come in the same millisecond, never flashes one). Additive (integration): `detail: "cancelled"` on a `failed` visual that was given up (the client drops its skeleton without a note); a `preparing` with the id of a panel already on the canvas is that panel being rebuilt (a model-path edit: "Updating…"), and its `ready` replaces it in place; a second `ready` with a turn's visual id is the planner's chart replacing the draft in place, and a `failed` after a `ready` (with a `canvas` without it) is a draft withdrawn.
- **On the turn's message** (additive `route` fields): `visual` ("none" | "suggest" | "requested", §3.4), `visual_status` ("ready", or for a requested one "failed" / "cancelled" / "none") with `visual_id` (ready; null again if a shown draft was withdrawn) or `visual_detail`, and `visual_plan` (how it was made: the draft, the planner's part, §12.1 "Instant draft, then refine"), written when the visual settles (usually after the message was saved and sent: the transcript reads it, a live turn patches it in from `visual ready`); `screen_visual_id` (the visual a question was about); `canvas_edit {op, kind?, periods?, outcome, source: "rules" | "model", visual_id?, detail?}` on an edit's reply.
- Additive changes are allowed; anything else changes this section first.

#### In the conversation (as built, phase 10 integration)

**When the planner runs, and why.** Ollama serves one request at a time on this machine, and the planner is one more call to the same model. Measured on the real model (`tests/integration/test_canvas_scheduling.py`: the voice answer prompt over a corpus table, the real planner over the corpus's 50-odd datasets, six questions, four schedules each, rotated):

| schedule of the planner's request | answer's first token p50 (max) | answer done p50 | plan ready p50 (max), from the answer's request |
|---|---|---|---|
| none (the answer alone) | 0.80 s (1.07) | 4.40 s | — |
| sent just before the answer's | **4.12 s** (5.27) | 7.49 s | 7.49 s (11.07) |
| sent on the answer's first delta | 0.79 s (1.05) | 4.23 s | 7.43 s (11.29) |
| sent once the answer's text is complete | 0.80 s (1.07) | 4.26 s | 7.47 s (11.29) |

Starting the visual with the turn costs the first token ~3.3 s; starting it on the first delta or after the answer costs the answer nothing and lands the plan at the same moment (the request just waits in Ollama's queue). **The planner runs once the answer's text is complete**, and only to refine a draft that isn't sure (next paragraph) (`ChatTurnService._start_visual`, just before the answer is saved, after any live-data continuation): same timing, the planner gets the answer's text (its figures; "Spoken answer" in the planner prompt), an answer cut while it is written never starts one, and a B5 language retry or a §3.7 continuation never queues behind it. The planner alone on an idle model: **3.35 s p50, 4.14 s p95** (that run); over the labelled sets the model's own time is 3.7 s p50 (tuning set) and 4.7 s p50 / 7.3 s p95 (hold-out), all within `planner_timeout_ms` (8 s); ~1,000 prompt tokens (1.6–1.7 s reading the catalog: only its system prompt is cached, warmed at startup with the router's and the answers') and 60–85 output tokens (2–3 s writing). Ranking the candidates in code takes 0.7 ms.

**Instant draft, then refine.** So the planner can't put a chart on screen as the answer starts; code can. `canvas/draft.py` picks and builds a **draft** without the model, in a few milliseconds (p50 3.6–3.7 ms, max 7.5 ms over the labelled sets), as soon as the turn's retrieval returns (`ChatTurnService._start_draft`, just before `sources`): the best candidate (`rank_candidates`: the turn's retrieved tables and their documents first, only the tables of the company the question names, below), its default chartable shape (periods → a line, two periods → bars; parts of a whole → a donut, two periods asked for → grouped or stacked bars; categories → bars; headline metrics → KPI tiles; dates → a timeline; a kind named in the question wins: "bar chart", "pie", "waterfall", "table", "exact figures", "गोल चार्ट"), its series and periods narrowed to the question's words (English and Hindi labels, a few synonyms: sales → revenue, debt → borrowings, budget ↔ बजट), the same metric of two companies side by side when the question names both; **from one document's tables** otherwise: the company the question names, else the document of the turn's top retrieved source, the passage the answer is drawn from (`TurnContext.scope`), unless the question compares documents or companies (it names two of them, or says "both companies", "each company", "दोनों कंपनियों": `TurnContext.compare`; "compare FY23 and FY24" compares periods, not documents). "The company's quarterly revenue" in a chat of both companies' documents drew the deck's Q4 table next to the other company's FY23 table in the final real run: the draft took the best-matching table of either document, the planner, offered the tables of both, one of each; now the planner is offered only that document's tables, a choice that still spans two is cut down to its first series' document (`planner.one_document`), and one that differs from the draft and mixes documents never replaces it (`refinement_loses`); then the planner's own validator and builder (`first_valid`, `builds`): every number is a cell. It is **confident** when one table is clear (it leads the runner-up, the tables that continue it aside, by 2 points or more, or it is the only one with the rows the question names, or the only timeline), the kind drawn is the kind asked for, and every series was named by the question (or the table has only one, or the question names the table itself: "the balance sheet", "Zephyra's Q4 results"). A confident draft is the turn's visual: no model call. Otherwise the planner, once the answer's text is complete, is offered the draft's candidates; when it chooses another chart the panel is rebuilt **in place** (same id: `visual ready` again, then `canvas`), when it says no table fits the draft is **withdrawn** (`visual failed`, `canvas` without it), when it times out or fails the draft stays. The planner's chart replaces the draft only when it covers the question at least as well (`draft.refinement_loses`): from the named company's documents and the answer's own, from one document's tables, from a table that matches as well (the draft's score, within the 2 points of an "unsure" draft), no fewer points of the periods the question names (a fiscal year named over quarterly tables means all its quarters: "FY23 and FY24" is eight points, and a chart of Q4 FY23 and Q4 FY24 shows two of them, not "both years"), no fewer points of a time series than the draft at all unless the question asks for fewer ("only", "just", "the latest", "सिर्फ़", a quarter or half named: `Cues.narrow`; then each period named counts once), and no fewer of the series it names; otherwise the draft stays (`planner: "kept"`, the reason in `reasons`: a misheard company name had the planner replace a correct 8-quarter draft with a 2-point bar chart; in the last real-model run "Show me Valmora's quarterly revenue and EBITDA for FY23 and FY24" had its 8-quarter draft replaced ~6 s later by a grouped bar of Q4 FY23 and Q4 FY24 in 2 of 4 runs, because a question that named periods skipped the points rule and "Q4 FY23" counted as showing FY23). One panel per turn, always. The message's `route.visual_plan` records how: `{draft: confident | refine | none, planner: skipped | same | changed | kept | none | failed | not_run | cancelled | planned, draft_ms, planner_ms, reasons}`. Delivery: the voice session holds the draft until the answer's first audio and sends it right after (§3.10: never before); SSE sends it with the answer's first delta. **The draft is the answer's evidence too** (quality round, item 1: the chart showed all eight quarters while the voice answer, given the best three passages without the quarterly table, said the report "does not provide quarterly revenue and EBITDA figures for FY24", citing the other company's deck): the turn waits for the draft (code, ~20 ms; at most 0.3 s) before `sources`, adds its tables to the answer's sources with the `[S#]` ids the visual's cells already use (the retrieved chunk, or the table as stored), and keeps the evidence within the answer's budget by dropping the weakest other passages (`ChatTurnService._draft_evidence`). The answer prompt then says the chart is on screen, drawn from those ids, with its one-line description (`prompts.on_screen_note`, instead of `VISUAL_NOTE`): agree with it, say what it shows with a key figure or two rather than every number, never say the documents don't have what it shows; a confident draft (no planner to redraw it) may be referred to ("The chart shows …"). The answer's coverage check (§3.4 "Answer checks") then holds the chart's rows as evidence: a denial of what it shows is asked again, then corrected.

Measured on the real models (`tests/integration/test_canvas_instant_visual.py`: the real voice session over the WebSocket with the real router, answer and planner on Ollama and the real Kokoro speech; scripted speech in and retrieval cached as in `test_answer_latency.py`; 10 EN / HI / Hinglish questions over the annual report, timed where the client receives each message):

| | before (#38: the planner after the answer) | now |
|---|---|---|
| visual on screen, from the answer's first audio | 3.6–5.1 s after the answer's text (5.5–9.7 s after its first delta) | **2–4 ms** (9 of 9 visuals: the draft, held until the first audio, then sent) |
| planner skipped (confident draft) | never | 9 of 9 (the turn's retrieved table ranks first) |
| refined visual, when the planner runs (a pass with every draft taken as unsure) | — | 2.5–7.3 s after the first audio, 2.7–4.1 s after the answer's text (planner call 2.6–5.0 s); it changed 3 of 4 drafts (line → bar, grouped bars → comparison: kinds the labels also accept), confirmed 1; a fifth answer abstained and its draft, withdrawn before the first audio, was never shown |
| answer's first delta, draft beside the answer / visual after it (same prompt, 6 runs each, alternating) | — | 613 / 614 ms p50 |
| spoken tail | when the visual landed while the answer still played | 9 of 9 visuals (5 of 5 in the refine pass) |

The one question of the ten without a visual (Hinglish, "… ki tulna dikhao") abstained on the cached retrieval: an abstained turn gets none. The draft is built in a thread (`asyncio.to_thread`), 17–25 ms with its database reads and write (one outlier at 270 ms), all before the answer's first token.

How often the planner is needed (`test_canvas_planner_eval.py`, no retrieval, so harder than a turn): skipped for a confident draft 24/30 (tuning set), 17/27 (hold-out v1) and 22 of hold-out v2's 36 (20 of its 32 chart questions); when it ran it changed the draft 3 of 6, 9 of 10, 10 of 14 times. Confident drafts were right 24/24, 16/17, 17/20.

**Only for answers from the documents.** A visual is prepared for `document_qa`, `mixed` and `correction` turns answered from sources (not abstained, not a mixed answer that fell back to general knowledge, not a question about the chart on screen), when `route.visual` says the words ask for one (requested, or suggested with a table among the sources: its draft starts when the retrieval returns), or when the answer's own figures (three or more, periods such as "FY24" not counted) suggest one and the answer drew on a table passage (started when the answer's text is complete: a confident draft is shown, an unsure one only through the planner). Requested visuals are announced (`preparing`, a skeleton, at once followed by the draft; `failed` shows a quiet note; without a draft the planner's heuristic fallback draws the best table's default chart when the model times out or fails); suggested ones appear only if they work (no skeleton, nothing on failure). An answer that turns out to say the documents don't cover it (B9) withdraws its draft (`failed {detail: "cancelled"}`, no note). Cells reuse the turn's own `[S#]` ids: the turn's sources are passed to the builder, and the turn's table chunks and their documents rank first among the candidates. General, conversation, acknowledgement, stop and abstained turns never get one (tested).

**How long a turn stays open for it.** SSE: the draft comes with the answer's first delta; the stream stays open after `agent_message` while the planner refines it (or plans one without a draft), at most `planner_timeout_ms` + 2 s (the heuristic fallback and building), then the draft stays (without one, `failed {detail: "timed out"}`); closing the stream cancels the planner (a draft that was sent stays, one that wasn't is withdrawn); the frontend lets the next question be asked as soon as `agent_message` arrives. Voice: `run(on_visual=…)` delivers the visual's events to the session as they come (§3.10 for when) and the turn ends with its answer as before. The memory summary and the next prompt's warm-up (§9.5) wait for the planner (they would queue behind it on the model); after a confident draft they go at once.

**A visual still being refined when the conversation moves on.** The next turn decides, per chat: an **edit** ("make it a bar chart", said while it is refined) waits for it; a turn that **needs the model** (any routed question, a document question from the fast path, "stop", a language request) **cancels** the planner (its request is closed, the model is free for the question: a follow-up asked while a visual was being planned got its first delta in 2.2 s against 2.6 s once it was done, real model), and the draft on screen stays the turn's visual (no `failed`; `visual_plan.planner: "cancelled"`); an acknowledgement, thanks or a greeting leaves it alone. The client's stop and the session closing cancel it too. A barge-in doesn't cancel it by itself (the interrupting question does, as above); one during the answer's generation never starts the planner (a draft already on screen stays); a cut before the answer's first audio withdraws the draft (nobody saw it: taken off the canvas, nothing sent).

**The spoken tail** ("It's on screen now." / "स्क्रीन पर दिखा दिया है।", pre-synthesized when a session's first visual starts). The answer never promises a visual (it may fail): the answer prompt says, when one was asked for, that the app draws charts itself, never to promise one or say it can't show charts (`VISUAL_NOTE`, in the user message, so the cached prefix is unchanged). Instead, *after* `visual ready`, if the answer is still being heard (its audio still being sent, or the client still has ≥ 0.3 s to play), not cut and the user isn't speaking, one more chunk follows the answer's last; otherwise nothing is said (the chart appearing says it, and the transcript shows "Chart added"). Chosen over "right after the answer, whenever it is ready" because speaking unprompted once the turn is over would talk into the user's next utterance and break the turn protocol. With the draft on screen from the answer's first audio, the tail follows the answer's last chunk after almost every visual (9 of 9 in the real run below); for a draft the planner is still checking it waits for the planner's verdict (it may withdraw it), so it never announces a visual that goes. Not part of the answer: never in `heard_text` or the message text; a barge-in during the tail finds the answer fully heard.

**Canvas edits** (§3.4 `canvas_edit`; `conversation.parse_edit`, `CanvasService.edit`). Keyword grammar, no model, English, Hindi and Hinglish: a kind change ("make that a bar chart", "show it as a table", "bar chart instead", "make the pie a bar chart", "isko bar chart mein dikhao", "इसे टेबल में दिखाओ"): everything but the kinds must be edit vocabulary (verbs, references, prepositions, politeness), so "show me revenue as a bar chart" is a new visual; remove ("remove the pie", "delete that chart", "हटा दो", "isko hata do"); pin / unpin ("pin this", "pin kar do", "इसे पिन करो", "pin hata do"); clear ("clear the canvas", "सब हटा दो": pinned panels stay); periods ("put FY23 next to it", "FY23 bhi dikhao", "FY23 भी जोड़ो", "compare it with FY23"; "only FY24"; a fiscal year asked for as a whole on a quarterly chart means its quarters: "only FY24" keeps "Q1 FY24" … "Q4 FY24", not the table's "Full year FY24" row, `service.match_periods`); a question is never an edit. "That / it / this / isko / इसे" is the visual the conversation touched last (added or edited most recently; the frontend doesn't report a focused panel); "the pie", "the line chart" pick by kind, "the second chart" / "दूसरा चार्ट" by screen order, "the revenue chart" by title and series words. Kind changes and periods are rebuilt from the stored spec through the validator and builder (every number still a cell; the id, position and pin kept; a kind that can't show it says "I can't show that chart that way.": a kind change keeps every table, series and period of the chart, so "show it as a table" of a chart of FY23's and FY24's quarterly tables is a table of both (`edits.change_kind`), and what the planner chooses for an edit the grammar can't read is merged with what was on screen unless the user asked to drop something, `edits.merge_planned`). A rebuilt visual is drawn in the chat's response language (the one asked for, else the sticky preference, else the latest answer's, `edits.edit_language`), not the language of the edit utterance: a Hindi edit said in an English chat leaves the labels and units English. What the grammar can't read ("add EBITDA to that chart", FY23 from another table) and what the router model calls `canvas_edit` goes to the planner, told what is on screen, with the visual's tables first (`visual preparing` with the panel's id, then the rebuilt visual). The reply is fixed and tiny: "Done." / "हो गया।", or "There's no chart on screen yet.", "It's already shown that way.", "That's already on the chart.", never an abstention. Accuracy: the grammar's 26 phrasings and 8 near misses (new visuals, questions, "show the table") are unit-tested; on the real model all spoken edits of the end-to-end run applied (kind changes in English and Hindi, remove) and a request for the kind the planner had already chosen got "It's already shown that way."

**Visuals as conversation context.** The router's input carries the canvas (§3.4: kinds, titles, x and series labels, highlights; no numbers; at most three panels) and what the utterance points at, found by code: an ordinal item ("the second bar" → Q2 FY24, "दूसरा बार"), the dip (the largest fall from the point before), the peak, the lowest, the highlighted point ("there"). A question about the screen (it names a chart, a part of one by its place, or its shape: "that dip") is a document question; its turn puts the tables of the visual it means first among the sources (they pass the gate: the user pointed at them) and the answer prompt gets that visual's line and the point. Real model: "What's the second bar on the chart?" → "The revenue for Q2 FY24 was ₹1,801 crore [S1]." (correct, cited, no new visual). Prompt cost: ~40 tokens a panel in the router's input, one line and the table (which is the evidence anyway) in the answer's, only when a question is about the screen.

**Speculative preparation** (workstream 10) is **not built** as such: preparing while the user speaks would save milliseconds (the candidates and the draft are code: a few milliseconds), and the planner's model call can't run then without competing with speech recognition and the router on the same machine, nor before the answer without delaying it (table above). The draft is the part worth doing early, and it is done at retrieval time.

**Accuracy, honestly** (`test_canvas_planner_eval.py`, real model, every corpus dataset a candidate, no retrieval: the question and its English query only, forced planning). Three paths: **old**, the planner alone with #38's candidates; **draft**, code alone; **draft + planner**, what a turn does now. Kind / table / series / all three right; series counts what is plotted including the x items (the slices of a donut, the steps of a waterfall: hold-out v2's definition; #38 counted only series and tiles, given in brackets for v1):

| set | old (#38) | draft | draft + planner |
|---|---|---|---|
| tuning (30, the planner's prompt and the draft tuned on it) | 30 / 30 / 30 / **30** | 30 / 29 / 30 / **29** | 30 / 30 / 30 / **30** |
| hold-out v1 (27; questions and misses known while the draft was written) | 25 / 20 / 26 / **19** (#38's rule: 15) | 25 / 22 / 25 / **21** (#38's rule: 20) | 26 / 25 / 27 / **24** (#38's rule: 23) |
| hold-out v2 (32 + 4 that ask for what no table holds; blind, run once) | 30 / 25 / 27 / **24** (strict: 17; "none": 0/4) | 26 / 30 / 27 / **22** (strict: 18; "none": 0/4) | 31 / 31 / 28 / **27** (strict: 22; "none": 0/4) |

Every visual of every path grounded. The other company's table, the commonest #38 miss (6 of v1's 12), is gone (0 on v1): a question that names a company is offered only its tables. What is left on v1: a confident draft from the wrong table of the right company (h04: the balance sheet for "how cash moved over the year"), a kind the planner changed (KPI tiles asked, bars drawn), the business overview for the shareholders. On v2 (blind, its one run, after the code was final): draft + planner 27/32 against the planner alone 24/32, with 14 planner calls instead of 36; by language EN 14/16, HI 6/8, Hinglish 7/8 (planner alone: 13/16, 6/8, 5/8). The draft alone is weaker on kinds (26/32: the table's default shape where the question wants a comparison or a bridge) and better on tables (30/32 against 25/32); 3 of its 20 confident drafts were wrong and so never refined (a donut of the balance sheet where particular rows were asked, the business overview for a people question, bars where another kind was wanted). No path left the 4 questions about something no table holds without a visual (0/4): the planner, forced, takes the nearest table, and the draft always does; in a conversation retrieval decides first (such a question usually abstains, and an abstained turn gets no visual), which this harness doesn't run. "Strict" counts every expected series shown; #38's series rule (series and tiles only, no x items) reads v2's labels badly, since they name the segments on the x axis: draft + planner 14/32 by it, the planner alone 16/32.

**The gate** (`route.visual`, keywords, EN / HI / Hinglish, §3.4): hold-out v1 25/27 want a visual (19/27 before; v1's questions were known), v2 31/32 (the old gate too: v2's questions mostly say "show" or "chart"; its 4 questions about what no table holds all ask to see something, so words can't tell: retrieval does). Over the four sets, precision 86 of 96 flagged, recall 86 of 89 chart questions (the old gate: 80 of 112, 80 of 89); false positives: the router eval's 69 routing cases 1 (5 before: "How does our revenue growth compare with India's GDP growth?" left, a comparison), the retrieval eval's 194 ordinary document questions 5 (23 before: "dividend per share", "each day", "what share of revenue came from exports" gone; all 5 left name two periods: "capex planned for FY25 and FY26").

**The planner before the draft** (#38, for the record): the tuning set (30 questions the prompt was tuned on): 30/30. A **hold-out set** (`tests/integration/canvas_planner_holdout.json`, 27 EN / HI / Hinglish questions over the eval corpus, written by a separate agent from the corpus's table inventory and the documents' text, without seeing the planner's prompt or the tuning set; forced planning so the planner itself is measured): kind 25/27 (93%), table 20/27 (74%), series 22/27 (81%), **all three right 15/27 (56%)**; every planned visual grounded. The commonest miss (6 of 12) is the other company's table for a question that names one ("Zephyra's segments" → Valmora's business overview): the planner is given every document's candidates and the company is only in the filenames. In a conversation the turn's retrieval (which since #37 keeps to the named company's passages) ranks its own table first, which the e2e run shows (3/3 visuals from the expected table), but the planner itself didn't check whose table it picks: fixed since (`services/subjects.py` in `rank_candidates`, above). 4 of the misses are "series" on waterfall / bridge / balance-sheet questions where the harness reads the column as the series (partly the labels, partly the choice). The gate: 19 of the 27 questions' words ask for a visual (`route.visual`); 8 don't ("Who owns Valmora? Break the shareholders down for me.", "Which of Valmora's three businesses contributes the most to sales?"): the conversation would not draw one unless the answer's figures suggest it.

**End to end on the real model** (`tests/integration/test_canvas_conversation_e2e.py`: the real text path with the real router, answer, planner, builder and canvas over the report's typed tables; retrieval cached as in `test_answer_latency.py`): a requested chart (bar of quarterly revenue, from the right table), an edit, a question about the chart (answered from its table), a suggested chart (comparison of segment revenue), a Hindi request (line, Hindi title), a Hindi edit to a table, "हटा दो": every visual grounded, every edit applied; visual ready 3.6–5.1 s after the answer's text, 5.5–9.7 s after its first delta (the planner then ran for every visual; that run predates the draft). With the draft, the voice run above has the chart on screen 2–4 ms after the answer's first audio: **the phase's "≤ 1 s after the spoken answer starts" is met**, for confident drafts at once and for the others with the draft (the planner's chart replaces it 2.5–7.3 s later). This e2e text run was not repeated with the draft.
