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

Chunks keep provenance: `document_id`, `version`, `chunk_id`, `page_start/end`, `heading_path`, `content_type` (paragraph/table/list), `language`. Tables are never split and are stored as markdown (+ optional text summary). Tables are **also persisted as typed, cell-level datasets** (§12.1, workstream 1) so later features (visual canvas, calculator) never re-parse text. Chunk sizing (~300–700 tokens, 50–100 overlap) is a starting point to tune with evals.

### 3.2 Retrieval (per document question)

```text
query (rewritten if it's a follow-up) → BGE-M3 dense + sparse
→ Qdrant prefetch (top `retrieval.prefetch_k` = 8 each, filtered to the project and the chat's documents) → RRF fusion
→ bge-reranker-v2-m3 → top 5 → context builder (dedupe, group by section, budget, [S#] ids)
→ confidence gate → answer or abstain
```

### 3.3 The conversation loop — what makes it feel like a conversation

**a) Short spoken replies, details on screen.** The voice answer prompt targets 1–3 sentences and offers more ("want the breakdown?"). Citations, tables and longer detail go to the transcript panel, not into speech.

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
    ]
    needs_retrieval: bool
    rewritten_query: str | None   # only for contextual follow-ups / corrections
    query_en: str | None          # English search query for non-English turns (reranker scores EN–EN best)
    tools: list[Literal["web_search"]] = []   # live data needed (§3.7)
    topic: str
    is_topic_shift: bool
    response_language: Literal["en", "hi"]
    confidence: float = Field(ge=0, le=1)
```

Router input: the new utterance, the last few turns, the session state, and — if the agent was interrupted — the partial answer that was played. Application code owns state; the model only proposes changes. Default language rule: answer in the language of the latest utterance unless asked otherwise.

**As built (phase 3).** The schema above is the persisted `route`; the model itself proposes only `{intent, query}` (`query` = the standalone English question) and application code validates it and derives the rest (topic, topic shift, languages, confidence):

- **Keyword fast path, no model:** stop phrases ("stop", "bas", "ruko", "रुको"); acknowledgements while the agent is idle ("okay", "yeah right", "haan theek hai"); thanks and greetings; a request that only changes the language; English standalone questions that name the documents. Fixed replies need no LLM at all.
- **Speculative retrieval** runs in parallel with the router call: reused when the route keeps the query (for Hindi the reranker scores the English query), discarded otherwise. An English standalone question whose retrieval scores ≥ 0.6 before the router answers skips the router.
- **Bounds:** `num_predict` 96 and `llm.router_timeout_ms` (2.5 s); on timeout or invalid output the turn falls back to a document question (`router.source: "fallback"`).
- **Validation overrides:** resume phrases always resume; a "stop"/"backchannel" that asks something is a question; a correction with nothing before it is a document question; "yes please" to the agent's offer searches for what was offered.

| Intent | Search | Answer (`route.answer`) | `abstained` |
|---|---|---|---|
| document_qa | yes (rewritten / English query) | grounded, `[S#]` | only if not covered |
| mixed | yes | grounded + marked general knowledge; not covered → general with `general_note: "not_covered"` | false |
| general_qa | no | general (says it isn't from the documents, no citations) | false |
| conversation | no | short LLM reply, or a fixed ack for thanks / greetings / language requests | false |
| clarification | no | one question back | false |
| resume_document | no (yes if it also asks something) | fixed text naming the document and topic, or grounded | false unless grounded and not covered |
| correction | inherits the last answered question's mode | as that mode | as that mode |
| backchannel | no | "Anything else?" / "और कुछ जानना है?"; nothing if that was just said | false |
| stop | no | silent: an `event` message "Stopped", nothing spoken | false |

**Language rule** (first that applies): a language the user asks for (it then sticks) → the request's forced `language` → the pinned preference → the utterance's language (romanized Hindi counts as Hindi and is answered in Devanagari) → the previous answer's language. Hindi answers keep citations.

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

- **Application code decides** when a tool runs (router output validated by Pydantic); the LLM never calls the network itself. A question-answering turn gets `tools=["web_search"]` when its utterance (or its standalone / English question) carries a live-data cue and the tool is available. Cues (`services/live_data.py`, EN / HI / Hinglish) are conservative: *strong* time cues ("right now", "real-time", "latest news", "current share price", "current repo rate", "price today", "the stock doing today", "लाइव") always count; *weak* time cues ("today", "this week", "आज", "अभी", "aaj", "abhi"; not before meeting words: "today's agenda", "आज की बैठक", "aaj ki meeting") and *topic* cues ("news", "latest update", "current price", "share price", "exchange rate", "weather", "ख़बर", "khabar") don't count when the question points at the documents ("the report", "the minutes", "according to the…", a filename) or at a past period / something stated ("FY24", "Q3", "at the end of the year", "on 31 March", "mentioned", "assumed", "used", "per unit", "buyback"). "Current ratio", "current rates of depreciation" (a "current … rate" needs a qualifier such as repo or exchange) and "how is the company doing now" (no stock subject) are document questions.
- Only the rewritten **search query** leaves the machine — never document text, never the transcript. It is built from the router's English standalone question: only the clauses asking for live data, phrases and parts pointing at the documents removed, filenames and document names removed, figures the user didn't say (an earlier answer's amounts, %, crore…) removed (years and FY tags stay), emails, URLs, long digit runs and ID-like tokens removed. **By design**, an entity the router resolves from the conversation (the company's name for "the stock") can be part of the query. A Hindi turn without an English query searches nothing.
- Web facts and document facts are labelled separately in the answer and in citations; the agent must not present web data as coming from the report. Web text enters prompts without square brackets (no planted `[S1]`), and earlier answers that used the web are marked so in the history.
- Partial answers are streamed: the provider interface yields results incrementally, and answer generation starts on the first useful result (target: first spoken words < 1 s after the filler). While the search runs, the model reads the prompt's start (system, history, document passages) so only the web snippets remain to read. Later results get at most `max_continuations` one-sentence continuations after the answer; **an answer is complete when its own text is**: stopped while only a continuation was still to come, it is saved complete, not interrupted (voice: speech after the answer has played ends the turn, it is not a barge-in).
- **Offline guard:** with `strict_offline: true` the tool is refused unless `"web_search"` is listed in `strict_offline_exceptions`. The UI shows a "searching the web" badge whenever it runs (`tool` events).
- **Without live data:** turned off (the default) → no search and nothing said about it (one prompt line: never guess current prices, rates or news). Turned on but unavailable now (SearXNG unreachable) → "I can't look up live data right now." only when the documents don't answer the question. Timeouts (`tools.web_search.timeout_s`) and failures end the search gracefully: "I couldn't get live data just now; from the report, …".
- **Page fetches** (top `fetch_pages` results): http(s) on ports 80/443 to public addresses only, connecting to the very address that was checked (no DNS rebinding), no environment proxies, a generic browser User-Agent, no cookies kept, at most 400 KB read and inflated (one gzip/deflate layer inflated by us with a cap; stacked or other encodings refused), text extracted off the event loop.

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
| `audio_chunk` | `turn_id`, `chunk_index`, `text`, `duration_ms`, (`filler: true`) | Announces one synthesized clause/sentence: its exact text and audio length. Its binary frames follow. Used for word-synced captions and to compute what was heard. `filler: true` marks the "Let me look that up." spoken when a turn searches the web (§3.7): not part of the answer, never counted in `heard_text`. |
| `tool` | `turn_id`, `name: "web_search"`, `phase`, `query`, … | Live-data tool progress (§3.7, same payload as the SSE `tool` event): drives the "searching the web" badge. |
| `agent_message` | `message: Message` | Final saved agent message (full text, typed citations, `route`, `latency`; `heard_text` set only if interrupted). |
| `barge_in` | `turn_id`, `decision: "stop" \| "resume"` | Server's decision after `barge_in_start`: **stop** = cancel generation + TTS, client stops playback and flushes queued audio for that turn; **resume** = it was a backchannel/noise, client restores volume. |
| `error` | `detail`, `stage: "stt" \| "retrieval" \| "llm" \| "tts" \| "storage" \| "audio"` | The turn failed at that stage; the session stays open and keeps listening. |

**Turn-taking rules.**

1. **End of turn**: the server's VAD (Silero, `vad.*` config) on the incoming audio is the source of truth. After `end_of_turn_ms` of silence the utterance is transcribed (STT; language restricted to the configured languages), saved as a user message, and answered with `ChatTurnService` using `modality="voice"`, `length="short"`. Utterances shorter than `vad.min_speech_ms` are ignored.
2. **Speaking**: answer deltas are cut into speakable chunks (first chunk at the first clause boundary or ≤ 5 words; then sentences; never right after an abbreviation such as "Rs." or "U.S."), each synthesized by Kokoro and streamed as `audio_chunk` + binary frames while generation continues.
3. **Barge-in** (§3.3 duck-then-decide): on `barge_in_start` the server keeps listening and decides:
   - **stop** when the transcript has real words beyond acknowledgements (more than `voice.barge_in.backchannel_max_words` of them, or an interruption cue such as "wait", "no", "रुको", a question word), or when speech is still going at the deadline. It cancels the answer and saves the agent message with **`heard_text`** = the text of fully played chunks plus the proportional share (whole words) of the chunk playing at the current playback position; the new utterance becomes the next user turn.
   - **resume** for acknowledgements and hums ("okay", "mm-hmm", "M M", "haan", "achha theek hai", "हम्म"), noise, or a short burst that has ended. A partial transcript with fewer than 2 real words never decides stop on its own.

   A decision is always sent within `voice.barge_in.decision_timeout_ms`. The server may also send `barge_in: stop` **without** a `barge_in_start` (the user spoke while the answer was still being written and the client's VAD didn't fire); the client treats it the same way.
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
- **Live data (§3.7).** A turn that searches the web sends `tool` messages after `turn` (never after a cut). Its filler `audio_chunk` (`filler: true`, chunk 0) and all its frames come right after `tool start` and **before `sources`**: the one audio that may precede `sources` (it carries no answer text and no citation; `sources` still precedes every `delta`). `state: speaking` comes with the answer's first audio, not the filler's. `tool done` may arrive between deltas, and a continuation (`tool results`, more `delta`s and `audio_chunk`s) may follow seconds after the answer's text: only `agent_message` ends the turn. The answer's last sentence is spoken as soon as its text is complete, not after the continuation. Speech after the answer has fully played, while only a continuation was still to come, is not a barge-in: no `barge_in` decision, no `interrupted` state; the answer's `agent_message` (complete, no `heard_text`) comes before the next `user_message`.

**Startup.** The backend preloads the embedder, reranker, STT and TTS models and the Ollama model (with the answers' `num_ctx`) at startup, in the background (~20–35 s, reported by `/health`), so the first spoken question isn't slowed by model loading (§9 measured ~3.7 s for a cold reranker).

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

| Component | When loaded | Approx. |
|---|---|---|
| macOS + browser + editor | always | ~5 GB |
| Docker VM (Qdrant) | always | ≤1.5 GB cap (Qdrant itself ~270 MB) |
| qwen3:8b (+ 8k-context KV cache) | conversation | ~6 GB (qwen3:4b ~3.5 GB) |
| BGE-M3 fp16 | conversation + ingestion | ~1.2 GB |
| Reranker fp16 | conversation | ~1.1 GB |
| Whisper | conversation | 0.5–1.6 GB |
| Kokoro | conversation | ~0.3 GB |
| Docling models | ingestion only, then unloaded | 1–2 GB |

Measured (§9.5): with Safari/Chrome/Claude app open, the machine was already swapping before our stack loaded. **Demo rule: close browsers other than the app's tab and heavy apps.** Kokoro runs on CPU to keep GPU memory free.

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
- **Reranker scores are not calibrated across languages.** A fixed 0.3 abstention cutoff would reject real Hindi questions (and, measured in phase 4 on the smoke documents, even Whisper's exact transcript of an English revenue question scored 0.282: answerable questions scored 0.074–0.983, unanswerable ones ≤ 0.013, so `retrieval.min_rerank_score` is **0.05** until phase 2 re-tunes it on the eval set). Changes: (1) the router also emits an **English search query** for non-English turns and the reranker scores that; (2) abstention uses **top score + gap to the next candidate + dense similarity**, thresholds tuned on the eval set.

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

Kokoro device: offline run measured MPS 0.31 s vs CPU 0.50 s full-sentence first audio (2 of 3 runs favour MPS). Config stays `cpu` to keep GPU memory for the LLM; revisit when tuning latency (MPS costs ~0.56 GB).

### Setup notes

- Ollama runs as a brew service (`brew services start ollama`), bound to `127.0.0.1:11434`.
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
| 3 + 7 Router, state, revisit features | ✅ done | #23 automatic titles, user summaries, transcript export; #24 router + conversation state + memory summary + drift and EN/HI/Hinglish switching: 43/43 intents on the labelled set with `qwen3:4b-instruct` (prompt tuned on that set), router p50 ≈ 0.65 s / p95 ≈ 1 s on routed turns, 0 on fast-path turns; #25 summary/export/title UI and routed-answer labels; #29 title job lane |
| 10 Live visual canvas | in progress | **part of the MVP** (user decision 2026-10-09); §12.1, contract v1 |

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
| 10 | **Live visual canvas** (§12.1, MVP): typed datasets, chartability, `VisualSpec` + validator + calculator, overview dashboard on upload, canvas panel, router/voice integration | every plotted number grounded to a cell (100%), hallucinated values 0; "show me revenue over five years" puts a cited chart on screen ≤ 1 s after the spoken answer starts; voice edits ("make it a bar chart", "remove that") work; overview dashboard after upload |

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

**Part of the MVP (user decision 2026-10-09)** as phase 10 (§10). Workstream 1 landed in phase 1 (every Docling table is stored as a cell-level dataset in `document_tables`). Workstreams 2–4, 6 (storage and API), 7 and 9 are built first and in parallel; 5 and 8 (router and voice integration) follow the quality round; 10 (speculative preparation) comes last.

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
- **Overview** (per project, built after ingestion): `GET /api/projects/{id}/overview` → `{panels: [Visual], status: "ready" | "building" | "none"}`.
- **Events** on both transports (SSE and the voice WebSocket, with `turn_id` on the WebSocket): `visual {phase: "preparing" | "ready" | "failed", visual_id, visual?: Visual, detail?}` and `canvas {panels: [Visual]}` (a snapshot after any change). A visual never delays the spoken answer; `preparing` lets the UI show a skeleton.
- Additive changes are allowed; anything else changes this section first.
