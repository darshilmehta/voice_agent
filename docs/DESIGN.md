# poc_gibberlink — Design

> **Status:** Phase −1 **complete** (2026-10-08): every local component downloaded and smoke-tested, including a full network-off run. Next: phase 0. No application code yet.
> **Last updated:** 2026-10-08 (config moved to JSON: local + cloud template)
> **Origin:** derived from [`blueprint.md`](blueprint.md). This document records what we actually decided; where the two disagree, this one wins.

---

## 1. What we are building

A fully local, ChatGPT "voice mode"-style agent that talks with you about your uploaded documents.

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
   │  SQLite: documents, jobs, sessions, turns              │
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
→ Qdrant prefetch (top 20 each, filtered to active documents) → RRF fusion
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

### 3.5 Conversation state and memory

```json
{
  "session_id": "…",
  "active_document_ids": ["doc_01"],
  "active_topic": "financial_performance",
  "previous_topic": "small_talk",
  "input_language": "hi",
  "response_language": "hi",
  "agent_state": "SPEAKING",
  "last_interrupted_turn_id": "t41",
  "retrieval_enabled": true
}
```

Agent states: `IDLE → LISTENING → THINKING → SPEAKING → (INTERRUPTED → LISTENING)`. Prompt context = recent turns + compact session summary + retrieved evidence + current utterance; never the full transcript.

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

- **Application code decides** when a tool runs (router output validated by Pydantic); the LLM never calls the network itself.
- Only the rewritten **search query** leaves the machine — never document text, never the transcript.
- Web facts and document facts are labelled separately in the answer and in citations; the agent must not present web data as coming from the report.
- Partial answers are streamed: the provider interface yields results incrementally, and answer generation starts on the first useful result (target: first spoken words < 1 s after the filler).
- **Offline guard:** with `strict_offline: true` the tool is refused unless `"web_search"` is listed in `strict_offline_exceptions`. The UI shows a "searching the web" badge whenever it runs.
- Timeouts (`tools.web_search.timeout_s`) end the search gracefully: "I couldn't get live data just now; from the report, …".

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

Local status: config has `provider: searxng`, `enabled: false` until phase 8; enabling it also requires `strict_offline_exceptions: ["web_search"]`. In the fully offline smoke run (test 12) web search is expected to be unavailable and must degrade to a document-only answer.

### 3.6 Failure handling

Degrade, never fake: LLM down → explicit error; Qdrant down → general chat only, document mode disabled with a message; Docling fails → document marked `FAILED` with the error; STT fails → text input still works; TTS fails → text still shown.

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
├── sqlite/app.db             documents, document_versions, ingestion_jobs, sessions, turns
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

Both files have **identical keys** (142) and are validated by one Pydantic schema; a test loads both so the template can't drift.

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
class JobQueue(Protocol):             # in_process | redis(stub)
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
│   │   ├── providers/       one folder per capability: base.py + implementations + cloud stubs
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
- **Reranker scores are not calibrated across languages.** A fixed 0.3 abstention cutoff would reject real Hindi questions. Changes: (1) the router also emits an **English search query** for non-English turns and the reranker scores that; (2) abstention uses **top score + gap to the next candidate + dense similarity**, thresholds tuned on the eval set.

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
| Instant pre-synthesized acknowledgement ("Sure,", "Let me check") | perceived wait ≈ 1 s |

Expected after these: **~2.5–3 s** to the first content audio on this Mac. The original 1.5–2.5 s target is likely out of reach with router + answer model on an M4 base; perceived latency is handled with acknowledgements.

- Still-open bug: "let's go back to the annual report" → router `backchannel`, answer claims no document access even with the prompt guard (4b ignores it). Fix in build: rule + session-state for resume phrases; never call the answer model without either sources or an explicit general-knowledge route.
- **Barge-in semantics (from the user's mic test):** ducking alone isn't enough — the agent kept talking and finished its thought. Required behavior: duck on speech onset → **stop and discard the rest of the answer** once speech is confirmed (≥ `minSpeechMs`, then transcript check) → restore volume on misfire. `mic.html` now implements duck → stop → restore.

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

### Setup notes for this machine

- Ollama runs as a brew service (`brew services start ollama`), bound to `127.0.0.1:11434`.
- Docker Desktop memory set to 1.5 GB (`MemoryMiB: 1536` in `~/Library/Group Containers/group.com.docker/settings-store.json`; backup alongside).
- The home router's DNS (`192.168.0.1`) stopped resolving during downloads; `1.1.1.1` / `8.8.8.8` were added as fallbacks. Irrelevant once everything is downloaded.
- Before going offline: install spaCy `en_core_web_sm` into the backend env (Kokoro's `misaki` otherwise fetches it at first run).

---

## 10. Build phases (after Phase −1 is green)

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Skeleton: config loader, provider registry, `/health`, `/api/config/public`, frontend shell with runtime config | `/health` reports every provider; both config files validate; `cloud.config.json` wiring test passes (placeholders raise NotImplemented) |
| 1 | Text-only document chat: upload → Docling → chunks → BGE-M3 → Qdrant → Qwen with citations | fact questions cited; out-of-document questions abstain; two docs distinguished |
| 2 | Hybrid retrieval + reranker + confidence gate | identifiers, paraphrases, numbers, page citations correct on eval set |
| 3 | Router + session state | unrelated questions skip retrieval; follow-ups and "back to the report" work |
| 4 | Voice in: browser mic → WebSocket → VAD → Whisper → transcript events | EN/HI transcription, silence handling, end-of-turn detection |
| 5 | Voice out: streamed answer → sentence TTS → browser playback | first audio after first sentence; text and audio in sync |
| 6 | Barge-in (duck-then-decide), corrections, stop | interrupt mid-answer; "no, I meant…" handled; backchannels ignored |
| 7 | Topic drift + EN/HI switching + code-mixing | drift script passes: document → general → Hindi → document |
| 8 | Live-data tools: web search with filler + streamed partial answers (§3.7) | mixed doc+live question answered with separate [S]/[W] citations; first words < 1 s after filler; barge-in cancels search; timeout falls back to document-only answer |
| 9 | Evals + observability + UI polish | retrieval Recall@5/MRR, groundedness, latency dashboard; demo script runs end to end |

Each phase is green (tests + acceptance criteria) before the next starts.

---

## 11. Open items

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

#### Suggested placement

After the core voice loop (phases 0–7): datasets hook in phase 1 (workstream 1), then a dedicated **"Visual canvas"** phase covering workstreams 2–9, with speculative preparation (10) last. To be scheduled once phases 0–7 are green.
