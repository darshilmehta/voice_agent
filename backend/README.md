# Backend

FastAPI service: configuration, provider registry, health, projects/chats/transcripts persistence, document upload and ingestion, text chat answered from the documents with citations, and the voice loop (a WebSocket per live voice session: VAD, STT, the same answer pipeline, TTS, barge-in). Architecture: [`../docs/DESIGN.md`](../docs/DESIGN.md).

## Run

```bash
uv sync
```

Ingestion, retrieval and speech run local models (Docling, BGE-M3, bge-reranker-v2-m3, Silero VAD, mlx-whisper or faster-whisper, Kokoro) from the optional `ml` dependency group. Without it the app starts and `/health` marks those providers `degraded`:

```bash
uv sync --group ml
```

At startup the conversation models (VAD, STT, TTS, embedder, reranker) load one after another in the background and Ollama is asked to load the chat model, so the first spoken question isn't slowed by loading; `/health` reports it under `preload` (`loading` → `ready`, or `degraded` with the reason per model). Documents uploaded meanwhile stay `PENDING` until the preload is over.

PyTorch work from different threads (ingestion, questions, speech) goes through one process-wide gate (`TorchGate` in `app/providers/models.py`): loading or freeing a model runs alone (a load changes process-wide torch state and copies weights to the GPU), inference on MPS runs one call at a time (a document is embedded one batch at a time, so a question waits for one batch at most), CPU inference runs alongside. Docling runs on the CPU, so a document being parsed doesn't hold up questions. Without the gate, concurrent MPS work aborts the process (Metal assertion, exit 134). mlx-whisper uses MLX's own GPU queue and isn't gated.

```bash
uv run python -m app
```

Host and port come from the config (`127.0.0.1:8000` locally). For auto-reload during development:

```bash
uv run uvicorn --factory app.main:create_app --reload
```

The config file is `../config/local.config.json` unless `APP_CONFIG_FILE` says otherwise ([`../config/README.md`](../config/README.md)). Configuration problems (missing `${VAR}` secrets, unknown keys, placeholder providers, anything that would reach the network under `strict_offline`) stop startup with exit code 2 and a message listing every problem.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Overall status plus every provider: `ok`, `degraded`, `down`, `disabled` or `not_implemented`, with a reason |
| `GET /api/config/public` | What the browser needs (title, languages, feature flags, limits, non-secret auth settings). Never secrets |
| `GET` / `POST /api/projects` | List projects (pinned first, then most recent activity; `?include_archived=true`) / create one |
| `GET` / `PATCH` / `DELETE /api/projects/{id}` | One project / rename, describe, `pinned`, `archived` / delete with everything inside it |
| `GET` / `POST /api/projects/{id}/documents` | The project's documents, newest first / upload one (below) |
| `GET` / `DELETE /api/documents/{id}` | One document (poll it for its status) / delete it with its vectors and stored files |
| `GET` / `POST /api/projects/{id}/chats` | The project's chats, most recent activity first / create one |
| `GET` / `PATCH` / `DELETE /api/chats/{id}` | One chat / rename, `pinned`, `archived`, `document_scope` (null = all documents), `language` / delete with its messages |
| `GET /api/chats/{id}/messages` | The transcript, paginated: `?after=<seq>` reads forward, `?before=<seq>` reads backward, `limit` ≤ 200 |
| `POST /api/chats/{id}/messages` | Ask in text: `{"text", "language": "en" \| "hi" \| null}` → the answer as Server-Sent Events (below) |
| `GET` / `POST /api/chats/{id}/summary` | The chat's user summary (`404` if none yet) / generate or refresh it (`?language=en\|hi`); see "Revisiting a chat" |
| `GET /api/chats/{id}/export` | Transcript download: `?format=md` (default) or `json`, as an attachment |
| `POST /api/chats/{id}/title:regenerate` | A new automatic title (`?force=true` also replaces a title the user set) |
| `GET /api/pins` | Pinned projects and chats for the sidebar, most recently pinned first |
| `WS /ws/chats/{id}/voice` | A live voice conversation in the chat (below) |

Lists return `{"items": [...]}`. Unknown ids are `404`, invalid input `422`, both with a `detail` message. `PATCH` changes only the fields sent; unknown fields are rejected. A transcript page is `{"items", "total", "has_more", "next_cursor"}`, items always in chronological order; pass `next_cursor` back as the same parameter (`after` or `before`) for the next page. To open a chat at its latest messages, request `before=<message_count + 1>`. A `503` means a dependency the operation needs is down (e.g. Qdrant during a delete); nothing was changed.

### Documents

Upload is `multipart/form-data` with one field, `file`. The extension must be in `ingestion.allowed_extensions` and the size within `server.max_upload_mb`; the content is sniffed (PDF signature, DOCX/PPTX package parts, text without binary data), the declared content type is ignored. Rejections are `422`. Identical content already in the project returns `200` with that document; a new file returns `202` with a new document in status `PENDING` (re-uploading a `FAILED` document's content retries it: `202`, same id). The original goes to the object store under a generated key (`<project>/<document>/v1.pdf`, never the uploaded name) and ingestion runs in the background job queue: `PENDING` → `PROCESSING` → `READY` (with `page_count`, `chunk_count`) or `FAILED` (with `error`). Each attempt is an `ingestion_jobs` row; parsed tables are stored cell by cell in `document_tables` (datasets for the visual canvas, docs/DESIGN.md §12.1). Ingestion models are released when the queue drains, and ingestions interrupted by a restart are queued again at startup. Deleting a document removes its vectors first (a `503` if Qdrant is down, nothing deleted), then its records, then its files; chats scoped to it drop it from `document_scope` (a scope left empty becomes null, all documents). Deleting a project does the same for everything in it.

### Chat

`POST /api/chats/{id}/messages` answers with `text/event-stream`. Each event is `event: <name>`, one `data: <JSON>` line and a blank line, in this order:

| Event | Data |
|---|---|
| `user_message` | the saved user message (same shape as the transcript's items) |
| `sources` | `{"sources": [Citation], "confidence": {"top_score", "gap", "dense_similarity", "above_threshold"}, "abstained": bool}` |
| `delta` | `{"text": "…"}`, zero or more: the answer as it is generated |
| `agent_message` | the saved agent message: final text and the citations it actually makes |
| `error` | `{"detail", "stage": "retrieval" \| "llm" \| "storage"}`; the stream ends (the user message stays saved) |

A `Citation` is `{"source_id": "S1", "document_id", "filename", "page_start", "page_end", "chunk_id", "snippet"}`; answers mark sources as `[S1]`. Unknown chat or invalid text fail before the stream (`404` / `422`).

Every turn is routed first (`services/router.py`, `services/planning.py`, docs/DESIGN.md §3.4): a keyword fast path recognises stop, acknowledgements ("okay", "theek hai"), thanks, greetings, pure language requests and English standalone questions that name the documents; anything else goes to the router model (`qwen3:4b-instruct`, JSON schema: intent + standalone English question, `llm.router_timeout_ms`), while retrieval of the raw utterance runs speculatively alongside it (a standalone question whose retrieval is confident before the router answers skips the router). The model only proposes; application code validates the proposal and decides the retrieval policy:

| Intent | What the turn does |
|---|---|
| `document_qa`, `correction` of a document question, `resume_document` with a question | search (rewritten / English query), grounded answer citing `[S#]`, or abstain |
| `mixed` | search; grounded answer that may add general knowledge, marked as such; not covered → general answer saying so |
| `general_qa`, `correction` of a general question | no search; general-knowledge answer, no citations, says it isn't from the documents |
| `conversation` | no search; one-line reply (thanks / greetings / language requests: fixed text, no model) |
| `clarification` | no search; one short question back |
| `resume_document` without a question | no search; fixed text: back to the document topic |
| `backchannel` | no search; "Anything else?" (nothing if that was just said) |
| `stop` | nothing is said; the turn ends with a `role: "event"` message |

Router failure or timeout falls back to phase 1: a document question on the raw utterance. Document search runs within the chat's project, narrowed to its `document_scope`, over `READY` documents only. If the best passage scores below `retrieval.min_rerank_score` the agent abstains with a fixed sentence in the user's language and no model call (`abstained: true`, no sources; only document questions abstain). Otherwise the top passages become numbered sources (deduplicated, grouped by section, within `retrieval.context_token_budget`) and the model answers from them only, citing `[S#]`; markers naming no source are removed from the saved text, and lists are normalized to `[S1][S2]`. The answer language (`services/language.py`): a language the user asks for ("answer in Hindi", which sticks), else `language` when given, else the language of the message (Devanagari or romanized Hindi → Hindi), else the previous answer's. Prompts carry the last 6 messages plus the chat's memory summary (refreshed in the background, `llm.memory_summary_*`), never the whole transcript. The conversation state (topics, languages, last interrupted answer, `retrieval_enabled`) is kept per chat in `chat_states`. The agent message records the route (`intent`, `rewritten_query`, `query_en`, `topic`, `is_topic_shift`, `response_language`, `answer`, `router`, `speculation`, top-level `abstained` and `stopped` booleans, retrieval confidence, model, prompt version) and latency (router, retrieval, rerank, first delta, total).

If the client disconnects mid-answer, generation is cancelled (the stream to Ollama is closed) and the text generated so far is saved as the agent message with `route.stopped: true`, citing only the sources that text cites; if no text was generated yet, no agent message is saved.

The turn itself is `ChatTurnService` (`app/services/chat_turns.py`), which knows nothing about HTTP: `begin(chat_id, text, language=…, modality="text" | "voice", length="short" | "full")` validates, and `run(turn)` yields typed events (`UserMessageEvent`, `SourcesEvent`, `DeltaEvent`, `AgentMessageEvent`, `ErrorEvent`, each with `name` and `payload()`). Cancelling the consumer or `aclose()` on the events stops the answer as above. This endpoint only serialises the events; the voice WebSocket runs the same turns with `modality="voice"` and speaks the deltas. `length="short"` (the default, and what this endpoint uses for now) asks for 1–3 speakable sentences with a 384-token cap; `"full"` asks for a fuller written answer (768).

### Voice

`WS /ws/chats/{id}/voice` is one live voice session (`app/services/voice/`; the endpoint in `app/api/voice.py` only adapts the WebSocket). A browser page from an origin not in `server.cors_allowed_origins` → close code `4403` (the CORS middleware doesn't cover WebSockets; requests without an `Origin` header are not browser pages and are allowed). Unknown chat → `4404`. Both are sent after accepting, so browsers see the code. One session per chat: opening a second one closes the first with `4409`, after the first has saved its state. A chat without `READY` documents still talks (its turns abstain). `end` closes with `1000`; server shutdown with `1001`. Messages over 64 KiB close the connection (uvicorn `ws_max_size`).

**Audio.** Client → server binary frames: PCM16 LE mono 16 kHz, 20–64 ms each (512 samples = 1,024 bytes typical), sent continuously while the mic is on; audio before `start` is ignored with an `audio` error. Input errors (malformed frames or control messages, VAD failures) are reported once per kind every 5 s, not once per message. Server → client binary frames: a 12-byte header (`uint32` LE `turn_id`, `chunk_index`, `seq`, `seq` counting the frames of one chunk from 0) + PCM16 LE mono 24 kHz, 200 ms per frame. Play only the current turn's frames.

**Control messages** (JSON text, field `type`). Client → server: `start {language: "en" | "hi" | null}` (null: detect per utterance among `stt.languages`), `barge_in_start {turn_id, played_ms}`, `playback {turn_id, played_ms}` (optional progress), `playback_done {turn_id}`, `stop {}`, `end {}`. Server → client:

| Message | Fields |
|---|---|
| `ready` | `session_id`, `input_sample_rate: 16000`, `output_sample_rate: 24000`, `language` (followed by the current `state`) |
| `state` | `state: listening \| thinking \| speaking \| interrupted` |
| `user_speech` | `phase: start \| end` (server VAD; `end` after `vad.end_of_turn_ms` of silence, or when a too-short burst is dropped) |
| `transcript_partial` | `text`: the speculative transcript taken during the end-of-turn silence, or the transcript of speech that interrupts the agent |
| `user_message` | `message`: the saved user message (`modality: "voice"`, the spoken `language`, STT timings in `latency`) |
| `turn` | `turn_id` of the answer that follows (1, 2, 3, … within the session) |
| `sources` | same payload as the SSE `sources` event |
| `delta` | `turn_id`, `text` |
| `audio_chunk` | `turn_id`, `chunk_index`, `text` (what is spoken: no `[S#]` markers), `duration_ms`; its binary frames follow, in order, before the next `audio_chunk` (other messages may come between them) |
| `agent_message` | `message`: the saved answer; `heard_text` set only if it was interrupted |
| `barge_in` | `turn_id`, `decision: stop \| resume`; after every `barge_in_start`, also when `stop` arrives while it is pending; also `stop` without a `barge_in_start` when an utterance that isn't a backchannel ends while the agent answers |
| `error` | `detail`, `stage: stt \| retrieval \| llm \| tts \| storage \| audio`; the session stays open |

**Ordering guarantees.** Within a turn: `user_message`, `turn`, `sources`, then `delta`s interleaved with `audio_chunk` + frames, then `agent_message` after the turn's last frame (so `playback_done` can follow it; if TTS fails partway, `error {stage: "tts"}` comes before `agent_message` and the text is still complete). Once `barge_in {decision: "stop"}` is sent, or `stop` is handled, nothing more of that turn is sent (no `delta`, `audio_chunk` or frame; the turn is muted before the decision goes out); its `agent_message` with `heard_text` follows — empty text if nothing had been generated — and comes before the next turn's `user_message`/`turn` (preceded by them if the cut came before they were sent). The user message is saved before the answer starts, so a cut turn always has both messages. `stop` while an utterance is still being transcribed (the client already shows `thinking`): `state interrupted`, then that utterance's `user_message`, `turn` and an empty stopped `agent_message`, then `state listening`; it isn't answered. A complete answer that is cut during playback is sent again as `agent_message` with the same id and `heard_text`. `turn_id` strictly increases within a session.

**Turn-taking.** The server's Silero VAD on the incoming audio is the source of truth: an utterance starts at speech probability ≥ `vad.threshold` (keeping ~190 ms before it) and ends after `vad.end_of_turn_ms` of silence (Silero's hysteresis: below threshold − 0.15); bursts under `vad.min_speech_ms` are ignored. At half the end-of-turn silence the utterance is transcribed speculatively and that transcript is used at the end of the turn unless speech resumed (§9.4). STT language detection is limited to `stt.languages` (or the `start` language). The transcript becomes a user message answered by `ChatTurnService` with `modality="voice"`, `length="short"`; the answer language is the spoken one. Utterances that are only hums ("hmm", "mm-hmm", "M M", "उम्म") or Whisper noise ("you") start no turn.

**Ignored utterances** always leave the client in a known state. Every `user_speech start` is followed by `user_speech end`; then:

| Case | What the client gets after `user_speech end` |
|---|---|
| shorter than `vad.min_speech_ms` | `barge_in {resume}` if a decision was pending, then the current `state` again (`listening`, `thinking` or `speaking`) |
| a backchannel or hum while the agent answers, with `barge_in_start` | `transcript_partial`s, `barge_in {resume}` (at the latest at the deadline, or twice the deadline while only acknowledgements are being said), then `state speaking` (or `thinking`) again |
| the same without `barge_in_start` (the browser VAD missed it) | `transcript_partial`, then `state speaking` (or `thinking`) again; no `barge_in` (none was started) |
| hums, noise, only acknowledgements ("Yeah, right.", "okay", "हाँ") or an empty transcript while the agent is silent | `state thinking` (sent at the end of the utterance), then `state listening` |
| transcription failed | `error {stage: "stt"}`, `barge_in {resume}` if one was pending, then the current `state` |

**Speaking.** Deltas are cut into chunks: the first at the first clause boundary (at least 2 words) or after 5 words, then whole sentences (a sentence over 30 words is cut at a clause); boundaries are punctuation followed by whitespace (`.!?।॥,;:`), so `18.2%` and `4,210` never split. Markers and markdown are not spoken; only the first `voice.max_spoken_sentences` sentences are spoken (the rest is on screen). Each chunk is synthesized with Kokoro while generation continues.

**Barge-in (duck, then decide).** On `barge_in_start` the server keeps listening and decides within `voice.barge_in.decision_timeout_ms`: once the new speech reaches `vad.min_speech_ms` it is transcribed (and again when it pauses). A backchannel is made of acknowledgements ("mm-hmm", "okay", "yeah", "haan", "achha theek hai", "ठीक है", …) and hums, however many; only other words count against `voice.barge_in.backchannel_max_words` ("yes please" passes), and an interruption cue ("stop", "wait", "no", a question word, "रुको", …) or the absence of any acknowledgement makes it not a backchannel. A transcript that isn't a backchannel is `stop` at once, unless it has fewer than 2 real words and the speech has already ended (Whisper mishears short snapshots: that waits for more evidence); hums, noise and empty transcripts never stop the answer by themselves. At the deadline, speech still going on is `stop`, a short burst that ended (or nothing) is `resume`; but if what has been said so far is only acknowledgements ("Yeah…" of "Yeah, right"), the decision waits up to twice `decision_timeout_ms` (transcribing again meanwhile) and is `resume` unless real words come: only those stop the answer. On `stop` generation and TTS are cancelled and the answer is saved with `heard_text` = the fully played chunks + the share of the chunk playing at `played_ms` in proportion to its duration, in whole words (rounded down); the new utterance becomes the next user turn when it ends. An utterance that ends while the agent is answering and isn't a backchannel stops the answer too, even without `barge_in_start` (also after a `resume`). `stop` cancels the same way (`route.interrupted: "stop"`; `played_ms` from the last `playback` report plus the time since, else from when each chunk should have played: from its arrival or the end of the previous one). When the session ends mid-answer (client gone, `end`, replaced by another session, server shutdown) the answer is saved the same way with `route.interrupted: "disconnect"`. Interrupted answers carry `route.stopped: true` and `route.interrupted: "barge_in" | "stop" | "disconnect"`; the next answer's prompt sees what was heard (nothing, if nothing was), not the whole answer. After an LLM error the cut-off fragment of the answer is not spoken.

A turn ends (`state: listening`) on `playback_done`, or 2 s after its audio should have finished playing if the client never says so (following the chunks as they were sent, gaps included, and pushed back by `playback` reports). Everything said is saved as messages of the chat, so the transcript view and the text endpoint see the same history.

### Revisiting a chat

**Titles.** A new chat is called "New chat" with `title_is_auto: true`. When its first agent message is saved (text or voice turn), `MessageService` runs the registered agent-message hook, which queues a job in the job queue's short lane (ingestion runs in the long lane, so a title never waits for a PDF conversion or for the startup model preload): the model writes a title of at most 6 words in the conversation's language (token cap, 20 s timeout, one retry); if it gives nothing usable the cleaned first question is used instead. The write only happens if nobody renamed the chat meanwhile (compare-and-set on the title and `title_is_auto`), so a user's title is never overwritten, and it changes `title` and `updated_at` for the sidebar's next fetch. A chat gets a title once: later answers don't regenerate it. `POST /api/chats/{id}/title:regenerate` returns the updated chat: `409` if the user set the title (unless `?force=true`), `422` without a user message, `503` if the model fails (nothing changes).

**Summaries.** `POST /api/chats/{id}/summary[?language=en|hi]` generates the summary (a chat with no messages is `422`, the model failing `503` with nothing stored) and `GET` returns the stored one. The response is `{id, chat_id, language, overview, key_points: [{text, sources: [{document_id, filename, page_start, page_end}]}], unanswered_questions: [{question, message_seq}], follow_ups: [str], content (the same as Markdown), covers_seq, message_count, stale, model, created_at}`. `stale` (`covers_seq < message_count`) means messages were added since: show "out of date" and POST again. Asking again for an unchanged chat in the same language returns the stored summary without calling the model. The model writes the overview, key points and follow-ups as JSON (validated with Pydantic); answers' per-answer `[S#]` markers are first renumbered chat-wide, and a key point's `sources` keep only numbers that were in the text the model saw, then become documents and pages. `unanswered_questions` are the user's questions of the turns the agent abstained on. Chats longer than the context budget (`llm.num_ctx` minus room for instructions and output) are summarised in windows and combined (map-reduce). The language is the chat's dominant one unless `language` is given.

**Export.** `GET /api/chats/{id}/export?format=md|json` returns the transcript with `Content-Disposition: attachment` (`fy24-margins-2026-10-08.md`: slugified title and creation date). Markdown has the title, project, creation date and languages, the summary (if any), every message with speaker, time and modality (an interrupted answer shows what was heard, then the full answer), citations as `[1]` and a sources list. JSON is `{"schema_version": 1, "exported_at", "chat", "messages", "sources", "summary"}`; `messages[].text` keeps the `[S#]` markers as saved and `messages[].citations[].ref` is the number in `sources` (the Markdown's `[n]`). No audio is stored, so none is exported. The models are in `app/services/export.py`.

## Database

Projects, documents, chats, messages and summaries live in SQLite (`data/sqlite/app.db`) through SQLAlchemy, so Postgres is a URL change plus implementing the `postgres` provider (docs/DESIGN.md §3.9, §6). Ids are readable and sortable: `prj_…`, `doc_…`, `cht_…`, `msg_…` (ULID layout). Deleting a project or chat cascades in the database (`ON DELETE CASCADE`, foreign keys enforced on every SQLite connection).

The schema is managed by Alembic. The app upgrades to the latest revision at startup, so there is nothing to run by hand. After changing `app/db/models.py`, generate a migration (it is linted and formatted automatically) and review it:

```bash
uv run alembic revision --autogenerate --rev-id 0003 -m "describe the change"
```

The CLI migrates the database the app config points at (`APP_CONFIG_FILE`; `METADATA_DB__URL` overrides it): `uv run alembic upgrade head`, `uv run alembic current`, `uv run alembic check`. A test fails if the models and the migrations disagree.

## Test and lint

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

Tests never touch the real Ollama, Qdrant, model files or database: HTTP is mocked and the project root is moved to a temp directory, so every test gets a fresh, migrated SQLite database. Tests that need the `ml` group skip without it.

Integration tests (real models, Docling and a running Qdrant; a temporary collection is created and dropped) are opt-in:

```bash
RUN_INTEGRATION=1 uv run --group ml pytest tests/integration -s
```

They read models from `MODELS_ROOT` (default `../data/models`) and the smoke-test documents from `SMOKE_DOCS` (default `../data/smoke/docs`, written by `scripts/smoke/05_docling.py`); Qdrant from `QDRANT_URL` (default `http://127.0.0.1:6333`). `test_chat_e2e.py` also needs Ollama with `qwen3:4b-instruct` and checks phase 1 end to end over HTTP: upload until `READY`, English and Hindi fact questions cited to page 2, an out-of-document question abstaining, two documents told apart, nothing leaking across projects. Timings, ranks and answers print in the summary.

`test_voice_e2e.py` (also needs Ollama, and the speech clips in `SMOKE_AUDIO`, default `../data/smoke/audio`, written by `scripts/smoke/09_kokoro.py`) runs the voice loop over the WebSocket with every real model preloaded: a spoken revenue question streamed in real time must come back transcribed, answered with citations and spoken; a spoken correction barges in mid-answer and must stop it (with `heard_text`) and be answered (Whisper must understand that spoken answer again); a Hindi question must be detected as Hindi. It prints the latency from the end of the user's speech to each stage.

`test_concurrency_e2e.py` (same needs) uploads the smoke PDF the moment the app starts, while the models preload (it must wait, `PENDING`, and then become `READY`), then times a text and a spoken question while idle and again while a 60-page PDF is being ingested.

`tests/integration/test_revisit_llm.py` needs only Ollama with `qwen3:4b-instruct` (no other model, no Qdrant): it titles and summarises scripted English and Hindi chats, and once in tiny windows to exercise map-reduce, printing the results to judge the prompts: `RUN_INTEGRATION=1 uv run pytest tests/integration/test_revisit_llm.py -s`.

## Layout

```text
alembic.ini          Alembic CLI config (the app migrates without it)
app/
  __main__.py        python -m app: validate config, then run uvicorn
  main.py            create_app(): lifespan, CORS, routers, error handlers
  settings.py        JSON config schema + loader (${VAR} secrets, SECTION__KEY env overrides)
  offline.py         strict_offline guard, HF offline env
  logging_setup.py   console or JSON logs
  api/               /health, /api/config/public, projects, documents (upload, delete), chats (+ messages, SSE), pins,
                     voice (WebSocket adapter); deps.py wires services
  db/
    models.py        SQLAlchemy models: projects, documents, document_versions, ingestion_jobs, document_tables, chats,
                     messages, chat_summaries
    types.py         UTC datetimes, JSON/JSONB, new_id()
    engine.py        metadata_db.url → engine (SQLite: foreign keys on, WAL)
    migrate.py       upgrade/downgrade from code
    migrations/      Alembic env.py + versions/
  domain/            Pydantic objects the services return (Project, Chat, Message, Citation, …)
  services/          business logic on the provider interfaces: projects, chats, messages, summaries, pins, documents
  providers/
    base.py          Provider, PlaceholderProvider, health model, local_snapshot()
    registry.py      capability → provider resolution, Container
    models.py        local-model health, lazy ML imports (ml group), LazyModelProvider (load/preload/unload)
    ingestion.py     DoclingParser: ParsedDocument (pages, items, tables + cells), chunking with provenance
    retrieval.py     BgeM3Embedder (dense + sparse), BgeReranker, QdrantStore (hybrid RRF search)
    llm.py           OllamaLLM: streamed chat + JSON-schema output (think off, num_ctx, keep_alive, temperatures)
    storage.py       SqliteDB (metadata), FilesystemStore (object store: safe generated keys, temp copies)
    runtime.py       InProcessJobQueue (long + short lanes of asyncio workers, idle hook, graceful shutdown), auth, sessions, events
    speech.py        SileroVAD (per-session streams), MlxWhisper / FasterWhisper, KokoroTTS, audio transport
    web_search.py
  services/
    ingestion.py     ingest_file: parse → chunk → embed → upsert (no DB writes)
    retrieval.py     hybrid search → rerank → top N + confidence signal
    document_pipeline.py  upload validation, dedupe, ingestion jobs, deletion of vectors/files/rows
    chat_turns.py         ChatTurnService: one turn (text or voice) as typed events: save → plan → retrieve →
                          gate → sources → answer; answer length is a parameter; stop = cancel
    router.py planning.py   turn router (fast path, router model, validation) and planning (speculative
                          retrieval, timeout/fallback, retrieval policy)
    conversation.py memory.py   per-chat conversation state; memory summary and when it may run
    sources.py prompts.py language.py   numbered sources + citation checks, prompts per answer mode, language rules
    preload.py            background model preloading at startup, reported in /health
    voice/                the voice loop: protocol.py (messages, frames, close codes), turn_taking.py (endpointing,
                          barge-in verdict), speech_text.py (speakable chunks, heard text, backchannels),
                          session.py (VoiceSession state machine, VoiceSessions: one per chat)
    titles.py             automatic chat titles: agent-message hook → job queue → LLM (fallback: the first question)
    chat_summary.py       user summaries: chat-wide source numbers, windows, map-reduce, grounding, Markdown
    chat_sources.py revisit_prompts.py markdown_text.py   chat-wide citation numbers; title/summary prompts; escaping
    export.py             transcript export (Markdown, JSON) and download file names
tests/
  integration/       opt-in, real models + Qdrant (RUN_INTEGRATION=1)
```

Each capability has a base class (for example `LLMClient`, `VectorStore`); its methods are added in the phase that builds it. A capability module becomes a package once it grows.
