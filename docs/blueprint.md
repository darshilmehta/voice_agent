# Local Document Conversation Agent — Project Blueprint

> **Status:** Recommended architecture and implementation blueprint  
> **Verified against current public project/model information:** 2026-10-08  
> **Goal:** Build a fully local demo where users upload documents, talk naturally with an AI agent about those documents, interrupt it, drift to unrelated topics, switch languages, return to the document, and optionally demonstrate two agents switching from natural speech to a GibberLink/ggwave data-over-sound channel.

---

## 1. Executive summary

This project is absolutely feasible as a local-first application with no per-token/API inference cost for the runtime.

The most important architectural decision is to **separate four concerns**:

1. **Document intelligence and ingestion** — turn PDFs/Office files/etc. into reliable structured content with page/section/table metadata.
2. **Retrieval-augmented generation (RAG)** — find the relevant document evidence using multilingual dense + sparse retrieval and reranking.
3. **Conversational voice agent** — handle speech recognition, interruption/barge-in, language switching, topic drift, conversation state, and answer generation.
4. **GibberLink/ggwave** — use data-over-sound only as an optional **agent-to-agent (A2A) transport**, not as the primary user↔agent voice channel.

That last point is important. GibberLink is based on ggwave, a small-data FSK data-over-sound protocol. The ggwave project describes typical throughput around **8–16 bytes/second** depending on protocol parameters, so it is excellent for compact machine messages but is not a sensible transport for a long natural-language conversation. The original GibberLink demo switches two AI agents to ggwave after they establish that the other side is also an AI agent. [ggwave](https://github.com/ggerganov/ggwave), [GibberLink](https://github.com/PennyroyalTea/gibberlink)

The resulting architecture should therefore be:

```text
                        ┌─────────────────────────────┐
                        │        Browser UI            │
                        │                             │
                        │  document upload             │
                        │  live microphone             │
                        │  transcript                  │
                        │  citations                   │
                        │  agent status                │
                        │  gibberlink visualization    │
                        └──────────────┬──────────────┘
                                       │
                         WebRTC media + WebSocket events
                                       │
                ┌──────────────────────▼─────────────────────┐
                │                 FastAPI                    │
                │                                            │
                │  conversation state                        │
                │  intent/topic router                       │
                │  RAG orchestration                         │
                │  voice orchestration                       │
                │  A2A/GibberLink control                   │
                └─────────┬───────────┬───────────┬──────────┘
                          │           │           │
                   ┌──────▼────┐ ┌───▼──────┐ ┌──▼──────────┐
                   │  Docling  │ │ Qdrant   │ │   Ollama    │
                   │ ingestion │ │ hybrid   │ │ Qwen /      │
                   │           │ │ retrieval│ │ gpt-oss     │
                   └─────┬─────┘ └────┬─────┘ └────┬────────┘
                         │            │             │
                         └────────────┴─────────────┘
                                      │
                             grounded answer
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                     faster-whisper          TTS
                           │                  Kokoro/Piper
                           │                     │
                           └─────── audio ───────┘

                     Optional A2A demo:

            Agent A  <── natural speech ──>  Agent B
                │                              │
                └──── handshake / consent ────┘
                              │
                       ggwave / GibberLink
                              │
                    compact binary messages
```

---

## 2. What the product should actually demonstrate

The strongest demo is not simply “chat with a PDF.”

It should demonstrate **stateful multimodal conversation over private knowledge**.

Example session:

```text
User: What was revenue in FY24?

Agent: FY24 revenue was $X, based on page 47 of the annual report.

User: And what was the EBITDA margin?

Agent: The EBITDA margin was Y%, also on page 47.

User: Actually, forget the report for a second. Explain what EBITDA means like I'm five.

Agent: Sure. EBITDA is a rough way to look at operating performance...

User: अच्छा, अब वापस रिपोर्ट पर चलो। FY25 revenue क्या था?

Agent: FY25 revenue was...

User: Also, do you think this company is growing fast?

Agent: Based on the report, revenue growth was...
```

Then a second demo can show the special A2A behavior:

```text
Agent A: Are you another AI agent?
Agent B: Yes.
Agent A: Do you agree to switch to data-over-sound?
Agent B: Yes.

         ↓

       GIBBERLINK MODE

         ↓

[compact ggwave payloads]

         ↓

Agent A and Agent B continue exchanging structured messages
```

The user should be able to **interrupt** and immediately return the system to human-facing natural speech.

---

# 3. Recommended technology stack

## Runtime stack

| Layer | Recommended choice | Why |
|---|---|---|
| Local LLM runtime | **Ollama** | Very simple local model serving and local HTTP API |
| Primary local LLM | **Qwen3 8B / larger Qwen3 variant** | Strong multilingual conversational behavior; good fit for local voice latency |
| Alternate reasoning model | **gpt-oss:20b** | Strong local reasoning/agentic capability; Apache 2.0 model; larger memory requirement |
| Document ingestion | **Docling** | Structured document parsing, tables, reading order, OCR, Office/PDF support |
| Vector DB | **Qdrant** | Local/self-hosted, hybrid dense+sparse retrieval, metadata filters, reranking support |
| Embeddings | **BAAI/bge-m3** | Multilingual, dense + sparse + multi-vector retrieval support |
| Sparse retrieval | **Qdrant BM25 / sparse vectors** | Exact terms, identifiers, numbers, names |
| Reranking | **BGE reranker / ColBERT-style local reranker** | Improve precision after candidate retrieval |
| STT | **faster-whisper** | Fast local Whisper inference with CTranslate2 |
| VAD | **Silero VAD** | Local speech segmentation and speech/non-speech detection |
| TTS | **Kokoro** first; Piper fallback | Good local TTS options; Kokoro is small and Apache 2.0 |
| Browser realtime media | **WebRTC** | Better fit for low-latency microphone/audio streaming and interruptions |
| Control/events | **WebSocket** | Conversation events, transcript deltas, status, citations, UI state |
| Backend | **Python + FastAPI** | Excellent ecosystem for ML/audio/RAG |
| Frontend | **React + Next.js** | Fast iteration and rich browser audio UX |
| Database | **SQLite** initially | Job state, documents, sessions, settings; no extra server required |
| A2A sound protocol | **ggwave / GibberLink concepts** | Experimental agent-to-agent data-over-sound demo |
| Containers | **Docker Compose** | Reproducible local setup |
| Dev assistant | **Claude Code** | Code generation, refactoring, debugging, tests |

### Why these choices are better than the first draft

The original concept was already viable, but for a reliable demo I would add/strengthen several things:

- **WebRTC for media**, rather than treating WebSockets as the audio transport.
- **Explicit conversation-state and intent routing**, rather than forcing every utterance through RAG.
- **Hybrid dense + sparse retrieval**, not embeddings alone.
- **Reranking**, because top-k vector search is not always precise enough.
- **Structured source metadata and citations** from ingestion onward.
- **Document/version/job records** in SQLite so ingestion is observable and repeatable.
- **A2A state machine for GibberLink**, rather than letting the LLM freely decide to emit encoded audio.
- **Grounded-answer checks**, so the agent can say it does not know instead of inventing an answer.
- **Evaluation datasets and retrieval metrics**, because reliability should be measured rather than judged from a few happy-path demos.

---

# 4. Local/free status

All major runtime components can be run locally without per-request inference charges.

### Fully local/open-source candidates

- Ollama local model serving
- Qwen3 open-weight local models
- gpt-oss open-weight local models
- Docling
- Qdrant
- BGE-M3
- faster-whisper
- Silero VAD
- Kokoro
- Piper
- ggwave
- FastAPI
- React / Next.js
- SQLite
- Docker

### Important licensing distinction

“Free to run locally” does not mean every model or voice has identical redistribution terms.

Examples verified in current public sources:

- Qwen3-8B: Apache 2.0 model listing. [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- gpt-oss-20b: Apache 2.0 model listing. [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- BGE-M3: MIT. [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- Docling codebase: MIT; its documentation explicitly notes that individual models have their own licenses. [Docling GitHub](https://github.com/docling-project/docling)
- Qdrant: Apache 2.0. [Qdrant](https://github.com/qdrant/qdrant)
- ggwave: MIT. [ggwave](https://github.com/ggerganov/ggwave)
- GibberLink repository: MIT. [GibberLink](https://github.com/PennyroyalTea/gibberlink)
- Kokoro-82M: Apache 2.0. [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Piper voices: voice-specific licensing must be checked from each voice/model card. [Piper project](https://github.com/rhasspy/piper)

Before commercial distribution, freeze exact dependency/model versions and perform a license audit.

---

# 5. Local LLM recommendation

## Primary model: Qwen3

Use Qwen3 as the default conversational model because the project needs:

- multilingual conversation
- topic switching
- instruction following
- structured output for routing/tool decisions
- reasonable latency
- local execution

Qwen3 was released as a family spanning small to very large models, and Qwen's model documentation includes broad multilingual coverage. The Qwen3-8B model is listed under Apache 2.0. [Qwen3 release](https://qwenlm.github.io/blog/qwen3/), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

### Suggested starting model

```bash
ollama run qwen3:8b
```

Use this as the default starting point unless hardware supports something larger.

### Bigger option

Try a larger Qwen3 model when latency remains acceptable.

### Alternative: gpt-oss:20b

```bash
ollama run gpt-oss:20b
```

Ollama currently lists the local 20B model at around 14 GB in its packaged format, with 128K context, and describes it as designed for lower-latency/local use. The model is Apache 2.0. [Ollama gpt-oss](https://ollama.com/library/gpt-oss), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

Do not run multiple large models concurrently on a constrained machine just because they exist. Start with one main LLM and keep the architecture model-agnostic.

### Separate coding model

Your application model and your coding model do not have to be the same.

For Claude Code-assisted development, Ollama currently documents local `qwen3-coder` and integration commands for Claude Code. Its local 30B model is much larger than Qwen3-8B and is better viewed as a developer-side option if your machine can accommodate it. [Ollama qwen3-coder](https://ollama.com/library/qwen3-coder)

---

# 6. The central architectural rule: never put everything into RAG

A common mistake in early RAG agents is:

```text
Every user message -> vector search -> LLM
```

Do not build the system this way.

Instead:

```text
                    User utterance
                          │
                          ▼
                  conversation router
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
       Document        General       Conversation
       question        question       continuation
           │              │              │
           ▼              ▼              ▼
          RAG        optional tools   direct LLM
           │              │              │
           └──────────────┴──────────────┘
                          │
                          ▼
                   answer synthesis
```

This solves topic drift naturally.

For example:

- “What was revenue?” → RAG
- “Explain EBITDA” → possibly general knowledge, unless the user says “according to the report”
- “Go back to the report” → restore document context
- “What’s the capital of Japan?” → general answer, no RAG
- “Now compare Japan with the country mentioned in the report” → mixed/contextual reasoning

The router should return a **structured object**, not prose.

Example:

```json
{
  "intent": "document_qa",
  "needs_retrieval": true,
  "document_scope": "current_workspace",
  "topic": "financial_performance",
  "language": "hi",
  "is_topic_shift": false,
  "is_followup": true,
  "confidence": 0.96
}
```

Validate this with Pydantic.

---

# 7. Document ingestion architecture

## Ingestion flow

```text
upload
  ↓
file validation
  ↓
content hash
  ↓
document record
  ↓
Docling conversion
  ↓
structured document model
  ↓
normalization
  ↓
semantic chunking
  ↓
metadata enrichment
  ↓
dense + sparse embeddings
  ↓
Qdrant upsert
  ↓
ingestion QA
  ↓
READY
```

## Why Docling

Docling is especially suitable because it is designed to turn complex documents into structured data, including reading order, tables, formulas and OCR, and it can run locally/offline after model assets are available. Its project page currently states Python 3.10+ and local/offline execution. [Docling](https://docling.ai/), [Docling GitHub](https://github.com/docling-project/docling)

## Files to support first

Phase 1:

- PDF
- DOCX
- PPTX
- TXT
- Markdown

Phase 2:

- XLSX
- HTML
- images / scanned PDFs
- EPUB
- email formats

Docling currently lists broad support across PDF/Office/HTML/images plus additional formats and chart understanding. [Docling GitHub](https://github.com/docling-project/docling)

---

# 8. Document model

Do not store only raw text.

Every chunk should preserve provenance.

Example:

```json
{
  "chunk_id": "doc_01_p47_c03",
  "document_id": "doc_01",
  "document_version": 2,
  "text": "Revenue increased by 34% ...",
  "page_start": 47,
  "page_end": 47,
  "section": "Financial Results",
  "heading_path": ["Annual Report", "Financial Results", "Revenue"],
  "content_type": "paragraph",
  "language": "en",
  "source_uri": "local://documents/annual_report.pdf",
  "hash": "..."
}
```

For tables:

```json
{
  "content_type": "table",
  "table_markdown": "| FY | Revenue | EBITDA |...",
  "page_start": 47,
  "section": "Financial Results"
}
```

Do not flatten every table into plain paragraphs. Preserve the table representation and optionally create a textual summary for semantic retrieval.

---

# 9. Chunking strategy

Do not use only a fixed character count.

Use a hierarchy such as:

```text
Document
  └── Section
       └── Subsection
            └── Paragraph / list / table
                 └── retrievable chunk
```

Recommended initial chunking policy:

- target roughly 300–700 tokens for ordinary prose
- overlap roughly 50–100 tokens when needed
- never split a table arbitrarily
- keep heading path attached to every chunk
- keep page number(s) attached
- create smaller chunks for dense factual material
- create larger parent context for narrative explanations

The right values should be tuned with evaluation data instead of treated as universal constants.

---

# 10. Retrieval architecture

Use **hybrid retrieval + reranking**.

```text
                     Query
                       │
             ┌─────────┴─────────┐
             │                   │
         dense search        sparse search
          BGE-M3              BM25 / sparse
             │                   │
             └─────────┬─────────┘
                       │
                     fusion
                       │
                     top 20
                       │
                    reranker
                       │
                      top 5
                       │
                     Qwen
```

This is not unnecessary complexity. Dense and sparse retrieval fail differently.

- Dense retrieval is good for meaning/paraphrases.
- Sparse retrieval is good for exact identifiers, terminology, numbers, names and strings.
- Reranking improves the ordering of the final candidates.

Qdrant explicitly documents dense+sparse hybrid search, RRF/DBSF fusion and multi-stage reranking workflows. [Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/), [Qdrant hybrid search](https://qdrant.tech/documentation/search/text-search/hybrid-search/), [Qdrant reranking tutorial](https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/)

---

# 11. Embedding model: BGE-M3

BGE-M3 is a strong fit for this project because its current model card describes it as:

- multilingual
- supporting more than 100 working languages
- supporting up to 8192-token inputs
- capable of dense retrieval
- capable of sparse retrieval
- capable of multi-vector/ColBERT-style retrieval
- explicitly recommending hybrid retrieval + reranking for RAG

[BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)

This is particularly valuable for the planned behavior:

```text
English query
    ↓
Hindi document
    ↓
Relevant retrieval
```

and:

```text
Gujarati query
    ↓
English report
    ↓
Relevant retrieval
```

You should still test these cross-language scenarios with your own evaluation set.

---

# 12. RAG context assembly

Do not simply concatenate the five top chunks.

Create a context builder that:

1. removes duplicate/near-duplicate chunks
2. groups chunks from the same section
3. preserves source identifiers
4. includes short neighboring context where helpful
5. enforces a context budget
6. marks each passage with a stable citation ID

Example prompt context:

```text
[S1]
Document: Annual Report 2025
Page: 47
Section: Financial Results > Revenue
Revenue increased by 34%...

[S2]
Document: Annual Report 2025
Page: 48
Section: Financial Results > EBITDA
EBITDA margin increased to...
```

Then instruct the model:

```text
Answer using the supplied sources when the question is about the documents.
Do not invent facts.
Every document-specific factual claim should be supported by one or more source IDs.
If the supplied evidence is insufficient, say so explicitly.
```

The UI can convert `[S1]` into a clickable “Annual Report 2025 — p.47”.

---

# 13. Retrieval confidence and abstention

A reliable RAG agent should be able to say:

> “I don't have enough evidence in the uploaded documents to answer that confidently.”

Do not make the model answer every query.

A retrieval gate can use:

```text
retrieval score
+ reranker score
+ agreement between retrievers
+ source diversity
+ query classification
```

Example policy:

```text
if not enough high-quality evidence:
    answer = abstain / ask clarification
else:
    synthesize answer
```

Thresholds must be tuned experimentally.

---

# 14. Language switching

The voice stack should not assume one fixed language.

Target flow:

```text
microphone
   ↓
VAD
   ↓
Whisper
   ↓
language detection
   ↓
conversation router
   ↓
Qwen
   ↓
response language selection
   ↓
TTS
```

Maintain language as conversation state:

```json
{
  "input_language": "hi",
  "response_language": "hi",
  "document_language": "en",
  "cross_lingual_retrieval": true
}
```

Default behavior:

> Respond in the language of the most recent user utterance unless the user explicitly asks for another language.

Also test code switching:

```text
"Tell me the revenue, but explain the margin wala part in Hindi."
```

Do not treat language selection as a separate model. Keep it as a routing/state concern around the core LLM.

---

# 15. Voice architecture

## STT

Use **faster-whisper**.

The current project describes it as a CTranslate2-based Whisper implementation that can be substantially faster and use less memory than the reference implementation, with additional gains from 8-bit quantization. [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

## VAD

Use **Silero VAD**.

It is a small local component whose repository is MIT licensed and is designed specifically for voice-activity detection. [Silero VAD](https://github.com/snakers4/silero-vad)

## TTS

### First choice: Kokoro

Kokoro-82M is currently listed as Apache 2.0 and is compact enough to be attractive for local use. [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)

### Fallback: Piper

Piper is useful when low CPU usage and many ready-made voices are important. Its project documentation warns that individual voice models may carry their own licenses, so check each voice card/model before redistribution. [Piper](https://github.com/rhasspy/piper)

---

# 16. Use WebRTC for the user audio path

For a genuinely conversational interface, use WebRTC for microphone/audio media rather than building a large custom audio stream over ordinary WebSocket messages.

Use:

```text
WebRTC
  → microphone audio
  → low-latency agent audio

WebSocket
  → transcript events
  → partial/final text
  → citations
  → state changes
  → audio playback state
  → errors
  → telemetry
```

This separation keeps media handling and application events clean.

---

# 17. Barge-in / interruption is a first-class feature

This is critical to making the demo feel like a real conversation.

When the agent is speaking:

```text
TTS playing
     │
     ├── user starts speaking
     │
     ▼
VAD detects speech
     │
     ▼
stop / fade current TTS
     │
     ▼
capture new utterance
     │
     ▼
process user input immediately
```

Do not wait for the old response to finish.

Recommended state machine:

```text
IDLE
  ↓
LISTENING
  ↓
THINKING
  ↓
SPEAKING
  ↓
INTERRUPTED ───────┐
  │                │
  └────→ LISTENING │
                   │
                   └→ ...
```

The UI should show the state clearly.

---

# 18. Conversation state machine

Do not let the LLM alone manage all state.

Use an explicit application state:

```json
{
  "session_id": "...",
  "active_document_ids": ["doc_01"],
  "active_topic": "financial_performance",
  "previous_topic": "company_overview",
  "input_language": "hi",
  "response_language": "hi",
  "mode": "human_agent",
  "a2a_mode": "disabled",
  "speaking": false,
  "retrieval_enabled": true
}
```

Application code owns this state. The LLM can propose changes through structured output, but the server validates and applies them.

This makes behavior deterministic enough to debug.

---

# 19. Topic drift

Track topic with lightweight state rather than trying to summarize the entire conversation after every turn.

Maintain:

```text
current topic
previous topic
document topic
last retrieval topic
last 3–8 user intents
```

Example:

```text
current topic = weather
previous topic = annual report
active document = annual_report.pdf
```

When the user says:

> “Okay, back to the report — what was EBITDA?”

The router should detect:

```json
{
  "is_topic_shift": true,
  "returning_to_previous_topic": true,
  "needs_retrieval": true
}
```

That should restore document QA without requiring the user to repeat the file name.

---

# 20. Conversational memory

Do not put an unlimited transcript into every LLM request.

Maintain three tiers:

### Short-term turn memory

Last few conversational turns.

### Session summary

Compact summary of what matters about the current conversation.

### Long-term/document memory

Document index in Qdrant, not conversational history.

This gives:

```text
Recent turns
     +
Session summary
     +
Retrieved document evidence
     +
Current user query
     ↓
LLM
```

---

# 21. Mixed questions

A strong agent should support questions that cross the boundary between document and general knowledge.

Example:

> “The report says the company had 18% EBITDA margin. Is that considered high for this industry?”

This contains:

- document evidence
- external/general contextual reasoning

The architecture should be able to classify it as:

```text
needs_document_context = true
needs_general_knowledge = true
```

For a strictly local/offline demo, the general-knowledge portion should be framed as model knowledge and clearly distinguished from claims sourced from the document.

Do not pretend the document contains information that it does not contain.

---

# 22. GibberLink / ggwave architecture

## Critical conceptual distinction

GibberLink is **not** the user voice interface.

It is a special A2A communication mode.

The original GibberLink project describes two independent conversational AI agents switching from English speech to the ggwave data-over-sound protocol after confirming the other side is an AI agent. The repository itself is MIT licensed. [GibberLink](https://github.com/PennyroyalTea/gibberlink)

The underlying ggwave library is a tiny FSK-based data-over-sound library with error correction and a documented bandwidth of roughly 8–16 bytes/sec depending on protocol parameters. [ggwave](https://github.com/ggerganov/ggwave)

Therefore:

```text
Human ↔ Agent
    = normal speech / WebRTC / STT / TTS

Agent A ↔ Agent B
    = normal speech initially
    = optional ggwave mode after handshake
```

---

# 23. Safer GibberLink mode: explicit handshake

Do not let the LLM spontaneously switch protocols from a single sentence.

Use a state machine:

```text
NORMAL_A2A
   │
   │ identify other side as AI
   ▼
AI_CONFIRMED
   │
   │ request switch
   ▼
SWITCH_REQUESTED
   │
   │ explicit acknowledgement
   ▼
GIBBERLINK
   │
   │ error / human joins / timeout
   ▼
NORMAL_A2A
```

The model can produce a **proposal** such as:

```json
{
  "action": "request_gibberlink",
  "reason": "other_party_identified_as_ai"
}
```

But only application code can actually transition state after the other party acknowledges.

---

# 24. What should travel through ggwave

Keep payloads compact.

Bad idea:

```text
"Let me explain the entire quarterly report..."
```

Better:

```json
{"t":"fact","id":"F31","v":"17.4%"}
```

or:

```json
{"t":"query","id":"Q81","r":"doc_01","p":47}
```

or compact binary/CBOR-style frames such as:

```text
TYPE | MESSAGE_ID | PAYLOAD | CHECKSUM
```

The receiving agent can look up large content from its own local state instead of sending the entire content through sound.

This is the most important way to make the GibberLink concept technically sensible.

---

# 25. Recommended A2A protocol layer

Create your own application-level envelope on top of ggwave.

Example:

```json
{
  "protocol": "local-a2a-v1",
  "session_id": "abc123",
  "seq": 42,
  "type": "DOCUMENT_FACT_REF",
  "payload": {
    "document_id": "doc01",
    "chunk_id": "doc01_p47_c03"
  },
  "ack": true
}
```

Do not expose arbitrary raw LLM output as the protocol payload.

This gives you:

- sequence numbers
- acknowledgements
- retries
- timeouts
- duplicate detection
- versioning
- telemetry

---

# 26. GibberLink demo topology

You have two good demo modes.

## Mode A — two browser devices

```text
Laptop A
  browser
  Agent A
  speaker + mic

       ⇅ sound

Laptop B
  browser
  Agent B
  speaker + mic
```

This is the most visually convincing demo.

## Mode B — one machine, two logical agents

```text
Agent A process
       │
       │ ggwave encoding
       ▼
virtual/local audio path
       │
       ▼
Agent B process
```

This is easier to automate and test.

Build Mode B first for deterministic testing. Add Mode A for the final “wow” presentation.

---

# 27. Data-over-sound reliability controls

Because sound is not guaranteed, add:

```text
message ID
sequence number
ACK
retry count
CRC/ECC handled by transport where applicable
TTL/timeout
mode switch timeout
fallback to normal speech/data channel
```

Example:

```text
send frame
   ↓
wait ACK
   ├── ACK → next frame
   └── timeout → retry
                   ├── success
                   └── failure → normal channel
```

Never make GibberLink the only path for critical state.

---

# 28. Project folder structure

Recommended monorepo:

```text
local-document-agent/
│
├── README.md
├── LICENSE
├── .env.example
├── docker-compose.yml
├── Makefile
│
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   ├── voice.py
│   │   │   └── health.py
│   │   │
│   │   ├── core/
│   │   │   ├── state.py
│   │   │   ├── events.py
│   │   │   └── errors.py
│   │   │
│   │   ├── db/
│   │   │   ├── models.py
│   │   │   ├── repository.py
│   │   │   └── migrations.py
│   │   │
│   │   ├── ingestion/
│   │   │   ├── loader.py
│   │   │   ├── docling_pipeline.py
│   │   │   ├── normalizer.py
│   │   │   ├── chunker.py
│   │   │   ├── metadata.py
│   │   │   └── ingestion_qa.py
│   │   │
│   │   ├── retrieval/
│   │   │   ├── embeddings.py
│   │   │   ├── sparse.py
│   │   │   ├── qdrant.py
│   │   │   ├── hybrid.py
│   │   │   ├── reranker.py
│   │   │   ├── context_builder.py
│   │   │   └── confidence.py
│   │   │
│   │   ├── agent/
│   │   │   ├── router.py
│   │   │   ├── prompts.py
│   │   │   ├── reasoning.py
│   │   │   ├── memory.py
│   │   │   └── tools.py
│   │   │
│   │   ├── voice/
│   │   │   ├── stt.py
│   │   │   ├── vad.py
│   │   │   ├── tts.py
│   │   │   └── sessions.py
│   │   │
│   │   ├── a2a/
│   │   │   ├── protocol.py
│   │   │   ├── state_machine.py
│   │   │   ├── frames.py
│   │   │   └── ggwave_transport.py
│   │   │
│   │   └── observability/
│   │       ├── logging.py
│   │       ├── metrics.py
│   │       └── tracing.py
│   │
│   └── tests/
│       ├── ingestion/
│       ├── retrieval/
│       ├── agent/
│       ├── voice/
│       └── a2a/
│
├── frontend/
│   ├── package.json
│   ├── app/
│   ├── components/
│   │   ├── DocumentUploader.tsx
│   │   ├── VoiceButton.tsx
│   │   ├── Transcript.tsx
│   │   ├── Citation.tsx
│   │   ├── AgentState.tsx
│   │   └── GibberLinkVisualizer.tsx
│   ├── lib/
│   │   ├── webrtc.ts
│   │   ├── websocket.ts
│   │   └── audio.ts
│   └── tests/
│
├── data/
│   ├── uploads/
│   ├── processed/
│   ├── sqlite/
│   └── qdrant/
│
├── evals/
│   ├── documents/
│   ├── questions.jsonl
│   ├── retrieval_cases.jsonl
│   ├── multilingual_cases.jsonl
│   ├── conversation_cases.jsonl
│   └── expected_answers.jsonl
│
└── scripts/
    ├── bootstrap.sh
    ├── ingest.py
    ├── evaluate_rag.py
    └── smoke_test.py
```

---

# 29. Docker Compose architecture

Use Compose for infrastructure, not for every local model unless necessary.

Example services:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage

  backend:
    build: ./backend
    depends_on:
      - qdrant
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    depends_on:
      - backend
    ports:
      - "3000:3000"
```

Keep Ollama installed on the host initially unless you have a strong reason to containerize the GPU runtime. Host-native GPU access is usually simpler for a local demo.

Qdrant itself supports local persistence and its current README notes write-ahead logging for durable updates. [Qdrant](https://github.com/qdrant/qdrant)

---

# 30. Initial installation sequence

## Prerequisites

Install:

- Git
- Python 3.10+
- Node.js LTS
- Docker Desktop / Docker Engine
- Ollama
- FFmpeg

Then:

```bash
ollama --version
python --version
node --version
docker --version
ffmpeg -version
```

## Pull initial model

```bash
ollama pull qwen3:8b
```

Optional larger model:

```bash
ollama pull gpt-oss:20b
```

## Start Qdrant

```bash
docker compose up -d qdrant
```

## Install backend dependencies

Use a modern Python environment manager if desired (for example `uv`) or a normal virtualenv.

Illustrative stack:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install fastapi uvicorn pydantic pydantic-settings qdrant-client
pip install docling sentence-transformers FlagEmbedding
pip install faster-whisper silero-vad
pip install ggwave
```

Actual versions should be pinned in `pyproject.toml` after the first known-good build rather than relying on floating versions forever.

---

# 31. MVP implementation phases

The order matters.

## Phase 0 — infrastructure smoke test

Goal:

```text
Ollama works
Qdrant works
FastAPI works
React works
```

Acceptance criteria:

- `ollama run qwen3:8b` returns an answer
- Qdrant is reachable on localhost
- FastAPI `/health` returns OK
- frontend opens

---

## Phase 1 — text-only document chat

Build:

```text
upload
  → Docling
  → chunk
  → BGE-M3
  → Qdrant
  → retrieve
  → Qwen
```

Do not add voice yet.

Acceptance tests:

1. Upload PDF.
2. Ask a fact question.
3. Answer contains citations.
4. Ask a question not present in the document.
5. Agent abstains instead of inventing.
6. Upload two documents and distinguish them.

---

## Phase 2 — hybrid retrieval + reranking

Upgrade:

```text
vector search
        ↓
+ sparse/BM25
        ↓
RRF
        ↓
reranker
```

Acceptance criteria:

- exact identifiers work
- paraphrased questions work
- numeric queries work
- page citations remain correct

---

## Phase 3 — conversation router

Add:

```text
intent
topic
follow-up
language
retrieval-needed
```

Acceptance criteria:

- unrelated question does not unnecessarily trigger RAG
- follow-up document question does
- explicit “back to the report” restores document context

---

## Phase 4 — local voice input

Add:

```text
WebRTC mic
  → VAD
  → faster-whisper
  → transcript
```

Acceptance criteria:

- normal conversational microphone input
- multiple speech lengths
- silence handling
- multilingual transcription
- partial/final transcript events

---

## Phase 5 — local voice output

Add:

```text
Qwen response
  → TTS
  → browser audio
```

Acceptance criteria:

- response automatically spoken
- text and audio remain synchronized enough for demo
- agent can be interrupted

---

## Phase 6 — barge-in

Implement:

```text
speaking
  + user speech
  = cancel current TTS
```

Acceptance criterion:

The user can interrupt the agent without waiting for the response to finish.

---

## Phase 7 — topic drift + language switching

Test deliberate drift:

```text
Document → general → joke → Hindi → document → Gujarati → document
```

Acceptance criteria:

- current topic updates correctly
- old document context can be restored
- language follows user intent
- no accidental retrieval on unrelated turns

---

## Phase 8 — A2A local agents

Create:

```text
Agent A
Agent B
```

Initially use ordinary text messages internally.

Then add TTS/STT to make the interaction audible.

Acceptance criterion:

Two agents can hold a short natural-language dialogue without deadlocking.

---

## Phase 9 — GibberLink / ggwave

Add explicit handshake:

```text
AI detected
  → request switch
  → ACK
  → ggwave mode
```

Use tiny structured payloads.

Acceptance criteria:

- mode transition is visible in UI
- payloads decode correctly
- acknowledgements work
- retries work
- timeout returns to normal channel

---

## Phase 10 — demo polish

Add:

- document cards
- source citations
- transcript
- live waveform
- speaking/listening indicator
- current language badge
- current topic badge
- RAG on/off indicator
- GibberLink mode indicator
- decoded message view
- agent A/B visualization

---

# 32. Evaluation: the part that makes the project reliable

Do not rely on manual testing alone.

Create a small evaluation corpus with 50–200 questions before calling the system “good.”

## Retrieval evaluation

Measure:

- Recall@5
- Recall@10
- MRR
- nDCG if desired
- citation hit rate

For every question, store expected relevant chunk IDs.

Example:

```json
{"id":"q001","question":"What was FY24 revenue?","relevant_chunks":["doc01_p47_c03"]}
```

## Generation evaluation

Measure:

- groundedness
- citation correctness
- answer completeness
- abstention quality
- contradiction rate

## Conversation evaluation

Test:

- follow-up references
- topic drift
- topic restoration
- interruptions
- repeated questions
- ambiguous questions
- language switching
- code switching

## Voice evaluation

Test:

- word error rate on your accents/noise
- latency to first transcript
- latency to first audio
- interruption response time
- false VAD triggers

## A2A evaluation

Test:

- handshake success
- decode success
- duplicate packets
- missing packets
- retry success
- timeout fallback

---

# 33. Observability

Add structured logs from day one.

Every request should have:

```text
session_id
turn_id
request_id
```

Log at minimum:

```text
STT latency
router latency
retrieval latency
reranker latency
LLM first-token latency
LLM total latency
TTS latency
source IDs
retrieval scores
mode transitions
language transitions
interruptions
errors
```

Do not log raw document content unnecessarily.

Useful debug event:

```json
{
  "turn_id": "t42",
  "intent": "document_qa",
  "language": "hi",
  "retrieval": {
    "dense_top": "doc01_p47_c03",
    "sparse_top": "doc01_p47_c03",
    "reranked": ["doc01_p47_c03", "doc01_p48_c01"]
  },
  "answer_sources": ["doc01_p47_c03"]
}
```

---

# 34. Failure handling

Every dependency should have a fallback.

### LLM unavailable

Show:

> Local language model is unavailable.

Do not return a fake answer.

### Qdrant unavailable

Allow normal conversation, but document-specific mode should explicitly fail or disable retrieval.

### Docling fails

Keep the upload record as `FAILED` with the error.

### Whisper fails

Keep text chat available.

### TTS fails

Keep text response visible.

### GibberLink fails

Fall back to ordinary A2A communication.

The system should degrade gracefully:

```text
full multimodal
   ↓ failure
text + RAG
   ↓ failure
text chat
```

---

# 35. Document lifecycle and idempotency

A robust ingestion system should detect duplicate files.

Compute a SHA-256 hash:

```text
file bytes
  ↓
SHA-256
  ↓
document content hash
```

If the exact same file is uploaded again:

```text
same hash
→ reuse document version
```

If the content changes:

```text
different hash
→ new document version
```

Store:

```text
document
version
hash
ingestion_status
created_at
updated_at
chunk_count
error_message
```

---

# 36. Security even for a local demo

Local does not automatically mean safe.

At minimum:

- restrict allowed file extensions
- enforce max file size
- store uploads outside the application code tree
- never execute uploaded content
- sanitize filenames
- reject unsupported MIME types
- limit decompression bombs / oversized archives
- keep temporary files isolated
- avoid exposing Qdrant publicly
- bind infrastructure to localhost where possible

If the application ever becomes remotely accessible, revisit this section completely.

---

# 37. Prompting strategy

Keep prompts modular.

## System prompt

Defines identity, safety, language behavior and grounding rules.

## Router prompt

Outputs strict JSON for intent/state decisions.

## RAG answer prompt

Receives retrieved evidence and citations.

## General chat prompt

Normal conversational response without document context.

## A2A agent prompt

Defines agent role, protocol awareness and mode transition proposal.

Do not make one giant prompt that tries to do everything.

---

# 38. Structured outputs

For router/tool decisions, require structured output.

Example Pydantic schema:

```python
class TurnRoute(BaseModel):
    intent: Literal[
        "document_qa",
        "general_qa",
        "conversation",
        "topic_shift",
        "resume_document",
        "clarification",
    ]
    needs_retrieval: bool
    topic: str
    language: str
    confidence: float = Field(ge=0, le=1)
```

Never parse critical control decisions from free-form prose such as:

```text
"I think maybe we should switch modes..."
```

---

# 39. Do not add LangChain on day one

LangChain/LlamaIndex can be useful later, but this project benefits from explicit control early on.

A direct architecture is easier to understand:

```text
FastAPI
  → router
  → retrieval
  → Qwen
  → voice
```

Once the pipeline is stable, decide whether an orchestration framework actually reduces code.

The most important thing is that the project has clear interfaces:

```python
DocumentStore
Retriever
Reranker
LLMClient
SpeechRecognizer
SpeechSynthesizer
ConversationStateStore
A2ATransport
```

Then a framework can be added without rewriting the core architecture.

---

# 40. Suggested Python interfaces

```python
class DocumentIngestor(Protocol):
    async def ingest(self, path: Path) -> IngestionResult: ...

class Retriever(Protocol):
    async def search(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int = 20,
    ) -> list[RetrievedChunk]: ...

class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]: ...

class LLMClient(Protocol):
    async def generate(self, messages: list[dict], **kwargs) -> str: ...

class SpeechRecognizer(Protocol):
    async def transcribe(self, audio: bytes) -> Transcript: ...

class SpeechSynthesizer(Protocol):
    async def synthesize(self, text: str, language: str) -> bytes: ...

class A2ATransport(Protocol):
    async def send(self, message: A2AMessage) -> None: ...
    async def receive(self) -> AsyncIterator[A2AMessage]: ...
```

This separation is one of the best things you can give Claude Code because it prevents the generated implementation from becoming an untestable monolith.

---

# 41. Suggested API surface

```text
GET    /health
GET    /models

POST   /documents/upload
GET    /documents
GET    /documents/{id}
DELETE /documents/{id}

POST   /chat/turn
POST   /chat/route
GET    /chat/{session_id}

WS     /ws/session/{session_id}

POST   /voice/session
POST   /voice/interrupt

POST   /a2a/session
POST   /a2a/handshake
POST   /a2a/gibberlink/enable
POST   /a2a/gibberlink/disable
```

The browser should not need to know how Qdrant or Docling work.

---

# 42. Example end-to-end request

User says:

> “What was EBITDA margin in FY24?”

System executes:

```text
1. Receive audio
2. VAD segments speech
3. Whisper transcribes
4. Detect language = English
5. Router = document_qa
6. Rewrite query if needed
7. Dense retrieval in Qdrant
8. Sparse retrieval in Qdrant
9. Fuse candidates
10. Rerank top 20
11. Build context from top 5
12. Qwen synthesizes answer
13. Validate citations
14. Stream text to browser
15. TTS response
16. Play audio
17. Store concise turn summary
```

---

# 43. Query rewriting

Add query rewriting only after baseline retrieval works.

Example:

```text
User:
"And what about margin?"

Conversation context:
"We were discussing EBITDA in FY24"

Rewritten retrieval query:
"FY24 EBITDA margin"
```

This is particularly important for voice conversations because people frequently use short follow-ups.

Do not rewrite everything. Only rewrite when the router marks the question as a contextual follow-up.

---

# 44. Parent-child retrieval

A useful later enhancement:

```text
small child chunk
     ↓
retrieval
     ↓
parent section context
```

Example:

```text
retrieved chunk = a 250-token fact
parent = full 900-token subsection
```

This can preserve precise retrieval while giving the LLM enough context to understand the fact.

---

# 45. Tables need special treatment

Many document QA failures come from tables.

Build table-aware indexing:

```text
table
 ├── raw structured table
 ├── markdown representation
 ├── text summary
 └── row/column metadata
```

For a question such as:

> “What was revenue in Germany in Q4?”

the retrieval layer should be able to locate the correct table and page rather than depending on prose surrounding it.

---

# 46. Numeric and exact-match robustness

Document QA frequently involves:

- percentages
- dates
- account numbers
- product codes
- names
- legal clauses
- years
- SKU-like identifiers

This is another reason not to use dense retrieval alone.

Use hybrid retrieval and preserve numbers exactly.

Also instruct the generation layer:

> Do not round numbers unless requested.

---

# 47. Citation verification

For high-quality answers, do not assume the model's citations are valid.

If the model says:

```text
Revenue was $4.2B [S1]
```

the backend should verify:

```text
S1 exists
S1 belongs to current document scope
S1 supports the claim
```

For the demo, a lightweight validation can ensure citation IDs exist.

A later version can add a local entailment/verification model.

---

# 48. Performance targets for a convincing local demo

These are engineering targets, not guarantees.

Aim for:

```text
Mic → first transcript:      < 1 sec
User stop → final transcript: ~0.5–1.5 sec
Query → first LLM token:      < 2–3 sec
Query → first TTS audio:      < 3–5 sec
Interrupt → TTS stop:         near-immediate
```

Exact latency depends heavily on CPU/GPU/model size and document retrieval settings.

Measure it; do not promise it.

---

# 49. Hardware strategy

The final model choice should be driven by your actual machine.

### Lower-end machine

```text
Qwen3 8B quantized
small/medium Whisper
BGE-M3 CPU/GPU as practical
Piper or Kokoro
Qdrant
```

### Mid-range GPU machine

```text
Qwen3 8B/14B
faster-whisper accelerated
BGE-M3 accelerated
local reranker
Kokoro
```

### High-memory machine

```text
Qwen3 14B+ or gpt-oss:20b
larger Whisper
BGE-M3 + reranker
higher-quality TTS
```

Do not select the biggest model available. For a voice agent, a smaller model with fast first-token latency often creates a better experience than a huge model that takes several seconds to respond.

---

# 50. Claude Code implementation strategy

Give Claude Code small, testable tasks instead of one huge prompt.

## Prompt pattern

Use:

```text
You are implementing [component].

Constraints:
- local only
- no external inference APIs
- Python + FastAPI
- typed interfaces
- unit tests required
- no hidden global state
- errors must be explicit

Deliver:
1. implementation
2. tests
3. README update
4. run instructions
```

### Recommended order

```text
1. project skeleton
2. config + health
3. SQLite models
4. Docling ingestion
5. chunk model
6. BGE-M3 embeddings
7. Qdrant storage
8. hybrid retrieval
9. reranker
10. RAG answer generation
11. router
12. session state
13. WebRTC
14. Whisper
15. VAD
16. TTS
17. barge-in
18. A2A
19. ggwave
20. evaluation harness
21. UI polish
```

Each phase should be green before moving on.

---

# 51. Claude Code testing requirements

Make Claude Code write tests for:

### Ingestion

- corrupt PDF
- duplicate document
- scanned PDF
- table extraction
- multiple pages
- Unicode text

### Retrieval

- exact identifier
- paraphrase
- multilingual query
- no-match query
- filter by document

### Agent

- document question
- general question
- topic drift
- topic restoration
- language switch
- follow-up pronoun/reference

### Voice

- empty audio
- noisy audio
- interrupted speech
- silence
- long utterance

### A2A

- handshake
- denial
- timeout
- retry
- duplicate message
- invalid payload
- version mismatch

---

# 52. Suggested UI

The home screen should feel like a real conversational workspace.

```text
┌──────────────────────────────────────────────────────────────┐
│ Local Document Agent                                         │
├───────────────────┬──────────────────────────────────────────┤
│ Documents         │ Conversation                             │
│                   │                                          │
│ ☑ Annual Report   │ User: What was FY24 revenue?             │
│ ☑ Contract.pdf    │                                          │
│                   │ Agent: FY24 revenue was...               │
│ + Upload          │                                          │
│                   │ User: Actually, explain EBITDA in Hindi │
│                   │                                          │
│                   │ Agent: ज़रूर...                           │
├───────────────────┴──────────────────────────────────────────┤
│  English  •  Document QA  •  Listening / Speaking           │
│                                                              │
│                    🎙 Talk                                   │
└──────────────────────────────────────────────────────────────┘
```

For the A2A demo, switch to:

```text
Agent A  ────────────── Agent B
   │                       │
 normal speech        normal speech
   │                       │
   └────── handshake ──────┘
            ↓
     GIBBERLINK MODE
            ↓
       sound / waveform
```

---

# 53. GibberLink visual effect

For the demo, show:

```text
NORMAL SPEECH
████████████████████

      ↓

AI HANDSHAKE
✓ AI identified
✓ protocol accepted

      ↓

GIBBERLINK
▰▰▰ ▰▰ ▰▰▰ ▰

Decoded message:
{"type":"FACT_REF","id":"F31"}
```

This makes the protocol transition obvious to a non-technical audience.

---

# 54. Interesting demo scenarios

## Scenario A — finance

Upload annual report.

Ask:

```text
Revenue?
EBITDA?
Gross margin?
What happened in FY24?
Compare two years.
```

Then drift:

```text
What does EBITDA mean?
Tell me a joke.
Switch to Hindi.
Back to the report.
```

## Scenario B — legal

Upload contract.

Ask:

```text
Termination clause?
Notice period?
Who carries liability?
Summarize in plain English.
Now explain the same thing in Hindi.
```

## Scenario C — multi-document

Upload:

```text
annual_report.pdf
investor_presentation.pdf
press_release.pdf
```

Ask:

> “Do the investor presentation and annual report agree about revenue?”

This shows cross-document retrieval.

## Scenario D — A2A

Agent A receives a question.
Agent B has the document.
They negotiate in natural speech and switch to ggwave.

---

# 55. Demo script for the final presentation

A polished 3–5 minute demo can be:

### Part 1 — ingest

Upload a PDF.

Show:

```text
Processing...
→ parsed
→ 132 chunks
→ 132 dense vectors
→ 132 sparse vectors
→ ready
```

### Part 2 — normal voice QA

Ask a document question.

Agent answers with page citation.

### Part 3 — interrupt

Start speaking while the agent is responding.

Agent stops immediately.

### Part 4 — drift

Ask an unrelated question.

Agent does not retrieve the document unnecessarily.

### Part 5 — language switch

Switch to Hindi or Gujarati.

Agent answers in that language.

### Part 6 — return to document

Say:

> “Back to the report — what was EBITDA margin?”

Correct document context returns.

### Part 7 — GibberLink

Introduce two local agents.

They confirm they are both AI agents.

Switch to GibberLink.

Show decoded structured frames.

### Part 8 — human interruption

Human speaks again.

Agents return to normal human-facing interaction.

---

# 56. What not to do

Avoid these early mistakes:

### Do not use a cloud LLM for hidden fallback

Otherwise the project is no longer truly local and private.

### Do not route every query to RAG

This hurts conversational quality.

### Do not put the whole transcript into the prompt forever

Latency and context usage will grow unnecessarily.

### Do not use only embeddings

Exact identifiers and numeric terms need lexical retrieval too.

### Do not make ggwave carry long natural-language messages

Use compact references/events.

### Do not make the model directly control transport state

Application code should own protocol transitions.

### Do not skip evaluation

A demo can look great while failing many realistic questions.

### Do not optimize model size before measuring latency

Measure end-to-end latency first.

---

# 57. Future enhancements

After the local demo works, interesting upgrades include:

- OCR fallback for difficult scans
- document image understanding
- chart/graph extraction
- page-image grounding
- semantic table retrieval
- query decomposition
- multi-hop document reasoning
- local citation verifier
- speaker diarization
- wake-word support
- personal voice profiles where licensing permits
- local speech emotion/prosody controls
- conversation bookmarks
- “pin this fact” functionality
- agent-to-agent task delegation
- encrypted A2A payloads
- signed A2A messages
- replayable conversation traces
- benchmark dashboards
- local evaluation viewer
- model routing based on latency/complexity

---

# 58. Advanced model routing

Later, you can use two local models without hard-wiring the whole application to them.

Example:

```text
simple conversational turn
    ↓
fast small model

complex document synthesis
    ↓
larger reasoning model

router / classification
    ↓
small model
```

But only add multi-model routing after the single-model pipeline is reliable.

A very good design is:

```text
LLMProvider interface
    ├── OllamaQwenProvider
    ├── OllamaGptOssProvider
    └── MockLLMProvider
```

Then you can benchmark them with the same evaluation set.

---

# 59. Model configuration philosophy

For voice conversation:

```text
temperature: relatively controlled
max tokens: limited
streaming: enabled
```

For router:

```text
temperature: near deterministic
structured output: required
```

For creative conversation:

```text
slightly higher temperature
```

For document QA:

```text
controlled generation
strict grounding
source IDs
abstention allowed
```

Exact values should be benchmarked rather than copied blindly.

---

# 60. State/event model

Use event-driven thinking internally.

Example events:

```text
DOCUMENT_UPLOADED
INGESTION_STARTED
INGESTION_PROGRESS
INGESTION_COMPLETED
USER_SPEECH_STARTED
USER_SPEECH_ENDED
TRANSCRIPT_PARTIAL
TRANSCRIPT_FINAL
ROUTE_DECIDED
RETRIEVAL_STARTED
RETRIEVAL_COMPLETED
LLM_STARTED
LLM_TOKEN
LLM_COMPLETED
TTS_STARTED
TTS_COMPLETED
INTERRUPTION_DETECTED
TOPIC_CHANGED
LANGUAGE_CHANGED
A2A_HANDSHAKE_STARTED
A2A_HANDSHAKE_ACCEPTED
GIBBERLINK_ENABLED
GIBBERLINK_FRAME_SENT
GIBBERLINK_FRAME_RECEIVED
GIBBERLINK_FRAME_ACKED
GIBBERLINK_FALLBACK
```

This gives the frontend everything needed to build a polished realtime experience.

---

# 61. Recommended minimum database tables

SQLite:

```text
users                 optional for local demo
workspaces

 documents
 document_versions
 ingestion_jobs
 chunks                optional if Qdrant is source of retrieval truth

 sessions
 turns
 session_summaries

 a2a_sessions
 a2a_messages
```

The local demo can simplify this to:

```text
documents
ingestion_jobs
sessions
turns
```

with Qdrant storing the chunk payloads.

---

# 62. Qdrant collection design

A simple initial point payload:

```json
{
  "document_id": "doc01",
  "version": 1,
  "chunk_id": "doc01_p47_c03",
  "page_start": 47,
  "page_end": 47,
  "section": "Financial Results",
  "language": "en",
  "content_type": "paragraph",
  "text": "..."
}
```

Vectors:

```text
dense
sparse
```

Optionally later:

```text
multi / ColBERT-style representation
```

Qdrant supports multiple named vectors and multi-stage queries, including dense+sparse fusion and later reranking. [Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/)

---

# 63. Why Qdrant is preferable here over a minimal vector store

For a demo that you eventually want to evolve, Qdrant gives you:

- durable local storage
- metadata filters
- dense vectors
- sparse vectors
- hybrid search
- multivectors
- reranking patterns
- a clean REST/API model
- easy local Docker deployment

The Qdrant project is Apache 2.0 licensed. [Qdrant GitHub](https://github.com/qdrant/qdrant)

---

# 64. Why BGE-M3 is preferable over a tiny English embedding model

Your feature requirements explicitly include language switching.

A multilingual embedding model is therefore a better default than a small English-only encoder.

BGE-M3's current model card explicitly lists support for more than 100 languages and supports dense/sparse/multi-vector modes. [BGE-M3](https://huggingface.co/BAAI/bge-m3)

That gives the retrieval layer a strong foundation for:

```text
English → English
English → Hindi
Hindi → English
Gujarati → English
Spanish → English
```

Still, build evaluation questions in every language you care about.

---

# 65. Model/artifact versioning

Create a file such as:

```yaml
models:
  llm:
    provider: ollama
    name: qwen3:8b
  embeddings:
    name: BAAI/bge-m3
  reranker:
    name: <chosen-local-reranker>
  stt:
    name: <chosen-whisper-model>
  tts:
    name: hexgrad/Kokoro-82M
```

When a model changes, retrieval embeddings may become incompatible.

Therefore store:

```text
embedding_model_name
embedding_model_version
embedding_dimension
chunking_version
```

with the document collection.

A model migration should be explicit.

---

# 66. Reproducibility

Pin:

- Python version
- Node version
- package versions
- Docker image tags
- model names
- model hashes where practical
- chunking configuration
- prompt versions

Add:

```text
config/versions.yaml
```

so the exact demo can be rebuilt.

---

# 67. Sources and current verification

The following public project/model sources were checked while preparing this blueprint.

### Ollama

- [Ollama transparent pricing / local-vs-cloud information](https://ollama.com/blog/transparent-pricing)
- [Ollama cloud models](https://registry.ollama.com/blog/cloud-models)
- [Ollama qwen3-coder](https://ollama.com/library/qwen3-coder)
- [Ollama gpt-oss](https://ollama.com/library/gpt-oss)

Important distinction: using Ollama with a truly local model is different from using an Ollama-hosted cloud model. Cloud models require hosted inference and should not be used in the strict local/offline path.

### Document ingestion

- [Docling](https://docling.ai/)
- [Docling GitHub](https://github.com/docling-project/docling)

Docling currently advertises local/offline usage, MIT-licensed code, and broad document-format support.

### Retrieval

- [Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Qdrant hybrid search](https://qdrant.tech/documentation/search/text-search/hybrid-search/)
- [Qdrant hybrid search with reranking](https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/)
- [Qdrant GitHub](https://github.com/qdrant/qdrant)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)

### LLMs

- [Qwen3 release](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

### Voice

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [Piper](https://github.com/rhasspy/piper)

### GibberLink / ggwave

- [GibberLink](https://github.com/PennyroyalTea/gibberlink)
- [ggwave](https://github.com/ggerganov/ggwave)

The current ggwave README describes a small FSK-based data-over-sound protocol with error correction and roughly 8–16 bytes/sec depending on protocol settings. That is why the recommended architecture uses compact machine payloads rather than long natural-language text over sound.

---

# 68. Final recommended architecture

```text
                           USER
                            │
                         microphone
                            │
                          WebRTC
                            │
                            ▼
                         VAD/STT
                            │
                            ▼
                    Conversation Router
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        Document QA     General chat    Conversation
             │              │              │
             ▼              │              │
       query rewrite        │              │
             │              │              │
             ▼              │              │
      BGE-M3 dense          │              │
      + sparse/BM25         │              │
             │              │              │
             ▼              │              │
           RRF              │              │
             │              │              │
             ▼              │              │
         reranker           │              │
             │              │              │
             └──────────────┴──────────────┘
                            │
                            ▼
                           Qwen
                            │
               ┌────────────┴────────────┐
               │                         │
             text                       state
               │                         │
               ▼                         ▼
              TTS                 topic/language/
               │                  retrieval state
               ▼
             WebRTC
               │
               ▼
             SPEAKER


       OPTIONAL A2A DEMO

             Agent A
                │
        normal speech first
                │
        AI identification
                │
       explicit handshake
                │
        ggwave/GibberLink
                │
          compact frames
                │
             Agent B
```

---

# 69. Final recommendation

The **best version of this project is not a PDF chatbot with a flashy audio effect**.

Build it as a small local conversational platform with clean interfaces between:

```text
Document Intelligence
        ↓
Retrieval
        ↓
Conversation Intelligence
        ↓
Voice
        ↓
Optional A2A Protocol
```

The most important design choices are:

1. **Qwen3 as the default local conversation model.**
2. **Docling for structured ingestion.**
3. **Qdrant + BGE-M3 hybrid retrieval.**
4. **Local reranking.**
5. **WebRTC + VAD + faster-whisper + Kokoro/Piper for voice.**
6. **Explicit conversation state for topic drift and language switching.**
7. **Abstention + citations for trustworthy document answers.**
8. **GibberLink/ggwave only as an optional A2A data-over-sound mode.**
9. **Compact structured A2A frames, not raw long-form speech encoded through ggwave.**
10. **Evaluation and observability from the start.**

That architecture gives you a credible local demo now and a path toward a serious product later.

---

# 70. Immediate next step

Before selecting the exact model variants and writing the first Claude Code prompt, capture the machine profile:

```text
OS:
CPU:
RAM:
GPU:
GPU VRAM:
Disk free space:
```

Use that information to choose the exact LLM, Whisper size, embedding execution mode, reranker, TTS model and whether the system should run everything concurrently or time-slice larger models.
