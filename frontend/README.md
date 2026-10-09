# Frontend

Next.js (App Router) + TypeScript. Voice conversation is the product (docs/DESIGN.md §1): the chat page opens in **voice mode**, with the voice presence field (§3.8), a large mic control, live captions and the answer's sources; the transcript is a panel beside it and typing is a fallback ("Type instead"). Around it: projects, chats and transcripts (§3.9): a sidebar with search, pinned items and each project's chats; project and chat pages; a paginated transcript view; document upload with live ingestion status; text chat with streamed, cited answers; and the system-status panel at `/status`.

## Run

Needs the backend on `http://localhost:8000` ([`../backend/README.md`](../backend/README.md)).

```bash
npm install
```

```bash
npm run dev
```

Open http://localhost:3000. The dev server binds to `127.0.0.1` only.

`npm run dev` and `npm run build` first run `scripts/copy-vad-assets.mjs`, which copies the browser VAD (Silero v5 from `@ricky0123/vad-web`) and the ONNX runtime it needs from `node_modules` into `public/vad/` (git-ignored). The app serves them itself; nothing is loaded from a CDN. Voice needs a secure page (https or localhost) and a browser with `getUserMedia` and AudioWorklet. **Container images** must also copy `public/` next to `server.js` (standalone output doesn't), or the VAD files 404 and interruptions then rely on the server alone.

## Configuration

One environment variable, read **at request time** (not baked in at build time), so the same build works against any backend:

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Where the browser reaches the backend |

The root layout reads it on the server for every request (`lib/backend-url.ts`, `dynamic = "force-dynamic"`) and passes it to a client context (`lib/backend-context.tsx`); every API call in the browser uses that value. Nothing uses `NEXT_PUBLIC_*`.

Everything else (title, languages, feature flags, upload limits, sign-in settings) comes from the backend's `GET /api/config/public`. The browser's origin must be listed in the backend config's `server.cors_allowed_origins`.

## Screens

| Route | What it shows |
|---|---|
| `/` | **Start a conversation** (a new chat in the project you chose or used last, opened in voice mode), recent chats to revisit, recent projects, or an invitation to create the first project |
| `/projects/[projectId]` | Rename, pin, archive, delete; the project's chats (archived ones folded away); its documents: upload, ingestion status, delete |
| `/chats/[chatId]` | Project › chat breadcrumb, rename, pin, archive, delete; the documents the chat answers from (toggle chips to narrow its scope); the conversation in voice mode (below), with the transcript as a panel (opens at the latest messages, loads earlier ones as you scroll up) |
| `/status` | Health of every backend component, and what this frontend is connected to |

The sidebar (a drawer below 820 px) has New project, search (⌘K / Ctrl+K; filters project names and chat titles), Pinned, Projects (each expands to its chats, newest activity first; each project's menu can upload documents) and a health indicator that opens `/status`.

Every change goes to the backend first; the entity it returns is applied to the shared store at once and the affected lists are fetched again, so the sidebar and pages always show what the server has, in its order.

### Documents

Drop files anywhere on the Documents card or choose them (several at once). Each file is checked in the browser against `limits` from `/api/config/public` (extension, size, not empty) before it is sent, then uploaded with progress, two at a time (`components/Uploads.tsx`; uploads keep going when you navigate away). The backend's answer decides what happens next: `202` adds the document (Queued → Processing → Ready with pages and chunks, or Failed with the reason), `200` means the project already has that exact file ("already uploaded"), `422` shows the backend's reason. While any loaded project has a document still queued or processing, its list is fetched every 2 s (`lib/workspace.tsx`); polling stops when everything is Ready or Failed. Deleting asks first, and says the document leaves search and every chat's scope.

### Chat

`POST /api/chats/{id}/messages` answers with Server-Sent Events, which `EventSource` can't request (it only GETs), so `lib/sse.ts` reads the `fetch` response body itself: UTF-8 decoded in streaming mode (a Hindi character split across network packets stays intact), LF/CRLF/CR line endings, comments ignored, one event per blank line. `lib/api.ts` turns the events into typed `user_message` → `sources` → `delta`… → `agent_message` (or `error`), and `lib/chat-turns.ts` keeps one turn at a time:

- the question shows at once (dimmed until the backend has saved it), then "Searching your documents…" until `sources`, then the answer streams with a caret (deltas applied once per animation frame) and is replaced by the saved message;
- **Stop** (or Esc) aborts the request: what arrived stays, marked stopped;
- an `error` event, a refused request or a stream that closes early shows on that turn with **Retry**, which asks the same text again (as a new message if the first one was saved, otherwise in place);
- answers the documents don't support (`abstained`) are shown apart, calmly, as "Not in your documents".

`[S1]` markers in answers become numbered chips; hovering, focusing or clicking one opens a popover with the document, pages and the cited passage (`components/Citations.tsx`; a passage that is a markdown table is shown as a compact table that scrolls sideways, read by `parseTableSnippet` in `lib/citations.ts` from the one-line snippet the backend stores), and each answer lists the sources it cites with the same numbers. The transcript follows a streaming answer while you're at the bottom, and stays put (with "Jump to latest") once you scroll up. Without a READY document in the chat's scope the composer says why and links to the project's documents instead of sending.

### Voice

The chat page (`components/voice/VoiceChat.tsx`) is the voice view. The stage has the **presence field** (`PresenceField.tsx`; the §3.8 particle field ported from `docs/prototypes/voice-presence.html`: one WebGL2 draw call of ~10,000 instanced strokes, a Canvas 2D fallback, `prefers-reduced-motion` and light/dark respected), **live captions** (`Captions.tsx`), the current answer's **source chips** (the same citation popover as the transcript), a polite `aria-live` **state label** and the mic controls: a large mic button (start and end the conversation), a stop button for the answer in progress, and "Type instead" (expands the text composer; typed turns stream into the transcript as before). A **new chat starts listening at once** (the click that created it allows audio, `lib/voice/autostart.ts`); reopening a chat does not, and the transcript panel is open by default when it has messages. Blocked, missing or unsupported microphones say so and offer Try again or typing.

Keys: Space or Enter on the mic button starts and ends the conversation; **Esc steps back one level**: it closes the transcript when that covers the stage (when the chat area is narrower than 880 px), else stops the answer in progress, else ends the conversation.

The audio engine is `lib/voice/` (protocol in docs/DESIGN.md §3.10, `WS /ws/chats/{id}/voice` on the backend origin):

- `session.ts` runs a session as a small external store: mic, socket, player and VAD; unexpected closes reconnect with backoff, ten attempts over about a minute (the mic and VAD keep running; the new server session starts its turn numbers over, the page forgets the dropped one's unfinished answer and fetches the transcript tail again), close 4403 (the socket's Origin isn't allowed: open the app from the address it is set up for), 4404 and 4409 end it with an explanation, release the mic and audio and don't reconnect, `error` messages show a notice and the session keeps listening; everything is torn down on navigation, `pagehide`, or when the backend turns `features.voice` off. A start that is stopped while the permission prompt is open (Esc, then the mic again) only ever releases its own microphone and audio context, never the newer start's.
- `capture.ts`: `getUserMedia` with echo cancellation, noise suppression and auto gain; an AudioWorklet downsamples to 16 kHz with a 32-tap windowed-sinc low-pass (cutoff 7 kHz, so sibilants above 8 kHz don't fold back into the speech band: -60 dB or better from 10 kHz up at 48 and 44.1 kHz) and sends 512-sample PCM16 frames once the server says `ready`; an AnalyserNode on the mic feeds the user's waves.
- `playback.ts`: 24 kHz frames are scheduled back to back on the AudioContext clock through a per-turn gain and a master gain (the duck, 0.2) into an AnalyserNode (the agent's waves). Only the current turn's frames play; others are dropped. The same schedule gives `played_ms` for `barge_in_start` and `playback` (output and base latency subtracted), and `chunkFraction` for captions. `playback_done` waits until every announced chunk has delivered its length in frames and the audio has played (after a 1.5 s grace if frames never complete).
- `vad.ts`: Silero v5 `MicVAD` on the shared mic stream. Speech onset while the agent speaks ducks the volume at once and sends `barge_in_start`; the server's `barge_in` decides (`stop` flushes that turn and freezes the caption, `resume`, or a VAD misfire, restores the volume; if the server never answers the volume returns after 3.5 s).
- `captions.ts`: each `audio_chunk` announces a clause's text and length; a word lights when the audio playing has reached its share of the chunk (by letters, plus a pause after punctuation), so captions follow what is actually heard.

**What the client relies on, and what it tolerates.** The backend guarantees (docs/DESIGN.md §3.10): `agent_message` is sent after the turn's last binary frame; after `barge_in: stop` or `stop` no further `delta`, `audio_chunk` or frames come for that turn, and its cut `agent_message` arrives before the next turn's `user_message` / `turn`; `turn_id` strictly increases within a session and a reconnect starts fresh; within a turn the order is `user_message`, `turn`, `sources`, then `delta` / `audio_chunk`, then `agent_message`. The client still doesn't trust them: it remembers the highest turn id and every cut turn, so a late `delta`, `audio_chunk`, frame or `sources` of a stopped or older turn is dropped and can't become the current turn (and its saved message is told apart from the new question's by `seq`); a `stop` pressed before the answer has an id (the question is still being transcribed or the turn id hasn't arrived) cancels the turn that arrives, until `agent_message`, `state: listening` or new user speech; `turn` arriving before `user_message` is accepted; words the user said that never became a message (a backchannel, noise, a failed transcription) leave the captions on `barge_in: resume`, `state: listening` or an `error`, and the speculative transcript of an ignored backchannel that arrives after `resume` is dropped. Found against the real backend: when a complete answer is cut during playback the server sends `agent_message` again with the same id and `heard_text` set, which updates the transcript entry in place (never twice); a `stop` that finds no answer in progress (the question is still being transcribed) is dropped by the server, so the client repeats it as soon as the turn id is known; and audio that starts playing while the user is already talking (they spoke up while the answer was being written) ducks and sends `barge_in_start` like any other interruption. From the backend's review fixes: `barge_in: stop` is accepted with or without a `barge_in_start` before it (it cuts that turn, and the user's words stay in the captions), and it is idempotent: after the user's own Stop the server also decides the pending barge-in with a `stop` for the same turn, which finds the turn already cut and does nothing, and a late one for an older turn never touches the newer turn's pending duck or timer. When the server ignores an utterance (it closes every `user_speech: start` with `end` and re-sends `state`, plus `barge_in: resume` if it treated it as a barge-in), the words and the pending "You" bubble go: on `state: listening`, or on the agent's own `speaking` / `thinking` while an answer is in progress; a real question keeps its words through `thinking`, because no answer of the agent's is in flight, and a real interruption keeps them because the turn is already cut. A saved answer says how it was cut in `route.interrupted` (`barge_in`, `stop` or `disconnect`): the transcript reads "Interrupted after: ...", "Stopped after: ..." or "Cut off when the connection dropped, after: ..." (and the same without a quote when nothing had been played), never "interrupted by you" for a cut the user didn't make; older messages without a reason read "Interrupted after: ...".

In development `window.__voice` is the live session (`getSnapshot()`, `stats`, `player`) and `window.__presence` the presence meters, for debugging and browser tests.

### Revisiting a chat: titles, summary, export

The transcript panel has two tabs, **Transcript | Summary** (`components/Summary.tsx`; the plain transcript layout, with voice off, has the same tabs above the transcript). The chat's **⋯** menu (header, sidebar, project page) has **Summarise**, **Regenerate title** and **Export transcript → Markdown / JSON**; the last three are greyed out for a chat with no messages.

- **Automatic titles.** A new chat is called "New chat" until the backend writes a title about a second after the first answer is saved. While the chat still has that placeholder (and `title_is_auto`), the page re-reads it 0.8, 1.8, 3.2 and 5 s after each saved answer (a voice `agent_message` or a finished typed turn; `lib/auto-title.ts`) and stops when it changes. The chat is refetched into the shared store without a loading state, and the header and sidebar fade the new text in (`components/TitleText.tsx`). **Regenerate title** calls `POST /api/chats/{id}/title:regenerate`; a `409` (the user chose the title) asks "Replace your title?" and retries with `?force=true`; `422` and `503` say what happened in a toast.
- **Summary.** `GET /api/chats/{id}/summary` when the Summary tab is first shown (a `404` "summary of chat … not found" is the normal "no summary yet", not an error), `POST …/summary[?language=en|hi]` to write or refresh one. It is slow on the local model, so the view says so, keeps working while you look at the transcript (the state lives in `lib/use-chat-summary.ts`, in the chat page), and a failed write leaves the summary that was there. It is rendered from the structured fields (overview, key points with source chips in the citation popover style, unanswered questions, follow-ups), never from the markdown `content`, and is never spoken. **Out of date: covers N of M messages** with a Refresh button shows when the backend says `stale` or the chat got more messages since the page fetched the summary; the **EN | हिंदी** switch writes the other language. A question the documents couldn't answer scrolls the transcript to the message that asked it (`message_seq`), loading earlier pages if needed, and highlights it.
- **Export.** `GET /api/chats/{id}/export?format=md|json` is fetched, turned into a blob and saved under the file name the server sends (`filename*=UTF-8''…` for Hindi titles; `lib/download.ts`). A browser can only read that header across origins if the backend lists `Content-Disposition` in `Access-Control-Expose-Headers`; without it the file is named after the chat's title. Errors show in a toast.

### Routed answers

Each agent message's `route` (docs/DESIGN.md §3.4; read by `lib/route.ts`, which tolerates older messages with none of it) shapes how the transcript shows the answer: a general-knowledge answer (`answer: "general"`, or `mixed` with a `general_note`) says "General knowledge, not from your documents"; a `mixed` answer that cites sources says "From your documents + general knowledge" (both in the same quiet style as "Not in your documents", which an answer given from general knowledge on purpose doesn't get); `ack` and `conversation` are light bubbles without a source list; a `clarification` says "Asked to clarify". When `rewritten_query` differs from what the user said (a correction, a follow-up, a word Whisper mis-heard), their message gets a tiny toggle beside its time and an "Understood as: …" line that shows on hover or focus. Topic shifts show nothing. On the voice stage, source chips (and the "Not in your documents" pill) belong only to answers that draw on the documents: once the saved answer is general, an acknowledgement or a clarification there are none, whatever retrieval found.

## Checks

```bash
npm run build && npm run typecheck
```

`build` also generates `next-env.d.ts`, which `typecheck` needs on a fresh clone.

## Layout

```text
app/          root layout (reads BACKEND_URL per request), routes, global styles (globals.css: tokens + primitives;
              styles/: shell, overlays, pages, chat, voice, summary)
components/   AppProviders, AppFrame (sidebar + drawer), Sidebar, views (Home, Project, Chat, Status), Transcript,
              Summary (panel tabs, the summary view), TitleText, Citations (chips, source list, popover),
              Documents (documents card), Uploads (upload queue), Actions (dialogs and entity actions), Dialog, Menu,
              Toast, StatusPanel, Icon, voice/ (VoiceChat: the voice view, PresenceField, Captions)
scripts/      copy-vad-assets.mjs (VAD + ONNX runtime from node_modules into public/vad/)
lib/          api.ts (typed backend calls, upload, streamed chat), sse.ts (Server-Sent Events reader), chat-turns.ts
              (questions and streaming answers), citations.ts ([S#] markers and citation shapes), backend-url.ts
              (runtime URL validation), backend-context.tsx (URL + public config), workspace.tsx (projects/chats/pins/
              documents store, ingestion polling), health.tsx (GET /health polling), format.ts, route.ts (what the
              router decided, "Understood as"), use-chat-summary.ts + summary-model.ts + summary-request.ts (summaries),
              auto-title.ts (the automatic title), download.ts (export file names and saving)
lib/voice/    protocol.ts (messages, frame header, socket URL), session.ts (the live session), capture.ts (mic +
              AudioWorklet), playback.ts (gapless 24 kHz playback, duck, progress), vad.ts (Silero barge-in), captions.ts
              (word timing), analysis.ts + presence-renderer.ts (voice levels, the WebGL2 field), autostart.ts,
              use-voice-session.ts
```
