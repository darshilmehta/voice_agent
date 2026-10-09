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
| `/projects/[projectId]` | Rename, pin, archive, delete; the project's overview dashboard once it is built ([Live visual canvas](#live-visual-canvas)); the project's chats (archived ones folded away); its documents: upload, ingestion status, delete |
| `/chats/[chatId]` | Project › chat breadcrumb, rename, pin, archive, delete; the documents the chat answers from (toggle chips to narrow its scope); the conversation in voice mode (below), with the transcript as a panel (opens at the latest messages, loads earlier ones as you scroll up) and, when the answer has evidence to show, the live visual canvas |
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

`[S1]` markers in answers become numbered chips (and `[W1]` markers, for live web results, "W1" chips in their own teal, see "Live web search" below); hovering, focusing or clicking one opens a popover with the document, pages and the cited passage (`components/Citations.tsx`; a document without pages, such as a DOCX, shows the passage's section instead, "§ 4.2.1 Hotels", the last heading of the citation's `section` with the whole path on hover, in the chip and the popover, and web chips are unchanged; a passage that is a markdown table is shown as a compact table that scrolls sideways, read by `parseTableSnippet` in `lib/citations.ts` from the one-line snippet the backend stores), and each answer lists the sources it cites with the same numbers. The transcript follows a streaming answer while you're at the bottom, and stays put (with "Jump to latest") once you scroll up. Without a READY document in the chat's scope the composer says why and links to the project's documents instead of sending.

### Voice

The chat page (`components/voice/VoiceChat.tsx`) is the voice view. The stage has the **presence field** (`PresenceField.tsx`; the §3.8 particle field ported from `docs/prototypes/voice-presence.html`: one WebGL2 draw call of ~10,000 instanced strokes, a Canvas 2D fallback, `prefers-reduced-motion` and light/dark respected), **live captions** (`Captions.tsx`), the current answer's **source chips** (the same citation popover as the transcript), a polite `aria-live` **state label** and the mic controls: a large mic button (start and end the conversation), a stop button for the answer in progress, and "Type instead" (expands the text composer; typed turns stream into the transcript as before). A **new chat starts listening at once** (the click that created it allows audio, `lib/voice/autostart.ts`); reopening a chat does not, and the transcript panel is open by default when it has messages. Blocked, missing or unsupported microphones say so and offer Try again or typing.

Keys: Space or Enter on the mic button starts and ends the conversation; **Esc steps back one level**: it closes the transcript when that covers the stage (when the chat area is narrower than 880 px), else stops the answer in progress, else ends the conversation.

The audio engine is `lib/voice/` (protocol in docs/DESIGN.md §3.10, `WS /ws/chats/{id}/voice` on the backend origin):

- `session.ts` runs a session as a small external store: mic, socket, player and VAD; unexpected closes reconnect with backoff, ten attempts over about a minute (the mic and VAD keep running; the new server session starts its turn numbers over, the page forgets the dropped one's unfinished answer and fetches the transcript tail again), close 4403 (the socket's Origin isn't allowed: open the app from the address it is set up for), 4404 and 4409 end it with an explanation, release the mic and audio and don't reconnect, `error` messages show a notice and the session keeps listening; everything is torn down on navigation, `pagehide`, or when the backend turns `features.voice` off. A start that is stopped while the permission prompt is open (Esc, then the mic again) only ever releases its own microphone and audio context, never the newer start's.
- `capture.ts`: `getUserMedia` with echo cancellation, noise suppression and auto gain; an AudioWorklet downsamples to 16 kHz with a 32-tap windowed-sinc low-pass (cutoff 7 kHz, so sibilants above 8 kHz don't fold back into the speech band: -60 dB or better from 10 kHz up at 48 and 44.1 kHz) and sends 512-sample PCM16 frames once the server says `ready`; an AnalyserNode on the mic feeds the user's waves.
- `playback.ts`: 24 kHz frames are scheduled back to back on the AudioContext clock through a per-turn gain and a master gain (the duck, 0.2) into an AnalyserNode (the agent's waves). Only the current turn's frames play; others are dropped. The same schedule gives `played_ms` for `barge_in_start` and `playback` (output and base latency subtracted), and `chunkFraction` for captions. `playback_done` waits until every announced chunk has delivered its length in frames and the audio has played (after a 1.5 s grace if frames never complete).
- `vad.ts`: Silero v5 `MicVAD` on the shared mic stream. Speech onset while the agent speaks ducks the volume at once and sends `barge_in_start`; the server's `barge_in` decides (`stop` flushes that turn and freezes the caption, `resume`, or a VAD misfire, restores the volume; if the server never answers the volume returns after 3.5 s).
- `captions.ts`: each `audio_chunk` announces a clause's text and length; a word lights when the audio playing has reached its share of the chunk (by letters, plus a pause after punctuation), so captions follow what is actually heard. A web search's filler chunk (`filler: true`) is left out of the spoken-vs-generated comparison, so generated-but-unspoken text still shows (dimmed) after it, and its words are an aside (below).

**What the client relies on, and what it tolerates.** The backend guarantees (docs/DESIGN.md §3.10): `agent_message` is sent after the turn's last binary frame; after `barge_in: stop` or `stop` no further `delta`, `audio_chunk` or frames come for that turn, and its cut `agent_message` arrives before the next turn's `user_message` / `turn`; `turn_id` strictly increases within a session and a reconnect starts fresh; within a turn the order is `user_message`, `turn`, `sources`, then `delta` / `audio_chunk`, then `agent_message`. The client still doesn't trust them: it remembers the highest turn id and every cut turn, so a late `delta`, `audio_chunk`, frame or `sources` of a stopped or older turn is dropped and can't become the current turn (and its saved message is told apart from the new question's by `seq`); a `stop` pressed before the answer has an id (the question is still being transcribed or the turn id hasn't arrived) cancels the turn that arrives, until `agent_message`, `state: listening` or new user speech; `turn` arriving before `user_message` is accepted; words the user said that never became a message (a backchannel, noise, a failed transcription) leave the captions on `barge_in: resume`, `state: listening` or an `error`, and the speculative transcript of an ignored backchannel that arrives after `resume` is dropped. Found against the real backend: when a complete answer is cut during playback the server sends `agent_message` again with the same id and `heard_text` set, which updates the transcript entry in place (never twice); a `stop` that finds no answer in progress (the question is still being transcribed) is dropped by the server, so the client repeats it as soon as the turn id is known; and audio that starts playing while the user is already talking (they spoke up while the answer was being written) ducks and sends `barge_in_start` like any other interruption. From the backend's review fixes: `barge_in: stop` is accepted with or without a `barge_in_start` before it (it cuts that turn, and the user's words stay in the captions), and it is idempotent: after the user's own Stop the server also decides the pending barge-in with a `stop` for the same turn, which finds the turn already cut and does nothing, and a late one for an older turn never touches the newer turn's pending duck or timer. When the server ignores an utterance (it closes every `user_speech: start` with `end` and re-sends `state`, plus `barge_in: resume` if it treated it as a barge-in), the words and the pending "You" bubble go: on `state: listening`, or on the agent's own `speaking` / `thinking` while an answer is in progress; a real question keeps its words through `thinking`, because no answer of the agent's is in flight, and a real interruption keeps them because the turn is already cut. A saved answer says how it was cut in `route.interrupted` (`barge_in`, `stop` or `disconnect`): the transcript reads "Interrupted after: ...", "Stopped after: ..." or "Cut off when the connection dropped, after: ..." (and the same without a quote when nothing had been played), never "interrupted by you" for a cut the user didn't make; older messages without a reason read "Interrupted after: ...".

In development `window.__voice` is the live session (`getSnapshot()`, `stats`, `player`) and `window.__presence` the presence meters, for debugging and browser tests.

### Revisiting a chat: titles, summary, export

The transcript panel has two tabs, **Transcript | Summary** (`components/Summary.tsx`; the plain transcript layout, with voice off, has the same tabs above the transcript). The chat's **⋯** menu (header, sidebar, project page) has **Summarise**, **Regenerate title** and **Export transcript → Markdown / JSON**; the last three are greyed out for a chat with no messages.

- **Automatic titles.** A new chat is called "New chat" until the backend writes a title, about a second after the first answer is saved (a few seconds if the model is busy). While the chat still has that placeholder (and `title_is_auto`), the page re-reads it 0.8, 1.8, 3.2 and 5 s after each saved answer (a voice `agent_message` or a finished typed turn; `lib/auto-title.ts`), then with a widening gap (2.5 s up to 10 s) until a minute has passed (12 reads; the schedule is `lib/title-watch.ts`). The next saved answer starts a new minute, and coming back to the tab or window reads once more; nothing is read while the tab is hidden. It stops at once when the title changes or the user renames the chat (a rename also discards any read still in flight, so an old title can't reappear), and on leaving the page. A chat that already has an answer when it opens (a reload, or coming back to it) is watched too. The chat is refetched into the shared store without a loading state, and the header and sidebar fade the new text in (`components/TitleText.tsx`). **Regenerate title** calls `POST /api/chats/{id}/title:regenerate`; a `409` (the user chose the title) asks "Replace your title?" and retries with `?force=true`; `422` and `503` say what happened in a toast.
- **Summary.** `GET /api/chats/{id}/summary` when the Summary tab is first shown (a `404` "summary of chat … not found" is the normal "no summary yet", not an error), `POST …/summary[?language=en|hi]` to write or refresh one. It is slow on the local model, so the view says so, keeps working while you look at the transcript (the state lives in `lib/use-chat-summary.ts`, in the chat page), and a failed write leaves the summary that was there. It is rendered from the structured fields (overview, key points with source chips in the citation popover style, unanswered questions, follow-ups), never from the markdown `content`, and is never spoken. **Out of date: covers N of M messages** with a Refresh button shows when the backend says `stale` or the chat got more messages since the page fetched the summary; the **EN | हिंदी** switch writes the other language. A question the documents couldn't answer scrolls the transcript to the message that asked it (`message_seq`), loading earlier pages if needed, and highlights it.
- **Export.** `GET /api/chats/{id}/export?format=md|json` is fetched, turned into a blob and saved under the file name the server sends (`filename*=UTF-8''…` for Hindi titles; `lib/download.ts`). A browser can only read that header across origins if the backend lists `Content-Disposition` in `Access-Control-Expose-Headers`; without it the file is named after the chat's title. Errors show in a toast.

### Routed answers

Each agent message's `route` (docs/DESIGN.md §3.4; read by `lib/route.ts`, which tolerates older messages with none of it) shapes how the transcript shows the answer: a general-knowledge answer (`answer: "general"`, or `mixed` with a `general_note`) says "General knowledge, not from your documents"; a `mixed` answer that cites sources says "From your documents + general knowledge" (both in the same quiet style as "Not in your documents", which an answer given from general knowledge on purpose doesn't get); `ack` and `conversation` are light bubbles without a source list; a `clarification` says "Asked to clarify". When `rewritten_query` differs from what the user said (a correction, a follow-up, a word Whisper mis-heard), their message gets a tiny toggle beside its time and an "Understood as: …" line that shows on hover or focus. Topic shifts show nothing. On the voice stage, source chips (and the "Not in your documents" pill) belong only to answers that draw on the documents: once the saved answer is general, an acknowledgement or a clarification there are none, whatever retrieval found.

### Live web search

When a question needs current data the backend searches the web (SearXNG, docs/DESIGN.md §3.7) while it retrieves from the documents, says a short filler, and cites documents as `[S#]` and web results as `[W#]`. Everything web-related is hidden when `GET /api/config/public` has `features.web_search: false`, except rendering the web citations of old messages (chips, popovers, the label of where an answer comes from).

- **`tool` events** (SSE `event: tool`, WebSocket `{"type":"tool","turn_id":N,…}`; `lib/web-search.ts` reads both): `start {query}`, `results {sources, elapsed_ms}` (the new web citations, numbered W1, W2… in arrival order and never renumbered), then `done` / `timeout` / `failed {count, elapsed_ms, detail?}`. A terminal event can arrive in the middle of the answer's deltas and a continuation can add `[W3]` seconds after the main answer, so only `agent_message` (voice) or the end of the stream (SSE) ends a turn. A `[W3]` that only a late `tool results` carries gets its chip from the turn's merged web sources; the saved message's `citations` are authoritative once it arrives.
- **The searching state** is on from `start` until a terminal event or any other end: `sources`, the saved answer, an error, a cut (barge-in stop, the Stop button), a dropped connection or a new turn (the error and cut paths may send no terminal `tool` event). While it is on, a **"Searching the web…" badge** with the query shows on the voice stage and under the streaming text answer, the voice state label says "Searching the web…" (the server stays in `speaking` through the silent search after the filler) and the presence field swirls as when thinking. Afterwards a quiet "Searched the web for “…”" (with "· timed out", "Web search failed for …" or "· no results" when it ended that way; the hover text has the result count and time) stays under the answer, because that query is exactly what left the machine. A saved message with no live record of it uses `route.web_search.query`.
- **The filler** ("Let me look that up." / "मैं देखता हूँ।") is the turn's first `audio_chunk` with `filler: true`. It isn't part of the answer, so it is left out of the cited-text fallback for the stage's chips and of the caption comparison, but its time counts in `played_ms` (nothing changes there). Its words show as a light aside above the captions, lighting up as they are spoken, and give way to the answer's captions once the answer has begun. It may arrive after `sources`; that is handled the same way.
- **Web citations** (`kind: "web"`, `url`, `title`, `site`, `published`; a citation without `kind` is a document passage): `[W1]` is a teal "W1" chip in the text; the source list under an answer groups documents first, then web results (a globe and the site name, with the W number); the popover shows title, site and date, the result's text and an external link (`target="_blank"`, `rel="noopener noreferrer"`, http(s) only: any other address is dropped and the popover has no link). That popover is a small non-modal dialog: Tab from the open chip moves into the link, Tab or Shift+Tab from the link goes back to the chip, Esc closes it and returns to the chip. A web citation is never looked up in the documents. The marker pattern accepts `[S#]`, `[W#]` and mixes (`[S1, W2]`; a bare number goes with the kind before it).
- **Labels** (`lib/route.ts`): `route.basis` (a sorted subset of `documents`, `web`, `general`) decides them when present, and a web citation adds "web": "General knowledge, not from your documents" (general only), "From your documents + general knowledge", "Live web results" (web only), "From your documents + live web results", "Live web results + general knowledge" and "From your documents + live web results + general knowledge" (documents only has no label). Without `basis` (older messages) the logic that predates it applies, and an answer with web citations never says "documents + general knowledge". A general answer shows its web chips, no document chips. `route.live_note` (`unavailable` / `failed`) adds a quiet line ("Live web search isn't available, so this answer has no web results.").
- **Summaries**: a key point's web source is `{document_id: "", filename: <site>}` with no pages and no link; it renders as a web chip (globe and site), not a document.

### Live visual canvas

The voice gives the short insight; the screen shows the evidence: charts, tiles, tables and timelines built by the backend from the documents' own tables (docs/DESIGN.md §12.1). The frontend **draws and formats; it never computes**: every number arrives in its unit's scale with the cell it came from, and every derived number arrives as a `Calculation` ("calculated", with its formula and the cells it used). Contract v1 is read by `lib/canvas/types.ts` (`normalizeVisual` tolerates a missing list or an unknown `kind`, which is then shown as a table).

**Where it shows.** A chat whose canvas has panels takes the stage and the presence field **docks**: it shrinks into a band at the top (the top-left corner at 640 px and below), keeps breathing as the presence indicator through every voice state, and eases back to full size when the last panel goes. The slot the rings are centred on is animated, and the field follows it every frame, so `PresenceField` needed no change. A *Hide visuals* button brings the full field back; a new or removed visual brings the canvas back. The transcript panel is unchanged. At 640 px and below the panels are a **swipeable sheet** (scroll-snap, dots or "3 of 15") above the voice controls, each panel scrolling inside itself; wider, they are a responsive grid. A canvas with nothing on it draws **nothing** (no chrome, no empty state, the presence field stays full size); a chat of a backend without the endpoint (404) or an unreachable one behaves the same. With voice turned off the canvas sits above the transcript.

**Events.** `GET /api/chats/{id}/canvas` on open; `visual` (`preparing` → skeleton, `ready` → panel, `failed` → a quiet note that goes away after 15 s) and `canvas` (a full snapshot) from both turn streams: the SSE `event: visual` / `event: canvas` (`lib/api.ts` `sendMessage` yields them, `lib/chat-turns.ts` publishes them) and the voice socket's `{type: "visual" | "canvas"}` (`lib/voice/session.ts`). Both go through `lib/canvas/events.ts` into one pure reducer (`lib/canvas/state.ts`), so the two paths cannot drift. A `failed` with `detail: "cancelled"` (the conversation moved on before the visual was planned) only drops its skeleton, without a note. An answer's visual comes after the answer (docs/DESIGN.md §12.1): the SSE stream stays open for it, so the composer accepts the next question as soon as `agent_message` arrives, and a `visual` that becomes ready marks the answer "Chart added" in the transcript (`route.visual_id`, which the backend also saves; the voice session patches its turn's message the same way, before or after its `agent_message`). Pin, unpin, remove and move are `POST …/canvas/ops`; the answer is the new canvas and replaces what the page holds. Move up / down are buttons (keyboard), the grip drags (mouse); after a move or a removal focus stays on a button of the panel that took its place and the change is announced.

**Kinds.** `kpi` tiles (value, unit small beside it, change, source), `line` (several units become stacked small charts, never two y-scales), `bar` (nominal categories in one colour; a long or long-named list lies down as rows, folds to twelve with "Show all", and emphasises the highlighted bar by greying the rest), `grouped_bar`, `stacked_bar` (negatives stack below zero), `waterfall` (start and end bars, floating increases and decreases; direction is also in the signed labels and the legend words), `donut` (at most eight parts, otherwise bars), `table`, `comparison` (A vs B with changes), `timeline`.

**Marks and colour.** The dataviz skill's method: bars at most 24 px with a 4 px rounded data end and a square baseline, 2 px lines, 10 px markers with a 2 px surface ring, 2 px surface gaps between touching marks, solid hairline grids, a legend for two or more series, value labels only where they fit (never a number on every point), text in text tokens (never a series colour), thin marks. The eight categorical hues are the skill's default palette mapped onto this app's surfaces as `--viz-*` tokens in `app/globals.css` and re-selected for dark; the six checks were run with `validate_palette.js` against this app's `--surface` (`#ffffff` and `#1d1d1b`): lightness band, chroma, adjacent colour-vision distance (worst 9.1 light / 8.4 dark, floor 8) and normal-vision distance (19.6 / 19.3, floor 15) pass; in light mode the aqua, yellow and magenta marks are below 3:1 on white, which the relief rule answers with a legend, selective labels, tooltips and the table view that every chart has. Waterfall polarity uses the skill's diverging pair (blue and red) with ink for totals. A ninth series is neutral grey, never a generated hue. Texture (the skill's opt-in channel for print and forced colours) is not drawn; `forced-colors` keeps the marks' own colours.

**Provenance.** Hovering or focusing a point shows a tooltip: the value, its label, every series at that position, and the source (document, page, the cell's text); clicking (or Enter) opens the existing citation popover (`components/Citations.tsx`, one export added) for that source, led by `Cell "604"` and followed by the cited passage. Tiles, table cells, timeline sources and the inputs of a calculation do the same on hover, focus and click. There is no document viewer in the app, so the popover is where a source opens. Calculated values (`series.calculated`, a tile's change, every `Calculation`) carry a dashed **calculated** badge, their `formula_text`, and their inputs as chips that open their cells; each panel's "How these were calculated (n)" lists them all. A `highlight` washes its positions, flags them, makes their labels bold (and greys or dims what it isn't about) and shows its note under the chart.

**Accessibility.** Each panel is an `article` named by its title and described by the visual's `summary`. Every chart has "View as table" (units in the headers, calculated columns marked, sources one click away). The marks are decorative to assistive technology; over them is one real button per data point (at least 24 px, bigger than the mark) with a full name ("FY22, EBITDA: ₹604 crore. Source: annual_report_FY24.pdf, Page 46"), one tab stop per chart, arrow keys between positions and series, Home/End/PageUp/PageDown, Enter for the source, Esc to hide the tooltip. Nothing is told by colour alone (legend words, signed labels, arrows and flags, dashed "calculated" badges, hollow markers for missing values). `prefers-reduced-motion` stops the docking ease and every transition. Hindi visuals use Hindi labels and words ("तालिका के रूप में देखें", "गणना की गई", "₹4,210 करोड़"), keep Latin digits as the documents print them, break and truncate labels on whole characters (`Intl.Segmenter`) and use a taller line height.

**Numbers.** `lib/canvas/format.ts`: Indian grouping (12,34,567) whenever the unit is INR or uses lakh/crore, western otherwise; the scale word is spelled out ("₹4,210 crore"); at most two decimals with trailing zeros dropped; a true minus (−); percent values are percent (18.4 is 18.4%); changes are "+18.3%", "+1.2 pp", "+₹520 crore", never coloured good or bad (the contract doesn't say which way is good). Missing values are "—" (and "Not reported" in a tooltip), drawn as a gap in a line and a hollow mark on a bar chart's baseline.

**Chart rendering: hand-rolled SVG, no new dependency.** The canvas needs five shapes (lines with gaps, bars with rounded ends and gaps, stacks, a waterfall, a ring), nice ticks, label fitting and a hit layer. What a library would add against that, bundled minified and tree-shaken (React external), measured with esbuild:

| Option | Added, gzipped |
|---|---|
| d3-scale 4.0.2 + d3-shape 3.2.0 (scales and paths only: no axes, tooltips, keyboard, labels) | 11.9 KB |
| visx 4.0.0 (scale, shape, axis) | 26.0 KB |
| Recharts 3.10.1 | 119.7 KB |
| ECharts 6.1.0 (tree-shaken, SVG renderer) | 195.4 KB |
| Vega-Lite 6.5.0 (+ vega) | 264.9 KB |
| **This implementation: all of the canvas** (every chart, panels, tooltips, keyboard, tables, tiles, timeline, both languages) | **22.1 KB JS + 3.0 KB CSS, lazy** |

The d3 primitives would save a few hundred lines of pure, tested layout math (`lib/canvas/scales.ts`) and still leave the part that matters here undone: a focusable button per point, provenance in every tooltip, per-mark rounded ends and surface gaps, small multiples, label fitting for Devanagari, "calculated" everywhere, themeing from CSS tokens. A chart library would draw its own tooltips, legends and ARIA, which then have to be fought into the citation popover and the house marks; Recharts, ECharts and Vega-Lite also cost more than the rest of the app's page JS. So: plain React SVG, a small `lib/canvas/scales.ts` (ticks, bar and stack geometry, waterfall steps, line runs, donut arcs, label fitting, tooltip placement, keyboard cursor), nothing added to `package.json`, no CDN, no pinning question.

**Bundle cost** (`next build`, gzipped, against `origin/main` at the same commit): the chart code is a separate chunk (20.7 KB JS, 3.0 KB CSS) that a page downloads **only when a canvas or an overview has a panel, or a visual is being prepared** (so the skeleton turns into the chart without a second wait); the contract reader loads with it (1.4 KB), the debug form only with `features.debug_panel` (0.9 KB). A chat or project with nothing to draw loads none of it (checked in the browser's resource list with the production build). Always loaded: +1.0 KB CSS (docking, shell, skeleton) and +3.9 KB JS on the chat page (+3.6 KB on the project page): the canvas state, the calls, the shell and the few words it needs.

**Overview dashboard.** The project page shows `GET /api/projects/{id}/overview` once `ready` (read-only panels: no pin, move or remove; each with "View as table" and its sources), "Building the overview…" while `building` (re-read every 3 s, up to ten minutes), and nothing for `none`, while loading, or when the endpoint is missing. **No "Show in chat" action:** contract v1 has no way to put an existing visual on a chat's canvas (the operations are remove, pin, unpin and move; `POST …/visuals` is debug-only and takes a spec, not a visual), so the button could only pretend. Ask the question in a chat and the visual is built there.

**Debug panel.** With `features.debug_panel` on, the chat header has "Debug: add a visual from a VisualSpec": paste JSON, `POST /api/chats/{id}/visuals`, and the visual joins the canvas like any other.

**Contract ambiguities and what the frontend does.**

- *Waterfall: which rows are totals?* Not in the contract. The first and last rows are the start and end bars, the rows between are increases and decreases; a row may say `kind: "total" | "delta"` (an optional extra, ignored when absent) for a bridge with a subtotal.
- *Comparison: how are A and B given?* `x` is null and `tiles` are "kpi / comparison only". The sides are the visual's `series` (FY23 | FY24), the metrics its `rows`, and each metric's change is the tile with the same label (or the same position when there are as many tiles as rows); a comparison with no rows lists its tiles with their changes.
- *Series with different units on one chart.* Each unit gets its own small chart (shared x, one crosshair on lines) instead of a second y-axis.
- *Donut shares.* A percentage would be a computation, so slices show the part's own figure; a share shows only if the backend sent a `share` calculation whose label names the part.
- *Percent scale.* `18.4` with a percent unit is 18.4%.
- *`position` for `move`.* The panel moves to the `position` of the neighbour it swaps with (or of the panel it is dropped on); the answer replaces the whole canvas.
- *`preparing` for an id already on the canvas.* The panel stays and says "Updating…" until its `ready`.
- *A calculation without a unit.* growth, CAGR and share are percentages, a ratio is a ratio, anything else follows the visual.

## Checks

```bash
npm run build && npm run typecheck
```

`build` also generates `next-env.d.ts`, which `typecheck` needs on a fresh clone.

## Layout

```text
app/          root layout (reads BACKEND_URL per request), routes, global styles (globals.css: tokens + primitives;
              styles/: shell, overlays, pages, chat, voice, summary, canvas + canvas-panels (the second loads with the
              chart code))
components/   AppProviders, AppFrame (sidebar + drawer), Sidebar, views (Home, Project, Chat, Status), Transcript,
              Summary (panel tabs, the summary view), TitleText, Citations (chips, source list, popover),
              Documents (documents card), Uploads (upload queue), Actions (dialogs and entity actions), Dialog, Menu,
              Toast, StatusPanel, WebSearchNote (the "Searching the web…" badge and "Searched the web for …"), Icon,
              voice/ (VoiceChat: the voice view, PresenceField, Captions)
scripts/      copy-vad-assets.mjs (VAD + ONNX runtime from node_modules into public/vad/)
lib/          api.ts (typed backend calls, upload, streamed chat), sse.ts (Server-Sent Events reader), chat-turns.ts
              (questions and streaming answers), citations.ts ([S#] / [W#] markers and citation shapes), web-search.ts
              (`tool` events and a turn's web search state, shared by SSE and voice), backend-url.ts
              (runtime URL validation), backend-context.tsx (URL + public config), workspace.tsx (projects/chats/pins/
              documents store, ingestion polling), health.tsx (GET /health polling), format.ts, route.ts (what the
              router decided, "Understood as"), use-chat-summary.ts + summary-model.ts + summary-request.ts (summaries),
              auto-title.ts + title-watch.ts (the automatic title and its timing), download.ts (export file names and saving)
components/canvas/  CanvasPanel (the shell: skeletons, failure notes, lazy loading), CanvasBoard (the lazy chunk: panels), Panel,
              ChartFrame (hit layer, keyboard, tooltip), charts/ (Line, Bar, Waterfall, Donut, Tiles, Timeline, DataTable),
              ProjectOverview, CanvasDebug (+ CanvasDebugForm), panel-context (provenance and the citation popover)
lib/canvas/   types.ts (contract v1 reader), state.ts (the reducer), events.ts, client.ts, use-canvas.ts, model.ts (series, units,
              provenance), format.ts (numbers, units, dates), scales.ts (layout math), labels.ts + labels-panel.ts (English and
              Hindi words: the few the shell needs, then the panels' own), order.ts
lib/voice/    protocol.ts (messages, frame header, socket URL), session.ts (the live session), capture.ts (mic +
              AudioWorklet), playback.ts (gapless 24 kHz playback, duck, progress), vad.ts (Silero barge-in), captions.ts
              (word timing), analysis.ts + presence-renderer.ts (voice levels, the WebGL2 field), autostart.ts,
              use-voice-session.ts
```
