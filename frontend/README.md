# Frontend

Next.js (App Router) + TypeScript. Phase 1 adds projects, chats and transcripts (docs/DESIGN.md §3.9): a sidebar with search, pinned items and each project's chats; project and chat pages; a paginated transcript view; document upload with live ingestion status; text chat with streamed, cited answers; and the phase 0 system-status panel at `/status`. Voice (§3.8) comes in later phases.

## Run

Needs the backend on `http://localhost:8000` ([`../backend/README.md`](../backend/README.md)).

```bash
npm install
```

```bash
npm run dev
```

Open http://localhost:3000. The dev server binds to `127.0.0.1` only.

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
| `/` | Welcome and recent projects, or an invitation to create the first one |
| `/projects/[projectId]` | Rename, pin, archive, delete; the project's chats (archived ones folded away); its documents: upload, ingestion status, delete |
| `/chats/[chatId]` | Project › chat breadcrumb, rename, pin, archive, delete; the documents the chat answers from (toggle chips to narrow its scope); the transcript (opens at the latest messages, loads earlier ones as you scroll up); the composer |
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

`[S1]` markers in answers become numbered chips; hovering, focusing or clicking one opens a popover with the document, pages and the cited passage (`components/Citations.tsx`), and each answer lists the sources it cites with the same numbers. The transcript follows a streaming answer while you're at the bottom, and stays put (with "Jump to latest") once you scroll up. Without a READY document in the chat's scope the composer says why and links to the project's documents instead of sending.

## Checks

```bash
npm run build && npm run typecheck
```

`build` also generates `next-env.d.ts`, which `typecheck` needs on a fresh clone.

## Layout

```text
app/          root layout (reads BACKEND_URL per request), routes, global styles (globals.css: tokens + primitives;
              styles/: shell, overlays, pages, chat)
components/   AppProviders, AppFrame (sidebar + drawer), Sidebar, views (Home, Project, Chat, Status), Transcript,
              Citations (chips, source list, popover), Documents (documents card), Uploads (upload queue),
              Actions (dialogs and entity actions), Dialog, Menu, Toast, StatusPanel, Icon
lib/          api.ts (typed backend calls, upload, streamed chat), sse.ts (Server-Sent Events reader), chat-turns.ts
              (questions and streaming answers), citations.ts ([S#] markers and citation shapes), backend-url.ts
              (runtime URL validation), backend-context.tsx (URL + public config), workspace.tsx (projects/chats/pins/
              documents store, ingestion polling), health.tsx (GET /health polling), format.ts
```
