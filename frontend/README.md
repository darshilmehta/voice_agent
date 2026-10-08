# Frontend

Next.js (App Router) + TypeScript. Phase 1 adds projects, chats and transcripts (docs/DESIGN.md §3.9): a sidebar with search, pinned items and each project's chats; project and chat pages; a paginated transcript view; and the phase 0 system-status panel at `/status`. Typing to the agent arrives with document ingestion; voice (§3.8) in later phases.

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
| `/projects/[projectId]` | Rename, pin, archive, delete; the project's chats (archived ones folded away); its documents, read-only until ingestion lands |
| `/chats/[chatId]` | Project › chat breadcrumb, rename, pin, archive, delete; documents in scope; the transcript (opens at the latest messages, loads earlier ones as you scroll up); a composer that stays disabled for now |
| `/status` | Health of every backend component, and what this frontend is connected to |

The sidebar (a drawer below 820 px) has New project, search (⌘K / Ctrl+K; filters project names and chat titles), Pinned, Projects (each expands to its chats, newest activity first) and a health indicator that opens `/status`.

Every change goes to the backend first; the entity it returns is applied to the shared store at once and the affected lists are fetched again, so the sidebar and pages always show what the server has, in its order.

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
              Actions (dialogs and entity actions), Dialog, Menu, Toast, StatusPanel, Icon
lib/          api.ts (typed backend calls), backend-url.ts (runtime URL validation), backend-context.tsx (URL + public
              config), workspace.tsx (projects/chats/pins store), health.tsx (GET /health polling), format.ts
```
