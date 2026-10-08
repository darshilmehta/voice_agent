# Frontend

Next.js (App Router) + TypeScript. Phase 0 is the shell: header from the backend's public config, placeholders for documents and conversation, and a live system-status panel built from `GET /health`.

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

Everything else (title, languages, feature flags, upload limits, sign-in settings) comes from the backend's `GET /api/config/public`. The browser's origin must be listed in the backend config's `server.cors_allowed_origins`.

## Checks

```bash
npm run build && npm run typecheck
```

`build` also generates `next-env.d.ts`, which `typecheck` needs on a fresh clone.

## Layout

```text
app/          layout, page (server: reads BACKEND_URL), global styles (light/dark tokens)
components/   AppShell (header, panels), StatusPanel (provider health)
lib/          api.ts (typed backend calls), backend-url.ts (runtime URL validation)
```
