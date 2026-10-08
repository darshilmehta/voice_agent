/** Typed access to the backend. Shapes mirror backend/app/api/*.py and backend/app/domain/projects.py. */

export type Language = "en" | "hi";

export interface PublicConfig {
  app_name: string;
  version: string;
  profile: "local" | "cloud";
  client: { app_title: string; languages: Language[]; default_language: Language };
  auth: { provider: string; issuer_url: string | null; client_id: string | null; audience: string | null };
  features: { voice: boolean; web_search: boolean; debug_panel: boolean };
  limits: { max_upload_mb: number; allowed_extensions: string[] };
}

export type HealthStatus = "ok" | "degraded" | "down" | "disabled" | "not_implemented";

export interface ProviderHealth {
  capability: string;
  provider: string;
  status: HealthStatus;
  detail: string;
  remote: boolean;
  implemented: boolean;
  latency_ms: number | null;
}

export interface HealthReport {
  status: "ok" | "degraded";
  version: string;
  profile: string;
  strict_offline: boolean;
  providers: ProviderHealth[];
}

// ------------------------------------------------------------------ projects, documents, chats, messages (§3.9)

/** ISO 8601 timestamp in UTC, as the backend serialises datetimes. */
export type Timestamp = string;

export interface Project {
  id: string;
  name: string;
  description: string | null;
  pinned_at: Timestamp | null;
  archived_at: Timestamp | null;
  created_at: Timestamp;
  updated_at: Timestamp;
  chat_count: number;
  document_count: number;
  pinned: boolean;
  archived: boolean;
}

export type DocumentStatus = "PENDING" | "PROCESSING" | "READY" | "FAILED";

export interface ProjectDocument {
  id: string;
  project_id: string;
  filename: string;
  mime: string;
  size_bytes: number;
  sha256: string;
  version: number;
  status: DocumentStatus | (string & {});
  page_count: number | null;
  chunk_count: number | null;
  error: string | null;
  created_at: Timestamp;
  updated_at: Timestamp;
}

export interface Chat {
  id: string;
  project_id: string;
  title: string;
  title_is_auto: boolean;
  pinned_at: Timestamp | null;
  archived_at: Timestamp | null;
  /** null = all of the project's documents. */
  document_scope: string[] | null;
  language: string | null;
  message_count: number;
  created_at: Timestamp;
  updated_at: Timestamp;
  last_message_at: Timestamp | null;
  pinned: boolean;
  archived: boolean;
}

export interface PinnedChat extends Chat {
  project_name: string;
}

export interface PinnedItems {
  projects: Project[];
  chats: PinnedChat[];
}

export type Role = "user" | "agent" | "event";
export type Modality = "voice" | "text";

/** A source an answer cites. The backend stores free-form JSON; today it writes document_id, page and chunk_id. */
export interface Citation {
  document_id?: string;
  page?: number;
  chunk_id?: string;
  [key: string]: unknown;
}

export interface Message {
  id: string;
  chat_id: string;
  seq: number;
  role: Role;
  modality: Modality;
  text: string;
  /** Agent answers cut off by a barge-in: what was actually played. */
  heard_text: string | null;
  language: string | null;
  citations: Citation[];
  route: Record<string, unknown> | null;
  latency: Record<string, unknown> | null;
  created_at: Timestamp;
  interrupted: boolean;
}

export interface MessagePage {
  items: Message[];
  total: number;
  has_more: boolean;
  next_cursor: number | null;
}

export interface ProjectPatch {
  name?: string;
  description?: string | null;
  pinned?: boolean;
  archived?: boolean;
}

export interface ChatPatch {
  title?: string;
  pinned?: boolean;
  archived?: boolean;
  document_scope?: string[] | null;
  language?: Language | null;
}

// ------------------------------------------------------------------ transport

export class BackendError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "BackendError";
  }

  /** The request never got an HTTP answer: the backend is down or unreachable. */
  get unreachable(): boolean {
    return this.status === undefined;
  }

  get notFound(): boolean {
    return this.status === 404;
  }
}

export const isAbort = (err: unknown) => err instanceof DOMException && err.name === "AbortError";

/** A readable message for any error thrown by the calls below. */
export const errorMessage = (err: unknown) =>
  err instanceof BackendError ? err.message : err instanceof Error ? err.message : String(err);

/** FastAPI errors are `{"detail": "…"}` or, for request validation, `{"detail": [{loc, msg}, …]}`. */
async function readDetail(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    const { detail } = body;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d: { loc?: unknown[]; msg?: string }) => {
          const field = (d.loc ?? []).filter((p) => p !== "body").join(".");
          const msg = (d.msg ?? "invalid").replace(/^Value error, /, "");
          return field ? `${field}: ${msg}` : msg;
        })
        .join("; ");
    }
  } catch {
    // not JSON
  }
  return null;
}

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
}

async function request<T>(base: string, path: string, { method = "GET", body, signal }: RequestOptions = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, {
      method,
      signal,
      cache: "no-store",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    if (isAbort(err)) throw err;
    throw new BackendError(`Can't reach the backend at ${base}`);
  }
  if (!res.ok) {
    const detail = await readDetail(res);
    throw new BackendError(detail ?? `${method} ${path} returned HTTP ${res.status}`, res.status);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const getJson = <T>(base: string, path: string, signal?: AbortSignal) => request<T>(base, path, { signal });

const id = encodeURIComponent;

export const fetchPublicConfig = (base: string, signal?: AbortSignal) =>
  getJson<PublicConfig>(base, "/api/config/public", signal);

export const fetchHealth = (base: string, signal?: AbortSignal) => getJson<HealthReport>(base, "/health", signal);

/** Every call bound to one backend URL; built once per app from the runtime BACKEND_URL. */
export function createApi(base: string) {
  return {
    base,
    publicConfig: (signal?: AbortSignal) => fetchPublicConfig(base, signal),
    health: (signal?: AbortSignal) => fetchHealth(base, signal),

    listProjects: (opts: { includeArchived?: boolean; signal?: AbortSignal } = {}) =>
      getJson<{ items: Project[] }>(
        base,
        `/api/projects${opts.includeArchived ? "?include_archived=true" : ""}`,
        opts.signal,
      ).then((r) => r.items),
    getProject: (projectId: string, signal?: AbortSignal) =>
      getJson<Project>(base, `/api/projects/${id(projectId)}`, signal),
    createProject: (body: { name: string; description?: string | null }) =>
      request<Project>(base, "/api/projects", { method: "POST", body }),
    updateProject: (projectId: string, patch: ProjectPatch) =>
      request<Project>(base, `/api/projects/${id(projectId)}`, { method: "PATCH", body: patch }),
    deleteProject: (projectId: string) =>
      request<void>(base, `/api/projects/${id(projectId)}`, { method: "DELETE" }),
    listDocuments: (projectId: string, signal?: AbortSignal) =>
      getJson<{ items: ProjectDocument[] }>(base, `/api/projects/${id(projectId)}/documents`, signal).then(
        (r) => r.items,
      ),

    listChats: (projectId: string, opts: { includeArchived?: boolean; signal?: AbortSignal } = {}) =>
      getJson<{ items: Chat[] }>(
        base,
        `/api/projects/${id(projectId)}/chats${opts.includeArchived ? "?include_archived=true" : ""}`,
        opts.signal,
      ).then((r) => r.items),
    getChat: (chatId: string, signal?: AbortSignal) => getJson<Chat>(base, `/api/chats/${id(chatId)}`, signal),
    createChat: (projectId: string, body: { title?: string; language?: Language } = {}) =>
      request<Chat>(base, `/api/projects/${id(projectId)}/chats`, { method: "POST", body }),
    updateChat: (chatId: string, patch: ChatPatch) =>
      request<Chat>(base, `/api/chats/${id(chatId)}`, { method: "PATCH", body: patch }),
    deleteChat: (chatId: string) => request<void>(base, `/api/chats/${id(chatId)}`, { method: "DELETE" }),

    /** One page of a transcript, chronological. `before` reads backward (open at the latest), `after` forward. */
    listMessages: (
      chatId: string,
      cursor: { before?: number; after?: number; limit?: number } = {},
      signal?: AbortSignal,
    ) => {
      const q = new URLSearchParams();
      if (cursor.before !== undefined) q.set("before", String(cursor.before));
      if (cursor.after !== undefined) q.set("after", String(cursor.after));
      if (cursor.limit !== undefined) q.set("limit", String(cursor.limit));
      const qs = q.toString() ? `?${q}` : "";
      return getJson<MessagePage>(base, `/api/chats/${id(chatId)}/messages${qs}`, signal);
    },

    pins: (signal?: AbortSignal) => getJson<PinnedItems>(base, "/api/pins", signal),
  };
}

export type Api = ReturnType<typeof createApi>;
