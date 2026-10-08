/** Typed access to the backend. Shapes mirror backend/app/api/*.py and backend/app/domain/projects.py. */

import { readSse } from "./sse";

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

/**
 * A source an answer cites: the answer text marks it as `[S1]`, `[S2]`, … Every field is optional in the type because
 * messages saved before typed citations hold free-form JSON (e.g. `document_id`, `page`, `chunk_id`); read them
 * through `lib/citations.ts`, which tolerates both.
 */
export interface Citation {
  source_id?: string;
  document_id?: string;
  filename?: string;
  page_start?: number | null;
  page_end?: number | null;
  chunk_id?: string;
  /** Up to ~300 characters of the cited passage. */
  snippet?: string;
  /** Older messages: a single page. */
  page?: number;
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

/** What retrieval found for a question: `sources` are numbered S1…, `abstained` means the evidence is too weak. */
export interface Confidence {
  top_score: number;
  gap: number;
  dense_similarity: number;
  above_threshold: boolean;
}

export interface SourcesPayload {
  sources: Citation[];
  confidence: Confidence | null;
  abstained: boolean;
}

export type StreamStage = "retrieval" | "llm" | "storage";

export interface StreamFailure {
  detail: string;
  stage: StreamStage | (string & {}) | null;
}

/** POST /api/chats/{id}/messages streams these, in this order (error can replace any step after user_message). */
export type ChatEvent =
  | { type: "user_message"; message: Message }
  | { type: "sources"; payload: SourcesPayload }
  | { type: "delta"; text: string }
  | { type: "agent_message"; message: Message }
  | { type: "error"; failure: StreamFailure };

export interface SendMessageBody {
  /** 1–4000 characters. */
  text: string;
  /** null: let the backend tell from the text. */
  language: Language | null;
}

export const MESSAGE_MAX_CHARS = 4000;

export interface UploadResult {
  document: ProjectDocument;
  /** The project already had this exact file (HTTP 200): `document` is the existing one. */
  duplicate: boolean;
}

export interface UploadOptions {
  /** Called with 0…1 as the request body goes out. */
  onProgress?: (fraction: number, loaded: number, total: number) => void;
  signal?: AbortSignal;
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
function detailOf(body: unknown): string | null {
  const detail = (body as { detail?: unknown } | null)?.detail;
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
  return null;
}

async function readDetail(res: Response): Promise<string | null> {
  try {
    return detailOf(await res.json());
  } catch {
    return null; // not JSON
  }
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
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

/**
 * POST one file as multipart/form-data (field `file`). XMLHttpRequest rather than fetch because only XHR reports
 * upload progress. 202 = new document (ingestion runs in the background), 200 = identical file already in the
 * project. Rejects with BackendError (422 carries the backend's reason) or an AbortError DOMException.
 */
function uploadDocument(base: string, projectId: string, file: File, opts: UploadOptions = {}): Promise<UploadResult> {
  const { onProgress, signal } = opts;
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Upload cancelled", "AbortError"));
    const xhr = new XMLHttpRequest();
    const path = `/api/projects/${id(projectId)}/documents`;
    xhr.open("POST", `${base}${path}`);
    xhr.responseType = "text";
    const onAbort = () => xhr.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    const done = () => signal?.removeEventListener("abort", onAbort);

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total, e.loaded, e.total);
      };
    }
    xhr.onload = () => {
      done();
      const body = parseJson(xhr.responseText);
      if (xhr.status === 200 || xhr.status === 201 || xhr.status === 202) {
        if (body && typeof body === "object" && "id" in body) {
          resolve({ document: body as ProjectDocument, duplicate: xhr.status === 200 });
        } else {
          reject(new BackendError(`POST ${path} returned an unexpected response`, xhr.status));
        }
        return;
      }
      const detail = detailOf(body);
      const fallback = xhr.status === 413 ? "The file is larger than the backend accepts" : `Upload failed (HTTP ${xhr.status})`;
      reject(new BackendError(detail ?? fallback, xhr.status));
    };
    xhr.onerror = () => {
      done();
      reject(new BackendError(`Can't reach the backend at ${base}`));
    };
    xhr.onabort = () => {
      done();
      reject(new DOMException("Upload cancelled", "AbortError"));
    };
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}

/** The connection breaking mid-stream surfaces as a bare TypeError ("network error"); say what happened instead. */
async function* readStream<T>(events: AsyncGenerator<T>): AsyncGenerator<T> {
  try {
    yield* events;
  } catch (err) {
    if (isAbort(err) || err instanceof BackendError) throw err;
    throw new BackendError("The connection to the backend was lost before the answer finished.");
  }
}

function parseEventData<T>(name: string, data: string): T {
  try {
    return JSON.parse(data) as T;
  } catch {
    throw new BackendError(`The answer stream sent an unreadable "${name}" event`);
  }
}

/**
 * POST a question and yield the answer as it streams (Server-Sent Events over the POST response; see lib/sse.ts).
 * Throws BackendError when the request is refused (HTTP status set) or the backend can't be reached, and an
 * AbortError when `signal` aborts. An `error` event is yielded, not thrown: the user message is already saved then.
 * Unknown event names are skipped.
 */
async function* sendMessage(
  base: string,
  chatId: string,
  body: SendMessageBody,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const path = `/api/chats/${id(chatId)}/messages`;
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, {
      method: "POST",
      signal,
      cache: "no-store",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    if (isAbort(err)) throw err;
    throw new BackendError(`Can't reach the backend at ${base}`);
  }
  if (!res.ok) {
    const detail = await readDetail(res);
    throw new BackendError(detail ?? `POST ${path} returned HTTP ${res.status}`, res.status);
  }
  if (!res.body || !(res.headers.get("content-type") ?? "").includes("text/event-stream")) {
    throw new BackendError(`POST ${path} didn't return an event stream`, res.status);
  }
  for await (const ev of readStream(readSse(res.body))) {
    switch (ev.event) {
      case "user_message":
        yield { type: "user_message", message: parseEventData<Message>(ev.event, ev.data) };
        break;
      case "sources": {
        const p = parseEventData<Partial<SourcesPayload>>(ev.event, ev.data);
        yield {
          type: "sources",
          payload: {
            sources: Array.isArray(p.sources) ? p.sources : [],
            confidence: p.confidence ?? null,
            abstained: p.abstained === true,
          },
        };
        break;
      }
      case "delta": {
        const d = parseEventData<{ text?: unknown }>(ev.event, ev.data);
        if (typeof d.text === "string" && d.text) yield { type: "delta", text: d.text };
        break;
      }
      case "agent_message":
        yield { type: "agent_message", message: parseEventData<Message>(ev.event, ev.data) };
        break;
      case "error": {
        const e = parseEventData<Partial<StreamFailure>>(ev.event, ev.data);
        yield {
          type: "error",
          failure: { detail: typeof e.detail === "string" ? e.detail : "Something went wrong", stage: e.stage ?? null },
        };
        break;
      }
    }
  }
}

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
    uploadDocument: (projectId: string, file: File, opts?: UploadOptions) => uploadDocument(base, projectId, file, opts),
    getDocument: (documentId: string, signal?: AbortSignal) =>
      getJson<ProjectDocument>(base, `/api/documents/${id(documentId)}`, signal),
    /** Removes the document's vectors, stored file and row, and drops it from every chat's document scope. */
    deleteDocument: (documentId: string) =>
      request<void>(base, `/api/documents/${id(documentId)}`, { method: "DELETE" }),

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
    /** Ask a question; yields the saved user message, sources, answer deltas and the saved answer (see ChatEvent). */
    sendMessage: (chatId: string, body: SendMessageBody, signal?: AbortSignal) => sendMessage(base, chatId, body, signal),

    pins: (signal?: AbortSignal) => getJson<PinnedItems>(base, "/api/pins", signal),
  };
}

export type Api = ReturnType<typeof createApi>;
