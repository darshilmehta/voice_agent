/** Typed access to the backend. Shapes mirror backend/app/api/*.py and backend/app/domain/projects.py. */

import { filenameFromDisposition } from "./download";
import { readSse } from "./sse";
import { normalizeSummary } from "./summary-model";
import { parseTool, type ToolEvent } from "./web-search";

export type Language = "en" | "hi";

export interface PublicConfig {
  app_name: string;
  version: string;
  profile: "local" | "cloud";
  client: { app_title: string; languages: Language[]; default_language: Language };
  auth: { provider: string; issuer_url: string | null; client_id: string | null; audience: string | null };
  features: { voice: boolean; web_search: boolean; debug_panel: boolean };
  limits: { max_upload_mb: number; allowed_extensions: string[] };
  /** The browser's side of noisy rooms (docs/DESIGN.md §3.10): its denoiser and noise-floor gate. Absent on older backends. */
  voice_input?: PublicVoiceInput;
}

/** Mirrors backend/app/api/public_config.py `PublicVoiceInput` (from `vad` and `voice.noise`). */
export interface PublicVoiceInput {
  denoise: "rnnoise" | "off";
  adaptive_gating: boolean;
  threshold: number;
  min_speech_ms: number;
  floor_window_ms: number;
  floor_percentile: number;
  start_snr_db: number;
  quiet_floor_dbfs: number;
  loud_floor_dbfs: number;
  noisy_threshold: number;
  noisy_min_speech_ms: number;
  assumed_user_dbfs: number;
  assumed_margin_db: number;
  far_field_hard_db: number;
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
 * A source an answer cites: the answer text marks it as `[S1]`, `[S2]`, … for a passage of the documents and `[W1]`,
 * `[W2]`, … for a live web result (docs/DESIGN.md §3.7). Every field is optional in the type because messages saved
 * before typed citations hold free-form JSON (e.g. `document_id`, `page`, `chunk_id`); read them through
 * `lib/citations.ts`, which tolerates both. A citation without `kind` is a document passage.
 */
export interface Citation {
  source_id?: string;
  document_id?: string;
  /** Web results: the site. */
  filename?: string;
  page_start?: number | null;
  page_end?: number | null;
  chunk_id?: string;
  /**
   * Where in the document the passage is: its heading path ("4. Travel > 4.2 Domestic > 4.2.1 Hotels"). What locates a
   * DOCX passage, which has no pages; a page wins where there is one. Absent or null for a passage without headings
   * and for web results.
   */
  section?: string | null;
  /** Up to ~300 characters of the cited passage (web results: of the result's text). */
  snippet?: string;
  /** Older messages: a single page. */
  page?: number;
  /** "web" for a live web result; absent (or "document") for a passage of the user's documents. */
  kind?: "document" | "web";
  /** Web results only: the page's address, title, site and publication date (when the engine gave one). */
  url?: string | null;
  title?: string | null;
  site?: string | null;
  published?: string | null;
  [key: string]: unknown;
}

/** A web result as the backend sends it (in `sources`, `tool results`, saved `citations`). */
export interface WebCitation extends Citation {
  source_id: string;
  kind: "web";
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

// ------------------------------------------------------------------ summaries, export, titles (docs/DESIGN.md §3.9)

/** A document page range a summary key point rests on, or (`web`) a web site it drew on: no pages, no link. */
export interface SummarySource {
  document_id: string | null;
  /** Web sources: the site. */
  filename: string;
  page_start: number | null;
  page_end: number | null;
  /** Documents without pages (DOCX): the heading path of the cited passage. */
  section?: string | null;
  /** A live web result (the backend sends `document_id: ""` and the site as `filename`). */
  web?: boolean;
}

export interface SummaryKeyPoint {
  text: string;
  sources: SummarySource[];
}

export interface UnansweredQuestion {
  question: string;
  /** `seq` of the user message that asked it; null when the summary can't say. */
  message_seq: number | null;
}

/** GET/POST /api/chats/{id}/summary. The UI renders these fields; `content` is the same summary as markdown. */
export interface ChatSummary {
  id: string;
  chat_id: string;
  language: Language | (string & {});
  overview: string;
  key_points: SummaryKeyPoint[];
  unanswered_questions: UnansweredQuestion[];
  follow_ups: string[];
  content: string;
  /** The last message (`seq`) the summary covers. */
  covers_seq: number;
  /** How many messages the summary was written from. */
  message_count: number;
  /** The chat has newer messages than the summary covers. */
  stale: boolean;
  model: string;
  created_at: Timestamp;
}

export type ExportFormat = "md" | "json";

export interface ExportedTranscript {
  blob: Blob;
  /** From the response's Content-Disposition; null when the browser can't read that header (CORS) or it has none. */
  filename: string | null;
}

/** The placeholder title a new chat has until the backend writes one from the first answer. */
export const PLACEHOLDER_TITLE = "New chat";

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

/**
 * POST /api/chats/{id}/messages streams these, in this order (error can replace any step after user_message). A turn
 * that searches the web also streams `tool` events (lib/web-search.ts): `start` first, `results` as web results arrive,
 * then `done` / `timeout` / `failed`, which can come in the middle of the answer's deltas. Only `agent_message` (or
 * `error`) ends the turn.
 */
export type ChatEvent =
  | { type: "user_message"; message: Message }
  | { type: "tool"; tool: ToolEvent }
  | { type: "sources"; payload: SourcesPayload }
  | { type: "delta"; text: string }
  | { type: "agent_message"; message: Message }
  | { type: "error"; failure: StreamFailure }
  /** A visual preparing / ready / failed, or the whole canvas (`event: visual` / `event: canvas`): read by lib/canvas. */
  | { type: "canvas"; name: "visual" | "canvas"; data: unknown };

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

  /** 409: the request conflicts with the chat's state (a title the user set). */
  get conflict(): boolean {
    return this.status === 409;
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

/** GET /api/chats/{id}/export: the transcript as a file. The body is read whole (it is a few hundred KB at most). */
async function exportChat(base: string, chatId: string, format: ExportFormat, signal?: AbortSignal): Promise<ExportedTranscript> {
  const path = `/api/chats/${id(chatId)}/export?format=${format}`;
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, { signal, cache: "no-store" });
  } catch (err) {
    if (isAbort(err)) throw err;
    throw new BackendError(`Can't reach the backend at ${base}`);
  }
  if (!res.ok) {
    const detail = await readDetail(res);
    throw new BackendError(detail ?? `GET ${path} returned HTTP ${res.status}`, res.status);
  }
  let blob: Blob;
  try {
    blob = await res.blob();
  } catch (err) {
    if (isAbort(err)) throw err;
    throw new BackendError("The connection to the backend was lost while the file was downloading.");
  }
  return { blob, filename: filenameFromDisposition(res.headers.get("content-disposition")) };
}

/** The 404 that means "no summary yet" (the chat exists), as opposed to an unknown chat. */
const NO_SUMMARY_YET = /^summary\b/i;

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
      case "tool": {
        // Progress of the web search, not the answer: one that can't be read is skipped rather than failing the turn.
        const tool = parseTool(parseJson(ev.data));
        if (tool) yield { type: "tool", tool };
        break;
      }
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
      case "visual":
      case "canvas": {
        // Visuals are optional: a payload that can't be read is skipped, never an error for the answer.
        let data: unknown = null;
        try {
          data = JSON.parse(ev.data);
        } catch {
          break;
        }
        yield { type: "canvas", name: ev.event, data };
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

    /**
     * Write a new automatic title from the chat's messages. Throws BackendError: 409 when the user chose the current
     * title (pass `force` to replace it anyway), 422 when there is no message yet, 503 when the model couldn't
     * produce one.
     */
    regenerateTitle: (chatId: string, opts: { force?: boolean } = {}) =>
      request<Chat>(base, `/api/chats/${id(chatId)}/title:regenerate${opts.force ? "?force=true" : ""}`, { method: "POST" }),

    /** The stored summary, or null when none was written yet. */
    getSummary: async (chatId: string, signal?: AbortSignal): Promise<ChatSummary | null> => {
      try {
        return normalizeSummary(await getJson<unknown>(base, `/api/chats/${id(chatId)}/summary`, signal));
      } catch (err) {
        if (err instanceof BackendError && err.notFound && NO_SUMMARY_YET.test(err.message)) return null;
        throw err;
      }
    },
    /**
     * Write (or refresh) the summary; slow, because the local model reads the whole chat. An unchanged chat in the
     * same language returns the stored one at once. 422: nothing to summarise, or a bad language; 503: no model.
     */
    createSummary: async (chatId: string, language?: Language, signal?: AbortSignal): Promise<ChatSummary> =>
      normalizeSummary(
        await request<unknown>(base, `/api/chats/${id(chatId)}/summary${language ? `?language=${language}` : ""}`, {
          method: "POST",
          signal,
        }),
      ),
    exportChat: (chatId: string, format: ExportFormat, signal?: AbortSignal) => exportChat(base, chatId, format, signal),

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
