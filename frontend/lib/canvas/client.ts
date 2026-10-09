/**
 * The canvas endpoints (contract v1). Kept apart from lib/api.ts so the canvas stays one additive module; it reuses
 * `BackendError` so failures read like every other backend call.
 *
 *   GET  /api/chats/{id}/canvas           → {panels: [Visual]}
 *   POST /api/chats/{id}/canvas/ops       {op: remove | pin | unpin | move, visual_id, position?} → the new canvas
 *   POST /api/chats/{id}/visuals          a VisualSpec (debug only) → the visual it built
 *   GET  /api/projects/{id}/overview      → {panels: [Visual], status: ready | building | none}
 */

import { BackendError, isAbort } from "../api";
import type { CanvasOp, Visual } from "./types";

export type OverviewStatus = "ready" | "building" | "none";

export interface Overview {
  panels: Visual[];
  status: OverviewStatus;
}

export interface OpRequest {
  op: CanvasOp;
  visual_id: string;
  /** For `move`: the position the panel should take. */
  position?: number;
}

const id = encodeURIComponent;

async function call(base: string, path: string, method: "GET" | "POST", body?: unknown, signal?: AbortSignal): Promise<unknown> {
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
    let detail: string | null = null;
    try {
      const j = (await res.json()) as { detail?: unknown };
      detail =
        typeof j.detail === "string"
          ? j.detail
          : Array.isArray(j.detail)
            ? j.detail.map((d: { msg?: string }) => d.msg ?? "invalid").join("; ")
            : null;
    } catch {
      // not JSON
    }
    throw new BackendError(detail ?? `${method} ${path} returned HTTP ${res.status}`, res.status);
  }
  if (res.status === 204) return null;
  try {
    return await res.json();
  } catch {
    throw new BackendError(`${method} ${path} returned something that isn't JSON`, res.status);
  }
}

/**
 * The panels of an answer. The contract reader (./types) is loaded only when there is something to read: a chat whose
 * canvas is empty, the common case, never downloads it.
 */
async function panelsOf(json: unknown): Promise<Visual[]> {
  const raw = (json as { panels?: unknown } | null)?.panels;
  if (!Array.isArray(raw) || raw.length === 0) return [];
  return (await import("./types")).normalizePanels(raw);
}

export function createCanvasApi(base: string) {
  return {
    getCanvas: async (chatId: string, signal?: AbortSignal) =>
      panelsOf(await call(base, `/api/chats/${id(chatId)}/canvas`, "GET", undefined, signal)),
    canvasOp: async (chatId: string, op: OpRequest) =>
      panelsOf(await call(base, `/api/chats/${id(chatId)}/canvas/ops`, "POST", op)),
    /** Debug only (`features.debug_panel`): build a visual from a pasted VisualSpec. */
    createVisual: async (chatId: string, spec: unknown) => {
      const json = await call(base, `/api/chats/${id(chatId)}/visuals`, "POST", spec);
      return (await import("./types")).normalizeVisual(json);
    },
    getOverview: async (projectId: string, signal?: AbortSignal): Promise<Overview> => {
      const json = (await call(base, `/api/projects/${id(projectId)}/overview`, "GET", undefined, signal)) as {
        status?: unknown;
      } | null;
      const status: OverviewStatus = json?.status === "ready" || json?.status === "building" ? json.status : "none";
      return { panels: status === "ready" ? await panelsOf(json) : [], status };
    },
  };
}

export type CanvasApi = ReturnType<typeof createCanvasApi>;
