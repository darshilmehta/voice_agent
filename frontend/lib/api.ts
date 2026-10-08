/** Typed access to the backend. Shapes mirror backend/app/api/*.py. */

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

export class BackendError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

async function getJson<T>(base: string, path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, { signal, cache: "no-store" });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new BackendError(`Can't reach the backend at ${base}`);
  }
  if (!res.ok) throw new BackendError(`${path} returned HTTP ${res.status}`, res.status);
  return (await res.json()) as T;
}

export const fetchPublicConfig = (base: string, signal?: AbortSignal) =>
  getJson<PublicConfig>(base, "/api/config/public", signal);

export const fetchHealth = (base: string, signal?: AbortSignal) => getJson<HealthReport>(base, "/health", signal);
