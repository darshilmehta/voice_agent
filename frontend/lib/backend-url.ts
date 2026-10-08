/**
 * Where the browser reaches the backend. Read from BACKEND_URL on every request (server side), never at
 * build time, so one frontend build runs against any backend (docs/DESIGN.md §6.5).
 */
const DEFAULT_BACKEND_URL = "http://localhost:8000";

export function backendUrl(): string {
  const raw = process.env.BACKEND_URL?.trim() || DEFAULT_BACKEND_URL;
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`BACKEND_URL is not a valid URL: ${raw}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`BACKEND_URL must be http(s): ${raw}`);
  }
  return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
}
