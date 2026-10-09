/**
 * Saving a transcript export: the server's file name (from Content-Disposition) and a blob download.
 *
 * Content-Disposition carries a Hindi title as `filename*=UTF-8''%E0%A4…` (RFC 8187) next to an ASCII `filename=`
 * fallback; the starred form wins. A browser can only read the header on a cross-origin response when the backend
 * lists it in `Access-Control-Expose-Headers`; when it can't, the file is named from the chat's title instead.
 */

/** Characters no file system accepts in a name (and control characters). */
const UNSAFE = /[\\/:*?"<>|\u0000-\u001f]+/g;

function cleanName(name: string): string | null {
  const base = name.trim().split(/[\\/]/).pop() ?? "";
  return base.replace(/[\u0000-\u001f]/g, "").trim() || null;
}

/** The file name in a Content-Disposition header, or null when it has none. Never contains a path separator. */
export function filenameFromDisposition(header: string | null | undefined): string | null {
  if (!header) return null;

  const star = /filename\*\s*=\s*([^;]*)/i.exec(header)?.[1]?.trim();
  if (star) {
    const parts = /^([\w-]+)'[^']*'(.*)$/.exec(star);
    const raw = (parts ? parts[2] : star).replace(/^"|"$/g, "");
    try {
      // Only UTF-8 is required by the RFC; anything else is decoded the same way and is at worst a little off.
      const name = cleanName(decodeURIComponent(raw));
      if (name) return name;
    } catch {
      // malformed percent-encoding: use the plain filename
    }
  }

  const plain = /filename\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^;]*))/i.exec(header);
  if (!plain) return null;
  const value = plain[1] !== undefined ? plain[1].replace(/\\(.)/g, "$1") : (plain[2] ?? "").trim();
  return cleanName(value);
}

const EXTENSION = { md: "md", json: "json" } as const;

/** A file name made from the chat's title, for when the server's can't be read: "FY24 margins.md". */
export function fallbackFilename(title: string, format: keyof typeof EXTENSION): string {
  const stem = title.replace(UNSAFE, " ").replace(/\s+/g, " ").trim().slice(0, 80).trim();
  return `${stem || "transcript"}.${EXTENSION[format]}`;
}

/** Hands a blob to the browser as a download. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  // The download has started by the next task; keep the URL alive a little longer for slow disks.
  window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
