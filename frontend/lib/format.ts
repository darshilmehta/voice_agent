/** Small formatting helpers for times, sizes and search matching. Everything renders client-side. */

/** Language names in their own script, for chips. */
export const LANGUAGE_NAMES: Record<string, string> = { en: "English", hi: "हिन्दी" };

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;

/** Compact age for dense lists: "now", "5m", "3h", "2d", "3w", then a date ("12 Mar", "12 Mar 2025"). */
export function shortAge(iso: string, now = Date.now()): string {
  const t = Date.parse(iso);
  const diff = Math.max(0, now - t);
  if (diff < MINUTE) return "now";
  if (diff < HOUR) return `${Math.floor(diff / MINUTE)}m`;
  if (diff < DAY) return `${Math.floor(diff / HOUR)}h`;
  if (diff < WEEK) return `${Math.floor(diff / DAY)}d`;
  if (diff < 5 * WEEK) return `${Math.floor(diff / WEEK)}w`;
  return shortDate(iso, now);
}

const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

/** "just now", "5 minutes ago", "yesterday", "3 weeks ago", or a date for anything older. */
export function relativeTime(iso: string, now = Date.now()): string {
  const t = Date.parse(iso);
  const diff = t - now;
  const abs = Math.abs(diff);
  if (abs < MINUTE) return "just now";
  if (abs < HOUR) return rtf.format(Math.round(diff / MINUTE), "minute");
  if (abs < DAY) return rtf.format(Math.round(diff / HOUR), "hour");
  if (abs < WEEK) return rtf.format(Math.round(diff / DAY), "day");
  if (abs < 5 * WEEK) return rtf.format(Math.round(diff / WEEK), "week");
  return `on ${shortDate(iso, now)}`;
}

export function shortDate(iso: string, now = Date.now()): string {
  const d = new Date(iso);
  const sameYear = d.getFullYear() === new Date(now).getFullYear();
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: sameYear ? undefined : "numeric" });
}

export function fullDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function clockTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

/** "Today", "Yesterday", or a weekday and date, for transcript day dividers. */
export function dayLabel(iso: string, now = Date.now()): string {
  const d = new Date(iso);
  const startOf = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((startOf(new Date(now)) - startOf(d)) / DAY);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  const sameYear = d.getFullYear() === new Date(now).getFullYear();
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: sameYear ? undefined : "numeric",
  });
}

export const dayKey = (iso: string) => new Date(iso).toDateString();

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** Case-insensitive search key. Kept length-preserving for the common case so match offsets map back. */
export const searchKey = (s: string) => s.normalize("NFC").toLocaleLowerCase();

export function matches(text: string, query: string): boolean {
  return !query || searchKey(text).includes(searchKey(query));
}
