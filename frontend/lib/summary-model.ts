/**
 * Chat summaries as the UI needs them (docs/DESIGN.md §3.9): reading the backend's JSON without trusting it, and
 * the words that say how current a summary is. Pure, so it can be checked without a browser.
 */

import type { ChatSummary, Language, SummarySource } from "./api";

const str = (v: unknown, fallback = ""): string => (typeof v === "string" ? v : fallback);
const int = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" ? (v as Record<string, unknown>) : {});

/** A summary with every list present and every field typed, whatever the backend left out. */
export function normalizeSummary(raw: unknown): ChatSummary {
  const r = obj(raw);
  return {
    id: str(r.id),
    chat_id: str(r.chat_id),
    language: str(r.language, "en"),
    overview: str(r.overview).trim(),
    key_points: list(r.key_points)
      .map((k) => {
        const o = obj(k);
        const sources: SummarySource[] = list(o.sources).map((s) => {
          const so = obj(s);
          const pageStart = int(so.page_start);
          const pageEnd = int(so.page_end);
          // A web result a key point drew on: the backend sends an explicitly empty `document_id` (not a missing
          // or null one) with the site as `filename`, no pages and no link (docs/DESIGN.md §3.7).
          const web = so.document_id === "" && pageStart === null && pageEnd === null;
          return {
            document_id: str(so.document_id) || null,
            filename: str(so.filename, web ? "Web" : "Document") || (web ? "Web" : "Document"),
            page_start: pageStart,
            page_end: pageEnd,
            ...(web ? { web: true } : {}),
          };
        });
        return { text: str(o.text).trim(), sources };
      })
      .filter((k) => k.text),
    unanswered_questions: list(r.unanswered_questions)
      .map((q) => {
        const o = obj(q);
        return { question: str(o.question).trim(), message_seq: int(o.message_seq) };
      })
      .filter((q) => q.question),
    follow_ups: list(r.follow_ups)
      .map((f) => str(f).trim())
      .filter(Boolean),
    content: str(r.content),
    covers_seq: int(r.covers_seq) ?? 0,
    message_count: int(r.message_count) ?? 0,
    stale: r.stale === true,
    model: str(r.model),
    created_at: str(r.created_at),
  };
}

/** The summary has nothing to show: the model returned an empty overview and no lists. */
export function isBlank(s: ChatSummary): boolean {
  return !s.overview && s.key_points.length === 0 && s.unanswered_questions.length === 0 && s.follow_ups.length === 0;
}

export const isLanguage = (v: unknown): v is Language => v === "en" || v === "hi";

/**
 * "Out of date: covers 8 of 14 messages". `covered` is how many messages the summary was written from and `now` how
 * many the chat has; when those don't show a gap (the backend flagged it stale for another reason) it says so without
 * numbers.
 */
export function outOfDateText(covered: number, now: number): string {
  return now > covered && covered > 0
    ? `Out of date: covers ${covered.toLocaleString()} of ${now.toLocaleString()} messages`
    : "Out of date: the chat has new messages";
}
