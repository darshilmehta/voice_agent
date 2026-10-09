/**
 * What the router decided about a turn (docs/DESIGN.md §3.4), as the transcript and the voice stage use it. Every
 * agent message's `route` carries `intent`, `answer` (what kind of reply it is), `general_note` and, for follow-ups
 * and corrections, `rewritten_query`. Messages saved before the router existed have none of these and read as they
 * always did; nothing here throws on odd input.
 *
 *   grounded       answered from the documents (cites them)
 *   mixed          documents + general knowledge; `general_note: "not_covered"` when the documents had nothing
 *   general        general knowledge, not from the documents
 *   conversation   chit-chat      ack  a short acknowledgement      resume  carries on the earlier answer
 *   clarification  the agent asked the user to clarify              silent  nothing to say
 */

import type { Message } from "./api";

export type AnswerKind = "grounded" | "mixed" | "general" | "conversation" | "ack" | "resume" | "clarification" | "silent";

const ANSWER_KINDS: ReadonlySet<string> = new Set<AnswerKind>([
  "grounded",
  "mixed",
  "general",
  "conversation",
  "ack",
  "resume",
  "clarification",
  "silent",
]);

type Routed = Pick<Message, "route">;

/** The kind of reply an agent message is, or null when its route doesn't say (older messages, unknown values). */
export function answerKindOf(m: Routed): AnswerKind | null {
  const answer = m.route?.answer;
  return typeof answer === "string" && ANSWER_KINDS.has(answer) ? (answer as AnswerKind) : null;
}

/** Why a mixed answer leans on general knowledge ("not_covered": the documents had nothing on the question). */
export function generalNoteOf(m: Routed): string | null {
  const note = m.route?.general_note;
  return typeof note === "string" && note.trim() ? note : null;
}

export type Basis = "general" | "mixed";

export const BASIS_LABEL: Record<Basis, string> = {
  general: "General knowledge, not from your documents",
  mixed: "From your documents + general knowledge",
};

/**
 * Where an answer's content comes from, when the user should be told: "general" for general-knowledge answers (and
 * mixed ones that carry a `general_note`, or that cite nothing, so nothing in them is from the documents as far as
 * the page can show), "mixed" for a mixed answer that cites sources. Null for everything else.
 */
export function basisOf(m: Routed, cites: boolean): Basis | null {
  const kind = answerKindOf(m);
  if (kind === "general") return "general";
  if (kind === "mixed") return generalNoteOf(m) !== null || !cites ? "general" : "mixed";
  return null;
}

/** Short acknowledgements and chit-chat: a plain, light bubble. */
export const isBrief = (kind: AnswerKind | null): boolean => kind === "ack" || kind === "conversation";

/**
 * Whether source chips belong with this kind of reply. Only answers that draw on the documents have any; a general
 * answer, an acknowledgement or a clarifying question must never show the sources of an earlier turn. Messages
 * with no route keep their citations.
 */
export function showsSources(kind: AnswerKind | null): boolean {
  return kind === null || kind === "grounded" || kind === "mixed" || kind === "resume";
}

// ------------------------------------------------------------------ "Understood as: …"

/** Compare questions without caring about case, spacing or punctuation. */
const sameText = (a: string, b: string): boolean => {
  const norm = (s: string) =>
    s
      .normalize("NFKC")
      .toLocaleLowerCase()
      .replace(/[\s.,;:!?'"“”‘’()।¿¡-]+/g, " ")
      .trim();
  return norm(a) === norm(b);
};

/** The question the router worked from, when it isn't what was said or typed. */
export function rewrittenQueryOf(agent: Routed, userText: string): string | null {
  const q = agent.route?.rewritten_query;
  if (typeof q !== "string") return null;
  const text = q.trim();
  return text && !sameText(text, userText) ? text : null;
}

/**
 * For each user message that the router understood differently ("no, I meant FY25" became "What was revenue in
 * FY25?"; Whisper heard the wrong word), the question it worked from, by user message id. A user message belongs to
 * the first agent message after it, before the next user message.
 */
export function understoodAsByUser(messages: readonly Message[]): Map<string, string> {
  const out = new Map<string, string>();
  const ordered = [...messages].sort((a, b) => a.seq - b.seq);
  let waiting: Message | null = null;
  for (const m of ordered) {
    if (m.role === "user") {
      waiting = m;
    } else if (m.role === "agent" && waiting) {
      const q = rewrittenQueryOf(m, waiting.text);
      if (q) out.set(waiting.id, q);
      waiting = null;
    }
  }
  return out;
}
