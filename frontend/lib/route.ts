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
 *
 * Live web search (§3.7) adds two fields: `basis`, a sorted subset of ["documents", "web", "general"] naming what the
 * answer drew on (it decides the label when present), and `live_note` ("unavailable" | "failed") when live data was
 * asked for and couldn't be had.
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

/** What an answer drew on (`route.basis`, a sorted subset of these). */
export type Provenance = "documents" | "web" | "general";

const PROVENANCE: ReadonlySet<string> = new Set<Provenance>(["documents", "web", "general"]);

/**
 * `route.basis`: what the answer drew on, when the backend says. Null when it is absent or isn't a list of the three
 * known names (older messages), so callers fall back to the logic that predates it. An empty list is a real answer:
 * it drew on nothing (an acknowledgement, a refusal).
 */
export function basisSetOf(m: Routed): Set<Provenance> | null {
  const basis = m.route?.basis;
  if (!Array.isArray(basis) || !basis.every((b) => typeof b === "string" && PROVENANCE.has(b))) return null;
  return new Set(basis as Provenance[]);
}

/** Why live data is missing from an answer that asked for it: no web search here, or it didn't give results. */
export type LiveNote = "unavailable" | "failed";

export function liveNoteOf(m: Routed): LiveNote | null {
  const note = m.route?.live_note;
  return note === "unavailable" || note === "failed" ? note : null;
}

export const LIVE_NOTE_TEXT: Record<LiveNote, string> = {
  unavailable: "Live web search isn't available, so this answer has no web results.",
  failed: "Live web search didn't return results this time.",
};

/** The short phrase for any answer that includes web results; the labels below say it together with the rest. */
export const WEB_NOTE = "Includes live web results";

/**
 * What the quiet label above an answer says, by what the answer drew on. `mixed` and `general` are the labels that
 * predate web search and keep their words; a web answer never says "documents + general knowledge" unless it also
 * says where the web results come in.
 */
export type Basis =
  | "general"
  | "mixed"
  | "web"
  | "documents_web"
  | "web_general"
  | "documents_web_general";

export const BASIS_LABEL: Record<Basis, string> = {
  general: "General knowledge, not from your documents",
  mixed: "From your documents + general knowledge",
  web: "Live web results",
  documents_web: "From your documents + live web results",
  web_general: "Live web results + general knowledge",
  documents_web_general: "From your documents + live web results + general knowledge",
};

/** The label for a set of provenances; null when the documents alone (or nothing) were used. */
export function labelFor(set: ReadonlySet<Provenance>): Basis | null {
  const documents = set.has("documents");
  const general = set.has("general");
  if (set.has("web")) {
    return documents && general ? "documents_web_general" : documents ? "documents_web" : general ? "web_general" : "web";
  }
  return general ? (documents ? "mixed" : "general") : null;
}

/** What the page can see an answer cites: passages of the documents and web results. */
export interface Cited {
  documents: boolean;
  web: boolean;
}

/**
 * Where an answer's content comes from, when the user should be told. `route.basis` says it when present and decides
 * it ("web" is also added when a citation is a web result). Without it, the logic that predates it: "general" for
 * general-knowledge answers (and mixed ones that carry a `general_note`, or that cite no document, so nothing in them
 * is from the documents as far as the page can show) and "mixed" for a mixed answer that cites a document; web
 * citations add "live web results" to that, and an answer with web results never claims general knowledge it can't
 * show. Null for everything else.
 */
export function basisOf(m: Routed, cited: Cited): Basis | null {
  const declared = basisSetOf(m);
  if (declared) {
    const set = new Set(declared);
    if (cited.web) set.add("web");
    return labelFor(set);
  }
  if (cited.web) return cited.documents ? "documents_web" : "web";
  const kind = answerKindOf(m);
  if (kind === "general") return "general";
  if (kind === "mixed") return generalNoteOf(m) !== null || !cited.documents ? "general" : "mixed";
  return null;
}

/** The label involves live web results. */
export const isWebBasis = (b: Basis | null): boolean =>
  b === "web" || b === "documents_web" || b === "web_general" || b === "documents_web_general";

/** Short acknowledgements and chit-chat: a plain, light bubble. */
export const isBrief = (kind: AnswerKind | null): boolean => kind === "ack" || kind === "conversation";

/**
 * Whether source chips can belong with this kind of reply. Answers that draw on the documents have them, and so can a
 * general answer, which may have used the web (only its web results: `sourcesToShow`); an acknowledgement or a
 * clarifying question must never show the sources of an earlier turn. Messages with no route keep their citations.
 */
export function showsSources(kind: AnswerKind | null): boolean {
  return kind === null || kind === "grounded" || kind === "mixed" || kind === "resume" || kind === "general";
}

/**
 * The sources to list for a reply of this kind: all of them where the documents are in play, only the web results for
 * a general answer (it never shows a document the answer didn't use, so a general answer that didn't use the web has
 * none), nothing for the rest.
 */
export function sourcesToShow<T extends { kind: string }>(kind: AnswerKind | null, refs: readonly T[]): T[] {
  if (kind === "general") return refs.filter((r) => r.kind === "web");
  return showsSources(kind) ? refs.slice() : [];
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
 * The visual an answer added to the chat's canvas (`route.visual_id`, docs/DESIGN.md §12.1): the transcript says
 * "Chart added". The backend writes it onto the saved message once the visual is ready, which is usually after the
 * answer arrived, so a live turn patches it in from the `visual` event (`withVisual`).
 */
export function visualIdOf(m: Routed): string | null {
  const id = m.route?.visual_id;
  return typeof id === "string" && id ? id : null;
}

/** `m` with the visual its turn produced (a `visual {phase: "ready"}` event), as the backend saves it. */
export function withVisual(m: Message, visualId: string): Message {
  return { ...m, route: { ...(m.route ?? {}), visual_id: visualId, visual_status: "ready" } };
}

/** The answer's visual was withdrawn after it was shown (a draft the planner found no table for, §12.1): no "Chart
 * added" any more. Other messages are returned as they are. */
export function withoutVisual(m: Message, visualId: string): Message {
  if (m.route?.visual_id !== visualId) return m;
  return { ...m, route: { ...m.route, visual_id: null, visual_status: "failed" } };
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
