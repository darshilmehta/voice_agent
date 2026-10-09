"use client";

/**
 * A chat's summary (docs/DESIGN.md §3.9) for the Summary view. It lives with the chat page rather than in the view, so
 * a summary that is being written (7 s or more on the local model, longer for long chats) keeps going when the reader
 * switches to the transcript and is there when they come back.
 *
 *   load      GET once, the first time the view is shown; "no summary yet" is a normal answer (null), not an error
 *   generate  POST, on request: the first summary, a refresh of a stale one, or the other language
 *   stale     the backend says so, or the chat got more messages since this page fetched the summary
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { BackendError, errorMessage, isAbort, type Chat, type ChatSummary, type Language } from "./api";
import { useBackend } from "./backend-context";
import { isLanguage } from "./summary-model";

export interface SummaryProblem {
  message: string;
  /** The backend can't be reached at all. */
  unreachable: boolean;
  /** The model isn't available (503): trying again later may work. */
  unavailable: boolean;
  /** The chat had nothing to summarise (422). */
  empty: boolean;
  /** The language that was being written, so "Try again" asks for that one (not the language still on screen). */
  language: Language | null;
}

interface Inner {
  /** The first read: idle until the view is shown. */
  load: "idle" | "loading" | "ready";
  summary: ChatSummary | null;
  /** `chat.message_count` when `summary` was fetched, to notice messages that arrive later. */
  fetchedAt: number;
  /** A summary is being written, in this language ("auto": whatever the backend picks). */
  writing: Language | "auto" | null;
  /** A language picked while there is no summary to switch yet. */
  choice: Language | null;
  problem: SummaryProblem | null;
  /** Where the problem came from, so the view can say what failed. */
  problemOf: "load" | "write" | null;
}

const INITIAL: Inner = { load: "idle", summary: null, fetchedAt: 0, writing: null, choice: null, problem: null, problemOf: null };

function problemFrom(err: unknown, language: Language | null = null): SummaryProblem {
  const status = err instanceof BackendError ? err.status : undefined;
  const message = errorMessage(err);
  return {
    message,
    unreachable: err instanceof BackendError && err.unreachable,
    unavailable: status === 503,
    empty: status === 422 && /nothing to summar/i.test(message),
    language,
  };
}

export interface ChatSummaryApi extends Inner {
  /** The language the Summary view shows. */
  language: Language;
  /** The summary is out of date (flagged by the backend, or the chat has grown since it was fetched). */
  stale: boolean;
  /** Messages the summary was written from, and messages the chat has now. */
  covered: number;
  now: number;
  /** Write or refresh the summary (in `language`, else the shown one). No-op while one is being written. */
  generate: (language?: Language) => void;
  /** The EN / HI switch: regenerates when a summary in the other language is shown, else remembers the choice. */
  selectLanguage: (language: Language) => void;
  /** The "Summarise" action: write one unless a current summary is already there. */
  summarise: () => void;
  /** Read it again after a failed first read. */
  reload: () => void;
}

const languageOf = (s: ChatSummary | null): Language | null => (s && isLanguage(s.language) ? s.language : null);
const chatLanguage = (c: Chat): Language | null => (isLanguage(c.language) ? c.language : null);

export function useChatSummary(chat: Chat, active: boolean): ChatSummaryApi {
  const { api } = useBackend();
  const chatId = chat.id;
  const [inner, setInner] = useState<Inner>(INITIAL);
  const innerRef = useRef(inner);
  const chatRef = useRef(chat);
  useEffect(() => {
    innerRef.current = inner;
    chatRef.current = chat;
  });
  const alive = useRef(true);
  /** Only the newest request may write its result: a read that was overtaken by a write is dropped. */
  const ticket = useRef(0);
  const started = useRef(false);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const apply = useCallback((my: number, change: (s: Inner) => Inner) => {
    if (alive.current && ticket.current === my) setInner(change);
  }, []);

  const read = useCallback(() => {
    const my = ++ticket.current;
    setInner((s) => ({ ...s, load: "loading", problem: null, problemOf: null }));
    api
      .getSummary(chatId)
      .then((summary) =>
        apply(my, (s) => ({ ...s, load: "ready", summary, fetchedAt: chatRef.current.message_count, problem: null, problemOf: null })),
      )
      .catch((err: unknown) => {
        if (isAbort(err)) return;
        apply(my, (s) => ({ ...s, load: "ready", problem: problemFrom(err), problemOf: "load" }));
      });
  }, [api, chatId, apply]);

  // The first time the view is shown.
  useEffect(() => {
    if (!active || started.current) return;
    started.current = true;
    read();
  }, [active, read]);

  const generate = useCallback(
    (language?: Language) => {
      const cur = innerRef.current;
      if (cur.writing) return;
      const c = chatRef.current;
      const lang = language ?? cur.choice ?? languageOf(cur.summary) ?? chatLanguage(c) ?? undefined;
      const my = ++ticket.current;
      started.current = true; // a write makes the first read unnecessary
      setInner((s) => ({ ...s, load: "ready", writing: lang ?? "auto", problem: null, problemOf: null }));
      api
        .createSummary(chatId, lang)
        .then((summary) =>
          apply(my, (s) => ({
            ...s,
            load: "ready",
            summary,
            fetchedAt: chatRef.current.message_count,
            writing: null,
            choice: null,
            problem: null,
            problemOf: null,
          })),
        )
        .catch((err: unknown) => {
          if (isAbort(err)) return;
          apply(my, (s) => ({
            ...s,
            load: "ready",
            writing: null,
            choice: null,
            problem: problemFrom(err, lang ?? null),
            problemOf: "write",
          }));
        });
    },
    [api, chatId, apply],
  );

  const selectLanguage = useCallback(
    (language: Language) => {
      const cur = innerRef.current;
      if (cur.writing) return;
      if (cur.summary && cur.summary.language !== language) generate(language);
      else setInner((s) => ({ ...s, choice: s.summary ? null : language }));
    },
    [generate],
  );

  const summarise = useCallback(() => {
    const cur = innerRef.current;
    const c = chatRef.current;
    if (cur.writing) return;
    const current = cur.summary && !cur.summary.stale && c.message_count <= cur.fetchedAt;
    if (!current) generate();
  }, [generate]);

  const summary = inner.summary;
  const language: Language =
    (inner.writing !== "auto" ? inner.writing : null) ?? inner.choice ?? languageOf(summary) ?? chatLanguage(chat) ?? "en";

  return {
    ...inner,
    language,
    stale: !!summary && (summary.stale || chat.message_count > inner.fetchedAt),
    // `covers_seq` is the last message the summary covers; `message_count` may be how many it was written from or how
    // many the chat had then (the contract doesn't say): the count of covered messages is the former either way, and the
    // chat's size is the larger of what we know now and what the summary reported.
    covered: summary ? summary.covers_seq || summary.message_count : 0,
    now: Math.max(chat.message_count, summary?.message_count ?? 0),
    generate,
    selectLanguage,
    summarise,
    reload: read,
  };
}
