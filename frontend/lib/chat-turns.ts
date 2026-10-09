"use client";

/**
 * Questions asked in this page view and their streaming answers ("turns"), shown after the saved transcript.
 *
 * One turn at a time. A turn goes sending → searching (the user message is saved) → answering (sources arrived,
 * text streams) → done (the saved agent message replaces the streamed text). It can end failed (an `error` event, a
 * refused request, or a stream that closed early) or stopped (the user aborted; what arrived is kept). Answer
 * deltas are applied once per animation frame, so fast token streams don't re-render per token.
 *
 * A turn that searches the web (docs/DESIGN.md §3.7) also gets `tool` events, kept in `turn.web`: the search (for the
 * "Searching the web…" badge and the note of what was searched) and the web results that arrived, so a `[W3]` that
 * only a late `tool results` carries still has its source. The search is over when sources arrive, the answer is
 * saved, the turn fails or is stopped: the error and abort paths may send no terminal `tool` event.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { errorMessage, isAbort, type Api, type Language, type Message, type SourcesPayload, type StreamFailure } from "./api";
import { splitCitations } from "./citations";
import { applyTool, endSearch, NO_WEB, type WebTurn } from "./web-search";

export type TurnPhase = "sending" | "searching" | "answering" | "done" | "failed" | "stopped";

export interface TurnFailure extends StreamFailure {
  /** The user message was saved before it failed (so a retry asks again as a new message). */
  saved: boolean;
}

export interface Turn {
  key: string;
  text: string;
  language: Language | null;
  /** When it was sent (ISO), for the day divider and time until the saved message arrives. */
  startedAt: string;
  phase: TurnPhase;
  user: Message | null;
  sources: SourcesPayload | null;
  /** Streamed answer so far. */
  answer: string;
  agent: Message | null;
  failure: TurnFailure | null;
  /** Retried as a newer turn: hide this one's Retry button. */
  retried: boolean;
  /** The live web search of this turn and the web results it brought. */
  web: WebTurn;
}

export const isActive = (t: Turn) => t.phase === "sending" || t.phase === "searching" || t.phase === "answering";

/** Answer text for screen readers: "[S1]" markers removed. */
export const spokenText = (text: string) =>
  splitCitations(text)
    .map((p) => (p.kind === "text" ? p.text : ""))
    .join("")
    .replace(/\s+([.,;:!?।])/g, "$1")
    .trim();

let counter = 0;

export interface ChatTurnsOptions {
  /** `features.web_search`: false keeps the search's badge and note off (web results still resolve their markers). */
  webSearch?: boolean;
}

export function useChatTurns(api: Api, chatId: string, onSettled?: () => void, options: ChatTurnsOptions = {}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [announcement, setAnnouncement] = useState("");
  const controller = useRef<AbortController | null>(null);
  const buffered = useRef<{ key: string; text: string } | null>(null);
  const frame = useRef<number | null>(null);
  const onSettledRef = useRef(onSettled);
  const webSearchRef = useRef(options.webSearch === true);
  useEffect(() => {
    onSettledRef.current = onSettled;
    webSearchRef.current = options.webSearch === true;
  });

  const update = useCallback((key: string, change: (t: Turn) => Turn) => {
    setTurns((list) => list.map((t) => (t.key === key ? change(t) : t)));
  }, []);

  const flush = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    const pending = buffered.current;
    buffered.current = null;
    if (pending) {
      update(pending.key, (t) => ({ ...t, phase: t.phase === "searching" || t.phase === "sending" ? "answering" : t.phase, answer: t.answer + pending.text }));
    }
  }, [update]);

  const run = useCallback(
    async (key: string, text: string, language: Language | null) => {
      const ctrl = new AbortController();
      controller.current = ctrl;
      let saved = false;
      let finished = false;
      try {
        for await (const ev of api.sendMessage(chatId, { text, language }, ctrl.signal)) {
          if (ev.type === "delta") {
            buffered.current = { key, text: (buffered.current?.text ?? "") + ev.text };
            if (frame.current === null) frame.current = requestAnimationFrame(flush);
            continue;
          }
          flush();
          if (ev.type === "user_message") {
            saved = true;
            update(key, (t) => ({ ...t, user: ev.message, phase: t.phase === "sending" ? "searching" : t.phase }));
          } else if (ev.type === "tool") {
            update(key, (t) => ({ ...t, web: applyTool(t.web, ev.tool, webSearchRef.current) }));
            if (ev.tool.phase === "start" && webSearchRef.current) setAnnouncement("Searching the web…");
          } else if (ev.type === "sources") {
            update(key, (t) => ({
              ...t,
              sources: ev.payload,
              web: endSearch(t.web),
              phase: t.phase === "done" ? t.phase : "answering",
            }));
          } else if (ev.type === "agent_message") {
            finished = true;
            update(key, (t) => ({ ...t, agent: ev.message, phase: "done", web: endSearch(t.web) }));
            setAnnouncement(`Answer: ${spokenText(ev.message.text)}`);
          } else if (ev.type === "error") {
            finished = true;
            update(key, (t) => ({ ...t, phase: "failed", failure: { ...ev.failure, saved }, web: endSearch(t.web) }));
            setAnnouncement(`Couldn't answer: ${ev.failure.detail}`);
          }
        }
        flush();
        if (!finished) {
          const detail = "The connection closed before the answer finished.";
          update(key, (t) => ({ ...t, phase: "failed", failure: { detail, stage: null, saved }, web: endSearch(t.web) }));
          setAnnouncement(`Couldn't answer: ${detail}`);
        }
      } catch (err) {
        flush();
        if (isAbort(err)) {
          update(key, (t) => ({ ...t, phase: "stopped", web: endSearch(t.web) }));
          setAnnouncement("Stopped.");
        } else {
          const detail = errorMessage(err);
          update(key, (t) => ({ ...t, phase: "failed", failure: { detail, stage: null, saved }, web: endSearch(t.web) }));
          setAnnouncement(`Couldn't answer: ${detail}`);
        }
      } finally {
        if (controller.current === ctrl) controller.current = null;
        onSettledRef.current?.();
      }
    },
    [api, chatId, flush, update],
  );

  const start = useCallback(
    (text: string, language: Language | null) => {
      const key = `turn-${++counter}`;
      const turn: Turn = {
        key,
        text,
        language,
        startedAt: new Date().toISOString(),
        phase: "sending",
        user: null,
        sources: null,
        answer: "",
        agent: null,
        failure: null,
        retried: false,
        web: NO_WEB,
      };
      setTurns((list) => [...list, turn]);
      void run(key, text, language);
    },
    [run],
  );

  /** Ask a question. False while another answer is still streaming. */
  const send = useCallback(
    (text: string, language: Language | null) => {
      if (controller.current) return false;
      start(text, language);
      return true;
    },
    [start],
  );

  /** Abort the streaming answer; whatever arrived stays on screen, marked stopped. */
  const stop = useCallback(() => controller.current?.abort(), []);

  /**
   * Ask a failed turn's question again. If its user message was saved, the transcript keeps it and the question is
   * asked again as a new turn; if the request never got that far, the same turn is reused.
   */
  const retry = useCallback(
    (key: string) => {
      if (controller.current) return;
      const turn = turns.find((t) => t.key === key);
      if (!turn || turn.phase !== "failed") return;
      if (turn.failure?.saved || turn.user) {
        update(key, (t) => ({ ...t, retried: true }));
        start(turn.text, turn.language);
      } else {
        update(key, (t) => ({
          ...t,
          phase: "sending",
          failure: null,
          answer: "",
          sources: null,
          web: NO_WEB,
          startedAt: new Date().toISOString(),
        }));
        void run(key, turn.text, turn.language);
      }
    },
    [turns, update, start, run],
  );

  // Leaving the chat stops its answer.
  useEffect(
    () => () => {
      controller.current?.abort();
      if (frame.current !== null) cancelAnimationFrame(frame.current);
    },
    [],
  );

  return { turns, busy: turns.some(isActive), send, stop, retry, announcement };
}
