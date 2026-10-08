"use client";

/**
 * A chat's transcript (docs/DESIGN.md §3.9): opens at the latest messages and loads earlier pages as you scroll up
 * (or with the button), keeping your place. Bubbles distinguish you and the agent, voice and text; agent answers
 * show their citations and, when cut off by a barge-in, what was actually heard.
 */

import { Fragment, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { BackendError, errorMessage, isAbort, type Chat, type Citation, type Message, type ProjectDocument } from "@/lib/api";
import { useBackend } from "@/lib/backend-context";
import { clockTime, dayKey, dayLabel, fullDateTime, plural } from "@/lib/format";

import { Icon } from "./Icon";
import { BackendDown, EmptyState } from "./States";

const PAGE = 50;

const LANGUAGE_LABEL: Record<string, string> = { en: "English", hi: "Hindi" };

interface TranscriptState {
  status: "loading" | "ready" | "error";
  items: Message[];
  total: number;
  hasMore: boolean;
  cursor: number | null;
  error: string | null;
  unreachable: boolean;
  loadingEarlier: boolean;
  earlierError: string | null;
}

const INITIAL: TranscriptState = {
  status: "loading",
  items: [],
  total: 0,
  hasMore: false,
  cursor: null,
  error: null,
  unreachable: false,
  loadingEarlier: false,
  earlierError: null,
};

export function Transcript({ chat, docsById }: { chat: Chat; docsById: Record<string, ProjectDocument> }) {
  const { api, config } = useBackend();
  const [s, setS] = useState<TranscriptState>(INITIAL);
  const [attempt, setAttempt] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const startRef = useRef<HTMLParagraphElement>(null);
  const earlierButtonRef = useRef<HTMLButtonElement>(null);
  const scrollToEnd = useRef(false);
  const anchor = useRef<{ height: number; top: number; refocus: boolean } | null>(null);
  const countRef = useRef(chat.message_count);
  useEffect(() => {
    countRef.current = chat.message_count;
  }, [chat.message_count]);

  const chatId = chat.id;
  const showDebug = config?.features.debug_panel ?? false;

  // Open at the latest messages: before = message_count + 1. If messages arrived since the chat was loaded, the
  // page's `total` says so and we ask again from the true end.
  useEffect(() => {
    const ctrl = new AbortController();
    setS(INITIAL);
    (async () => {
      try {
        let before = countRef.current + 1;
        let page = await api.listMessages(chatId, { before, limit: PAGE }, ctrl.signal);
        if (page.total + 1 > before) {
          before = page.total + 1;
          page = await api.listMessages(chatId, { before, limit: PAGE }, ctrl.signal);
        }
        scrollToEnd.current = true;
        setS({
          ...INITIAL,
          status: "ready",
          items: page.items,
          total: page.total,
          hasMore: page.has_more,
          cursor: page.next_cursor,
        });
      } catch (err) {
        if (isAbort(err) || ctrl.signal.aborted) return;
        setS({
          ...INITIAL,
          status: "error",
          error: errorMessage(err),
          unreachable: err instanceof BackendError && err.unreachable,
        });
      }
    })();
    return () => ctrl.abort();
  }, [api, chatId, attempt]);

  const busy = useRef(false);
  const loadEarlier = useCallback(
    async (fromButton = false) => {
      if (busy.current || !s.hasMore || s.cursor === null) return;
      busy.current = true;
      const el = scrollRef.current;
      anchor.current = el ? { height: el.scrollHeight, top: el.scrollTop, refocus: fromButton } : null;
      setS((x) => ({ ...x, loadingEarlier: true, earlierError: null }));
      try {
        const page = await api.listMessages(chatId, { before: s.cursor, limit: PAGE });
        setS((x) => ({
          ...x,
          items: [...page.items, ...x.items],
          total: page.total,
          hasMore: page.has_more,
          cursor: page.next_cursor,
          loadingEarlier: false,
        }));
      } catch (err) {
        anchor.current = null;
        setS((x) => ({ ...x, loadingEarlier: false, earlierError: errorMessage(err) }));
      } finally {
        busy.current = false;
      }
    },
    [api, chatId, s.hasMore, s.cursor],
  );

  // Keep the reader's place: scroll to the end after the first page, and hold position when older pages arrive.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (scrollToEnd.current) {
      scrollToEnd.current = false;
      el.scrollTop = el.scrollHeight;
      return;
    }
    const a = anchor.current;
    if (a) {
      anchor.current = null;
      el.scrollTop = a.top + (el.scrollHeight - a.height);
      if (a.refocus) (earlierButtonRef.current ?? startRef.current)?.focus({ preventScroll: true });
    }
  }, [s.items]);

  // Scrolling near the top loads the previous page.
  const loadEarlierRef = useRef(loadEarlier);
  useEffect(() => {
    loadEarlierRef.current = loadEarlier;
  }, [loadEarlier]);
  const canAutoLoad = s.status === "ready" && s.hasMore && !s.earlierError;
  useEffect(() => {
    const target = sentinelRef.current;
    const root = scrollRef.current;
    if (!canAutoLoad || !target || !root) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) void loadEarlierRef.current();
      },
      { root, rootMargin: "240px 0px 0px 0px" },
    );
    io.observe(target);
    return () => io.disconnect();
  }, [canAutoLoad]);

  const remaining = Math.max(0, s.total - s.items.length);

  return (
    <div ref={scrollRef} className="transcript" aria-busy={s.status === "loading" || s.loadingEarlier}>
      <section className="transcript-inner" aria-label="Transcript">
        {s.status === "loading" && <TranscriptSkeleton />}

        {s.status === "error" &&
          (s.unreachable ? (
            <BackendDown onRetry={() => setAttempt((n) => n + 1)} />
          ) : (
            <div className="notice" role="alert">
              <div className="notice-body">
                <strong>Couldn't load the transcript</strong>
                <p>{s.error}</p>
              </div>
              <button type="button" className="btn btn-sm" onClick={() => setAttempt((n) => n + 1)}>
                <Icon name="refresh" />
                Retry
              </button>
            </div>
          ))}

        {s.status === "ready" && s.items.length === 0 && (
          <div className="transcript-empty">
            <EmptyState icon="chat" title="No messages yet">
              <p>
                Text chat arrives with document ingestion, and voice in a later phase. Every message — yours and the
                agent's, typed or spoken — will be kept here.
              </p>
            </EmptyState>
          </div>
        )}

        {s.status === "ready" && s.items.length > 0 && (
          <>
            <div ref={sentinelRef} className="tx-sentinel" aria-hidden />
            {s.hasMore ? (
              <div className="tx-earlier">
                <button
                  ref={earlierButtonRef}
                  type="button"
                  className="btn btn-sm"
                  onClick={() => void loadEarlier(true)}
                  disabled={s.loadingEarlier}
                >
                  {s.loadingEarlier ? "Loading earlier messages…" : `Load earlier messages (${remaining})`}
                </button>
                {s.earlierError && (
                  <p className="form-error" role="alert">
                    {s.earlierError}
                  </p>
                )}
              </div>
            ) : (
              <p ref={startRef} className="tx-start" tabIndex={-1}>
                Start of the chat · {plural(s.total, "message")}
              </p>
            )}
            <ol className="tx-list">
              {s.items.map((m, i) => (
                <Fragment key={m.id}>
                  {(i === 0 || dayKey(m.created_at) !== dayKey(s.items[i - 1].created_at)) && (
                    <li className="tx-day">
                      <span>{dayLabel(m.created_at)}</span>
                    </li>
                  )}
                  <MessageItem message={m} docsById={docsById} showDebug={showDebug} />
                </Fragment>
              ))}
            </ol>
          </>
        )}
      </section>
    </div>
  );
}

function TranscriptSkeleton() {
  return (
    <div className="tx-list" aria-hidden>
      {["agent", "user", "agent"].map((role, i) => (
        <div key={i} className={`msg msg-${role}`}>
          <div className="skel skel-meta-sm" />
          <div className={`bubble bubble-skeleton ${i === 2 ? "tall" : ""}`} />
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ one message

function lastWords(text: string, n: number): string {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length > n ? `…${words.slice(-n).join(" ")}` : words.join(" ");
}

function citationParts(c: Citation, docsById: Record<string, ProjectDocument>) {
  const doc = c.document_id ? docsById[c.document_id] : undefined;
  const fallback = [c.filename, c.document_name, c.title].find((v): v is string => typeof v === "string");
  const name = doc?.filename ?? fallback ?? "Document";
  const page = typeof c.page === "number" ? c.page : null;
  return { key: `${c.document_id ?? name}:${page ?? ""}`, name, page, chunk: typeof c.chunk_id === "string" ? c.chunk_id : null };
}

function MessageItem({
  message: m,
  docsById,
  showDebug,
}: {
  message: Message;
  docsById: Record<string, ProjectDocument>;
  showDebug: boolean;
}) {
  if (m.role === "event") {
    return (
      <li className="msg msg-event">
        <span>{m.text}</span>
        <time dateTime={m.created_at} title={fullDateTime(m.created_at)}>
          {clockTime(m.created_at)}
        </time>
      </li>
    );
  }

  const isUser = m.role === "user";
  const heard = m.heard_text;
  const cutOff = m.role === "agent" && heard !== null && heard !== m.text;
  // What was played, when it is the start of the answer; the rest is shown dimmed as not heard.
  const played = heard?.trimEnd() ?? "";
  const heardPrefix = cutOff && m.text.startsWith(played) ? played : null;

  const seen = new Set<string>();
  const citations = m.citations
    .map((c) => citationParts(c, docsById))
    .filter((c) => (seen.has(c.key) ? false : (seen.add(c.key), true)));

  return (
    <li className={`msg msg-${m.role}`}>
      <div className="msg-meta">
        <span className="msg-who">{isUser ? "You" : "Agent"}</span>
        <span className="msg-mode">
          <Icon name={m.modality === "voice" ? "mic" : "keyboard"} size={13} />
          {m.modality === "voice" ? "Voice" : "Text"}
        </span>
        {m.language && (
          <span className="msg-lang" title={LANGUAGE_LABEL[m.language] ?? m.language}>
            {m.language.toUpperCase()}
          </span>
        )}
        <time dateTime={m.created_at} title={fullDateTime(m.created_at)}>
          {clockTime(m.created_at)}
        </time>
      </div>

      <div className="bubble" lang={m.language ?? undefined}>
        {heardPrefix !== null ? (
          <>
            {heardPrefix}
            <span className="unheard">
              <span className="visually-hidden"> [not played:] </span>
              {m.text.slice(heardPrefix.length)}
            </span>
          </>
        ) : (
          m.text
        )}
      </div>

      {cutOff && (
        <p className="msg-note">
          <Icon name="pulse" size={13} />
          {heard ? (
            <span>
              Interrupted after: <q>{lastWords(heard, 8)}</q>
            </span>
          ) : (
            <span>Interrupted before it was played</span>
          )}
        </p>
      )}

      {citations.length > 0 && (
        <ul className="cites" aria-label="Sources">
          {citations.map((c) => (
            <li key={c.key} className="cite" title={c.chunk ? `${c.name}, chunk ${c.chunk}` : c.name}>
              <Icon name="doc" size={12} />
              <span className="cite-name">{c.name}</span>
              {c.page !== null && <span className="cite-page">p.{c.page}</span>}
            </li>
          ))}
        </ul>
      )}

      {showDebug && (m.route || m.latency) && (
        <details className="msg-debug">
          <summary>Route and timings</summary>
          <pre>{JSON.stringify({ route: m.route, latency: m.latency }, null, 2)}</pre>
        </details>
      )}
    </li>
  );
}
