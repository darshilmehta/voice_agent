"use client";

/**
 * A chat's transcript (docs/DESIGN.md §3.9): opens at the latest messages and loads earlier pages as you scroll up
 * (or with the button), keeping your place. Bubbles distinguish you and the agent, voice and text; agent answers
 * show numbered citation chips with their sources, answers the documents don't support are set apart ("Not in
 * your documents"), and answers cut off by a barge-in show what was actually heard.
 *
 * Questions asked on this page ("turns", lib/chat-turns.ts) follow the saved messages: the question, a quiet
 * "Searching your documents…" until sources arrive, the answer streaming in with a caret, then the saved answer.
 * Failures show inline on their turn with Retry. While you're at the bottom the view follows new text; scroll up
 * and it stays where you are, with a button to jump back to the latest.
 */

import { Fragment, memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  BackendError,
  errorMessage,
  isAbort,
  type Chat,
  type Message,
  type ProjectDocument,
  type SourcesPayload,
} from "@/lib/api";
import { useBackend } from "@/lib/backend-context";
import type { Turn } from "@/lib/chat-turns";
import { citedIds, toSourceRefs, type SourceRef } from "@/lib/citations";
import { clockTime, dayKey, dayLabel, fullDateTime, plural } from "@/lib/format";

import { AnswerText, CitationPopoverProvider, SourceList } from "./Citations";
import { Icon } from "./Icon";
import { BackendDown, EmptyState } from "./States";

const PAGE = 50;
/** Within this many pixels of the end counts as "at the bottom": new text keeps the view pinned there. */
const PIN_SLACK = 48;
const JUMP_AFTER = 200;
const NO_CITATIONS: never[] = [];

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

interface TranscriptProps {
  chat: Chat;
  docsById: Record<string, ProjectDocument>;
  /** Questions asked on this page, after the saved messages. */
  turns?: Turn[];
  onRetry?: (turnKey: string) => void;
  /** An answer is streaming: Retry waits. */
  busy?: boolean;
  /** Messages the voice session saved while this page was open (the session's `user_message` / `agent_message`). */
  voiceMessages?: Message[];
  /** The voice turn in progress, before it is saved: what the user is saying, what the agent has written so far. */
  voice?: { userText: string | null; agentText: string | null; sources: SourcesPayload | null } | null;
}

const NO_MESSAGES: Message[] = [];

export function Transcript({
  chat,
  docsById,
  turns = [],
  onRetry,
  busy = false,
  voiceMessages = NO_MESSAGES,
  voice = null,
}: TranscriptProps) {
  const { api, config } = useBackend();
  const [s, setS] = useState<TranscriptState>(INITIAL);
  const [attempt, setAttempt] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const startRef = useRef<HTMLParagraphElement>(null);
  const earlierButtonRef = useRef<HTMLButtonElement>(null);
  const scrollToEnd = useRef(false);
  const anchor = useRef<{ height: number; top: number; refocus: boolean } | null>(null);
  const pinned = useRef(true);
  const [showJump, setShowJump] = useState(false);
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

  const busyLoading = useRef(false);
  const loadEarlier = useCallback(
    async (fromButton = false) => {
      if (busyLoading.current || !s.hasMore || s.cursor === null) return;
      busyLoading.current = true;
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
        busyLoading.current = false;
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
      pinned.current = true;
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

  // Asking a question brings you to the end, even if you had scrolled up.
  const turnCount = turns.length;
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el || turnCount === 0) return;
    pinned.current = true;
    el.scrollTop = el.scrollHeight;
  }, [turnCount]);

  // While pinned, follow the content as it grows (streamed text, sources, notes) and the view as it shrinks (a
  // taller composer).
  useEffect(() => {
    const el = scrollRef.current;
    const inner = innerRef.current;
    if (!el || !inner) return;
    const ro = new ResizeObserver(() => {
      if (pinned.current) el.scrollTop = el.scrollHeight;
    });
    ro.observe(inner);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Only the reader scrolling up unpins. (A scroll event can arrive after more content was added, so "not at the
  // bottom" alone doesn't mean they left it.)
  const lastTop = useRef(0);
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const top = el.scrollTop;
    const gap = el.scrollHeight - top - el.clientHeight;
    if (gap <= PIN_SLACK) pinned.current = true;
    else if (top < lastTop.current - 1) pinned.current = false;
    lastTop.current = top;
    const far = gap > JUMP_AFTER && !pinned.current;
    setShowJump((v) => (v === far ? v : far));
  };

  const jumpToLatest = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinned.current = true;
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  };

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

  // Messages saved by this page's turns are shown with their turn, not again from a page load.
  const turnMessageIds = useMemo(() => {
    const ids = new Set<string>();
    for (const t of turns) {
      if (t.user) ids.add(t.user.id);
      if (t.agent) ids.add(t.agent.id);
    }
    return ids;
  }, [turns]);
  // Likewise for what the voice session saved on this page: shown after the history, in time order with the turns.
  const voiceIds = useMemo(() => new Set(voiceMessages.map((m) => m.id)), [voiceMessages]);
  const history =
    turnMessageIds.size || voiceIds.size
      ? s.items.filter((m) => !turnMessageIds.has(m.id) && !voiceIds.has(m.id))
      : s.items;
  const savedInTurns = turns.reduce((n, t) => n + (t.user ? 1 : 0) + (t.agent ? 1 : 0), 0);
  const remaining = Math.max(0, s.total - s.items.length);
  const total = remaining + history.length + savedInTurns + voiceMessages.length;
  const ready = s.status === "ready";
  const tail = useMemo(
    () =>
      [
        ...turns.map((turn) => ({ kind: "turn" as const, at: Date.parse(turn.user?.created_at ?? turn.startedAt), turn })),
        ...voiceMessages.map((message) => ({ kind: "voice" as const, at: Date.parse(message.created_at), message })),
      ].sort((a, b) => a.at - b.at),
    [turns, voiceMessages],
  );
  const liveVoice = voice && (voice.userText !== null || voice.agentText !== null) ? voice : null;
  const hasContent = history.length > 0 || tail.length > 0 || liveVoice !== null;

  let lastDay = history.length ? dayKey(history[history.length - 1].created_at) : null;

  return (
    <div
      ref={scrollRef}
      className="transcript"
      data-scroll-root
      onScroll={onScroll}
      aria-busy={s.status === "loading" || s.loadingEarlier}
    >
      <CitationPopoverProvider>
        <section ref={innerRef} className="transcript-inner" aria-label="Transcript">
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

          {ready && !hasContent && (
            <div className="transcript-empty">
              <EmptyState icon="chat" title="No messages yet">
                <p>
                  Ask a question about this project's documents. Answers cite the pages they come from, and every
                  message — yours and the agent's — is kept here.
                </p>
              </EmptyState>
            </div>
          )}

          {ready && hasContent && (
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
                  Start of the chat · {plural(total, "message")}
                </p>
              )}
            </>
          )}

          {hasContent && s.status !== "loading" && (
            <ol className="tx-list">
              {history.map((m, i) => (
                <Fragment key={m.id}>
                  {(i === 0 || dayKey(m.created_at) !== dayKey(history[i - 1].created_at)) && (
                    <li className="tx-day">
                      <span>{dayLabel(m.created_at)}</span>
                    </li>
                  )}
                  <MessageItem message={m} docsById={docsById} showDebug={showDebug} />
                </Fragment>
              ))}
              {tail.map((item) => {
                if (item.kind === "voice") {
                  const m = item.message;
                  const divider = dayKey(m.created_at) !== lastDay;
                  lastDay = dayKey(m.created_at);
                  return (
                    <Fragment key={m.id}>
                      {divider && (
                        <li className="tx-day">
                          <span>{dayLabel(m.created_at)}</span>
                        </li>
                      )}
                      <MessageItem message={m} docsById={docsById} showDebug={showDebug} />
                    </Fragment>
                  );
                }
                const t = item.turn;
                const when = t.user?.created_at ?? t.startedAt;
                const divider = dayKey(when) !== lastDay;
                lastDay = dayKey(when);
                return (
                  <Fragment key={t.key}>
                    {divider && (
                      <li className="tx-day">
                        <span>{dayLabel(when)}</span>
                      </li>
                    )}
                    {t.user ? (
                      <MessageItem message={t.user} docsById={docsById} showDebug={showDebug} />
                    ) : (
                      <PendingQuestion turn={t} />
                    )}
                    <LiveAnswer
                      turn={t}
                      docsById={docsById}
                      showDebug={showDebug}
                      busy={busy}
                      onRetry={onRetry}
                    />
                  </Fragment>
                );
              })}
              {liveVoice?.userText != null && (
                <li className="msg msg-user" data-pending>
                  <div className="msg-meta">
                    <span className="msg-who">You</span>
                    <span className="msg-mode">
                      <Icon name="mic" size={13} />
                      Voice
                    </span>
                    <span className="msg-status">Listening…</span>
                  </div>
                  <div className="bubble">{liveVoice.userText || "…"}</div>
                </li>
              )}
              {liveVoice?.agentText != null && (
                <LiveVoiceAnswer text={liveVoice.agentText} sources={liveVoice.sources} docsById={docsById} />
              )}
            </ol>
          )}

          <div className="tx-jump-wrap">
            {showJump && (
              <button type="button" className="tx-jump" onClick={jumpToLatest}>
                <Icon name="arrowDown" size={14} />
                {busy ? "Follow the answer" : "Jump to latest"}
              </button>
            )}
          </div>
        </section>
      </CitationPopoverProvider>
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

/** Saved answers say whether they abstained in `route` (tolerant of where the backend puts the flag). */
function abstainedFlag(m: Message): boolean {
  const route = m.route as Record<string, unknown> | null;
  if (!route) return false;
  const nested = (route.retrieval ?? route.confidence) as Record<string, unknown> | undefined;
  return route.abstained === true || nested?.abstained === true;
}

/** Answers the user stopped mid-stream are saved with what was written so far and `route.stopped`. */
function stoppedFlag(m: Message): boolean {
  const route = m.route as Record<string, unknown> | null;
  return route?.stopped === true;
}

function bySourceId(refs: SourceRef[]): Record<string, SourceRef> {
  const out: Record<string, SourceRef> = {};
  for (const r of refs) if (r.sourceId && !out[r.sourceId]) out[r.sourceId] = r;
  return out;
}

/** Sources for chips: the message's own citations first, then (live turns) what retrieval returned. */
function useSources(text: string, citations: unknown, live: SourcesPayload | null | undefined, docsById: Record<string, ProjectDocument>) {
  return useMemo(() => {
    const own = toSourceRefs(citations, docsById);
    const map = { ...bySourceId(live ? toSourceRefs(live.sources, docsById) : []), ...bySourceId(own) };
    // The list under the bubble shows what the text cites; older citations without markers are listed as they are.
    const cited = citedIds(text)
      .map((id) => map[id])
      .filter((r): r is SourceRef => !!r);
    const listed = own.length ? [...own, ...cited.filter((c) => !own.some((o) => o.key === c.key))] : cited;
    return { map, listed };
  }, [text, citations, live, docsById]);
}

function MessageMeta({ m, isUser }: { m: Message; isUser: boolean }) {
  return (
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
  );
}

const MessageItem = memo(function MessageItem({
  message: m,
  docsById,
  showDebug,
  live,
}: {
  message: Message;
  docsById: Record<string, ProjectDocument>;
  showDebug: boolean;
  /** For an answer given on this page: what retrieval returned (fills in chips, says whether it abstained). */
  live?: SourcesPayload | null;
}) {
  const { map, listed } = useSources(m.text, m.citations, live, docsById);

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
  const abstained = !isUser && (live?.abstained ?? abstainedFlag(m));
  const heard = m.heard_text;
  const cutOff = m.role === "agent" && heard !== null && heard !== m.text;
  // What was played, when it is the start of the answer; the rest is shown dimmed as not heard.
  const played = heard?.trimEnd() ?? "";
  const heardPrefix = cutOff && m.text.startsWith(played) ? played : null;
  const render = (text: string) => (isUser ? text : <AnswerText text={text} sources={map} />);

  return (
    <li className={`msg msg-${m.role}`}>
      <MessageMeta m={m} isUser={isUser} />

      <div className={abstained ? "bubble bubble-abstained" : "bubble"} lang={m.language ?? undefined}>
        {abstained && <AbstainLabel />}
        {heardPrefix !== null ? (
          <>
            {render(heardPrefix)}
            <span className="unheard">
              <span className="visually-hidden"> [not played:] </span>
              {render(m.text.slice(heardPrefix.length))}
            </span>
          </>
        ) : (
          render(m.text)
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

      {!isUser && <SourceList sources={listed} />}

      {!isUser && stoppedFlag(m) && (
        <p className="msg-note msg-note-quiet">
          <Icon name="stop" size={12} />
          Stopped. The rest of this answer wasn&apos;t written.
        </p>
      )}

      {showDebug && (m.route || m.latency || live) && (
        <details className="msg-debug">
          <summary>Route and timings</summary>
          <pre>
            {JSON.stringify(
              {
                route: m.route,
                latency: m.latency,
                ...(live ? { retrieval: { confidence: live.confidence, abstained: live.abstained, sources: live.sources.length } } : {}),
              },
              null,
              2,
            )}
          </pre>
        </details>
      )}
    </li>
  );
});

function AbstainLabel() {
  return (
    <span className="abstain-label">
      <Icon name="info" size={14} />
      Not in your documents
    </span>
  );
}

// ------------------------------------------------------------------ a voice answer being written

/** The agent's answer to a spoken question while it is still being generated (the saved message replaces it). */
function LiveVoiceAnswer({
  text,
  sources,
  docsById,
}: {
  text: string;
  sources: SourcesPayload | null;
  docsById: Record<string, ProjectDocument>;
}) {
  const { map, listed } = useSources(text, NO_CITATIONS, sources, docsById);
  if (!text) return null;
  return (
    <li className="msg msg-agent" aria-busy="true">
      <div className="msg-meta">
        <span className="msg-who">Agent</span>
        <span className="msg-mode">
          <Icon name="mic" size={13} />
          Voice
        </span>
        <span className="msg-status">Answering…</span>
      </div>
      <div className="bubble">
        <AnswerText text={text} sources={map} streaming />
      </div>
      <SourceList sources={listed} />
    </li>
  );
}

// ------------------------------------------------------------------ a question asked on this page

/** The question before the backend has saved it. */
function PendingQuestion({ turn }: { turn: Turn }) {
  const notSent = turn.phase === "failed" && !turn.failure?.saved;
  const status = notSent ? "Not sent" : turn.phase === "stopped" ? "Stopped" : "Sending…";
  return (
    <li className="msg msg-user" data-pending={notSent ? undefined : true}>
      <div className="msg-meta">
        <span className="msg-who">You</span>
        <span className="msg-mode">
          <Icon name="keyboard" size={13} />
          Text
        </span>
        <span className={notSent ? "msg-status msg-status-bad" : "msg-status"}>{status}</span>
      </div>
      <div className="bubble" lang={turn.language ?? undefined}>
        {turn.text}
      </div>
    </li>
  );
}

const STAGE_TITLE: Record<string, string> = {
  retrieval: "Couldn't search your documents",
  llm: "The language model couldn't answer",
  storage: "The answer couldn't be saved",
};

function failureTitle(turn: Turn): string {
  const f = turn.failure;
  if (!f) return "Couldn't answer";
  if (f.stage && STAGE_TITLE[f.stage]) return STAGE_TITLE[f.stage];
  if (!f.saved) return "Your question wasn't sent";
  return "The answer was cut off";
}

function LiveAnswer({
  turn,
  docsById,
  showDebug,
  busy,
  onRetry,
}: {
  turn: Turn;
  docsById: Record<string, ProjectDocument>;
  showDebug: boolean;
  busy: boolean;
  onRetry?: (key: string) => void;
}) {
  const { map, listed } = useSources(turn.answer, NO_CITATIONS, turn.sources, docsById);

  if (turn.phase === "done" && turn.agent) {
    return <MessageItem message={turn.agent} docsById={docsById} showDebug={showDebug} live={turn.sources} />;
  }
  if (turn.phase === "sending") return null;

  const abstained = turn.sources?.abstained ?? false;
  const hasText = turn.answer.length > 0;
  const streaming = turn.phase === "answering" && hasText;
  const waiting = (turn.phase === "searching" || turn.phase === "answering") && !hasText;
  const status =
    turn.phase === "searching"
      ? "Searching…"
      : turn.phase === "answering"
        ? "Answering…"
        : turn.phase === "stopped"
          ? "Stopped"
          : turn.phase === "failed"
            ? "No answer"
            : null;

  return (
    <li className="msg msg-agent" aria-busy={turn.phase === "searching" || turn.phase === "answering" || undefined}>
      <div className="msg-meta">
        <span className="msg-who">Agent</span>
        {status && <span className="msg-status">{status}</span>}
      </div>

      {waiting && (
        <div className="bubble bubble-waiting">
          <span className="dots" aria-hidden>
            <i />
            <i />
            <i />
          </span>
          {turn.phase === "searching" ? "Searching your documents…" : "Writing the answer…"}
        </div>
      )}

      {hasText && (
        <div
          className={`bubble${abstained ? " bubble-abstained" : ""}${turn.phase === "failed" ? " bubble-partial" : ""}`}
          lang={turn.language ?? undefined}
        >
          {abstained && <AbstainLabel />}
          <AnswerText text={turn.answer} sources={map} streaming={streaming} />
        </div>
      )}

      {hasText && <SourceList sources={listed} />}

      {turn.phase === "stopped" && (
        <p className="msg-note msg-note-quiet">
          <Icon name="stop" size={12} />
          {hasText ? "Stopped. The rest of this answer wasn't written." : "Stopped before the answer started."}
        </p>
      )}

      {turn.phase === "failed" && turn.failure && (
        <div className="turn-error" role="alert">
          <Icon name="alert" className="turn-error-icon" />
          <div className="turn-error-body">
            <strong>{failureTitle(turn)}</strong>
            <p>{turn.failure.detail}</p>
          </div>
          {turn.retried ? (
            <span className="turn-error-note">Asked again below</span>
          ) : (
            onRetry && (
              <button type="button" className="btn btn-sm" onClick={() => onRetry(turn.key)} disabled={busy}>
                <Icon name="refresh" />
                Retry
              </button>
            )
          )}
        </div>
      )}
    </li>
  );
}
