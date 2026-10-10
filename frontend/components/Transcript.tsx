"use client";

/**
 * A chat's transcript (docs/DESIGN.md §3.9): opens at the latest messages and loads earlier pages as you scroll up
 * (or with the button), keeping your place. Bubbles distinguish you and the agent, voice and text; agent answers
 * show numbered citation chips with their sources, answers the documents don't support are set apart ("Not in
 * your documents"), and answers cut off by a barge-in show what was actually heard.
 *
 * What the router decided (lib/route.ts, `route` of each agent message) shapes the rest: general-knowledge and mixed
 * answers carry a quiet label saying where they come from, short acknowledgements are light bubbles without a source
 * list, a clarifying question says it asked one, and a user message the router read differently shows
 * "Understood as: …" (hidden until hovered or focused). The summary's unanswered questions scroll to a message here
 * (`focus`, loading earlier pages when it isn't loaded yet) and highlight it.
 *
 * Questions asked on this page ("turns", lib/chat-turns.ts) follow the saved messages: the question, a quiet
 * "Searching your documents…" until sources arrive, the answer streaming in with a caret, then the saved answer.
 * Failures show inline on their turn with Retry. While you're at the bottom the view follows new text; scroll up
 * and it stays where you are, with a button to jump back to the latest.
 *
 * Live web search (docs/DESIGN.md §3.7): a turn that searches the web shows a "Searching the web…" badge until the
 * answer starts, then a quiet "Searched the web for “…”" (exactly what left the machine); web results are `[W#]`
 * chips and a globe chip in the source list (documents first, then web); the label says when the answer includes live
 * web results; and `route.live_note` explains missing live data. With `features.web_search` off the badge, the note
 * and the hint are hidden; web citations still render, as on any old message.
 */

import { Fragment, memo, useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  BackendError,
  errorMessage,
  isAbort,
  type Chat,
  type Citation,
  type Message,
  type ProjectDocument,
  type SourcesPayload,
} from "@/lib/api";
import { useBackend } from "@/lib/backend-context";
import { PRODUCT_NAME } from "@/lib/brand";
import type { Turn } from "@/lib/chat-turns";
import { citedIds, toSourceRefs, type SourceRef } from "@/lib/citations";
import { clockTime, dayKey, dayLabel, fullDateTime, plural } from "@/lib/format";
import { cutNote, cutReasonOf, stoppedNote } from "@/lib/interruption";
import {
  BASIS_LABEL,
  LIVE_NOTE_TEXT,
  WEB_NOTE,
  answerKindOf,
  basisOf,
  isBrief,
  isFixedReply,
  isWebBasis,
  liveNoteOf,
  showsSources,
  sourcesToShow,
  understoodAsByUser,
  visualIdOf,
  visualReuseOf,
  visualUpdatedOf,
  type Basis,
} from "@/lib/route";
import { searchOfRoute, type WebSearchState } from "@/lib/web-search";

import { AnswerText, CitationPopoverProvider, SourceList } from "./Citations";
import { Icon } from "./Icon";
import { BackendDown, EmptyState } from "./States";
import { WebSearchNote } from "./WebSearchNote";

const PAGE = 50;
/** Within this many pixels of the end counts as "at the bottom": new text keeps the view pinned there. */
const PIN_SLACK = 48;
const JUMP_AFTER = 200;
const NO_CITATIONS: never[] = [];
const NO_SEARCHES: Record<string, WebSearchState> = {};

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
  voice?: {
    userText: string | null;
    agentText: string | null;
    sources: SourcesPayload | null;
    /** The turn's web search, and the web results that arrived (a `[W3]` only a late `tool results` carries). */
    search: WebSearchState | null;
    webSources: Citation[];
  } | null;
  /** What the web search of each message the voice session saved was (by message id): the note of what was searched. */
  voiceSearches?: Record<string, WebSearchState>;
  /**
   * Scroll to the message with this `seq` and highlight it (loading earlier pages until it is there). `n` tells one
   * request from the next; `onFocusDone` is called with it once the message was shown, or can't be found.
   */
  focus?: { seq: number; n: number } | null;
  onFocusDone?: (n: number, found: boolean) => void;
}

const NO_MESSAGES: Message[] = [];
/** How long a message jumped to from the summary stays highlighted. */
const FLASH_MS = 2200;

export function Transcript({
  chat,
  docsById,
  turns = [],
  onRetry,
  busy = false,
  voiceMessages = NO_MESSAGES,
  voice = null,
  voiceSearches = NO_SEARCHES,
  focus = null,
  onFocusDone,
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
  const webOn = config?.features.web_search === true;

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

  // A jump from the summary: scroll to the message, loading earlier pages until it is there, and flash it. Scrolls
  // this view only (scrollIntoView would also move the clipped panel around it) and puts keyboard focus on the
  // message, since the button that asked is no longer shown.
  const [flashSeq, setFlashSeq] = useState<number | null>(null);
  const onFocusDoneRef = useRef(onFocusDone);
  useEffect(() => {
    onFocusDoneRef.current = onFocusDone;
  });
  const handledFocus = useRef(0);
  useEffect(() => {
    if (!focus || handledFocus.current === focus.n || s.status !== "ready") return;
    const root = scrollRef.current;
    if (!root) return;
    const done = (found: boolean) => {
      handledFocus.current = focus.n;
      onFocusDoneRef.current?.(focus.n, found);
    };
    const target = root.querySelector<HTMLElement>(`[data-seq="${focus.seq}"]`);
    if (target) {
      const box = root.getBoundingClientRect();
      const at = target.getBoundingClientRect();
      const top = root.scrollTop + (at.top - box.top) - Math.max(0, (root.clientHeight - at.height) / 2);
      // A short way is scrolled smoothly; a long one (a message on an earlier page) jumps, so the view can't be pulled
      // back to the end by the content that was just added above it while it is still under way.
      const goal = Math.max(0, top);
      const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches && Math.abs(goal - root.scrollTop) < root.clientHeight * 2;
      pinned.current = false; // don't let the view snap back to the end
      root.scrollTo({ top: goal, behavior: smooth ? "smooth" : "auto" });
      target.focus({ preventScroll: true });
      setFlashSeq(focus.seq);
      done(true);
    } else if (s.hasMore && !s.loadingEarlier && !s.earlierError) {
      void loadEarlier(); // the effect runs again when the page arrives
    } else if (!s.hasMore || s.earlierError) {
      done(false);
    }
  }, [focus, s.status, s.items, s.hasMore, s.loadingEarlier, s.earlierError, loadEarlier]);
  useEffect(() => {
    if (flashSeq === null) return;
    const timer = window.setTimeout(() => setFlashSeq(null), FLASH_MS);
    return () => window.clearTimeout(timer);
  }, [flashSeq]);

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
  // "Understood as": what the router worked from when it isn't what was said, for the user messages it applies to.
  const understood = useMemo(() => {
    const all: Message[] = [...history, ...voiceMessages];
    for (const t of turns) {
      if (t.user) all.push(t.user);
      if (t.agent) all.push(t.agent);
    }
    return understoodAsByUser(all);
  }, [history, voiceMessages, turns]);
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
                  message — yours and Docent's — is kept here.
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
                  <MessageItem
                    message={m}
                    docsById={docsById}
                    showDebug={showDebug}
                    webOn={webOn}
                    understood={understood.get(m.id)}
                    flash={m.seq === flashSeq}
                  />
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
                      <MessageItem
                        message={m}
                        docsById={docsById}
                        showDebug={showDebug}
                        webOn={webOn}
                        search={voiceSearches[m.id]}
                        understood={understood.get(m.id)}
                        flash={m.seq === flashSeq}
                      />
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
                      <MessageItem
                        message={t.user}
                        docsById={docsById}
                        showDebug={showDebug}
                        webOn={webOn}
                        understood={understood.get(t.user.id)}
                        flash={t.user.seq === flashSeq}
                      />
                    ) : (
                      <PendingQuestion turn={t} />
                    )}
                    <LiveAnswer
                      turn={t}
                      docsById={docsById}
                      showDebug={showDebug}
                      webOn={webOn}
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
                <LiveVoiceAnswer
                  text={liveVoice.agentText}
                  sources={liveVoice.sources}
                  extra={liveVoice.webSources}
                  search={webOn ? liveVoice.search : null}
                  docsById={docsById}
                />
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

/**
 * Sources for chips: the message's own citations first (the saved ones are authoritative), then (live turns) what
 * retrieval returned, then the web results that arrived with the search (`extra`: a `[W3]` that only a continuation's
 * `tool results` carries).
 */
function useSources(
  text: string,
  citations: unknown,
  live: SourcesPayload | null | undefined,
  docsById: Record<string, ProjectDocument>,
  extra?: Citation[],
) {
  return useMemo(() => {
    const own = toSourceRefs(citations, docsById);
    const map = {
      ...bySourceId(extra ? toSourceRefs(extra, docsById) : []),
      ...bySourceId(live ? toSourceRefs(live.sources, docsById) : []),
      ...bySourceId(own),
    };
    // The list under the bubble shows what the text cites; older citations without markers are listed as they are.
    const cited = citedIds(text)
      .map((id) => map[id])
      .filter((r): r is SourceRef => !!r);
    const listed = own.length ? [...own, ...cited.filter((c) => !own.some((o) => o.key === c.key))] : cited;
    return { map, listed };
  }, [text, citations, live, docsById, extra]);
}

function MessageMeta({ m, isUser, extra }: { m: Message; isUser: boolean; extra?: ReactNode }) {
  return (
    <div className="msg-meta">
      <span className="msg-who">{isUser ? "You" : PRODUCT_NAME}</span>
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
      {extra}
    </div>
  );
}

const MessageItem = memo(function MessageItem({
  message: m,
  docsById,
  showDebug,
  webOn = false,
  live,
  extra,
  search,
  understood,
  flash = false,
}: {
  message: Message;
  docsById: Record<string, ProjectDocument>;
  showDebug: boolean;
  /** `features.web_search`: the note of what was searched and the hint about missing live data show only when on. */
  webOn?: boolean;
  /** For an answer given on this page: what retrieval returned (fills in chips, says whether it abstained). */
  live?: SourcesPayload | null;
  /** For an answer given on this page: the web results the search brought (the saved citations win over them). */
  extra?: Citation[];
  /** The live web search of this answer, when this page saw it (the saved route may carry only the query). */
  search?: WebSearchState | null;
  /**
   * A user message the router understood differently (a correction, a follow-up, a word Whisper mis-heard): the
   * question it worked from. Quiet by default: "Understood as: …" shows when the message is hovered or focused, and
   * the toggle beside the time keeps it open; with no hover (touch) the toggle is always there, very small.
   */
  understood?: string;
  /** Highlight it for a moment (it was jumped to from the summary). */
  flash?: boolean;
}) {
  const { map, listed } = useSources(m.text, m.citations, live, docsById, extra);
  // "Understood as: …" is hidden until the message is hovered or focused; the toggle in the meta row keeps it open.
  const [understoodOpen, setUnderstoodOpen] = useState(false);
  const understoodId = useId();

  if (m.role === "event") {
    return (
      <li className="msg msg-event" data-seq={m.seq} tabIndex={-1} data-flash={flash || undefined}>
        <span>{m.text}</span>
        <time dateTime={m.created_at} title={fullDateTime(m.created_at)}>
          {clockTime(m.created_at)}
        </time>
      </li>
    );
  }

  const isUser = m.role === "user";
  // What the router decided shapes the labels: general-knowledge and mixed answers say where they come from, a short
  // acknowledgement is a light bubble with no sources, a clarifying question says it asked one.
  const kind = isUser ? null : answerKindOf(m);
  const hasWeb = listed.some((r) => r.kind === "web");
  const basis = isUser ? null : basisOf(m, { documents: listed.some((r) => r.kind === "document"), web: hasWeb });
  const brief = isBrief(kind);
  // What was searched (exactly what left the machine) and why live data is missing: only while web search is on.
  const searched = webOn && !isUser ? (search && search.status !== "ended" ? search : (searchOfRoute(m.route) ?? search ?? null)) : null;
  const liveNote = webOn && !isUser ? liveNoteOf(m) : null;
  // The documents didn't cover it: "Not in your documents", unless it was answered from general knowledge on purpose.
  const abstained = !isUser && !basis && !isFixedReply(kind) && (live?.abstained ?? abstainedFlag(m));
  const heard = m.heard_text;
  const cutOff = m.role === "agent" && heard !== null && heard !== m.text;
  // What was played, when it is the start of the answer; the rest is shown dimmed as not heard.
  const played = heard?.trimEnd() ?? "";
  const heardPrefix = cutOff && m.text.startsWith(played) ? played : null;
  const render = (text: string) => (isUser ? text : <AnswerText text={text} sources={map} />);
  // Stopped before anything was written: the saved answer is empty, so there is no bubble to show, only the note.
  const blank = !isUser && m.text.trim() === "";
  const reason = cutReasonOf(m);
  const cut = cutNote(reason, heard ?? "");
  const stopped = stoppedFlag(m);
  // The connection dropped while the answer was going out and nothing says how much of it was heard.
  const droppedUnheard = !isUser && !stopped && reason === "disconnect" && heard === null;

  const bubbleClass = ["bubble", abstained && "bubble-abstained", brief && "bubble-light"].filter(Boolean).join(" ");

  return (
    <li className={`msg msg-${m.role}`} data-seq={m.seq} tabIndex={-1} data-flash={flash || undefined}>
      <MessageMeta
        m={m}
        isUser={isUser}
        extra={
          isUser && understood ? (
            <button
              type="button"
              className="understood-toggle"
              aria-expanded={understoodOpen}
              aria-controls={understoodId}
              title="How this was understood"
              onClick={() => setUnderstoodOpen((v) => !v)}
            >
              <Icon name="info" size={12} />
              <span className="visually-hidden">How this was understood</span>
            </button>
          ) : undefined
        }
      />

      {!blank && (
        <div className={bubbleClass} lang={m.language ?? undefined}>
          {abstained && <AbstainLabel />}
          {basis && <BasisLabel basis={basis} />}
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
      )}

      {cutOff && (
        <p className="msg-note">
          <Icon name="pulse" size={13} />
          <span>
            {cut.lead}
            {cut.quote !== null && (
              <>
                {" "}
                <q>{lastWords(cut.quote, 8)}</q>
              </>
            )}
          </span>
        </p>
      )}

      {!isUser && visualIdOf(m) && !visualReuseOf(m) && (
        <p className="msg-note">
          <Icon name="chart" size={13} />
          <span>Chart added to the canvas</span>
        </p>
      )}
      {!isUser && visualIdOf(m) && visualReuseOf(m) && (
        <p className="msg-note msg-note-quiet">
          <Icon name="chart" size={12} />
          {visualReuseOf(m) === "extends" ? "Chart updated on the canvas" : "Already on the canvas"}
        </p>
      )}
      {!isUser && !visualIdOf(m) && visualUpdatedOf(m) && (
        <p className="msg-note msg-note-quiet">
          <Icon name="chart" size={12} />
          Chart updated
        </p>
      )}

      {isUser && understood && (
        <p id={understoodId} className="understood-line" data-open={understoodOpen || undefined}>
          Understood as: <q>{understood}</q>
        </p>
      )}

      {kind === "clarification" && !blank && (
        <p className="msg-note msg-note-quiet">
          <Icon name="info" size={12} />
          Asked to clarify
        </p>
      )}

      {searched && <WebSearchNote search={searched} />}
      {liveNote && (
        <p className="msg-note msg-note-quiet">
          <Icon name="info" size={12} />
          {LIVE_NOTE_TEXT[liveNote]}
        </p>
      )}

      {!isUser && showsSources(kind) &&<SourceList sources={sourcesToShow(kind, listed)} />}

      {/* A cut voice answer already says how it was cut (and is usually complete, so "wasn't written" would be wrong). */}
      {!isUser && (stopped || droppedUnheard) && !cutOff && (
        <p className="msg-note msg-note-quiet">
          <Icon name="stop" size={12} />
          {stoppedNote(reason, blank, stopped)}
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

/** Where an answer's content comes from, when it isn't (only) the documents. Same quiet treatment as the abstention. */
function BasisLabel({ basis }: { basis: Basis }) {
  const web = isWebBasis(basis);
  return (
    <span className="abstain-label basis-label" data-basis={basis} title={web ? WEB_NOTE : undefined}>
      <Icon name={web ? "globe" : basis === "mixed" ? "doc" : "info"} size={14} />
      {BASIS_LABEL[basis]}
    </span>
  );
}

// ------------------------------------------------------------------ a voice answer being written

/** The agent's answer to a spoken question while it is still being generated (the saved message replaces it). */
function LiveVoiceAnswer({
  text,
  sources,
  extra,
  search,
  docsById,
}: {
  text: string;
  sources: SourcesPayload | null;
  extra: Citation[];
  search: WebSearchState | null;
  docsById: Record<string, ProjectDocument>;
}) {
  const { map, listed } = useSources(text, NO_CITATIONS, sources, docsById, extra);
  if (!text) return null;
  return (
    <li className="msg msg-agent" aria-busy="true">
      <div className="msg-meta">
        <span className="msg-who">{PRODUCT_NAME}</span>
        <span className="msg-mode">
          <Icon name="mic" size={13} />
          Voice
        </span>
        <span className="msg-status">Answering…</span>
      </div>
      <div className="bubble">
        <AnswerText text={text} sources={map} streaming />
      </div>
      {search && <WebSearchNote search={search} />}
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
  webOn,
  busy,
  onRetry,
}: {
  turn: Turn;
  docsById: Record<string, ProjectDocument>;
  showDebug: boolean;
  webOn: boolean;
  busy: boolean;
  onRetry?: (key: string) => void;
}) {
  const { map, listed } = useSources(turn.answer, NO_CITATIONS, turn.sources, docsById, turn.web.sources);

  if (turn.phase === "done" && turn.agent) {
    return (
      <MessageItem
        message={turn.agent}
        docsById={docsById}
        showDebug={showDebug}
        webOn={webOn}
        live={turn.sources}
        extra={turn.web.sources}
        search={turn.web.search}
      />
    );
  }
  if (turn.phase === "sending") return null;

  // Not in the documents, but the web had it: that is an answer, not an abstention.
  const webResults = turn.web.sources.length > 0 || (turn.sources?.sources ?? []).some((c) => c.kind === "web");
  const abstained = !webResults && (turn.sources?.abstained ?? false);
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
        <span className="msg-who">{PRODUCT_NAME}</span>
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

      {webOn && turn.web.search && <WebSearchNote search={turn.web.search} />}

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
