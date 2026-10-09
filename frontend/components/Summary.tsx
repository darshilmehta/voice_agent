"use client";

/**
 * The Summary view of a chat (docs/DESIGN.md §3.9), next to the transcript in the panel you open to revisit a
 * conversation: "Transcript | Summary" tabs, and the summary rendered from its structured fields (overview, key points
 * with source chips, questions the documents couldn't answer, follow-ups), never from the markdown `content`.
 *
 * A summary is text to read; nothing here is spoken. Writing one takes a while on the local model, so the view says
 * so, keeps working while you look at the transcript, and shows an out-of-date summary with a Refresh button instead
 * of replacing it silently. A question the documents couldn't answer takes you to the message that asked it.
 */

import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

import type { Chat, Language, ProjectDocument, SummarySource } from "@/lib/api";
import { toSourceRef, type SourceRef } from "@/lib/citations";
import { plural, relativeTime } from "@/lib/format";
import { isBlank, outOfDateText } from "@/lib/summary-model";
import type { ChatSummaryApi } from "@/lib/use-chat-summary";

import { CitationPopoverProvider, SourceList } from "./Citations";
import { Icon } from "./Icon";
import { BackendDown, EmptyState } from "./States";

export type PanelView = "transcript" | "summary";

export interface PanelIds {
  tab: Record<PanelView, string>;
  panel: Record<PanelView, string>;
}

export function usePanelIds(): PanelIds {
  const base = useId();
  return {
    tab: { transcript: `${base}-tab-transcript`, summary: `${base}-tab-summary` },
    panel: { transcript: `${base}-panel-transcript`, summary: `${base}-panel-summary` },
  };
}

// ------------------------------------------------------------------ the tabs and the panes they switch

const TABS: { id: PanelView; label: string }[] = [
  { id: "transcript", label: "Transcript" },
  { id: "summary", label: "Summary" },
];

/**
 * "Transcript | Summary" (WAI-ARIA tabs: arrow keys, Home and End move between them and show that view). A dot on
 * Summary says one is being written or is out of date.
 */
export function PanelTabs({
  view,
  onChange,
  ids,
  summaryBadge,
}: {
  view: PanelView;
  onChange: (view: PanelView) => void;
  ids: PanelIds;
  summaryBadge: "writing" | "stale" | null;
}) {
  const refs = useRef<Partial<Record<PanelView, HTMLButtonElement | null>>>({});

  const onKeyDown = (e: KeyboardEvent) => {
    const at = TABS.findIndex((t) => t.id === view);
    let next = at;
    if (e.key === "ArrowRight") next = (at + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (at - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    onChange(TABS[next].id);
    refs.current[TABS[next].id]?.focus();
  };

  return (
    <div role="tablist" aria-label="Conversation views" className="ptabs" onKeyDown={onKeyDown}>
      {TABS.map((t) => {
        const selected = t.id === view;
        const badge = t.id === "summary" ? summaryBadge : null;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[t.id] = el;
            }}
            type="button"
            role="tab"
            id={ids.tab[t.id]}
            aria-selected={selected}
            aria-controls={ids.panel[t.id]}
            tabIndex={selected ? 0 : -1}
            className="ptab"
            onClick={() => onChange(t.id)}
          >
            {t.label}
            {badge && <span className="ptab-dot" data-kind={badge} aria-hidden />}
            {badge && <span className="visually-hidden">{badge === "writing" ? " (being written)" : " (out of date)"}</span>}
          </button>
        );
      })}
    </div>
  );
}

/**
 * The two views stacked in one body: the transcript stays mounted underneath (it keeps its place and what it loaded)
 * and the summary covers it. The covered one is inert, so keyboard and screen readers only reach the shown one.
 */
export function PanelViews({
  view,
  ids,
  transcript,
  summary,
}: {
  view: PanelView;
  ids: PanelIds;
  transcript: ReactNode;
  summary: ReactNode;
}) {
  return (
    <div className="pv">
      <div
        className="pv-pane pv-transcript"
        role="tabpanel"
        id={ids.panel.transcript}
        aria-labelledby={ids.tab.transcript}
        inert={view !== "transcript"}
      >
        {transcript}
      </div>
      <div
        className="pv-pane pv-summary"
        role="tabpanel"
        id={ids.panel.summary}
        aria-labelledby={ids.tab.summary}
        hidden={view !== "summary"}
        data-scroll-root
      >
        {summary}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ the summary

const LANGUAGES: { id: Language; label: string; name: string }[] = [
  { id: "en", label: "EN", name: "English" },
  { id: "hi", label: "हिंदी", name: "Hindi" },
];
const LANGUAGE_NAME: Record<Language, string> = { en: "English", hi: "Hindi" };

const SOURCE_NOTE = "Cited by this summary. The passage isn't stored with it; the transcript has the full answer.";

function sourceRefs(sources: SummarySource[], docsById: Record<string, ProjectDocument>): SourceRef[] {
  const seen = new Set<string>();
  const out: SourceRef[] = [];
  for (const s of sources) {
    const ref = toSourceRef(
      { document_id: s.document_id ?? undefined, filename: s.filename, page_start: s.page_start, page_end: s.page_end },
      docsById,
    );
    if (seen.has(ref.key)) continue;
    seen.add(ref.key);
    out.push({ ...ref, note: SOURCE_NOTE });
  }
  return out;
}

export function SummaryView({
  chat,
  s,
  docsById,
  onShowMessage,
}: {
  chat: Chat;
  s: ChatSummaryApi;
  docsById: Record<string, ProjectDocument>;
  /** Show the message with this `seq` in the transcript. */
  onShowMessage: (seq: number) => void;
}) {
  const { summary, writing, problem, problemOf } = s;
  const empty = !summary && (chat.message_count === 0 || problem?.empty === true);
  // A refresh keeps the old summary on screen; a first summary or another language replaces it, so show placeholders.
  const placeholder = writing !== null && (!summary || (writing !== "auto" && writing !== summary.language));

  let body: ReactNode;
  if (placeholder) {
    body = <WritingState language={writing === "auto" ? null : writing} />;
  } else if (!summary && s.load !== "ready") {
    body = <SummarySkeleton label="Loading the summary…" />;
  } else if (!summary && problemOf === "load" && problem) {
    body = problem.unreachable ? (
      <BackendDown message={problem.message} onRetry={s.reload} />
    ) : (
      <Problem title="Couldn't load the summary" detail={problem.message} onRetry={s.reload} />
    );
  } else if (empty) {
    body = (
      <EmptyState icon="list" title="Nothing to summarise yet">
        <p>Once you've talked about your documents, a summary of the conversation appears here.</p>
      </EmptyState>
    );
  } else if (!summary) {
    body = (
      <>
        {problemOf === "write" && problem && <WriteProblem problem={problem} onRetry={() => s.generate(problem.language ?? undefined)} />}
        <EmptyState
          icon="list"
          title="No summary yet"
          action={
            <button type="button" className="btn btn-primary" onClick={() => s.generate()}>
              <Icon name="sparkle" />
              Summarise this chat
            </button>
          }
        >
          <p>
            A short overview, the key points with their pages, and what the documents couldn't answer. Writing it runs
            on the local model, so it can take a moment.
          </p>
        </EmptyState>
      </>
    );
  } else {
    body = <SummaryContent s={s} chat={chat} docsById={docsById} onShowMessage={onShowMessage} />;
  }

  return (
    <div className="sm" aria-busy={writing !== null || undefined}>
      <div className="sm-inner">
        <div className="sm-bar">
          <span className="sm-bar-label">Language</span>
          <LanguageSwitch value={s.language} disabled={writing !== null || empty} onChange={s.selectLanguage} />
        </div>
        {body}
      </div>
    </div>
  );
}

function LanguageSwitch({
  value,
  disabled,
  onChange,
}: {
  value: Language;
  disabled: boolean;
  onChange: (language: Language) => void;
}) {
  return (
    <div role="group" aria-label="Summary language" className="seg">
      {LANGUAGES.map((l) => (
        <button
          key={l.id}
          type="button"
          className="seg-btn"
          lang={l.id}
          title={l.name}
          aria-pressed={value === l.id}
          aria-disabled={disabled || undefined}
          onClick={() => !disabled && value !== l.id && onChange(l.id)}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
}

function SummaryContent({
  s,
  chat,
  docsById,
  onShowMessage,
}: {
  s: ChatSummaryApi;
  chat: Chat;
  docsById: Record<string, ProjectDocument>;
  onShowMessage: (seq: number) => void;
}) {
  const summary = s.summary;
  if (!summary) return null;
  const lang = summary.language;
  const refreshing = s.writing !== null;
  const showStale = s.stale && !refreshing;
  const blank = isBlank(summary);

  return (
    <>
      {refreshing && (
        <div className="sm-banner" role="status">
          <span className="sm-spinner" aria-hidden />
          <span>Refreshing the summary… It runs on the local model, so this can take a moment.</span>
        </div>
      )}
      {showStale && (
        <div className="sm-banner sm-banner-stale" role="status">
          <Icon name="info" size={15} />
          <span>{outOfDateText(s.covered, s.now)}</span>
          <button type="button" className="btn btn-sm" onClick={() => s.generate()}>
            <Icon name="refresh" size={14} />
            Refresh
          </button>
        </div>
      )}
      {s.problemOf === "write" && s.problem && (
        <WriteProblem problem={s.problem} onRetry={() => s.generate(s.problem?.language ?? undefined)} />
      )}

      <CitationPopoverProvider>
        <article className="sm-body" data-refreshing={refreshing || undefined}>
          {blank && (
            <Problem
              title="The summary came back empty"
              detail="The model didn't find anything to say about this chat. Try writing it again."
              onRetry={() => s.generate()}
            />
          )}

          {summary.overview && (
            <section className="sm-section" aria-labelledby={`${summary.id}-overview`}>
              <h3 id={`${summary.id}-overview`}>Overview</h3>
              <p className="sm-overview" lang={lang}>
                {summary.overview}
              </p>
            </section>
          )}

          {summary.key_points.length > 0 && (
            <section className="sm-section" aria-labelledby={`${summary.id}-points`}>
              <h3 id={`${summary.id}-points`}>Key points</h3>
              <ul className="sm-points">
                {summary.key_points.map((k, i) => (
                  <li key={i} className="sm-point">
                    <p lang={lang}>{k.text}</p>
                    <SourceList sources={sourceRefs(k.sources, docsById)} />
                  </li>
                ))}
              </ul>
            </section>
          )}

          {summary.unanswered_questions.length > 0 && (
            <section className="sm-section" aria-labelledby={`${summary.id}-unanswered`}>
              <h3 id={`${summary.id}-unanswered`}>Not answered from your documents</h3>
              <ul className="sm-questions">
                {summary.unanswered_questions.map((q, i) => (
                  <li key={i}>
                    {q.message_seq === null ? (
                      <p className="sm-q sm-q-plain" lang={lang}>
                        <Icon name="info" size={14} />
                        <span>{q.question}</span>
                      </p>
                    ) : (
                      <button
                        type="button"
                        className="sm-q"
                        lang={lang}
                        title="Show this question in the transcript"
                        aria-label={`${q.question} (show in the transcript)`}
                        onClick={() => onShowMessage(q.message_seq as number)}
                      >
                        <Icon name="info" size={14} />
                        <span className="sm-q-text">{q.question}</span>
                        <Icon name="chevron" size={14} className="sm-q-go" />
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {summary.follow_ups.length > 0 && (
            <section className="sm-section" aria-labelledby={`${summary.id}-follow`}>
              <h3 id={`${summary.id}-follow`}>Follow-ups to try</h3>
              <ul className="sm-follow">
                {summary.follow_ups.map((f, i) => (
                  <li key={i} lang={lang}>
                    {f}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <footer className="sm-foot">
            {[
              LANGUAGE_NAME[lang as Language] ?? lang,
              plural(s.covered, "message"),
              summary.model ? `Written by ${summary.model}` : null,
              summary.created_at && !Number.isNaN(Date.parse(summary.created_at)) ? relativeTime(summary.created_at) : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </footer>
        </article>
      </CitationPopoverProvider>
    </>
  );
}

// ------------------------------------------------------------------ states

/** Seconds since `running` became true, ticking once a second. */
function useElapsed(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) {
      setSeconds(0);
      return;
    }
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  return seconds;
}

function SummarySkeleton({ label }: { label?: string }) {
  return (
    <div className="sm-skeleton" role={label ? "status" : undefined} aria-label={label}>
      <div aria-hidden>
        <div className="skel sm-skel-h" />
        <div className="skel sm-skel-line" />
        <div className="skel sm-skel-line" />
        <div className="skel sm-skel-line short" />
        <div className="skel sm-skel-h" />
        {[0, 1, 2].map((i) => (
          <div key={i} className="sm-skel-point">
            <div className="skel sm-skel-line" />
            <div className="skel sm-skel-chip" />
          </div>
        ))}
      </div>
    </div>
  );
}

function WritingState({ language }: { language: Language | null }) {
  const seconds = useElapsed(true);
  const name = language ? ` ${LANGUAGE_NAME[language]}` : "";
  return (
    <>
      <div className="sm-writing" role="status">
        <strong>
          <span className="sm-spinner" aria-hidden />
          Writing the{name} summary…
        </strong>
        <p>
          It can take a moment because the summary is written by the language model running on this machine. Longer
          conversations take longer.
        </p>
        {seconds >= 20 && <p>Still working. Long chats are read in pieces and then combined.</p>}
      </div>
      <SummarySkeleton />
    </>
  );
}

function Problem({ title, detail, onRetry }: { title: string; detail: string; onRetry: () => void }) {
  return (
    <div className="notice" role="alert">
      <div className="notice-body">
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>
      <button type="button" className="btn btn-sm" onClick={onRetry}>
        <Icon name="refresh" />
        Try again
      </button>
    </div>
  );
}

/** The summary couldn't be written; whatever summary was there stays. */
function WriteProblem({ problem, onRetry }: { problem: NonNullable<ChatSummaryApi["problem"]>; onRetry: () => void }) {
  if (problem.unreachable) return <BackendDown message={problem.message} onRetry={onRetry} />;
  const detail = problem.unavailable
    ? "The language model isn't available right now. Make sure it is running, then try again."
    : problem.message;
  return <Problem title="Couldn't write the summary" detail={detail} onRetry={onRetry} />;
}
