"use client";

/**
 * The voice-first chat page (docs/DESIGN.md §1, §3.8, §3.9). Voice conversation is the product; text is for
 * revisiting and for when speaking isn't possible.
 *
 *   stage:  presence field · live captions · what was searched on the web and the current answer's sources · state
 *           label · mic controls · "Type instead"
 *   panel:  the transcript and the summary ("Transcript | Summary" tabs, components/Summary.tsx), beside the stage (or
 *           over it on narrow screens); the stage keeps its mic either way
 *
 * Keys: Space or Enter on the focused mic button starts and ends the conversation; Esc steps back one level (closes
 * the transcript when it covers the stage, else stops the answer in progress, else ends the conversation).
 */

import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { Chat, ProjectDocument, SourcesPayload } from "@/lib/api";
import { useBackend } from "@/lib/backend-context";
import { citedIds, toSourceRefs, type SourceRef } from "@/lib/citations";
import { LIVE_NOTE_TEXT, answerKindOf, basisOf, liveNoteOf, showsSources, sourcesToShow } from "@/lib/route";
import { consumeAutoStart } from "@/lib/voice/autostart";
import { MIC_ERROR_TEXT, voiceSupport, type MicErrorKind } from "@/lib/voice/capture";
import { searchingNow, type VoiceSession, type VoiceSnapshot } from "@/lib/voice/session";

import { CitationPopoverProvider, SourceList } from "../Citations";
import { Icon } from "../Icon";
import { WebSearchNote } from "../WebSearchNote";
import { Captions } from "./Captions";
import { PresenceField } from "./PresenceField";

// ------------------------------------------------------------------ state label

function stateText(s: VoiceSnapshot, unavailable: boolean): string {
  switch (s.phase) {
    case "starting":
      return "Waiting for the microphone…";
    case "connecting":
      return "Connecting…";
    case "reconnecting":
      return "Connection lost · reconnecting…";
    case "idle":
      if (s.fatal) return "Voice stopped";
      if (s.micError || unavailable) return "Microphone unavailable";
      if (s.needsGesture) return "Tap the microphone to start";
      return s.turn || s.messages.length > 0 ? "Paused · tap the microphone to talk" : "Tap the microphone to talk";
    case "live":
      if (s.ducked && s.audible) return "Interrupted · listening";
      // The server stays in `speaking` through the silent search after the filler: the label follows the search then.
      if (!s.audible && searchingNow(s)) return "Searching the web…";
      if (s.audible || s.serverState === "speaking") return "Speaking";
      if (s.serverState === "thinking") return "Thinking…";
      if (s.serverState === "interrupted") return "Interrupted · listening";
      return "Listening…";
  }
}

const STAGE_TITLE: Record<string, string> = {
  stt: "Couldn't make out what you said",
  retrieval: "Couldn't search your documents",
  llm: "The language model couldn't answer",
  tts: "Couldn't speak the answer (it's in the transcript)",
  storage: "The answer couldn't be saved",
  audio: "There was an audio problem",
};

// ------------------------------------------------------------------ sources of the current answer

/**
 * The sources the current answer cites: chips appear as the text that cites them is announced, then the saved ones.
 * Only an answer that draws on the documents or the web has any: once the saved answer says it is a general-knowledge
 * answer (with no web results), an acknowledgement or a clarifying question (`route.answer`, lib/route.ts) there are no
 * chips, whatever retrieval found for it, so the sources of the previous question can't stay on screen under an answer
 * that doesn't use them. The same goes for "Not in your documents": an answer given from general knowledge or from the
 * web on purpose isn't an abstention. Web results (`[W#]`) come from the `sources` message and the search's `tool
 * results` while the answer is written (a `[W3]` that only a late `tool results` carries still has its chip), and from
 * the saved citations, which are authoritative.
 */
function useAnswerSources(turn: VoiceSnapshot["turn"], docsById: Record<string, ProjectDocument>): {
  refs: SourceRef[];
  abstained: boolean;
} {
  return useMemo(() => {
    if (!turn) return { refs: [], abstained: false };
    const message = turn.message;
    const kind = message ? answerKindOf(message) : null;
    const saved = message ? toSourceRefs(message.citations, docsById) : [];
    const savedWeb = saved.some((r) => r.kind === "web");
    if (message && !showsSources(kind)) return { refs: [], abstained: false };
    const live: SourcesPayload["sources"] = turn.sources?.sources ?? [];
    const basis = message ? basisOf(message, { documents: saved.some((r) => r.kind === "document"), web: savedWeb }) : null;
    const webResults = turn.web.sources.length > 0 || live.some((c) => c.kind === "web");
    const abstained = !basis && !webResults && (turn.sources?.abstained ?? false);
    if (saved.length > 0) return { refs: sourcesToShow(kind, saved), abstained };
    // The filler ("Let me look that up.") is spoken but isn't the answer: it can't cite anything.
    const spoken = turn.chunks.filter((c) => !c.filler);
    const text = turn.message?.text ?? (spoken.length ? spoken.map((c) => c.text).join(" ") : turn.deltaText);
    const byId = new Map(
      [...toSourceRefs(turn.web.sources, docsById), ...toSourceRefs(live, docsById)].map((r) => [r.sourceId, r] as const),
    );
    const refs = citedIds(text)
      .map((id) => byId.get(id))
      .filter((r): r is SourceRef => !!r);
    return { refs, abstained };
  }, [turn, docsById]);
}

// ------------------------------------------------------------------ the page

export interface VoiceChatProps {
  chat: Chat;
  docsById: Record<string, ProjectDocument>;
  session: VoiceSession;
  snapshot: VoiceSnapshot;
  /** Starts listening as soon as the page is shown (a new chat; the click that made it allows audio). */
  autoStart: boolean;
  panelOpen: boolean;
  onPanelOpenChange: (open: boolean) => void;
  typeOpen: boolean;
  onTypeOpenChange: (open: boolean) => void;
  /** Messages in the transcript, for the toggle's badge. */
  messageCount: number;
  /** The panel's body: the transcript and the summary (components/Summary.tsx). */
  panel: ReactNode;
  /** The panel header's tabs ("Transcript | Summary"). */
  panelTabs: ReactNode;
  /** The text composer (shown under "Type instead"). */
  composer: ReactNode;
  /** Language of the conversation, for captions. */
  language: string | null;
  /**
   * The live visual canvas (docs §12.1), or null when there is none yet. While it shows, the presence field docks: it
   * shrinks into a band at the top, keeps breathing as the presence indicator, and the visuals take the stage.
   */
  canvas?: ReactNode;
  /** Visuals on the canvas, for the toggle's badge. */
  canvasCount?: number;
}

export function VoiceChat({
  chat,
  docsById,
  session,
  snapshot,
  autoStart,
  panelOpen,
  onPanelOpenChange,
  typeOpen,
  onTypeOpenChange,
  messageCount,
  panel,
  panelTabs,
  composer,
  language,
  canvas = null,
  canvasCount = 0,
}: VoiceChatProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const dockRef = useRef<HTMLDivElement>(null);
  const slotRef = useRef<HTMLDivElement>(null);
  const micRef = useRef<HTMLButtonElement>(null);
  const panelToggleRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const typeToggleRef = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const typeId = useId();

  // What this browser can do, and whether the microphone is already blocked (known without asking).
  const [support, setSupport] = useState<{ ok: true } | { ok: false; kind: MicErrorKind } | null>(null);
  const [permission, setPermission] = useState<"granted" | "denied" | "prompt">("prompt");
  useEffect(() => {
    const s = voiceSupport();
    setSupport(s);
    if (!s.ok) return;
    let status: PermissionStatus | null = null;
    let alive = true;
    const sync = () => status && alive && setPermission(status.state);
    navigator.permissions
      ?.query({ name: "microphone" as PermissionName })
      .then((p) => {
        status = p;
        sync();
        p.addEventListener("change", sync);
      })
      .catch(() => undefined); // Safari can't be asked: the first start finds out
    return () => {
      alive = false;
      status?.removeEventListener("change", sync);
    };
  }, []);

  // The canvas takes the stage and the presence field docks; the toggle lets the person bring the full field back. A
  // visual arriving or leaving brings the canvas back.
  const [visualsHidden, setVisualsHidden] = useState(false);
  useEffect(() => setVisualsHidden(false), [canvasCount]);
  const docked = canvas !== null && !visualsHidden;
  const canvasId = useId();

  const unsupported = support !== null && !support.ok;
  const blocked = permission === "denied" && snapshot.phase === "idle";
  const live = snapshot.phase !== "idle";

  // Browsers without voice get the text composer straight away.
  useEffect(() => {
    if (unsupported) onTypeOpenChange(true);
  }, [unsupported, onTypeOpenChange]);

  // A new chat starts listening (once the click that created it has allowed audio). Deferred a tick so React's
  // development double-mount doesn't start twice.
  const autoRef = useRef(autoStart);
  useEffect(() => {
    if (!autoRef.current || support === null || !support.ok) return;
    const timer = window.setTimeout(() => {
      autoRef.current = false;
      consumeAutoStart(chat.id);
      void session.start({ auto: true });
    }, 0);
    micRef.current?.focus({ preventScroll: true });
    return () => window.clearTimeout(timer);
  }, [session, support, chat.id]);

  // ---- layout: the transcript sits beside the stage, or over it when there isn't room
  const [overlay, setOverlay] = useState(false);
  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const measure = () => {
      setOverlay(root.clientWidth < 880);
      root.style.setProperty("--vc-dock", `${dockRef.current?.offsetHeight ?? 150}px`);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(root);
    if (dockRef.current) ro.observe(dockRef.current);
    return () => ro.disconnect();
  }, []);

  // The panel mounts the first time it opens and stays (keeps its place and what it loaded).
  const [panelMounted, setPanelMounted] = useState(panelOpen);
  useEffect(() => {
    if (panelOpen) setPanelMounted(true);
  }, [panelOpen]);

  const togglePanel = useCallback(() => {
    onPanelOpenChange(!panelOpen);
  }, [onPanelOpenChange, panelOpen]);
  const wasOpen = useRef(panelOpen);
  useEffect(() => {
    // Focus follows the panel: into it when opened by the button, back to the button when closed.
    if (panelOpen && !wasOpen.current) {
      panelRef.current?.querySelector<HTMLElement>("[role=tab][aria-selected=true]")?.focus({ preventScroll: true });
    }
    if (!panelOpen && wasOpen.current && rootRef.current?.contains(document.activeElement)) {
      panelToggleRef.current?.focus({ preventScroll: true });
    }
    wasOpen.current = panelOpen;
  }, [panelOpen]);

  const agentBusy = live && (snapshot.audible || snapshot.serverState === "thinking" || snapshot.serverState === "speaking");

  // Esc steps back one level.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // A held key repeats: only the first press counts (stopping the answer and then ending the conversation is two presses).
      if (e.key !== "Escape" || e.defaultPrevented || e.repeat) return;
      // A dialog, a menu or the sidebar drawer is open: Esc belongs to it.
      if (document.querySelector("dialog[open], [role=menu], .app[data-drawer=open]")) return;
      const target = e.target instanceof Element ? e.target : null;
      if (target?.closest("textarea, input, [contenteditable=true]")) return; // the composer has its own Esc
      if (panelOpen && overlay) {
        e.preventDefault();
        onPanelOpenChange(false);
      } else if (agentBusy) {
        e.preventDefault();
        session.stopAnswer();
      } else if (snapshot.phase !== "idle") {
        e.preventDefault();
        void session.stop();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [panelOpen, overlay, agentBusy, snapshot.phase, session, onPanelOpenChange]);

  // Typing instead: expanding moves focus to the box, collapsing returns it to the toggle.
  const wasTyping = useRef(typeOpen);
  useEffect(() => {
    if (typeOpen && !wasTyping.current) {
      document.getElementById(typeId)?.querySelector<HTMLElement>("textarea")?.focus({ preventScroll: true });
    }
    if (!typeOpen && wasTyping.current && document.activeElement === document.body) typeToggleRef.current?.focus();
    wasTyping.current = typeOpen;
  }, [typeOpen, typeId]);

  const { refs: sourceRefs, abstained } = useAnswerSources(snapshot.turn, docsById);
  // What was searched on the web (the session leaves it out when `features.web_search` is off), and the hint about live
  // data that couldn't be had. The previous answer's note goes while the user is speaking the next question.
  const { config } = useBackend();
  const userTalking = snapshot.caption === "user" && !snapshot.userFinal;
  const search = userTalking ? null : (snapshot.turn?.web.search ?? null);
  const liveNote = config?.features.web_search === true && snapshot.turn?.message && !userTalking ? liveNoteOf(snapshot.turn.message) : null;
  const searching = searchingNow(snapshot);

  // ---- the mic
  const micError = snapshot.micError ?? (blocked ? { kind: "denied" as const, message: MIC_ERROR_TEXT.denied } : null);
  // Buttons that are only briefly unavailable use aria-disabled, not `disabled`: a button that becomes disabled drops
  // keyboard focus, and the person pressing Space on the mic would lose their place.
  const starting = snapshot.phase === "starting";
  const onMic = () => {
    if (starting) return;
    if (live) void session.stop();
    else void session.start();
  };
  const onStopAnswer = () => {
    if (!agentBusy) return;
    session.stopAnswer();
    micRef.current?.focus({ preventScroll: true }); // the stop button is about to go dim
  };
  const label = stateText(snapshot, micError !== null);
  const tone = live ? (searching && !snapshot.audible ? "searching" : (snapshot.serverState ?? "listening")) : "idle";
  const micLabel = live ? "End the voice conversation" : "Start a voice conversation";
  const notice = snapshot.notice;
  const showRetry = micError && micError.kind !== "unsupported" && micError.kind !== "insecure";

  return (
    <div
      ref={rootRef}
      className="vc"
      data-panel={panelOpen ? "open" : "closed"}
      data-overlay={overlay || undefined}
      data-phase={snapshot.phase}
      data-canvas={canvas !== null ? "on" : undefined}
      data-docked={docked || undefined}
    >
      <section className="vc-stage" aria-label="Voice conversation">
        <PresenceField session={session} anchorRef={slotRef} />

        <div className="vc-top">
          {canvas !== null && (
            <button
              type="button"
              className="btn btn-sm vc-panel-toggle"
              aria-expanded={docked}
              aria-controls={canvasId}
              onClick={() => setVisualsHidden((v) => !v)}
            >
              <Icon name="sparkle" size={15} />
              {docked ? "Hide visuals" : "Show visuals"}
              {canvasCount > 0 && <span className="vc-count">{canvasCount}</span>}
            </button>
          )}
          <button
            ref={panelToggleRef}
            type="button"
            className="btn btn-sm vc-panel-toggle"
            aria-expanded={panelOpen}
            aria-controls={panelId}
            onClick={togglePanel}
          >
            <Icon name="chat" size={15} />
            Transcript
            {messageCount > 0 && <span className="vc-count">{messageCount > 99 ? "99+" : messageCount}</span>}
          </button>
        </div>

        <div className="vc-display">
          <div ref={slotRef} className="vc-slot" />
          {canvas !== null && (
            <div id={canvasId} className="vc-visuals" data-scroll-root hidden={!docked}>
              {canvas}
            </div>
          )}
          <div className="vc-say">
            <Captions
              session={session}
              snapshot={snapshot}
              language={language}
              hint={
                unsupported
                  ? "Voice isn't available here. Type your question below."
                  : live
                    ? "Ask about your documents."
                    : "Tap the microphone, then ask about your documents."
              }
            />
            <CitationPopoverProvider>
              <div className="vc-sources" aria-live="off">
                {search && <WebSearchNote search={search} stage />}
                {liveNote && (
                  <span className="vc-abstain">
                    <Icon name="info" size={13} />
                    {LIVE_NOTE_TEXT[liveNote]}
                  </span>
                )}
                {snapshot.caption === "user" ? null : abstained && snapshot.turn?.message ? (
                  <span className="vc-abstain">
                    <Icon name="info" size={13} />
                    Not in your documents
                  </span>
                ) : (
                  <SourceList sources={sourceRefs} />
                )}
              </div>
            </CitationPopoverProvider>
          </div>
        </div>

        <div ref={dockRef} className="vc-dock">
          <p className="vc-state" role="status" aria-live="polite" data-tone={tone}>
            {label}
          </p>

          {(micError || snapshot.fatal) && (
            <div className="vc-alert" role={snapshot.micError || snapshot.fatal ? "alert" : undefined}>
              <Icon name="alert" size={15} className="vc-alert-icon" />
              <span>{snapshot.fatal ?? micError?.message}</span>
              {(snapshot.fatal ? snapshot.fatalRetry : showRetry) && !unsupported && (
                <button type="button" className="btn btn-sm" onClick={() => void session.start()} aria-disabled={starting || undefined}>
                  <Icon name="refresh" size={14} />
                  {snapshot.fatal ? "Reconnect" : "Try again"}
                </button>
              )}
            </div>
          )}

          {notice && (
            <div key={notice.key} className="vc-alert vc-notice" role="alert">
              <Icon name="alert" size={15} className="vc-alert-icon" />
              <span>
                <strong>{STAGE_TITLE[notice.stage ?? ""] ?? "Something went wrong"}.</strong> {notice.detail}
              </span>
              <button type="button" className="icon-btn icon-btn-sm" aria-label="Dismiss" onClick={() => session.dismissNotice()}>
                <Icon name="close" size={14} />
              </button>
            </div>
          )}

          <div className="vc-controls">
            <button
              type="button"
              className="vc-round vc-small"
              onClick={onStopAnswer}
              aria-disabled={!agentBusy || undefined}
              tabIndex={agentBusy ? undefined : -1}
              aria-label="Stop the answer"
              title="Stop the answer (Esc)"
            >
              <Icon name="stop" size={16} />
            </button>

            <button
              ref={micRef}
              type="button"
              className="vc-round vc-mic"
              data-live={live || undefined}
              data-busy={snapshot.phase === "starting" || snapshot.phase === "connecting" || snapshot.phase === "reconnecting" || undefined}
              aria-label={micLabel}
              title={micLabel}
              disabled={unsupported}
              aria-disabled={starting || undefined}
              onClick={onMic}
            >
              <Icon name="mic" size={28} />
            </button>

            <button
              ref={typeToggleRef}
              type="button"
              className="vc-round vc-small"
              aria-expanded={typeOpen}
              aria-controls={typeId}
              aria-label="Type instead"
              title="Type instead"
              onClick={() => onTypeOpenChange(!typeOpen)}
            >
              <Icon name="keyboard" size={18} />
            </button>
          </div>
          <p className="vc-hint" aria-hidden="true">
            {live ? "Tap to end · Esc stops" : "Tap to talk"}
          </p>

          <div id={typeId} className="vc-type" hidden={!typeOpen}>
            {typeOpen && (
              <>
                <p className="vc-type-label">
                  <Icon name="keyboard" size={14} />
                  Type instead
                </p>
                {composer}
              </>
            )}
          </div>
        </div>
      </section>

      <aside ref={panelRef} id={panelId} className="vc-panel" aria-label="Transcript and summary" hidden={!panelOpen}>
        <div className="vc-panel-head">
          {panelTabs}
          <button type="button" className="icon-btn" aria-label="Close the transcript" onClick={() => onPanelOpenChange(false)}>
            <Icon name="close" size={16} />
          </button>
        </div>
        <div className="vc-panel-body">{panelMounted && panel}</div>
      </aside>
    </div>
  );
}
