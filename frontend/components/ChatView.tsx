"use client";

/**
 * /chats/[chatId]: breadcrumb (project › chat), rename/pin/archive/delete, the documents the chat answers from
 * (toggle a chip to narrow or widen the chat's document scope), then the conversation.
 *
 * Voice first (docs §1, §3.9): the page opens in voice mode (components/voice/VoiceChat: presence field, mic, live
 * captions, the answer's sources); the transcript is a panel beside it, open by default when the chat already has
 * messages; "Type instead" expands the text composer: Enter sends, Shift+Enter adds a line, Stop (or Esc) ends a
 * streaming answer. Without a READY document in scope the composer explains why it can't send and links to
 * uploading. When the backend turns `features.voice` off, the page is the plain transcript + composer.
 */

import Link from "next/link";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { MESSAGE_MAX_CHARS, errorMessage, type Chat, type Language, type ProjectDocument } from "@/lib/api";
import { useBackend, useDocumentTitle } from "@/lib/backend-context";
import { useChatTurns } from "@/lib/chat-turns";
import { LANGUAGE_NAMES } from "@/lib/format";
import { wantsAutoStart } from "@/lib/voice/autostart";
import { useVoiceSession } from "@/lib/voice/use-voice-session";
import { isIngesting, keys, projectName, slotOf, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useEntityActions } from "./Actions";
import { DOCUMENT_STATUS } from "./Documents";
import { Icon } from "./Icon";
import { Menu } from "./Menu";
import { LoadFailed } from "./States";
import { useToast } from "./Toast";
import { Transcript } from "./Transcript";
import { VoiceChat } from "./voice/VoiceChat";

export function ChatView({ chatId }: { chatId: string }) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const chat = state.chats[chatId];
  const slot = slotOf(state, keys.entity(chatId));
  useDocumentTitle(chat?.title ?? null);

  useEffect(() => {
    void ws.loadChat(chatId, true);
  }, [chatId, ws]);

  const projectId = chat?.project_id;
  useEffect(() => {
    if (!projectId) return;
    void ws.loadProject(projectId);
    void ws.loadDocuments(projectId);
  }, [projectId, ws]);

  if (!chat) {
    if (slot.status === "error") {
      return (
        <div className="page">
          <LoadFailed slot={slot} kind="chat" onRetry={() => void ws.loadChat(chatId, true)} />
        </div>
      );
    }
    return <ChatSkeleton />;
  }
  // Keyed by chat: a different chat starts with its own transcript and no answer in flight.
  return <ChatPage key={chat.id} chat={chat} />;
}

const LANGUAGES: readonly Language[] = ["en", "hi"];

interface Blocked {
  reason: string;
  /** Composer placeholder while blocked. */
  hint: string;
  /** Text of the link to the project's documents. */
  link: string;
  projectId: string;
}

/** Why the chat can't answer yet, or null when at least one document it uses is READY. */
function readiness(chat: Chat, docs: ProjectDocument[] | null): { canAsk: boolean; blocked: Blocked | null } {
  if (docs === null) return { canAsk: false, blocked: null };
  const inScope = chat.document_scope === null ? docs : docs.filter((d) => chat.document_scope?.includes(d.id));
  if (inScope.some((d) => d.status === "READY")) return { canAsk: true, blocked: null };
  const scoped = chat.document_scope !== null;
  const projectId = chat.project_id;
  if (docs.length === 0) {
    const reason = "This project has no documents yet, so there's nothing to answer from.";
    return { canAsk: false, blocked: { reason, hint: "Upload a document to start asking", link: "Upload documents", projectId } };
  }
  if (scoped && inScope.length === 0) {
    // e.g. the only documents it used were deleted
    const reason = "This chat doesn't use any of the project's documents now. Turn one on above.";
    return { canAsk: false, blocked: { reason, hint: "Choose a document to answer from", link: "Manage documents", projectId } };
  }
  if (inScope.some(isIngesting)) {
    const reason = `${scoped ? "The documents this chat uses are" : "Your documents are"} still being processed. You can ask once one is ready.`;
    return { canAsk: false, blocked: { reason, hint: "Waiting for a document to be ready…", link: "See their progress", projectId } };
  }
  const reason = scoped
    ? "None of the documents this chat uses is ready. Turn on a ready one above."
    : "None of this project's documents could be processed.";
  return { canAsk: false, blocked: { reason, hint: "No ready documents to answer from", link: "Manage documents", projectId } };
}

function ChatPage({ chat }: { chat: Chat }) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const { api } = useBackend();
  const actions = useEntityActions();
  const docs = state.documents[chat.project_id] ?? null;
  const docsById = useMemo(() => Object.fromEntries((docs ?? []).map((d) => [d.id, d])), [docs]);
  const menu = actions.chatMenu(chat).filter((i) => i.id === "archive" || i.id === "delete");
  const project = projectName(state, chat.project_id);

  // After each answer: message count, activity time and (later) the automatic title change.
  const chatId = chat.id;
  const projectId = chat.project_id;
  const onSettled = useCallback(() => {
    void ws.loadChat(chatId, true);
    void ws.loadChats(projectId, true);
    void ws.loadProjects(true);
  }, [ws, chatId, projectId]);
  const conversation = useChatTurns(api, chatId, onSettled);
  const { canAsk, blocked } = readiness(chat, docs);
  const language = LANGUAGES.includes(chat.language as Language) ? (chat.language as Language) : null;

  // Voice first (docs §1, §3.9): the page opens in voice mode, the transcript is a panel (open when reopening a chat
  // that has messages), and typing is a collapsed fallback. A new chat starts listening at once.
  const { config } = useBackend();
  const voiceEnabled = config?.features.voice !== false;
  const { session, snapshot } = useVoiceSession(chat.id, language);
  const [autoStart] = useState(() => wantsAutoStart(chat.id));
  const [panelOpen, setPanelOpen] = useState(() => chat.message_count > 0);
  const [typeOpen, setTypeOpen] = useState(false);

  // Messages the voice session saves change the chat's count, activity and (later) title.
  const savedByVoice = snapshot.messages.length;
  useEffect(() => {
    if (savedByVoice > 0) onSettled();
  }, [savedByVoice, onSettled]);

  // The backend turned voice off while the page was open (the public config was reloaded): let go of the microphone
  // and the socket.
  useEffect(() => {
    if (!voiceEnabled) void session.stop();
  }, [voiceEnabled, session]);

  // After a dropped connection the voice service starts a new session, which never sent what the old one saved (an
  // answer that was being given, the last question): read the end of the transcript again.
  const resyncs = snapshot.resyncs;
  const knownCount = useRef(chat.message_count);
  useEffect(() => {
    knownCount.current = chat.message_count;
  }, [chat.message_count]);
  useEffect(() => {
    if (resyncs === 0) return;
    const ctrl = new AbortController();
    const seen = Math.max(knownCount.current, ...session.getSnapshot().messages.map((m) => m.seq));
    api
      .listMessages(chatId, { after: seen, limit: 50 }, ctrl.signal)
      .then((page) => {
        session.mergeMessages(page.items);
        onSettled();
      })
      .catch(() => undefined); // the next save refreshes it anyway
    return () => ctrl.abort();
  }, [resyncs, api, chatId, session, onSettled]);

  // What's being said right now but not saved yet, for the transcript.
  const voiceTurn = snapshot.turn;
  const voiceLive = useMemo(() => {
    const speaking = snapshot.caption === "user" && !snapshot.userFinal && (snapshot.userSpeaking || snapshot.userText !== "");
    const answering =
      !!voiceTurn && !voiceTurn.message && voiceTurn.deltaText !== "" && (snapshot.audible || snapshot.serverState !== "listening");
    return {
      userText: speaking ? snapshot.userText : null,
      agentText: answering ? voiceTurn.deltaText : null,
      sources: voiceTurn?.sources ?? null,
    };
  }, [snapshot.caption, snapshot.userFinal, snapshot.userSpeaking, snapshot.userText, snapshot.audible, snapshot.serverState, voiceTurn]);

  const transcript = (
    <Transcript
      chat={chat}
      docsById={docsById}
      turns={conversation.turns}
      busy={conversation.busy}
      onRetry={canAsk ? conversation.retry : undefined}
      voiceMessages={snapshot.messages}
      voice={voiceLive}
    />
  );
  const composer = (
    <Composer
      busy={conversation.busy}
      loading={docs === null}
      blocked={blocked}
      onSend={(text) => {
        const sent = conversation.send(text, language);
        if (sent) setPanelOpen(true); // the typed answer shows in the transcript
        return sent;
      }}
      onStop={conversation.stop}
      announcement={conversation.announcement}
    />
  );

  return (
    <div className="chat-page">
      <header className="chat-head">
        <div className="chat-titlebar">
          <nav aria-label="Breadcrumb" className="crumbs">
            <Link href={`/projects/${chat.project_id}`} className="crumb">
              <Icon name="folder" size={14} />
              <span className="crumb-text">{project ?? "Project"}</span>
            </Link>
            <Icon name="chevron" size={14} className="crumb-sep" />
          </nav>
          <h1 className="chat-title" title={chat.title}>
            {chat.title}
          </h1>
          {chat.archived && <span className="tag">Archived</span>}
          <div className="head-actions">
            <button
              type="button"
              className="icon-btn"
              aria-pressed={chat.pinned}
              aria-label="Pin chat"
              title={chat.pinned ? "Unpin" : "Pin to sidebar"}
              onClick={() => void actions.setChatPinned(chat, !chat.pinned)}
            >
              <Icon name="star" filled={chat.pinned} className={chat.pinned ? "star-on" : undefined} />
            </button>
            <button
              type="button"
              className="icon-btn"
              aria-label="Rename chat"
              title="Rename"
              onClick={() => actions.renameChat(chat)}
            >
              <Icon name="pencil" />
            </button>
            <Menu label="More chat actions" items={menu} className="icon-btn" />
          </div>
        </div>
        <ScopeBar chat={chat} docs={docs} />
      </header>

      {chat.archived && (
        <div className="banner banner-inset">
          <Icon name="archive" />
          <span>This chat is archived and hidden from the sidebar.</span>
          <button type="button" className="btn btn-sm" onClick={() => void actions.setChatArchived(chat, false)}>
            Restore
          </button>
        </div>
      )}

      {voiceEnabled ? (
        <VoiceChat
          chat={chat}
          docsById={docsById}
          session={session}
          snapshot={snapshot}
          autoStart={autoStart}
          panelOpen={panelOpen}
          onPanelOpenChange={setPanelOpen}
          typeOpen={typeOpen}
          onTypeOpenChange={setTypeOpen}
          messageCount={chat.message_count + snapshot.messages.filter((m) => m.seq > chat.message_count).length}
          transcript={transcript}
          composer={composer}
          language={language}
        />
      ) : (
        <>
          {transcript}
          {composer}
        </>
      )}
    </div>
  );
}

const MAX_DOC_CHIPS = 4;

/**
 * Documents the chat answers from. Each chip is a toggle: pressed = used by this chat. Leaving every document
 * pressed stores `document_scope: null` ("all documents", including ones uploaded later).
 */
function ScopeBar({ chat, docs }: { chat: Chat; docs: ProjectDocument[] | null }) {
  const { config } = useBackend();
  const ws = useWorkspaceActions();
  const toast = useToast();
  const [expanded, setExpanded] = useState(false);
  const [saving, setSaving] = useState<string | null>(null);
  const groupRef = useRef<HTMLDivElement>(null);
  const scope = chat.document_scope;
  const all = docs ?? [];
  const used = (id: string) => scope === null || scope.includes(id);
  const usedCount = all.filter((d) => used(d.id)).length;
  const shown = expanded ? all : all.slice(0, MAX_DOC_CHIPS);
  const languages = chat.language ? [chat.language] : (config?.client.languages ?? []);

  // Buttons stay enabled (aria-disabled) while saving, so keyboard focus isn't dropped; clicks are ignored then.
  const save = async (next: string[] | null, docId: string) => {
    setSaving(docId);
    try {
      await ws.updateChat(chat.id, { document_scope: next });
      // "Use all" disappears once it has worked: keep focus in the group.
      if (docId === "all") groupRef.current?.querySelector<HTMLElement>(".chip-toggle")?.focus();
    } catch (err) {
      toast({ tone: "error", message: `Couldn't change the chat's documents: ${errorMessage(err)}` });
    } finally {
      setSaving(null);
    }
  };

  const toggle = (doc: ProjectDocument) => {
    if (saving !== null) return;
    const current = scope ?? all.map((d) => d.id);
    const next = current.includes(doc.id) ? current.filter((id) => id !== doc.id) : [...current, doc.id];
    if (next.length === 0) {
      toast({ message: "A chat answers from at least one document. Turn another one on first." });
      return;
    }
    void save(all.every((d) => next.includes(d.id)) ? null : next, doc.id);
  };

  return (
    <div ref={groupRef} className="scope" aria-label="Documents this chat answers from, and language" role="group">
      {docs === null ? (
        <span className="chip chip-skeleton" aria-hidden />
      ) : all.length === 0 ? (
        <Link href={`/projects/${chat.project_id}#documents`} className="chip chip-muted chip-link">
          <Icon name="upload" size={13} />
          No documents yet · upload
        </Link>
      ) : (
        <>
          <span className="scope-label">{scope === null ? "All documents" : `Using ${usedCount} of ${all.length}`}</span>
          {shown.map((d) => {
            const on = used(d.id);
            const status = d.status === "READY" ? null : (DOCUMENT_STATUS[d.status] ?? { label: d.status, tone: "idle" });
            return (
              <button
                key={d.id}
                type="button"
                className="chip chip-toggle"
                aria-pressed={on}
                aria-disabled={saving !== null || undefined}
                data-saving={saving === d.id || undefined}
                title={`${d.filename}${status ? ` (${status.label.toLowerCase()})` : ""}. ${on ? "Used by this chat: click to leave it out." : "Not used by this chat: click to include it."}`}
                onClick={() => toggle(d)}
              >
                {status ? (
                  <span className={`dot ${status.tone}${isIngesting(d) ? " dot-busy" : ""}`} aria-hidden />
                ) : (
                  <Icon name={on ? "check" : "doc"} size={13} />
                )}
                <span className="chip-text">{d.filename}</span>
                {status && <span className="visually-hidden">, {status.label}</span>}
              </button>
            );
          })}
          {all.length > MAX_DOC_CHIPS && (
            <button type="button" className="chip chip-muted" aria-expanded={expanded} onClick={() => setExpanded((v) => !v)}>
              {expanded ? "Show fewer" : `+${all.length - MAX_DOC_CHIPS} more`}
            </button>
          )}
          {scope !== null && (
            <button
              type="button"
              className="link-btn scope-reset"
              aria-disabled={saving !== null || undefined}
              onClick={() => saving === null && void save(null, "all")}
            >
              Use all
            </button>
          )}
        </>
      )}
      {languages.length > 0 && (
        <span
          className="chip scope-lang"
          title={
            chat.language
              ? `Answers in ${LANGUAGE_NAMES[chat.language] ?? chat.language}`
              : "Answers in the language you use"
          }
        >
          {languages.map((l) => l.toUpperCase()).join(" · ")}
        </span>
      )}
    </div>
  );
}

const MAX_ROWS_PX = 200;
const COUNT_FROM = MESSAGE_MAX_CHARS - 500;

function Composer({
  busy,
  loading = false,
  blocked,
  onSend,
  onStop,
  announcement = "",
}: {
  busy: boolean;
  /** Documents are still loading: we don't know yet whether the chat can answer. */
  loading?: boolean;
  blocked: Blocked | null;
  onSend?: (text: string) => boolean;
  onStop?: () => void;
  announcement?: string;
}) {
  const [text, setText] = useState("");
  const [nudge, setNudge] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const buttonFocused = useRef(false);

  // The stop button turns back into a disabled Send when the answer ends; don't leave keyboard focus on it.
  useEffect(() => {
    if (!busy && buttonFocused.current) {
      buttonFocused.current = false;
      inputRef.current?.focus({ preventScroll: true });
    }
  }, [busy]);
  const noteId = useId();
  const countId = useId();
  const trimmed = text.trim();
  const length = trimmed.length;
  const over = length > MESSAGE_MAX_CHARS;
  const disabled = !onSend;
  const canSend = !disabled && !busy && !loading && !blocked && length > 0 && !over;

  // Grow with the text up to a limit, then scroll.
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_ROWS_PX)}px`;
  }, [text]);

  const submit = () => {
    if (!canSend) {
      if (blocked && length > 0) setNudge((n) => n + 1);
      return;
    }
    if (onSend?.(trimmed)) setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter while composing (Hindi and other input methods) picks a candidate; it must not send.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape" && busy) {
      e.preventDefault();
      onStop?.();
    }
  };

  const placeholder = disabled
    ? "Loading the chat…"
    : blocked
      ? blocked.hint
      : "Ask about your documents…";

  return (
    <form
      className="composer"
      onSubmit={(e: FormEvent) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="composer-box" aria-disabled={disabled || undefined} data-busy={busy || undefined}>
        <textarea
          ref={inputRef}
          className="composer-input"
          rows={1}
          value={text}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          aria-label="Message"
          aria-describedby={`${noteId}${length >= COUNT_FROM ? ` ${countId}` : ""}`}
          aria-invalid={over || undefined}
          enterKeyHint="send"
        />
        <button
          type="button"
          className={busy ? "composer-send composer-stop" : "composer-send"}
          disabled={!busy && !canSend}
          onClick={busy ? onStop : submit}
          onFocus={() => (buttonFocused.current = true)}
          onBlur={(e) => {
            // Losing focus because it just became disabled still counts as "was on the button".
            if (!e.currentTarget.disabled) buttonFocused.current = false;
          }}
          onPointerDown={() => (buttonFocused.current = true)}
          aria-label={busy ? "Stop the answer" : "Send"}
          title={busy ? "Stop (Esc)" : "Send (Enter)"}
        >
          <Icon name={busy ? "stop" : "arrowUp"} size={16} />
        </button>
      </div>
      <div className="composer-foot">
        {blocked ? (
          <p id={noteId} key={nudge} className={nudge ? "composer-note composer-blocked pulse-once" : "composer-note composer-blocked"}>
            <Icon name="info" size={13} />
            <span>
              {blocked.reason}{" "}
              <Link href={`/projects/${blocked.projectId}#documents`}>{blocked.link}</Link>
            </span>
          </p>
        ) : (
          <p id={noteId} className="composer-note">
            {busy ? "Answering… Esc or the stop button ends it." : "Enter to send · Shift+Enter for a new line"}
          </p>
        )}
        {length >= COUNT_FROM && (
          <span id={countId} className={over ? "composer-count over" : "composer-count"}>
            {over
              ? `${(length - MESSAGE_MAX_CHARS).toLocaleString()} characters over the ${MESSAGE_MAX_CHARS.toLocaleString()} limit`
              : `${length.toLocaleString()} / ${MESSAGE_MAX_CHARS.toLocaleString()}`}
          </span>
        )}
      </div>
      <p className="visually-hidden" role="status" aria-live="polite">
        {announcement}
      </p>
    </form>
  );
}

function ChatSkeleton() {
  return (
    <div className="chat-page" aria-busy="true" aria-label="Loading chat">
      <header className="chat-head">
        <div className="chat-titlebar">
          <div className="skel skel-crumb" />
          <div className="skel skel-chat-title" />
        </div>
        <div className="scope">
          <span className="chip chip-skeleton" aria-hidden />
        </div>
      </header>
      <div className="transcript" />
      <Composer busy={false} blocked={null} />
    </div>
  );
}

