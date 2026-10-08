"use client";

/**
 * /chats/[chatId]: breadcrumb (project › chat), rename/pin/archive/delete, documents in scope, the transcript and
 * a composer that stays disabled until text chat arrives with document ingestion. The voice view (§3.8) comes later.
 */

import Link from "next/link";
import { useEffect, useMemo } from "react";

import type { Chat, ProjectDocument } from "@/lib/api";
import { useBackend, useDocumentTitle } from "@/lib/backend-context";
import { LANGUAGE_NAMES } from "@/lib/format";
import { keys, projectName, slotOf, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useEntityActions } from "./Actions";
import { Icon } from "./Icon";
import { Menu } from "./Menu";
import { LoadFailed } from "./States";
import { Transcript } from "./Transcript";

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
  return <ChatPage chat={chat} />;
}

function ChatPage({ chat }: { chat: Chat }) {
  const state = useWorkspace();
  const actions = useEntityActions();
  const docs = state.documents[chat.project_id] ?? null;
  const docsById = useMemo(() => Object.fromEntries((docs ?? []).map((d) => [d.id, d])), [docs]);
  const menu = actions.chatMenu(chat).filter((i) => i.id === "archive" || i.id === "delete");
  const project = projectName(state, chat.project_id);

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
        <ScopeBar chat={chat} docs={docs} docsById={docsById} />
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

      <Transcript chat={chat} docsById={docsById} />
      <Composer />
    </div>
  );
}

const MAX_DOC_CHIPS = 4;

function ScopeBar({
  chat,
  docs,
  docsById,
}: {
  chat: Chat;
  docs: ProjectDocument[] | null;
  docsById: Record<string, ProjectDocument>;
}) {
  const { config } = useBackend();
  const scoped = chat.document_scope;
  const names = scoped ? scoped.map((id) => docsById[id]?.filename ?? "Unknown document") : (docs ?? []).map((d) => d.filename);
  const shown = names.slice(0, MAX_DOC_CHIPS);
  const languages = chat.language ? [chat.language] : (config?.client.languages ?? []);

  return (
    <div className="scope" aria-label="Documents in scope and language" role="group">
      {docs === null ? (
        <span className="chip chip-skeleton" aria-hidden />
      ) : names.length === 0 ? (
        <span className="chip chip-muted">
          <Icon name="doc" size={13} />
          No documents in this project yet
        </span>
      ) : (
        <>
          <span className="scope-label">{scoped ? "Answers from" : "All documents"}</span>
          {shown.map((name, i) => (
            <span key={`${name}-${i}`} className="chip" title={name}>
              <Icon name="doc" size={13} />
              <span className="chip-text">{name}</span>
            </span>
          ))}
          {names.length > shown.length && <span className="chip chip-muted">+{names.length - shown.length} more</span>}
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

function Composer() {
  return (
    <div className="composer">
      <div className="composer-box" aria-disabled="true">
        <textarea
          className="composer-input"
          rows={1}
          disabled
          placeholder="Chat arrives with document ingestion"
          aria-label="Message"
          aria-describedby="composer-note"
        />
        <button type="button" className="composer-send" disabled aria-label="Send">
          <Icon name="arrowUp" size={16} />
        </button>
      </div>
      <p id="composer-note" className="composer-note">
        Typing to the agent arrives with document ingestion, speaking in a later phase. Every chat's transcript is kept.
      </p>
    </div>
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
      <Composer />
    </div>
  );
}
