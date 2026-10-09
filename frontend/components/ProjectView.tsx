"use client";

/**
 * /projects/[projectId]: header with rename/pin/archive/delete, the project's chats, and its documents (upload,
 * ingestion status, delete; components/Documents.tsx).
 */

import Link from "next/link";
import { useEffect, useMemo } from "react";

import type { Chat, Project } from "@/lib/api";
import { useDocumentTitle } from "@/lib/backend-context";
import { fullDateTime, plural, relativeTime, shortDate } from "@/lib/format";
import { chatActivity, chatsOf, keys, slotOf, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useEntityActions } from "./Actions";
import { ProjectOverview } from "./canvas/ProjectOverview";
import { DocumentsCard } from "./Documents";
import { Icon } from "./Icon";
import { Menu } from "./Menu";
import { BackendDown, EmptyState, LoadFailed } from "./States";

export function ProjectView({ projectId }: { projectId: string }) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const project = state.projects[projectId];
  const slot = slotOf(state, keys.entity(projectId));
  useDocumentTitle(project?.name ?? null);

  useEffect(() => {
    void ws.loadProject(projectId, true);
    void ws.loadChats(projectId);
    void ws.loadDocuments(projectId);
  }, [projectId, ws]);

  if (!project) {
    if (slot.status === "error") {
      return (
        <div className="page">
          <LoadFailed slot={slot} kind="project" onRetry={() => void ws.loadProject(projectId, true)} />
        </div>
      );
    }
    return <ProjectSkeleton />;
  }
  return <ProjectPage project={project} />;
}

function ProjectPage({ project }: { project: Project }) {
  const actions = useEntityActions();
  const docs = useWorkspace().documents[project.id];
  const docsById = useMemo(() => Object.fromEntries((docs ?? []).map((d) => [d.id, d])), [docs]);
  const menu = actions.projectMenu(project).filter((i) => i.id === "archive" || i.id === "delete");

  return (
    <div className="page">
      <header className="page-head">
        <p className="eyebrow">
          <Icon name="folder" />
          Project
        </p>
        <div className="title-row">
          <h1 className="page-title">{project.name}</h1>
          {project.archived && <span className="tag">Archived</span>}
          <div className="head-actions">
            <button
              type="button"
              className="btn btn-sm"
              aria-pressed={project.pinned}
              onClick={() => void actions.setProjectPinned(project, !project.pinned)}
            >
              <Icon name="star" filled={project.pinned} className={project.pinned ? "star-on" : undefined} />
              {project.pinned ? "Pinned" : "Pin"}
            </button>
            <button type="button" className="btn btn-sm" onClick={() => actions.renameProject(project)}>
              <Icon name="pencil" />
              Rename
            </button>
            <Menu label="More project actions" items={menu} className="icon-btn" />
          </div>
        </div>
        {project.description && <p className="lede">{project.description}</p>}
        <p className="meta">
          {plural(project.chat_count, "chat")} · {plural(project.document_count, "document")} ·{" "}
          <span title={fullDateTime(project.created_at)}>created {shortDate(project.created_at)}</span> ·{" "}
          <span title={fullDateTime(project.updated_at)}>updated {relativeTime(project.updated_at)}</span>
        </p>
      </header>

      {project.archived && (
        <div className="banner">
          <Icon name="archive" />
          <span>This project is archived, so it's hidden from the sidebar and the home page.</span>
          <button type="button" className="btn btn-sm" onClick={() => void actions.setProjectArchived(project, false)}>
            Restore
          </button>
        </div>
      )}

      <ProjectOverview projectId={project.id} docsById={docsById} />

      <div className="project-layout">
        <ChatsCard project={project} />
        <DocumentsCard project={project} />
      </div>
    </div>
  );
}

function ChatsCard({ project }: { project: Project }) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const actions = useEntityActions();
  const slot = slotOf(state, keys.chats(project.id));
  const chats = chatsOf(state, project.id);
  const active = chats?.filter((c) => !c.archived) ?? [];
  const archived = chats?.filter((c) => c.archived) ?? [];
  const creating = actions.creatingChatIn === project.id;

  return (
    <section className="card" aria-labelledby="chats-title">
      <div className="card-head">
        <h2 id="chats-title">Chats</h2>
        <button
          type="button"
          className="btn btn-sm btn-primary"
          onClick={() => void actions.newChat(project.id)}
          disabled={creating}
        >
          <Icon name="plus" />
          {creating ? "Creating…" : "New chat"}
        </button>
      </div>

      {chats === null ? (
        slot.status === "error" ? (
          slot.unreachable ? (
            <BackendDown onRetry={() => void ws.loadChats(project.id, true)} />
          ) : (
            <p className="form-error">{slot.error}</p>
          )
        ) : (
          <div className="rows" aria-hidden>
            {Array.from({ length: Math.min(Math.max(project.chat_count, 1), 4) }, (_, i) => (
              <div key={i} className="row-skeleton" />
            ))}
          </div>
        )
      ) : active.length === 0 ? (
        <EmptyState icon="chat" title="No chats yet" compact>
          <p>Start a chat to ask about this project's documents. Every chat is kept with its full transcript.</p>
        </EmptyState>
      ) : (
        <ChatRows chats={active} />
      )}

      {archived.length > 0 && (
        <details className="archived">
          <summary>
            <Icon name="chevron" size={14} className="summary-chevron" />
            Archived chats ({archived.length})
          </summary>
          <ChatRows chats={archived} />
        </details>
      )}
    </section>
  );
}

function ChatRows({ chats }: { chats: Chat[] }) {
  const actions = useEntityActions();
  return (
    <ul className="rows">
      {chats.map((c) => (
        <li key={c.id} className="row">
          <Link href={`/chats/${c.id}`} className="row-link">
            <Icon name="chat" className="row-icon" />
            <span className="row-main">
              <span className="row-title">
                {c.title}
                {c.pinned && (
                  <Icon name="star" filled size={12} className="star-on" role="img" aria-hidden={false} aria-label="Pinned" />
                )}
              </span>
              <span className="row-meta">
                {plural(c.message_count, "message")} · {relativeTime(chatActivity(c))}
              </span>
            </span>
          </Link>
          <Menu label={`Actions for chat ${c.title}`} items={actions.chatMenu(c)} className="icon-btn row-more" />
        </li>
      ))}
    </ul>
  );
}

function ProjectSkeleton() {
  return (
    <div className="page" aria-busy="true" aria-label="Loading project">
      <div className="page-head">
        <div className="skel skel-eyebrow" />
        <div className="skel skel-title" />
        <div className="skel skel-meta" />
      </div>
      <div className="project-layout">
        <div className="card card-skeleton" />
        <div className="card card-skeleton" />
      </div>
    </div>
  );
}
