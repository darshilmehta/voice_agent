"use client";

/** /projects/[projectId]: header with rename/pin/archive/delete, the project's chats, its documents (read-only). */

import Link from "next/link";
import { useEffect, useState } from "react";

import type { Chat, Project, ProjectDocument } from "@/lib/api";
import { useBackend, useDocumentTitle } from "@/lib/backend-context";
import { formatBytes, fullDateTime, plural, relativeTime, shortDate } from "@/lib/format";
import { chatActivity, chatsOf, keys, slotOf, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useEntityActions } from "./Actions";
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

const STATUS_LABEL: Record<string, { label: string; tone: string }> = {
  READY: { label: "Ready", tone: "ok" },
  PROCESSING: { label: "Processing", tone: "degraded" },
  PENDING: { label: "Queued", tone: "idle" },
  FAILED: { label: "Failed", tone: "down" },
};

function DocumentsCard({ project }: { project: Project }) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const { config } = useBackend();
  const slot = slotOf(state, keys.docs(project.id));
  const docs = state.documents[project.id] ?? null;
  const [showWhy, setShowWhy] = useState(false);
  const formats = config ? config.limits.allowed_extensions.join(" ") : ".pdf .docx .pptx .txt .md";
  const limit = config ? ` up to ${config.limits.max_upload_mb} MB each` : "";

  return (
    <section className="card" aria-labelledby="docs-title">
      <div className="card-head">
        <h2 id="docs-title">Documents</h2>
        <button
          type="button"
          className="btn btn-sm"
          aria-disabled="true"
          aria-describedby="upload-note"
          onClick={() => setShowWhy(true)}
        >
          <Icon name="upload" />
          Upload
        </button>
      </div>

      {docs === null ? (
        slot.status === "error" ? (
          slot.unreachable ? (
            <BackendDown onRetry={() => void ws.loadDocuments(project.id, true)} />
          ) : (
            <p className="form-error">{slot.error}</p>
          )
        ) : (
          <div className="rows" aria-hidden>
            <div className="row-skeleton" />
          </div>
        )
      ) : docs.length === 0 ? (
        <EmptyState icon="doc" title="No documents yet" compact>
          <p id="upload-note" className={showWhy ? "pulse-once" : undefined}>
            Uploading and indexing arrive with document ingestion, which is coming next. You'll be able to add{" "}
            {formats} files{limit}.
          </p>
        </EmptyState>
      ) : (
        <>
          <ul className="rows">
            {docs.map((d) => (
              <DocumentRow key={d.id} doc={d} />
            ))}
          </ul>
          <p id="upload-note" className={showWhy ? "note pulse-once" : "note"}>
            Uploading more documents arrives with document ingestion, coming next.
          </p>
        </>
      )}
    </section>
  );
}

function DocumentRow({ doc }: { doc: ProjectDocument }) {
  const status = STATUS_LABEL[doc.status] ?? { label: doc.status, tone: "idle" };
  const facts = [
    doc.page_count !== null ? plural(doc.page_count, "page") : null,
    doc.chunk_count !== null ? plural(doc.chunk_count, "chunk") : null,
    formatBytes(doc.size_bytes),
    `added ${shortDate(doc.created_at)}`,
  ].filter(Boolean);
  return (
    <li className="row row-static">
      <Icon name="doc" className="row-icon" />
      <span className="row-main">
        <span className="row-title">{doc.filename}</span>
        <span className="row-meta">
          <span className="status-pill">
            <span className={`dot ${status.tone}`} aria-hidden />
            {status.label}
          </span>
          {" · "}
          {facts.join(" · ")}
        </span>
        {doc.error && <span className="row-error">{doc.error}</span>}
      </span>
    </li>
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
