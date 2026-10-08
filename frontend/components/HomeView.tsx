"use client";

/** "/": welcome, then recent projects, or an empty state inviting the first project. */

import Link from "next/link";

import type { Project } from "@/lib/api";
import { useBackend, useDocumentTitle } from "@/lib/backend-context";
import { LANGUAGE_NAMES, plural, relativeTime } from "@/lib/format";
import { keys, slotOf, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useEntityActions } from "./Actions";
import { Icon } from "./Icon";
import { BackendDown, EmptyState } from "./States";

const RECENT = 6;

export function HomeView() {
  const { config, configState, appTitle } = useBackend();
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const actions = useEntityActions();
  useDocumentTitle(null);

  const slot = slotOf(state, keys.projects);
  const projects = (state.projectIds ?? [])
    .map((id) => state.projects[id])
    .filter((p): p is Project => !!p && !p.archived);
  const recent = [...projects].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at)).slice(0, RECENT);
  const archivedOnly = state.projectIds !== null && projects.length === 0 && state.projectIds.length > 0;

  return (
    <div className="page">
      {state.projectIds === null &&
        slot.status === "error" &&
        (slot.unreachable ? (
          <BackendDown />
        ) : (
          <div className="notice" role="alert">
            <div className="notice-body">
              <strong>Couldn't load your projects</strong>
              <p>{slot.error}</p>
            </div>
            <button type="button" className="btn btn-sm" onClick={() => void ws.loadProjects(true)}>
              <Icon name="refresh" />
              Retry
            </button>
          </div>
        ))}

      <header className="hero">
        <h1>{appTitle}</h1>
        <p className="lede">
          Talk with your documents by voice or text. Answers cite the pages they come from.
        </p>
        <div className="chips" aria-label="Setup">
          {config ? (
            <>
              <span className="chip">
                <strong>{config.profile === "local" ? "Local" : "Cloud"}</strong>
                {config.profile === "local" && "· runs on this machine"}
              </span>
              {config.client.languages.map((lang) => (
                <span key={lang} className="chip" lang={lang}>
                  {LANGUAGE_NAMES[lang] ?? lang}
                </span>
              ))}
            </>
          ) : (
            configState.kind === "loading" && <span className="chip chip-skeleton" aria-hidden />
          )}
        </div>
      </header>

      <section aria-labelledby="recent-title" className="home-section">
        <div className="section-head">
          <h2 id="recent-title">Recent projects</h2>
          {projects.length > 0 && (
            <button type="button" className="btn btn-sm" onClick={actions.newProject}>
              <Icon name="plus" />
              New project
            </button>
          )}
        </div>

        {state.projectIds === null ? (
          slot.status === "error" ? null : (
            <div className="project-grid" aria-hidden>
              {Array.from({ length: 3 }, (_, i) => (
                <div key={i} className="pcard pcard-skeleton" />
              ))}
            </div>
          )
        ) : projects.length === 0 ? (
          <div className="card">
            <EmptyState
              icon="folder"
              title={archivedOnly ? "All your projects are archived" : "Create your first project"}
              action={
                <button type="button" className="btn btn-primary" onClick={actions.newProject}>
                  <Icon name="plus" />
                  New project
                </button>
              }
            >
              <p>
                A project groups the documents about one subject — an annual report, a set of contracts — with the
                chats you have about them.
                {archivedOnly && " Show archived projects from the sidebar to restore one."}
              </p>
            </EmptyState>
          </div>
        ) : (
          <ul className="project-grid">
            {recent.map((p) => (
              <li key={p.id}>
                <ProjectCard project={p} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function ProjectCard({ project }: { project: Project }) {
  return (
    <Link href={`/projects/${project.id}`} className="pcard">
      <span className="pcard-top">
        <span className="pcard-icon" aria-hidden>
          <Icon name="folder" />
        </span>
        {project.pinned && (
          <span className="pcard-pin" title="Pinned">
            <Icon name="star" filled size={14} />
            <span className="visually-hidden">Pinned</span>
          </span>
        )}
      </span>
      <span className="pcard-name">{project.name}</span>
      <span className="pcard-desc">{project.description || "No description"}</span>
      <span className="pcard-meta">
        {plural(project.chat_count, "chat")} · {plural(project.document_count, "document")} · updated{" "}
        {relativeTime(project.updated_at)}
      </span>
    </Link>
  );
}
