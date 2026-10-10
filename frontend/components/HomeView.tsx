"use client";

/**
 * "/": welcome; the primary action, "Start a conversation" (a new chat in the chosen project, opened in voice mode,
 * docs/DESIGN.md §3.9); recent chats to revisit; recent projects, or an empty state inviting the first project.
 */

import Link from "next/link";
import { useEffect, useId, useMemo, useState } from "react";

import type { Chat, Project } from "@/lib/api";
import { useBackend, useDocumentTitle } from "@/lib/backend-context";
import { TAGLINE } from "@/lib/brand";
import { LANGUAGE_NAMES, plural, relativeTime } from "@/lib/format";
import { chatActivity, keys, slotOf, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useEntityActions } from "./Actions";
import { Icon } from "./Icon";
import { BackendDown, EmptyState } from "./States";

const RECENT = 6;
const RECENT_CHATS = 8;
const LAST_PROJECT_KEY = "voice-agent.last-project";

/** The project a conversation was last started in (a per-browser convenience; storage may be unavailable). */
function rememberedProject(): string | null {
  try {
    return window.localStorage.getItem(LAST_PROJECT_KEY);
  } catch {
    return null;
  }
}

export function HomeView() {
  const { config, configState, appTitle } = useBackend();
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const actions = useEntityActions();
  useDocumentTitle(null, true);

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
        <p className="tagline">{TAGLINE}</p>
        <p className="lede">
          Talk with your documents out loud. Answers are short, spoken, and cite the pages they come from; typing is
          there when you can't speak.
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

      {projects.length > 0 && (
        <>
          <StartConversation projects={recent} all={projects} />
          <RecentChats projects={recent} />
        </>
      )}

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

/** The primary action: pick a project (the one used last by default), start talking. */
function StartConversation({ projects, all }: { projects: Project[]; all: Project[] }) {
  const actions = useEntityActions();
  const selectId = useId();
  const [chosen, setChosen] = useState<string | null>(null);
  const [remembered, setRemembered] = useState<string | null>(null);
  useEffect(() => setRemembered(rememberedProject()), []);

  // Your choice, else the project you last talked in, else the most recently active one.
  const byId = useMemo(() => new Map(all.map((p) => [p.id, p])), [all]);
  const project =
    (chosen ? byId.get(chosen) : undefined) ?? (remembered ? byId.get(remembered) : undefined) ?? projects[0] ?? all[0];
  const creating = actions.creatingChatIn === project.id;

  const start = () => {
    try {
      window.localStorage.setItem(LAST_PROJECT_KEY, project.id);
    } catch {
      // not remembered; nothing else depends on it
    }
    void actions.newChat(project.id);
  };

  return (
    <section aria-labelledby="start-title" className="home-section">
      <div className="start-card">
        <div className="start-copy">
          <h2 id="start-title">Start a conversation</h2>
          <p>
            Opens a new chat in voice mode and asks for your microphone. Ask about the documents in{" "}
            {all.length > 1 ? "the project below" : <strong>{project.name}</strong>}, out loud.
          </p>
          {all.length > 1 && (
            <div className="start-project">
              <label htmlFor={selectId} className="field-label">
                Project
              </label>
              <select
                id={selectId}
                className="input start-select"
                value={project.id}
                onChange={(e) => setChosen(e.target.value)}
              >
                {all.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
          )}
          {project.document_count === 0 && (
            <p className="start-note">
              <Icon name="info" size={13} />
              <span>
                This project has no documents yet, so answers will say they aren't in your documents.{" "}
                <Link href={`/projects/${project.id}#documents`}>Upload some</Link>
              </span>
            </p>
          )}
        </div>
        <button type="button" className="btn btn-primary start-btn" onClick={start} disabled={creating}>
          <Icon name="mic" size={20} />
          {creating ? "Starting…" : "Start a conversation"}
        </button>
      </div>
    </section>
  );
}

/** Conversations to revisit, newest first, across the most recently active projects. */
function RecentChats({ projects }: { projects: Project[] }) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const ids = projects.map((p) => p.id).join(",");
  useEffect(() => {
    for (const id of ids.split(",").filter(Boolean)) void ws.loadChats(id);
  }, [ids, ws]);

  const chats = projects
    .flatMap((p) => (state.chatIds[p.id] ?? []).map((id) => state.chats[id]))
    .filter((c): c is Chat => !!c && !c.archived && c.message_count > 0)
    .sort((a, b) => Date.parse(chatActivity(b)) - Date.parse(chatActivity(a)))
    .slice(0, RECENT_CHATS);
  if (chats.length === 0) return null;

  return (
    <section aria-labelledby="recent-chats-title" className="home-section">
      <div className="section-head">
        <h2 id="recent-chats-title">Recent chats</h2>
      </div>
      <ul className="rows">
        {chats.map((c) => (
          <li key={c.id} className="row">
            <Link href={`/chats/${c.id}`} className="row-link">
              <Icon name="chat" className="row-icon" />
              <span className="row-main">
                <span className="row-title">{c.title}</span>
                <span className="row-meta">
                  {state.projects[c.project_id]?.name ?? "Project"} · {plural(c.message_count, "message")} ·{" "}
                  {relativeTime(chatActivity(c))}
                </span>
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
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
