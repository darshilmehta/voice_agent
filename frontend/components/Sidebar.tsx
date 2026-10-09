"use client";

/**
 * The left sidebar (docs/DESIGN.md §3.9): new project, search, PINNED, PROJECTS (each expands to its chats, newest
 * activity first, plus "New chat") and the health indicator. Search filters project names and chat titles on the
 * client; while it is active, every project's chats are loaded so all chats are searchable.
 */

import Link from "next/link";
import { useParams, usePathname } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";

import type { Chat, Project } from "@/lib/api";
import { useBackend } from "@/lib/backend-context";
import { matches, shortAge } from "@/lib/format";
import {
  chatActivity,
  chatsOf,
  keys,
  pinnedChats,
  pinnedProjects,
  projectName,
  slotOf,
  useWorkspace,
  useWorkspaceActions,
} from "@/lib/workspace";

import { useEntityActions } from "./Actions";
import { HealthIndicator } from "./HealthIndicator";
import { Highlight } from "./Highlight";
import { Icon } from "./Icon";
import { Menu, type MenuItem } from "./Menu";
import { useReconnect } from "./States";
import { TitleText } from "./TitleText";

const EXPANDED_KEY = "sidebar.expanded";
const ARCHIVED_KEY = "sidebar.showArchived";

function readStored<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // private mode or storage blocked: the sidebar just won't remember
  }
}

/**
 * State remembered in this browser (best effort). Read after mount, so server and first client render agree;
 * written only when it changes, never while restoring.
 */
function useRemembered<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(fallback);
  const fallbackRef = useRef(fallback);
  useEffect(() => {
    setValue(readStored(key, fallbackRef.current));
  }, [key]);
  const update = useCallback(
    (next: (prev: T) => T) =>
      setValue((prev) => {
        const v = next(prev);
        if (v !== prev) store(key, v);
        return v;
      }),
    [key],
  );
  return [value, update] as const;
}

/** Re-render every minute so "2m" ages stay current. */
function useNow(intervalMs = 60_000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(t);
  }, [intervalMs]);
  return now;
}

interface SidebarProps {
  /** Present when the sidebar is a drawer (narrow screens): closes it. */
  onClose?: () => void;
  /** Ask the frame to show the sidebar (⌘K on narrow screens). */
  onRequestOpen: () => void;
  searchRef: RefObject<HTMLInputElement | null>;
}

export function Sidebar({ onClose, onRequestOpen, searchRef }: SidebarProps) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const actions = useEntityActions();
  const { appTitle } = useBackend();
  const reconnect = useReconnect();
  const pathname = usePathname();
  const params = useParams<{ projectId?: string; chatId?: string }>();
  const now = useNow();

  const [query, setQuery] = useState("");
  // Which projects are open and whether archived items show, remembered per browser.
  const [expanded, setExpanded] = useRemembered<string[]>(EXPANDED_KEY, []);
  const [showArchived, setShowArchived] = useRemembered<boolean>(ARCHIVED_KEY, false);

  const activeChatId = params.chatId ?? null;
  const activeChat = activeChatId ? state.chats[activeChatId] : undefined;
  const activeProjectId = params.projectId ?? activeChat?.project_id ?? null;
  const q = query.trim();

  // The project you're in is always open.
  useEffect(() => {
    if (activeProjectId) setExpanded((ids) => (ids.includes(activeProjectId) ? ids : [...ids, activeProjectId]));
  }, [activeProjectId, setExpanded]);

  // ⌘K / Ctrl+K focuses search.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        onRequestOpen();
        requestAnimationFrame(() => searchRef.current?.focus());
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onRequestOpen, searchRef]);

  const projectsSlot = slotOf(state, keys.projects);
  const allProjects = useMemo(
    () => (state.projectIds ?? []).map((id) => state.projects[id]).filter((p): p is Project => !!p),
    [state.projectIds, state.projects],
  );
  const archivedCount = allProjects.filter((p) => p.archived).length;
  const visible = (archived: boolean, id: string) => !archived || showArchived || id === activeProjectId || id === activeChatId;

  // Load chats of open projects, and of every project while searching.
  const wanted = q ? allProjects.map((p) => p.id) : expanded;
  const wantedKey = wanted.join(",");
  useEffect(() => {
    for (const id of wantedKey ? wantedKey.split(",") : []) {
      const slot = slotOf(state, keys.chats(id));
      if (!state.chatIds[id] && slot.status === "idle" && state.projects[id]) void ws.loadChats(id);
    }
  }, [wantedKey, state, ws]);

  const toggle = (id: string) => setExpanded((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]));

  // ---- what to show
  const pinnedP = pinnedProjects(state).filter((p) => matches(p.name, q));
  const pinnedC = pinnedChats(state).filter((c) => matches(c.title, q));

  const tree = allProjects
    .filter((p) => visible(p.archived, p.id))
    .map((p) => {
      const chats = (chatsOf(state, p.id) ?? []).filter((c) => visible(c.archived, c.id));
      const hits = q ? chats.filter((c) => matches(c.title, q)) : chats;
      return { project: p, chats: hits, nameHit: matches(p.name, q) };
    })
    .filter((n) => !q || n.nameHit || n.chats.length > 0);

  const searching = q && allProjects.some((p) => slotOf(state, keys.chats(p.id)).status === "loading");
  const nothingFound = q && tree.length === 0 && pinnedP.length === 0 && pinnedC.length === 0 && !searching;

  return (
    <div className="sb">
      <div className="sb-head">
        <Link href="/" className="sb-brand" aria-current={pathname === "/" ? "page" : undefined}>
          <span className="sb-logo" aria-hidden>
            <Icon name="sparkle" size={14} />
          </span>
          <span className="sb-brand-name">{appTitle}</span>
        </Link>
        {onClose && (
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close sidebar" data-drawer-focus>
            <Icon name="close" />
          </button>
        )}
      </div>

      <div className="sb-tools">
        <button type="button" className="btn sb-new" onClick={actions.newProject}>
          <Icon name="plus" />
          New project
        </button>
        <div className="sb-search">
          <Icon name="search" />
          <input
            ref={searchRef}
            type="search"
            className="sb-search-input"
            placeholder="Search"
            aria-label="Search projects and chats"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape" && query) {
                e.preventDefault();
                e.stopPropagation();
                setQuery("");
              }
            }}
          />
          {!query && (
            <kbd className="sb-kbd" aria-hidden>
              ⌘K
            </kbd>
          )}
        </div>
      </div>

      <nav className="sb-nav" aria-label="Projects and chats">
        {(pinnedP.length > 0 || pinnedC.length > 0) && (
          <Section title="Pinned">
            <ul className="sb-list">
              {pinnedP.map((p) => (
                <SidebarItem
                  key={p.id}
                  href={`/projects/${p.id}`}
                  current={pathname === `/projects/${p.id}`}
                  icon="folder"
                  label={<Highlight text={p.name} query={q} />}
                  menuLabel={`Actions for project ${p.name}`}
                  menu={actions.projectMenu(p)}
                />
              ))}
              {pinnedC.map((c) => (
                <SidebarItem
                  key={c.id}
                  href={`/chats/${c.id}`}
                  current={c.id === activeChatId}
                  icon="chat"
                  label={<TitleText title={c.title}><Highlight text={c.title} query={q} /></TitleText>}
                  sub={projectName(state, c.project_id) ?? undefined}
                  menuLabel={`Actions for chat ${c.title}`}
                  menu={actions.chatMenu(c)}
                />
              ))}
            </ul>
          </Section>
        )}

        <Section title="Projects">
          {state.projectIds === null ? (
            projectsSlot.status === "error" ? (
              <div className="sb-msg sb-msg-error" role="alert">
                <span>{projectsSlot.unreachable ? "Can't reach the backend." : "Couldn't load projects."}</span>
                <button type="button" className="link-btn" onClick={reconnect}>
                  Retry
                </button>
              </div>
            ) : (
              <SkeletonRows count={4} />
            )
          ) : allProjects.length === 0 ? (
            <p className="sb-msg">No projects yet.</p>
          ) : (
            <ul className="sb-list">
              {tree.map(({ project, chats }) => (
                <ProjectNode
                  key={project.id}
                  project={project}
                  chats={chats}
                  chatsLoaded={!!state.chatIds[project.id]}
                  chatsFailed={slotOf(state, keys.chats(project.id)).status === "error"}
                  expanded={q ? chats.length > 0 : expanded.includes(project.id)}
                  onToggle={() => toggle(project.id)}
                  searching={!!q}
                  query={q}
                  current={pathname === `/projects/${project.id}`}
                  activeChatId={activeChatId}
                  now={now}
                />
              ))}
            </ul>
          )}
          {searching && <p className="sb-msg">Searching chats…</p>}
          {nothingFound && <p className="sb-msg">No projects or chats match “{q}”.</p>}
          {!q && archivedCount > 0 && (
            <button
              type="button"
              className="sb-archived-toggle"
              aria-pressed={showArchived}
              onClick={() => setShowArchived((v) => !v)}
            >
              <Icon name="archive" />
              {showArchived ? "Hide archived" : `Show archived (${archivedCount})`}
            </button>
          )}
        </Section>
      </nav>

      <div className="sb-foot">
        <HealthIndicator />
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  const id = useId();
  return (
    <section className="sb-section" aria-labelledby={id}>
      <h2 id={id} className="sb-heading">
        {title}
      </h2>
      {children}
    </section>
  );
}

function SkeletonRows({ count, as: Tag = "div" }: { count: number; as?: "div" | "li" }) {
  return (
    <Tag className="sb-skeletons" aria-hidden>
      {Array.from({ length: count }, (_, i) => (
        <span key={i} className="sb-skel" style={{ width: `${88 - ((i * 17) % 40)}%` }} />
      ))}
    </Tag>
  );
}

function SidebarItem({
  href,
  current,
  icon,
  label,
  sub,
  menuLabel,
  menu,
  meta,
  archived,
}: {
  href: string;
  current: boolean;
  icon?: "folder" | "chat";
  label: ReactNode;
  sub?: string;
  menuLabel: string;
  menu: MenuItem[];
  meta?: string;
  archived?: boolean;
}) {
  return (
    <li className="sb-item" data-current={current || undefined}>
      <Link href={href} className="sb-link" aria-current={current ? "page" : undefined}>
        {icon && <Icon name={icon} className="sb-icon" />}
        <span className="sb-label">{label}</span>
        {sub && <span className="sb-sub">{sub}</span>}
        {archived && <span className="tag">Archived</span>}
        {meta && <span className="sb-meta">{meta}</span>}
      </Link>
      <Menu label={menuLabel} items={menu} className="icon-btn icon-btn-sm sb-more" iconSize={15} />
    </li>
  );
}

function ProjectNode({
  project,
  chats,
  chatsLoaded,
  chatsFailed,
  expanded,
  onToggle,
  searching,
  query,
  current,
  activeChatId,
  now,
}: {
  project: Project;
  chats: Chat[];
  chatsLoaded: boolean;
  chatsFailed: boolean;
  expanded: boolean;
  onToggle: () => void;
  searching: boolean;
  query: string;
  current: boolean;
  activeChatId: string | null;
  now: number;
}) {
  const actions = useEntityActions();
  const ws = useWorkspaceActions();
  const listId = useId();
  const creating = actions.creatingChatIn === project.id;

  return (
    <li className="sb-project" data-expanded={expanded || undefined}>
      <div className="sb-item" data-current={current || undefined}>
        <button
          type="button"
          className="sb-twisty"
          aria-expanded={expanded}
          aria-controls={listId}
          aria-label={`Chats in ${project.name}`}
          onClick={onToggle}
          disabled={searching}
        >
          <Icon name="chevron" size={14} />
        </button>
        <Link href={`/projects/${project.id}`} className="sb-link" aria-current={current ? "page" : undefined}>
          <span className="sb-label">
            <Highlight text={project.name} query={query} />
          </span>
          {project.archived && <span className="tag">Archived</span>}
        </Link>
        <Menu
          label={`Actions for project ${project.name}`}
          items={actions.projectMenu(project)}
          className="icon-btn icon-btn-sm sb-more"
          iconSize={15}
        />
      </div>
      <ul id={listId} className="sb-chats" hidden={!expanded}>
        {expanded && !chatsLoaded && !chatsFailed && <SkeletonRows count={2} as="li" />}
        {expanded && chatsFailed && !chatsLoaded && (
          <li className="sb-msg sb-msg-error">
            <span>Couldn't load chats.</span>
            <button type="button" className="link-btn" onClick={() => void ws.loadChats(project.id, true)}>
              Retry
            </button>
          </li>
        )}
        {chats.map((c) => (
          <SidebarItem
            key={c.id}
            href={`/chats/${c.id}`}
            current={c.id === activeChatId}
            label={<TitleText title={c.title}><Highlight text={c.title} query={query} /></TitleText>}
            meta={shortAge(chatActivity(c), now)}
            archived={c.archived}
            menuLabel={`Actions for chat ${c.title}`}
            menu={actions.chatMenu(c)}
          />
        ))}
        {!searching && chatsLoaded && (
          <li className="sb-item">
            <button
              type="button"
              className="sb-link sb-newchat"
              onClick={() => void actions.newChat(project.id)}
              disabled={creating}
            >
              <Icon name="plus" className="sb-icon" />
              <span className="sb-label">{creating ? "Creating…" : "New chat"}</span>
            </button>
          </li>
        )}
      </ul>
    </li>
  );
}
