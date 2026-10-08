"use client";

/**
 * One client-side store for projects, chats, pins and documents (docs/DESIGN.md §3.9), shared by the sidebar and
 * the pages so an edit anywhere shows up everywhere at once.
 *
 * Update model (used for every mutation): call the backend, apply the entity it returns to the store right away,
 * then refetch the lists whose order or counts may have changed. Nothing is shown as saved before the server
 * confirms it, and the lists always end up in the server's order.
 *
 * Lists are fetched with include_archived=true and filtered in the UI, so the sidebar, the project page and the
 * "show archived" toggle all read the same data.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";

import {
  BackendError,
  errorMessage,
  isAbort,
  type Chat,
  type ChatPatch,
  type PinnedItems,
  type Project,
  type ProjectDocument,
  type ProjectPatch,
} from "./api";
import { useBackend } from "./backend-context";

// ------------------------------------------------------------------ state

export type LoadStatus = "idle" | "loading" | "ready" | "error";

export interface SlotMeta {
  status: LoadStatus;
  error: string | null;
  unreachable: boolean;
  notFound: boolean;
  /** Deleted from this app: views say so instead of "not found" while navigation moves away. */
  deleted: boolean;
}

const IDLE: SlotMeta = { status: "idle", error: null, unreachable: false, notFound: false, deleted: false };

/** Slot keys: "projects", "pins", "chats:<projectId>", "docs:<projectId>", "entity:<id>". */
export type SlotKey = string;
export const keys = {
  projects: "projects",
  pins: "pins",
  chats: (projectId: string) => `chats:${projectId}`,
  docs: (projectId: string) => `docs:${projectId}`,
  entity: (id: string) => `entity:${id}`,
};

export interface WorkspaceState {
  projects: Record<string, Project>;
  chats: Record<string, Chat>;
  /** All projects (archived too), server order: pinned first, then most recent activity. null until loaded. */
  projectIds: string[] | null;
  /** Each project's chats (archived too), most recent activity first. */
  chatIds: Record<string, string[]>;
  /** Chats from GET /api/pins (we may not have loaded their project's chat list). */
  pinnedChatIds: string[];
  /** Project names that came with pinned chats. */
  projectNames: Record<string, string>;
  documents: Record<string, ProjectDocument[]>;
  slots: Record<SlotKey, SlotMeta>;
}

const initialState: WorkspaceState = {
  projects: {},
  chats: {},
  projectIds: null,
  chatIds: {},
  pinnedChatIds: [],
  projectNames: {},
  documents: {},
  slots: {},
};

type Action =
  | { type: "loading"; key: SlotKey }
  | { type: "failed"; key: SlotKey; error: string; unreachable: boolean; notFound: boolean }
  | { type: "projectList"; items: Project[] }
  | { type: "pins"; data: PinnedItems }
  | { type: "chatList"; projectId: string; items: Chat[] }
  | { type: "documents"; projectId: string; items: ProjectDocument[] }
  | { type: "project"; project: Project }
  | { type: "chat"; chat: Chat }
  | { type: "removeProject"; id: string }
  | { type: "removeChat"; id: string };

const ready: SlotMeta = { ...IDLE, status: "ready" };

function byId<T extends { id: string }>(items: T[]): Record<string, T> {
  return Object.fromEntries(items.map((i) => [i.id, i]));
}

function without<T>(record: Record<string, T>, drop: (key: string, value: T) => boolean): Record<string, T> {
  return Object.fromEntries(Object.entries(record).filter(([k, v]) => !drop(k, v)));
}

function reducer(state: WorkspaceState, action: Action): WorkspaceState {
  switch (action.type) {
    case "loading": {
      const prev = state.slots[action.key] ?? IDLE;
      return { ...state, slots: { ...state.slots, [action.key]: { ...prev, status: "loading" } } };
    }
    case "failed": {
      const { key, error, unreachable, notFound } = action;
      const prev = state.slots[key] ?? IDLE;
      return { ...state, slots: { ...state.slots, [key]: { ...prev, status: "error", error, unreachable, notFound } } };
    }
    case "projectList": {
      const entitySlots = Object.fromEntries(action.items.map((p) => [keys.entity(p.id), ready]));
      return {
        ...state,
        projects: { ...state.projects, ...byId(action.items) },
        projectIds: action.items.map((p) => p.id),
        slots: { ...state.slots, ...entitySlots, [keys.projects]: ready },
      };
    }
    case "pins": {
      const { projects, chats } = action.data;
      const plainChats = chats.map(({ project_name: _, ...chat }) => chat as Chat);
      return {
        ...state,
        projects: { ...state.projects, ...byId(projects) },
        chats: { ...state.chats, ...byId(plainChats) },
        pinnedChatIds: chats.map((c) => c.id),
        projectNames: { ...state.projectNames, ...Object.fromEntries(chats.map((c) => [c.project_id, c.project_name])) },
        slots: { ...state.slots, [keys.pins]: ready },
      };
    }
    case "chatList": {
      const { projectId, items } = action;
      return {
        ...state,
        chats: { ...state.chats, ...byId(items) },
        chatIds: { ...state.chatIds, [projectId]: items.map((c) => c.id) },
        slots: { ...state.slots, [keys.chats(projectId)]: ready },
      };
    }
    case "documents":
      return {
        ...state,
        documents: { ...state.documents, [action.projectId]: action.items },
        slots: { ...state.slots, [keys.docs(action.projectId)]: ready },
      };
    case "project": {
      const { project } = action;
      const ids = state.projectIds;
      return {
        ...state,
        projects: { ...state.projects, [project.id]: project },
        projectIds: ids && !ids.includes(project.id) ? [project.id, ...ids] : ids,
        slots: { ...state.slots, [keys.entity(project.id)]: ready },
      };
    }
    case "chat": {
      const { chat } = action;
      const ids = state.chatIds[chat.project_id];
      return {
        ...state,
        chats: { ...state.chats, [chat.id]: chat },
        chatIds: ids && !ids.includes(chat.id) ? { ...state.chatIds, [chat.project_id]: [chat.id, ...ids] } : state.chatIds,
        slots: { ...state.slots, [keys.entity(chat.id)]: ready },
      };
    }
    case "removeProject": {
      const { id } = action;
      const goneChats = new Set(Object.values(state.chats).filter((c) => c.project_id === id).map((c) => c.id));
      const gone: SlotMeta = { ...IDLE, status: "error", notFound: true, deleted: true };
      const slots = { ...state.slots, [keys.entity(id)]: gone };
      for (const chatId of goneChats) slots[keys.entity(chatId)] = gone;
      return {
        ...state,
        projects: without(state.projects, (k) => k === id),
        chats: without(state.chats, (k) => goneChats.has(k)),
        projectIds: state.projectIds?.filter((p) => p !== id) ?? null,
        chatIds: without(state.chatIds, (k) => k === id),
        pinnedChatIds: state.pinnedChatIds.filter((c) => !goneChats.has(c)),
        documents: without(state.documents, (k) => k === id),
        slots,
      };
    }
    case "removeChat": {
      const { id } = action;
      const chat = state.chats[id];
      const list = chat ? state.chatIds[chat.project_id] : undefined;
      return {
        ...state,
        chats: without(state.chats, (k) => k === id),
        chatIds: chat && list ? { ...state.chatIds, [chat.project_id]: list.filter((c) => c !== id) } : state.chatIds,
        pinnedChatIds: state.pinnedChatIds.filter((c) => c !== id),
        slots: { ...state.slots, [keys.entity(id)]: { ...IDLE, status: "error", notFound: true, deleted: true } },
      };
    }
  }
}

// ------------------------------------------------------------------ selectors

export const slotOf = (state: WorkspaceState, key: SlotKey): SlotMeta => state.slots[key] ?? IDLE;

const time = (iso: string | null) => (iso ? Date.parse(iso) : 0);

/** Most recent activity of a chat: its last message, or its creation. */
export const chatActivity = (chat: Chat) => chat.last_message_at ?? chat.created_at;

export function pinnedProjects(state: WorkspaceState): Project[] {
  return Object.values(state.projects)
    .filter((p) => p.pinned && !p.archived)
    .sort((a, b) => time(b.pinned_at) - time(a.pinned_at));
}

/** Pinned, non-archived chats of non-archived projects, most recently pinned first (same rule as GET /api/pins). */
export function pinnedChats(state: WorkspaceState): Chat[] {
  const ids = new Set(state.pinnedChatIds);
  for (const c of Object.values(state.chats)) if (c.pinned) ids.add(c.id);
  return [...ids]
    .map((id) => state.chats[id])
    .filter((c): c is Chat => !!c && c.pinned && !c.archived && !state.projects[c.project_id]?.archived)
    .sort((a, b) => time(b.pinned_at) - time(a.pinned_at));
}

export function projectName(state: WorkspaceState, projectId: string): string | null {
  return state.projects[projectId]?.name ?? state.projectNames[projectId] ?? null;
}

export function chatsOf(state: WorkspaceState, projectId: string): Chat[] | null {
  const ids = state.chatIds[projectId];
  return ids ? ids.map((id) => state.chats[id]).filter((c): c is Chat => !!c) : null;
}

// ------------------------------------------------------------------ actions

export interface WorkspaceActions {
  loadProjects: (force?: boolean) => Promise<void>;
  loadPins: (force?: boolean) => Promise<void>;
  loadChats: (projectId: string, force?: boolean) => Promise<void>;
  loadDocuments: (projectId: string, force?: boolean) => Promise<void>;
  loadProject: (projectId: string, force?: boolean) => Promise<void>;
  loadChat: (chatId: string, force?: boolean) => Promise<void>;
  /** Mutations throw BackendError; callers show the message. */
  createProject: (name: string, description?: string | null) => Promise<Project>;
  updateProject: (projectId: string, patch: ProjectPatch) => Promise<Project>;
  deleteProject: (projectId: string) => Promise<void>;
  createChat: (projectId: string) => Promise<Chat>;
  updateChat: (chatId: string, patch: ChatPatch) => Promise<Chat>;
  deleteChat: (chatId: string) => Promise<void>;
}

const StateContext = createContext<WorkspaceState | null>(null);
const ActionsContext = createContext<WorkspaceActions | null>(null);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const { api } = useBackend();
  const [state, dispatch] = useReducer(reducer, initialState);
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  });

  // Per-key request bookkeeping: concurrent loads of the same thing share one request, and only the newest
  // request for a key may write its result (a revalidation after a mutation beats an older in-flight load).
  const inflight = useRef(new Map<SlotKey, Promise<void>>());
  const generation = useRef(new Map<SlotKey, number>());

  const run = useCallback(<T,>(key: SlotKey, fetcher: () => Promise<T>, apply: (data: T) => void, force = false) => {
    const pending = inflight.current.get(key);
    if (pending && !force) return pending;
    const gen = (generation.current.get(key) ?? 0) + 1;
    generation.current.set(key, gen);
    dispatch({ type: "loading", key });
    const promise: Promise<void> = fetcher()
      .then((data) => {
        if (generation.current.get(key) === gen) apply(data);
      })
      .catch((err: unknown) => {
        if (isAbort(err) || generation.current.get(key) !== gen) return;
        dispatch({
          type: "failed",
          key,
          error: errorMessage(err),
          unreachable: err instanceof BackendError && err.unreachable,
          notFound: err instanceof BackendError && err.notFound,
        });
      })
      .finally(() => {
        if (inflight.current.get(key) === promise) inflight.current.delete(key);
      });
    inflight.current.set(key, promise);
    return promise;
  }, []);

  const actions = useMemo<WorkspaceActions>(() => {
    const loadProjects = (force = false) =>
      run(keys.projects, () => api.listProjects({ includeArchived: true }), (items) => dispatch({ type: "projectList", items }), force);
    const loadPins = (force = false) => run(keys.pins, () => api.pins(), (data) => dispatch({ type: "pins", data }), force);
    const loadChats = (projectId: string, force = false) =>
      run(
        keys.chats(projectId),
        () => api.listChats(projectId, { includeArchived: true }),
        (items) => dispatch({ type: "chatList", projectId, items }),
        force,
      );
    const loadDocuments = (projectId: string, force = false) =>
      run(keys.docs(projectId), () => api.listDocuments(projectId), (items) => dispatch({ type: "documents", projectId, items }), force);
    const loadProject = (projectId: string, force = false) =>
      run(keys.entity(projectId), () => api.getProject(projectId), (project) => dispatch({ type: "project", project }), force);
    const loadChat = (chatId: string, force = false) =>
      run(keys.entity(chatId), () => api.getChat(chatId), (chat) => dispatch({ type: "chat", chat }), force);

    return {
      loadProjects,
      loadPins,
      loadChats,
      loadDocuments,
      loadProject,
      loadChat,

      async createProject(name, description) {
        const project = await api.createProject({ name, description: description || null });
        dispatch({ type: "project", project });
        void loadProjects(true);
        return project;
      },
      async updateProject(projectId, patch) {
        const project = await api.updateProject(projectId, patch);
        dispatch({ type: "project", project });
        void loadProjects(true);
        if ("pinned" in patch || "archived" in patch) void loadPins(true);
        return project;
      },
      async deleteProject(projectId) {
        await api.deleteProject(projectId);
        dispatch({ type: "removeProject", id: projectId });
        void loadProjects(true);
        void loadPins(true);
      },
      async createChat(projectId) {
        const chat = await api.createChat(projectId);
        dispatch({ type: "chat", chat });
        void loadChats(projectId, true);
        void loadProject(projectId, true); // chat count, activity
        void loadProjects(true);
        return chat;
      },
      async updateChat(chatId, patch) {
        const chat = await api.updateChat(chatId, patch);
        dispatch({ type: "chat", chat });
        void loadChats(chat.project_id, true);
        if ("pinned" in patch || "archived" in patch) void loadPins(true);
        return chat;
      },
      async deleteChat(chatId) {
        const projectId = stateRef.current.chats[chatId]?.project_id;
        await api.deleteChat(chatId);
        dispatch({ type: "removeChat", id: chatId });
        if (projectId) {
          void loadChats(projectId, true);
          void loadProject(projectId, true);
        }
        void loadProjects(true);
        void loadPins(true);
      },
    };
  }, [api, run]);

  // The sidebar needs every project and everything pinned from the start.
  useEffect(() => {
    void actions.loadProjects();
    void actions.loadPins();
  }, [actions]);

  return (
    <ActionsContext value={actions}>
      <StateContext value={state}>{children}</StateContext>
    </ActionsContext>
  );
}

export function useWorkspace(): WorkspaceState {
  const ctx = useContext(StateContext);
  if (!ctx) throw new Error("useWorkspace() must be used inside <WorkspaceProvider>");
  return ctx;
}

export function useWorkspaceActions(): WorkspaceActions {
  const ctx = useContext(ActionsContext);
  if (!ctx) throw new Error("useWorkspaceActions() must be used inside <WorkspaceProvider>");
  return ctx;
}
