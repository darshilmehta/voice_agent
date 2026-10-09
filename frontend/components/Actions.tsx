"use client";

/**
 * Everything a user can do to a project, chat or document, in one place, so the sidebar, page headers and lists
 * offer the same actions with the same dialogs and feedback. Dialogs: new project, rename, delete (confirm). Pin and
 * archive act at once and report through a toast (archive offers Undo).
 */

import { useParams, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useMemo, useState, type FormEvent, type ReactNode } from "react";

import { errorMessage, type Chat, type Project, type ProjectDocument } from "@/lib/api";
import { plural } from "@/lib/format";
import { requestAutoStart } from "@/lib/voice/autostart";
import { useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { Dialog } from "./Dialog";
import type { MenuItem } from "./Menu";
import { useToast } from "./Toast";
import { useUploads } from "./Uploads";

type DialogState =
  | { kind: "newProject" }
  | { kind: "renameProject"; project: Project }
  | { kind: "deleteProject"; project: Project }
  | { kind: "renameChat"; chat: Chat }
  | { kind: "deleteChat"; chat: Chat }
  | { kind: "deleteDocument"; document: ProjectDocument };

export interface EntityActions {
  newProject: () => void;
  renameProject: (p: Project) => void;
  deleteProject: (p: Project) => void;
  setProjectPinned: (p: Project, pinned: boolean) => Promise<void>;
  setProjectArchived: (p: Project, archived: boolean) => Promise<void>;
  newChat: (projectId: string) => Promise<void>;
  renameChat: (c: Chat) => void;
  deleteChat: (c: Chat) => void;
  setChatPinned: (c: Chat, pinned: boolean) => Promise<void>;
  setChatArchived: (c: Chat, archived: boolean) => Promise<void>;
  deleteDocument: (d: ProjectDocument) => void;
  /** The "⋯" menu items for a project / chat. */
  projectMenu: (p: Project) => MenuItem[];
  chatMenu: (c: Chat) => MenuItem[];
  creatingChatIn: string | null;
}

const ActionsContext = createContext<EntityActions | null>(null);

export function ActionsProvider({ children }: { children: ReactNode }) {
  const ws = useWorkspaceActions();
  const state = useWorkspace();
  const toast = useToast();
  const { pick: pickFiles } = useUploads();
  const router = useRouter();
  const params = useParams<{ projectId?: string; chatId?: string }>();
  const [dialog, setDialog] = useState<DialogState | null>(null);
  const [creatingChatIn, setCreatingChatIn] = useState<string | null>(null);

  const activeChatId = params.chatId ?? null;
  const activeProjectId = params.projectId ?? (activeChatId ? state.chats[activeChatId]?.project_id : undefined) ?? null;

  const closeDialog = useCallback(() => setDialog(null), []);

  const setProjectPinned = useCallback(
    async (p: Project, pinned: boolean) => {
      try {
        await ws.updateProject(p.id, { pinned });
        toast({ message: pinned ? `Pinned “${p.name}”` : `Unpinned “${p.name}”` });
      } catch (err) {
        toast({ tone: "error", message: `Couldn't ${pinned ? "pin" : "unpin"} the project: ${errorMessage(err)}` });
      }
    },
    [ws, toast],
  );

  const setProjectArchived = useCallback(
    async (p: Project, archived: boolean) => {
      try {
        await ws.updateProject(p.id, { archived });
        toast({
          message: archived ? `Archived “${p.name}”` : `Restored “${p.name}”`,
          action: archived
            ? { label: "Undo", onClick: () => void ws.updateProject(p.id, { archived: false }).catch(() => undefined) }
            : undefined,
        });
      } catch (err) {
        toast({ tone: "error", message: `Couldn't ${archived ? "archive" : "restore"} the project: ${errorMessage(err)}` });
      }
    },
    [ws, toast],
  );

  const setChatPinned = useCallback(
    async (c: Chat, pinned: boolean) => {
      try {
        await ws.updateChat(c.id, { pinned });
        toast({ message: pinned ? `Pinned “${c.title}”` : `Unpinned “${c.title}”` });
      } catch (err) {
        toast({ tone: "error", message: `Couldn't ${pinned ? "pin" : "unpin"} the chat: ${errorMessage(err)}` });
      }
    },
    [ws, toast],
  );

  const setChatArchived = useCallback(
    async (c: Chat, archived: boolean) => {
      try {
        await ws.updateChat(c.id, { archived });
        toast({
          message: archived ? `Archived “${c.title}”` : `Restored “${c.title}”`,
          action: archived
            ? { label: "Undo", onClick: () => void ws.updateChat(c.id, { archived: false }).catch(() => undefined) }
            : undefined,
        });
      } catch (err) {
        toast({ tone: "error", message: `Couldn't ${archived ? "archive" : "restore"} the chat: ${errorMessage(err)}` });
      }
    },
    [ws, toast],
  );

  const newChat = useCallback(
    async (projectId: string) => {
      setCreatingChatIn(projectId);
      try {
        const chat = await ws.createChat(projectId);
        // A new chat opens in voice mode and asks for the microphone (docs §3.9).
        requestAutoStart(chat.id);
        router.push(`/chats/${chat.id}`);
      } catch (err) {
        toast({ tone: "error", message: `Couldn't create a chat: ${errorMessage(err)}` });
      } finally {
        setCreatingChatIn(null);
      }
    },
    [ws, router, toast],
  );

  const value = useMemo<EntityActions>(() => {
    const renameProject = (project: Project) => setDialog({ kind: "renameProject", project });
    const deleteProject = (project: Project) => setDialog({ kind: "deleteProject", project });
    const renameChat = (chat: Chat) => setDialog({ kind: "renameChat", chat });
    const deleteChat = (chat: Chat) => setDialog({ kind: "deleteChat", chat });
    return {
      newProject: () => setDialog({ kind: "newProject" }),
      renameProject,
      deleteProject,
      setProjectPinned,
      setProjectArchived,
      newChat,
      renameChat,
      deleteChat,
      setChatPinned,
      setChatArchived,
      deleteDocument: (document) => setDialog({ kind: "deleteDocument", document }),
      creatingChatIn,
      projectMenu: (p) => [
        { id: "upload", label: "Upload documents…", icon: "upload", onSelect: () => pickFiles(p.id) },
        { id: "rename", label: "Rename…", icon: "pencil", onSelect: () => renameProject(p) },
        {
          id: "pin",
          label: p.pinned ? "Unpin" : "Pin to sidebar",
          icon: "star",
          onSelect: () => void setProjectPinned(p, !p.pinned),
        },
        {
          id: "archive",
          label: p.archived ? "Restore from archive" : "Archive",
          icon: p.archived ? "unarchive" : "archive",
          onSelect: () => void setProjectArchived(p, !p.archived),
        },
        { id: "delete", label: "Delete…", icon: "trash", danger: true, separated: true, onSelect: () => deleteProject(p) },
      ],
      chatMenu: (c) => [
        { id: "rename", label: "Rename…", icon: "pencil", onSelect: () => renameChat(c) },
        { id: "pin", label: c.pinned ? "Unpin" : "Pin to sidebar", icon: "star", onSelect: () => void setChatPinned(c, !c.pinned) },
        {
          id: "archive",
          label: c.archived ? "Restore from archive" : "Archive",
          icon: c.archived ? "unarchive" : "archive",
          onSelect: () => void setChatArchived(c, !c.archived),
        },
        { id: "delete", label: "Delete…", icon: "trash", danger: true, separated: true, onSelect: () => deleteChat(c) },
      ],
    };
  }, [setProjectPinned, setProjectArchived, newChat, setChatPinned, setChatArchived, creatingChatIn, pickFiles]);

  return (
    <ActionsContext value={value}>
      {children}
      {dialog?.kind === "newProject" && (
        <ProjectFormDialog
          onClose={closeDialog}
          onSubmit={async (name, description) => {
            const project = await ws.createProject(name, description);
            closeDialog();
            router.push(`/projects/${project.id}`);
          }}
        />
      )}
      {dialog?.kind === "renameProject" && (
        <ProjectFormDialog
          project={dialog.project}
          onClose={closeDialog}
          onSubmit={async (name, description) => {
            await ws.updateProject(dialog.project.id, { name, description: description || null });
            closeDialog();
          }}
        />
      )}
      {dialog?.kind === "renameChat" && (
        <ChatRenameDialog
          chat={dialog.chat}
          onClose={closeDialog}
          onSubmit={async (title) => {
            await ws.updateChat(dialog.chat.id, { title });
            closeDialog();
          }}
        />
      )}
      {dialog?.kind === "deleteProject" && (
        <ConfirmDeleteDialog
          title="Delete project?"
          confirmLabel="Delete project"
          onClose={closeDialog}
          onConfirm={async () => {
            const { project } = dialog;
            await ws.deleteProject(project.id);
            closeDialog();
            if (activeProjectId === project.id) router.replace("/");
            toast({ message: `Deleted “${project.name}”` });
          }}
        >
          <p>
            <strong>“{dialog.project.name}”</strong> and everything inside it will be deleted from this machine
            {projectContents(dialog.project)}
          </p>
          <p>This can't be undone.</p>
        </ConfirmDeleteDialog>
      )}
      {dialog?.kind === "deleteChat" && (
        <ConfirmDeleteDialog
          title="Delete chat?"
          confirmLabel="Delete chat"
          onClose={closeDialog}
          onConfirm={async () => {
            const { chat } = dialog;
            await ws.deleteChat(chat.id);
            closeDialog();
            if (activeChatId === chat.id) router.replace(`/projects/${chat.project_id}`);
            toast({ message: `Deleted “${chat.title}”` });
          }}
        >
          <p>
            <strong>“{dialog.chat.title}”</strong> will be deleted from this machine
            {dialog.chat.message_count > 0
              ? ` with its ${plural(dialog.chat.message_count, "message")} and any summaries.`
              : ". It has no messages yet."}
          </p>
          <p>This can't be undone.</p>
        </ConfirmDeleteDialog>
      )}
      {dialog?.kind === "deleteDocument" && (
        <ConfirmDeleteDialog
          title="Delete document?"
          confirmLabel="Delete document"
          onClose={closeDialog}
          onConfirm={async () => {
            const { document } = dialog;
            await ws.deleteDocument(document);
            closeDialog();
            toast({ message: `Deleted “${document.filename}”` });
          }}
        >
          <p>
            <strong>“{dialog.document.filename}”</strong> will be deleted from this machine and removed from search, so
            chats can no longer answer from it. Chats limited to selected documents won't include it any more.
          </p>
          <p>Answers already given keep their text. This can't be undone.</p>
        </ConfirmDeleteDialog>
      )}
    </ActionsContext>
  );
}

export function useEntityActions(): EntityActions {
  const ctx = useContext(ActionsContext);
  if (!ctx) throw new Error("useEntityActions() must be used inside <ActionsProvider>");
  return ctx;
}

// ------------------------------------------------------------------ dialogs

/** ": 3 chats with all their messages and summaries, and 2 documents with their indexed content." */
function projectContents(p: Project): string {
  const parts = [
    p.chat_count > 0 ? `${plural(p.chat_count, "chat")} with all their messages and summaries` : null,
    p.document_count > 0 ? `${plural(p.document_count, "document")} with their indexed content` : null,
  ].filter(Boolean);
  return parts.length ? `: ${parts.join(", and ")}.` : ". It has no chats or documents yet.";
}

function useSubmit(onSubmit: () => Promise<void>) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await onSubmit();
    } catch (err) {
      setError(errorMessage(err));
      setBusy(false);
    }
  };
  return { busy, error, run };
}

const NAME_MAX = 200;
const DESCRIPTION_MAX = 4000;

function ProjectFormDialog({
  project,
  onClose,
  onSubmit,
}: {
  project?: Project;
  onClose: () => void;
  onSubmit: (name: string, description: string) => Promise<void>;
}) {
  const [name, setName] = useState(project?.name ?? "");
  const [description, setDescription] = useState(project?.description ?? "");
  const { busy, error, run } = useSubmit(() => onSubmit(name.trim(), description.trim()));
  const editing = !!project;
  const unchanged = editing && name.trim() === project.name && description.trim() === (project.description ?? "");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || busy) return;
    if (unchanged) return onClose();
    void run();
  };

  return (
    <Dialog
      title={editing ? "Rename project" : "New project"}
      description={editing ? undefined : "A project holds documents and the chats about them."}
      onClose={onClose}
      busy={busy}
    >
      <form className="form" onSubmit={submit} noValidate>
        <label className="field">
          <span className="field-label">Name</span>
          <input
            data-autofocus
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={NAME_MAX}
            required
            placeholder="e.g. Annual report FY24"
            autoComplete="off"
            aria-invalid={!!error || undefined}
          />
        </label>
        <label className="field">
          <span className="field-label">
            Description <span className="field-hint">optional</span>
          </span>
          <textarea
            className="input textarea"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            maxLength={DESCRIPTION_MAX}
            rows={3}
            placeholder="What is this project about?"
          />
        </label>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={!name.trim() || busy}>
            {busy ? (editing ? "Saving…" : "Creating…") : editing ? "Save" : "Create project"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

function ChatRenameDialog({
  chat,
  onClose,
  onSubmit,
}: {
  chat: Chat;
  onClose: () => void;
  onSubmit: (title: string) => Promise<void>;
}) {
  const [title, setTitle] = useState(chat.title);
  const { busy, error, run } = useSubmit(() => onSubmit(title.trim()));

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!title.trim() || busy) return;
    if (title.trim() === chat.title) return onClose();
    void run();
  };

  return (
    <Dialog title="Rename chat" onClose={onClose} busy={busy}>
      <form className="form" onSubmit={submit} noValidate>
        <label className="field">
          <span className="field-label">Title</span>
          <input
            data-autofocus
            className="input"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            maxLength={NAME_MAX}
            required
            autoComplete="off"
            aria-invalid={!!error || undefined}
          />
        </label>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={!title.trim() || busy}>
            {busy ? "Saving…" : "Save"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

function ConfirmDeleteDialog({
  title,
  confirmLabel,
  onClose,
  onConfirm,
  children,
}: {
  title: string;
  confirmLabel: string;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  children: ReactNode;
}) {
  const { busy, error, run } = useSubmit(onConfirm);
  return (
    <Dialog title={title} description={children} onClose={onClose} busy={busy} role="alertdialog">
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      <div className="dialog-actions">
        <button type="button" className="btn" onClick={onClose} disabled={busy} data-autofocus>
          Cancel
        </button>
        <button type="button" className="btn btn-danger" onClick={() => void run()} disabled={busy}>
          {busy ? "Deleting…" : confirmLabel}
        </button>
      </div>
    </Dialog>
  );
}
