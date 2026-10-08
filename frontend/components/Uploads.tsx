"use client";

/**
 * Document uploads, app-wide, so they keep going when you navigate away and the sidebar can start one.
 *
 * Each file is checked against GET /api/config/public limits first (extension, size, not empty), then sent with
 * per-file progress, two at a time. A new document (HTTP 202) goes straight into the workspace store, where the
 * documents list shows its ingestion status from then on; an identical file (HTTP 200) is reported as already
 * uploaded; anything refused stays listed with the reason until dismissed (failed uploads can be retried).
 */

import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";

import { BackendError, errorMessage, isAbort, type ProjectDocument, type PublicConfig } from "@/lib/api";
import { useBackend } from "@/lib/backend-context";
import { formatBytes, plural } from "@/lib/format";
import { projectName, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useToast } from "./Toast";

export type UploadPhase = "rejected" | "waiting" | "uploading" | "duplicate" | "failed";

export interface UploadItem {
  key: string;
  projectId: string;
  name: string;
  size: number;
  phase: UploadPhase;
  /** 0…1 of the request body sent. */
  progress: number;
  /** Why it was rejected or failed. */
  message: string | null;
  /** For duplicates: the document already in the project. */
  existing: ProjectDocument | null;
}

interface UploadsValue {
  itemsFor: (projectId: string) => UploadItem[];
  /** Check and queue files for a project. */
  add: (projectId: string, files: Iterable<File>) => void;
  /** Open the file picker for a project (must run inside a click handler). */
  pick: (projectId: string) => void;
  cancel: (key: string) => void;
  retry: (key: string) => void;
  dismiss: (key: string) => void;
  /** "PDF, DOCX, PPTX, TXT or MD", and the input's accept attribute. */
  formats: string;
  accept: string;
  maxMb: number | null;
}

const UploadsContext = createContext<UploadsValue | null>(null);

const PARALLEL = 2;
const FALLBACK_EXTENSIONS = [".pdf", ".docx", ".pptx", ".txt", ".md"];

const extensionOf = (name: string) => {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot).toLowerCase() : "";
};

/** [".pdf", ".docx", ".md"] → "PDF, DOCX or MD" */
export function humanFormats(extensions: string[]): string {
  const names = extensions.map((e) => e.replace(/^\./, "").toUpperCase());
  return names.length > 1 ? `${names.slice(0, -1).join(", ")} or ${names[names.length - 1]}` : (names[0] ?? "");
}

/** Why a file can't be uploaded, or null. Same rules as the backend, so most refusals never leave the browser. */
function checkFile(file: File, config: PublicConfig | null): string | null {
  const extensions = (config?.limits.allowed_extensions ?? FALLBACK_EXTENSIONS).map((e) => e.toLowerCase());
  const ext = extensionOf(file.name);
  if (!ext) return `It has no file extension. Supported: ${humanFormats(extensions)}.`;
  if (!extensions.includes(ext)) return `${ext} files aren't supported. Use ${humanFormats(extensions)}.`;
  if (file.size === 0) return "The file is empty.";
  const maxMb = config?.limits.max_upload_mb;
  if (maxMb && file.size > maxMb * 1024 * 1024) {
    return `It's ${formatBytes(file.size)}; the limit is ${maxMb} MB per file.`;
  }
  return null;
}

export function UploadsProvider({ children }: { children: ReactNode }) {
  const { api, config } = useBackend();
  const ws = useWorkspaceActions();
  const state = useWorkspace();
  const toast = useToast();
  const router = useRouter();
  const pathname = usePathname();
  const [items, setItems] = useState<UploadItem[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const pickFor = useRef<string | null>(null);
  /** Files waiting, uploading or failed (kept for Retry). */
  const files = useRef(new Map<string, { file: File; projectId: string }>());
  const controllers = useRef(new Map<string, AbortController>());
  const queue = useRef<string[]>([]);
  const nextKey = useRef(1);
  const stateRef = useRef(state);
  const pathRef = useRef(pathname);
  useEffect(() => {
    stateRef.current = state;
    pathRef.current = pathname;
  });

  const patch = useCallback((key: string, change: Partial<UploadItem>) => {
    setItems((list) => list.map((i) => (i.key === key ? { ...i, ...change } : i)));
  }, []);

  const drop = useCallback((key: string) => {
    files.current.delete(key);
    setItems((list) => list.filter((i) => i.key !== key));
  }, []);

  const pump = useCallback(() => {
    while (controllers.current.size < PARALLEL && queue.current.length > 0) {
      const key = queue.current.shift() as string;
      const entry = files.current.get(key);
      if (!entry) continue;
      const { file, projectId } = entry;
      const ctrl = new AbortController();
      controllers.current.set(key, ctrl);
      patch(key, { phase: "uploading", progress: 0, message: null });
      let last = 0;
      api
        .uploadDocument(projectId, file, {
          signal: ctrl.signal,
          onProgress: (fraction) => {
            // Progress events come fast; a percent at a time is plenty.
            if (fraction - last >= 0.01 || fraction === 1) {
              last = fraction;
              patch(key, { progress: fraction });
            }
          },
        })
        .then(({ document, duplicate }) => {
          if (duplicate) {
            files.current.delete(key);
            patch(key, { phase: "duplicate", progress: 1, existing: document });
          } else {
            drop(key);
            ws.documentAdded(document);
          }
        })
        .catch((err: unknown) => {
          if (isAbort(err)) drop(key);
          // The backend refused this file (type, size, content): trying again won't help. Otherwise it may.
          else if (err instanceof BackendError && err.status !== undefined && err.status < 500) {
            files.current.delete(key);
            patch(key, { phase: "rejected", message: errorMessage(err) });
          } else patch(key, { phase: "failed", message: errorMessage(err) });
        })
        .finally(() => {
          controllers.current.delete(key);
          pump();
        });
    }
  }, [api, patch, drop, ws]);

  const add = useCallback(
    (projectId: string, list: Iterable<File>) => {
      const fresh: UploadItem[] = [];
      for (const file of list) {
        const key = `upload-${nextKey.current++}`;
        const problem = checkFile(file, config);
        fresh.push({
          key,
          projectId,
          name: file.name,
          size: file.size,
          phase: problem ? "rejected" : "waiting",
          progress: 0,
          message: problem,
          existing: null,
        });
        if (!problem) {
          files.current.set(key, { file, projectId });
          queue.current.push(key);
        }
      }
      if (fresh.length === 0) return;
      setItems((cur) => [...cur, ...fresh]);
      pump();

      // Started from elsewhere (the sidebar): say what happened and offer the way to the project.
      if (pathRef.current !== `/projects/${projectId}`) {
        const accepted = fresh.filter((i) => i.phase === "waiting").length;
        const rejected = fresh.length - accepted;
        const name = projectName(stateRef.current, projectId) ?? "the project";
        const view = { label: "View", onClick: () => router.push(`/projects/${projectId}#documents`) };
        if (accepted === 0) {
          const first = fresh[0];
          toast({
            tone: "error",
            message: rejected === 1 ? `Can't upload ${first.name}: ${first.message}` : `None of the ${rejected} files can be uploaded`,
            action: view,
          });
        } else {
          toast({
            message: `Uploading ${plural(accepted, "file")} to “${name}”${rejected ? ` · ${rejected} skipped` : ""}`,
            action: view,
          });
        }
      }
    },
    [config, pump, router, toast],
  );

  const pick = useCallback((projectId: string) => {
    pickFor.current = projectId;
    inputRef.current?.click();
  }, []);

  const onPicked = (e: ChangeEvent<HTMLInputElement>) => {
    const projectId = pickFor.current;
    const chosen = e.target.files ? Array.from(e.target.files) : [];
    e.target.value = ""; // choosing the same file again still fires change
    if (projectId && chosen.length) add(projectId, chosen);
  };

  const cancel = useCallback((key: string) => {
    const ctrl = controllers.current.get(key);
    if (ctrl) ctrl.abort();
    else {
      queue.current = queue.current.filter((k) => k !== key);
      drop(key);
    }
  }, [drop]);

  const retry = useCallback(
    (key: string) => {
      if (!files.current.has(key)) return;
      patch(key, { phase: "waiting", progress: 0, message: null });
      queue.current.push(key);
      pump();
    },
    [patch, pump],
  );

  const dismiss = useCallback(
    (key: string) => {
      if (controllers.current.has(key)) return;
      queue.current = queue.current.filter((k) => k !== key);
      drop(key);
    },
    [drop],
  );

  // Leaving the page would cut uploads off: ask first.
  const busy = items.some((i) => i.phase === "uploading" || i.phase === "waiting");
  useEffect(() => {
    if (!busy) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [busy]);

  useEffect(() => {
    const ctrls = controllers.current;
    return () => ctrls.forEach((c) => c.abort());
  }, []);

  const extensions = config?.limits.allowed_extensions ?? FALLBACK_EXTENSIONS;
  const value = useMemo<UploadsValue>(
    () => ({
      itemsFor: (projectId) => items.filter((i) => i.projectId === projectId),
      add,
      pick,
      cancel,
      retry,
      dismiss,
      formats: humanFormats(extensions),
      accept: extensions.join(","),
      maxMb: config?.limits.max_upload_mb ?? null,
    }),
    [items, add, pick, cancel, retry, dismiss, extensions, config],
  );

  return (
    <UploadsContext value={value}>
      {children}
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={value.accept}
        hidden
        tabIndex={-1}
        aria-hidden
        onChange={onPicked}
      />
    </UploadsContext>
  );
}

export function useUploads(): UploadsValue {
  const ctx = useContext(UploadsContext);
  if (!ctx) throw new Error("useUploads() must be used inside <UploadsProvider>");
  return ctx;
}
