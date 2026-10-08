"use client";

/** Shared empty, error and not-found states. */

import Link from "next/link";
import { useCallback, useEffect, useRef, type ReactNode } from "react";

import { useBackend } from "@/lib/backend-context";
import { useHealth } from "@/lib/health";
import { useWorkspaceActions, type SlotMeta } from "@/lib/workspace";

import { Icon, type IconName } from "./Icon";

/** Fetch everything app-wide again: public config, health, projects and pins. */
export function useReconnect() {
  const { reloadConfig } = useBackend();
  const { check } = useHealth();
  const ws = useWorkspaceActions();
  return useCallback(() => {
    reloadConfig();
    void check();
    void ws.loadProjects(true);
    void ws.loadPins(true);
  }, [reloadConfig, check, ws]);
}

/** When the health poll sees the backend come back after being unreachable, reload app-wide data. */
export function AutoReconnect() {
  const { report, error } = useHealth();
  const reconnect = useReconnect();
  const wasDown = useRef(false);
  useEffect(() => {
    if (error && !report) wasDown.current = true;
    else if (report && wasDown.current) {
      wasDown.current = false;
      reconnect();
    }
  }, [report, error, reconnect]);
  return null;
}

/** The backend can't be reached: say where we looked and how to start it (same advice as phase 0). */
export function BackendDown({ message, onRetry }: { message?: string; onRetry?: () => void }) {
  const { backendUrl } = useBackend();
  const reconnect = useReconnect();
  return (
    <div className="notice" role="alert">
      <div className="notice-body">
        <strong>{message ?? `Can't reach the backend at ${backendUrl}`}</strong>
        <p>
          Start it with <code>cd backend &amp;&amp; uv run python -m app</code>, then retry.
        </p>
      </div>
      <button
        type="button"
        className="btn btn-sm"
        onClick={() => {
          reconnect();
          onRetry?.();
        }}
      >
        <Icon name="refresh" />
        Retry
      </button>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  action,
  compact = false,
}: {
  icon: IconName;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  compact?: boolean;
}) {
  return (
    <div className={compact ? "empty-state compact" : "empty-state"}>
      <span className="empty-icon" aria-hidden>
        <Icon name={icon} size={compact ? 20 : 24} />
      </span>
      <strong className="empty-title">{title}</strong>
      {children && <div className="empty-text">{children}</div>}
      {action}
    </div>
  );
}

/** What to show when loading an entity failed: backend down, gone, or another error. */
export function LoadFailed({ slot, kind, onRetry }: { slot: SlotMeta; kind: "project" | "chat"; onRetry: () => void }) {
  if (slot.unreachable) return <BackendDown onRetry={onRetry} />;
  if (slot.notFound) {
    return (
      <div className="page-state">
        <EmptyState
          icon={kind === "project" ? "folder" : "chat"}
          title={slot.deleted ? `This ${kind} was deleted` : `${kind === "project" ? "Project" : "Chat"} not found`}
          action={
            <Link href="/" className="btn">
              Back to home
            </Link>
          }
        >
          {slot.deleted ? null : `It may have been deleted, or the link is wrong.`}
        </EmptyState>
      </div>
    );
  }
  return (
    <div className="notice" role="alert">
      <div className="notice-body">
        <strong>Couldn't load this {kind}</strong>
        <p>{slot.error}</p>
      </div>
      <button type="button" className="btn btn-sm" onClick={onRetry}>
        <Icon name="refresh" />
        Retry
      </button>
    </div>
  );
}
