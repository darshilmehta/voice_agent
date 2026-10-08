"use client";

/** One-at-a-time notifications in a polite live region ("Project archived · Undo", "Couldn't pin: …"). */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { Icon } from "./Icon";

interface ToastInput {
  message: string;
  tone?: "default" | "error";
  action?: { label: string; onClick: () => void };
}

interface Toast extends ToastInput {
  id: number;
}

const ToastContext = createContext<((t: ToastInput) => void) | null>(null);

const DURATION_MS = 5000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<Toast | null>(null);
  const [paused, setPaused] = useState(false);
  const nextId = useRef(1);

  const show = useCallback((t: ToastInput) => setToast({ ...t, id: nextId.current++ }), []);
  const dismiss = useCallback(() => setToast(null), []);

  useEffect(() => {
    if (!toast || paused) return;
    const timer = window.setTimeout(dismiss, toast.tone === "error" ? DURATION_MS * 1.6 : DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [toast, paused, dismiss]);

  const value = useMemo(() => show, [show]);

  return (
    <ToastContext value={value}>
      {children}
      <div
        className="toast-region"
        role="status"
        aria-live="polite"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        onFocus={() => setPaused(true)}
        onBlur={() => setPaused(false)}
      >
        {toast && (
          <div key={toast.id} className={`toast ${toast.tone === "error" ? "toast-error" : ""}`}>
            {toast.tone === "error" && <Icon name="alert" />}
            <span className="toast-msg">{toast.message}</span>
            {toast.action && (
              <button
                type="button"
                className="toast-action"
                onClick={() => {
                  toast.action?.onClick();
                  dismiss();
                }}
              >
                {toast.action.label}
              </button>
            )}
            <button type="button" className="icon-btn icon-btn-sm" onClick={dismiss} aria-label="Dismiss">
              <Icon name="close" size={14} />
            </button>
          </div>
        )}
      </div>
    </ToastContext>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast() must be used inside <ToastProvider>");
  return ctx;
}
