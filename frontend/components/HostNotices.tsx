"use client";

/**
 * Warnings about this machine from GET /health (`warnings`, docs/DESIGN.md §8): today, Ollama's prompt cache running
 * uncapped. The app never changes the user's Ollama or system settings; it shows the one command that fixes it. A
 * notice can be dismissed (remembered in this browser, per warning); it goes by itself once the backend stops
 * reporting it.
 */

import { useCallback, useState } from "react";

import { useHealth } from "@/lib/health";

import { Icon } from "./Icon";

const KEY = "host-notice-dismissed:";

function dismissedBefore(code: string): boolean {
  try {
    return window.localStorage.getItem(KEY + code) === "1";
  } catch {
    return false;
  }
}

export function HostNotices() {
  const { report } = useHealth();
  const [dismissed, setDismissed] = useState<ReadonlySet<string>>(() => new Set());
  const [copied, setCopied] = useState<string | null>(null);

  const dismiss = useCallback((code: string) => {
    setDismissed((prev) => new Set(prev).add(code));
    try {
      window.localStorage.setItem(KEY + code, "1");
    } catch {
      // private mode: dismissed for this page only
    }
  }, []);

  const copy = useCallback(async (code: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(code);
    } catch {
      setCopied(null);
    }
  }, []);

  const shown = (report?.warnings ?? []).filter((w) => !dismissed.has(w.code) && !dismissedBefore(w.code));
  if (shown.length === 0) return null;
  return (
    <div className="host-notices">
      {shown.map((w) => (
        <div key={w.code} className="host-notice" role="status">
          <Icon name="alert" size={15} />
          <div className="host-notice-body">
            <span>{w.message} Run once in a terminal (lasts until reboot):</span>
            <span className="host-notice-fix">
              <code>{w.fix}</code>
              <button type="button" className="btn btn-sm" onClick={() => void copy(w.code, w.fix)}>
                {copied === w.code ? "Copied" : "Copy"}
              </button>
            </span>
          </div>
          <button type="button" className="icon-btn" aria-label="Dismiss this notice" onClick={() => dismiss(w.code)}>
            <Icon name="close" size={15} />
          </button>
        </div>
      ))}
    </div>
  );
}
