"use client";

/** GET /health, shared by the sidebar indicator and the status panel, rechecked every 30 s and on tab focus. */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { errorMessage, isAbort, type HealthReport } from "./api";
import { useBackend } from "./backend-context";

const POLL_MS = 30_000;

interface HealthContextValue {
  report: HealthReport | null;
  error: string | null;
  checking: boolean;
  checkedAt: Date | null;
  check: () => Promise<void>;
}

const HealthContext = createContext<HealthContextValue | null>(null);

export function HealthProvider({ children }: { children: ReactNode }) {
  const { api } = useBackend();
  const [report, setReport] = useState<HealthReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const check = useCallback(async () => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    setChecking(true);
    try {
      setReport(await api.health(ctrl.signal));
      setError(null);
    } catch (err) {
      if (isAbort(err) || ctrl.signal.aborted) return;
      setReport(null);
      setError(errorMessage(err));
    } finally {
      if (!ctrl.signal.aborted) {
        setChecking(false);
        setCheckedAt(new Date());
      }
    }
  }, [api]);

  useEffect(() => {
    void check();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void check();
    }, POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") void check();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      inflight.current?.abort();
    };
  }, [check]);

  const value = useMemo(
    () => ({ report, error, checking, checkedAt, check }),
    [report, error, checking, checkedAt, check],
  );
  return <HealthContext value={value}>{children}</HealthContext>;
}

export function useHealth(): HealthContextValue {
  const ctx = useContext(HealthContext);
  if (!ctx) throw new Error("useHealth() must be used inside <HealthProvider>");
  return ctx;
}
