"use client";

import { useEffect, useState } from "react";

import { BackendError, fetchPublicConfig, type Language, type PublicConfig } from "@/lib/api";

import { StatusPanel } from "./StatusPanel";

const LANGUAGE_NAMES: Record<Language, string> = { en: "English", hi: "हिन्दी" };

type ConfigState = { kind: "loading" } | { kind: "ready"; config: PublicConfig } | { kind: "error"; message: string };

export function AppShell({ backendUrl }: { backendUrl: string }) {
  const [state, setState] = useState<ConfigState>({ kind: "loading" });

  useEffect(() => {
    const ctrl = new AbortController();
    fetchPublicConfig(backendUrl, ctrl.signal)
      .then((config) => {
        setState({ kind: "ready", config });
        document.title = config.client.app_title;
      })
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        setState({ kind: "error", message: err instanceof BackendError ? err.message : String(err) });
      });
    return () => ctrl.abort();
  }, [backendUrl]);

  const config = state.kind === "ready" ? state.config : null;

  return (
    <main className="page">
      <header className="header">
        <h1>{config?.client.app_title ?? "Document Voice Agent"}</h1>
        {config && (
          <div className="chips">
            <span className="chip">
              <strong>{config.profile === "local" ? "Local" : "Cloud"}</strong>
              {config.profile === "local" && "· runs on this machine"}
            </span>
            {config.client.languages.map((lang) => (
              <span key={lang} className="chip" lang={lang}>
                {LANGUAGE_NAMES[lang]}
              </span>
            ))}
          </div>
        )}
      </header>

      {state.kind === "error" && (
        <div className="notice" role="alert">
          <strong>{state.message}</strong>
          <p>
            Start it with <code>cd backend &amp;&amp; uv run python -m app</code>, then reload this page.
          </p>
        </div>
      )}

      <div className="grid">
        <section className="card placeholder" aria-labelledby="docs-title">
          <h2 id="docs-title">Documents</h2>
          <p className="sub">
            {config
              ? `${config.limits.allowed_extensions.join(" ")} · up to ${config.limits.max_upload_mb} MB`
              : "Your uploaded files"}
          </p>
          <div className="empty">Uploading and indexing documents arrives in phase 1.</div>
        </section>

        <section className="card placeholder" aria-labelledby="conv-title">
          <h2 id="conv-title">Conversation</h2>
          <p className="sub">Ask by voice or text; answers cite their pages.</p>
          <div className="empty">
            Text chat over your documents arrives in phase 1, voice in phases 4–6.
          </div>
        </section>

        {(config?.features.debug_panel ?? true) && <StatusPanel backendUrl={backendUrl} />}
      </div>
    </main>
  );
}
