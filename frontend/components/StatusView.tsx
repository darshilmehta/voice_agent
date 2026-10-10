"use client";

/** /status: the phase 0 system-status panel plus what this frontend is connected to. */

import { useBackend, useDocumentTitle } from "@/lib/backend-context";
import { LANGUAGE_NAMES } from "@/lib/format";
import { useHealth } from "@/lib/health";

import { BackendDown } from "./States";
import { StatusPanel } from "./StatusPanel";

export function StatusView() {
  const { backendUrl, config, configState } = useBackend();
  const { error } = useHealth();
  useDocumentTitle("System status");

  return (
    <div className="page">
      <header className="page-head">
        <h1 className="page-title">System status</h1>
        <p className="lede">Every component Docent depends on, checked live by the backend.</p>
      </header>

      {error && <BackendDown />}

      <div className="status-layout">
        <StatusPanel />

        <section className="card" aria-labelledby="conn-title">
          <h2 id="conn-title">Connection</h2>
          <dl className="facts">
            <div>
              <dt>Backend</dt>
              <dd>
                <code>{backendUrl}</code>
              </dd>
            </div>
            {config ? (
              <>
                <div>
                  <dt>Profile</dt>
                  <dd>{config.profile === "local" ? "Local · runs on this machine" : "Cloud"}</dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd>{config.version}</dd>
                </div>
                <div>
                  <dt>Languages</dt>
                  <dd>
                    {config.client.languages.map((l) => (
                      <span key={l} lang={l}>
                        {LANGUAGE_NAMES[l] ?? l}
                        {l === config.client.default_language ? " (default)" : ""}
                        {l !== config.client.languages.at(-1) && ", "}
                      </span>
                    ))}
                  </dd>
                </div>
                <div>
                  <dt>Uploads</dt>
                  <dd>
                    {config.limits.allowed_extensions.join(" ")} · up to {config.limits.max_upload_mb} MB
                  </dd>
                </div>
                <div>
                  <dt>Features</dt>
                  <dd>
                    Voice {config.features.voice ? "on" : "off"} · web search {config.features.web_search ? "on" : "off"}
                  </dd>
                </div>
              </>
            ) : (
              <div>
                <dt>Config</dt>
                <dd>{configState.kind === "error" ? configState.message : "Loading…"}</dd>
              </div>
            )}
          </dl>
        </section>
      </div>
    </div>
  );
}
