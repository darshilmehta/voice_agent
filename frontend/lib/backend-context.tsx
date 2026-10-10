"use client";

/**
 * The backend URL and public config, shared by every client component.
 *
 * The root layout reads BACKEND_URL on the server for each request (lib/backend-url.ts) and hands it to this
 * provider, so the browser never depends on a build-time value (docs/DESIGN.md §6.5). Title, languages, feature
 * flags and limits come from GET /api/config/public.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { createApi, errorMessage, isAbort, type Api, type PublicConfig } from "./api";
import { PRODUCT_NAME, TAGLINE } from "./brand";

export const DEFAULT_TITLE = PRODUCT_NAME;

type ConfigState =
  | { kind: "loading" }
  | { kind: "ready"; config: PublicConfig }
  | { kind: "error"; message: string; unreachable: boolean };

interface BackendContextValue {
  backendUrl: string;
  api: Api;
  configState: ConfigState;
  config: PublicConfig | null;
  appTitle: string;
  reloadConfig: () => void;
}

const BackendContext = createContext<BackendContextValue | null>(null);

export function BackendProvider({ backendUrl, children }: { backendUrl: string; children: ReactNode }) {
  const api = useMemo(() => createApi(backendUrl), [backendUrl]);
  const [configState, setConfigState] = useState<ConfigState>({ kind: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    api
      .publicConfig(ctrl.signal)
      .then((config) => setConfigState({ kind: "ready", config }))
      .catch((err: unknown) => {
        if (isAbort(err) || ctrl.signal.aborted) return;
        setConfigState({
          kind: "error",
          message: errorMessage(err),
          unreachable: (err as { unreachable?: boolean }).unreachable ?? false,
        });
      });
    return () => ctrl.abort();
  }, [api, attempt]);

  const reloadConfig = useCallback(() => setAttempt((n) => n + 1), []);
  const config = configState.kind === "ready" ? configState.config : null;

  const value = useMemo<BackendContextValue>(
    () => ({
      backendUrl,
      api,
      configState,
      config,
      appTitle: config?.client.app_title ?? DEFAULT_TITLE,
      reloadConfig,
    }),
    [backendUrl, api, configState, config, reloadConfig],
  );

  return <BackendContext value={value}>{children}</BackendContext>;
}

export function useBackend(): BackendContextValue {
  const ctx = useContext(BackendContext);
  if (!ctx) throw new Error("useBackend() must be used inside <BackendProvider>");
  return ctx;
}

/**
 * Sets the browser tab title to "<page> · <app title>" while the calling page is shown ("<app title> · <tagline>"
 * on the home page, which passes ``home``). Next re-applies the root layout's static metadata title after client
 * navigations, so the title is re-asserted whenever <head> changes.
 */
export function useDocumentTitle(page: string | null | undefined, home = false) {
  const { appTitle } = useBackend();
  const wanted = page ? `${page} · ${appTitle}` : home ? `${appTitle} · ${TAGLINE}` : appTitle;
  useEffect(() => {
    const apply = () => {
      if (document.title !== wanted) document.title = wanted;
    };
    apply();
    const observer = new MutationObserver(apply);
    observer.observe(document.head, { subtree: true, childList: true, characterData: true });
    return () => observer.disconnect();
  }, [wanted]);
}
