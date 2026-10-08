"use client";

/** Client-side app state for every page: backend URL + config, health, the workspace store, toasts, actions. */

import type { ReactNode } from "react";

import { BackendProvider } from "@/lib/backend-context";
import { HealthProvider } from "@/lib/health";
import { WorkspaceProvider } from "@/lib/workspace";

import { ActionsProvider } from "./Actions";
import { AppFrame } from "./AppFrame";
import { AutoReconnect } from "./States";
import { ToastProvider } from "./Toast";

export function AppProviders({ backendUrl, children }: { backendUrl: string; children: ReactNode }) {
  return (
    <BackendProvider backendUrl={backendUrl}>
      <HealthProvider>
        <WorkspaceProvider>
          <ToastProvider>
            <ActionsProvider>
              <AutoReconnect />
              <AppFrame>{children}</AppFrame>
            </ActionsProvider>
          </ToastProvider>
        </WorkspaceProvider>
      </HealthProvider>
    </BackendProvider>
  );
}
