"use client";

/**
 * Debug panel gate: only when the public config has `features.debug_panel` does the chat header load and show the form
 * that pastes a VisualSpec into `POST /api/chats/{id}/visuals` (CanvasDebugForm.tsx). Every other session never
 * downloads it.
 */

import { lazy, Suspense } from "react";

import { useBackend } from "@/lib/backend-context";

const Form = lazy(() => import("./CanvasDebugForm"));

export function CanvasDebug({ chatId }: { chatId: string }) {
  const { config } = useBackend();
  if (config?.features.debug_panel !== true) return null;
  return (
    <Suspense fallback={null}>
      <Form chatId={chatId} />
    </Suspense>
  );
}
