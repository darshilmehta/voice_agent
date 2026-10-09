"use client";

/**
 * The debug form (loaded only when the public config has `features.debug_panel`, see CanvasDebug.tsx): paste a
 * VisualSpec as JSON and send it to `POST /api/chats/{id}/visuals` to see what the backend builds from it on this
 * chat's canvas.
 */

import { useId, useMemo, useState, type FormEvent } from "react";

import { errorMessage } from "@/lib/api";
import { useBackend } from "@/lib/backend-context";
import { createCanvasApi } from "@/lib/canvas/client";
import { publishCanvasEvent } from "@/lib/canvas/events";

export default function CanvasDebugForm({ chatId }: { chatId: string }) {
  const { backendUrl } = useBackend();
  const client = useMemo(() => createCanvasApi(backendUrl), [backendUrl]);
  const [text, setText] = useState("");
  const [state, setState] = useState<{ kind: "idle" | "busy" | "ok" | "error"; message: string }>({ kind: "idle", message: "" });
  const fieldId = useId();

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    let spec: unknown;
    try {
      spec = JSON.parse(text);
    } catch (err) {
      setState({ kind: "error", message: `That isn't valid JSON: ${(err as Error).message}` });
      return;
    }
    setState({ kind: "busy", message: "Sending…" });
    try {
      const visual = await client.createVisual(chatId, spec);
      // The canvas picks it up like any other `ready` visual; without a visual in the answer it reads the canvas again.
      publishCanvasEvent(chatId, { type: "visual", phase: "ready", visualId: visual?.id ?? "debug", visual, detail: null, turnId: null });
      setState({ kind: "ok", message: visual ? `Added “${visual.title}”.` : "Sent. The canvas will refresh." });
    } catch (err) {
      setState({ kind: "error", message: errorMessage(err) });
    }
  };

  return (
    <details className="cv-debug">
      <summary>Debug: add a visual from a VisualSpec</summary>
      <form className="form" onSubmit={submit}>
        <label className="field" htmlFor={fieldId}>
          <span className="field-label">VisualSpec (JSON)</span>
          <textarea
            id={fieldId}
            className="input textarea"
            rows={6}
            spellCheck={false}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder='{"kind": "line", "title": "Revenue", …}'
          />
        </label>
        <div className="cv-debug-row">
          <button type="submit" className="btn btn-sm btn-primary" disabled={state.kind === "busy" || text.trim() === ""}>
            Add to the canvas
          </button>
          {state.message && (
            <span className={state.kind === "error" ? "form-error" : "cv-debug-msg"} role="status">
              {state.message}
            </span>
          )}
        </div>
      </form>
    </details>
  );
}
