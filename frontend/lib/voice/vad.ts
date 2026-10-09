/**
 * Browser voice-activity detection for barge-in (docs/DESIGN.md §3.3, §9.5): Silero v5 through @ricky0123/vad-web's
 * MicVAD, running on the same microphone stream and AudioContext as the uplink. It tells the page the moment the user
 * starts speaking, so the agent can be ducked locally (< 200 ms) before the server has heard a word.
 *
 * vad-web and onnxruntime-web are served from this app (`public/vad/`, copied from node_modules by
 * scripts/copy-vad-assets.mjs) and loaded here as two classic scripts, on first use only. Nothing comes from a CDN,
 * and the ~14 MB WASM binary stays out of the app bundle (bundlers handle onnxruntime's dynamic WASM import badly).
 */

import type { MicVAD } from "@ricky0123/vad-web";

const ASSET_BASE = "/vad/";

// What the two scripts put on window: `ort` (onnxruntime-web, WASM only) and `vad` (vad-web's UMD bundle).
declare global {
  interface Window {
    ort?: { env: { wasm: { numThreads?: number }; logLevel?: string } };
    vad?: { MicVAD: typeof MicVAD };
  }
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.async = false;
    el.onload = () => resolve();
    el.onerror = () => {
      el.remove();
      reject(new Error(`Couldn't load ${src}`));
    };
    document.head.appendChild(el);
  });
}

let scripts: Promise<void> | null = null;

/** Loads the VAD scripts once per page. A failed load can be retried by the next call. */
export function loadVadScripts(): Promise<void> {
  if (window.vad) return Promise.resolve();
  scripts ??= (async () => {
    await loadScript(`${ASSET_BASE}ort.wasm.min.js`); // window.ort, which vad's bundle expects to find
    await loadScript(`${ASSET_BASE}bundle.min.js`); // window.vad
  })().catch((err) => {
    scripts = null;
    throw err;
  });
  return scripts;
}

export interface BargeInVadEvents {
  /** Speech began (the very first speech frame): duck the agent. */
  onSpeechStart: () => void;
  /** Speech lasted long enough to be real (`minSpeechMs`). */
  onSpeechRealStart: () => void;
  /** It was too short to be speech (a cough, a click, "mm"): restore the volume. */
  onMisfire: () => void;
  onSpeechEnd: () => void;
}

export interface BargeInVad {
  destroy: () => Promise<void>;
}

/**
 * Frees a VAD whose `start()` failed. MicVAD.destroy() assumes the audio nodes exist and throws before it releases the
 * model when they don't (the failure came while they were being made), so the model is released directly then.
 */
async function releaseUnstarted(vad: MicVAD): Promise<void> {
  try {
    await vad.destroy();
    return;
  } catch {
    // no audio nodes to tear down
  }
  try {
    await (vad as unknown as { model: { release: () => Promise<void> } }).model.release();
  } catch {
    // already released
  }
}

/**
 * Starts the VAD on an existing stream and context. Tuning follows the smoke tests (§9.5): thresholds 0.5 / 0.35,
 * ~600 ms of trailing silence ends a segment, anything under ~250 ms is a misfire.
 */
export async function startBargeInVad(
  ctx: AudioContext,
  stream: MediaStream,
  events: BargeInVadEvents,
): Promise<BargeInVad> {
  await loadVadScripts();
  const lib = window.vad;
  if (!lib) throw new Error("The VAD script didn't register");
  const base = new URL(ASSET_BASE, window.location.origin).href;
  const vad = await lib.MicVAD.new({
    model: "v5",
    audioContext: ctx,
    baseAssetPath: base,
    onnxWASMBasePath: base,
    // The page owns the microphone: the VAD listens to its stream and never opens or stops one itself.
    getStream: async () => stream,
    pauseStream: async () => undefined,
    resumeStream: async () => stream,
    startOnLoad: false,
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.35,
    redemptionMs: 600,
    minSpeechMs: 250,
    ortConfig: (ort: NonNullable<Window["ort"]>) => {
      ort.env.logLevel = "error";
      ort.env.wasm.numThreads = 1; // no cross-origin isolation here, so no threads
    },
    onSpeechStart: events.onSpeechStart,
    onSpeechRealStart: events.onSpeechRealStart,
    onVADMisfire: events.onMisfire,
    onSpeechEnd: events.onSpeechEnd,
  } as Parameters<typeof MicVAD.new>[0]);
  try {
    await vad.start();
  } catch (err) {
    // The model and its ONNX session are loaded by now: release them, or they stay in memory for good.
    await releaseUnstarted(vad);
    throw err;
  }
  return { destroy: () => vad.destroy().catch(() => undefined) };
}
