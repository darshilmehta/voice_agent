/**
 * Microphone capture for the voice session: getUserMedia (echo cancellation, noise suppression, auto gain), an
 * AudioWorklet that downsamples to 16 kHz and cuts PCM16 frames, and an AnalyserNode that feeds the presence field
 * with the user's voice (after the browser's echo cancellation, so the agent's own voice doesn't drive the user's waves).
 */

import { INPUT_SAMPLE_RATE, UPLINK_FRAME_SAMPLES } from "./protocol";

/**
 * The worklet's source. It lives in a string, not a file, so the app doesn't depend on a bundler or a public path for
 * it; it is loaded from a Blob URL. Each output sample is the average of the input samples it covers (a box filter with
 * fractional edges): enough anti-aliasing for speech at 48 kHz → 16 kHz, no state besides one partial sample.
 */
const WORKLET_SOURCE = `
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.ratio = sampleRate / (o.targetRate || 16000);
    this.frame = o.frameSamples || 512;
    this.out = new Int16Array(this.frame);
    this.pos = 0;
    this.acc = 0;
    this.accW = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const r = this.ratio;
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i];
      let w = 1;
      while (w > 1e-9) {
        const room = r - this.accW;
        if (w < room) {
          this.acc += s * w;
          this.accW += w;
          w = 0;
        } else {
          this.acc += s * room;
          w -= room;
          let v = this.acc / r;
          v = v < -1 ? -1 : v > 1 ? 1 : v;
          this.out[this.pos++] = v < 0 ? v * 32768 : v * 32767;
          this.acc = 0;
          this.accW = 0;
          if (this.pos === this.frame) {
            const buffer = this.out.buffer;
            this.port.postMessage(buffer, [buffer]);
            this.out = new Int16Array(this.frame);
            this.pos = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
`;

export type MicErrorKind = "denied" | "no-device" | "busy" | "unsupported" | "insecure" | "failed";

export class MicError extends Error {
  constructor(
    readonly kind: MicErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "MicError";
  }
}

export const MIC_ERROR_TEXT: Record<MicErrorKind, string> = {
  denied: "Microphone access is blocked. Allow it for this site in your browser's settings, then try again.",
  "no-device": "No microphone was found. Connect one, then try again.",
  busy: "The microphone is in use by another app or couldn't be started.",
  unsupported: "Voice isn't supported in this browser. You can type instead.",
  insecure: "Microphone access needs a secure page (https or localhost). You can type instead.",
  failed: "Couldn't start the microphone.",
};

/** What this browser can do, checked before asking for anything. */
export function voiceSupport(): { ok: true } | { ok: false; kind: MicErrorKind } {
  if (typeof window === "undefined") return { ok: false, kind: "unsupported" };
  if (!window.isSecureContext) return { ok: false, kind: "insecure" };
  const ok =
    !!navigator.mediaDevices?.getUserMedia &&
    typeof AudioContext !== "undefined" &&
    typeof AudioWorkletNode !== "undefined" &&
    typeof WebSocket !== "undefined";
  return ok ? { ok: true } : { ok: false, kind: "unsupported" };
}

function classify(err: unknown): MicError {
  const name = err instanceof DOMException || err instanceof Error ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
    case "PermissionDeniedError":
      return new MicError("denied", MIC_ERROR_TEXT.denied);
    case "NotFoundError":
    case "OverconstrainedError":
    case "DevicesNotFoundError":
      return new MicError("no-device", MIC_ERROR_TEXT["no-device"]);
    case "NotReadableError":
    case "AbortError":
    case "TrackStartError":
      return new MicError("busy", MIC_ERROR_TEXT.busy);
    default:
      return new MicError("failed", MIC_ERROR_TEXT.failed);
  }
}

/** The mic's permission state without prompting, where the browser can say ("prompt" otherwise). */
export async function micPermission(): Promise<"granted" | "denied" | "prompt"> {
  try {
    const status = await navigator.permissions.query({ name: "microphone" as PermissionName });
    return status.state;
  } catch {
    return "prompt"; // Safari and others can't be asked
  }
}

export interface MicCapture {
  stream: MediaStream;
  /** The user's voice, for the presence field. */
  analyser: AnalyserNode;
  stop: () => void;
}

/**
 * Opens the microphone and streams PCM16 16 kHz frames to `onFrame` until `stop()`. Throws MicError. The frame
 * callback gets an ArrayBuffer it owns.
 */
export async function openMic(
  ctx: AudioContext,
  onFrame: (pcm: ArrayBuffer) => void,
  /** The device went away or access was revoked while open (not called by `stop()`). */
  onEnded?: () => void,
): Promise<MicCapture> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (err) {
    throw classify(err);
  }
  try {
    const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" }));
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.3;
    const node = new AudioWorkletNode(ctx, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { targetRate: INPUT_SAMPLE_RATE, frameSamples: UPLINK_FRAME_SAMPLES },
    });
    node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => onFrame(e.data);
    if (onEnded) for (const t of stream.getAudioTracks()) t.addEventListener("ended", onEnded, { once: true });
    // Browsers only pull a node that leads to the destination; a zero-gain sink keeps it running silently.
    const sink = ctx.createGain();
    sink.gain.value = 0;
    source.connect(analyser);
    source.connect(node);
    node.connect(sink);
    sink.connect(ctx.destination);
    return {
      stream,
      analyser,
      stop: () => {
        node.port.onmessage = null;
        for (const t of stream.getTracks()) t.stop();
        try {
          source.disconnect();
          node.disconnect();
          sink.disconnect();
          analyser.disconnect();
        } catch {
          // already disconnected
        }
      },
    };
  } catch (err) {
    for (const t of stream.getTracks()) t.stop();
    throw err instanceof MicError ? err : new MicError("failed", MIC_ERROR_TEXT.failed);
  }
}
