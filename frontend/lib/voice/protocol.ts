/**
 * The voice session protocol (docs/DESIGN.md §3.10): one WebSocket per live voice session on a chat.
 *
 * Audio is binary, control is JSON text. Client → server audio is raw PCM16 LE mono 16 kHz, 20–64 ms per frame.
 * Server → client audio is a 12-byte header (uint32 LE turn_id, chunk_index, seq) followed by PCM16 LE mono 24 kHz.
 */

import type { Language, Message, SourcesPayload } from "../api";

export const INPUT_SAMPLE_RATE = 16_000;
export const OUTPUT_SAMPLE_RATE = 24_000;
/** Samples per uplink frame: 512 at 16 kHz is 32 ms, 1,024 bytes. */
export const UPLINK_FRAME_SAMPLES = 512;
export const AUDIO_HEADER_BYTES = 12;

/** WebSocket close codes the backend uses. */
export const CLOSE_UNKNOWN_CHAT = 4404;
export const CLOSE_REPLACED = 4409;

export type AgentState = "listening" | "thinking" | "speaking" | "interrupted";
export type ErrorStage = "stt" | "retrieval" | "llm" | "tts" | "storage" | "audio";

// ------------------------------------------------------------------ client → server

export type ClientMessage =
  | { type: "start"; language: Language | null }
  | { type: "barge_in_start"; turn_id: number; played_ms: number }
  | { type: "playback"; turn_id: number; played_ms: number }
  | { type: "playback_done"; turn_id: number }
  | { type: "stop" }
  | { type: "end" };

// ------------------------------------------------------------------ server → client

export type ServerMessage =
  | { type: "ready"; session_id: string; input_sample_rate: number; output_sample_rate: number; language: string | null }
  | { type: "state"; state: AgentState }
  | { type: "user_speech"; phase: "start" | "end" }
  | { type: "transcript_partial"; text: string }
  | { type: "user_message"; message: Message }
  | { type: "turn"; turn_id: number }
  | ({ type: "sources" } & Partial<SourcesPayload>)
  | { type: "delta"; turn_id: number; text: string }
  | { type: "audio_chunk"; turn_id: number; chunk_index: number; text: string; duration_ms: number }
  | { type: "agent_message"; message: Message }
  | { type: "barge_in"; turn_id: number; decision: "stop" | "resume" }
  | { type: "error"; detail: string; stage: ErrorStage | (string & {}) | null };

const SERVER_TYPES = new Set([
  "ready",
  "state",
  "user_speech",
  "transcript_partial",
  "user_message",
  "turn",
  "sources",
  "delta",
  "audio_chunk",
  "agent_message",
  "barge_in",
  "error",
]);

/** A parsed control message, or null for anything that isn't one (unknown types are ignored, not fatal). */
export function parseServerMessage(text: string): ServerMessage | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object") return null;
  const type = (value as { type?: unknown }).type;
  return typeof type === "string" && SERVER_TYPES.has(type) ? (value as ServerMessage) : null;
}

// ------------------------------------------------------------------ audio frames

export interface AudioFrameHeader {
  turnId: number;
  chunkIndex: number;
  seq: number;
}

/** The header of a server audio frame, or null when the frame is too short to hold one. */
export function readAudioHeader(frame: ArrayBuffer): AudioFrameHeader | null {
  if (frame.byteLength < AUDIO_HEADER_BYTES) return null;
  const view = new DataView(frame);
  return { turnId: view.getUint32(0, true), chunkIndex: view.getUint32(4, true), seq: view.getUint32(8, true) };
}

/** PCM16 LE mono 24 kHz as floats in [-1, 1], from the bytes after the header. */
export function pcm16ToFloat32(frame: ArrayBuffer, offset = AUDIO_HEADER_BYTES): Float32Array {
  const samples = Math.max(0, Math.floor((frame.byteLength - offset) / 2));
  const view = new DataView(frame, offset, samples * 2);
  const out = new Float32Array(samples);
  for (let i = 0; i < samples; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

// ------------------------------------------------------------------ endpoint

/** `ws://` / `wss://` URL of a chat's voice socket on the backend origin (keeps any path prefix on BACKEND_URL). */
export function voiceSocketUrl(backendUrl: string, chatId: string): string {
  const url = new URL(backendUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/ws/chats/${encodeURIComponent(chatId)}/voice`;
  url.search = "";
  url.hash = "";
  return url.toString();
}
