/**
 * The room's noise floor, for the browser VAD (docs/DESIGN.md §3.10 "Noisy rooms"). The same algorithm and the same
 * thresholds (`voice_input` in /api/config/public, from `voice.noise`) as the server's gate
 * (backend/app/services/voice/noise.py), so both sides agree on what counts as speech in a noisy room:
 *
 *   every 32 ms frame → its level, 10·log10(mean square) dBFS → a low percentile of the last few seconds = the floor
 *   the louder the floor (quiet_floor_dbfs → loud_floor_dbfs), the higher the VAD threshold and the longer the speech
 *   it needs (→ noisy_threshold, noisy_min_speech_ms); speech only counts once a frame is start_snr_db above the floor
 */

import type { PublicVoiceInput } from "../api";

export const DB_MIN = -100;
const MIN_FLOOR_FRAMES = 31; // ~1 s of frames before the floor is measured rather than assumed
const QUIET_FLOOR_DBFS = -70;
const RECOMPUTE_EVERY = 4;
const FRAME_MS = 32;

/** A frame's level in dBFS (10·log10 of its mean square), at least DB_MIN. */
export function frameDbfs(frame: Float32Array): number {
  if (frame.length === 0) return DB_MIN;
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  const power = sum / frame.length;
  return power > 0 ? Math.max(DB_MIN, 10 * Math.log10(power)) : DB_MIN;
}

/** The `p`-th percentile (0–100) of `values`, linearly interpolated (numpy's default). */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const at = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(at);
  const hi = Math.ceil(at);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo);
}

/** A rolling low percentile of frame levels: what the room sounds like when nobody near the mic is talking. */
export class NoiseFloor {
  private levels: number[] = [];
  private since = 0;
  private value = QUIET_FLOOR_DBFS;
  private readonly size: number;

  constructor(
    windowMs: number,
    private readonly pct: number,
  ) {
    this.size = Math.max(MIN_FLOOR_FRAMES, Math.round(windowMs / FRAME_MS));
  }

  push(dbfs: number): void {
    this.levels.push(dbfs);
    if (this.levels.length > this.size) this.levels.shift();
    this.since++;
    if (this.levels.length >= MIN_FLOOR_FRAMES && (this.since >= RECOMPUTE_EVERY || this.levels.length === MIN_FLOOR_FRAMES)) {
      this.since = 0;
      this.value = percentile(this.levels, this.pct);
    }
  }

  get dbfs(): number {
    return this.value;
  }
}

export interface VadTuning {
  positiveSpeechThreshold: number;
  negativeSpeechThreshold: number;
  minSpeechMs: number;
}

/** How noisy the room is: 0 at or below quiet_floor_dbfs, 1 at or above loud_floor_dbfs. */
export function noisiness(floorDbfs: number, cfg: PublicVoiceInput): number {
  const span = cfg.loud_floor_dbfs - cfg.quiet_floor_dbfs;
  return Math.min(1, Math.max(0, (floorDbfs - cfg.quiet_floor_dbfs) / span));
}

/** The VAD's thresholds for a room with this floor (Silero's off threshold stays 0.15 below the on one). */
export function vadTuning(floorDbfs: number, cfg: PublicVoiceInput): VadTuning {
  const k = cfg.adaptive_gating ? noisiness(floorDbfs, cfg) : 0;
  const top = Math.max(cfg.noisy_threshold, cfg.threshold);
  const on = cfg.threshold + (top - cfg.threshold) * k;
  const minTop = Math.max(cfg.noisy_min_speech_ms, cfg.min_speech_ms);
  return {
    positiveSpeechThreshold: on,
    negativeSpeechThreshold: Math.max(0.01, on - 0.15),
    minSpeechMs: Math.round(cfg.min_speech_ms + (minTop - cfg.min_speech_ms) * k),
  };
}

/** Speech that may duck the agent: the VAD says speech at the room's threshold, and the frame is above the floor. */
export function opensSpeech(probability: number, dbfs: number, floorDbfs: number, cfg: PublicVoiceInput): boolean {
  if (!cfg.adaptive_gating) return probability >= cfg.threshold;
  return probability >= vadTuning(floorDbfs, cfg).positiveSpeechThreshold && dbfs >= floorDbfs + cfg.start_snr_db;
}

/** The settings a backend without `voice_input` implies: main's VAD tuning, no denoiser, no gate. */
export const DEFAULT_VOICE_INPUT: PublicVoiceInput = {
  denoise: "off",
  adaptive_gating: false,
  threshold: 0.5,
  min_speech_ms: 250,
  floor_window_ms: 8000,
  floor_percentile: 20,
  start_snr_db: 9,
  quiet_floor_dbfs: -60,
  loud_floor_dbfs: -35,
  noisy_threshold: 0.8,
  noisy_min_speech_ms: 400,
  assumed_user_dbfs: -26,
  assumed_margin_db: 2,
  far_field_hard_db: 10,
};

/** An utterance's level: this percentile of its speech frames' levels (its loud part), as on the server. */
export const LEVEL_PERCENTILE = 80;

/**
 * The user's own speech level, learnt from their turns (each `user_message` of this session): rises quickly, falls
 * slowly, as on the server (UserLevel). Until the first turn, `assumed_user_dbfs` with `assumed_margin_db` more room.
 */
export class UserLevel {
  dbfs: number | null = null;

  update(level: number | null): void {
    if (level === null || level <= DB_MIN) return;
    if (this.dbfs === null) {
      this.dbfs = level;
      return;
    }
    this.dbfs += (level - this.dbfs) * (level > this.dbfs ? 0.5 : 0.15);
  }
}

/** The quietest level that can still be the user, near the mic (below it: the TV, the next table). */
export function nearFieldMin(cfg: PublicVoiceInput, userDbfs: number | null): number {
  return userDbfs !== null ? userDbfs - cfg.far_field_hard_db : cfg.assumed_user_dbfs - cfg.far_field_hard_db - cfg.assumed_margin_db;
}
