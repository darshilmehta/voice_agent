/**
 * The agent's voice: 24 kHz PCM16 frames from the socket, scheduled back to back on the AudioContext clock (gapless,
 * sample-accurate), through a per-turn gain (so one turn can be faded out and flushed) and a master gain (the duck
 * while the user may be interrupting) into an AnalyserNode that feeds the presence field.
 *
 * Only the current turn is played; frames of any other turn (stale audio after an interruption) are dropped. The same
 * schedule is the clock for everything the page says about playback: how many milliseconds of a turn were actually
 * heard (barge-in, `playback` progress) and how far into each announced chunk the voice is (live captions).
 */

import { OUTPUT_SAMPLE_RATE } from "./protocol";

/** Master volume while the user may be interrupting (docs §3.3: duck, then decide). */
export const DUCK_LEVEL = 0.2;
/** How far ahead of "now" the first audio of a turn (or audio after an underrun) is scheduled, to absorb jitter. */
const START_LEAD_S = 0.1;

interface Slot {
  turnId: number;
  chunkIndex: number;
  start: number;
  dur: number;
  source: AudioBufferSourceNode;
}

interface ChunkSpan {
  start: number;
  end: number;
  /** Announced length (`audio_chunk.duration_ms`), seconds; null until announced. */
  announced: number | null;
}

interface TurnBus {
  gain: GainNode;
  chunks: Map<number, ChunkSpan>;
  /** Everything ever scheduled for the turn, for "how much was played". */
  played: { start: number; dur: number }[];
  /** Ctx time the turn was cut off, or null. Progress stops there. */
  frozenAt: number | null;
  /** The server has sent everything for this turn (agent_message). */
  complete: boolean;
  doneFired: boolean;
}

export interface PlayerEvents {
  /** The agent's audio started or stopped being audible. */
  onAudible?: (audible: boolean) => void;
  /** The last scheduled audio of a completed turn has finished playing. */
  onTurnPlayed?: (turnId: number) => void;
}

export class AgentPlayer {
  readonly analyser: AnalyserNode;
  private master: GainNode;
  private turn: number | null = null;
  private buses = new Map<number, TurnBus>();
  private slots: Slot[] = [];
  private nextTime = 0;
  private cancelled = new Set<number>();
  private audible = false;
  private _ducked = false;
  private poll: number | null = null;
  /** Frames dropped because they belonged to another turn (visible in the debug log). */
  stale = 0;
  underruns = 0;

  constructor(
    private ctx: AudioContext,
    private events: PlayerEvents = {},
    /** Sample rate of the frames from the server (the `ready` message confirms it). */
    public sampleRate = OUTPUT_SAMPLE_RATE,
  ) {
    this.master = ctx.createGain();
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.3;
    this.master.connect(this.analyser);
    this.analyser.connect(ctx.destination);
  }

  get currentTurn(): number | null {
    return this.turn;
  }

  get ducked(): boolean {
    return this._ducked;
  }

  /** Seconds of audio the listener hears "now": the context clock minus the output device's latency. */
  private heardNow(): number {
    const latency = this.ctx.outputLatency || this.ctx.baseLatency || 0;
    return this.ctx.currentTime - Math.min(latency, 0.25);
  }

  /** A new agent turn: older turns' audio is cut off (the server moved on without a stop). */
  setTurn(turnId: number): void {
    if (this.turn === turnId) return;
    if (this.turn !== null && this.turn !== turnId) this.stopTurn(this.turn, { restore: false });
    this.turn = turnId;
    this.cancelled.delete(turnId);
    this.nextTime = 0;
  }

  /** `audio_chunk`: the announced length of a chunk, for caption timing. */
  announce(turnId: number, chunkIndex: number, durationMs: number): void {
    if (turnId !== this.turn || this.cancelled.has(turnId)) return;
    const bus = this.bus(turnId);
    const span = bus.chunks.get(chunkIndex);
    if (span) span.announced = durationMs / 1000;
    else bus.chunks.set(chunkIndex, { start: Infinity, end: 0, announced: durationMs / 1000 });
  }

  private bus(turnId: number): TurnBus {
    let bus = this.buses.get(turnId);
    if (!bus) {
      const gain = this.ctx.createGain();
      gain.connect(this.master);
      bus = { gain, chunks: new Map(), played: [], frozenAt: null, complete: false, doneFired: false };
      this.buses.set(turnId, bus);
      // Keep only a few old turns around.
      for (const [id, old] of this.buses) {
        if (this.buses.size <= 3) break;
        if (id !== this.turn && !this.slots.some((s) => s.turnId === id)) {
          old.gain.disconnect();
          this.buses.delete(id);
        }
      }
    }
    return bus;
  }

  /** Schedule one frame (float samples at the output rate) of `turnId`'s chunk. False if it was dropped as stale. */
  enqueue(turnId: number, chunkIndex: number, pcm: Float32Array): boolean {
    if (turnId !== this.turn || this.cancelled.has(turnId)) {
      this.stale++;
      return false;
    }
    if (pcm.length === 0) return true;
    const ctx = this.ctx;
    const buffer = ctx.createBuffer(1, pcm.length, this.sampleRate);
    buffer.copyToChannel(pcm as Float32Array<ArrayBuffer>, 0);
    const bus = this.bus(turnId);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(bus.gain);

    const now = ctx.currentTime;
    let start = this.nextTime;
    if (start < now + 0.005) {
      if (bus.played.length > 0 && start > 0) this.underruns++;
      start = now + START_LEAD_S;
    }
    source.start(start);
    this.nextTime = start + buffer.duration;
    const slot: Slot = { turnId, chunkIndex, start, dur: buffer.duration, source };
    this.slots.push(slot);
    bus.played.push({ start, dur: buffer.duration });
    const span = bus.chunks.get(chunkIndex);
    if (span) {
      span.start = Math.min(span.start, start);
      span.end = Math.max(span.end, start + buffer.duration);
    } else {
      bus.chunks.set(chunkIndex, { start, end: start + buffer.duration, announced: null });
    }
    source.onended = () => {
      this.slots = this.slots.filter((s) => s !== slot);
      this.check();
    };
    this.setAudible(true);
    this.ensurePoll();
    return true;
  }

  /** The server has sent everything for the turn; `onTurnPlayed` fires once its audio has finished playing. */
  completeTurn(turnId: number): void {
    const bus = this.buses.get(turnId);
    if (!bus) {
      // No audio ever arrived for this turn: nothing to wait for.
      queueMicrotask(() => this.events.onTurnPlayed?.(turnId));
      return;
    }
    bus.complete = true;
    this.check();
  }

  private check(): void {
    const active = this.slots.some((s) => !this.cancelled.has(s.turnId));
    if (!active) this.setAudible(false);
    for (const [id, bus] of this.buses) {
      if (bus.complete && !bus.doneFired && !this.slots.some((s) => s.turnId === id)) {
        bus.doneFired = true;
        if (!this.cancelled.has(id)) this.events.onTurnPlayed?.(id);
      }
    }
  }

  /** Audio is still audible while the schedule hasn't run out (also covers a browser that skips `onended`). */
  private ensurePoll(): void {
    if (this.poll !== null) return;
    this.poll = window.setInterval(() => {
      const now = this.ctx.currentTime;
      this.slots = this.slots.filter((s) => s.start + s.dur > now - 0.05 || this.cancelled.has(s.turnId));
      this.check();
      if (this.slots.length === 0 && this.poll !== null) {
        window.clearInterval(this.poll);
        this.poll = null;
      }
    }, 120);
  }

  private setAudible(value: boolean): void {
    if (this.audible === value) return;
    this.audible = value;
    this.events.onAudible?.(value);
  }

  /** Is the agent's voice playing (or about to) right now? */
  get isAudible(): boolean {
    return this.audible;
  }

  // ------------------------------------------------------------------ duck / stop

  /** Drop the volume at once (the user may be interrupting). */
  duck(level = DUCK_LEVEL): void {
    const g = this.master.gain;
    const t = this.ctx.currentTime;
    g.cancelScheduledValues(t);
    g.setTargetAtTime(level, t, 0.03);
    this._ducked = true;
  }

  /** Back to full volume (it was a cough, a backchannel, or the server said "resume"). */
  restore(): void {
    const g = this.master.gain;
    const t = this.ctx.currentTime;
    g.cancelScheduledValues(t);
    g.setTargetAtTime(1, t, 0.08);
    this._ducked = false;
  }

  /** Fade out and discard everything of a turn, including audio that arrives later. */
  stopTurn(turnId: number, opts: { restore?: boolean } = {}): void {
    const bus = this.buses.get(turnId);
    const t = this.ctx.currentTime;
    this.cancelled.add(turnId);
    if (bus) {
      bus.frozenAt ??= this.heardNow();
      bus.gain.gain.cancelScheduledValues(t);
      bus.gain.gain.setTargetAtTime(0, t, 0.01);
    }
    for (const s of this.slots.filter((x) => x.turnId === turnId)) {
      s.source.onended = null;
      try {
        s.source.stop(t + 0.06);
      } catch {
        // already stopped
      }
    }
    this.slots = this.slots.filter((s) => s.turnId !== turnId);
    if (this.turn === turnId) this.nextTime = 0;
    this.check();
    if (opts.restore !== false) {
      // The duck was for this interruption; the next turn plays at full volume.
      this.master.gain.cancelScheduledValues(t);
      this.master.gain.setTargetAtTime(1, t + 0.1, 0.05);
      this._ducked = false;
    }
  }

  /** Cut off whatever is playing (leaving the page, ending the session). */
  stopAll(): void {
    for (const id of new Set([...this.buses.keys(), ...(this.turn !== null ? [this.turn] : [])])) this.stopTurn(id);
    this.turn = null;
  }

  // ------------------------------------------------------------------ progress

  /** Milliseconds of the turn the listener has actually heard, frozen at the moment it was cut off. */
  playedMs(turnId: number | null): number {
    if (turnId === null) return 0;
    const bus = this.buses.get(turnId);
    if (!bus) return 0;
    const now = Math.min(this.heardNow(), bus.frozenAt ?? Infinity);
    let sec = 0;
    for (const p of bus.played) sec += Math.max(0, Math.min(p.dur, now - p.start));
    return Math.round(sec * 1000);
  }

  /**
   * How far the voice is into an announced chunk, 0…1 (0 until its audio starts, 1 once it ended; frozen if the turn
   * was cut off). Drives the word-by-word highlight of live captions.
   */
  chunkFraction(turnId: number, chunkIndex: number): number {
    const bus = this.buses.get(turnId);
    const span = bus?.chunks.get(chunkIndex);
    if (!bus || !span || !Number.isFinite(span.start)) return 0;
    const now = Math.min(this.heardNow(), bus.frozenAt ?? Infinity);
    const length = span.announced ?? Math.max(0.001, span.end - span.start);
    return Math.min(1, Math.max(0, (now - span.start) / length));
  }

  dispose(): void {
    if (this.poll !== null) window.clearInterval(this.poll);
    this.poll = null;
    this.stopAll();
    try {
      this.master.disconnect();
      this.analyser.disconnect();
    } catch {
      // already disconnected
    }
  }
}
