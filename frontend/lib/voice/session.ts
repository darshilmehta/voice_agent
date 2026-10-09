/**
 * A live voice session on one chat (docs/DESIGN.md §3.3, §3.8, §3.10): the microphone, the WebSocket, the agent's
 * voice and the browser VAD, tied together.
 *
 *   mic ──► AudioWorklet (16 kHz PCM16 frames) ──► WebSocket ──► server (VAD, STT, answer, TTS)
 *    └────► AnalyserNode ────────────────────────────────────► presence field (user's waves)
 *    └────► Silero VAD ──► speech onset while the agent speaks ──► duck locally + barge_in_start
 *   server ──► audio_chunk + 24 kHz frames ──► AgentPlayer (gapless, GainNode) ──► speakers + AnalyserNode (agent's waves)
 *
 * The session is a small external store (`subscribe` / `getSnapshot`) for React. Things that change every frame
 * (the voice's levels, how far into a sentence playback is) are not in the snapshot: the presence field and the
 * captions read them from `player` and the analysers on each animation frame.
 *
 * A turn that searches the web (docs/DESIGN.md §3.7) also gets `tool` messages and a filler `audio_chunk`
 * (`filler: true`): `turn.web` holds the search (`searchingNow` says whether it is running: the label and the presence
 * follow it, since the server stays in `speaking` through the silent search) and every web result that arrived.
 *
 * Failure handling: an unexpected socket close reconnects with backoff (the microphone and VAD keep running); close
 * 4404 (no such chat) and 4409 (replaced by another session on this chat) end the session with an explanation;
 * `error` messages are shown and the session keeps listening.
 */

import type { Language, Message, SourcesPayload } from "../api";
import { publishRawCanvasEvent } from "../canvas/events";
import { withVisual } from "../route";
import { applyTool, endSearch, NO_WEB, parseTool, type WebSearchState, type WebTurn } from "../web-search";
import { MIC_ERROR_TEXT, MicError, openMic, voiceSupport, type MicCapture, type MicErrorKind } from "./capture";
import { AgentPlayer } from "./playback";
import {
  CLOSE_ORIGIN_NOT_ALLOWED,
  CLOSE_REPLACED,
  CLOSE_UNKNOWN_CHAT,
  OUTPUT_SAMPLE_RATE,
  parseServerMessage,
  pcm16ToFloat32,
  readAudioHeader,
  voiceSocketUrl,
  type AgentState,
  type ClientMessage,
  type ServerMessage,
} from "./protocol";
import { startBargeInVad, type BargeInVad } from "./vad";

export type Phase = "idle" | "starting" | "connecting" | "live" | "reconnecting";

export interface ChunkInfo {
  index: number;
  text: string;
  durationMs: number;
  /** The short "Let me look that up." spoken while the web is searched: not part of the answer's text (§3.7). */
  filler: boolean;
}

/** The agent's current answer: what it said, how, and from which sources. */
export interface TurnState {
  /** The server's turn id; null between the user's message and the `turn` message. */
  id: number | null;
  /** `seq` of the user message this turn answers (an older agent message can't belong to it). */
  userSeq: number | null;
  /** The turn id arrived before the user message it answers (`userSeq` is then only the previous turn's). */
  unasked: boolean;
  chunks: ChunkInfo[];
  /** Text as generated (`delta`), which may run ahead of the audio. */
  deltaText: string;
  sources: SourcesPayload | null;
  /** The saved answer. */
  message: Message | null;
  /** Cut off by the user (barge-in stop or the stop button). */
  cut: boolean;
  /** The turn's live web search (`tool` messages) and the web results that arrived for it. */
  web: WebTurn;
}

export interface Notice {
  detail: string;
  stage: string | null;
  key: number;
}

export interface VoiceSnapshot {
  phase: Phase;
  /** Why the microphone can't be used (blocked, missing, unsupported browser), or null. */
  micError: { kind: MicErrorKind; message: string } | null;
  /** The session ended with a problem that isn't the microphone (replaced elsewhere, connection lost), or null. */
  fatal: string | null;
  /** Starting again could help (not for a chat that no longer exists). */
  fatalRetry: boolean;
  /** Audio can't start without a click (the page was opened directly, not by a click). */
  needsGesture: boolean;
  /** A turn failed (the session continues). */
  notice: Notice | null;
  serverState: AgentState | null;
  userSpeaking: boolean;
  /** Whose words the captions show. */
  caption: "user" | "agent" | null;
  /** The user's transcript: partial while they speak, final once saved. */
  userText: string;
  userFinal: boolean;
  /** The agent was ducked and the server hasn't decided yet. */
  ducked: boolean;
  /** The agent's voice is playing. */
  audible: boolean;
  turn: TurnState | null;
  /** Messages the server saved during this page view, in order. */
  messages: Message[];
  /** What the web search of each saved answer was (by message id), for the transcript's note of what was searched. */
  searches: Record<string, WebSearchState>;
  vad: "off" | "loading" | "on" | "unavailable";
  attempt: number;
  /**
   * Counts the times the socket dropped and came back as a new server session. Whatever the server saved for the
   * dropped one (an answer, a transcript) never reached this page: the page fetches the transcript tail again.
   */
  resyncs: number;
}

const INITIAL: VoiceSnapshot = {
  phase: "idle",
  micError: null,
  fatal: null,
  fatalRetry: true,
  needsGesture: false,
  notice: null,
  serverState: null,
  userSpeaking: false,
  caption: null,
  userText: "",
  userFinal: false,
  ducked: false,
  audible: false,
  turn: null,
  messages: [],
  searches: {},
  vad: "off",
  attempt: 0,
  resyncs: 0,
};

export interface VoiceOptions {
  chatId: string;
  backendUrl: string;
  /** null: the server detects the language of each utterance. */
  language: Language | null;
  /** `features.web_search`: false keeps the search's badge, label and note off (web results still resolve their markers). */
  webSearch?: boolean;
}

/**
 * The agent is searching the web right now: a search started and nothing has ended it. Nothing is, once the answer
 * has sources or is saved, the turn is cut, or the session isn't live. The server stays in `speaking` through the
 * silent search after the filler, so the label and the presence follow this instead.
 */
export function searchingNow(s: Pick<VoiceSnapshot, "phase" | "turn">): boolean {
  const t = s.turn;
  return s.phase === "live" && !!t && !t.cut && !t.message && t.web.search?.status === "searching";
}

// 0.4 s doubling up to 8 s: ten attempts keep trying for about a minute, which covers a backend restart with its
// models loading (a restart measured against the real backend took 13 s before the socket accepted again).
const MAX_RECONNECTS = 10;
const CONNECT_TIMEOUT_MS = 8000;
const PROGRESS_MS = 250;
/** If the server never decides about a barge-in, the agent comes back to full volume. */
const DUCK_FAILSAFE_MS = 3500;
const MAX_UPLINK_BUFFER = 1_000_000;

const freshTurn = (id: number | null, userSeq: number | null = null): TurnState => ({
  id,
  userSeq,
  unasked: false,
  chunks: [],
  deltaText: "",
  sources: null,
  message: null,
  cut: false,
  web: NO_WEB,
});

export class VoiceSession {
  private snap: VoiceSnapshot = INITIAL;
  private listeners = new Set<() => void>();
  private notifyQueued = false;

  private ctx: AudioContext | null = null;
  private mic: MicCapture | null = null;
  private _player: AgentPlayer | null = null;
  private vad: BargeInVad | null = null;
  private ws: WebSocket | null = null;
  private ready = false;
  private closing = false;
  /** Bumped by every start and stop: async work started under an older run gives up. */
  private run = 0;

  private reconnectTimer: number | null = null;
  private connectTimer: number | null = null;
  private progressTimer: number | null = null;
  private duckTimer: number | null = null;
  /** barge_in_start was sent and the server hasn't decided yet. */
  private bargePending = false;
  /** The turn that barge-in is about. */
  private bargeTurn: number | null = null;
  private noticeKey = 0;

  // Turn bookkeeping. Turn ids strictly increase within a server session (a reconnect starts a new one), and after
  // a stop nothing more arrives for the stopped turn: these make a misbehaving server harmless.
  /** The highest turn id seen in this server session; a message for an id at or below it, other than the current turn's, is old. */
  private maxTurnId = -1;
  /** Turns that were cut off (stop, barge-in, replaced): whatever else arrives for them is ignored. */
  private cutTurns = new Set<number>();
  /** Turns whose visual is ready (§12.1), by turn id: their answers say "Chart added". */
  private visualOfTurn = new Map<number, string>();
  /** Stop was pressed while the answer had no id yet (or before the question was saved): it cancels the turn that arrives. */
  private stopPending = false;
  /** The socket dropped: the next `ready` is a new server session. */
  private reconnected = false;
  /** The server answered `barge_in: resume`: the utterance behind it was ignored (until the next speech or message). */
  private discardedUtterance = false;
  /** The browser VAD hears the user speaking right now. */
  private vadSpeaking = false;

  /** Counters for the debug hook (window.__voice in development). */
  readonly stats = { uplinkFrames: 0, downlinkFrames: 0, staleFrames: 0 };

  constructor(private opts: VoiceOptions) {}

  // ------------------------------------------------------------------ store

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): VoiceSnapshot => this.snap;

  private set(patch: Partial<VoiceSnapshot>): void {
    this.snap = { ...this.snap, ...patch };
    if (this.notifyQueued) return;
    this.notifyQueued = true;
    queueMicrotask(() => {
      this.notifyQueued = false;
      for (const l of this.listeners) l();
    });
  }

  private patchTurn(change: (t: TurnState) => TurnState, id?: number | null): void {
    const base = this.snap.turn ?? freshTurn(id ?? null);
    this.set({ turn: change(base) });
  }

  /** The turn with its web search ended, when one was running (the session ended, the connection dropped, an error). */
  private searchEnded(turn: TurnState | null): TurnState | null {
    return turn && turn.web.search?.status === "searching" ? { ...turn, web: endSearch(turn.web) } : turn;
  }

  setOptions(opts: Partial<VoiceOptions>): void {
    this.opts = { ...this.opts, ...opts };
  }

  // ------------------------------------------------------------------ read by the presence field and captions

  get player(): AgentPlayer | null {
    return this._player;
  }
  /** The agent's voice (what the speakers play). */
  get agentAnalyser(): AnalyserNode | null {
    return this._player?.analyser ?? null;
  }
  /** The user's voice (the microphone, after echo cancellation). */
  get userAnalyser(): AnalyserNode | null {
    return this.mic?.analyser ?? null;
  }
  get sampleRate(): number {
    return this.ctx?.sampleRate ?? 48000;
  }
  get active(): boolean {
    return this.snap.phase !== "idle";
  }

  // ------------------------------------------------------------------ lifecycle

  /** Open the microphone and the socket. Call from a click (or soon after one) so the browser lets audio play. */
  async start(opts: { auto?: boolean } = {}): Promise<void> {
    if (this.snap.phase !== "idle") return;
    const run = ++this.run;
    this.closing = false;
    this.set({
      phase: "starting",
      micError: null,
      fatal: null,
      needsGesture: false,
      notice: null,
      serverState: null,
      userSpeaking: false,
      userText: "",
      userFinal: false,
      caption: null,
      turn: null,
      ducked: false,
      audible: false,
      attempt: 0,
    });

    const support = voiceSupport();
    if (!support.ok) {
      this.set({ phase: "idle", micError: { kind: support.kind, message: MIC_ERROR_TEXT[support.kind] } });
      return;
    }

    // This run's resources live in locals until the run is known to still be current: a run that was stopped while
    // it waited (the permission prompt can stay open for a long time) releases only its own, never a newer run's.
    let ctx: AudioContext | null = null;
    let mic: MicCapture | null = null;
    const current = () => run === this.run;
    const discard = async () => {
      mic?.stop();
      mic = null;
      if (ctx) {
        ctx.onstatechange = null;
        if (this.ctx === ctx) this.ctx = null; // the owner is gone; stop() may already have torn it down
        if (ctx.state !== "closed") await ctx.close().catch(() => undefined);
      }
    };

    try {
      const context = new AudioContext({ latencyHint: "interactive" });
      ctx = context;
      this.ctx = context; // published at once so that stop() can close it while we wait
      const running = () => context.state === "running";
      if (!running()) {
        // Without a recent click the browser keeps audio suspended: don't ask for the mic just to be stuck.
        await Promise.race([context.resume().catch(() => undefined), new Promise((r) => setTimeout(r, opts.auto ? 400 : 1500))]);
        if (!running()) {
          await discard();
          if (current()) this.set({ phase: "idle", needsGesture: true });
          return;
        }
      }
      if (!current()) return void (await discard());
      // Browsers suspend audio when the tab is hidden or a call comes in (Safari: "interrupted"): pick it up again.
      context.onstatechange = () => {
        if (this.ctx === context && context.state !== "running" && context.state !== "closed") void context.resume().catch(() => undefined);
      };

      mic = await openMic(
        context,
        (pcm) => this.sendAudio(pcm),
        () => {
          if (this.ctx === context && current()) void this.fail("The microphone was disconnected or its access was taken away.");
        },
      );
      if (!current()) return void (await discard());

      this.mic = mic;
      this._player = new AgentPlayer(context, {
        onAudible: (audible) => this.onAudible(audible),
        onTurnPlayed: (turnId) => this.onTurnPlayed(turnId),
      });
    } catch (err) {
      await discard();
      if (!current()) return;
      const kind: MicErrorKind = err instanceof MicError ? err.kind : "failed";
      this.set({ phase: "idle", micError: { kind, message: err instanceof MicError ? err.message : MIC_ERROR_TEXT.failed } });
      return;
    }

    window.addEventListener("pagehide", this.onPageHide);
    window.addEventListener("online", this.onOnline);
    this.set({ phase: "connecting" });
    this.connect();
  }

  /** End the session: tell the server, close the socket, release the microphone and the audio. */
  async stop(): Promise<void> {
    if (this.snap.phase === "idle") return;
    this.run++;
    this.closing = true;
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) this.send({ type: "end" });
    await this.teardown();
    this.set({
      phase: "idle",
      serverState: null,
      userSpeaking: false,
      ducked: false,
      audible: false,
      vad: "off",
      attempt: 0,
      notice: null,
      turn: this.searchEnded(this.snap.turn),
    });
  }

  /** The session can't go on: say why. */
  private async fail(message: string, retry = true): Promise<void> {
    this.run++;
    this.closing = true;
    await this.teardown();
    this.set({
      phase: "idle",
      fatal: message,
      fatalRetry: retry,
      serverState: null,
      userSpeaking: false,
      ducked: false,
      audible: false,
      vad: "off",
      attempt: 0,
      turn: this.searchEnded(this.snap.turn),
    });
  }

  /** Leaving the page. */
  destroy(): void {
    void this.stop();
  }

  private async teardown(): Promise<void> {
    window.removeEventListener("pagehide", this.onPageHide);
    window.removeEventListener("online", this.onOnline);
    for (const t of [this.reconnectTimer, this.connectTimer, this.duckTimer]) if (t !== null) window.clearTimeout(t);
    if (this.progressTimer !== null) window.clearInterval(this.progressTimer);
    this.reconnectTimer = this.connectTimer = this.duckTimer = this.progressTimer = null;
    this.bargePending = false;
    this.bargeTurn = null;
    this.reconnected = false;
    this.stopPending = false;
    this.discardedUtterance = false;
    this.vadSpeaking = false;
    this.maxTurnId = -1;
    this.cutTurns.clear();

    const { ws, vad, mic, ctx } = this;
    const player = this._player;
    this.ws = null;
    this.vad = null;
    this.mic = null;
    this.ctx = null;
    this._player = null;
    this.ready = false;

    if (ws) {
      ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
      try {
        ws.close(1000);
      } catch {
        // already closed
      }
    }
    mic?.stop();
    player?.dispose();
    await vad?.destroy().catch(() => undefined);
    if (ctx) ctx.onstatechange = null;
    if (ctx && ctx.state !== "closed") await ctx.close().catch(() => undefined);
  }

  private onPageHide = () => {
    void this.stop();
  };

  private onOnline = () => {
    if (this.snap.phase === "reconnecting" && this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
      this.connect();
    }
  };

  // ------------------------------------------------------------------ socket

  private connect(): void {
    let ws: WebSocket;
    try {
      ws = new WebSocket(voiceSocketUrl(this.opts.backendUrl, this.opts.chatId));
    } catch {
      void this.fail("The voice connection address isn't valid.");
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    this.ready = false;
    this.connectTimer = window.setTimeout(() => {
      // Connected (or connecting) but never said "ready": try again.
      if (this.ws === ws && !this.ready) ws.close();
    }, CONNECT_TIMEOUT_MS);
    ws.onopen = () => this.send({ type: "start", language: this.opts.language });
    ws.onmessage = (e: MessageEvent) => {
      if (this.ws !== ws) return;
      if (typeof e.data === "string") {
        const msg = parseServerMessage(e.data);
        if (msg) this.onControl(msg);
      } else if (e.data instanceof ArrayBuffer) {
        this.onAudio(e.data);
      }
    };
    ws.onclose = (e: CloseEvent) => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.ready = false;
      if (this.connectTimer !== null) window.clearTimeout(this.connectTimer);
      this.connectTimer = null;
      if (this.closing) return;
      if (e.code === CLOSE_ORIGIN_NOT_ALLOWED)
        void this.fail("Voice isn't allowed from this address. Open the app from the address it is set up for.", false);
      else if (e.code === CLOSE_UNKNOWN_CHAT) void this.fail("This chat no longer exists.", false);
      else if (e.code === CLOSE_REPLACED)
        void this.fail("This conversation was opened in another tab or window, so voice stopped here.");
      else this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.snap.phase === "idle") return;
    const attempt = this.snap.attempt + 1;
    if (attempt > MAX_RECONNECTS) {
      void this.fail("Lost the connection to the voice service.");
      return;
    }
    // Whatever the agent was saying belongs to a session that no longer exists.
    this.cancelPlayback();
    this.reconnected = true;
    this.set({
      phase: "reconnecting",
      attempt,
      serverState: null,
      userSpeaking: false,
      ducked: false,
      turn: this.searchEnded(this.snap.turn),
    });
    const delay = Math.min(8000, 400 * 2 ** (attempt - 1)) + Math.random() * 200;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private send(msg: ClientMessage): void {
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }

  private sendAudio(pcm: ArrayBuffer): void {
    const ws = this.ws;
    if (!this.ready || !ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > MAX_UPLINK_BUFFER) return;
    ws.send(pcm);
    this.stats.uplinkFrames++;
  }

  // ------------------------------------------------------------------ server → client

  private onAudio(frame: ArrayBuffer): void {
    const header = readAudioHeader(frame);
    const player = this._player;
    if (!header || !player) return;
    this.stats.downlinkFrames++;
    // Only the current agent turn is played; stale audio (after an interruption) is dropped.
    if (!player.enqueue(header.turnId, header.chunkIndex, pcm16ToFloat32(frame))) this.stats.staleFrames++;
  }

  /**
   * The turn a message belongs to, creating it if it's a new one; null for a turn that has been left behind.
   *
   * Ids strictly increase within a server session, so a turn that was cut off, or any id at or below the highest seen
   * (other than the current turn's), is a late message from the past and must not become the current turn: a late
   * `delta` of a stopped turn used to take the new question's place and silence its answer. A new turn that was
   * stopped before it had an id (Stop pressed while thinking) is cancelled the moment it gets one.
   */
  private adopt(turnId: number): TurnState | null {
    const cur = this.snap.turn;
    if (cur && cur.id === turnId) return cur;
    if (this.cutTurns.has(turnId) || turnId <= this.maxTurnId) return null;
    this.maxTurnId = turnId;
    this._player?.setTurn(turnId);
    const placeholder = cur && cur.id === null ? cur : null;
    // The question this answers (its user_message came first) stays attached, so an older turn's saved answer that
    // arrives late can still be told apart.
    const next: TurnState = placeholder
      ? { ...placeholder, id: turnId }
      : { ...freshTurn(turnId, cur?.userSeq ?? null), unasked: true }; // `turn` came before its user_message
    if (placeholder?.cut || this.stopPending) {
      this.cutTurns.add(turnId);
      this._player?.stopTurn(turnId);
      next.cut = true;
      // The server drops a `stop` that finds no answer in progress (the question was still being transcribed), so
      // it would go on to write and speak this one for nobody: now that the turn is known, say stop again.
      this.send({ type: "stop" });
    }
    this.set({ turn: next });
    return next;
  }

  /** The agent's words own the captions again, unless the user is talking right now. */
  private agentCaption(): Partial<VoiceSnapshot> {
    return this.snap.userSpeaking ? {} : { caption: "agent" };
  }

  /** A new server session numbers its turns from the start again; forget the old one's. */
  private resetTurns(): void {
    this.maxTurnId = -1;
    this.cutTurns.clear();
    this.stopPending = false;
    this._player?.reset();
    this.clearBarge();
  }

  /** The agent is in the middle of an answer (being written or spoken), not finished or cut off. */
  private answering(): boolean {
    const t = this.snap.turn;
    return !!t && t.id !== null && !t.cut && (!t.message || this.snap.audible);
  }

  /** The utterance was ignored by the server: drop its words, and any transcript of it that is still on its way. */
  private ignoreUtterance(): void {
    this.discardedUtterance = true;
    this.dropUserCaption();
  }

  /** The user's words were never saved as a message (a backchannel, noise, a failed transcription): stop showing them. */
  private dropUserCaption(): void {
    const t = this.snap.turn;
    const hasWords = !!t && (t.chunks.length > 0 || t.deltaText !== "" || !!t.message);
    this.set({ userText: "", userFinal: false, caption: hasWords ? "agent" : null });
  }

  /** Messages the page fetched itself (the transcript tail after a reconnect), merged by id, in order. */
  mergeMessages(items: Message[]): void {
    if (items.length === 0) return;
    const byId = new Map(this.snap.messages.map((m) => [m.id, m] as const));
    for (const m of items) byId.set(m.id, m); // what the server saved is the truth: it wins over a copy we had
    this.set({ messages: [...byId.values()].sort((a, b) => a.seq - b.seq) });
  }

  private onControl(msg: ServerMessage): void {
    switch (msg.type) {
      case "ready": {
        this.ready = true;
        if (this.connectTimer !== null) window.clearTimeout(this.connectTimer);
        this.connectTimer = null;
        // Every `ready` is a new server session: its turn numbers start over, and what the old one was doing is gone.
        const again = this.reconnected;
        this.reconnected = false;
        this.resetTurns();
        if (msg.output_sample_rate && this._player) this._player.sampleRate = msg.output_sample_rate || OUTPUT_SAMPLE_RATE;
        const stale = this.snap.turn;
        this.set({
          phase: "live",
          attempt: 0,
          fatal: null,
          ...(again
            ? {
                // An answer the dropped session was giving is over; the server saved what it had, and the page
                // fetches the transcript again (`resyncs`) so it shows up.
                turn: stale && stale.message ? stale : null,
                userText: "",
                userFinal: false,
                userSpeaking: false,
                caption: stale && stale.message ? "agent" : null,
                resyncs: this.snap.resyncs + 1,
              }
            : {}),
        });
        if (this.progressTimer === null) this.progressTimer = window.setInterval(() => this.reportProgress(), PROGRESS_MS);
        if (!this.vad && this.snap.vad === "off") void this.startVad();
        break;
      }
      case "state": {
        // `listening` means the agent isn't doing anything: a search still marked running was ended without a word.
        this.set(msg.state === "listening" ? { serverState: msg.state, turn: this.searchEnded(this.snap.turn) } : { serverState: msg.state });
        // Words the user said that never became a message: a Stop that nothing answered is over, and the words go.
        // The server closes every `user_speech: start` with `end` and re-sends the state: `listening` when nothing was
        // going on, the agent's own state when it was answering (the speech was ignored). Anything but that agent
        // state or `listening` after the end of speech is the question being worked on: its words stay.
        const s = this.snap;
        const unsaved = s.caption === "user" && !s.userFinal && !s.userSpeaking;
        if (msg.state === "listening") {
          this.stopPending = false;
          if (unsaved) this.ignoreUtterance();
        } else if (unsaved && (msg.state === "speaking" || msg.state === "thinking") && this.answering()) {
          this.ignoreUtterance();
        }
        break;
      }
      case "user_speech":
        if (msg.phase === "start") {
          this.stopPending = false; // a new utterance: an earlier Stop no longer applies
          this.discardedUtterance = false;
          this.set({ userSpeaking: true, caption: "user", userText: "", userFinal: false });
          // The browser VAD normally got here first; this covers a VAD that isn't running.
          this.beginBargeIn();
        } else {
          this.set({ userSpeaking: false });
        }
        break;
      case "transcript_partial":
        // The server transcribes speculatively, so the text of an utterance it has already decided to ignore ("okay"
        // → `barge_in: resume`) can still arrive after the decision, and after `user_speech: end`.
        if (this.discardedUtterance) break;
        this.set({ userText: msg.text, userFinal: false, caption: "user" });
        break;
      case "user_message": {
        // A new turn begins. If `turn` already arrived (it may come with the message) and nothing has been said
        // for it, keep its id; otherwise start fresh and wait for `turn`.
        const cur = this.snap.turn;
        const early =
          !!cur && cur.unasked && cur.id !== null && !cur.message && cur.chunks.length === 0 && cur.deltaText === "";
        this.discardedUtterance = false;
        this.set({
          notice: null, // a new turn: the last one's failure has been seen
          messages: this.withMessage(msg.message),
          userText: msg.message.text,
          userFinal: true,
          userSpeaking: false,
          caption: "user",
          turn: early ? { ...cur, userSeq: msg.message.seq, unasked: false } : freshTurn(null, msg.message.seq),
        });
        break;
      }
      case "turn":
        this.adopt(msg.turn_id);
        break;
      case "tool": {
        // The turn's web search (§3.7). Its results are merged into the turn's sources (a continuation can cite `[W3]`,
        // which only a later `tool results` carries); the search state is what the badge, the label and the presence show.
        const tool = parseTool(msg);
        if (!tool || typeof msg.turn_id !== "number") break;
        const turn = this.adopt(msg.turn_id);
        if (!turn || turn.cut) break; // a turn that was cut off, or a late message of an old one
        this.set({ turn: { ...turn, web: applyTool(turn.web, tool, this.opts.webSearch === true) } });
        break;
      }
      case "sources":
        // `sources` has no turn id: it belongs to the turn in progress, unless that one was cut off. The answer is
        // starting, so the search is no longer what is being waited for (it may still bring more results).
        if (this.snap.turn?.cut) break;
        this.patchTurn((t) => ({
          ...t,
          sources: { sources: msg.sources ?? [], confidence: msg.confidence ?? null, abstained: msg.abstained === true },
          web: endSearch(t.web),
        }));
        break;
      case "delta": {
        const turn = this.adopt(msg.turn_id);
        if (!turn || turn.cut) break; // a turn that was cut off, or a late message of an old one
        this.set({ turn: { ...turn, deltaText: turn.deltaText + msg.text }, ...this.agentCaption() });
        break;
      }
      case "audio_chunk": {
        const turn = this.adopt(msg.turn_id);
        if (!turn || turn.cut) break;
        this._player?.announce(msg.turn_id, msg.chunk_index, msg.duration_ms);
        const chunk: ChunkInfo = {
          index: msg.chunk_index,
          text: msg.text,
          durationMs: msg.duration_ms,
          filler: msg.filler === true,
        };
        const chunks = [...turn.chunks.filter((c) => c.index !== chunk.index), chunk].sort((a, b) => a.index - b.index);
        this.set({ turn: { ...turn, chunks }, ...this.agentCaption() });
        break;
      }
      case "agent_message": {
        this.stopPending = false;
        const turn = this.snap.turn ?? freshTurn(null);
        // Its visual may have become ready first (§12.1): the saved message didn't know yet ("Chart added").
        const visualId = turn.id !== null ? this.visualOfTurn.get(turn.id) : undefined;
        const m = visualId ? withVisual(msg.message, visualId) : msg.message;
        // The server sends an answer again, with the same id, when a complete answer is cut during playback (it now
        // has `heard_text`): that updates the turn's message and the transcript in place.
        const resent = turn.message !== null && turn.message.id === m.id;
        if (!resent && ((turn.userSeq !== null && m.seq < turn.userSeq) || turn.message !== null)) {
          // The saved answer of an earlier turn (an interrupted one arriving after the next question), or a second
          // answer for a turn that has one: it only joins the transcript.
          this.set({ messages: this.withMessage(m) });
          break;
        }
        const cut = turn.cut || m.heard_text !== null;
        if (cut && turn.id !== null) {
          this.cutTurns.add(turn.id);
          this._player?.stopTurn(turn.id);
        }
        // The answer is saved: the search is over, and what it was stays with the message (the transcript's note).
        const web = endSearch(turn.web);
        this.set({
          messages: this.withMessage(m),
          searches: web.search ? { ...this.snap.searches, [m.id]: web.search } : this.snap.searches,
          turn: { ...turn, message: m, cut, web },
          caption: "agent",
        });
        if (!cut && !resent && turn.id !== null) this._player?.completeTurn(turn.id);
        break;
      }
      case "barge_in":
        if (msg.decision === "stop") {
          this.applyStop(msg.turn_id);
        } else {
          // A backchannel or noise: the agent carries on, and the user's words (never saved) leave the captions.
          this.resumeVolume();
          this.discardedUtterance = true;
          if (!this.snap.userFinal) this.dropUserCaption();
        }
        break;
      case "visual":
      case "canvas":
        // The live canvas follows these (lib/canvas). They may come after the turn's agent_message, or for a turn
        // that was cut; a visual that is ready only marks its answer in the transcript ("Chart added").
        publishRawCanvasEvent(this.opts.chatId, msg.type, msg);
        if (msg.type === "visual" && msg.phase === "ready" && typeof msg.turn_id === "number") {
          this.noteVisual(msg.turn_id, msg.visual_id);
        }
        break;
      case "error":
        // The error path may send no terminal `tool` event: whatever search was running is over.
        this.set({ notice: { detail: msg.detail, stage: msg.stage, key: ++this.noticeKey }, turn: this.searchEnded(this.snap.turn) });
        // A failed transcription (or any failure before the question was saved) leaves no message: clear the words.
        if (!this.snap.userFinal && this.snap.caption === "user" && !this.snap.userSpeaking) this.dropUserCaption();
        break;
    }
  }

  /** A turn's visual became ready: remember it, and mark the turn's saved answer if it is there already. */
  private noteVisual(turnId: number, visualId: string): void {
    this.visualOfTurn.set(turnId, visualId);
    const turn = this.snap.turn;
    const answer = turn && turn.id === turnId ? turn.message : null;
    if (turn && answer) {
      const marked = withVisual(answer, visualId);
      this.set({ messages: this.withMessage(marked), turn: { ...turn, message: marked } });
    }
  }

  /** The saved messages with `m` added, or, when one with its id is already there, replaced in place (never twice). */
  private withMessage(m: Message): Message[] {
    const at = this.snap.messages.findIndex((x) => x.id === m.id);
    if (at === -1) return [...this.snap.messages, m];
    const next = this.snap.messages.slice();
    next[at] = m;
    return next;
  }

  // ------------------------------------------------------------------ playback, barge-in

  private onAudible(audible: boolean): void {
    this.set({ audible });
    // The answer starts playing while the user is still talking (they spoke up while it was being written): that is
    // an interruption like any other, so duck at once and let the server decide, instead of talking over them.
    if (audible && (this.vadSpeaking || this.snap.userSpeaking)) this.beginBargeIn();
  }

  /** All of a turn's audio has played: tell the server. */
  private onTurnPlayed(turnId: number): void {
    const turn = this.snap.turn;
    if (turn && turn.id === turnId && !turn.cut) this.send({ type: "playback_done", turn_id: turnId });
  }

  private reportProgress(): void {
    const player = this._player;
    const turn = this.snap.turn;
    if (!player || !turn || turn.id === null || turn.cut || !player.isAudible) return;
    this.send({ type: "playback", turn_id: turn.id, played_ms: player.playedMs(turn.id) });
  }

  private currentTurnId(): number | null {
    return this._player?.currentTurn ?? this.snap.turn?.id ?? null;
  }

  /** The user started talking while the agent speaks: duck now, let the server decide (§3.3). */
  private beginBargeIn(): void {
    const player = this._player;
    if (!player || this.bargePending || this.snap.phase !== "live") return;
    const turnId = this.currentTurnId();
    if (turnId === null || this.snap.turn?.cut) return;
    if (!player.isAudible && this.snap.serverState !== "speaking") return;
    this.bargePending = true;
    this.bargeTurn = turnId;
    if (player.isAudible) {
      player.duck();
      this.set({ ducked: true });
    }
    // If the server never answers, don't leave the agent quiet.
    this.duckTimer = window.setTimeout(() => this.resumeVolume(), DUCK_FAILSAFE_MS);
    this.send({ type: "barge_in_start", turn_id: turnId, played_ms: player.playedMs(turnId) });
  }

  private clearBarge(): void {
    this.bargePending = false;
    this.bargeTurn = null;
    if (this.duckTimer !== null) window.clearTimeout(this.duckTimer);
    this.duckTimer = null;
  }

  /** Server: it was noise or a backchannel. Back to full volume. */
  private resumeVolume(): void {
    this.clearBarge();
    this._player?.restore();
    if (this.snap.ducked) this.set({ ducked: false });
  }

  /**
   * Server: the interruption is real. Stop that turn and discard what's queued.
   *
   * It comes after a `barge_in_start` (we ducked and asked), but also without one (the user spoke while the answer was
   * still being written), and after our own Stop (the server decides a pending barge-in on `stop` too). So it must be
   * safe to receive for a turn that is already cut: then there is nothing left to do, and a newer turn's pending
   * barge-in (its duck, its timer) is not touched.
   */
  private applyStop(turnId: number): void {
    this.discardedUtterance = false; // this utterance is a real one
    if (this.cutTurns.has(turnId)) return; // our own Stop, or an earlier stop, already cut it
    const ofNewerTurn = this.bargePending && this.bargeTurn !== null && this.bargeTurn !== turnId;
    if (!ofNewerTurn) this.clearBarge();
    this.cutTurns.add(turnId); // whatever else arrives for it is ignored, even if it comes before the turn is known
    this._player?.stopTurn(turnId, { restore: !ofNewerTurn });
    const cur = this.snap.turn;
    // Only the turn it names is cut: with the next question already placed, this is an older turn's stop.
    this.set({
      ducked: ofNewerTurn ? this.snap.ducked : false,
      turn: cur && cur.id === turnId ? { ...cur, cut: true, web: endSearch(cur.web) } : cur,
    });
  }

  private cancelPlayback(): void {
    this.clearBarge();
    this._player?.stopAll();
    this.set({ ducked: false });
  }

  /** Stop button / Esc: cut the answer off, keep listening. */
  stopAnswer(): void {
    if (this.snap.phase !== "live") return;
    const cur = this.snap.turn;
    this.clearBarge();
    const playing = cur?.id != null && !!this._player?.isAudible && !this._player.isCancelled(cur.id);
    if (cur && cur.id !== null && !cur.cut && (!cur.message || playing)) {
      // The answer in progress (still being written, or written and still being spoken) has an id: cut it off.
      this.cutTurns.add(cur.id);
      this._player?.stopTurn(cur.id);
      this.set({ ducked: false, turn: { ...cur, cut: true, web: endSearch(cur.web) } });
    } else if (cur && cur.id === null && !cur.cut) {
      // The question is saved but its answer has no id yet: mark it, so the turn is cancelled when it gets one.
      this.stopPending = true;
      this.set({ ducked: false, turn: { ...cur, cut: true, web: endSearch(cur.web) } });
    } else {
      // Nothing in flight we know of (the question is still being transcribed): cancel the turn that arrives.
      this.stopPending = true;
      this.set({ ducked: false });
    }
    this.send({ type: "stop" });
  }

  dismissNotice(): void {
    if (this.snap.notice) this.set({ notice: null });
  }

  // ------------------------------------------------------------------ browser VAD

  private async startVad(): Promise<void> {
    const { ctx, mic } = this;
    const run = this.run;
    if (!ctx || !mic) return;
    this.set({ vad: "loading" });
    try {
      const vad = await startBargeInVad(ctx, mic.stream, {
        onSpeechStart: () => {
          this.vadSpeaking = true;
          this.beginBargeIn();
        },
        onSpeechRealStart: () => undefined, // the server decides once it has heard enough
        onMisfire: () => {
          // Too short to be speech (a cough, a click): the agent carries on at full volume.
          this.vadSpeaking = false;
          if (this.bargePending) this.resumeVolume();
        },
        onSpeechEnd: () => {
          this.vadSpeaking = false;
        },
      });
      if (run !== this.run) {
        await vad.destroy();
        return;
      }
      this.vad = vad;
      this.set({ vad: "on" });
    } catch (err) {
      console.warn("Browser VAD unavailable; interruptions rely on the server:", err);
      if (run === this.run) this.set({ vad: "unavailable" });
    }
  }
}
