"use client";

/**
 * Live captions under the presence field. The agent's words light up one by one as they are spoken, following the
 * audio that is actually playing (lib/voice/captions.ts); the user's words appear in the warm colour while they
 * speak, then as the saved transcript. After an interruption the caption keeps only what was heard, then "…".
 *
 * Captions are visual, with no live region: the spoken answer is already audio, and state changes are announced by
 * the state label. The whole conversation is in the transcript.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { chunkWords, plainSpeech, spokenCount, turnWords, type CaptionWord } from "@/lib/voice/captions";
import type { VoiceSession, VoiceSnapshot } from "@/lib/voice/session";

const split = (text: string): string[] => plainSpeech(text).split(" ").filter(Boolean);

export function Captions({
  session,
  snapshot,
  language,
  hint,
}: {
  session: VoiceSession;
  snapshot: VoiceSnapshot;
  language: string | null;
  /** Shown while nothing has been said yet. */
  hint: string;
}) {
  const { turn, caption, userText, userFinal, audible } = snapshot;
  const boxRef = useRef<HTMLParagraphElement>(null);
  const [spoken, setSpoken] = useState(0);

  // While the user speaks (or has just spoken) their words own the captions, even before the first partial arrives.
  const showUser = caption === "user";
  const userWords = showUser ? split(userText) : [];

  const heardText = turn?.message?.heard_text ?? null;
  /** Cut off by the user: show what was heard, then "…". */
  const interrupted = !!turn && (turn.cut || heardText !== null);
  /** Interrupted, and no audio was announced for it: the server's "what was heard" is all there is. */
  const trustServer = !!turn?.message && heardText !== null && turn.chunks.length === 0;
  const agentWords = useMemo<CaptionWord[]>(() => {
    if (!turn) return [];
    if (trustServer) return chunkWords(heardText ?? "", 0);
    if (turn.message && turn.chunks.length === 0) return chunkWords(turn.message.text, 0); // spoken without caption timing
    return turnWords(turn.chunks, turn.deltaText);
  }, [turn, trustServer, heardText]);

  const settled = !!turn?.message && !audible && !interrupted; // saved, and the audio has played out (or never came)
  const agentId = turn?.id ?? null;
  const showAgent = !showUser && agentWords.length > 0;

  // Follow the playback clock: how many words have been spoken so far. Only re-renders when the count changes.
  useEffect(() => {
    if (!showAgent || trustServer) return;
    const player = session.player;
    if (settled || !player || agentId === null) {
      // Nothing is playing any more: a finished answer shows in full; a cut-off one stays as it was.
      if (!interrupted) setSpoken(agentWords.length);
      return;
    }
    const count = () => spokenCount(agentWords, (chunk) => player.chunkFraction(agentId, chunk));
    if (interrupted) {
      setSpoken(count()); // the player froze its clock when the turn was cut off
      return;
    }
    let raf = 0;
    const tick = () => {
      setSpoken(count());
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [session, showAgent, trustServer, agentId, agentWords, settled, interrupted]);

  const words: { text: string; on: boolean; pending: boolean }[] = showUser
    ? userWords.map((text) => ({ text, on: true, pending: false }))
    : agentWords.map((w, i) => ({
        text: w.text,
        on: trustServer || settled || i < spoken,
        pending: w.chunk < 0 && !settled,
      }));
  // After an interruption only what was heard stays, followed by "…".
  const cutOff = showAgent && interrupted;
  const visible = cutOff ? words.filter((w) => w.on) : words;

  // Keep the word being spoken in view when the answer is longer than the caption area.
  const onCount = visible.filter((w) => w.on).length;
  useEffect(() => {
    const box = boxRef.current;
    if (!box || box.scrollHeight <= box.clientHeight) return;
    const el = box.querySelector<HTMLElement>(`[data-i="${Math.max(0, onCount - 1)}"]`);
    if (el) box.scrollTop = Math.max(0, el.offsetTop - box.clientHeight / 2);
  }, [onCount, visible.length]);

  const lang = showUser ? (language ?? undefined) : (turn?.message?.language ?? language ?? undefined);

  return (
    <p
      ref={boxRef}
      className={`vc-caption${showUser ? " is-user" : ""}${visible.length === 0 ? " is-empty" : ""}`}
      lang={lang}
      data-final={showUser ? userFinal || undefined : undefined}
    >
      <span className="vc-words">
        {visible.length === 0 && !cutOff ? (
          <span className="vc-caption-hint">{showUser ? "…" : hint}</span>
        ) : (
          <>
            {visible.map((w, i) => (
              <span key={i} data-i={i} className={`vc-w${w.on ? " on" : ""}${w.pending ? " pending" : ""}`}>
                {w.text}{" "}
              </span>
            ))}
            {cutOff && <span className="vc-w cut">…</span>}
          </>
        )}
      </span>
    </p>
  );
}
