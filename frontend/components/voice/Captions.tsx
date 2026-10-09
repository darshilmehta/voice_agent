"use client";

/**
 * Live captions under the presence field. The agent's words light up one by one as they are spoken, following the
 * audio that is actually playing (lib/voice/captions.ts); the user's words appear in the warm colour while they
 * speak, then as the saved transcript. After an interruption the caption keeps only what was heard, then "…".
 *
 * A turn that searches the web first says a short filler ("Let me look that up."): it shows as a light aside above the
 * answer while it is spoken and while the search runs, and gives way to the answer's captions once that has started
 * (the filler is not part of the answer: the transcript doesn't have it).
 *
 * Captions are visual, with no live region: the spoken answer is already audio, and state changes are announced by
 * the state label. The whole conversation is in the transcript.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { agentCaptionWords, plainSpeech, spokenCount, type CaptionWord } from "@/lib/voice/captions";
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
  // The agent's words, with their timing (lib/voice/captions.ts). `trustServer`: interrupted with no answer audio ever
  // announced, so the server's "what was heard" is all there is.
  const { words: agentWords, trustServer } = useMemo<{ words: CaptionWord[]; trustServer: boolean }>(
    () => (turn ? agentCaptionWords(turn) : { words: [], trustServer: false }),
    [turn],
  );

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

  const words: { text: string; on: boolean; pending: boolean; aside: boolean }[] = showUser
    ? userWords.map((text) => ({ text, on: true, pending: false, aside: false }))
    : agentWords.map((w, i) => ({
        text: w.text,
        on: trustServer || settled || i < spoken,
        pending: w.chunk < 0 && !settled,
        aside: w.aside === true,
      }));
  // After an interruption only what was heard stays, followed by "…".
  const cutOff = showAgent && interrupted;
  const heardWords = cutOff ? words.filter((w) => w.on) : words;
  // The filler is an aside: once it has been said and the answer has begun (its words are there, or were heard), it
  // gives way to the answer.
  const aside = heardWords.filter((w) => w.aside);
  const answer = heardWords.filter((w) => !w.aside);
  const hideAside = aside.length > 0 && aside.every((w) => w.on) && answer.length > 0;
  const visible = hideAside ? answer : heardWords;

  // Keep the word being spoken in view when the answer is longer than the caption area.
  const onCount = visible.filter((w) => w.on).length;
  useEffect(() => {
    const box = boxRef.current;
    if (!box || box.scrollHeight <= box.clientHeight) return;
    const el = box.querySelector<HTMLElement>(`[data-i="${Math.max(0, onCount - 1)}"]`);
    if (el) box.scrollTop = Math.max(0, el.offsetTop - box.clientHeight / 2);
  }, [onCount, visible.length]);

  const lang = showUser ? (language ?? undefined) : (turn?.message?.language ?? language ?? undefined);
  const indexed = visible.map((w, i) => ({ w, i }));
  const asideShown = indexed.filter((x) => x.w.aside);
  const answerShown = indexed.filter((x) => !x.w.aside);

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
            {asideShown.length > 0 && (
              <span className="vc-aside">
                {asideShown.map(({ w, i }) => (
                  <span key={i} data-i={i} className={`vc-w${w.on ? " on" : ""}`}>
                    {w.text}{" "}
                  </span>
                ))}
              </span>
            )}
            {answerShown.map(({ w, i }) => (
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
