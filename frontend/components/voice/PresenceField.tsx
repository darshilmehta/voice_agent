"use client";

/**
 * The voice presence field (docs/DESIGN.md §3.8): thousands of short strokes that organise into rings around a small
 * breathing core. The agent widens and cools the rings and sends waves outward; the user tightens and warms them and
 * pulls waves inward; thinking swirls them; a barge-in duck dims the field as the audio ducks. Idle, it breathes.
 *
 * The canvas fills the stage (ambient strokes swirl around the pointer); the rings are centred on `anchorRef`, the
 * empty slot the layout keeps clear of the captions. Both voices are read from AnalyserNodes every frame, in the
 * browser, with no round trip. Light theme draws with normal blending, dark theme additively.
 */

import { useEffect, useRef, type RefObject } from "react";

import { clamp, createMeter, measure } from "@/lib/voice/analysis";
import { PresenceRenderer, type FrameState, type Layout } from "@/lib/voice/presence-renderer";
import type { VoiceSession } from "@/lib/voice/session";

/** Ring radius unit relative to the slot: the outer ring's diameter is ~74% of the slot's short side. */
const SCALE = 0.8;

function parseColor(value: string): [number, number, number] {
  const hex = value.trim().replace("#", "");
  const full = hex.length === 3 ? hex.replace(/./g, (c) => c + c) : hex;
  if (/^[0-9a-f]{6}$/i.test(full)) {
    return [parseInt(full.slice(0, 2), 16) / 255, parseInt(full.slice(2, 4), 16) / 255, parseInt(full.slice(4, 6), 16) / 255];
  }
  return [0.97, 0.97, 0.96];
}

export function PresenceField({ session, anchorRef }: { session: VoiceSession; anchorRef: RefObject<HTMLElement | null> }) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const stage = host.parentElement ?? host;
    const renderer = new PresenceRenderer(host);
    const dark = window.matchMedia("(prefers-color-scheme: dark)");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");

    let layout: Layout = { width: 1, height: 1, cx: 0, cy: 0, scale: 1 };
    let dpr = 1;
    let background = parseColor(getComputedStyle(document.documentElement).getPropertyValue("--bg"));
    const refreshBackground = () => {
      background = parseColor(getComputedStyle(document.documentElement).getPropertyValue("--bg"));
    };
    const relayout = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = host.getBoundingClientRect();
      const { width, height } = renderer.resize(rect.width, rect.height, dpr);
      const anchor = anchorRef.current?.getBoundingClientRect();
      const slot = anchor ?? rect;
      const cx = (slot.left - rect.left + slot.width / 2) * dpr;
      const cy = (slot.top - rect.top + slot.height / 2) * dpr;
      layout = { width, height, cx, cy, scale: Math.max(40, Math.min(slot.width, slot.height) * SCALE * dpr) };
    };
    relayout();
    const ro = new ResizeObserver(relayout);
    ro.observe(host);
    if (anchorRef.current) ro.observe(anchorRef.current);
    const onScheme = () => refreshBackground();
    dark.addEventListener("change", onScheme);

    // The pointer: ambient strokes swirl into small rings around it, then settle when it stops moving.
    const pointer = { x: -1e4, y: -1e4, movedAt: -10, on: 0 };
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      const rect = host.getBoundingClientRect();
      pointer.x = (e.clientX - rect.left) * dpr;
      pointer.y = (e.clientY - rect.top) * dpr;
      pointer.movedAt = performance.now() / 1000;
    };
    stage.addEventListener("pointermove", onMove);

    const agent = createMeter();
    const user = createMeter();
    // Development aid: the live meters (levels and ripples) for debugging and browser tests.
    if (process.env.NODE_ENV !== "production") (window as unknown as { __presence?: unknown }).__presence = { agent, user, renderer };
    let last = performance.now() / 1000;
    let motionTime = 0;
    let rot = 0;
    let think = 0;
    let duck = 0;
    let cutAt = -10;
    let wasCut = false;
    let raf = 0;
    let visible = true;
    const io = new IntersectionObserver(([entry]) => (visible = entry?.isIntersecting ?? true));
    io.observe(host);

    let skip = false;
    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (!visible) return;
      // Nobody is talking and the field is only breathing: half the frame rate is plenty (and kinder to laptops).
      if (agent.env < 0.01 && user.env < 0.01 && pointer.on < 0.02 && session.getSnapshot().phase === "idle") {
        skip = !skip;
        if (skip) return;
      }
      const t = performance.now() / 1000;
      const dt = Math.min(0.1, t - last);
      last = t;
      const snap = session.getSnapshot();
      const motion = reduce.matches ? 0 : 1;
      const sr = session.sampleRate;
      measure(session.agentAnalyser, agent, dt, t, sr, motion > 0);
      measure(session.userAnalyser, user, dt, t, sr, motion > 0);

      const thinking = snap.phase === "live" && snap.serverState === "thinking" && !snap.audible;
      think += ((thinking ? 1 : 0) - think) * (1 - Math.exp(-dt / 0.35));
      // Ducked while the server decides; a short hold after the agent is cut off so the dip is seen.
      const cut = snap.turn?.cut === true;
      if (cut && !wasCut) cutAt = t;
      wasCut = cut;
      const duckTarget = snap.ducked || t - cutAt < 0.25 ? 1 : 0;
      duck += (duckTarget - duck) * (1 - Math.exp(-dt / 0.08));
      motionTime += dt * motion;
      rot += dt * (0.015 + 0.55 * think) * motion;
      pointer.on += ((t - pointer.movedAt < 1.2 ? 1 : 0) - pointer.on) * (1 - Math.exp(-dt / 0.4));

      const state: FrameState = {
        clock: t,
        time: motionTime,
        rot,
        motion,
        agent: clamp(agent.env, 0, 1),
        user: clamp(user.env, 0, 1),
        agentTone: agent.tone,
        think,
        duck,
        dark: dark.matches,
        waveOut: agent.ripples,
        waveIn: user.ripples,
        pointer,
        background,
      };
      renderer.render(state, layout);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      dark.removeEventListener("change", onScheme);
      stage.removeEventListener("pointermove", onMove);
      renderer.dispose();
    };
  }, [session, anchorRef]);

  return <div ref={hostRef} className="vc-field" aria-hidden="true" />;
}
