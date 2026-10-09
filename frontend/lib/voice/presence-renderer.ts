/**
 * The voice presence field (docs/DESIGN.md §3.8, v2), ported from docs/prototypes/voice-presence.html.
 *
 * One WebGL2 draw call of ~10,000 instanced strokes (capsules with an antialiased edge); all motion is computed in the
 * vertex shader from a handful of uniforms per frame. Three kinds of instances:
 *   0  ambient strokes across the stage; near the pointer they swirl into small rings and settle when it stops
 *   1  presence rings around the centre: the voice field (agent ripples travel out, user ripples travel in)
 *   2  the core: a small glowing heart that breathes with the agent
 *
 * Without WebGL2 a Canvas 2D fallback draws the rings with the same signals. The renderer owns its canvas: if the
 * shaders fail after a WebGL context was obtained, the canvas can't give a 2D context any more, so a fresh one is made.
 */

import { clamp } from "./analysis";

export interface FrameState {
  /** Seconds on a monotonic clock; waves compare their start times against it. */
  clock: number;
  /** Seconds of motion, frozen under reduced motion. */
  time: number;
  rot: number;
  /** 0 under reduced motion, 1 otherwise. */
  motion: number;
  agent: number;
  user: number;
  agentTone: number;
  think: number;
  duck: number;
  dark: boolean;
  /** Ripple start times (seconds, `clock` base). */
  waveOut: readonly number[];
  waveIn: readonly number[];
  pointer: { x: number; y: number; on: number };
  /** Background colour, 0…1 per channel. */
  background: readonly [number, number, number];
}

export interface Layout {
  /** Canvas size in device pixels. */
  width: number;
  height: number;
  /** Ring centre in device pixels, from the top-left. */
  cx: number;
  cy: number;
  /** Ring radius unit in device pixels (the outer ring is ~0.46 of it). */
  scale: number;
}

const VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 aInst;   // kind, a, b, seed
uniform vec2 uRes, uCenter, uPointer;
uniform float uScale, uTime, uClock, uRot, uMotion, uAgent, uUser, uAgentTone, uThink, uDuck, uDark, uPointerOn;
uniform vec4 uWaveOut, uWaveIn;
out vec2 vLocal;
out float vAspect, vKind;
out vec4 vColor;

const float TAU = 6.2831853;
const float QUAD = 1.35;  // overdraw so the capsule edge antialiases inside the quad

vec3 spectrum(float p) {
  p = fract(p) * 5.0;
  vec3 c0 = vec3(0.26, 0.52, 0.96), c1 = vec3(0.47, 0.37, 0.95), c2 = vec3(0.80, 0.34, 0.80),
       c3 = vec3(0.95, 0.41, 0.36), c4 = vec3(0.98, 0.66, 0.25);
  if (p < 1.0) return mix(c0, c1, p);
  if (p < 2.0) return mix(c1, c2, p - 1.0);
  if (p < 3.0) return mix(c2, c3, p - 2.0);
  if (p < 4.0) return mix(c3, c4, p - 3.0);
  return mix(c4, c0, p - 4.0);
}
vec3 cool(float p) { return mix(vec3(0.22, 0.47, 0.98), vec3(0.56, 0.39, 0.98), 0.5 + 0.5 * sin(p * TAU)); }
vec3 warm(float p) { return mix(vec3(0.96, 0.42, 0.33), vec3(0.99, 0.66, 0.26), 0.5 + 0.5 * sin(p * TAU)); }

float wave(float r, float start, float r0, float speed, float dir) {
  float age = uClock - start;
  if (age < 0.0 || age > 1.3) return 0.0;
  return exp(-pow((r - (r0 + dir * age * speed)) / 0.035, 2.0)) * (1.0 - age / 1.3);
}
float waves(float r, vec4 s, float r0, float speed, float dir) {
  return wave(r, s.x, r0, speed, dir) + wave(r, s.y, r0, speed, dir) + wave(r, s.z, r0, speed, dir) + wave(r, s.w, r0, speed, dir);
}

void main() {
  float kind = aInst.x, seed = aInst.w, t = uTime;
  float energy = clamp(uAgent + uUser, 0.0, 1.0);
  float fade = mix(1.0, 0.85, uDark);
  vec2 pos;
  float ang, len, wid, alpha;
  vec3 col;

  if (kind < 0.5) {
    vec2 p = aInst.yz + uMotion * 0.005 * vec2(sin(t * 0.31 + seed * TAU), cos(t * 0.27 + seed * 9.1));
    vec2 toP = p - (uPointer - uCenter) / uScale;
    float dp = length(toP);
    float near = uPointerOn * exp(-pow(dp / 0.17, 2.0));
    float pa = atan(toP.y, toP.x);
    ang = mix(seed * TAU, pa, near);
    pos = p + vec2(cos(pa), sin(pa)) * near * 0.01 * sin(dp * 55.0 - uClock * 5.0) * uMotion;
    len = 0.0032 + 0.0065 * near;
    wid = 0.0011 + 0.0004 * near;
    col = mix(mix(vec3(0.6), vec3(0.5), uDark), spectrum(pa / TAU + 0.55), near);
    alpha = (0.18 + 0.62 * near) * fade;
  } else if (kind < 1.5) {
    float r0 = aInst.y, th = aInst.z + uRot;
    float Rb = 0.25 + 0.012 * sin(t * 1.26) * uMotion + 0.06 * uAgent - 0.05 * uUser * (1.0 - uAgent) - 0.04 * uDuck;
    float W = 0.028 + 0.07 * uAgent + 0.032 * uUser;
    float wo = waves(r0, uWaveOut, 0.13, 0.42, 1.0) * uMotion;
    float wi = waves(r0, uWaveIn, 0.5, 0.34, -1.0) * uMotion;
    float r = r0 + 0.024 * wo - 0.02 * wi + 0.004 * sin(th * 9.0 + t * 1.1 + seed * TAU) * uMotion;
    float band = exp(-pow((r0 - Rb) / W, 2.0));
    alpha = (0.07 + band * (0.3 + 0.42 * energy) + 0.45 * (wo + wi)) * (1.0 - 0.45 * uDuck) * fade;
    len = 0.0045 + band * (0.008 + 0.016 * uAgent + 0.01 * uUser) + 0.01 * (wo + wi);
    wid = 0.0016 + 0.0008 * band;
    ang = th + 0.7 * uThink;
    float p = th / TAU;
    col = spectrum(p);
    col = mix(col, cool(p), clamp(uAgent * 1.3, 0.0, 1.0) * 0.6);
    col = mix(col, warm(p), clamp(uUser * 1.3, 0.0, 1.0) * 0.75);
    col *= 0.9 + 0.25 * uAgentTone * uAgent;
    pos = vec2(cos(th), sin(th)) * r;
  } else {
    float cr = 0.07 + 0.006 * sin(t * 1.26) * uMotion + 0.04 * uAgent - 0.012 * uUser - 0.02 * uDuck;
    pos = vec2(0.0);
    ang = 0.0;
    len = cr;
    wid = cr;
    col = mix(vec3(0.33, 0.47, 0.98), vec3(0.58, 0.40, 0.98), 0.5 + 0.5 * sin(t * 0.4));
    col = mix(col, vec3(0.98, 0.55, 0.36), clamp(uUser * 1.2, 0.0, 1.0) * 0.35);
    alpha = (0.55 + 0.4 * uAgent + 0.2 * uThink) * (1.0 - 0.45 * uDuck);
  }

  vec2 dir = vec2(cos(ang), sin(ang)), nrm = vec2(-dir.y, dir.x);
  vec2 world = pos + dir * aCorner.x * len * QUAD + nrm * aCorner.y * wid * QUAD;
  gl_Position = vec4((uCenter + world * uScale) / uRes * 2.0 - 1.0, 0.0, 1.0);
  vLocal = vec2(aCorner.x * len, aCorner.y * wid) * QUAD / wid;
  vAspect = len / wid;
  vColor = vec4(col, clamp(alpha, 0.0, 1.0));
  vKind = kind;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec2 vLocal;
in float vAspect, vKind;
in vec4 vColor;
out vec4 outColor;
void main() {
  if (vKind > 1.5) {
    float d = length(vLocal);
    float a = pow(clamp(1.0 - d, 0.0, 1.0), 1.8);
    outColor = vec4(mix(vColor.rgb, vec3(1.0), 0.35 * (1.0 - d) * (1.0 - d)), vColor.a * a);
    return;
  }
  float e = max(vAspect - 1.0, 0.0);
  float d = length(vec2(vLocal.x - clamp(vLocal.x, -e, e), vLocal.y));  // capsule SDF, 1 = edge
  float aa = fwidth(d) + 1e-4;
  outColor = vec4(vColor.rgb, vColor.a * (1.0 - smoothstep(1.0 - aa, 1.0 + aa, d)));
}`;

const UNIFORMS = [
  "uRes",
  "uCenter",
  "uPointer",
  "uScale",
  "uTime",
  "uClock",
  "uRot",
  "uMotion",
  "uAgent",
  "uUser",
  "uAgentTone",
  "uThink",
  "uDuck",
  "uDark",
  "uPointerOn",
  "uWaveOut",
  "uWaveIn",
] as const;

const TAU = Math.PI * 2;

/** Instance data, [kind, a, b, seed] per stroke. Built once per page (it's random but static). */
let instances: Float32Array | null = null;

function buildInstances(): Float32Array {
  const out: number[] = [];
  const step = 0.05;
  // The ambient grid is bigger than the prototype's: the stage here is whatever room the page leaves, not a screen.
  for (let y = -1.7; y <= 1.7; y += step) {
    for (let x = -2.2; x <= 2.2; x += step) {
      const px = x + (Math.random() - 0.5) * step * 0.9;
      const py = y + (Math.random() - 0.5) * step * 0.9;
      if (Math.hypot(px, py) < 0.52) continue; // keep the presence area for the rings
      out.push(0, px, py, Math.random());
    }
  }
  for (let r = 0.11; r <= 0.46; r += 0.0125) {
    const n = Math.floor((TAU * r) / 0.0115);
    const off = Math.random() * TAU;
    for (let i = 0; i < n; i++) {
      out.push(1, r + (Math.random() - 0.5) * 0.003, off + (i / n) * TAU + (Math.random() - 0.5) * 0.01, Math.random());
    }
  }
  out.push(2, 0, 0, 0);
  return new Float32Array(out);
}

function getInstances(): Float32Array {
  return (instances ??= buildInstances());
}

export class PresenceRenderer {
  canvas: HTMLCanvasElement;
  readonly host: HTMLElement;
  private gl: WebGL2RenderingContext | null = null;
  private ctx2d: CanvasRenderingContext2D | null = null;
  private loc: Record<string, WebGLUniformLocation | null> = {};
  private count = 0;
  private lost = false;
  private onLost = (e: Event) => {
    e.preventDefault();
    this.lost = true;
  };
  private onRestored = () => {
    this.lost = false;
    this.initGL();
  };

  constructor(host: HTMLElement) {
    this.host = host;
    this.count = getInstances().length / 4;
    this.canvas = this.makeCanvas();
    host.appendChild(this.canvas);
    const gl = this.canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false, alpha: false });
    if (gl) {
      this.gl = gl;
      if (!this.initGL()) {
        // A canvas that already has a WebGL context can't give a 2D one: swap in a fresh canvas.
        this.gl = null;
        this.swapCanvas();
      }
    }
    if (!this.gl) this.ctx2d = this.canvas.getContext("2d", { alpha: false });
    this.canvas.addEventListener("webglcontextlost", this.onLost);
    this.canvas.addEventListener("webglcontextrestored", this.onRestored);
  }

  get webgl(): boolean {
    return this.gl !== null;
  }

  private makeCanvas(): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-hidden", "true");
    canvas.className = "vc-canvas";
    return canvas;
  }

  private swapCanvas() {
    const fresh = this.makeCanvas();
    this.canvas.replaceWith(fresh);
    this.canvas = fresh;
  }

  private initGL(): boolean {
    const gl = this.gl;
    if (!gl) return false;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type);
      if (!s) throw new Error("createShader failed");
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
      return s;
    };
    try {
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? "link error");
      gl.useProgram(prog);
      for (const n of UNIFORMS) this.loc[n] = gl.getUniformLocation(prog, n);
    } catch (e) {
      console.warn("WebGL2 particles unavailable, using the 2D fallback:", e);
      return false;
    }
    const data = getInstances();
    gl.bindVertexArray(gl.createVertexArray());
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enable(gl.BLEND);
    return true;
  }

  resize(cssWidth: number, cssHeight: number, dpr: number): { width: number; height: number } {
    const width = Math.max(1, Math.round(cssWidth * dpr));
    const height = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    return { width, height };
  }

  render(s: FrameState, l: Layout): void {
    if (this.lost) return;
    const { gl } = this;
    const [r, g, b] = s.background;
    if (gl) {
      gl.viewport(0, 0, l.width, l.height);
      if (s.dark) gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      else gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(r, g, b, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const u = this.loc;
      // WebGL's origin is the bottom-left corner.
      gl.uniform2f(u.uRes, l.width, l.height);
      gl.uniform2f(u.uCenter, l.cx, l.height - l.cy);
      gl.uniform2f(u.uPointer, s.pointer.x, l.height - s.pointer.y);
      gl.uniform1f(u.uScale, l.scale);
      gl.uniform1f(u.uTime, s.time % 1000);
      gl.uniform1f(u.uClock, s.clock % 1000);
      gl.uniform1f(u.uRot, s.rot % TAU);
      gl.uniform1f(u.uMotion, s.motion);
      gl.uniform1f(u.uAgent, s.agent);
      gl.uniform1f(u.uUser, s.user);
      gl.uniform1f(u.uAgentTone, s.agentTone);
      gl.uniform1f(u.uThink, s.think);
      gl.uniform1f(u.uDuck, s.duck);
      gl.uniform1f(u.uDark, s.dark ? 1 : 0);
      gl.uniform1f(u.uPointerOn, s.pointer.on * s.motion);
      gl.uniform4fv(u.uWaveOut, s.waveOut.map((x) => x % 1000));
      gl.uniform4fv(u.uWaveIn, s.waveIn.map((x) => x % 1000));
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
      return;
    }
    const c = this.ctx2d;
    if (!c) return;
    // 2D fallback: the rings only, same signals.
    c.fillStyle = `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`;
    c.fillRect(0, 0, l.width, l.height);
    const data = getInstances();
    const Rb = 0.25 + 0.085 * s.agent - 0.06 * s.user - 0.04 * s.duck;
    const W = 0.028 + 0.11 * s.agent + 0.045 * s.user;
    c.lineCap = "round";
    c.lineWidth = Math.max(1, l.scale * 0.003);
    const hue = s.user > s.agent ? 20 : 225;
    for (let i = 0; i < this.count; i += 3) {
      if (data[i * 4] !== 1) continue;
      const r0 = data[i * 4 + 1];
      const th = data[i * 4 + 2] + s.rot;
      const band = Math.exp(-(((r0 - Rb) / W) ** 2));
      const a = (0.07 + band * (0.32 + 0.6 * clamp(s.agent + s.user, 0, 1))) * (1 - 0.45 * s.duck);
      c.strokeStyle = `hsla(${hue}, 80%, ${s.dark ? 66 : 58}%, ${a})`;
      const x = l.cx + Math.cos(th) * r0 * l.scale;
      const y = l.cy - Math.sin(th) * r0 * l.scale;
      const len = l.scale * (0.004 + band * 0.012);
      c.beginPath();
      c.moveTo(x - Math.cos(th) * len, y + Math.sin(th) * len);
      c.lineTo(x + Math.cos(th) * len, y - Math.sin(th) * len);
      c.stroke();
    }
  }

  dispose(): void {
    this.canvas.removeEventListener("webglcontextlost", this.onLost);
    this.canvas.removeEventListener("webglcontextrestored", this.onRestored);
    this.gl?.getExtension("WEBGL_lose_context")?.loseContext();
    this.canvas.remove();
    this.gl = null;
    this.ctx2d = null;
  }
}
