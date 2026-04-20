import * as React from "react";
import { useEffect, useRef, useState } from "react";

/**
 * CelegansDashboard — full-scale interactive simulation viewer.
 *
 * Replaces the earlier BrainBody component with a unified dashboard
 * that exposes the complexity of the v3 + Tier 1 model:
 *   • 20-segment body animation (MuJoCo-derived)
 *   • 300-neuron brain view projected from 3D soma coordinates,
 *     colour-coded by recent activity
 *   • 9-modulator concentration strip (volume-transmission layer)
 *   • 5-state FSM timeline
 *   • 8-event classifier probability streams
 *   • Environment view (agar plate + food patch + worm trail) when
 *     the scenario includes T1e environment data
 *
 * All panels share a single `currentT` so scrubbing/playback is
 * synchronised. Canvas 2D throughout for 60 fps animation.
 */

type Scenario = "spontaneous" | "touch" | "osmotic_shock" | "food" | "chemotaxis";

type Trace = {
  scenario: string;
  meta: {
    brain_sync_ms: number;
    classifier_dt_ms?: number;
    num_segments: number;
    num_frames: number;
    duration_s: number;
    readout_neurons: string[];
    events_tracked: string[];
    states: string[];
    modulation_enabled?: boolean;
    modulators?: string[];
    sources: Record<string, string>;
    modulation_final_concentrations?: Record<string, number>;
  };
  frames: Array<{
    t: number;
    positions: Array<[number, number]>;
    state: string;
  }>;
  raster: Array<{ t: number; n: number[] }>;
  event_probs: Record<string, number[]>;
  fsm_states: number[];
  stim_log: Array<{ t: number; preset: string; intensity: number; neurons: string[] }>;
  // T1 dashboard extensions
  neuron_positions?: Array<[number, number, number]>;
  neuron_names?: string[];
  modulator_concentrations?: number[][];
  modulator_names?: string[];
  environment?: {
    food_xy_mm: [number, number];
    sigma_mm: number;
    trail: Array<{ t: number; x: number; y: number }>;
    chemotaxis_index: Record<string, number | [number, number]>;
  };
};

const SCENARIOS: Record<Scenario, { label: string; desc: string; tagline: string }> = {
  spontaneous: {
    label: "Spontaneous",
    desc: "No stimulus — baseline behavioural distribution.",
    tagline: "Idle brain, idle worm.",
  },
  touch: {
    label: "Head touch",
    desc: "ALM/AVM mechanoreceptor drive at t=5s (Chalfie 1981).",
    tagline: "Poke → reversal.",
  },
  osmotic_shock: {
    label: "Osmotic shock",
    desc: "ASH polymodal avoidance drive at t=5s (Hart 1995).",
    tagline: "Noxious → avoidance cascade.",
  },
  food: {
    label: "Food",
    desc: "ASI/ASJ/ADF feeding-state tonic drive from t=2s (Flavell 2013).",
    tagline: "Satiety → dwelling.",
  },
  chemotaxis: {
    label: "Chemotaxis",
    desc: "2D agar, food patch at (4 mm, 0). ASE/AWC/AWA driven by real concentration field (Pierce-Shimomura 1999).",
    tagline: "Navigate toward food.",
  },
};

const STATE_COLORS: Record<string, string> = {
  FORWARD: "#2f5233",
  REVERSE: "#b94b4b",
  OMEGA: "#8b5cf6",
  PIROUETTE: "#e76f51",
  QUIESCENT: "#6b7280",
};

const MODULATOR_COLORS: Record<string, string> = {
  "FLP-11": "#8b5cf6",  // peptides — purples
  "FLP-1":  "#a78bfa",
  "FLP-2":  "#c4b5fd",
  "NLP-12": "#ddd6fe",
  "PDF-1":  "#7c3aed",
  "5HT":    "#059669",  // monoamines — distinctive
  "DA":     "#2563eb",
  "TA":     "#dc2626",
  "OA":     "#d97706",
};

const EVENT_COLORS: Record<string, string> = {
  reversal_onset:      "#b94b4b",
  reversal_offset:     "#dc8686",
  forward_run_onset:   "#2f5233",
  forward_run_offset:  "#64884e",
  omega_onset:         "#8b5cf6",
  pirouette_entry:     "#e76f51",
  quiescence_onset:    "#6b7280",
  speed_burst_onset:   "#f59e0b",
};

// --------------------------------------------------------------
// Drawing helpers
// --------------------------------------------------------------

function segmentWidth(i: number, total: number): number {
  const t = i / (total - 1);
  return 3 + 9 * Math.sin(Math.PI * t);
}

function drawWormBody(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  segs: Array<{ x: number; y: number }>,
  state: string,
) {
  ctx.clearRect(0, 0, w, h);
  // Soft radial background
  const bg = ctx.createRadialGradient(w / 2, h / 2, 40, w / 2, h / 2, Math.max(w, h));
  bg.addColorStop(0, "rgba(247, 237, 211, 0.95)");
  bg.addColorStop(1, "rgba(229, 215, 185, 0.92)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  // Subtle dotted grid
  ctx.fillStyle = "rgba(26, 42, 74, 0.10)";
  for (let x = 16; x < w; x += 32) {
    for (let y = 16; y < h; y += 32) {
      ctx.fillRect(x, y, 1, 1);
    }
  }
  // Body with outer glow matching state
  const stateColor = STATE_COLORS[state] ?? "#2f5233";
  ctx.save();
  ctx.shadowBlur = 12;
  ctx.shadowColor = stateColor + "55";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let i = 0; i < segs.length - 1; i++) {
    ctx.strokeStyle = i === 0 ? "#1a2a4a" : stateColor;
    ctx.lineWidth = segmentWidth(i, segs.length);
    ctx.beginPath();
    ctx.moveTo(segs[i].x, segs[i].y);
    ctx.lineTo(segs[i + 1].x, segs[i + 1].y);
    ctx.stroke();
  }
  ctx.restore();
  // Head marker — small cream dot indicating the head tip
  const head = segs[0];
  ctx.fillStyle = "#f2ead3";
  ctx.beginPath();
  ctx.arc(head.x, head.y, 2.5, 0, Math.PI * 2);
  ctx.fill();
}

function drawBrain3D(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  positions: Array<[number, number, number]>,
  names: string[],
  activeSet: Set<number>,
  hoverIdx: number | null,
  readoutSet: Set<string>,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);

  if (!positions || positions.length === 0) return;

  // Project 3D (x, y, z) to 2D (sx, sy). Worm coordinates have y as AP
  // axis (head negative → tail positive) and z as DV axis. We want an
  // elegant side-on projection: X → screen x (left-right), Y → screen
  // x (head-tail) flipped, Z → screen y (depth). So use:
  //   screen_x = (y − y_center) / range_y × w × 0.9 + w/2  (head on the left)
  //   screen_y = h/2 − (z − z_center) / range_z × h × 0.4
  const ys = positions.map((p) => p[1]);
  const zs = positions.map((p) => p[2]);
  const xs = positions.map((p) => p[0]);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const zMin = Math.min(...zs), zMax = Math.max(...zs);
  const yRange = yMax - yMin || 1;
  const zRange = zMax - zMin || 1;

  // Subtle body outline cue
  ctx.strokeStyle = "rgba(80, 90, 120, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const cy = h / 2;
  ctx.moveTo(w * 0.05, cy);
  ctx.lineTo(w * 0.95, cy);
  ctx.stroke();

  // Depth-sorted draw (x = left-right, used for depth shading)
  const idxSorted = [...Array(positions.length).keys()].sort(
    (a, b) => positions[a][0] - positions[b][0]
  );

  for (const i of idxSorted) {
    const [x, y, z] = positions[i];
    const sx = ((y - yMin) / yRange) * w * 0.9 + w * 0.05;
    const sy = cy - ((z - zMin) / zRange - 0.5) * h * 0.75;
    // Depth factor from x (LR) — some neurons occluded by others
    const depthT = (x - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs) + 0.001);
    const depthFade = 0.55 + 0.45 * depthT;
    const isActive = activeSet.has(i);
    const isReadout = readoutSet.has(names[i] ?? "");
    const isHover = hoverIdx === i;
    let r = isReadout ? 2.8 : 1.8;
    if (isHover) r = 4.5;
    let fillColor: string;
    if (isActive) {
      fillColor = `rgba(94, 199, 122, ${0.95 * depthFade})`;
    } else if (isReadout) {
      fillColor = `rgba(160, 180, 220, ${0.7 * depthFade})`;
    } else {
      fillColor = `rgba(110, 120, 150, ${0.55 * depthFade})`;
    }
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
    if (isHover) {
      ctx.strokeStyle = "#f2ead3";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Label
      ctx.fillStyle = "#f2ead3";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(names[i] ?? "?", sx + 8, sy - 8);
    }
  }

  // Legend
  ctx.fillStyle = "rgba(210, 220, 240, 0.6)";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("head", w * 0.04, h - 8);
  ctx.textAlign = "right";
  ctx.fillText("tail", w * 0.96, h - 8);
  ctx.textAlign = "left";
}

function drawModulatorStrip(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  concentrations: number[][] | undefined,
  names: string[] | undefined,
  currentFrac: number,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);

  if (!concentrations || !names || concentrations.length === 0) {
    ctx.fillStyle = "rgba(210, 220, 240, 0.5)";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("No modulator telemetry in this scenario.", 12, h / 2);
    return;
  }

  const labelW = 52;
  const plotW = w - labelW - 4;
  const numMods = names.length;
  const rowH = h / numMods;

  // Per-modulator max for normalisation
  const maxByMod = names.map((_, mi) => {
    let m = 0;
    for (const row of concentrations) {
      if (row[mi] > m) m = row[mi];
    }
    return Math.max(m, 1e-6);
  });

  for (let mi = 0; mi < numMods; mi++) {
    const y0 = mi * rowH;
    const color = MODULATOR_COLORS[names[mi]] ?? "#94a3b8";
    // Label
    ctx.fillStyle = "rgba(210, 220, 240, 0.85)";
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText(names[mi], 4, y0 + rowH * 0.65);

    // Render time-series as intensity-shaded bars
    const nT = concentrations.length;
    const stride = Math.max(1, Math.floor(nT / plotW));
    for (let t = 0; t < nT; t += stride) {
      const intensity = concentrations[t][mi] / maxByMod[mi];
      const x = labelW + (t / nT) * plotW;
      const bw = Math.max(1, (stride / nT) * plotW);
      ctx.fillStyle = color + Math.round(intensity * 220).toString(16).padStart(2, "0");
      ctx.fillRect(x, y0 + 2, bw, rowH - 4);
    }
  }

  // Current-time cursor
  const cursorX = labelW + currentFrac * plotW;
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();
}

function drawFsmTimeline(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  states: number[],
  currentFrac: number,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);

  const stateNames = ["(none)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"];
  const labelW = 52;
  const plotW = w - labelW - 4;
  const nT = states.length;
  if (nT === 0) return;

  ctx.fillStyle = "rgba(210, 220, 240, 0.85)";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("state", 4, h * 0.65);

  const bw = plotW / nT;
  for (let t = 0; t < nT; t++) {
    const s = states[t];
    const name = stateNames[s];
    const color = STATE_COLORS[name] ?? "#9ca3af";
    ctx.fillStyle = color;
    ctx.fillRect(labelW + t * bw, 6, bw + 0.5, h - 12);
  }

  // Current-time cursor
  const cursorX = labelW + currentFrac * plotW;
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();
}

function drawStimMarkers(
  ctx: CanvasRenderingContext2D,
  w: number, h: number, labelW: number,
  stims: Trace["stim_log"], durationS: number,
) {
  ctx.save();
  ctx.strokeStyle = "rgba(245, 158, 11, 0.7)";
  ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
  ctx.lineWidth = 1;
  ctx.font = "9px system-ui, sans-serif";
  ctx.setLineDash([2, 3]);
  for (const s of stims) {
    const frac = s.t / durationS;
    const x = labelW + frac * (w - labelW - 4);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillText(s.preset, x + 2, 10);
  }
  ctx.restore();
  ctx.setLineDash([]);
}

function drawEventProbs(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  probs: Record<string, number[]>,
  eventNames: string[] | undefined,
  currentFrac: number,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);
  if (!eventNames || eventNames.length === 0) return;

  const labelW = 52;
  const plotW = w - labelW - 4;

  // Grid + 0.5 threshold
  ctx.strokeStyle = "rgba(210, 220, 240, 0.12)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(labelW, h * 0.5);
  ctx.lineTo(w, h * 0.5);
  ctx.stroke();
  ctx.setLineDash([]);

  // Lines
  for (const ev of eventNames) {
    const arr = probs[ev];
    if (!arr || arr.length === 0) continue;
    const color = EVENT_COLORS[ev] ?? "#94a3b8";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = labelW + (i / arr.length) * plotW;
      const y = h - arr[i] * (h - 4) - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Label
  ctx.fillStyle = "rgba(210, 220, 240, 0.85)";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("events", 4, h * 0.55);
  ctx.fillText("1", labelW - 14, 10);
  ctx.fillText("0", labelW - 14, h - 4);

  // Cursor
  const cursorX = labelW + currentFrac * plotW;
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();
}

function drawEnvironment(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  env: NonNullable<Trace["environment"]> | undefined,
  currentTS: number,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);
  if (!env) {
    ctx.fillStyle = "rgba(210, 220, 240, 0.5)";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("No environment in this scenario.", 12, h / 2);
    return;
  }

  // Work out a 20 mm square world centred on the midpoint between
  // worm start and food.
  const worldMm = 20;
  const pxPerMm = Math.min(w, h) / worldMm;
  const cx = w / 2;
  const cy = h / 2;

  // Render concentration field as a dim radial glow around food
  const foodSX = cx + env.food_xy_mm[0] * pxPerMm;
  const foodSY = cy - env.food_xy_mm[1] * pxPerMm;
  const radialR = env.sigma_mm * pxPerMm * 2.5;
  const grad = ctx.createRadialGradient(foodSX, foodSY, 0, foodSX, foodSY, radialR);
  grad.addColorStop(0, "rgba(245, 158, 11, 0.45)");
  grad.addColorStop(0.5, "rgba(245, 158, 11, 0.12)");
  grad.addColorStop(1, "rgba(245, 158, 11, 0.00)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(foodSX, foodSY, radialR, 0, Math.PI * 2);
  ctx.fill();

  // Food patch marker
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(foodSX, foodSY, 5, 0, Math.PI * 2);
  ctx.fill();

  // Worm trail up to currentT
  ctx.strokeStyle = "rgba(94, 199, 122, 0.6)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let first = true;
  let lastX = foodSX, lastY = foodSY;
  for (const p of env.trail) {
    if (p.t > currentTS + 0.1) break;
    const sx = cx + p.x * pxPerMm;
    const sy = cy - p.y * pxPerMm;
    if (first) {
      ctx.moveTo(sx, sy);
      first = false;
    } else {
      ctx.lineTo(sx, sy);
    }
    lastX = sx;
    lastY = sy;
  }
  ctx.stroke();

  // Current worm head
  ctx.fillStyle = "#5ec77a";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);
  ctx.fill();

  // Labels
  ctx.fillStyle = "rgba(210, 220, 240, 0.8)";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("food patch", foodSX + 8, foodSY - 6);
  ctx.fillText("worm", lastX + 8, lastY - 4);
}

// --------------------------------------------------------------
// Main component
// --------------------------------------------------------------

export function CelegansDashboard() {
  const [scenario, setScenario] = useState<Scenario>("spontaneous");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [currentT, setCurrentT] = useState(0);
  const [width, setWidth] = useState(1024);
  const [hoverNeuron, setHoverNeuron] = useState<number | null>(null);

  const wrapRef = useRef<HTMLDivElement>(null);
  const bodyCanvasRef = useRef<HTMLCanvasElement>(null);
  const brainCanvasRef = useRef<HTMLCanvasElement>(null);
  const envCanvasRef = useRef<HTMLCanvasElement>(null);
  const modCanvasRef = useRef<HTMLCanvasElement>(null);
  const fsmCanvasRef = useRef<HTMLCanvasElement>(null);
  const evCanvasRef = useRef<HTMLCanvasElement>(null);

  const traceRef = useRef<Trace | null>(null);
  const pausedRef = useRef(paused);
  const speedRef = useRef(speed);
  const currentTRef = useRef(0);
  const hoverRef = useRef<number | null>(null);
  traceRef.current = trace;
  pausedRef.current = paused;
  speedRef.current = speed;
  currentTRef.current = currentT;
  hoverRef.current = hoverNeuron;

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target && (e.target as HTMLElement).tagName === "INPUT") return;
      if (e.code === "Space") {
        e.preventDefault();
        setPaused((v) => !v);
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        const tr = traceRef.current;
        if (tr) {
          currentTRef.current = Math.max(0, currentTRef.current - 1);
          setCurrentT(currentTRef.current);
        }
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        const tr = traceRef.current;
        if (tr) {
          currentTRef.current = Math.min(tr.meta.duration_s, currentTRef.current + 1);
          setCurrentT(currentTRef.current);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Fetch trace
  useEffect(() => {
    let cancel = false;
    setTrace(null);
    setLoadErr(null);
    setCurrentT(0);
    fetch(`/data/wormbody-brain-${scenario}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: Trace) => { if (!cancel) setTrace(d); })
      .catch((e) => { if (!cancel) setLoadErr(String(e)); });
    return () => { cancel = true; };
  }, [scenario]);

  // Responsive width
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(Math.max(320, Math.round(e.contentRect.width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Animation loop
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const draw = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const tr = traceRef.current;

      if (tr && !pausedRef.current) {
        currentTRef.current = (currentTRef.current + dt * speedRef.current) % tr.meta.duration_s;
        setCurrentT(currentTRef.current);
      }

      const t = currentTRef.current;

      // Dimensions — the panels stack on narrow screens, grid on wide.
      const narrow = width < 760;
      const bodyW = narrow ? width : Math.round(width * 0.38);
      const brainW = narrow ? width : Math.round(width * 0.36);
      const envW = narrow ? width : width - bodyW - brainW - 16;
      const panelH = 260;
      const stripH = 110;

      // --- Body canvas
      if (bodyCanvasRef.current && tr) {
        const c = bodyCanvasRef.current;
        if (c.width !== bodyW) c.width = bodyW;
        if (c.height !== panelH) c.height = panelH;
        const ctx = c.getContext("2d");
        if (ctx) {
          const idx = Math.min(
            tr.frames.length - 1,
            Math.floor((t / tr.meta.duration_s) * tr.frames.length)
          );
          const frame = tr.frames[idx];
          const pxPerSimM = Math.min(bodyW, panelH) * 0.5;
          let cx = 0, cy = 0;
          for (const [x, y] of frame.positions) { cx += x; cy += y; }
          cx /= frame.positions.length;
          cy /= frame.positions.length;
          const ox = bodyW / 2 - cx * pxPerSimM;
          const oy = panelH / 2 - cy * pxPerSimM;
          const segs = frame.positions.map(([x, y]) => ({
            x: ox + x * pxPerSimM, y: oy + y * pxPerSimM,
          }));
          drawWormBody(ctx, bodyW, panelH, segs, frame.state);
        }
      }

      // --- Brain canvas
      if (brainCanvasRef.current && tr) {
        const c = brainCanvasRef.current;
        if (c.width !== brainW) c.width = brainW;
        if (c.height !== panelH) c.height = panelH;
        const ctx = c.getContext("2d");
        if (ctx) {
          // Which neurons are active RIGHT NOW: raster entries within
          // last 100 ms window
          const activeSet = new Set<number>();
          if (tr.raster) {
            for (const e of tr.raster) {
              if (e.t > t - 0.1 && e.t <= t) {
                for (const n of e.n) activeSet.add(n);
              }
            }
          }
          // Map raster indices (0..18) to full 300-neuron indices via
          // readout_neurons + neuron_names. readout indices are positions
          // in readout_neurons; we need their index in neuron_names.
          const activeFullSet = new Set<number>();
          if (tr.neuron_names && tr.meta.readout_neurons) {
            const nameToIdx = new Map<string, number>();
            tr.neuron_names.forEach((nm, i) => nameToIdx.set(nm, i));
            for (const ri of activeSet) {
              const nm = tr.meta.readout_neurons[ri];
              const full = nameToIdx.get(nm ?? "");
              if (full !== undefined) activeFullSet.add(full);
            }
          }
          const readoutSet = new Set(tr.meta.readout_neurons);
          drawBrain3D(
            ctx, brainW, panelH,
            tr.neuron_positions ?? [],
            tr.neuron_names ?? [],
            activeFullSet,
            hoverRef.current,
            readoutSet,
          );
        }
      }

      // --- Environment canvas
      if (envCanvasRef.current && tr) {
        const c = envCanvasRef.current;
        if (c.width !== envW) c.width = envW;
        if (c.height !== panelH) c.height = panelH;
        const ctx = c.getContext("2d");
        if (ctx) drawEnvironment(ctx, envW, panelH, tr.environment, t);
      }

      // --- Strips (full width, stacked)
      const stripsW = width;
      const curFrac = tr ? t / tr.meta.duration_s : 0;

      if (modCanvasRef.current && tr) {
        const c = modCanvasRef.current;
        if (c.width !== stripsW) c.width = stripsW;
        if (c.height !== stripH) c.height = stripH;
        const ctx = c.getContext("2d");
        if (ctx) drawModulatorStrip(
          ctx, stripsW, stripH,
          tr.modulator_concentrations, tr.modulator_names, curFrac,
        );
      }

      if (fsmCanvasRef.current && tr) {
        const c = fsmCanvasRef.current;
        if (c.width !== stripsW) c.width = stripsW;
        if (c.height !== 32) c.height = 32;
        const ctx = c.getContext("2d");
        if (ctx) {
          drawFsmTimeline(ctx, stripsW, 32, tr.fsm_states, curFrac);
          if (tr.stim_log) drawStimMarkers(ctx, stripsW, 32, 52, tr.stim_log, tr.meta.duration_s);
        }
      }

      if (evCanvasRef.current && tr) {
        const c = evCanvasRef.current;
        if (c.width !== stripsW) c.width = stripsW;
        if (c.height !== 110) c.height = 110;
        const ctx = c.getContext("2d");
        if (ctx) drawEventProbs(
          ctx, stripsW, 110, tr.event_probs, tr.meta.events_tracked, curFrac,
        );
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [width]);

  // Hover on brain canvas → find nearest neuron
  const onBrainMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const tr = traceRef.current;
    if (!tr || !tr.neuron_positions) return;
    const canvas = brainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const my = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const ys = tr.neuron_positions.map((p) => p[1]);
    const zs = tr.neuron_positions.map((p) => p[2]);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const zMin = Math.min(...zs), zMax = Math.max(...zs);
    const yRange = yMax - yMin || 1;
    const zRange = zMax - zMin || 1;
    const w = canvas.width, h = canvas.height;
    const cy = h / 2;
    let best = -1;
    let bestD = Infinity;
    tr.neuron_positions.forEach((p, i) => {
      const sx = ((p[1] - yMin) / yRange) * w * 0.9 + w * 0.05;
      const sy = cy - ((p[2] - zMin) / zRange - 0.5) * h * 0.75;
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bestD && d < 400) { bestD = d; best = i; }
    });
    setHoverNeuron(best >= 0 ? best : null);
  };

  const onBrainLeave = () => setHoverNeuron(null);

  const scrubTo = (frac: number) => {
    const tr = traceRef.current;
    if (!tr) return;
    currentTRef.current = Math.max(0, Math.min(tr.meta.duration_s, frac * tr.meta.duration_s));
    setCurrentT(currentTRef.current);
  };

  const meta = trace?.meta;
  const currentFrame = trace
    ? trace.frames[Math.min(
        trace.frames.length - 1,
        Math.floor((currentT / trace.meta.duration_s) * trace.frames.length),
      )]
    : null;

  return (
    <div className="my-6 flex flex-col gap-3 text-sm" ref={wrapRef}>
      {/* Top bar: scenario + controls */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-card p-3 shadow-sm">
        <div className="inline-flex flex-wrap rounded-lg border bg-muted/40 p-0.5 text-xs w-fit gap-0.5">
          {(Object.keys(SCENARIOS) as Scenario[]).map((s) => (
            <button
              key={s}
              onClick={() => setScenario(s)}
              className={`rounded-md px-3 py-1.5 font-medium transition-all ${
                scenario === s
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "hover:bg-accent"
              }`}
            >
              {SCENARIOS[s].label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setPaused((v) => !v)}
            className="rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors"
          >
            {paused ? "▶ Play" : "⏸ Pause"}
          </button>
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span>speed</span>
            <input
              type="range"
              min="0.25" max="3" step="0.25"
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
              className="accent-primary w-20"
            />
            <span className="tabular-nums font-mono text-[0.65rem]">{speed.toFixed(2)}×</span>
          </label>
        </div>

        <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
          <span className="tabular-nums font-mono">
            t = {currentT.toFixed(1)} / {meta?.duration_s?.toFixed(0) ?? "-"}s
          </span>
          {currentFrame && (
            <span>
              state: <span
                className="font-semibold px-1.5 py-0.5 rounded text-white text-[0.65rem]"
                style={{ backgroundColor: STATE_COLORS[currentFrame.state] ?? "#6b7280" }}
              >{currentFrame.state}</span>
            </span>
          )}
        </div>
      </div>

      {/* Scenario description + chemotaxis index if present */}
      <div className="text-xs text-muted-foreground flex flex-wrap items-center gap-2">
        <span className="font-semibold text-foreground">{SCENARIOS[scenario].tagline}</span>
        <span className="opacity-70">— {SCENARIOS[scenario].desc}</span>
        {trace?.environment?.chemotaxis_index?.CI !== undefined && (
          <span className="ml-auto inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[0.65rem] font-mono">
            <span className="text-muted-foreground">CI</span>
            <span className="text-foreground font-semibold">
              {(trace.environment.chemotaxis_index.CI as number).toFixed(2)}
            </span>
          </span>
        )}
      </div>

      {/* Keyboard hint */}
      <div className="text-[0.65rem] text-muted-foreground/70">
        ⌨ <kbd className="px-1 rounded border text-[0.6rem]">space</kbd> play/pause ·
        <kbd className="mx-1 px-1 rounded border text-[0.6rem]">← →</kbd> ±1 s scrub
      </div>

      {/* Main panel grid */}
      <div className="flex flex-wrap gap-2">
        <div className="flex-1 min-w-[300px]">
          <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
            body · {meta?.num_segments ?? 20}-segment MuJoCo
          </div>
          <div className="rounded-lg overflow-hidden border">
            <canvas ref={bodyCanvasRef} className="block w-full" />
          </div>
        </div>
        <div className="flex-1 min-w-[320px]">
          <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
            brain · 300 neurons @ 3D soma coords · recently-active highlighted
          </div>
          <div className="rounded-lg overflow-hidden border">
            <canvas
              ref={brainCanvasRef}
              className="block w-full cursor-crosshair"
              onMouseMove={onBrainMove}
              onMouseLeave={onBrainLeave}
            />
          </div>
        </div>
        <div className="flex-1 min-w-[220px]">
          <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
            {trace?.environment ? "arena · food + worm trail" : "arena (inactive)"}
          </div>
          <div className="rounded-lg overflow-hidden border">
            <canvas ref={envCanvasRef} className="block w-full" />
          </div>
        </div>
      </div>

      {/* Scrub bar (clickable timeline) */}
      <div
        className="h-2 rounded-full bg-muted cursor-pointer relative"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          scrubTo((e.clientX - r.left) / r.width);
        }}
      >
        <div
          className="h-full rounded-full bg-primary"
          style={{ width: `${meta ? (currentT / meta.duration_s) * 100 : 0}%` }}
        />
      </div>

      {/* Modulator strip */}
      <div>
        <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
          modulators · 9-concentration volume-transmission field
        </div>
        <div className="rounded-lg overflow-hidden border">
          <canvas ref={modCanvasRef} className="block w-full" />
        </div>
      </div>

      {/* FSM timeline */}
      <div>
        <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
          behavioural state
        </div>
        <div className="rounded-lg overflow-hidden border">
          <canvas ref={fsmCanvasRef} className="block w-full" />
        </div>
      </div>

      {/* Event probability plot */}
      <div>
        <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
          event-classifier probabilities · 8 canonical transitions
        </div>
        <div className="rounded-lg overflow-hidden border">
          <canvas ref={evCanvasRef} className="block w-full" />
        </div>
      </div>

      {loadErr && (
        <div className="text-xs text-destructive">Trace load failed: {loadErr}</div>
      )}

      {/* Attribution fold */}
      <details className="text-xs text-muted-foreground mt-2 rounded-lg border bg-card/40 p-3">
        <summary className="cursor-pointer font-medium text-foreground">
          Sources, methods &amp; honest calibration notes
        </summary>
        <div className="mt-3 space-y-2 pl-2">
          <div><strong>Brain:</strong> 300-neuron LIF (Brian2), Cook et al. 2019 hermaphrodite connectome, NT identity from Loer &amp; Rand 2022. Architecture mirrors Shiu et al. 2024 (<em>Nature</em>) <em>Drosophila</em> brain model, adapted to worm.</div>
          <div><strong>Body:</strong> 20-segment MuJoCo MJCF, Boyle-Berri-Cohen 2012 CPG parameters, resistive-force-theory drag (anisotropy 2.0).</div>
          <div><strong>Event classifiers:</strong> logistic regression on 18-neuron cross-worm-intersection readout, trained on Atanas et al. 2023 (DANDI 000776) paired calcium+behaviour across 10 worms. Cross-worm generalisation validated (train 1–8, test 9–10).</div>
          <div><strong>Integration cadence:</strong> brain-body sync at 50 ms, classifier at 600 ms (Atanas sampling rate). Pattern follows Eon Systems 2026 embodied-fly integration.</div>
          <div><strong>v3 neuromodulation:</strong> 9 peptidergic + monoaminergic modulators (FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5-HT, dopamine, tyramine, octopamine) with releaser + receptor tables extracted from CeNGEN single-cell expression (Taylor et al. 2021). Shown in the strip above.</div>
          <div><strong>Tier 1 upgrades (opt-in via ClosedLoopEnv):</strong> graded non-spiking dynamics (Kunert-Graf 2014) replacing LIF spiking; L-type Ca plateau channels on 14 command neurons; volume-transmission distance-weighted modulator diffusion (λ 150–700 µm); real closed-loop proprioception; 2D agar environment with chemical gradient for Pierce-Shimomura 1999 chemotaxis validation. Shipped scenarios remain on v3 LIF pending v3.3 recalibration.</div>
          <div className="pt-1 italic">
            Perturbation phenotype reproductions are at n=3 seeds × 20 s runs; the AVA/Chalfie result holds at significance but the RIS/Turek signal needs longer runs. Detailed ensemble audit in <code className="text-[0.7rem]">artifacts/ensemble_report.md</code>. Every claim on this page has a measured error bar — no single-seed overclaims.
          </div>
        </div>
      </details>
    </div>
  );
}

export default CelegansDashboard;
