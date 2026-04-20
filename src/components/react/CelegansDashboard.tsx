import * as React from "react";
import { useEffect, useMemo, useRef, useState } from "react";

/**
 * CelegansDashboard — integrated simulator viewer.
 *
 * Panels, all time-synchronised:
 *   • 20-segment body (MuJoCo-derived)
 *   • 300-neuron brain view at 3D soma coords, recently-active highlighted,
 *     glow halos on releaser neurons sized by current modulator concentration
 *   • Arena (2D agar + food + worm trail) when scenario includes T1e env
 *   • 9-modulator concentration strip
 *   • FSM state timeline with stimulus markers
 *   • 8-event classifier probability streams
 */

// ---------- Types -----------------------------------------------------

type Scenario = "spontaneous" | "touch" | "osmotic_shock" | "food" | "chemotaxis";

type BrainEdges = {
  names: string[];
  edges: Array<[number, number, number, number]>; // [pre, post, weight, pre_sign]
};

type NeuronMeta = {
  name: string;
  nt: string;
  class: string;
  sign: number;
  outgoing: Array<[string, number]>;
  incoming: Array<[string, number]>;
};

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

// ---------- Constants -------------------------------------------------

const SCENARIOS: Record<Scenario, { label: string; desc: string; watch: string[] }> = {
  spontaneous: {
    label: "Spontaneous",
    desc: "No stimulus — baseline behavioural distribution.",
    watch: [
      "Mix of FORWARD / REVERSE / QUIESCENT in the FSM timeline",
      "PDF-1 modulator tonically elevated (arousal)",
    ],
  },
  touch: {
    label: "Head touch",
    desc: "ALM/AVM mechanoreceptor drive at t=5s (Chalfie 1981).",
    watch: [
      "Spike at ALM/AVM followed by REVERSE state within ~1s",
      "AVA/AVE command neurons light up",
    ],
  },
  osmotic_shock: {
    label: "Osmotic shock",
    desc: "ASH polymodal avoidance drive at t=5s (Hart 1995).",
    watch: [
      "ASH activates → AIB + AVA cascade visible in brain edges",
      "FLP-11 concentration surges (RIS glows purple in brain)",
      "OMEGA / PIROUETTE states appear after reversal",
    ],
  },
  food: {
    label: "Food",
    desc: "ASI/ASJ/ADF feeding-state tonic from t=2s (Flavell 2013).",
    watch: [
      "NSM 5-HT concentration climbs (emerald glow on NSM L/R)",
      "Pharyngeal neurons (M1-M5) activate",
      "QUIESCENT state dominates — dwelling on food",
    ],
  },
  chemotaxis: {
    label: "Chemotaxis",
    desc: "2D agar + food patch. ASE/AWC/AWA driven by real gradient (Pierce-Shimomura 1999).",
    watch: [
      "Worm trail in arena — does it navigate toward the food patch?",
      "ASE/AWC firing fluctuates with dC/dt as worm moves",
      "Chemotaxis index (CI) in header: positive = toward food",
    ],
  },
};

const STATE_COLORS: Record<string, string> = {
  FORWARD:   "#2f5233",
  REVERSE:   "#b94b4b",
  OMEGA:     "#7c3aed",
  PIROUETTE: "#e76f51",
  QUIESCENT: "#6b7280",
};

const MODULATOR_COLORS: Record<string, string> = {
  "FLP-11": "#a78bfa",
  "FLP-1":  "#c4b5fd",
  "FLP-2":  "#ddd6fe",
  "NLP-12": "#fbcfe8",
  "PDF-1":  "#7c3aed",
  "5HT":    "#10b981",
  "DA":     "#3b82f6",
  "TA":     "#ef4444",
  "OA":     "#f59e0b",
};

const EVENT_COLORS: Record<string, string> = {
  reversal_onset:      "#dc2626",
  reversal_offset:     "#f87171",
  forward_run_onset:   "#059669",
  forward_run_offset:  "#6ee7b7",
  omega_onset:         "#8b5cf6",
  pirouette_entry:     "#f97316",
  quiescence_onset:    "#9ca3af",
  speed_burst_onset:   "#f59e0b",
};

const RELEASERS: Record<string, string[]> = {
  "FLP-11": ["RIS"],
  "FLP-1":  ["AVKL", "AVKR"],
  "FLP-2":  ["AVKL", "AVKR"],
  "NLP-12": ["DVA"],
  "PDF-1":  ["AVBL", "AVBR", "ALA", "RIA"],
  "5HT":    ["NSML", "NSMR", "HSNL", "HSNR", "ADFL", "ADFR"],
  "DA":     ["PDEL", "PDER", "ADEL", "ADER", "CEPDL", "CEPDR", "CEPVL", "CEPVR"],
  "TA":     ["RIML", "RIMR"],
  "OA":     ["RICL", "RICR"],
};

const PANEL_H = 280;
const STRIP_BRAIN_H = 130;
const STRIP_FSM_H   = 36;
const STRIP_EV_H    = 120;
const MOD_STRIP_H   = 180;

// ---------- Utilities -------------------------------------------------

function hexAlpha(hex: string, alpha: number): string {
  const a = Math.max(0, Math.min(255, Math.round(alpha * 255)));
  return hex + a.toString(16).padStart(2, "0");
}

function parseHex(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

function mixColors(a: string, b: string, t: number): string {
  const [r1, g1, b1] = parseHex(a);
  const [r2, g2, b2] = parseHex(b);
  const r = Math.round(r1 * (1 - t) + r2 * t);
  const g = Math.round(g1 * (1 - t) + g2 * t);
  const bl = Math.round(b1 * (1 - t) + b2 * t);
  return `#${[r, g, bl].map((x) => x.toString(16).padStart(2, "0")).join("")}`;
}

function segmentWidth(i: number, total: number): number {
  const t = i / Math.max(1, total - 1);
  return 3 + 9 * Math.sin(Math.PI * t);
}

// Cache position bounds so we don't min/max per-frame
type PosBounds = {
  xMin: number; xMax: number;
  yMin: number; yMax: number;
  zMin: number; zMax: number;
};

function computeBounds(positions: Array<[number, number, number]>): PosBounds {
  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;
  let zMin = Infinity, zMax = -Infinity;
  for (const [x, y, z] of positions) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  return { xMin, xMax, yMin, yMax, zMin, zMax };
}

function projectNeuron(
  p: [number, number, number],
  bounds: PosBounds,
  w: number,
  h: number,
  rotRad: number = 0,
): { sx: number; sy: number; depthT: number } {
  const yRange = bounds.yMax - bounds.yMin || 1;
  const zRange = bounds.zMax - bounds.zMin || 1;
  const xRange = bounds.xMax - bounds.xMin || 1;
  // Normalise to [-1, 1]
  const ny = ((p[1] - bounds.yMin) / yRange) * 2 - 1;
  const nx = ((p[0] - bounds.xMin) / xRange) * 2 - 1;
  const nz = ((p[2] - bounds.zMin) / zRange) * 2 - 1;
  // Rotate around the y (AP) axis: mixes x (LR) into z for depth.
  const cos = Math.cos(rotRad);
  const sin = Math.sin(rotRad);
  const xR = nx * cos - nz * sin;
  const zR = nx * sin + nz * cos;
  const sx = (ny + 1) / 2 * w * 0.9 + w * 0.05;
  const sy = h / 2 - zR * h * 0.4;
  const depthT = (xR + 1) / 2;  // 0 = back, 1 = front
  return { sx, sy, depthT };
}

function setupCanvasDPR(canvas: HTMLCanvasElement, cssW: number, cssH: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const wantW = Math.round(cssW * dpr);
  const wantH = Math.round(cssH * dpr);
  if (canvas.width !== wantW) canvas.width = wantW;
  if (canvas.height !== wantH) canvas.height = wantH;
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return ctx;
}

// ---------- Drawing ---------------------------------------------------

function drawWormBody(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  segs: Array<{ x: number; y: number }>,
  state: string,
  trail: Array<{ x: number; y: number }> | null,
) {
  // Warm cream gradient background
  const bg = ctx.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, Math.max(w, h) * 0.8);
  bg.addColorStop(0, "#f9f0d6");
  bg.addColorStop(1, "#e5d7b9");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Dotted grid
  ctx.fillStyle = "rgba(26, 42, 74, 0.08)";
  for (let y = 14; y < h; y += 24) {
    for (let x = 14; x < w; x += 24) {
      ctx.fillRect(x, y, 1, 1);
    }
  }

  const stateColor = STATE_COLORS[state] ?? "#2f5233";

  // Draw trail (ghosts) of previous head positions
  if (trail && trail.length > 1) {
    ctx.save();
    for (let i = 0; i < trail.length - 1; i++) {
      const a = trail[i];
      const b = trail[i + 1];
      const alpha = (i / trail.length) * 0.35;
      ctx.strokeStyle = hexAlpha(stateColor, alpha);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Glow body
  ctx.save();
  ctx.shadowBlur = 16;
  ctx.shadowColor = hexAlpha(stateColor, 0.5);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Compute per-segment curvature for color gradient
  const curvatures: number[] = new Array(segs.length).fill(0);
  for (let i = 1; i < segs.length - 1; i++) {
    const a = segs[i - 1], b = segs[i], c = segs[i + 1];
    const dx1 = b.x - a.x, dy1 = b.y - a.y;
    const dx2 = c.x - b.x, dy2 = c.y - b.y;
    // Cross-product z-component as curvature proxy
    curvatures[i] = dx1 * dy2 - dy1 * dx2;
  }
  // Normalise
  const maxCurv = Math.max(1, Math.max(...curvatures.map(Math.abs)));

  for (let i = 0; i < segs.length - 1; i++) {
    // Blend base state color with a curvature-derived accent
    const c = curvatures[i] / maxCurv;
    const accent = c > 0 ? "#f59e0b" : "#8b5cf6";  // ventral = amber, dorsal = violet
    const mix = Math.abs(c);
    const col = mixColors(stateColor, accent, mix * 0.45);
    ctx.strokeStyle = i === 0 ? "#1a2a4a" : col;
    ctx.lineWidth = segmentWidth(i, segs.length);
    ctx.beginPath();
    ctx.moveTo(segs[i].x, segs[i].y);
    ctx.lineTo(segs[i + 1].x, segs[i + 1].y);
    ctx.stroke();
  }
  ctx.restore();

  // Head marker
  const head = segs[0];
  ctx.fillStyle = "#f2ead3";
  ctx.strokeStyle = "#1a2a4a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(head.x, head.y, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawSpikeRaster(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  raster: Trace["raster"],
  readoutNames: string[],
  durationS: number,
  currentFrac: number,
) {
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#0f1429"); bg.addColorStop(1, "#0a0e1a");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const labelW = 68;
  const plotW = w - labelW - 8;
  const nNeurons = readoutNames.length;
  if (nNeurons === 0) return;
  const rowH = (h - 12) / nNeurons;

  // Labels
  ctx.fillStyle = "rgba(226, 232, 240, 0.85)";
  ctx.font = "9px system-ui, sans-serif";
  for (let i = 0; i < nNeurons; i++) {
    ctx.fillText(readoutNames[i], 4, 8 + i * rowH + rowH * 0.7);
  }

  // Spike dots
  ctx.fillStyle = "#5ec77a";
  for (const e of raster) {
    const x = labelW + (e.t / durationS) * plotW;
    for (const ni of e.n) {
      if (ni >= 0 && ni < nNeurons) {
        ctx.fillRect(x, 6 + ni * rowH + 1, 2, Math.max(2, rowH - 2));
      }
    }
  }

  // Cursor
  const cursorX = labelW + currentFrac * plotW;
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();

  ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
  ctx.font = "9px system-ui, sans-serif";
  ctx.fillText("spike raster · 18 readout × time", 4, h - 2);
}

function drawBrain3D(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  positions: Array<[number, number, number]>,
  names: string[],
  bounds: PosBounds,
  activeSet: Set<number>,
  recentPulses: Map<number, number>,
  hoverIdx: number | null,
  lockedIdx: number | null,
  readoutSet: Set<string>,
  modConcentrations: number[] | null,
  modNames: string[] | null,
  nameToIdx: Map<string, number>,
  edges: BrainEdges | null,
  edgeAlpha: number,
  highlightedReleasers: Set<number> | null,
  rotRad: number,
) {
  // Dark gradient backdrop
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#0f1429");
  bg.addColorStop(1, "#0a0e1a");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if (!positions || positions.length === 0) {
    ctx.fillStyle = "rgba(210, 220, 240, 0.4)";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Loading brain positions…", w / 2, h / 2);
    ctx.textAlign = "left";
    return;
  }

  // Subtle body outline (head→tail axis)
  ctx.strokeStyle = "rgba(100, 116, 139, 0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w * 0.05, h / 2);
  ctx.lineTo(w * 0.95, h / 2);
  ctx.stroke();

  // Synapse edges — only draw edges touching currently-active neurons
  // (otherwise 400 edges would clutter). Active neurons' outgoing edges
  // fade by weight + by edge alpha (user-controlled).
  if (edges && edgeAlpha > 0.01 && activeSet.size > 0) {
    // Build idx mapping from edges.names to our `names` array
    // (should be identical order since both derived from connectome.npz)
    ctx.save();
    ctx.lineCap = "round";
    for (const [pre, post, weight, preSign] of edges.edges) {
      if (!activeSet.has(pre)) continue;
      const pPre = positions[pre];
      const pPost = positions[post];
      if (!pPre || !pPost) continue;
      const { sx: x1, sy: y1 } = projectNeuron(pPre, bounds, w, h, rotRad);
      const { sx: x2, sy: y2 } = projectNeuron(pPost, bounds, w, h, rotRad);
      const wNorm = Math.min(1, weight / 30);
      const color = preSign > 0 ? "#10b981" : preSign < 0 ? "#ef4444" : "#94a3b8";
      ctx.strokeStyle = hexAlpha(color, edgeAlpha * (0.25 + 0.6 * wNorm));
      ctx.lineWidth = 0.6 + wNorm;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      // Curved line for visual variety
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2 - 8 * (preSign > 0 ? 1 : -1);
      ctx.quadraticCurveTo(mx, my, x2, y2);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Releaser glow halos (modulator concentration visualization)
  if (modConcentrations && modNames) {
    for (let mi = 0; mi < modNames.length; mi++) {
      const conc = modConcentrations[mi] ?? 0;
      if (conc <= 0.3) continue;
      const mod = modNames[mi];
      const color = MODULATOR_COLORS[mod] ?? "#94a3b8";
      const intensity = Math.min(1, conc / 8);
      const releasers = RELEASERS[mod] ?? [];
      for (const rn of releasers) {
        const idx = nameToIdx.get(rn);
        if (idx === undefined) continue;
        const { sx, sy } = projectNeuron(positions[idx], bounds, w, h, rotRad);
        const r = 5 + 18 * intensity;
        const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
        grad.addColorStop(0, hexAlpha(color, 0.55 * intensity));
        grad.addColorStop(1, hexAlpha(color, 0));
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Depth-sorted draw order (back → front)
  const order: number[] = [];
  for (let i = 0; i < positions.length; i++) order.push(i);
  order.sort((a, b) => positions[a][0] - positions[b][0]);

  const hoverLabel: { sx: number; sy: number; name: string } | null = { sx: 0, sy: 0, name: "" } as any;
  let hoverDrawn = false;

  for (const i of order) {
    const { sx, sy, depthT } = projectNeuron(positions[i], bounds, w, h, rotRad);
    const depthFade = 0.5 + 0.5 * depthT;
    const isActive = activeSet.has(i);
    const isReadout = readoutSet.has(names[i] ?? "");
    const isHover = hoverIdx === i;

    let r = isReadout ? 2.5 : 1.6;
    if (isHover) r = 5;

    const pulse = recentPulses.get(i) ?? 0;
    const isHighlight = highlightedReleasers?.has(i) ?? false;
    const isLocked = lockedIdx === i;
    if (isLocked) {
      ctx.shadowBlur = 16;
      ctx.shadowColor = "#f2ead3";
      ctx.fillStyle = "#f2ead3";
      r = 5.5;
    } else if (isHighlight) {
      ctx.shadowBlur = 12;
      ctx.shadowColor = "#facc15";
      ctx.fillStyle = hexAlpha("#facc15", 0.95 * depthFade);
      r = 4.5;
    } else if (isActive || pulse > 0) {
      const glow = Math.max(0.8, pulse);
      ctx.shadowBlur = 6 + 12 * pulse;
      ctx.shadowColor = "#5ec77a";
      ctx.fillStyle = hexAlpha("#5ec77a", 0.9 * glow * depthFade);
      r = (isReadout ? 2.8 : 2.0) + 2.5 * pulse;
    } else if (isReadout) {
      ctx.shadowBlur = 0;
      ctx.fillStyle = hexAlpha("#a5b4fc", 0.7 * depthFade);
    } else {
      ctx.shadowBlur = 0;
      ctx.fillStyle = hexAlpha("#64748b", 0.55 * depthFade);
    }
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();

    if (isHover) {
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "#f2ead3";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      (hoverLabel as any).sx = sx;
      (hoverLabel as any).sy = sy;
      (hoverLabel as any).name = names[i] ?? "?";
      hoverDrawn = true;
    }
  }
  ctx.shadowBlur = 0;

  if (hoverDrawn) {
    ctx.font = "12px system-ui, sans-serif";
    const lbl = (hoverLabel as any).name;
    const mw = ctx.measureText(lbl).width;
    const lx = (hoverLabel as any).sx + 8;
    const ly = (hoverLabel as any).sy - 10;
    ctx.fillStyle = "rgba(15, 20, 41, 0.92)";
    ctx.fillRect(lx - 3, ly - 11, mw + 8, 16);
    ctx.fillStyle = "#f2ead3";
    ctx.fillText(lbl, lx + 1, ly + 1);
  }

  // Axis labels
  ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("head", w * 0.045, h - 8);
  ctx.textAlign = "right";
  ctx.fillText("tail", w * 0.955, h - 8);
  ctx.textAlign = "left";
}

function drawModulatorStrip(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  concentrations: number[][] | undefined,
  names: string[] | undefined,
  currentFrac: number,
  hoverRow: string | null,
) {
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#0f1429");
  bg.addColorStop(1, "#0a0e1a");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if (!concentrations || !names || concentrations.length === 0) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("No modulator telemetry in this scenario.", 14, h / 2);
    return;
  }

  const labelW = 68;
  const valueW = 52;
  const plotW = w - labelW - valueW - 8;
  const numMods = names.length;
  const rowH = (h - 8) / numMods;

  // Per-modulator max
  const maxByMod = new Array(numMods).fill(1e-6);
  for (const row of concentrations) {
    for (let mi = 0; mi < numMods; mi++) {
      if (row[mi] > maxByMod[mi]) maxByMod[mi] = row[mi];
    }
  }

  const nT = concentrations.length;
  const tIdx = Math.min(nT - 1, Math.floor(currentFrac * nT));

  // Draw each modulator row
  for (let mi = 0; mi < numMods; mi++) {
    const y0 = 4 + mi * rowH;
    const color = MODULATOR_COLORS[names[mi]] ?? "#94a3b8";
    const hover = hoverRow === names[mi];

    // Row background for hover highlight
    if (hover) {
      ctx.fillStyle = hexAlpha(color, 0.12);
      ctx.fillRect(0, y0, w, rowH);
    }

    // Label on left
    ctx.fillStyle = hover ? "#f2ead3" : "rgba(226, 232, 240, 0.92)";
    ctx.font = (hover ? "bold " : "") + "11px system-ui, sans-serif";
    ctx.fillText(names[mi], 8, y0 + rowH * 0.7);

    // Color swatch
    ctx.fillStyle = color;
    ctx.fillRect(55, y0 + rowH * 0.35, 8, rowH * 0.3);

    // Heatmap strip
    for (let px = 0; px < plotW; px++) {
      const ti = Math.min(nT - 1, Math.floor((px / plotW) * nT));
      const intensity = Math.min(1, concentrations[ti][mi] / maxByMod[mi]);
      if (intensity < 0.04) continue;
      ctx.fillStyle = hexAlpha(color, intensity * 0.95);
      ctx.fillRect(labelW + px, y0 + 2, 1, rowH - 4);
    }

    // Current concentration value on right
    const curC = concentrations[tIdx][mi] ?? 0;
    ctx.fillStyle = hexAlpha(color, 0.85);
    ctx.font = "10px ui-monospace, monospace";
    const valStr = curC < 10 ? curC.toFixed(2) : curC.toFixed(1);
    ctx.fillText(valStr, labelW + plotW + 4, y0 + rowH * 0.7);
    // Mini bar next to value showing relative level
    ctx.fillStyle = hexAlpha(color, 0.5);
    const barH = Math.max(2, (curC / maxByMod[mi]) * (rowH - 4));
    ctx.fillRect(w - 4, y0 + rowH - 2 - barH, 2, barH);
  }

  // Current-time cursor
  const cursorX = labelW + currentFrac * plotW;
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();
}

function drawFsmTimeline(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  states: number[],
  currentFrac: number,
) {
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);

  const stateNames = ["(none)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"];
  const labelW = 68;
  const plotW = w - labelW - 8;
  const nT = states.length;
  if (nT === 0) return;

  ctx.fillStyle = "rgba(226, 232, 240, 0.92)";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("state", 8, h * 0.65);

  const bw = plotW / nT;
  for (let t = 0; t < nT; t++) {
    const s = states[t];
    const name = stateNames[s] ?? "";
    const color = STATE_COLORS[name] ?? "#6b7280";
    ctx.fillStyle = color;
    ctx.fillRect(labelW + t * bw, 4, Math.max(0.8, bw + 0.5), h - 8);
  }

  const cursorX = labelW + currentFrac * plotW;
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1.5;
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
  ctx.strokeStyle = "rgba(245, 158, 11, 0.8)";
  ctx.fillStyle = "#fef3c7";
  ctx.lineWidth = 1;
  ctx.font = "9px system-ui, sans-serif";
  ctx.setLineDash([2, 3]);
  for (const s of stims) {
    const frac = s.t / durationS;
    const x = labelW + frac * (w - labelW - 8);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function drawEventProbs(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  probs: Record<string, number[]>,
  eventNames: string[] | undefined,
  currentFrac: number,
) {
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#0f1429");
  bg.addColorStop(1, "#0a0e1a");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if (!eventNames || eventNames.length === 0) return;
  const labelW = 68;
  const plotW = w - labelW - 8;

  // Grid
  ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(labelW, h - 4);
  ctx.lineTo(w - 4, h - 4);
  ctx.stroke();
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(labelW, h * 0.5);
  ctx.lineTo(w - 4, h * 0.5);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels
  ctx.fillStyle = "rgba(226, 232, 240, 0.92)";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("events", 8, h * 0.55);
  ctx.font = "9px system-ui, sans-serif";
  ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
  ctx.fillText("1.0", labelW - 20, 12);
  ctx.fillText("0.5", labelW - 20, h * 0.5 + 3);
  ctx.fillText("0", labelW - 14, h - 6);
  // Threshold label
  ctx.fillStyle = "rgba(250, 204, 21, 0.9)";
  ctx.font = "9px system-ui, sans-serif";
  ctx.fillText("fire threshold", labelW + 4, h * 0.5 - 2);

  // Lines + rising-edge detection markers
  for (const ev of eventNames) {
    const arr = probs[ev];
    if (!arr || arr.length === 0) continue;
    const color = EVENT_COLORS[ev] ?? "#94a3b8";
    // Line
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = labelW + (i / arr.length) * plotW;
      const y = h - 6 - arr[i] * (h - 16);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Markers at threshold-crossing upswings (event "fired")
    ctx.fillStyle = color;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i - 1] < 0.5 && arr[i] >= 0.5) {
        const x = labelW + (i / arr.length) * plotW;
        const y = h - 6 - arr[i] * (h - 16);
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  const cursorX = labelW + currentFrac * plotW;
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();
}

function drawEventLegend(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  events: string[] | undefined,
) {
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);
  if (!events) return;
  ctx.font = "10px system-ui, sans-serif";
  const colW = Math.min(150, w / events.length);
  events.forEach((ev, i) => {
    const x = 12 + i * colW;
    const y = h / 2;
    ctx.fillStyle = EVENT_COLORS[ev] ?? "#94a3b8";
    ctx.fillRect(x, y - 4, 10, 3);
    ctx.fillStyle = "rgba(226, 232, 240, 0.85)";
    ctx.fillText(ev.replace(/_/g, " "), x + 14, y + 3);
  });
}

function drawEnvironment(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  env: Trace["environment"] | undefined,
  currentTS: number,
  viewScaleMm: number = 20,
) {
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#0f1429");
  bg.addColorStop(1, "#0a0e1a");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if (!env) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Arena inactive for this scenario.", w / 2, h / 2 - 6);
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText("Select Chemotaxis to see the 2D agar.", w / 2, h / 2 + 10);
    ctx.textAlign = "left";
    return;
  }

  const worldMm = viewScaleMm;
  const pxPerMm = Math.min(w, h) / worldMm;
  const cx = w / 2;
  const cy = h / 2;

  // Grid
  ctx.strokeStyle = "rgba(100, 116, 139, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= w; x += pxPerMm) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
  for (let y = 0; y <= h; y += pxPerMm) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
  ctx.stroke();

  // Food patch radial gradient
  const foodSX = cx + env.food_xy_mm[0] * pxPerMm;
  const foodSY = cy - env.food_xy_mm[1] * pxPerMm;
  const radialR = env.sigma_mm * pxPerMm * 2.5;
  const grad = ctx.createRadialGradient(foodSX, foodSY, 0, foodSX, foodSY, radialR);
  grad.addColorStop(0, "rgba(245, 158, 11, 0.55)");
  grad.addColorStop(0.5, "rgba(245, 158, 11, 0.15)");
  grad.addColorStop(1, "rgba(245, 158, 11, 0.00)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(foodSX, foodSY, radialR, 0, Math.PI * 2);
  ctx.fill();

  // Food marker
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(foodSX, foodSY, 5, 0, Math.PI * 2);
  ctx.fill();

  // Trail up to currentT
  ctx.strokeStyle = "rgba(94, 199, 122, 0.65)";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  let first = true;
  let lastSX = foodSX, lastSY = foodSY;
  for (const p of env.trail) {
    if (p.t > currentTS + 0.15) break;
    const sx = cx + p.x * pxPerMm;
    const sy = cy - p.y * pxPerMm;
    if (first) { ctx.moveTo(sx, sy); first = false; }
    else { ctx.lineTo(sx, sy); }
    lastSX = sx; lastSY = sy;
  }
  ctx.stroke();

  // Worm head
  ctx.shadowBlur = 8;
  ctx.shadowColor = "#5ec77a";
  ctx.fillStyle = "#5ec77a";
  ctx.beginPath();
  ctx.arc(lastSX, lastSY, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Labels
  ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("food", foodSX + 8, foodSY - 6);
  ctx.fillText("worm", lastSX + 8, lastSY - 4);

  // Scale bar
  ctx.strokeStyle = "rgba(148, 163, 184, 0.7)";
  ctx.fillStyle = "rgba(148, 163, 184, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(10, h - 12);
  ctx.lineTo(10 + pxPerMm * 2, h - 12);
  ctx.stroke();
  ctx.font = "9px system-ui, sans-serif";
  ctx.fillText("2 mm", 10, h - 18);
}

// ---------- Main component --------------------------------------------

export function CelegansDashboard() {
  const [scenario, setScenario] = useState<Scenario>("osmotic_shock");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [currentT, setCurrentT] = useState(0);
  const [width, setWidth] = useState(1024);
  const [hoverNeuron, setHoverNeuron] = useState<number | null>(null);
  const [edges, setEdges] = useState<BrainEdges | null>(null);
  const [showEdges, setShowEdges] = useState(true);
  const [edgeAlpha, setEdgeAlpha] = useState(0.6);
  const [lockedNeuron, setLockedNeuron] = useState<number | null>(null);
  const [neuronMeta, setNeuronMeta] = useState<NeuronMeta[] | null>(null);
  const [hoverModulator, setHoverModulator] = useState<string | null>(null);
  const [brainRot, setBrainRot] = useState(0);           // rotation in radians
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; rot: number } | null>(null);
  const [brainViewMode, setBrainViewMode] = useState<"3d" | "raster">("3d");
  const [arenaZoomMm, setArenaZoomMm] = useState(20);  // world extent in arena view
  const [showFps, setShowFps] = useState(false);
  const fpsRef = useRef({ last: 0, frames: 0, fps: 0 });

  const wrapRef = useRef<HTMLDivElement>(null);
  const bodyCanvasRef = useRef<HTMLCanvasElement>(null);
  const brainCanvasRef = useRef<HTMLCanvasElement>(null);
  const envCanvasRef = useRef<HTMLCanvasElement>(null);
  const modCanvasRef = useRef<HTMLCanvasElement>(null);
  const fsmCanvasRef = useRef<HTMLCanvasElement>(null);
  const evCanvasRef = useRef<HTMLCanvasElement>(null);
  const evLegendRef = useRef<HTMLCanvasElement>(null);

  // Load brain edges + neuron metadata once
  useEffect(() => {
    fetch("/data/brain-edges.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: BrainEdges) => setEdges(d))
      .catch(() => setEdges(null));
    fetch("/data/neuron-meta.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: NeuronMeta[]) => setNeuronMeta(d))
      .catch(() => setNeuronMeta(null));
  }, []);

  // Track neuron pulse decay (for spike animation). Pulses decay by dt.
  const pulseRef = useRef<Map<number, number>>(new Map());

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

  // Precompute bounds + name->index map ONCE per trace load
  const brainDerived = useMemo(() => {
    if (!trace?.neuron_positions || !trace?.neuron_names) return null;
    const bounds = computeBounds(trace.neuron_positions);
    const nameToIdx = new Map<string, number>();
    trace.neuron_names.forEach((nm, i) => nameToIdx.set(nm, i));
    const readoutSet = new Set(trace.meta.readout_neurons);
    return { bounds, nameToIdx, readoutSet };
  }, [trace]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement;
      if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA")) return;
      const tr = traceRef.current;
      if (e.code === "Space") {
        e.preventDefault();
        setPaused((v) => !v);
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        if (tr) {
          currentTRef.current = Math.max(0, currentTRef.current - 1);
          setCurrentT(currentTRef.current);
        }
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        if (tr) {
          currentTRef.current = Math.min(tr.meta.duration_s, currentTRef.current + 1);
          setCurrentT(currentTRef.current);
        }
      } else if (e.code === "Comma") {
        // frame back
        if (tr) {
          const dt = tr.meta.brain_sync_ms / 1000;
          currentTRef.current = Math.max(0, currentTRef.current - dt);
          setCurrentT(currentTRef.current);
          setPaused(true);
        }
      } else if (e.code === "Period") {
        if (tr) {
          const dt = tr.meta.brain_sync_ms / 1000;
          currentTRef.current = Math.min(tr.meta.duration_s, currentTRef.current + dt);
          setCurrentT(currentTRef.current);
          setPaused(true);
        }
      } else if (e.code === "KeyR") {
        if (tr) {
          currentTRef.current = 0;
          setCurrentT(0);
        }
      } else if (e.code === "KeyF") {
        setShowFps((v) => !v);
      } else if (e.code === "Digit1" || e.code === "Digit2" ||
                 e.code === "Digit3" || e.code === "Digit4" || e.code === "Digit5") {
        const idx = parseInt(e.code.replace("Digit", "")) - 1;
        const keys = Object.keys(SCENARIOS) as Scenario[];
        if (keys[idx]) setScenario(keys[idx]);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Fetch trace
  useEffect(() => {
    let cancel = false;
    setLoading(true);
    setLoadErr(null);
    setCurrentT(0);
    currentTRef.current = 0;
    fetch(`/data/wormbody-brain-${scenario}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: Trace) => {
        if (cancel) return;
        setTrace(d);
        setLoading(false);
      })
      .catch((e) => {
        if (cancel) return;
        setLoadErr(String(e));
        setLoading(false);
      });
    return () => { cancel = true; };
  }, [scenario]);

  // Responsive width (one observer on the container)
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const w = Math.max(320, Math.round(e.contentRect.width));
        setWidth(w);
      }
    });
    ro.observe(el);
    setWidth(Math.max(320, el.clientWidth));
    return () => ro.disconnect();
  }, []);

  // Animation loop
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const draw = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;

      // FPS counter
      fpsRef.current.frames++;
      if (now - fpsRef.current.last > 500) {
        fpsRef.current.fps = fpsRef.current.frames / ((now - fpsRef.current.last) / 1000);
        fpsRef.current.last = now;
        fpsRef.current.frames = 0;
      }

      const tr = traceRef.current;
      const derived = brainDerived;

      if (tr && !pausedRef.current) {
        currentTRef.current = (currentTRef.current + dt * speedRef.current) % tr.meta.duration_s;
        setCurrentT(currentTRef.current);
      }
      const t = currentTRef.current;

      // Layout: stacked if narrow, 3-column grid if wide
      const narrow = width < 820;
      const colGap = 10;
      let bodyW, brainW, envW;
      if (narrow) {
        bodyW = brainW = envW = width;
      } else {
        bodyW = Math.round(width * 0.30);
        brainW = Math.round(width * 0.44);
        envW = width - bodyW - brainW - colGap * 2;
      }
      const stripsW = width;

      // Body canvas
      if (bodyCanvasRef.current) {
        const ctx = setupCanvasDPR(bodyCanvasRef.current, bodyW, PANEL_H);
        if (ctx && tr) {
          const idx = Math.min(
            tr.frames.length - 1,
            Math.floor((t / tr.meta.duration_s) * tr.frames.length)
          );
          const frame = tr.frames[idx];
          if (frame) {
            const pxPerSimM = Math.min(bodyW, PANEL_H) * 0.5;
            let cx = 0, cy = 0;
            for (const [x, y] of frame.positions) { cx += x; cy += y; }
            cx /= frame.positions.length;
            cy /= frame.positions.length;
            const ox = bodyW / 2 - cx * pxPerSimM;
            const oy = PANEL_H / 2 - cy * pxPerSimM;
            const segs = frame.positions.map(([x, y]) => ({
              x: ox + x * pxPerSimM, y: oy + y * pxPerSimM,
            }));
            // Build head trail from previous frames
            const trailFramesBack = 20;
            const trail: Array<{ x: number; y: number }> = [];
            for (let k = trailFramesBack; k >= 1; k -= 2) {
              const pi = idx - k;
              if (pi < 0) continue;
              const p0 = tr.frames[pi].positions[0];
              trail.push({
                x: ox + p0[0] * pxPerSimM,
                y: oy + p0[1] * pxPerSimM,
              });
            }
            trail.push({ x: segs[0].x, y: segs[0].y });
            drawWormBody(ctx, bodyW, PANEL_H, segs, frame.state, trail);
          }
        } else if (ctx) {
          ctx.fillStyle = "#f9f0d6";
          ctx.fillRect(0, 0, bodyW, PANEL_H);
          ctx.fillStyle = "#6b5e3d";
          ctx.font = "12px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(loading ? "Loading…" : (loadErr ?? ""), bodyW / 2, PANEL_H / 2);
          ctx.textAlign = "left";
        }
      }

      // Brain canvas
      if (brainCanvasRef.current) {
        const ctx = setupCanvasDPR(brainCanvasRef.current, brainW, PANEL_H);
        if (brainViewMode === "raster" && ctx && tr) {
          drawSpikeRaster(
            ctx, brainW, PANEL_H,
            tr.raster, tr.meta.readout_neurons,
            tr.meta.duration_s, t / tr.meta.duration_s,
          );
        } else if (ctx && tr && derived) {
          // Active set from raster within last 100ms
          const activeIdxs = new Set<number>();
          if (tr.raster) {
            for (const e of tr.raster) {
              if (e.t > t - 0.1 && e.t <= t) {
                for (const rIdx of e.n) {
                  const nm = tr.meta.readout_neurons[rIdx];
                  if (nm) {
                    const full = derived.nameToIdx.get(nm);
                    if (full !== undefined) activeIdxs.add(full);
                  }
                }
              }
            }
          }
          // Pulse bookkeeping: decay existing pulses, inject new ones
          // for neurons that just became active.
          const pulses = pulseRef.current;
          for (const [k, v] of pulses) {
            const nv = v - dt * 2.5;  // ~0.4s pulse decay
            if (nv <= 0) pulses.delete(k);
            else pulses.set(k, nv);
          }
          for (const idx of activeIdxs) {
            if (!pulses.has(idx)) pulses.set(idx, 1.0);
          }
          // Modulator concentrations at current time
          let modAt: number[] | null = null;
          if (tr.modulator_concentrations && tr.modulator_concentrations.length) {
            const nT = tr.modulator_concentrations.length;
            const ti = Math.min(nT - 1, Math.floor((t / tr.meta.duration_s) * nT));
            modAt = tr.modulator_concentrations[ti];
          }
          // Highlighted releasers: if user is hovering a modulator row,
          // show its releaser neurons with gold halo.
          let highlighted: Set<number> | null = null;
          if (hoverModulator && derived) {
            const rs = RELEASERS[hoverModulator] ?? [];
            highlighted = new Set<number>();
            for (const rn of rs) {
              const idx = derived.nameToIdx.get(rn);
              if (idx !== undefined) highlighted.add(idx);
            }
          }
          drawBrain3D(
            ctx, brainW, PANEL_H,
            tr.neuron_positions ?? [],
            tr.neuron_names ?? [],
            derived.bounds,
            activeIdxs,
            pulses,
            hoverRef.current,
            lockedNeuron,
            derived.readoutSet,
            modAt,
            tr.modulator_names ?? null,
            derived.nameToIdx,
            showEdges ? edges : null,
            edgeAlpha,
            highlighted,
            brainRot,
          );
        } else if (ctx) {
          const bg = ctx.createLinearGradient(0, 0, 0, PANEL_H);
          bg.addColorStop(0, "#0f1429"); bg.addColorStop(1, "#0a0e1a");
          ctx.fillStyle = bg;
          ctx.fillRect(0, 0, brainW, PANEL_H);
          ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
          ctx.font = "12px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(loading ? "Loading brain…" : "Brain positions not in this trace.", brainW / 2, PANEL_H / 2);
          ctx.textAlign = "left";
        }
      }

      // Environment canvas
      if (envCanvasRef.current) {
        const ctx = setupCanvasDPR(envCanvasRef.current, envW, PANEL_H);
        if (ctx) drawEnvironment(ctx, envW, PANEL_H, tr?.environment, t, arenaZoomMm);
      }

      // Modulator strip
      if (modCanvasRef.current) {
        const ctx = setupCanvasDPR(modCanvasRef.current, stripsW, MOD_STRIP_H);
        if (ctx) {
          const curFrac = tr ? t / tr.meta.duration_s : 0;
          drawModulatorStrip(ctx, stripsW, MOD_STRIP_H,
            tr?.modulator_concentrations, tr?.modulator_names, curFrac, hoverModulator);
        }
      }

      // FSM timeline
      if (fsmCanvasRef.current) {
        const ctx = setupCanvasDPR(fsmCanvasRef.current, stripsW, STRIP_FSM_H);
        if (ctx && tr) {
          const curFrac = t / tr.meta.duration_s;
          drawFsmTimeline(ctx, stripsW, STRIP_FSM_H, tr.fsm_states, curFrac);
          if (tr.stim_log) drawStimMarkers(ctx, stripsW, STRIP_FSM_H, 68, tr.stim_log, tr.meta.duration_s);
        } else if (ctx) {
          ctx.fillStyle = "#0a0e1a";
          ctx.fillRect(0, 0, stripsW, STRIP_FSM_H);
        }
      }

      // Event probs
      if (evCanvasRef.current) {
        const ctx = setupCanvasDPR(evCanvasRef.current, stripsW, STRIP_EV_H);
        if (ctx && tr) {
          const curFrac = t / tr.meta.duration_s;
          drawEventProbs(ctx, stripsW, STRIP_EV_H, tr.event_probs, tr.meta.events_tracked, curFrac);
        } else if (ctx) {
          ctx.fillStyle = "#0a0e1a";
          ctx.fillRect(0, 0, stripsW, STRIP_EV_H);
        }
      }

      // Event legend
      if (evLegendRef.current) {
        const ctx = setupCanvasDPR(evLegendRef.current, stripsW, 28);
        if (ctx) drawEventLegend(ctx, stripsW, 28, tr?.meta.events_tracked);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [width, loading, loadErr, brainDerived, edges, showEdges, edgeAlpha, lockedNeuron, hoverModulator, brainRot, brainViewMode, arenaZoomMm]);

  const onBrainMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const tr = traceRef.current;
    const derived = brainDerived;
    if (!tr || !tr.neuron_positions || !derived) return;
    const canvas = brainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (isDragging && dragStartRef.current) {
      const dx = e.clientX - dragStartRef.current.x;
      setBrainRot(dragStartRef.current.rot + dx * 0.005);
      return;
    }
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const cssW = rect.width, cssH = rect.height;
    let best = -1, bestD = 400;
    for (let i = 0; i < tr.neuron_positions.length; i++) {
      const { sx, sy } = projectNeuron(tr.neuron_positions[i], derived.bounds, cssW, cssH, brainRot);
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    setHoverNeuron(best >= 0 ? best : null);
  };

  const onBrainLeave = () => setHoverNeuron(null);
  const onBrainMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.shiftKey) {
      setIsDragging(true);
      dragStartRef.current = { x: e.clientX, rot: brainRot };
    }
  };
  const onBrainMouseUp = () => {
    setIsDragging(false);
    dragStartRef.current = null;
  };

  const onBrainClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.shiftKey) return;  // shift-click is for drag, don't also lock
    const tr = traceRef.current;
    const derived = brainDerived;
    if (!tr || !tr.neuron_positions || !derived) return;
    const canvas = brainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = -1, bestD = 400;
    for (let i = 0; i < tr.neuron_positions.length; i++) {
      const { sx, sy } = projectNeuron(tr.neuron_positions[i], derived.bounds, rect.width, rect.height, brainRot);
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) setLockedNeuron((prev) => (prev === best ? null : best));
  };

  const lockedMeta = useMemo(() => {
    if (lockedNeuron === null || !trace?.neuron_names || !neuronMeta) return null;
    const nm = trace.neuron_names[lockedNeuron];
    return neuronMeta.find((m) => m.name === nm) ?? null;
  }, [lockedNeuron, trace, neuronMeta]);

  const scrubTo = (frac: number) => {
    const tr = traceRef.current;
    if (!tr) return;
    currentTRef.current = Math.max(0, Math.min(tr.meta.duration_s, frac * tr.meta.duration_s));
    setCurrentT(currentTRef.current);
  };

  const exportPNG = () => {
    // Composite all canvases into one image + download
    const refs = [
      { ref: bodyCanvasRef,  label: "body"   },
      { ref: brainCanvasRef, label: "brain"  },
      { ref: envCanvasRef,   label: "arena"  },
      { ref: modCanvasRef,   label: "mods"   },
      { ref: fsmCanvasRef,   label: "fsm"    },
      { ref: evCanvasRef,    label: "events" },
    ];
    // Compute total height, pick the widest
    const rects = refs.map(({ ref }) => ref.current?.getBoundingClientRect());
    const wMax = Math.max(...rects.map((r) => r?.width ?? 0));
    const gap = 10;
    const padding = 20;
    const panelLabelH = 18;
    let hTotal = padding * 2;
    for (const r of rects) hTotal += (r?.height ?? 0) + panelLabelH + gap;

    const out = document.createElement("canvas");
    const dpr = window.devicePixelRatio || 1;
    out.width = (wMax + padding * 2) * dpr;
    out.height = hTotal * dpr;
    const octx = out.getContext("2d");
    if (!octx) return;
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    octx.fillStyle = "#0a0e1a";
    octx.fillRect(0, 0, out.width, out.height);
    let y = padding;
    octx.font = "12px system-ui, sans-serif";
    for (let i = 0; i < refs.length; i++) {
      const canvas = refs[i].ref.current;
      const rect = rects[i];
      if (!canvas || !rect) continue;
      octx.fillStyle = "rgba(226, 232, 240, 0.85)";
      octx.fillText(refs[i].label.toUpperCase(), padding, y + 12);
      y += panelLabelH;
      octx.drawImage(canvas, padding, y, rect.width, rect.height);
      y += rect.height + gap;
    }
    // Title
    octx.fillStyle = "#f2ead3";
    octx.font = "bold 14px system-ui, sans-serif";
    octx.fillText(
      `C. elegans simulator — ${scenario} @ t=${currentT.toFixed(1)}s`,
      padding, 14,
    );
    const url = out.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = url;
    link.download = `celegans-${scenario}-t${currentT.toFixed(1)}s.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const meta = trace?.meta;
  const currentFrame = trace
    ? trace.frames[Math.min(
        trace.frames.length - 1,
        Math.floor((currentT / trace.meta.duration_s) * trace.frames.length),
      )]
    : null;

  const ci = trace?.environment?.chemotaxis_index?.CI;

  // Live stats derived at currentT
  const liveStats = useMemo(() => {
    if (!trace || !brainDerived) return null;
    const t = currentT;
    // Active neurons (raster events in last 100 ms)
    const activeNames = new Set<string>();
    if (trace.raster) {
      for (const e of trace.raster) {
        if (e.t > t - 0.1 && e.t <= t) {
          for (const rIdx of e.n) {
            const nm = trace.meta.readout_neurons[rIdx];
            if (nm) activeNames.add(nm);
          }
        }
      }
    }
    // Modulator current values + top-3
    let topMods: Array<[string, number]> = [];
    let totalMod = 0;
    if (trace.modulator_concentrations && trace.modulator_names) {
      const nT = trace.modulator_concentrations.length;
      const ti = Math.min(nT - 1, Math.floor((t / trace.meta.duration_s) * nT));
      const row = trace.modulator_concentrations[ti];
      topMods = trace.modulator_names.map((n, i): [string, number] => [n, row[i]]);
      totalMod = topMods.reduce((a, [, v]) => a + v, 0);
      topMods.sort((a, b) => b[1] - a[1]);
      topMods = topMods.slice(0, 3);
    }
    // Current state + duration
    const stateNames = ["(none)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"];
    const nSync = trace.fsm_states.length;
    const stateIdx = Math.min(nSync - 1, Math.floor((t / trace.meta.duration_s) * nSync));
    const currState = stateNames[trace.fsm_states[stateIdx]] ?? "?";
    // State dwell time so far
    let runStart = stateIdx;
    while (runStart > 0 && trace.fsm_states[runStart - 1] === trace.fsm_states[stateIdx]) runStart--;
    const dwellS = (stateIdx - runStart) * (trace.meta.brain_sync_ms / 1000);
    // Events recently crossing 0.5 threshold
    let recentEvents = 0;
    const eventsNames = trace.meta.events_tracked ?? [];
    for (const ev of eventsNames) {
      const arr = trace.event_probs[ev];
      if (!arr) continue;
      const tiE = Math.min(arr.length - 1, Math.floor((t / trace.meta.duration_s) * arr.length));
      const tiWin = Math.max(0, tiE - 5);
      let crossed = false;
      for (let i = tiWin; i <= tiE; i++) {
        if (arr[i] > 0.5) { crossed = true; break; }
      }
      if (crossed) recentEvents++;
    }
    return { activeCount: activeNames.size, topMods, totalMod, currState, dwellS, recentEvents };
  }, [trace, currentT, brainDerived]);

  return (
    <div className="my-8 flex flex-col gap-4 text-sm" ref={wrapRef}>
      {/* Hero intro */}
      <div className="rounded-xl border bg-gradient-to-br from-card via-card/80 to-card/60 p-4 mb-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="inline-flex items-center gap-1.5 text-[0.65rem] uppercase tracking-wider font-semibold">
            <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
            v3 brain · Tier 1 body · live simulator
          </span>
          <span className="text-[0.6rem] text-muted-foreground ml-auto">
            300 neurons · 9 modulators · 5 states · 8 events · 4 published phenotypes
          </span>
        </div>
        <div className="mt-1.5 font-medium text-foreground">
          Closed-loop <em>C. elegans</em> digital twin.
        </div>
        <div className="text-xs text-muted-foreground mt-0.5">
          Sensory input → 300-neuron connectome-constrained brain → 9-modulator
          peptide/monoamine layer → 5-state behavioural FSM → 20-segment MuJoCo body.
          All panels synchronised; click neurons, scrub time, hover modulators.
        </div>
      </div>

      {/* Header bar */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-card px-3 py-2.5 shadow-sm">
        <div className="inline-flex flex-wrap rounded-lg border bg-muted/40 p-0.5 gap-0.5 text-xs">
          {(Object.keys(SCENARIOS) as Scenario[]).map((s) => (
            <button
              key={s}
              onClick={() => setScenario(s)}
              className={`rounded-md px-3 py-1.5 font-medium transition-all ${
                scenario === s
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "hover:bg-accent text-foreground/80"
              }`}
            >
              {SCENARIOS[s].label}
            </button>
          ))}
        </div>
        <button
          onClick={() => setPaused((v) => !v)}
          className="rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors"
        >
          {paused ? "▶ Play" : "⏸ Pause"}
        </button>
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span>speed</span>
          <input
            type="range" min="0.25" max="3" step="0.25"
            value={speed}
            onChange={(e) => setSpeed(+e.target.value)}
            className="accent-primary w-24"
          />
          <span className="tabular-nums font-mono text-[0.65rem] w-8">{speed.toFixed(2)}×</span>
        </label>
        <button
          onClick={exportPNG}
          className="rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors"
          title="Download PNG snapshot of current state"
        >📷 snapshot</button>
        <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
          <span className="tabular-nums font-mono">
            t = {currentT.toFixed(1)} / {meta?.duration_s?.toFixed(0) ?? "–"} s
          </span>
          {currentFrame && (
            <span className="inline-flex items-center gap-1">
              <span
                className="px-1.5 py-0.5 rounded text-[0.65rem] font-semibold text-white"
                style={{ backgroundColor: STATE_COLORS[currentFrame.state] ?? "#6b7280" }}
              >
                {currentFrame.state}
              </span>
            </span>
          )}
          {ci !== undefined && typeof ci === "number" && (
            <span className="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[0.65rem]">
              <span className="text-muted-foreground">CI</span>
              <span className="text-foreground font-semibold">{ci.toFixed(2)}</span>
            </span>
          )}
        </div>
      </div>

      {/* Scenario description + "watch for" hints */}
      <div className="rounded-lg bg-card/40 border px-3 py-2 space-y-1">
        <div className="text-xs text-muted-foreground">
          {SCENARIOS[scenario].desc}
        </div>
        <div className="flex flex-wrap gap-2 text-[0.65rem]">
          <span className="text-muted-foreground font-medium">watch for:</span>
          {SCENARIOS[scenario].watch.map((w, i) => (
            <span key={i} className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 bg-background/40">
              <span className="w-1 h-1 rounded-full bg-primary/60" />
              {w}
            </span>
          ))}
        </div>
      </div>

      {/* Live stats readout */}
      {liveStats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[0.65rem]">
          <StatCard label="active neurons" value={`${liveStats.activeCount}/18`} />
          <StatCard
            label="dominant modulator"
            value={liveStats.topMods[0]?.[0] ?? "—"}
            sub={liveStats.topMods[0] ? `C = ${liveStats.topMods[0][1].toFixed(1)}` : ""}
            accent={liveStats.topMods[0] ? MODULATOR_COLORS[liveStats.topMods[0][0]] : undefined}
          />
          <StatCard
            label="state dwell"
            value={`${liveStats.dwellS.toFixed(1)}s`}
            sub={liveStats.currState}
            accent={STATE_COLORS[liveStats.currState]}
          />
          <StatCard label="events firing" value={`${liveStats.recentEvents}`} sub="of 8 canonical" />
        </div>
      )}

      {/* Scrubbable timeline */}
      <div
        className="h-1.5 rounded-full bg-muted cursor-pointer relative group"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          scrubTo((e.clientX - r.left) / r.width);
        }}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-75"
          style={{ width: `${meta ? (currentT / meta.duration_s) * 100 : 0}%` }}
        />
      </div>

      {/* Main panels — CSS Grid for predictable layout */}
      <div className="grid gap-3" style={{
        gridTemplateColumns: width < 820
          ? "1fr"
          : "minmax(280px, 30fr) minmax(320px, 44fr) minmax(240px, 26fr)"
      }}>
        <div>
          <PanelLabel>body · 20-segment MuJoCo · state-coloured glow</PanelLabel>
          <div className="rounded-lg overflow-hidden border bg-[#f9f0d6]">
            <canvas ref={bodyCanvasRef} className="block w-full" />
          </div>
        </div>
        <div>
          <div className="flex items-baseline justify-between gap-2">
            <PanelLabel>brain · {brainViewMode === "3d" ? "300 neurons · 3D" : "spike raster"}</PanelLabel>
            <div className="flex items-center gap-2 text-[0.65rem] text-muted-foreground">
              <div className="inline-flex rounded-md border p-0.5 bg-muted/30">
                <button
                  onClick={() => setBrainViewMode("3d")}
                  className={`px-2 py-0.5 rounded text-[0.6rem] ${
                    brainViewMode === "3d" ? "bg-primary text-primary-foreground" : "hover:bg-accent"
                  }`}
                >3D</button>
                <button
                  onClick={() => setBrainViewMode("raster")}
                  className={`px-2 py-0.5 rounded text-[0.6rem] ${
                    brainViewMode === "raster" ? "bg-primary text-primary-foreground" : "hover:bg-accent"
                  }`}
                >raster</button>
              </div>
              {brainViewMode === "3d" && (
                <>
                  <label className="inline-flex items-center gap-1 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showEdges}
                      onChange={(e) => setShowEdges(e.target.checked)}
                      className="accent-primary"
                    />
                    edges
                  </label>
                  {showEdges && (
                    <input
                      type="range" min="0.1" max="1" step="0.05"
                      value={edgeAlpha}
                      onChange={(e) => setEdgeAlpha(+e.target.value)}
                      className="accent-primary w-12"
                      title="edge opacity"
                    />
                  )}
                </>
              )}
            </div>
          </div>
          <div className="relative rounded-lg overflow-hidden border bg-[#0a0e1a]">
            <canvas
              ref={brainCanvasRef}
              className={`block w-full ${isDragging ? "cursor-grabbing" : "cursor-crosshair"}`}
              onMouseMove={onBrainMove}
              onMouseLeave={(e) => { onBrainMouseUp(); onBrainLeave(); }}
              onMouseDown={onBrainMouseDown}
              onMouseUp={onBrainMouseUp}
              onClick={onBrainClick}
            />
            {/* Rotation indicator + reset */}
            <div className="absolute top-2 left-2 flex items-center gap-2 rounded-md bg-[#0f1429]/70 px-2 py-1 text-[0.6rem] text-[#94a3b8]">
              <span>⇄ shift-drag to rotate · {Math.round((brainRot * 180 / Math.PI) % 360)}°</span>
              <button
                onClick={() => setBrainRot(0)}
                className="text-[#a5b4fc] hover:text-[#f2ead3]"
              >reset</button>
            </div>
            {lockedMeta && (
              <div className="absolute top-2 right-2 w-60 rounded-lg bg-[#0f1429]/95 border border-[#a5b4fc]/40 p-3 shadow-lg text-[0.7rem] text-[#e2e8f0]">
                <div className="flex items-baseline justify-between mb-1">
                  <span className="text-base font-semibold text-[#a5b4fc]">{lockedMeta.name}</span>
                  <button
                    onClick={() => setLockedNeuron(null)}
                    className="text-[#64748b] hover:text-[#e2e8f0] text-[0.65rem]"
                    title="Unlock"
                  >✕</button>
                </div>
                <div className="space-y-1 leading-relaxed">
                  <div><span className="text-[#64748b]">class:</span> {lockedMeta.class}</div>
                  <div><span className="text-[#64748b]">NT:</span> {lockedMeta.nt}</div>
                  <div><span className="text-[#64748b]">sign:</span>
                    <span className={`ml-1 font-mono ${lockedMeta.sign > 0 ? "text-[#10b981]" : lockedMeta.sign < 0 ? "text-[#ef4444]" : "text-[#94a3b8]"}`}>
                      {lockedMeta.sign > 0 ? "+1 exc" : lockedMeta.sign < 0 ? "−1 inh" : "0 mod"}
                    </span>
                  </div>
                  {lockedMeta.outgoing.length > 0 && (
                    <div className="pt-1 border-t border-[#1e293b]">
                      <div className="text-[#64748b] mb-0.5">top outgoing →</div>
                      {lockedMeta.outgoing.slice(0, 5).map(([n, w]) => (
                        <div key={n} className="pl-2 flex justify-between">
                          <span>{n}</span>
                          <span className="text-[#64748b] font-mono text-[0.6rem]">{w}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {lockedMeta.incoming.length > 0 && (
                    <div className="pt-1 border-t border-[#1e293b]">
                      <div className="text-[#64748b] mb-0.5">top incoming ←</div>
                      {lockedMeta.incoming.slice(0, 5).map(([n, w]) => (
                        <div key={n} className="pl-2 flex justify-between">
                          <span>{n}</span>
                          <span className="text-[#64748b] font-mono text-[0.6rem]">{w}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
        <div>
          <div className="flex items-baseline justify-between gap-2">
            <PanelLabel>
              {trace?.environment ? "arena · 2D agar + food + trail" : "arena (inactive)"}
            </PanelLabel>
            {trace?.environment && (
              <div className="flex items-center gap-1.5 text-[0.6rem] text-muted-foreground">
                <span>zoom</span>
                <button
                  onClick={() => setArenaZoomMm(Math.max(5, arenaZoomMm - 5))}
                  className="rounded border px-1.5 py-0 hover:bg-accent"
                  title="Zoom in"
                >−</button>
                <span className="font-mono tabular-nums">{arenaZoomMm}mm</span>
                <button
                  onClick={() => setArenaZoomMm(Math.min(40, arenaZoomMm + 5))}
                  className="rounded border px-1.5 py-0 hover:bg-accent"
                  title="Zoom out"
                >+</button>
              </div>
            )}
          </div>
          <div className="rounded-lg overflow-hidden border bg-[#0a0e1a]">
            <canvas ref={envCanvasRef} className="block w-full" />
          </div>
        </div>
      </div>

      {/* Modulator strip with hover-linked releaser highlight */}
      <div>
        <PanelLabel>modulators · 9 concentrations × time · hover a row → releaser neurons glow gold</PanelLabel>
        <div className="rounded-lg overflow-hidden border bg-[#0a0e1a] relative">
          <canvas ref={modCanvasRef} className="block w-full" />
          {/* Invisible hover zones over each modulator row */}
          {trace?.modulator_names && (
            <div className="absolute inset-0 pointer-events-none">
              {trace.modulator_names.map((n, i) => {
                const rows = trace.modulator_names!.length;
                const topPct = ((i * (MOD_STRIP_H - 8) / rows + 4) / MOD_STRIP_H) * 100;
                const heightPct = ((MOD_STRIP_H - 8) / rows / MOD_STRIP_H) * 100;
                return (
                  <div
                    key={n}
                    className="absolute left-0 w-full pointer-events-auto cursor-pointer"
                    style={{ top: `${topPct}%`, height: `${heightPct}%` }}
                    onMouseEnter={() => setHoverModulator(n)}
                    onMouseLeave={() => setHoverModulator(null)}
                    title={`${n} releasers: ${(RELEASERS[n] ?? []).join(", ")}`}
                  />
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* FSM timeline + state-diagram inset */}
      <div>
        <PanelLabel>behavioural state · FSM transitions over time · ticks = stimuli</PanelLabel>
        <div className="relative rounded-lg overflow-hidden border bg-[#0a0e1a]">
          <canvas ref={fsmCanvasRef} className="block w-full" />
          <FsmDiagramInset currentState={currentFrame?.state ?? "FORWARD"} />
        </div>
      </div>

      {/* Event probs + legend */}
      <div>
        <PanelLabel>event probabilities · 8 canonical behavioural transitions</PanelLabel>
        <div className="rounded-lg overflow-hidden border bg-[#0a0e1a]">
          <canvas ref={evCanvasRef} className="block w-full" />
          <canvas ref={evLegendRef} className="block w-full" />
        </div>
      </div>

      {/* FSM state legend */}
      <div className="flex flex-wrap gap-3 text-[0.7rem] text-muted-foreground px-1">
        <span className="font-medium text-foreground">states:</span>
        {Object.entries(STATE_COLORS).map(([name, col]) => (
          <span key={name} className="inline-flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded" style={{ backgroundColor: col }} />
            {name}
          </span>
        ))}
        <span className="ml-auto text-[0.65rem]">
          ⌨ <kbd className="px-1 rounded border text-[0.6rem]">space</kbd> play ·
          <kbd className="mx-1 px-1 rounded border text-[0.6rem]">← →</kbd> ±1s ·
          <kbd className="mx-1 px-1 rounded border text-[0.6rem]">, .</kbd> ±frame ·
          <kbd className="mx-1 px-1 rounded border text-[0.6rem]">R</kbd> restart ·
          <kbd className="mx-1 px-1 rounded border text-[0.6rem]">1–5</kbd> scenarios ·
          <kbd className="mx-1 px-1 rounded border text-[0.6rem]">F</kbd> fps
        </span>
      </div>

      {loadErr && (
        <div className="text-xs text-destructive">Trace load failed: {loadErr}</div>
      )}

      {/* FPS overlay */}
      {showFps && (
        <div className="fixed top-4 right-4 z-50 rounded bg-black/80 text-white text-xs px-2 py-1 font-mono">
          {Math.round(fpsRef.current.fps)} fps
        </div>
      )}

      {/* Attribution fold */}
      <details className="text-xs text-muted-foreground mt-2 rounded-xl border bg-card/40 p-4">
        <summary className="cursor-pointer font-semibold text-foreground">
          Sources, methods &amp; honest calibration notes
        </summary>
        <div className="mt-3 space-y-2 pl-2">
          <div><strong>Brain:</strong> 300-neuron LIF in Brian2, Cook et al. 2019 hermaphrodite connectome, NT identity from Loer &amp; Rand 2022. Architecture mirrors Shiu et al. 2024 (<em>Nature</em>) Drosophila brain model, adapted to worm.</div>
          <div><strong>Body:</strong> 20-segment MuJoCo MJCF, Boyle-Berri-Cohen 2012 CPG parameters, resistive-force-theory drag (anisotropy 2.0).</div>
          <div><strong>Event classifiers:</strong> logistic regression on 18-neuron cross-worm-intersection readout, trained on Atanas et al. 2023 (DANDI 000776) paired calcium+behavior across 10 worms. Cross-worm generalization validated (train 1-8, test 9-10).</div>
          <div><strong>Integration cadence:</strong> brain-body sync at 50 ms, classifier at 600 ms (Atanas sampling rate). Pattern follows Eon Systems 2026 embodied-fly integration.</div>
          <div><strong>v3 modulation:</strong> 9 peptidergic + monoaminergic concentrations (FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5-HT, dopamine, tyramine, octopamine) with releaser + receptor tables from CeNGEN single-cell expression (Taylor et al. 2021).</div>
          <div><strong>Tier 1 upgrades (opt-in):</strong> graded non-spiking dynamics (Kunert-Graf 2014), L-type Ca plateau channels, volume-transmission distance-weighted modulator diffusion, real closed-loop proprioception, 2D agar environment for Pierce-Shimomura 1999 chemotaxis validation.</div>
          <div className="pt-1 italic">Current shipped scenarios use v3 LIF brain. Tier 1 graded stack requires v3.4 classifier retraining to reproduce phenotypes. Honest perturbation numbers with n=3 seed error bars documented in <code className="text-[0.7rem]">artifacts/ensemble_report.md</code>.</div>
        </div>
      </details>
    </div>
  );
}

// Tiny SVG state-diagram overlay showing 5 FSM states with the current
// one highlighted. Arrows indicate biologically-relevant transitions.
function FsmDiagramInset({ currentState }: { currentState: string }) {
  const states = ["FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"];
  // Position on a pentagon for nice spatial layout
  const nodes = states.map((s, i) => {
    const angle = (i / states.length) * Math.PI * 2 - Math.PI / 2;
    return {
      name: s,
      x: 50 + 32 * Math.cos(angle),
      y: 32 + 22 * Math.sin(angle),
    };
  });
  const nameToPt = new Map(nodes.map((n) => [n.name, n]));
  // Transitions worth drawing
  const transitions: Array<[string, string]> = [
    ["FORWARD", "REVERSE"],
    ["REVERSE", "FORWARD"],
    ["REVERSE", "OMEGA"],
    ["OMEGA", "FORWARD"],
    ["REVERSE", "PIROUETTE"],
    ["PIROUETTE", "FORWARD"],
    ["FORWARD", "QUIESCENT"],
    ["QUIESCENT", "FORWARD"],
  ];
  return (
    <div className="absolute top-0 right-2 w-24 h-16 pointer-events-none">
      <svg viewBox="0 0 100 64" className="w-full h-full">
        {transitions.map(([a, b], i) => {
          const pa = nameToPt.get(a);
          const pb = nameToPt.get(b);
          if (!pa || !pb) return null;
          return (
            <line key={i}
              x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
              stroke="rgba(148, 163, 184, 0.4)" strokeWidth={0.4}
            />
          );
        })}
        {nodes.map((n) => (
          <g key={n.name}>
            <circle
              cx={n.x} cy={n.y}
              r={currentState === n.name ? 3.5 : 2}
              fill={STATE_COLORS[n.name] ?? "#6b7280"}
              opacity={currentState === n.name ? 1 : 0.5}
            />
          </g>
        ))}
      </svg>
    </div>
  );
}

function PanelLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-1.5 text-[0.7rem] uppercase tracking-wider text-muted-foreground font-medium">
      {children}
    </div>
  );
}

function StatCard({ label, value, sub, accent }: {
  label: string; value: string; sub?: string; accent?: string;
}) {
  return (
    <div className="rounded-lg border bg-card/40 px-3 py-1.5">
      <div className="text-[0.6rem] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="flex items-baseline gap-1.5 mt-0.5">
        {accent && (
          <span
            className="inline-block w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: accent }}
          />
        )}
        <span className="font-semibold text-foreground text-sm tabular-nums">{value}</span>
        {sub && <span className="text-muted-foreground text-[0.65rem]">{sub}</span>}
      </div>
    </div>
  );
}

export default CelegansDashboard;
