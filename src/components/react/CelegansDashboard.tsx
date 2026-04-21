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

/** P0 #2 — CeNGEN panel: per-neuron gene expression subset. */
type CengenPanel = {
  _meta: {
    description: string;
    groups: string[];
    genes_by_group: Record<string, string[]>;
    gene_max_tpm: Record<string, number>;
    total_neurons: number;
    total_panel_genes: number;
  };
  /** neuron name → gene → TPM (sparse: only genes > 0.05 TPM kept) */
  expression: Record<string, Record<string, number>>;
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
  /** P0 #1 — full-network raster, indices refer to all_neurons (or neuron_names). */
  full_raster?: Array<{ t: number; n: number[] }>;
  validated_readout_set?: string[];
  all_neurons?: string[];
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

const SCENARIOS: Record<Scenario, {
  label: string; desc: string; watch: string[];
  moments: Array<{ t: number; label: string }>;
  lit?: string;
}> = {
  spontaneous: {
    label: "Spontaneous",
    desc: "No stimulus — baseline behavioural distribution.",
    watch: [
      "Mix of FORWARD / REVERSE / QUIESCENT in the FSM timeline",
      "PDF-1 modulator tonically elevated (arousal)",
    ],
    moments: [
      { t: 0, label: "start" },
      { t: 10, label: "mid" },
      { t: 25, label: "late" },
    ],
    lit: "Gray, Hill & Bargmann 2005 — roaming/dwelling distribution baseline",
  },
  touch: {
    label: "Head touch",
    desc: "ALM/AVM mechanoreceptor drive at t=5s (Chalfie 1981).",
    watch: [
      "Spike at ALM/AVM followed by REVERSE state within ~1s",
      "AVA/AVE command neurons light up",
    ],
    moments: [
      { t: 4.5, label: "pre-touch" },
      { t: 5.0, label: "touch@t=5s" },
      { t: 5.8, label: "reversal onset" },
      { t: 8, label: "post-recovery" },
    ],
    lit: "Chalfie 1985 — AVA ablation abolishes reversal. v3 reproduces at ΔREV = −0.57 ± 0.37 (n=3 seeds).",
  },
  osmotic_shock: {
    label: "Osmotic shock",
    desc: "ASH polymodal avoidance drive at t=5s (Hart 1995).",
    watch: [
      "ASH activates → AIB + AVA cascade visible in brain edges",
      "FLP-11 concentration surges (RIS glows purple in brain)",
      "OMEGA / PIROUETTE states appear after reversal",
    ],
    moments: [
      { t: 4.5, label: "pre-shock" },
      { t: 5.0, label: "ASH fires" },
      { t: 6.0, label: "reversal + FLP-11 surge" },
      { t: 8.5, label: "omega entry" },
    ],
    lit: "Hart 1995 — ASH drives avoidance. Turek 2016 — RIS/FLP-11 pathway (ΔQUI ≈ −0.24 ± 0.33 at n=3 seeds).",
  },
  food: {
    label: "Food",
    desc: "ASI/ASJ/ADF feeding-state tonic from t=2s (Flavell 2013).",
    watch: [
      "NSM 5-HT concentration climbs (emerald glow on NSM L/R)",
      "Pharyngeal neurons (M1-M5) activate",
      "QUIESCENT state dominates — dwelling on food",
    ],
    moments: [
      { t: 0, label: "pre-food" },
      { t: 2.0, label: "food on" },
      { t: 10, label: "5-HT elevated" },
      { t: 25, label: "dwelling" },
    ],
    lit: "Flavell 2013 — NSM 5-HT signalling induces dwelling on food.",
  },
  chemotaxis: {
    label: "Chemotaxis",
    desc: "2D agar + food patch. ASE/AWC/AWA driven by real gradient (Pierce-Shimomura 1999).",
    watch: [
      "Worm trail in arena — does it navigate toward the food patch?",
      "ASE/AWC firing fluctuates with dC/dt as worm moves",
      "Chemotaxis index (CI) in header: positive = toward food",
    ],
    moments: [
      { t: 0, label: "start at origin" },
      { t: 15, label: "early navigation" },
      { t: 30, label: "mid-run" },
      { t: 55, label: "final approach" },
    ],
    lit: "Pierce-Shimomura, Morse & Lockery 1999 — klinotaxis navigation via biased random walk. Chalasani 2007 — AWC OFF-cell dynamics.",
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

// Named biological circuits — shown as badges when their member
// neurons light up together.
const CIRCUITS: Record<string, { members: string[]; color: string; desc: string }> = {
  "reversal":  { members: ["AVAL", "AVAR", "AIBL", "AIBR", "AVEL", "AVER"], color: "#b94b4b", desc: "AVA/AIB/AVE command interneurons" },
  "forward":   { members: ["AVBL", "AVBR", "PVCL", "PVCR", "RIBL", "RIBR"], color: "#2f5233", desc: "AVB/PVC/RIB forward command" },
  "head-nose": { members: ["ASHL", "ASHR", "OLQDL", "OLQDR", "FLPL", "FLPR"], color: "#7c3aed", desc: "ASH/OLQ/FLP polymodal nose sensors" },
  "feeding":   { members: ["NSML", "NSMR", "M3L", "M3R", "M4"], color: "#10b981", desc: "NSM serotonergic + pharyngeal motor" },
  "omega":     { members: ["SMDVL", "SMDVR", "RIVL", "RIVR", "RMEL"], color: "#8b5cf6", desc: "SMDV/RIV omega-turn circuit" },
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
const STRIP_FSM_H   = 48;
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

  // Compass rose in corner — dorsal (up) / ventral (down) axes
  ctx.save();
  ctx.fillStyle = "rgba(26, 42, 74, 0.42)";
  ctx.font = "7.5px ui-monospace, monospace";
  ctx.fillText("D", 8, 14);
  ctx.fillText("V", 8, h - 10);
  ctx.strokeStyle = "rgba(26, 42, 74, 0.25)";
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  ctx.moveTo(11, 16);
  ctx.lineTo(11, h - 14);
  ctx.stroke();
  ctx.restore();

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

  // Velocity arrow: from head in direction of recent head motion (trail → head)
  if (trail && trail.length >= 2) {
    const prev = trail[trail.length - 2];
    const dx = head.x - prev.x;
    const dy = head.y - prev.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > 1.5) {
      const nx = dx / len;
      const ny = dy / len;
      const arrowLen = Math.min(18, len * 3);
      const ax = head.x + nx * arrowLen;
      const ay = head.y + ny * arrowLen;
      ctx.save();
      ctx.strokeStyle = hexAlpha("#1a2a4a", 0.7);
      ctx.fillStyle = hexAlpha("#1a2a4a", 0.7);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(head.x, head.y);
      ctx.lineTo(ax, ay);
      ctx.stroke();
      // Arrowhead
      const perpX = -ny;
      const perpY = nx;
      const tipSize = 4;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - nx * tipSize + perpX * tipSize * 0.6, ay - ny * tipSize + perpY * tipSize * 0.6);
      ctx.lineTo(ax - nx * tipSize - perpX * tipSize * 0.6, ay - ny * tipSize - perpY * tipSize * 0.6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }
}

function drawSpikeRaster(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  raster: Trace["raster"],
  readoutNames: string[],
  durationS: number,
  currentFrac: number,
  neuronMeta: NeuronMeta[] | null,
  lockedReadoutIdx: number | null,
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

  // Row highlight for locked neuron
  if (lockedReadoutIdx !== null && lockedReadoutIdx >= 0 && lockedReadoutIdx < nNeurons) {
    ctx.fillStyle = "rgba(242, 234, 211, 0.1)";
    ctx.fillRect(0, 6 + lockedReadoutIdx * rowH, w, rowH);
    ctx.strokeStyle = "rgba(242, 234, 211, 0.5)";
    ctx.lineWidth = 0.8;
    ctx.strokeRect(0, 6 + lockedReadoutIdx * rowH, w, rowH);
  }

  // Labels
  ctx.font = "9px system-ui, sans-serif";
  for (let i = 0; i < nNeurons; i++) {
    ctx.fillStyle = i === lockedReadoutIdx ? "#f2ead3" : "rgba(226, 232, 240, 0.85)";
    if (i === lockedReadoutIdx) ctx.font = "bold 9px system-ui, sans-serif";
    else ctx.font = "9px system-ui, sans-serif";
    ctx.fillText(readoutNames[i], 4, 8 + i * rowH + rowH * 0.7);
  }

  // Per-neuron NT color lookup
  const rowColor = readoutNames.map((nm) => {
    if (!neuronMeta) return "#5ec77a";
    const m = neuronMeta.find((mm) => mm.name === nm);
    const nt = m?.nt ?? "";
    if (nt.includes("Acetylcholine") || nt.startsWith("ACh")) return "#38bdf8";
    if (nt.includes("Glutamate")) return "#a3e635";
    if (nt.startsWith("GABA")) return "#f87171";
    if (nt.includes("Dopamine") || nt.includes("Serotonin") ||
        nt.includes("Octopamine") || nt.includes("Tyramine")) return "#c084fc";
    return "#5ec77a";
  });

  // Spike dots, NT-colored
  for (const e of raster) {
    const x = labelW + (e.t / durationS) * plotW;
    for (const ni of e.n) {
      if (ni >= 0 && ni < nNeurons) {
        ctx.fillStyle = rowColor[ni];
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
  dimMask: Set<number> | null,
  ntByIdx: string[] | null,
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
  // fade by weight + by edge alpha (user-controlled). A travelling pulse
  // (bright dot along the curve) visualises the signal propagating from
  // pre to post — pulse progress = (time_since_spike) / TRAVEL_S.
  if (edges && edgeAlpha > 0.01 && (activeSet.size > 0 || lockedIdx !== null)) {
    ctx.save();
    ctx.lineCap = "round";
    for (const [pre, post, weight, preSign] of edges.edges) {
      // If a neuron is locked, show only its incoming + outgoing edges.
      // Otherwise, show edges from currently-active neurons.
      if (lockedIdx !== null) {
        if (pre !== lockedIdx && post !== lockedIdx) continue;
      } else {
        if (!activeSet.has(pre)) continue;
      }
      const pPre = positions[pre];
      const pPost = positions[post];
      if (!pPre || !pPost) continue;
      const { sx: x1, sy: y1 } = projectNeuron(pPre, bounds, w, h, rotRad);
      const { sx: x2, sy: y2 } = projectNeuron(pPost, bounds, w, h, rotRad);
      const wNorm = Math.min(1, weight / 30);
      const color = preSign > 0 ? "#10b981" : preSign < 0 ? "#ef4444" : "#94a3b8";
      // Pulse recency — use the pre's pulse decay value as a proxy for
      // "just fired", so bright edges correlate with fresh spikes.
      const pulse = recentPulses.get(pre) ?? 0;
      const isLockedEdge = lockedIdx !== null;
      const brightness = (isLockedEdge ? 0.5 : 0.25) + 0.6 * wNorm + 0.3 * pulse;
      ctx.strokeStyle = hexAlpha(color, edgeAlpha * Math.min(1, brightness));
      ctx.lineWidth = (isLockedEdge ? 1.0 : 0.6) + wNorm + 0.8 * pulse;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2 - 8 * (preSign > 0 ? 1 : -1);
      ctx.quadraticCurveTo(mx, my, x2, y2);
      ctx.stroke();
      // Travelling pulse dot along the quadratic curve.
      // progress = 1 - pulse (pulse=1 at spike, 0 at end of decay)
      if (pulse > 0.05) {
        const tp = 1 - pulse;  // 0..1 along curve
        // Quadratic Bezier at param tp: (1-tp)^2 * P0 + 2(1-tp)tp * P1 + tp^2 * P2
        const bx = (1 - tp) * (1 - tp) * x1 + 2 * (1 - tp) * tp * mx + tp * tp * x2;
        const by = (1 - tp) * (1 - tp) * y1 + 2 * (1 - tp) * tp * my + tp * tp * y2;
        ctx.shadowBlur = 6;
        ctx.shadowColor = color;
        ctx.fillStyle = hexAlpha(color, Math.min(1, 0.9 * pulse));
        ctx.beginPath();
        ctx.arc(bx, by, 1.2 + 1.2 * pulse, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
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
    const isDimmed = dimMask !== null && dimMask.has(i);
    const depthFade = (0.5 + 0.5 * depthT) * (isDimmed ? 0.2 : 1);
    const isActive = activeSet.has(i);
    const isReadout = readoutSet.has(names[i] ?? "");
    const isHover = hoverIdx === i;

    let r = isReadout ? 2.5 : 1.6;
    if (isHover) r = 5;

    const pulse = recentPulses.get(i) ?? 0;
    const isHighlight = highlightedReleasers?.has(i) ?? false;
    const isLocked = lockedIdx === i;
    if (isLocked) {
      // Pulsing outline — sine oscillation based on performance.now().
      const phase = 0.5 + 0.5 * Math.sin(performance.now() / 400);
      ctx.save();
      ctx.strokeStyle = hexAlpha("#f2ead3", 0.5 + 0.5 * phase);
      ctx.lineWidth = 1 + phase;
      ctx.beginPath();
      ctx.arc(sx, sy, 10 + 2 * phase, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      ctx.shadowBlur = 18;
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
      // Per-NT spike color: ACh cyan, Glu lime, GABA red, modulatory purple
      let spikeColor = "#5ec77a";  // default green
      const nt = ntByIdx?.[i] ?? "";
      if (nt.includes("Acetylcholine") || nt.startsWith("ACh")) spikeColor = "#38bdf8";
      else if (nt.includes("Glutamate")) spikeColor = "#a3e635";
      else if (nt.startsWith("GABA")) spikeColor = "#f87171";
      else if (
        nt.includes("Dopamine") || nt.includes("Serotonin") ||
        nt.includes("Octopamine") || nt.includes("Tyramine")
      ) spikeColor = "#c084fc";
      // Expanding spike ring — radius grows with (1-pulse), alpha fades out
      if (pulse > 0.05) {
        const ringR = 3 + (1 - pulse) * 14;
        ctx.save();
        ctx.strokeStyle = hexAlpha(spikeColor, 0.6 * pulse * depthFade);
        ctx.lineWidth = 1.2 * pulse;
        ctx.beginPath();
        ctx.arc(sx, sy, ringR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      ctx.shadowBlur = 6 + 12 * pulse;
      ctx.shadowColor = spikeColor;
      ctx.fillStyle = hexAlpha(spikeColor, 0.9 * glow * depthFade);
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
    const name = (hoverLabel as any).name;
    const nt = ntByIdx?.[hoverIdx ?? -1] ?? "";
    const ntShort = nt.replace(/\s*\([^)]+\)/, "").replace(/unc-17,.*/, "ACh");
    const lbl = name + (ntShort ? ` · ${ntShort}` : "");
    const mw = ctx.measureText(lbl).width;
    const lx = (hoverLabel as any).sx + 8;
    const ly = (hoverLabel as any).sy - 10;
    ctx.fillStyle = "rgba(15, 20, 41, 0.92)";
    ctx.strokeStyle = "rgba(165, 180, 252, 0.4)";
    ctx.lineWidth = 1;
    ctx.fillRect(lx - 3, ly - 11, mw + 8, 16);
    ctx.strokeRect(lx - 3, ly - 11, mw + 8, 16);
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

  // Mini edge-color legend in the lower-right
  if (edges && edgeAlpha > 0.01) {
    const legX = w - 96;
    const legY = h - 44;
    ctx.save();
    ctx.fillStyle = "rgba(15, 20, 41, 0.72)";
    ctx.fillRect(legX - 4, legY - 4, 96, 42);
    ctx.strokeStyle = "rgba(100, 116, 139, 0.3)";
    ctx.strokeRect(legX - 4, legY - 4, 96, 42);
    ctx.font = "8px ui-monospace, monospace";
    ctx.fillStyle = "rgba(226, 232, 240, 0.85)";
    ctx.fillText("edges:", legX, legY + 4);
    // Excitatory
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(legX, legY + 14);
    ctx.lineTo(legX + 16, legY + 14);
    ctx.stroke();
    ctx.fillText("exc", legX + 20, legY + 17);
    // Inhibitory
    ctx.strokeStyle = "#ef4444";
    ctx.beginPath();
    ctx.moveTo(legX, legY + 24);
    ctx.lineTo(legX + 16, legY + 24);
    ctx.stroke();
    ctx.fillText("inh", legX + 20, legY + 27);
    // Gap/mod
    ctx.strokeStyle = "#94a3b8";
    ctx.beginPath();
    ctx.moveTo(legX + 40, legY + 24);
    ctx.lineTo(legX + 56, legY + 24);
    ctx.stroke();
    ctx.fillText("mod", legX + 60, legY + 27);
    ctx.restore();
  }
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
      ctx.fillStyle = hexAlpha(color, intensity * 0.6);
      ctx.fillRect(labelW + px, y0 + 2, 1, rowH - 4);
    }
    // Line overlay — concentration trajectory as a bright curve
    ctx.save();
    ctx.strokeStyle = hexAlpha(color, 0.9);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let px = 0; px < plotW; px += 1) {
      const ti = Math.min(nT - 1, Math.floor((px / plotW) * nT));
      const frac = Math.min(1, concentrations[ti][mi] / maxByMod[mi]);
      const yy = y0 + rowH - 2 - frac * (rowH - 4);
      if (px === 0) ctx.moveTo(labelW + px, yy);
      else ctx.lineTo(labelW + px, yy);
    }
    ctx.stroke();
    ctx.restore();

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

function drawTimeTicks(
  ctx: CanvasRenderingContext2D,
  w: number, h: number, labelW: number,
  durationS: number, tickEveryS: number,
) {
  ctx.save();
  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
  ctx.lineWidth = 1;
  ctx.font = "8px ui-monospace, monospace";
  const plotW = w - labelW - 8;
  for (let t = 0; t <= durationS; t += tickEveryS) {
    const x = labelW + (t / durationS) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, h - 1);
    ctx.lineTo(x, h - 4);
    ctx.stroke();
    ctx.fillText(`${t}s`, x + 2, h - 6);
  }
  ctx.restore();
}

function drawFsmTimeline(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  states: number[],
  currentFrac: number,
  durationS: number,
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

  // Time axis ticks — 5s cadence
  const tickEvery = durationS > 45 ? 10 : durationS > 15 ? 5 : 2;
  drawTimeTicks(ctx, w, h, labelW, durationS, tickEvery);

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
  // Label each stim with its preset name
  ctx.fillStyle = "#fbbf24";
  ctx.font = "9px system-ui, sans-serif";
  for (const s of stims) {
    const frac = s.t / durationS;
    const x = labelW + frac * (w - labelW - 8);
    const label = s.preset.replace(/_/g, " ");
    ctx.fillText(label, x + 3, 10);
  }
  ctx.restore();
}

function drawEventFireMarkers(
  ctx: CanvasRenderingContext2D,
  w: number, h: number, labelW: number,
  probs: Record<string, number[]>,
  eventNames: string[] | undefined,
  durationS: number,
) {
  if (!eventNames) return;
  ctx.save();
  for (const ev of eventNames) {
    const arr = probs[ev];
    if (!arr || arr.length < 2) continue;
    const col = EVENT_COLORS[ev] ?? "#94a3b8";
    for (let i = 1; i < arr.length; i++) {
      if (arr[i - 1] < 0.5 && arr[i] >= 0.5) {
        const frac = i / arr.length;
        const x = labelW + frac * (w - labelW - 8);
        // Small caret at the top of the FSM strip
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - 3, 4);
        ctx.lineTo(x + 3, 4);
        ctx.closePath();
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

function drawEventProbs(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  probs: Record<string, number[]>,
  eventNames: string[] | undefined,
  currentFrac: number,
  durationS: number,
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

  // Time ticks at bottom
  const tickEvery = durationS > 45 ? 10 : durationS > 15 ? 5 : 2;
  ctx.save();
  ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
  ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
  ctx.font = "8px ui-monospace, monospace";
  for (let t = 0; t <= durationS; t += tickEvery) {
    const x = labelW + (t / durationS) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, h - 6);
    ctx.lineTo(x, h - 2);
    ctx.stroke();
  }
  ctx.restore();

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

  // Concentration contour rings — iso-lines at C = 0.75, 0.50, 0.25 of peak.
  // For a Gaussian with σ, C/Cmax = exp(-d²/(2σ²)) → d = σ*sqrt(-2*ln(frac))
  ctx.save();
  ctx.strokeStyle = "rgba(251, 191, 36, 0.4)";
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 0.8;
  ctx.font = "8px ui-monospace, monospace";
  ctx.fillStyle = "rgba(251, 191, 36, 0.7)";
  for (const frac of [0.75, 0.5, 0.25]) {
    const d_mm = env.sigma_mm * Math.sqrt(-2 * Math.log(frac));
    const rPx = d_mm * pxPerMm;
    ctx.beginPath();
    ctx.arc(foodSX, foodSY, rPx, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillText(`${Math.round(frac * 100)}%`, foodSX + rPx - 12, foodSY - 2);
  }
  ctx.setLineDash([]);
  ctx.restore();

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

  // Time tick marks along the trail — every 10 s
  ctx.save();
  ctx.fillStyle = "rgba(94, 199, 122, 0.85)";
  ctx.font = "8px ui-monospace, monospace";
  let nextTick = 10;
  for (const p of env.trail) {
    if (p.t > currentTS + 0.15) break;
    if (p.t >= nextTick) {
      const sx = cx + p.x * pxPerMm;
      const sy = cy - p.y * pxPerMm;
      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(226, 232, 240, 0.75)";
      ctx.fillText(`${nextTick}s`, sx + 4, sy + 3);
      ctx.fillStyle = "rgba(94, 199, 122, 0.85)";
      nextTick += 10;
    }
  }
  ctx.restore();

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

  // Compute current distance to food + local concentration (Gaussian model)
  let wormX = 0, wormY = 0;
  for (const p of env.trail) {
    if (p.t > currentTS + 0.15) break;
    wormX = p.x; wormY = p.y;
  }
  const dx = wormX - env.food_xy_mm[0];
  const dy = wormY - env.food_xy_mm[1];
  const distMm = Math.sqrt(dx * dx + dy * dy);
  const cRel = Math.exp(-(distMm * distMm) / (2 * env.sigma_mm * env.sigma_mm));

  // Telemetry box in top-right of arena
  ctx.save();
  ctx.fillStyle = "rgba(15, 20, 41, 0.78)";
  ctx.fillRect(w - 108, 4, 104, 34);
  ctx.strokeStyle = "rgba(100, 116, 139, 0.3)";
  ctx.strokeRect(w - 108, 4, 104, 34);
  ctx.fillStyle = "rgba(226, 232, 240, 0.95)";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText(`d(food) = ${distMm.toFixed(2)} mm`, w - 102, 17);
  ctx.fillText(`C(worm) = ${(cRel * 100).toFixed(1)}%`, w - 102, 30);
  ctx.restore();

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
  // Cache of state distributions per scenario (computed from JSON on first load)
  const [scenarioStats, setScenarioStats] = useState<Record<Scenario, {
    fwd: number; rev: number; omg: number; pir: number; qui: number;
  }> | null>(null);
  const [brainRot, setBrainRot] = useState(0);           // rotation in radians
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; rot: number } | null>(null);
  const [brainViewMode, setBrainViewMode] = useState<"3d" | "raster">("3d");
  const [arenaZoomMm, setArenaZoomMm] = useState(20);  // world extent in arena view
  const [showFps, setShowFps] = useState(false);
  const fpsRef = useRef({ last: 0, frames: 0, fps: 0 });
  const [copiedLink, setCopiedLink] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [copiedCite, setCopiedCite] = useState(false);
  const [cengenPanel, setCengenPanel] = useState<CengenPanel | null>(null);

  // Load CeNGEN panel (P0 #2) once
  useEffect(() => {
    fetch("/data/cengen-panel.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: CengenPanel) => setCengenPanel(d))
      .catch(() => setCengenPanel(null));
  }, []);

  // Read scenario + time from URL hash (#scenario=touch&t=5.2) on first mount
  const didInitFromUrl = useRef(false);
  useEffect(() => {
    if (didInitFromUrl.current || typeof window === "undefined") return;
    didInitFromUrl.current = true;
    const hash = window.location.hash.replace(/^#/, "");
    if (!hash) return;
    const params = new URLSearchParams(hash);
    const s = params.get("scenario") as Scenario | null;
    let shouldScroll = false;
    if (s && (["spontaneous", "touch", "osmotic_shock", "food", "chemotaxis"] as Scenario[]).includes(s)) {
      setScenario(s);
      shouldScroll = true;
    }
    const tRaw = params.get("t");
    if (tRaw) {
      const tN = parseFloat(tRaw);
      if (isFinite(tN)) {
        currentTRef.current = tN;
        setCurrentT(tN);
        setPaused(true);
        shouldScroll = true;
      }
    }
    // Scroll dashboard into view after a brief delay (wait for layout)
    if (shouldScroll) {
      setTimeout(() => {
        wrapRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 200);
    }
  }, []);

  // Respect prefers-reduced-motion — disable the pulse/trail/edge animations
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReducedMotion(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  const wrapRef = useRef<HTMLDivElement>(null);
  const bodyCanvasRef = useRef<HTMLCanvasElement>(null);
  const brainCanvasRef = useRef<HTMLCanvasElement>(null);
  const envCanvasRef = useRef<HTMLCanvasElement>(null);
  const modCanvasRef = useRef<HTMLCanvasElement>(null);
  const fsmCanvasRef = useRef<HTMLCanvasElement>(null);
  const evCanvasRef = useRef<HTMLCanvasElement>(null);
  const evLegendRef = useRef<HTMLCanvasElement>(null);

  // Fetch state distribution for all scenarios once (for picker sparklines)
  const [scenarioTimelines, setScenarioTimelines] = useState<Record<Scenario, number[]> | null>(null);
  useEffect(() => {
    const scenarioKeys: Scenario[] = ["spontaneous", "touch", "osmotic_shock", "food", "chemotaxis"];
    const stats: Partial<Record<Scenario, any>> = {};
    const timelines: Partial<Record<Scenario, number[]>> = {};
    let pending = scenarioKeys.length;
    scenarioKeys.forEach((s) => {
      fetch(`/data/wormbody-brain-${s}.json`)
        .then((r) => r.ok ? r.json() : null)
        .then((d: Trace | null) => {
          if (d && d.fsm_states) {
            const total = d.fsm_states.length || 1;
            const counts = [0, 0, 0, 0, 0, 0];
            for (const x of d.fsm_states) if (x >= 0 && x <= 5) counts[x]++;
            stats[s] = {
              fwd: counts[1] / total, rev: counts[2] / total,
              omg: counts[3] / total, pir: counts[4] / total,
              qui: counts[5] / total,
            };
            // Downsample FSM sequence to 40 bins (mode per bin)
            const nBins = 40;
            const bins: number[] = new Array(nBins);
            for (let b = 0; b < nBins; b++) {
              const start = Math.floor((b / nBins) * total);
              const end = Math.floor(((b + 1) / nBins) * total);
              const counts2 = [0, 0, 0, 0, 0, 0];
              for (let i = start; i < end; i++) {
                const x = d.fsm_states[i];
                if (x >= 0 && x <= 5) counts2[x]++;
              }
              let mx = 0, mi = 0;
              for (let i = 0; i < 6; i++) if (counts2[i] > mx) { mx = counts2[i]; mi = i; }
              bins[b] = mi;
            }
            timelines[s] = bins;
          }
          pending--;
          if (pending === 0) {
            setScenarioStats(stats as any);
            setScenarioTimelines(timelines as any);
          }
        })
        .catch(() => {
          pending--;
          if (pending === 0) {
            setScenarioStats(stats as any);
            setScenarioTimelines(timelines as any);
          }
        });
    });
  }, []);

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

  // NT-type filter — empty set means "show all"
  const [ntFilter, setNtFilter] = useState<Set<string>>(new Set());

  // Hovered circuit (for cross-panel highlight of its members)
  const [hoverCircuit, setHoverCircuit] = useState<string | null>(null);

  // Precompute bounds + name->index map ONCE per trace load
  const brainDerived = useMemo(() => {
    if (!trace?.neuron_positions || !trace?.neuron_names) return null;
    const bounds = computeBounds(trace.neuron_positions);
    const nameToIdx = new Map<string, number>();
    trace.neuron_names.forEach((nm, i) => nameToIdx.set(nm, i));
    const readoutSet = new Set(trace.meta.readout_neurons);
    return { bounds, nameToIdx, readoutSet };
  }, [trace]);

  // Map NT category -> set of raw NT strings from neuronMeta
  const NT_CATEGORIES: Record<string, string[]> = {
    "ACh":    ["Acetylcholine (ACh)", "ACh (unc-17, no cho-1)"],
    "Glu":    ["Glutamate (Glu)"],
    "GABA":   ["GABA"],
    "Modulatory": ["Dopamine (DA)", "Serotonin / 5HT", "Octopamine (OA)", "Tyramine (TA)"],
    "Unknown": ["Unknown", "unknown"],
  };

  // Build per-index NT lookup once per trace+meta load
  const ntByIdx = useMemo(() => {
    if (!trace?.neuron_names || !neuronMeta) return null;
    const byName = new Map<string, string>();
    for (const m of neuronMeta) byName.set(m.name, m.nt);
    return trace.neuron_names.map((nm) => byName.get(nm) ?? "");
  }, [trace, neuronMeta]);

  // Dim mask: indices NOT in active NT filter become dimmed
  const ntDimMask = useMemo(() => {
    if (ntFilter.size === 0 || !trace?.neuron_names || !neuronMeta) return null;
    const allowed = new Set<string>();
    Array.from(ntFilter).forEach((cat) => {
      for (const nt of (NT_CATEGORIES[cat] ?? [])) allowed.add(nt);
    });
    const nameToNT = new Map<string, string>();
    for (const m of neuronMeta) nameToNT.set(m.name, m.nt);
    const dim = new Set<number>();
    trace.neuron_names.forEach((nm, i) => {
      const nt = nameToNT.get(nm);
      if (!nt || !allowed.has(nt)) dim.add(i);
    });
    return dim;
  }, [ntFilter, trace, neuronMeta]);

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
      } else if (e.code === "KeyE") {
        setShowEdges((v) => !v);
      } else if (e.code === "KeyV") {
        setBrainViewMode((v) => (v === "3d" ? "raster" : "3d"));
      } else if (e.code === "BracketLeft" || e.code === "BracketRight") {
        // Cycle locked neuron through the 18-readout set
        if (!tr) return;
        const readout = tr.meta.readout_neurons;
        const lockedName = lockedNeuron !== null ? tr.neuron_names?.[lockedNeuron] : null;
        const curR = lockedName ? readout.indexOf(lockedName) : -1;
        const dir = e.code === "BracketRight" ? 1 : -1;
        const next = ((curR < 0 ? 0 : curR) + dir + readout.length) % readout.length;
        const nextName = readout[next];
        const derived = brainDerived;
        const gIdx = derived?.nameToIdx.get(nextName);
        if (gIdx !== undefined) setLockedNeuron(gIdx);
      } else if (e.code === "Slash" && e.shiftKey) {
        // Shift+/ → "?"
        setShowHelp((v) => !v);
      } else if (e.code === "Escape") {
        setShowHelp(false);
        setLockedNeuron(null);
      } else if (e.code === "Digit1" || e.code === "Digit2" ||
                 e.code === "Digit3" || e.code === "Digit4" || e.code === "Digit5") {
        const idx = parseInt(e.code.replace("Digit", "")) - 1;
        const keys = Object.keys(SCENARIOS) as Scenario[];
        if (keys[idx]) setScenario(keys[idx]);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [brainDerived, lockedNeuron]);

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

      if (tr && !pausedRef.current && !reducedMotion) {
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
          // Map global lockedNeuron index → readout index, if applicable
          const lockedName = lockedNeuron !== null ? tr.neuron_names?.[lockedNeuron] : null;
          const lockedRIdx = lockedName ? tr.meta.readout_neurons.indexOf(lockedName) : -1;
          drawSpikeRaster(
            ctx, brainW, PANEL_H,
            tr.raster, tr.meta.readout_neurons,
            tr.meta.duration_s, t / tr.meta.duration_s,
            neuronMeta,
            lockedRIdx >= 0 ? lockedRIdx : null,
          );
        } else if (ctx && tr && derived) {
          // Active set from raster within last 100 ms. Prefer the full-
          // network raster (P0 #1) — indices there already refer to
          // the full neuron list. Fall back to the 18-readout raster
          // for older JSON exports.
          const activeIdxs = new Set<number>();
          if (tr.full_raster && tr.full_raster.length > 0) {
            for (const e of tr.full_raster) {
              if (e.t > t - 0.1 && e.t <= t) {
                for (const fIdx of e.n) activeIdxs.add(fIdx);
              }
            }
          } else if (tr.raster) {
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
          // Highlighted neurons: releasers of hovered modulator,
          // or members of hovered circuit.
          let highlighted: Set<number> | null = null;
          if ((hoverModulator || hoverCircuit) && derived) {
            highlighted = new Set<number>();
            if (hoverModulator) {
              const rs = RELEASERS[hoverModulator] ?? [];
              for (const rn of rs) {
                const idx = derived.nameToIdx.get(rn);
                if (idx !== undefined) highlighted.add(idx);
              }
            }
            if (hoverCircuit && CIRCUITS[hoverCircuit]) {
              for (const nm of CIRCUITS[hoverCircuit].members) {
                const idx = derived.nameToIdx.get(nm);
                if (idx !== undefined) highlighted.add(idx);
              }
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
            ntDimMask,
            ntByIdx,
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
          drawFsmTimeline(ctx, stripsW, STRIP_FSM_H, tr.fsm_states, curFrac, tr.meta.duration_s);
          drawEventFireMarkers(ctx, stripsW, STRIP_FSM_H, 68, tr.event_probs, tr.meta.events_tracked, tr.meta.duration_s);
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
          drawEventProbs(ctx, stripsW, STRIP_EV_H, tr.event_probs, tr.meta.events_tracked, curFrac, tr.meta.duration_s);
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
  }, [width, loading, loadErr, brainDerived, edges, showEdges, edgeAlpha, lockedNeuron, hoverModulator, brainRot, brainViewMode, arenaZoomMm, ntDimMask, hoverCircuit, reducedMotion, ntByIdx, neuronMeta]);

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

  // Touch: two-finger drag rotates the brain (mobile equivalent of shift-drag)
  const onBrainTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (e.touches.length === 2) {
      const avg = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      setIsDragging(true);
      dragStartRef.current = { x: avg, rot: brainRot };
    }
  };
  const onBrainTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (e.touches.length === 2 && dragStartRef.current) {
      const avg = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const dx = avg - dragStartRef.current.x;
      setBrainRot(dragStartRef.current.rot + dx * 0.005);
    }
  };
  const onBrainTouchEnd = () => {
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
    // Raster mode: map y → row → readout neuron → global index
    if (brainViewMode === "raster") {
      const rasterH = rect.height;
      const nNeurons = tr.meta.readout_neurons.length;
      const rowH = (rasterH - 12) / Math.max(1, nNeurons);
      const row = Math.floor((my - 6) / rowH);
      if (row >= 0 && row < nNeurons) {
        const nm = tr.meta.readout_neurons[row];
        const gIdx = derived.nameToIdx.get(nm);
        if (gIdx !== undefined) {
          setLockedNeuron((prev) => (prev === gIdx ? null : gIdx));
        }
      }
      return;
    }
    // 3D mode: project and pick
    const PICK_RADIUS_SQ = 120;
    let best = -1, bestD = PICK_RADIUS_SQ;
    for (let i = 0; i < tr.neuron_positions.length; i++) {
      const { sx, sy } = projectNeuron(tr.neuron_positions[i], derived.bounds, rect.width, rect.height, brainRot);
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) {
      setLockedNeuron((prev) => (prev === best ? null : best));
    } else if (lockedNeuron !== null) {
      // Clicked empty space → unlock
      setLockedNeuron(null);
    }
  };

  const lockedMeta = useMemo(() => {
    if (lockedNeuron === null || !trace?.neuron_names || !neuronMeta) return null;
    const nm = trace.neuron_names[lockedNeuron];
    return neuronMeta.find((m) => m.name === nm) ?? null;
  }, [lockedNeuron, trace, neuronMeta]);

  // Firing-rate history for the locked neuron, if it's a readout neuron.
  // Bins the raster into 0.5 s buckets and computes spike-count per bin.
  const lockedRateHist = useMemo(() => {
    if (lockedNeuron === null || !trace) return null;
    const nm = trace.neuron_names?.[lockedNeuron];
    if (!nm) return null;
    const rIdx = trace.meta.readout_neurons.indexOf(nm);
    if (rIdx < 0) return null;  // not a readout — no raster data
    const BIN_S = 0.5;
    const nBins = Math.max(1, Math.ceil(trace.meta.duration_s / BIN_S));
    const bins = new Array<number>(nBins).fill(0);
    for (const e of trace.raster) {
      if (e.n.includes(rIdx)) {
        const bi = Math.min(nBins - 1, Math.floor(e.t / BIN_S));
        bins[bi]++;
      }
    }
    return { bins, binS: BIN_S, maxRate: Math.max(1, ...bins) };
  }, [lockedNeuron, trace]);

  const scrubTo = (frac: number) => {
    const tr = traceRef.current;
    if (!tr) return;
    currentTRef.current = Math.max(0, Math.min(tr.meta.duration_s, frac * tr.meta.duration_s));
    setCurrentT(currentTRef.current);
  };

  const exportCSV = () => {
    const tr = traceRef.current;
    if (!tr) return;
    const stateNames = ["(none)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"];
    const events = tr.meta.events_tracked ?? [];
    const mods = tr.modulator_names ?? [];
    const header = ["t_s", "state"]
      .concat(events.map((e) => `p_${e}`))
      .concat(mods.map((m) => `c_${m}`));
    const rows: string[] = [header.join(",")];
    const nSync = tr.fsm_states.length;
    const dt = tr.meta.brain_sync_ms / 1000;
    for (let i = 0; i < nSync; i++) {
      const t = i * dt;
      const state = stateNames[tr.fsm_states[i]] ?? "";
      const evalEvent = (ev: string) => {
        const arr = tr.event_probs[ev];
        if (!arr || arr.length === 0) return "";
        const ei = Math.min(arr.length - 1, Math.floor((t / tr.meta.duration_s) * arr.length));
        return arr[ei].toFixed(3);
      };
      const evCols = events.map(evalEvent);
      let modCols: string[] = [];
      if (tr.modulator_concentrations && tr.modulator_concentrations.length > 0) {
        const nMT = tr.modulator_concentrations.length;
        const mi = Math.min(nMT - 1, Math.floor((t / tr.meta.duration_s) * nMT));
        modCols = mods.map((_, k) => (tr.modulator_concentrations![mi][k] ?? 0).toFixed(3));
      } else {
        modCols = mods.map(() => "");
      }
      rows.push([t.toFixed(3), state, ...evCols, ...modCols].join(","));
    }
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `celegans-${scenario}-trace.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
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
    // Metadata footer
    octx.fillStyle = "rgba(226, 232, 240, 0.6)";
    octx.font = "10px ui-monospace, monospace";
    const ts = new Date().toISOString().slice(0, 19).replace("T", " ") + " UTC";
    octx.fillText(`exported ${ts} · rohitravi.com/projects/c-elegans-multimodal`, padding, hTotal - 6);
    const url = out.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = url;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    link.download = `celegans-${scenario}-t${currentT.toFixed(1)}s-${stamp}.png`;
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
    // Active neurons (raster events in last 400 ms — wider for circuit
    // detection). Prefer the full raster (P0 #1) for the 'wide' set so
    // non-readout circuit members like RIS/PVC get counted.
    const activeNames = new Set<string>();     // of 18 readout, last 100 ms
    const activeAllNames = new Set<string>();  // of 300, last 100 ms
    const activeWideNames = new Set<string>(); // of 300, last 400 ms
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
    const fullNames = trace.all_neurons ?? trace.neuron_names;
    if (trace.full_raster && fullNames) {
      for (const e of trace.full_raster) {
        if (e.t > t - 0.1 && e.t <= t) {
          for (const fIdx of e.n) {
            const nm = fullNames[fIdx];
            if (nm) activeAllNames.add(nm);
          }
        }
        if (e.t > t - 0.4 && e.t <= t) {
          for (const fIdx of e.n) {
            const nm = fullNames[fIdx];
            if (nm) activeWideNames.add(nm);
          }
        }
      }
    } else if (trace.raster) {
      // Fallback: readout-only wide window
      for (const e of trace.raster) {
        if (e.t > t - 0.4 && e.t <= t) {
          for (const rIdx of e.n) {
            const nm = trace.meta.readout_neurons[rIdx];
            if (nm) activeWideNames.add(nm);
          }
        }
      }
    }
    // Circuit activation: fraction of circuit members firing in window
    const activeCircuits: Array<{ name: string; frac: number; color: string; desc: string }> = [];
    for (const [name, c] of Object.entries(CIRCUITS)) {
      const fires = c.members.filter((m) => activeWideNames.has(m)).length;
      const frac = fires / c.members.length;
      if (frac > 0.0) activeCircuits.push({ name, frac, color: c.color, desc: c.desc });
    }
    activeCircuits.sort((a, b) => b.frac - a.frac);
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
    const activeEvents: Array<{ name: string; prob: number }> = [];
    for (const ev of eventsNames) {
      const arr = trace.event_probs[ev];
      if (!arr) continue;
      const tiE = Math.min(arr.length - 1, Math.floor((t / trace.meta.duration_s) * arr.length));
      const tiWin = Math.max(0, tiE - 5);
      let crossed = false;
      let maxP = 0;
      for (let i = tiWin; i <= tiE; i++) {
        if (arr[i] > 0.5) { crossed = true; }
        if (arr[i] > maxP) maxP = arr[i];
      }
      if (crossed) {
        recentEvents++;
        activeEvents.push({ name: ev, prob: maxP });
      }
    }
    activeEvents.sort((a, b) => b.prob - a.prob);
    // Rolling history for sparklines — 40 samples over the last 4 s
    const HIST_N = 40;
    const HIST_WINDOW_S = 4;
    const hist = {
      active: new Array<number>(HIST_N).fill(0),
      topMod: new Array<number>(HIST_N).fill(0),
      events: new Array<number>(HIST_N).fill(0),
    };
    for (let k = 0; k < HIST_N; k++) {
      const tk = Math.max(0, t - HIST_WINDOW_S + (k / (HIST_N - 1)) * HIST_WINDOW_S);
      // Active neurons at tk — count spikes within last 100ms window
      let ak = 0;
      const seen = new Set<number>();
      if (trace.raster) {
        for (const e of trace.raster) {
          if (e.t > tk - 0.1 && e.t <= tk) {
            for (const ni of e.n) if (!seen.has(ni)) { seen.add(ni); ak++; }
          }
        }
      }
      hist.active[k] = ak;
      if (trace.modulator_concentrations && trace.modulator_names && topMods[0]) {
        const nT = trace.modulator_concentrations.length;
        const tiK = Math.min(nT - 1, Math.floor((tk / trace.meta.duration_s) * nT));
        const mi = trace.modulator_names.indexOf(topMods[0][0]);
        if (mi >= 0) hist.topMod[k] = trace.modulator_concentrations[tiK][mi];
      }
      let ek = 0;
      const eventsNames2 = trace.meta.events_tracked ?? [];
      for (const ev of eventsNames2) {
        const arr = trace.event_probs[ev];
        if (!arr) continue;
        const tiE = Math.min(arr.length - 1, Math.floor((tk / trace.meta.duration_s) * arr.length));
        if (arr[tiE] > 0.5) ek++;
      }
      hist.events[k] = ek;
    }
    return {
      activeCount: activeNames.size,
      activeAllCount: activeAllNames.size,
      topMods, totalMod, currState, dwellS, recentEvents,
      activeCircuits, hist, activeEvents,
    };
  }, [trace, currentT, brainDerived]);

  // Transition opacity — fades panels while a new scenario is loading
  const panelOpacity = loading ? 0.35 : 1;
  const panelTransition = "opacity 260ms cubic-bezier(0.4, 0, 0.2, 1)";

  // Scrub timeline preview — what state/time + active counts at hover x
  const [scrubHover, setScrubHover] = useState<{ x: number; t: number; state: string; active: number; events: number } | null>(null);

  // Neuron search
  const [searchQ, setSearchQ] = useState("");
  const searchMatches = useMemo(() => {
    if (!searchQ.trim() || !trace?.neuron_names) return [];
    const q = searchQ.trim().toUpperCase();
    const names = trace.neuron_names;
    const out: Array<{ name: string; idx: number; meta: NeuronMeta | null }> = [];
    for (let i = 0; i < names.length && out.length < 8; i++) {
      const nm = names[i];
      if (nm.toUpperCase().includes(q)) {
        const m = neuronMeta?.find((mm) => mm.name === nm) ?? null;
        out.push({ name: nm, idx: i, meta: m });
      }
    }
    return out;
  }, [searchQ, trace, neuronMeta]);

  return (
    <div
      className="my-8 flex flex-col gap-4 text-sm"
      ref={wrapRef}
      role="region"
      aria-label="C. elegans closed-loop brain-body simulator"
    >
      {/* Hero intro */}
      <div className="rounded-xl border bg-gradient-to-br from-card via-card/80 to-card/60 p-4 mb-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="inline-flex items-center gap-1.5 text-[0.65rem] uppercase tracking-wider font-semibold">
            <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
            v3 brain · Tier 1 body · live simulator
          </span>
          <span className="text-[0.6rem] text-muted-foreground ml-auto flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
            {trace ? (
              <>
                <span>{trace.neuron_names?.length ?? 300} neurons</span>
                <span>·</span>
                <span>{trace.modulator_names?.length ?? 9} modulators</span>
                <span>·</span>
                <span>{(trace.meta.states?.length ?? 6) - 1} states</span>
                <span>·</span>
                <span>{trace.meta.events_tracked?.length ?? 8} events</span>
                <span>·</span>
                <span>{trace.meta.duration_s.toFixed(0)}s scenario</span>
                <span>·</span>
                <span>{(1000 / trace.meta.brain_sync_ms).toFixed(0)} Hz sync</span>
              </>
            ) : (
              <>300 neurons · 9 modulators · 5 states · 8 events · 4 published phenotypes</>
            )}
          </span>
        </div>
        <div className="mt-1.5 font-medium text-foreground">
          Closed-loop <em>C. elegans</em> digital twin.
        </div>
        <div className="text-xs text-muted-foreground mt-0.5">
          Sensory input → {trace?.neuron_names?.length ?? 300}-neuron connectome-constrained brain → 9-modulator
          peptide/monoamine layer → 5-state behavioural FSM → 20-segment MuJoCo body.
          All panels synchronised; click neurons, scrub time, hover modulators.
        </div>
      </div>

      {/* Header bar */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-card px-3 py-2.5 shadow-sm">
        <div className="inline-flex flex-wrap rounded-lg border bg-muted/40 p-0.5 gap-0.5 text-xs">
          {(Object.keys(SCENARIOS) as Scenario[]).map((s, sIdx) => {
            const stats = scenarioStats?.[s];
            const timeline = scenarioTimelines?.[s];
            const stateNames = ["(none)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"];
            return (
              <button
                key={s}
                onClick={() => setScenario(s)}
                className={`rounded-md px-3 py-1.5 font-medium transition-all flex items-center gap-1.5 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1 ${
                  scenario === s
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "hover:bg-accent text-foreground/80"
                }`}
                title={stats ? `Press ${sIdx + 1} · FWD ${(stats.fwd*100).toFixed(0)}% · REV ${(stats.rev*100).toFixed(0)}% · OMG ${(stats.omg*100).toFixed(0)}% · PIR ${(stats.pir*100).toFixed(0)}% · QUI ${(stats.qui*100).toFixed(0)}%` : undefined}
                aria-label={`Select scenario: ${SCENARIOS[s].label}. Keyboard shortcut: ${sIdx + 1}. ${SCENARIOS[s].desc}`}
                aria-pressed={scenario === s}
              >
                <kbd
                  className={`px-1 rounded text-[0.55rem] font-mono font-normal ${
                    scenario === s
                      ? "bg-primary-foreground/20 text-primary-foreground/90"
                      : "bg-muted text-muted-foreground/80"
                  }`}
                >{sIdx + 1}</kbd>
                <span>{SCENARIOS[s].label}</span>
                {/* Mini FSM timeline — mode-per-bin over the whole scenario */}
                {timeline && (
                  <span
                    className="inline-flex h-2 w-12 rounded overflow-hidden ring-1 ring-border/40 transition-opacity"
                    style={{ opacity: scenario === s ? 1 : 0.6 }}
                  >
                    {timeline.map((si, i) => (
                      <span
                        key={i}
                        className="h-full"
                        style={{
                          flex: 1,
                          backgroundColor: STATE_COLORS[stateNames[si]] ?? "#1e293b",
                        }}
                      />
                    ))}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <button
          onClick={() => setPaused((v) => !v)}
          className="rounded-md border px-2 sm:px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
          aria-label={paused ? "Play simulation" : "Pause simulation"}
          aria-pressed={!paused}
        >
          <span className="sm:hidden">{paused ? "▶" : "⏸"}</span>
          <span className="hidden sm:inline">{paused ? "▶ Play" : "⏸ Pause"}</span>
        </button>
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span>speed</span>
          <input
            type="range" min="0.25" max="3" step="0.25"
            value={speed}
            onChange={(e) => setSpeed(+e.target.value)}
            className="accent-primary w-24 focus:outline-none focus:ring-2 focus:ring-primary rounded"
            aria-label={`Playback speed: ${speed.toFixed(2)} times normal`}
          />
          <span className="tabular-nums font-mono text-[0.65rem] w-8">{speed.toFixed(2)}×</span>
        </label>
        <button
          onClick={exportPNG}
          className="rounded-md border px-2 sm:px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
          title="Download PNG snapshot of current state"
          aria-label="Download PNG snapshot of all panels at current time"
        ><span className="sm:hidden">📷</span><span className="hidden sm:inline">📷 snapshot</span></button>
        <button
          onClick={() => {
            const url = new URL(window.location.href);
            url.hash = `scenario=${scenario}&t=${currentT.toFixed(2)}`;
            navigator.clipboard?.writeText(url.toString()).then(() => {
              setCopiedLink(true);
              setTimeout(() => setCopiedLink(false), 1600);
            });
          }}
          className="rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
          title="Copy a link to this exact moment"
          aria-label="Copy shareable link to current scenario and time"
        ><span className="sm:hidden">{copiedLink ? "✓" : "🔗"}</span><span className="hidden sm:inline">{copiedLink ? "✓ copied" : "🔗 link"}</span></button>
        <button
          onClick={exportCSV}
          className="rounded-md border px-2 sm:px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
          aria-label="Download scenario trace as CSV"
          title="Export trace (state, event probs, modulators) as CSV"
        ><span className="sm:hidden">⇩</span><span className="hidden sm:inline">⇩ csv</span></button>
        <button
          onClick={() => {
            setLockedNeuron(null);
            setBrainRot(0);
            setNtFilter(new Set());
            setSearchQ("");
            setBrainViewMode("3d");
            setShowEdges(true);
            setEdgeAlpha(0.6);
            setArenaZoomMm(20);
          }}
          className="rounded-md border px-2 sm:px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
          aria-label="Reset brain view and filters to defaults"
          title="Clear locks, filters, rotation, zoom"
        ><span className="sm:hidden">↺</span><span className="hidden sm:inline">↺ reset</span></button>
        <button
          onClick={() => setShowHelp((v) => !v)}
          className="rounded-md border px-2 sm:px-3 py-1.5 text-xs font-medium hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
          aria-label="Show help overlay"
          title="Help / shortcuts (?)"
          aria-expanded={showHelp}
        ><span className="sm:hidden">?</span><span className="hidden sm:inline">? help</span></button>
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

      {/* Scenario description + "watch for" hints + jump-to-moment buttons */}
      <div className="rounded-lg bg-card/40 border px-3 py-2 space-y-1">
        <div className="text-xs text-muted-foreground">
          {SCENARIOS[scenario].desc}
        </div>
        {SCENARIOS[scenario].lit && (
          <div className="text-[0.65rem] text-muted-foreground border-l-2 border-primary/40 pl-2 italic">
            <span className="not-italic text-foreground/70 font-medium mr-1">Compare to literature:</span>
            {SCENARIOS[scenario].lit}
          </div>
        )}
        <div className="flex flex-wrap gap-2 text-[0.65rem]">
          <span className="text-muted-foreground font-medium">watch for:</span>
          {SCENARIOS[scenario].watch.map((w, i) => (
            <span key={i} className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 bg-background/40">
              <span className="w-1 h-1 rounded-full bg-primary/60" />
              {w}
            </span>
          ))}
        </div>
        <div className="flex flex-wrap gap-1.5 text-[0.65rem] pt-0.5">
          <span className="text-muted-foreground font-medium">jump to:</span>
          {SCENARIOS[scenario].moments.map((m) => (
            <button
              key={m.label}
              onClick={() => {
                currentTRef.current = m.t;
                setCurrentT(m.t);
                setPaused(true);
              }}
              className="rounded-md border px-2 py-0.5 hover:bg-accent hover:border-primary transition-colors font-mono focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1"
              title={`Jump to t=${m.t.toFixed(1)}s and pause`}
            >
              {m.t.toFixed(1)}s · {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Natural-language narrative status */}
      {liveStats && (
        <div className="rounded-lg bg-gradient-to-r from-card/60 to-card/20 border px-3 py-2 text-[0.75rem] leading-relaxed">
          <span className="text-muted-foreground">At </span>
          <span className="font-mono font-semibold text-foreground">t={currentT.toFixed(1)}s</span>
          <span className="text-muted-foreground">, the worm is in </span>
          <span
            className="font-semibold px-1 rounded text-white"
            style={{ backgroundColor: STATE_COLORS[liveStats.currState] ?? "#6b7280" }}
          >
            {liveStats.currState}
          </span>
          <span className="text-muted-foreground"> (dwell </span>
          <span className="font-mono text-foreground">{liveStats.dwellS.toFixed(1)}s</span>
          <span className="text-muted-foreground">). </span>
          {liveStats.activeCount > 0 && (
            <>
              <span className="font-mono text-foreground">{liveStats.activeCount}/18</span>
              <span className="text-muted-foreground"> readout neurons firing</span>
              {liveStats.activeCircuits.length > 0 && (
                <>
                  <span className="text-muted-foreground">; </span>
                  {liveStats.activeCircuits.slice(0, 2).map((c, i) => (
                    <React.Fragment key={c.name}>
                      {i > 0 && <span className="text-muted-foreground"> and </span>}
                      <span
                        className="font-semibold"
                        style={{ color: c.color }}
                      >
                        {c.name}
                      </span>
                      <span className="text-muted-foreground"> circuit ({Math.round(c.frac * 100)}%)</span>
                    </React.Fragment>
                  ))}
                </>
              )}
              <span className="text-muted-foreground">. </span>
            </>
          )}
          {liveStats.topMods[0] && liveStats.topMods[0][1] > 0.2 && (
            <>
              <span className="text-muted-foreground">Dominant modulator </span>
              <span
                className="font-semibold"
                style={{ color: MODULATOR_COLORS[liveStats.topMods[0][0]] ?? "#94a3b8" }}
              >
                {liveStats.topMods[0][0]}
              </span>
              <span className="text-muted-foreground"> (C=</span>
              <span className="font-mono">{liveStats.topMods[0][1].toFixed(1)}</span>
              <span className="text-muted-foreground">).</span>
            </>
          )}
        </div>
      )}

      {/* Live stats readout */}
      {liveStats && (
        <div className="space-y-2">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[0.65rem]">
            <StatCard
              label={trace?.full_raster ? "active neurons" : "active readouts"}
              value={trace?.full_raster
                ? `${liveStats.activeAllCount}/${trace.all_neurons?.length ?? trace.neuron_names?.length ?? 300}`
                : `${liveStats.activeCount}/18`}
              sub={trace?.full_raster
                ? `incl. ${liveStats.activeCount}/18 validated`
                : undefined}
              spark={liveStats.hist.active}
              sparkColor="#5ec77a"
            />
            <StatCard
              label="dominant modulator"
              value={liveStats.topMods[0]?.[0] ?? "—"}
              sub={liveStats.topMods[0] ? `C = ${liveStats.topMods[0][1].toFixed(1)}` : ""}
              accent={liveStats.topMods[0] ? MODULATOR_COLORS[liveStats.topMods[0][0]] : undefined}
              spark={liveStats.hist.topMod}
              sparkColor={liveStats.topMods[0] ? MODULATOR_COLORS[liveStats.topMods[0][0]] : "#94a3b8"}
            />
            <StatCard
              label="state dwell"
              value={`${liveStats.dwellS.toFixed(1)}s`}
              sub={liveStats.currState}
              accent={STATE_COLORS[liveStats.currState]}
            />
            <StatCard
              label="events firing"
              value={`${liveStats.recentEvents}`}
              sub="of 8 canonical"
              spark={liveStats.hist.events}
              sparkColor="#f59e0b"
            />
          </div>
          {/* Active-event badges — which classifier events are firing right now */}
          {liveStats.activeEvents.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 text-[0.65rem]">
              <span className="text-muted-foreground font-medium">events firing:</span>
              {liveStats.activeEvents.map((e) => {
                const col = EVENT_COLORS[e.name] ?? "#94a3b8";
                return (
                  <span
                    key={e.name}
                    className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5"
                    style={{
                      backgroundColor: hexAlpha(col, 0.12),
                      borderColor: hexAlpha(col, 0.6),
                      color: "var(--foreground)",
                    }}
                  >
                    <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ backgroundColor: col }} />
                    <span className="font-semibold">{e.name.replace(/_/g, " ")}</span>
                    <span className="text-muted-foreground font-mono text-[0.6rem]">
                      p={e.prob.toFixed(2)}
                    </span>
                  </span>
                );
              })}
            </div>
          )}
          {/* Active-circuit badges */}
          {liveStats.activeCircuits.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 text-[0.65rem]">
              <span className="text-muted-foreground font-medium">active circuits:</span>
              {liveStats.activeCircuits.map((c) => (
                <span
                  key={c.name}
                  title={c.desc}
                  onMouseEnter={() => setHoverCircuit(c.name)}
                  onMouseLeave={() => setHoverCircuit(null)}
                  className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 cursor-pointer transition-transform hover:scale-105"
                  style={{
                    backgroundColor: hexAlpha(c.color, hoverCircuit === c.name ? 0.25 : 0.10),
                    borderColor: c.color,
                    borderWidth: hoverCircuit === c.name ? "1.5px" : "1px",
                    color: "var(--foreground)",
                  }}
                >
                  <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ backgroundColor: c.color }} />
                  <span className="font-semibold">{c.name}</span>
                  <span className="text-muted-foreground font-mono text-[0.6rem]">
                    {Math.round(c.frac * 100)}%
                  </span>
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Scrubbable timeline — thicker + drag-capable */}
      <div
        className="h-2.5 rounded-full bg-muted cursor-pointer relative group hover:bg-muted/70 transition-colors"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          scrubTo((e.clientX - r.left) / r.width);
        }}
        onPointerDown={(e) => {
          // Enable click-and-drag scrubbing
          const el = e.currentTarget;
          const r = el.getBoundingClientRect();
          const update = (clientX: number) => {
            const frac = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
            scrubTo(frac);
          };
          update(e.clientX);
          const move = (ev: PointerEvent) => update(ev.clientX);
          const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
          };
          window.addEventListener("pointermove", move);
          window.addEventListener("pointerup", up);
        }}
        onMouseMove={(e) => {
          const tr = traceRef.current;
          if (!tr) return;
          const r = e.currentTarget.getBoundingClientRect();
          const frac = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
          const tHover = frac * tr.meta.duration_s;
          const nSync = tr.fsm_states.length;
          const sIdx = Math.min(nSync - 1, Math.floor(frac * nSync));
          const stateNames = ["(none)", "FORWARD", "REVERSE", "OMEGA", "PIROUETTE", "QUIESCENT"];
          const st = stateNames[tr.fsm_states[sIdx]] ?? "?";
          // Count active neurons in ±100 ms window at tHover
          let active = 0;
          if (tr.raster) {
            const seen = new Set<number>();
            for (const ev of tr.raster) {
              if (ev.t > tHover - 0.1 && ev.t <= tHover) {
                for (const ni of ev.n) if (!seen.has(ni)) { seen.add(ni); active++; }
              }
            }
          }
          // Event firings at tHover
          let events = 0;
          for (const ev of (tr.meta.events_tracked ?? [])) {
            const arr = tr.event_probs[ev];
            if (!arr) continue;
            const ei = Math.min(arr.length - 1, Math.floor(frac * arr.length));
            if (arr[ei] > 0.5) events++;
          }
          setScrubHover({ x: e.clientX - r.left, t: tHover, state: st, active, events });
        }}
        onMouseLeave={() => setScrubHover(null)}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-75 relative"
          style={{ width: `${meta ? (currentT / meta.duration_s) * 100 : 0}%` }}
        >
          {/* Draggable knob */}
          <div className="absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-primary border-2 border-background shadow-md group-hover:scale-125 transition-transform" />
        </div>
        {/* Stim markers on the scrub bar itself */}
        {trace?.stim_log?.map((s, i) => (
          <div
            key={i}
            className="absolute top-0 h-full w-0.5 bg-amber-400/80 pointer-events-none"
            style={{ left: `${(s.t / (meta?.duration_s ?? 1)) * 100}%` }}
            title={`stim t=${s.t.toFixed(1)}s ${s.preset}`}
          />
        ))}
        {/* Hover preview tooltip */}
        {scrubHover && (
          <div
            className="absolute -top-9 pointer-events-none flex flex-col items-center z-10"
            style={{ left: scrubHover.x, transform: "translateX(-50%)" }}
          >
            <div
              className="rounded bg-[#0f1429]/95 border border-border px-2 py-0.5 shadow-md text-[0.65rem] font-mono whitespace-nowrap flex items-center gap-1.5"
            >
              <span className="text-foreground tabular-nums">{scrubHover.t.toFixed(2)}s</span>
              <span
                className="px-1 rounded text-[0.55rem] font-semibold text-white"
                style={{ backgroundColor: STATE_COLORS[scrubHover.state] ?? "#6b7280" }}
              >
                {scrubHover.state}
              </span>
              {scrubHover.active > 0 && (
                <span className="text-[#5ec77a] text-[0.55rem]">● {scrubHover.active}</span>
              )}
              {scrubHover.events > 0 && (
                <span className="text-[#f59e0b] text-[0.55rem]">⚡{scrubHover.events}</span>
              )}
            </div>
            <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[4px] border-t-border" />
          </div>
        )}
      </div>

      {/* Main panels — CSS Grid for predictable layout */}
      <div className="grid gap-3" style={{
        gridTemplateColumns: width < 820
          ? "1fr"
          : "minmax(280px, 30fr) minmax(320px, 44fr) minmax(240px, 26fr)",
        opacity: panelOpacity,
        transition: panelTransition,
      }}>
        <div>
          <PanelLabel>body · 20-segment MuJoCo · state-coloured glow</PanelLabel>
          <div className="rounded-lg overflow-hidden border bg-[#f9f0d6]">
            <canvas
              ref={bodyCanvasRef}
              className="block w-full"
              role="img"
              aria-label={`20-segment worm body in state ${currentFrame?.state ?? "unknown"} at time ${currentT.toFixed(1)} seconds`}
            />
          </div>
        </div>
        <div>
          <div className="flex items-baseline justify-between gap-2">
            <PanelLabel>brain · {brainViewMode === "3d"
              ? `${trace?.neuron_names?.length ?? 300} neurons · 3D${liveStats ? ` · ${trace?.full_raster ? liveStats.activeAllCount : liveStats.activeCount} firing` : ""}${ntFilter.size > 0 ? ` · filter: ${Array.from(ntFilter).join("/")}` : ""}`
              : `spike raster · 18 readout${liveStats ? ` · ${liveStats.activeCount} active` : ""}`}
            </PanelLabel>
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
              <button
                onClick={() => {
                  const el = brainCanvasRef.current?.parentElement;
                  if (!el) return;
                  if (document.fullscreenElement) {
                    document.exitFullscreen();
                  } else {
                    el.requestFullscreen?.().catch(() => {/* ignore */});
                  }
                }}
                className="rounded px-1 text-[0.6rem] hover:bg-accent"
                title="Toggle fullscreen for brain view"
                aria-label="Toggle brain-view fullscreen"
              >⛶</button>
            </div>
          </div>
          {/* NT filter chips (below title row, above canvas) */}
          {brainViewMode === "3d" && (
            <div className="flex items-center gap-1 text-[0.6rem] mb-1 flex-wrap">
              <span className="text-muted-foreground font-medium">NT filter:</span>
              {Object.keys(NT_CATEGORIES).map((cat) => {
                const active = ntFilter.has(cat);
                const colors: Record<string, string> = {
                  "ACh": "#38bdf8",
                  "Glu": "#a3e635",
                  "GABA": "#f87171",
                  "Modulatory": "#c084fc",
                  "Unknown": "#94a3b8",
                };
                const col = colors[cat] ?? "#94a3b8";
                return (
                  <button
                    key={cat}
                    onClick={() => {
                      const next = new Set(ntFilter);
                      if (active) next.delete(cat); else next.add(cat);
                      setNtFilter(next);
                    }}
                    className="inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 transition-colors"
                    style={{
                      backgroundColor: active ? hexAlpha(col, 0.18) : "transparent",
                      borderColor: active ? col : "hsl(var(--border))",
                      color: active ? "var(--foreground)" : "hsl(var(--muted-foreground))",
                    }}
                  >
                    <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ backgroundColor: col }} />
                    {cat}
                  </button>
                );
              })}
              {ntFilter.size > 0 && (
                <button
                  onClick={() => setNtFilter(new Set())}
                  className="text-muted-foreground hover:text-foreground underline-offset-2 hover:underline ml-1"
                >clear</button>
              )}
            </div>
          )}
          <div className="relative rounded-lg overflow-hidden border bg-[#0a0e1a]">
            <canvas
              ref={brainCanvasRef}
              className={`block w-full ${isDragging ? "cursor-grabbing" : "cursor-crosshair"}`}
              onMouseMove={onBrainMove}
              onMouseLeave={(e) => { onBrainMouseUp(); onBrainLeave(); }}
              onMouseDown={onBrainMouseDown}
              onMouseUp={onBrainMouseUp}
              onClick={onBrainClick}
              onTouchStart={onBrainTouchStart}
              onTouchMove={onBrainTouchMove}
              onTouchEnd={onBrainTouchEnd}
              onTouchCancel={onBrainTouchEnd}
              role="img"
              aria-label={`Brain view: ${brainViewMode === "3d" ? "300 neurons in 3D with active neurons highlighted in green" : "Spike raster of 18 readout neurons over time"}. Click neurons to lock, shift-drag to rotate.`}
            />
            {/* Rotation indicator + reset */}
            <div className="absolute top-2 left-2 flex items-center gap-2 rounded-md bg-[#0f1429]/70 px-2 py-1 text-[0.6rem] text-[#94a3b8]">
              <span>⇄ shift-drag to rotate · {Math.round((brainRot * 180 / Math.PI) % 360)}°</span>
              <button
                onClick={() => setBrainRot(0)}
                className="text-[#a5b4fc] hover:text-[#f2ead3]"
              >reset</button>
            </div>
            {/* Neuron search */}
            {brainViewMode === "3d" && (
              <div className="absolute bottom-2 left-2 w-44">
                <input
                  type="text"
                  value={searchQ}
                  onChange={(e) => setSearchQ(e.target.value)}
                  placeholder="search neurons…"
                  className="w-full rounded-md bg-[#0f1429]/80 border border-[#1e293b] px-2 py-1 text-[0.65rem] text-[#e2e8f0] placeholder-[#64748b] focus:outline-none focus:border-[#a5b4fc]"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && searchMatches[0]) {
                      setLockedNeuron(searchMatches[0].idx);
                      setSearchQ("");
                    } else if (e.key === "Escape") {
                      setSearchQ("");
                    }
                  }}
                />
                {searchMatches.length > 0 && (
                  <div className="mt-1 rounded-md bg-[#0f1429]/95 border border-[#1e293b] overflow-hidden shadow-lg">
                    {searchMatches.map((m) => (
                      <button
                        key={m.name}
                        onClick={() => { setLockedNeuron(m.idx); setSearchQ(""); }}
                        className="block w-full text-left px-2 py-1 text-[0.65rem] text-[#e2e8f0] hover:bg-[#1e293b] flex justify-between items-center gap-2"
                      >
                        <span className="font-mono font-semibold">{m.name}</span>
                        <span className="text-[0.55rem] text-[#64748b] truncate">
                          {m.meta?.nt.replace(/\s*\([^)]+\)/, "") ?? "?"}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {lockedMeta && (
              <div className="absolute top-2 right-2 w-64 max-h-[95%] overflow-y-auto rounded-lg bg-[#0f1429]/95 border border-[#a5b4fc]/40 p-3 shadow-lg text-[0.7rem] text-[#e2e8f0]">
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
                      {lockedMeta.outgoing.slice(0, 5).map(([n, w]) => {
                        const active = liveStats ? liveStats.topMods !== null : false;
                        const readoutIdx = trace?.meta.readout_neurons.indexOf(n) ?? -1;
                        const firing = readoutIdx >= 0 && trace?.raster.some(
                          (e) => e.t > currentT - 0.1 && e.t <= currentT && e.n.includes(readoutIdx)
                        );
                        return (
                          <div key={n} className="pl-2 flex justify-between items-center">
                            <span className="flex items-center gap-1">
                              {firing && <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#5ec77a] animate-pulse" title="firing now" />}
                              {n}
                            </span>
                            <span className="text-[#64748b] font-mono text-[0.6rem]">{w}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {lockedMeta.incoming.length > 0 && (
                    <div className="pt-1 border-t border-[#1e293b]">
                      <div className="text-[#64748b] mb-0.5">top incoming ←</div>
                      {lockedMeta.incoming.slice(0, 5).map(([n, w]) => {
                        const readoutIdx = trace?.meta.readout_neurons.indexOf(n) ?? -1;
                        const firing = readoutIdx >= 0 && trace?.raster.some(
                          (e) => e.t > currentT - 0.1 && e.t <= currentT && e.n.includes(readoutIdx)
                        );
                        return (
                          <div key={n} className="pl-2 flex justify-between items-center">
                            <span className="flex items-center gap-1">
                              {firing && <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#5ec77a] animate-pulse" title="firing now" />}
                              {n}
                            </span>
                            <span className="text-[#64748b] font-mono text-[0.6rem]">{w}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {/* Ego-network mini diagram */}
                  <div className="pt-1 border-t border-[#1e293b]">
                    <div className="text-[#64748b] mb-0.5">ego network</div>
                    <EgoNetwork meta={lockedMeta} />
                  </div>
                  {/* CeNGEN gene-expression ring — P0 #2 */}
                  {cengenPanel && (
                    <div className="pt-1 border-t border-[#1e293b]">
                      <div className="text-[#64748b] mb-0.5 flex justify-between items-baseline">
                        <span>CeNGEN expression</span>
                        <span className="font-mono text-[0.55rem]">Taylor 2021</span>
                      </div>
                      <CengenRing panel={cengenPanel} neuronName={lockedMeta.name} />
                    </div>
                  )}
                  {lockedRateHist && (
                    <div className="pt-1 border-t border-[#1e293b]">
                      <div className="text-[#64748b] mb-0.5 flex justify-between items-center">
                        <span>firing rate · 0.5 s bins</span>
                        <span className="font-mono text-[0.6rem]">peak {lockedRateHist.maxRate}</span>
                      </div>
                      <div className="relative h-8 bg-[#0a0e1a] rounded mt-0.5 overflow-hidden">
                        <svg
                          viewBox="0 0 100 100"
                          preserveAspectRatio="none"
                          className="absolute inset-0 w-full h-full"
                          aria-hidden="true"
                        >
                          {lockedRateHist.bins.map((r, i) => {
                            const x = (i / lockedRateHist.bins.length) * 100;
                            const w = 100 / lockedRateHist.bins.length;
                            const h = (r / lockedRateHist.maxRate) * 100;
                            return (
                              <rect
                                key={i}
                                x={x} y={100 - h}
                                width={Math.max(0.5, w - 0.3)} height={h}
                                fill="#5ec77a" opacity="0.85"
                              />
                            );
                          })}
                          {/* Current-time cursor */}
                          <line
                            x1={(currentT / (trace?.meta.duration_s ?? 1)) * 100}
                            y1={0}
                            x2={(currentT / (trace?.meta.duration_s ?? 1)) * 100}
                            y2={100}
                            stroke="#f2ead3"
                            strokeWidth="0.6"
                            vectorEffect="non-scaling-stroke"
                          />
                        </svg>
                      </div>
                    </div>
                  )}
                  {!lockedRateHist && (
                    <div className="pt-1 border-t border-[#1e293b] text-[0.6rem] text-[#64748b] italic">
                      not in 18-neuron readout — no raster data
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
      <div style={{ opacity: panelOpacity, transition: panelTransition }}>
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
      <div style={{ opacity: panelOpacity, transition: panelTransition }}>
        <PanelLabel>behavioural state · FSM transitions over time · ticks = stimuli</PanelLabel>
        <div className="relative rounded-lg overflow-hidden border bg-[#0a0e1a]">
          <canvas ref={fsmCanvasRef} className="block w-full" />
          <FsmDiagramInset currentState={currentFrame?.state ?? "FORWARD"} />
        </div>
      </div>

      {/* Event probs + legend */}
      <div style={{ opacity: panelOpacity, transition: panelTransition }}>
        <PanelLabel>event probabilities · 8 canonical behavioural transitions</PanelLabel>
        <div className="rounded-lg overflow-hidden border bg-[#0a0e1a]">
          <canvas ref={evCanvasRef} className="block w-full" />
          <canvas ref={evLegendRef} className="block w-full" />
        </div>
      </div>

      {/* Legends row */}
      <div className="rounded-lg border bg-card/30 p-2.5 space-y-2">
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[0.65rem] text-muted-foreground">
          <span className="font-semibold text-foreground w-16">states:</span>
          {Object.entries(STATE_COLORS).map(([name, col]) => (
            <span key={name} className="inline-flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded" style={{ backgroundColor: col }} />
              {name}
            </span>
          ))}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[0.65rem] text-muted-foreground">
          <span className="font-semibold text-foreground w-16">NT (spike glow):</span>
          {[
            { name: "ACh",  col: "#38bdf8" },
            { name: "Glu",  col: "#a3e635" },
            { name: "GABA", col: "#f87171" },
            { name: "Modulatory", col: "#c084fc" },
          ].map(({ name, col }) => {
            const active = ntFilter.has(name);
            return (
              <button
                key={name}
                onClick={() => {
                  const next = new Set(ntFilter);
                  if (active) next.delete(name); else next.add(name);
                  setNtFilter(next);
                }}
                className="inline-flex items-center gap-1 cursor-pointer rounded px-0.5 hover:bg-accent/50 transition-colors"
                style={{ color: active ? "var(--foreground)" : undefined }}
                title={`Click to ${active ? "remove" : "add"} ${name} filter`}
              >
                <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: col, boxShadow: active ? `0 0 4px ${col}` : undefined }} />
                {name}
              </button>
            );
          })}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[0.65rem] text-muted-foreground">
          <span className="font-semibold text-foreground w-16">modulators:</span>
          {Object.entries(MODULATOR_COLORS).map(([name, col]) => (
            <span
              key={name}
              className="inline-flex items-center gap-1 cursor-help"
              title={`releasers: ${(RELEASERS[name] ?? []).join(", ")}`}
              onMouseEnter={() => setHoverModulator(name)}
              onMouseLeave={() => setHoverModulator(null)}
            >
              <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: col }} />
              {name}
            </span>
          ))}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[0.65rem] text-muted-foreground">
          <span className="font-semibold text-foreground w-16">events:</span>
          {Object.entries(EVENT_COLORS).map(([name, col]) => (
            <span key={name} className="inline-flex items-center gap-1">
              <span className="inline-block w-3 h-2 rounded" style={{ backgroundColor: col }} />
              {name.replace(/_/g, " ")}
            </span>
          ))}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[0.65rem] text-muted-foreground pt-1 border-t border-border/50">
          <span className="ml-auto">
            ⌨ <kbd className="px-1 rounded border text-[0.6rem]">space</kbd> play ·
            <kbd className="mx-1 px-1 rounded border text-[0.6rem]">← →</kbd> ±1s ·
            <kbd className="mx-1 px-1 rounded border text-[0.6rem]">, .</kbd> ±frame ·
            <kbd className="mx-1 px-1 rounded border text-[0.6rem]">R</kbd> restart ·
            <kbd className="mx-1 px-1 rounded border text-[0.6rem]">1–5</kbd> scenarios ·
            <kbd className="mx-1 px-1 rounded border text-[0.6rem]">F</kbd> fps ·
            <kbd className="mx-1 px-1 rounded border text-[0.6rem]">shift-drag</kbd> rotate brain
          </span>
        </div>
      </div>

      {loadErr && (
        <div className="text-xs text-destructive">Trace load failed: {loadErr}</div>
      )}

      {/* Shimmer animation styles */}
      <style>{`
        @keyframes celegans-shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
        .celegans-shimmer {
          position: absolute;
          inset: 0;
          background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(165, 180, 252, 0.12) 50%,
            transparent 100%
          );
          animation: celegans-shimmer 1.8s infinite cubic-bezier(0.4, 0, 0.2, 1);
          pointer-events: none;
        }
      `}</style>

      {/* Loading shimmer overlay — shows on top of panels during scenario switch */}
      {loading && (
        <div className="fixed top-2 left-1/2 -translate-x-1/2 z-40 rounded-full bg-card border shadow-md px-3 py-1 text-[0.65rem] font-medium text-muted-foreground flex items-center gap-2 overflow-hidden relative" aria-live="polite">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          loading {SCENARIOS[scenario].label.toLowerCase()}…
          <div className="celegans-shimmer" />
        </div>
      )}

      {/* FPS overlay */}
      {showFps && (
        <div className="fixed top-4 right-4 z-50 rounded bg-black/80 text-white text-xs px-2 py-1 font-mono">
          {Math.round(fpsRef.current.fps)} fps
        </div>
      )}

      {/* Help overlay */}
      {showHelp && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          onClick={() => setShowHelp(false)}
          role="dialog"
          aria-labelledby="dashboard-help-title"
          aria-modal="true"
        >
          <div
            className="max-w-xl w-full rounded-xl bg-card border shadow-2xl p-5 text-sm"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-3">
              <h2 id="dashboard-help-title" className="text-lg font-semibold">Dashboard guide</h2>
              <button
                onClick={() => setShowHelp(false)}
                className="text-muted-foreground hover:text-foreground text-xl leading-none"
                aria-label="Close help"
              >×</button>
            </div>
            <div className="space-y-2.5 text-[0.8rem]">
              <div>
                <div className="font-semibold text-foreground mb-0.5">Panels</div>
                <ul className="pl-4 list-disc text-muted-foreground space-y-0.5">
                  <li><b>Body</b> — MuJoCo 20-segment worm, colored by FSM state.</li>
                  <li><b>Brain</b> — 300 neurons in 3D. Active = bright glow (colored by NT).</li>
                  <li><b>Arena</b> — 2D agar field + food patch (chemotaxis scenario only).</li>
                  <li><b>Modulator strip</b> — 9 peptide/monoamine concentrations × time.</li>
                  <li><b>FSM + events</b> — behavioural-state timeline with stim + event carets.</li>
                </ul>
              </div>
              <div>
                <div className="font-semibold text-foreground mb-0.5">Interactions</div>
                <ul className="pl-4 list-disc text-muted-foreground space-y-0.5">
                  <li>Click a neuron to lock — shows class/NT, 1-hop edges, firing-rate history.</li>
                  <li>Shift-drag the brain to rotate around the AP axis.</li>
                  <li>Hover a circuit or modulator badge to highlight its neurons.</li>
                  <li>Type in the search box to jump to a neuron by name.</li>
                  <li>Scrub the timeline — hover shows state + time preview.</li>
                </ul>
              </div>
              <div>
                <div className="font-semibold text-foreground mb-0.5">Keyboard</div>
                <div className="text-muted-foreground text-[0.75rem] flex flex-wrap gap-x-3 gap-y-1">
                  <span><kbd className="px-1 rounded border">space</kbd> play/pause</span>
                  <span><kbd className="px-1 rounded border">← →</kbd> ±1 s</span>
                  <span><kbd className="px-1 rounded border">, .</kbd> frame step</span>
                  <span><kbd className="px-1 rounded border">R</kbd> restart</span>
                  <span><kbd className="px-1 rounded border">1–5</kbd> scenarios</span>
                  <span><kbd className="px-1 rounded border">F</kbd> fps</span>
                  <span><kbd className="px-1 rounded border">E</kbd> edges</span>
                  <span><kbd className="px-1 rounded border">V</kbd> view</span>
                  <span><kbd className="px-1 rounded border">[ ]</kbd> next neuron</span>
                  <span><kbd className="px-1 rounded border">?</kbd> this help</span>
                  <span><kbd className="px-1 rounded border">Esc</kbd> close</span>
                </div>
              </div>
              <div className="pt-1 text-muted-foreground text-[0.7rem] italic">
                All scenarios are pre-rendered from Brian2 + MuJoCo simulations.
                v3 LIF brain; Tier 1 graded-brain stack available in the Python
                backend but awaits v3.4 classifier retraining before shipping.
              </div>
            </div>
          </div>
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
          <div className="pt-2 border-t border-border/40">
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="text-foreground font-semibold">Cite this simulator</span>
              <button
                onClick={() => {
                  const cite = `Ravi, R. (${new Date().getFullYear()}). Connectome-constrained C. elegans digital twin (v3). https://rohitravi.com/projects/c-elegans-multimodal`;
                  navigator.clipboard?.writeText(cite).then(() => {
                    setCopiedCite(true);
                    setTimeout(() => setCopiedCite(false), 1600);
                  });
                }}
                className="rounded-md border px-2 py-0.5 text-[0.65rem] hover:bg-accent focus:outline-none focus:ring-2 focus:ring-primary"
                aria-label="Copy citation to clipboard"
              >{copiedCite ? "✓ copied" : "copy citation"}</button>
            </div>
            <code className="block bg-background/60 rounded p-2 text-[0.7rem] whitespace-pre-wrap">
              Ravi, R. ({new Date().getFullYear()}). Connectome-constrained C. elegans digital twin (v3). https://rohitravi.com/projects/c-elegans-multimodal
            </code>
          </div>
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

/**
 * P0 #2 — Polar bar chart of CeNGEN expression for the locked neuron.
 * Each group (NT marker, channels, receptors, peptides, etc.) occupies
 * an angular wedge; individual genes within the group are radial bars.
 * Bar height is log(TPM + 1) normalised to the gene's cross-neuron max
 * so rarely-expressed genes still show rather than vanishing under
 * high-TPM neighbours.
 */
function CengenRing({ panel, neuronName }: {
  panel: CengenPanel; neuronName: string;
}) {
  const row = panel.expression[neuronName];
  if (!row) {
    return (
      <div className="text-[0.6rem] text-[#64748b] italic py-1">
        no CeNGEN data for {neuronName} (ventral-cord motor neurons were
        not individually profiled in the Taylor 2021 release)
      </div>
    );
  }
  const W = 220, H = 160;
  const cx = W / 2, cy = H / 2 + 6;
  const rInner = 16, rOuter = 64;

  // Flatten panel genes into angular positions, keeping group ordering
  type Slot = { gene: string; group: string; angle: number };
  const slots: Slot[] = [];
  const groupColors: Record<string, string> = {
    "NT marker":    "#10b981",
    "Voltage Ca":   "#f59e0b",
    "Voltage K":    "#38bdf8",
    "Voltage Na":   "#6366f1",
    "AChR":         "#0ea5e9",
    "GluR/GluCl":   "#a3e635",
    "GABAR":        "#f87171",
    "Peptide rx":   "#c084fc",
    "Monoamine rx": "#ec4899",
    "Peptide gene": "#a78bfa",
    "Sensory GCY":  "#fbbf24",
    "CNG":          "#fde047",
    "TRP":          "#fb923c",
    "DEG/ENaC":     "#64748b",
    "Innexin":      "#94a3b8",
    "Insulin/FOXO": "#14b8a6",
  };
  const groups = panel._meta.groups;
  // Compute total gene count for angular normalisation
  let total = 0;
  for (const g of groups) total += panel._meta.genes_by_group[g]?.length ?? 0;
  let slotIdx = 0;
  const groupArcs: Array<{ name: string; startAngle: number; endAngle: number; color: string }> = [];
  for (const g of groups) {
    const genes = panel._meta.genes_by_group[g] ?? [];
    const startAngle = (slotIdx / total) * Math.PI * 2 - Math.PI / 2;
    for (const gene of genes) {
      const angle = (slotIdx / total) * Math.PI * 2 - Math.PI / 2;
      slots.push({ gene, group: g, angle });
      slotIdx++;
    }
    const endAngle = (slotIdx / total) * Math.PI * 2 - Math.PI / 2;
    groupArcs.push({ name: g, startAngle, endAngle, color: groupColors[g] ?? "#94a3b8" });
  }

  const maxMap = panel._meta.gene_max_tpm;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      {/* Group arc backgrounds */}
      {groupArcs.map((g) => {
        if (g.endAngle === g.startAngle) return null;
        const r = rOuter + 5;
        const x1 = cx + r * Math.cos(g.startAngle);
        const y1 = cy + r * Math.sin(g.startAngle);
        const x2 = cx + r * Math.cos(g.endAngle);
        const y2 = cy + r * Math.sin(g.endAngle);
        const large = g.endAngle - g.startAngle > Math.PI ? 1 : 0;
        return (
          <path
            key={`g-${g.name}`}
            d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`}
            stroke={g.color} strokeWidth={2} fill="none" opacity={0.7}
          />
        );
      })}
      {/* Inner circle */}
      <circle cx={cx} cy={cy} r={rInner} fill="#0f1429" stroke="rgba(100,116,139,0.4)" />
      <text x={cx} y={cy + 3} textAnchor="middle" fontSize="8.5" fill="#e2e8f0" fontWeight="bold">
        {neuronName}
      </text>
      {/* Bars */}
      {slots.map((s) => {
        const tpm = row[s.gene] ?? 0;
        const maxT = maxMap[s.gene] ?? 1;
        const norm = tpm > 0 ? Math.log(tpm + 1) / Math.log(maxT + 1) : 0;
        const r0 = rInner + 1;
        const r1 = r0 + norm * (rOuter - rInner - 2);
        const x1 = cx + r0 * Math.cos(s.angle);
        const y1 = cy + r0 * Math.sin(s.angle);
        const x2 = cx + r1 * Math.cos(s.angle);
        const y2 = cy + r1 * Math.sin(s.angle);
        const color = groupColors[s.group] ?? "#94a3b8";
        return (
          <g key={`bar-${s.gene}`}>
            <line
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={color} strokeWidth={1.1}
              opacity={tpm > 0 ? 0.9 : 0.15}
            />
            <title>{`${s.gene} (${s.group}): ${tpm.toFixed(2)} TPM`}</title>
          </g>
        );
      })}
      {/* Group labels — positioned at mid-arc, only for groups with ≥2 genes */}
      {groupArcs.map((g) => {
        const n = panel._meta.genes_by_group[g.name]?.length ?? 0;
        if (n < 2) return null;
        const mid = (g.startAngle + g.endAngle) / 2;
        const lx = cx + (rOuter + 12) * Math.cos(mid);
        const ly = cy + (rOuter + 12) * Math.sin(mid);
        return (
          <text
            key={`lbl-${g.name}`}
            x={lx} y={ly}
            textAnchor={Math.cos(mid) > 0.2 ? "start" : Math.cos(mid) < -0.2 ? "end" : "middle"}
            fontSize="6.5" fill={g.color}
            dominantBaseline="middle"
          >
            {g.name}
          </text>
        );
      })}
    </svg>
  );
}

function EgoNetwork({ meta }: { meta: NeuronMeta }) {
  const outN = meta.outgoing.slice(0, 5);
  const inN = meta.incoming.slice(0, 5);
  const nOut = outN.length;
  const nIn = inN.length;
  const W = 220, H = 110;
  const cx = W / 2, cy = H / 2;
  const ringR = 42;
  const outMax = Math.max(1, ...outN.map(([, w]) => w));
  const inMax = Math.max(1, ...inN.map(([, w]) => w));
  // Outgoing on right hemisphere (angles -π/2 to π/2), incoming on left
  const outPts = outN.map(([n, w], i) => {
    const angle = -Math.PI / 2 + (i + 1) / (nOut + 1) * Math.PI;
    return { name: n, w, x: cx + ringR * Math.cos(angle), y: cy + ringR * Math.sin(angle) };
  });
  const inPts = inN.map(([n, w], i) => {
    const angle = Math.PI / 2 + (i + 1) / (nIn + 1) * Math.PI;
    return { name: n, w, x: cx + ringR * Math.cos(angle), y: cy + ringR * Math.sin(angle) };
  });
  const sign = meta.sign;
  const coreColor = sign > 0 ? "#10b981" : sign < 0 ? "#ef4444" : "#a5b4fc";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" aria-hidden="true">
      {/* Outgoing edges */}
      {outPts.map((p) => (
        <line
          key={`o-${p.name}`}
          x1={cx} y1={cy} x2={p.x} y2={p.y}
          stroke="#10b981"
          strokeOpacity={0.25 + 0.65 * (p.w / outMax)}
          strokeWidth={0.5 + 1.2 * (p.w / outMax)}
        />
      ))}
      {/* Incoming edges */}
      {inPts.map((p) => (
        <line
          key={`i-${p.name}`}
          x1={p.x} y1={p.y} x2={cx} y2={cy}
          stroke="#a5b4fc"
          strokeOpacity={0.25 + 0.65 * (p.w / inMax)}
          strokeWidth={0.5 + 1.2 * (p.w / inMax)}
        />
      ))}
      {/* Partner nodes */}
      {[...outPts, ...inPts].map((p, i) => (
        <g key={`n-${p.name}-${i}`}>
          <circle cx={p.x} cy={p.y} r={3} fill="#64748b" />
          <text x={p.x} y={p.y - 5} textAnchor="middle" fontSize="7" fill="#cbd5e1">
            {p.name}
          </text>
        </g>
      ))}
      {/* Center (locked) */}
      <circle cx={cx} cy={cy} r={6} fill={coreColor} />
      <text x={cx} y={cy + 16} textAnchor="middle" fontSize="7.5" fill="#e2e8f0" fontWeight="bold">
        {meta.name}
      </text>
      {/* Hemisphere labels */}
      <text x={2} y={H - 4} fontSize="6.5" fill="#64748b">← in</text>
      <text x={W - 16} y={H - 4} fontSize="6.5" fill="#64748b">out →</text>
    </svg>
  );
}

function PanelLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-1.5 text-[0.7rem] uppercase tracking-wider text-muted-foreground font-medium">
      {children}
    </div>
  );
}

function StatCard({ label, value, sub, accent, spark, sparkColor }: {
  label: string; value: string; sub?: string; accent?: string;
  spark?: number[]; sparkColor?: string;
}) {
  return (
    <div className="rounded-lg border bg-card/40 px-3 py-1.5 relative overflow-hidden">
      {spark && spark.length > 1 && (
        <Sparkline values={spark} color={sparkColor ?? "#94a3b8"} />
      )}
      <div className="relative z-10">
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
    </div>
  );
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  const n = values.length;
  const max = Math.max(1, ...values);
  const min = Math.min(...values);
  const rng = Math.max(1e-6, max - min);
  // Normalise so max -> 0 (top) and min -> 1 (bottom) of sparkline band
  const pts = values.map((v, i) => {
    const x = (i / (n - 1)) * 100;
    const y = 100 - ((v - min) / rng) * 100;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const areaPath = `M 0,100 L ${pts.join(" L ")} L 100,100 Z`;
  const linePath = `M ${pts.join(" L ")}`;
  const gradId = `sparkGrad-${color.replace("#", "")}`;
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className="absolute inset-0 w-full h-full opacity-[0.35]"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.55" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#${gradId})`} />
      <path d={linePath} fill="none" stroke={color} strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export default CelegansDashboard;
