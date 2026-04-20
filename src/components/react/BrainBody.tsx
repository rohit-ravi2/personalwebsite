import * as React from "react";
import { useEffect, useRef, useState } from "react";

/**
 * BrainBody — Phase 3c closed-loop demo.
 *
 * Shows a Brian2 LIF brain (300 neurons, Cook 2019 connectome +
 * Loer & Rand 2022 NT identity) driving a MuJoCo C. elegans body
 * through event classifiers fit from Atanas 2023 calcium imaging.
 *
 * Architecture mirrors the Shiu et al. 2024 Drosophila pattern:
 *   sensory → LIF brain → motor-neuron readout → event classifier
 *   → FSM → state-specific CPG → MuJoCo body.
 *
 * Four pre-rendered scenarios:
 *   spontaneous  — no stimulus
 *   touch        — ALM/AVM mechanoreceptor stim
 *   osmotic_shock — ASH polymodal avoidance stim
 *   food         — ASI/ASJ/ADF feeding-state tonic stim
 */

type Scenario = "spontaneous" | "touch" | "osmotic_shock" | "food";

type BrainBodyTrace = {
  scenario: string;
  meta: {
    brain_sync_ms: number;
    num_segments: number;
    num_frames: number;
    duration_s: number;
    readout_neurons: string[];
    events_tracked: string[];
    states: string[];
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
};

const NUM_SEGMENTS = 20;

const STATE_COLORS: Record<string, string> = {
  FORWARD: "#2f5233",
  REVERSE: "#b94b4b",
  OMEGA: "#8b5cf6",
  PIROUETTE: "#e76f51",
  QUIESCENT: "#6b7280",
};

const SCENARIO_META: Record<Scenario, { label: string; desc: string }> = {
  spontaneous: {
    label: "Spontaneous",
    desc: "No stimulus. Baseline behavioural distribution.",
  },
  touch: {
    label: "Head touch",
    desc: "Touch stimulus at t=5s → ALM/AVM → expected reversal (Chalfie 1981).",
  },
  osmotic_shock: {
    label: "Osmotic shock",
    desc: "High-osmolarity stim at t=5s → ASH polymodal avoidance (Hart 1995).",
  },
  food: {
    label: "Food",
    desc: "Tonic food signal from t=2s → ASI/ASJ/ADF feeding-state neurons.",
  },
};

function segmentWidth(i: number, total: number): number {
  const t = i / (total - 1);
  return 4 + 10 * Math.sin(Math.PI * t);
}

function drawWorm(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  segs: Array<{ x: number; y: number }>,
  state: string,
) {
  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createRadialGradient(w / 2, h / 2, 40, w / 2, h / 2, Math.max(w, h));
  bg.addColorStop(0, "rgba(247, 237, 211, 0.95)");
  bg.addColorStop(1, "rgba(229, 215, 185, 0.92)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Subtle grid
  ctx.strokeStyle = "rgba(26, 42, 74, 0.06)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= w; x += 32) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
  for (let y = 0; y <= h; y += 32) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
  ctx.stroke();

  const color = STATE_COLORS[state] ?? "#2f5233";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let i = 0; i < segs.length - 1; i++) {
    ctx.strokeStyle = i === 0 ? "#1a2a4a" : color;
    ctx.lineWidth = segmentWidth(i, segs.length);
    ctx.beginPath();
    ctx.moveTo(segs[i].x, segs[i].y);
    ctx.lineTo(segs[i + 1].x, segs[i + 1].y);
    ctx.stroke();
  }
}

function drawRaster(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  raster: Array<{ t: number; n: number[] }>,
  currentT: number,
  windowS: number,
  nNeurons: number,
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1117";
  ctx.fillRect(0, 0, w, h);

  // Time axis: rightmost = currentT, leftmost = currentT - windowS
  const tMin = currentT - windowS;
  const tMax = currentT;
  const pxPerS = w / windowS;
  const rowH = h / Math.max(nNeurons, 1);

  ctx.fillStyle = "#5ec77a";
  for (const tick of raster) {
    if (tick.t < tMin || tick.t > tMax) continue;
    const x = (tick.t - tMin) * pxPerS;
    for (const neuron of tick.n) {
      const y = neuron * rowH;
      ctx.fillRect(x, y, 2, Math.max(1.5, rowH - 1));
    }
  }
  // "Now" line
  ctx.strokeStyle = "#f2ead3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w - 1, 0);
  ctx.lineTo(w - 1, h);
  ctx.stroke();
}

export function BrainBody() {
  const bodyCanvasRef = useRef<HTMLCanvasElement>(null);
  const rasterCanvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [scenario, setScenario] = useState<Scenario>("spontaneous");
  const [trace, setTrace] = useState<BrainBodyTrace | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [currentT, setCurrentT] = useState(0);
  const [width, setWidth] = useState(760);

  const traceRef = useRef<BrainBodyTrace | null>(null);
  const pausedRef = useRef(paused);
  const speedRef = useRef(playbackSpeed);
  const currentTRef = useRef(0);
  traceRef.current = trace;
  pausedRef.current = paused;
  speedRef.current = playbackSpeed;
  currentTRef.current = currentT;

  // Fetch trace when scenario changes
  useEffect(() => {
    let cancel = false;
    setTrace(null);
    setLoadErr(null);
    setCurrentT(0);
    fetch(`/data/wormbody-brain-${scenario}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: BrainBodyTrace) => {
        if (cancel) return;
        setTrace(d);
      })
      .catch((e) => {
        if (cancel) return;
        setLoadErr(String(e));
      });
    return () => { cancel = true; };
  }, [scenario]);

  // Responsive width
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setWidth(Math.max(320, Math.round(e.contentRect.width)));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Animation loop
  useEffect(() => {
    const bodyCanvas = bodyCanvasRef.current;
    const rasterCanvas = rasterCanvasRef.current;
    if (!bodyCanvas || !rasterCanvas) return;
    const bctx = bodyCanvas.getContext("2d");
    const rctx = rasterCanvas.getContext("2d");
    if (!bctx || !rctx) return;

    let rafId = 0;
    let last = performance.now();

    const bodyW = Math.floor(width * 0.6);
    const rasterW = width - bodyW - 12;
    const H = 360;

    const draw = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;

      const tr = traceRef.current;
      if (bodyCanvas.width !== bodyW) bodyCanvas.width = bodyW;
      if (bodyCanvas.height !== H) bodyCanvas.height = H;
      if (rasterCanvas.width !== rasterW) rasterCanvas.width = rasterW;
      if (rasterCanvas.height !== H) rasterCanvas.height = H;

      if (!tr) {
        bctx.fillStyle = "rgba(247, 237, 211, 0.95)";
        bctx.fillRect(0, 0, bodyW, H);
        bctx.fillStyle = "#6b5e3d";
        bctx.font = "14px sans-serif";
        bctx.textAlign = "center";
        bctx.fillText(
          loadErr ? `Trace load failed: ${loadErr}` : "Loading trace…",
          bodyW / 2, H / 2,
        );
        rctx.fillStyle = "#0f1117";
        rctx.fillRect(0, 0, rasterW, H);
        rafId = requestAnimationFrame(draw);
        return;
      }

      if (!pausedRef.current) {
        currentTRef.current = (currentTRef.current + dt * speedRef.current)
                               % tr.meta.duration_s;
        setCurrentT(currentTRef.current);
      }

      const t = currentTRef.current;
      // Find nearest body frame
      const idx = Math.min(
        tr.frames.length - 1,
        Math.floor((t / tr.meta.duration_s) * tr.frames.length),
      );
      const frame = tr.frames[idx];

      // Render body — center + scale
      const pxPerSimM = Math.min(bodyW, H) * 0.5;
      let cx = 0, cy = 0;
      for (const [x, y] of frame.positions) { cx += x; cy += y; }
      cx /= frame.positions.length;
      cy /= frame.positions.length;
      const ox = bodyW / 2 - cx * pxPerSimM;
      const oy = H / 2 - cy * pxPerSimM;
      const segs = frame.positions.map(([x, y]) => ({
        x: ox + x * pxPerSimM, y: oy + y * pxPerSimM,
      }));
      drawWorm(bctx, bodyW, H, segs, frame.state);

      // Render raster
      drawRaster(
        rctx, rasterW, H, tr.raster, t,
        /* windowS */ 10,
        tr.meta.readout_neurons.length,
      );

      rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [width, loadErr]);

  const currentFrame = trace
    ? trace.frames[Math.min(
        trace.frames.length - 1,
        Math.floor((currentT / trace.meta.duration_s) * trace.frames.length),
      )]
    : null;

  return (
    <div className="my-6 flex flex-col gap-3" ref={wrapRef}>
      {/* Scenario toggle */}
      <div className="inline-flex flex-wrap rounded-lg border p-0.5 text-xs w-fit self-start gap-0.5">
        {(Object.keys(SCENARIO_META) as Scenario[]).map((s) => (
          <button
            key={s}
            onClick={() => setScenario(s)}
            className={`rounded-md px-3 py-1.5 font-medium transition-colors ${
              scenario === s ? "bg-primary text-primary-foreground" : "hover:bg-accent"
            }`}
          >
            {SCENARIO_META[s].label}
          </button>
        ))}
      </div>

      {/* Canvas pair */}
      <div className="flex gap-3 rounded-lg border overflow-hidden">
        <canvas ref={bodyCanvasRef} className="block" />
        <canvas ref={rasterCanvasRef} className="block" />
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <button onClick={() => setPaused((v) => !v)}
                className="rounded border px-3 py-1 hover:bg-accent">
          {paused ? "Play" : "Pause"}
        </button>
        <label className="flex items-center gap-1">
          <span>Speed · <span className="tabular-nums">{playbackSpeed.toFixed(1)}×</span></span>
          <input type="range" min="0.25" max="3.0" step="0.25"
                 value={playbackSpeed}
                 onChange={(e) => setPlaybackSpeed(+e.target.value)}
                 className="accent-primary w-32"/>
        </label>
        <span className="tabular-nums text-muted-foreground ml-auto">
          t = {currentT.toFixed(1)}s / {trace?.meta.duration_s ?? 0}s
          {currentFrame && (
            <> · state = <span style={{ color: STATE_COLORS[currentFrame.state] }}>
              {currentFrame.state}
            </span></>
          )}
        </span>
      </div>

      {/* Scenario description */}
      <div className="text-xs text-muted-foreground">
        <strong>{SCENARIO_META[scenario].label}:</strong>{" "}
        {SCENARIO_META[scenario].desc}
      </div>

      {/* Attribution */}
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer font-medium">Sources &amp; attribution</summary>
        <div className="mt-2 space-y-1 pl-3">
          <div><strong>Brain:</strong> 300-neuron LIF in Brian2, wiring from Cook et al. 2019 (Nature) hermaphrodite connectome, neurotransmitter identity from Loer &amp; Rand 2022 (WormAtlas). Architecture mirrors Shiu et al. 2024 (<em>Nature</em>) Drosophila brain model.</div>
          <div><strong>Body:</strong> 20-segment MuJoCo wormbody with Boyle-Berri-Cohen 2012 CPG parameters and resistive-force-theory drag.</div>
          <div><strong>Event classifiers:</strong> Logistic regression fit on Atanas et al. 2023 (DANDI 000776) paired calcium+behavior recordings across 10 worms. Cross-worm generalization validated (train 1-8, test 9-10) on 18-neuron strict intersection readout.</div>
          <div><strong>Integration pattern:</strong> brain-body sync cadence after Eon Systems 2026 embodied-fly demonstration.</div>
          <div className="pt-1 italic">v1.5: per-neuron affine distribution calibration maps Brian2 synthetic-calcium moments onto the Atanas ΔF/F training distribution; biological face-validity improved (osmotic → omega+pirouette surge, food → quiescence bias). Remaining: classifier joint structure still differs post-moment-match, so event thresholds remain empirically set. Per-edge receptor mapping beyond the 22 sign overrides is v2.</div>
        </div>
      </details>
    </div>
  );
}

export default BrainBody;
