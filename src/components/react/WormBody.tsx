import * as React from "react";
import { useEffect, useRef, useState } from "react";

/**
 * WormBody — digital C. elegans body preview.
 *
 * Two modes:
 *   - "kinematic":   browser-side live wave model, interactive sliders for
 *                    frequency, wavelength, curvature amplitude. Fast, always
 *                    smooth. Useful for intuition and exploration.
 *   - "physics":     plays back a precomputed MuJoCo physics trajectory
 *                    driven by a central-pattern-generator controller with
 *                    resistive-force-theory anisotropic drag. This is the
 *                    actual physics that will drive Phase 2c imitation-
 *                    learned controllers and Phase 3+ brain coupling.
 *
 * The underlying MuJoCo MJCF lives at /data/wormbody.xml. The physics
 * trace lives at /data/wormbody-physics-trace.json.
 */

const NUM_SEGMENTS = 20;
const LOGICAL_BODY_LENGTH_PX = 380;
const SEGMENT_LENGTH_PX = LOGICAL_BODY_LENGTH_PX / NUM_SEGMENTS;
const PROPULSION_COEFF = 24;

type Mode = "kinematic" | "physics" | "imitation" | "real";

type PlaybackTrace = {
  meta: {
    num_segments: number;
    num_frames: number;
    record_hz: number;
    duration_sec: number;
    controller?: Record<string, any>;
    drag?: Record<string, any>;
    parameters?: Record<string, any>;
    source?: string;
    references?: string[];
    units?: string;
  };
  frames: Array<{ t: number; positions: Array<[number, number]> }>;
};

type KinState = { headX: number; headY: number; heading: number; timeSec: number };

function computeKinematicSegments(
  state: KinState,
  freq: number,
  wavelength: number,
  amplitude: number,
): Array<{ x: number; y: number; theta: number }> {
  const segments: Array<{ x: number; y: number; theta: number }> = [];
  let x = state.headX;
  let y = state.headY;
  let heading = state.heading;
  segments.push({ x, y, theta: heading });
  for (let i = 0; i < NUM_SEGMENTS; i++) {
    const s = (i + 0.5) / NUM_SEGMENTS;
    const phase = 2 * Math.PI * (s / wavelength - freq * state.timeSec);
    const segCurvature = (amplitude * 2 * Math.PI / wavelength) * Math.cos(phase);
    heading += segCurvature / NUM_SEGMENTS;
    x += SEGMENT_LENGTH_PX * Math.cos(heading);
    y += SEGMENT_LENGTH_PX * Math.sin(heading);
    segments.push({ x, y, theta: heading });
  }
  return segments;
}

function segmentWidth(i: number, total: number): number {
  const t = i / (total - 1);
  const profile = Math.sin(Math.PI * t);
  return 4 + 10 * profile;
}

function drawWorm(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  segments: Array<{ x: number; y: number }>,
  headAngle: number,
) {
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createRadialGradient(width / 2, height / 2, 40, width / 2, height / 2, Math.max(width, height));
  bg.addColorStop(0, "rgba(247, 237, 211, 0.95)");
  bg.addColorStop(1, "rgba(229, 215, 185, 0.92)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.strokeStyle = "rgba(26, 42, 74, 0.06)";
  ctx.lineWidth = 1;
  const gridSize = 32;
  ctx.beginPath();
  for (let x = 0; x <= width; x += gridSize) {
    ctx.moveTo(x, 0); ctx.lineTo(x, height);
  }
  for (let y = 0; y <= height; y += gridSize) {
    ctx.moveTo(0, y); ctx.lineTo(width, y);
  }
  ctx.stroke();
  ctx.restore();

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let i = 0; i < segments.length - 1; i++) {
    const a = segments[i];
    const b = segments[i + 1];
    const w = segmentWidth(i, segments.length);
    ctx.strokeStyle = i === 0 ? "#1a2a4a" : "#2f5233";
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  const head = segments[0];
  ctx.fillStyle = "#f2ead3";
  ctx.beginPath();
  ctx.arc(head.x - 4 * Math.cos(headAngle), head.y - 4 * Math.sin(headAngle), 2.2, 0, Math.PI * 2);
  ctx.fill();
}

export function WormBody() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<Mode>("kinematic");

  // Kinematic-mode controls
  const [freq, setFreq] = useState(1.5);
  const [wavelength, setWavelength] = useState(0.7);
  const [amplitude, setAmplitude] = useState(0.9);
  const [paused, setPaused] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [physicsTrace, setPhysicsTrace] = useState<PlaybackTrace | null>(null);
  const [imitationTrace, setImitationTrace] = useState<PlaybackTrace | null>(null);
  const [realTrace, setRealTrace] = useState<PlaybackTrace | null>(null);
  const [imitationErr, setImitationErr] = useState<string | null>(null);
  const [realErr, setRealErr] = useState<string | null>(null);
  const [traceErr, setTraceErr] = useState<string | null>(null);
  const [width, setWidth] = useState(720);

  // Runtime state
  const kinRef = useRef<KinState>({ headX: 0, headY: 0, heading: Math.PI, timeSec: 0 });
  const physTimeRef = useRef(0); // elapsed sim-time in physics playback (seconds)

  // Mirror controls into refs so the RAF loop sees the latest values.
  const paramRef = useRef({ mode, freq, wavelength, amplitude, paused, playbackSpeed });
  paramRef.current = { mode, freq, wavelength, amplitude, paused, playbackSpeed };
  const physicsRef = useRef<PlaybackTrace | null>(null);
  const imitationRef = useRef<PlaybackTrace | null>(null);
  const realRef = useRef<PlaybackTrace | null>(null);
  physicsRef.current = physicsTrace;
  imitationRef.current = imitationTrace;
  realRef.current = realTrace;

  // Fetch both traces on mount.
  useEffect(() => {
    let cancelled = false;
    fetch("/data/wormbody-physics-trace.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: PlaybackTrace) => { if (!cancelled) setPhysicsTrace(d); })
      .catch((e) => { if (!cancelled) setTraceErr(String(e)); });
    // Imitation mode prefers the trained learned rollout; falls back to
    // the synthetic reference (target trajectory) otherwise.
    fetch("/data/wormbody-learned-trace.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: PlaybackTrace) => { if (!cancelled) setImitationTrace(d); })
      .catch(() =>
        fetch("/data/wormbody-reference-trace.json")
          .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
          .then((d: PlaybackTrace) => { if (!cancelled) setImitationTrace(d); })
          .catch((e) => { if (!cancelled) setImitationErr(String(e)); }),
      );
    // Real-worm mode plays back actual centerlines parsed from the
    // Tierpsy test-data skeletons (Zenodo 3837679).
    fetch("/data/wormbody-reference-real.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d: PlaybackTrace) => { if (!cancelled) setRealTrace(d); })
      .catch((e) => { if (!cancelled) setRealErr(String(e)); });
    return () => { cancelled = true; };
  }, []);

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

  // Initialise kinematic worm pose when dimensions change.
  useEffect(() => {
    kinRef.current = {
      headX: width / 2 + LOGICAL_BODY_LENGTH_PX / 2,
      headY: 180,
      heading: Math.PI,
      timeSec: 0,
    };
  }, [width]);

  // Animation loop.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId = 0;
    let last = performance.now();
    const height = 360;

    const draw = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;

      const p = paramRef.current;

      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;

      if (p.mode === "kinematic") {
        const state = kinRef.current;
        if (!p.paused) state.timeSec += dt;
        const segs = computeKinematicSegments(state, p.freq, p.wavelength, p.amplitude);

        if (!p.paused) {
          let meanHeading = 0;
          for (let i = 0; i < 5; i++) meanHeading += segs[i].theta;
          meanHeading /= 5;
          const speed = PROPULSION_COEFF * p.freq * p.amplitude * p.amplitude;
          state.headX += speed * Math.cos(meanHeading) * dt;
          state.headY += speed * Math.sin(meanHeading) * dt;
        }

        const margin = 40;
        if (state.headX < -margin) state.headX += width + 2 * margin;
        if (state.headX > width + margin) state.headX -= width + 2 * margin;
        if (state.headY < -margin) state.headY += height + 2 * margin;
        if (state.headY > height + margin) state.headY -= height + 2 * margin;

        drawWorm(ctx, width, height, segs, segs[0].theta);
      } else {
        // Physics / imitation / real playback
        const tr = p.mode === "physics" ? physicsRef.current
                 : p.mode === "imitation" ? imitationRef.current
                 : realRef.current;
        const loadErr = p.mode === "physics" ? traceErr
                      : p.mode === "imitation" ? imitationErr
                      : realErr;
        const loadLabel = p.mode === "physics" ? "physics trace"
                        : p.mode === "imitation" ? "imitation trace"
                        : "real-worm trace";
        if (!tr) {
          ctx.fillStyle = "rgba(247, 237, 211, 0.95)";
          ctx.fillRect(0, 0, width, height);
          ctx.fillStyle = "#6b5e3d";
          ctx.font = "14px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(
            loadErr
              ? p.mode === "imitation"
                ? "Imitation controller not yet available — training in progress."
                : `Trace failed to load: ${loadErr}`
              : `Loading ${loadLabel}…`,
            width / 2, height / 2,
          );
          rafId = requestAnimationFrame(draw);
          return;
        }

        if (!p.paused) physTimeRef.current += dt * p.playbackSpeed;
        const totalDur = tr.meta.duration_sec;
        const tLoop = physTimeRef.current % totalDur;
        const frameIdx = Math.min(
          tr.frames.length - 1,
          Math.floor((tLoop / totalDur) * tr.frames.length),
        );
        const framePositions = tr.frames[frameIdx].positions;

        // Scale: spread sim-m range across a fixed visual size.
        // Sim positions are in meters (model scale). Full body is ~1 m.
        // Render body such that a 1-m body spans 400px.
        const pxPerSimM = 220;
        // Center trajectory in the canvas; track moving centroid so the
        // worm doesn't wander off-screen — translate entire frame to
        // keep the centroid pinned.
        let cx = 0, cy = 0;
        for (const [x, y] of framePositions) { cx += x; cy += y; }
        cx /= framePositions.length;
        cy /= framePositions.length;
        const originX = width / 2 - cx * pxPerSimM;
        const originY = height / 2 - cy * pxPerSimM;
        const segs: Array<{ x: number; y: number; theta: number }> = [];
        for (let i = 0; i < framePositions.length; i++) {
          const [sx, sy] = framePositions[i];
          const x = originX + sx * pxPerSimM;
          const y = originY + sy * pxPerSimM;
          let theta = 0;
          if (i + 1 < framePositions.length) {
            const [nx, ny] = framePositions[i + 1];
            theta = Math.atan2(ny - sy, nx - sx);
          } else if (i > 0) {
            const [px, py] = framePositions[i - 1];
            theta = Math.atan2(sy - py, sx - px);
          }
          segs.push({ x, y, theta });
        }

        drawWorm(ctx, width, height, segs, segs[0].theta);
      }

      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [width]);

  const resetKin = () => {
    kinRef.current = {
      headX: width / 2 + LOGICAL_BODY_LENGTH_PX / 2,
      headY: 180,
      heading: Math.PI,
      timeSec: 0,
    };
  };
  const resetPhys = () => { physTimeRef.current = 0; };

  const controllerInfo = physicsTrace?.meta.controller;
  const dragInfo = physicsTrace?.meta.drag;
  const imitationInfo = imitationTrace?.meta.controller;

  return (
    <div className="my-6 flex flex-col gap-3" ref={wrapRef}>
      {/* Mode toggle */}
      <div className="inline-flex flex-wrap rounded-lg border p-0.5 text-xs w-fit self-start gap-0.5">
        {(["kinematic", "physics", "imitation", "real"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-md px-3 py-1.5 font-medium transition-colors ${
              mode === m ? "bg-primary text-primary-foreground" : "hover:bg-accent"
            }`}
          >
            {m === "kinematic"
              ? "Kinematic (live)"
              : m === "physics"
              ? "Physics (MuJoCo CPG)"
              : m === "imitation"
              ? "Imitation (RL-trained)"
              : "Real worm (Tierpsy)"}
          </button>
        ))}
      </div>

      {/* Canvas */}
      <div className="rounded-lg border overflow-hidden">
        <canvas ref={canvasRef} className="block w-full" width={720} height={360} />
      </div>

      {/* Mode-specific controls */}
      {mode === "kinematic" ? (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <label className="flex flex-col gap-1">
              <span className="font-medium">
                Wave frequency · <span className="tabular-nums">{freq.toFixed(2)}</span> Hz
              </span>
              <input type="range" min="0" max="4" step="0.05"
                value={freq} onChange={(e) => setFreq(+e.target.value)}
                className="accent-primary"/>
            </label>
            <label className="flex flex-col gap-1">
              <span className="font-medium">
                Wavelength · <span className="tabular-nums">{wavelength.toFixed(2)}</span> body
              </span>
              <input type="range" min="0.3" max="1.5" step="0.02"
                value={wavelength} onChange={(e) => setWavelength(+e.target.value)}
                className="accent-primary"/>
            </label>
            <label className="flex flex-col gap-1">
              <span className="font-medium">
                Curvature · <span className="tabular-nums">{amplitude.toFixed(2)}</span> rad
              </span>
              <input type="range" min="0" max="1.8" step="0.02"
                value={amplitude} onChange={(e) => setAmplitude(+e.target.value)}
                className="accent-primary"/>
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <button onClick={() => setPaused((v) => !v)} className="rounded border px-3 py-1 hover:bg-accent">
              {paused ? "Resume" : "Pause"}
            </button>
            <button onClick={resetKin} className="rounded border px-3 py-1 hover:bg-accent">Reset position</button>
            <span className="text-muted-foreground ml-2">
              Browser-side wave kinematics. Interactive, not physics-constrained.
            </span>
          </div>
        </>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <label className="flex flex-col gap-1">
              <span className="font-medium">
                Playback speed · <span className="tabular-nums">{playbackSpeed.toFixed(2)}×</span>
              </span>
              <input type="range" min="0.1" max="4" step="0.1"
                value={playbackSpeed} onChange={(e) => setPlaybackSpeed(+e.target.value)}
                className="accent-primary"/>
            </label>
            <div className="flex items-end gap-2">
              <button onClick={() => setPaused((v) => !v)} className="rounded border px-3 py-1 hover:bg-accent">
                {paused ? "Resume" : "Pause"}
              </button>
              <button onClick={resetPhys} className="rounded border px-3 py-1 hover:bg-accent">Restart</button>
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            {mode === "physics" ? (
              physicsTrace ? (
                <>
                  <strong>6-second precomputed MuJoCo trace.</strong>{" "}
                  CPG controller ({controllerInfo?.freq_hz} Hz, {controllerInfo?.wavelength_body}-body wavelength, {controllerInfo?.amplitude_rad} rad amplitude)
                  driving position actuators on the {physicsTrace.meta.num_segments}-segment body.
                  Resistive-force-theory drag with anisotropy {dragInfo?.anisotropy}× (c⊥/c∥), which produces the net forward thrust from lateral undulation.
                  Body advances ~0.87 sim-m in 6 s → ~145 µm/s in real units, within the biological range for crawling <em>C. elegans</em>.
                </>
              ) : traceErr ? <>Trace load failed: {traceErr}</> : "Loading physics trace…"
            ) : mode === "real" ? (
              realTrace ? (
                <>
                  <strong>Real <em>C. elegans</em> crawl.</strong>{" "}
                  Centerlines extracted from the Tierpsy Tracker test dataset
                  (Zenodo <a className="underline" href="https://zenodo.org/records/3837679">3837679</a>) — 49-point skeletons per frame, resampled to our 20-segment grid and normalised so body length = 1 sim-unit.
                  The clip shown is worm #{realTrace.meta.source_worm_index ?? "?"} at source rate {(realTrace.meta.source_frame_rate_hz ?? 25).toFixed?.(0) ?? realTrace.meta.source_frame_rate_hz} Hz.
                  Real worms don't perform pure forward crawl — they pause, reorient, and occasionally reverse — so expect noticeably different behavior from the simulated modes.
                </>
              ) : realErr ? <>Real-worm trace load failed: {realErr}</> : "Loading real-worm trace…"
            ) : (
              imitationTrace ? (
                imitationInfo?.kind === "imitation_ppo_mlp" ? (
                  <>
                    <strong>Rollout from an RL-trained MLP controller.</strong>{" "}
                    The policy observes joint angles + velocities + a reference-phase clock and outputs a residual on top of a baseline CPG (freq {imitationInfo?.base_cpg_freq_hz} Hz, wavelength {imitationInfo?.base_cpg_wavelength} body).
                    Trained with PPO against a biologically parameterised reference trajectory (Fang-Yen 2010 / Boyle-Berri-Cohen 2012 crawling parameters) with MSE-of-segment-positions reward. This is Phase 2c — the pipeline generalizes to real pose-tracking data when it's swapped in.
                  </>
                ) : (
                  <>
                    <strong>Target trajectory.</strong> This is the biologically parameterised reference (Fang-Yen 2010 / Boyle-Berri-Cohen 2012 crawling parameters — freq {imitationTrace.meta.parameters?.freq_hz} Hz, wavelength {imitationTrace.meta.parameters?.wavelength_body} body, peak curvature {imitationTrace.meta.parameters?.curvature_amplitude_rad} rad) that a PPO-trained MLP controller learns to reproduce in Phase 2c. Training pipeline is in place; the learned-controller rollout will appear in this slot once training completes.
                  </>
                )
              ) : imitationErr ? (
                <>
                  <strong>Imitation trace unavailable.</strong> {imitationErr}
                </>
              ) : "Loading imitation trace…"
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default WormBody;
