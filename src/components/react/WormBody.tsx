import * as React from "react";
import { useEffect, useRef, useState } from "react";

/**
 * WormBody — Phase 1a of the digital C. elegans project.
 *
 * Browser-side kinematic visualisation of a 20-segment worm performing
 * sinusoidal traveling-wave locomotion. The underlying body geometry
 * is the MuJoCo MJCF shipped at /data/wormbody.xml — that's the
 * scientific spec future phases will use for physics + RL. Here we
 * render a visual preview using a kinematic wave model so visitors
 * can see what the worm looks like moving without needing WASM MuJoCo
 * in the page.
 *
 * Wave equation: θ(s, t) = amplitude · sin(2π (s/λ − f·t))
 *   where s ∈ [0,1] is fractional arc length along the body and
 *   θ is the cumulative heading change per unit arc length.
 * Propulsion: forward speed ≈ k · frequency · amplitude² (empirical
 *   scaling from resistive-force-theory derivations on low-Re media).
 */

const NUM_SEGMENTS = 20;
const LOGICAL_BODY_LENGTH_PX = 380;
const SEGMENT_LENGTH_PX = LOGICAL_BODY_LENGTH_PX / NUM_SEGMENTS;
const PROPULSION_COEFF = 24; // px/sec per (freq * amp²)

type WormState = {
  headX: number;
  headY: number;
  heading: number;
  timeSec: number;
};

function computeSegments(
  state: WormState,
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
    // Curvature integrated over this segment: the wave's second-derivative
    // contribution scaled by amplitude gives the heading rate per arc-unit.
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
  // Tapered body: narrower at head (i=0) and tail (i=total-1), thickest mid-body.
  const t = i / (total - 1);
  const profile = Math.sin(Math.PI * t); // 0→1→0
  return 4 + 10 * profile;
}

export function WormBody() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [freq, setFreq] = useState(1.5); // Hz
  const [wavelength, setWavelength] = useState(0.7); // fraction of body length
  const [amplitude, setAmplitude] = useState(0.9); // radians per 2π·s (wave curvature amplitude)
  const [paused, setPaused] = useState(false);
  const [width, setWidth] = useState(720);

  // Runtime state lives in a ref so control changes don't blow away the worm's pose.
  const stateRef = useRef<WormState>({
    headX: 0,
    headY: 0,
    heading: 0,
    timeSec: 0,
  });
  // Parameter refs mirror React state for the animation loop (closure capture).
  const paramRef = useRef({ freq, wavelength, amplitude, paused });
  paramRef.current = { freq, wavelength, amplitude, paused };

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

  // Initialise worm pose at center of canvas whenever dimensions change.
  useEffect(() => {
    stateRef.current = {
      headX: width / 2 + LOGICAL_BODY_LENGTH_PX / 2,
      headY: 180,
      heading: Math.PI, // initial heading points "backward" so the body trails to the right-start position
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

      const state = stateRef.current;
      const p = paramRef.current;

      if (!p.paused) {
        state.timeSec += dt;
      }

      const segs = computeSegments(state, p.freq, p.wavelength, p.amplitude);

      // Forward propulsion: move head in the direction of the averaged body axis.
      // Propulsion is driven by wave frequency × amplitude²; pause disables motion
      // but leaves the wave evolving so you can still see the shape.
      if (!p.paused) {
        // Average heading from first 5 segments (the anterior half directs motion).
        let meanHeading = 0;
        for (let i = 0; i < 5; i++) meanHeading += segs[i].theta;
        meanHeading /= 5;
        const speed = PROPULSION_COEFF * p.freq * p.amplitude * p.amplitude;
        state.headX += speed * Math.cos(meanHeading) * dt;
        state.headY += speed * Math.sin(meanHeading) * dt;
      }

      // Wrap around (so the worm keeps going rather than leaving the canvas).
      const margin = 40;
      const effectiveW = width;
      if (state.headX < -margin) state.headX += effectiveW + 2 * margin;
      if (state.headX > effectiveW + margin) state.headX -= effectiveW + 2 * margin;
      if (state.headY < -margin) state.headY += height + 2 * margin;
      if (state.headY > height + margin) state.headY -= height + 2 * margin;

      // Resize canvas each frame if width changed.
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;

      // Substrate (agar-ish): subtle speckled cream with a faint radial vignette.
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createRadialGradient(width / 2, height / 2, 40, width / 2, height / 2, Math.max(width, height));
      bg.addColorStop(0, "rgba(247, 237, 211, 0.95)");
      bg.addColorStop(1, "rgba(229, 215, 185, 0.92)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);
      // Fine grid to suggest texture / scale
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

      // Worm body: draw as a tapered chain of capsules.
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let i = 0; i < segs.length - 1; i++) {
        const a = segs[i];
        const b = segs[i + 1];
        const w = segmentWidth(i, segs.length);
        ctx.strokeStyle = i === 0 ? "#1a2a4a" : "#2f5233";
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      // Head marker — small dot indicating anterior.
      const head = segs[0];
      const headAngle = segs[0].theta;
      ctx.fillStyle = "#f2ead3";
      ctx.beginPath();
      ctx.arc(head.x - 4 * Math.cos(headAngle), head.y - 4 * Math.sin(headAngle), 2.2, 0, Math.PI * 2);
      ctx.fill();

      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [width]);

  const reset = () => {
    stateRef.current = {
      headX: width / 2 + LOGICAL_BODY_LENGTH_PX / 2,
      headY: 180,
      heading: Math.PI,
      timeSec: 0,
    };
  };

  return (
    <div className="my-6 flex flex-col gap-3" ref={wrapRef}>
      <div className="rounded-lg border overflow-hidden">
        <canvas ref={canvasRef} className="block w-full" width={720} height={360} />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <label className="flex flex-col gap-1">
          <span className="font-medium">
            Wave frequency · <span className="tabular-nums">{freq.toFixed(2)}</span> Hz
          </span>
          <input
            type="range" min="0" max="4" step="0.05"
            value={freq} onChange={(e) => setFreq(+e.target.value)}
            className="accent-primary"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-medium">
            Wavelength · <span className="tabular-nums">{wavelength.toFixed(2)}</span> body
          </span>
          <input
            type="range" min="0.3" max="1.5" step="0.02"
            value={wavelength} onChange={(e) => setWavelength(+e.target.value)}
            className="accent-primary"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-medium">
            Curvature · <span className="tabular-nums">{amplitude.toFixed(2)}</span> rad
          </span>
          <input
            type="range" min="0" max="1.8" step="0.02"
            value={amplitude} onChange={(e) => setAmplitude(+e.target.value)}
            className="accent-primary"
          />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button
          onClick={() => setPaused((v) => !v)}
          className="rounded border px-3 py-1 hover:bg-accent"
        >
          {paused ? "Resume" : "Pause"}
        </button>
        <button
          onClick={reset}
          className="rounded border px-3 py-1 hover:bg-accent"
        >
          Reset position
        </button>
        <span className="text-muted-foreground ml-2">
          20-segment kinematic preview. The <a href="/data/wormbody.xml" className="underline">MJCF body spec</a> (loadable in MuJoCo) is the
          scientific artifact the next phase will train on.
        </span>
      </div>
    </div>
  );
}

export default WormBody;
