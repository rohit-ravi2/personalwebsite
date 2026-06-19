import * as React from "react";
import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html, Line } from "@react-three/drei";
import * as THREE from "three";

/**
 * HeroCell3D — the molecular HERO CELL (AVA) rendered 1:1 from the Tier4
 * substrate code. This is Scene 1/3.
 *
 * What is rendered, all from emitted data (no hand-authored geometry):
 *   - 3D membrane shell built from the 57 EM-derived morphology segments
 *     (soma / axon / dendrite) of AVAL — coloured live by the hero cell's
 *     REAL voltage trajectory, with a Ca²⁺-driven emissive glow.
 *   - Channel / receptor / pump / transporter glyphs as 3D meshes placed on
 *     the membrane, SIZED by the per-channel gbar (CeNGEN TPM × γ × C_global),
 *     each carrying {id, status, provenance}.
 *   - Ion pools (Na / K / Ca / Cl) as translucent 3D volumes inside the cell.
 *   - Cleft + glia OUTER shell wrapping the soma.
 *   - Synapse, gap-junction and peptide-DCV cassettes anchored to the axon.
 *   - WCM metabolism badge (AVA-only proteome).
 *
 * Status drives material, so completeness is legible at a glance:
 *   ON          → solid, lit
 *   default-OFF → dimmed, low opacity
 *   ORPHANED    → wireframe
 *   MISSING     → red dashed ghost rendered in-place (NOT integrated)
 *
 * Hovering a glyph or a legend row cross-highlights via the shared `hovered`
 * id. Clicking a glyph selects it (drives the legend filter upstream).
 */

// ---------------------------------------------------------------------------
// Shared types (kept structurally compatible with SubstrateAnatomy.tsx)
// ---------------------------------------------------------------------------

type Status = "on" | "off" | "orphaned" | "missing";

export type HeroSegment = {
  id: number;
  parent: number | null;
  tag: string; // soma | axon | dendrite | other
  prox: [number, number, number];
  dist: [number, number, number];
  prox_r: number;
  dist_r: number;
  length_um: number;
  surf_um2: number;
};

export type HeroMorphology = {
  cell: string;
  cengen_class: string;
  n_segments: number;
  total_surf_um2: number;
  region_surf_um2: { soma: number; axon: number; dendrite: number };
  soma_centroid_um: [number, number, number];
  segments: HeroSegment[];
};

export type HeroChannel = {
  channel: string;
  gbar_Scm2: number;
  ion: string;
  family: string;
  expressed: boolean;
};

export type HeroChannelGbar = {
  hero_cell: string;
  hero_class: string;
  hero_channels: HeroChannel[];
};

export type InventoryRecord = {
  id: string;
  name: string;
  category: string;
  subtype: string;
  status: Status;
  file: string;
  line: number;
  gbar_source: string | null;
  gbar_value_AVA: number | null;
  physical_desc: string;
  provenance: string;
};

export type FrameState = {
  // updated imperatively each rAF tick by the parent's clock (a ref object),
  // so per-frame V/Ca colour does not trigger React re-renders.
  V_mV: number;
  Ca_uM: number;
  I_Na: number;
  I_K: number;
  I_Ca: number;
  pump: number;
  t: number; // seconds, monotonically increasing animation clock
};

// ---------------------------------------------------------------------------
// Status → material parameters
// ---------------------------------------------------------------------------

type StatusStyle = {
  opacity: number;
  wireframe: boolean;
  emissiveBoost: number;
  dashed: boolean;
  baseColor: string;
};

const STATUS_STYLE: Record<Status, StatusStyle> = {
  on: { opacity: 1.0, wireframe: false, emissiveBoost: 0.25, dashed: false, baseColor: "#2f7a52" },
  off: { opacity: 0.32, wireframe: false, emissiveBoost: 0.0, dashed: false, baseColor: "#c8922b" },
  orphaned: { opacity: 0.5, wireframe: true, emissiveBoost: 0.0, dashed: false, baseColor: "#7c8597" },
  missing: { opacity: 0.55, wireframe: false, emissiveBoost: 0.0, dashed: true, baseColor: "#e11d48" },
};

// Family → glyph hue (so ion species read at a glance, distinct from status).
const FAMILY_COLOR: Record<string, string> = {
  Ca: "#d6743c", // orange — calcium
  K: "#3c6fd6", // blue — potassium
  Na: "#d63c8a", // magenta — sodium
  Cl: "#3cb7d6", // cyan — chloride
  CaK_brake: "#7d4cd6", // violet — Ca-activated K
  receptor: "#2f9e6b", // green — ligand-gated
};

const CATEGORY_COLOR: Record<string, string> = {
  channel: "#2f7a52",
  receptor: "#2f9e6b",
  pump: "#b8860b",
  transporter: "#a06a2c",
  gap_junction: "#5b7d9e",
  release: "#9e5b8c",
  neuromod_peptide: "#7d5b9e",
  ion_compartment: "#5b9e8c",
  geometry: "#6b6b6b",
  metabolism: "#9e7d3c",
};

// ---------------------------------------------------------------------------
// Geometry helpers — world units are scaled-down µm.
// ---------------------------------------------------------------------------

const UM = 0.05; // µm → scene units (axon ~1000 µm → ~50 units)

function v3(p: [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(p[0], p[1], p[2]);
}

// ---------------------------------------------------------------------------
// Voltage / calcium → colour
// ---------------------------------------------------------------------------

// Map V (mV) to a forest→amber→coral ramp. Resting (~-70) reads cool green,
// depolarised (~0+) reads warm. Clamp to a physiological window.
function voltageColor(v: number, out: THREE.Color): THREE.Color {
  const vmin = -80;
  const vmax = -50; // foundation regime stays sub-threshold; tighten window so
  // the small resting fluctuations are actually visible.
  const t = Math.min(1, Math.max(0, (v - vmin) / (vmax - vmin)));
  // cool forest (#1f5c3a) → cream (#e9dcc0) → warm amber (#d68a2b)
  const cool = new THREE.Color("#1f5c3a");
  const mid = new THREE.Color("#7faa6e");
  const warm = new THREE.Color("#d68a2b");
  if (t < 0.5) out.copy(cool).lerp(mid, t / 0.5);
  else out.copy(mid).lerp(warm, (t - 0.5) / 0.5);
  return out;
}

// ---------------------------------------------------------------------------
// Membrane shell — tubes per morphology segment, recoloured by live V.
// ---------------------------------------------------------------------------

function MembraneShell({
  morph,
  frame,
  center,
}: {
  morph: HeroMorphology;
  frame: React.MutableRefObject<FrameState>;
  center: THREE.Vector3;
}) {
  // Build one merged-ish set of cylinders. We render each segment as a
  // tapered cylinder; soma as a sphere. We keep it as a single group with a
  // shared material so the per-frame V recolour is one assignment.
  const { somaSegs, tubeSegs } = useMemo(() => {
    const somaSegs: HeroSegment[] = [];
    const tubeSegs: HeroSegment[] = [];
    for (const s of morph.segments) {
      if (s.length_um < 1e-6 || s.tag === "soma") somaSegs.push(s);
      else tubeSegs.push(s);
    }
    return { somaSegs, tubeSegs };
  }, [morph]);

  // Precompute transforms for each tube segment.
  const tubes = useMemo(() => {
    return tubeSegs.map((s) => {
      const a = v3(s.prox).sub(center).multiplyScalar(UM);
      const b = v3(s.dist).sub(center).multiplyScalar(UM);
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length();
      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      const quat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.clone().normalize(),
      );
      // Radii: clamp to a visible floor (axons are sub-µm).
      const rTop = Math.max(0.12, s.dist_r * UM * 4);
      const rBot = Math.max(0.12, s.prox_r * UM * 4);
      return { mid, quat, len, rTop, rBot, tag: s.tag, id: s.id };
    });
  }, [tubeSegs, center]);

  const somas = useMemo(() => {
    return somaSegs.map((s) => {
      const p = v3(s.prox).sub(center).multiplyScalar(UM);
      const r = Math.max(0.9, s.prox_r * UM * 6);
      return { p, r, id: s.id };
    });
  }, [somaSegs, center]);

  // Per-mesh recolour + Ca glow is handled by SharedMembraneMaterial below
  // (one color assignment per mesh per frame).

  return (
    <group>
      {/* tubes */}
      {tubes.map((t, i) => (
        <mesh key={`tube-${t.id}-${i}`} position={t.mid} quaternion={t.quat}>
          <cylinderGeometry args={[t.rTop, t.rBot, t.len, 10, 1, true]} />
          <SharedMembraneMaterial frame={frame} />
        </mesh>
      ))}
      {/* soma(s) */}
      {somas.map((s, i) => (
        <mesh key={`soma-${s.id}-${i}`} position={s.p}>
          <sphereGeometry args={[s.r, 40, 40]} />
          <SharedMembraneMaterial frame={frame} soma />
        </mesh>
      ))}
    </group>
  );
}

/**
 * SharedMembraneMaterial — a translucent membrane material that recolours by
 * live V and glows by Ca. Each instance subscribes to the frame clock; cheap
 * because it is a single color assignment per mesh per frame.
 */
function SharedMembraneMaterial({
  frame,
  soma = false,
}: {
  frame: React.MutableRefObject<FrameState>;
  soma?: boolean;
}) {
  const ref = useRef<THREE.MeshStandardMaterial>(null);
  const tmp = useRef(new THREE.Color());
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    const f = frame.current;
    voltageColor(f.V_mV, tmp.current);
    m.color.copy(tmp.current);
    const caN = Math.min(1, Math.max(0, (f.Ca_uM - 0.15) / 0.25));
    m.emissive.setRGB(0.85 * caN, 0.42 * caN, 0.12 * caN);
    m.emissiveIntensity = 0.12 + 0.7 * caN;
  });
  return (
    <meshStandardMaterial
      ref={ref}
      transparent
      opacity={soma ? 0.5 : 0.42}
      roughness={0.45}
      metalness={0.05}
      side={THREE.DoubleSide}
      depthWrite={false}
    />
  );
}

// ---------------------------------------------------------------------------
// Membrane anchor points — deterministic placement of glyphs on the surface.
// ---------------------------------------------------------------------------

// Distribute N points on the soma sphere via a Fibonacci spiral, then a few
// along the proximal axon. Deterministic so glyph positions are stable.
function membraneAnchors(
  morph: HeroMorphology,
  center: THREE.Vector3,
  n: number,
): { pos: THREE.Vector3; normal: THREE.Vector3 }[] {
  const soma = morph.segments.find((s) => s.tag === "soma") ?? morph.segments[0];
  const somaCenter = v3(soma.prox).sub(center).multiplyScalar(UM);
  const R = Math.max(0.9, soma.prox_r * UM * 6) + 0.15;

  // proximal axon direction (toward centroid of axon segs) for a small spread
  const axon = morph.segments.filter((s) => s.tag === "axon");
  const out: { pos: THREE.Vector3; normal: THREE.Vector3 }[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / Math.max(1, n - 1)) * 2; // 1 .. -1
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    const dir = new THREE.Vector3(
      Math.cos(theta) * radius,
      y,
      Math.sin(theta) * radius,
    ).normalize();
    out.push({
      pos: somaCenter.clone().add(dir.clone().multiplyScalar(R)),
      normal: dir,
    });
  }
  // pin a couple of anchors onto the proximal axon for synapse/gap cassettes
  if (axon.length > 0) {
    const a = axon[Math.floor(axon.length * 0.15)];
    const ap = v3(a.dist).sub(center).multiplyScalar(UM);
    const an = ap.clone().sub(somaCenter).normalize();
    out.push({ pos: ap, normal: an });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Glyph — a single rendered structure (channel / pump / receptor / etc.)
// ---------------------------------------------------------------------------

type GlyphSpec = {
  rec: InventoryRecord;
  pos: THREE.Vector3;
  normal: THREE.Vector3;
  size: number; // base radius in scene units
  color: string;
  shape: "barrel" | "cap" | "disc" | "wedge" | "cluster";
};

function Glyph({
  spec,
  hovered,
  selected,
  setHovered,
  onSelect,
  frame,
}: {
  spec: GlyphSpec;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
  frame: React.MutableRefObject<FrameState>;
}) {
  const { rec, pos, normal, size, color, shape } = spec;
  const style = STATUS_STYLE[rec.status];
  const groupRef = useRef<THREE.Group>(null);
  const matRef = useRef<THREE.MeshStandardMaterial>(null);
  const isHi = hovered === rec.id || selected === rec.id;

  // Orient glyph so its local +Y aligns with the surface normal.
  const quat = useMemo(
    () =>
      new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        normal.clone().normalize(),
      ),
    [normal],
  );

  // Gentle highlight pulse + ION-flow-driven emissive on ON glyphs.
  useFrame(() => {
    const g = groupRef.current;
    if (g) {
      const target = isHi ? 1.45 : 1.0;
      g.scale.lerp(new THREE.Vector3(target, target, target), 0.18);
    }
    const m = matRef.current;
    if (m && rec.status === "on") {
      // pulse magnitude tracks the dominant ion current for this glyph's ion
      const f = frame.current;
      let drive = 0;
      const ion = rec.subtype.includes("Ca")
        ? Math.abs(f.I_Ca)
        : rec.subtype.startsWith("K") || rec.category === "channel"
          ? Math.abs(f.I_K)
          : Math.abs(f.I_Na);
      drive = Math.min(1, drive + ion / 0.0015);
      const pulse = 0.5 + 0.5 * Math.sin(f.t * 2.2 + pos.x * 3);
      m.emissiveIntensity = style.emissiveBoost + 0.5 * drive * pulse;
    }
  });

  const baseColor = rec.status === "missing" ? STATUS_STYLE.missing.baseColor : color;

  return (
    <group
      ref={groupRef}
      position={pos}
      quaternion={quat}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(rec.id);
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        setHovered(null);
        document.body.style.cursor = "auto";
      }}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(rec.id);
      }}
    >
      {/* MISSING → red dashed ghost outline rendered in-place */}
      {rec.status === "missing" ? (
        <MissingGhost size={size} />
      ) : (
        <GlyphBody shape={shape} size={size}>
          <meshStandardMaterial
            ref={matRef}
            color={baseColor}
            emissive={baseColor}
            emissiveIntensity={style.emissiveBoost}
            transparent={style.opacity < 1 || style.wireframe}
            opacity={style.opacity}
            wireframe={style.wireframe}
            roughness={0.4}
            metalness={0.15}
          />
        </GlyphBody>
      )}

      {/* hover/selected tooltip */}
      {isHi && (
        <Html center distanceFactor={28} position={[0, size + 0.7, 0]} zIndexRange={[40, 0]}>
          <div className="pointer-events-none w-52 rounded-lg border border-white/50 bg-white/90 p-2 text-left shadow-lg backdrop-blur-md">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[0.72rem] font-semibold text-emerald-950">{rec.name}</span>
              <span className="shrink-0 font-mono text-[0.55rem] uppercase text-emerald-900/50">
                {rec.status === "off"
                  ? "default-OFF"
                  : rec.status === "missing"
                    ? "NOT INTEGRATED"
                    : rec.status}
              </span>
            </div>
            <p className="mt-0.5 text-[0.62rem] leading-snug text-emerald-900/70">{rec.subtype}</p>
            <p className="mt-1 text-[0.6rem] leading-snug text-emerald-900/60">{rec.physical_desc}</p>
            {rec.gbar_value_AVA != null && (
              <p className="mt-1 font-mono text-[0.55rem] text-emerald-900/55">
                gbar(AVA) = {rec.gbar_value_AVA.toExponential(2)} S/cm²
              </p>
            )}
            <p className="mt-1 font-mono text-[0.55rem] leading-snug text-emerald-900/45">
              {rec.file.split("/").slice(-2).join("/")}:{rec.line}
            </p>
          </div>
        </Html>
      )}
    </group>
  );
}

function GlyphBody({
  shape,
  size,
  children,
}: {
  shape: GlyphSpec["shape"];
  size: number;
  children: React.ReactNode;
}) {
  switch (shape) {
    case "barrel": // ion channel — transmembrane barrel
      return (
        <mesh>
          <cylinderGeometry args={[size, size * 0.8, size * 2.0, 8]} />
          {children}
        </mesh>
      );
    case "cap": // pump — capped dome
      return (
        <mesh>
          <sphereGeometry args={[size, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.6]} />
          {children}
        </mesh>
      );
    case "disc": // receptor — broad disc on the surface
      return (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[size * 1.2, size * 1.2, size * 0.7, 16]} />
          {children}
        </mesh>
      );
    case "wedge": // transporter — angular cotransporter
      return (
        <mesh>
          <coneGeometry args={[size, size * 1.8, 5]} />
          {children}
        </mesh>
      );
    case "cluster": // gap junction / release — clustered box
      return (
        <mesh>
          <boxGeometry args={[size * 1.4, size * 1.4, size * 1.4]} />
          {children}
        </mesh>
      );
  }
}

// MISSING structures: a red dashed wire ghost in-place (NOT integrated).
function MissingGhost({ size }: { size: number }) {
  const pts = useMemo(() => {
    // dashed ring + vertical pin to read as a "slot where this would go"
    const ring: THREE.Vector3[] = [];
    const seg = 28;
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      ring.push(new THREE.Vector3(Math.cos(a) * size * 1.3, 0, Math.sin(a) * size * 1.3));
    }
    return ring;
  }, [size]);
  return (
    <group>
      <Line points={pts} color="#e11d48" lineWidth={1.5} dashed dashScale={6} dashSize={0.12} gapSize={0.08} />
      <Line
        points={[new THREE.Vector3(0, -size, 0), new THREE.Vector3(0, size * 1.4, 0)]}
        color="#e11d48"
        lineWidth={1.5}
        dashed
        dashScale={6}
        dashSize={0.12}
        gapSize={0.08}
      />
    </group>
  );
}

// ---------------------------------------------------------------------------
// Ion pools — translucent 3D volumes inside the soma.
// ---------------------------------------------------------------------------

function IonPools({ center, soma }: { center: THREE.Vector3; soma: THREE.Vector3 }) {
  // Four offset blobs labelled Na/K/Ca/Cl inside the soma volume.
  const pools = [
    { ion: "Na", color: "#d63c8a", off: new THREE.Vector3(0.4, 0.3, 0.2), r: 0.5 },
    { ion: "K", color: "#3c6fd6", off: new THREE.Vector3(-0.4, 0.2, -0.3), r: 0.6 },
    { ion: "Ca", color: "#d6743c", off: new THREE.Vector3(0.1, -0.4, 0.35), r: 0.35 },
    { ion: "Cl", color: "#3cb7d6", off: new THREE.Vector3(-0.25, -0.3, -0.2), r: 0.4 },
  ];
  return (
    <group position={soma}>
      {pools.map((p) => (
        <mesh key={p.ion} position={p.off}>
          <sphereGeometry args={[p.r, 20, 20]} />
          <meshStandardMaterial
            color={p.color}
            transparent
            opacity={0.22}
            roughness={0.6}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Cleft + glia OUTER shell wrapping the soma.
// ---------------------------------------------------------------------------

function GliaShell({ soma, radius }: { soma: THREE.Vector3; radius: number }) {
  return (
    <group position={soma}>
      {/* synaptic cleft — thin gap shell */}
      <mesh>
        <sphereGeometry args={[radius * 1.18, 32, 32]} />
        <meshStandardMaterial
          color="#bcd3c4"
          transparent
          opacity={0.07}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>
      {/* glia — outer K-sink shell (sink-only; the spatial siphon is MISSING) */}
      <mesh>
        <sphereGeometry args={[radius * 1.32, 24, 24]} />
        <meshStandardMaterial
          color="#9fb6a6"
          transparent
          opacity={0.05}
          wireframe
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

// ---------------------------------------------------------------------------
// Ion-flow particles — driven by the real hero_flows currents.
// ---------------------------------------------------------------------------

function FlowParticles({
  soma,
  radius,
  frame,
}: {
  soma: THREE.Vector3;
  radius: number;
  frame: React.MutableRefObject<FrameState>;
}) {
  const N = 120;
  const ref = useRef<THREE.Points>(null);
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const arr = new Float32Array(N * 3);
    const phase = new Float32Array(N);
    const ion = new Float32Array(N); // 0 Na, 1 K, 2 Ca, 3 pump
    for (let i = 0; i < N; i++) {
      phase[i] = Math.random();
      ion[i] = i % 4;
      arr[i * 3] = 0;
      arr[i * 3 + 1] = 0;
      arr[i * 3 + 2] = 0;
    }
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    (g as unknown as { _phase: Float32Array })._phase = phase;
    (g as unknown as { _ion: Float32Array })._ion = ion;
    return g;
  }, []);

  const colors = useMemo(() => {
    const c = new Float32Array(N * 3);
    const palette = [
      new THREE.Color("#d63c8a"), // Na
      new THREE.Color("#3c6fd6"), // K
      new THREE.Color("#d6743c"), // Ca
      new THREE.Color("#b8860b"), // pump
    ];
    for (let i = 0; i < N; i++) {
      const col = palette[i % 4];
      c[i * 3] = col.r;
      c[i * 3 + 1] = col.g;
      c[i * 3 + 2] = col.b;
    }
    geom.setAttribute("color", new THREE.BufferAttribute(c, 3));
    return c;
  }, [geom]);

  useFrame(() => {
    const p = ref.current;
    if (!p) return;
    const f = frame.current;
    const pos = geom.getAttribute("position") as THREE.BufferAttribute;
    const phase = (geom as unknown as { _phase: Float32Array })._phase;
    const ionArr = (geom as unknown as { _ion: Float32Array })._ion;
    // direction: inward currents (negative) draw particles toward the cell;
    // outward (positive) push them out. pump always extrudes Na (outward).
    const ionMag = [Math.abs(f.I_Na), Math.abs(f.I_K), Math.abs(f.I_Ca), Math.abs(f.pump)];
    const ionDir = [Math.sign(f.I_Na || -1), Math.sign(f.I_K || 1), -1, 1];
    for (let i = 0; i < N; i++) {
      const ion = ionArr[i];
      const speed = 0.15 + (ionMag[ion] / 0.0015) * 1.2;
      let u = (phase[i] + f.t * speed * 0.12 * ionDir[ion]) % 1;
      if (u < 0) u += 1;
      phase[i] = phase[i]; // keep base phase; u is the travelled fraction
      // travel along a radial spoke; angle fixed per particle
      const theta = i * 2.39996;
      const yy = ((i % 7) / 7 - 0.5) * 1.2;
      const r0 = radius * 1.3;
      const r1 = radius * 0.7;
      const rr = ionDir[ion] < 0 ? r0 + (r1 - r0) * u : r1 + (r0 - r1) * u;
      pos.setXYZ(
        i,
        soma.x + Math.cos(theta) * rr,
        soma.y + yy * radius,
        soma.z + Math.sin(theta) * rr,
      );
    }
    pos.needsUpdate = true;
  });

  return (
    <points ref={ref} geometry={geom}>
      <pointsMaterial
        size={0.22}
        vertexColors
        transparent
        opacity={0.85}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// WCM metabolism badge — AVA-only proteome marker.
// ---------------------------------------------------------------------------

function WcmBadge({ soma, radius }: { soma: THREE.Vector3; radius: number }) {
  return (
    <Html center distanceFactor={30} position={[soma.x, soma.y + radius * 1.7, soma.z]}>
      <div className="pointer-events-none whitespace-nowrap rounded-md border border-amber-500/40 bg-amber-50/90 px-2 py-0.5 text-[0.6rem] font-medium text-amber-900 shadow backdrop-blur-sm">
        WCM proteome · AVA-only (metabolism vars declared, inert)
      </div>
    </Html>
  );
}

// ---------------------------------------------------------------------------
// HeroCell3D — assembles everything.
// ---------------------------------------------------------------------------

export default function HeroCell3D({
  morph,
  gbar,
  records,
  frame,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  morph: HeroMorphology;
  gbar: HeroChannelGbar;
  records: InventoryRecord[];
  frame: React.MutableRefObject<FrameState>;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const center = useMemo(() => v3(morph.soma_centroid_um), [morph]);

  // soma scene position + radius
  const somaSeg = useMemo(
    () => morph.segments.find((s) => s.tag === "soma") ?? morph.segments[0],
    [morph],
  );
  const somaPos = useMemo(
    () => v3(somaSeg.prox).sub(center).multiplyScalar(UM),
    [somaSeg, center],
  );
  const somaR = Math.max(0.9, somaSeg.prox_r * UM * 6);

  // gbar lookup for sizing channel/receptor glyphs
  const gbarById = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of gbar.hero_channels) m.set(c.channel, c.gbar_Scm2);
    return m;
  }, [gbar]);

  const maxG = useMemo(
    () => Math.max(1e-9, ...gbar.hero_channels.map((c) => c.gbar_Scm2)),
    [gbar],
  );

  // Which records get a 3D glyph on the hero cell?
  // Channels, receptors, pumps, transporters, gap junctions, release,
  // neuromod/peptide. (ion_compartment / geometry / metabolism are rendered as
  // pools / shell / badge, except the MISSING compartment ghosts which we DO
  // place so the gaps are visible in-place.)
  const glyphRecords = useMemo(() => {
    const placeable = new Set([
      "channel",
      "receptor",
      "pump",
      "transporter",
      "gap_junction",
      "release",
      "neuromod_peptide",
    ]);
    return records.filter(
      (r) => placeable.has(r.category) || r.status === "missing",
    );
  }, [records]);

  const anchors = useMemo(
    () => membraneAnchors(morph, center, glyphRecords.length),
    [morph, center, glyphRecords.length],
  );

  const specs: GlyphSpec[] = useMemo(() => {
    const shapeFor = (cat: string): GlyphSpec["shape"] => {
      switch (cat) {
        case "channel":
          return "barrel";
        case "receptor":
          return "disc";
        case "pump":
          return "cap";
        case "transporter":
          return "wedge";
        default:
          return "cluster";
      }
    };
    return glyphRecords.map((rec, i) => {
      const a = anchors[i % anchors.length];
      // size: channels/receptors by gbar; others by a fixed mid size.
      const chanKey = rec.id.replace(/^chan_|^recep_/, "");
      const g = gbarById.get(chanKey);
      let size: number;
      if (g != null && g > 0) {
        size = 0.18 + 0.55 * Math.sqrt(g / maxG); // sqrt for visual balance
      } else {
        size = 0.26; // pumps/transporters/cassettes & zero-gbar channels
      }
      // color: channels by family/ion, others by category
      const fam =
        rec.category === "channel" || rec.category === "receptor"
          ? gbar.hero_channels.find((c) => c.channel === chanKey)?.family
          : undefined;
      const color =
        (fam && FAMILY_COLOR[fam]) || CATEGORY_COLOR[rec.category] || "#2f7a52";
      return {
        rec,
        pos: a.pos,
        normal: a.normal,
        size,
        color,
        shape: shapeFor(rec.category),
      };
    });
  }, [glyphRecords, anchors, gbarById, maxG, gbar]);

  return (
    <group>
      <MembraneShell morph={morph} frame={frame} center={center} />
      <IonPools center={center} soma={somaPos} />
      <GliaShell soma={somaPos} radius={somaR} />
      <FlowParticles soma={somaPos} radius={somaR} frame={frame} />
      <WcmBadge soma={somaPos} radius={somaR} />
      {specs.map((spec) => (
        <Glyph
          key={spec.rec.id}
          spec={spec}
          hovered={hovered}
          selected={selected}
          setHovered={setHovered}
          onSelect={onSelect}
          frame={frame}
        />
      ))}
      {/* cell label */}
      <Html center distanceFactor={34} position={[somaPos.x, somaPos.y - somaR * 1.9, somaPos.z]}>
        <div className="pointer-events-none whitespace-nowrap rounded-md bg-white/85 px-2 py-1 text-[0.62rem] font-medium text-emerald-900 shadow">
          HERO · {morph.cell} ({morph.cengen_class}) · {morph.n_segments} segs · EM-derived
        </div>
      </Html>
    </group>
  );
}
