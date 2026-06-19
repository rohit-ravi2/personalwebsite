import * as React from "react";
import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html, Line } from "@react-three/drei";
import * as THREE from "three";
import {
  type GlyphSignature,
  type GlyphSignatures,
  ParametricGlyphBody,
  StructureGlyph,
  signatureKey,
} from "./GlyphGeometry";

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

// Re-exported for any consumer that imports the signatures type from HeroCell3D.
export type { GlyphSignatures } from "./GlyphGeometry";

type GlyphSpec = {
  rec: InventoryRecord;
  pos: THREE.Vector3;
  normal: THREE.Vector3;
  size: number; // base radius in scene units
  color: string;
  shape: "barrel" | "cap" | "disc" | "wedge" | "cluster";
  // AlphaFold/PDB-derived shape signature (when a structure exists for this
  // record); drives a structure-derived LatheGeometry instead of a primitive.
  sig?: GlyphSignature;
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
  const { rec, pos, normal, size, color, shape, sig } = spec;
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
      {/* MISSING → red dashed ghost outline rendered in-place.
          Otherwise: if an AlphaFold/PDB signature exists, revolve a
          structure-derived LatheGeometry; else a refined parametric glyph. */}
      {rec.status === "missing" ? (
        <MissingGhost size={size} />
      ) : sig ? (
        <StructureGlyph
          sig={sig}
          size={size}
          showPore={rec.category === "channel" || rec.category === "receptor"}
        >
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
        </StructureGlyph>
      ) : (
        <ParametricGlyphBody shape={shape} size={size}>
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
        </ParametricGlyphBody>
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

// (GlyphBody superseded by ParametricGlyphBody / StructureGlyph in
// ./GlyphGeometry — see the AlphaFold-informed glyph upgrade.)

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
// Ion compartments — every ion_compartment inventory record rendered 1:1 as a
// dedicated, status-coded, click-cross-highlightable 3D structure (not just the
// generic Na/K/Ca/Cl blobs + one glia shell). Each glyph carries its own
// {id, status, provenance} and reacts to the shared `hovered`/`selected` id
// exactly like the membrane glyphs.
//
// Structure family is chosen per record so the spatial topology is legible:
//   ion_singlepool   → 4 well-mixed Na/K/Ca/Cl blobs inside the soma (ON)
//   ion_tissue_buf   → thin inline buffer shell hugging the membrane (ON)
//   ion_spatial_k    → 3 nested ghost shells (submembrane / bulk / K_out) (OFF)
//   ion_spatial_ca   → 2 nested ghost shells (submembrane / bulk)        (OFF)
//   ion_glia_buffer  → outer wireframe sink shell, sink-only              (ORPHANED)
//   ion_compartments → wireframe lattice box (Contract-B interface)       (ORPHANED)
//   ion_osmotic      → pulsing volume-regulation halo                     (ORPHANED)
//   ion_perisynaptic → submembrane band girdle                           (ORPHANED)
//   ion_ephaptic     → external field arc                                 (ORPHANED)
// ---------------------------------------------------------------------------

const ION_BLOB_COLOR: Record<string, string> = {
  Na: "#d63c8a",
  K: "#3c6fd6",
  Ca: "#d6743c",
  Cl: "#3cb7d6",
};

// A wrapper that makes any compartment sub-tree interactive + status-coded and
// wires it to the shared hover/select id, with a tooltip carrying provenance.
function CompartmentGlyph({
  rec,
  soma,
  radius,
  hovered,
  selected,
  setHovered,
  onSelect,
  labelOffset,
  children,
}: {
  rec: InventoryRecord;
  soma: THREE.Vector3;
  radius: number;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
  labelOffset: number;
  children: (style: StatusStyle, isHi: boolean) => React.ReactNode;
}) {
  const style = STATUS_STYLE[rec.status];
  const isHi = hovered === rec.id || selected === rec.id;
  const groupRef = useRef<THREE.Group>(null);
  useFrame(() => {
    const g = groupRef.current;
    if (!g) return;
    // subtle scale pop on highlight so the compartment "lifts" like a glyph
    const target = isHi ? 1.08 : 1.0;
    g.scale.lerp(new THREE.Vector3(target, target, target), 0.15);
  });
  return (
    <group
      ref={groupRef}
      position={soma}
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
      {children(style, isHi)}
      {isHi && (
        <Html
          center
          distanceFactor={28}
          position={[0, radius * labelOffset, 0]}
          zIndexRange={[40, 0]}
        >
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
            <p className="mt-1 font-mono text-[0.55rem] leading-snug text-emerald-900/45">
              {rec.file.split("/").slice(-2).join("/")}:{rec.line}
            </p>
          </div>
        </Html>
      )}
    </group>
  );
}

// Resolve the ion_compartment records into a lookup so each glyph renders the
// REAL emitted status/provenance (no hand-authored status).
function IonCompartments({
  records,
  soma,
  radius,
  frame,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  records: InventoryRecord[];
  soma: THREE.Vector3;
  radius: number;
  frame: React.MutableRefObject<FrameState>;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const byId = useMemo(() => {
    const m = new Map<string, InventoryRecord>();
    for (const r of records) if (r.category === "ion_compartment") m.set(r.id, r);
    return m;
  }, [records]);

  const common = { soma, radius, hovered, selected, setHovered, onSelect };
  const osmoticRef = useRef<THREE.Mesh>(null);

  // Osmotic halo gentle breathing to read as "volume regulation".
  useFrame(() => {
    const m = osmoticRef.current;
    if (!m) return;
    const f = frame.current;
    const s = 1 + 0.04 * Math.sin(f.t * 0.9);
    m.scale.setScalar(s);
  });

  const singlepool = byId.get("ion_singlepool");
  const tissueBuf = byId.get("ion_tissue_buf");
  const spatialK = byId.get("ion_spatial_k");
  const spatialCa = byId.get("ion_spatial_ca");
  const gliaBuf = byId.get("ion_glia_buffer");
  const compartments = byId.get("ion_compartments");
  const osmotic = byId.get("ion_osmotic");
  const perisyn = byId.get("ion_perisynaptic");
  const ephaptic = byId.get("ion_ephaptic");

  // deterministic offsets for the four well-mixed single-pool blobs
  const blobOffsets: Record<string, THREE.Vector3> = {
    Na: new THREE.Vector3(0.4, 0.3, 0.2),
    K: new THREE.Vector3(-0.4, 0.2, -0.3),
    Ca: new THREE.Vector3(0.1, -0.4, 0.35),
    Cl: new THREE.Vector3(-0.25, -0.3, -0.2),
  };
  const blobR: Record<string, number> = { Na: 0.5, K: 0.6, Ca: 0.35, Cl: 0.4 };

  return (
    <group>
      {/* ion_singlepool — ON: the 4 well-mixed intracellular blobs (status-coded) */}
      {singlepool && (
        <CompartmentGlyph rec={singlepool} {...common} labelOffset={1.0}>
          {(style) => (
            <>
              {(["Na", "K", "Ca", "Cl"] as const).map((ion) => (
                <mesh key={ion} position={blobOffsets[ion]}>
                  <sphereGeometry args={[blobR[ion], 20, 20]} />
                  <meshStandardMaterial
                    color={ION_BLOB_COLOR[ion]}
                    transparent
                    opacity={0.16 + 0.14 * style.opacity}
                    roughness={0.6}
                    depthWrite={false}
                  />
                </mesh>
              ))}
            </>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_tissue_buf — ON: thin inline buffer shell hugging the membrane */}
      {tissueBuf && (
        <CompartmentGlyph rec={tissueBuf} {...common} labelOffset={1.12}>
          {(style) => (
            <mesh>
              <sphereGeometry args={[radius * 1.08, 28, 28]} />
              <meshStandardMaterial
                color={style.baseColor}
                transparent
                opacity={0.06 + 0.05 * style.opacity}
                roughness={0.5}
                side={THREE.BackSide}
                depthWrite={false}
              />
            </mesh>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_spatial_k — OFF: 3-pool nested ghost shells (submembrane/bulk/K_out) */}
      {spatialK && (
        <CompartmentGlyph rec={spatialK} {...common} labelOffset={1.5}>
          {(style) => (
            <>
              {[0.78, 1.0, 1.42].map((f, i) => (
                <mesh key={i}>
                  <sphereGeometry args={[radius * f, 22, 22]} />
                  <meshStandardMaterial
                    color={ION_BLOB_COLOR.K}
                    transparent
                    opacity={style.opacity * (0.1 - i * 0.02)}
                    wireframe={i === 2}
                    side={THREE.DoubleSide}
                    depthWrite={false}
                  />
                </mesh>
              ))}
            </>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_spatial_ca — OFF: 2-pool nested ghost shells (submembrane/bulk) */}
      {spatialCa && (
        <CompartmentGlyph rec={spatialCa} {...common} labelOffset={1.32}>
          {(style) => (
            <>
              {[0.7, 0.95].map((f, i) => (
                <mesh key={i}>
                  <sphereGeometry args={[radius * f, 22, 22]} />
                  <meshStandardMaterial
                    color={ION_BLOB_COLOR.Ca}
                    transparent
                    opacity={style.opacity * (0.12 - i * 0.04)}
                    wireframe={i === 1}
                    side={THREE.DoubleSide}
                    depthWrite={false}
                  />
                </mesh>
              ))}
            </>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_glia_buffer — ORPHANED: outer wireframe sink shell (no return path) */}
      {gliaBuf && (
        <CompartmentGlyph rec={gliaBuf} {...common} labelOffset={1.6}>
          {(style) => (
            <mesh>
              <sphereGeometry args={[radius * 1.34, 24, 24]} />
              <meshStandardMaterial
                color={style.baseColor}
                transparent
                opacity={0.06 + 0.06 * style.opacity}
                wireframe
                depthWrite={false}
              />
            </mesh>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_compartments — ORPHANED: wireframe lattice box (Contract-B interface) */}
      {compartments && (
        <CompartmentGlyph rec={compartments} {...common} labelOffset={1.7}>
          {(style) => (
            <mesh>
              <boxGeometry
                args={[radius * 2.7, radius * 2.7, radius * 2.7, 3, 3, 3]}
              />
              <meshStandardMaterial
                color={style.baseColor}
                transparent
                opacity={0.18 + 0.2 * style.opacity}
                wireframe
                depthWrite={false}
              />
            </mesh>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_osmotic — ORPHANED: pulsing volume-regulation halo */}
      {osmotic && (
        <CompartmentGlyph rec={osmotic} {...common} labelOffset={1.85}>
          {(style) => (
            <mesh ref={osmoticRef}>
              <sphereGeometry args={[radius * 1.22, 26, 26]} />
              <meshStandardMaterial
                color={style.baseColor}
                transparent
                opacity={0.05 + 0.07 * style.opacity}
                wireframe
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_perisynaptic — ORPHANED: submembrane band girdle around the soma */}
      {perisyn && (
        <CompartmentGlyph rec={perisyn} {...common} labelOffset={1.45}>
          {(style) => (
            <mesh rotation={[Math.PI / 2, 0, 0]}>
              {/* a thin equatorial band: torus hugging just outside the membrane */}
              <torusGeometry args={[radius * 1.1, radius * 0.12, 12, 48]} />
              <meshStandardMaterial
                color={style.baseColor}
                transparent
                opacity={0.18 + 0.25 * style.opacity}
                wireframe={style.wireframe}
                roughness={0.5}
                depthWrite={false}
              />
            </mesh>
          )}
        </CompartmentGlyph>
      )}

      {/* ion_ephaptic — ORPHANED: external field arc (extracellular coupling) */}
      {ephaptic && (
        <CompartmentGlyph rec={ephaptic} {...common} labelOffset={1.95}>
          {(style) => <EphapticArc radius={radius} style={style} />}
        </CompartmentGlyph>
      )}
    </group>
  );
}

// Ephaptic coupling: a set of field-line arcs bowing out from the membrane,
// rendered as dashed lines per status (extracellular potential coupling).
function EphapticArc({ radius, style }: { radius: number; style: StatusStyle }) {
  const arcs = useMemo(() => {
    const out: THREE.Vector3[][] = [];
    const nArcs = 5;
    for (let a = 0; a < nArcs; a++) {
      const ang = (a / nArcs) * Math.PI * 2;
      const pts: THREE.Vector3[] = [];
      const seg = 24;
      for (let i = 0; i <= seg; i++) {
        const t = i / seg; // 0..1 along the arc
        const phi = Math.PI * t; // 0..pi latitude sweep
        const rr = radius * (1.0 + 0.55 * Math.sin(phi)); // bulge out at equator
        pts.push(
          new THREE.Vector3(
            Math.cos(ang) * rr * Math.sin(phi),
            radius * Math.cos(phi) * 1.1,
            Math.sin(ang) * rr * Math.sin(phi),
          ),
        );
      }
      out.push(pts);
    }
    return out;
  }, [radius]);
  return (
    <group>
      {arcs.map((pts, i) => (
        <Line
          key={i}
          points={pts}
          color={style.baseColor}
          lineWidth={1.4}
          transparent
          opacity={0.35 + 0.4 * style.opacity}
          dashed
          dashScale={5}
          dashSize={0.14}
          gapSize={0.1}
        />
      ))}
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
// Geometry markers — the 3 geometry records made individually cross-
// highlightable in 3D (not just in the legend).
//
//   geo_percell    (ON)          — per-cell C_m / surf / vol; the membrane shell
//                                  itself. Marker pinned at the soma equator.
//   geo_morphology (ON)          — EM-derived µm morphology frame; marker at the
//                                  soma centroid (the frame origin).
//   geo_em_override (default-OFF) — opt-in EM C_m override, VB6 ONLY. Surfaced
//                                  here as an explicit dimmed/amber marker so the
//                                  default-OFF, AVA-does-NOT-use-it fact is
//                                  visible in the scene, not buried in the legend.
//
// Each marker shares the `hovered`/`selected`/`onSelect` ids with the legend so
// clicking the row highlights the marker and vice-versa.
// ---------------------------------------------------------------------------

function GeometryMarker({
  rec,
  pos,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  rec: InventoryRecord;
  pos: THREE.Vector3;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const isHi = hovered === rec.id || selected === rec.id;
  const style = STATUS_STYLE[rec.status];
  const color = rec.status === "on" ? CATEGORY_COLOR.geometry : style.baseColor;

  useFrame(() => {
    const g = groupRef.current;
    if (!g) return;
    const target = isHi ? 1.5 : 1.0;
    g.scale.lerp(new THREE.Vector3(target, target, target), 0.18);
  });

  // Short label so the EM-override default-OFF status is legible at a glance.
  const tag =
    rec.id === "geo_em_override"
      ? "EM override · VB6 · OFF"
      : rec.id === "geo_morphology"
        ? "EM µm frame"
        : "per-cell Cm/surf/vol";

  return (
    <group
      ref={groupRef}
      position={pos}
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
      {/* small diamond marker (octahedron) — wireframe when default-OFF */}
      <mesh>
        <octahedronGeometry args={[0.32, 0]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isHi ? 0.6 : style.emissiveBoost}
          transparent
          opacity={rec.status === "on" ? (isHi ? 1 : 0.85) : style.opacity}
          wireframe={rec.status !== "on"}
          roughness={0.45}
          metalness={0.1}
        />
      </mesh>

      {/* persistent compact tag (always shown so the default-OFF EM override is
          discoverable without hovering); upgrades to a full panel on highlight */}
      {isHi ? (
        <Html center distanceFactor={28} position={[0, 0.9, 0]} zIndexRange={[40, 0]}>
          <div className="pointer-events-none w-52 rounded-lg border border-white/50 bg-white/90 p-2 text-left shadow-lg backdrop-blur-md">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[0.72rem] font-semibold text-emerald-950">{rec.name}</span>
              <span className="shrink-0 font-mono text-[0.55rem] uppercase text-emerald-900/50">
                {rec.status === "off" ? "default-OFF" : rec.status}
              </span>
            </div>
            <p className="mt-0.5 text-[0.62rem] leading-snug text-emerald-900/70">{rec.subtype}</p>
            <p className="mt-1 text-[0.6rem] leading-snug text-emerald-900/60">{rec.physical_desc}</p>
            <p className="mt-1 font-mono text-[0.55rem] leading-snug text-emerald-900/45">
              {rec.file.split("/").slice(-2).join("/")}:{rec.line}
            </p>
          </div>
        </Html>
      ) : (
        <Html center distanceFactor={32} position={[0, 0.55, 0]} zIndexRange={[20, 0]}>
          <div
            className={`pointer-events-none whitespace-nowrap rounded border px-1.5 py-0.5 text-[0.55rem] font-medium shadow-sm backdrop-blur-sm ${
              rec.status === "on"
                ? "border-emerald-600/30 bg-white/80 text-emerald-900/80"
                : "border-amber-500/45 bg-amber-50/85 text-amber-900/90"
            }`}
          >
            {tag}
          </div>
        </Html>
      )}
    </group>
  );
}

function GeometryMarkers({
  records,
  somaPos,
  somaR,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  records: InventoryRecord[];
  somaPos: THREE.Vector3;
  somaR: number;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const geoRecs = useMemo(
    () => records.filter((r) => r.category === "geometry"),
    [records],
  );

  // Deterministic, distinct anchor per geometry record around the soma so the
  // three markers don't overlap.
  const placement: Record<string, THREE.Vector3> = useMemo(() => {
    const m: Record<string, THREE.Vector3> = {};
    // geo_morphology → soma centroid (frame origin)
    m["geo_morphology"] = somaPos.clone();
    // geo_percell → soma equator, +X side (the membrane shell / passive ID)
    m["geo_percell"] = somaPos
      .clone()
      .add(new THREE.Vector3(somaR * 1.45, 0, 0));
    // geo_em_override → opposite side, slightly up; default-OFF & VB6-only
    m["geo_em_override"] = somaPos
      .clone()
      .add(new THREE.Vector3(-somaR * 1.35, somaR * 0.9, 0));
    return m;
  }, [somaPos, somaR]);

  return (
    <group>
      {geoRecs.map((rec) => (
        <GeometryMarker
          key={rec.id}
          rec={rec}
          pos={placement[rec.id] ?? somaPos}
          hovered={hovered}
          selected={selected}
          setHovered={setHovered}
          onSelect={onSelect}
        />
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Metabolism markers — ALL THREE metabolism records get a 3D presence so the
// audit is legible in-place (not just in the legend):
//
//   met_wcm_ava_proteome  (status ON)  → amber WCM proteome badge above soma
//   met_atp_vars          (status OFF) → dimmed mitochondrion glyph INSIDE the
//                                        soma (declared but inert state vars)
//   met_icel1314          (status OFF) → ghosted reaction-network badge beside
//                                        the soma (46-reaction FBA, flag-OFF)
//
// Each is hover/click cross-highlighted via the shared `hovered`/`selected`
// ids, exactly like the membrane & geometry markers, so clicking the legend
// row lights up the marker and vice-versa.
// ---------------------------------------------------------------------------

// Shared tooltip body for the metabolism markers (mirrors the glyph tooltip).
function MetaTooltip({ rec, offsetY }: { rec: InventoryRecord; offsetY: number }) {
  return (
    <Html center distanceFactor={28} position={[0, offsetY, 0]} zIndexRange={[40, 0]}>
      <div className="pointer-events-none w-56 rounded-lg border border-white/50 bg-white/90 p-2 text-left shadow-lg backdrop-blur-md">
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
        <p className="mt-1 font-mono text-[0.55rem] leading-snug text-emerald-900/45">
          {rec.file.split("/").slice(-2).join("/")}:{rec.line}
        </p>
      </div>
    </Html>
  );
}

// Dimmed mitochondrion glyph — a cristae-ribbed capsule sitting inside the soma
// volume. Rendered for the OFF `met_atp_vars` record (ATP/NADH/mito state vars
// declared but inert), so it reads as "the powerhouse is modelled but dark".
function MitochondrionGlyph({
  rec,
  soma,
  radius,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  rec: InventoryRecord;
  soma: THREE.Vector3;
  radius: number;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const isHi = hovered === rec.id || selected === rec.id;
  const style = STATUS_STYLE[rec.status]; // off → dimmed
  const color = CATEGORY_COLOR.metabolism;

  // Offset to a quiet corner of the soma interior so it does not sit on the
  // ion pools; oblong long-axis tilted for readability.
  const center = useMemo(
    () =>
      new THREE.Vector3(
        soma.x - radius * 0.18,
        soma.y - radius * 0.12,
        soma.z + radius * 0.22,
      ),
    [soma, radius],
  );
  const len = radius * 0.95;
  const rad = radius * 0.26;

  useFrame(() => {
    const g = groupRef.current;
    if (!g) return;
    const target = isHi ? 1.4 : 1.0;
    g.scale.lerp(new THREE.Vector3(target, target, target), 0.18);
  });

  // A few cristae discs along the long axis to read as a mitochondrion.
  const cristae = useMemo(() => {
    const out: number[] = [];
    const n = 4;
    for (let i = 0; i < n; i++) out.push((i / (n - 1) - 0.5) * len * 0.9);
    return out;
  }, [len]);

  return (
    <group
      ref={groupRef}
      position={center}
      rotation={[0.5, 0.3, 0.9]}
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
      {/* outer membrane — dimmed capsule */}
      <mesh>
        <capsuleGeometry args={[rad, len, 8, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isHi ? 0.35 : 0.05}
          transparent
          opacity={isHi ? 0.6 : style.opacity}
          roughness={0.5}
          metalness={0.1}
        />
      </mesh>
      {/* cristae — ribbed inner discs, faint */}
      {cristae.map((y, i) => (
        <mesh key={i} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[rad * 0.82, rad * 0.12, 6, 14]} />
          <meshStandardMaterial
            color={color}
            transparent
            opacity={isHi ? 0.5 : 0.22}
            roughness={0.6}
          />
        </mesh>
      ))}
      {isHi && <MetaTooltip rec={rec} offsetY={len * 0.5 + radius * 0.5} />}
    </group>
  );
}

// Ghosted reaction-network badge — a small wireframe node-and-edge graph that
// reads as a metabolic flux network, for the OFF `met_icel1314` record
// (46-reaction iCEL1314 FBA subset; build flag default False).
function ReactionNetworkBadge({
  rec,
  soma,
  radius,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  rec: InventoryRecord;
  soma: THREE.Vector3;
  radius: number;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const isHi = hovered === rec.id || selected === rec.id;
  const color = CATEGORY_COLOR.metabolism;

  // Deterministic little graph: nodes on a jittered spiral + a few edges.
  const { nodes, edges } = useMemo(() => {
    const nN = 7;
    const nodes: THREE.Vector3[] = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < nN; i++) {
      const a = golden * i;
      const rr = 0.45 + (0.35 * ((i * 7) % 5)) / 5;
      nodes.push(
        new THREE.Vector3(Math.cos(a) * rr, ((i % 3) - 1) * 0.35, Math.sin(a) * rr),
      );
    }
    const edges: [THREE.Vector3, THREE.Vector3][] = [];
    for (let i = 0; i < nN - 1; i++) edges.push([nodes[i], nodes[i + 1]]);
    edges.push([nodes[0], nodes[3]]);
    edges.push([nodes[2], nodes[5]]);
    edges.push([nodes[1], nodes[6]]);
    return { nodes, edges };
  }, []);

  // Sit just outside the soma, opposite the WCM badge.
  const center = useMemo(
    () =>
      new THREE.Vector3(soma.x + radius * 1.55, soma.y - radius * 0.2, soma.z),
    [soma, radius],
  );
  const s = radius * 0.7;

  useFrame(() => {
    const g = groupRef.current;
    if (!g) return;
    const target = isHi ? 1.35 : 1.0;
    g.scale.lerp(new THREE.Vector3(target * s, target * s, target * s), 0.18);
    g.rotation.y += 0.0015; // slow idle drift so the graph reads as 3D
  });

  return (
    <group
      ref={groupRef}
      position={center}
      scale={[s, s, s]}
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
      {/* edges + nodes are in local units; the group scale (s) + hover pulse
          are both applied in useFrame so they compose cleanly. */}
      {edges.map((e, i) => (
        <Line
          key={`e-${i}`}
          points={[e[0], e[1]]}
          color={color}
          lineWidth={isHi ? 1.6 : 1.0}
          transparent
          opacity={isHi ? 0.75 : 0.4}
        />
      ))}
      {/* nodes — faint wireframe spheres (reactions/metabolites) */}
      {nodes.map((p, i) => (
        <mesh key={`n-${i}`} position={p}>
          <sphereGeometry args={[0.12, 10, 10]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={isHi ? 0.4 : 0.0}
            transparent
            opacity={isHi ? 0.85 : 0.5}
            wireframe={!isHi}
            roughness={0.6}
          />
        </mesh>
      ))}
      <Html center distanceFactor={30} position={[0, 1.1, 0]}>
        <div
          className={`pointer-events-none whitespace-nowrap rounded-md border px-2 py-0.5 text-[0.55rem] font-medium shadow backdrop-blur-sm ${
            isHi
              ? "border-amber-600/60 bg-amber-50 text-amber-900"
              : "border-amber-700/25 bg-amber-50/55 text-amber-900/55"
          }`}
        >
          iCEL1314 · 46-rxn FBA (default-OFF)
        </div>
      </Html>
      {isHi && <MetaTooltip rec={rec} offsetY={1.7} />}
    </group>
  );
}

// WCM proteome badge — AVA-only proteome marker (status ON). Record-linked so it
// cross-highlights with the legend like the other metabolism markers.
function WcmBadge({
  rec,
  soma,
  radius,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  rec: InventoryRecord | undefined;
  soma: THREE.Vector3;
  radius: number;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const isHi = rec != null && (hovered === rec.id || selected === rec.id);
  return (
    <Html center distanceFactor={30} position={[soma.x, soma.y + radius * 1.7, soma.z]}>
      <div
        className={`whitespace-nowrap rounded-md border px-2 py-0.5 text-[0.6rem] font-medium shadow backdrop-blur-sm transition-colors ${
          isHi
            ? "border-amber-600/70 bg-amber-100 text-amber-950"
            : "border-amber-500/40 bg-amber-50/90 text-amber-900"
        } ${rec ? "cursor-pointer" : "pointer-events-none"}`}
        onMouseEnter={() => rec && setHovered(rec.id)}
        onMouseLeave={() => rec && setHovered(null)}
        onClick={() => rec && onSelect(rec.id)}
      >
        WCM proteome · AVA-only (metabolism vars declared, inert)
      </div>
    </Html>
  );
}

// MetabolismMarkers — picks the 3 metabolism records out of the inventory and
// renders the right marker per id, so the metabolism audit is fully present in
// 3D (the OFF mitochondrion + OFF reaction network, plus the ON WCM badge).
function MetabolismMarkers({
  records,
  somaPos,
  somaR,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  records: InventoryRecord[];
  somaPos: THREE.Vector3;
  somaR: number;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const byId = useMemo(() => {
    const m = new Map<string, InventoryRecord>();
    for (const r of records) if (r.category === "metabolism") m.set(r.id, r);
    return m;
  }, [records]);

  const atp = byId.get("met_atp_vars");
  const icel = byId.get("met_icel1314");
  const wcm = byId.get("met_wcm_ava_proteome");

  return (
    <group>
      {atp && (
        <MitochondrionGlyph
          rec={atp}
          soma={somaPos}
          radius={somaR}
          hovered={hovered}
          selected={selected}
          setHovered={setHovered}
          onSelect={onSelect}
        />
      )}
      {icel && (
        <ReactionNetworkBadge
          rec={icel}
          soma={somaPos}
          radius={somaR}
          hovered={hovered}
          selected={selected}
          setHovered={setHovered}
          onSelect={onSelect}
        />
      )}
      <WcmBadge
        rec={wcm}
        soma={somaPos}
        radius={somaR}
        hovered={hovered}
        selected={selected}
        setHovered={setHovered}
        onSelect={onSelect}
      />
    </group>
  );
}

// ---------------------------------------------------------------------------
// HeroCell3D — assembles everything.
// ---------------------------------------------------------------------------

export default function HeroCell3D({
  morph,
  gbar,
  signatures,
  records,
  frame,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  morph: HeroMorphology;
  gbar: HeroChannelGbar;
  signatures?: GlyphSignatures;
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
      // AlphaFold/PDB-derived shape signature, keyed by bare channel id.
      const sig = signatures?.signatures?.[signatureKey(rec.id)];
      return {
        rec,
        pos: a.pos,
        normal: a.normal,
        size,
        color,
        shape: shapeFor(rec.category),
        sig,
      };
    });
  }, [glyphRecords, anchors, gbarById, maxG, gbar, signatures]);

  return (
    <group>
      <MembraneShell morph={morph} frame={frame} center={center} />
      <IonCompartments
        records={records}
        soma={somaPos}
        radius={somaR}
        frame={frame}
        hovered={hovered}
        selected={selected}
        setHovered={setHovered}
        onSelect={onSelect}
      />
      <FlowParticles soma={somaPos} radius={somaR} frame={frame} />
      <GeometryMarkers
        records={records}
        somaPos={somaPos}
        somaR={somaR}
        hovered={hovered}
        selected={selected}
        setHovered={setHovered}
        onSelect={onSelect}
      />
      <MetabolismMarkers
        records={records}
        somaPos={somaPos}
        somaR={somaR}
        hovered={hovered}
        selected={selected}
        setHovered={setHovered}
        onSelect={onSelect}
      />
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
