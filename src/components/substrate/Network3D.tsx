import * as React from "react";
import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";

/**
 * Network3D — the 300-cell C. elegans connectome rendered in 3D, Scene 2/3.
 *
 * Everything is from emitted, read-only substrate data (no hand-authored
 * geometry or positions):
 *   - 300 soma rendered as an InstancedMesh at REAL EM µm soma centroids
 *     (network_positions.json — morphology/loader.py:soma_centroid_um).
 *   - Per-cell colour driven LIVE by the REAL whole-network Brian2 voltage
 *     trajectory (trajectory.json — V_mV[frame][cell]); the emissive soma glow
 *     is driven by each cell's REAL total transmembrane ionic-current magnitude
 *     (trajectory.json net_flows — per-cell I_Na/I_K/I_Ca/pump, network_builder
 *     .py:380-423), falling back to Ca²⁺ when net_flows is absent. A second
 *     "channel family" colour mode tints each soma by its dominant gbar family
 *     (hero_channel_gbar.cell_channel_table).
 *   - Connectome edges from network_edges.json (artifacts/connectome.npz):
 *       · gap junctions  — ohmic, innexin-typed, symmetric (blue)
 *       · chem synapses  — signed (excitatory green / inhibitory rose), width
 *                          by |w|, neurotransmitter-typed
 *     Both layers are independently toggleable.
 *   - Network-wide ion-flow particle layer: packets travel presynaptic →
 *     postsynaptic along the strongest chem edges, packet speed driven by the
 *     source cell's REAL total ionic-current magnitude (net_flows), falling back
 *     to the source-cell depolarisation-rate proxy when net_flows is absent.
 *
 * Per-frame recolour / particle motion is written imperatively inside
 * useFrame so it never triggers a React re-render.
 */

// ---------------------------------------------------------------------------
// Types (structurally compatible with SubstrateAnatomy.tsx)
// ---------------------------------------------------------------------------

export type NetworkCell = {
  cell: string;
  x: number;
  y: number;
  z: number;
  class: string;
  has_morphology: boolean;
  // EM sphere-equiv soma radius (µm); null when the cell has no soma segments.
  soma_r_um: number | null;
};

export type NetworkPositions = {
  n_cells: number;
  n_with_position: number;
  n_with_soma_r?: number;
  soma_r_um_median?: number | null;
  cells: NetworkCell[];
};

export type ChemEdge = {
  s: number;
  t: number;
  w: number;
  raw: number;
  sign: number;
  nt: string;
};
export type GapEdge = { a: number; b: number; w: number };

export type NetworkEdges = {
  n_nodes: number;
  names: string[];
  n_chem_edges: number;
  n_gap_edges: number;
  chem: ChemEdge[];
  gap: GapEdge[];
};

export type Trajectory = {
  real: boolean;
  illustrative: boolean;
  dt_ms: number;
  n_frames: number;
  n_cells: number;
  hero: string;
  cells: string[];
  V_mV: number[][];
  Ca_uM: number[][];
  // Per-cell ionic currents for the FULL network (FIX round 4), [frame][cell],
  // integer units of (1/net_flow_scale) uA/cm^2. Optional — falls back to the
  // voltage-derived proxy when absent (older trajectory.json).
  net_flow_scale?: number;
  net_flows?: { I_Na: number[][]; I_K: number[][]; I_Ca: number[][]; pump: number[][] };
};

export type CellChannelRow = {
  cell: string;
  class: string;
  n_channels_expressed: number;
  gbar_by_family: Record<string, number>;
  dominant_family: string;
  cm_pF?: number;
  e_leak_mV?: number;
  // MODULE names (gbar>0) of every channel/receptor this cell expresses. Match
  // the inventory record id suffix (chan_<module>) so a clicked channel/receptor
  // legend row can cross-highlight the exact set of expressing somata.
  channels_expressed?: string[];
};

export type NetworkColorMode = "voltage" | "family";

// Shared clock ref — SubstrateAnatomy's TrajectoryClock advances `frameIndex`
// for the whole network (integer trajectory frame) on its rAF tick.
export type NetClock = {
  frameIndex: number; // current integer trajectory frame
  frameFloat: number; // fractional, for sub-frame interpolation
  t: number; // seconds, monotonic animation clock
};

// ---------------------------------------------------------------------------
// Palette (house: forest / cream / amber, with ion-family + signal accents)
// ---------------------------------------------------------------------------

const FAMILY_COLOR: Record<string, string> = {
  Ca: "#d6743c",
  K: "#3c6fd6",
  Na: "#d63c8a",
  Cl: "#3cb7d6",
  CaK_brake: "#7d4cd6",
  receptor: "#2f9e6b",
  none: "#8a8f7a",
};

const SCALE = 0.045; // µm → scene units (A-P span ~720 µm → ~32 units)

// Voltage → forest→cream→amber→coral ramp. Network regime visits a wider
// window than the hero (down to ~-91, up to ~+1) so use the full span.
const V_MIN = -85;
const V_MAX = -30;
function voltageColor(v: number, out: THREE.Color): THREE.Color {
  const t = Math.min(1, Math.max(0, (v - V_MIN) / (V_MAX - V_MIN)));
  const cool = COLOR_COOL;
  const mid = COLOR_MID;
  const warm = COLOR_WARM;
  if (t < 0.5) out.copy(cool).lerp(mid, t / 0.5);
  else out.copy(mid).lerp(warm, (t - 0.5) / 0.5);
  return out;
}
const COLOR_COOL = new THREE.Color("#1f5c3a");
const COLOR_MID = new THREE.Color("#e2d4a8");
const COLOR_WARM = new THREE.Color("#d6622b");

// ---------------------------------------------------------------------------
// Soma instances — one InstancedMesh, recoloured per frame from trajectory.
// ---------------------------------------------------------------------------

function SomaInstances({
  positions,
  traj,
  clock,
  colorMode,
  familyByCell,
  hovered,
  selectedIndices,
  setHovered,
  onSelect,
  trajIndexByCell,
}: {
  positions: NetworkPositions;
  traj: Trajectory;
  clock: React.MutableRefObject<NetClock>;
  colorMode: NetworkColorMode;
  familyByCell: Map<string, string>;
  hovered: string | null;
  // Cell-instance indices to cross-highlight. For a single selected cell this
  // is one index; for a selected CHANNEL/RECEPTOR record it is every soma
  // expressing that channel (gbar>0). Empty when nothing maps into the network.
  selectedIndices: number[];
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
  trajIndexByCell: Int32Array;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const n = positions.cells.length;
  const tmpColor = useRef(new THREE.Color());
  const tmpMat = useRef(new THREE.Matrix4());
  const dummy = useRef(new THREE.Object3D());
  const selSetRef = useRef<Set<number>>(new Set());

  // Static base layout: place each instance at its scaled µm centroid.
  const center = useMemo(() => {
    const c = new THREE.Vector3();
    positions.cells.forEach((p) => c.add(new THREE.Vector3(p.x, p.y, p.z)));
    c.multiplyScalar(1 / Math.max(1, n));
    return c;
  }, [positions, n]);

  // Geometry base radius (the shared sphere); per-instance size variation is
  // applied as a scale factor below so the InstancedMesh stays one draw call.
  const baseR = 0.42;

  // Per-instance soma-size scale — REAL EM-measured variation.
  // network_positions.json carries soma_r_um (sphere-equiv radius from the
  // cell's soma surface area, morphology/loader.py:soma_segments). We scale
  // each glyph by soma_r_um / median, clamped to keep the dense neuropil
  // legible. Cells with no soma data (soma_r_um null) fall back to 1.0.
  const sizeScale = useMemo(() => {
    const med = positions.soma_r_um_median ?? null;
    const arr = new Float32Array(n);
    positions.cells.forEach((p, i) => {
      const r = p.soma_r_um;
      if (med && med > 0 && r != null && r > 0) {
        // sqrt-compress the ratio so the ~3.7× radius spread reads as a
        // clear-but-not-overwhelming ~1.9× glyph spread; clamp the tails.
        const s = Math.sqrt(r / med);
        arr[i] = Math.min(1.6, Math.max(0.62, s));
      } else {
        arr[i] = 1.0;
      }
    });
    return arr;
  }, [positions, n]);

  // Family colours (static, mode === "family").
  const familyColors = useMemo(() => {
    const arr = new Float32Array(n * 3);
    positions.cells.forEach((p, i) => {
      const fam = familyByCell.get(p.cell) ?? "none";
      const c = new THREE.Color(FAMILY_COLOR[fam] ?? FAMILY_COLOR.none);
      arr[i * 3] = c.r;
      arr[i * 3 + 1] = c.g;
      arr[i * 3 + 2] = c.b;
    });
    return arr;
  }, [positions, familyByCell, n]);

  // Per-cell total transmembrane ionic-current MAGNITUDE per frame, normalised
  // to [0,1] (FIX round 4). When trajectory.json carries net_flows (real per-
  // cell I_Na/I_K/I_Ca/pump), the soma glow is driven by actual ionic flux
  // |I_Na|+|I_K|+|I_Ca|+|pump| rather than Ca²⁺ alone — current-accurate. The
  // 95th-percentile of the magnitude is used as the normaliser so a few high-
  // flux cells don't wash the rest out. Null when net_flows is absent (older
  // trajectory.json) → caller falls back to the Ca²⁺-derived glow.
  const fluxByFrameCell = useMemo<Float32Array | null>(() => {
    const nf = traj.net_flows;
    const scale = traj.net_flow_scale ?? 20.0;
    if (!nf || !nf.I_Na) return null;
    const F = traj.n_frames;
    const C = traj.n_cells;
    const out = new Float32Array(F * C); // raw |I| magnitude in uA/cm^2
    const samples: number[] = [];
    for (let f = 0; f < F; f++) {
      const na = nf.I_Na[f];
      const k = nf.I_K[f];
      const ca = nf.I_Ca[f];
      const pu = nf.pump[f];
      if (!na) continue;
      for (let c = 0; c < C; c++) {
        const mag =
          (Math.abs(na[c] ?? 0) +
            Math.abs(k[c] ?? 0) +
            Math.abs(ca[c] ?? 0) +
            Math.abs(pu[c] ?? 0)) /
          scale; // int units → uA/cm^2
        out[f * C + c] = mag;
        if ((f & 7) === 0) samples.push(mag); // subsample for the percentile
      }
    }
    // Robust normaliser: 95th percentile of sampled magnitudes (min 1 uA/cm^2).
    samples.sort((a, b) => a - b);
    const p95 = samples.length
      ? samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.95))]
      : 1;
    const norm = Math.max(1, p95);
    for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i] / norm);
    return out;
  }, [traj]);

  // Initialise transforms + instanceColor buffer once.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const d = dummy.current;
    positions.cells.forEach((p, i) => {
      d.position.set(
        (p.x - center.x) * SCALE,
        (p.y - center.y) * SCALE,
        (p.z - center.z) * SCALE,
      );
      d.scale.setScalar(sizeScale[i]);
      d.updateMatrix();
      mesh.setMatrixAt(i, d.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (!mesh.instanceColor) {
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(n * 3),
        3,
      );
    }
    mesh.instanceColor.needsUpdate = true;
  }, [positions, center, n, sizeScale]);

  // Per-frame recolour (voltage / Ca) or static family tint + hover scale.
  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh || !mesh.instanceColor) return;
    const fi = clock.current.frameIndex % traj.n_frames;
    const Vrow = traj.V_mV[fi];
    const Carow = traj.Ca_uM[fi];
    const ic = mesh.instanceColor.array as Float32Array;
    const hiIdx =
      hovered != null ? positions.cells.findIndex((c) => c.cell === hovered) : -1;
    // Selected set: a single soma (cell name) OR every soma expressing a
    // selected channel/receptor record. A soft sinusoidal pulse keeps a large
    // multi-cell selection legible as one coordinated group.
    const selSet = selSetRef.current;
    selSet.clear();
    for (const idx of selectedIndices) if (idx >= 0) selSet.add(idx);
    const pulse = 0.5 + 0.5 * Math.sin(clock.current.t * 4.0);

    for (let i = 0; i < n; i++) {
      if (colorMode === "voltage") {
        const ti = trajIndexByCell[i];
        const v = ti >= 0 ? Vrow?.[ti] : undefined;
        if (Number.isFinite(v)) {
          voltageColor(v as number, tmpColor.current);
          // Glow brightens active cells. Current-accurate when net_flows is
          // present (total ionic flux magnitude); else Ca²⁺-derived fallback.
          if (fluxByFrameCell && ti >= 0) {
            const fluxN = fluxByFrameCell[fi * traj.n_cells + ti] ?? 0;
            tmpColor.current.offsetHSL(0, 0, 0.18 * fluxN);
          } else {
            const ca = ti >= 0 ? Carow?.[ti] : 0;
            const caN = Math.min(1, Math.max(0, ((ca ?? 0) - 0.2) / 1.5));
            tmpColor.current.offsetHSL(0, 0, 0.12 * caN);
          }
        } else {
          tmpColor.current.set("#3a3f33"); // no-data soma — dark olive
        }
        ic[i * 3] = tmpColor.current.r;
        ic[i * 3 + 1] = tmpColor.current.g;
        ic[i * 3 + 2] = tmpColor.current.b;
      } else {
        ic[i * 3] = familyColors[i * 3];
        ic[i * 3 + 1] = familyColors[i * 3 + 1];
        ic[i * 3 + 2] = familyColors[i * 3 + 2];
      }
      // Selected (incl. channel-expressing) somata pulse warm-amber so a
      // multi-cell channel/receptor selection reads as one highlighted set.
      if (selSet.has(i)) {
        const k = 0.35 + 0.45 * pulse;
        ic[i * 3] = ic[i * 3] * (1 - k) + 0.92 * k;
        ic[i * 3 + 1] = ic[i * 3 + 1] * (1 - k) + 0.62 * k;
        ic[i * 3 + 2] = ic[i * 3 + 2] * (1 - k) + 0.18 * k;
      }
    }
    mesh.instanceColor.needsUpdate = true;

    // Hover / select pop — rescale highlighted instances. The selected set may
    // be one soma or many (every cell expressing a clicked channel); selected
    // somata gently pulse in size, the hovered soma pops to a fixed size.
    const highlighted = new Set<number>(selSet);
    if (hiIdx >= 0) highlighted.add(hiIdx);
    for (const idx of highlighted) {
      const p = positions.cells[idx];
      const d = dummy.current;
      d.position.set(
        (p.x - center.x) * SCALE,
        (p.y - center.y) * SCALE,
        (p.z - center.z) * SCALE,
      );
      const selPop = selSet.has(idx) ? 1.75 + 0.45 * pulse : 1.0;
      const hoverPop = idx === hiIdx ? 1.7 : 1.0;
      d.scale.setScalar(Math.max(selPop, hoverPop) * sizeScale[idx]);
      d.updateMatrix();
      mesh.setMatrixAt(idx, d.matrix);
    }
    // Reset any previously-scaled instance that is no longer highlighted.
    const prev = (mesh as unknown as { _prevHi?: number[] })._prevHi ?? [];
    for (const idx of prev) {
      if (highlighted.has(idx) || idx < 0) continue;
      const p = positions.cells[idx];
      const d = dummy.current;
      d.position.set(
        (p.x - center.x) * SCALE,
        (p.y - center.y) * SCALE,
        (p.z - center.z) * SCALE,
      );
      d.scale.setScalar(sizeScale[idx]);
      d.updateMatrix();
      mesh.setMatrixAt(idx, d.matrix);
    }
    (mesh as unknown as { _prevHi?: number[] })._prevHi = Array.from(highlighted);
    mesh.instanceMatrix.needsUpdate = true;
  });

  // Picking via instanced raycast.
  const onMove = (e: { instanceId?: number; stopPropagation: () => void }) => {
    e.stopPropagation();
    if (e.instanceId == null) return;
    const cell = positions.cells[e.instanceId];
    if (cell) {
      setHovered(cell.cell);
      document.body.style.cursor = "pointer";
    }
  };

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, n]}
      onPointerMove={onMove}
      onPointerOut={(e) => {
        e.stopPropagation();
        setHovered(null);
        document.body.style.cursor = "auto";
      }}
      onClick={(e) => {
        e.stopPropagation();
        if (e.instanceId == null) return;
        const cell = positions.cells[e.instanceId];
        if (cell) onSelect(cell.cell);
      }}
    >
      <sphereGeometry args={[baseR, 14, 14]} />
      <meshStandardMaterial
        roughness={0.45}
        metalness={0.08}
        toneMapped={false}
      />
    </instancedMesh>
  );
}

// ---------------------------------------------------------------------------
// Edge layers — gap (ohmic, symmetric) and chem (signed).
//
// HYBRID rendering for depth legibility (FIX round 1, 2026-06-19):
//   · The strongest edges (top-N by |w|) are upgraded to solid 3D geometry so
//     they read with depth/occlusion instead of flat 1px lines:
//       - chem  → CURVED ARC tubes (quadratic Bézier bowed off the chord);
//                 the bow encodes pre→post direction (arc bows toward post) and
//                 separates the dense central neuropil so edges don't overlap.
//       - gap   → STRAIGHT tubes (ohmic, symmetric — no direction to encode).
//     All strong tubes for a family are merged into ONE BufferGeometry with
//     per-vertex colour (signed/innexin-weighted) — a single draw call, no
//     per-edge React nodes.
//   · The long tail stays as LineSegments (cheap, fills in the fine structure).
//
// Tube radius scales with |w| so the dominant routes are visually heaviest.
// ---------------------------------------------------------------------------

// How many of the strongest edges per family get promoted to tubes. Sized from
// the weight distribution (chem |w|≥~12 ≈ top 400; gap w≥~3 ≈ top 400) — heavy
// enough to be the visible backbone, light enough to stay one merged mesh.
const CHEM_TUBE_COUNT = 400;
const GAP_TUBE_COUNT = 400;

// Build a merged tube geometry from a list of edges. Each edge becomes either a
// straight tube (bow = 0) or a quadratic-Bézier arc (bow > 0) bowed off the
// chord. Vertex colours are baked in (alpha pre-multiplied) to match the line
// fallback's additive look. Returns one BufferGeometry (single draw call).
function buildTubeGeometry(
  tubes: Array<{
    a: THREE.Vector3;
    b: THREE.Vector3;
    color: THREE.Color;
    alpha: number;
    radius: number;
    bow: number; // 0 = straight, >0 = arc height as a fraction of chord length
  }>,
  radialSegments: number,
  pathSegments: number,
): THREE.BufferGeometry {
  if (tubes.length === 0) return new THREE.BufferGeometry();
  const geoms: THREE.BufferGeometry[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  const altUp = new THREE.Vector3(1, 0, 0);
  for (const tu of tubes) {
    const chord = new THREE.Vector3().subVectors(tu.b, tu.a);
    const len = chord.length();
    if (len < 1e-5) continue;
    let curve: THREE.Curve<THREE.Vector3>;
    if (tu.bow > 0) {
      // Pick an offset direction roughly perpendicular to the chord so the arc
      // bows out of the dense central line; deterministic (no flicker).
      const dir = chord.clone().normalize();
      let perp = new THREE.Vector3().crossVectors(dir, up);
      if (perp.lengthSq() < 1e-4) perp = new THREE.Vector3().crossVectors(dir, altUp);
      perp.normalize();
      const mid = new THREE.Vector3()
        .addVectors(tu.a, tu.b)
        .multiplyScalar(0.5)
        .addScaledVector(perp, len * tu.bow);
      curve = new THREE.QuadraticBezierCurve3(tu.a.clone(), mid, tu.b.clone());
    } else {
      curve = new THREE.LineCurve3(tu.a.clone(), tu.b.clone());
    }
    const seg = tu.bow > 0 ? pathSegments : 1;
    const g = new THREE.TubeGeometry(curve, seg, tu.radius, radialSegments, false);
    const vc = g.getAttribute("position").count;
    const colArr = new Float32Array(vc * 3);
    const r = tu.color.r * tu.alpha;
    const gg = tu.color.g * tu.alpha;
    const bb = tu.color.b * tu.alpha;
    for (let i = 0; i < vc; i++) {
      colArr[i * 3] = r;
      colArr[i * 3 + 1] = gg;
      colArr[i * 3 + 2] = bb;
    }
    g.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    geoms.push(g);
  }
  if (geoms.length === 0) return new THREE.BufferGeometry();
  const merged = mergeBufferGeometries(geoms);
  geoms.forEach((g) => g.dispose());
  return merged ?? new THREE.BufferGeometry();
}

// Minimal non-indexed BufferGeometry merge (position + color only). Avoids
// pulling in three/examples BufferGeometryUtils for one call. All inputs are
// non-indexed TubeGeometries with matching attributes.
function mergeBufferGeometries(
  geoms: THREE.BufferGeometry[],
): THREE.BufferGeometry | null {
  let totalPos = 0;
  for (const g of geoms) {
    const p = g.getAttribute("position");
    const idx = g.getIndex();
    totalPos += idx ? idx.count : p.count;
  }
  const pos = new Float32Array(totalPos * 3);
  const col = new Float32Array(totalPos * 3);
  let o = 0;
  for (const g of geoms) {
    const p = g.getAttribute("position") as THREE.BufferAttribute;
    const c = g.getAttribute("color") as THREE.BufferAttribute;
    const idx = g.getIndex();
    if (idx) {
      for (let i = 0; i < idx.count; i++) {
        const v = idx.getX(i);
        pos[o * 3] = p.getX(v);
        pos[o * 3 + 1] = p.getY(v);
        pos[o * 3 + 2] = p.getZ(v);
        col[o * 3] = c.getX(v);
        col[o * 3 + 1] = c.getY(v);
        col[o * 3 + 2] = c.getZ(v);
        o++;
      }
    } else {
      for (let i = 0; i < p.count; i++) {
        pos[o * 3] = p.getX(i);
        pos[o * 3 + 1] = p.getY(i);
        pos[o * 3 + 2] = p.getZ(i);
        col[o * 3] = c.getX(i);
        col[o * 3 + 1] = c.getY(i);
        col[o * 3 + 2] = c.getZ(i);
        o++;
      }
    }
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  merged.setAttribute("color", new THREE.BufferAttribute(col, 3));
  merged.computeVertexNormals();
  return merged;
}

function buildEdgeGeometry(
  segments: Array<{
    a: THREE.Vector3;
    b: THREE.Vector3;
    color: THREE.Color;
    alpha: number;
  }>,
): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(segments.length * 6);
  const col = new Float32Array(segments.length * 6);
  segments.forEach((s, i) => {
    pos[i * 6] = s.a.x;
    pos[i * 6 + 1] = s.a.y;
    pos[i * 6 + 2] = s.a.z;
    pos[i * 6 + 3] = s.b.x;
    pos[i * 6 + 4] = s.b.y;
    pos[i * 6 + 5] = s.b.z;
    const r = s.color.r * s.alpha;
    const gg = s.color.g * s.alpha;
    const bb = s.color.b * s.alpha;
    for (const off of [0, 3]) {
      col[i * 6 + off] = r;
      col[i * 6 + off + 1] = gg;
      col[i * 6 + off + 2] = bb;
    }
  });
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return g;
}

function EdgeLayers({
  edges,
  positions,
  center,
  showGap,
  showChem,
  showEdges,
  revealedNodes,
}: {
  edges: NetworkEdges;
  positions: NetworkPositions;
  center: THREE.Vector3;
  showGap: boolean;
  showChem: boolean;
  // master edges flag: true → render the full connectome; false → declutter and
  // render ONLY edges that touch a revealed node (selected/hovered cell + 1-hop).
  showEdges: boolean;
  revealedNodes: Set<number>;
}) {
  // When decluttered, keep an edge only if one endpoint is a revealed node.
  const keepGap = useMemo(
    () =>
      showEdges
        ? (_g: GapEdge) => true
        : (g: GapEdge) => revealedNodes.has(g.a) || revealedNodes.has(g.b),
    [showEdges, revealedNodes],
  );
  const keepChem = useMemo(
    () =>
      showEdges
        ? (_c: ChemEdge) => true
        : (c: ChemEdge) => revealedNodes.has(c.s) || revealedNodes.has(c.t),
    [showEdges, revealedNodes],
  );
  // Map edge node indices (network_edges.names order) to scene positions.
  const nodePos = useMemo(() => {
    const byName = new Map(positions.cells.map((c) => [c.cell, c]));
    return edges.names.map((nm) => {
      const c = byName.get(nm);
      if (!c) return new THREE.Vector3(0, 0, 0);
      return new THREE.Vector3(
        (c.x - center.x) * SCALE,
        (c.y - center.y) * SCALE,
        (c.z - center.z) * SCALE,
      );
    });
  }, [edges.names, positions, center]);

  const maxGap = useMemo(
    () => Math.max(1, ...edges.gap.map((g) => g.w)),
    [edges.gap],
  );
  const maxChem = useMemo(
    () => Math.max(1, ...edges.chem.map((c) => Math.abs(c.w))),
    [edges.chem],
  );

  // Split each family into the strongest edges (→ tubes) and the long tail
  // (→ lines). Strongest = largest |w|; the tail keeps the fine connectome
  // structure cheap. Self-loops are dropped in both layers.
  const blue = useMemo(() => new THREE.Color("#4f86c6"), []);
  const excit = useMemo(() => new THREE.Color("#3a9e63"), []); // depol (sign +1)
  const inhib = useMemo(() => new THREE.Color("#d6486a"), []); // hyperpol (-1)

  // --- GAP: straight tubes for the strong, lines for the tail. ---
  const { gapTubeGeom, gapLineGeom } = useMemo(() => {
    const valid = edges.gap.filter((g) => g.a !== g.b && keepGap(g));
    const sorted = [...valid].sort((a, b) => b.w - a.w);
    const strong = sorted.slice(0, GAP_TUBE_COUNT);
    const tail = sorted.slice(GAP_TUBE_COUNT);
    const tubes = strong.map((g) => {
      const wn = g.w / maxGap;
      return {
        a: nodePos[g.a],
        b: nodePos[g.b],
        color: blue,
        alpha: 0.45 + 0.5 * wn,
        radius: 0.018 + 0.075 * wn,
        bow: 0, // ohmic / symmetric — no direction, keep straight
      };
    });
    const lineSegs = tail.map((g) => ({
      a: nodePos[g.a],
      b: nodePos[g.b],
      color: blue,
      alpha: 0.16 + 0.4 * (g.w / maxGap),
    }));
    return {
      gapTubeGeom: buildTubeGeometry(tubes, 6, 1),
      gapLineGeom: buildEdgeGeometry(lineSegs),
    };
  }, [edges.gap, nodePos, maxGap, blue, keepGap]);

  // --- CHEM: curved-arc tubes for the strong (arc bows to convey direction +
  // depth), lines for the tail. ---
  const { chemTubeGeom, chemLineGeom } = useMemo(() => {
    const valid = edges.chem.filter((c) => c.s !== c.t && keepChem(c));
    const sorted = [...valid].sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const strong = sorted.slice(0, CHEM_TUBE_COUNT);
    const tail = sorted.slice(CHEM_TUBE_COUNT);
    const tubes = strong.map((c) => {
      const wn = Math.abs(c.w) / maxChem;
      return {
        a: nodePos[c.s],
        b: nodePos[c.t],
        color: c.sign >= 0 ? excit : inhib,
        alpha: 0.4 + 0.5 * wn,
        radius: 0.016 + 0.06 * wn,
        bow: 0.16, // arc height ≈16% of chord — depth + pre→post separation
      };
    });
    const lineSegs = tail.map((c) => ({
      a: nodePos[c.s],
      b: nodePos[c.t],
      color: c.sign >= 0 ? excit : inhib,
      alpha: 0.09 + 0.4 * (Math.abs(c.w) / maxChem),
    }));
    return {
      chemTubeGeom: buildTubeGeometry(tubes, 5, 10),
      chemLineGeom: buildEdgeGeometry(lineSegs),
    };
  }, [edges.chem, nodePos, maxChem, excit, inhib, keepChem]);

  useEffect(() => {
    return () => {
      gapTubeGeom.dispose();
      gapLineGeom.dispose();
      chemTubeGeom.dispose();
      chemLineGeom.dispose();
    };
  }, [gapTubeGeom, gapLineGeom, chemTubeGeom, chemLineGeom]);

  return (
    <group>
      {showChem && (
        <>
          <mesh geometry={chemTubeGeom}>
            <meshStandardMaterial
              vertexColors
              transparent
              opacity={0.9}
              roughness={0.5}
              metalness={0.05}
              depthWrite
              toneMapped={false}
            />
          </mesh>
          <lineSegments geometry={chemLineGeom}>
            <lineBasicMaterial
              vertexColors
              transparent
              opacity={0.5}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
              toneMapped={false}
            />
          </lineSegments>
        </>
      )}
      {showGap && (
        <>
          <mesh geometry={gapTubeGeom}>
            <meshStandardMaterial
              vertexColors
              transparent
              opacity={0.92}
              roughness={0.45}
              metalness={0.08}
              depthWrite
              toneMapped={false}
            />
          </mesh>
          <lineSegments geometry={gapLineGeom}>
            <lineBasicMaterial
              vertexColors
              transparent
              opacity={0.6}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
              toneMapped={false}
            />
          </lineSegments>
        </>
      )}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Network ion-flow particles — packets travel pre→post along the strongest
// chem edges; speed driven by the source cell's real total ionic-current
// magnitude (net_flows), with a source-cell depolarisation-rate fallback.
// ---------------------------------------------------------------------------

function NetworkFlow({
  edges,
  positions,
  center,
  traj,
  clock,
  trajIndexByCell,
  active,
}: {
  edges: NetworkEdges;
  positions: NetworkPositions;
  center: THREE.Vector3;
  traj: Trajectory;
  clock: React.MutableRefObject<NetClock>;
  trajIndexByCell: Int32Array; // maps names[] index → trajectory cell index
  active: boolean;
}) {
  // Pick the strongest chem edges (by |w|) to carry flow packets — keeps the
  // particle count bounded while showing the dominant routes.
  const FLOW_EDGES = 360;
  const ref = useRef<THREE.Points>(null);

  const { geom, edgeMeta } = useMemo(() => {
    const byName = new Map(positions.cells.map((c) => [c.cell, c]));
    const nodePos = edges.names.map((nm) => {
      const c = byName.get(nm);
      return c
        ? new THREE.Vector3(
            (c.x - center.x) * SCALE,
            (c.y - center.y) * SCALE,
            (c.z - center.z) * SCALE,
          )
        : new THREE.Vector3();
    });
    const top = [...edges.chem]
      .filter((c) => c.s !== c.t)
      .sort((a, b) => Math.abs(b.w) - Math.abs(a.w))
      .slice(0, FLOW_EDGES);

    const g = new THREE.BufferGeometry();
    const arr = new Float32Array(top.length * 3);
    const colArr = new Float32Array(top.length * 3);
    const excit = new THREE.Color("#7ee0a3");
    const inhib = new THREE.Color("#f08aa6");
    const meta = top.map((c, i) => {
      const a = nodePos[c.s];
      const b = nodePos[c.t];
      arr[i * 3] = a.x;
      arr[i * 3 + 1] = a.y;
      arr[i * 3 + 2] = a.z;
      const col = c.sign >= 0 ? excit : inhib;
      colArr[i * 3] = col.r;
      colArr[i * 3 + 1] = col.g;
      colArr[i * 3 + 2] = col.b;
      return {
        a,
        b,
        srcTraj: trajIndexByCell[c.s],
        phase: Math.random(),
      };
    });
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    return { geom: g, edgeMeta: meta };
  }, [edges, positions, center, trajIndexByCell]);

  useEffect(() => () => geom.dispose(), [geom]);

  // Track previous-frame V per source so we can estimate depolarisation rate.
  const prevV = useRef<Float32Array>(new Float32Array(edgeMeta.length));

  // CURRENT-ACCURATE drive (FIX round 4): when trajectory.json carries per-cell
  // net_flows, the packet speed is driven by the SOURCE CELL's real total ionic-
  // current magnitude (|I_Na|+|I_K|+|I_Ca|+|pump|), normalised by its 95th
  // percentile, rather than the voltage-derived proxy. A presynaptic cell that is
  // actually moving charge (large transmembrane current) pushes its packets
  // faster — physically the right driver for ion flow. Falls back to the V proxy
  // when net_flows is absent.
  const hasFlows = !!traj.net_flows?.I_Na;
  const fluxNorm = useMemo(() => {
    const nf = traj.net_flows;
    const scale = traj.net_flow_scale ?? 20.0;
    if (!nf?.I_Na) return 1;
    const samples: number[] = [];
    for (let f = 0; f < traj.n_frames; f += 8) {
      const na = nf.I_Na[f], k = nf.I_K[f], ca = nf.I_Ca[f], pu = nf.pump[f];
      if (!na) continue;
      for (const m of edgeMeta) {
        const c = m.srcTraj;
        if (c < 0) continue;
        samples.push(
          (Math.abs(na[c] ?? 0) + Math.abs(k[c] ?? 0) +
            Math.abs(ca[c] ?? 0) + Math.abs(pu[c] ?? 0)) / scale,
        );
      }
    }
    samples.sort((a, b) => a - b);
    const p95 = samples.length
      ? samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.95))]
      : 1;
    return Math.max(1e-3, p95);
  }, [traj, edgeMeta]);

  useFrame((_s, delta) => {
    const p = ref.current;
    if (!p || !active) return;
    const fi = clock.current.frameIndex % traj.n_frames;
    const Vrow = traj.V_mV[fi];
    const nf = traj.net_flows;
    const fScale = traj.net_flow_scale ?? 20.0;
    const naRow = hasFlows ? nf!.I_Na[fi] : undefined;
    const kRow = hasFlows ? nf!.I_K[fi] : undefined;
    const caRow = hasFlows ? nf!.I_Ca[fi] : undefined;
    const puRow = hasFlows ? nf!.pump[fi] : undefined;
    const pos = geom.getAttribute("position") as THREE.BufferAttribute;
    const t = clock.current.t;
    for (let i = 0; i < edgeMeta.length; i++) {
      const m = edgeMeta[i];
      let speed: number;
      if (hasFlows && naRow && m.srcTraj >= 0) {
        const c = m.srcTraj;
        const mag =
          (Math.abs(naRow[c] ?? 0) +
            Math.abs(kRow![c] ?? 0) +
            Math.abs(caRow![c] ?? 0) +
            Math.abs(puRow![c] ?? 0)) /
          fScale; // uA/cm^2
        const drive = Math.min(1.4, mag / fluxNorm);
        speed = 0.12 + drive * 1.0;
      } else {
        // Voltage-derived fallback (older trajectory.json without net_flows).
        const v = m.srcTraj >= 0 ? Vrow?.[m.srcTraj] : -70;
        const vv = Number.isFinite(v) ? (v as number) : -70;
        const dRate = Math.abs(vv - prevV.current[i]);
        prevV.current[i] = vv;
        const depol = Math.min(1, Math.max(0, (vv + 75) / 45));
        speed = 0.12 + depol * 0.9 + Math.min(0.6, dRate * 0.4);
      }
      let u = (m.phase + t * speed * 0.25) % 1;
      if (u < 0) u += 1;
      pos.setXYZ(
        i,
        m.a.x + (m.b.x - m.a.x) * u,
        m.a.y + (m.b.y - m.a.y) * u,
        m.a.z + (m.b.z - m.a.z) * u,
      );
    }
    pos.needsUpdate = true;
  });

  if (!active) return null;
  return (
    <points ref={ref} geometry={geom}>
      <pointsMaterial
        size={0.5}
        vertexColors
        transparent
        opacity={0.95}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        toneMapped={false}
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// Hover tooltip — DOM overlay anchored to the hovered soma's world position.
// ---------------------------------------------------------------------------

function HoverLabel({
  cell,
  center,
  familyByCell,
  channelRowByCell,
}: {
  cell: NetworkCell;
  center: THREE.Vector3;
  familyByCell: Map<string, string>;
  channelRowByCell: Map<string, CellChannelRow>;
}) {
  const fam = familyByCell.get(cell.cell) ?? "none";
  const row = channelRowByCell.get(cell.cell);
  const pos = new THREE.Vector3(
    (cell.x - center.x) * SCALE,
    (cell.y - center.y) * SCALE,
    (cell.z - center.z) * SCALE,
  );
  return (
    <Html center distanceFactor={36} position={[pos.x, pos.y + 0.9, pos.z]} zIndexRange={[50, 0]}>
      <div className="pointer-events-none w-48 rounded-lg border border-white/50 bg-white/92 p-2 text-left shadow-lg backdrop-blur-md">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[0.74rem] font-semibold text-emerald-950">
            {cell.cell}
          </span>
          <span className="font-mono text-[0.55rem] text-emerald-900/50">
            {cell.class}
          </span>
        </div>
        {row && (
          <>
            <p className="mt-0.5 text-[0.62rem] text-emerald-900/70">
              dominant: <span style={{ color: FAMILY_COLOR[fam] }}>{fam}</span> ·{" "}
              {row.n_channels_expressed} channels
            </p>
            {row.cm_pF != null && (
              <p className="font-mono text-[0.55rem] text-emerald-900/55">
                Cm {row.cm_pF.toFixed(2)} pF · Eleak {row.e_leak_mV?.toFixed(1)} mV
              </p>
            )}
          </>
        )}
        <p className="mt-0.5 font-mono text-[0.55rem] text-emerald-900/45">
          µm ({cell.x.toFixed(0)}, {cell.y.toFixed(0)}, {cell.z.toFixed(0)})
        </p>
      </div>
    </Html>
  );
}

// ---------------------------------------------------------------------------
// Network3D — assembles the scene.
// ---------------------------------------------------------------------------

export default function Network3D({
  positions,
  edges,
  traj,
  channelTable,
  clock,
  colorMode,
  showGap,
  showChem,
  showFlow,
  showEdges,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  positions: NetworkPositions;
  edges: NetworkEdges;
  traj: Trajectory;
  channelTable: CellChannelRow[];
  clock: React.MutableRefObject<NetClock>;
  colorMode: NetworkColorMode;
  showGap: boolean;
  showChem: boolean;
  showFlow: boolean;
  showEdges: boolean;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const { camera } = useThree();

  const center = useMemo(() => {
    const c = new THREE.Vector3();
    positions.cells.forEach((p) => c.add(new THREE.Vector3(p.x, p.y, p.z)));
    c.multiplyScalar(1 / Math.max(1, positions.cells.length));
    return c;
  }, [positions]);

  // Frame the long A-P axis nicely on mount.
  useEffect(() => {
    camera.position.set(0, 0, 34);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  const familyByCell = useMemo(() => {
    const m = new Map<string, string>();
    for (const r of channelTable) m.set(r.cell, r.dominant_family);
    return m;
  }, [channelTable]);

  const channelRowByCell = useMemo(() => {
    const m = new Map<string, CellChannelRow>();
    for (const r of channelTable) m.set(r.cell, r);
    return m;
  }, [channelTable]);

  // Map network_edges.names[] index → trajectory.cells[] index (usually same
  // order, but resolve by name to be safe).
  const trajIndexByCell = useMemo(() => {
    const trajIdx = new Map(traj.cells.map((c, i) => [c, i]));
    const arr = new Int32Array(edges.names.length);
    edges.names.forEach((nm, i) => {
      arr[i] = trajIdx.has(nm) ? (trajIdx.get(nm) as number) : -1;
    });
    return arr;
  }, [edges.names, traj.cells]);

  // Soma instance i (positions.cells order) → trajectory index.
  const somaTrajIndex = useMemo(() => {
    const trajIdx = new Map(traj.cells.map((c, i) => [c, i]));
    const arr = new Int32Array(positions.cells.length);
    positions.cells.forEach((c, i) => {
      arr[i] = trajIdx.has(c.cell) ? (trajIdx.get(c.cell) as number) : -1;
    });
    return arr;
  }, [positions.cells, traj.cells]);

  const hoveredCell = useMemo(
    () => positions.cells.find((c) => c.cell === hovered) ?? null,
    [positions.cells, hovered],
  );

  // Resolve `selected` into the set of soma instance indices to cross-highlight
  // (FIX round 3, 2026-06-19 — the legend↔network cross-highlight gap).
  //   · `selected` is a CELL NAME (e.g. "AVAL")  → that one soma.
  //   · `selected` is a CHANNEL/RECEPTOR record id ("chan_<module>", emitted by
  //     emit_substrate_inventory.py) → every soma whose channels_expressed
  //     (gbar>0, from emit_viz_assets.py) contains that module. This is exact,
  //     not a family approximation. Pump / transporter / ion-compartment /
  //     geometry / metabolism records carry no per-soma expression in this data,
  //     so they stay hero-scoped and map to the empty set here (network
  //     selection is cell- + channel/receptor-scoped).
  const selectedIndices = useMemo(() => {
    if (!selected) return [];
    // Direct cell-name match first.
    const direct = positions.cells.findIndex((c) => c.cell === selected);
    if (direct >= 0) return [direct];
    // Channel/receptor record id → expressing cells.
    if (selected.startsWith("chan_")) {
      const mod = selected.slice("chan_".length);
      const idxByCell = new Map(positions.cells.map((c, i) => [c.cell, i]));
      const out: number[] = [];
      for (const r of channelTable) {
        if (r.channels_expressed?.includes(mod)) {
          const i = idxByCell.get(r.cell);
          if (i != null) out.push(i);
        }
      }
      return out;
    }
    return [];
  }, [selected, positions.cells, channelTable]);

  // Edge-node adjacency (edges.names order) for the 1-hop reveal-on-select.
  const adjacency = useMemo(() => {
    const adj: Set<number>[] = edges.names.map(() => new Set<number>());
    for (const g of edges.gap) {
      if (g.a === g.b) continue;
      adj[g.a]?.add(g.b);
      adj[g.b]?.add(g.a);
    }
    for (const c of edges.chem) {
      if (c.s === c.t) continue;
      adj[c.s]?.add(c.t);
      adj[c.t]?.add(c.s);
    }
    return adj;
  }, [edges]);

  // Map a cell name → its edge-node index (edges.names order).
  const edgeIdxByName = useMemo(() => {
    const m = new Map<string, number>();
    edges.names.forEach((nm, i) => m.set(nm, i));
    return m;
  }, [edges.names]);

  // Revealed nodes = (selected cells ∪ hovered cell) and their 1-hop neighbours,
  // in edges.names order. When `showEdges` is true this is unused (all edges
  // render); when false it is the declutter focus set.
  const revealedNodes = useMemo(() => {
    const out = new Set<number>();
    if (showEdges) return out;
    const seeds: number[] = [];
    // selected cells (selectedIndices are positions.cells indices → resolve name)
    for (const pi of selectedIndices) {
      const nm = positions.cells[pi]?.cell;
      const ei = nm != null ? edgeIdxByName.get(nm) : undefined;
      if (ei != null) seeds.push(ei);
    }
    if (hovered) {
      const ei = edgeIdxByName.get(hovered);
      if (ei != null) seeds.push(ei);
    }
    for (const s of seeds) {
      out.add(s);
      const nbrs = adjacency[s];
      if (nbrs) for (const n of nbrs) out.add(n);
    }
    return out;
  }, [showEdges, selectedIndices, hovered, positions.cells, edgeIdxByName, adjacency]);

  return (
    <group>
      <EdgeLayers
        edges={edges}
        positions={positions}
        center={center}
        showGap={showGap}
        showChem={showChem}
        showEdges={showEdges}
        revealedNodes={revealedNodes}
      />
      <SomaInstances
        positions={positions}
        traj={traj}
        clock={clock}
        colorMode={colorMode}
        familyByCell={familyByCell}
        hovered={hovered}
        selectedIndices={selectedIndices}
        setHovered={setHovered}
        onSelect={onSelect}
        trajIndexByCell={somaTrajIndex}
      />
      <NetworkFlow
        edges={edges}
        positions={positions}
        center={center}
        traj={traj}
        clock={clock}
        trajIndexByCell={trajIndexByCell}
        active={showFlow}
      />
      {hoveredCell && (
        <HoverLabel
          cell={hoveredCell}
          center={center}
          familyByCell={familyByCell}
          channelRowByCell={channelRowByCell}
        />
      )}
      {/* network caption */}
      <Html center distanceFactor={60} position={[0, -16, 0]}>
        <div className="pointer-events-none whitespace-nowrap rounded-md bg-white/85 px-2 py-1 text-[0.62rem] font-medium text-emerald-900 shadow">
          NETWORK · {positions.n_cells} cells (real EM µm) ·{" "}
          {edges.n_chem_edges} chem + {edges.n_gap_edges} gap edges
        </div>
      </Html>
    </group>
  );
}
