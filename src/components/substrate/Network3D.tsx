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
 *     trajectory (trajectory.json — V_mV[frame][cell]); Ca²⁺ drives an
 *     emissive boost. A second "channel family" colour mode tints each soma by
 *     its dominant gbar family (hero_channel_gbar.cell_channel_table).
 *   - Connectome edges from network_edges.json (artifacts/connectome.npz):
 *       · gap junctions  — ohmic, innexin-typed, symmetric (blue)
 *       · chem synapses  — signed (excitatory green / inhibitory rose), width
 *                          by |w|, neurotransmitter-typed
 *     Both layers are independently toggleable.
 *   - Network-wide ion-flow particle layer: packets travel presynaptic →
 *     postsynaptic along the most active chem edges, speed/brightness driven by
 *     the live source-cell depolarisation rate from the real trajectory.
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
};

export type NetworkPositions = {
  n_cells: number;
  n_with_position: number;
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
};

export type CellChannelRow = {
  cell: string;
  class: string;
  n_channels_expressed: number;
  gbar_by_family: Record<string, number>;
  dominant_family: string;
  cm_pF?: number;
  e_leak_mV?: number;
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
  selected,
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
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
  trajIndexByCell: Int32Array;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const n = positions.cells.length;
  const tmpColor = useRef(new THREE.Color());
  const tmpMat = useRef(new THREE.Matrix4());
  const dummy = useRef(new THREE.Object3D());

  // Static base layout: place each instance at its scaled µm centroid.
  const center = useMemo(() => {
    const c = new THREE.Vector3();
    positions.cells.forEach((p) => c.add(new THREE.Vector3(p.x, p.y, p.z)));
    c.multiplyScalar(1 / Math.max(1, n));
    return c;
  }, [positions, n]);

  // Per-instance base radius — fixed; soma differences are not in this data.
  const baseR = 0.42;

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
      d.scale.setScalar(1);
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
  }, [positions, center, n]);

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
    const selIdx =
      selected != null ? positions.cells.findIndex((c) => c.cell === selected) : -1;

    for (let i = 0; i < n; i++) {
      if (colorMode === "voltage") {
        const ti = trajIndexByCell[i];
        const v = ti >= 0 ? Vrow?.[ti] : undefined;
        if (Number.isFinite(v)) {
          voltageColor(v as number, tmpColor.current);
          // Ca²⁺ brightens (mild) so active cells pop.
          const ca = ti >= 0 ? Carow?.[ti] : 0;
          const caN = Math.min(1, Math.max(0, ((ca ?? 0) - 0.2) / 1.5));
          tmpColor.current.offsetHSL(0, 0, 0.12 * caN);
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
    }
    mesh.instanceColor.needsUpdate = true;

    // Hover / select pop — rescale just the highlighted instances.
    for (const idx of [hiIdx, selIdx]) {
      if (idx < 0) continue;
      const p = positions.cells[idx];
      const d = dummy.current;
      d.position.set(
        (p.x - center.x) * SCALE,
        (p.y - center.y) * SCALE,
        (p.z - center.z) * SCALE,
      );
      d.scale.setScalar(idx === selIdx ? 2.1 : 1.7);
      d.updateMatrix();
      mesh.setMatrixAt(idx, d.matrix);
    }
    // Reset any previously-scaled instance that is no longer highlighted.
    const prev = (mesh as unknown as { _prevHi?: number[] })._prevHi ?? [];
    for (const idx of prev) {
      if (idx === hiIdx || idx === selIdx || idx < 0) continue;
      const p = positions.cells[idx];
      const d = dummy.current;
      d.position.set(
        (p.x - center.x) * SCALE,
        (p.y - center.y) * SCALE,
        (p.z - center.z) * SCALE,
      );
      d.scale.setScalar(1);
      d.updateMatrix();
      mesh.setMatrixAt(idx, d.matrix);
    }
    (mesh as unknown as { _prevHi?: number[] })._prevHi = [hiIdx, selIdx];
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
// Edge layers — gap (ohmic, symmetric) and chem (signed). Built once as
// LineSegments with per-vertex colour; toggled by `visible`.
// ---------------------------------------------------------------------------

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
}: {
  edges: NetworkEdges;
  positions: NetworkPositions;
  center: THREE.Vector3;
  showGap: boolean;
  showChem: boolean;
}) {
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

  const gapGeom = useMemo(() => {
    const blue = new THREE.Color("#4f86c6");
    const segs = edges.gap
      .filter((g) => g.a !== g.b)
      .map((g) => ({
        a: nodePos[g.a],
        b: nodePos[g.b],
        color: blue,
        alpha: 0.18 + 0.55 * (g.w / maxGap),
      }));
    return buildEdgeGeometry(segs);
  }, [edges.gap, nodePos, maxGap]);

  const chemGeom = useMemo(() => {
    const excit = new THREE.Color("#3a9e63"); // depolarising (sign +1)
    const inhib = new THREE.Color("#d6486a"); // hyperpolarising (sign -1)
    const segs = edges.chem
      .filter((c) => c.s !== c.t)
      .map((c) => ({
        a: nodePos[c.s],
        b: nodePos[c.t],
        color: c.sign >= 0 ? excit : inhib,
        alpha: 0.1 + 0.5 * (Math.abs(c.w) / maxChem),
      }));
    return buildEdgeGeometry(segs);
  }, [edges.chem, nodePos, maxChem]);

  useEffect(() => {
    return () => {
      gapGeom.dispose();
      chemGeom.dispose();
    };
  }, [gapGeom, chemGeom]);

  return (
    <group>
      {showChem && (
        <lineSegments geometry={chemGeom}>
          <lineBasicMaterial
            vertexColors
            transparent
            opacity={0.55}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </lineSegments>
      )}
      {showGap && (
        <lineSegments geometry={gapGeom}>
          <lineBasicMaterial
            vertexColors
            transparent
            opacity={0.7}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </lineSegments>
      )}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Network ion-flow particles — packets travel pre→post along the most active
// chem edges; speed/brightness driven by the live source-cell depolarisation.
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

  useFrame((_s, delta) => {
    const p = ref.current;
    if (!p || !active) return;
    const fi = clock.current.frameIndex % traj.n_frames;
    const Vrow = traj.V_mV[fi];
    const pos = geom.getAttribute("position") as THREE.BufferAttribute;
    const t = clock.current.t;
    for (let i = 0; i < edgeMeta.length; i++) {
      const m = edgeMeta[i];
      const v = m.srcTraj >= 0 ? Vrow?.[m.srcTraj] : -70;
      const vv = Number.isFinite(v) ? (v as number) : -70;
      // depolarisation drive: how far above rest, plus instantaneous rate.
      const dRate = Math.abs(vv - prevV.current[i]);
      prevV.current[i] = vv;
      const depol = Math.min(1, Math.max(0, (vv + 75) / 45));
      const speed = 0.12 + depol * 0.9 + Math.min(0.6, dRate * 0.4);
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

  return (
    <group>
      <EdgeLayers
        edges={edges}
        positions={positions}
        center={center}
        showGap={showGap}
        showChem={showChem}
      />
      <SomaInstances
        positions={positions}
        traj={traj}
        clock={clock}
        colorMode={colorMode}
        familyByCell={familyByCell}
        hovered={hovered}
        selected={selected}
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
