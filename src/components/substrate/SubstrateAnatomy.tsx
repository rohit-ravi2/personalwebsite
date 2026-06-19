import * as React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import HeroCell3D, { type FrameState } from "./HeroCell3D";
import { type GlyphSignatures } from "./GlyphGeometry";
import Network3D, {
  type NetClock,
  type NetworkColorMode,
  type CellChannelRow,
} from "./Network3D";

/**
 * SubstrateAnatomy — full-bleed Three.js visualization of the C. elegans
 * Tier4 whole-cell electrical substrate, rendered 1:1 from code.
 *
 * Two views:
 *   (A) HERO CELL  — AVA at full molecular detail (per-segment morphology +
 *                    channel/receptor/pump complement with ON/OFF/orphaned status)
 *   (B) NETWORK    — 300 cells at real µm soma positions + connectome edges,
 *                    coloured by a REAL substrate voltage trajectory.
 *
 * All data fetched from /data/*.json, emitted read-only from the substrate
 * repo (tests/emit_*.py). Every rendered structure carries a status:
 *   ON         — integrated + active on the default assemble path
 *   default-OFF — integrated but a build flag is needed to flip it on
 *   ORPHANED   — module exists / registers but is NOT on the assemble path
 *   MISSING    — known-missing (NOT integrated); diagnosed by audit
 *
 * This is the SCAFFOLD: data plumbing, state, view toggle, completeness
 * legend, and a 3D <Canvas> shell with placeholder geometry. Detailed
 * hero/network/flow geometry lands in subsequent passes.
 */

// ----------------------------------------------------------------------------
// Types — mirror the emitted JSON schemas exactly (see tests/emit_*.py).
// ----------------------------------------------------------------------------

type Status = "on" | "off" | "orphaned" | "missing";

type InventoryRecord = {
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

type SubstrateInventory = {
  schema_version: number;
  generated: string;
  generator: string;
  repo: string;
  summary: {
    n_channels_in_registry: number;
    n_channel_records: number;
    n_receptor_records: number;
    channels_plus_receptors: number;
    category_counts: Record<string, number>;
    status_counts: Record<string, number>;
    note: string;
  };
  legend: Record<Status, string>;
  records: InventoryRecord[];
  missing: InventoryRecord[];
};

type HeroSegment = {
  id: number;
  parent: number | null;
  tag: string; // soma | axon | dendrite
  prox: [number, number, number];
  dist: [number, number, number];
  prox_r: number;
  dist_r: number;
  length_um: number;
  surf_um2: number;
};

type HeroMorphology = {
  provenance: Record<string, unknown>;
  cell: string;
  cengen_class: string;
  n_segments: number;
  total_surf_um2: number;
  region_surf_um2: { soma: number; axon: number; dendrite: number };
  soma_centroid_um: [number, number, number];
  segments: HeroSegment[];
};

type HeroChannel = {
  channel: string;
  gbar_Scm2: number;
  ion: string;
  family: string;
  expressed: boolean;
};

type HeroChannelGbar = {
  provenance: Record<string, unknown>;
  hero_cell: string;
  hero_class: string;
  n_channels_registry: number;
  hero_cm_pF: number;
  hero_surf_cm2: number;
  hero_channels: HeroChannel[];
  n_cells: number;
  cell_channel_table: CellChannelRow[];
};

type NetworkCell = {
  cell: string;
  x: number;
  y: number;
  z: number;
  class: string;
  has_morphology: boolean;
};

type NetworkPositions = {
  provenance: Record<string, unknown>;
  n_cells: number;
  n_with_position: number;
  cells: NetworkCell[];
};

type ChemEdge = { s: number; t: number; w: number; raw: number; sign: number; nt: string };
type GapEdge = { a: number; b: number; w: number };

type NetworkEdges = {
  provenance: Record<string, unknown>;
  n_nodes: number;
  names: string[];
  n_chem_edges: number;
  n_gap_edges: number;
  chem: ChemEdge[];
  gap: GapEdge[];
};

type Trajectory = {
  real: boolean;
  illustrative: boolean;
  regime: string;
  dt_ms: number;
  n_frames: number;
  n_cells: number;
  hero: string;
  cells: string[];
  finite_cells: number;
  V_mV: number[][]; // [frame][cell]
  Ca_uM: number[][]; // [frame][cell]
  hero_flows: { I_Na: number[]; I_K: number[]; I_Ca: number[]; pump: number[] };
  units: Record<string, string>;
  provenance: Record<string, unknown>;
};

type DataBundle = {
  inventory: SubstrateInventory;
  heroMorph: HeroMorphology;
  heroGbar: HeroChannelGbar;
  positions: NetworkPositions;
  edges: NetworkEdges;
  trajectory: Trajectory;
  glyphSignatures: GlyphSignatures;
};

// ----------------------------------------------------------------------------
// Status display config — house palette (forest / cream / amber) + signal reds.
// ----------------------------------------------------------------------------

const STATUS_META: Record<
  Status,
  { label: string; chip: string; dot: string; ring: string }
> = {
  on: {
    label: "ON",
    chip: "bg-emerald-500/12 text-emerald-800 border-emerald-600/40",
    dot: "bg-emerald-500",
    ring: "ring-emerald-500/50",
  },
  off: {
    label: "default-OFF",
    chip: "bg-amber-500/12 text-amber-800 border-amber-600/40",
    dot: "bg-amber-500",
    ring: "ring-amber-500/50",
  },
  orphaned: {
    label: "ORPHANED",
    chip: "bg-slate-500/12 text-slate-700 border-slate-500/40",
    dot: "bg-slate-400",
    ring: "ring-slate-400/50",
  },
  missing: {
    label: "NOT INTEGRATED",
    chip: "bg-rose-500/12 text-rose-800 border-rose-600/45",
    dot: "bg-rose-500",
    ring: "ring-rose-500/50",
  },
};

const STATUS_ORDER: Status[] = ["on", "off", "orphaned", "missing"];

const DATA_FILES: Record<keyof DataBundle, string> = {
  inventory: "/data/substrate_inventory.json",
  heroMorph: "/data/hero_morphology.json",
  heroGbar: "/data/hero_channel_gbar.json",
  positions: "/data/network_positions.json",
  edges: "/data/network_edges.json",
  trajectory: "/data/trajectory.json",
  glyphSignatures: "/data/glyph_signatures.json",
};

// ----------------------------------------------------------------------------
// Data hook
// ----------------------------------------------------------------------------

type LoadState =
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "ready"; data: DataBundle };

function useSubstrateData(): LoadState {
  const [state, setState] = useState<LoadState>({ phase: "loading" });

  useEffect(() => {
    let cancelled = false;
    const entries = Object.entries(DATA_FILES) as Array<[keyof DataBundle, string]>;

    Promise.all(
      entries.map(async ([key, url]) => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
        const json = await res.json();
        return [key, json] as const;
      }),
    )
      .then((pairs) => {
        if (cancelled) return;
        const data = Object.fromEntries(pairs) as unknown as DataBundle;
        setState({ phase: "ready", data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setState({
          phase: "error",
          message: err instanceof Error ? err.message : String(err),
        });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}

// ----------------------------------------------------------------------------
// Small UI atoms (house style: frosted-glass cards, forest/cream/amber)
// ----------------------------------------------------------------------------

function StatusChip({ status, count }: { status: Status; count?: number }) {
  const m = STATUS_META[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[0.7rem] font-medium ${m.chip}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${m.dot}`} />
      {m.label}
      {count != null && <span className="font-mono opacity-70">{count}</span>}
    </span>
  );
}

function GlassCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-2xl border border-white/40 bg-white/55 shadow-sm backdrop-blur-md ${className}`}
    >
      {children}
    </div>
  );
}

// ----------------------------------------------------------------------------
// 3D scaffold — placeholder scenes (real hero/network geometry lands next pass)
// ----------------------------------------------------------------------------

/**
 * TrajectoryClock — advances the shared `frame` ref from the REAL hero
 * trajectory each rAF tick. Lives inside <Canvas>. Writes V/Ca/currents
 * imperatively so per-frame recolour does not re-render React.
 */
function TrajectoryClock({
  traj,
  frame,
  netClock,
  playing,
  speed,
}: {
  traj: Trajectory;
  frame: React.MutableRefObject<FrameState>;
  netClock: React.MutableRefObject<NetClock>;
  playing: boolean;
  speed: number;
}) {
  const heroIdx = useMemo(
    () => Math.max(0, traj.cells.indexOf(traj.hero)),
    [traj],
  );
  const tRef = useRef(0); // seconds of animation clock
  const frameFloat = useRef(0); // fractional trajectory frame index

  useFrame((_state, delta) => {
    tRef.current += delta;
    if (playing) {
      // 1 trajectory frame per (dt_ms) ms model time; play at `speed`× a
      // compressed wall-clock (so 1333×6ms = 8s model time is watchable).
      frameFloat.current += delta * 30 * speed;
      if (frameFloat.current >= traj.n_frames) frameFloat.current = 0;
    }
    const fi = Math.floor(frameFloat.current) % traj.n_frames;
    const f = frame.current;
    const v = traj.V_mV[fi]?.[heroIdx];
    const ca = traj.Ca_uM[fi]?.[heroIdx];
    f.V_mV = Number.isFinite(v) ? v : -70;
    f.Ca_uM = Number.isFinite(ca) ? ca : 0.2;
    f.I_Na = traj.hero_flows.I_Na[fi] ?? 0;
    f.I_K = traj.hero_flows.I_K[fi] ?? 0;
    f.I_Ca = traj.hero_flows.I_Ca[fi] ?? 0;
    f.pump = traj.hero_flows.pump[fi] ?? 0;
    f.t = tRef.current;
    // Drive the shared network clock from the same frame index.
    const nc = netClock.current;
    nc.frameIndex = fi;
    nc.frameFloat = frameFloat.current;
    nc.t = tRef.current;
  });

  return null;
}

function Scene({
  view,
  data,
  frame,
  netClock,
  playing,
  speed,
  hovered,
  selected,
  setHovered,
  onSelect,
  colorMode,
  showGap,
  showChem,
  showFlow,
}: {
  view: View;
  data: DataBundle;
  frame: React.MutableRefObject<FrameState>;
  netClock: React.MutableRefObject<NetClock>;
  playing: boolean;
  speed: number;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
  colorMode: NetworkColorMode;
  showGap: boolean;
  showChem: boolean;
  showFlow: boolean;
}) {
  // Hero glyph set = intrinsic + ligand-gated + pumps/transporters/etc. that
  // live on a single cell, PLUS the MISSING ghosts. (Network-only / pure ion
  // bookkeeping records are surfaced in the legend, not as hero glyphs.)
  const heroRecords = useMemo(
    () => [...data.inventory.records, ...data.inventory.missing],
    [data.inventory],
  );

  return (
    <Canvas
      camera={{ position: [0, 0, view === "hero" ? 14 : 28], fov: 45, near: 0.1, far: 2000 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
    >
      <color attach="background" args={["#f5efe1"]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[10, 20, 15]} intensity={0.9} />
      <directionalLight position={[-15, -10, -10]} intensity={0.3} />
      <hemisphereLight args={["#ffffff", "#cdbfa0", 0.4]} />
      <TrajectoryClock
        traj={data.trajectory}
        frame={frame}
        netClock={netClock}
        playing={playing}
        speed={speed}
      />
      {view === "hero" ? (
        <HeroCell3D
          morph={data.heroMorph}
          gbar={data.heroGbar}
          signatures={data.glyphSignatures}
          records={heroRecords}
          frame={frame}
          hovered={hovered}
          selected={selected}
          setHovered={setHovered}
          onSelect={onSelect}
        />
      ) : (
        <Network3D
          positions={data.positions}
          edges={data.edges}
          traj={data.trajectory}
          channelTable={data.heroGbar.cell_channel_table}
          clock={netClock}
          colorMode={colorMode}
          showGap={showGap}
          showChem={showChem}
          showFlow={showFlow}
          hovered={hovered}
          selected={selected}
          setHovered={setHovered}
          onSelect={onSelect}
        />
      )}
      <OrbitControls enableDamping dampingFactor={0.08} makeDefault />
    </Canvas>
  );
}

// ----------------------------------------------------------------------------
// Completeness legend — the audit, rendered. Status chips + grouped records.
// ----------------------------------------------------------------------------

// Human-readable category order + display labels (audit-narrative order).
const CATEGORY_ORDER: string[] = [
  "channel",
  "receptor",
  "pump",
  "transporter",
  "gap_junction",
  "release",
  "ion_compartment",
  "neuromod_peptide",
  "geometry",
  "metabolism",
];

const CATEGORY_LABEL: Record<string, string> = {
  channel: "Ion channels",
  receptor: "Ligand-gated receptors",
  pump: "Pumps (active transport)",
  transporter: "Cotransporters / exchangers",
  gap_junction: "Gap junctions",
  release: "Synaptic release",
  ion_compartment: "Ion compartments & buffering",
  neuromod_peptide: "Neuromodulation / peptides",
  geometry: "Geometry",
  metabolism: "Metabolism",
};

/**
 * ProvenanceRow — a single inventory record. Hover highlights, CLICK selects
 * (cross-highlighting the matching 3D glyph / edges via the shared `selected`
 * id) and pins an expanded provenance panel (file:line, gbar, full provenance).
 */
function ProvenanceRow({
  r,
  hovered,
  selected,
  setHovered,
  onSelect,
}: {
  r: InventoryRecord;
  hovered: string | null;
  selected: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const m = STATUS_META[r.status];
  const isSel = selected === r.id;
  const isHi = hovered === r.id || isSel;
  return (
    <li
      onMouseEnter={() => setHovered(r.id)}
      onMouseLeave={() => setHovered(null)}
      onClick={() => onSelect(r.id)}
      className={`group cursor-pointer rounded-lg border px-2 py-1 transition-colors ${
        isSel
          ? `border-transparent ring-1 ${m.ring} bg-white/80`
          : isHi
            ? `border-transparent ring-1 ${m.ring} bg-white/60`
            : "border-transparent hover:bg-white/60"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[0.78rem] font-medium text-emerald-950">
          <span className={`h-1.5 w-1.5 rounded-full ${m.dot}`} />
          {r.name}
        </span>
        <span className="shrink-0 font-mono text-[0.6rem] text-emerald-900/45">
          {m.label}
        </span>
      </div>
      <p className="mt-0.5 pl-3 text-[0.68rem] leading-snug text-emerald-900/60">
        {r.subtype}
      </p>
      {/* Provenance: peek on hover (file:line), pin full panel on select. */}
      {isSel ? (
        <div className="mt-1.5 ml-3 space-y-1 rounded-md border border-white/60 bg-white/70 px-2 py-1.5">
          <p className="text-[0.66rem] leading-snug text-emerald-900/75">
            {r.physical_desc}
          </p>
          <p className="font-mono text-[0.58rem] leading-snug text-emerald-900/55">
            {r.file}:{r.line}
          </p>
          {r.gbar_source && (
            <p className="font-mono text-[0.58rem] leading-snug text-emerald-900/45">
              gbar = {r.gbar_source}
              {r.gbar_value_AVA != null && (
                <>
                  {" "}
                  · AVA{" "}
                  <span className="text-emerald-800/70">
                    {r.gbar_value_AVA.toExponential(2)} S/cm²
                  </span>
                </>
              )}
            </p>
          )}
          <p className="font-mono text-[0.55rem] leading-snug text-emerald-900/40">
            {r.provenance}
          </p>
        </div>
      ) : (
        isHi && (
          <p className="mt-1 pl-3 font-mono text-[0.6rem] leading-snug text-emerald-900/45">
            {r.file.split("/").slice(-2).join("/")}:{r.line}
          </p>
        )
      )}
    </li>
  );
}

function CompletenessLegend({
  data,
  filter,
  setFilter,
  hovered,
  setHovered,
  selected,
  onSelect,
}: {
  data: DataBundle;
  filter: Status | null;
  setFilter: (s: Status | null) => void;
  hovered: string | null;
  setHovered: (id: string | null) => void;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const inv = data.inventory;

  // Present structures (everything that is actually in the model), grouped and
  // ordered by the audit narrative. The MISSING records get their own section.
  const presentByCategory = useMemo(() => {
    const groups: Record<string, InventoryRecord[]> = {};
    for (const r of inv.records) {
      if (filter && r.status !== filter) continue;
      (groups[r.category] ??= []).push(r);
    }
    // status order within a category: on → off → orphaned
    for (const recs of Object.values(groups)) {
      recs.sort(
        (a, b) =>
          STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status),
      );
    }
    const ordered: [string, InventoryRecord[]][] = [];
    for (const cat of CATEGORY_ORDER) {
      if (groups[cat]?.length) ordered.push([cat, groups[cat]]);
    }
    // any category not in CATEGORY_ORDER (defensive)
    for (const [cat, recs] of Object.entries(groups)) {
      if (!CATEGORY_ORDER.includes(cat) && recs.length)
        ordered.push([cat, recs]);
    }
    return ordered;
  }, [inv.records, filter]);

  const showMissing = !filter || filter === "missing";
  const presentCount = inv.records.length;

  return (
    <GlassCard className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-white/40 p-4">
        <h3 className="text-sm font-semibold text-emerald-900">
          Completeness audit — {presentCount + inv.missing.length} structures
        </h3>
        <p className="mt-0.5 text-[0.7rem] text-emerald-900/60">
          {inv.summary.n_channels_in_registry}-channel registry ·{" "}
          {inv.summary.n_channel_records} channels +{" "}
          {inv.summary.n_receptor_records} receptors · emitted {inv.generated}{" "}
          from <span className="font-mono">{inv.repo}</span>
        </p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {STATUS_ORDER.map((s) => (
            <button
              key={s}
              onClick={() => setFilter(filter === s ? null : s)}
              className={`transition-opacity ${
                filter && filter !== s ? "opacity-40" : "opacity-100"
              }`}
              title={inv.legend[s]}
            >
              <StatusChip status={s} count={inv.summary.status_counts[s]} />
            </button>
          ))}
        </div>
        {filter ? (
          <p className="mt-2 text-[0.7rem] italic text-emerald-900/70">
            {inv.legend[filter]}
          </p>
        ) : (
          <p className="mt-2 text-[0.66rem] leading-snug text-emerald-900/55">
            Click any row to cross-highlight its structure in the 3D scene and
            pin its <span className="font-mono">file:line</span> provenance.
          </p>
        )}
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {/* PRESENT structures, grouped by category */}
        {presentByCategory.map(([cat, recs]) => (
          <div key={cat}>
            <h4 className="mb-1 flex items-baseline justify-between text-[0.7rem] font-semibold uppercase tracking-wide text-emerald-900/55">
              <span>{CATEGORY_LABEL[cat] ?? cat.replace(/_/g, " ")}</span>
              <span className="font-mono text-[0.6rem] text-emerald-900/40">
                {recs.length}
              </span>
            </h4>
            <ul className="space-y-1">
              {recs.map((r) => (
                <ProvenanceRow
                  key={r.id}
                  r={r}
                  hovered={hovered}
                  selected={selected}
                  setHovered={setHovered}
                  onSelect={onSelect}
                />
              ))}
            </ul>
          </div>
        ))}

        {/* MISSING structures — the foundation gaps, prominent + cited. */}
        {showMissing && inv.missing.length > 0 && (
          <div className="rounded-xl border border-rose-600/30 bg-rose-500/[0.06] p-3">
            <div className="mb-1 flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-rose-500" />
              <h4 className="text-[0.74rem] font-bold uppercase tracking-wide text-rose-800">
                Not integrated — foundation gaps · {inv.missing.length}
              </h4>
            </div>
            <p className="mb-2 text-[0.66rem] leading-snug text-rose-900/70">
              Known-missing structure the substrate work is converging on, shown
              in red rather than hidden. Cited to{" "}
              <span className="font-mono">
                docs/SUBSTRATE_LIMITER_AUDIT_2026-06-19.md
              </span>{" "}
              and{" "}
              <span className="font-mono">
                docs/K_PARTITION_DISCRIMINATOR_VERDICT.md
              </span>
              .
            </p>
            <ul className="space-y-1">
              {inv.missing.map((r) => (
                <ProvenanceRow
                  key={r.id}
                  r={r}
                  hovered={hovered}
                  selected={selected}
                  setHovered={setHovered}
                  onSelect={onSelect}
                />
              ))}
            </ul>
          </div>
        )}

        {/* Ultra-res Blender poster — download card with thumbnail. */}
        <PosterCard />
      </div>
    </GlassCard>
  );
}

/**
 * PosterCard — links to the offline Blender (Cycles + CUDA) ultra-res render of
 * the hero cell. The on-page 3D is the interactive view; this is the poster.
 */
function PosterCard() {
  return (
    <a
      href="/images/substrate_hero_poster.png"
      target="_blank"
      rel="noopener noreferrer"
      download
      className="group block overflow-hidden rounded-xl border border-white/50 bg-white/55 transition-colors hover:bg-white/75"
    >
      <div className="relative aspect-[16/10] w-full overflow-hidden bg-[#0c0f0d]">
        <img
          src="/images/substrate_hero_poster_thumb.png"
          alt="Ultra-resolution Blender Cycles render of the AVA hero cell substrate"
          loading="lazy"
          className="h-full w-full object-cover opacity-95 transition-transform duration-500 group-hover:scale-[1.03]"
        />
        <span className="absolute right-2 top-2 rounded-full bg-black/55 px-2 py-0.5 text-[0.6rem] font-medium text-white/90 backdrop-blur-sm">
          Blender · Cycles · CUDA
        </span>
      </div>
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <div>
          <p className="text-[0.74rem] font-semibold text-emerald-900">
            Download ultra-res poster
          </p>
          <p className="text-[0.62rem] text-emerald-900/55">
            Offline path-traced hero render · full PNG
          </p>
        </div>
        <span className="shrink-0 rounded-lg border border-emerald-700/30 bg-emerald-700/10 px-2.5 py-1 text-[0.66rem] font-medium text-emerald-800 transition-colors group-hover:bg-emerald-700 group-hover:text-white">
          ↓ PNG
        </span>
      </div>
    </a>
  );
}

// ----------------------------------------------------------------------------
// Top-level component
// ----------------------------------------------------------------------------

type View = "hero" | "network";

export default function SubstrateAnatomy() {
  const load = useSubstrateData();
  const [view, setView] = useState<View>("hero");
  const [filter, setFilter] = useState<Status | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  // Network view controls.
  const [colorMode, setColorMode] = useState<NetworkColorMode>("voltage");
  const [showGap, setShowGap] = useState(true);
  const [showChem, setShowChem] = useState(true);
  const [showFlow, setShowFlow] = useState(true);
  // Shared per-frame state, written imperatively by TrajectoryClock inside the
  // canvas and read by hero geometry — never triggers a React re-render.
  const frame = useRef<FrameState>({
    V_mV: -70,
    Ca_uM: 0.2,
    I_Na: 0,
    I_K: 0,
    I_Ca: 0,
    pump: 0,
    t: 0,
  });
  // Shared network frame index — also written by TrajectoryClock; read by the
  // 300-cell soma recolour + ion-flow without re-rendering React.
  const netClock = useRef<NetClock>({ frameIndex: 0, frameFloat: 0, t: 0 });

  const onSelect = (id: string) => setSelected((cur) => (cur === id ? null : id));

  return (
    // Full-bleed break-out from the `prose mx-auto` MDX container.
    <div className="not-prose relative left-1/2 right-1/2 -mx-[50vw] w-screen max-w-[100vw] px-3 py-6 sm:px-6">
      <div className="mx-auto max-w-[1400px]">
        <header className="mb-4">
          <h2 className="text-lg font-semibold text-emerald-900">
            Tier4 Substrate Anatomy — rendered 1:1 from code
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-emerald-900/70">
            An <em>as-is</em> anatomy of the <em>C. elegans</em> Tier4 whole-cell
            electrical substrate: every channel, receptor, pump, transporter, gap
            junction, and ion compartment actually present in the Brian2 model,
            each carrying an honest status read straight from the assemble path.
            Voltage colour and Ca²⁺ glow are driven by a real Brian2 trajectory —
            never illustrative unless flagged.
          </p>
          {load.phase === "ready" && (
            <p className="mt-2 max-w-3xl font-mono text-[0.68rem] leading-snug text-emerald-900/45">
              {load.data.inventory.records.length} structures present (
              {load.data.inventory.summary.status_counts.on} ON ·{" "}
              {load.data.inventory.summary.status_counts.off} default-OFF ·{" "}
              {load.data.inventory.summary.status_counts.orphaned} orphaned) +{" "}
              {load.data.inventory.missing.length} not-integrated · emitted
              read-only from the substrate repo by tests/emit_*.py.
            </p>
          )}
        </header>

        {load.phase === "loading" && (
          <GlassCard className="flex h-[60vh] items-center justify-center">
            <p className="animate-pulse text-sm text-emerald-900/60">
              Loading substrate data…
            </p>
          </GlassCard>
        )}

        {load.phase === "error" && (
          <GlassCard className="flex h-[40vh] flex-col items-center justify-center gap-2 p-6 text-center">
            <p className="text-sm font-medium text-rose-700">
              Could not load substrate data.
            </p>
            <p className="max-w-md font-mono text-[0.7rem] text-rose-600/80">
              {load.message}
            </p>
          </GlassCard>
        )}

        {load.phase === "ready" && (
          <Ready
            data={load.data}
            view={view}
            setView={setView}
            filter={filter}
            setFilter={setFilter}
            hovered={hovered}
            setHovered={setHovered}
            selected={selected}
            onSelect={onSelect}
            frame={frame}
            netClock={netClock}
            playing={playing}
            setPlaying={setPlaying}
            speed={speed}
            setSpeed={setSpeed}
            colorMode={colorMode}
            setColorMode={setColorMode}
            showGap={showGap}
            setShowGap={setShowGap}
            showChem={showChem}
            setShowChem={setShowChem}
            showFlow={showFlow}
            setShowFlow={setShowFlow}
          />
        )}
      </div>
    </div>
  );
}

/**
 * LiveHud — DOM overlay (outside the canvas) that reads the shared frame ref
 * on its own requestAnimationFrame loop and shows live V / Ca / currents.
 * Kept out of React state so it can update at frame rate without re-rendering
 * the rest of the tree.
 */
function LiveHud({ frame }: { frame: React.MutableRefObject<FrameState> }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const el = ref.current;
      if (el) {
        const f = frame.current;
        el.innerHTML =
          `<span class="text-emerald-900/55">V</span> <b>${f.V_mV.toFixed(1)}</b> mV` +
          ` &nbsp; <span class="text-emerald-900/55">Ca</span> <b>${f.Ca_uM.toFixed(3)}</b> µM` +
          ` &nbsp; <span class="text-emerald-900/55">I_K</span> ${(f.I_K * 1e3).toFixed(2)}` +
          ` &nbsp; <span class="text-emerald-900/55">I_Na</span> ${(f.I_Na * 1e3).toFixed(2)}` +
          ` &nbsp; <span class="text-emerald-900/55">pump</span> ${(f.pump * 1e3).toFixed(2)} <span class="text-emerald-900/40">µA/cm²</span>`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frame]);
  return (
    <div className="pointer-events-none absolute left-2 top-2 rounded-lg bg-white/80 px-2.5 py-1 backdrop-blur-sm">
      <div
        ref={ref}
        className="font-mono text-[0.62rem] leading-snug text-emerald-900/80"
      />
    </div>
  );
}

function Ready({
  data,
  view,
  setView,
  filter,
  setFilter,
  hovered,
  setHovered,
  selected,
  onSelect,
  frame,
  netClock,
  playing,
  setPlaying,
  speed,
  setSpeed,
  colorMode,
  setColorMode,
  showGap,
  setShowGap,
  showChem,
  setShowChem,
  showFlow,
  setShowFlow,
}: {
  data: DataBundle;
  view: View;
  setView: (v: View) => void;
  filter: Status | null;
  setFilter: (s: Status | null) => void;
  hovered: string | null;
  setHovered: (id: string | null) => void;
  selected: string | null;
  onSelect: (id: string) => void;
  frame: React.MutableRefObject<FrameState>;
  netClock: React.MutableRefObject<NetClock>;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  speed: number;
  setSpeed: (s: number) => void;
  colorMode: NetworkColorMode;
  setColorMode: (m: NetworkColorMode) => void;
  showGap: boolean;
  setShowGap: (b: boolean) => void;
  showChem: boolean;
  setShowChem: (b: boolean) => void;
  showFlow: boolean;
  setShowFlow: (b: boolean) => void;
}) {
  const traj = data.trajectory;
  const dataIsReal = traj.real && !traj.illustrative;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_360px]">
      {/* 3D canvas column */}
      <div className="flex flex-col gap-3">
        {/* Toolbar: view toggle + provenance badges */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex rounded-xl border border-white/50 bg-white/55 p-1 backdrop-blur-md">
            {(["hero", "network"] as View[]).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                  view === v
                    ? "bg-emerald-700 text-primary-foreground shadow-sm"
                    : "text-emerald-900/70 hover:bg-white/60"
                }`}
              >
                {v === "hero"
                  ? `Hero cell · ${data.heroMorph.cell}`
                  : `Network · ${data.positions.n_cells} cells`}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[0.7rem] font-medium ${
                dataIsReal
                  ? "border-emerald-600/40 bg-emerald-500/12 text-emerald-800"
                  : "border-amber-600/40 bg-amber-500/12 text-amber-800"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  dataIsReal ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              {dataIsReal ? "REAL trajectory" : "ILLUSTRATIVE"}
            </span>
            <span className="hidden font-mono text-[0.65rem] text-emerald-900/50 sm:inline">
              {traj.n_frames}f · {traj.dt_ms}ms · {traj.finite_cells} finite
            </span>
          </div>
        </div>

        {/* The 3D viewport */}
        <GlassCard className="relative overflow-hidden">
          <div className="h-[68vh] min-h-[420px] w-full">
            <Scene
              view={view}
              data={data}
              frame={frame}
              netClock={netClock}
              playing={playing}
              speed={speed}
              hovered={hovered}
              selected={selected}
              setHovered={setHovered}
              onSelect={onSelect}
              colorMode={colorMode}
              showGap={showGap}
              showChem={showChem}
              showFlow={showFlow}
            />
          </div>

          {/* live V/Ca HUD — reads the shared frame ref on its own rAF loop */}
          {view === "hero" && <LiveHud frame={frame} />}

          {/* network legend + colour-mode (network) */}
          {view === "network" && (
            <div className="pointer-events-none absolute left-2 top-2 flex flex-col gap-1.5">
              <div className="pointer-events-auto inline-flex w-fit rounded-lg border border-white/50 bg-white/85 p-0.5 backdrop-blur-sm">
                {(
                  [
                    ["voltage", "V colour"],
                    ["family", "channel family"],
                  ] as [NetworkColorMode, string][]
                ).map(([m, label]) => (
                  <button
                    key={m}
                    onClick={() => setColorMode(m)}
                    className={`rounded-md px-2 py-0.5 text-[0.62rem] font-medium transition-colors ${
                      colorMode === m
                        ? "bg-emerald-700 text-white"
                        : "text-emerald-900/70 hover:bg-emerald-700/10"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="pointer-events-none w-fit rounded-lg bg-white/82 px-2 py-1 font-mono text-[0.55rem] leading-relaxed text-emerald-900/70 backdrop-blur-sm">
                {colorMode === "voltage" ? (
                  <>
                    <span className="text-emerald-900/50">soma V:</span>{" "}
                    <span style={{ color: "#1f5c3a" }}>■</span> −85
                    <span style={{ color: "#d6622b" }}> ■</span> −30 mV · Ca²⁺ glow
                  </>
                ) : (
                  <span>
                    <span style={{ color: "#7d4cd6" }}>■</span> CaK-brake ·{" "}
                    <span style={{ color: "#3c6fd6" }}>■</span> K (dominant gbar)
                  </span>
                )}
              </div>
            </div>
          )}

          {/* playback + speed controls (both views) */}
          <div className="absolute right-2 top-2 flex flex-col items-end gap-1.5">
            <div className="flex items-center gap-1.5 rounded-lg bg-white/85 px-2 py-1 backdrop-blur-sm">
              <button
                onClick={() => setPlaying(!playing)}
                className="rounded px-2 py-0.5 text-[0.65rem] font-medium text-emerald-900 hover:bg-emerald-700/10"
              >
                {playing ? "⏸ pause" : "▶ play"}
              </button>
              {[0.5, 1, 2, 4].map((s) => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`rounded px-1.5 py-0.5 text-[0.6rem] font-mono ${
                    speed === s
                      ? "bg-emerald-700 text-white"
                      : "text-emerald-900/70 hover:bg-emerald-700/10"
                  }`}
                >
                  {s}×
                </button>
              ))}
            </div>

            {/* edge / flow toggles (network) */}
            {view === "network" && (
              <div className="flex items-center gap-1 rounded-lg bg-white/85 px-1.5 py-1 backdrop-blur-sm">
                {(
                  [
                    ["gap", showGap, setShowGap, "gap", "#4f86c6"],
                    ["chem", showChem, setShowChem, "chem", "#3a9e63"],
                    ["flow", showFlow, setShowFlow, "ion-flow", "#d6622b"],
                  ] as [string, boolean, (b: boolean) => void, string, string][]
                ).map(([key, val, setter, label, dot]) => (
                  <button
                    key={key}
                    onClick={() => setter(!val)}
                    className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[0.6rem] font-medium transition-opacity ${
                      val
                        ? "text-emerald-900 opacity-100"
                        : "text-emerald-900/45 opacity-60"
                    } hover:bg-emerald-700/10`}
                  >
                    <span
                      className="h-1.5 w-1.5 rounded-full"
                      style={{
                        backgroundColor: val ? dot : "transparent",
                        border: `1px solid ${dot}`,
                      }}
                    />
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* regime caption */}
          <div className="pointer-events-none absolute bottom-2 left-2 right-2 rounded-lg bg-white/75 px-2.5 py-1 font-mono text-[0.6rem] leading-snug text-emerald-900/60 backdrop-blur-sm">
            regime: {traj.regime}
          </div>
        </GlassCard>

        {/* Scene status note */}
        <p className="text-[0.7rem] italic text-emerald-900/45">
          {view === "hero" ? (
            <>
              Hero molecular geometry is live: AVA membrane recoloured by the real
              voltage trajectory (Ca²⁺ glow), with channel / receptor / pump /
              transporter glyphs sized by gbar and styled by status (ON · default-OFF
              dimmed · ORPHANED wireframe · NOT-INTEGRATED red dashed). The three
              geometry records are pinned as cross-highlightable diamond markers on
              the soma — incl. the default-OFF <span className="font-mono">geo_em_override</span>{" "}
              (opt-in EM C_m override, VB6-only; AVA does not use it), shown amber/wireframe.
            </>
          ) : (
            <>
              Network is live: {data.positions.n_cells} soma at real EM µm
              centroids, each recoloured by the whole-network Brian2 voltage
              trajectory (or tinted by dominant channel family). Gap junctions
              (ohmic, innexin-typed) and signed chem synapses (excitatory green /
              inhibitory rose) are independently toggleable; ion-flow packets
              travel pre→post along the strongest chem edges, driven by live
              source-cell depolarisation. Hover a soma for its identity, channel
              count and passive parameters.
            </>
          )}
        </p>
      </div>

      {/* Completeness legend column */}
      <div className="h-[68vh] min-h-[420px] lg:h-auto">
        <CompletenessLegend
          data={data}
          filter={filter}
          setFilter={setFilter}
          hovered={hovered}
          setHovered={setHovered}
          selected={selected}
          onSelect={onSelect}
        />
      </div>
    </div>
  );
}
