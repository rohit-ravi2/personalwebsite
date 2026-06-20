import * as React from "react";
import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Billboard, Html, Line } from "@react-three/drei";
import * as THREE from "three";
import { type FrameState } from "./HeroCell3D";

/**
 * SchematicHero — the DEFAULT hero view: an exploded, labelled, category-lane
 * cutaway of the AVA whole-cell electrical substrate.
 *
 * Where the dense HeroCell3D render packs every structure onto one realistic
 * membrane (beautiful, but structures overlap and "what is MISSING" is hard to
 * read), this view trades realism for LEGIBILITY:
 *
 *   - Three clearly-separated horizontal ZONES (a cutaway cross-section):
 *       EXTRACELLULAR (top)  · cleft / glia + missing K-siphon ghost
 *       MEMBRANE (middle)    · channels / pumps / transporters / receptors /
 *                              gap-junctions in SPACED, LABELLED rows by family
 *       CYTOPLASM (bottom)   · ion pools, release, peptide-DCV, metabolism,
 *                              + missing grounded K_in source ghost
 *   - ONE representative, individually-spaced, LABELLED glyph per TYPE. gbar is
 *     encoded as a small BAR + number on each glyph, never as instance count.
 *   - STATUS per glyph: on solid · off dim · orphaned wireframe · missing red
 *     dashed ghost in its category slot.
 *   - Clicking a legend row / zone header ISOLATES that category (others recede).
 *   - Hover → tooltip with name + status + file:line + physical_desc.
 *   - The real trajectory still drives a SUBTLE per-glyph tint/glow.
 *
 * Every glyph maps 1:1 to an inventory record (records + missing[]); nothing is
 * fabricated. The four `missing` records render as red-dashed ghosts in their
 * category slot.
 */

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

type Status = "on" | "off" | "orphaned" | "missing";

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

// ---------------------------------------------------------------------------
// Status → material style
// ---------------------------------------------------------------------------

type StatusStyle = {
  opacity: number;
  wireframe: boolean;
  emissiveBoost: number;
  baseColor: string;
};

const STATUS_STYLE: Record<Status, StatusStyle> = {
  on: { opacity: 1.0, wireframe: false, emissiveBoost: 0.22, baseColor: "#2f7a52" },
  off: { opacity: 0.34, wireframe: false, emissiveBoost: 0.0, baseColor: "#c8922b" },
  orphaned: { opacity: 0.5, wireframe: true, emissiveBoost: 0.0, baseColor: "#7c8597" },
  missing: { opacity: 0.55, wireframe: false, emissiveBoost: 0.0, baseColor: "#e11d48" },
};

// Ion-family → glyph hue (so species read at a glance, distinct from status).
const FAMILY_COLOR: Record<string, string> = {
  Ca: "#d6743c",
  K: "#3c6fd6",
  Na: "#d63c8a",
  Cl: "#3cb7d6",
  CaK_brake: "#7d4cd6",
  cation: "#7a8a55",
  receptor: "#2f9e6b",
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
// ROW TAXONOMY — the spaced, labelled sub-family rows in the MEMBRANE zone, and
// the slot grouping for the extracellular / cytoplasm zones. Each record is
// assigned to exactly one row key by a deterministic classifier so the layout
// is stable and 1:1 with the inventory.
// ---------------------------------------------------------------------------

type ZoneKey = "extracellular" | "membrane" | "cytoplasm";

type RowDef = {
  key: string;
  zone: ZoneKey;
  label: string;
  // glyph family colour hint for the row header dot
  hint: string;
};

// Ordered top→bottom within each zone.
const ROWS: RowDef[] = [
  // EXTRACELLULAR
  { key: "ec_glia", zone: "extracellular", label: "Cleft · glia · buffering", hint: "#5b9e8c" },
  // MEMBRANE — sub-family rows
  { key: "m_ca", zone: "membrane", label: "Ca channels (EGL19 · UNC2 · CCA1)", hint: FAMILY_COLOR.Ca },
  { key: "m_k", zone: "membrane", label: "K channels (IRK · SHL1 · EXP2 · KQT …)", hint: FAMILY_COLOR.K },
  { key: "m_kca", zone: "membrane", label: "K-Ca brakes (SLO1 · SLO2 · KCNL1)", hint: FAMILY_COLOR.CaK_brake },
  { key: "m_na", zone: "membrane", label: "Na / leak / HCN (NCA · NaP · DEG-ENaC · HCN)", hint: FAMILY_COLOR.Na },
  { key: "m_cl", zone: "membrane", label: "Cl channels (CLH · ANO · BEST)", hint: FAMILY_COLOR.Cl },
  { key: "m_recep", zone: "membrane", label: "Receptors (NMDA · AMPA · GABA-A · GluCl · nAChR …)", hint: FAMILY_COLOR.receptor },
  { key: "m_pump", zone: "membrane", label: "Pumps · transporters (eat6 · PMCA · KCC2 · ABTS1 · NCX · nkcc1 · NHX)", hint: CATEGORY_COLOR.pump },
  { key: "m_gj", zone: "membrane", label: "Gap junctions (unc-7 / unc-9 innexin)", hint: CATEGORY_COLOR.gap_junction },
  // CYTOPLASM
  { key: "c_ion", zone: "cytoplasm", label: "Ion pools · compartments (Na · K · Ca · Cl)", hint: CATEGORY_COLOR.ion_compartment },
  { key: "c_release", zone: "cytoplasm", label: "Release machinery (graded · NT pools · STD)", hint: CATEGORY_COLOR.release },
  { key: "c_pep", zone: "cytoplasm", label: "Peptide / neuromod (DA · 5HT · peptide DCV)", hint: CATEGORY_COLOR.neuromod_peptide },
  { key: "c_meta", zone: "cytoplasm", label: "Metabolism (WCM-AVA · ATP · iCEL1314) · geometry", hint: CATEGORY_COLOR.metabolism },
];

const ZONE_META: Record<ZoneKey, { label: string; sub: string; color: string }> = {
  extracellular: { label: "EXTRACELLULAR", sub: "cleft · glia · K buffering", color: "#5b9e8c" },
  membrane: { label: "MEMBRANE", sub: "channels · receptors · pumps · gap junctions", color: "#2f7a52" },
  cytoplasm: { label: "CYTOPLASM", sub: "ion pools · release · peptides · metabolism", color: "#9e7d3c" },
};

/**
 * Assign an inventory record to a layout row. Deterministic, total (every record
 * lands somewhere), and 1:1. Uses category + subtype + id so e.g. the K-Ca
 * brakes split out of the generic K row.
 */
function rowForRecord(r: InventoryRecord): string {
  const sub = r.subtype.toLowerCase();
  const cat = r.category;

  if (cat === "gap_junction") return "m_gj";
  if (cat === "pump" || cat === "transporter") return "m_pump";
  if (cat === "receptor") return "m_recep";
  if (cat === "release") return "c_release";
  if (cat === "neuromod_peptide") return "c_pep";
  if (cat === "metabolism" || cat === "geometry") return "c_meta";

  if (cat === "ion_compartment") {
    // The glial / extracellular buffering records sit in the EC zone; the
    // intracellular pools + multi-compartment interfaces sit in the cytoplasm.
    if (/glia|tissue|ephaptic/.test(r.id) || /glia/.test(sub)) return "ec_glia";
    if (r.id === "missing_glial_k_siphon") return "ec_glia";
    return "c_ion";
  }

  if (cat === "channel") {
    // MISSING Ca-K brake-relief ghost belongs in the K-Ca brake row (audit).
    if (r.id === "missing_grounded_brake_relief" || /brake/.test(sub)) return "m_kca";
    if (/k\(ca/.test(sub) || /bk|sk|slack|slo/.test(sub)) return "m_kca";
    // cation / Ih (HCN) and Na / leak / persistent → the Na/leak/HCN row.
    if (sub.startsWith("cation") || /ih|hcn|nalcn|enac|persistent/.test(sub)) return "m_na";
    if (sub.startsWith("na")) return "m_na";
    // Ca-Cl (ANO/BEST, anoctamin/bestrophin) → Cl row; pure Ca → Ca row.
    if (/ca-cl|tmem16|bestrophin/.test(sub)) return "m_cl";
    if (sub.startsWith("ca") || /cav/.test(sub)) return "m_ca";
    if (sub.startsWith("cl") || /clc|cl /.test(sub)) return "m_cl";
    if (sub.startsWith("k") || /kv|kir|k2p|herg|eag|kcnq|shaw|shaker/.test(sub)) return "m_k";
    return "m_k"; // defensive: any unclassified channel → K row
  }

  // The MISSING channel ghosts (grounded γ_NCA, brake relief) carry category
  // "channel" already handled above. Defensive catch-all → membrane Na row.
  return "m_na";
}

// ---------------------------------------------------------------------------
// Layout geometry — a flat-ish 2.5D board. X = position within a row, Y = zone
// stacking, small Z jitter so it still reads as 3D under gentle rotation.
// ---------------------------------------------------------------------------

const ZONE_Y: Record<ZoneKey, number> = {
  extracellular: 9.2,
  membrane: 0,
  cytoplasm: -9.2,
};

const ROW_GAP = 2.35; // vertical spacing between rows inside a zone
const COL_GAP = 2.5; // horizontal spacing between glyphs in a row
const GLYPH_Z = 0.0;

type PlacedGlyph = {
  rec: InventoryRecord;
  rowKey: string;
  zone: ZoneKey;
  pos: THREE.Vector3;
  color: string;
  gbarNorm: number | null; // 0..1 within its row family (for the bar)
};

// ---------------------------------------------------------------------------
// Trajectory → subtle tint. Small, so animation does not reintroduce clutter.
// ---------------------------------------------------------------------------

function voltageTint(v: number, out: THREE.Color): THREE.Color {
  const t = Math.min(1, Math.max(0, (v + 80) / 30)); // -80..-50
  const cool = new THREE.Color("#1f5c3a");
  const warm = new THREE.Color("#d68a2b");
  out.copy(cool).lerp(warm, t);
  return out;
}

// ---------------------------------------------------------------------------
// One schematic glyph — a clean primitive keyed to category, with a status
// material, a gbar bar, a persistent HTML label, and a hover tooltip.
// ---------------------------------------------------------------------------

function categoryShape(cat: string): "barrel" | "disc" | "cap" | "wedge" | "node" | "pool" {
  switch (cat) {
    case "channel":
      return "barrel";
    case "receptor":
      return "disc";
    case "pump":
      return "cap";
    case "transporter":
      return "wedge";
    case "gap_junction":
      return "node";
    case "ion_compartment":
      return "pool";
    default:
      return "node";
  }
}

function GlyphBody({
  shape,
  size,
  children,
}: {
  shape: ReturnType<typeof categoryShape>;
  size: number;
  children: React.ReactNode;
}) {
  switch (shape) {
    case "barrel":
      return (
        <mesh>
          <cylinderGeometry args={[size * 0.62, size * 0.62, size * 1.5, 24, 1]} />
          {children}
        </mesh>
      );
    case "disc":
      return (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[size * 0.85, size * 0.85, size * 0.5, 28]} />
          {children}
        </mesh>
      );
    case "cap":
      return (
        <mesh>
          <sphereGeometry args={[size * 0.8, 28, 20, 0, Math.PI * 2, 0, Math.PI * 0.62]} />
          {children}
        </mesh>
      );
    case "wedge":
      return (
        <mesh>
          <coneGeometry args={[size * 0.72, size * 1.4, 12, 1]} />
          {children}
        </mesh>
      );
    case "pool":
      return (
        <mesh>
          <sphereGeometry args={[size * 0.8, 24, 24]} />
          {children}
        </mesh>
      );
    case "node":
    default:
      return (
        <mesh>
          <icosahedronGeometry args={[size * 0.78, 1]} />
          {children}
        </mesh>
      );
  }
}

// MISSING → a red dashed ring + pin marking the empty slot the structure goes in.
function MissingGhost({ size }: { size: number }) {
  const ring = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    const seg = 28;
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * size * 0.9, Math.sin(a) * size * 0.9, 0));
    }
    return pts;
  }, [size]);
  return (
    <group>
      <Line points={ring} color="#e11d48" lineWidth={1.6} dashed dashScale={5} dashSize={0.14} gapSize={0.1} />
      <Line
        points={[new THREE.Vector3(0, -size * 1.1, 0), new THREE.Vector3(0, size * 1.1, 0)]}
        color="#e11d48"
        lineWidth={1.6}
        dashed
        dashScale={5}
        dashSize={0.14}
        gapSize={0.1}
      />
    </group>
  );
}

function SchematicGlyph({
  g,
  frame,
  hovered,
  selected,
  activeCategory,
  setHovered,
  onSelect,
}: {
  g: PlacedGlyph;
  frame: React.MutableRefObject<FrameState>;
  hovered: string | null;
  selected: string | null;
  activeCategory: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const { rec } = g;
  const style = STATUS_STYLE[rec.status];
  const shape = categoryShape(rec.category);
  const groupRef = useRef<THREE.Group>(null);
  const matRef = useRef<THREE.MeshStandardMaterial>(null);
  const baseColRef = useRef(new THREE.Color(g.color));
  const tmp = useRef(new THREE.Color());
  const isHi = hovered === rec.id || selected === rec.id;

  // Isolation: when a category is active and this glyph isn't in it, recede.
  const isolated = activeCategory != null && rec.category !== activeCategory;
  const size = 0.72;

  useFrame(() => {
    const grp = groupRef.current;
    if (grp) {
      const target = isHi ? 1.4 : isolated ? 0.82 : 1.0;
      grp.scale.lerp(new THREE.Vector3(target, target, target), 0.18);
    }
    const m = matRef.current;
    if (m && rec.status === "on" && !isolated) {
      // SUBTLE V tint blended into the family colour; small Ca emissive lift.
      const f = frame.current;
      voltageTint(f.V_mV, tmp.current);
      m.color.copy(baseColRef.current).lerp(tmp.current, 0.22);
      const caN = Math.min(1, Math.max(0, (f.Ca_uM - 0.15) / 0.3));
      m.emissiveIntensity = style.emissiveBoost + 0.35 * caN * (isHi ? 1.4 : 1);
    } else if (m) {
      m.color.copy(baseColRef.current);
    }
  });

  // dim opacity further when isolated-out, but keep faintly visible
  const opacity = isolated ? Math.min(style.opacity, 0.12) : style.opacity;
  const labelStrong = isHi || (!isolated && (activeCategory == null || rec.category === activeCategory));

  return (
    <group
      ref={groupRef}
      position={g.pos}
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
      {rec.status === "missing" ? (
        <MissingGhost size={size} />
      ) : (
        <GlyphBody shape={shape} size={size}>
          <meshStandardMaterial
            ref={matRef}
            color={g.color}
            emissive={g.color}
            emissiveIntensity={style.emissiveBoost}
            transparent
            opacity={opacity}
            wireframe={style.wireframe}
            roughness={0.42}
            metalness={0.12}
            depthWrite={!isolated}
          />
        </GlyphBody>
      )}

      {/* gbar BAR — a small vertical bar beside the glyph encoding gbar_value_AVA
          (normalised within its row family). Number lives in the label/tooltip. */}
      {g.gbarNorm != null && g.gbarNorm > 0 && rec.status !== "missing" && !isolated && (
        <group position={[size * 0.95, -size * 0.75, 0]}>
          {/* track */}
          <mesh position={[0, size * 0.75, 0]}>
            <boxGeometry args={[0.1, size * 1.5, 0.1]} />
            <meshStandardMaterial color="#cbbfa6" transparent opacity={0.35} />
          </mesh>
          {/* fill */}
          <mesh position={[0, (size * 1.5 * g.gbarNorm) / 2, 0]}>
            <boxGeometry args={[0.16, Math.max(0.04, size * 1.5 * g.gbarNorm), 0.16]} />
            <meshStandardMaterial color={g.color} emissive={g.color} emissiveIntensity={0.25} />
          </mesh>
        </group>
      )}

      {/* persistent compact label (Billboard so it always faces camera) */}
      <Billboard position={[0, -size * 1.45, 0]}>
        <Html center distanceFactor={18} zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
          <div
            className={`whitespace-nowrap rounded px-1.5 py-0.5 text-center font-medium shadow-sm backdrop-blur-sm transition-opacity ${
              rec.status === "missing"
                ? "border border-rose-500/50 bg-rose-50/90 text-rose-800"
                : rec.status === "orphaned"
                  ? "border border-slate-400/50 bg-white/80 text-slate-700"
                  : rec.status === "off"
                    ? "border border-amber-500/40 bg-amber-50/85 text-amber-900/90"
                    : "border border-emerald-600/30 bg-white/85 text-emerald-900"
            }`}
            style={{
              fontSize: "0.5rem",
              opacity: labelStrong ? 1 : 0.25,
            }}
          >
            {shortName(rec)}
          </div>
        </Html>
      </Billboard>

      {/* hover/select tooltip — full provenance */}
      {isHi && (
        <Html center distanceFactor={16} position={[0, size * 1.7, 0]} zIndexRange={[60, 0]}>
          <div className="pointer-events-none w-56 rounded-lg border border-white/50 bg-white/92 p-2 text-left shadow-lg backdrop-blur-md">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[0.72rem] font-semibold text-emerald-950">{rec.name}</span>
              <span className="shrink-0 font-mono text-[0.55rem] uppercase text-emerald-900/50">
                {statusLabel(rec.status)}
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

function statusLabel(s: Status): string {
  return s === "off" ? "default-OFF" : s === "missing" ? "NOT INTEGRATED" : s;
}

// A short label for the persistent on-glyph text (gene/abbrev). Strips the
// parenthetical and long suffixes so it reads at the default camera.
function shortName(rec: InventoryRecord): string {
  let n = rec.name;
  // keep gene-ish prefix before a parenthetical, e.g. "NMDA (nmr-1/2)" → "NMDA"
  const paren = n.indexOf(" (");
  if (paren > 0) n = n.slice(0, paren);
  if (n.length > 22) n = n.slice(0, 21) + "…";
  return n;
}

// ---------------------------------------------------------------------------
// Zone slab + header — a translucent panel behind each zone and a Billboard
// header that ISOLATES that zone's categories when clicked.
// ---------------------------------------------------------------------------

function ZoneSlab({
  zone,
  width,
  height,
  onSelect,
}: {
  zone: ZoneKey;
  width: number;
  height: number;
  onSelect: () => void;
}) {
  const meta = ZONE_META[zone];
  const y = ZONE_Y[zone];
  return (
    <group position={[0, y, -1.2]}>
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerOver={() => (document.body.style.cursor = "pointer")}
        onPointerOut={() => (document.body.style.cursor = "auto")}
      >
        <planeGeometry args={[width, height]} />
        <meshStandardMaterial
          color={meta.color}
          transparent
          opacity={0.06}
          roughness={0.9}
          depthWrite={false}
        />
      </mesh>
      {/* header pinned to the left edge */}
      <Html
        position={[-width / 2 + 0.4, height / 2 - 0.9, 0.1]}
        distanceFactor={22}
        zIndexRange={[8, 0]}
      >
        <div
          className="cursor-pointer select-none whitespace-nowrap rounded-md border bg-white/70 px-2 py-1 shadow-sm backdrop-blur-sm"
          style={{ borderColor: `${meta.color}55` }}
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
        >
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: meta.color }} />
            <span className="text-[0.62rem] font-bold tracking-wide" style={{ color: meta.color }}>
              {meta.label}
            </span>
          </div>
          <p className="mt-0.5 text-[0.5rem] text-emerald-900/55">{meta.sub}</p>
        </div>
      </Html>
    </group>
  );
}

// Row label — a small Billboard tag pinned to the left of each row.
function RowLabel({ row, x, y }: { row: RowDef; x: number; y: number }) {
  return (
    <Html position={[x, y + 0.95, 0]} distanceFactor={20} zIndexRange={[7, 0]} style={{ pointerEvents: "none" }}>
      <div className="flex items-center gap-1 whitespace-nowrap" style={{ transform: "translateX(0.5rem)" }}>
        <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: row.hint }} />
        <span className="text-[0.5rem] font-medium text-emerald-900/55">{row.label}</span>
      </div>
    </Html>
  );
}

// ---------------------------------------------------------------------------
// SchematicHero — assembles the lanes.
// ---------------------------------------------------------------------------

export default function SchematicHero({
  records,
  frame,
  hovered,
  selected,
  activeCategory,
  setHovered,
  onSelect,
}: {
  records: InventoryRecord[]; // includes the 4 missing[] ghosts
  frame: React.MutableRefObject<FrameState>;
  hovered: string | null;
  selected: string | null;
  // when a legend category filter is active, isolate that category in the scene
  activeCategory: string | null;
  setHovered: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  // 1. bucket every record into a row.
  const byRow = useMemo(() => {
    const m = new Map<string, InventoryRecord[]>();
    for (const row of ROWS) m.set(row.key, []);
    for (const r of records) {
      const k = rowForRecord(r);
      (m.get(k) ?? m.set(k, []).get(k)!).push(r);
    }
    // stable order inside a row: on → off → orphaned → missing, then by gbar desc
    const ord: Status[] = ["on", "off", "orphaned", "missing"];
    for (const recs of m.values()) {
      recs.sort((a, b) => {
        const so = ord.indexOf(a.status) - ord.indexOf(b.status);
        if (so !== 0) return so;
        return (b.gbar_value_AVA ?? 0) - (a.gbar_value_AVA ?? 0);
      });
    }
    return m;
  }, [records]);

  // 2. compute per-zone row layout + glyph placements.
  const { glyphs, rowAnchors, boardWidth, zoneHeights } = useMemo(() => {
    const glyphs: PlacedGlyph[] = [];
    const rowAnchors: { row: RowDef; x: number; y: number }[] = [];

    // group rows by zone, keep ROWS order
    const rowsByZone: Record<ZoneKey, RowDef[]> = {
      extracellular: [],
      membrane: [],
      cytoplasm: [],
    };
    for (const row of ROWS) rowsByZone[row.zone].push(row);

    let maxCols = 1;
    for (const recs of byRow.values()) maxCols = Math.max(maxCols, recs.length);
    const boardWidth = Math.max(10, (maxCols + 1) * COL_GAP + 4);

    const zoneHeights: Record<ZoneKey, number> = {
      extracellular: 0,
      membrane: 0,
      cytoplasm: 0,
    };

    (Object.keys(rowsByZone) as ZoneKey[]).forEach((zone) => {
      const rows = rowsByZone[zone];
      const n = rows.length;
      const totalH = (n - 1) * ROW_GAP;
      zoneHeights[zone] = totalH + 3.2; // pad for header + labels
      const yBase = ZONE_Y[zone] + totalH / 2;
      rows.forEach((row, ri) => {
        const y = yBase - ri * ROW_GAP;
        const recs = byRow.get(row.key) ?? [];
        // left-pad so rows start past the row label gutter
        const xStart = -((recs.length - 1) * COL_GAP) / 2;
        rowAnchors.push({ row, x: -boardWidth / 2 + 0.5, y });
        recs.forEach((rec, ci) => {
          const x = xStart + ci * COL_GAP;
          // small deterministic z jitter for 2.5D depth under rotation
          const z = GLYPH_Z + ((ci % 2 === 0 ? 1 : -1) * 0.18) + (ri % 2) * 0.1;
          // gbar normalisation within the row family
          const fam = recs
            .map((rr) => rr.gbar_value_AVA ?? 0)
            .reduce((a, b) => Math.max(a, b), 0);
          const gbarNorm =
            rec.gbar_value_AVA != null && fam > 0
              ? Math.sqrt(rec.gbar_value_AVA / fam)
              : rec.gbar_value_AVA === 0
                ? 0
                : null;
          // colour: channels/receptors by ion family from subtype; else category
          const color = colorForRecord(rec);
          glyphs.push({
            rec,
            rowKey: row.key,
            zone,
            pos: new THREE.Vector3(x, y, z),
            color,
            gbarNorm,
          });
        });
      });
    });

    return { glyphs, rowAnchors, boardWidth, zoneHeights };
  }, [byRow]);

  return (
    <group>
      {/* zone slabs (click to isolate that zone's leading category) */}
      {(Object.keys(ZONE_META) as ZoneKey[]).map((zone) => (
        <ZoneSlab
          key={zone}
          zone={zone}
          width={boardWidth}
          height={zoneHeights[zone]}
          onSelect={() => {
            // isolate by zone → pick the representative category for that zone
            // (handled upstream via onSelect of a representative record). Here we
            // toggle the first record in that zone so the legend + scene align.
            const first = glyphs.find((g) => g.zone === zone);
            if (first) onSelect(first.rec.id);
          }}
        />
      ))}

      {/* row labels */}
      {rowAnchors.map((a) => (
        <RowLabel key={a.row.key} row={a.row} x={a.x} y={a.y} />
      ))}

      {/* glyphs */}
      {glyphs.map((g) => (
        <SchematicGlyph
          key={g.rec.id}
          g={g}
          frame={frame}
          hovered={hovered}
          selected={selected}
          activeCategory={activeCategory}
          setHovered={setHovered}
          onSelect={onSelect}
        />
      ))}
    </group>
  );
}

// colour for a record: ion family (channels/receptors) by subtype, else category
function colorForRecord(rec: InventoryRecord): string {
  if (rec.status === "missing") return STATUS_STYLE.missing.baseColor;
  const sub = rec.subtype.toLowerCase();
  if (rec.category === "channel") {
    if (/k\(ca|bk|sk|slack|slo/.test(sub)) return FAMILY_COLOR.CaK_brake;
    if (sub.startsWith("cation") || /ih|hcn/.test(sub)) return FAMILY_COLOR.cation;
    if (sub.startsWith("na") || /nalcn|enac|persistent/.test(sub)) return FAMILY_COLOR.Na;
    if (/ca-cl|tmem16|bestrophin/.test(sub)) return FAMILY_COLOR.Cl;
    if (sub.startsWith("ca")) return FAMILY_COLOR.Ca;
    if (sub.startsWith("cl") || /clc/.test(sub)) return FAMILY_COLOR.Cl;
    return FAMILY_COLOR.K;
  }
  if (rec.category === "receptor") return FAMILY_COLOR.receptor;
  return CATEGORY_COLOR[rec.category] ?? "#2f7a52";
}
