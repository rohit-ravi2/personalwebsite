import * as React from "react";
import { useMemo } from "react";
import * as THREE from "three";

/**
 * GlyphGeometry — AlphaFold-informed, high-fidelity glyph geometry for the
 * Tier4 hero cell. Kept in its own module so it can be authored independently
 * of HeroCell3D's scene assembly.
 *
 * Two tiers of geometry, both higher-poly than the original primitives:
 *
 *   1. STRUCTURE-DERIVED (preferred): for any channel/receptor/gap-junction
 *      that has an AlphaFold/PDB signature (emitted by
 *      visualization/emit_glyph_signatures.py from the CA backbone), we build a
 *      THREE.LatheGeometry by revolving the structure's height-binned radial
 *      silhouette (outer_profile). A central pore is carved using pore_frac so
 *      ion channels read as conducting barrels. The glyph SHAPE is therefore
 *      literally derived from the folded protein, not a hand-authored primitive.
 *
 *   2. PARAMETRIC FALLBACK: for structures with no PDB, we render a refined,
 *      beveled, high-segment-count parametric glyph keyed to category
 *      (barrel / cap / disc / wedge / cluster). Segment counts are raised and
 *      shapes beveled so the molecular detail rises to use spare GPU headroom
 *      without OOM risk (geometry is tiny; ~70 glyphs total).
 */

// Re-exported so SubstrateAnatomy can type the data bundle without coupling to
// HeroCell3D's internal module graph.
export type GlyphSignature = {
  pdb: string;
  gene: string;
  /** Structural provenance: "exact" (channel's own gene PDB), "paralog" (close
   * same-subfamily worm paralog), or "family" (distant family fold proxy).
   * Optional for back-compat with schema_version 1 signatures. */
  match?: "exact" | "paralog" | "family";
  /** Human-readable note on the structure source / fold rationale. */
  note?: string;
  n_ca: number;
  n_residues: number;
  n_chains: number;
  height_A: number;
  radius_A: number;
  aspect: number;
  pore_frac: number;
  outer_profile: number[];
  inner_profile: number[];
};

export type GlyphSignatures = {
  schema_version: number;
  generated: string;
  generator: string;
  n_bins: number;
  note: string;
  signatures: Record<string, GlyphSignature>;
  missing_pdb: { id: string; stem: string }[];
};

export type GlyphShape = "barrel" | "cap" | "disc" | "wedge" | "cluster";

// ---------------------------------------------------------------------------
// TRUE backbone surface meshes (emitted by visualization/emit_backbone_meshes.py
// from the CA backbone of the EXACT-fold PDBs). Unlike the lathe signature
// (a revolved silhouette), these are indexed triangle tubes swept along the real
// folded CA trace, per chain, so the EXACT glyphs render the actual backbone
// shape rather than a rotationally-symmetric profile.
// ---------------------------------------------------------------------------

export type BackboneChainMesh = {
  chain: string;
  n_spine: number;
  n_vertices: number;
  n_triangles: number;
  /** flat [x,y,z, x,y,z, ...] in a normalised frame (longest extent == 1,
   * centered, principal axis aligned to +Y). */
  positions: number[];
  normals: number[];
  indices: number[];
};

export type BackboneMesh = {
  pdb: string;
  gene: string;
  match: "exact";
  note?: string;
  n_ca: number;
  n_chains: number;
  chains: BackboneChainMesh[];
};

export type BackboneMeshes = {
  schema_version: number;
  generated: string;
  generator: string;
  decimation: { max_spine: number; radial_div: number; smooth_passes: number };
  n_meshes: number;
  total_vertices: number;
  total_triangles: number;
  note: string;
  meshes: Record<string, BackboneMesh>;
};

// ---------------------------------------------------------------------------
// Innexin gap-junction HEMICHANNEL signature (emitted by
// visualization/emit_hemichannel_signature.py from the AF2-multimer unc7/unc9
// heterodimer). The per-subunit silhouette is structure-derived; the C6
// hexameric ring is imposed from innexin biology (see ring_provenance), so the
// gap junction renders as a real ring-of-subunits hemichannel instead of a
// generic "cluster" icosahedron.
// ---------------------------------------------------------------------------

export type HemichannelSubunit = {
  chain: string;
  n_ca: number;
  outer_profile: number[];
  pore_frac: number;
  aspect: number;
  height_A: number;
  radius_A: number;
};

export type HemichannelSignature = {
  schema_version: number;
  generated: string;
  generator: string;
  source_pdb: string;
  n_subunits: number;
  n_bins: number;
  ring_radius_A: number;
  inter_chain_centroid_A: number;
  subunits: Record<string, HemichannelSubunit>;
  ring_provenance: string;
  note: string;
};

// Map an inventory record id (chan_*/recep_*) to the signature key used by the
// emitter (bare channel id, e.g. "slo1").
export function signatureKey(recId: string): string {
  return recId.replace(/^chan_/, "").replace(/^recep_/, "");
}

// ---------------------------------------------------------------------------
// Structure-derived lathe geometry from a radial silhouette.
// ---------------------------------------------------------------------------

/**
 * Build a LatheGeometry from a normalised outer_profile (radii in [0,1] along
 * the protein's principal axis). The profile is smoothed lightly, scaled so the
 * widest point matches `size`, and the overall height set from the structure's
 * aspect ratio so e.g. the tall NMDA receptor reads taller than the squat
 * KCNL-1. The lathe is centered on the local origin and oriented along +Y
 * (matching the membrane-normal convention used by the glyph group quaternion).
 */
export function buildLatheGeometry(
  sig: GlyphSignature,
  size: number,
): THREE.LatheGeometry {
  const prof = sig.outer_profile;
  const n = prof.length;
  // height in scene units: tie to aspect so tall channels look tall, but clamp
  // so nothing dominates the soma. radius spans ~`size`.
  const radius = size * 1.15;
  const height = size * 2.0 * Math.min(2.6, Math.max(0.7, sig.aspect));

  // light 1-2-1 smoothing of the silhouette to remove single-bin spikes
  const sm = prof.map((_, i) => {
    const a = prof[Math.max(0, i - 1)];
    const b = prof[i];
    const c = prof[Math.min(n - 1, i + 1)];
    return (a + 2 * b + c) / 4;
  });

  const pts: THREE.Vector2[] = [];
  // bottom cap point on the axis so the lathe is a closed solid
  pts.push(new THREE.Vector2(0.0001, -height / 2));
  for (let i = 0; i < n; i++) {
    const y = -height / 2 + (i / (n - 1)) * height;
    const r = Math.max(0.04, sm[i]) * radius;
    pts.push(new THREE.Vector2(r, y));
  }
  pts.push(new THREE.Vector2(0.0001, height / 2));

  // high angular resolution -> smooth revolved surface (cheap: ~48 segments)
  const geo = new THREE.LatheGeometry(pts, 48);
  geo.computeVertexNormals();
  return geo;
}

/**
 * StructureGlyph — renders the structure-derived lathe plus a thin central pore
 * tube (so conducting channels show a lumen). `children` is the shared material.
 */
export function StructureGlyph({
  sig,
  size,
  showPore,
  children,
}: {
  sig: GlyphSignature;
  size: number;
  showPore: boolean;
  children: React.ReactNode;
}) {
  const geo = useMemo(() => buildLatheGeometry(sig, size), [sig, size]);
  const poreR = useMemo(
    () => Math.max(0.02, sig.pore_frac) * size * 0.9,
    [sig, size],
  );
  const height = size * 2.0 * Math.min(2.6, Math.max(0.7, sig.aspect));
  return (
    <group>
      <mesh geometry={geo}>{children}</mesh>
      {showPore && (
        <mesh>
          <cylinderGeometry args={[poreR, poreR, height * 1.02, 20, 1, true]} />
          <meshStandardMaterial
            color="#0b1f17"
            transparent
            opacity={0.55}
            side={THREE.BackSide}
            roughness={0.9}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
}

// ---------------------------------------------------------------------------
// TRUE backbone-mesh glyph (EXACT folds only). Builds a THREE.BufferGeometry per
// chain from the emitted indexed triangle tube and renders the real folded
// backbone path. The mesh frame is already normalised (longest extent == 1,
// centered, principal axis -> +Y), so we scale by `size` to match the lathe
// glyphs' footprint, and add the same dark pore lumen down the +Y axis so
// conducting channels still read as barrels.
// ---------------------------------------------------------------------------

function buildBackboneGeometry(chain: BackboneChainMesh): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(chain.positions, 3),
  );
  if (chain.normals && chain.normals.length === chain.positions.length) {
    geo.setAttribute(
      "normal",
      new THREE.Float32BufferAttribute(chain.normals, 3),
    );
  }
  geo.setIndex(chain.indices);
  if (!chain.normals || chain.normals.length !== chain.positions.length) {
    geo.computeVertexNormals();
  }
  return geo;
}

export function BackboneGlyph({
  mesh,
  size,
  showPore,
  children,
}: {
  mesh: BackboneMesh;
  size: number;
  showPore: boolean;
  children: React.ReactNode;
}) {
  // The emitted frame is normalised to extent 1; match the lathe footprint by
  // scaling so the glyph spans ~size*2 (lathe height ~ size*2). The principal
  // axis is +Y, so the multi-chain pore-forming bundle stands membrane-normal.
  const s = size * 2.0;
  const geos = useMemo(
    () => mesh.chains.map((c) => buildBackboneGeometry(c)),
    [mesh],
  );
  return (
    <group scale={[s, s, s]}>
      {geos.map((geo, i) => (
        <mesh key={i} geometry={geo}>
          {children}
        </mesh>
      ))}
      {showPore && (
        <mesh>
          {/* normalised frame: full height ~1, so pore tube spans ~1.05 */}
          <cylinderGeometry args={[0.07, 0.07, 1.05, 20, 1, true]} />
          <meshStandardMaterial
            color="#0b1f17"
            transparent
            opacity={0.55}
            side={THREE.BackSide}
            roughness={0.9}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Refined parametric fallback glyphs (higher poly + beveled).
// ---------------------------------------------------------------------------

// A beveled "barrel" built as a lathe with a slight waist + chamfered rims,
// far smoother than the old 8-sided cylinder.
function beveledBarrelPoints(size: number): THREE.Vector2[] {
  const h = size * 2.0;
  const r = size;
  return [
    new THREE.Vector2(0.0001, -h / 2),
    new THREE.Vector2(r * 0.7, -h / 2),
    new THREE.Vector2(r * 0.92, -h / 2 + size * 0.18), // chamfered lower rim
    new THREE.Vector2(r * 0.86, -h * 0.18),
    new THREE.Vector2(r * 0.8, 0), // gentle waist
    new THREE.Vector2(r * 0.86, h * 0.18),
    new THREE.Vector2(r * 0.92, h / 2 - size * 0.18), // chamfered upper rim
    new THREE.Vector2(r * 0.7, h / 2),
    new THREE.Vector2(0.0001, h / 2),
  ];
}

export function ParametricGlyphBody({
  shape,
  size,
  children,
}: {
  shape: GlyphShape;
  size: number;
  children: React.ReactNode;
}) {
  const barrelGeo = useMemo(
    () =>
      shape === "barrel"
        ? new THREE.LatheGeometry(beveledBarrelPoints(size), 40)
        : null,
    [shape, size],
  );

  switch (shape) {
    case "barrel": // ion channel — beveled transmembrane barrel + lumen
      return (
        <group>
          <mesh geometry={barrelGeo!}>{children}</mesh>
          <mesh>
            <cylinderGeometry
              args={[size * 0.22, size * 0.22, size * 2.05, 18, 1, true]}
            />
            <meshStandardMaterial
              color="#0b1f17"
              transparent
              opacity={0.5}
              side={THREE.BackSide}
              roughness={0.9}
              depthWrite={false}
            />
          </mesh>
        </group>
      );
    case "cap": // pump — smooth capped dome on a short collar
      return (
        <group>
          <mesh>
            <sphereGeometry args={[size, 40, 28, 0, Math.PI * 2, 0, Math.PI * 0.62]} />
            {children}
          </mesh>
          <mesh position={[0, -size * 0.28, 0]}>
            <cylinderGeometry args={[size * 0.78, size * 0.6, size * 0.5, 32]} />
            {children}
          </mesh>
        </group>
      );
    case "disc": { // receptor — rounded torus-rimmed disc (pentameric read)
      return (
        <group rotation={[Math.PI / 2, 0, 0]}>
          <mesh>
            <cylinderGeometry args={[size * 1.18, size * 1.18, size * 0.7, 36]} />
            {children}
          </mesh>
          <mesh>
            <torusGeometry args={[size * 1.18, size * 0.16, 16, 40]} />
            {children}
          </mesh>
        </group>
      );
    }
    case "wedge": // transporter — chamfered cotransporter prism
      return (
        <mesh>
          <coneGeometry args={[size, size * 1.8, 10, 1]} />
          {children}
        </mesh>
      );
    case "cluster": // gap junction / release — rounded clustered node
      return (
        <mesh>
          <icosahedronGeometry args={[size * 1.05, 1]} />
          {children}
        </mesh>
      );
  }
}

// ---------------------------------------------------------------------------
// Innexin gap-junction hemichannel glyph — a C6 ring of structure-derived
// innexin subunits around a central pore.
// ---------------------------------------------------------------------------

/**
 * Build a LatheGeometry for ONE innexin subunit from a hemichannel subunit
 * silhouette. Unlike buildLatheGeometry (which spans the whole glyph radius),
 * each subunit is a slimmer revolved lobe (it is one of N around the ring), so
 * the radius is scaled down and the subunit reads as a distinct lobe rather
 * than filling the whole footprint.
 */
function buildSubunitLathe(
  sub: HemichannelSubunit,
  subRadius: number,
  height: number,
): THREE.LatheGeometry {
  const prof = sub.outer_profile;
  const n = prof.length;
  const sm = prof.map((_, i) => {
    const a = prof[Math.max(0, i - 1)];
    const b = prof[i];
    const c = prof[Math.min(n - 1, i + 1)];
    return (a + 2 * b + c) / 4;
  });
  const pts: THREE.Vector2[] = [];
  pts.push(new THREE.Vector2(0.0001, -height / 2));
  for (let i = 0; i < n; i++) {
    const y = -height / 2 + (i / (n - 1)) * height;
    const r = Math.max(0.04, sm[i]) * subRadius;
    pts.push(new THREE.Vector2(r, y));
  }
  pts.push(new THREE.Vector2(0.0001, height / 2));
  const geo = new THREE.LatheGeometry(pts, 28);
  geo.computeVertexNormals();
  return geo;
}

/**
 * HemichannelGlyph — renders an innexin gap-junction hemichannel (innexon) as a
 * hexameric (C6, n_subunits from the signature) ring of structure-derived
 * subunit lathes around an open central pore. Replaces the generic "cluster"
 * icosahedron for unc-7 / unc-9 gap junctions. The ring axis is the local +Y
 * (membrane normal), matching the other glyphs' orientation convention.
 *
 * The subunit silhouette is taken from the AF2-multimer (real fold); the ring
 * count + radial placement encode the hexameric innexon biology. A thin pore
 * cylinder marks the conduction pathway down the channel axis.
 */
export function HemichannelGlyph({
  hemi,
  size,
  children,
}: {
  hemi: HemichannelSignature;
  size: number;
  children: React.ReactNode;
}) {
  // Prefer the larger unc-7 fold as the representative subunit silhouette; fall
  // back to whatever subunit is present.
  const sub = useMemo(() => {
    const subs = hemi.subunits;
    return subs["unc-7"] ?? Object.values(subs)[0];
  }, [hemi]);

  const n = Math.max(3, hemi.n_subunits || 6);
  // Geometry budget: the whole hemichannel footprint ~ size*1.2. Subunit lobes
  // sit on a ring of radius ringR with each lobe radius subR; choose so lobes
  // touch into a closed wall without overlapping too hard.
  const ringR = size * 0.62;
  const subR = size * 0.46;
  // squat assembly: membrane-spanning, slightly taller than wide
  const height = size * 1.7;

  const geo = useMemo(
    () => (sub ? buildSubunitLathe(sub, subR, height) : null),
    [sub, subR, height],
  );

  const angles = useMemo(
    () => Array.from({ length: n }, (_, i) => (i / n) * Math.PI * 2),
    [n],
  );

  if (!geo || !sub) {
    // defensive: degrade to a clustered node if the signature is malformed
    return (
      <mesh>
        <icosahedronGeometry args={[size * 1.05, 1]} />
        {children}
      </mesh>
    );
  }

  return (
    <group>
      {angles.map((a, i) => (
        <mesh
          key={i}
          geometry={geo}
          position={[Math.cos(a) * ringR, 0, Math.sin(a) * ringR]}
        >
          {children}
        </mesh>
      ))}
      {/* open central pore down the channel axis (BackSide dark lumen) */}
      <mesh>
        <cylinderGeometry
          args={[size * 0.2, size * 0.2, height * 1.04, 24, 1, true]}
        />
        <meshStandardMaterial
          color="#0b1f17"
          transparent
          opacity={0.6}
          side={THREE.BackSide}
          roughness={0.9}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}
