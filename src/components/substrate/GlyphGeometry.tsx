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
