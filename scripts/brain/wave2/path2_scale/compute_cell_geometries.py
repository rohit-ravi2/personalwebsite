"""
Compute per-cell surface area + volume from NeuroML morphology segments.

Source: simulation/parsed/out_morphology_segments.csv (Cook/Witvliet-derived,
parsed by c302). Each cell has segments with (proximal, distal) endpoints
and diameter.

For each segment we treat it as a truncated cone:
  segment_length = ||P_dist - P_prox||
  surface_area   = π · (r_prox + r_dist) · √((r_prox - r_dist)² + L²)
  volume         = (π · L / 3) · (r_prox² + r_prox·r_dist + r_dist²)

Sum across all segments to get whole-cell surface area + volume.

Each segment is paired: proximal row followed by distal row. We pair them
by sequential order within a neuron (which is how the c302 parser emits
them).

Output: per-cell {surf_cm2, vol_L, total_segs} written to
cell_morphology_data.py for runtime import.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

SRC = Path("/mnt/ssd4tb/Desktop/C-Elegans/simulation/parsed/out_morphology_segments.csv")
OUT = Path(__file__).resolve().parent / "cell_morphology_data.py"


def main():
    rows_by_neuron = {}
    with SRC.open() as f:
        r = csv.DictReader(f)
        for row in r:
            n = row["Neuron"]
            rows_by_neuron.setdefault(n, []).append(row)

    print(f"Parsing {len(rows_by_neuron)} neurons...")
    cells = {}
    for neuron, rows in rows_by_neuron.items():
        # Pair proximal/distal sequentially
        surf_um2 = 0.0
        vol_um3 = 0.0
        n_segs = 0
        i = 0
        while i < len(rows) - 1:
            a, b = rows[i], rows[i+1]
            if not (a["Tag"].endswith("proximal") and b["Tag"].endswith("distal")):
                i += 1
                continue
            try:
                xa, ya, za, da = float(a["x"]), float(a["y"]), float(a["z"]), float(a["diameter"])
                xb, yb, zb, db = float(b["x"]), float(b["y"]), float(b["z"]), float(b["diameter"])
            except ValueError:
                i += 2
                continue
            L = math.sqrt((xb-xa)**2 + (yb-ya)**2 + (zb-za)**2)
            ra, rb = da/2, db/2
            slant = math.sqrt((ra-rb)**2 + L**2)
            surf_um2 += math.pi * (ra + rb) * slant
            vol_um3 += (math.pi * L / 3.0) * (ra*ra + ra*rb + rb*rb)
            n_segs += 1
            i += 2
        # Surface area in cm²: 1 μm² = 1e-8 cm²
        surf_cm2 = surf_um2 * 1e-8
        # Volume in L: 1 μm³ = 1e-15 L
        vol_L = vol_um3 * 1e-15
        cells[neuron] = {
            "surf_cm2": surf_cm2,
            "surf_um2": surf_um2,
            "vol_L": vol_L,
            "n_segments": n_segs,
        }

    # Stats
    surfs_um2 = [c["surf_um2"] for c in cells.values()]
    print(f"\nSurface area distribution (μm²):")
    print(f"  min {min(surfs_um2):.1f}, max {max(surfs_um2):.1f}, "
          f"mean {sum(surfs_um2)/len(surfs_um2):.1f}, "
          f"median {sorted(surfs_um2)[len(surfs_um2)//2]:.1f}")

    # Compare to old default
    print(f"\nOld default: 100 μm² (Layer 1)")
    print(f"Real range:  {min(surfs_um2):.0f} – {max(surfs_um2):.0f} μm²")

    # Compare to Nicoletti cells (AVAL etc.)
    for n in ["AVAL", "AVAR", "AIYL", "AIYR", "RIML", "RIMR", "HSNL", "RIBL"]:
        if n in cells:
            c = cells[n]
            print(f"  {n}: {c['surf_um2']:.1f} μm² ({c['n_segments']} segs)")

    # Write output module
    with OUT.open("w") as f:
        f.write('"""Per-cell morphology data — auto-generated from\n')
        f.write('simulation/parsed/out_morphology_segments.csv (c302 parser).\n')
        f.write('Surface area in cm², volume in liters.\n')
        f.write('"""\n')
        f.write('from __future__ import annotations\n\n')
        f.write('CELL_MORPHOLOGY: dict[str, dict[str, float]] = {\n')
        for n in sorted(cells):
            c = cells[n]
            f.write(f"    {n!r}: {{'surf_cm2': {c['surf_cm2']:.6e}, "
                    f"'surf_um2': {c['surf_um2']:.2f}, "
                    f"'vol_L': {c['vol_L']:.6e}, "
                    f"'n_segments': {c['n_segments']}}},\n")
        f.write('}\n')
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
