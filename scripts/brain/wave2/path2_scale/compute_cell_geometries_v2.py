"""
Parse full morphology from .cell.nml files in simulation/cells/.

Each cell file has full morphology (soma + axon segments) with proximal and
distal endpoints + diameters. We sum truncated-cone surface area and volume
across all segments.

This supersedes compute_cell_geometries.py which used the partial CSV.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

CELLS_DIR = Path("/mnt/ssd4tb/Desktop/C-Elegans/simulation/cells")
OUT = Path(__file__).resolve().parent / "cell_morphology_data.py"


def parse_cell(path: Path) -> dict:
    """Return {surf_cm2, surf_um2, vol_L, n_segs} for a cell's .nml file."""
    text = path.read_text()
    # Strip namespace for easier parsing
    text = re.sub(r'\sxmlns="[^"]+"', '', text)
    root = ET.fromstring(text)

    # NeuroML structure: <morphology><segment id="..."><proximal x= y= z= diameter=/>
    # <distal x= y= z= diameter=/></segment></morphology>
    # parent segments inherit the distal of parent as their proximal — handle this.
    segments = {}  # id -> (proximal, distal, parent_id)
    for seg in root.iter("segment"):
        seg_id = int(seg.get("id"))
        parent_elem = seg.find("parent")
        parent_id = int(parent_elem.get("segment")) if parent_elem is not None else None
        prox_elem = seg.find("proximal")
        dist_elem = seg.find("distal")
        dist = (float(dist_elem.get("x")), float(dist_elem.get("y")),
                float(dist_elem.get("z")), float(dist_elem.get("diameter")))
        if prox_elem is not None:
            prox = (float(prox_elem.get("x")), float(prox_elem.get("y")),
                    float(prox_elem.get("z")), float(prox_elem.get("diameter")))
        else:
            prox = None
        segments[seg_id] = {"prox": prox, "dist": dist, "parent": parent_id}

    # Resolve proximal-via-parent
    for sid, s in segments.items():
        if s["prox"] is None and s["parent"] is not None:
            parent = segments.get(s["parent"])
            if parent is not None:
                s["prox"] = parent["dist"]

    surf_um2 = 0.0
    vol_um3 = 0.0
    n_segs = 0
    for sid, s in segments.items():
        if s["prox"] is None:
            continue
        xa, ya, za, da = s["prox"]
        xb, yb, zb, db = s["dist"]
        L = math.sqrt((xb-xa)**2 + (yb-ya)**2 + (zb-za)**2)
        ra, rb = da/2, db/2
        if L < 1e-6 and abs(ra - rb) < 1e-6:
            # degenerate point segment — surface = sphere of ra
            surf_um2 += 4 * math.pi * ra * ra
            vol_um3 += (4.0/3.0) * math.pi * ra**3
        else:
            slant = math.sqrt((ra-rb)**2 + L**2)
            surf_um2 += math.pi * (ra + rb) * slant
            vol_um3 += (math.pi * L / 3.0) * (ra*ra + ra*rb + rb*rb)
        n_segs += 1

    return {
        "surf_cm2": surf_um2 * 1e-8,
        "surf_um2": surf_um2,
        "vol_L": vol_um3 * 1e-15,
        "n_segments": n_segs,
    }


def main():
    # Use the non-_D variant (full cell, not just soma)
    cell_files = sorted([p for p in CELLS_DIR.glob("*.cell.nml")
                         if "_D.cell.nml" not in p.name])
    print(f"Parsing {len(cell_files)} cell files...")

    cells = {}
    for path in cell_files:
        neuron = path.stem.split(".")[0]
        try:
            cells[neuron] = parse_cell(path)
        except Exception as e:
            print(f"  FAILED {neuron}: {e}")

    surfs = [c["surf_um2"] for c in cells.values()]
    n_segs = [c["n_segments"] for c in cells.values()]
    print(f"\nParsed {len(cells)} cells")
    print(f"Surface area μm²: min {min(surfs):.0f}, max {max(surfs):.0f}, "
          f"mean {sum(surfs)/len(surfs):.0f}, "
          f"median {sorted(surfs)[len(surfs)//2]:.0f}")
    print(f"Segments per cell: min {min(n_segs)}, max {max(n_segs)}, "
          f"mean {sum(n_segs)/len(n_segs):.0f}")

    print(f"\nNicoletti cells (compare):")
    print(f"  AVAL Nicoletti: 1123 μm² (from C=9.66pF/0.86μF/cm²)")
    for n in ["AVAL", "AVAR", "AIYL", "AIYR", "RIML", "RIMR", "HSNL", "RIBL",
              "RIPL", "VD1", "ASEL", "AWCL"]:
        if n in cells:
            c = cells[n]
            print(f"  {n}: {c['surf_um2']:.0f} μm² ({c['n_segments']} segs)")

    with OUT.open("w") as f:
        f.write('"""Per-cell morphology from simulation/cells/*.cell.nml.\n')
        f.write('Full cell incl. axon. Surface area in cm², volume in liters.\n')
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
