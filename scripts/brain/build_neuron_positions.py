#!/usr/bin/env python3
"""Phase T1c step 1 — extract per-neuron 3D soma positions from
OpenWorm morphology data for volume-transmission calculations.

OpenWorm's c302 parsed morphologies include 3D coordinates for every
neuron. We take the median of all segment coordinates per neuron as
a proxy for soma position (more stable than taking the first
"proximal" point which may be a process tip).

Output: artifacts/neuron_positions.npz with:
  names:     (N,)  neuron names (in connectome.npz order)
  positions: (N, 3) float32, (x, y, z) in µm
  valid:     (N,)  bool, True where position is known
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

MORPH_CSV = Path("/home/rohit/Desktop/C-Elegans/simulation/parsed/"
                 "out_morphology_segments.csv")
CONN = Path(__file__).resolve().parent / "artifacts" / "connectome.npz"
OUT = Path(__file__).resolve().parent / "artifacts" / "neuron_positions.npz"


def main():
    conn = np.load(CONN, allow_pickle=True)
    conn_names = [str(s) for s in conn["names"]]
    N = len(conn_names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(MORPH_CSV)
    print(f"Loaded {len(df)} morphology segments for "
          f"{df['Neuron'].nunique()} unique neurons")

    # Per-neuron median (x, y, z) — more robust to outlier segment tips
    # than using "proximal"-only points.
    med = df.groupby("Neuron")[["x", "y", "z"]].median()

    positions = np.full((N, 3), np.nan, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    matched = 0
    for i, name in enumerate(conn_names):
        if name in med.index:
            positions[i] = med.loc[name].values
            valid[i] = True
            matched += 1

    print(f"Matched {matched}/{N} connectome neurons to positions")

    # Fill in missing positions with interpolation from the centroid
    # of matched neurons — so volume-transmission code doesn't crash
    # on missing data; modulation affects them but at large distances.
    centroid = np.nanmedian(positions[valid], axis=0)
    positions[~valid] = centroid
    print(f"  centroid (fallback for missing): "
          f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) µm")

    # Basic sanity checks on key neurons
    for n in ["AVAL", "AVAR", "RIS", "NSML", "PDEL", "ASHL"]:
        if n in conn_names:
            idx = conn_names.index(n)
            p = positions[idx]
            print(f"  {n}: ({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f}) "
                  f"{'(inferred)' if not valid[idx] else ''}")

    np.savez_compressed(
        OUT,
        names=np.array(conn_names, dtype=object),
        positions=positions.astype(np.float32),
        valid=valid,
    )
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
