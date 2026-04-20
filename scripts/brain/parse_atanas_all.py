#!/usr/bin/env python3
"""Parse all 10 Atanas 2023 worms into compact per-worm .npz artifacts.

Extends `parse_atanas.py` (single-worm) to iterate every NWB file
under `data/external/atanas2023/`, skipping ones already parsed. Each
worm writes `artifacts/atanas_worm_NN.npz` with the same schema as
the worm_01 output.

The per-worm NeuroPAL-identified neuron subset varies (that's why
each worm needs its own .npz). Downstream harness code handles
per-worm alignment to the Cook connectome.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np


NWB_DIR = Path(
    "/home/rohit/Desktop/website/personalwebsite/data/external/atanas2023"
)
ART = Path(__file__).resolve().parent / "artifacts"


def _decode_labels(ds: h5py.Dataset) -> np.ndarray:
    raw = ds[:]
    return np.array(
        [s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
         for s in raw], dtype=object,
    )


def parse_one(nwb_path: Path, out_path: Path) -> dict:
    """Parse a single Atanas NWB file. Returns a summary dict."""
    with h5py.File(nwb_path, "r") as f:
        sig = f["/processing/CalciumActivity/SignalCalciumImResponseSeries"]
        neural = sig["data"][:].astype(np.float32)
        t_neural = sig["timestamps"][:].astype(np.float32)

        raw = f["/processing/CalciumActivity/SignalRawFluor/"
                "SignalCalciumImResponseSeries"]
        neural_raw = raw["data"][:].astype(np.float32)

        labels_ds = f["/processing/CalciumActivity/NeuronIDs/labels"]
        neuron_ids = _decode_labels(labels_ds)

        bh_paths = {
            "velocity":  "/processing/Behavior/velocity/velocity",
            "head_curv": "/processing/Behavior/head_curvature/head_curvature",
            "body_curv": "/processing/Behavior/body_curvature/body_curvature",
            "ang_vel":   "/processing/Behavior/angular_velocity/angular_velocity",
            "reversal":  "/processing/Behavior/reversal_events/reversal_events",
            "pumping":   "/processing/Behavior/pumping/pumping",
        }
        behavior = {}
        for key, path in bh_paths.items():
            if path in f:
                g = f[path]
                data = g["data"][:].astype(np.float32)
                if data.shape[0] == neural.shape[0]:
                    behavior[key] = data

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        neural=neural,
        neural_raw=neural_raw,
        neuron_ids=neuron_ids,
        t=t_neural,
        **behavior,
    )
    named = sum(1 for s in neuron_ids if s not in ("-", "", "nan")
                and not str(s).startswith("?"))
    return {
        "file": nwb_path.name,
        "T": neural.shape[0],
        "N_rois": neural.shape[1],
        "N_named": named,
        "duration_min": (t_neural[-1] - t_neural[0]) / 60,
        "behavior_keys": list(behavior.keys()),
        "out_size_kb": out_path.stat().st_size / 1024,
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    nwbs = sorted(NWB_DIR.glob("atanas_worm_*.nwb"))
    print(f"Found {len(nwbs)} NWB files")

    summary = []
    for nwb in nwbs:
        # Extract index from filename: atanas_worm_NN[_sub-xxx].nwb
        m = re.match(r"atanas_worm_(\d+)", nwb.stem)
        if not m:
            print(f"  skip {nwb.name} (unrecognised name)")
            continue
        n = int(m.group(1))
        out = ART / f"atanas_worm_{n:02d}.npz"
        if out.exists():
            print(f"  [skip] {out.name} already exists "
                  f"({out.stat().st_size/1024:.0f} KB)")
            continue
        print(f"  parsing {nwb.name} → {out.name} "
              f"({nwb.stat().st_size/1e9:.1f} GB)…")
        info = parse_one(nwb, out)
        info["worm_index"] = n
        summary.append(info)
        print(f"    T={info['T']}  ROIs={info['N_rois']}  "
              f"named={info['N_named']}  "
              f"dur={info['duration_min']:.1f} min  "
              f"out={info['out_size_kb']:.0f} KB")

    print(f"\n=== Parsed {len(summary)} new worms ===")
    for s in summary:
        print(f"  worm_{s['worm_index']:02d}: {s['N_named']}/{s['N_rois']} "
              f"named, {s['duration_min']:.1f} min")


if __name__ == "__main__":
    main()
