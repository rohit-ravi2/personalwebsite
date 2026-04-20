#!/usr/bin/env python3
"""Phase 3a step 4 — extract paired (neural, behavior) from an Atanas
2023 NWB file.

Atanas et al. 2023 (DANDI 000776) records GCaMP7f whole-brain imaging
(~1.4 Hz volumes) + NIR behavior video + NeuroPAL-based identification
of ~40/60 neurons per worm (in this worm, 134 ROIs get confident IDs).
The NWB schema is ndx-multichannel-volume + standard /processing
Behavior containers.

What we take for Phase 3b fitting:
  - /processing/CalciumActivity/SignalCalciumImResponseSeries/data
        (T_neural, N_neural)  ΔF/F signal per identified neuron
  - /processing/CalciumActivity/NeuronIDs/labels
        (N_neural,)  NeuroPAL names — may contain "-" for unidentified
  - /processing/CalciumActivity/SignalCalciumImResponseSeries/timestamps
        (T_neural,)  seconds since session start
  - /processing/Behavior/{velocity,head_curvature,body_curvature,
        angular_velocity,reversal_events}/…/data
        (T_neural,)  already time-aligned to neural samples

Output: scripts/brain/artifacts/atanas_worm_01.npz with arrays
  - neural       (T, N)    float32  ΔF/F signal
  - neural_raw   (T, N)    float32  raw fluorescence
  - neuron_ids   (N,)      str      NeuroPAL labels
  - t            (T,)      float32  seconds
  - velocity     (T,)      float32  body velocity
  - head_curv    (T,)      float32
  - body_curv    (T,)      float32
  - ang_vel      (T,)      float32
  - reversal     (T,)      float32  reversal-event indicator / probability

Run:
    /home/rohit/miniconda3/envs/ml/bin/python \\
        scripts/brain/parse_atanas.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


NWB = Path(
    "/home/rohit/Desktop/website/personalwebsite/data/external/atanas2023/"
    "atanas_worm_01.nwb"
)
OUT = Path(__file__).resolve().parent / "artifacts" / "atanas_worm_01.npz"


def _decode_labels(ds: h5py.Dataset) -> np.ndarray:
    """NWB stores object-dtype string arrays; decode to plain Python str."""
    raw = ds[:]
    return np.array([
        s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
        for s in raw
    ], dtype=object)


def main() -> None:
    if not NWB.exists():
        raise SystemExit(f"NWB missing: {NWB}")
    print(f"Parsing {NWB.name} ({NWB.stat().st_size / 1e9:.1f} GB)")

    with h5py.File(NWB, "r") as f:
        # Neural — ΔF/F (processed signal)
        sig = f["/processing/CalciumActivity/SignalCalciumImResponseSeries"]
        neural = sig["data"][:].astype(np.float32)            # (T, N)
        t_neural = sig["timestamps"][:].astype(np.float32)    # (T,)
        roi_ids = sig["rois"][:].astype(np.int64)             # (N,)

        # Raw fluorescence (for reference / optional ΔF/F recomputation)
        raw = f["/processing/CalciumActivity/SignalRawFluor/"
                "SignalCalciumImResponseSeries"]
        neural_raw = raw["data"][:].astype(np.float32)

        # Neuron IDs — NeuroPAL-confirmed labels per ROI (may be "-")
        labels_ds = f["/processing/CalciumActivity/NeuronIDs/labels"]
        neuron_ids = _decode_labels(labels_ds)

        # Behavior — already aligned to neural timestamps (both 1600 samples)
        bh_paths = {
            "velocity":   "/processing/Behavior/velocity/velocity",
            "head_curv":  "/processing/Behavior/head_curvature/head_curvature",
            "body_curv":  "/processing/Behavior/body_curvature/body_curvature",
            "ang_vel":    "/processing/Behavior/angular_velocity/angular_velocity",
            "reversal":   "/processing/Behavior/reversal_events/reversal_events",
            "pumping":    "/processing/Behavior/pumping/pumping",
        }
        behavior = {}
        bh_shape_warning = []
        for key, path in bh_paths.items():
            g = f[path]
            data = g["data"][:].astype(np.float32)
            ts = g["timestamps"][:].astype(np.float32)
            if data.shape[0] == neural.shape[0]:
                behavior[key] = data
                # sanity: timestamps should match
                if not np.allclose(ts, t_neural, atol=1e-2):
                    bh_shape_warning.append(f"{key}: timestamps diverge")
            else:
                bh_shape_warning.append(
                    f"{key}: shape {data.shape} != neural T={neural.shape[0]}; skipped"
                )

    # Report
    print(f"\nNeural trace:   {neural.shape}  (T_frames, N_neurons)")
    print(f"  timespan:     {t_neural[0]:.1f}–{t_neural[-1]:.1f} s "
          f"({(t_neural[-1] - t_neural[0]) / 60:.1f} min)")
    sample_dt = float(np.median(np.diff(t_neural)))
    print(f"  sample dt:    {sample_dt*1000:.0f} ms (~{1/sample_dt:.2f} Hz)")
    print(f"  value range:  [{neural.min():.3f}, {neural.max():.3f}]")

    # Count named vs unidentified
    named_mask = np.array([s not in ("-", "", "nan") and not s.startswith("?")
                           for s in neuron_ids])
    print(f"\nNeuron IDs:     {len(neuron_ids)} ROIs")
    print(f"  identified:   {named_mask.sum()}")
    print(f"  unidentified: {(~named_mask).sum()}")
    print(f"  sample names: {[str(s) for s in neuron_ids[named_mask][:12]]}")

    # Cross-reference with our 300-neuron connectome
    conn = np.load(
        Path(__file__).resolve().parent / "artifacts" / "connectome.npz",
        allow_pickle=True,
    )
    conn_names = set(str(s) for s in conn["names"])
    id_set = set(str(s) for s in neuron_ids[named_mask])
    overlap = id_set & conn_names
    print(f"\nConnectome intersection:")
    print(f"  IDs in Cook 2019 connectome:  {len(overlap)}")
    print(f"  IDs NOT in connectome:        "
          f"{sorted(id_set - conn_names)[:10]}")

    # Behavior
    print(f"\nBehavior features:")
    for k, v in behavior.items():
        print(f"  {k:<10} shape={v.shape}  "
              f"range=[{v.min():.3f}, {v.max():.3f}]  "
              f"mean={v.mean():.3f}")
    if bh_shape_warning:
        print("  WARN:", *bh_shape_warning, sep="\n  ")

    # Save compact .npz
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        neural=neural,
        neural_raw=neural_raw,
        neuron_ids=neuron_ids,
        t=t_neural,
        **behavior,
    )
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
