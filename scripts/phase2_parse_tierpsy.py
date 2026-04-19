#!/usr/bin/env python3
"""Convert Tierpsy `MANUAL_FEATS_skeletons.hdf5` centerlines into our
reference trajectory JSON schema — the same format
`scripts/phase2_reference.py` produces synthetically. The PPO imitation
loop in `scripts/phase2_train_imitation.py` will consume either
interchangeably.

Input: data/external/wormpose/tierpsy_test_data/data/MANUAL_FEATS/
       Results/MANUAL_FEATS_skeletons.hdf5
Output: public/data/wormbody-reference-real.json
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = (
    REPO
    / "data"
    / "external"
    / "wormpose"
    / "tierpsy_test_data"
    / "data"
    / "MANUAL_FEATS"
    / "Results"
    / "MANUAL_FEATS_skeletons.hdf5"
)
OUT = REPO / "public" / "data" / "wormbody-reference-real.json"

NUM_SEGMENTS = 20
CLIP_DURATION_S = 6.0
TARGET_FRAME_HZ = 60
SIM_M_PER_BODY = 1.0  # our MJCF body spans 1 sim-m (20 segments × 0.05 m)


def _resample_centerline(pts_49: np.ndarray) -> np.ndarray:
    """Resample a 49-point centerline to NUM_SEGMENTS points by
    arclength-uniform interpolation. Returns shape (NUM_SEGMENTS, 2)."""
    diffs = np.diff(pts_49, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-6:
        return np.tile(pts_49[0], (NUM_SEGMENTS, 1))
    target_s = np.linspace(0.0, total, NUM_SEGMENTS)
    x_i = np.interp(target_s, cum, pts_49[:, 0])
    y_i = np.interp(target_s, cum, pts_49[:, 1])
    return np.stack([x_i, y_i], axis=1)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Tierpsy source missing: {SRC}")
    with h5py.File(SRC, "r") as f:
        skeletons = f["skeleton"][:]  # (N_frames_all, 49, 2)
        traj = f["trajectories_data"][:]
        # Tierpsy stores one long table of frames across all tracked worms in
        # the recording; worm_index_joined separates individuals, frame_number
        # is the per-video frame index, and has_skeleton flags fits we trust.
        is_good = traj["has_skeleton"].astype(bool)
        worm_idx = traj["worm_index_joined"]
        frame_num = traj["frame_number"]

        unique_worms, counts = np.unique(worm_idx, return_counts=True)
        print(f"Found {len(unique_worms)} tracked worms; sample counts {counts[:8]}")

        # Pick the longest uninterrupted good streak. That's the cleanest
        # clip for an imitation target.
        best_worm = None
        best_run = 0
        best_start = 0
        for w in unique_worms:
            mask = (worm_idx == w) & is_good
            if mask.sum() < best_run:
                continue
            idxs = np.where(mask)[0]
            if idxs.size < 30:
                continue
            # Longest run of contiguous frame_num
            fn = frame_num[idxs]
            breaks = np.where(np.diff(fn) != 1)[0]
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks + 1, [len(idxs)]])
            for s, e in zip(starts, ends):
                run_len = e - s
                if run_len > best_run:
                    best_run = run_len
                    best_worm = w
                    best_start = idxs[s]
        assert best_worm is not None, "no usable track"
        print(f"Best streak: worm {best_worm}, length {best_run} frames, start row {best_start}")

        # Extract the streak
        streak_rows = np.arange(best_start, best_start + best_run)
        sk = skeletons[streak_rows]  # (best_run, 49, 2)
        # Source frame rate per Tierpsy convention: check timestamp if avail
        if "timestamp" in f and "time" in f["timestamp"]:
            ts = f["timestamp/time"][:]
            src_hz = 1.0 / np.median(np.diff(ts))
        else:
            src_hz = 30.0
        print(f"Source frame rate ≈ {src_hz:.1f} Hz")

    # Take first CLIP_DURATION_S seconds of the streak (or all, whichever smaller)
    max_src_frames = int(CLIP_DURATION_S * src_hz)
    clip = sk[:max_src_frames]
    print(f"Clip: {clip.shape[0]} source frames ({clip.shape[0] / src_hz:.2f} s)")

    # Resample time axis to TARGET_FRAME_HZ
    n_target = int(CLIP_DURATION_S * TARGET_FRAME_HZ)
    src_times = np.arange(clip.shape[0]) / src_hz
    tgt_times = np.linspace(0.0, clip.shape[0] / src_hz, n_target)

    # Resample centerlines: for each target frame, find two source frames
    # bracketing it and linearly interpolate skeleton positions, then
    # arclength-resample to NUM_SEGMENTS points.
    frames: list[dict] = []
    for tf, tt in enumerate(tgt_times):
        # Nearest source frame
        src_idx = int(np.clip(tt * src_hz, 0, clip.shape[0] - 1))
        pts_49 = clip[src_idx]
        # Drop any NaN points (Tierpsy tags unresolved segments with NaN)
        finite_mask = np.isfinite(pts_49).all(axis=1)
        if finite_mask.sum() < 10:
            # Skip dead frames; carry forward prior skeleton if available.
            if frames:
                frames.append({"t": round(float(tt), 4), "positions": frames[-1]["positions"]})
            continue
        pts_49_clean = pts_49[finite_mask]
        # Resample to 20 points by arclength
        pts_20 = _resample_centerline(pts_49_clean)
        # Tierpsy pixel coordinates → our sim-m scale. Normalise so body length = 1 sim-m.
        # Use median body length from the clip as the scale factor.
        seg_diffs = np.diff(pts_20, axis=0)
        body_len_px = float(np.sum(np.linalg.norm(seg_diffs, axis=1)))
        if body_len_px < 1e-3:
            continue
        scale = SIM_M_PER_BODY / body_len_px
        pts_scaled = pts_20 * scale
        frames.append({
            "t": round(float(tt), 4),
            "positions": [[float(x), float(y)] for x, y in pts_scaled],
        })

    # Re-center: subtract the first frame's centroid so trajectories start at origin.
    p0 = np.array(frames[0]["positions"])
    c0 = p0.mean(axis=0)
    for fr in frames:
        for i, xy in enumerate(fr["positions"]):
            fr["positions"][i] = [float(xy[0] - c0[0]), float(xy[1] - c0[1])]

    meta = {
        "num_segments": NUM_SEGMENTS,
        "num_frames": len(frames),
        "record_hz": TARGET_FRAME_HZ,
        "duration_sec": CLIP_DURATION_S,
        "source": "Tierpsy test data (Zenodo 3837679) — MANUAL_FEATS_skeletons.hdf5",
        "source_worm_index": int(best_worm),
        "source_frame_rate_hz": float(src_hz),
        "units": "MJCF scaled: 1 sim-m ≈ 1 body length",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.name}: {len(frames)} frames @ {TARGET_FRAME_HZ} Hz, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
