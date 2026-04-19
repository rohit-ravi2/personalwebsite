#!/usr/bin/env python3
"""Phase 2b — synthetic reference trajectory for imitation learning.

The Phase 2c controller learns to drive the MuJoCo body so its segment
positions match a reference kinematic pattern. Ideally that reference
comes from real C. elegans tracking video (Atanas 2023 /
OpenWorm Movement DB / WormPose etc.), but with those sources
inaccessible right now we generate a biologically parameterized
reference from the published crawling literature:

  - Wave frequency ~1 Hz (Fang-Yen 2010)
  - Wavelength ~0.65 body lengths (Boyle, Berri & Cohen 2012)
  - Peak body curvature consistent with unbiased forward crawling on
    agar (~0.3 rad per segment hinge)
  - Forward speed emerges from resistive-force-theory coupling

The reference lives in the same sim-unit frame as the MuJoCo body so
the imitation reward is MSE of segment positions in sim-m. When a real
pose dataset becomes available we drop in a different producer with
the same schema and the training loop is unchanged.

Output: public/data/wormbody-reference-trace.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "public" / "data" / "wormbody-reference-trace.json"

# Match the MuJoCo body geometry exactly.
NUM_SEGMENTS = 20
SEGMENT_LENGTH_M = 5.0e-2  # = scripts/build_wormbody.SEGMENT_LENGTH_M

# Published C. elegans crawling parameters (forward crawl on agar).
FREQ_HZ = 1.0
WAVELENGTH_BODY = 0.65
CURVATURE_AMPLITUDE_RAD = 0.35
FORWARD_SPEED_BODY_PER_SEC = 0.22  # ~220 µm/s → 0.22 body-lengths/sec

DURATION_S = 6.0
RECORD_HZ = 60


def compute_reference() -> list[dict]:
    frames: list[dict] = []
    n_frames = int(round(DURATION_S * RECORD_HZ))
    body_length = NUM_SEGMENTS * SEGMENT_LENGTH_M
    for frame_idx in range(n_frames):
        t = frame_idx / RECORD_HZ
        # Head moves steadily along +x world axis. Worm body trails to
        # +x as segments are generated from head rearward.
        head_x = -FORWARD_SPEED_BODY_PER_SEC * body_length * t
        head_y = 0.0
        heading = math.pi  # body extends along -x from head
        positions: list[list[float]] = []
        x, y, h = head_x, head_y, heading
        positions.append([x, y])
        for j in range(1, NUM_SEGMENTS):
            s = (j - 0.5) / (NUM_SEGMENTS - 1)  # fractional position along body
            phase = 2 * math.pi * (s / WAVELENGTH_BODY - FREQ_HZ * t)
            # curvature = d heading / d arclength
            seg_curvature = (CURVATURE_AMPLITUDE_RAD * 2 * math.pi / WAVELENGTH_BODY) * math.cos(phase)
            h += seg_curvature / (NUM_SEGMENTS - 1)
            x += SEGMENT_LENGTH_M * math.cos(h)
            y += SEGMENT_LENGTH_M * math.sin(h)
            positions.append([x, y])
        frames.append({"t": round(t, 4), "positions": positions})
    return frames


def main() -> None:
    frames = compute_reference()
    # Re-center so the first-frame centroid is at origin (matches the
    # sim-trace convention so rewards are on comparable coordinate frames).
    import numpy as np
    p0 = np.array(frames[0]["positions"])
    c0 = p0.mean(axis=0)
    for f in frames:
        for i in range(len(f["positions"])):
            f["positions"][i] = [
                f["positions"][i][0] - float(c0[0]),
                f["positions"][i][1] - float(c0[1]),
            ]

    meta = {
        "num_segments": NUM_SEGMENTS,
        "num_frames": len(frames),
        "record_hz": RECORD_HZ,
        "duration_sec": DURATION_S,
        "source": "synthetic (biologically parameterised)",
        "parameters": {
            "freq_hz": FREQ_HZ,
            "wavelength_body": WAVELENGTH_BODY,
            "curvature_amplitude_rad": CURVATURE_AMPLITUDE_RAD,
            "forward_speed_body_per_sec": FORWARD_SPEED_BODY_PER_SEC,
        },
        "references": [
            "Fang-Yen et al. 2010, PNAS — body-wave mechanics during crawling",
            "Boyle, Berri & Cohen 2012, Front. Comp. Neurosci. — neuromechanical model parameters",
        ],
        "units": "MJCF scaled: 1 sim-m ≈ 1 real-mm",
    }
    payload = {"meta": meta, "frames": frames}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(
        f"wrote {OUT.name}: {len(frames)} frames @ {RECORD_HZ} Hz, "
        f"{NUM_SEGMENTS} segments, {size_kb:.1f} KB"
    )


if __name__ == "__main__":
    main()
