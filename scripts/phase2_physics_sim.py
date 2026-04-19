#!/usr/bin/env python3
"""Phase 2a — MuJoCo physics sim of the wormbody with a tuned CPG controller.

Loads public/data/wormbody.xml (the artifact from Phase 1a), drives the
19 hinge joints with a central-pattern-generator-style traveling wave,
and applies per-segment anisotropic drag from low-Reynolds resistive
force theory (c_perp ≈ 2·c_parallel). The asymmetric drag converts the
lateral wave into net forward propulsion — this is the physics that
makes worms crawl on agar / swim in viscous fluid.

Output: public/data/wormbody-physics-trace.json, played back by the
WormBody React component.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[1]
MJCF = REPO / "public" / "data" / "wormbody.xml"
OUT = REPO / "public" / "data" / "wormbody-physics-trace.json"

SIM_DURATION_S = 6.0
RECORD_HZ = 60
CPG_FREQ_HZ = 1.3
CPG_WAVELENGTH = 0.85
CPG_AMPLITUDE_RAD = 0.4     # target joint angle amplitude ∈ joint range [±0.5]

# Resistive-force-theory coefficients (units: N·s/m per segment).
# Ratio c_perp / c_para ≈ 2 is the canonical value for slender bodies
# in viscous media, and is the source of the propulsive asymmetry.
DRAG_PARA = 2.0
DRAG_PERP = 4.0


def cpg_controls(model: mujoco.MjModel, t: float) -> np.ndarray:
    """Traveling-wave target joint angles (position actuators)."""
    nu = model.nu
    ctrl = np.zeros(nu, dtype=np.float64)
    for j in range(nu):
        s = (j + 0.5) / nu
        phase = 2 * math.pi * (s / CPG_WAVELENGTH - CPG_FREQ_HZ * t)
        ctrl[j] = CPG_AMPLITUDE_RAD * math.sin(phase)
    return ctrl


def _apply_resistive_drag(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apply anisotropic drag to each body segment.

    For each segment, decompose the segment's linear velocity into
    components parallel and perpendicular to its body axis, then
    apply drag forces: F = -c_para · v_para - c_perp · v_perp.
    Forces are set via data.xfrc_applied (world-frame Cartesian forces
    + torques on each body).
    """
    # Segment bodies are indices 1..nbody-1.
    for bid in range(1, model.nbody):
        # The segment's x-axis in world frame == its rotation matrix column 0.
        xmat = data.xmat[bid].reshape(3, 3)
        axis = xmat[:, 0]  # body-axis direction in world frame

        # Linear velocity of the segment's CoM (world frame).
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        v_lin = vel[3:6]

        v_para = np.dot(v_lin, axis) * axis
        v_perp = v_lin - v_para

        force = -DRAG_PARA * v_para - DRAG_PERP * v_perp
        # Apply to body CoM; xfrc_applied layout: [fx, fy, fz, tx, ty, tz].
        data.xfrc_applied[bid, 0:3] = force
        data.xfrc_applied[bid, 3:6] = 0.0


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MJCF.as_posix())
    data = mujoco.MjData(model)

    print(
        f"Loaded {MJCF.name}: {model.nbody} bodies, {model.njnt} joints, "
        f"{model.nu} actuators, timestep {model.opt.timestep*1000:.2f} ms"
    )

    # Settle briefly under gravity so the body rests on the substrate.
    for _ in range(200):
        _apply_resistive_drag(model, data)
        mujoco.mj_step(model, data)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    sim_dt = model.opt.timestep
    total_steps = int(round(SIM_DURATION_S / sim_dt))
    record_stride = max(1, int(round(1.0 / RECORD_HZ / sim_dt)))

    frames: list[dict] = []
    seg_body_ids = list(range(1, model.nbody))

    for step in range(total_steps):
        data.ctrl[:] = cpg_controls(model, data.time)
        _apply_resistive_drag(model, data)
        mujoco.mj_step(model, data)

        if step % record_stride == 0:
            positions = []
            for bid in seg_body_ids:
                p = data.xpos[bid]
                positions.append([float(p[0]), float(p[1])])
            frames.append({
                "t": round(float(data.time), 4),
                "positions": positions,
            })

    # Normalise so the trajectory starts at the origin.
    p0 = np.array(frames[0]["positions"])
    centroid0 = p0.mean(axis=0)
    for frame in frames:
        for i, xy in enumerate(frame["positions"]):
            frame["positions"][i] = [xy[0] - centroid0[0], xy[1] - centroid0[1]]

    meta = {
        "num_segments": model.nbody - 1,
        "num_frames": len(frames),
        "record_hz": RECORD_HZ,
        "duration_sec": SIM_DURATION_S,
        "controller": {
            "kind": "cpg",
            "freq_hz": CPG_FREQ_HZ,
            "wavelength_body": CPG_WAVELENGTH,
            "amplitude_rad": CPG_AMPLITUDE_RAD,
        },
        "drag": {
            "kind": "resistive_force_theory",
            "c_para": DRAG_PARA,
            "c_perp": DRAG_PERP,
            "anisotropy": DRAG_PERP / DRAG_PARA,
        },
        "units": "MJCF scaled: 1 sim-m ≈ 1 real-mm of worm body",
    }
    payload = {"meta": meta, "frames": frames}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(
        f"wrote {OUT.name}: {len(frames)} frames @ {RECORD_HZ} Hz, "
        f"{model.nbody - 1} segments, {size_kb:.1f} KB"
    )

    p_first = np.array(frames[0]["positions"]).mean(axis=0)
    p_last = np.array(frames[-1]["positions"]).mean(axis=0)
    displacement = np.linalg.norm(p_last - p_first)
    print(
        f"CoM displacement over {SIM_DURATION_S}s: {displacement:.4f} sim-m "
        f"(avg {displacement / SIM_DURATION_S:.4f} m/s)"
    )


if __name__ == "__main__":
    main()
