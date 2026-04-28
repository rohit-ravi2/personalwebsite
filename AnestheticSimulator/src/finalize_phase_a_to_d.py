"""Wave P consolidation — run scan_pose -> phase_c -> phase_d on whatever
docking results currently exist, and write a single milestone summary.

Idempotent: safe to run multiple times. Each invocation overwrites prior
intermediate CSVs based on the current state of `artifacts/binding/poses/`.

Usage:
    conda activate wave-p-docking
    python src/finalize_phase_a_to_d.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "WAVE_P_PHASE_ABCD_MILESTONE.md"


def run(cmd: list[str], label: str) -> str:
    print(f"\n=== {label} ===")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr else "")
    print(out)
    return out


def main() -> int:
    py = sys.executable
    out_scan = run([py, "src/scan_pose_affinities.py"], "scan_pose_affinities")
    out_c = run([py, "src/phase_c_occupancy.py"], "phase_c_occupancy")
    out_d = run([py, "src/phase_d_kinetic_shifts.py"], "phase_d_kinetic_shifts")

    poses = sorted((ROOT / "artifacts" / "binding" / "poses").glob("*_out.pdbqt"))
    n_poses = len(poses)
    occupancy = ROOT / "artifacts" / "occupancy" / "best_pocket_per_target.csv"
    n_occ = sum(1 for _ in open(occupancy)) - 1 if occupancy.exists() else 0
    kinetics = ROOT / "artifacts" / "kinetics" / "kinetic_shifts_at_1xEC50.csv"
    n_kin = sum(1 for _ in open(kinetics)) - 1 if kinetics.exists() else 0

    # Pull Gate C.1 result text from out_c
    gate_c1 = "UNKNOWN"
    for line in out_c.splitlines():
        if "PASS — multi-target framing supported" in line:
            gate_c1 = "PASS"
        elif "FAIL — single-target framing implied" in line:
            gate_c1 = "FAIL"

    SUMMARY.write_text(
        f"""# Wave P — Phases A+B+C+D consolidated milestone

## Pipeline state

- Vina dockings completed: {n_poses}
- (anesthetic, target) occupancy rows: {n_occ}
- Kinetic-shift rows for Wave 2 perturbation: {n_kin}

## Gate C.1 — multi-target framing falsifiability check

Verdict: **{gate_c1}**

(See `artifacts/occupancy/gate_c1_summary.md` for the engaged-target list and
top-15 (anesthetic, target) occupancy ranking.)

## Drop-in artifacts for Wave 2 perturbation runs

- `artifacts/kinetics/wave2_overlay.json` — by_anesthetic → by_target →
  parameter shifts (g_max factor, τ_decay factor, n_Ca delta, rate factor)
  with evidence-grade tags (LITERATURE / ANALOGY / CONSERVATIVE / DEFERRED).
- `artifacts/occupancy/occupancy_matrix.csv` — wide-form gene × anesthetic
  occupancy at 1× clinical EC50.
- `artifacts/occupancy/best_pocket_per_target.csv` — long form with pocket id,
  predicted Kd, and per-dose occupancy at 0.5×/1×/2×/5× EC50.

## Per-stage stdout snapshots (last invocation)

### scan_pose_affinities

```
{out_scan[-2000:]}
```

### phase_c_occupancy

```
{out_c[-3000:]}
```

### phase_d_kinetic_shifts

```
{out_d[-2500:]}
```
"""
    )
    print(f"\nMilestone summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
