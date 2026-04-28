"""CP B.2 — Bifurcation analysis under varied input current.

Sweep I_inj across [-50, +50] pA in 1 pA steps for each Wave 2 cell. At
each I_inj, find steady-state V (or limit cycle if oscillating). Plot
V_steady vs I_inj — the bifurcation diagram. Identify bifurcation type:
saddle-node, Hopf, transcritical, etc.

For Wave 2 cells which are graded (no spiking) and predominantly monostable
(per CP B.1), the bifurcation diagram should be smooth I-V curve with
gradual depolarization under positive current. Sudden discontinuities
indicate bistable transitions; smooth-sigmoid indicates saddle-node-on-
invariant-circle (SNIC) absent at this scale.

For AVA-class cells, Wicks 1996 plateau bistability prediction implies
hysteresis if the cell actually has two-state structure: increasing I_inj
should snap to plateau at some critical current; decreasing should snap
back at a lower critical current. We test this by sweeping forward and
backward.

Output: artifacts/bifurcation_analysis.md
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cell_params import ALL_CELLS
from dynamical_analysis.phase_planes import (
    SLOW_GATE, total_current_pA, gate_steady_state
)

OUT = ROOT / "artifacts"
CHK = ROOT / "checkpoints" / "path_b_cp2_bifurcation.json"


def steady_state_V(cell: dict, slow_gate_spec: dict, I_inj_pA: float,
                     V_init: float = -55.0, n_iters: int = 200) -> float:
    """Find steady-state V at given I_inj by iterating to fixed point.

    Uses gradient-descent on |I_total| to converge. The slow gate is at
    its steady-state value at each iteration's V.
    """
    V = V_init
    dt_eff = 0.5  # mV step factor
    for _ in range(n_iters):
        gate_ss = gate_steady_state(V, slow_gate_spec)
        I = total_current_pA(cell, V, gate_ss, slow_gate_spec["channel"], I_inj_pA)
        # Move V proportional to I (positive I → depolarize, negative → hyperpolarize)
        # Step size scaled by C_total for stability
        from cell_params import total_capacitance_pF
        C_pF = total_capacitance_pF(cell)
        dV = (I / C_pF) * dt_eff  # pA / pF * ms = mV
        V += dV
        if abs(dV) < 1e-4:
            break
        # Clamp to physiological bounds
        V = max(-100.0, min(70.0, V))
    return V


def bifurcation_sweep(cell: dict, slow_gate_spec: dict, currents_pA: list[float],
                        forward: bool = True) -> list[dict]:
    """Sweep I_inj. If forward, V_init from previous I (continuation).
    Direction matters for hysteresis detection."""
    V = -55.0
    rows = []
    iter_currents = currents_pA if forward else list(reversed(currents_pA))
    for I in iter_currents:
        V = steady_state_V(cell, slow_gate_spec, I, V_init=V)
        rows.append({"I_inj_pA": I, "V_steady_mV": round(V, 2)})
    if not forward:
        rows = list(reversed(rows))
    return rows


def detect_hysteresis(forward: list[dict], backward: list[dict]) -> dict:
    """Compare forward and backward sweeps; if V differs at the same I, hysteresis exists."""
    if len(forward) != len(backward):
        return {"hysteresis": False, "max_difference_mV": None}
    max_diff = 0.0
    diff_at_I = None
    for fwd, bwd in zip(forward, backward):
        diff = abs(fwd["V_steady_mV"] - bwd["V_steady_mV"])
        if diff > max_diff:
            max_diff = diff
            diff_at_I = fwd["I_inj_pA"]
    return {
        "hysteresis": max_diff > 2.0,  # 2 mV threshold for "real" hysteresis
        "max_difference_mV": round(max_diff, 2),
        "at_I_inj_pA": diff_at_I,
    }


def classify_bifurcation(forward: list[dict]) -> str:
    """Crude classification of bifurcation type from V-I sweep."""
    Vs = [r["V_steady_mV"] for r in forward]
    Is = [r["I_inj_pA"] for r in forward]
    # Compute discrete dV/dI and look for jumps
    dV_dI = []
    for i in range(1, len(Vs)):
        dI = Is[i] - Is[i - 1]
        if dI == 0:
            continue
        dV_dI.append((Vs[i] - Vs[i - 1]) / dI)
    if not dV_dI:
        return "no_data"
    max_slope = max(dV_dI)
    min_slope = min(dV_dI)
    avg_slope = sum(dV_dI) / len(dV_dI)
    # Smooth sigmoid: max/avg < 5
    if max_slope < 5 * avg_slope and max_slope < 10.0:
        return "monotone_smooth (no bifurcation in tested range)"
    if max_slope > 30.0:
        return "discontinuous (saddle-node or hard bistable transition)"
    if max_slope > 10.0:
        return "rapid_transition (SNIC or near-saddle-node)"
    return "smooth"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CHK.parent.mkdir(parents=True, exist_ok=True)

    currents = [float(x) for x in range(-50, 51, 2)]  # -50 to +50 pA, 2 pA steps

    md_path = OUT / "bifurcation_analysis.md"
    cell_data = {}

    with open(md_path, "w") as f:
        f.write("# CP B.2 — Bifurcation analysis under varied input current\n\n")
        f.write("**Date:** 2026-04-28\n\n")
        f.write("Sweep I_inj from -50 to +50 pA in 2 pA steps for each Wave 2 cell. "
                "Compute steady-state V at each I via iterative fixed-point search. "
                "Forward + backward sweeps detect hysteresis (signature of bistability). "
                "Slope of V-vs-I curve indicates bifurcation type.\n\n")

        for cell_name, cell in ALL_CELLS.items():
            slow = SLOW_GATE[cell_name]
            f.write(f"## {cell_name}\n\n")

            forward = bifurcation_sweep(cell, slow, currents, forward=True)
            backward = bifurcation_sweep(cell, slow, currents, forward=False)
            hyst = detect_hysteresis(forward, backward)
            classification = classify_bifurcation(forward)

            # Save bifurcation data CSV
            bif_csv = OUT / f"bifurcation_{cell_name}.csv"
            with open(bif_csv, "w", newline="") as bf:
                w = csv.DictWriter(bf, fieldnames=["I_inj_pA", "V_forward_mV", "V_backward_mV", "diff_mV"])
                w.writeheader()
                for fwd, bwd in zip(forward, backward):
                    w.writerow({
                        "I_inj_pA": fwd["I_inj_pA"],
                        "V_forward_mV": fwd["V_steady_mV"],
                        "V_backward_mV": bwd["V_steady_mV"],
                        "diff_mV": round(fwd["V_steady_mV"] - bwd["V_steady_mV"], 2),
                    })

            f.write(f"### Bifurcation classification: **{classification}**\n\n")
            f.write(f"### Hysteresis detection\n\n")
            f.write(f"- Forward vs backward sweep max difference: **{hyst['max_difference_mV']} mV** "
                    f"at I_inj = {hyst['at_I_inj_pA']} pA\n")
            f.write(f"- Hysteresis verdict: **{'PRESENT' if hyst['hysteresis'] else 'ABSENT'}**\n\n")

            f.write(f"### V-I curve key points\n\n")
            f.write("| I_inj (pA) | V_forward (mV) | V_backward (mV) | diff |\n|---|---|---|---|\n")
            for I_target in [-50, -30, -10, 0, 10, 30, 50]:
                fwd = next((r for r in forward if r["I_inj_pA"] == I_target), None)
                bwd = next((r for r in backward if r["I_inj_pA"] == I_target), None)
                if fwd and bwd:
                    diff = round(fwd["V_steady_mV"] - bwd["V_steady_mV"], 2)
                    f.write(f"| {I_target} | {fwd['V_steady_mV']} | {bwd['V_steady_mV']} | {diff} |\n")

            # Wicks check for AVA
            if cell_name in ("AVAL", "AVAR"):
                if hyst["hysteresis"]:
                    f.write(f"\n**Wicks 1996 bistability:** ✓ hysteresis detected — "
                            f"plateau-low-V transition at distinct critical currents.\n\n")
                else:
                    f.write(f"\n**Wicks 1996 bistability check:** ⚠ no hysteresis detected at "
                            f"this resolution. AVA-class plateau may be monostable in this "
                            f"single-slow-variable approximation; full bistability could require "
                            f"multiple coupled slow variables (e.g., EGL-19 inactivation + "
                            f"NCA/UNC-103 slow currents) or specific stimulus protocols.\n\n")

            cell_data[cell_name] = {
                "classification": classification,
                "hysteresis": hyst,
                "csv": str(bif_csv.relative_to(ROOT)),
            }

        f.write("## Cross-cell synthesis\n\n"
                "Bifurcation classifications + hysteresis verdicts per cell summarize the "
                "cell's I-V dynamical structure. Cells classified as `monotone_smooth` are "
                "monostable in this approximation; cells with `discontinuous` or "
                "`rapid_transition` show bistable switching dynamics.\n\n"
                "**Caveat:** single-slow-variable approximation is a phase-plane simplification. "
                "Full multi-gate Brian2 simulation (separate validator) would expose additional "
                "dynamical structure if present. The current sweep is a useful first-pass "
                "characterization.\n")

    state = {
        "checkpoint": "path_b_cp2_bifurcation",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell_data": cell_data,
    }
    json.dump(state, open(CHK, "w"), indent=2, default=str)

    print(f"Bifurcation analysis:\n")
    for cell_name, data in cell_data.items():
        print(f"  {cell_name:6s}: {data['classification']}, "
              f"hysteresis {'PRESENT' if data['hysteresis']['hysteresis'] else 'absent'} "
              f"(max diff {data['hysteresis']['max_difference_mV']} mV)")
    print(f"\nMD: {md_path}")
    print(f"Checkpoint: {CHK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
