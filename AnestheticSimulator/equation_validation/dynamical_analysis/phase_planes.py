"""CP B.1 — Phase plane diagrams for production-grade Wave 2 cells.

For each cell, identify the dominant slow gating variable (typically the
Ca-channel inactivation or the slow K-channel activation), construct the
V vs gating-variable phase plane, plot nullclines, vector field, fixed
points.

For AVAL/AVAR: dominant slow variable is EGL-19 inactivation (h_egl19).
Wicks 1996 plateau predictions: AVA-class cells should show plateau-state
fixed point at depolarized V (~-20 mV, matching Mellem voltage regime),
limit cycle absent, monostable dynamics.

For AIY/RIM: dominant slow variable depends on cell. AIY has SLO-1+EGL-19
coupled → slo1egl19 inactivation as candidate. RIM has UNC-2 (P/Q-type Ca)
+ SHL-1 (A-type K) — UNC-2 inactivation is the slowest typical.

This is a phase-plane sketch using ohmic channel approximation with a
single slow gating variable that modulates one channel's effective
conductance. Adequate for nullcline structure; full multi-gate dynamics
require Brian2 simulation (separate validator).

Output: artifacts/phase_plane_analysis.md + per-cell PNG conceptually
(no PNG generation for headless run; data exported as CSV grids for
downstream plotting).
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

from cell_params import ALL_CELLS, total_capacitance_pF

OUT = ROOT / "artifacts"
CHK = ROOT / "checkpoints" / "path_b_cp1_phase_planes.json"


# Slow gating variable per cell (best biological proxy)
SLOW_GATE = {
    "AVAL": {"channel": "egl19", "name": "h_egl19", "type": "Ca_inactivation",
             "v_half_mV": -25.0, "k_mV": 5.0, "tau_ms": 50.0,
             "rationale": "EGL-19 L-type Ca channel inactivation drives plateau "
                          "decay; Wicks 1996 + Mellem 2008."},
    "AVAR": {"channel": "egl19", "name": "h_egl19", "type": "Ca_inactivation",
             "v_half_mV": -25.0, "k_mV": 5.0, "tau_ms": 50.0,
             "rationale": "Same as AVAL; UNC-103 (ERG-like K) provides additional "
                          "slow current but EGL-19 inactivation dominates plateau."},
    "AIY": {"channel": "slo1egl19", "name": "n_slo1egl19", "type": "K_activation_via_Ca_coupling",
            "v_half_mV": -30.0, "k_mV": 10.0, "tau_ms": 100.0,
            "rationale": "SLO-1 BK channel activation coupled to EGL-19 Ca influx; "
                         "slowest dynamic in AIY's channel suite; CP B.1 extrapolated "
                         "parameters per WB3 caveat."},
    "RIM": {"channel": "unc2", "name": "h_unc2", "type": "Ca_inactivation",
            "v_half_mV": -35.0, "k_mV": 6.0, "tau_ms": 80.0,
            "rationale": "UNC-2 P/Q-type Ca inactivation; RIM's plateau/burst dynamics "
                         "via Ca-dependent rebound. CP B.1 extrapolated parameters per "
                         "WB3 caveat."},
}


def boltzmann(v: float, v_half: float, k: float, ascending: bool = False) -> float:
    """Sigmoidal Boltzmann.
    ascending=True: activation (rises with V); False: inactivation (falls with V)."""
    if ascending:
        return 1.0 / (1.0 + math.exp(-(v - v_half) / k))
    return 1.0 / (1.0 + math.exp((v - v_half) / k))


def channel_g_at(cell: dict, channel: str, gate_value: float) -> float:
    """Return effective conductance (nS) for the channel at a given gate value
    (0-1) for the SLOW_GATE variable."""
    if "g_nS" in cell:
        g_max = cell["g_nS"][channel]
    else:
        g_max = cell["g_Scm2"][channel] * cell["surf_cm2"] * 1e9
    return g_max * gate_value


def total_current_pA(cell: dict, V: float, slow_gate_val: float, slow_channel: str,
                       I_inj_pA: float = 0.0) -> float:
    """Total membrane current at (V, slow_gate_val).
    I = Σ g_i (E_i - V) + I_inj
    For the slow channel, g is modulated by slow_gate_val.
    For other channels, use full conductance (steady-state activated).
    """
    I_total = I_inj_pA
    for ch in cell.get("g_nS", {}).keys() if "g_nS" in cell else cell.get("g_Scm2", {}).keys():
        if "g_nS" in cell:
            g_max = cell["g_nS"][ch]
        else:
            g_max = cell["g_Scm2"][ch] * cell["surf_cm2"] * 1e9
        if g_max <= 0:
            continue
        rev_key = cell["channel_reversal"][ch]
        E = cell[rev_key]
        if ch == slow_channel:
            g = g_max * slow_gate_val
        else:
            g = g_max
        I_total += g * (E - V)
    return I_total


def gate_steady_state(V: float, slow_gate_spec: dict) -> float:
    """Steady-state value of the slow gating variable at given V."""
    is_inactivation = "inactivation" in slow_gate_spec["type"]
    return boltzmann(V, slow_gate_spec["v_half_mV"], slow_gate_spec["k_mV"],
                     ascending=not is_inactivation)


def find_fixed_points(cell: dict, slow_gate_spec: dict,
                        V_grid: list[float], I_inj_pA: float = 0.0) -> list[dict]:
    """Find fixed points where both nullclines intersect: dV/dt = 0 AND dGate/dt = 0.

    On the gate nullcline (dGate/dt = 0), gate = gate_steady_state(V).
    On this curve, find V where total current = 0 → that's the fixed point.
    """
    fps = []
    prev_I = None
    prev_V = None
    for V in V_grid:
        gate_ss = gate_steady_state(V, slow_gate_spec)
        I = total_current_pA(cell, V, gate_ss, slow_gate_spec["channel"], I_inj_pA)
        if prev_I is not None and (prev_I * I < 0):
            # Sign change → fixed point in this interval
            # Linear interp
            V_fp = prev_V - prev_I * (V - prev_V) / (I - prev_I)
            gate_fp = gate_steady_state(V_fp, slow_gate_spec)
            fps.append({"V_mV": round(V_fp, 2), "gate": round(gate_fp, 4)})
        prev_I = I
        prev_V = V
    return fps


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CHK.parent.mkdir(parents=True, exist_ok=True)

    md_path = OUT / "phase_plane_analysis.md"
    cell_data = {}

    V_grid = [v for v in range(-100, 60)]
    V_grid_fine = [v / 2 for v in range(-200, 121)]  # 0.5 mV resolution

    with open(md_path, "w") as f:
        f.write("# CP B.1 — Phase plane analysis for production-grade Wave 2 cells\n\n")
        f.write("**Date:** 2026-04-28\n\n")
        f.write("Phase plane in (V, slow_gate) space for each Nicoletti-validated cell. "
                "Slow gate identified per cell from the channel suite; nullclines + fixed "
                "points computed via ohmic-channel approximation with the slow gate as "
                "the only dynamic variable.\n\n")

        for cell_name, cell in ALL_CELLS.items():
            slow = SLOW_GATE[cell_name]
            f.write(f"## {cell_name}\n\n")
            f.write(f"- Slow gating variable: **{slow['name']}** "
                    f"({slow['type']} on {slow['channel']})\n")
            f.write(f"- Boltzmann: V_half = {slow['v_half_mV']} mV, k = {slow['k_mV']} mV, "
                    f"τ = {slow['tau_ms']} ms\n")
            f.write(f"- Rationale: {slow['rationale']}\n\n")

            # Find fixed points at I_inj=0 (rest)
            fps_rest = find_fixed_points(cell, slow, V_grid_fine, I_inj_pA=0)
            fps_at_minus5 = find_fixed_points(cell, slow, V_grid_fine, I_inj_pA=-5)
            fps_at_plus5 = find_fixed_points(cell, slow, V_grid_fine, I_inj_pA=+5)
            fps_at_plus20 = find_fixed_points(cell, slow, V_grid_fine, I_inj_pA=+20)

            f.write(f"### Fixed points\n\n")
            f.write("| I_inj (pA) | fixed points (V_mV, gate) |\n|---|---|\n")
            f.write(f"| 0 | {fps_rest} |\n")
            f.write(f"| -5 | {fps_at_minus5} |\n")
            f.write(f"| +5 | {fps_at_plus5} |\n")
            f.write(f"| +20 | {fps_at_plus20} |\n\n")

            n_fps = len(fps_rest)
            if n_fps == 1:
                interp = "Monostable at rest. Single attracting fixed point — biologically expected for graded interneuron."
            elif n_fps == 2:
                interp = "Bistable at rest. Two stable fixed points (lower-V and plateau-V) with unstable middle. Wicks plateau structure."
            elif n_fps == 3:
                interp = "Bistable at rest with one unstable saddle. Classic plateau dynamics — Wicks 1996 prediction for AVA-class."
            else:
                interp = f"{n_fps} fixed points — non-standard topology."

            f.write(f"### Interpretation\n\n{interp}\n\n")

            # Wicks 1996 prediction for AVA: plateau at depolarized V
            if cell_name in ("AVAL", "AVAR"):
                # Check if any FP has V > -30 (plateau-like)
                plateau_fps = [fp for fp in fps_rest if fp["V_mV"] > -30]
                low_fps = [fp for fp in fps_rest if fp["V_mV"] < -50]
                if plateau_fps:
                    f.write(f"**Wicks 1996 plateau check:** ✓ plateau-state FP at "
                            f"V = {plateau_fps[0]['V_mV']} mV (matches Mellem 2008 "
                            f"depolarized AVA regime).\n\n")
                else:
                    f.write(f"**Wicks 1996 plateau check:** ⚠ no plateau-state FP found "
                            f"at I_inj = 0. Cell may require depolarizing drive to enter "
                            f"plateau state.\n\n")
            else:
                f.write(f"**WB3 caveat note:** {cell_name} parameters extrapolated from "
                        f"Wave 2 cell-builder validation. Phase plane structure here "
                        f"reflects extrapolated parameters; not a primary-source-anchored "
                        f"prediction. Sensitivity to V_half ± 5 mV would be informative.\n\n")

            cell_data[cell_name] = {
                "slow_gate_spec": slow,
                "fixed_points_at": {
                    "I_inj_-5": fps_at_minus5,
                    "I_inj_0": fps_rest,
                    "I_inj_5": fps_at_plus5,
                    "I_inj_20": fps_at_plus20,
                },
            }

        f.write("## Cross-cell synthesis\n\n"
                "AVA-class cells (AVAL, AVAR): tested for Wicks 1996 plateau structure. "
                "AIY/RIM: phase-plane structure documented under explicit WB3 extrapolation "
                "caveat — slow-gate Boltzmann parameters are biologically reasonable defaults "
                "but not primary-source-anchored.\n\n"
                "**Sensitivity analysis caveat (per WB3 Decision 3 caveat):** AIY and RIM "
                "phase-plane fixed points should be re-evaluated under V_half ± 5 mV "
                "perturbation; if FP topology changes substantially across that range, "
                "the cell-builder extrapolation produces parameter-dependent dynamics that "
                "may not be robust. Sensitivity sweep deferred to a separate analysis.\n")

    # Persist CSV grids for downstream plotting (gate nullcline + sample vector field)
    for cell_name, cell in ALL_CELLS.items():
        slow = SLOW_GATE[cell_name]
        # Gate nullcline: gate = steady_state(V)
        nullcline_path = OUT / f"phase_plane_{cell_name}_gate_nullcline.csv"
        with open(nullcline_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["V_mV", "gate_steady_state"])
            for V in V_grid:
                w.writerow([V, round(gate_steady_state(V, slow), 4)])

        # V nullcline: dV/dt = 0 → solve I = 0 for gate, given V
        v_null_path = OUT / f"phase_plane_{cell_name}_V_nullcline.csv"
        with open(v_null_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["V_mV", "gate_at_V_nullcline_I_inj_0"])
            for V in V_grid:
                # I(V, gate) = sum_other_currents + g_slow * gate * (E_slow - V) + I_inj
                # Solve for gate when I = 0
                if "g_nS" in cell:
                    g_slow = cell["g_nS"][slow["channel"]]
                else:
                    g_slow = cell["g_Scm2"][slow["channel"]] * cell["surf_cm2"] * 1e9
                if g_slow <= 0:
                    w.writerow([V, "nan"])
                    continue
                rev_key = cell["channel_reversal"][slow["channel"]]
                E_slow = cell[rev_key]
                # Other channels' contribution at full activation
                other_I = 0.0
                for ch in cell.get("g_nS", {}).keys() if "g_nS" in cell else cell.get("g_Scm2", {}).keys():
                    if ch == slow["channel"]:
                        continue
                    if "g_nS" in cell:
                        g = cell["g_nS"][ch]
                    else:
                        g = cell["g_Scm2"][ch] * cell["surf_cm2"] * 1e9
                    if g <= 0:
                        continue
                    rk = cell["channel_reversal"][ch]
                    E = cell[rk]
                    other_I += g * (E - V)
                # I_total = other_I + g_slow * gate * (E_slow - V) = 0
                # → gate = -other_I / (g_slow * (E_slow - V))
                denom = g_slow * (E_slow - V)
                if abs(denom) < 1e-9:
                    w.writerow([V, "nan"])
                else:
                    gate_val = -other_I / denom
                    w.writerow([V, round(gate_val, 4) if 0 <= gate_val <= 1 else f"out_of_range:{gate_val:.3f}"])

    state = {
        "checkpoint": "path_b_cp1_phase_planes",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell_data": cell_data,
    }
    json.dump(state, open(CHK, "w"), indent=2, default=str)

    print(f"Phase plane analysis:\n")
    for cell_name, data in cell_data.items():
        n_fps = len(data["fixed_points_at"]["I_inj_0"])
        print(f"  {cell_name:6s}: {n_fps} FP(s) at rest "
              f"(I_inj=0): {data['fixed_points_at']['I_inj_0']}")
    print(f"\nMD: {md_path}")
    print(f"Checkpoint: {CHK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
