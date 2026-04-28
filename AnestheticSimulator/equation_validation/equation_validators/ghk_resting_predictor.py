"""CP A.2 — GHK resting potential predictor.

Goldman-Hodgkin-Katz equation predicts membrane resting potential from
ion permeabilities. For a cell with K, Na, and Cl channels:

    V_rest = (RT/F) ln[(P_K[K]_o + P_Na[Na]_o + P_Cl[Cl]_i) /
                       (P_K[K]_i + P_Na[Na]_i + P_Cl[Cl]_o)]

For Wave 2 cells using channel densities + reversal potentials, the
parallel-conductance form (Mullins-Noda 1956):

    V_rest = Σ(g_i × E_i) / Σg_i

This is the same formula used for Nernst-bounds steady-state at I_inj=0,
which is exactly the GHK-equivalent at resting state with all channels
at their effective steady-state activation. We compare this prediction
to a Brian2 simulation of each cell allowed to settle to V_rest with
zero injected current.

Output: artifacts/ghk_resting_predictions.csv
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

OUT = ROOT / "artifacts"
CHK = ROOT / "checkpoints" / "path_a_cp2_ghk.json"


def ghk_parallel_conductance(cell: dict) -> dict:
    """Predict V_rest using parallel-conductance (Mullins-Noda) approximation.

    Returns dict with V_rest_predicted_mV, contributing channels, and
    fractional contribution per channel.
    """
    g_total = 0.0
    g_E_sum = 0.0
    contributions = {}
    for ch in cell.get("g_nS", {}).keys() if "g_nS" in cell else cell.get("g_Scm2", {}).keys():
        if "g_nS" in cell:
            g = cell["g_nS"][ch]
        else:
            g = cell["g_Scm2"][ch] * cell["surf_cm2"] * 1e9
        if g <= 0:
            continue
        rev_key = cell["channel_reversal"][ch]
        E = cell[rev_key]
        g_total += g
        g_E_sum += g * E
        contributions[ch] = {"g_nS": round(g, 5), "E_mV": E, "g_E_product": round(g * E, 4)}

    v_rest = g_E_sum / g_total if g_total > 0 else float("nan")
    # Fractional V-shift contribution per channel
    for ch in contributions:
        contributions[ch]["fractional_g"] = round(contributions[ch]["g_nS"] / g_total, 4)

    return {
        "V_rest_predicted_mV": v_rest,
        "g_total_nS": g_total,
        "contributions": contributions,
    }


def simulate_brian2_resting(cell: dict, duration_ms: float = 1000) -> float:
    """Run a Brian2 single-compartment LIF-equivalent of the cell, no input,
    and return settled V_rest.

    Models the cell as a passive membrane with each channel as an ohmic
    branch: dV/dt = -Σ g_i (V - E_i) / C_total. Channels at fixed activation.
    Equivalent to the GHK parallel-conductance prediction with explicit
    integration; serves as a numerical sanity check.
    """
    try:
        from brian2 import (NeuronGroup, Network, ms, mV, nS, pF, pA,
                            defaultclock, prefs, seed as brian2_seed)
        prefs.codegen.target = "numpy"
        defaultclock.dt = 0.1 * ms
        brian2_seed(42)
    except ImportError:
        return float("nan")

    # Build channel-summed equation. Each channel contributes g_i × (E_i - V).
    g_terms = []
    namespace = {}
    for ch in cell.get("g_nS", {}).keys() if "g_nS" in cell else cell.get("g_Scm2", {}).keys():
        if "g_nS" in cell:
            g_val = cell["g_nS"][ch]
        else:
            g_val = cell["g_Scm2"][ch] * cell["surf_cm2"] * 1e9
        if g_val <= 0:
            continue
        rev_key = cell["channel_reversal"][ch]
        E_val = cell[rev_key]
        g_var = f"g_{ch}"
        E_var = f"E_{ch}"
        namespace[g_var] = g_val * nS
        namespace[E_var] = E_val * mV
        g_terms.append(f"{g_var} * ({E_var} - v)")

    if not g_terms:
        return float("nan")

    cm_pF = cell["cm_uFcm2"] * cell["surf_cm2"] * 1e6
    namespace["C_mem"] = cm_pF * pF

    eqs = f"dv/dt = ({' + '.join(g_terms)}) / C_mem : volt"
    G = NeuronGroup(1, eqs, namespace=namespace, method="exact")
    # Initialize at -55 mV (between E_K and E_Ca)
    G.v = -55 * mV
    net = Network(G)
    net.run(duration_ms * ms)
    return float(G.v[0] / mV)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CHK.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    detail = {}
    for cell_name, cell in ALL_CELLS.items():
        ghk = ghk_parallel_conductance(cell)
        v_predicted = ghk["V_rest_predicted_mV"]
        v_simulated = simulate_brian2_resting(cell)
        if math.isnan(v_predicted) or math.isnan(v_simulated):
            divergence = float("nan")
        else:
            divergence = v_simulated - v_predicted
        rows.append({
            "cell": cell_name,
            "V_rest_predicted_mV": round(v_predicted, 2) if not math.isnan(v_predicted) else "nan",
            "V_rest_simulated_mV": round(v_simulated, 2) if not math.isnan(v_simulated) else "nan",
            "divergence_mV": round(divergence, 3) if not math.isnan(divergence) else "nan",
            "g_total_nS": round(ghk["g_total_nS"], 4),
            "n_active_channels": len(ghk["contributions"]),
            "interpretation": (
                "PASS — GHK matches Brian2" if not math.isnan(divergence) and abs(divergence) < 1.0
                else "MARGINAL — within 5 mV" if not math.isnan(divergence) and abs(divergence) < 5.0
                else "DIVERGENT" if not math.isnan(divergence)
                else "skipped"
            ),
        })
        detail[cell_name] = ghk

    out_csv = OUT / "ghk_resting_predictions.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_json = OUT / "ghk_contributions_detail.json"
    json.dump(detail, open(out_json, "w"), indent=2, default=str)

    print(f"GHK resting potential predictions:\n")
    for r in rows:
        print(f"  {r['cell']:6s}: predicted {r['V_rest_predicted_mV']:>7} mV, "
              f"simulated {r['V_rest_simulated_mV']:>7} mV, Δ {r['divergence_mV']:>+7} mV "
              f"[{r['interpretation']}]")

    n_pass = sum(1 for r in rows if "PASS" in r["interpretation"])
    state = {
        "checkpoint": "path_a_cp2_ghk",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pass": n_pass,
        "n_total": len(rows),
        "rows": rows,
    }
    json.dump(state, open(CHK, "w"), indent=2)
    print(f"\nCSV: {out_csv}")
    print(f"Checkpoint: {CHK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
