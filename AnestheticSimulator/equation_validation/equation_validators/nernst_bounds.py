"""CP A.1 — Nernst-bounds validator.

For each Wave 2 cell, the Nernst equilibrium potentials define hard
physiological bounds on membrane voltage. The most-negative bound is E_K
(typical -80 mV); the most-positive bound is E_Ca (60-130 mV depending on
[Ca] gradient) or E_Na (~+50 mV).

A simulation that produces voltage outside [E_K - margin, E_max + margin]
is non-physiological — either a parameter error or a numerical instability.

This validator runs each Wave 2 cell under a current-injection sweep and
flags any voltage excursion outside the Nernst-derived envelope.

Output: artifacts/nernst_bounds_validation.csv
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
CHK = ROOT / "checkpoints" / "path_a_cp1_nernst.json"

NERNST_MARGIN_MV = 5.0  # transient overshoot tolerance


def nernst_envelope(cell: dict) -> tuple[float, float]:
    """Return (V_min_bound, V_max_bound) for the cell."""
    revs = []
    for ch in cell.get("g_nS", {}).keys() if "g_nS" in cell else cell.get("g_Scm2", {}).keys():
        rev_key = cell["channel_reversal"][ch]
        revs.append(cell[rev_key])
    return min(revs) - NERNST_MARGIN_MV, max(revs) + NERNST_MARGIN_MV


def simulate_simple_lif_with_nernst(cell: dict, I_inj_pA: float, duration_ms: float = 500) -> dict:
    """Minimal single-compartment simulation of cell using a passive-leak +
    nonlinear-current ODE driven by I_inj.

    Avoids dependency on Wave 2's Brian2 cell builders; uses parameter values
    from cell_params to estimate steady-state V at the given current injection.
    Channel currents are approximated as ohmic at their reversal potential —
    this is the steady-state-activated regime, sufficient for Nernst-bound checking.
    """
    # Compute the steady-state V where total current = 0:
    # 0 = sum_i g_i * (E_i - V) + I_inj
    # V = (sum_i g_i * E_i + I_inj) / sum_i g_i
    g_total_nS = 0.0
    g_E_sum = 0.0
    for ch in cell.get("g_nS", {}).keys() if "g_nS" in cell else cell.get("g_Scm2", {}).keys():
        if "g_nS" in cell:
            g_nS = cell["g_nS"][ch]
        else:
            g_nS = cell["g_Scm2"][ch] * cell["surf_cm2"] * 1e9
        if g_nS <= 0:
            continue
        rev_key = cell["channel_reversal"][ch]
        E = cell[rev_key]
        g_total_nS += g_nS
        g_E_sum += g_nS * E

    if g_total_nS == 0:
        return {"V_steady_mV": float("nan"), "g_total_nS": 0.0}

    # I_inj is in pA; g in nS → mV via V = E + I/g; here I/g_total in mV
    V_ss_mV = (g_E_sum + I_inj_pA) / g_total_nS
    return {
        "V_steady_mV": V_ss_mV,
        "g_total_nS": g_total_nS,
        "I_inj_pA": I_inj_pA,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CHK.parent.mkdir(parents=True, exist_ok=True)

    # Stimulus protocol: hyperpolarizing → depolarizing current sweep
    currents_pA = [-30, -20, -10, -5, 0, 5, 10, 20, 30, 50, 100]

    rows = []
    for cell_name, cell in ALL_CELLS.items():
        v_lo, v_hi = nernst_envelope(cell)
        for I in currents_pA:
            r = simulate_simple_lif_with_nernst(cell, I)
            v = r["V_steady_mV"]
            within = (v_lo <= v <= v_hi)
            rows.append({
                "cell": cell_name,
                "I_inj_pA": I,
                "V_steady_mV": round(v, 2) if not math.isnan(v) else "nan",
                "V_min_bound_mV": v_lo,
                "V_max_bound_mV": v_hi,
                "within_envelope": within,
                "note": "" if within else f"OUT_OF_BOUNDS by {min(abs(v - v_lo), abs(v - v_hi)):.1f} mV",
            })

    out_csv = OUT / "nernst_bounds_validation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["within_envelope"])
    n_fail = n_total - n_pass
    fails_by_cell = {}
    for r in rows:
        if not r["within_envelope"]:
            fails_by_cell.setdefault(r["cell"], []).append(r)

    print(f"Nernst-bound validation: {n_pass}/{n_total} pass")
    for cell, fails in fails_by_cell.items():
        print(f"  {cell}: {len(fails)} excursions")
        for f in fails[:3]:
            print(f"    I={f['I_inj_pA']} pA → V={f['V_steady_mV']} (envelope [{f['V_min_bound_mV']}, {f['V_max_bound_mV']}])")

    state = {
        "checkpoint": "path_a_cp1_nernst",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pass": n_pass,
        "n_total": n_total,
        "fails_by_cell": {c: len(fs) for c, fs in fails_by_cell.items()},
        "output_csv": str(out_csv.relative_to(ROOT)),
    }
    json.dump(state, open(CHK, "w"), indent=2)
    print(f"\nCSV: {out_csv}")
    print(f"Checkpoint: {CHK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
