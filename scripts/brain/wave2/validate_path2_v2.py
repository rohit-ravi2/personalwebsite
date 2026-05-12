"""
Path 2 v2 four-tier cell validation — Deliverable 5 (Group D).

Validates Layer 1 cells with v2 parameters (per-family C_global from
Group C calibration + refined γ from Group B) against four-tier hierarchy
per `docs/channel_parameter_derivation_methodology.md` §4.0:

  Tier A — First-principles consistency
    Mass conservation, no runaway dynamics, ion concentrations
    physiological at rest

  Tier B — Cell-level measurements
    V_rest in published range per cell class

  Tier C — Phenotype categories (deferred to follow-up — requires VC sims)
    Plateau vs graded distinction

  Tier D — Cross-cell consistency
    Cells with similar gene expression show similar behavior;
    differential expression produces differential behavior

Acceptance per §4.0: v2 passes if all cells satisfy Tier A + Tier B
AND ≥2/4 cells satisfy Tier C AND Tier D shows expected distinctions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import json
import numpy as np

from layer1_cells import (
    AVAL_SPEC, AVAR_SPEC, AIY_SPEC, RIM_SPEC, CellSpec,
    build_layer1_cell,
)
from channels.derived_channel_parameters import (
    get_derived_gbar, C_GLOBAL_PER_FAMILY, GAMMA_PS,
)


# V_rest published ranges per cell (Tier B)
V_REST_RANGE = {
    "AVAL": (-50.0, -30.0),  # AVA-class envelope
    "AVAR": (-35.0, -15.0),  # AVA-class envelope (more depolarized variant)
    "AIY":  (-95.0, -55.0),  # AIY-class
    "RIM":  (-65.0, -40.0),  # RIM-class
}

CELL_SPECS = {"AVAL": AVAL_SPEC, "AVAR": AVAR_SPEC, "AIY": AIY_SPEC, "RIM": RIM_SPEC}


def build_path2_cell(cell_name: str):
    """Build Path 2 v2 cell with current calibrated parameters."""
    nicoletti_spec = CELL_SPECS[cell_name]
    ch_p2_map = {
        "egl19": "EGL-19", "irk": "IRK", "nca": "NCA",
        "unc103": "UNC-103", "shl1": "SHL-1", "cca1": "CCA-1",
        "unc2": "UNC-2", "egl2": "EGL-2", "kqt1": "KQT-1",
    }
    path2_channels = {}
    for ch_name in nicoletti_spec.channels:
        ch_p2 = ch_p2_map.get(ch_name, ch_name.upper())
        path2_channels[ch_name] = get_derived_gbar(ch_p2, cell_name)
    spec = replace(nicoletti_spec, channels=path2_channels)
    return build_layer1_cell(spec)


def validate_cell(cell_name: str, sim_ms: float = 5000.0):
    """Run all four tiers for a single cell."""
    print(f"\n{'='*72}\n{cell_name} — v2 4-tier validation\n{'='*72}")

    bundle = build_path2_cell(cell_name)
    from brian2 import ms
    bundle["network"].run(sim_ms * ms)

    mon = bundle["monitor"]
    initial = {ion: float(getattr(mon, ion)[0][0]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}
    final = {ion: float(getattr(mon, ion)[0][-1]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}
    V_final = float(mon.v[0][-1] / 1e-3)

    # Tier A — First-principles consistency
    tier_A = {
        "K_in_physiological_100_140": 100 <= final["K_in"] <= 140,
        "Na_in_physiological_5_15":    5 <= final["Na_in"] <= 15,
        "Cl_in_physiological_3_10":    3 <= final["Cl_in"] <= 10,
        "Ca_in_physiological_50_200nM": 50e-6 <= final["Ca_in"] <= 200e-6,
        "no_runaway_ion_drift":         all(abs(final[ion]/initial[ion] - 1) < 0.5 for ion in initial),
    }
    tier_A_pass = all(tier_A.values())

    # Tier B — Cell-level measurements
    v_min, v_max = V_REST_RANGE[cell_name]
    tier_B = {
        "V_rest_in_published_range": v_min <= V_final <= v_max,
        "K_drift_within_5pct":       abs(100 * (final["K_in"]/initial["K_in"] - 1)) < 5.0,
        "Na_drift_within_5pct":      abs(100 * (final["Na_in"]/initial["Na_in"] - 1)) < 5.0,
    }
    tier_B_pass = all(tier_B.values())

    # Tier C — Phenotype categories (deferred; requires VC sim — placeholder)
    tier_C = {"plateau_capability": "deferred (requires VC sim)"}

    print(f"\n  After 5s rest:")
    print(f"    V_rest = {V_final:.2f} mV  (range: [{v_min}, {v_max}])")
    print(f"    [K]_in = {final['K_in']:.2f} mM  (Δ {100*(final['K_in']/initial['K_in']-1):+.2f}%)")
    print(f"    [Na]_in = {final['Na_in']:.2f} mM  (Δ {100*(final['Na_in']/initial['Na_in']-1):+.2f}%)")
    print(f"    [Cl]_in = {final['Cl_in']:.2f} mM")
    print(f"    [Ca]_in = {final['Ca_in']*1e6:.1f} nM")
    print(f"\n  Tier A — First-principles consistency:")
    for k, v in tier_A.items():
        print(f"    {k:<35} {'PASS' if v else 'FAIL'}")
    print(f"    Overall Tier A: {'PASS' if tier_A_pass else 'FAIL'}")
    print(f"\n  Tier B — Cell-level measurements:")
    for k, v in tier_B.items():
        print(f"    {k:<35} {'PASS' if v else 'FAIL'}")
    print(f"    Overall Tier B: {'PASS' if tier_B_pass else 'FAIL'}")
    print(f"\n  Tier C — Phenotype categories: deferred (requires VC sim)")

    return {
        "name": cell_name,
        "V_rest_mV": V_final,
        "final": final,
        "initial": initial,
        "tier_A": tier_A, "tier_A_pass": tier_A_pass,
        "tier_B": tier_B, "tier_B_pass": tier_B_pass,
        "tier_C": tier_C,
    }


def cross_cell_consistency(results: list[dict]) -> dict:
    """Tier D — cross-cell consistency check."""
    by_cell = {r["name"]: r for r in results}
    v_aval = by_cell["AVAL"]["V_rest_mV"]
    v_avar = by_cell["AVAR"]["V_rest_mV"]
    v_aiy = by_cell["AIY"]["V_rest_mV"]
    v_rim = by_cell["RIM"]["V_rest_mV"]

    checks = {
        "AVA_class_similar_within_25mV": abs(v_aval - v_avar) < 25.0,
        "AIY_more_hyperpolarized_than_AVA": v_aiy < min(v_aval, v_avar),
        "AVAL_distinct_from_AIY_by_20mV": abs(v_aval - v_aiy) > 20.0,
    }

    print(f"\n{'='*72}\nTier D — Cross-cell consistency\n{'='*72}")
    print(f"  V_rest by cell: AVAL={v_aval:.1f}, AVAR={v_avar:.1f}, AIY={v_aiy:.1f}, RIM={v_rim:.1f}")
    for k, v in checks.items():
        print(f"    {k:<40} {'PASS' if v else 'FAIL'}")
    tier_D_pass = all(checks.values())
    print(f"    Overall Tier D: {'PASS' if tier_D_pass else 'FAIL'}")
    return {"checks": checks, "tier_D_pass": tier_D_pass}


def summary_report(results, tier_D):
    print(f"\n{'#'*72}\n# v2 Validation summary\n{'#'*72}")
    print(f"\n{'cell':<6} {'V_rest':<10} {'Tier A':<10} {'Tier B':<10} {'verdict':<15}")
    print('-' * 60)
    tier_a_pass_count = 0
    tier_b_pass_count = 0
    for r in results:
        a = "PASS" if r["tier_A_pass"] else "FAIL"
        b = "PASS" if r["tier_B_pass"] else "FAIL"
        if r["tier_A_pass"]: tier_a_pass_count += 1
        if r["tier_B_pass"]: tier_b_pass_count += 1
        verdict = "PASS" if r["tier_A_pass"] and r["tier_B_pass"] else "FAIL"
        print(f"{r['name']:<6} {r['V_rest_mV']:>+7.2f}  {a:<10} {b:<10} {verdict:<15}")

    n_total = len(results)
    print(f"\nTier A passes: {tier_a_pass_count}/{n_total}")
    print(f"Tier B passes: {tier_b_pass_count}/{n_total}")
    print(f"Tier D consistency: {'PASS' if tier_D['tier_D_pass'] else 'FAIL'}")

    # v2 acceptance per §4.0
    print(f"\n{'='*72}\nv2 Acceptance per §4.0:")
    print(f"  Required: Tier A pass for all + Tier B pass for all + Tier D consistency")
    print(f"            (Tier C deferred — kinetic audit work block)")

    if tier_a_pass_count == n_total and tier_b_pass_count == n_total and tier_D["tier_D_pass"]:
        print(f"  v2 PASSES → Ship Path 2 v2 as methodology demonstration")
    else:
        print(f"  v2 PARTIAL → ", end="")
        if tier_a_pass_count == n_total and tier_b_pass_count >= 2:
            print(f"Tier A all + Tier B partial; ship as v2 with documented Tier B exceptions")
        else:
            print(f"Diagnose failure pattern for v3 routing")


def main():
    print(f"\n{'#'*72}\n# §7.3.5 v2 Group D — 4-tier validation (Path 2 v2)\n{'#'*72}")
    print(f"\nv2 parameters in use:")
    print(f"  C_global per family:")
    for f, c in C_GLOBAL_PER_FAMILY.items():
        print(f"    {f}: {c:.2e}")
    print(f"  γ values (refined where applicable):")
    for ch, g in GAMMA_PS.items():
        marker = " (v2 refit)" if ch in ("IRK", "NCA") else ""
        print(f"    {ch:<8} {g} pS{marker}")

    results = [validate_cell(name) for name in ("AVAL", "AVAR", "AIY", "RIM")]
    tier_D = cross_cell_consistency(results)
    summary_report(results, tier_D)


if __name__ == "__main__":
    main()
