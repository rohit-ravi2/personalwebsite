"""
Diagnose Layer 2 NaN issue.

Tests, in order:
  1. Substrate-only (synapses zeroed): does the 300-cell substrate run
     stable in isolation?
  2. Substrate + gap junctions only (no chemical synapses)
  3. Full network

For each, run 500 ms (short) and report which cells went NaN first.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from layer2.assemble import assemble_layer2_network


def run_with_modifications(scale_chem: float, scale_gap: float, sim_ms: float = 500.0):
    """scale_chem / scale_gap are conductance per unit weight, in pS."""
    from brian2 import ms, second
    bundle = assemble_layer2_network(chem_scale=scale_chem * 1e-12,
                                      gap_scale=scale_gap * 1e-12)
    print(f"\n  chem_scale={scale_chem} pS, gap_scale={scale_gap} pS, sim={sim_ms}ms")
    bundle["network"].run(sim_ms * ms)
    mon = bundle["monitor"]
    names = bundle["meta"]["cell_names"]
    V_final = np.asarray(mon.v[:, -1] / 1e-3)
    Ca_final = np.asarray(mon.Ca_in[:, -1]) * 1e3
    K_final = np.asarray(mon.K_in[:, -1])
    Na_final = np.asarray(mon.Na_in[:, -1])
    nan_v = np.isnan(V_final)
    nan_ca = np.isnan(Ca_final)

    print(f"  NaN V: {nan_v.sum()}/{len(names)}, NaN Ca: {nan_ca.sum()}/{len(names)}")
    if not nan_v.all():
        V_ok = V_final[~nan_v]
        Ca_ok = Ca_final[~nan_v]
        K_ok = K_final[~nan_v]
        Na_ok = Na_final[~nan_v]
        print(f"  V: min {V_ok.min():+.1f}, max {V_ok.max():+.1f}, med {np.median(V_ok):+.1f} mV")
        print(f"  Ca: max {Ca_ok.max():.3f} μM, med {np.median(Ca_ok):.4f}")
        print(f"  K: min {K_ok.min():.1f}, max {K_ok.max():.1f} mM")
        print(f"  Na: min {Na_ok.min():.2f}, max {Na_ok.max():.2f} mM")

    # Which cells went NaN first?
    if nan_v.any():
        nan_names = [n for n, isnan in zip(names, nan_v) if isnan][:10]
        print(f"  Sample NaN cells: {nan_names}")

    return {"n_nan_V": int(nan_v.sum()),
            "n_nan_Ca": int(nan_ca.sum()),
            "n_total": len(names)}


def main():
    print("=" * 80)
    print("Layer 2 stability diagnostic")
    print("=" * 80)

    print("\n--- Test 1: Substrate only (no synapses) ---")
    run_with_modifications(scale_chem=0.0, scale_gap=0.0)

    print("\n--- Test 2: Gap junctions @ 1 pS ---")
    run_with_modifications(scale_chem=0.0, scale_gap=1.0)

    print("\n--- Test 3: Chemical synapses @ 1 pS ---")
    run_with_modifications(scale_chem=1.0, scale_gap=0.0)

    print("\n--- Test 4: Both @ 1 pS ---")
    run_with_modifications(scale_chem=1.0, scale_gap=1.0)

    print("\n--- Test 5: Chem 1 pS + Gap 0.01 pS ---")
    run_with_modifications(scale_chem=1.0, scale_gap=0.01)

    print("\n--- Test 6: Chem 1 pS + Gap 0.001 pS ---")
    run_with_modifications(scale_chem=1.0, scale_gap=0.001)


if __name__ == "__main__":
    main()
