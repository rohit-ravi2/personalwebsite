"""
Layer 1 §7.3 — Per-cell integration validation.

Tests AVAL, AVAR, RIM, AIY cells composed with ion_dynamics + pumps + Nicoletti
channels. Each cell runs for 5s; reports steady-state ion concentrations + V_rest.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np

from layer1_cells import build_layer1_cell, CELL_SPECS


def run_cell_and_report(spec_name: str, sim_ms: float = 5000.0):
    print(f"\n{'='*72}\nTesting {spec_name}\n{'='*72}")
    spec = CELL_SPECS[spec_name]
    try:
        bundle = build_layer1_cell(spec)
    except Exception as e:
        print(f"  BUILD FAILED: {type(e).__name__}: {e}")
        return None
    f_K, f_Na = bundle["leak_split"]
    print(f"  LEAK split: f_K = {f_K:.3f}, f_Na = {f_Na:.3f} (from e_leak = {spec.e_leak_mV} mV)")
    print(f"  Pump params (TPM-scaled from AVAL anchor):")
    for k, v in bundle["pump_params"].items():
        print(f"    {k:<20} = {v:.4e}")
    from brian2 import ms
    try:
        bundle["network"].run(sim_ms * ms)
    except Exception as e:
        print(f"  RUN FAILED: {type(e).__name__}: {e}")
        return None
    mon = bundle["monitor"]
    initial = {ion: float(getattr(mon, ion)[0][0]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}
    final = {ion: float(getattr(mon, ion)[0][-1]) for ion in ("K_in", "Na_in", "Cl_in", "Ca_in")}
    deltas_pct = {ion: 100 * (final[ion] / initial[ion] - 1) for ion in initial}
    V_final = float(mon.v[0][-1] / 1e-3)
    print(f"\n  After 5s simulation:")
    print(f"    V_rest    = {V_final:.2f} mV (started {spec.v_init_mV})")
    print(f"    [K]_in    = {final['K_in']:.3f} mM (Δ {deltas_pct['K_in']:+.2f}%)")
    print(f"    [Na]_in   = {final['Na_in']:.3f} mM (Δ {deltas_pct['Na_in']:+.2f}%)")
    print(f"    [Cl]_in   = {final['Cl_in']:.3f} mM (Δ {deltas_pct['Cl_in']:+.2f}%)")
    print(f"    [Ca]_in   = {final['Ca_in']*1e6:.1f} nM (Δ {deltas_pct['Ca_in']:+.2f}%)")
    print(f"    Dynamic Nernst: E_K={float(mon.E_K_mV[0][-1]):.1f}, E_Na={float(mon.E_Na_mV[0][-1]):.1f}, "
          f"E_Cl={float(mon.E_Cl_mV[0][-1]):.1f}, E_Ca={float(mon.E_Ca_mV[0][-1]):.1f}")
    return {
        "name": spec_name, "V_rest_mV": V_final, "initial": initial, "final": final,
        "deltas_pct": deltas_pct, "leak_split": (f_K, f_Na),
        "pump_params": bundle["pump_params"],
        "rest_published": spec.rest_published_mV,
    }


def summary(results: list) -> None:
    print("\n" + "=" * 72)
    print("Summary — Layer 1 §7.3 per-cell integration")
    print("=" * 72)
    print(f"\n{'cell':<6} {'V_rest':>8} {'[K]':>10} {'[Na]':>10} {'[Cl]':>10} {'[Ca]':>10} {'pub V':>14}  verdict")
    print("-" * 86)
    for r in results:
        if r is None:
            continue
        d = r["deltas_pct"]
        f = r["final"]
        pub_min, pub_max = r["rest_published"]
        v_pass = pub_min <= r["V_rest_mV"] <= pub_max
        k_pass = abs(d["K_in"]) < 2.0
        cl_pass = 3.0 <= f["Cl_in"] <= 7.0
        ca_pass = f["Ca_in"] < 5.0e-4
        all_pass = v_pass and k_pass and cl_pass and ca_pass
        verdict = "PASS" if all_pass else "FINDING"
        print(f"{r['name']:<6} {r['V_rest_mV']:>+7.2f}  "
              f"{f['K_in']:>6.2f}({d['K_in']:+4.1f}%)  "
              f"{f['Na_in']:>6.2f}({d['Na_in']:+4.1f}%)  "
              f"{f['Cl_in']:>6.2f}({d['Cl_in']:+4.1f}%)  "
              f"{f['Ca_in']*1e6:>5.1f}nM  "
              f"[{pub_min:+5.0f},{pub_max:+4.0f}]  {verdict}")


def main():
    print("#" * 72)
    print("# Layer 1 §7.3 — Per-cell integration validation")
    print("#" * 72)
    results = [run_cell_and_report(name) for name in ("AVAL", "AVAR", "RIM", "AIY")]
    summary(results)


if __name__ == "__main__":
    main()
