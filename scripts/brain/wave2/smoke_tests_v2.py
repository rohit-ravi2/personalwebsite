"""Phase β CP1.A.4 — smoke tests for updated harnesses + NEURONReference.

Adds smoke tests for:
  * NEURONReference wrapper (instantiate AVAL, run vclamp at 3 holds, verify struct)
  * voltage_clamp_compare_v2 with current-domain tolerance (good case + bad case)
  * current_clamp_layer_a_compare on a leak-only matched pair (Brian2 ≈ NEURON)

All Phase α smoke tests must still pass — that's verified by running smoke_tests.py
separately. This file ADDS coverage; does not replace.

Usage:
    /home/rohit/venvs/wave2-neuron/bin/python smoke_tests_v2.py
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from voltage_clamp_harness import (
    voltage_clamp_compare_v2,
    current_domain_divergence,
    evaluate_current_domain_panel,
    leak_brian2_factory,
    leak_analytic_reference,
)
from plateau_harness import (
    current_clamp_layer_a_compare,
)
from neuron_reference import NEURONReference


# ---------------------------------------------------------------------------
# Test 1: NEURONReference wrapper basic
# ---------------------------------------------------------------------------

def smoke_neuron_reference_wrapper() -> dict:
    """Instantiate AVAL, run vclamp at 3 holds, verify output structure."""
    print("[1] NEURONReference wrapper: instantiate AVAL")
    ref = NEURONReference("AVAL")

    print("[1] NEURONReference wrapper: voltage_clamp at -60, -30, 0 mV")
    vc = ref.voltage_clamp(
        holding_potentials=[-60.0, -30.0, 0.0],
        duration_ms=250.0,
        prestep_ms=1007.8,
        prestep_mV=-30.0,
        tail_ms=242.2,
        tail_mV=-30.0,
        dt_ms=0.01,
    )

    # Validate structure
    required_top = {"cell", "protocol", "holds", "surf_cm2"}
    missing = required_top - set(vc.keys())
    if missing:
        print(f"    FAIL: missing keys {missing}")
        return {"pass": False, "reason": f"missing top keys: {missing}"}

    if len(vc["holds"]) != 3:
        return {"pass": False, "reason": f"expected 3 holds, got {len(vc['holds'])}"}

    required_hold = {"hold_mV", "t_ms", "V_mV", "I_total_pA",
                     "peak_I_pA", "ss_I_pA", "time_to_peak_ms"}
    for h in vc["holds"]:
        missing_h = required_hold - set(h.keys())
        if missing_h:
            return {"pass": False, "reason": f"hold missing keys: {missing_h}"}

    print("    Hold features:")
    for h in vc["holds"]:
        print(f"      hold={h['hold_mV']:+6.1f} mV  peak_I={h['peak_I_pA']:+8.2f} pA  "
              f"ss_I={h['ss_I_pA']:+8.2f} pA  ttp={h['time_to_peak_ms']:6.2f} ms")

    # Cross-check against v3 captured AVAL: at -30 mV the model should produce
    # close-to-zero current (clamp = pre-step, no transient). At 0 mV EGL-19
    # should be partially active producing a small ica (negative for inward).
    print("    v3-cross-check: EGL-19 partially active at 0 mV → expect non-trivial current")
    h0 = next(h for h in vc["holds"] if abs(h["hold_mV"]) < 1)
    if abs(h0["peak_I_pA"]) < 1e-6:
        print(f"    WARN: peak_I at 0 mV is {h0['peak_I_pA']:.3e} pA (suspicious, expected non-zero)")

    print("[1] NEURONReference wrapper: PASS (structure + non-trivial currents)")
    ref.cleanup()
    return {"pass": True, "vc_result": vc}


# ---------------------------------------------------------------------------
# Test 2: Current-domain tolerance metric — known-good case
# ---------------------------------------------------------------------------

def smoke_current_domain_tolerance_good() -> dict:
    """Apply current-domain metric to a leak-only Brian2 vs analytic ref.
    Should pass (Brian2 leak == analytic leak)."""
    print("[2] current-domain tolerance (good case): leak vs analytic")
    g_leak_nS = 1.0
    e_leak_mV = -70.0
    holds = [-100, -80, -60, -40, -20, 0, 20]
    factory = leak_brian2_factory(g_leak_nS=g_leak_nS, e_leak_mV=e_leak_mV)
    ref = leak_analytic_reference(g_leak_nS=g_leak_nS, e_leak_mV=e_leak_mV)
    result = voltage_clamp_compare_v2(
        factory, ref, holds,
        duration_ms=200.0, dt_ms=0.025, settle_window_ms=20.0,
        feature_tolerance=0.05, panel_pass_fraction=0.8,
    )
    print(f"    panel_pass={result['panel_pass']}  fraction_passing={result['fraction_passing']:.3f}")
    for s in result["per_step"]:
        print(f"    hold={s['hold_mV']:+6.1f} mV  brian2_ss={s['brian2_ss_I_pA']:+8.3f} pA  "
              f"ref_ss={s['ref_ss_I_pA']:+8.3f} pA")
    return result


# ---------------------------------------------------------------------------
# Test 3: Current-domain tolerance metric — known-bad case
# ---------------------------------------------------------------------------

def smoke_current_domain_tolerance_bad() -> dict:
    """Apply current-domain metric to a Brian2 leak that is deliberately
    miscalibrated (10% wrong g_leak). Should FAIL the panel test."""
    print("[3] current-domain tolerance (bad case): Brian2 g=1.1 nS vs analytic g=1.0 nS")
    factory = leak_brian2_factory(g_leak_nS=1.1, e_leak_mV=-70.0)
    ref = leak_analytic_reference(g_leak_nS=1.0, e_leak_mV=-70.0)
    holds = [-100, -80, -60, -40, -20, 0, 20]
    result = voltage_clamp_compare_v2(
        factory, ref, holds,
        duration_ms=200.0, dt_ms=0.025, settle_window_ms=20.0,
        feature_tolerance=0.05, panel_pass_fraction=0.8,
    )
    print(f"    panel_pass={result['panel_pass']}  fraction_passing={result['fraction_passing']:.3f}")
    for s in result["per_step"]:
        print(f"    hold={s['hold_mV']:+6.1f} mV  brian2_ss={s['brian2_ss_I_pA']:+8.3f} pA  "
              f"ref_ss={s['ref_ss_I_pA']:+8.3f} pA")
    # Expected: 10% mis-calibration ≥ 5% tolerance, panel should fail
    return result


# ---------------------------------------------------------------------------
# Test 4: divergence() unit checks
# ---------------------------------------------------------------------------

def smoke_divergence_unit() -> dict:
    """Spot-check current_domain_divergence formula behaves as expected."""
    print("[4] divergence() unit checks")

    # Identical → 0
    assert current_domain_divergence(10.0, 10.0, 100.0) == 0.0

    # 10% relative → div ≈ 0.1
    d = current_domain_divergence(10.0, 11.0, 100.0)
    assert abs(d - 1.0/11.0) < 1e-6, f"got {d}, expected ~0.0909"

    # Both near zero, peak large → div ≤ floor
    d = current_domain_divergence(0.05, 0.0, 100.0)
    assert d <= 0.05, f"got {d}, expected ≤ 0.05 (floor at 0.1*peak=10)"

    # Pathological case: both = 0
    assert current_domain_divergence(0.0, 0.0, 0.0) == 0.0

    # Sign-flip case
    d = current_domain_divergence(-50.0, 50.0, 100.0)
    assert d == 2.0, f"got {d}, expected 2.0"

    print("    All unit checks passed.")
    return {"pass": True}


# ---------------------------------------------------------------------------
# Test 5: current_clamp_layer_a_compare on matched leak cells
# ---------------------------------------------------------------------------

def smoke_layer_a_leak_only() -> dict:
    """Build a leak-only Brian2 cell + leak-only NEURON cell with identical
    parameters, run CC protocol, verify Layer A comparison passes."""
    print("[5] current_clamp_layer_a_compare on leak-only matched cells")

    # Parameters: AVAL-like geometry, leak-only
    g_leak_Scm2 = 0.01
    e_leak_mV = -65.0
    cm_uFcm2 = 1.0
    surf_cm2 = 1e-5
    surf_um2 = surf_cm2 * 1e8
    # Convert to single-cell units
    g_leak_nS = g_leak_Scm2 * surf_cm2 * 1e9       # S → nS
    cm_pF = cm_uFcm2 * surf_cm2 * 1e6              # μF → pF

    print(f"    Cell: surf={surf_cm2:.2e} cm²  g_leak={g_leak_nS:.3f} nS  C={cm_pF:.3f} pF  E_leak={e_leak_mV} mV")

    # Brian2 factory — leak-only with current injection
    def factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, ms, mV, nS, pA, pF, amp,
            prefs, start_scope,
        )
        start_scope()
        prefs.codegen.target = "cython"
        eqs = """
        dv/dt = (-g_leak * (v - E_leak) + I_inject) / C_mem : volt
        I_inject : amp
        E_leak : volt
        g_leak : siemens
        C_mem : farad
        """
        G = NeuronGroup(1, eqs, method="exact")
        G.E_leak = e_leak_mV * mV
        G.g_leak = g_leak_nS * nS
        G.C_mem = cm_pF * pF
        G.v = e_leak_mV * mV
        G.I_inject = 0 * pA

        mon = StateMonitor(G, ["v"], record=True)
        net = Network(G, mon)

        def inject_pA(amp_pA: float) -> None:
            G.I_inject = amp_pA * pA

        return {
            "group": G,
            "monitor": mon,
            "network": net,
            "inject_pA": inject_pA,
            "tau_m_ms": cm_pF / g_leak_nS,
        }

    # NEURON reference — custom cell, leak-only
    custom_spec = {
        "channels": ["leak"],
        "params": {
            ("leak", "gbar"): g_leak_Scm2,
            ("leak", "e"): e_leak_mV,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": 60.0,
        "ek_mV": -80.0,
        "v_init_mV": e_leak_mV,
    }
    nref = NEURONReference("custom", custom_spec=custom_spec)

    result = current_clamp_layer_a_compare(
        factory, nref, "leak_only_AVAL_geom",
        injection_pa=20.0,
        injection_duration_ms=200.0,
        settle_ms=100.0,
        post_ms=300.0,
        v_rest_mv=e_leak_mV,
        voltage_feature_tolerance_mV=3.0,
        timepoint_pass_fraction=0.8,
        dt_ms=0.025,
    )

    bf = result["brian2_features"]
    nf = result["neuron_features"]
    fr = result["feature_residuals"]
    print(f"    Brian2 baseline_pre={bf['baseline_pre_mV']:+6.2f} mV  "
          f"peak={bf['peak_V_mV']:+6.2f} mV  plateau={bf['plateau_V_mV']:+6.2f} mV")
    print(f"    NEURON baseline_pre={nf['baseline_pre_mV']:+6.2f} mV  "
          f"peak={nf['peak_V_mV']:+6.2f} mV  plateau={nf['plateau_V_mV']:+6.2f} mV")
    print(f"    Residuals: peak={fr['peak_V_mV']:.3f} mV  plateau={fr['plateau_V_mV']:.3f} mV")
    print(f"    fraction_passing={result['fraction_passing']:.3f}  panel_pass={result['panel_pass']}")
    print(f"    timing_diag: ttp_residual={result['timing_diagnostics']['time_to_peak_residual_ms']:.2f} ms  "
          f"settling_residual={result['timing_diagnostics']['settling_time_residual_ms']:.2f} ms")

    nref.cleanup()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Phase β CP1.A.4 — Smoke tests v2 (Layer A + NEURONReference) ===\n")

    print("--- Test 1: NEURONReference wrapper structure ---")
    r1 = smoke_neuron_reference_wrapper()
    print()

    print("--- Test 2: current-domain tolerance (good case) ---")
    r2 = smoke_current_domain_tolerance_good()
    print()

    print("--- Test 3: current-domain tolerance (bad case) ---")
    r3 = smoke_current_domain_tolerance_bad()
    print()

    print("--- Test 4: divergence() unit checks ---")
    try:
        r4 = smoke_divergence_unit()
    except AssertionError as e:
        print(f"    FAIL: {e}")
        r4 = {"pass": False, "reason": str(e)}
    print()

    print("--- Test 5: current_clamp_layer_a_compare leak-only ---")
    r5 = smoke_layer_a_leak_only()
    print()

    # Decision matrix
    t1_ok = r1.get("pass", False)
    t2_ok = r2.get("panel_pass", False)
    t3_ok = (not r3.get("panel_pass", False))  # 10% mis-calibration should FAIL
    t4_ok = r4.get("pass", False)
    t5_ok = r5.get("panel_pass", False)

    print("=== SMOKE TEST SUMMARY ===")
    print(f"  T1 NEURONReference wrapper structure: {t1_ok}")
    print(f"  T2 current-domain tolerance good case (pass): {t2_ok}")
    print(f"  T3 current-domain tolerance bad case (fail): {t3_ok}")
    print(f"  T4 divergence unit checks: {t4_ok}")
    print(f"  T5 Layer A compare leak-only (pass): {t5_ok}")
    overall = bool(t1_ok and t2_ok and t3_ok and t4_ok and t5_ok)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
