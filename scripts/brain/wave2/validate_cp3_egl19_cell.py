"""
Phase β CP3 — EGL-19 in cell context (Gate 2a).

Constructs a minimal Brian2 cell (leak + cadiff + caintra1 + EGL-19) and
equivalent NEURON reference, runs voltage-clamp + current-clamp protocols
on both, compares via Layer A harness.

Spec note: EGL-19 alone won't produce Mellem-style sustained plateau (no
SLO-1, no termination). That's expected, NOT failure. CP3 validates
implementation correctness, not phenotype reproduction.

Configuration
-------------

Both cells use AVAL geometry + AVAL leak/EGL-19 conductances. caintra1 is
included as bookkeeping (EGL-19 doesn't read cai for inactivation in
Nicoletti's parameterization). cadiff is OMITTED (it would conflict with
caintra1 on the cai writer in NEURON; for CP3 we use caintra1 alone since
that's what AIY/RIM use as their pool).

Voltage-clamp validation: same as CP2 protocol, but on the [leak + EGL-19 +
caintra1] cell (caintra1 is silent for V dynamics).

Current-clamp validation: 50 pA injection × 100 ms (Mellem-style step),
settle 200 ms, post 1500 ms, v_rest = -25 mV. Compare V trajectories.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from neuron_reference import NEURONReference
from voltage_clamp_harness import voltage_clamp_compare_v2
from plateau_harness import current_clamp_layer_a_compare
from channels.egl19 import EGL19_EQS, egl19_apply_params, egl19_init_states
from calcium_pool import caintra1_eqs


def build_brian2_cp3_cell_factory(
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_egl19_Scm2: float = 0.104385e-9 / 1123.84e-8,
    eca_mV: float = 60.0,
    v_init_mV: float = -60.0,
    vol_cm3: float = 129.6e-12,        # AVAL volume
    fca: float = 0.001,
    tca_ms: float = 50.0,
    ca_eq_mM: float = 5e-8,
    voltage_clamp_mode: bool = True,
):
    """Brian2 factory for [leak + EGL-19 + caintra1] cell.

    voltage_clamp_mode=True wires set_v + clamp via network_operation.
    voltage_clamp_mode=False wires inject_pA for current-clamp.
    """
    pool = caintra1_eqs(vol_cm3=vol_cm3, surf_cm2=surf_cm2, fca=fca,
                        tca_ms=tca_ms, ca_eq_mM=ca_eq_mM)

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, pA, pF, prefs, start_scope, defaultclock,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Build full cell eqs
        cell_eqs = """
        v_mV = v / mV : 1
        # Channel currents (sum to total ica for Ca pool):
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        # Leak:
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        # Total membrane current density:
        i_total_mAcm2 = i_leak_mAcm2 + ica_mAcm2 : 1
        # Total membrane current (pA):
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA : amp
        # I_inject (pA, set externally for current-clamp):
        I_inject : amp
        # Membrane potential dynamics — used in current-clamp;
        # voltage-clamp suppresses this via network_operation forcing v.
        dv/dt = (I_inject - I_total) / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        # Cell parameters:
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + EGL19_EQS + pool["eqs"]

        G = NeuronGroup(1, cell_eqs, method="euler")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        G.I_inject = 0 * pA
        egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_init_states(G, v_mV=v_init_mV)
        # Pool params
        for k, v in pool["params"].items():
            setattr(G, k, v)
        for k, (unit_str, v) in pool.get("params_with_units", {}).items():
            from brian2 import ms as _ms, second as _sec
            unit_map = {"ms": _ms, "second": _sec}
            setattr(G, k, v * unit_map[unit_str])
        for k, v in pool["init"].items():
            setattr(G, k, v)

        clamp = {"v_target_mV": v_init_mV}

        if voltage_clamp_mode:
            @network_operation(dt=0.025 * ms)
            def _clamp():
                G.v = clamp["v_target_mV"] * mV
            mon_vars = ["v", "I_total", "ica_egl19_mAcm2", "i_leak_mAcm2",
                        "m_egl19", "h_egl19", "cai_mM"]
            mon = StateMonitor(G, mon_vars, record=True)
            net = Network(G, mon, _clamp)

            def set_v(v_mV: float) -> None:
                clamp["v_target_mV"] = float(v_mV)
                G.v = float(v_mV) * mV

            return {
                "group": G,
                "monitor": mon,
                "network": net,
                "set_v": set_v,
            }
        else:
            mon = StateMonitor(G, ["v", "I_total", "ica_egl19_mAcm2",
                                    "i_leak_mAcm2", "m_egl19", "h_egl19",
                                    "cai_mM"], record=True)
            net = Network(G, mon)

            def inject_pA(amp_pA: float) -> None:
                G.I_inject = float(amp_pA) * pA

            def set_v(v_mV: float) -> None:
                G.v = float(v_mV) * mV

            return {
                "group": G,
                "monitor": mon,
                "network": net,
                "inject_pA": inject_pA,
                "set_v": set_v,
            }

    return _factory


def build_neuron_cp3_cell(
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_egl19_Scm2: float = 0.104385e-9 / 1123.84e-8,
    eca_mV: float = 60.0,
) -> NEURONReference:
    """NEURON [leak + egl19 + caintra1] reference cell."""
    custom_spec = {
        "channels": ["leak", "egl19", "caintra1"],
        "params": {
            ("leak", "gbar"): g_leak_Scm2,
            ("leak", "e"): e_leak_mV,
            ("egl19", "gbar"): g_egl19_Scm2,
            ("caintra1", "vol"): 129.6e-12,
            ("caintra1", "surf"): surf_cm2,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": eca_mV,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
    }
    return NEURONReference("custom", custom_spec=custom_spec)


def main():
    print("=== CP3: EGL-19 Gate 2a evaluation in cell context ===\n")

    surf = 1123.84e-8
    g_leak = 0.150164e-9 / surf
    g_egl19 = 0.104385e-9 / surf
    e_leak = -39.0
    cm = 0.859551

    print(f"Cell: AVAL geometry — leak + EGL-19 + caintra1")
    print(f"  surf={surf:.3e} cm², cm={cm:.3f} μF/cm²")
    print(f"  g_leak={g_leak:.3e} S/cm², e_leak={e_leak} mV")
    print(f"  g_egl19={g_egl19:.3e} S/cm², eca=60 mV\n")

    # --- Voltage-clamp protocol ---
    print("=== CP3 VC: voltage-clamp Layer A comparison ===")
    print("Building NEURON reference...")
    nref_vc = build_neuron_cp3_cell(
        surf_cm2=surf, cm_uFcm2=cm, g_leak_Scm2=g_leak, e_leak_mV=e_leak,
        g_egl19_Scm2=g_egl19,
    )
    print("Building Brian2 factory...")
    factory_vc = build_brian2_cp3_cell_factory(
        surf_cm2=surf, cm_uFcm2=cm, g_leak_Scm2=g_leak, e_leak_mV=e_leak,
        g_egl19_Scm2=g_egl19, voltage_clamp_mode=True,
    )

    holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]
    print(f"Running voltage-clamp at {len(holds)} holds...")
    vc_result = voltage_clamp_compare_v2(
        factory_vc, nref_vc, holds,
        duration_ms=200.0,
        dt_ms=0.025,
        settle_window_ms=20.0,
        feature_tolerance=0.05,
        panel_pass_fraction=0.8,
        feature_keys=("peak_I_pA", "ss_I_pA"),
        skip_initial_transient_ms=2.0,
        brian2_prestep_ms=50.0,
        brian2_prestep_mV=-60.0,
    )
    print(f"VC result: panel_pass={vc_result['panel_pass']}  "
          f"holds_passing={vc_result['n_holds_passing']}/{vc_result['n_holds']}  "
          f"frac={vc_result['fraction_passing']:.3f}")
    for s in vc_result["per_step"]:
        print(f"  hold={s['hold_mV']:+6.1f} mV  brian2_peak={s['brian2_peak_I_pA']:+8.2f}  "
              f"nrn_peak={s['ref_peak_I_pA']:+8.2f}  "
              f"brian2_ss={s['brian2_ss_I_pA']:+8.2f}  nrn_ss={s['ref_ss_I_pA']:+8.2f}")
    nref_vc.cleanup()
    print()

    # --- Current-clamp protocol ---
    print("=== CP3 CC: current-clamp Layer A comparison ===")
    print("Building NEURON reference (fresh instance)...")
    nref_cc = build_neuron_cp3_cell(
        surf_cm2=surf, cm_uFcm2=cm, g_leak_Scm2=g_leak, e_leak_mV=e_leak,
        g_egl19_Scm2=g_egl19,
    )
    print("Building Brian2 factory (current-clamp mode)...")
    factory_cc = build_brian2_cp3_cell_factory(
        surf_cm2=surf, cm_uFcm2=cm, g_leak_Scm2=g_leak, e_leak_mV=e_leak,
        g_egl19_Scm2=g_egl19, voltage_clamp_mode=False, v_init_mV=-25.0,
    )

    print("Running current-clamp comparison: 50 pA × 100 ms, settle 200 ms, post 1500 ms, v_rest -25 mV...")
    cc_result = current_clamp_layer_a_compare(
        factory_cc, nref_cc, "AVA_egl19_only_cp3",
        injection_pa=50.0,
        injection_duration_ms=100.0,
        settle_ms=200.0,
        post_ms=1500.0,
        v_rest_mv=-25.0,
        voltage_feature_tolerance_mV=3.0,
        timepoint_pass_fraction=0.8,
        dt_ms=0.025,
    )

    bf = cc_result["brian2_features"]
    nf = cc_result["neuron_features"]
    fr = cc_result["feature_residuals"]
    td = cc_result["timing_diagnostics"]
    print(f"\nBrian2 features: baseline_pre={bf['baseline_pre_mV']:+6.2f}  "
          f"peak={bf['peak_V_mV']:+6.2f}  plateau={bf['plateau_V_mV']:+6.2f}  "
          f"baseline_post={bf['baseline_post_mV']:+6.2f}")
    print(f"NEURON features: baseline_pre={nf['baseline_pre_mV']:+6.2f}  "
          f"peak={nf['peak_V_mV']:+6.2f}  plateau={nf['plateau_V_mV']:+6.2f}  "
          f"baseline_post={nf['baseline_post_mV']:+6.2f}")
    print(f"Residuals: peak={fr['peak_V_mV']:.3f} mV  plateau={fr['plateau_V_mV']:.3f} mV  "
          f"baseline={fr['baseline_pre_mV']:.3f} mV")
    print(f"Per-timepoint: {cc_result['n_timepoints_passing']}/{cc_result['n_timepoints']} pass = "
          f"{cc_result['fraction_passing']:.3f}")
    print(f"feature_pass={cc_result['feature_pass']}  panel_pass={cc_result['panel_pass']}")
    print(f"Timing diag (warn-only): ttp_residual={td['time_to_peak_residual_ms']:.2f} ms  "
          f"settling_residual={td['settling_time_residual_ms']:.2f} ms")
    nref_cc.cleanup()

    # Save results
    out_path = Path(__file__).parent / "artifacts" / "cp3_validation_results.json"
    serializable = {
        "vc": {
            "panel_pass": vc_result["panel_pass"],
            "n_holds": vc_result["n_holds"],
            "n_holds_passing": vc_result["n_holds_passing"],
            "fraction_passing": vc_result["fraction_passing"],
            "per_step": vc_result["per_step"],
            "tolerance_metric": vc_result["tolerance_metric"],
            "warnings": vc_result["warnings"],
        },
        "cc": {
            "panel_pass": cc_result["panel_pass"],
            "feature_pass": cc_result["feature_pass"],
            "fraction_passing": cc_result["fraction_passing"],
            "brian2_features": cc_result["brian2_features"],
            "neuron_features": cc_result["neuron_features"],
            "feature_residuals": cc_result["feature_residuals"],
            "timing_diagnostics": cc_result["timing_diagnostics"],
            "warnings": cc_result["warnings"],
            "protocol": cc_result["protocol"],
        },
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nWrote: {out_path}")

    overall_pass = vc_result["panel_pass"] and cc_result["panel_pass"]
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
