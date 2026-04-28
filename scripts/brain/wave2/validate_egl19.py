"""
Phase β CP2 — EGL-19 isolated voltage-clamp validation.

Strategy
--------
Build a Brian2 cell with [leak + EGL-19] (no Ca pool — EGL-19 doesn't depend
on cai). Build equivalent NEURON cell. Run voltage-clamp protocol on both,
compare ica trajectories using current-domain tolerance.

Acceptance criteria (CP2)
- EGL-19 Brian2 implementation files exist (channel module + parameters): yes
- Voltage-clamp validation passes >80% of holding potentials within tolerance
- IV curve from Brian2 implementation matches NEURON reference within tolerance
- Time-to-peak and inactivation kinetics reported (warn-only diagnostics)

Cell construction
-----------------
NEURON section: [leak + egl19] only. Use AVAL leak + AVAL egl19 gbar for
realistic operating regime. Voltage-clamp at multiple holds (-60 to +30 mV
in 10 mV steps).

Brian2 cell: same passive parameters, EGL_19_EQS attached, same gbars.
Voltage-clamped via network_operation.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from neuron_reference import NEURONReference, _nicoletti_env
from voltage_clamp_harness import (
    voltage_clamp_compare_v2, current_domain_divergence,
    evaluate_current_domain_panel,
)
from channels.egl19 import EGL19_EQS, EGL19_PARAMS, egl19_apply_params, egl19_init_states


def build_brian2_egl19_cell_factory(
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,  # AVAL rescaled
    e_leak_mV: float = -39.0,
    g_egl19_Scm2: float = 0.104385 * 1e-9 / 1123.84e-8,  # AVAL rescaled
    eca_mV: float = 60.0,
    v_init_mV: float = -60.0,
):
    """Brian2 factory returning leak+EGL-19 cell with voltage clamp interface."""

    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope, defaultclock,
        )
        start_scope()
        prefs.codegen.target = "cython"

        # Cell-level eqs with v in mV (numeric), I_total in pA (numeric).
        # surf_cm2_param is the surface area used to convert ica_mAcm2 → I_total_pA.
        # We use v as a state variable (volt) but expose v_mV for channel-side use.
        cell_eqs = f"""
        # v_mV is just numerical mV value of v (for channel-side eqs that work in mV-numeric)
        v_mV = v / mV : 1
        # Total ica = sum of channel ica (we have just EGL-19 here)
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        # Membrane current density: leak + ica (mA/cm²-numeric)
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_total_mAcm2 = i_leak_mAcm2 + ica_mAcm2 : 1
        # I_total in pA: × surf (cm²) × 1e-3 (mA→A) × 1e12 (A→pA)
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA : amp
        # V dynamics (membrane equation) — only used outside clamp.
        # During voltage clamp the network_operation forces v to the held value.
        # Capacitance: C_pF = cm_uFcm2 * surf_cm2 * 1e6 (pF)
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        # Cell parameters:
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + EGL19_EQS

        from brian2 import pA, pF
        G = NeuronGroup(1, cell_eqs, method="euler")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_init_states(G, v_mV=v_init_mV)

        clamp = {"v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            G.v = clamp["v_target_mV"] * mV

        mon = StateMonitor(G, ["v", "I_total", "ica_egl19_mAcm2",
                                "i_leak_mAcm2", "m_egl19", "h_egl19"],
                            record=True)
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

    return _factory


def build_neuron_egl19_cell(
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_egl19_Scm2: float = 0.104385 * 1e-9 / 1123.84e-8,
    eca_mV: float = 60.0,
) -> NEURONReference:
    """Construct a NEURON section with [leak + egl19] only."""
    custom_spec = {
        "channels": ["leak", "egl19"],
        "params": {
            ("leak", "gbar"): g_leak_Scm2,
            ("leak", "e"): e_leak_mV,
            ("egl19", "gbar"): g_egl19_Scm2,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": eca_mV,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
    }
    return NEURONReference("custom", custom_spec=custom_spec)


def main():
    print("=== CP2: EGL-19 isolated voltage-clamp validation ===\n")

    # AVAL geometry + AVAL leak/egl19 gbars
    surf = 1123.84e-8
    g_leak_Scm2 = 0.150164 * 1e-9 / surf
    g_egl19_Scm2 = 0.104385 * 1e-9 / surf
    e_leak_mV = -39.0
    cm_uFcm2 = 0.859551

    print(f"Cell config: surf={surf:.3e} cm², cm={cm_uFcm2:.3f} μF/cm²")
    print(f"             g_leak={g_leak_Scm2:.3e} S/cm², e_leak={e_leak_mV} mV")
    print(f"             g_egl19={g_egl19_Scm2:.3e} S/cm², eca=60 mV\n")

    print("Building NEURON reference [leak + egl19]...")
    nref = build_neuron_egl19_cell(
        surf_cm2=surf, cm_uFcm2=cm_uFcm2,
        g_leak_Scm2=g_leak_Scm2, e_leak_mV=e_leak_mV,
        g_egl19_Scm2=g_egl19_Scm2,
    )

    print("Building Brian2 factory [leak + egl19]...")
    factory = build_brian2_egl19_cell_factory(
        surf_cm2=surf, cm_uFcm2=cm_uFcm2,
        g_leak_Scm2=g_leak_Scm2, e_leak_mV=e_leak_mV,
        g_egl19_Scm2=g_egl19_Scm2,
    )

    holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    print(f"\nRunning voltage-clamp comparison at {len(holds)} holds...")
    print("(Using 50 ms pre-step at -60 mV + 2 ms skip on initial capacitive transient)")
    result = voltage_clamp_compare_v2(
        factory, nref, holds,
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

    print(f"\nResults: panel_pass={result['panel_pass']}  "
          f"holds_passing={result['n_holds_passing']}/{result['n_holds']}  "
          f"frac={result['fraction_passing']:.3f}")

    print("\nPer-hold detail:")
    for s, e in zip(result["per_step"], result["evaluation"]["per_step_evaluations"]):
        peak_div = e["feature_results"]["peak_I_pA"]["divergence"]
        ss_div = e["feature_results"]["ss_I_pA"]["divergence"]
        peak_pass = e["feature_results"]["peak_I_pA"]["pass"]
        ss_pass = e["feature_results"]["ss_I_pA"]["pass"]
        print(f"  hold={s['hold_mV']:+6.1f} mV  "
              f"brian2_peak={s['brian2_peak_I_pA']:+8.2f} pA  "
              f"nrn_peak={s['ref_peak_I_pA']:+8.2f} pA  "
              f"peak_div={peak_div:.3f} ({'P' if peak_pass else 'F'})  "
              f"brian2_ss={s['brian2_ss_I_pA']:+8.2f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+8.2f}  "
              f"ss_div={ss_div:.3f} ({'P' if ss_pass else 'F'})")

    # IV curve diagnostic
    iv_brian2 = [s["brian2_ss_I_pA"] for s in result["per_step"]]
    iv_neuron = [s["ref_ss_I_pA"] for s in result["per_step"]]
    print("\nIV curve (SS):")
    print("  V (mV) | Brian2 (pA) | NEURON (pA) | Δ (pA)")
    for h, ib, in_ in zip(holds, iv_brian2, iv_neuron):
        print(f"  {h:+6.1f} | {ib:+10.2f}  | {in_:+10.2f}  | {ib - in_:+8.2f}")

    nref.cleanup()

    # Save results
    out_path = Path(__file__).parent / "artifacts" / "egl19_validation_results.json"
    serializable = {
        "panel_pass": result["panel_pass"],
        "n_holds": result["n_holds"],
        "n_holds_passing": result["n_holds_passing"],
        "fraction_passing": result["fraction_passing"],
        "tolerance_metric": result["tolerance_metric"],
        "per_step": result["per_step"],
        "evaluation": {
            "per_step_evaluations": result["evaluation"]["per_step_evaluations"],
            "per_feature_peak": result["evaluation"]["per_feature_peak"],
        },
        "warnings": result["warnings"],
        "iv_curve": {
            "holds_mV": holds,
            "brian2_ss_pA": iv_brian2,
            "neuron_ss_pA": iv_neuron,
        },
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nWrote: {out_path}")

    return 0 if result["panel_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
