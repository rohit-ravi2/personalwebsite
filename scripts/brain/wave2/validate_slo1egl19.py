"""
Phase β run #2 Phase E — SLO-1+EGL-19 coupled voltage-clamp validation.

Validates the coupled-channel translation. Cell construction:
[leak + egl19 + slo1egl19] at AVAL geometry. Voltage-clamp at 11 holds.
NEURON reference uses Nicoletti's actual mod files (slo1egl19 EXTERNAL hooks
into egl19 automatically via SUFFIX matching).

Pass criterion: panel_pass >80% holds.
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
from channels import egl19 as egl19_mod
from channels import slo1_egl19_coupled as slo1egl19_mod


def build_brian2_factory(
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_egl19_Scm2: float = 0.104385e-9 / 1123.84e-8,
    g_slo1egl19_Scm2: float = 0.11,
    eca_mV: float = 60.0,
    ek_mV: float = -80.0,
    v_init_mV: float = -60.0,
):
    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network, network_operation,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        cell_eqs = f"""
        v_mV = v / mV : 1
        # Sum: ica from EGL-19 (Ca current), ik from slo1egl19 (K current).
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        ik_total_mAcm2 = ik_slo1egl19_mAcm2 : 1
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_total_mAcm2 = i_leak_mAcm2 + ica_mAcm2 + ik_total_mAcm2 : 1
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + egl19_mod.EGL19_EQS + slo1egl19_mod.SLO1_EGL19_EQS

        G = NeuronGroup(1, cell_eqs, method="euler")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV

        egl19_mod.egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_mod.egl19_init_states(G, v_mV=v_init_mV)

        slo1egl19_mod.slo1egl19_apply_params(G, gbar_Scm2=g_slo1egl19_Scm2, ek_mV=ek_mV)
        slo1egl19_mod.slo1egl19_init_states(G, v_mV=v_init_mV)

        clamp = {"v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            G.v = clamp["v_target_mV"] * mV

        mon = StateMonitor(G, ["v", "I_total", "ica_egl19_mAcm2", "ik_slo1egl19_mAcm2",
                                "i_leak_mAcm2", "m_egl19", "h_egl19", "m_slo1egl19",
                                "slo1egl19_caCALC"],
                            record=True)
        net = Network(G, mon, _clamp)

        def set_v(v_mV):
            clamp["v_target_mV"] = float(v_mV)
            G.v = float(v_mV) * mV

        return {"group": G, "monitor": mon, "network": net, "set_v": set_v}

    return _factory


def build_neuron_cell(
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_egl19_Scm2: float = 0.104385e-9 / 1123.84e-8,
    g_slo1egl19_Scm2: float = 0.11,
    eca_mV: float = 60.0,
    ek_mV: float = -80.0,
):
    custom_spec = {
        "channels": ["leak", "egl19", "slo1egl19"],
        "params": {
            ("leak", "gbar"): g_leak_Scm2,
            ("leak", "e"): e_leak_mV,
            ("egl19", "gbar"): g_egl19_Scm2,
            ("slo1egl19", "gbar"): g_slo1egl19_Scm2,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": eca_mV,
        "ek_mV": ek_mV,
        "v_init_mV": -60.0,
    }
    return NEURONReference("custom", custom_spec=custom_spec)


def main():
    print("=== Phase E: SLO-1+EGL-19 coupled voltage-clamp validation ===\n")

    surf = 1123.84e-8
    g_leak = 0.150164e-9 / surf
    g_egl19 = 0.104385e-9 / surf

    # Use modest gbar for slo1egl19 to keep currents in reasonable range
    g_slo1egl19 = 0.11

    print(f"Cell: AVAL-like [leak + egl19 + slo1egl19] @ surf={surf:.3e} cm²")
    print(f"  g_leak={g_leak:.3e}  g_egl19={g_egl19:.3e}  g_slo1egl19={g_slo1egl19:.3e}\n")

    print("Building NEURON [leak + egl19 + slo1egl19]...")
    nref = build_neuron_cell(g_egl19_Scm2=g_egl19, g_slo1egl19_Scm2=g_slo1egl19)

    print("Building Brian2 factory...")
    factory = build_brian2_factory(g_egl19_Scm2=g_egl19, g_slo1egl19_Scm2=g_slo1egl19)

    holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    print(f"\nRunning voltage-clamp comparison at {len(holds)} holds...")
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
        print(f"  hold={s['hold_mV']:+5.0f} mV  "
              f"brian2_peak={s['brian2_peak_I_pA']:+11.2f}  "
              f"nrn_peak={s['ref_peak_I_pA']:+11.2f}  "
              f"div={peak_div:.4f}({'P' if peak_pass else 'F'})  "
              f"brian2_ss={s['brian2_ss_I_pA']:+11.2f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+11.2f}  "
              f"div={ss_div:.4f}({'P' if ss_pass else 'F'})")

    nref.cleanup()

    out_path = Path(__file__).parent / "artifacts" / "slo1egl19_validation_results.json"
    serializable = {
        "panel_pass": result["panel_pass"],
        "n_holds": result["n_holds"],
        "n_holds_passing": result["n_holds_passing"],
        "fraction_passing": result["fraction_passing"],
        "tolerance_metric": result["tolerance_metric"],
        "per_step": [
            {k: v for k, v in s.items()
             if k in ("hold_mV", "brian2_peak_I_pA", "ref_peak_I_pA",
                       "brian2_ss_I_pA", "ref_ss_I_pA")}
            for s in result["per_step"]
        ],
        "per_step_evaluations": result["evaluation"]["per_step_evaluations"],
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  → saved {out_path}")
    return result["panel_pass"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
