"""
Phase β run #2 Phase D — SLO-1 isolated voltage-clamp validation.

Tests SLO-1 isolated translation against NEURON across multiple cai values
(static): 5e-5, 1e-4, 5e-4, 1e-3 mM. This confirms Ca-dependence is captured.

Per-cai run: 11 holds, voltage-clamp Layer A, current-domain tolerance.
Acceptance: panel_pass at every cai level.
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
from voltage_clamp_harness import voltage_clamp_compare_v2
from channels import slo1_iso as channel_mod


def build_brian2_slo1iso_factory(
    cai_mM: float,
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_channel_Scm2: float = 0.11,
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
        ik_total_mAcm2 = ik_slo1iso_mAcm2 : 1
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_total_mAcm2 = i_leak_mAcm2 + ik_total_mAcm2 : 1
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + channel_mod.SLO1_ISO_EQS

        G = NeuronGroup(1, cell_eqs, method="euler")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        channel_mod.slo1iso_apply_params(G, gbar_Scm2=g_channel_Scm2, ek_mV=ek_mV,
                                          cai_mM=cai_mM)
        channel_mod.slo1iso_init_states(G, v_mV=v_init_mV)

        clamp = {"v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            G.v = clamp["v_target_mV"] * mV

        mon = StateMonitor(G, ["v", "I_total", "ik_slo1iso_mAcm2", "i_leak_mAcm2", "m_slo1iso"], record=True)
        net = Network(G, mon, _clamp)

        def set_v(v_mV):
            clamp["v_target_mV"] = float(v_mV)
            G.v = float(v_mV) * mV

        return {"group": G, "monitor": mon, "network": net, "set_v": set_v}

    return _factory


def build_neuron_slo1iso_cell(
    cai_mM: float,
    surf_cm2: float = 1123.84e-8,
    cm_uFcm2: float = 0.859551,
    g_leak_Scm2: float = 0.150164 * 1e-9 / 1123.84e-8,
    e_leak_mV: float = -39.0,
    g_channel_Scm2: float = 0.11,
    ek_mV: float = -80.0,
):
    custom_spec = {
        "channels": ["leak", "slo1iso"],
        "params": {
            ("leak", "gbar"): g_leak_Scm2,
            ("leak", "e"): e_leak_mV,
            ("slo1iso", "gbar"): g_channel_Scm2,
        },
        "surf_cm2": surf_cm2,
        "cm_uFcm2": cm_uFcm2,
        "eca_mV": 60.0,
        "ek_mV": ek_mV,
        "v_init_mV": -60.0,
        "cai_mM_static": cai_mM,
    }
    return NEURONReference("custom", custom_spec=custom_spec)


def main():
    print("=== Phase D: SLO-1 isolated voltage-clamp validation ===")
    print("Testing translation across multiple static cai values.\n")

    surf = 1123.84e-8
    cm_uFcm2 = 0.859551
    g_leak = 0.150164e-9/surf
    e_leak = -39.0
    g_chan = 0.11

    holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]
    cai_values_mM = [5e-5, 1e-4, 5e-4, 1e-3]

    overall_results = {}
    for cai_mM in cai_values_mM:
        print(f"\n--- cai = {cai_mM:.4e} mM ---")
        nref = build_neuron_slo1iso_cell(cai_mM=cai_mM, surf_cm2=surf, cm_uFcm2=cm_uFcm2,
                                          g_leak_Scm2=g_leak, e_leak_mV=e_leak,
                                          g_channel_Scm2=g_chan)
        # NEURONReference doesn't natively support setting cai per-segment after init.
        # Need to inject this as a custom_spec hook OR set h.cai0_ca_ion before init.
        # Use the latter approach: set h.cai0_ca_ion which propagates to cai default.
        with _nicoletti_env():
            nref._h.cai0_ca_ion = cai_mM
            # Also explicitly set on each segment after build
            for seg in nref._soma:
                seg.cai = cai_mM

        factory = build_brian2_slo1iso_factory(cai_mM=cai_mM, surf_cm2=surf,
                                                cm_uFcm2=cm_uFcm2, g_leak_Scm2=g_leak,
                                                e_leak_mV=e_leak, g_channel_Scm2=g_chan)

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

        print(f"  panel_pass={result['panel_pass']}  "
              f"holds_passing={result['n_holds_passing']}/{result['n_holds']}")
        # Show a few representative holds
        for s, e in zip(result["per_step"], result["evaluation"]["per_step_evaluations"]):
            if s["hold_mV"] in (-60, -20, 0, 20, 40):
                pdiv = e["feature_results"]["peak_I_pA"]["divergence"]
                sdiv = e["feature_results"]["ss_I_pA"]["divergence"]
                print(f"  hold={s['hold_mV']:+5.0f} brian2_peak={s['brian2_peak_I_pA']:+10.3f} "
                      f"nrn_peak={s['ref_peak_I_pA']:+10.3f} (div={pdiv:.4f})  "
                      f"brian2_ss={s['brian2_ss_I_pA']:+10.3f} nrn_ss={s['ref_ss_I_pA']:+10.3f} (div={sdiv:.4f})")
        overall_results[f"cai_{cai_mM}"] = {
            "panel_pass": result["panel_pass"],
            "n_holds_passing": result["n_holds_passing"],
            "n_holds": result["n_holds"],
            "fraction_passing": result["fraction_passing"],
        }
        nref.cleanup()

    # Summary
    print("\n=== SLO-1 isolated validation summary ===")
    all_pass = all(r["panel_pass"] for r in overall_results.values())
    for cai, r in overall_results.items():
        print(f"  {cai}: pass={r['panel_pass']}  "
              f"holds={r['n_holds_passing']}/{r['n_holds']}")
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")

    out_path = Path(__file__).parent / "artifacts" / "slo1iso_validation_results.json"
    with open(out_path, "w") as f:
        json.dump({"overall_pass": all_pass, "per_cai": overall_results}, f, indent=2)
    print(f"  → saved {out_path}")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
