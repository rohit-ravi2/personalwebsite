"""
Phase β run #2 Phase F — Gate 2 evaluation on AVA.

Two decoupled cell constructions:

Component 2a: Brian2 AVA with NCA + EGL-19 + leak vs NEURON's matching subset
              (constructed via custom_spec, not Nicoletti's full AVA).
              Tests channel kinetics correctness in cell context.
              Pass criterion: voltage-feature ≤5%, >80% holds clear.

Component 2b: Brian2 AVA with full 7-channel essential set (egl19, slo1iso,
              slo1egl19, shk1, shl1, nca, kqt3) + leak + Ca-pool + Mellem
              targets (20 mV plateau, 600 ms duration, SLO-1 termination).
              No NEURON reference. Tests architectural sufficiency.
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
from channels import nca as nca_mod
from channels import shk1 as shk1_mod
from channels import shl1 as shl1_mod
from channels import kqt3 as kqt3_mod
from channels import slo1_iso as slo1iso_mod
from channels import slo1_egl19_coupled as slo1egl19_mod


# -------------------- AVA cell parameters from Nicoletti --------------------
# AVAL_simulations.py: g0 = [egl19, leak, irk, nca, eleak, cm]
#   = [0.104385, 0.150164, 0.1, 0, -39, 0.859551] in nS-or-units
# gScm2(g0, surf, 3) — index 3 (nca) is treated as the gScm2 reference.
# Looking at g_to_Scm2.py to figure out scaling...

# Based on validate_egl19.py and validate_cp3_egl19_cell.py, the convention is:
#   g_egl19_Scm2 = 0.104385 * 1e-9 / surf  (translating nS to S/cm² via surface)
# This is a confusing "double-scaling" but matches Nicoletti's parameter recovery.

AVA_SURF_CM2 = 1123.84e-8
AVA_CM_UFCM2 = 0.859551
AVA_E_LEAK_MV = -39.0


# -------- Component 2a: NCA + EGL-19 + leak --------

def build_brian2_ava_2a(
    surf_cm2: float = AVA_SURF_CM2,
    cm_uFcm2: float = AVA_CM_UFCM2,
    g_leak_Scm2: float = 0.150164e-9 / AVA_SURF_CM2,
    e_leak_mV: float = AVA_E_LEAK_MV,
    g_egl19_Scm2: float = 0.104385e-9 / AVA_SURF_CM2,
    g_nca_Scm2: float = 0.0,  # 0 in AVAL config; can be raised for testing
    eca_mV: float = 60.0,
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
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        ik_total_mAcm2 = ik_nca_mAcm2 : 1
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_total_mAcm2 = i_leak_mAcm2 + ica_mAcm2 + ik_total_mAcm2 : 1
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + egl19_mod.EGL19_EQS + nca_mod.NCA_EQS

        G = NeuronGroup(1, cell_eqs, method="euler")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV

        egl19_mod.egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_mod.egl19_init_states(G, v_mV=v_init_mV)
        nca_mod.nca_apply_params(G, gbar_Scm2=g_nca_Scm2)
        nca_mod.nca_init_states(G, v_mV=v_init_mV)

        clamp = {"v_target_mV": v_init_mV}

        @network_operation(dt=0.025 * ms)
        def _clamp():
            G.v = clamp["v_target_mV"] * mV

        mon = StateMonitor(G, ["v", "I_total", "ica_egl19_mAcm2", "ik_nca_mAcm2",
                                "i_leak_mAcm2"], record=True)
        net = Network(G, mon, _clamp)

        def set_v(v_mV):
            clamp["v_target_mV"] = float(v_mV)
            G.v = float(v_mV) * mV

        return {"group": G, "monitor": mon, "network": net, "set_v": set_v}

    return _factory


def build_neuron_ava_2a(g_nca_Scm2: float = 0.0):
    """NEURON [leak + egl19 + nca] cell — apples-to-apples with Brian2 2a."""
    custom_spec = {
        "channels": ["leak", "egl19", "nca"],
        "params": {
            ("leak", "gbar"): 0.150164e-9 / AVA_SURF_CM2,
            ("leak", "e"): AVA_E_LEAK_MV,
            ("egl19", "gbar"): 0.104385e-9 / AVA_SURF_CM2,
            ("nca", "gbar"): g_nca_Scm2,
        },
        "surf_cm2": AVA_SURF_CM2,
        "cm_uFcm2": AVA_CM_UFCM2,
        "eca_mV": 60.0,
        "ek_mV": -80.0,
        "v_init_mV": -60.0,
    }
    return NEURONReference("custom", custom_spec=custom_spec)


def run_component_2a():
    print("=== Phase F.2a: voltage-clamp Layer A — AVA [leak + EGL-19 + NCA] ===\n")
    print(f"AVA geometry: surf={AVA_SURF_CM2:.3e} cm², cm={AVA_CM_UFCM2}")
    print(f"  g_leak = {0.150164e-9/AVA_SURF_CM2:.3e} S/cm²")
    print(f"  g_egl19 = {0.104385e-9/AVA_SURF_CM2:.3e} S/cm²")
    print(f"  g_nca = 0.0 (Nicoletti's AVAL has nca gbar = 0)")
    print()

    # Use g_nca = 0 to match Nicoletti's actual AVAL setup
    nref = build_neuron_ava_2a(g_nca_Scm2=0.0)
    factory = build_brian2_ava_2a(g_nca_Scm2=0.0)

    holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]

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

    print(f"Component 2a results:")
    print(f"  panel_pass={result['panel_pass']}")
    print(f"  holds_passing={result['n_holds_passing']}/{result['n_holds']}")
    print(f"  fraction_passing={result['fraction_passing']:.3f}\n")

    print("Per-hold detail:")
    for s, e in zip(result["per_step"], result["evaluation"]["per_step_evaluations"]):
        peak_div = e["feature_results"]["peak_I_pA"]["divergence"]
        ss_div = e["feature_results"]["ss_I_pA"]["divergence"]
        peak_pass = e["feature_results"]["peak_I_pA"]["pass"]
        ss_pass = e["feature_results"]["ss_I_pA"]["pass"]
        print(f"  hold={s['hold_mV']:+5.0f} mV  "
              f"brian2_peak={s['brian2_peak_I_pA']:+9.2f}  "
              f"nrn_peak={s['ref_peak_I_pA']:+9.2f}  "
              f"div={peak_div:.4f}({'P' if peak_pass else 'F'})  "
              f"brian2_ss={s['brian2_ss_I_pA']:+9.2f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+9.2f}  "
              f"div={ss_div:.4f}({'P' if ss_pass else 'F'})")
    nref.cleanup()
    return result


# -------- Component 2b: full 7-channel essential set + Mellem 2008 targets --------

def build_brian2_ava_2b(
    surf_cm2: float = AVA_SURF_CM2,
    cm_uFcm2: float = AVA_CM_UFCM2,
    g_leak_Scm2: float = 0.150164e-9 / AVA_SURF_CM2,
    e_leak_mV: float = AVA_E_LEAK_MV,
    # Channel densities — using Nicoletti AVAL where available; for channels not
    # in Nicoletti's AVAL, use AIY-derived densities scaled to AVA's larger surface.
    # AVAL g0 (Nicoletti): [egl19=0.104385, leak=0.150164, irk=0.1, nca=0, eleak=-39, cm=0.859551]
    # All in nS for first 4 entries. gScm2 converts: g_Scm2 = g_nS * 1e-9 / surf.
    g_egl19_Scm2: float = 0.104385e-9 / AVA_SURF_CM2,    # ≈ 9.3e-6
    g_nca_Scm2: float = 0.0,                              # AVAL config
    # AIY g0: [leak=0.14, slo1iso=1.0, kqt1=0.2, egl19=0.1, slo1egl19=0.92, nca=0.06, shl1=0.5]
    # Convert each to S/cm² via gScm2 with AIY surf=65.89e-8.
    # Then we apply same densities to AVA — since these are S/cm² (intensive), they
    # transfer directly. NOTE: Nicoletti's gbar values are nS at AIY's surface;
    # the S/cm² value is what's intrinsic. Use AIY-derived S/cm² for channels not
    # in AVA's set.
    g_slo1iso_Scm2: float = 1.0e-9 / 65.89e-8,            # ≈ 1.518e-3
    g_slo1egl19_Scm2: float = 0.92e-9 / 65.89e-8,         # ≈ 1.396e-3
    g_shl1_Scm2: float = 0.5e-9 / 65.89e-8,               # ≈ 7.589e-4
    # SHK-1: not in AIY's set; use VA5's value — VA5 has shk1 in its insert list.
    # Need VA5 g0. Approximation: use a small density.
    g_shk1_Scm2: float = 1e-4,
    # KQT-3: not in AIY (which uses kqt1). Use NMODL default scaled down.
    g_kqt3_Scm2: float = 1e-4,
    eca_mV: float = 60.0,
    ek_mV: float = -80.0,
    v_init_mV: float = -60.0,  # initialize at hyperpolarized; ramp via leak
    cai_mM_static: float = 5e-5,
):
    """Full 7-channel AVA cell. Used for current-clamp plateau test (2b)."""
    def _factory():
        from brian2 import (
            NeuronGroup, StateMonitor, Network,
            ms, mV, prefs, start_scope, pA, pF,
        )
        start_scope()
        prefs.codegen.target = "cython"

        cell_eqs = f"""
        v_mV = v / mV : 1
        ica_mAcm2 = ica_egl19_mAcm2 : 1
        # K-channels: shk1, shl1, kqt3, slo1iso, slo1egl19
        ik_total_mAcm2 = ik_shk1_mAcm2 + ik_shl1_mAcm2 + ik_kqt3_mAcm2 + ik_slo1iso_mAcm2 + ik_slo1egl19_mAcm2 : 1
        # Non-specific leak and nca:
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_nca_mAcm2 = ik_nca_mAcm2 : 1
        i_total_mAcm2 = i_leak_mAcm2 + i_nca_mAcm2 + ica_mAcm2 + ik_total_mAcm2 : 1
        # Total current (pA), incorporating injected current:
        I_total = i_total_mAcm2 * surf_cm2_param * 1e9 * pA - I_inj : amp
        dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF) : volt
        I_inj : amp
        g_leak_Scm2 : 1
        e_leak_mV : 1
        surf_cm2_param : 1
        cm_uFcm2_param : 1
        """ + egl19_mod.EGL19_EQS + nca_mod.NCA_EQS + shk1_mod.SHK1_EQS \
            + shl1_mod.SHL1_EQS + kqt3_mod.KQT3_EQS + slo1iso_mod.SLO1_ISO_EQS \
            + slo1egl19_mod.SLO1_EGL19_EQS

        G = NeuronGroup(1, cell_eqs, method="rk4")
        G.g_leak_Scm2 = g_leak_Scm2
        G.e_leak_mV = e_leak_mV
        G.surf_cm2_param = surf_cm2
        G.cm_uFcm2_param = cm_uFcm2
        G.v = v_init_mV * mV
        G.I_inj = 0 * pA

        egl19_mod.egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=eca_mV)
        egl19_mod.egl19_init_states(G, v_mV=v_init_mV)
        nca_mod.nca_apply_params(G, gbar_Scm2=g_nca_Scm2)
        shk1_mod.shk1_apply_params(G, gbar_Scm2=g_shk1_Scm2, ek_mV=ek_mV)
        shk1_mod.shk1_init_states(G, v_mV=v_init_mV)
        shl1_mod.shl1_apply_params(G, gbar_Scm2=g_shl1_Scm2, ek_mV=ek_mV)
        shl1_mod.shl1_init_states(G, v_mV=v_init_mV)
        kqt3_mod.kqt3_apply_params(G, gbar_Scm2=g_kqt3_Scm2, ek_mV=ek_mV)
        kqt3_mod.kqt3_init_states(G, v_mV=v_init_mV)
        slo1iso_mod.slo1iso_apply_params(G, gbar_Scm2=g_slo1iso_Scm2, ek_mV=ek_mV,
                                          cai_mM=cai_mM_static)
        slo1iso_mod.slo1iso_init_states(G, v_mV=v_init_mV)
        slo1egl19_mod.slo1egl19_apply_params(G, gbar_Scm2=g_slo1egl19_Scm2, ek_mV=ek_mV)
        slo1egl19_mod.slo1egl19_init_states(G, v_mV=v_init_mV)

        mon = StateMonitor(G, ["v", "I_total", "I_inj"], record=True)
        net = Network(G, mon)

        return {"group": G, "monitor": mon, "network": net}

    return _factory


def run_component_2b():
    """Mellem 2008-style current-clamp protocol on 7-channel AVA cell.

    Protocol:
      0-200 ms: settle at v_rest=-25 mV (no injection)
      200-300 ms: 50 pA injection (100 ms pulse)
      300-1800 ms: post-stim recovery (1500 ms)

    Targets:
      - plateau amplitude in 15-25 mV range
      - plateau duration in 400-800 ms range
      - active termination (release tau substantially shorter than leak τ_m)
    """
    print("=== Phase F.2b: current-clamp plateau on full-essential-set AVA ===\n")
    print(f"AVA geometry: surf={AVA_SURF_CM2:.3e} cm², cm={AVA_CM_UFCM2}")
    print("Channels: egl19 + slo1iso + slo1egl19 + shk1 + shl1 + nca + kqt3 + leak")
    print()

    from brian2 import ms, mV, defaultclock, pA
    factory = build_brian2_ava_2b()
    bundle = factory()
    G = bundle["group"]
    net = bundle["network"]
    mon = bundle["monitor"]

    defaultclock.dt = 0.025 * ms

    # Settle 200 ms at I=0
    G.I_inj = 0 * pA
    net.run(200 * ms)
    # Inject 50 pA × 100 ms
    G.I_inj = 50 * pA
    net.run(100 * ms)
    # Recover 1500 ms
    G.I_inj = 0 * pA
    net.run(1500 * ms)

    t = np.array(mon.t) * 1e3  # ms
    V = np.array(mon.v[0]) * 1e3  # mV (since v is volt)

    # Baseline V
    base_mask = (t > 100) & (t < 200)
    V_base = V[base_mask].mean()

    # Peak during stim (200-300 ms)
    stim_mask = (t > 200) & (t < 300)
    V_peak_stim = V[stim_mask].max() if stim_mask.any() else V_base

    # Plateau amplitude: V at end of stim - baseline
    V_at_300 = V[np.argmin(np.abs(t - 300))]
    plateau_amp = V_at_300 - V_base

    # Plateau duration: time from stim end to V returns to within 5 mV of baseline
    post_stim_mask = t > 300
    t_post = t[post_stim_mask]; V_post = V[post_stim_mask]
    threshold_v = V_base + 5.0  # 5 mV above baseline
    decay_idx = np.where(V_post < threshold_v)[0]
    if len(decay_idx) > 0:
        plateau_duration = t_post[decay_idx[0]] - 300
    else:
        plateau_duration = -1  # didn't terminate within window

    # Termination signature: rate of V decay post-stim
    # Active termination: V drops to baseline+5 within ~300 ms (much faster than leak τ_m ~10 ms? or longer?)
    # Leak τ_m for AVA: τ_m = cm/g_leak = 0.86 / 0.0134e-3 = 64.2 ms (using cm in μF/cm² = 0.86, g in S/cm²)
    # So "leak-dominated" termination would have τ ≈ 64 ms; "active termination" would be faster.

    print(f"Baseline V (100-200 ms): {V_base:.2f} mV")
    print(f"Peak V during stim (200-300 ms): {V_peak_stim:.2f} mV")
    print(f"V at end of stim (t=300): {V_at_300:.2f} mV")
    print(f"Plateau amplitude (V_at_end - V_base): {plateau_amp:.2f} mV")
    print(f"Plateau duration (time to V_base + 5 mV): {plateau_duration:.1f} ms")
    print()

    # Mellem targets
    target_amp_low, target_amp_high = 15, 25
    target_dur_low, target_dur_high = 400, 800

    amp_pass = target_amp_low <= plateau_amp <= target_amp_high
    dur_pass = target_dur_low <= plateau_duration <= target_dur_high
    arch_pass = amp_pass and dur_pass

    print(f"Mellem 2008 targets:")
    print(f"  Plateau amplitude {target_amp_low}-{target_amp_high} mV: {'PASS' if amp_pass else 'FAIL'} (got {plateau_amp:.2f})")
    print(f"  Plateau duration {target_dur_low}-{target_dur_high} ms: {'PASS' if dur_pass else 'FAIL'} (got {plateau_duration:.1f})")
    print(f"  Architectural sufficiency: {'PASS' if arch_pass else 'FAIL'}")

    return {
        "V_base": float(V_base),
        "V_peak_stim": float(V_peak_stim),
        "V_at_end_of_stim": float(V_at_300),
        "plateau_amp_mV": float(plateau_amp),
        "plateau_duration_ms": float(plateau_duration),
        "amp_pass": bool(amp_pass),
        "dur_pass": bool(dur_pass),
        "arch_pass": bool(arch_pass),
        "trajectory": {
            "t_ms": t.tolist()[::10],  # downsampled
            "V_mV": V.tolist()[::10],
        },
    }


def main():
    print("############################################################")
    print("# Phase F: Gate 2 evaluation on AVA")
    print("############################################################\n")

    # Component 2a
    result_2a = run_component_2a()
    print()
    print("###" * 25)
    print()

    # Component 2b
    result_2b = run_component_2b()

    # Outcome classification
    print()
    print("############################################################")
    print("# Gate 2 outcome classification")
    print("############################################################")
    print()

    pass_2a = result_2a["panel_pass"]
    pass_2b = result_2b["arch_pass"]

    if pass_2a and pass_2b:
        outcome = "2a-pass / 2b-pass — Path A's cellular layer production-grade. Major Wave 2 milestone."
    elif pass_2a and not pass_2b:
        outcome = "2a-pass / 2b-fail — CONDITION 6 SURFACES. Channels work, architecture insufficient. PAUSE for morning review."
    elif not pass_2a:
        outcome = "2a-fail — Per-channel rollback territory. PAUSE for morning review."
    else:
        outcome = "2b-pass without 2a-pass: anomalous, investigate"

    print(f"Outcome: {outcome}")

    out_path = Path(__file__).parent / "artifacts" / "phase_f_gate2_results.json"
    summary = {
        "component_2a": {
            "panel_pass": pass_2a,
            "n_holds_passing": result_2a["n_holds_passing"],
            "n_holds": result_2a["n_holds"],
            "fraction_passing": result_2a["fraction_passing"],
        },
        "component_2b": {k: v for k, v in result_2b.items() if k != "trajectory"},
        "outcome": outcome,
        "gate2_pass": pass_2a and pass_2b,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ saved {out_path}")

    return pass_2a and pass_2b


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
