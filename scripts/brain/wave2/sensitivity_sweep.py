"""
Density-sensitivity sweep for Phase F Component 2b (Mellem 2008 plateau).

Purpose
-------
Phase β run #2 produced 2a-pass / 2b-fail (amplitude 46.8 mV vs target 15-25 mV;
duration 21.4 ms vs target 400-800 ms). Per the Wave 2 architectural plan this
triggers Condition 6 (channels-correct, architecture-insufficient).

Before authorizing a 3-4 week morphology integration fork the user wants a
density-sensitivity check to distinguish:
- "Wrong densities masquerading as architecture-insufficient" (Condition 6
   false alarm — density-tunable)
- "True architectural insufficiency" (Condition 6 confirmed; morphology fork
   warranted)

Sweep design
------------
Two-axis grid over the 5 non-Nicoletti-AVA channel densities:

  Axis 1 (terminator): scale [SLO-1 isolated, SLO-1+EGL-19 coupled] together
      by factor in {0.5, 1.0, 2.0, 4.0}.
  Axis 2 (voltage-gated K): scale [SHK-1, SHL-1, KQT-3] together by factor
      in {0.5, 1.0, 2.0, 4.0}.

EGL-19, NCA, and leak are held at Nicoletti's published AVAL values throughout.

Mellem 2008 protocol (identical to Phase F 2b):
  - 200 ms settle at I = 0
  - 100 ms × 50 pA injection
  - 1500 ms post-stim recovery

For each combination we record:
  - plateau amplitude (mV, V_at_end_of_stim - V_base)
  - plateau duration (ms; time post-stim until V drops below V_base + 5 mV)
  - peak V during stim
  - release-tau ratio (release_tau_ms / leak_tau_ms) and architectural_signature
  - finite/instability flag

Verdict assignment is performed at the end based on whether any combination
achieves amplitude in [15, 25] mV AND duration in [400, 800] ms (or is
"close" — see verdict logic below).
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

# Import constants and channel densities consistent with Phase F.
from validate_phase_f_gate2 import (
    AVA_SURF_CM2,
    AVA_CM_UFCM2,
    AVA_E_LEAK_MV,
)
from channels import egl19 as egl19_mod
from channels import nca as nca_mod
from channels import shk1 as shk1_mod
from channels import shl1 as shl1_mod
from channels import kqt3 as kqt3_mod
from channels import slo1_iso as slo1iso_mod
from channels import slo1_egl19_coupled as slo1egl19_mod


# Baseline (factor=1.0) densities from Phase F Component 2b
BASELINE_DENSITIES = {
    # Principled (NOT swept):
    "g_egl19_Scm2": 0.104385e-9 / AVA_SURF_CM2,   # ~ 9.288e-6
    "g_nca_Scm2": 0.0,                            # AVAL g0
    "g_leak_Scm2": 0.150164e-9 / AVA_SURF_CM2,    # ~ 1.336e-5
    # Terminator block (Axis 1):
    "g_slo1iso_Scm2": 1.0e-9 / 65.89e-8,          # ~ 1.518e-3
    "g_slo1egl19_Scm2": 0.92e-9 / 65.89e-8,       # ~ 1.396e-3
    # Voltage-gated K block (Axis 2):
    "g_shk1_Scm2": 1e-4,
    "g_shl1_Scm2": 1e-4,                          # NOTE: doc says shl1 1e-4 but
    # Phase F code used 0.5e-9/65.89e-8 = 7.589e-4 for shl1. We follow the CODE
    # (validate_phase_f_gate2.py) since that is what produced the 46.8 mV / 21.4
    # ms numbers — SHL1 baseline is 7.589e-4, not 1e-4.
    "g_kqt3_Scm2": 1e-4,
}
# Override SHL-1 baseline to match the Phase F code (the empirical 2b run).
BASELINE_DENSITIES["g_shl1_Scm2"] = 0.5e-9 / 65.89e-8   # ~ 7.589e-4

# Constants
ECA_MV = 60.0
EK_MV = -80.0
V_INIT_MV = -60.0
CAI_MM_STATIC = 5e-5

# Mellem-protocol parameters (matches Phase F)
SETTLE_MS = 200.0
STIM_MS = 100.0
RECOVER_MS = 1500.0
STIM_AMP_PA = 50.0
DT_MS = 0.025

# Targets
TARGET_AMP_RANGE = (15.0, 25.0)
TARGET_DUR_RANGE = (400.0, 800.0)


def build_brian2_ava_2b_sweep(term_factor: float, kv_factor: float):
    """Construct a Brian2 AVA cell with terminator and Kv densities scaled.

    Mirrors validate_phase_f_gate2.build_brian2_ava_2b but parametrizes the
    sweep axes.
    """
    g_slo1iso_Scm2 = BASELINE_DENSITIES["g_slo1iso_Scm2"] * term_factor
    g_slo1egl19_Scm2 = BASELINE_DENSITIES["g_slo1egl19_Scm2"] * term_factor
    g_shk1_Scm2 = BASELINE_DENSITIES["g_shk1_Scm2"] * kv_factor
    g_shl1_Scm2 = BASELINE_DENSITIES["g_shl1_Scm2"] * kv_factor
    g_kqt3_Scm2 = BASELINE_DENSITIES["g_kqt3_Scm2"] * kv_factor

    g_egl19_Scm2 = BASELINE_DENSITIES["g_egl19_Scm2"]
    g_nca_Scm2 = BASELINE_DENSITIES["g_nca_Scm2"]
    g_leak_Scm2 = BASELINE_DENSITIES["g_leak_Scm2"]

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
        ik_total_mAcm2 = ik_shk1_mAcm2 + ik_shl1_mAcm2 + ik_kqt3_mAcm2 + ik_slo1iso_mAcm2 + ik_slo1egl19_mAcm2 : 1
        i_leak_mAcm2 = g_leak_Scm2 * (v_mV - e_leak_mV) : 1
        i_nca_mAcm2 = ik_nca_mAcm2 : 1
        i_total_mAcm2 = i_leak_mAcm2 + i_nca_mAcm2 + ica_mAcm2 + ik_total_mAcm2 : 1
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
        G.e_leak_mV = AVA_E_LEAK_MV
        G.surf_cm2_param = AVA_SURF_CM2
        G.cm_uFcm2_param = AVA_CM_UFCM2
        G.v = V_INIT_MV * mV
        G.I_inj = 0 * pA

        egl19_mod.egl19_apply_params(G, gbar_Scm2=g_egl19_Scm2, eca_mV=ECA_MV)
        egl19_mod.egl19_init_states(G, v_mV=V_INIT_MV)
        nca_mod.nca_apply_params(G, gbar_Scm2=g_nca_Scm2)
        shk1_mod.shk1_apply_params(G, gbar_Scm2=g_shk1_Scm2, ek_mV=EK_MV)
        shk1_mod.shk1_init_states(G, v_mV=V_INIT_MV)
        shl1_mod.shl1_apply_params(G, gbar_Scm2=g_shl1_Scm2, ek_mV=EK_MV)
        shl1_mod.shl1_init_states(G, v_mV=V_INIT_MV)
        kqt3_mod.kqt3_apply_params(G, gbar_Scm2=g_kqt3_Scm2, ek_mV=EK_MV)
        kqt3_mod.kqt3_init_states(G, v_mV=V_INIT_MV)
        slo1iso_mod.slo1iso_apply_params(G, gbar_Scm2=g_slo1iso_Scm2, ek_mV=EK_MV,
                                          cai_mM=CAI_MM_STATIC)
        slo1iso_mod.slo1iso_init_states(G, v_mV=V_INIT_MV)
        slo1egl19_mod.slo1egl19_apply_params(G, gbar_Scm2=g_slo1egl19_Scm2, ek_mV=EK_MV)
        slo1egl19_mod.slo1egl19_init_states(G, v_mV=V_INIT_MV)

        mon = StateMonitor(G, ["v", "I_total", "I_inj"], record=True)
        net = Network(G, mon)

        return {"group": G, "monitor": mon, "network": net}

    return _factory


def _leak_tau_ms() -> float:
    """Pure-leak τ_m for the AVA cell with Nicoletti g_leak.

    τ_m = cm / g_leak  (with cm in μF/cm² and g_leak in S/cm²).
    Converting: τ_s = (cm * 1e-6 F/cm²) / (g S/cm²); τ_ms = τ_s * 1e3.
    Net factor: 1e-3.
    """
    g_leak = BASELINE_DENSITIES["g_leak_Scm2"]  # S/cm²
    return (AVA_CM_UFCM2 / g_leak) * 1e-3  # ~64 ms


def run_one(term_factor: float, kv_factor: float) -> dict:
    """Run one sweep cell at given factor combination. Return measurements."""
    from brian2 import ms, defaultclock, pA

    factory = build_brian2_ava_2b_sweep(term_factor, kv_factor)
    bundle = factory()
    G = bundle["group"]
    net = bundle["network"]
    mon = bundle["monitor"]

    defaultclock.dt = DT_MS * ms

    issues = []
    finite_ok = True

    try:
        G.I_inj = 0 * pA
        net.run(SETTLE_MS * ms)
        G.I_inj = STIM_AMP_PA * pA
        net.run(STIM_MS * ms)
        G.I_inj = 0 * pA
        net.run(RECOVER_MS * ms)
    except Exception as e:
        issues.append(f"exception_during_run: {e}")
        return {
            "term_factor": term_factor,
            "kv_factor": kv_factor,
            "g_slo1iso_Scm2": BASELINE_DENSITIES["g_slo1iso_Scm2"] * term_factor,
            "g_slo1egl19_Scm2": BASELINE_DENSITIES["g_slo1egl19_Scm2"] * term_factor,
            "g_shk1_Scm2": BASELINE_DENSITIES["g_shk1_Scm2"] * kv_factor,
            "g_shl1_Scm2": BASELINE_DENSITIES["g_shl1_Scm2"] * kv_factor,
            "g_kqt3_Scm2": BASELINE_DENSITIES["g_kqt3_Scm2"] * kv_factor,
            "V_base_mV": float("nan"),
            "V_peak_stim_mV": float("nan"),
            "V_at_end_of_stim_mV": float("nan"),
            "plateau_amp_mV": float("nan"),
            "plateau_duration_ms": float("nan"),
            "tau_release_ms": float("nan"),
            "leak_tau_ms": _leak_tau_ms(),
            "release_tau_ratio": float("nan"),
            "architectural_signature": "exception",
            "amp_pass": False,
            "dur_pass": False,
            "arch_pass": False,
            "finite_ok": False,
            "issues": issues,
        }

    t = np.array(mon.t) * 1e3  # ms
    V = np.array(mon.v[0]) * 1e3  # mV

    if not np.all(np.isfinite(V)):
        finite_ok = False
        issues.append("non_finite_V")

    # Baseline (last 100 ms of settle)
    base_mask = (t > 100) & (t < SETTLE_MS)
    V_base = float(np.mean(V[base_mask])) if base_mask.any() else float(V[0])

    # Peak during stim
    stim_t0 = SETTLE_MS
    stim_t1 = SETTLE_MS + STIM_MS
    stim_mask = (t > stim_t0) & (t < stim_t1)
    V_peak_stim = float(np.max(V[stim_mask])) if stim_mask.any() else V_base

    # Plateau amplitude: V at end of stim - baseline
    end_idx = int(np.argmin(np.abs(t - stim_t1)))
    V_at_end = float(V[end_idx])
    plateau_amp = V_at_end - V_base

    # Plateau duration: post-stim time until V < V_base + 5 mV
    post_mask = t > stim_t1
    t_post = t[post_mask]
    V_post = V[post_mask]
    threshold = V_base + 5.0
    decay_idx = np.where(V_post < threshold)[0]
    if len(decay_idx) > 0:
        plateau_duration = float(t_post[decay_idx[0]] - stim_t1)
    else:
        # V never crossed below threshold within recovery window
        plateau_duration = float(t_post[-1] - stim_t1) if len(t_post) > 0 else -1.0
        issues.append("plateau_did_not_terminate")

    # Release-tau: fit exponential decay to V_post until close to V_base.
    tau_release_ms = float("nan")
    arch_signature = "unknown"
    leak_tau_ms = _leak_tau_ms()
    if finite_ok and len(V_post) > 50 and plateau_amp > 0.5:
        # Fit window: from stim end to either threshold-crossing or 800 ms post,
        # whichever is shorter.
        fit_end_t = stim_t1 + min(800.0, t_post[-1] - stim_t1)
        fit_mask = (t_post >= stim_t1) & (t_post <= fit_end_t)
        V_fit = V_post[fit_mask]
        t_fit = t_post[fit_mask] - stim_t1
        delta = V_fit - V_base
        # use only positive delta (depolarized w.r.t. baseline) above noise floor
        valid = delta > 0.5
        if valid.sum() >= 5:
            log_delta = np.log(delta[valid])
            try:
                slope, _ = np.polyfit(t_fit[valid], log_delta, 1)
                if slope < 0:
                    tau_release_ms = float(-1.0 / slope)
                else:
                    tau_release_ms = float("inf")
            except Exception as e:
                issues.append(f"release_tau_fit_failed: {e}")

        if np.isfinite(tau_release_ms) and leak_tau_ms > 0:
            ratio = tau_release_ms / leak_tau_ms
            if ratio < 0.6:
                arch_signature = "active_termination"
            elif ratio < 1.4:
                arch_signature = "leak_dominated"
            else:
                arch_signature = "no_termination"
        else:
            ratio = float("nan")
    else:
        ratio = float("nan")

    amp_pass = TARGET_AMP_RANGE[0] <= plateau_amp <= TARGET_AMP_RANGE[1]
    dur_pass = TARGET_DUR_RANGE[0] <= plateau_duration <= TARGET_DUR_RANGE[1]
    arch_pass = bool(amp_pass and dur_pass)

    return {
        "term_factor": term_factor,
        "kv_factor": kv_factor,
        "g_slo1iso_Scm2": BASELINE_DENSITIES["g_slo1iso_Scm2"] * term_factor,
        "g_slo1egl19_Scm2": BASELINE_DENSITIES["g_slo1egl19_Scm2"] * term_factor,
        "g_shk1_Scm2": BASELINE_DENSITIES["g_shk1_Scm2"] * kv_factor,
        "g_shl1_Scm2": BASELINE_DENSITIES["g_shl1_Scm2"] * kv_factor,
        "g_kqt3_Scm2": BASELINE_DENSITIES["g_kqt3_Scm2"] * kv_factor,
        "V_base_mV": V_base,
        "V_peak_stim_mV": V_peak_stim,
        "V_at_end_of_stim_mV": V_at_end,
        "plateau_amp_mV": plateau_amp,
        "plateau_duration_ms": plateau_duration,
        "tau_release_ms": tau_release_ms,
        "leak_tau_ms": leak_tau_ms,
        "release_tau_ratio": ratio,
        "architectural_signature": arch_signature,
        "amp_pass": bool(amp_pass),
        "dur_pass": bool(dur_pass),
        "arch_pass": arch_pass,
        "finite_ok": bool(finite_ok),
        "issues": issues,
    }


def assign_verdict(rows: list[dict]) -> tuple[str, dict]:
    """Apply the verdict logic from the work block spec."""
    finite = [r for r in rows if r["finite_ok"]]
    if not finite:
        return "VERDICT_NEITHER_TUNABLE", {"reason": "all_runs_unstable"}

    any_arch_pass = [r for r in finite if r["arch_pass"]]
    amp_pass_rows = [r for r in finite if r["amp_pass"]]
    dur_pass_rows = [r for r in finite if r["dur_pass"]]
    max_dur = max((r["plateau_duration_ms"] for r in finite), default=-1)
    max_dur_with_amp_pass = max(
        (r["plateau_duration_ms"] for r in amp_pass_rows), default=-1
    )

    if any_arch_pass:
        verdict = "VERDICT_DENSITY_TUNABLE"
        details = {
            "passing_combinations": [
                {"term_factor": r["term_factor"], "kv_factor": r["kv_factor"]}
                for r in any_arch_pass
            ],
            "n_passing": len(any_arch_pass),
        }
        return verdict, details

    # No combination passes both.
    if amp_pass_rows and max_dur_with_amp_pass <= 200.0:
        verdict = "VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS"
        details = {
            "n_amp_pass": len(amp_pass_rows),
            "max_dur_among_amp_pass_ms": max_dur_with_amp_pass,
        }
        return verdict, details

    if amp_pass_rows and max_dur_with_amp_pass > 200.0:
        # Some amp-passing combos extend into 200-400 ms range — close to target
        # but not in it. Treat as duration_fails but flag the closeness.
        verdict = "VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS"
        details = {
            "n_amp_pass": len(amp_pass_rows),
            "max_dur_among_amp_pass_ms": max_dur_with_amp_pass,
            "note": "some amp-passing combos reach 200-400 ms duration — close but not pass",
        }
        return verdict, details

    if dur_pass_rows:
        verdict = "VERDICT_DURATION_TUNABLE_AMPLITUDE_FAILS"
        details = {
            "n_dur_pass": len(dur_pass_rows),
            "passing_durations_with_failed_amp": [
                {"term": r["term_factor"], "kv": r["kv_factor"],
                 "amp": r["plateau_amp_mV"], "dur": r["plateau_duration_ms"]}
                for r in dur_pass_rows
            ],
        }
        return verdict, details

    verdict = "VERDICT_NEITHER_TUNABLE"
    details = {
        "max_dur_overall_ms": max_dur,
        "min_amp_overall_mV": min(r["plateau_amp_mV"] for r in finite),
        "max_amp_overall_mV": max(r["plateau_amp_mV"] for r in finite),
    }
    return verdict, details


def main():
    print("############################################################")
    print("# Density-sensitivity sweep — Phase F Component 2b")
    print("############################################################\n")

    term_factors = [0.5, 1.0, 2.0, 4.0]
    kv_factors = [0.5, 1.0, 2.0, 4.0]
    print(f"Sweep grid: {len(term_factors)} terminator × {len(kv_factors)} Kv "
          f"= {len(term_factors)*len(kv_factors)} runs")
    print(f"  terminator factors (SLO-1 iso + SLO-1+EGL-19): {term_factors}")
    print(f"  Kv factors (SHK-1 + SHL-1 + KQT-3): {kv_factors}")
    print(f"  baseline densities (factor=1.0):")
    print(f"    g_slo1iso = {BASELINE_DENSITIES['g_slo1iso_Scm2']:.3e} S/cm²")
    print(f"    g_slo1egl19 = {BASELINE_DENSITIES['g_slo1egl19_Scm2']:.3e}")
    print(f"    g_shk1 = {BASELINE_DENSITIES['g_shk1_Scm2']:.3e}")
    print(f"    g_shl1 = {BASELINE_DENSITIES['g_shl1_Scm2']:.3e}")
    print(f"    g_kqt3 = {BASELINE_DENSITIES['g_kqt3_Scm2']:.3e}")
    print(f"  principled (NOT swept):")
    print(f"    g_leak = {BASELINE_DENSITIES['g_leak_Scm2']:.3e}")
    print(f"    g_egl19 = {BASELINE_DENSITIES['g_egl19_Scm2']:.3e}")
    print(f"    g_nca = {BASELINE_DENSITIES['g_nca_Scm2']:.3e}")
    print(f"  leak τ_m = {_leak_tau_ms():.1f} ms")
    print(f"  Mellem protocol: {SETTLE_MS} ms settle, {STIM_AMP_PA} pA × "
          f"{STIM_MS} ms, {RECOVER_MS} ms recover")
    print(f"  Targets: amp [{TARGET_AMP_RANGE[0]}, {TARGET_AMP_RANGE[1]}] mV, "
          f"dur [{TARGET_DUR_RANGE[0]}, {TARGET_DUR_RANGE[1]}] ms")
    print()

    rows = []
    n = 0
    total = len(term_factors) * len(kv_factors)
    t_start = time.time()
    for tf in term_factors:
        for kf in kv_factors:
            n += 1
            t0 = time.time()
            try:
                row = run_one(tf, kf)
            except Exception as e:
                row = {
                    "term_factor": tf, "kv_factor": kf,
                    "finite_ok": False,
                    "issues": [f"top_level_exception: {e}"],
                    "amp_pass": False, "dur_pass": False, "arch_pass": False,
                    "plateau_amp_mV": float("nan"),
                    "plateau_duration_ms": float("nan"),
                    "V_base_mV": float("nan"),
                    "V_peak_stim_mV": float("nan"),
                    "V_at_end_of_stim_mV": float("nan"),
                    "tau_release_ms": float("nan"),
                    "release_tau_ratio": float("nan"),
                    "architectural_signature": "exception",
                    "leak_tau_ms": _leak_tau_ms(),
                }
            elapsed = time.time() - t0
            print(f"  [{n}/{total}] term={tf:.2f} kv={kf:.2f}  "
                  f"amp={row['plateau_amp_mV']:7.2f} mV  "
                  f"dur={row['plateau_duration_ms']:7.1f} ms  "
                  f"τrel={row['tau_release_ms']:.1f} ms  "
                  f"sig={row['architectural_signature']:18s}  "
                  f"arch_pass={row['arch_pass']}  "
                  f"({elapsed:.1f}s)")
            if row.get("issues"):
                for iss in row["issues"]:
                    print(f"      issue: {iss}")
            rows.append(row)

    total_time = time.time() - t_start
    print(f"\nSweep complete in {total_time:.1f}s")

    verdict, details = assign_verdict(rows)
    print(f"\n############################################################")
    print(f"# Verdict: {verdict}")
    print(f"############################################################")
    print(json.dumps(details, indent=2))

    # Save raw results
    out_dir = Path(__file__).parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "density_sensitivity_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "verdict": verdict,
            "verdict_details": details,
            "sweep_axes": {
                "terminator_factors": term_factors,
                "kv_factors": kv_factors,
            },
            "baseline_densities": BASELINE_DENSITIES,
            "principled_unchanged": {
                "g_egl19_Scm2": BASELINE_DENSITIES["g_egl19_Scm2"],
                "g_nca_Scm2": BASELINE_DENSITIES["g_nca_Scm2"],
                "g_leak_Scm2": BASELINE_DENSITIES["g_leak_Scm2"],
            },
            "protocol": {
                "settle_ms": SETTLE_MS,
                "stim_ms": STIM_MS,
                "recover_ms": RECOVER_MS,
                "stim_amp_pA": STIM_AMP_PA,
                "dt_ms": DT_MS,
            },
            "targets": {
                "amplitude_mV": list(TARGET_AMP_RANGE),
                "duration_ms": list(TARGET_DUR_RANGE),
            },
            "leak_tau_ms": _leak_tau_ms(),
            "rows": rows,
        }, f, indent=2)
    print(f"\n→ saved {json_path}")
    return verdict, details, rows


if __name__ == "__main__":
    main()
