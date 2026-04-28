"""
Run Phase F Component 2b plateau protocol on the Ca-coupled AVA cell.

Tests the hypothesis (raised by `density_sensitivity_analysis.md`) that
adding the dynamic Ca-pool `caintra1` and coupling SLO-1 isolated to its
[Ca]_i state (instead of the static cai = 5e-5 mM in Phase F's published
build) is sufficient to bring plateau duration into the Mellem 2008
400-800 ms target range — without escalating to multi-compartment morphology.

Procedure
---------
1. Build the Ca-coupled cell via `ca_coupled_cell.build_brian2_ava_ca_coupled`.
2. Run Mellem 2008 protocol identically to Phase F 2b:
     - 200 ms settle at I = 0
     - 100 ms × 50 pA injection
     - 1500 ms post-stim recovery
     - dt = 0.025 ms, RK4
3. Extract metrics:
     - V_base (last 100 ms of settle)
     - V_peak during stim
     - Plateau amplitude (V at end of stim - V_base)
     - Plateau duration (time post-stim until V drops below V_base + 5 mV)
     - Release-tau ratio (architectural signature)
     - Pool diagnostics: peak [Ca]_i, [Ca]_i at end of stim, [Ca]_i decay tau,
       SLO-1 m gate trajectory
4. Classify verdict:
     VERDICT_CA_COUPLING_SUFFICIENT — amp ∈ [15,25] AND dur ∈ [400,800]
     VERDICT_CA_COUPLING_PARTIAL — dur > 100 ms (ie. >~5x the static-cai run)
                                   but not in [400,800]
     VERDICT_CA_COUPLING_INSUFFICIENT — dur ≤ 100 ms (no meaningful improvement)
     VERDICT_NUMERICAL_ISSUES — NaN/Inf or instability detected
5. Write artifacts: ca_coupling_test_results.{json, md}

Optional secondary sweep (--sweep flag) varies caintra1 buffer (fca) and
efflux (tca_ms) at small grid to characterize sensitivity within the new
regime. Bonus, not required for primary verdict.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_WAVE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_WAVE2_DIR))

import numpy as np

from ca_coupled_cell import (
    build_brian2_ava_ca_coupled,
    AVA_SURF_CM2,
    AVA_VOL_CM3,
    AVA_CM_UFCM2,
    leak_tau_ms,
)


# Mellem 2008 protocol — IDENTICAL to Phase F 2b for direct comparability.
SETTLE_MS = 200.0
STIM_MS = 100.0
RECOVER_MS = 1500.0
STIM_AMP_PA = 50.0
DT_MS = 0.025

# Targets
TARGET_AMP_RANGE = (15.0, 25.0)
TARGET_DUR_RANGE = (400.0, 800.0)


def _run_protocol(factory) -> dict:
    """Execute Mellem 2008 protocol and capture all monitored states."""
    from brian2 import ms, defaultclock, pA

    bundle = factory()
    G = bundle["group"]
    net = bundle["network"]
    mon = bundle["monitor"]

    defaultclock.dt = DT_MS * ms

    G.I_inj = 0 * pA
    net.run(SETTLE_MS * ms)
    G.I_inj = STIM_AMP_PA * pA
    net.run(STIM_MS * ms)
    G.I_inj = 0 * pA
    net.run(RECOVER_MS * ms)

    t_ms = np.array(mon.t) * 1e3
    V_mV = np.array(mon.v[0]) * 1e3
    cai_mM = np.array(mon.cai_mM[0])
    ica_egl19 = np.array(mon.ica_egl19_mAcm2[0])
    m_slo1iso = np.array(mon.m_slo1iso[0])
    ik_slo1iso = np.array(mon.ik_slo1iso_mAcm2[0])
    ik_slo1egl19 = np.array(mon.ik_slo1egl19_mAcm2[0])

    return {
        "t_ms": t_ms,
        "V_mV": V_mV,
        "cai_mM": cai_mM,
        "ica_egl19_mAcm2": ica_egl19,
        "m_slo1iso": m_slo1iso,
        "ik_slo1iso_mAcm2": ik_slo1iso,
        "ik_slo1egl19_mAcm2": ik_slo1egl19,
        "channel_densities": bundle["channel_densities"],
        "geometry": bundle["geometry"],
        "ca_pool_settings": bundle["ca_pool_settings"],
        "pool_params_effective": bundle["pool_params_effective"],
    }


def _measure_plateau(traj: dict) -> dict:
    """Extract Mellem-style plateau metrics + Ca-pool diagnostics."""
    t = traj["t_ms"]
    V = traj["V_mV"]
    cai = traj["cai_mM"]

    issues = []
    finite_ok = bool(np.all(np.isfinite(V)) and np.all(np.isfinite(cai)))
    if not finite_ok:
        issues.append("non_finite_states")

    base_mask = (t > 100) & (t < SETTLE_MS)
    V_base = float(np.mean(V[base_mask])) if base_mask.any() else float(V[0])
    cai_base = float(np.mean(cai[base_mask])) if base_mask.any() else float(cai[0])

    stim_t0 = SETTLE_MS
    stim_t1 = SETTLE_MS + STIM_MS
    stim_mask = (t > stim_t0) & (t < stim_t1)
    V_peak_stim = float(np.max(V[stim_mask])) if stim_mask.any() else V_base
    cai_peak_stim = float(np.max(cai[stim_mask])) if stim_mask.any() else cai_base

    end_idx = int(np.argmin(np.abs(t - stim_t1)))
    V_at_end = float(V[end_idx])
    cai_at_end = float(cai[end_idx])
    plateau_amp = V_at_end - V_base

    # Plateau duration: post-stim time until V < V_base + 5 mV
    post_mask = t > stim_t1
    t_post = t[post_mask]
    V_post = V[post_mask]
    cai_post = cai[post_mask]
    threshold = V_base + 5.0
    decay_idx = np.where(V_post < threshold)[0]
    if len(decay_idx) > 0:
        plateau_duration = float(t_post[decay_idx[0]] - stim_t1)
    else:
        plateau_duration = float(t_post[-1] - stim_t1) if len(t_post) > 0 else -1.0
        issues.append("plateau_did_not_terminate")

    # Release-tau (V): exponential fit to V_post over the first 800 ms
    tau_release_ms = float("nan")
    arch_signature = "unknown"
    leak_tau = leak_tau_ms()
    if finite_ok and len(V_post) > 50 and plateau_amp > 0.5:
        fit_end_t = stim_t1 + min(800.0, t_post[-1] - stim_t1)
        fit_mask = (t_post >= stim_t1) & (t_post <= fit_end_t)
        V_fit = V_post[fit_mask]
        t_fit = t_post[fit_mask] - stim_t1
        delta = V_fit - V_base
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
        if np.isfinite(tau_release_ms) and leak_tau > 0:
            ratio = tau_release_ms / leak_tau
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

    # Ca-pool decay tau: exp fit to cai_post (similar window).
    tau_cai_decay_ms = float("nan")
    if finite_ok and len(cai_post) > 50 and (cai_at_end - cai_base) > 1e-7:
        fit_end_t = stim_t1 + min(800.0, t_post[-1] - stim_t1)
        fit_mask = (t_post >= stim_t1) & (t_post <= fit_end_t)
        cai_fit = cai_post[fit_mask]
        t_fit = t_post[fit_mask] - stim_t1
        delta_ca = cai_fit - cai_base
        valid_ca = delta_ca > 1e-8
        if valid_ca.sum() >= 5:
            log_delta_ca = np.log(delta_ca[valid_ca])
            try:
                slope_ca, _ = np.polyfit(t_fit[valid_ca], log_delta_ca, 1)
                if slope_ca < 0:
                    tau_cai_decay_ms = float(-1.0 / slope_ca)
                else:
                    tau_cai_decay_ms = float("inf")
            except Exception as e:
                issues.append(f"cai_decay_fit_failed: {e}")

    amp_pass = TARGET_AMP_RANGE[0] <= plateau_amp <= TARGET_AMP_RANGE[1]
    dur_pass = TARGET_DUR_RANGE[0] <= plateau_duration <= TARGET_DUR_RANGE[1]
    arch_pass = bool(amp_pass and dur_pass)

    return {
        "V_base_mV": V_base,
        "V_peak_stim_mV": V_peak_stim,
        "V_at_end_of_stim_mV": V_at_end,
        "plateau_amp_mV": float(plateau_amp),
        "plateau_duration_ms": float(plateau_duration),
        "tau_release_ms": tau_release_ms,
        "leak_tau_ms": leak_tau,
        "release_tau_ratio": float(ratio) if isinstance(ratio, (int, float)) and np.isfinite(ratio) else (
            float("nan")
        ),
        "architectural_signature": arch_signature,
        "amp_pass": bool(amp_pass),
        "dur_pass": bool(dur_pass),
        "arch_pass": arch_pass,
        "finite_ok": finite_ok,
        "issues": issues,
        # Ca-pool diagnostics
        "cai_base_mM": cai_base,
        "cai_peak_stim_mM": cai_peak_stim,
        "cai_at_end_of_stim_mM": cai_at_end,
        "cai_decay_tau_ms": tau_cai_decay_ms,
        "cai_fold_change_peak_vs_base":
            float(cai_peak_stim / cai_base) if cai_base > 0 else float("inf"),
    }


def _classify(metrics: dict) -> tuple[str, dict]:
    """Map metrics → coarse verdict per the work-block spec."""
    if not metrics["finite_ok"]:
        return "VERDICT_NUMERICAL_ISSUES", {"issues": metrics["issues"]}
    amp = metrics["plateau_amp_mV"]
    dur = metrics["plateau_duration_ms"]
    if metrics["arch_pass"]:
        return "VERDICT_CA_COUPLING_SUFFICIENT", {
            "amp_mV": amp,
            "duration_ms": dur,
            "release_tau_ratio": metrics["release_tau_ratio"],
            "signature": metrics["architectural_signature"],
        }
    # Phase F 2b reference: 21.4 ms duration with static cai.
    # Partial threshold: dur > 100 ms (~5x baseline, ie. material improvement).
    if dur > 100.0:
        return "VERDICT_CA_COUPLING_PARTIAL", {
            "amp_mV": amp,
            "duration_ms": dur,
            "amp_pass": metrics["amp_pass"],
            "dur_pass": metrics["dur_pass"],
            "improvement_factor_vs_phase_f_2b": dur / 21.4,
            "release_tau_ratio": metrics["release_tau_ratio"],
            "signature": metrics["architectural_signature"],
        }
    return "VERDICT_CA_COUPLING_INSUFFICIENT", {
        "amp_mV": amp,
        "duration_ms": dur,
        "improvement_factor_vs_phase_f_2b": dur / 21.4 if dur > 0 else 0.0,
        "release_tau_ratio": metrics["release_tau_ratio"],
        "signature": metrics["architectural_signature"],
    }


def run_primary() -> dict:
    """Single Ca-coupled cell run + verdict classification. The load-bearing test."""
    print("########################################################")
    print("# Ca-coupling integration test — Phase F 2b on coupled cell")
    print("########################################################\n")

    print(f"AVA geometry: surf={AVA_SURF_CM2:.3e} cm², vol={AVA_VOL_CM3:.3e} cm³, "
          f"cm={AVA_CM_UFCM2}")
    print("Channels: leak + egl19 + slo1iso(DYNAMIC Ca) + slo1egl19(V-only) + "
          "shk1 + shl1 + nca + kqt3 + caintra1")
    print(f"Mellem protocol: {SETTLE_MS} ms settle, {STIM_AMP_PA} pA × {STIM_MS} ms, "
          f"{RECOVER_MS} ms recover, dt={DT_MS} ms, RK4")
    print(f"Targets: amp [{TARGET_AMP_RANGE[0]}, {TARGET_AMP_RANGE[1]}] mV, "
          f"dur [{TARGET_DUR_RANGE[0]}, {TARGET_DUR_RANGE[1]}] ms")
    print(f"Phase F 2b reference (static cai): 46.85 mV / 21.4 ms\n")

    factory = build_brian2_ava_ca_coupled()
    t0 = time.time()
    try:
        traj = _run_protocol(factory)
    except Exception as e:
        tb = traceback.format_exc()
        elapsed = time.time() - t0
        print(f"!! exception during run after {elapsed:.1f}s:\n{tb}")
        return {
            "verdict": "VERDICT_NUMERICAL_ISSUES",
            "verdict_details": {"exception": str(e), "traceback": tb},
            "elapsed_sec": elapsed,
            "metrics": None,
            "traj": None,
        }
    elapsed = time.time() - t0

    metrics = _measure_plateau(traj)
    verdict, details = _classify(metrics)

    print(f"V_base = {metrics['V_base_mV']:.2f} mV")
    print(f"V_peak (stim) = {metrics['V_peak_stim_mV']:.2f} mV")
    print(f"V_at_end_of_stim = {metrics['V_at_end_of_stim_mV']:.2f} mV")
    print(f"Plateau amplitude = {metrics['plateau_amp_mV']:.2f} mV  "
          f"(target [{TARGET_AMP_RANGE[0]}, {TARGET_AMP_RANGE[1]}], "
          f"{'PASS' if metrics['amp_pass'] else 'FAIL'})")
    print(f"Plateau duration = {metrics['plateau_duration_ms']:.1f} ms  "
          f"(target [{TARGET_DUR_RANGE[0]}, {TARGET_DUR_RANGE[1]}], "
          f"{'PASS' if metrics['dur_pass'] else 'FAIL'})")
    print(f"τ_release (V) = {metrics['tau_release_ms']:.2f} ms  "
          f"(leak τ_m = {metrics['leak_tau_ms']:.2f} ms; "
          f"ratio = {metrics['release_tau_ratio']:.2f}; "
          f"signature = {metrics['architectural_signature']})")
    print(f"[Ca]_i base = {metrics['cai_base_mM']:.3e} mM, "
          f"peak = {metrics['cai_peak_stim_mM']:.3e} mM "
          f"({metrics['cai_fold_change_peak_vs_base']:.1f}x base), "
          f"decay τ = {metrics['cai_decay_tau_ms']:.2f} ms")
    print(f"\n>>> VERDICT: {verdict}")
    print(json.dumps(details, indent=2))
    print(f"\n(elapsed {elapsed:.1f}s)\n")

    # Downsampled trajectory for artifact (every 10 samples = 0.25 ms)
    DS = 10
    return {
        "verdict": verdict,
        "verdict_details": details,
        "metrics": metrics,
        "channel_densities": traj["channel_densities"],
        "geometry": traj["geometry"],
        "ca_pool_settings": traj["ca_pool_settings"],
        "pool_params_effective": traj["pool_params_effective"],
        "elapsed_sec": elapsed,
        "trajectory_downsampled": {
            "stride": DS,
            "t_ms": traj["t_ms"][::DS].tolist(),
            "V_mV": traj["V_mV"][::DS].tolist(),
            "cai_mM": traj["cai_mM"][::DS].tolist(),
            "ica_egl19_mAcm2": traj["ica_egl19_mAcm2"][::DS].tolist(),
            "m_slo1iso": traj["m_slo1iso"][::DS].tolist(),
            "ik_slo1iso_mAcm2": traj["ik_slo1iso_mAcm2"][::DS].tolist(),
            "ik_slo1egl19_mAcm2": traj["ik_slo1egl19_mAcm2"][::DS].tolist(),
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
    }


def run_secondary_sweep() -> list[dict]:
    """Optional small sweep over caintra1 (fca, tca) and SLO-1 conductance.

    8 cells: 4 (fca, tca) combinations × 2 SLO-1 gbar scales. Tests sensitivity
    within the new Ca-coupled regime.
    """
    print("\n########################################################")
    print("# Secondary sweep — caintra1 (fca, tca) × SLO-1 gbar")
    print("########################################################\n")

    # Baseline values
    fca_default = 0.001
    tca_default = 50.0
    g_slo1iso_default = 1.0e-9 / 65.89e-8

    # Two probe sets:
    #   (A) physiological-near regime: fca ∈ {1x, 10x} × tca ∈ {1x, 5x}, slo×{1, 4}
    #       — characterizes sensitivity at biologically plausible parameter values.
    #   (B) loop-engagement regime: fca ∈ {100x, 1000x, 10000x, 100000x} at slo×1
    #       — probes the *upper bound* of what Ca-coupling alone can achieve.
    #         These are unphysiological but mechanistically informative.
    grid_A = [
        ({"fca": fca_default,         "tca_ms": tca_default},      sf)
        for sf in (1.0, 4.0)
    ] + [
        ({"fca": fca_default * 10,    "tca_ms": tca_default},      sf)
        for sf in (1.0, 4.0)
    ] + [
        ({"fca": fca_default,         "tca_ms": tca_default * 5},  sf)
        for sf in (1.0, 4.0)
    ] + [
        ({"fca": fca_default * 10,    "tca_ms": tca_default * 5},  sf)
        for sf in (1.0, 4.0)
    ]
    grid_B = [
        ({"fca": fca_default * 100,    "tca_ms": tca_default * 5}, 1.0),
        ({"fca": fca_default * 1000,   "tca_ms": tca_default * 5}, 1.0),
        ({"fca": fca_default * 10000,  "tca_ms": tca_default * 5}, 1.0),
        ({"fca": fca_default * 100000, "tca_ms": tca_default * 5}, 1.0),
    ]
    cell_specs = grid_A + grid_B

    rows = []
    n_total = len(cell_specs)
    for n, (params, sf) in enumerate(cell_specs, start=1):
        t0 = time.time()
        try:
            factory = build_brian2_ava_ca_coupled(
                g_slo1iso_Scm2=g_slo1iso_default * sf,
                caintra1_fca=params["fca"],
                caintra1_tca_ms=params["tca_ms"],
            )
            traj = _run_protocol(factory)
            metrics = _measure_plateau(traj)
            verdict, details = _classify(metrics)
        except Exception as e:
            metrics = {
                "plateau_amp_mV": float("nan"),
                "plateau_duration_ms": float("nan"),
                "amp_pass": False, "dur_pass": False, "arch_pass": False,
                "architectural_signature": "exception",
                "release_tau_ratio": float("nan"),
                "cai_peak_stim_mM": float("nan"),
                "cai_decay_tau_ms": float("nan"),
                "finite_ok": False,
                "issues": [f"top_level_exception: {e}"],
            }
            verdict = "VERDICT_NUMERICAL_ISSUES"
            details = {"exception": str(e)}
        elapsed = time.time() - t0
        print(f"  [{n}/{n_total}] fca={params['fca']:9.4f} tca={params['tca_ms']:5.0f}ms "
              f"slo1x{sf:.1f}  amp={metrics['plateau_amp_mV']:7.2f}  "
              f"dur={metrics['plateau_duration_ms']:7.1f}ms  "
              f"cai_pk={metrics['cai_peak_stim_mM']:.2e}mM  "
              f"sig={metrics['architectural_signature']:18s}  "
              f"verdict={verdict}  ({elapsed:.1f}s)")
        rows.append({
            "fca": params["fca"],
            "tca_ms": params["tca_ms"],
            "slo1iso_factor": sf,
            "verdict": verdict,
            "verdict_details": details,
            "metrics": metrics,
            "elapsed_sec": elapsed,
        })
    return rows


def write_artifacts(primary: dict, secondary: list[dict] | None) -> None:
    out_dir = _WAVE2_DIR / "artifacts"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "ca_coupling_test_results.json"
    md_path = out_dir / "ca_coupling_test_results.md"

    payload = {
        "primary": primary,
        "secondary_sweep": secondary,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved {json_path}")

    metrics = primary.get("metrics") or {}
    verdict = primary["verdict"]
    details = primary["verdict_details"]
    elapsed = primary.get("elapsed_sec", float("nan"))

    md_lines = []
    md_lines.append(
        "# Ca-coupling integration test results — Phase F 2b on Ca-coupled cell\n"
    )
    md_lines.append(f"## Verdict: {verdict}\n")
    md_lines.append("**Date:** 2026-04-26\n")
    md_lines.append(
        "**Trigger:** density-sensitivity sweep "
        "(`density_sensitivity_analysis.md`) confirmed VERDICT_AMPLITUDE_TUNABLE_"
        "DURATION_FAILS for the Phase F static-cai cell. The load-bearing finding "
        "was that terminator scaling has near-zero effect on phenotype because "
        "SLO-1 isolated reads `cai_static = 5e-5 mM` (per F12) — no Ca-feedback "
        "loop exists. This work block tests the cheaper architectural extension "
        "*before* triggering the morphology fork: add caintra1 dynamic [Ca]_i "
        "and couple SLO-1 isolated to it.\n"
    )
    md_lines.append("---\n")

    md_lines.append("## Architecture decisions\n")
    md_lines.append(
        "1. **Ca-pool: `caintra1`** (Nicoletti's AIY/RIM convention), not "
        "`cadiff`. Geometry scaled to AVA (vol=129.6e-12 cm³, surf=1123.84e-8 "
        "cm²); caintra1's empirical effective coefficient is rescaled "
        "linearly per the formula structure (see `calcium_pool.py`).\n"
        "2. **SLO-1 isolated → dynamic [Ca]_i** via a new module "
        "`channels/slo1_iso_dynamic_ca.py` whose eqs is identical to the "
        "static variant except `cai_mM : 1` is *not* declared as a parameter "
        "— it is supplied as a state by the pool.\n"
        "3. **SLO-1+EGL-19 coupled keeps closed-form `calcium(V)`** (Option A "
        "per the work-block spec). Rationale: Nicoletti's coupled variant "
        "encodes a *nanodomain* Ca, not bulk [Ca]_i; replacing it would change "
        "two things at once and confound the test. Option A isolates the "
        "variable.\n"
        "4. **Conductances unchanged from Phase F 2b baseline** (g_egl19, "
        "g_leak, g_nca at Nicoletti AVAL g0; g_slo1iso, g_slo1egl19, g_shl1, "
        "g_shk1, g_kqt3 at Phase F 2b baseline). Any phenotype change is "
        "therefore attributable to the Ca-coupling change.\n"
        "5. **Ca-pool parameters at Nicoletti caintra1.mod defaults**: "
        "fca=0.001, tca=50 ms, ca_eq=5e-8 mM (NEURON numerical default).\n"
    )

    md_lines.append("\n## Primary run — single coupled cell, Mellem protocol\n")
    md_lines.append("\n**Protocol** (identical to Phase F 2b for direct comparability):\n")
    md_lines.append(
        f"- {SETTLE_MS} ms settle at I=0\n"
        f"- {STIM_MS} ms × {STIM_AMP_PA} pA injection\n"
        f"- {RECOVER_MS} ms post-stim recovery\n"
        f"- Brian2 RK4, dt={DT_MS} ms\n"
    )

    if metrics:
        md_lines.append("\n**Phenotype:**\n")
        md_lines.append(
            f"| Metric | Value | Phase F 2b (static cai) | Mellem target |\n"
            f"|---|---|---|---|\n"
            f"| Plateau amplitude (mV) | {metrics['plateau_amp_mV']:.2f} | 46.85 | "
            f"[{TARGET_AMP_RANGE[0]}, {TARGET_AMP_RANGE[1]}] |\n"
            f"| Plateau duration (ms) | {metrics['plateau_duration_ms']:.1f} | 21.4 | "
            f"[{TARGET_DUR_RANGE[0]}, {TARGET_DUR_RANGE[1]}] |\n"
            f"| τ_release (ms) | {metrics['tau_release_ms']:.2f} | — | — |\n"
            f"| Architectural signature | {metrics['architectural_signature']} | "
            f"no_termination | active_termination |\n"
            f"| amp_pass | {metrics['amp_pass']} | False | — |\n"
            f"| dur_pass | {metrics['dur_pass']} | False | — |\n"
            f"| **arch_pass** | **{metrics['arch_pass']}** | **False** | — |\n"
        )
        md_lines.append("\n**Ca-pool diagnostics:**\n")
        md_lines.append(
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| [Ca]_i baseline (mM) | {metrics['cai_base_mM']:.3e} |\n"
            f"| [Ca]_i peak during stim (mM) | {metrics['cai_peak_stim_mM']:.3e} |\n"
            f"| [Ca]_i at end of stim (mM) | {metrics['cai_at_end_of_stim_mM']:.3e} |\n"
            f"| [Ca]_i fold-change peak vs base | "
            f"{metrics['cai_fold_change_peak_vs_base']:.1f}× |\n"
            f"| [Ca]_i decay τ (ms) | {metrics['cai_decay_tau_ms']:.2f} |\n"
        )
        if metrics["issues"]:
            md_lines.append("\n**Issues during run:**\n")
            for iss in metrics["issues"]:
                md_lines.append(f"- {iss}\n")
        md_lines.append(f"\n_Elapsed: {elapsed:.1f}s_\n")

    md_lines.append("\n## Verdict reasoning\n")
    if verdict == "VERDICT_CA_COUPLING_SUFFICIENT":
        md_lines.append(
            "Both amplitude AND duration entered Mellem's target ranges. "
            "Single-compartment AVA + dynamic caintra1 + Ca-coupled SLO-1 "
            "isolated suffices to produce Mellem 2008 plateau dynamics. "
            "The morphology fork is **avoided** — Path A's cellular layer "
            "reaches Mellem dynamics without architectural escalation. "
            "This is a major Wave 2 milestone.\n"
        )
    elif verdict == "VERDICT_CA_COUPLING_PARTIAL":
        md_lines.append(
            "Plateau duration improved substantially over the static-cai "
            "baseline (21.4 ms → "
            f"{metrics['plateau_duration_ms']:.1f} ms, "
            f"factor {details.get('improvement_factor_vs_phase_f_2b', 0):.1f}×) "
            "but did not reach Mellem's [400, 800] ms target. The Ca-coupling "
            "loop is empirically load-bearing — it is the dominant ingredient "
            "the static-cai build was missing — but is not sufficient on its "
            "own to close the gap. The residual gap is a quantified case for "
            "morphology integration: morphology may either be needed in "
            "addition to Ca-coupling, or there may be further density tuning "
            "that closes the gap in the new regime.\n"
        )
    elif verdict == "VERDICT_CA_COUPLING_INSUFFICIENT":
        md_lines.append(
            "Plateau duration showed minimal improvement over the static-cai "
            "baseline. The dynamic Ca-pool was *not* the missing ingredient. "
            "The morphology fork is robustly justified — single-compartment "
            "architecture cannot reach Mellem dynamics regardless of Ca "
            "dynamics in the bulk-pool encoding tested here.\n"
        )
    elif verdict == "VERDICT_NUMERICAL_ISSUES":
        md_lines.append(
            "Numerical instability prevented clean evaluation. See `issues` "
            "in the metrics block above. Surfaced for review.\n"
        )

    if secondary:
        md_lines.append("\n## Secondary sweep — caintra1 + SLO-1 sensitivity\n")
        md_lines.append(
            "Bonus diagnostic — varies caintra1's `fca` (Ca buffer factor) and "
            "`tca` (efflux time constant) at small grid × SLO-1 conductance ×{1, 4}.\n"
        )
        md_lines.append(
            "\n| fca | tca (ms) | slo1_factor | amp (mV) | dur (ms) | "
            "signature | verdict |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for r in secondary:
            m = r["metrics"]
            md_lines.append(
                f"| {r['fca']:.4f} | {r['tca_ms']:.0f} | {r['slo1iso_factor']:.1f} | "
                f"{m['plateau_amp_mV']:.2f} | {m['plateau_duration_ms']:.1f} | "
                f"{m['architectural_signature']} | {r['verdict']} |\n"
            )

    md_lines.append("\n---\n")
    md_lines.append("## Files produced\n")
    md_lines.append(
        "```\n"
        "wave2/\n"
        "├── ca_coupled_cell.py                          [cell builder]\n"
        "├── run_ca_coupling_test.py                     [this driver]\n"
        "├── channels/\n"
        "│   └── slo1_iso_dynamic_ca.py                  [Ca-dynamic SLO-1 iso]\n"
        "└── artifacts/\n"
        "    ├── ca_coupling_test_results.md             [this file]\n"
        "    └── ca_coupling_test_results.json           [raw + downsampled traj]\n"
        "```\n"
    )

    with open(md_path, "w") as f:
        f.writelines(md_lines)
    print(f"→ saved {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="Run optional secondary sweep over caintra1 params + slo1 g.")
    args = parser.parse_args()

    primary = run_primary()
    secondary = run_secondary_sweep() if args.sweep else None
    write_artifacts(primary, secondary)
    return primary["verdict"]


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict == "VERDICT_CA_COUPLING_SUFFICIENT" else 1)
