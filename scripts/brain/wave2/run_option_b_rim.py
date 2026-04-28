"""
Wave 2 cellular extension RIM CP5 + CP6 + CP7 — RIM Layer A validation.

Two components, both apples-to-apples Brian2 vs Nicoletti's actual RIM:

Component CP5 — voltage-clamp Layer A
    Brian2 7-channel RIM cell vs NEURON RIM
    (NEURONReference("RIM") — Nicoletti's wrapper). 11 holds.
    Pass: voltage-feature ≤5% relative + >80% holds clear.

Component CP6 — current-clamp via direct upstream invocation
    Brian2 7-channel RIM cell vs Nicoletti's published RIM trajectories,
    by direct invocation of `RIM_simulation_iclamp.py` protocol.
    11 current steps -15 to +35 pA per `RIM_simulation.py` line 20.

    Protocol (from RIM_simulation_iclamp.py lines 60-71):
        delay = 5000 ms
        duration = 5000 ms
        simdur = 14000 ms (post-stim recovery 4000 ms)
        h.finitialize(-60)
        h.dt = 0.04 ms

    Pass: voltage-feature ≤3 mV at peak + plateau, >80% timepoints within 3 mV.

CP7 — verdict
    PRODUCTION_GRADE (both pass) / PARTIAL (one fails) / IMPLEMENTATION_BUG /
    DEEPER_FINDING.

Output:
    artifacts/option_b_rim_results.json
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from neuron_reference import NEURONReference, _nicoletti_env
from voltage_clamp_harness import voltage_clamp_compare_v2
from option_alpha_rim_cell import (
    build_brian2_rim_7channel,
    RIM_SURF_CM2, RIM_CM_UFCM2, RIM_E_LEAK_MV, RIM_ECA_MV, RIM_EK_MV,
    RIM_G_SCM2,
)


# ---------------------------------------------------------------------------
# Component CP5 — voltage clamp
# ---------------------------------------------------------------------------

def run_component_cp5_voltage_clamp():
    """Brian2 7-channel RIM vs NEURON RIM (Nicoletti's published cell)."""
    print("=" * 70)
    print("CP5 — VC Brian2 7-channel RIM vs NEURON RIM")
    print("=" * 70)
    print(f"\nRIM geometry: surf={RIM_SURF_CM2:.3e} cm², cm={RIM_CM_UFCM2}")
    print("Channels:     [shl1, egl2, irk, cca1, unc2, egl19, leak]")
    print(f"\nDensities (S/cm² already; no gScm2 rescale):")
    for k, v in RIM_G_SCM2.items():
        print(f"  {k:6s}: {v:.4e}")
    print(f"\neca = {RIM_ECA_MV} mV (F18 refinement: symmetric USEION ca contract)")
    print(f"ek  = {RIM_EK_MV} mV")
    print(f"eleak = {RIM_E_LEAK_MV} mV")
    print()

    print("Building NEURON RIM reference...")
    nref = NEURONReference("RIM")
    print(f"  Built. surf={nref._surf_cm2():.3e} cm², C_m={nref._cm_pF:.3f} pF\n")

    factory = build_brian2_rim_7channel(record_components=False)

    # 11 holds — same convention as AVAL/AIY precedent.
    holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    print(f"Running voltage-clamp comparison at {len(holds)} holds...")
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

    print(f"\nCP5 results:")
    print(f"  panel_pass: {result['panel_pass']}")
    print(f"  holds passing: {result['n_holds_passing']}/{result['n_holds']} "
          f"({result['fraction_passing']:.1%})")

    print("\nPer-hold detail:")
    for s, e in zip(result["per_step"], result["evaluation"]["per_step_evaluations"]):
        peak_div = e["feature_results"]["peak_I_pA"]["divergence"]
        ss_div = e["feature_results"]["ss_I_pA"]["divergence"]
        peak_pass = e["feature_results"]["peak_I_pA"]["pass"]
        ss_pass = e["feature_results"]["ss_I_pA"]["pass"]
        print(f"  hold={s['hold_mV']:+5.0f} mV  "
              f"b2_peak={s['brian2_peak_I_pA']:+10.2f}  "
              f"nrn_peak={s['ref_peak_I_pA']:+10.2f}  "
              f"div={peak_div:.4f}({'P' if peak_pass else 'F'})  "
              f"b2_ss={s['brian2_ss_I_pA']:+10.2f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+10.2f}  "
              f"div={ss_div:.4f}({'P' if ss_pass else 'F'})")
    nref.cleanup()
    return result


# ---------------------------------------------------------------------------
# Component CP6 — current clamp via direct upstream invocation
# ---------------------------------------------------------------------------

def run_neuron_rim_current_clamp_upstream(injection_pa_list, dt_ms=0.04):
    """Directly invoke Nicoletti's RIM exact section construction.

    Per RIM_simulation_iclamp.py:
      stim.delay = 5000 ms
      stim.dur   = 5000 ms
      simdur     = 14000 ms
      h.finitialize(-60)
      h.dt = 0.04
    """
    print(f"\n  Invoking upstream RIM via direct NEURON construction "
          f"(matching RIM_simulation_iclamp.py exactly)...")

    with _nicoletti_env():
        from neuron import h, gui  # noqa: F401
        import math

        # Nicoletti's exact RIM g vector (already in S/cm²)
        rim_g = [
            RIM_G_SCM2["shl1"], RIM_G_SCM2["egl2"], RIM_G_SCM2["irk"],
            RIM_G_SCM2["cca1"], RIM_G_SCM2["unc2"], RIM_G_SCM2["egl19"],
            RIM_G_SCM2["leak"], RIM_E_LEAK_MV, RIM_CM_UFCM2,
        ]

        cm_uFcm2 = float(rim_g[8])
        e_leak = float(rim_g[7])
        surf = RIM_SURF_CM2
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_rim_cp6")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        # Insertion order per RIM_simulation_iclamp.py lines 31-37:
        soma.insert("shl1")
        soma.insert("egl2")
        soma.insert("irk")
        soma.insert("cca1")
        soma.insert("unc2")
        soma.insert("egl19")
        soma.insert("leak")

        for seg in soma:
            seg.shl1.gbar = float(rim_g[0])
            seg.egl2.gbar = float(rim_g[1])
            seg.irk.gbar = float(rim_g[2])
            seg.cca1.gbar = float(rim_g[3])
            seg.unc2.gbar = float(rim_g[4])
            seg.egl19.gbar = float(rim_g[5])
            seg.leak.gbar = float(rim_g[6])
            seg.leak.e = e_leak
            seg.eca = 60.0
            seg.ek = -80.0

        stim = h.IClamp(soma(0.5))
        stim.delay = 5000.0
        stim.dur = 5000.0
        simdur = 14000.0

        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)

        sweeps = []
        for inj_pa in injection_pa_list:
            stim.amp = inj_pa * 1e-3   # NEURON IClamp.amp is in nA
            h.tstop = simdur
            h.dt = dt_ms
            h.v_init = -60
            h.finitialize(-60)
            h.run()

            t_arr = np.array(t_vec.to_python())
            v_arr = np.array(v_vec.to_python())
            sweeps.append({
                "injection_pa": float(inj_pa),
                "t_ms": t_arr,
                "V_mV": v_arr,
                "stim_onset_ms": 5000.0,
                "stim_offset_ms": 10000.0,
                "simdur_ms": simdur,
            })

        del soma
        del stim

    return sweeps


def run_brian2_rim_current_clamp(injection_pa_list, dt_ms=0.04):
    """Run Brian2 7-channel RIM cell under same 5000 ms current-clamp protocol."""
    print(f"  Running Brian2 7-channel RIM cell under same protocol (dt={dt_ms} ms)...")
    from brian2 import ms, mV, defaultclock, pA

    sweeps = []
    for inj_pa in injection_pa_list:
        factory = build_brian2_rim_7channel(record_components=False)
        bundle = factory()
        defaultclock.dt = dt_ms * ms
        bundle["disable_clamp"]()

        # Settle 5000 ms at I=0 (matches Nicoletti's stim.delay)
        bundle["inject_pA"](0.0)
        bundle["network"].run(5000.0 * ms)
        # Inject 5000 ms
        bundle["inject_pA"](inj_pa)
        bundle["network"].run(5000.0 * ms)
        # Recovery 4000 ms (total 14000 ms)
        bundle["inject_pA"](0.0)
        bundle["network"].run(4000.0 * ms)

        mon = bundle["monitor"]
        t_arr = np.asarray(mon.t) * 1e3
        v_arr = np.asarray(mon.v[0]) * 1e3
        sweeps.append({
            "injection_pa": float(inj_pa),
            "t_ms": t_arr,
            "V_mV": v_arr,
            "stim_onset_ms": 5000.0,
            "stim_offset_ms": 10000.0,
            "simdur_ms": 14000.0,
        })

    return sweeps


def extract_features(sweep):
    """Extract voltage features from a single CC sweep."""
    t = sweep["t_ms"] - sweep["stim_onset_ms"]
    v = sweep["V_mV"]
    stim_dur = sweep["stim_offset_ms"] - sweep["stim_onset_ms"]

    pre_mask = (t >= -300.0) & (t < 0)
    step_mask = (t >= 0) & (t < stim_dur)
    plateau_mask = (t >= stim_dur - 1000.0) & (t < stim_dur)
    nico_peak_mask = (t >= 0) & (t < 300.0)
    post_mask = t >= stim_dur

    baseline_pre = float(np.mean(v[pre_mask])) if pre_mask.any() else float(v[0])

    if step_mask.any():
        v_step = v[step_mask]
        t_step = t[step_mask]
        delta = v_step - baseline_pre
        peak_idx = int(np.argmax(np.abs(delta)))
        peak_V = float(v_step[peak_idx])
        time_to_peak = float(t_step[peak_idx])
    else:
        peak_V = baseline_pre
        time_to_peak = 0.0

    plateau_V = float(np.median(v[plateau_mask])) if plateau_mask.any() else peak_V
    baseline_post = float(np.mean(v[post_mask][-min(len(v[post_mask]), 500):])) if post_mask.any() else float(v[-1])

    if nico_peak_mask.any():
        v_nico = v[nico_peak_mask]
        nico_peak_min = float(np.min(v_nico))
        nico_peak_max = float(np.max(v_nico))
    else:
        nico_peak_min = baseline_pre
        nico_peak_max = baseline_pre

    return {
        "baseline_pre_mV": baseline_pre,
        "peak_V_mV": peak_V,
        "plateau_V_mV": plateau_V,
        "baseline_post_mV": baseline_post,
        "time_to_peak_ms": time_to_peak,
        "nico_peak_min_mV": nico_peak_min,
        "nico_peak_max_mV": nico_peak_max,
    }


def run_component_cp6_current_clamp():
    """Current-clamp comparison: Brian2 7-channel vs NEURON RIM (upstream)."""
    print("\n" + "=" * 70)
    print("CP6 — CC Brian2 7-channel RIM vs NEURON RIM (upstream)")
    print("=" * 70)
    print("\nProtocol (from RIM_simulation_iclamp.py + RIM_simulation.py):")
    print("  delay = 5000 ms, duration = 5000 ms, simdur = 14000 ms")
    print("  injection levels: 11 steps from -15 to +35 pA "
          "(per RIM_simulation.py line 20: linspace(-0.015, 0.035, 11) nA)")
    print("  v_init = -60 mV, dt = 0.04 ms")
    print()

    injection_pa = list(np.linspace(-15.0, 35.0, 11))
    print(f"Injection levels (pA): {[f'{x:+.1f}' for x in injection_pa]}")

    print("\n[NEURON]")
    nrn_sweeps = run_neuron_rim_current_clamp_upstream(injection_pa)

    print("\n[Brian2]")
    b2_sweeps = run_brian2_rim_current_clamp(injection_pa)

    print("\n" + "-" * 70)
    print("Per-sweep comparison:")
    print("-" * 70)

    voltage_feature_tolerance_mV = 3.0
    timepoint_pass_fraction = 0.8

    per_sweep_results = []
    aggregate_pass_count = 0
    n_total_timepoints = 0
    n_total_passing = 0

    for nrn, b2 in zip(nrn_sweeps, b2_sweeps):
        inj = nrn["injection_pa"]
        nrn_feat = extract_features(nrn)
        b2_feat = extract_features(b2)

        residuals = {
            "baseline_pre_mV": abs(b2_feat["baseline_pre_mV"] - nrn_feat["baseline_pre_mV"]),
            "peak_V_mV": abs(b2_feat["peak_V_mV"] - nrn_feat["peak_V_mV"]),
            "plateau_V_mV": abs(b2_feat["plateau_V_mV"] - nrn_feat["plateau_V_mV"]),
            "baseline_post_mV": abs(b2_feat["baseline_post_mV"] - nrn_feat["baseline_post_mV"]),
        }
        feature_pass = (
            residuals["peak_V_mV"] <= voltage_feature_tolerance_mV
            and residuals["plateau_V_mV"] <= voltage_feature_tolerance_mV
        )

        nrn_t = nrn["t_ms"]; nrn_v = nrn["V_mV"]
        b2_t = b2["t_ms"];   b2_v = b2["V_mV"]
        common_start = max(nrn_t[0], b2_t[0])
        common_end = min(nrn_t[-1], b2_t[-1])
        common_grid = np.linspace(common_start, common_end, 5000)
        b2_v_interp = np.interp(common_grid, b2_t, b2_v)
        nrn_v_interp = np.interp(common_grid, nrn_t, nrn_v)
        residuals_t = np.abs(b2_v_interp - nrn_v_interp)
        n_pass = int(np.sum(residuals_t <= voltage_feature_tolerance_mV))
        n_total = len(residuals_t)
        frac = n_pass / n_total
        timepoint_pass = frac >= timepoint_pass_fraction

        sweep_pass = bool(feature_pass and timepoint_pass)
        if sweep_pass:
            aggregate_pass_count += 1
        n_total_timepoints += n_total
        n_total_passing += n_pass

        print(f"\nInjection: {inj:+6.1f} pA")
        print(f"  Brian2: pre={b2_feat['baseline_pre_mV']:+6.2f}  "
              f"peak={b2_feat['peak_V_mV']:+6.2f}  plat={b2_feat['plateau_V_mV']:+6.2f}  "
              f"post={b2_feat['baseline_post_mV']:+6.2f}")
        print(f"  NEURON: pre={nrn_feat['baseline_pre_mV']:+6.2f}  "
              f"peak={nrn_feat['peak_V_mV']:+6.2f}  plat={nrn_feat['plateau_V_mV']:+6.2f}  "
              f"post={nrn_feat['baseline_post_mV']:+6.2f}")
        print(f"  Δ: pre={residuals['baseline_pre_mV']:.3f}  "
              f"peak={residuals['peak_V_mV']:.3f}  "
              f"plateau={residuals['plateau_V_mV']:.3f}  "
              f"post={residuals['baseline_post_mV']:.3f}")
        print(f"  feature_pass={feature_pass}, timepoint_pass={timepoint_pass} "
              f"({n_pass}/{n_total} = {frac:.1%}); sweep_pass={sweep_pass}")

        per_sweep_results.append({
            "injection_pa": inj,
            "brian2_features": b2_feat,
            "neuron_features": nrn_feat,
            "feature_residuals": residuals,
            "feature_pass": bool(feature_pass),
            "timepoint_pass": bool(timepoint_pass),
            "sweep_pass": sweep_pass,
            "fraction_timepoints_passing": float(frac),
            "n_timepoints": n_total,
            "n_timepoints_passing": n_pass,
            "trajectories": {
                "nrn_t_ms": nrn_t.tolist()[::40],
                "nrn_V_mV": nrn_v.tolist()[::40],
                "b2_t_ms": b2_t.tolist()[::40],
                "b2_V_mV": b2_v.tolist()[::40],
            },
        })

    print()
    print("-" * 70)
    aggregate_frac = n_total_passing / max(1, n_total_timepoints)
    sweep_frac = aggregate_pass_count / len(per_sweep_results)
    aggregate_pass = sweep_frac >= timepoint_pass_fraction
    print(f"Aggregate: {aggregate_pass_count}/{len(per_sweep_results)} sweeps pass "
          f"({sweep_frac:.1%})")
    print(f"Aggregate timepoint pass: {n_total_passing}/{n_total_timepoints} "
          f"({aggregate_frac:.1%})")
    print(f"Component CP6 panel_pass: {aggregate_pass}")

    return {
        "panel_pass": bool(aggregate_pass),
        "n_sweeps": len(per_sweep_results),
        "n_sweeps_passing": int(aggregate_pass_count),
        "fraction_sweeps_passing": float(sweep_frac),
        "aggregate_timepoint_pass_fraction": float(aggregate_frac),
        "voltage_feature_tolerance_mV": voltage_feature_tolerance_mV,
        "timepoint_pass_fraction": timepoint_pass_fraction,
        "per_sweep": per_sweep_results,
    }


def write_status(cp: str, payload: dict):
    out_path = Path(__file__).parent / "artifacts" / "checkpoints" / f"{cp}_status.json"
    out_path.parent.mkdir(exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    args = sys.argv[1:]
    do_cp5 = (not args) or "cp5" in args
    do_cp6 = (not args) or "cp6" in args

    result_5 = None
    result_6 = None

    if do_cp5:
        result_5 = run_component_cp5_voltage_clamp()
        write_status("rim_CP5", {
            "checkpoint": "rim_CP5",
            "panel_pass": bool(result_5["panel_pass"]),
            "n_holds": result_5["n_holds"],
            "n_holds_passing": result_5["n_holds_passing"],
            "fraction_passing": float(result_5["fraction_passing"]),
        })

    if do_cp6:
        result_6 = run_component_cp6_current_clamp()
        write_status("rim_CP6", {
            "checkpoint": "rim_CP6",
            "panel_pass": bool(result_6["panel_pass"]),
            "n_sweeps": result_6["n_sweeps"],
            "n_sweeps_passing": result_6["n_sweeps_passing"],
            "fraction_sweeps_passing": result_6["fraction_sweeps_passing"],
            "aggregate_timepoint_pass_fraction": result_6["aggregate_timepoint_pass_fraction"],
        })

    # CP7 verdict
    print("\n" + "=" * 70)
    print("CP7 — RIM outcome classification")
    print("=" * 70)

    if not (do_cp5 and do_cp6):
        print("(Partial run — full verdict requires both CP5 and CP6)")
        return None, None

    pass_5 = bool(result_5["panel_pass"])
    pass_6 = bool(result_6["panel_pass"])

    if pass_5 and pass_6:
        verdict = "VERDICT_RIM_PRODUCTION_GRADE"
        verdict_msg = (
            "Both apples-to-apples comparisons pass. Brian2 7-channel RIM matches "
            "Nicoletti's NEURON RIM within tolerance for both voltage-clamp and "
            "5000 ms current-clamp protocols."
        )
    elif pass_5 and not pass_6:
        verdict = "VERDICT_RIM_PARTIAL_VC_PASS_CC_FAIL"
        verdict_msg = (
            "Voltage-clamp passes (channel kinetics correct in cell context). "
            "Current-clamp 5000 ms protocol diverges. Investigate as DEEPER_FINDING "
            "candidate."
        )
    elif not pass_5 and pass_6:
        verdict = "VERDICT_RIM_ANOMALOUS_VC_FAIL_CC_PASS"
        verdict_msg = "Anomalous: voltage-clamp Layer A fails but current-clamp passes."
    else:
        verdict = "VERDICT_RIM_PARTIAL_BOTH_FAIL"
        verdict_msg = "Both components fail."

    print(f"\nVerdict: {verdict}")
    print(f"  {verdict_msg}")

    # Save final results
    out_path = Path(__file__).parent / "artifacts" / "option_b_rim_results.json"
    summary = {
        "checkpoint": "wave2_RIM_CP5_CP6_CP7",
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "component_cp5_voltage_clamp": {
            "panel_pass": pass_5,
            "n_holds": result_5["n_holds"],
            "n_holds_passing": result_5["n_holds_passing"],
            "fraction_passing": float(result_5["fraction_passing"]),
            "tolerance_metric": result_5["tolerance_metric"],
            "per_step": result_5["per_step"],
            "per_step_evaluations": result_5["evaluation"]["per_step_evaluations"],
        },
        "component_cp6_current_clamp": result_6,
        "elapsed_s": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {time.time() - t_start:.1f} s")

    write_status("rim_CP7", {
        "checkpoint": "rim_CP7",
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "elapsed_s": time.time() - t_start,
    })

    return verdict, summary


if __name__ == "__main__":
    verdict, summary = main()
    if verdict == "VERDICT_RIM_PRODUCTION_GRADE":
        sys.exit(0)
    elif verdict and verdict.startswith("VERDICT_RIM_PARTIAL"):
        sys.exit(2)
    elif verdict is None:
        sys.exit(0)
    else:
        sys.exit(1)
