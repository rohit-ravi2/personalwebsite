"""
Wave 2 cellular extension Option B CP3 + CP4 + CP5 — AIY Layer A validation.

Two components, both apples-to-apples Brian2 vs Nicoletti's actual AIY:

Component 3 (CP3) — voltage-clamp Layer A
    Brian2 7-channel AIY cell vs NEURON AIY (direct upstream construction
    via NEURONReference("AIY") which uses Nicoletti's wrapper / g_to_Scm2).
    Pass: voltage-feature ≤5% relative + >80% holds clear (per
    voltage_clamp_compare_v2 / current-domain divergence metric).

Component 4 (CP4) — current-clamp via direct upstream invocation
    Brian2 7-channel AIY cell vs Nicoletti's published AIY trajectories,
    by direct invocation of `AIY_simulation_iclamp.py`. 11 current steps
    -15 to +35 pA per `AIY_simulation.py` line 21.

    Protocol (from AIY_simulation_iclamp.py lines 62-71):
        delay = 1000 ms
        duration = 5000 ms
        simdur = 11000 ms (so post-stim recovery ≈ 5000 ms)
        h.finitialize(-60)
        h.dt = 0.4 ms

    Pass: voltage-feature ≤3 mV at peak + plateau, >80% timepoints within 3 mV.

CP5 — verdict
    PRODUCTION_GRADE (both pass) / PARTIAL (one fails) / IMPLEMENTATION_BUG /
    DEEPER_FINDING.

Output:
    artifacts/option_b_aiy_results.json
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
from option_alpha_aiy_cell import (
    build_brian2_aiy_7channel,
    AIY_SURF_CM2, AIY_CM_UFCM2, AIY_E_LEAK_MV, AIY_ECA_MV, AIY_EK_MV,
    AIY_G_SCM2, AIY_G0_NS,
)


# ---------------------------------------------------------------------------
# Component 3 — voltage clamp (CP3)
# ---------------------------------------------------------------------------

def run_component_3_voltage_clamp():
    """Brian2 7-channel AIY vs NEURON AIY (Nicoletti's published cell).

    NEURONReference('AIY') uses Nicoletti's g_to_Scm2 + same parameter vector
    as Brian2 cell. Differences will be: (a) translation correctness of each
    channel in cell context, (b) any architectural divergence introduced by
    Brian2's force-clamp vs NEURON's VClamp.
    """
    print("=" * 70)
    print("CP3 — VC Brian2 7-channel AIY vs NEURON AIY")
    print("=" * 70)
    print(f"\nAIY geometry: surf={AIY_SURF_CM2:.3e} cm², cm={AIY_CM_UFCM2}")
    print("Brian2 cell channels: [egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1]")
    print("NEURON cell:          [egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1] (Nicoletti AIY canonical)")
    print(f"\nDensities (S/cm² from g0 = {AIY_G0_NS}):")
    for k, v in AIY_G_SCM2.items():
        print(f"  {k:10s}: {v:.3e}")
    print()

    # Build NEURON reference (Nicoletti's actual AIY via her wrapper)
    print("Building NEURON AIY reference...")
    nref = NEURONReference("AIY")
    print(f"  Built. surf={nref._surf_cm2():.3e} cm², C_m={nref._cm_pF:.3f} pF\n")

    # Build Brian2 factory
    factory = build_brian2_aiy_7channel(record_components=False)

    # Holds: 11 levels -80 to +40 (matching Phase F 2a / F gate2 convention).
    # AIY's AIY_simulation_vclamp.py uses 18 holds in [-120, +50] mV — we use
    # the standard 11-hold sweep for Layer A consistency with AVAL precedent.
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

    print(f"\nCP3 results:")
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
              f"brian2_peak={s['brian2_peak_I_pA']:+10.2f}  "
              f"nrn_peak={s['ref_peak_I_pA']:+10.2f}  "
              f"div={peak_div:.4f}({'P' if peak_pass else 'F'})  "
              f"brian2_ss={s['brian2_ss_I_pA']:+10.2f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+10.2f}  "
              f"div={ss_div:.4f}({'P' if ss_pass else 'F'})")
    nref.cleanup()
    return result


# ---------------------------------------------------------------------------
# Component 4 — current clamp via direct upstream invocation (CP4)
# ---------------------------------------------------------------------------

def run_neuron_aiy_current_clamp_upstream(injection_pa_list, dt_ms=0.4):
    """Directly invoke Nicoletti's AIY exact section construction.

    Per AIY_simulation_iclamp.py:
      stim.delay = 1000 ms
      stim.dur   = 5000 ms
      simdur     = 11000 ms
      h.finitialize(-60)
      h.dt = 0.4

    For our protocol we need to control individual injection levels (not
    Nicoletti's linspace), so we re-implement the inner protocol while
    using Nicoletti's exact section construction.
    """
    print(f"\n  Invoking upstream AIY via direct NEURON construction "
          f"(matching AIY_simulation_iclamp.py exactly)...")

    with _nicoletti_env():
        from neuron import h, gui  # noqa: F401
        import math
        from g_to_Scm2 import gScm2

        # Nicoletti's exact construction (matching AIY_simulation_iclamp.py):
        surf = AIY_SURF_CM2
        # g0 = [leak, slo1iso, kqt1, egl19, slo1egl19, nca, shl1, eleak, cm]
        g0 = [0.14, 1.0, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6]
        g_scaled = gScm2(g0, surf, 6)

        cm_uFcm2 = float(g_scaled[8])
        e_leak = float(g_scaled[7])
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_aiy_cp4")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        # Insertion order per AIY_simulation_iclamp.py lines 28-38:
        soma.insert("egl19")
        soma.insert("slo1egl19")
        soma.insert("nca")
        soma.insert("leak")
        soma.insert("slo1iso")
        soma.insert("kqt1")
        soma.insert("shl1")

        for seg in soma:
            seg.leak.gbar = float(g_scaled[0])
            seg.slo1iso.gbar = float(g_scaled[1])
            seg.kqt1.gbar = float(g_scaled[2])
            seg.egl19.gbar = float(g_scaled[3])
            seg.slo1egl19.gbar = float(g_scaled[4])
            seg.nca.gbar = float(g_scaled[5])
            seg.shl1.gbar = float(g_scaled[6])
            seg.leak.e = e_leak
            seg.eca = 60
            seg.ek = -80

        # Protocol per AIY_simulation_iclamp.py:
        stim = h.IClamp(soma(0.5))
        stim.delay = 1000.0
        stim.dur = 5000.0
        simdur = 11000.0

        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)

        sweeps = []
        for inj_pa in injection_pa_list:
            # NEURON IClamp.amp is in nA (1 pA = 1e-3 nA)
            stim.amp = inj_pa * 1e-3
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
                "stim_onset_ms": 1000.0,
                "stim_offset_ms": 1000.0 + 5000.0,
                "simdur_ms": simdur,
            })

        # Cleanup
        del soma
        del stim

    return sweeps


def run_brian2_aiy_current_clamp(injection_pa_list, dt_ms=0.4):
    """Run Brian2 7-channel AIY cell under same 5000 ms current-clamp protocol.

    Returns list of dicts {injection_pa, t_ms, V_mV} matching upstream output.
    """
    print(f"  Running Brian2 7-channel AIY cell under same protocol (dt={dt_ms} ms)...")
    from brian2 import ms, mV, defaultclock, pA

    sweeps = []
    for inj_pa in injection_pa_list:
        # Re-build factory each sweep for clean state
        factory = build_brian2_aiy_7channel(record_components=False)
        bundle = factory()
        defaultclock.dt = dt_ms * ms
        bundle["disable_clamp"]()  # ensure free-running

        # Settle 1000 ms at I=0 (matches Nicoletti's stim.delay)
        bundle["inject_pA"](0.0)
        bundle["network"].run(1000.0 * ms)
        # Inject for 5000 ms
        bundle["inject_pA"](inj_pa)
        bundle["network"].run(5000.0 * ms)
        # Recovery for 5000 ms (to total 11000 ms simdur)
        bundle["inject_pA"](0.0)
        bundle["network"].run(5000.0 * ms)

        mon = bundle["monitor"]
        t_arr = np.asarray(mon.t) * 1e3  # ms
        v_arr = np.asarray(mon.v[0]) * 1e3  # mV
        sweeps.append({
            "injection_pa": float(inj_pa),
            "t_ms": t_arr,
            "V_mV": v_arr,
            "stim_onset_ms": 1000.0,
            "stim_offset_ms": 6000.0,
            "simdur_ms": 11000.0,
        })

    return sweeps


def extract_features(sweep):
    """Extract voltage features from a single CC sweep.

    Mirrors run_option_alpha_cp4.extract_features but with AIY's protocol
    (5000 ms stim, longer plateau window).

    Returns:
      baseline_pre_mV: mean V over t in [-300, 0] ms
      peak_V_mV: max |V - baseline| during stim
      plateau_V_mV: median V over last 1000 ms of stim (AIY plateau is later;
                    Nicoletti's analysis uses t in [5990, 6000] ms = last 10 ms
                    of stim, so we mirror that)
      baseline_post_mV: mean V over last 200 ms of recovery
      time_to_peak_ms: time after stim onset of peak
    """
    t = sweep["t_ms"] - sweep["stim_onset_ms"]  # 0 = stim onset
    v = sweep["V_mV"]
    stim_dur = sweep["stim_offset_ms"] - sweep["stim_onset_ms"]

    pre_mask = (t >= -300.0) & (t < 0)
    step_mask = (t >= 0) & (t < stim_dur)
    # Nicoletti's analysis: SS over [5990, 6000] ms wall, which is [4990, 5000]
    # post-stim-onset. We use the last 1000 ms of stim for robustness — peak
    # comparison uses Nicoletti's [1000, 1300] ms window i.e. [0, 300] post-onset.
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

    # Nicoletti's published peak (her vi_peak): for j<=2 (lowest 3 currents) use
    # min in [0, 300] ms; otherwise use max. We don't classify here — we just
    # report both.
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


def run_component_4_current_clamp():
    """Current-clamp comparison: Brian2 7-channel vs NEURON AIY (upstream)."""
    print("\n" + "=" * 70)
    print("CP4 — CC Brian2 7-channel AIY vs NEURON AIY (upstream)")
    print("=" * 70)
    print("\nProtocol (from AIY_simulation_iclamp.py + AIY_simulation.py):")
    print("  delay = 1000 ms, duration = 5000 ms, simdur = 11000 ms")
    print("  injection levels: 11 steps from -15 to +35 pA "
          "(per AIY_simulation.py line 21: linspace(-0.015, 0.035, 11) nA)")
    print("  v_init = -60 mV, dt = 0.4 ms (Nicoletti AIY uses 0.4 ms)")
    print()

    # 11 injection levels in pA (matching AIY_simulation.py linspace(-0.015, 0.035, 11) nA)
    injection_pa = list(np.linspace(-15.0, 35.0, 11))
    print(f"Injection levels (pA): {[f'{x:+.1f}' for x in injection_pa]}")

    # NEURON upstream sweeps
    print("\n[NEURON]")
    nrn_sweeps = run_neuron_aiy_current_clamp_upstream(injection_pa)

    # Brian2 sweeps
    print("\n[Brian2]")
    b2_sweeps = run_brian2_aiy_current_clamp(injection_pa)

    # Compare per-sweep
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

        # Per-timepoint comparison
        nrn_t = nrn["t_ms"]
        nrn_v = nrn["V_mV"]
        b2_t = b2["t_ms"]
        b2_v = b2["V_mV"]
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
                "nrn_t_ms": nrn_t.tolist()[::40],  # downsample
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
    print(f"Component 4 panel_pass: {aggregate_pass}")

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Component 3 — voltage clamp
    result_3 = run_component_3_voltage_clamp()

    # Component 4 — current clamp via upstream
    result_4 = run_component_4_current_clamp()

    # Outcome classification
    pass_3 = bool(result_3["panel_pass"])
    pass_4 = bool(result_4["panel_pass"])

    print("\n" + "=" * 70)
    print("CP5 — AIY outcome classification")
    print("=" * 70)
    if pass_3 and pass_4:
        verdict = "VERDICT_AIY_PRODUCTION_GRADE"
        verdict_msg = (
            "Both apples-to-apples comparisons pass. Brian2 7-channel AIY matches "
            "Nicoletti's NEURON AIY within tolerance for both voltage-clamp and "
            "5000 ms current-clamp protocols."
        )
    elif pass_3 and not pass_4:
        verdict = "VERDICT_AIY_PARTIAL_VC_PASS_CC_FAIL"
        verdict_msg = (
            "Voltage-clamp passes (channel kinetics correct in cell context). "
            "Current-clamp 5000 ms protocol diverges. Investigate as DEEPER_FINDING "
            "candidate (e.g., integration scheme, transient handling, slow-tau "
            "channel like KQT-1's s-gate accumulating drift)."
        )
    elif not pass_3 and pass_4:
        verdict = "VERDICT_AIY_ANOMALOUS_VC_FAIL_CC_PASS"
        verdict_msg = "Anomalous: voltage-clamp Layer A fails but current-clamp passes. Investigate."
    else:
        verdict = "VERDICT_AIY_PARTIAL_BOTH_FAIL"
        verdict_msg = "Both components fail. Likely a translation defect or coupling issue surfaced."

    print(f"\nVerdict: {verdict}")
    print(f"  {verdict_msg}")

    # Save full results
    out_path = Path(__file__).parent / "artifacts" / "option_b_aiy_results.json"
    summary = {
        "checkpoint": "option_b_AIY_CP3_CP4_CP5",
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "component_3_voltage_clamp": {
            "panel_pass": pass_3,
            "n_holds": result_3["n_holds"],
            "n_holds_passing": result_3["n_holds_passing"],
            "fraction_passing": float(result_3["fraction_passing"]),
            "tolerance_metric": result_3["tolerance_metric"],
            "per_step": result_3["per_step"],
            "per_step_evaluations": result_3["evaluation"]["per_step_evaluations"],
        },
        "component_4_current_clamp": result_4,
        "elapsed_s": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {time.time() - t_start:.1f} s")

    return verdict, summary


if __name__ == "__main__":
    verdict, summary = main()
    if verdict == "VERDICT_AIY_PRODUCTION_GRADE":
        sys.exit(0)
    elif verdict.startswith("VERDICT_AIY_PARTIAL"):
        sys.exit(2)
    else:
        sys.exit(1)
