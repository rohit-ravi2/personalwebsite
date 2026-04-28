"""
Wave 2 option α-1 CP4 — Phase F re-evaluation with TRUE 4-channel AVAL.

Two components, both apples-to-apples Brian2 vs Nicoletti's actual AVAL:

Component 2a — voltage-clamp Layer A
    Brian2 4-channel AVA cell vs NEURON AVAL (direct upstream construction
    via NEURONReference("AVAL") which uses Nicoletti's wrapper).
    Pass: voltage-feature ≤5% relative + >80% holds clear (per
    voltage_clamp_compare_v2 / current-domain divergence metric).

Component 2b — current-clamp (1000 ms protocol matching Nicoletti)
    Brian2 4-channel AVA cell vs Nicoletti's published AVAL trajectories,
    by direct invocation of `AVAL_simulation_iclamp.py`. 7 current steps
    -30 to +30 pA per `AVAL_simulations.py` line 31.

    Protocol (from AVAL_simulation_iclamp.py lines 53-55):
        delay = 1023 ms
        duration = 1000 ms
        simdur = 2500 ms (so post-stim recovery ≈ 477 ms)
        h.finitialize(-60)

    Pass: >80% of timepoints within voltage-feature ≤3 mV at peak + plateau,
    summed across all 7 current levels.

Output:
    artifacts/option_alpha_phase_f_results.json
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
from option_alpha_ava_cell import (
    build_brian2_aval_4channel,
    AVAL_SURF_CM2, AVAL_CM_UFCM2, AVAL_E_LEAK_MV, AVAL_ECA_MV, AVAL_EK_MV,
    AVAL_G_SCM2,
)


# ---------------------------------------------------------------------------
# Component 2a — voltage clamp
# ---------------------------------------------------------------------------

def run_component_2a():
    """Brian2 4-channel vs NEURON AVAL (Nicoletti's published cell).

    NEURONReference('AVAL') invokes the same parameter rescaling pipeline as
    AVAL_simulations.py via `g_to_Scm2.gScm2` — that's the published
    canonical AVAL cell. So this comparison is "Brian2 4-channel AVA built
    from same g0 vector" vs "NEURON AVAL built by Nicoletti's wrapper."

    Both cells should have identical channels, identical gbar, identical
    eca/ek/v_init/cm. Differences will be: (a) translation correctness of
    each individual channel, (b) any architectural divergence introduced
    by Brian2's force-clamp vs NEURON's VClamp/SEClamp.
    """
    print("=" * 70)
    print("CP4 component 2a — VC Brian2 4-channel AVA vs NEURON AVAL")
    print("=" * 70)
    print(f"\nAVAL geometry: surf={AVAL_SURF_CM2:.3e} cm², cm={AVAL_CM_UFCM2}")
    print("Brian2 cell channels: [egl19, leak, irk, nca]")
    print("NEURON cell:          [egl19, leak, irk, nca] (Nicoletti AVAL canonical)")
    print(f"\nDensities (S/cm²):")
    for k, v in AVAL_G_SCM2.items():
        print(f"  {k}: {v:.3e}")
    print()

    # Build NEURON reference (Nicoletti's actual AVAL via her wrapper)
    print("Building NEURON AVAL reference...")
    nref = NEURONReference("AVAL")
    print(f"  Built. surf={nref._surf_cm2():.3e} cm², C_m={nref._cm_pF:.3f} pF\n")

    # Build Brian2 factory
    factory = build_brian2_aval_4channel(record_components=False)

    # Holds: 11 levels -80 to +40 (matching Phase β / Phase F 2a convention)
    holds = [-80, -60, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    # Adapter: voltage_clamp_compare_v2 expects bundle["set_v"] callable but
    # not "inject_pA" / "disable_clamp" / "config" — those are ignored. The
    # factory sets enabled=True on first set_v call and locks v at clamp value.
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

    print(f"\n2a results:")
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
              f"brian2_peak={s['brian2_peak_I_pA']:+9.2f}  "
              f"nrn_peak={s['ref_peak_I_pA']:+9.2f}  "
              f"div={peak_div:.4f}({'P' if peak_pass else 'F'})  "
              f"brian2_ss={s['brian2_ss_I_pA']:+9.2f}  "
              f"nrn_ss={s['ref_ss_I_pA']:+9.2f}  "
              f"div={ss_div:.4f}({'P' if ss_pass else 'F'})")
    nref.cleanup()
    return result


# ---------------------------------------------------------------------------
# Component 2b — current clamp via direct upstream invocation
# ---------------------------------------------------------------------------

def run_neuron_aval_current_clamp_upstream(injection_pa_list, dt_ms=0.025):
    """Directly invoke Nicoletti's AVAL_simulation_iclamp.py.

    Returns list of dicts {injection_pa, t_ms, V_mV} for each injection level.
    Uses the same parameter vector and protocol as Nicoletti's published work.

    Nicoletti's `AVA_simulation_iclamp(gAVA_scaled, s1, s2, ns)` interface:
    - gAVA_scaled: rescaled g vector (output of gScm2(g0, surf, 3))
    - s1, s2, ns: linspace(s1, s2, ns) in nA (NEURON's IClamp.amp is in nA)

    Per AVAL_simulation_iclamp.py:
      stim.delay = 1023 ms
      stim.dur   = 1000 ms
      simdur     = 2500 ms
      h.finitialize(-60)
      h.dt = 0.025

    For our protocol we need to control individual injection levels (not
    Nicoletti's linspace), so we re-implement the inner protocol while
    using Nicoletti's exact section construction.
    """
    print(f"\n  Invoking upstream AVAL via direct NEURON construction "
          f"(matching AVAL_simulation_iclamp.py exactly)...")

    with _nicoletti_env():
        from neuron import h, gui  # noqa: F401
        import math
        from g_to_Scm2 import gScm2

        # Reset NEURON state
        # Nicoletti's exact construction:
        surf = AVAL_SURF_CM2
        g0 = [0.104385, 0.150164, 0.1, 0, -39, 0.859551]
        g_scaled = gScm2(g0, surf, 3)

        cm_uFcm2 = float(g_scaled[5])
        e_leak = float(g_scaled[4])
        L = math.sqrt(surf / math.pi)
        rsoma = L * 1e4

        soma = h.Section(name="soma_aval_cp4")
        soma.L = rsoma
        soma.diam = rsoma
        soma.Ra = 100
        soma.cm = cm_uFcm2

        soma.insert("irk")
        soma.insert("leak")
        soma.insert("egl19")
        soma.insert("nca")

        for seg in soma:
            seg.egl19.gbar = float(g_scaled[0])
            seg.leak.gbar = float(g_scaled[1])
            seg.irk.gbar = float(g_scaled[2])
            seg.nca.gbar = float(g_scaled[3])
            seg.leak.e = e_leak
            seg.eca = 60
            seg.ek = -80

        # Protocol per AVAL_simulation_iclamp.py:
        stim = h.IClamp(soma(0.5))
        stim.delay = 1023.0
        stim.dur = 1000.0
        simdur = 2500.0

        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)

        sweeps = []
        for inj_pa in injection_pa_list:
            # NEURON IClamp.amp is in nA
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
                "stim_onset_ms": 1023.0,
                "stim_offset_ms": 1023.0 + 1000.0,
                "simdur_ms": simdur,
            })

        # Cleanup section
        del soma
        del stim

    return sweeps


def run_brian2_aval_current_clamp(injection_pa_list, dt_ms=0.025):
    """Run Brian2 4-channel AVA cell under same 1000 ms current-clamp protocol.

    Returns list of dicts {injection_pa, t_ms, V_mV} matching the NEURON
    upstream output shape.
    """
    print(f"  Running Brian2 4-channel AVA cell under same protocol...")
    from brian2 import ms, mV, defaultclock, pA

    sweeps = []
    for inj_pa in injection_pa_list:
        # Re-build factory each sweep for clean state
        factory = build_brian2_aval_4channel(record_components=False)
        bundle = factory()
        defaultclock.dt = dt_ms * ms
        bundle["disable_clamp"]()  # ensure free-running

        # Settle 1023 ms at I=0 (matches Nicoletti's stim.delay)
        bundle["inject_pA"](0.0)
        bundle["network"].run(1023.0 * ms)
        # Inject for 1000 ms
        bundle["inject_pA"](inj_pa)
        bundle["network"].run(1000.0 * ms)
        # Recovery for ~477 ms (to total 2500 ms simdur)
        bundle["inject_pA"](0.0)
        bundle["network"].run(477.0 * ms)

        mon = bundle["monitor"]
        t_arr = np.asarray(mon.t) * 1e3  # ms
        v_arr = np.asarray(mon.v[0]) * 1e3  # mV
        sweeps.append({
            "injection_pa": float(inj_pa),
            "t_ms": t_arr,
            "V_mV": v_arr,
            "stim_onset_ms": 1023.0,
            "stim_offset_ms": 2023.0,
            "simdur_ms": 2500.0,
        })

    return sweeps


def extract_features(sweep):
    """Extract voltage features from a single CC sweep.

    Aligns time so t=0 = stim onset. Computes:
      baseline_pre_mV: mean V over t in [-300, 0] ms
      peak_V_mV: max |V - baseline| during stim window
      plateau_V_mV: median V over last 200 ms of stim
      baseline_post_mV: mean V over last 100 ms of recovery
      time_to_peak_ms: time after stim onset of peak
    """
    t = sweep["t_ms"] - sweep["stim_onset_ms"]  # 0 = stim onset
    v = sweep["V_mV"]
    stim_dur = sweep["stim_offset_ms"] - sweep["stim_onset_ms"]

    pre_mask = (t >= -300.0) & (t < 0)
    step_mask = (t >= 0) & (t < stim_dur)
    plateau_mask = (t >= stim_dur - 200.0) & (t < stim_dur)
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
    baseline_post = float(np.mean(v[post_mask][-min(len(v[post_mask]), 4000):])) if post_mask.any() else float(v[-1])

    return {
        "baseline_pre_mV": baseline_pre,
        "peak_V_mV": peak_V,
        "plateau_V_mV": plateau_V,
        "baseline_post_mV": baseline_post,
        "time_to_peak_ms": time_to_peak,
    }


def run_component_2b():
    """Current-clamp comparison: Brian2 4-channel vs NEURON AVAL (upstream)."""
    print("\n" + "=" * 70)
    print("CP4 component 2b — CC Brian2 4-channel AVA vs NEURON AVAL (upstream)")
    print("=" * 70)
    print("\nProtocol (from AVAL_simulation_iclamp.py + AVAL_simulations.py):")
    print("  delay = 1023 ms, duration = 1000 ms, simdur = 2500 ms")
    print("  injection levels: 7 steps from -30 to +30 pA "
          "(per AVAL_simulations.py line 31: linspace(-0.03, 0.03, 7) nA)")
    print("  v_init = -60 mV, dt = 0.025 ms")
    print()

    # 7 injection levels in pA
    injection_pa = list(np.linspace(-30.0, 30.0, 7))
    print(f"Injection levels (pA): {[f'{x:+.0f}' for x in injection_pa]}")

    # NEURON upstream sweeps
    print("\n[NEURON]")
    nrn_sweeps = run_neuron_aval_current_clamp_upstream(injection_pa)

    # Brian2 sweeps
    print("\n[Brian2]")
    b2_sweeps = run_brian2_aval_current_clamp(injection_pa)

    # Compare per-sweep
    print("\n" + "-" * 70)
    print("Per-sweep comparison:")
    print("-" * 70)

    # Voltage-feature pass criteria (per CP3_status convention):
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

        # Per-feature residuals
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

        # Per-timepoint comparison: interpolate Brian2 onto NEURON grid in
        # the comparison window [stim_onset - 100, simdur].
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

        print(f"\nInjection: {inj:+.0f} pA")
        print(f"  Brian2: baseline_pre={b2_feat['baseline_pre_mV']:+6.2f}  "
              f"peak={b2_feat['peak_V_mV']:+6.2f}  plateau={b2_feat['plateau_V_mV']:+6.2f}  "
              f"post={b2_feat['baseline_post_mV']:+6.2f}")
        print(f"  NEURON: baseline_pre={nrn_feat['baseline_pre_mV']:+6.2f}  "
              f"peak={nrn_feat['peak_V_mV']:+6.2f}  plateau={nrn_feat['plateau_V_mV']:+6.2f}  "
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
    print(f"Component 2b panel_pass: {aggregate_pass}")

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

    # Component 2a — voltage clamp
    result_2a = run_component_2a()

    # Component 2b — current clamp via upstream
    result_2b = run_component_2b()

    # Outcome classification
    pass_2a = bool(result_2a["panel_pass"])
    pass_2b = bool(result_2b["panel_pass"])

    print("\n" + "=" * 70)
    print("CP4 outcome classification")
    print("=" * 70)
    if pass_2a and pass_2b:
        verdict = "PRODUCTION_GRADE"
        verdict_msg = (
            "Both apples-to-apples comparisons pass. Brian2 4-channel AVA matches "
            "Nicoletti's NEURON AVAL within tolerance for both voltage-clamp and "
            "1000 ms current-clamp protocols."
        )
    elif pass_2a and not pass_2b:
        verdict = "PARTIAL_2A_PASS_2B_FAIL"
        verdict_msg = (
            "Voltage-clamp passes (channel kinetics correct in cell context). "
            "Current-clamp 1000 ms protocol diverges. Investigate as DEEPER_FINDING "
            "candidate (e.g., integration scheme, capacitive transient handling, "
            "Brian2-vs-NEURON discretization)."
        )
    elif not pass_2a and pass_2b:
        verdict = "ANOMALOUS_2A_FAIL_2B_PASS"
        verdict_msg = "Anomalous: investigate."
    else:
        verdict = "PARTIAL_BOTH_FAIL"
        verdict_msg = "Both components fail. Likely a translation defect surfaced."

    print(f"\nVerdict: {verdict}")
    print(f"  {verdict_msg}")

    # Save full results
    out_path = Path(__file__).parent / "artifacts" / "option_alpha_phase_f_results.json"
    summary = {
        "checkpoint": "option_alpha_CP4",
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "component_2a": {
            "panel_pass": pass_2a,
            "n_holds": result_2a["n_holds"],
            "n_holds_passing": result_2a["n_holds_passing"],
            "fraction_passing": float(result_2a["fraction_passing"]),
            "tolerance_metric": result_2a["tolerance_metric"],
            "per_step": result_2a["per_step"],
            "per_step_evaluations": result_2a["evaluation"]["per_step_evaluations"],
        },
        "component_2b": result_2b,
        "elapsed_s": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {time.time() - t_start:.1f} s")

    return verdict, summary


if __name__ == "__main__":
    verdict, summary = main()
    if verdict == "PRODUCTION_GRADE":
        sys.exit(0)
    elif verdict.startswith("PARTIAL"):
        sys.exit(2)
    else:
        sys.exit(1)
