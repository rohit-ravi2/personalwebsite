"""
Phase δ WB3 CP4 — Touch cascade validation + W_graded_I retune (Caveat 2).

CP4.1: Touch cascade under cross_coupling="graded_b2".
  - Run touch_anterior 30 s scenario.
  - Profile firing rates pre / peri / post touch for:
      ALM/AVM/AVD/PVC/AVA/AVB/AIB/AIY/RIM (sensory + interneurons)
  - Compare AVA Δ peri-touch to per-edge LIF baseline (+7.5 Hz from
    Stage IV, `stage_iv_touch_cascade.py` Component 1).

CP4.2: W_graded_I retune (Caveat 2 — only if AVA Δ <+5 Hz).
  - Document trajectory: starting value, test outcome, retune target,
    rationale.
  - Test ladder: 0.3 → 1.0 → 3.0 → 10.0 pA.
  - Final value: smallest W_graded_I bringing AVA Δ peri-touch ≥+5 Hz.
  - Do NOT exceed 10 pA; surface if architectural issue.

CP4.3: Wave 2 mechanistic resolution check.
  - AVAL ≠ AVAR distinguishability (different rest, different plateau).
  - Plateau dynamics realistic.
  - Behavioral state distribution comparison vs per-edge LIF baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
WAVE2_DIR = THIS_DIR.parent
BRAIN_DIR = WAVE2_DIR.parent
sys.path.insert(0, str(BRAIN_DIR))
sys.path.insert(0, str(WAVE2_DIR))
sys.path.insert(0, str(THIS_DIR))

from brian2 import ms, mV, pA, defaultclock

from wave2_hybrid_brain import Wave2HybridBrain


# Stage IV reference: per-edge LIF baseline AVA Δ peri-touch
# (per stage_IV_touch_cascade_findings.md table @ line 49-50)
STAGE_IV_BASELINE = {
    "AVAL_baseline_Hz": 28.50,
    "AVAL_touch_Hz": 36.00,
    "AVAL_recovery_Hz": 31.00,
    "AVAL_delta_touch_Hz": 7.50,
    "AVAR_baseline_Hz": 28.50,
    "AVAR_touch_Hz": 34.50,
    "AVAR_recovery_Hz": 30.50,
    "AVAR_delta_touch_Hz": 6.00,
}

KEY_CELLS = [
    "ALML", "ALMR", "AVM", "PVCL", "PVCR", "AVDL", "AVDR",
    "AVEL", "AVER", "AVAL", "AVAR", "AVBL", "AVBR",
    "AIBL", "AIBR", "RIML", "RIMR", "AIYL", "AIYR",
]


def run_touch_cascade(W_graded_I_pA, seed=42, wave2_active=None,
                      v_half_overrides=None):
    """Run a 30 s touch_anterior scenario and return per-cell rates.

    Protocol (matching Stage IV Component 1):
      - 0-3 s: spontaneous baseline
      - 3-5 s: 200 Hz Poisson on ALML/ALMR/AVM (touch_anterior stim)
      - 5-7 s: spontaneous recovery
    Total 7 s, but our cross-coupled brain settles slowly; we add 2 s
    pre-baseline so that "baseline" measurement starts at t=2 s for
    initial transients to dissipate.
    """
    if wave2_active is None:
        wave2_active = ["AVAL", "AVAR"]
    defaultclock.dt = 0.1 * ms
    brain = Wave2HybridBrain(
        wave2_active=wave2_active,
        cross_coupling="graded_b2",
        W_graded_I_pA=W_graded_I_pA,
        v_half_overrides=v_half_overrides,
        seed=seed,
    )

    # Phase 1: settle for 2 s (initial gap-junction transient settles).
    brain.run(2000)

    # Phase 2: 3 s spontaneous baseline measurement.
    brain.run(3000)
    baseline_rates = brain.firing_rates(window_ms=2000)

    # Phase 3: 2 s touch.
    for n in ["ALML", "ALMR", "AVM"]:
        if n in brain.idx:
            brain.inject_poisson(n, 200, weight_mv=8)
    brain.run(2000)
    touch_rates = brain.firing_rates(window_ms=2000)

    # Phase 4: 2 s recovery (run after stim to let activity settle).
    brain.run(2000)
    recovery_rates = brain.firing_rates(window_ms=2000)

    # W2 voltages snapshot (at end of recovery)
    v_w2 = {name: float(grp.v[0] / mV) for name, grp in brain.wave2_groups.items()}

    out = {
        "W_graded_I_pA": float(W_graded_I_pA),
        "wave2_active": list(wave2_active),
        "v_half_overrides": dict(v_half_overrides or {}),
        "soft_cap_warnings_total": int(brain.soft_cap_warning_count()),
        "wave2_voltages_end_mV": v_w2,
    }
    for c in KEY_CELLS:
        if c in brain.idx:
            i = brain.idx[c]
            b = float(baseline_rates[i])
            s = float(touch_rates[i])
            r = float(recovery_rates[i])
            out[c] = {
                "baseline_Hz": b,
                "touch_Hz": s,
                "recovery_Hz": r,
                "delta_touch_Hz": s - b,
            }
    # Wave2 pseudo-spike counts (graded_b2)
    out["wave2_pseudo_spike_counts_total"] = {
        name: len(brain.wave2_pseudo_spikes.get(name, []))
        for name in wave2_active
    }
    return out


def cp4_1_touch_cascade_default(seed=42):
    """CP4.1: cascade at default W_graded_I = 0.3 pA."""
    print("\n" + "=" * 70)
    print("CP4.1: Touch cascade @ W_graded_I = 0.3 pA (default)")
    print("=" * 70)
    t0 = time.time()
    out = run_touch_cascade(W_graded_I_pA=0.3, seed=seed)
    out["wall_time_s"] = time.time() - t0
    print(_format_cascade_report(out))
    return out


def cp4_2_w_graded_i_retune(seed=42, baseline_result=None,
                             ladder=(1.0, 3.0, 10.0), target_delta_Hz=5.0):
    """CP4.2: W_graded_I retune trajectory (Caveat 2).

    Only triggered if baseline_result['AVAL']['delta_touch_Hz'] < target_delta_Hz.

    Returns dict trajectory + final value.
    """
    print("\n" + "=" * 70)
    print(f"CP4.2: W_graded_I retune (target AVA Δ ≥ {target_delta_Hz} Hz)")
    print("=" * 70)

    trajectory = []
    if baseline_result is not None:
        trajectory.append({
            "W_graded_I_pA": 0.3,
            "AVAL_delta_Hz": baseline_result["AVAL"]["delta_touch_Hz"],
            "AVAR_delta_Hz": baseline_result["AVAR"]["delta_touch_Hz"],
            "rationale": "Mellem-calibrated starting point (Decision 4 default)",
            "outcome": "below_target" if baseline_result["AVAL"]["delta_touch_Hz"] < target_delta_Hz else "target_met",
        })

    final_W_graded_I = 0.3
    final_result = baseline_result
    if baseline_result and baseline_result["AVAL"]["delta_touch_Hz"] >= target_delta_Hz:
        print(f"  AVA Δ {baseline_result['AVAL']['delta_touch_Hz']:.2f} Hz ≥ "
              f"{target_delta_Hz} Hz at W_graded_I = 0.3 pA. No retune needed.")
    else:
        for W_pA in ladder:
            print(f"\n  Ladder step: W_graded_I = {W_pA} pA")
            t0 = time.time()
            r = run_touch_cascade(W_graded_I_pA=W_pA, seed=seed)
            r["wall_time_s"] = time.time() - t0
            ava_delta = r["AVAL"]["delta_touch_Hz"]
            avar_delta = r["AVAR"]["delta_touch_Hz"]
            outcome = "target_met" if ava_delta >= target_delta_Hz else "below_target"
            trajectory.append({
                "W_graded_I_pA": W_pA,
                "AVAL_delta_Hz": ava_delta,
                "AVAR_delta_Hz": avar_delta,
                "rationale": (
                    "Per-cell Mellem injection scale doesn't account for "
                    "cumulative summation across many active inputs OR "
                    "not all weights saturate to σ=1 in physiological regime; "
                    "scale up empirically until cascade matches per-edge LIF "
                    "baseline (+7.5 Hz from Stage IV)"
                ),
                "outcome": outcome,
                "soft_cap_warnings_total": r["soft_cap_warnings_total"],
            })
            print(_format_cascade_report(r))
            final_W_graded_I = W_pA
            final_result = r
            if ava_delta >= target_delta_Hz:
                print(f"  TARGET MET at W_graded_I = {W_pA} pA (AVAL Δ {ava_delta:.2f} Hz)")
                break

        if final_result and final_result["AVAL"]["delta_touch_Hz"] < target_delta_Hz:
            # Did not meet target up to 10 pA; surface as architectural issue.
            print(f"\n  *** WARNING: AVA Δ < {target_delta_Hz} Hz at W_graded_I = "
                  f"{ladder[-1]} pA. Architectural issue — not pushing past 10 pA "
                  "(per Caveat 2). Document and surface.")

    return {
        "starting_value_pA": 0.3,
        "target_delta_Hz": float(target_delta_Hz),
        "final_value_pA": float(final_W_graded_I),
        "final_AVAL_delta_Hz": (
            float(final_result["AVAL"]["delta_touch_Hz"])
            if final_result else None
        ),
        "trajectory": trajectory,
        "final_result": final_result,
    }


def cp4_3_mechanistic_resolution(final_result):
    """CP4.3: Wave 2 mechanistic resolution check.

    - AVAL ≠ AVAR distinguishability (different baseline rates, different
      V trajectories peri-touch).
    - Compare baseline distribution to Stage IV LIF reference.
    """
    print("\n" + "=" * 70)
    print("CP4.3: Wave 2 mechanistic resolution check")
    print("=" * 70)

    aval = final_result["AVAL"]
    avar = final_result["AVAR"]
    v_w2 = final_result["wave2_voltages_end_mV"]

    distinguishable = abs(aval["touch_Hz"] - avar["touch_Hz"]) > 0.5

    print(f"  AVAL: baseline {aval['baseline_Hz']:.2f} Hz, "
          f"touch {aval['touch_Hz']:.2f} Hz, V_end {v_w2.get('AVAL', float('nan')):+.1f} mV")
    print(f"  AVAR: baseline {avar['baseline_Hz']:.2f} Hz, "
          f"touch {avar['touch_Hz']:.2f} Hz, V_end {v_w2.get('AVAR', float('nan')):+.1f} mV")
    print(f"  AVAL ≠ AVAR distinguishable: {distinguishable}")

    # Comparison vs Stage IV LIF reference (per-edge LIF baseline)
    delta_diff = abs(aval["delta_touch_Hz"] - STAGE_IV_BASELINE["AVAL_delta_touch_Hz"])
    print(f"\n  Stage IV per-edge LIF baseline: AVAL Δ +{STAGE_IV_BASELINE['AVAL_delta_touch_Hz']:.2f} Hz")
    print(f"  WB3 graded_b2 cascade:           AVAL Δ {aval['delta_touch_Hz']:+.2f} Hz")
    print(f"  Difference vs Stage IV:          {delta_diff:.2f} Hz")

    return {
        "AVAL_AVAR_distinguishable": distinguishable,
        "AVAL_v_end_mV": v_w2.get("AVAL"),
        "AVAR_v_end_mV": v_w2.get("AVAR"),
        "AVAL_delta_touch_Hz": aval["delta_touch_Hz"],
        "AVAR_delta_touch_Hz": avar["delta_touch_Hz"],
        "stage_IV_AVAL_delta_Hz": STAGE_IV_BASELINE["AVAL_delta_touch_Hz"],
        "delta_difference_Hz": delta_diff,
    }


def _format_cascade_report(out):
    """Format cascade rates table."""
    lines = [f"\n  W_graded_I = {out['W_graded_I_pA']} pA, soft-cap warnings {out['soft_cap_warnings_total']}"]
    lines.append(f"  {'cell':<8}{'baseline':>10}{'touch':>10}{'recovery':>10}{'Δ_touch':>10}")
    for c in KEY_CELLS:
        if c in out:
            m = out[c]
            lines.append(
                f"  {c:<6}{m['baseline_Hz']:>10.2f}{m['touch_Hz']:>10.2f}"
                f"{m['recovery_Hz']:>10.2f}{m['delta_touch_Hz']:>+10.2f}"
            )
    return "\n".join(lines)


def main():
    out = {}

    # CP4.1
    cp4_1 = cp4_1_touch_cascade_default(seed=42)
    out["cp4_1_default"] = cp4_1

    # CP4.2 — only retune if AVA Δ <+5 Hz under default
    cp4_2 = cp4_2_w_graded_i_retune(seed=42, baseline_result=cp4_1)
    out["cp4_2_retune"] = cp4_2

    # CP4.3 — mechanistic resolution on the FINAL result (post-retune if applied)
    final_result = cp4_2["final_result"] or cp4_1
    cp4_3 = cp4_3_mechanistic_resolution(final_result)
    out["cp4_3_mechanistic"] = cp4_3

    # Persist
    out_path = WAVE2_DIR / "artifacts" / "phase_delta_wb3_cp4_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[CP4 results written] {out_path}")

    # Acceptance summary
    print("\n" + "=" * 70)
    print("CP4 Acceptance Summary")
    print("=" * 70)
    final_AVAL = cp4_2["final_AVAL_delta_Hz"]
    final_W = cp4_2["final_value_pA"]
    print(f"  Final W_graded_I: {final_W} pA")
    print(f"  Final AVAL Δ:     {final_AVAL:+.2f} Hz")
    print(f"  Stage IV baseline: +{STAGE_IV_BASELINE['AVAL_delta_touch_Hz']:.2f} Hz")
    print(f"  AVAL ≠ AVAR distinguishable: {cp4_3['AVAL_AVAR_distinguishable']}")
    return out


if __name__ == "__main__":
    main()
