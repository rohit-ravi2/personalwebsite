"""Wave2HybridBrain investigation thread W2 — M2-pure cascade test.

Catalog §7.2 Wave2HybridBrain investigation thread. Runs Wave2HybridBrain
under M2-pure sign mode (per-edge, no DOCUMENTED_SIGN_EXCEPTIONS) and
compares cascade firing + AVA σ Δ peri-touch to:
  - Pure LIFBrain M2-pure baseline (Phase 1A: AVAL Δ +60.20 Hz)
  - Wave2HybridBrain M2-current baseline (CP2 D7-followup peredge:
    AVAL σ_baseline 0.6420, Δ_touch -0.0165)

Test purpose: disambiguate which factor caused C-37 Falsified-but-cited:
  (a) DOCUMENTED_SIGN_EXCEPTIONS (verified in Phase 1A by M1 + M2-current
      both failing cascade firing under same exceptions)
  (b) Wave 2 cellular substitution itself (changes the network's
      recurrent feedback dynamics)

If Wave2HybridBrain M2-pure shows AVA σ Δ > 0 peri-touch with cascade
firing through LIF cells: factor (a) was the cause; (b) is fine. The
graded_b2 cross-coupling implementation is sound.

If Wave2HybridBrain M2-pure shows AVA σ Δ ≈ 0 like M2-current: factor
(b) is also operative. Wave 2 substitution itself breaks recurrent
feedback. Investigation continues per W2-3, W2-4 (alternative coupling
parameters / driving-force asymmetry tests).

Protocol matches CP2 D7-followup: 30s touch_anterior at W_graded_I = 0.3 pA
(Mellem-calibrated default). Records both σ-magnitude readout and LIF
cascade rates.
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

from brian2 import ms, mV, defaultclock

from wave2_hybrid_brain import Wave2HybridBrain


KEY_LIF_CELLS = [
    "ALML", "ALMR", "AVM",
    "PVCL", "PVCR", "AVDL", "AVDR", "AVEL", "AVER",
    "AVBL", "AVBR", "AIBL", "AIBR", "RIML", "RIMR", "AIYL", "AIYR",
]
WAVE2_CELLS = ["AVAL", "AVAR"]


def run_w2_m2pure(seed=42):
    """Run Wave2HybridBrain under M2-pure (per-edge + no exceptions).

    Same protocol as CP2 D7-followup peredge run, but with sign_exceptions={}.
    """
    defaultclock.dt = 0.1 * ms
    brain = Wave2HybridBrain(
        wave2_active=WAVE2_CELLS,
        cross_coupling="graded_b2",
        W_graded_I_pA=0.3,
        seed=seed,
        use_per_edge_glu_signs=True,
        sign_exceptions={},  # M2-pure — explicit empty
    )

    # 2 s settle
    brain.run(2000)

    # 3 s baseline measurement
    brain.run(3000)
    rates_baseline = brain.firing_rates(window_ms=2000)
    sigma_baseline = brain.wave2_activities(window_ms=2000)

    # 2 s touch
    for n in ["ALML", "ALMR", "AVM"]:
        if n in brain.idx:
            brain.inject_poisson(n, 200, weight_mv=8)
    brain.run(2000)
    rates_touch = brain.firing_rates(window_ms=2000)
    sigma_touch = brain.wave2_activities(window_ms=2000)

    # 2 s recovery
    brain.run(2000)
    rates_recovery = brain.firing_rates(window_ms=2000)
    sigma_recovery = brain.wave2_activities(window_ms=2000)

    v_w2 = {name: float(grp.v[0] / mV) for name, grp in brain.wave2_groups.items()}

    out = {
        "config": "Wave2HybridBrain M2-pure (use_per_edge_glu_signs=True, sign_exceptions={})",
        "seed": seed,
        "W_graded_I_pA": 0.3,
        "wave2_active": list(WAVE2_CELLS),
        "soft_cap_warnings_total": int(brain.soft_cap_warning_count()),
        "wave2_voltages_end_mV": v_w2,
        "lif_cells": {},
        "wave2_cells": {},
        "wave2_legacy_pseudo_spike_counts": {
            name: len(brain.wave2_pseudo_spikes.get(name, []))
            for name in WAVE2_CELLS
        },
    }
    for c in KEY_LIF_CELLS:
        if c in brain.idx:
            i = brain.idx[c]
            out["lif_cells"][c] = {
                "baseline_Hz": float(rates_baseline[i]),
                "touch_Hz": float(rates_touch[i]),
                "recovery_Hz": float(rates_recovery[i]),
                "delta_touch_Hz": float(rates_touch[i] - rates_baseline[i]),
            }
    for c in WAVE2_CELLS:
        if c in brain.idx:
            i = brain.idx[c]
            sb = float(sigma_baseline.get(c, 0.0))
            st = float(sigma_touch.get(c, 0.0))
            sr = float(sigma_recovery.get(c, 0.0))
            out["wave2_cells"][c] = {
                "sigma_baseline": sb,
                "sigma_touch": st,
                "sigma_recovery": sr,
                "sigma_delta_touch": st - sb,
                "sigma_delta_post": sr - st,
                "firing_rate_baseline_proxy": float(rates_baseline[i]),
                "firing_rate_touch_proxy": float(rates_touch[i]),
                "firing_rate_delta_proxy": float(rates_touch[i] - rates_baseline[i]),
            }
    return out


def main():
    print("=" * 78)
    print("  Wave2HybridBrain investigation W2 — M2-pure cascade test")
    print("=" * 78)
    print("  Protocol: 30 s touch_anterior, AVAL+AVAR active as Wave 2 cells,")
    print("            per-edge sign mode, NO DOCUMENTED_SIGN_EXCEPTIONS")
    print("  Reference baselines:")
    print("    Pure LIFBrain M2-pure:    AVAL Δ +60.2 Hz (Phase 1A)")
    print("    W2HybridBrain M2-current: AVAL σ Δ -0.0165 (CP2 D7-followup)")
    print("=" * 78)

    t0 = time.time()
    result = run_w2_m2pure(seed=42)
    result["wall_time_s"] = time.time() - t0

    # Print report
    print(f"\n  Soft-cap warnings: {result['soft_cap_warnings_total']}")
    print(f"  AVAL V_end: {result['wave2_voltages_end_mV'].get('AVAL', float('nan')):+.1f} mV")
    print(f"  AVAR V_end: {result['wave2_voltages_end_mV'].get('AVAR', float('nan')):+.1f} mV")

    print(f"\n  LIF cascade (firing_rates() Hz, genuine spike rates):")
    print(f"  {'cell':<8}{'baseline':>10}{'touch':>10}{'Δ_touch':>10}")
    for c in KEY_LIF_CELLS:
        if c in result["lif_cells"]:
            m = result["lif_cells"][c]
            print(
                f"  {c:<6}{m['baseline_Hz']:>10.2f}{m['touch_Hz']:>10.2f}"
                f"{m['delta_touch_Hz']:>+10.2f}"
            )

    print(f"\n  Wave 2 cells (σ-magnitude readout):")
    print(f"  {'cell':<8}{'σ_base':>10}{'σ_touch':>10}{'σ_recov':>10}"
          f"{'Δ_touch':>10}{'Δ_post':>10}")
    for c in WAVE2_CELLS:
        if c in result["wave2_cells"]:
            m = result["wave2_cells"][c]
            print(
                f"  {c:<6}{m['sigma_baseline']:>10.4f}"
                f"{m['sigma_touch']:>10.4f}{m['sigma_recovery']:>10.4f}"
                f"{m['sigma_delta_touch']:>+10.4f}"
                f"{m['sigma_delta_post']:>+10.4f}"
            )

    # Acceptance summary
    print("\n" + "=" * 78)
    print("  W2 ACCEPTANCE SUMMARY")
    print("=" * 78)

    # AVDL Δ peri-touch — proxy for "cascade firing in LIF"
    avdl_delta = result["lif_cells"].get("AVDL", {}).get("delta_touch_Hz", float("nan"))
    aval_delta = result["wave2_cells"].get("AVAL", {}).get("sigma_delta_touch", float("nan"))
    aval_delta_post = result["wave2_cells"].get("AVAL", {}).get("sigma_delta_post", float("nan"))

    print(f"  AVDL Δ peri-touch:  {avdl_delta:+.2f} Hz   "
          f"(reference pure-LIF M2-pure: +60.4 Hz)")
    print(f"  AVAL σ Δ peri-touch: {aval_delta:+.4f}    "
          f"(reference M2-current: -0.0165)")
    print(f"  AVAL σ Δ post-touch: {aval_delta_post:+.4f}    (drift baseline)")
    print()
    if avdl_delta > 30 and aval_delta > 0.05:
        print("  VERDICT: cascade fires + AVA σ activates. "
              "Factor (a) [DOCUMENTED_SIGN_EXCEPTIONS] was the C-37 cause.")
        print("           Wave 2 substitution itself is fine.")
    elif avdl_delta > 30 and aval_delta < 0.05:
        print("  VERDICT: cascade fires in LIF but AVA σ does NOT activate.")
        print("           Factor (b) [Wave 2 substitution] is operative.")
        print("           Continue investigation — W2-3 (parameters), W2-4 (E_exc).")
    else:
        print("  VERDICT: cascade does not fire in LIF either. Anomalous result;")
        print("           check sign_exceptions plumbing in Wave2HybridBrain.")

    out_path = WAVE2_DIR / "artifacts" / "wb_investigation_w2_m2pure_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[results written] {out_path}")

    return result


if __name__ == "__main__":
    main()
