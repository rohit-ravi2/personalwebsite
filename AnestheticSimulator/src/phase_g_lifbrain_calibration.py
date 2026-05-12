"""Phase G CP2 — behavioral threshold calibration on production LIFBrain.

Halothane dose-response sweep on M2-pure + recalibrated-stack production
substrate. Anchors against Crowder 1996 PMID 8873562: ~50% behavioral
suppression at clinical EC50 (~3% atm / ~280 µM aqueous).

Primary readout (per Phase G LIFBrain pre-flight Decision 5): FWD state
fraction. 50% suppression = FWD fraction at dose=k is 0.5 × FWD fraction
at dose=0 (no perturbation baseline).

Secondary diagnostic: AVA + AVB command interneuron firing rates.

Target outcomes:
  - 50%-suppression dose within 2× of clinical EC50 → calibration success
  - within 5× → partial calibration (document gap honestly)
  - beyond 5× → HARD STOP (write CALIBRATION_GAP.md and pause)

Compute budget: 6 dose points × 5 seeds × ~70s wall = ~35 min.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_g_lifbrain_substrate import (
    make_lifbrain_substrate,
    lifbrain_behavioral_readout,
    run_one_dose,
    PHASE_G_DIR,
    OVERLAY_V2,
)

# Dose multipliers per CP2 spec (extended to 6 points for cleaner curve)
DOSES = (0.001, 0.01, 0.1, 1.0, 10.0)
N_SEEDS = 5
DURATION_S = 30.0
SCENARIO = "spontaneous"


def run_baseline_unperturbed(seeds: list[int]) -> list[dict]:
    """No-perturbation baseline runs for the suppression denominator."""
    print(f"\n  Baseline (no perturbation), {len(seeds)} seeds × {DURATION_S}s:")
    results = []
    for s in seeds:
        t0 = time.time()
        env = make_lifbrain_substrate(s)
        env.run(DURATION_S, stim_schedule=[])
        readout = lifbrain_behavioral_readout(env)
        wall = time.time() - t0
        results.append({
            "seed": s,
            "fwd_fraction": readout["fwd_fraction"],
            "mean_firing_rate_hz": readout["mean_firing_rate_hz"],
            "command_rates_hz": readout["command_interneuron_rates_hz"],
            "fsm_state_fractions": readout["fsm_state_fractions"],
            "wall_s": round(wall, 1),
        })
        print(f"    seed={s}: FWD={readout['fwd_fraction']:.3f} "
              f"mean_rate={readout['mean_firing_rate_hz']:.2f} Hz "
              f"({wall:.1f}s wall)")
    return results


def run_dose_response_halothane(seeds: list[int], doses: tuple) -> list[dict]:
    """Per-dose runs for halothane on production substrate."""
    results = []
    for dose in doses:
        print(f"\n  Halothane dose={dose}× clinical EC50, {len(seeds)} seeds:")
        for s in seeds:
            res = run_one_dose("halothane", dose=dose, seed=s,
                                duration_s=DURATION_S)
            r = res["readout"]
            results.append({
                "anesthetic": "halothane",
                "dose_multiplier": dose,
                "seed": s,
                "fwd_fraction": r["fwd_fraction"],
                "mean_firing_rate_hz": r["mean_firing_rate_hz"],
                "command_rates_hz": r["command_interneuron_rates_hz"],
                "fsm_state_fractions": r["fsm_state_fractions"],
                "perturbation_summary": res["perturbation_summary"],
                "wall_s": res["wall_clock"]["total_s"],
            })
            print(f"    seed={s}: FWD={r['fwd_fraction']:.3f} "
                  f"mean_rate={r['mean_firing_rate_hz']:.2f} Hz "
                  f"({res['wall_clock']['total_s']:.1f}s wall)")
    return results


def fit_ec50_from_fwd(baseline_fwd_mean: float,
                        dose_means: dict[float, float]) -> dict:
    """Identify simulator's 50%-suppression dose from FWD fraction data.

    Suppression = 1 - (FWD_at_dose / FWD_baseline). 50% suppression =
    FWD_at_dose = 0.5 × FWD_baseline.

    Simple linear interpolation between bracket doses; if no bracket
    contains 50%, report the limit.
    """
    target_fwd = 0.5 * baseline_fwd_mean

    sorted_doses = sorted(dose_means.keys())
    fwd_at_dose = [dose_means[d] for d in sorted_doses]

    # Check whether suppression is monotonic
    is_monotonic = all(fwd_at_dose[i+1] <= fwd_at_dose[i] + 0.05
                        for i in range(len(fwd_at_dose) - 1))

    # Bracket-search for 50% crossing
    ec50_dose = None
    for i in range(len(sorted_doses) - 1):
        if fwd_at_dose[i] >= target_fwd and fwd_at_dose[i + 1] <= target_fwd:
            # Linear interp in log-dose space
            d_lo, d_hi = sorted_doses[i], sorted_doses[i + 1]
            f_lo, f_hi = fwd_at_dose[i], fwd_at_dose[i + 1]
            if abs(f_lo - f_hi) > 1e-6:
                frac = (f_lo - target_fwd) / (f_lo - f_hi)
                import math
                log_d_lo, log_d_hi = math.log10(d_lo), math.log10(d_hi)
                log_ec50 = log_d_lo + frac * (log_d_hi - log_d_lo)
                ec50_dose = 10 ** log_ec50
                break

    # Calibration verdict
    if ec50_dose is None:
        if fwd_at_dose[0] < target_fwd:
            verdict = "EC50 below lowest tested dose"
        elif fwd_at_dose[-1] > target_fwd:
            verdict = "EC50 above highest tested dose (insufficient suppression)"
        else:
            verdict = "non-monotonic — no clean bracket"
    else:
        fold_off = max(ec50_dose, 1.0 / ec50_dose)
        if fold_off <= 2.0:
            verdict = "SUCCESS (within 2× of clinical EC50)"
        elif fold_off <= 5.0:
            verdict = "PARTIAL (within 5× of clinical EC50)"
        else:
            verdict = "HARD STOP — beyond 5× of clinical EC50"

    return {
        "baseline_fwd_fraction": baseline_fwd_mean,
        "target_fwd_for_50pct_suppression": target_fwd,
        "doses_tested": sorted_doses,
        "fwd_at_each_dose": fwd_at_dose,
        "is_monotonic_suppression": is_monotonic,
        "ec50_simulator": ec50_dose,
        "ec50_clinical_reference": 1.0,
        "fold_off_clinical": (max(ec50_dose, 1.0 / ec50_dose)
                               if ec50_dose else None),
        "verdict": verdict,
    }


def main() -> int:
    seeds = list(range(42, 42 + N_SEEDS))
    print("=" * 78)
    print("  Phase G CP2 — behavioral threshold calibration vs Crowder 1996")
    print(f"  Substrate: production LIFBrain (M2-pure + recalibrated stack)")
    print(f"  Anesthetic: halothane")
    print(f"  Doses: {DOSES}× clinical EC50")
    print(f"  n_seeds × duration: {N_SEEDS} × {DURATION_S}s")
    print(f"  Primary readout: FWD state fraction (Crowder swimming-behavior analog)")
    print("=" * 78)

    t_start = time.time()

    # Baseline (no perturbation)
    baseline_results = run_baseline_unperturbed(seeds)
    baseline_fwd = [b["fwd_fraction"] for b in baseline_results]
    baseline_fwd_mean = statistics.mean(baseline_fwd)
    baseline_fwd_sem = (statistics.stdev(baseline_fwd) / math.sqrt(len(baseline_fwd))
                         if len(baseline_fwd) > 1 else 0.0)
    print(f"\n  Baseline FWD fraction (no perturbation): "
          f"{baseline_fwd_mean:.3f} ± {baseline_fwd_sem:.3f}")

    # Halothane dose-response
    dose_results = run_dose_response_halothane(seeds, DOSES)

    # Aggregate per-dose
    by_dose = {}
    for r in dose_results:
        by_dose.setdefault(r["dose_multiplier"], []).append(r)
    dose_means = {
        d: statistics.mean([r["fwd_fraction"] for r in rs])
        for d, rs in by_dose.items()
    }
    dose_sems = {
        d: (statistics.stdev([r["fwd_fraction"] for r in rs])
            / math.sqrt(len(rs))) if len(rs) > 1 else 0.0
        for d, rs in by_dose.items()
    }

    print(f"\n  Per-dose FWD fraction (mean ± SEM):")
    print(f"  {'dose':>10}  {'FWD_mean':>10}  {'FWD_sem':>10}  {'suppression':>12}")
    for d in sorted(dose_means):
        suppr = 1 - (dose_means[d] / baseline_fwd_mean) if baseline_fwd_mean > 0 else 0
        print(f"  {d:>10}x  {dose_means[d]:>10.3f}  {dose_sems[d]:>10.3f}  "
              f"{suppr:>11.1%}")

    # Fit + verdict
    ec50_analysis = fit_ec50_from_fwd(baseline_fwd_mean, dose_means)
    print(f"\n  EC50 analysis:")
    print(f"    Target FWD (50% suppression): {ec50_analysis['target_fwd_for_50pct_suppression']:.3f}")
    print(f"    Simulator EC50: {ec50_analysis['ec50_simulator']}")
    print(f"    Fold-off from clinical: {ec50_analysis['fold_off_clinical']}")
    print(f"    Monotonic suppression: {ec50_analysis['is_monotonic_suppression']}")
    print(f"    Verdict: {ec50_analysis['verdict']}")

    # Secondary diagnostic: command interneuron rates
    print(f"\n  Secondary diagnostic — AVA/AVB at each dose:")
    print(f"  {'dose':>10}  {'AVAL':>8}  {'AVAR':>8}  {'AVBL':>8}  {'AVBR':>8}")
    for d in sorted(by_dose.keys()):
        avg = {cn: statistics.mean([r["command_rates_hz"].get(cn, 0)
                                      for r in by_dose[d]])
               for cn in ["AVAL", "AVAR", "AVBL", "AVBR"]}
        print(f"  {d:>10}x  {avg['AVAL']:>8.1f}  {avg['AVAR']:>8.1f}  "
              f"{avg['AVBL']:>8.1f}  {avg['AVBR']:>8.1f}")
    base_cmd = {cn: statistics.mean([b["command_rates_hz"].get(cn, 0)
                                      for b in baseline_results])
                for cn in ["AVAL", "AVAR", "AVBL", "AVBR"]}
    print(f"  {'baseline':>10}  {base_cmd['AVAL']:>8.1f}  {base_cmd['AVAR']:>8.1f}  "
          f"{base_cmd['AVBL']:>8.1f}  {base_cmd['AVBR']:>8.1f}")

    # Persist
    out = {
        "_meta": {
            "substrate": "production LIFBrain (M2-pure + Phase 2 recalibrated stack)",
            "anesthetic": "halothane",
            "primary_readout": "FWD state fraction",
            "secondary_readout": "AVA/AVB command interneuron firing rates",
            "reference": "Crowder 1996 PMID 8873562",
            "doses_tested": list(DOSES),
            "n_seeds": N_SEEDS,
            "duration_s": DURATION_S,
            "scenario": SCENARIO,
            "total_wall_min": round((time.time() - t_start) / 60, 1),
        },
        "baseline_unperturbed": {
            "results": baseline_results,
            "fwd_mean": baseline_fwd_mean,
            "fwd_sem": baseline_fwd_sem,
            "cmd_means_hz": base_cmd,
        },
        "halothane_dose_response": dose_results,
        "per_dose_aggregate": {
            str(d): {"fwd_mean": dose_means[d], "fwd_sem": dose_sems[d]}
            for d in sorted(dose_means)
        },
        "ec50_analysis": ec50_analysis,
    }
    out_path = PHASE_G_DIR / "phase_g_lifbrain_cp2_calibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  CP2 results JSON: {out_path}")

    # Markdown summary
    md_path = PHASE_G_DIR / "phase_g_lifbrain_cp2_calibration.md"
    with open(md_path, "w") as f:
        f.write("# Phase G CP2 — behavioral threshold calibration on production LIFBrain\n\n")
        f.write(f"**Date:** 2026-05-12 | **Substrate:** production LIFBrain (M2-pure + Phase 2 recalibrated stack) "
                f"| **Reference:** Crowder 1996 PMID 8873562\n\n")
        f.write(f"**Total wall time:** {out['_meta']['total_wall_min']:.1f} min\n\n")
        f.write(f"## Calibration verdict\n\n")
        f.write(f"**{ec50_analysis['verdict']}**\n\n")
        if ec50_analysis['ec50_simulator']:
            f.write(f"- Simulator 50%-suppression dose: **{ec50_analysis['ec50_simulator']:.3f}× clinical EC50**\n")
            f.write(f"- Fold-off from Crowder 1996 anchor: {ec50_analysis['fold_off_clinical']:.2f}×\n")
        else:
            f.write(f"- Simulator EC50: not bracketed by tested doses\n")
        f.write(f"- Baseline FWD fraction: {baseline_fwd_mean:.3f} ± {baseline_fwd_sem:.3f}\n")
        f.write(f"- Target FWD for 50% suppression: {ec50_analysis['target_fwd_for_50pct_suppression']:.3f}\n\n")
        f.write(f"## Dose-response\n\n")
        f.write("| dose × EC50 | FWD mean | FWD SEM | suppression % | AVAL Hz | AVAR Hz | AVBL Hz | AVBR Hz |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for d in sorted(dose_means):
            suppr = 1 - (dose_means[d] / baseline_fwd_mean) if baseline_fwd_mean > 0 else 0
            avg = {cn: statistics.mean([r["command_rates_hz"].get(cn, 0)
                                          for r in by_dose[d]])
                   for cn in ["AVAL", "AVAR", "AVBL", "AVBR"]}
            f.write(f"| {d:.3f} | {dose_means[d]:.3f} | {dose_sems[d]:.3f} | "
                    f"{suppr:.1%} | {avg['AVAL']:.1f} | {avg['AVAR']:.1f} | "
                    f"{avg['AVBL']:.1f} | {avg['AVBR']:.1f} |\n")
        f.write(f"| **baseline (0×)** | {baseline_fwd_mean:.3f} | {baseline_fwd_sem:.3f} | 0% | "
                f"{base_cmd['AVAL']:.1f} | {base_cmd['AVAR']:.1f} | "
                f"{base_cmd['AVBL']:.1f} | {base_cmd['AVBR']:.1f} |\n")
        f.write(f"\nMonotonic FWD suppression with dose: **{ec50_analysis['is_monotonic_suppression']}**\n\n")
    print(f"  CP2 markdown: {md_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total wall time: {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
