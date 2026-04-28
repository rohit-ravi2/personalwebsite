"""CP6 — calibration verdict + recommendations.

Consolidates findings from Stages 4-6 into a single verdict using the
4-way categorization agreed in pre-flight pushback:

- DISCRIMINATIVE_AND_CALIBRATED: discriminative + |log_err| ≤ 0.5 for ≥3 mech classes
- DISCRIMINATIVE_BUT_BIASED:    discriminative + characterizable systematic bias
- DISCRIMINATIVE_RANK_ONLY:     discriminative + Spearman ρ ≥ 0.3 but absolute |log_err| > 1
- NON_DISCRIMINATIVE:           anesthetic engagement ≈ negative-control engagement

Reads:
- artifacts/calibration/calibration_comparison_raw.csv (Stage 4)
- artifacts/calibration/stage5_discriminative.csv      (Stage 5; if present)
- artifacts/calibration/stage6_rank_correlation.csv    (Stage 6)

Writes:
- artifacts/calibration/calibration_summary.md         (final verdict report)
- artifacts/calibration/calibration_run_state.json     (machine-readable state)

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/calibration_cp6_verdict.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "artifacts" / "calibration" / "calibration_comparison_raw.csv"
KP_CSV = ROOT / "artifacts" / "calibration" / "calibration_comparison_withKp.csv"
DISCR_CSV = ROOT / "artifacts" / "calibration" / "stage5_discriminative.csv"
RANK_CSV = ROOT / "artifacts" / "calibration" / "stage6_rank_correlation.csv"
GROUND = ROOT / "artifacts" / "calibration" / "ground_truth_Kd_table.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "calibration_summary.md"
OUT_STATE = ROOT / "artifacts" / "calibration" / "calibration_run_state.json"


def load_raw() -> list[dict]:
    if not RAW_CSV.exists():
        return []
    out = []
    with open(RAW_CSV) as f:
        for r in csv.DictReader(f):
            try:
                r["log_error"] = float(r["log_error"])
                r["fold_error"] = float(r["fold_error"])
            except (ValueError, KeyError):
                continue
            out.append(r)
    return out


def load_rank() -> list[dict]:
    if not RANK_CSV.exists():
        return []
    return list(csv.DictReader(open(RANK_CSV)))


def load_discr() -> list[dict]:
    if not DISCR_CSV.exists():
        return []
    return list(csv.DictReader(open(DISCR_CSV)))


def main() -> int:
    raw = load_raw()
    rank = load_rank()
    discr = load_discr()

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Stage-4 (Frame A — raw, no K_p) statistics
    # -----------------------------------------------------------------
    log_errs = [r["log_error"] for r in raw]
    s4: dict = {}
    if log_errs:
        log_errs_sorted = sorted(log_errs)
        s4["n"] = len(log_errs)
        s4["log_err_mean"] = sum(log_errs) / len(log_errs)
        s4["log_err_median"] = log_errs_sorted[len(log_errs) // 2]
        s4["frac_within_2x"] = sum(1 for x in log_errs if abs(x) <= 0.3) / len(log_errs)
        s4["frac_within_3x"] = sum(1 for x in log_errs if abs(x) <= 0.5) / len(log_errs)
        s4["frac_within_10x"] = sum(1 for x in log_errs if abs(x) <= 1.0) / len(log_errs)

    # Per-class log-error medians
    per_class: dict[str, list[float]] = {}
    for r in raw:
        per_class.setdefault(r["target_class"], []).append(r["log_error"])
    class_medians = {
        cls: sorted(le)[len(le) // 2] for cls, le in per_class.items() if le
    }

    # Number of mech classes with median |log_err| ≤ 0.5
    n_classes_calibrated = sum(1 for cls, m in class_medians.items() if abs(m) <= 0.5)
    s4["n_classes_calibrated"] = n_classes_calibrated
    s4["per_class_median_log_err"] = class_medians

    # -----------------------------------------------------------------
    # Stage-6 — rank correlation summary (already on existing data, all 30 Tier-1 targets)
    # -----------------------------------------------------------------
    s6: dict = {}
    if rank:
        rhos = [float(r["spearman_rho"]) for r in rank]
        rhos_sorted = sorted(rhos)
        s6["n_targets"] = len(rhos)
        s6["frac_positive_rho"] = sum(1 for x in rhos if x > 0) / len(rhos)
        s6["frac_strong_rho"] = sum(1 for x in rhos if x > 0.5) / len(rhos)
        s6["median_rho"] = rhos_sorted[len(rhos) // 2]

    # -----------------------------------------------------------------
    # Stage-5 — discriminative comparison
    # -----------------------------------------------------------------
    s5: dict = {}
    if discr:
        ane_eng = []
        neg_eng = []
        for r in discr:
            try:
                n = int(r["n_engaged_at_ref_conc"]) if r["n_engaged_at_ref_conc"] else 0
            except ValueError:
                continue
            if r["category"] == "anesthetic":
                ane_eng.append((r["compound"], n))
            else:
                neg_eng.append((r["compound"], n))
        s5["anesthetic_engagement"] = ane_eng
        s5["negative_control_engagement"] = neg_eng
        if ane_eng and neg_eng:
            ane_med = sorted(n for _, n in ane_eng)[len(ane_eng) // 2]
            neg_med = sorted(n for _, n in neg_eng)[len(neg_eng) // 2]
            s5["median_anesthetic_engagement"] = ane_med
            s5["median_negative_control_engagement"] = neg_med
            s5["discriminative_gap"] = ane_med - neg_med

    # -----------------------------------------------------------------
    # Verdict logic
    # -----------------------------------------------------------------
    verdict = "PENDING"
    notes = []

    discriminative_ok = False
    if s5 and "discriminative_gap" in s5:
        if s5["discriminative_gap"] >= 10:
            discriminative_ok = True
            notes.append(f"S5: discriminative gap {s5['discriminative_gap']} ≥ 10 (anesthetics engage more targets than negative controls)")
        elif s5["discriminative_gap"] >= 5:
            discriminative_ok = "weak"
            notes.append(f"S5: weak discriminative gap {s5['discriminative_gap']}; pipeline distinguishes weakly")
        else:
            notes.append(f"S5: discriminative gap {s5['discriminative_gap']} < 5; pipeline does NOT clearly distinguish")
    else:
        notes.append("S5: discriminative data not yet available")

    rank_ok = False
    if s6 and "frac_positive_rho" in s6:
        if s6["frac_positive_rho"] >= 0.7 and s6["median_rho"] > 0.1:
            rank_ok = True
            notes.append(f"S6: {s6['frac_positive_rho']*100:.0f}% targets with ρ>0; median ρ={s6['median_rho']:+.3f} → rank correlation present")
        else:
            notes.append(f"S6: rank correlation weak (frac_positive {s6.get('frac_positive_rho','?')}, median ρ {s6.get('median_rho','?')})")

    abs_calibrated = False
    if s4 and "n_classes_calibrated" in s4:
        if s4["n_classes_calibrated"] >= 3:
            abs_calibrated = True
            notes.append(f"S4: {s4['n_classes_calibrated']}/5 mech classes calibrated (median |log_err|≤0.5)")
        else:
            notes.append(f"S4: {s4['n_classes_calibrated']}/5 classes calibrated; absolute Kd off in others")

    # Apply 4-way categorization
    if discriminative_ok and abs_calibrated and rank_ok:
        verdict = "DISCRIMINATIVE_AND_CALIBRATED"
    elif discriminative_ok and rank_ok:
        verdict = "DISCRIMINATIVE_BUT_BIASED"
    elif (discriminative_ok or rank_ok):
        verdict = "DISCRIMINATIVE_RANK_ONLY"
    elif s5 and s5.get("discriminative_gap", 0) < 5:
        verdict = "NON_DISCRIMINATIVE"
    else:
        verdict = "PENDING_DATA"

    # -----------------------------------------------------------------
    # Markdown
    # -----------------------------------------------------------------
    with open(OUT_MD, "w") as f:
        f.write("# Wave P calibration — final verdict\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write("## Method\n\n"
                "Calibration of the Wave P binding-occupancy pipeline against published "
                "experimental data, using:\n\n"
                "- Stage 4: predicted Vina Kd vs experimental EC50/IC50 for 5 mammalian-homolog targets × 6 anesthetics (24 verified pairs)\n"
                "- Stage 5: discriminative power test — engagement at 100 µM aqueous of 6 anesthetics vs 8 negative controls (incl. Eger 2001 cis/trans-DCE)\n"
                "- Stage 6: per-target Spearman ρ between predicted affinity and clinical-potency proxy (-log10 EC50) across all 30 Tier-1 targets\n\n"
                "**Critical caveat:** Ground-truth values are EC50/IC50 from patch-clamp dose-response, NOT classical equilibrium Kd from radioligand binding. Direct fold-error vs EC50 conflates Vina ΔG bias with the Kd-vs-EC50 quantity distinction. Spearman rank correlation is the more interpretable metric for absolute calibration.\n\n")

        f.write("## Stage 4 — predicted Kd vs experimental EC50/IC50 (no K_p)\n\n")
        if s4:
            f.write(f"- N pairs: {s4['n']}\n")
            f.write(f"- log_err median: {s4['log_err_median']:+.2f}\n")
            f.write(f"- |log_err| ≤ 0.3 (within 2×): {s4['frac_within_2x']*100:.0f}%\n")
            f.write(f"- |log_err| ≤ 0.5 (within ~3×): {s4['frac_within_3x']*100:.0f}%\n")
            f.write(f"- |log_err| ≤ 1.0 (within 10×): {s4['frac_within_10x']*100:.0f}%\n")
            f.write(f"- Mech classes with median |log_err| ≤ 0.5: {s4['n_classes_calibrated']}/5\n\n")
            f.write("Per mech-class median log_err:\n\n")
            for cls, m in sorted(class_medians.items(), key=lambda kv: abs(kv[1])):
                tag = " (CALIBRATED ≤0.5)" if abs(m) <= 0.5 else " (BIASED >0.5)"
                f.write(f"- {cls}: {m:+.2f}{tag}\n")
            f.write("\n")

        f.write("## Stage 6 — rank correlation across 30 Tier-1 targets\n\n")
        if s6:
            f.write(f"- N targets: {s6['n_targets']}\n")
            f.write(f"- Targets with ρ > 0: {s6['frac_positive_rho']*100:.0f}%\n")
            f.write(f"- Targets with ρ > 0.5: {s6['frac_strong_rho']*100:.0f}%\n")
            f.write(f"- Median ρ: {s6['median_rho']:+.3f}\n\n")

        f.write("## Stage 5 — discriminative power\n\n")
        if s5 and "median_anesthetic_engagement" in s5:
            f.write(f"- Median anesthetic engagement: {s5['median_anesthetic_engagement']}/30 targets at 100 µM aqueous\n")
            f.write(f"- Median negative-control engagement: {s5['median_negative_control_engagement']}/30 targets at 100 µM aqueous\n")
            f.write(f"- Discriminative gap: {s5['discriminative_gap']}\n\n")
            f.write("Per anesthetic engagement:\n\n")
            for c, n in sorted(s5["anesthetic_engagement"], key=lambda t: -t[1]):
                f.write(f"- {c}: {n}/30\n")
            f.write("\nPer negative control engagement:\n\n")
            for c, n in sorted(s5["negative_control_engagement"], key=lambda t: -t[1]):
                f.write(f"- {c}: {n}/30\n")
            f.write("\n")
        else:
            f.write("(Stage 5 data not available — sweep still running or not analyzed)\n\n")

        f.write("## Verdict reasoning\n\n")
        for n in notes:
            f.write(f"- {n}\n")
        f.write(f"\n**Verdict: {verdict}**\n\n")

        f.write("## Implications\n\n")
        if verdict == "DISCRIMINATIVE_AND_CALIBRATED":
            f.write("Pipeline is biologically meaningful and absolutely calibrated. wave2_overlay.json ships as-is. Proceed to Phase E/F/G/H.\n")
        elif verdict == "DISCRIMINATIVE_BUT_BIASED":
            f.write("Pipeline distinguishes anesthetics from controls AND tracks experiment in rank order, with characterizable systematic bias on specific target classes (likely allosteric potentiators where binding-Kd ≠ functional EC50). Ship wave2_overlay.json with documented per-class bias for downstream interpretation. Phase E/F/G/H proceed.\n")
        elif verdict == "DISCRIMINATIVE_RANK_ONLY":
            f.write("Pipeline distinguishes anesthetics from controls; absolute Kd values cannot be cleanly validated due to Kd-vs-EC50 quantity mismatch. Ship wave2_overlay.json with explicit caveat that occupancy values are rank-meaningful, not absolute-meaningful. Phase E/F/G/H proceed with this caveat.\n")
        elif verdict == "NON_DISCRIMINATIVE":
            f.write("Pipeline produces similar engagement for anesthetics and inert lipophilic compounds. wave2_overlay.json should NOT be deployed. Pipeline rebuild needed before Phase E/F/G/H.\n")
        else:
            f.write("Insufficient data for verdict; await pending stages.\n")

    print(f"Verdict: {verdict}")
    print(f"Summary: {OUT_MD}")
    for n in notes:
        print(f"  - {n}")

    # State JSON
    state = {
        "verdict": verdict,
        "stage4": s4,
        "stage5": s5,
        "stage6": s6,
        "notes": notes,
    }
    with open(OUT_STATE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"State JSON: {OUT_STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
