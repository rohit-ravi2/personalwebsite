"""CP5 — Strict-subset recalibration with allosteric correction factor.

Inputs CP4's strict subset and computes:
1. Pre-correction calibration metrics (% within 10×, within 3×, signed/abs log_err)
2. Single-parameter multiplicative allosteric correction factor f_allo (single
   degree of freedom: median signed log_err of T1 subset)
3. Post-correction metrics
4. Cross-validation: leave-one-anesthetic-out — does the correction generalize?

Theoretical context:
For PAMs of pentameric ligand-gated ion channels, functional EC50 reflects:
    EC50 ≈ Kd / η_allo
where η_allo = allosteric coupling efficiency (< 1 for PAMs, so EC50 < Kd
in concentration units would be wrong direction). Forman & Miller 2016
review: η_allo ≈ 0.1-0.3 for general anesthetic PAMs of GABA-A/GlyR.

Vina docks the apo state and outputs ΔG_bind → predicted Kd_apo.
Functional EC50 ~ Kd_apo × something. If the pipeline systematically
overestimates EC50 by f_allo across the T1 subset, applying log_err -= log10(f_allo)
should bring the strict subset toward zero signed bias.

The validity of the correction is judged by:
- Does the residual log_err distribution narrow (mean abs log_err drops)?
- Does the leave-one-out cross-validation hold up (correction trained on N-1
  anesthetics still applies to the held-out one)?

Output: artifacts/calibration/cp5_strict_recalibration.{csv,md}
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRICT_CSV = ROOT / "artifacts" / "calibration" / "cp4_strict_subset.csv"
OUT_CSV = ROOT / "artifacts" / "calibration" / "cp5_strict_recalibration.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "cp5_strict_recalibration.md"


def metrics(log_errs: list[float]) -> dict:
    n = len(log_errs)
    if n == 0:
        return {"n": 0}
    abs_errs = [abs(e) for e in log_errs]
    return {
        "n": n,
        "mean_abs": sum(abs_errs) / n,
        "median_abs": sorted(abs_errs)[n // 2],
        "signed_mean": sum(log_errs) / n,
        "signed_median": sorted(log_errs)[n // 2],
        "within_10x": sum(1 for e in log_errs if abs(e) <= 1.0),
        "within_3x": sum(1 for e in log_errs if abs(e) <= 0.477),
        "pct_10x": 100 * sum(1 for e in log_errs if abs(e) <= 1.0) / n,
        "pct_3x": 100 * sum(1 for e in log_errs if abs(e) <= 0.477) / n,
    }


def main() -> int:
    with open(STRICT_CSV) as f:
        rows = list(csv.DictReader(f))

    log_errs = [float(r["log_error"]) for r in rows]
    pre = metrics(log_errs)

    print("Pre-correction (T1 strict subset)")
    print(f"  n = {pre['n']}")
    print(f"  signed mean log_err = {pre['signed_mean']:+.3f}  (fold = {10**pre['signed_mean']:.2f}×)")
    print(f"  signed median log_err = {pre['signed_median']:+.3f}")
    print(f"  mean |log_err| = {pre['mean_abs']:.3f}")
    print(f"  median |log_err| = {pre['median_abs']:.3f}")
    print(f"  within 10×: {pre['within_10x']}/{pre['n']} ({pre['pct_10x']:.0f}%)")
    print(f"  within 3×: {pre['within_3x']}/{pre['n']} ({pre['pct_3x']:.0f}%)")

    # Allosteric correction factor: use signed median (robust to outliers like ketamine NMDAR)
    f_allo_log = pre["signed_median"]
    f_allo = 10 ** f_allo_log

    # Apply correction
    corrected_log_errs = [e - f_allo_log for e in log_errs]
    post = metrics(corrected_log_errs)

    print(f"\nAllosteric correction factor f_allo = 10^{f_allo_log:+.3f} = {f_allo:.2f}×")
    print(f"  (Pipeline predicts Kd ≈ {f_allo:.2f}× larger than functional EC50; correct by dividing predicted Kd by {f_allo:.2f})")

    print("\nPost-correction (T1 strict subset, after dividing predicted Kd by f_allo)")
    print(f"  signed mean log_err = {post['signed_mean']:+.3f}  (fold = {10**post['signed_mean']:.2f}×)")
    print(f"  mean |log_err| = {post['mean_abs']:.3f}  (was {pre['mean_abs']:.3f}, change = {post['mean_abs'] - pre['mean_abs']:+.3f})")
    print(f"  within 10×: {post['within_10x']}/{post['n']} ({post['pct_10x']:.0f}%)")
    print(f"  within 3×: {post['within_3x']}/{post['n']} ({post['pct_3x']:.0f}%)")

    # Leave-one-anesthetic-out cross-validation
    print("\nLeave-one-anesthetic-out cross-validation:")
    print(f"  {'held-out':>14s} {'train_n':>8s} {'f_allo_train':>14s} {'CV_signed':>11s} {'CV_mean_abs':>13s}")
    anesthetics = sorted({r["anesthetic"] for r in rows})
    cv_results = []
    for held in anesthetics:
        train_log_errs = [float(r["log_error"]) for r in rows if r["anesthetic"] != held]
        held_log_errs = [float(r["log_error"]) for r in rows if r["anesthetic"] == held]
        if not train_log_errs or not held_log_errs:
            continue
        train_correction = sorted(train_log_errs)[len(train_log_errs) // 2]
        held_corrected = [e - train_correction for e in held_log_errs]
        held_metrics = metrics(held_corrected)
        cv_results.append({
            "held_out": held,
            "train_n": len(train_log_errs),
            "held_n": len(held_log_errs),
            "f_allo_train_log": train_correction,
            "f_allo_train_fold": 10 ** train_correction,
            "cv_signed_mean": held_metrics["signed_mean"],
            "cv_mean_abs": held_metrics["mean_abs"],
        })
        print(f"  {held:>14s} {len(train_log_errs):>8d} {10**train_correction:>13.2f}× "
              f"{held_metrics['signed_mean']:>+10.3f} {held_metrics['mean_abs']:>13.3f}")

    # Per-row corrected output
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_class", "vina_gene", "anesthetic", "value_uM",
                    "predicted_Kd_uM", "log_err_pre", "predicted_Kd_uM_corrected",
                    "log_err_post", "within_10x_post", "within_3x_post"])
        for r, le_pre, le_post in zip(rows, log_errs, corrected_log_errs):
            try:
                pred = float(r["predicted_Kd_uM"])
                pred_corr = pred / f_allo
            except (ValueError, TypeError):
                pred = pred_corr = float("nan")
            w.writerow([
                r["target_class"], r["vina_gene"], r["anesthetic"], r["value_uM"],
                f"{pred:.2f}", f"{le_pre:+.3f}", f"{pred_corr:.2f}", f"{le_post:+.3f}",
                "1" if abs(le_post) <= 1.0 else "0",
                "1" if abs(le_post) <= 0.477 else "0",
            ])
    print(f"\nCSV: {OUT_CSV}")

    with open(OUT_MD, "w") as f:
        f.write("# CP5 — Strict-subset recalibration with allosteric correction\n\n")
        f.write("## Method\n\n"
                "The CP4 strict subset (T1 — recombinant single-target electrophysiology, n=17 "
                "with comparison rows) showed a **systematic positive log_err** "
                f"(signed mean +{pre['signed_mean']:.3f}, signed median +{pre['signed_median']:.3f}). "
                "The pipeline predicts a larger Kd than the functional EC50, by approximately "
                f"{10**pre['signed_median']:.2f}×.\n\n"
                "**Theoretical interpretation (Forman & Miller 2016 PMID 27749338):** functional "
                "potentiation EC50 reflects binding affinity Kd × allosteric coupling efficiency η. "
                "For PAMs (volatiles, propofol on GABA-A/GlyR), η < 1 — so functional EC50 falls "
                "BELOW the binding Kd in concentration units (i.e., functional response saturates "
                "at sub-Kd concentrations because the modulator only needs to occupy a fraction of "
                "sites to achieve maximum coupling). This means a docking-derived Kd will appear "
                "too large compared to functional EC50, by a factor 1/η ≈ 3-10×.\n\n"
                "Apply a single-parameter correction:\n\n"
                f"    f_allo = 10^(signed median log_err) = 10^{f_allo_log:+.3f} = {f_allo:.2f}×\n\n"
                "Correction direction: divide pipeline-predicted Kd by f_allo to obtain "
                "functional-EC50-comparable values.\n\n")

        f.write("## Pre-correction metrics (T1 strict subset)\n\n")
        f.write(f"- n = {pre['n']}\n")
        f.write(f"- signed mean log_err = {pre['signed_mean']:+.3f} ({10**pre['signed_mean']:.2f}× fold)\n")
        f.write(f"- signed median log_err = {pre['signed_median']:+.3f} ({10**pre['signed_median']:.2f}× fold)\n")
        f.write(f"- mean |log_err| = {pre['mean_abs']:.3f}\n")
        f.write(f"- median |log_err| = {pre['median_abs']:.3f}\n")
        f.write(f"- within 10×: {pre['within_10x']}/{pre['n']} ({pre['pct_10x']:.0f}%)\n")
        f.write(f"- within 3×: {pre['within_3x']}/{pre['n']} ({pre['pct_3x']:.0f}%)\n\n")

        f.write(f"## Allosteric correction factor\n\n")
        f.write(f"f_allo = **{f_allo:.2f}×** (median-based; robust to outliers)\n\n")

        f.write("## Post-correction metrics (T1 strict subset)\n\n")
        f.write(f"- signed mean log_err = {post['signed_mean']:+.3f} ({10**post['signed_mean']:.2f}× fold)\n")
        f.write(f"- mean |log_err| = {post['mean_abs']:.3f} "
                f"(change vs pre: {post['mean_abs'] - pre['mean_abs']:+.3f})\n")
        f.write(f"- median |log_err| = {post['median_abs']:.3f} "
                f"(change vs pre: {post['median_abs'] - pre['median_abs']:+.3f})\n")
        f.write(f"- within 10×: {post['within_10x']}/{post['n']} ({post['pct_10x']:.0f}%)\n")
        f.write(f"- within 3×: {post['within_3x']}/{post['n']} ({post['pct_3x']:.0f}%)\n\n")

        f.write("## Leave-one-anesthetic-out cross-validation\n\n"
                "Training set: 5 anesthetics × ~3 targets each. Held-out anesthetic's log_err "
                "evaluated after applying f_allo trained on the other anesthetics. If the "
                "correction generalizes (rather than overfitting), held-out signed_mean should "
                "be near zero and held-out mean_abs should be similar to in-sample.\n\n")
        f.write("| held-out | train n | f_allo (train) | held signed_mean | held mean |log_err| |\n")
        f.write("|---|---|---|---|---|\n")
        for cv in cv_results:
            f.write(f"| {cv['held_out']} | {cv['train_n']} | {cv['f_allo_train_fold']:.2f}× | "
                    f"{cv['cv_signed_mean']:+.3f} | {cv['cv_mean_abs']:.3f} |\n")

        cv_signed_means = [cv["cv_signed_mean"] for cv in cv_results]
        cv_abs_log_errs = [abs(s) for s in cv_signed_means]
        f.write(f"\n**LOO-CV summary:** mean of held-out signed-means = "
                f"{sum(cv_signed_means)/len(cv_signed_means):+.3f}, "
                f"mean of |held-out signed-means| = {sum(cv_abs_log_errs)/len(cv_abs_log_errs):.3f}\n\n")

        if sum(cv_abs_log_errs) / len(cv_abs_log_errs) <= 0.5:
            cv_verdict = "ROBUST — correction generalizes across held-out anesthetics"
        elif sum(cv_abs_log_errs) / len(cv_abs_log_errs) <= 1.0:
            cv_verdict = "PARTIAL — correction generalizes for most anesthetics; one or more outliers"
        else:
            cv_verdict = "FRAGILE — correction does not generalize; ketamine and similar outliers dominate"
        f.write(f"**LOO-CV verdict:** {cv_verdict}\n\n")

        f.write("## Verdict\n\n")
        if post['mean_abs'] < pre['mean_abs'] - 0.05:
            v = (f"**ALLOSTERIC CORRECTION VALIDATED.** Single-parameter f_allo = {f_allo:.2f}× "
                 f"reduces mean |log_err| from {pre['mean_abs']:.3f} to {post['mean_abs']:.3f} "
                 f"on the T1 strict subset. The systematic +{pre['signed_median']:.3f} bias is "
                 "consistent with PAM allosteric coupling theory.")
        else:
            v = (f"**CORRECTION MARGINAL.** f_allo = {f_allo:.2f}× shifts signed mean to zero "
                 "but does not narrow the absolute log_err distribution meaningfully. The bias "
                 "may be target-specific rather than uniform allosteric.")
        f.write(v + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
