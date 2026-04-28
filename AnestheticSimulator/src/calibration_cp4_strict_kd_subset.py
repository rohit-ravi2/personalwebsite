"""CP4 — Strict-Kd ground-truth subset construction.

The current ground_truth_Kd_table.csv contains 30 anchor entries, all of which
are FUNCTIONAL EC50/IC50 (electrophysiology potentiation/block, mitochondrial
O2 consumption) — NOT strict radioligand-displacement or photoaffinity Kd.

This script:
1. Annotates each entry with a directness tier (T1/T2/T3) reflecting how close
   the measurement is to a true binding Kd.
2. Identifies the T1 strict-functional subset (recombinant single-target
   electrophysiology — most direct).
3. Documents the systematic allosteric bias: positive allosteric modulator
   functional EC50 is typically 5-10× larger than direct-binding Kd
   (Forman & Miller 2016 review; Husain 2003 photoaffinity).
4. Outputs strict_subset_for_cp5.csv used by CP5 recalibration.

Directness tiers:
- T1: Recombinant single-target electrophysiology (HEK/oocyte) — cleanest
      functional readout; closest to channel-level EC50; allosteric coupling
      adds ~3-10× bias for PAMs vs direct Kd
- T2: Native tissue / native multi-subunit electrophysiology — adds endogenous
      modulation noise; ~10-30× bias possible
- T3: Whole-mitochondrial / O2 consumption — most indirect; multiple steps
      between binding and readout; ~10-100× bias possible
- NO_KD: Strict radioligand or photoaffinity Kd (none in current table)

Outputs: artifacts/calibration/cp4_directness_tiers.csv,
         artifacts/calibration/cp4_strict_subset.csv,
         artifacts/calibration/cp4_strict_kd_summary.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "artifacts" / "calibration" / "ground_truth_Kd_table.csv"
COMPARISON = ROOT / "artifacts" / "calibration" / "calibration_comparison_raw.csv"
OUT_TIERS = ROOT / "artifacts" / "calibration" / "cp4_directness_tiers.csv"
OUT_STRICT = ROOT / "artifacts" / "calibration" / "cp4_strict_subset.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "cp4_strict_kd_summary.md"


def classify_tier(method: str, conditions: str, value_type: str) -> tuple[str, str]:
    """Return (tier, rationale) for a calibration entry."""
    method_l = method.lower()
    conditions_l = conditions.lower()
    value_type_l = value_type.lower() if value_type else ""

    if "photoaffinity" in method_l or "radioligand" in method_l or "displacement" in method_l:
        return ("STRICT_KD", "direct binding measurement")

    if "o2_consumption" in method_l or "mitochondria" in conditions_l:
        return ("T3", "isolated mitochondrial O2 assay; multiple steps between binding and readout")

    if "patch-clamp" in method_l and "recombinant" in conditions_l:
        if "oocyte" in conditions_l or "hek" in conditions_l:
            return ("T1", "recombinant single-target electrophys (oocyte/HEK); cleanest functional EC50")
        return ("T1", "recombinant electrophys; cleanest functional EC50")

    if "patch-clamp" in method_l and "native" in conditions_l:
        return ("T2", "native-tissue electrophys; endogenous modulation noise")

    return ("T2", f"unclassified method={method!r} conditions={conditions!r}")


def main() -> int:
    with open(GROUND_TRUTH) as f:
        gt_rows = list(csv.DictReader(f))

    with open(COMPARISON) as f:
        comp_rows = list(csv.DictReader(f))

    # Index comparison by (target_class, anesthetic) for joining
    comp_by_key = {}
    for r in comp_rows:
        comp_by_key[(r["target_class"], r["anesthetic"])] = r

    # Annotate each ground truth row with tier
    annotated = []
    for r in gt_rows:
        tier, rationale = classify_tier(r["experimental_method"], r["conditions"], r["value_type"])
        comp = comp_by_key.get((r["target_class"], r["anesthetic"]), {})
        annotated.append({
            "target_class": r["target_class"],
            "vina_gene": comp.get("vina_gene", ""),
            "mammalian_homolog": r["mammalian_homolog"],
            "anesthetic": r["anesthetic"],
            "value_uM": r["value_uM"],
            "value_type": r["value_type"],
            "experimental_method": r["experimental_method"],
            "conditions": r["conditions"],
            "directness_tier": tier,
            "tier_rationale": rationale,
            "predicted_Kd_uM": comp.get("predicted_Kd_uM", ""),
            "log_error": comp.get("log_error", ""),
            "anchor_PMID": r["anchor_PMID"],
        })

    # Tier counts
    tier_counts = {}
    for r in annotated:
        tier_counts[r["directness_tier"]] = tier_counts.get(r["directness_tier"], 0) + 1

    print("Directness tier distribution:")
    for tier in sorted(tier_counts):
        print(f"  {tier}: {tier_counts[tier]} entries")

    # Filter strict subset — exclude entries with no value (no_significant_effect)
    # and exclude T3 (mitochondrial). T1 and STRICT_KD form the strict subset.
    def has_numeric_value(r):
        v = r["value_uM"]
        if v == "" or v is None:
            return False
        try:
            float(v)
            return True
        except (ValueError, TypeError):
            return False

    strict_rows = [r for r in annotated
                   if r["directness_tier"] in ("T1", "STRICT_KD")
                   and has_numeric_value(r)
                   and r["log_error"] not in ("", None)]
    t2_rows = [r for r in annotated
               if r["directness_tier"] == "T2"
               and has_numeric_value(r)
               and r["log_error"] not in ("", None)]
    t3_rows = [r for r in annotated
               if r["directness_tier"] == "T3"
               and has_numeric_value(r)
               and r["log_error"] not in ("", None)]

    print(f"\nStrict subset (T1 + STRICT_KD with numeric values + comparison): {len(strict_rows)}")
    print(f"T2 with comparison: {len(t2_rows)}")
    print(f"T3 with comparison: {len(t3_rows)}")

    # Compute strict-subset stats
    def stats(rows, label):
        if not rows:
            print(f"\n{label}: empty subset")
            return None
        log_errs = [float(r["log_error"]) for r in rows]
        abs_errs = [abs(e) for e in log_errs]
        within_10x = sum(1 for e in log_errs if abs(e) <= 1.0)
        within_3x = sum(1 for e in log_errs if abs(e) <= 0.477)
        n = len(rows)
        print(f"\n{label} (n={n}):")
        print(f"  mean |log_err|: {sum(abs_errs)/n:.3f}")
        print(f"  median |log_err|: {sorted(abs_errs)[n // 2]:.3f}")
        print(f"  within 10× (|log_err| ≤ 1.0): {within_10x}/{n} ({100*within_10x/n:.0f}%)")
        print(f"  within 3× (|log_err| ≤ 0.477): {within_3x}/{n} ({100*within_3x/n:.0f}%)")
        print(f"  signed mean log_err (positive = pipeline overestimates Kd, weaker than measured): {sum(log_errs)/n:+.3f}")
        return {
            "n": n,
            "mean_abs_log_err": sum(abs_errs) / n,
            "median_abs_log_err": sorted(abs_errs)[n // 2],
            "within_10x_count": within_10x,
            "within_10x_pct": 100 * within_10x / n,
            "within_3x_count": within_3x,
            "within_3x_pct": 100 * within_3x / n,
            "signed_mean_log_err": sum(log_errs) / n,
        }

    strict_stats = stats(strict_rows, "STRICT subset (T1 + STRICT_KD)")
    t2_stats = stats(t2_rows, "T2 (native-tissue)")
    t3_stats = stats(t3_rows, "T3 (mitochondrial)")

    # Write outputs
    OUT_TIERS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TIERS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(annotated[0].keys()))
        w.writeheader()
        w.writerows(annotated)

    with open(OUT_STRICT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strict_rows[0].keys()))
        w.writeheader()
        w.writerows(strict_rows)
    print(f"\nTiers CSV: {OUT_TIERS}")
    print(f"Strict subset CSV: {OUT_STRICT}")

    with open(OUT_MD, "w") as f:
        f.write("# CP4 — Strict-Kd ground-truth subset construction\n\n")
        f.write("## Directness tier framework\n\n"
                "The current ground-truth table (`ground_truth_Kd_table.csv`) labeled itself "
                "as a 'Kd table' but contains **zero strict-Kd entries**. All 30 anchor values "
                "are FUNCTIONAL readouts (electrophysiology EC50/IC50, mitochondrial O2 "
                "consumption IC50). True strict Kd would require radioligand displacement or "
                "photoaffinity binding (e.g., Hall 1994 propofol-azi-octanol, Husain 2003 "
                "etomidate-TDBzl photoaffinity, Eckenhoff 1996 halothane photoaffinity).\n\n"
                "**Why this matters:** for positive allosteric modulators (PAMs) of "
                "GABA-A/GlyR, functional potentiation EC50 is typically **3-10× larger** "
                "than direct-binding Kd because the allosteric coupling efficiency is < 1 "
                "(Forman & Miller 2016 review *Anesth Analg* 123:1297; Husain 2003 PMID "
                "12707441). For ion-channel blockers (open-channel block), functional IC50 "
                "tracks Kd more closely. For mitochondrial O2 assays, the Kd-readout chain "
                "is even less direct — multiple coupling steps, possible non-binding effects.\n\n"
                "**Directness tier assignments:**\n\n"
                "- **T1** — recombinant single-target electrophysiology (HEK/oocyte). "
                "Cleanest functional readout; PAM allosteric bias ~3-10×; channel-block "
                "bias ~1-3×.\n"
                "- **T2** — native-tissue electrophysiology. Endogenous modulation noise; "
                "additional ~3-10× bias possible.\n"
                "- **T3** — isolated mitochondrial O2 consumption assay. Multiple "
                "intervening steps; ~10-100× bias possible; not a binding measurement.\n"
                "- **STRICT_KD** — radioligand/photoaffinity displacement Kd. None present.\n\n")
        f.write("## Tier distribution\n\n")
        f.write("| Tier | Count | Description |\n|---|---|---|\n")
        tier_descriptions = {
            "T1": "recombinant single-target electrophys",
            "T2": "native-tissue / mixed",
            "T3": "mitochondrial O2 consumption",
            "STRICT_KD": "radioligand/photoaffinity Kd",
        }
        for tier in sorted(tier_counts):
            f.write(f"| {tier} | {tier_counts[tier]} | {tier_descriptions.get(tier, '')} |\n")
        f.write(f"\n## Subset statistics (entries with numeric value AND comparison row)\n\n")

        if strict_stats:
            f.write(f"### Strict subset (T1 + STRICT_KD)\n\n")
            f.write(f"- n = {strict_stats['n']}\n")
            f.write(f"- mean |log_err| = {strict_stats['mean_abs_log_err']:.3f}\n")
            f.write(f"- median |log_err| = {strict_stats['median_abs_log_err']:.3f}\n")
            f.write(f"- within 10× (|log_err| ≤ 1.0): {strict_stats['within_10x_count']}/{strict_stats['n']} ({strict_stats['within_10x_pct']:.0f}%)\n")
            f.write(f"- within 3× (|log_err| ≤ 0.477): {strict_stats['within_3x_count']}/{strict_stats['n']} ({strict_stats['within_3x_pct']:.0f}%)\n")
            f.write(f"- signed mean log_err: {strict_stats['signed_mean_log_err']:+.3f} "
                    f"({'pipeline systematically overestimates Kd (weaker than measured)' if strict_stats['signed_mean_log_err'] > 0.2 else 'pipeline systematically underestimates Kd (tighter than measured)' if strict_stats['signed_mean_log_err'] < -0.2 else 'no systematic bias'})\n\n")

        if t3_stats:
            f.write(f"### T3 (mitochondrial) subset\n\n")
            f.write(f"- n = {t3_stats['n']}\n")
            f.write(f"- mean |log_err| = {t3_stats['mean_abs_log_err']:.3f}\n")
            f.write(f"- within 10×: {t3_stats['within_10x_count']}/{t3_stats['n']} ({t3_stats['within_10x_pct']:.0f}%)\n\n")

        f.write("## Per-entry tiers and errors\n\n")
        f.write("| Target | Anesthetic | Tier | EC50/IC50 (µM) | Predicted Kd (µM) | log_err |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in annotated:
            try:
                pred = float(r["predicted_Kd_uM"])
                pred_str = f"{pred:.1f}"
                le = float(r["log_error"])
                le_str = f"{le:+.2f}"
            except (ValueError, TypeError):
                pred_str = "—"
                le_str = "—"
            try:
                v = float(r["value_uM"])
                v_str = f"{v:g}"
            except (ValueError, TypeError):
                v_str = "—"
            f.write(f"| {r['mammalian_homolog']} | {r['anesthetic']} | {r['directness_tier']} | "
                    f"{v_str} | {pred_str} | {le_str} |\n")

        f.write("\n## CP5 recalibration plan\n\n"
                "CP5 will recalibrate the pipeline using only the **T1 strict subset** as ground "
                "truth. The headline metric `% within 10×` and `% within 3×` will be recomputed; "
                "the systematic log_err signed mean will reveal whether the pipeline has a "
                "predictable allosteric bias that can be corrected by a multiplicative factor.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
