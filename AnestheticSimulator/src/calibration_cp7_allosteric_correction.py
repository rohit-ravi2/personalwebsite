"""CP7 — Allosteric correction + chemical class stratification.

Apply the CP5 allosteric correction factor (f_allo = 2.5×) to the full
calibration set, then stratify by chemical class to test whether the universal
correction is robust across anesthetic chemotypes or whether class-specific
corrections are needed.

Chemical classes:
- ALKANE_HALOGENATED: halothane (CHBrClCF3) — single anesthetic alkane in set
- ETHER_HALOGENATED: isoflurane, sevoflurane (halogenated ethers)
- IV_PHENOL: propofol (alkylphenol)
- IV_IMIDAZOLE: etomidate (carboxylated imidazole)
- IV_ARYLCYCLOHEXYLAMINE: ketamine (NMDAR antagonist)

Halogenated non-immobilizer comparison (from negative_vina_results.csv):
- hexafluoroethane (CF3CF3) — non-anesthetic per Eger 2001 despite high lipid
  solubility; pipeline expectation: should engage few/no targets at clinical
  concentrations or with very weak affinity even at saturating concentrations.

Outputs:
- artifacts/calibration/cp7_corrected.csv (per-row corrected predictions)
- artifacts/calibration/cp7_class_stratified.csv (per-class metrics)
- artifacts/calibration/cp7_summary.md
- artifacts/kinetics/wave2_overlay_v2.json (corrected occupancies for downstream)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP_CSV = ROOT / "artifacts" / "calibration" / "calibration_comparison_raw.csv"
TIERS_CSV = ROOT / "artifacts" / "calibration" / "cp4_directness_tiers.csv"
NEG_CSV = ROOT / "artifacts" / "calibration" / "negative_vina_results.csv"
OVERLAY_V1 = ROOT / "artifacts" / "kinetics" / "wave2_overlay.json"
OUT_CORR = ROOT / "artifacts" / "calibration" / "cp7_corrected.csv"
OUT_STRAT = ROOT / "artifacts" / "calibration" / "cp7_class_stratified.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "cp7_summary.md"
OUT_OVERLAY = ROOT / "artifacts" / "kinetics" / "wave2_overlay_v2.json"

F_ALLO_LOG = 0.399  # CP5 median-based correction factor
F_ALLO = 10 ** F_ALLO_LOG  # 2.50

CLASS_MAP = {
    "halothane": "ALKANE_HALOGENATED",
    "isoflurane": "ETHER_HALOGENATED",
    "sevoflurane": "ETHER_HALOGENATED",
    "propofol": "IV_PHENOL",
    "etomidate": "IV_IMIDAZOLE",
    "ketamine": "IV_ARYLCYCLOHEXYLAMINE",
}


def metrics(log_errs: list[float]) -> dict:
    n = len(log_errs)
    if n == 0:
        return {"n": 0, "mean_abs": float("nan"), "median_abs": float("nan"),
                "signed_mean": float("nan"), "within_10x": 0, "within_3x": 0,
                "pct_10x": 0.0, "pct_3x": 0.0}
    abs_errs = [abs(e) for e in log_errs]
    return {
        "n": n,
        "mean_abs": sum(abs_errs) / n,
        "median_abs": sorted(abs_errs)[n // 2],
        "signed_mean": sum(log_errs) / n,
        "within_10x": sum(1 for e in log_errs if abs(e) <= 1.0),
        "within_3x": sum(1 for e in log_errs if abs(e) <= 0.477),
        "pct_10x": 100 * sum(1 for e in log_errs if abs(e) <= 1.0) / n,
        "pct_3x": 100 * sum(1 for e in log_errs if abs(e) <= 0.477) / n,
    }


def main() -> int:
    # Load comparison rows (full set including T2 & T3)
    with open(COMP_CSV) as f:
        comp_rows = list(csv.DictReader(f))

    # Load tier annotations to filter
    with open(TIERS_CSV) as f:
        tier_rows = {(r["target_class"], r["anesthetic"]): r["directness_tier"]
                     for r in csv.DictReader(f)}

    # Apply correction: log_err_corr = log_err - f_allo_log; predicted_Kd /= f_allo
    corrected = []
    for r in comp_rows:
        try:
            le_pre = float(r["log_error"])
            pred = float(r["predicted_Kd_uM"])
        except (ValueError, TypeError):
            continue
        tier = tier_rows.get((r["target_class"], r["anesthetic"]), "UNKNOWN")
        chem_class = CLASS_MAP.get(r["anesthetic"], "UNKNOWN")
        le_post = le_pre - F_ALLO_LOG
        pred_corr = pred / F_ALLO
        corrected.append({
            "target_class": r["target_class"],
            "vina_gene": r["vina_gene"],
            "anesthetic": r["anesthetic"],
            "chem_class": chem_class,
            "directness_tier": tier,
            "experimental_value_uM": r["experimental_value_uM"],
            "value_type": r["experimental_value_type"],
            "predicted_Kd_uM_pre": f"{pred:.2f}",
            "predicted_Kd_uM_post": f"{pred_corr:.2f}",
            "log_err_pre": f"{le_pre:+.3f}",
            "log_err_post": f"{le_post:+.3f}",
            "within_10x_post": "1" if abs(le_post) <= 1.0 else "0",
            "within_3x_post": "1" if abs(le_post) <= 0.477 else "0",
        })

    # Per-chemical-class metrics, T1 only (cleanest)
    print(f"Allosteric correction f_allo = {F_ALLO:.2f}× (log {F_ALLO_LOG:+.3f})")
    print()
    print("Per-chemical-class metrics — T1 strict subset only")
    print(f"  {'class':>26s} {'n':>3s} {'pre_signed':>11s} {'post_signed':>12s} "
          f"{'pre_meanAbs':>12s} {'post_meanAbs':>13s} {'post_pct10x':>12s}")

    class_rows = {}
    for r in corrected:
        if r["directness_tier"] != "T1":
            continue
        class_rows.setdefault(r["chem_class"], []).append(r)

    strat_data = []
    for chem_class, rows in sorted(class_rows.items()):
        pre = [float(r["log_err_pre"]) for r in rows]
        post = [float(r["log_err_post"]) for r in rows]
        pre_m = metrics(pre)
        post_m = metrics(post)
        strat_data.append({
            "chem_class": chem_class,
            "n": pre_m["n"],
            "pre_signed_mean": pre_m["signed_mean"],
            "post_signed_mean": post_m["signed_mean"],
            "pre_mean_abs": pre_m["mean_abs"],
            "post_mean_abs": post_m["mean_abs"],
            "post_pct_10x": post_m["pct_10x"],
            "post_pct_3x": post_m["pct_3x"],
        })
        print(f"  {chem_class:>26s} {pre_m['n']:>3d} {pre_m['signed_mean']:>+10.3f} "
              f"{post_m['signed_mean']:>+11.3f} {pre_m['mean_abs']:>12.3f} "
              f"{post_m['mean_abs']:>13.3f} {post_m['pct_10x']:>11.0f}%")

    # Halogenated non-immobilizer baseline: hexafluoroethane vs anesthetic alkane (halothane)
    print("\nHalogenated non-immobilizer baseline (hexafluoroethane vs halothane)")
    with open(NEG_CSV) as f:
        neg_rows = list(csv.DictReader(f))

    # Compute Kd for hexafluoroethane and halothane across all targets where
    # hexafluoroethane has a Vina score; compute engagement at 1 mM aqueous (clinical).
    R_KCAL = 1.9872041e-3
    T_K = 298.0
    RT = R_KCAL * T_K

    def kd_uM(dg: float) -> float:
        return math.exp(dg / RT) * 1e6

    # Build target → (hfe_aff, halothane_aff) map. halothane is in mammalian set
    # (calibration_comparison_raw.csv via Vina), but we want C. elegans targets;
    # use negative_vina_results.csv ligand=halothane if present; otherwise compute
    # from existing infrastructure.
    hfe_by_gene = {}
    for r in neg_rows:
        if r["ligand"] == "hexafluoroethane":
            try:
                hfe_by_gene[r["gene"]] = float(r["affinity_kcal_per_mol"])
            except ValueError:
                continue
    halothane_by_gene = {}
    for r in neg_rows:
        if r["ligand"] == "halothane":
            try:
                halothane_by_gene[r["gene"]] = float(r["affinity_kcal_per_mol"])
            except ValueError:
                continue

    print(f"  hexafluoroethane targets: {len(hfe_by_gene)}, halothane in negative set: {len(halothane_by_gene)}")

    # If halothane is not in negative set, compare hfe distribution alone vs cis-DCE
    cis_dce_by_gene = {}
    for r in neg_rows:
        if r["ligand"] == "cis_12_dichloroethylene":
            try:
                cis_dce_by_gene[r["gene"]] = float(r["affinity_kcal_per_mol"])
            except ValueError:
                continue

    common = set(hfe_by_gene) & set(cis_dce_by_gene)
    print(f"  common targets (hfe ∩ cis-DCE): {len(common)}")
    print(f"  median Kd hfe: {sorted([kd_uM(hfe_by_gene[g])/F_ALLO for g in common])[len(common)//2]:.0f} µM (post-correction)")
    print(f"  median Kd cis-DCE: {sorted([kd_uM(cis_dce_by_gene[g])/F_ALLO for g in common])[len(common)//2]:.0f} µM (post-correction)")

    # Engagement at 1 mM aqueous (clinical halogenated alkane scale)
    conc_uM = 1000
    hfe_eng = sum(1 for g in common if conc_uM / (conc_uM + kd_uM(hfe_by_gene[g]) / F_ALLO) > 0.10)
    cis_eng = sum(1 for g in common if conc_uM / (conc_uM + kd_uM(cis_dce_by_gene[g]) / F_ALLO) > 0.10)
    print(f"  At {conc_uM} µM aqueous, post-CP5-correction:")
    print(f"    hexafluoroethane engaged ≥10%: {hfe_eng}/{len(common)}")
    print(f"    cis-DCE engaged ≥10%: {cis_eng}/{len(common)}")

    # Write output CSVs
    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CORR, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corrected[0].keys()))
        w.writeheader()
        w.writerows(corrected)

    with open(OUT_STRAT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strat_data[0].keys()))
        w.writeheader()
        for row in strat_data:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"\nCSVs: {OUT_CORR}, {OUT_STRAT}")

    # Build wave2_overlay_v2.json — re-scale occupancies using corrected Kd
    overlay_v1 = json.load(open(OVERLAY_V1))
    overlay_v2 = {"by_anesthetic": {}, "_meta": {
        "version": "v2",
        "correction": f"divided predicted Kd by f_allo = {F_ALLO:.2f}× (CP5 median allosteric correction)",
        "correction_log": F_ALLO_LOG,
        "source_doc": "artifacts/calibration/cp5_strict_recalibration.md",
    }}
    for anesth, target_dict in overlay_v1["by_anesthetic"].items():
        overlay_v2["by_anesthetic"][anesth] = {}
        for target, info in target_dict.items():
            new_info = json.loads(json.dumps(info))  # deep copy
            old_occ = info.get("occupancy_1xEC50")
            if old_occ is not None and old_occ > 0:
                # Hill: occ = c / (c + Kd). c = clinical EC50 ≈ Kd_v1 × old_occ / (1 - old_occ)
                # So Kd_v1 = c × (1 - old_occ) / old_occ; Kd_v2 = Kd_v1 / F_ALLO; new_occ = c / (c + Kd_v2)
                # Equivalently: new_occ / (1 - new_occ) = (old_occ / (1 - old_occ)) × F_ALLO
                if old_occ < 1.0:
                    ratio_v1 = old_occ / (1 - old_occ)
                    ratio_v2 = ratio_v1 * F_ALLO
                    new_occ = ratio_v2 / (1 + ratio_v2)
                else:
                    new_occ = old_occ
                new_info["occupancy_1xEC50_v1"] = old_occ
                new_info["occupancy_1xEC50"] = new_occ
                new_info["correction_applied"] = "f_allo_2.50x_CP5"
            overlay_v2["by_anesthetic"][anesth][target] = new_info

    with open(OUT_OVERLAY, "w") as f:
        json.dump(overlay_v2, f, indent=2)
    print(f"Overlay v2: {OUT_OVERLAY}")

    # Markdown summary
    with open(OUT_MD, "w") as f:
        f.write(f"# CP7 — Allosteric correction + chemical class stratification\n\n")
        f.write(f"## Correction applied\n\n")
        f.write(f"- f_allo = **{F_ALLO:.2f}×** (CP5 median-based)\n")
        f.write(f"- Direction: divide pipeline-predicted Kd by f_allo\n")
        f.write(f"- Rationale: T1 strict subset signed median log_err = +{F_ALLO_LOG:.3f} "
                f"(positive bias = pipeline overestimates Kd; consistent with PAM allosteric "
                f"coupling theory η ~ 0.4)\n\n")

        f.write("## Per-chemical-class metrics (T1 strict subset only)\n\n")
        f.write("| chem_class | n | pre signed_mean | post signed_mean | pre mean |log_err| | post mean |log_err| | post % within 10× |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in strat_data:
            f.write(f"| {s['chem_class']} | {s['n']} | {s['pre_signed_mean']:+.3f} | "
                    f"{s['post_signed_mean']:+.3f} | {s['pre_mean_abs']:.3f} | "
                    f"{s['post_mean_abs']:.3f} | {s['post_pct_10x']:.0f}% |\n")

        f.write("\n## Class-specific bias interpretation\n\n")
        f.write("After universal f_allo correction, residual signed_mean per chemical class:\n\n")
        for s in strat_data:
            interp = (
                "near-zero — universal correction sufficient" if abs(s["post_signed_mean"]) <= 0.2
                else f"positive residual ({s['post_signed_mean']:+.2f}) — needs class-specific tightening" if s["post_signed_mean"] > 0.2
                else f"negative residual ({s['post_signed_mean']:+.2f}) — pipeline now underestimates Kd; may reflect ion-channel-block kinetics (η > 1 effective)"
            )
            f.write(f"- **{s['chem_class']}**: signed_mean = {s['post_signed_mean']:+.3f} → {interp}\n")

        f.write(f"\n## Halogenated non-immobilizer baseline\n\n"
                f"Hexafluoroethane is a halogenated alkane non-immobilizer per Eger 2001 — "
                f"used as a negative-control test of whether Wave P discriminates the "
                f"non-immobilizer class from clinical alkanes by binding profile.\n\n"
                f"At {conc_uM} µM aqueous (clinical-range halogenated alkane concentration), "
                f"post-CP5-correction:\n\n"
                f"- hexafluoroethane engages ≥10%: **{hfe_eng}/{len(common)}** common targets\n"
                f"- cis-DCE engages ≥10% (anesthetic positive control): **{cis_eng}/{len(common)}** common targets\n\n")

        if hfe_eng < cis_eng - 5:
            f.write("**Discriminative finding:** hexafluoroethane engages substantially fewer targets "
                    f"than cis-DCE at clinical concentration; pipeline distinguishes Eger non-immobilizer "
                    f"from anesthetic positive control.\n\n")
        elif abs(hfe_eng - cis_eng) <= 3:
            f.write("**Non-discriminative:** hexafluoroethane and cis-DCE engage similar numbers of "
                    f"targets; Wave P does not robustly distinguish Eger non-immobilizer alkane from "
                    f"anesthetic positive control by binding profile alone. CP3's FAIL on cis/trans-DCE "
                    f"is reinforced — pipeline lacks the chemical specificity to act as Eger's anesthetic "
                    f"vs non-immobilizer classifier.\n\n")
        else:
            f.write(f"**Inverted discrimination:** hexafluoroethane engages {hfe_eng - cis_eng} MORE "
                    f"targets than cis-DCE — pipeline biased toward bulk lipophilicity over "
                    f"shape/conformational specificity.\n\n")

        f.write(f"## Outputs\n\n"
                f"- `cp7_corrected.csv` — per-row pre/post-correction comparison (n={len(corrected)})\n"
                f"- `cp7_class_stratified.csv` — per-chemical-class metrics\n"
                f"- `wave2_overlay_v2.json` — corrected occupancies for downstream Phase E/F/G use\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
