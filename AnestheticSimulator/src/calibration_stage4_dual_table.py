"""Stage 4 — dual K_p calibration tables (with and without K_p amplification).

Compares Vina-predicted affinity for mammalian-homolog targets against
literature EC50/IC50 values. Produces TWO calibration tables to diagnose
whether systematic bias lives in K_p over-application:

(A) "raw" — predicted Kd from Vina ΔG, compared directly to experimental EC50/IC50
(B) "with K_p" — effective concentration = K_p × aqueous EC50, compared at
                 occupancy_at_experimental_concentration scale

Per-pair fold-error and Spearman rank correlation computed for both frames.
The diagnostic question: does removing K_p improve agreement with experiment?

Output:
- artifacts/calibration/calibration_comparison_raw.csv     (no K_p amplification)
- artifacts/calibration/calibration_comparison_withKp.csv  (current pipeline, with K_p)
- artifacts/calibration/stage4_summary.md

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/calibration_stage4_dual_table.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND = ROOT / "artifacts" / "calibration" / "ground_truth_Kd_table.csv"
MAMMALIAN_VINA = ROOT / "artifacts" / "calibration" / "mammalian_vina_results.csv"
PANEL = ROOT / "anesthetics" / "anesthetic_panel.csv"
OUT_RAW = ROOT / "artifacts" / "calibration" / "calibration_comparison_raw.csv"
OUT_KP = ROOT / "artifacts" / "calibration" / "calibration_comparison_withKp.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "stage4_summary.md"

R_KCAL = 1.9872041e-3
T_K = 298.0
RT = R_KCAL * T_K


def affinity_to_kd_uM(dg: float) -> float:
    return math.exp(dg / RT) * 1e6


def occupancy(conc_uM: float, kd_uM: float) -> float:
    if kd_uM <= 0: return 1.0
    if conc_uM <= 0: return 0.0
    return conc_uM / (conc_uM + kd_uM)


def spearman(x: list[float], y: list[float]) -> tuple[float, int]:
    if len(x) != len(y) or len(x) < 2:
        return float("nan"), len(x)
    def ranks(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j+1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    cov = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    sx = math.sqrt(sum((a-mx)**2 for a in rx))
    sy = math.sqrt(sum((b-my)**2 for b in ry))
    if sx == 0 or sy == 0: return float("nan"), len(x)
    return cov / (sx*sy), len(x)


def load_panel() -> dict[str, dict]:
    out = {}
    with open(PANEL) as f:
        for r in csv.DictReader(f):
            name = r["name"].strip().lower()
            try:
                ec50 = float(r["clinical_aqueous_EC50_uM"])
            except (ValueError, KeyError):
                ec50 = float("nan")
            try:
                kp = float(r["oil_water_partition_coefficient"])
            except (ValueError, KeyError):
                kp = 1.0
            out[name] = {"ec50_uM": ec50, "Kp": kp}
    return out


def load_ground_truth() -> list[dict]:
    out = []
    with open(GROUND) as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def load_vina_best() -> dict[tuple[str, str], dict]:
    """Best (most negative) affinity per (anesthetic, target) pair."""
    best: dict[tuple[str, str], dict] = {}
    with open(MAMMALIAN_VINA) as f:
        for r in csv.DictReader(f):
            try:
                aff = float(r["affinity_kcal_per_mol"])
            except ValueError:
                continue
            ane = r["ligand"].strip().lower()
            gene = r["gene"].strip()  # human gene name e.g. "GABRA1"
            key = (ane, gene)
            if key not in best or aff < best[key]["affinity"]:
                best[key] = {
                    "affinity": aff,
                    "pocket_id": r.get("pocket_id", ""),
                    "druggability": float(r.get("druggability_score") or 0),
                }
    return best


def main() -> int:
    panel = load_panel()
    truth = load_ground_truth()
    best = load_vina_best()

    print(f"Ground-truth rows: {len(truth)}")
    print(f"Vina best per pair: {len(best)}")

    # Build comparison table
    raw_rows = []
    kp_rows = []
    for r in truth:
        gene = r["mammalian_homolog"].split("_")[0] if "_" in r["mammalian_homolog"] else r["mammalian_homolog"]
        # Map experimental homolog name to Vina gene
        # GABA-A_α1β2γ2 -> we have GABRA1 docking
        # GlyR_α1 -> GLRA1
        # nAChR_α4β2 -> CHRNA4
        # TREK-1_KCNK2 -> KCNK2
        # NDUFS2_ComplexI -> NDUFS2
        homolog_map = {
            "GABA-A": "GABRA1", "GlyR": "GLRA1", "nAChR": "CHRNA4",
            "TREK-1": "KCNK2", "NDUFS2": "NDUFS2",
        }
        prefix = r["mammalian_homolog"].split("_")[0]
        vina_gene = homolog_map.get(prefix, prefix)

        ane = r["anesthetic"].strip().lower()
        if not r["value_uM"]:
            continue  # "no_significant_effect" entries skip numeric comparison
        try:
            exp_value = float(r["value_uM"])
        except ValueError:
            continue

        vina = best.get((ane, vina_gene))
        if not vina:
            continue

        ane_meta = panel.get(ane, {})
        clinical_ec50 = ane_meta.get("ec50_uM")
        kp = ane_meta.get("Kp", 1.0)

        pred_kd = affinity_to_kd_uM(vina["affinity"])

        # Frame A — raw: compare predicted Kd directly to experimental EC50/IC50
        # Fold error: predicted_Kd / experimental_value (or its reciprocal, larger of the two)
        if exp_value > 0 and pred_kd > 0:
            ratio = pred_kd / exp_value
            fold = ratio if ratio >= 1 else 1 / ratio
            log_err = math.log10(pred_kd) - math.log10(exp_value)
        else:
            fold = float("nan")
            log_err = float("nan")

        raw_rows.append({
            "target_class": r["target_class"],
            "mammalian_homolog": r["mammalian_homolog"],
            "vina_gene": vina_gene,
            "anesthetic": ane,
            "vina_affinity_kcal_per_mol": vina["affinity"],
            "predicted_Kd_uM": pred_kd,
            "experimental_value_uM": exp_value,
            "experimental_value_type": r["value_type"],
            "fold_error": fold,
            "log_error": log_err,
            "anchor_PMID": r.get("anchor_PMID", ""),
        })

        # Frame B — with K_p: compute predicted occupancy at experimental concentration
        # using K_p × experimental_concentration as effective.
        if exp_value > 0 and pred_kd > 0:
            # Effective concentration assuming pocket sees K_p * bath
            eff_conc = kp * exp_value
            pred_occ_at_exp_conc = occupancy(eff_conc, pred_kd)
            # Experimentally, at exp_value (which is the EC50/IC50), expected occupancy is 0.5
            # by definition (assuming the EC50 reflects 50% effect).
            kp_log_err = math.log10(eff_conc) - math.log10(pred_kd)
        else:
            pred_occ_at_exp_conc = float("nan")
            kp_log_err = float("nan")

        kp_rows.append({
            **raw_rows[-1],
            "Kp_oil_water": kp,
            "effective_concentration_uM": kp * exp_value if exp_value > 0 else "",
            "predicted_occupancy_at_eff_conc": pred_occ_at_exp_conc,
            "expected_occupancy_at_EC50": 0.5,
            "log_err_eff_vs_Kd": kp_log_err,
        })

    # Write
    with open(OUT_RAW, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    print(f"Raw (no K_p): {OUT_RAW}  ({len(raw_rows)} rows)")
    with open(OUT_KP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kp_rows[0].keys()))
        w.writeheader()
        w.writerows(kp_rows)
    print(f"With K_p:     {OUT_KP}  ({len(kp_rows)} rows)")

    # Summary statistics
    print()
    print("=" * 70)
    print("STAGE 4 — calibration: predicted Kd vs experimental EC50/IC50")
    print("=" * 70)
    print()
    print("Frame A — raw (predicted Kd vs experimental EC50/IC50):")
    raw_log_errs = [r["log_error"] for r in raw_rows if not math.isnan(r["log_error"])]
    if raw_log_errs:
        print(f"  N pairs: {len(raw_log_errs)}")
        print(f"  log_err mean: {sum(raw_log_errs)/len(raw_log_errs):+.2f}")
        print(f"  log_err median: {sorted(raw_log_errs)[len(raw_log_errs)//2]:+.2f}")
        n_within_1 = sum(1 for x in raw_log_errs if abs(x) <= 1)
        print(f"  |log_err| ≤ 1 (within 10×): {n_within_1}/{len(raw_log_errs)}")
        n_within_05 = sum(1 for x in raw_log_errs if abs(x) <= 0.5)
        print(f"  |log_err| ≤ 0.5 (within ~3×): {n_within_05}/{len(raw_log_errs)}")
        n_within_03 = sum(1 for x in raw_log_errs if abs(x) <= 0.3)
        print(f"  |log_err| ≤ 0.3 (within 2×): {n_within_03}/{len(raw_log_errs)}")
    print()
    # Spearman
    pred = [math.log10(r["predicted_Kd_uM"]) for r in raw_rows
            if not math.isnan(r["log_error"])]
    exp_v = [math.log10(r["experimental_value_uM"]) for r in raw_rows
             if not math.isnan(r["log_error"])]
    rho, n = spearman(pred, exp_v)
    print(f"  Spearman ρ (log_pred_Kd vs log_exp_value): {rho:+.3f} (n={n})")
    print(f"    (Strong positive = pipeline ranks targets/anesthetics in same order as experiment.)")
    print()

    # Per-class breakdown
    print("Per mechanism class:")
    by_class: dict[str, list[dict]] = {}
    for r in raw_rows:
        by_class.setdefault(r["target_class"], []).append(r)
    for cls, rs in sorted(by_class.items()):
        log_errs = [r["log_error"] for r in rs if not math.isnan(r["log_error"])]
        if log_errs:
            print(f"  {cls:25s}  n={len(log_errs)}  median_log_err={sorted(log_errs)[len(log_errs)//2]:+.2f}  "
                  f"mean={sum(log_errs)/len(log_errs):+.2f}")

    # Per-anesthetic breakdown
    print()
    print("Per anesthetic:")
    by_ane: dict[str, list[dict]] = {}
    for r in raw_rows:
        by_ane.setdefault(r["anesthetic"], []).append(r)
    for ane, rs in sorted(by_ane.items()):
        log_errs = [r["log_error"] for r in rs if not math.isnan(r["log_error"])]
        if log_errs:
            print(f"  {ane:12s}  n={len(log_errs)}  median_log_err={sorted(log_errs)[len(log_errs)//2]:+.2f}")

    # Per-pair table
    print()
    print("All pairs sorted by absolute log_err:")
    print(f"  {'gene':10s} {'anesthetic':12s} {'pred_Kd':>9s} {'exp':>9s} {'type':22s} {'log_err':>8s}")
    for r in sorted(raw_rows, key=lambda r: abs(r['log_error']) if not math.isnan(r['log_error']) else 999):
        print(f"  {r['vina_gene']:10s} {r['anesthetic']:12s} {r['predicted_Kd_uM']:>9.2f} "
              f"{r['experimental_value_uM']:>9.1f} {r['experimental_value_type']:22s} "
              f"{r['log_error']:>+8.2f}")

    # Markdown summary
    with open(OUT_MD, "w") as f:
        f.write("# Stage 4 — dual K_p calibration tables\n\n")
        f.write("## Method\n\n")
        f.write("Compare Vina-predicted Kd (from `Kd = exp(ΔG/RT)` at 298 K) for "
                "mammalian-homolog targets against published experimental EC50/IC50 values.\n\n"
                "Frame A — raw: predicted_Kd vs experimental_value, no K_p amplification.\n"
                "Frame B — with K_p: predicted occupancy at K_p × experimental_concentration.\n\n"
                "**Caveat (load-bearing):** All ground-truth values are EC50/IC50 from "
                "patch-clamp dose-response, NOT classical equilibrium Kd from radioligand "
                "binding. Direct fold-error against EC50/IC50 conflates Vina ΔG bias with "
                "Kd-vs-EC50 quantity mismatch. Spearman rank correlation is the more "
                "interpretable metric.\n\n")
        f.write(f"## Frame A — raw (predicted Kd vs experimental value)\n\n")
        f.write(f"- N pairs: {len(raw_log_errs)}\n")
        if raw_log_errs:
            f.write(f"- log_err mean: {sum(raw_log_errs)/len(raw_log_errs):+.2f}\n")
            f.write(f"- log_err median: {sorted(raw_log_errs)[len(raw_log_errs)//2]:+.2f}\n")
            f.write(f"- |log_err| ≤ 0.3 (within 2×): {sum(1 for x in raw_log_errs if abs(x)<=0.3)}/{len(raw_log_errs)}\n")
            f.write(f"- |log_err| ≤ 0.5 (within ~3×): {sum(1 for x in raw_log_errs if abs(x)<=0.5)}/{len(raw_log_errs)}\n")
            f.write(f"- |log_err| ≤ 1.0 (within 10×): {sum(1 for x in raw_log_errs if abs(x)<=1)}/{len(raw_log_errs)}\n")
        f.write(f"- Spearman ρ (log_pred_Kd vs log_exp_value): {rho:+.3f}\n\n")
        f.write("## Per-pair table\n\n")
        f.write("| target | anesthetic | pred Kd µM | exp µM | type | log err |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in sorted(raw_rows, key=lambda r: abs(r['log_error']) if not math.isnan(r['log_error']) else 999):
            f.write(f"| {r['vina_gene']} | {r['anesthetic']} | {r['predicted_Kd_uM']:.2f} | "
                    f"{r['experimental_value_uM']:.1f} | {r['experimental_value_type']} | "
                    f"{r['log_error']:+.2f} |\n")
    print(f"\nMarkdown: {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
