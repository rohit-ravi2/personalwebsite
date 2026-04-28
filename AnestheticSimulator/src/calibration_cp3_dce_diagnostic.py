"""CP3 — DCE concentration sweep diagnostic.

Tests pipeline conformational specificity via Eger 2001 cis/trans-1,2-DCE pair.
cis-DCE is anesthetic; trans-DCE is non-anesthetic, near-identical lipid solubility.

Reuses already-docked DCE poses from poses_negative/. Computes engagement at
expanded concentration grid: 0.1, 0.3, 1.0, 3.0, 10.0, 30.0 mM aqueous.

Output: artifacts/calibration/dce_diagnostic_summary.{csv,md}
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEG_VINA = ROOT / "artifacts" / "calibration" / "negative_vina_results.csv"
OUT_CSV = ROOT / "artifacts" / "calibration" / "dce_concentration_sweep.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "dce_diagnostic_summary.md"

R_KCAL = 1.9872041e-3
T_K = 298.0
RT = R_KCAL * T_K

CONCENTRATIONS_uM = [100, 300, 1000, 3000, 10_000, 30_000]


def kd_uM(dg: float) -> float:
    return math.exp(dg / RT) * 1e6


def main() -> int:
    rows = list(csv.DictReader(open(NEG_VINA)))
    cis_rows = [r for r in rows if r["ligand"] == "cis_12_dichloroethylene"]
    trans_rows = [r for r in rows if r["ligand"] == "trans_12_dichloroethylene"]

    # Best per (gene) pair
    def best_per_gene(rows):
        best = {}
        for r in rows:
            try:
                aff = float(r["affinity_kcal_per_mol"])
            except ValueError:
                continue
            g = r["gene"]
            if g not in best or aff < best[g]["aff"]:
                best[g] = {"aff": aff, "kd_uM": kd_uM(aff)}
        return best

    cis_best = best_per_gene(cis_rows)
    trans_best = best_per_gene(trans_rows)
    print(f"cis-DCE targets: {len(cis_best)}, trans-DCE targets: {len(trans_best)}")

    # Engagement count per concentration
    print(f"\n{'conc_uM':>10s} {'cis engaged':>13s} {'trans engaged':>15s} {'gap (cis-trans)':>18s}")
    sweep_rows = []
    for c in CONCENTRATIONS_uM:
        cis_eng = sum(1 for d in cis_best.values() if c / (c + d["kd_uM"]) > 0.10)
        trans_eng = sum(1 for d in trans_best.values() if c / (c + d["kd_uM"]) > 0.10)
        gap = cis_eng - trans_eng
        sweep_rows.append({
            "conc_uM": c,
            "cis_engaged_count": cis_eng,
            "trans_engaged_count": trans_eng,
            "gap_cis_minus_trans": gap,
        })
        print(f"  {c:>10d}  {cis_eng:>13d}  {trans_eng:>15d}  {gap:>18d}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"\nCSV: {OUT_CSV}")

    # Diagnostic verdict
    max_gap = max(r["gap_cis_minus_trans"] for r in sweep_rows)
    min_gap = min(r["gap_cis_minus_trans"] for r in sweep_rows)
    print(f"\nMax cis-trans gap across concentrations: {max_gap}")
    print(f"Min cis-trans gap (most-negative; trans engages more): {min_gap}")

    if max_gap >= 5:
        verdict = "PASS — pipeline shows conformational specificity (cis engages more than trans at some concentration)"
    elif max_gap >= 2:
        verdict = "WEAK PASS — small differential, may be noise"
    elif min_gap <= -5:
        verdict = "INVERTED — trans engages more than cis (pipeline scrambles isomers; biologically wrong)"
    else:
        verdict = "FAIL — no conformational specificity (pipeline responds to bulk lipophilicity, not shape)"

    print(f"\nVerdict: {verdict}")

    # Eger 2001 anchor: cis-DCE clinical EC50 for nematode immobilization is in the
    # 1-10 mM aqueous range based on extrapolation from mammalian MAC values.
    # cis-DCE is reported anesthetic per Eger; trans-DCE is non-anesthetic.
    # Source: Eger EI, et al. Anesth Analg 2001;92:1395 (need PMID verification).
    eger_band = (1000, 10_000)
    in_band = [r for r in sweep_rows if eger_band[0] <= r["conc_uM"] <= eger_band[1]]
    print(f"\nIn Eger anesthetic-range (1-10 mM):")
    for r in in_band:
        print(f"  {r['conc_uM']} µM: cis {r['cis_engaged_count']}/30, trans {r['trans_engaged_count']}/30, gap {r['gap_cis_minus_trans']}")

    with open(OUT_MD, "w") as f:
        f.write("# CP3 — DCE concentration sweep diagnostic\n\n")
        f.write("## Method\n\n"
                "Reuse existing Vina poses for cis-1,2-dichloroethylene (anesthetic) and "
                "trans-1,2-dichloroethylene (non-anesthetic per Eger 2001) against the 30 "
                "Tier-1 C. elegans targets. Compute engagement count at varying aqueous "
                "concentrations using Hill occupancy (no K_p amplification — DCE K_p data unverified).\n\n"
                "**Diagnostic claim:** if pipeline distinguishes cis (anesthetic) from "
                "trans (non-anesthetic), it's measuring target-specific shape fitting, not "
                "bulk lipophilicity. If they engage similarly across concentrations, "
                "pipeline lacks conformational specificity.\n\n")
        f.write("## Sweep results\n\n")
        f.write("| conc (µM) | cis engaged / 30 | trans engaged / 30 | gap (cis - trans) |\n")
        f.write("|---|---|---|---|\n")
        for r in sweep_rows:
            f.write(f"| {r['conc_uM']} | {r['cis_engaged_count']} | {r['trans_engaged_count']} | "
                    f"{r['gap_cis_minus_trans']} |\n")
        f.write(f"\n## Verdict: **{verdict}**\n\n")
        f.write(f"- Max gap (cis − trans): {max_gap}\n")
        f.write(f"- Min gap: {min_gap}\n")
        f.write(f"- Eger 2001 anesthetic concentration range (1-10 mM aqueous):\n")
        for r in in_band:
            f.write(f"  - {r['conc_uM']} µM: cis {r['cis_engaged_count']}, trans {r['trans_engaged_count']}, gap {r['gap_cis_minus_trans']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
