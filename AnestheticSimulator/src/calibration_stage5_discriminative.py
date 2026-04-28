"""Stage 5 — discriminative power test: anesthetics vs negative controls.

Compares Tier-1 target engagement (>10% occupancy at clinical/comparable
concentrations) for anesthetics vs negative controls. Includes the Eger-2001
cis/trans-1,2-DCE diagnostic pair: nearly-identical lipid solubility but only
cis is anesthetic. If pipeline ranks them the same, it's responding to bulk
lipophilicity rather than target-specific binding.

For negative controls, "comparable concentration" ≈ 100 µM (sub-narcotic
range used in the literature for these molecules). Sweeping concentrations
0.1× / 1× / 10× / 100× of this reference lets us see dose-response of
engagement.

Output: artifacts/calibration/stage5_discriminative.{csv,md}

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/calibration_stage5_discriminative.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_VINA = ROOT / "artifacts" / "calibration" / "negative_vina_results.csv"
ANESTH_BEST = ROOT / "artifacts" / "occupancy" / "best_pocket_per_target.csv"
NEG_PANEL = ROOT / "anesthetics" / "negative_control_panel.csv"
TIER1 = ROOT / "targets" / "tier1_targets.csv"
OUT_CSV = ROOT / "artifacts" / "calibration" / "stage5_discriminative.csv"
OUT_MD = ROOT / "artifacts" / "calibration" / "stage5_discriminative.md"

# Reference concentration for "is this binding occupied?" test
# Anesthetics use clinical EC50 (varied 0.3-5000 µM). For negative controls,
# we use 100 µM as the reference: roughly the geometric mean of anesthetic
# clinical aqueous EC50, and well within the range where any genuine
# discriminative pocket-fit signal should manifest.
NEG_REF_CONC_uM = 100.0

R_KCAL = 1.9872041e-3
T_K = 298.0
RT = R_KCAL * T_K


def affinity_to_kd_uM(dg: float) -> float:
    return math.exp(dg / RT) * 1e6


def occupancy(conc_uM: float, kd_uM: float) -> float:
    if kd_uM <= 0: return 1.0
    if conc_uM <= 0: return 0.0
    return conc_uM / (conc_uM + kd_uM)


def load_compartments() -> dict[str, str]:
    out = {}
    if not TIER1.exists():
        return out
    with open(TIER1) as f:
        for r in csv.DictReader(f):
            out[r["gene_name"].strip()] = r.get("pocket_compartment", "membrane_embedded").strip()
    return out


def best_per_pair(csv_path: Path, ligand_col: str) -> dict[tuple[str, str], dict]:
    best: dict[tuple[str, str], dict] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                aff = float(r["affinity_kcal_per_mol"])
            except (ValueError, KeyError):
                continue
            ligand = r[ligand_col].strip().lower()
            gene = r["gene"].strip()
            key = (ligand, gene)
            if key not in best or aff < best[key]["affinity"]:
                best[key] = {"affinity": aff, "pocket_id": r.get("pocket_id", ""),
                             "druggability": float(r.get("druggability_score") or 0)}
    return best


def main() -> int:
    if not NEGATIVE_VINA.exists():
        print(f"Negative control results not yet at {NEGATIVE_VINA} — sweep still running?")
        return 1

    compartments = load_compartments()
    neg_best = best_per_pair(NEGATIVE_VINA, "ligand")
    print(f"Negative-control unique (ligand, gene) pairs: {len(neg_best)}")

    # For each negative control, occupancy at NEG_REF_CONC_uM (no K_p, since we
    # want raw aqueous comparison; K_p effect explored in Stage 4)
    neg_engagement: dict[str, set[str]] = {}
    for (lig, gene), e in neg_best.items():
        kd = affinity_to_kd_uM(e["affinity"])
        # No K_p amplification for negative controls (most have unknown K_p; raw aq comparison)
        occ = occupancy(NEG_REF_CONC_uM, kd)
        if occ > 0.10:
            neg_engagement.setdefault(lig, set()).add(gene)

    # Anesthetic engagement at 1× clinical EC50 — already computed in Phase C
    # Reload from best_pocket_per_target.csv
    anes_engagement: dict[str, set[str]] = {}
    if ANESTH_BEST.exists():
        with open(ANESTH_BEST) as f:
            for r in csv.DictReader(f):
                ane = r["anesthetic"].strip().lower()
                gene = r["gene"].strip()
                try:
                    occ = float(r.get("occupancy_1.0xEC50") or 0)
                except (ValueError, TypeError):
                    occ = 0.0
                if occ > 0.10:
                    anes_engagement.setdefault(ane, set()).add(gene)

    # Also compute anesthetic engagement at NEG_REF_CONC_uM (for fair comparison
    # at the same concentration as negative controls)
    anes_at_ref: dict[str, set[str]] = {}
    if ANESTH_BEST.exists():
        with open(ANESTH_BEST) as f:
            for r in csv.DictReader(f):
                ane = r["anesthetic"].strip().lower()
                gene = r["gene"].strip()
                try:
                    aff = float(r["best_affinity_kcal_per_mol"])
                except (ValueError, KeyError):
                    continue
                kd = affinity_to_kd_uM(aff)
                occ_at_ref = occupancy(NEG_REF_CONC_uM, kd)
                if occ_at_ref > 0.10:
                    anes_at_ref.setdefault(ane, set()).add(gene)

    print()
    print("=" * 70)
    print(f"STAGE 5 — discriminative power: engagement at {NEG_REF_CONC_uM} µM aqueous")
    print("(No K_p amplification — fair raw-aqueous comparison)")
    print("=" * 70)
    print()
    print(f"Anesthetics — targets with >10% occupancy at {NEG_REF_CONC_uM} µM:")
    for ane in sorted(anes_at_ref):
        n = len(anes_at_ref[ane])
        print(f"  {ane:15s}  {n:>3d} / 30 targets")
    print()
    print(f"Negative controls — targets with >10% occupancy at {NEG_REF_CONC_uM} µM:")
    for lig in sorted(neg_engagement.keys() | {l for l, _ in neg_best.keys()}):
        n = len(neg_engagement.get(lig, set()))
        print(f"  {lig:30s}  {n:>3d} / 30 targets")

    print()
    print("=" * 70)
    print("Eger 2001 diagnostic — cis/trans-1,2-dichloroethylene")
    print("=" * 70)
    cis = neg_engagement.get("cis_12_dichloroethylene", set())
    trans = neg_engagement.get("trans_12_dichloroethylene", set())
    print(f"  cis  (anesthetic):       {len(cis):>3d} / 30 engaged")
    print(f"  trans (NON-anesthetic):  {len(trans):>3d} / 30 engaged")
    print(f"  cis − trans (specific):  {len(cis - trans):>3d}")
    print(f"  trans − cis (artifact):  {len(trans - cis):>3d}")
    print(f"  shared:                  {len(cis & trans):>3d}")
    diff = len(cis) - len(trans)
    if abs(diff) >= 5:
        print(f"  → discriminates (Δ={diff:+d}); shape sensitivity present")
    else:
        print(f"  → does NOT discriminate (Δ={diff:+d}); pipeline responds to bulk lipophilicity, not shape")

    # Write CSV — engagement table
    rows = []
    all_compounds = sorted(set(anes_at_ref.keys()) | set(neg_engagement.keys()) | {l for l, _ in neg_best.keys()})
    for c in all_compounds:
        ane_n = len(anes_at_ref.get(c, set())) if c in anes_at_ref else None
        neg_n = len(neg_engagement.get(c, set())) if c in neg_engagement or c in {l for l, _ in neg_best.keys()} else None
        category = "anesthetic" if c in anes_at_ref else "negative_control"
        rows.append({
            "compound": c,
            "category": category,
            "n_engaged_at_ref_conc": ane_n if category == "anesthetic" else neg_n,
            "ref_concentration_uM": NEG_REF_CONC_uM,
            "Kp_amplification": "no (raw aqueous)",
        })
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nEngagement table: {OUT_CSV}")

    # Verdict logic
    median_anes_engaged = sorted(len(s) for s in anes_at_ref.values())[len(anes_at_ref)//2] if anes_at_ref else 0
    median_neg_engaged = sorted(len(neg_engagement.get(c, set())) for c in {l for l, _ in neg_best.keys()})[len(neg_best)//(2*30) or 0] if neg_best else 0
    discriminative = median_anes_engaged - median_neg_engaged

    print()
    print(f"Median anesthetic engagement: {median_anes_engaged}/30 targets")
    print(f"Median negative-control engagement: {median_neg_engaged}/30 targets")
    print(f"Discriminative gap: {discriminative}")
    if discriminative >= 10:
        verdict = "DISCRIMINATIVE — pipeline distinguishes anesthetics from inert lipophilic compounds"
    elif discriminative >= 5:
        verdict = "WEAKLY DISCRIMINATIVE"
    else:
        verdict = "NON-DISCRIMINATIVE — pipeline does not distinguish anesthetics from controls"
    print(f"VERDICT: {verdict}")

    # Markdown
    with open(OUT_MD, "w") as f:
        f.write("# Stage 5 — discriminative power test\n\n")
        f.write(f"## Method\n\n"
                f"For each compound (6 anesthetics + 8 negative controls), count "
                f"how many of 30 Tier-1 targets show >10% occupancy at "
                f"{NEG_REF_CONC_uM} µM aqueous (no K_p amplification — fair raw-aqueous "
                f"comparison since negative controls don't have lipid:water partition data).\n\n"
                f"Discriminative pipeline: anesthetics show substantially higher engagement "
                f"than negative controls. Non-discriminative: similar engagement counts.\n\n"
                f"Eger 2001 diagnostic: cis-1,2-dichloroethylene (anesthetic) vs trans "
                f"(NOT anesthetic). Same lipid solubility, different shape. If pipeline "
                f"distinguishes them, it's measuring target-specific fit, not bulk lipophilicity.\n\n")
        f.write(f"## Engagement counts at {NEG_REF_CONC_uM} µM aqueous\n\n")
        f.write("| compound | category | targets engaged (>10% occ) / 30 |\n")
        f.write("|---|---|---|\n")
        for r in sorted(rows, key=lambda r: (r["category"], -(r["n_engaged_at_ref_conc"] or 0))):
            f.write(f"| {r['compound']} | {r['category']} | {r['n_engaged_at_ref_conc']} |\n")
        f.write(f"\n## cis/trans-1,2-DCE diagnostic\n\n")
        f.write(f"- cis (anesthetic): {len(cis)}/30 engaged\n")
        f.write(f"- trans (non-anesthetic): {len(trans)}/30 engaged\n")
        f.write(f"- difference: {diff}\n")
        f.write(f"- {'pipeline distinguishes shape' if abs(diff) >= 5 else 'pipeline does NOT distinguish shape — likely responding to bulk lipophilicity'}\n\n")
        f.write(f"## Verdict\n\n**{verdict}**\n\n"
                f"- Median anesthetic engagement: {median_anes_engaged}/30\n"
                f"- Median negative-control engagement: {median_neg_engaged}/30\n"
                f"- Gap: {discriminative}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
