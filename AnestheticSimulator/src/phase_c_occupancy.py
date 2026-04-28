"""Phase C — convert Vina affinities to fractional occupancy at clinical EC50.

Implementation status: SHIPPED.

For every (anesthetic, target, pocket) docked in Phase B, compute:
- Predicted Kd from Vina ΔG via Kd = exp(ΔG/RT) at 298 K
- Fractional occupancy F = [drug_eff] / ([drug_eff] + Kd) at clinical aqueous EC50
- For membrane-embedded targets, [drug_eff] = K_p × [drug]_aqueous (membrane partition)

Outputs:
- artifacts/occupancy/best_pocket_per_target.csv  - long form, one row per (anesthetic, gene)
- artifacts/occupancy/occupancy_matrix.csv        - wide form (gene × anesthetic) at 1×EC50
- artifacts/occupancy/gate_c1_summary.md          - prereg gate evaluation

Gate C.1 prereg: end-of-run PASS/FAIL on multi-target check —
"≥ 5 targets show > 10% occupancy at 1× clinical EC50 for at least one anesthetic"
which is the load-bearing falsifiability test for the multi-target framing.

Usage:
    conda activate wave-p-docking
    python src/phase_c_occupancy.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VINA_RESULTS = ROOT / "artifacts" / "binding" / "vina_results.csv"
PANEL = ROOT / "anesthetics" / "anesthetic_panel.csv"
TIER1 = ROOT / "targets" / "tier1_targets.csv"
OUT_BEST = ROOT / "artifacts" / "occupancy" / "best_pocket_per_target.csv"
OUT_MATRIX = ROOT / "artifacts" / "occupancy" / "occupancy_matrix.csv"
OUT_SUMMARY = ROOT / "artifacts" / "occupancy" / "gate_c1_summary.md"

# Thermodynamic constants
R_KCAL = 1.9872041e-3   # kcal/(mol·K)
T_K = 298.0
RT = R_KCAL * T_K       # ≈ 0.5925 kcal/mol


def affinity_to_kd_uM(delta_g_kcal_per_mol: float) -> float:
    """Vina affinity (kcal/mol, negative = better) → Kd in micromolar.

    Kd = exp(ΔG/RT) with ΔG in kcal/mol gives Kd in molar (1 M std state).
    Convert to µM by multiplying by 1e6.
    """
    return math.exp(delta_g_kcal_per_mol / RT) * 1e6


def occupancy(conc_uM: float, kd_uM: float, n_hill: float = 1.0) -> float:
    if kd_uM <= 0:
        return 1.0
    if conc_uM <= 0:
        return 0.0
    c_n = conc_uM ** n_hill
    return c_n / (c_n + kd_uM ** n_hill)


def load_anesthetics() -> dict[str, dict]:
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
            out[name] = {"ec50_uM": ec50, "Kp_oil_water": kp}
    return out


def load_target_compartments() -> dict[str, str]:
    out = {}
    if not TIER1.exists():
        return out
    with open(TIER1) as f:
        for r in csv.DictReader(f):
            gene = r["gene_name"].strip()
            comp = r.get("pocket_compartment", "membrane_embedded").strip()
            out[gene] = comp
    return out


def main() -> int:
    if not VINA_RESULTS.exists():
        print(f"Vina results not found at {VINA_RESULTS}; run phase_b_dock_pipeline.py first.")
        return 1
    OUT_BEST.parent.mkdir(parents=True, exist_ok=True)

    anesthetics = load_anesthetics()
    compartments = load_target_compartments()

    # Group dockings by (anesthetic, gene); pick best (most negative) affinity
    best: dict[tuple[str, str], dict] = {}
    n_rows = 0
    with open(VINA_RESULTS) as f:
        for r in csv.DictReader(f):
            n_rows += 1
            try:
                aff = float(r["affinity_kcal_per_mol"])
            except ValueError:
                continue
            ane = r["anesthetic"].strip().lower()
            gene = r["gene"].strip()
            key = (ane, gene)
            if key not in best or aff < best[key]["affinity"]:
                best[key] = {
                    "affinity": aff,
                    "pocket_id": r.get("pocket_id", ""),
                    "druggability": float(r.get("druggability_score", 0) or 0),
                    "uniprot_acc": r.get("uniprot_acc", ""),
                }

    print(f"Vina rows read: {n_rows}")
    print(f"Unique (anesthetic, gene) pairs with valid affinity: {len(best)}")

    DOSES = [0.5, 1.0, 2.0, 5.0]
    matrix_rows = []

    for (ane, gene), entry in sorted(best.items()):
        ane_meta = anesthetics.get(ane, {})
        ec50 = ane_meta.get("ec50_uM", float("nan"))
        kp = ane_meta.get("Kp_oil_water", 1.0)
        compartment = compartments.get(gene, "membrane_embedded")

        kd_uM = affinity_to_kd_uM(entry["affinity"])
        row = {
            "anesthetic": ane,
            "gene": gene,
            "uniprot_acc": entry["uniprot_acc"],
            "best_pocket_id": entry["pocket_id"],
            "best_pocket_druggability": f"{entry['druggability']:.3f}",
            "best_affinity_kcal_per_mol": f"{entry['affinity']:.2f}",
            "predicted_Kd_uM": f"{kd_uM:.4g}",
            "ec50_clinical_aqueous_uM": f"{ec50:.1f}" if not math.isnan(ec50) else "",
            "Kp_oil_water": f"{kp:.0f}",
            "pocket_compartment": compartment,
        }
        for d in DOSES:
            if math.isnan(ec50):
                row[f"occupancy_{d}xEC50"] = ""
                continue
            conc_aq = d * ec50
            if "membrane_embedded" in compartment.lower():
                conc_eff = kp * conc_aq
            else:
                conc_eff = conc_aq
            occ = occupancy(conc_eff, kd_uM)
            row[f"occupancy_{d}xEC50"] = f"{occ:.3f}"
        matrix_rows.append(row)

    if not matrix_rows:
        print("No usable pairs.")
        return 1

    fieldnames = list(matrix_rows[0].keys())
    with open(OUT_BEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)
    print(f"Best-pocket-per-target table: {OUT_BEST}")

    all_genes = sorted({r["gene"] for r in matrix_rows})
    all_anes = sorted({r["anesthetic"] for r in matrix_rows})
    wide_fields = ["gene"] + all_anes
    with open(OUT_MATRIX, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wide_fields)
        writer.writeheader()
        for gene in all_genes:
            row = {"gene": gene}
            for ane in all_anes:
                m = next((r for r in matrix_rows
                          if r["anesthetic"] == ane and r["gene"] == gene), None)
                row[ane] = m["occupancy_1.0xEC50"] if m else ""
            writer.writerow(row)
    print(f"Occupancy matrix at 1x EC50: {OUT_MATRIX}")

    # Gate C.1 evaluation
    gene_high_occ_at_1xec50 = set()
    for r in matrix_rows:
        try:
            occ = float(r["occupancy_1.0xEC50"])
        except (ValueError, KeyError):
            continue
        if occ > 0.10:
            gene_high_occ_at_1xec50.add(r["gene"])

    n_targets = len(gene_high_occ_at_1xec50)
    multi_target_pass = n_targets >= 5

    print()
    print("=" * 60)
    print("GATE C.1 — multi-target framing falsifiability check")
    print("=" * 60)
    print(f"Targets with >10% occupancy at 1x EC50 (≥1 anesthetic): {n_targets}")
    if multi_target_pass:
        print("  PASS — multi-target framing supported (≥5 targets engaged)")
    else:
        print("  FAIL — single-target framing implied (premise FALSIFIED at this gate)")
    print()
    print("Engaged targets:", ", ".join(sorted(gene_high_occ_at_1xec50)))

    ranked = []
    for r in matrix_rows:
        try:
            occ = float(r["occupancy_1.0xEC50"])
        except (ValueError, KeyError):
            continue
        ranked.append((occ, r))
    ranked.sort(key=lambda t: -t[0])
    print("\nTop 15 (anesthetic, target) pairs by 1x-EC50 occupancy:")
    print(f"{'anesthetic':12s} {'gene':10s} {'ΔG':>7} {'Kd_uM':>10} {'occ@1x':>8}")
    for occ, r in ranked[:15]:
        print(f"{r['anesthetic']:12s} {r['gene']:10s} {r['best_affinity_kcal_per_mol']:>7} "
              f"{r['predicted_Kd_uM']:>10} {occ:>8.3f}")

    with open(OUT_SUMMARY, "w") as f:
        f.write("# Gate C.1 — Wave P multi-target framing falsifiability check\n\n")
        f.write(f"- Vina rows: {n_rows}\n")
        f.write(f"- Unique (anesthetic, gene) pairs with valid affinity: {len(best)}\n")
        f.write(f"- Targets with >10% occupancy at 1x EC50: {n_targets}\n")
        f.write(f"- Verdict: **{'PASS' if multi_target_pass else 'FAIL'}**\n\n")
        f.write("Engaged targets: " + ", ".join(sorted(gene_high_occ_at_1xec50)) + "\n\n")
        f.write("## Top 15 (anesthetic, target) pairs by 1x-EC50 occupancy\n\n")
        f.write("| anesthetic | gene | ΔG kcal/mol | Kd µM | occ@1xEC50 |\n")
        f.write("|---|---|---|---|---|\n")
        for occ, r in ranked[:15]:
            f.write(f"| {r['anesthetic']} | {r['gene']} | "
                    f"{r['best_affinity_kcal_per_mol']} | "
                    f"{r['predicted_Kd_uM']} | {occ:.3f} |\n")
    print(f"\nSummary md: {OUT_SUMMARY}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
