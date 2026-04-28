"""Phase D — translate per-(anesthetic, target) occupancy into channel kinetic shifts.

Implementation status: SHIPPED.

For each target, apply mechanism-class-specific shift rules (literature-grounded
where data exists; conservative defaults flagged "ANALOGY" otherwise) to produce
a per-(anesthetic, target) kinetic-delta matrix. This matrix is the direct input
to Wave 2 channel-parameter perturbation: each Brian2 channel parameter (g_max,
τ_decay, n_Ca_cooperativity, rate_complex_I) gets multiplied by the appropriate
factor derived from occupancy at the target dose.

Mechanism classes (from `targets/tier1_targets_corrected.csv`):

| class | Effect | Translation rule | Source |
|---|---|---|---|
| gaba_potentiation     | potentiate, slow τ_decay | τ_decay × (1 + 3 × occ)   | Hales & Lambert; Mihic |
| glucl_potentiation    | partial potentiation     | τ_decay × (1 + 1.5 × occ) | ANALOGY to GABA-A |
| nachr_antagonism      | open-channel block       | g_max × (1 - 0.7 × occ)   | Forman 1996 |
| k2p_potentiation      | activate K2P             | g_max × (1 + 2 × occ)     | Patel & Honoré 1999 |
| nca_block             | NALCN-complex block      | g_max × (1 - occ)         | Lu 2007; Sedensky 1987 |
| snare_cooperativity   | reduce Ca-cooperativity  | n_Ca → n_Ca - 1.5 × occ   | Stewart 2000; van Swinderen 1999 |
| complex_i_block       | inhibit Complex I rate   | rate × (1 - 0.3 × occ)    | Hanley 2002; Kayser 2001 |

Evidence-grade tags: LITERATURE / ANALOGY / CONSERVATIVE / DEFERRED.

Inputs:
- artifacts/occupancy/best_pocket_per_target.csv  (Phase C)
- targets/tier1_targets_corrected.csv             (Phase A audit output)

Outputs:
- artifacts/kinetics/kinetic_shifts_at_1xEC50.csv
- artifacts/kinetics/wave2_overlay.json    - drop-in for Wave 2 channel runs
- artifacts/kinetics/phase_d_summary.md

Usage:
    conda activate wave-p-docking
    python src/phase_d_kinetic_shifts.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEST_OCC = ROOT / "artifacts" / "occupancy" / "best_pocket_per_target.csv"
TIER1_CORRECTED = ROOT / "targets" / "tier1_targets_corrected.csv"
TIER1_ORIG = ROOT / "targets" / "tier1_targets.csv"
KINETICS_OUT = ROOT / "artifacts" / "kinetics" / "kinetic_shifts_at_1xEC50.csv"
WAVE2_OVERLAY = ROOT / "artifacts" / "kinetics" / "wave2_overlay.json"
PHASE_D_MD = ROOT / "artifacts" / "kinetics" / "phase_d_summary.md"


# Mechanism-class shift definitions
# Each entry: list of (parameter_name, formula_lambda(occupancy) -> factor, evidence_grade, source_note)
SHIFTS = {
    "gaba_potentiation": [
        ("tau_decay_factor", lambda occ: 1 + 3.0 * occ, "LITERATURE",
         "Hales & Lambert 1991 — halothane potentiates GABA-A τ_decay 2-4× at clinical doses"),
    ],
    "glucl_potentiation": [
        ("tau_decay_factor", lambda occ: 1 + 1.5 * occ, "ANALOGY",
         "Transfer from GABA-A; Hibbs 2011 PMID 21572436 confirms GluCl pocket conserved"),
    ],
    "nachr_antagonism": [
        ("g_max_factor", lambda occ: max(0.0, 1 - 0.7 * occ), "LITERATURE",
         "Forman 1996 / Wachtel 1995 — halothane open-channel block at high occupancy"),
    ],
    "k2p_potentiation": [
        ("g_max_factor", lambda occ: 1 + 2.0 * occ, "LITERATURE",
         "Patel & Honoré 1999 — TASK / TREK-1 activated by volatile anesthetics"),
    ],
    "nca_block": [
        ("g_max_factor", lambda occ: max(0.0, 1 - 1.0 * occ), "LITERATURE",
         "Lu 2007 / Sedensky & Meneely 1987 PMID 3576211 — unc-79/80 resistance, NCA block"),
    ],
    "snare_cooperativity": [
        ("n_Ca_delta", lambda occ: -1.5 * occ, "LITERATURE",
         "Stewart 2000 PMID 11095753 + van Swinderen 1999 PMID 10051668 — SNARE Ca-cooperativity reduction"),
    ],
    "complex_i_block": [
        ("rate_factor", lambda occ: max(0.0, 1 - 0.3 * occ), "LITERATURE",
         "Hanley 2002 / Kayser 2001 PMID 11278828 — Complex I sensitive to halogenated anesthetics"),
    ],
    "complex_ii_block": [
        # Complex II (MEV-1 / SDHC) is anesthetic-resistant in mev-1 mutants
        # → anesthetics affect Complex II only weakly. Conservative rate decrement.
        ("rate_factor", lambda occ: max(0.0, 1 - 0.10 * occ), "CONSERVATIVE",
         "Senoo-Matsuda 2001 / Kayser 2003 PMID 12878724 — mev-1 hypersensitivity is small "
         "→ Complex II is a weak target; small rate decrement"),
    ],
}


def load_target_classes() -> dict[str, str]:
    out = {}
    src = TIER1_CORRECTED if TIER1_CORRECTED.exists() else TIER1_ORIG
    with open(src) as f:
        for r in csv.DictReader(f):
            gene = r["gene_name"].strip()
            cls = r.get("mechanism_class", "").strip()
            if cls:
                out[gene] = cls
    return out


def load_occupancy() -> list[dict]:
    if not BEST_OCC.exists():
        return []
    out = []
    with open(BEST_OCC) as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def main() -> int:
    target_class = load_target_classes()
    occ_rows = load_occupancy()
    if not occ_rows:
        print("No occupancy data — run phase_c_occupancy.py first")
        return 1

    print(f"Loaded {len(occ_rows)} occupancy rows; {len(target_class)} target→class mappings")
    KINETICS_OUT.parent.mkdir(parents=True, exist_ok=True)

    shift_rows = []
    overlay: dict = {"by_anesthetic": {}}

    for r in occ_rows:
        gene = r["gene"].strip()
        ane = r["anesthetic"].strip()
        cls = target_class.get(gene, "unknown")
        try:
            occ_1x = float(r.get("occupancy_1.0xEC50", "") or 0.0)
        except ValueError:
            occ_1x = 0.0

        rules = SHIFTS.get(cls)
        if rules is None:
            shift_rows.append({
                "anesthetic": ane, "gene": gene,
                "mechanism_class": cls or "unknown",
                "occupancy_1xEC50": f"{occ_1x:.3f}",
                "parameter": "", "shift_value": "",
                "evidence_grade": "DEFERRED",
                "source": f"No shift rule defined for class '{cls or 'unknown'}'",
            })
            continue

        for param_name, formula, grade, source in rules:
            value = formula(occ_1x)
            shift_rows.append({
                "anesthetic": ane, "gene": gene,
                "mechanism_class": cls,
                "occupancy_1xEC50": f"{occ_1x:.3f}",
                "parameter": param_name,
                "shift_value": f"{value:.4f}",
                "evidence_grade": grade,
                "source": source,
            })
            ane_block = overlay["by_anesthetic"].setdefault(ane, {})
            tgt_block = ane_block.setdefault(gene, {
                "mechanism_class": cls,
                "occupancy_1xEC50": occ_1x,
                "parameters": {},
            })
            tgt_block["parameters"][param_name] = {
                "value": value, "evidence_grade": grade, "source": source,
            }

    fieldnames = ["anesthetic", "gene", "mechanism_class", "occupancy_1xEC50",
                  "parameter", "shift_value", "evidence_grade", "source"]
    with open(KINETICS_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(shift_rows)
    print(f"Per-row kinetic shifts:  {KINETICS_OUT}  ({len(shift_rows)} rows)")

    with open(WAVE2_OVERLAY, "w") as f:
        json.dump(overlay, f, indent=2, default=str)
    print(f"Wave 2 overlay JSON:     {WAVE2_OVERLAY}")

    by_class: dict[str, int] = {}
    by_grade: dict[str, int] = {}
    for r in shift_rows:
        by_class[r["mechanism_class"]] = by_class.get(r["mechanism_class"], 0) + 1
        by_grade[r["evidence_grade"]] = by_grade.get(r["evidence_grade"], 0) + 1

    print("\n=== Phase D summary ===")
    print(f"Total (anesthetic × target × parameter) shifts: {len(shift_rows)}")
    print("By mechanism class:")
    for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {k:25s}  {v} rows")
    print("By evidence grade:")
    for k, v in sorted(by_grade.items(), key=lambda kv: -kv[1]):
        print(f"  {k:15s}  {v} rows")

    def shift_mag(r):
        try:
            v = float(r["shift_value"]) if r["shift_value"] else 0.0
        except ValueError:
            return 0.0
        return abs(v) if r["parameter"] == "n_Ca_delta" else abs(v - 1.0)

    ranked = [r for r in shift_rows if r["shift_value"]]
    ranked.sort(key=lambda r: -shift_mag(r))

    print("\nTop 12 largest-magnitude shifts at 1× EC50:")
    print(f"{'anesthetic':12s} {'gene':10s} {'class':25s} {'param':20s} {'value':>8} {'grade':10s}")
    for r in ranked[:12]:
        print(f"  {r['anesthetic']:11s} {r['gene']:10s} {r['mechanism_class']:25s} "
              f"{r['parameter']:20s} {r['shift_value']:>8} {r['evidence_grade']:10s}")

    with open(PHASE_D_MD, "w") as f:
        f.write("# Phase D — kinetic-shift translation summary\n\n")
        f.write(f"- Source: {BEST_OCC}\n")
        f.write(f"- Total rows: {len(shift_rows)}\n")
        f.write(f"- Wave 2 overlay JSON: {WAVE2_OVERLAY}\n\n")
        f.write("## By mechanism class\n\n")
        for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]):
            f.write(f"- {k}: {v} rows\n")
        f.write("\n## By evidence grade\n\n")
        for k, v in sorted(by_grade.items(), key=lambda kv: -kv[1]):
            f.write(f"- {k}: {v} rows\n")
        f.write("\n## Top 12 largest-magnitude shifts at 1× EC50\n\n")
        f.write("| anesthetic | gene | class | parameter | shift | grade |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in ranked[:12]:
            f.write(f"| {r['anesthetic']} | {r['gene']} | {r['mechanism_class']} | "
                    f"{r['parameter']} | {r['shift_value']} | {r['evidence_grade']} |\n")
    print(f"\nMarkdown summary: {PHASE_D_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
