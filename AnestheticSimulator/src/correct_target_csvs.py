"""Apply Phase A audit results to correct Tier-1 / Tier-2 target CSVs.

Reads `artifacts/structures/uniprot_id_audit.csv` and writes corrected
`targets/tier1_targets_corrected.csv` with verified UniProt IDs, verified
WormBase sequence IDs, AF DB pLDDT, and structure path columns added.

Preserves the original CSV unchanged so the audit history is visible.

Usage:
    python src/correct_target_csvs.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIER1_CSV = ROOT / "targets" / "tier1_targets.csv"
TIER1_OUT = ROOT / "targets" / "tier1_targets_corrected.csv"
AUDIT_CSV = ROOT / "artifacts" / "structures" / "uniprot_id_audit.csv"


def load_audit() -> dict[str, dict]:
    out = {}
    with open(AUDIT_CSV) as f:
        for row in csv.DictReader(f):
            out[row["gene_name"]] = row
    return out


def main() -> int:
    audit = load_audit()
    out_rows = []

    with open(TIER1_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        # Add new columns, keep originals for traceability
        new_fields = [
            "verified_uniprot_id",
            "verified_wormbase_seq_id",
            "verified_protein_name",
            "alphafold_pdb_path",
            "alphafold_global_plddt",
            "alphafold_high_conf_frac",
            "uniprot_id_correction_status",
        ]
        fieldnames_out = fieldnames + new_fields

        for row in reader:
            gene = row.get("gene_name", "").strip()
            a = audit.get(gene, {})
            row["verified_uniprot_id"] = a.get("verified_uniprot_id", "")
            row["verified_wormbase_seq_id"] = a.get("verified_wormbase_id", "")
            row["verified_protein_name"] = a.get("verified_protein_name", "")
            row["alphafold_pdb_path"] = a.get("alphafold_pdb_path", "")
            row["alphafold_global_plddt"] = a.get("alphafold_global_plddt", "")
            row["alphafold_high_conf_frac"] = a.get("alphafold_frac_high_confidence", "")
            csv_id = row.get("uniprot_id", "").strip()
            ver_id = a.get("verified_uniprot_id", "").strip()
            if not ver_id:
                row["uniprot_id_correction_status"] = "NO_HIT"
            elif csv_id == ver_id:
                row["uniprot_id_correction_status"] = "OK"
            else:
                row["uniprot_id_correction_status"] = f"CORRECTED:{csv_id}->{ver_id}"
            out_rows.append(row)

    with open(TIER1_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(out_rows)

    n_corrected = sum(1 for r in out_rows if r["uniprot_id_correction_status"].startswith("CORRECTED"))
    n_ok = sum(1 for r in out_rows if r["uniprot_id_correction_status"] == "OK")
    n_nohit = sum(1 for r in out_rows if r["uniprot_id_correction_status"] == "NO_HIT")
    n_with_struct = sum(1 for r in out_rows if r["alphafold_pdb_path"])

    print(f"Wrote: {TIER1_OUT}")
    print(f"Rows: {len(out_rows)}")
    print(f"  UniProt OK as in CSV: {n_ok}")
    print(f"  UniProt CORRECTED:    {n_corrected}")
    print(f"  UniProt NO_HIT:       {n_nohit}")
    print(f"  AF DB structure:      {n_with_struct}/{len(out_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
