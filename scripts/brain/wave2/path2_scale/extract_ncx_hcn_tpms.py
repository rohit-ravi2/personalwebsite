"""
Extract per-cell TPM values for NCX paralogs + HCN homologs from CeNGEN T2.

NCX (Na/Ca exchanger): ncx-1, ncx-2, ncx-3, ncx-4, ncx-5, ncx-9
HCN homologs in C. elegans: cng-1, cng-2, cng-3, tax-2, tax-4
  (cyclic-nucleotide-gated; closely related to HCN, depolarization-driven inward
  cation current at hyperpolarized V in some)
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path("/mnt/ssd4tb/Desktop/C-Elegans/data/expression/cengen/thresholded/021821_medium_threshold2.csv")
OUT_PATH = Path(__file__).resolve().parent / "ncx_hcn_tpm_data.py"

GENES = ["ncx-1", "ncx-2", "ncx-3", "ncx-4", "ncx-9",
         "cng-1", "cng-2", "cng-3", "tax-2", "tax-4"]


def main():
    with CSV_PATH.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        cengen_classes = [c.strip().strip('"') for c in header[3:] if c.strip()]
        rows = {}
        for row in reader:
            if len(row) < 3:
                continue
            gene = row[1].strip().strip('"')
            if gene in GENES:
                vals = {}
                for cls, raw in zip(cengen_classes, row[3:]):
                    try:
                        v = float(raw) if raw else 0.0
                    except ValueError:
                        v = 0.0
                    vals[cls] = v
                rows[gene] = vals

    print(f"Genes found: {sorted(rows.keys())}")
    for g, vals in rows.items():
        nonzero = sum(1 for v in vals.values() if v > 0)
        print(f"  {g}: {nonzero}/{len(vals)} non-zero, range {min(vals.values()):.1f} - {max(vals.values()):.1f}")

    with OUT_PATH.open("w") as f:
        f.write('"""CeNGEN T2 TPM for NCX paralogs + HCN homologs.\n')
        f.write('Used for cell-specific scaling of NCX I_max + HCN gbar.\n')
        f.write('"""\n')
        f.write('from __future__ import annotations\n\n')
        for gene in GENES:
            key = gene.replace("-", "_").upper() + "_TPM"
            f.write(f"\n{key}: dict[str, float] = {{\n")
            vals = rows.get(gene, {})
            for cls in sorted(vals):
                f.write(f"    {cls!r:14s}: {vals[cls]:.4f},\n")
            f.write("}\n")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
