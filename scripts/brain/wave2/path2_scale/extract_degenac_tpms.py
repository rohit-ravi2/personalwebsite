"""
Extract per-cell TPM for the DEG/ENaC family genes from CeNGEN T2.

We aggregate the depolarizing-leak members:
  unc-8, del-1, del-2, del-3, asic-1, asic-2, deg-1, acd-1, acd-3
Excluded (purely mechanosensory):
  mec-4, mec-10, deg-3, mec-6
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path("/mnt/ssd4tb/Desktop/C-Elegans/data/expression/cengen/thresholded/021821_medium_threshold2.csv")
OUT_PATH = Path(__file__).resolve().parent / "degenac_tpm_data.py"

GENES = ["unc-8", "del-1", "del-2", "del-3", "asic-1", "asic-2",
         "deg-1", "acd-1", "acd-3"]


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

    for g, vals in rows.items():
        nonzero = sum(1 for v in vals.values() if v > 0)
        max_tpm = max(vals.values())
        print(f"  {g:8s}: {nonzero}/{len(vals)} non-zero, max {max_tpm:.0f} TPM")

    with OUT_PATH.open("w") as f:
        f.write('"""CeNGEN T2 TPM for DEG/ENaC family (depolarizing-leak subset).\n')
        f.write('Excludes mechanosensory-specific mec-4/mec-10/deg-3/mec-6.\n')
        f.write('"""\n')
        f.write('from __future__ import annotations\n\n')
        for gene in GENES:
            key = gene.replace("-", "_").upper() + "_TPM"
            f.write(f"\n{key}: dict[str, float] = {{\n")
            vals = rows.get(gene, {})
            for cls in sorted(vals):
                f.write(f"    {cls!r:14s}: {vals[cls]:.4f},\n")
            f.write("}\n")
        f.write('\n\nDEGENAC_TABLES = (\n')
        for gene in GENES:
            key = gene.replace("-", "_").upper() + "_TPM"
            f.write(f"    {key},\n")
        f.write(')\n')
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
