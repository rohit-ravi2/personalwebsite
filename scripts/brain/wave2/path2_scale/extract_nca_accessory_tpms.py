"""
Extract per-cell TPM for NCA accessory proteins (UNC-79, UNC-80, NLF-1).

In real C. elegans biology, NCA-1 and NCA-2 (pore-forming subunits) require
UNC-79 + UNC-80 + NLF-1 (accessory proteins) for stable activatable open
state. Cells with high NCA TPM but low UNC-79/UNC-80 may have non-functional
NCA, while cells with the full complement have plateau-supporting current.

This is the molecular composition that differentiates plateau cells from
phasic cells beyond what CeNGEN gene-level expression alone reveals.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path("/mnt/ssd4tb/Desktop/C-Elegans/data/expression/cengen/thresholded/021821_medium_threshold2.csv")
OUT_PATH = Path(__file__).resolve().parent / "nca_accessory_tpm_data.py"

GENES = ["unc-79", "unc-80", "nlf-1", "nca-1", "nca-2"]


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
        print(f"  {g}: {nonzero}/{len(vals)} non-zero, range {min(vals.values()):.1f} - {max(vals.values()):.1f}")

    with OUT_PATH.open("w") as f:
        f.write('"""CeNGEN T2 TPM for NCA accessory proteins.\n')
        f.write('UNC-79 + UNC-80 + NLF-1 required for functional NCA channels.\n')
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
