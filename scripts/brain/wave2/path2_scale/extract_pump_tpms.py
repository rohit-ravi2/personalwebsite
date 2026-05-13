"""
Extract per-cell TPM values for the four pump genes (eat-6 / mca-3 / kcc-2 /
abts-1) from CeNGEN T2 medium threshold, and emit pump_tpm_data.py with
dicts keyed by CeNGEN class name. Run once.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path("/mnt/ssd4tb/Desktop/C-Elegans/data/expression/cengen/thresholded/021821_medium_threshold2.csv")
OUT_PATH = Path(__file__).resolve().parent / "pump_tpm_data.py"

PUMP_GENES = ["eat-6", "mca-3", "kcc-2", "abts-1"]


def extract():
    with CSV_PATH.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        # header: ["", "gene_name", "Wormbase_ID", "ADA", "ADE", ...]
        cengen_classes = header[3:]
        # Filter to non-empty class names
        cengen_classes = [c.strip().strip('"') for c in cengen_classes if c.strip()]

        rows = {}
        for row in reader:
            if len(row) < 3:
                continue
            gene = row[1].strip().strip('"')
            if gene in PUMP_GENES:
                # Parse numeric values per cell
                vals = {}
                for cls, raw in zip(cengen_classes, row[3:]):
                    raw = raw.strip()
                    try:
                        v = float(raw) if raw else 0.0
                    except ValueError:
                        v = 0.0
                    vals[cls] = v
                rows[gene] = vals

    return cengen_classes, rows


def main():
    cengen_classes, rows = extract()
    print(f"Found {len(rows)} pump genes across {len(cengen_classes)} CeNGEN classes")
    for g, vals in rows.items():
        nonzero = sum(1 for v in vals.values() if v > 0)
        print(f"  {g}: {nonzero}/{len(vals)} non-zero, range "
              f"{min(vals.values()):.1f} – {max(vals.values()):.1f} TPM")

    # Write the data module
    with OUT_PATH.open("w") as f:
        f.write('"""CeNGEN T2 pump-gene TPM lookup — auto-generated from\n')
        f.write('021821_medium_threshold2.csv. Per cell, per pump gene.\n')
        f.write('"""\n')
        f.write('from __future__ import annotations\n\n')
        f.write('# Pump-gene TPMs (CeNGEN T2 medium threshold)\n')
        for gene in PUMP_GENES:
            key = gene.replace("-", "_").upper() + "_TPM"
            f.write(f"\n{key}: dict[str, float] = {{\n")
            vals = rows[gene]
            for cls in sorted(vals):
                f.write(f"    {cls!r:14s}: {vals[cls]:.4f},\n")
            f.write("}\n")

        f.write('\n\n# AVA-class anchor values (for relative scaling)\n')
        for gene in PUMP_GENES:
            key = gene.replace("-", "_").upper() + "_TPM"
            ava_val = rows[gene].get("AVA", 0.0)
            f.write(f"{gene.replace('-','_').upper()}_AVA = {ava_val:.4f}\n")

    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
