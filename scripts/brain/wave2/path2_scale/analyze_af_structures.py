"""
Analyze AlphaFold structures for the NCA complex.

Parses PDB files for NCA-2 (Q22573/G5EDM1), UNC-79 (P42173), NLF-1 (M4Q8W4).
UNC-80 (Q9XV66) too large (3263 aa) — not in AF database.

Extracts:
  - Per-residue pLDDT (confidence) from B-factor column
  - Average pLDDT (overall structure confidence)
  - Mean pLDDT in well-folded segments (proxy for stable interaction interface
    availability)
  - Total residue count

Outputs an AF-derived metadata dict for use in accessory-factor weighting.

Biological interpretation:
  - UNC-79 binds NCA directly (obligate primary partner per Yeh 2008,
    Humphrey 2007; NALCN cryo-EM structures show direct contact).
  - UNC-80 binds via UNC-79 (secondary).
  - NLF-1 modulates function (peripheral assembly per Xie 2013).

AF data tells us:
  - How structured each protein is (high pLDDT = compact, stable interface
    available; low pLDDT = disordered, less reliable assembly)
  - Mean pLDDT scales as a confidence weight for that protein's contribution
    to NCA complex assembly.
"""
from __future__ import annotations

from pathlib import Path
import statistics

PDB_DIR = Path(__file__).resolve().parent / "alphafold"
OUT = Path(__file__).resolve().parent / "af_complex_metadata.py"

PROTEINS = {
    "nca-2":   ("G5EDM1", "NCA-2: pore-forming subunit"),
    "unc-79":  ("P42173", "UNC-79: obligate primary accessory (direct NCA binding)"),
    "nlf-1":   ("M4Q8W4", "NLF-1: peripheral modulator"),
    # UNC-80 (Q9XV66, 3263 aa) not in AF DB — too large for single fragment
}


def parse_pdb_plddt(pdb_path: Path) -> dict:
    """Extract per-CA-atom pLDDT (B-factor) from AF PDB."""
    if not pdb_path.exists():
        return {"error": f"file not found: {pdb_path}"}
    plddt = []
    residues = set()
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                bfac = float(line[60:66])
                res_id = int(line[22:26])
                plddt.append(bfac)
                residues.add(res_id)
            except ValueError:
                continue
    if not plddt:
        return {"error": "no CA atoms"}
    return {
        "n_residues": len(residues),
        "mean_plddt": statistics.mean(plddt),
        "median_plddt": statistics.median(plddt),
        "min_plddt": min(plddt),
        "max_plddt": max(plddt),
        # Fraction of residues with pLDDT > 70 (well-structured)
        "fraction_high_confidence": sum(1 for v in plddt if v > 70) / len(plddt),
        # Fraction "very high" > 90 (interface-quality)
        "fraction_very_high": sum(1 for v in plddt if v > 90) / len(plddt),
    }


def main():
    results = {}
    print("=== AlphaFold Structure Analysis: NCA Complex ===\n")
    print(f"{'protein':<10s} {'uniprot':<10s} {'n_res':>6s} {'mean_pLDDT':>10s} "
          f"{'frac>70':>8s} {'frac>90':>8s}")
    print("-" * 70)
    for gene, (uniprot, description) in PROTEINS.items():
        pdb = PDB_DIR / f"AF-{uniprot}.pdb"
        stats = parse_pdb_plddt(pdb)
        if "error" in stats:
            print(f"{gene:<10s} {uniprot:<10s} {stats['error']}")
            continue
        results[gene] = {**stats, "uniprot": uniprot, "description": description}
        print(f"{gene:<10s} {uniprot:<10s} {stats['n_residues']:>6d} "
              f"{stats['mean_plddt']:>10.1f} "
              f"{stats['fraction_high_confidence']:>8.2f} "
              f"{stats['fraction_very_high']:>8.2f}")

    # Compute relative confidence weights for each accessory protein
    # Higher pLDDT (more structured) = more reliable interface for complex
    # assembly
    print("\n=== Derived weights for accessory-factor formula ===")
    if "unc-79" in results and "nlf-1" in results:
        u79_w = results["unc-79"]["fraction_high_confidence"]
        nlf_w = results["nlf-1"]["fraction_high_confidence"]
        # UNC-80 not in AF DB — assume similar to UNC-79 (related architecture)
        u80_w = u79_w
        total = u79_w + u80_w + nlf_w
        # Biological prior: UNC-79 is obligate primary (Yeh 2008)
        # Structural data: confidence weights from AF pLDDT folded fraction
        # Combine: UNC-79 gets BOTH the obligate role + its structural weight
        bio_unc79_priority = 0.6   # obligate primary partner
        bio_unc80_priority = 0.3
        bio_nlf1_priority  = 0.1
        # Net weight = biological priority × structural confidence
        w_u79 = bio_unc79_priority * u79_w / max(total/3, 0.1)
        w_u80 = bio_unc80_priority * u80_w / max(total/3, 0.1)
        w_nlf = bio_nlf1_priority  * nlf_w / max(total/3, 0.1)
        w_sum = w_u79 + w_u80 + w_nlf
        # Normalize
        w_u79 /= w_sum; w_u80 /= w_sum; w_nlf /= w_sum
        print(f"  UNC-79 weight: {w_u79:.3f}")
        print(f"  UNC-80 weight: {w_u80:.3f}")
        print(f"  NLF-1  weight: {w_nlf:.3f}")
        results["weights"] = {"unc79": w_u79, "unc80": w_u80, "nlf1": w_nlf}

    # Write metadata module
    with OUT.open("w") as f:
        f.write('"""AF-derived metadata for NCA complex assembly.\n')
        f.write('Used to weight accessory-protein contribution per-cell.\n')
        f.write('"""\n')
        f.write('from __future__ import annotations\n\n')
        f.write(f'AF_METADATA = {results!r}\n')
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
