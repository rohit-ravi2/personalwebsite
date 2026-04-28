"""Phase A — fpocket cavity detection on downloaded AlphaFold DB structures.

Runs fpocket on every PDB in `artifacts/structures/` and parses per-pocket
metrics into a single summary table. Pocket score, druggability score,
pocket volume, residue list per pocket are recorded so downstream Phase B
can constrain Vina dockings to specific pockets.

Output:
- artifacts/structures/<gene>_<acc>_pockets/  - fpocket native output dir
- artifacts/structures/pocket_summary.csv     - flat summary table

Usage:
    conda activate wave-p-docking
    python src/phase_a_pocket_detect.py
    python src/phase_a_pocket_detect.py --target UNC-49     # just one
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRUCTURES_DIR = ROOT / "artifacts" / "structures"
SUMMARY_CSV = STRUCTURES_DIR / "pocket_summary.csv"


def parse_fpocket_info(info_path: Path) -> list[dict]:
    """Parse fpocket's _info.txt file into per-pocket dicts."""
    if not info_path.exists():
        return []
    text = info_path.read_text()
    pockets = []
    cur = None
    for line in text.splitlines():
        m_p = re.match(r"^Pocket (\d+) :", line)
        if m_p:
            if cur:
                pockets.append(cur)
            cur = {"pocket_id": int(m_p.group(1))}
            continue
        if cur is None:
            continue
        m_kv = re.match(r"^\s+([\w\s\-\.\(\)/]+?)\s*:\s*(.+)$", line)
        if m_kv:
            key = m_kv.group(1).strip()
            val = m_kv.group(2).strip()
            try:
                val_num = float(val)
                cur[key] = val_num
            except ValueError:
                cur[key] = val
    if cur:
        pockets.append(cur)
    return pockets


def run_fpocket(pdb_path: Path) -> tuple[bool, list[dict], str]:
    """Run fpocket on a PDB. Returns (success, pockets, message)."""
    out_dir_name = f"{pdb_path.stem}_out"
    out_dir = pdb_path.parent / out_dir_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    proc = subprocess.run(
        ["fpocket", "-f", str(pdb_path)],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        return False, [], f"fpocket exit {proc.returncode}: {proc.stderr[:200]}"

    info_path = out_dir / f"{pdb_path.stem}_info.txt"
    pockets = parse_fpocket_info(info_path)
    if not pockets:
        return False, [], f"no pockets parsed from {info_path}"

    return True, pockets, f"{len(pockets)} pockets"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", help="Only run on this gene name (substring match)")
    ap.add_argument("--top", type=int, default=5,
                    help="Save top-N pockets per target into summary (default 5)")
    args = ap.parse_args()

    pdbs = sorted(STRUCTURES_DIR.glob("*.pdb"))
    if args.target:
        pdbs = [p for p in pdbs if args.target.lower() in p.stem.lower()]

    if not pdbs:
        print(f"No PDBs found at {STRUCTURES_DIR}")
        return 1

    print(f"Targets to process: {len(pdbs)}")
    summary_rows = []
    n_ok = 0

    for pdb in pdbs:
        gene_acc = pdb.stem  # e.g. "UNC-49_G5EBQ0"
        print(f"  {gene_acc:30s}", end=" ", flush=True)
        ok, pockets, msg = run_fpocket(pdb)
        if not ok:
            print(f"FAIL — {msg}")
            summary_rows.append({
                "gene_acc": gene_acc, "pocket_id": "", "score": "",
                "druggability_score": "", "n_alpha_spheres": "",
                "volume": "", "n_residues": "",
                "status": f"FAIL: {msg}",
            })
            continue
        n_ok += 1

        # Sort by druggability score desc; ties by pocket_score desc
        def keyfn(p):
            return (
                -float(p.get("Druggability Score", 0) or 0),
                -float(p.get("Score", 0) or 0),
            )
        pockets_sorted = sorted(pockets, key=keyfn)
        top = pockets_sorted[:args.top]
        top_drug = top[0].get("Druggability Score", "—") if top else "—"
        print(f"{len(pockets)} pockets, top druggability {top_drug}")

        for rank, p in enumerate(top, 1):
            summary_rows.append({
                "gene_acc": gene_acc,
                "pocket_id": p.get("pocket_id", ""),
                "rank": rank,
                "score": p.get("Score", ""),
                "druggability_score": p.get("Druggability Score", ""),
                "n_alpha_spheres": p.get("Number of Alpha Spheres", ""),
                "volume": p.get("Pocket volume (Monte Carlo)", p.get("Pocket volume (convex hull)", "")),
                "hydrophobicity_score": p.get("Hydrophobicity score", ""),
                "polarity_score": p.get("Polarity score", ""),
                "mean_local_hydrophobic_density": p.get("Mean local hydrophobic density", ""),
                "status": "OK",
            })

    # Always include a header even if no rows (empty summary still useful)
    fieldnames = [
        "gene_acc", "pocket_id", "rank", "score", "druggability_score",
        "n_alpha_spheres", "volume", "hydrophobicity_score", "polarity_score",
        "mean_local_hydrophobic_density", "status",
    ]
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            # ensure all keys exist
            for k in fieldnames:
                row.setdefault(k, "")
            writer.writerow(row)

    print()
    print(f"Targets with pockets:    {n_ok}/{len(pdbs)}")
    print(f"Summary CSV:             {SUMMARY_CSV}")
    if n_ok > 0:
        # Top-druggability summary
        from collections import defaultdict
        per_target_top = defaultdict(lambda: -1.0)
        for r in summary_rows:
            if r.get("rank") == 1 and r.get("druggability_score"):
                try:
                    per_target_top[r["gene_acc"]] = float(r["druggability_score"])
                except (TypeError, ValueError):
                    pass
        sorted_tt = sorted(per_target_top.items(), key=lambda kv: -kv[1])
        print("\nTop-pocket druggability (Vina-relevance proxy, > 0.5 is high):")
        for gene_acc, score in sorted_tt[:15]:
            mark = "***" if score >= 0.5 else ("**" if score >= 0.3 else "")
            print(f"  {gene_acc:30s}  {score:.3f} {mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
