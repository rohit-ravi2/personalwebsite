"""Extract affinities from negative-control pose files (works on partial data).

Mirrors scan_pose_affinities.py but for the calibration sweep.
Output: artifacts/calibration/negative_vina_results.csv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSES = ROOT / "artifacts" / "calibration" / "poses_negative"
LOG = ROOT / "artifacts" / "calibration" / "negative_sweep.log"
OUT = ROOT / "artifacts" / "calibration" / "negative_vina_results.csv"


def parse_pose_filename(name: str) -> dict | None:
    # e.g. "cis_12_dichloroethylene_SNB-1_O02495_p2_out.pdbqt"
    m = re.match(r"^(?P<lig>[\w]+?)_(?P<gene>[A-Z][\w-]*?)_(?P<acc>[A-Z0-9]+)_p(?P<pocket>\d+)_out\.pdbqt$", name)
    if not m:
        # Try with underscores in ligand name (cis_12_dichloroethylene)
        m = re.match(r"^(?P<lig>.+?)_(?P<gene>[A-Z][\w-]*?)_(?P<acc>[A-Z0-9]+)_p(?P<pocket>\d+)_out\.pdbqt$", name)
        if not m:
            return None
    return {
        "ligand": m.group("lig"),
        "gene": m.group("gene"),
        "uniprot_acc": m.group("acc"),
        "pocket_id": int(m.group("pocket")),
    }


def best_affinity(path: Path) -> float | None:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    try:
                        return float(parts[3])
                    except (IndexError, ValueError):
                        pass
    except OSError:
        return None
    return None


def parse_log_drug() -> dict[tuple[str, str, int], float]:
    if not LOG.exists():
        return {}
    pat = re.compile(
        r"^\s+(?P<lig>\S+)\s+->\s+(?P<gene_acc>\S+)\s+p\s*(?P<pid>\d+)\s+drug=(?P<drug>[\d.]+)"
    )
    out: dict[tuple[str, str, int], float] = {}
    with open(LOG) as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            out[(m.group("lig"), m.group("gene_acc"), int(m.group("pid")))] = float(m.group("drug"))
    return out


def main() -> int:
    pose_files = sorted(POSES.glob("*_out.pdbqt"))
    print(f"Pose files: {len(pose_files)}")
    drug_map = parse_log_drug()

    rows = []
    for p in pose_files:
        meta = parse_pose_filename(p.name)
        if meta is None:
            continue
        aff = best_affinity(p)
        if aff is None:
            continue
        gene_acc = f"{meta['gene']}_{meta['uniprot_acc']}"
        drug = drug_map.get((meta["ligand"], gene_acc, meta["pocket_id"]), "")
        rows.append({
            "ligand": meta["ligand"],
            "gene": meta["gene"],
            "uniprot_acc": meta["uniprot_acc"],
            "pocket_id": meta["pocket_id"],
            "druggability_score": drug,
            "affinity_kcal_per_mol": aff,
            "status": "OK",
            "message": "",
        })
    if not rows:
        print("No rows parsed")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} rows)")

    # Quick summary
    by_ligand = {}
    for r in rows:
        by_ligand.setdefault(r["ligand"], []).append(r)
    print("\nLigands docked:")
    for lig, rs in sorted(by_ligand.items()):
        n_targets = len(set(r["gene"] for r in rs))
        print(f"  {lig:30s}  {len(rs):>3d} dockings  {n_targets} targets covered  best ΔG={min(r['affinity_kcal_per_mol'] for r in rs):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
