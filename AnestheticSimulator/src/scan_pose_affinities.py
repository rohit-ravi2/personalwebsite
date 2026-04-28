"""Extract affinity from each completed Vina pose file.

Independent of the main dock pipeline's CSV writeout. Lets us evaluate
partial Phase B output without waiting for the full sweep to finish.

Output: artifacts/binding/vina_results_from_poses.csv

Usage:
    conda activate wave-p-docking
    python src/scan_pose_affinities.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSES_DIR = ROOT / "artifacts" / "binding" / "poses"
OUT = ROOT / "artifacts" / "binding" / "vina_results_from_poses.csv"
LOG = ROOT / "artifacts" / "binding" / "full_sweep.log"


def parse_pose_filename(name: str) -> dict | None:
    # e.g. "halothane_NCA-2_G5EDM1_p175_out.pdbqt"
    m = re.match(r"^(?P<ane>[\w-]+?)_(?P<gene>[A-Z][\w-]*?)_(?P<acc>[A-Z0-9]+)_p(?P<pocket>\d+)_out\.pdbqt$", name)
    if not m:
        return None
    return {
        "anesthetic": m.group("ane"),
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
                        continue
    except OSError:
        return None
    return None


def parse_log_for_druggability() -> dict[tuple[str, str, int], float]:
    """Build a (anesthetic, gene_acc, pocket_id) -> druggability map from the sweep log."""
    if not LOG.exists():
        return {}
    cur_drug: float | None = None
    out: dict[tuple[str, str, int], float] = {}
    pat = re.compile(
        r"^\s+(?P<ane>\S+)\s+->\s+(?P<gene_acc>\S+)\s+pocket\s*(?P<pid>\d+)\s+drug=(?P<drug>[\d.]+)"
    )
    with open(LOG) as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            key = (m.group("ane"), m.group("gene_acc"), int(m.group("pid")))
            out[key] = float(m.group("drug"))
    return out


def main() -> int:
    if not POSES_DIR.is_dir():
        print(f"No poses dir at {POSES_DIR}")
        return 1
    pose_files = sorted(POSES_DIR.glob("*_out.pdbqt"))
    print(f"Pose files found: {len(pose_files)}")
    drug_map = parse_log_for_druggability()
    print(f"Druggability entries from log: {len(drug_map)}")

    rows = []
    for p in pose_files:
        meta = parse_pose_filename(p.name)
        if meta is None:
            continue
        aff = best_affinity(p)
        if aff is None:
            continue
        gene_acc = f"{meta['gene']}_{meta['uniprot_acc']}"
        drug = drug_map.get((meta["anesthetic"], gene_acc, meta["pocket_id"]), "")
        rows.append({
            "anesthetic": meta["anesthetic"],
            "gene": meta["gene"],
            "uniprot_acc": meta["uniprot_acc"],
            "pocket_id": meta["pocket_id"],
            "druggability_score": drug,
            "affinity_kcal_per_mol": aff,
        })

    if not rows:
        print("No affinities parsed.")
        return 1

    fieldnames = list(rows[0].keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {OUT} ({len(rows)} rows)")

    # Also produce a vina_results.csv compatible file (matches dock_pipeline schema)
    compat = ROOT / "artifacts" / "binding" / "vina_results.csv"
    fieldnames_compat = ["anesthetic", "gene", "uniprot_acc", "pocket_id",
                         "druggability_score", "center_x", "center_y", "center_z",
                         "size_x", "size_y", "size_z",
                         "affinity_kcal_per_mol", "status", "message"]
    with open(compat, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_compat)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames_compat}
            row["status"] = "OK"
            writer.writerow(row)
    print(f"Wrote compat schema (consumable by phase_c_occupancy.py): {compat} ({len(rows)} rows)")

    # Summary
    by_target = {}
    for r in rows:
        by_target.setdefault(r["gene"], []).append(r["affinity_kcal_per_mol"])
    print(f"\nTargets covered: {len(by_target)}")
    print("Genes (count of dockings):")
    for g in sorted(by_target):
        print(f"  {g:10s}  {len(by_target[g])} dockings  best ΔG={min(by_target[g]):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
