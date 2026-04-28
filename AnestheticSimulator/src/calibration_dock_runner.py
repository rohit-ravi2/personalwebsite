"""Calibration docking runner — generalized version of phase_b_dock_pipeline.py.

Accepts CLI flags --structures-dir, --ligands-dir, --pockets-dir, --out-dir,
--results-csv so the same logic can drive:

(1) Mammalian homolog × anesthetic dockings (Stage 3a)
(2) Negative-control × C. elegans Tier-1 dockings (Stage 3b)

Pocket detection and box-center logic identical to phase_b_dock_pipeline.py.

Usage:
    # Stage 3a — mammalian homolog × anesthetic panel
    python src/calibration_dock_runner.py \\
        --structures-dir artifacts/calibration/structures \\
        --ligands-dir anesthetics/anesthetic_smiles \\
        --pockets-dir artifacts/calibration/structures \\
        --receptors-dir artifacts/calibration/receptors \\
        --poses-dir artifacts/calibration/poses_mammalian \\
        --results-csv artifacts/calibration/mammalian_vina_results.csv

    # Stage 3b — negative controls × C. elegans Tier-1
    python src/calibration_dock_runner.py \\
        --structures-dir artifacts/structures \\
        --ligands-dir anesthetics/negative_controls \\
        --pockets-dir artifacts/structures \\
        --receptors-dir artifacts/binding/receptors \\
        --poses-dir artifacts/calibration/poses_negative \\
        --results-csv artifacts/calibration/negative_vina_results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prepare_receptor(pdb_in: Path, pdbqt_out: Path) -> tuple[bool, str]:
    if pdbqt_out.exists():
        return True, "cached"
    pdbqt_out.parent.mkdir(parents=True, exist_ok=True)
    base = pdbqt_out.with_suffix("")
    proc = subprocess.run(
        ["mk_prepare_receptor.py", "--read_pdb", str(pdb_in),
         "-o", str(base), "-p"],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-300:]
    if not pdbqt_out.exists():
        cand = base.with_suffix(".pdbqt")
        if cand.exists():
            shutil.move(str(cand), str(pdbqt_out))
        else:
            return False, "PDBQT not produced"
    return True, "prepared"


def parse_pocket_atom_pdb(p: Path) -> np.ndarray:
    coords = []
    with open(p) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    return np.array(coords, dtype=np.float32)


def parse_druggability(info_path: Path) -> dict[int, float]:
    if not info_path.exists():
        return {}
    out: dict[int, float] = {}
    cur_id = None
    with open(info_path) as f:
        for line in f:
            m_p = re.match(r"^Pocket (\d+) :", line)
            if m_p:
                cur_id = int(m_p.group(1))
                continue
            if cur_id is None:
                continue
            m_d = re.search(r"Druggability Score\s*:\s*([\d.]+)", line)
            if m_d:
                out[cur_id] = float(m_d.group(1))
    return out


def find_pocket_atms(target_dir: Path, top_k: int) -> list[tuple[Path, float]]:
    pockets_dir = target_dir / "pockets"
    if not pockets_dir.is_dir():
        return []
    info_files = list(target_dir.glob("*_info.txt"))
    drug = parse_druggability(info_files[0]) if info_files else {}
    pocket_files = []
    for pf in pockets_dir.glob("pocket*_atm.pdb"):
        m = re.match(r"pocket(\d+)_atm", pf.stem)
        if not m: continue
        pid = int(m.group(1))
        pocket_files.append((pf, drug.get(pid, 0.0), pid))
    pocket_files.sort(key=lambda t: (-t[1], t[2]))
    return [(pf, ds) for pf, ds, _ in pocket_files[:top_k]]


def pocket_box(coords: np.ndarray, padding: float = 6.0,
               min_size: float = 22.0) -> tuple[tuple, tuple]:
    centroid = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 2 * padding, min_size)
    return tuple(map(float, centroid)), tuple(map(float, size))


def run_vina(receptor: Path, ligand: Path, center: tuple, size: tuple,
             out_pdbqt: Path, log: Path,
             exhaustiveness: int = 8, num_modes: int = 9) -> tuple[bool, float | None, str]:
    cmd = [
        "vina",
        "--receptor", str(receptor), "--ligand", str(ligand),
        "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}", "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.2f}", "--size_y", f"{size[1]:.2f}", "--size_z", f"{size[2]:.2f}",
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
        "--cpu", "4", "--out", str(out_pdbqt),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    log.write_text(proc.stdout + "\n--STDERR--\n" + proc.stderr)
    if proc.returncode != 0:
        return False, None, f"vina exit {proc.returncode}"
    aff = None
    if out_pdbqt.exists():
        with open(out_pdbqt) as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    try: aff = float(parts[3]); break
                    except (IndexError, ValueError): pass
    if aff is None:
        return False, None, "no REMARK VINA RESULT"
    return True, aff, f"affinity={aff:.2f} kcal/mol elapsed={dt:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structures-dir", required=True, type=Path)
    ap.add_argument("--ligands-dir", required=True, type=Path)
    ap.add_argument("--pockets-dir", required=True, type=Path,
                    help="Where {target}_out/ fpocket dirs live (often == structures-dir)")
    ap.add_argument("--receptors-dir", required=True, type=Path)
    ap.add_argument("--poses-dir", required=True, type=Path)
    ap.add_argument("--results-csv", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--target", help="Substring filter on target name")
    ap.add_argument("--ligand", help="Substring filter on ligand name")
    args = ap.parse_args()

    args.poses_dir.mkdir(parents=True, exist_ok=True)
    args.receptors_dir.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    targets = sorted(args.structures_dir.glob("*.pdb"))
    if args.target:
        targets = [p for p in targets if args.target.lower() in p.stem.lower()]
    ligands = sorted(args.ligands_dir.glob("*.pdbqt"))
    if args.ligand:
        ligands = [p for p in ligands if args.ligand.lower() in p.stem.lower()]
    if not targets or not ligands:
        print(f"No targets/ligands matched ({len(targets)}/{len(ligands)})")
        return 1

    print(f"Targets: {len(targets)}  Ligands: {len(ligands)}  Top-K: {args.top_k}")
    print(f"Total dockings: {len(targets) * len(ligands) * args.top_k}")

    rows = []
    n_ok = 0
    for tgt in targets:
        gene_acc = tgt.stem
        gene = gene_acc.split("_")[0]
        receptor_pdbqt = args.receptors_dir / f"{gene_acc}.pdbqt"
        ok, msg = prepare_receptor(tgt, receptor_pdbqt)
        if not ok:
            print(f"  RECEPTOR FAIL {gene}: {msg}")
            continue

        target_out = args.pockets_dir / f"{gene_acc}_out"
        # Run fpocket if not already
        if not target_out.exists():
            print(f"  fpocket on {gene_acc} ...", end=" ", flush=True)
            proc = subprocess.run(["fpocket", "-f", str(tgt)], capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(f"FAIL")
                continue
            print("ok")

        pocket_files = find_pocket_atms(target_out, args.top_k)
        if not pocket_files:
            print(f"  NO POCKETS  {gene}")
            continue

        for pf, drug_score in pocket_files:
            rank = int(re.match(r"pocket(\d+)_atm", pf.stem).group(1))
            coords = parse_pocket_atom_pdb(pf)
            if len(coords) == 0:
                continue
            center, size = pocket_box(coords)

            for lig in ligands:
                lig_name = lig.stem
                pose_out = args.poses_dir / f"{lig_name}_{gene_acc}_p{rank}_out.pdbqt"
                log_out = args.poses_dir / f"{lig_name}_{gene_acc}_p{rank}.log"
                print(f"  {lig_name:25s} -> {gene_acc:30s} p{rank:>2d} drug={drug_score:.3f}", flush=True)
                ok, aff, msg = run_vina(receptor_pdbqt, lig, center, size, pose_out, log_out,
                                         exhaustiveness=args.exhaustiveness)
                rows.append({
                    "ligand": lig_name, "gene": gene,
                    "uniprot_acc": gene_acc.split("_", 1)[1] if "_" in gene_acc else "",
                    "pocket_id": rank, "druggability_score": drug_score,
                    "affinity_kcal_per_mol": aff if aff is not None else "",
                    "status": "OK" if ok else "FAIL", "message": msg,
                })
                if ok:
                    n_ok += 1
                    print(f"      {msg}")

    fieldnames = list(rows[0].keys()) if rows else ["ligand", "gene", "uniprot_acc",
                                                       "pocket_id", "druggability_score",
                                                       "affinity_kcal_per_mol", "status", "message"]
    with open(args.results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"Successful: {n_ok}/{len(rows)}")
    print(f"Results: {args.results_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
