"""Phase B step 2 — Vina docking pipeline (anesthetic x target x top-K pockets).

For each anesthetic ligand and each target structure:
1. Prepare receptor PDBQT via Meeko (one-time per target).
2. For top-K (default 3) fpocket-detected pockets, compute pocket centroid + box.
3. Run Vina dock.
4. Parse best-mode affinity, write to results CSV.

Outputs:
- artifacts/binding/receptors/<gene>.pdbqt
- artifacts/binding/poses/<anesthetic>_<gene>_p<rank>_out.pdbqt
- artifacts/binding/vina_results.csv

Usage:
    conda activate wave-p-docking
    python src/phase_b_dock_pipeline.py --target UNC-49 --anesthetic halothane
    python src/phase_b_dock_pipeline.py                     # everything
    python src/phase_b_dock_pipeline.py --top-k 1           # only top pocket
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
STRUCTURES_DIR = ROOT / "artifacts" / "structures"
LIGANDS_DIR = ROOT / "anesthetics" / "anesthetic_smiles"
RECEPTORS_DIR = ROOT / "artifacts" / "binding" / "receptors"
POSES_DIR = ROOT / "artifacts" / "binding" / "poses"
VINA_RESULTS = ROOT / "artifacts" / "binding" / "vina_results.csv"

VINA_NUM_MODES = 9
DEFAULT_BOX_SIZE = 22.0     # Angstroms; covers small-molecule anesthetics + slack


def prepare_receptor(pdb_in: Path, pdbqt_out: Path) -> tuple[bool, str]:
    if pdbqt_out.exists():
        return True, "cached"
    pdbqt_out.parent.mkdir(parents=True, exist_ok=True)
    base = pdbqt_out.with_suffix("")  # strip .pdbqt — meeko will re-add
    proc = subprocess.run(
        ["mk_prepare_receptor.py",
         "--read_pdb", str(pdb_in),
         "-o", str(base),
         "-p"],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-300:]
    if not pdbqt_out.exists():
        # meeko sometimes writes to base.pdbqt
        candidate = base.with_suffix(".pdbqt")
        if candidate.exists():
            shutil.move(str(candidate), str(pdbqt_out))
        else:
            return False, "PDBQT not produced"
    return True, "prepared"


def parse_pocket_atom_pdb(atm_pdb: Path) -> np.ndarray:
    """Load atom coordinates from fpocket pocket<i>_atm.pdb file."""
    coords = []
    with open(atm_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue
    return np.array(coords, dtype=np.float32)


def pocket_box(coords: np.ndarray, padding: float = 6.0,
               min_size: float = DEFAULT_BOX_SIZE) -> tuple[tuple, tuple]:
    """Return (center_xyz, size_xyz) for the Vina search box."""
    centroid = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 2 * padding, min_size)
    return tuple(map(float, centroid)), tuple(map(float, size))


def parse_druggability(info_path: Path) -> dict[int, float]:
    """Parse druggability score per pocket from fpocket _info.txt."""
    if not info_path.exists():
        return {}
    out = {}
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
    """Return up to top_k pocket atom PDBs ranked by druggability descending."""
    pockets_dir = target_dir / "pockets"
    if not pockets_dir.is_dir():
        return []
    info_files = list(target_dir.glob("*_info.txt"))
    drug = parse_druggability(info_files[0]) if info_files else {}

    pocket_files = []
    for pf in pockets_dir.glob("pocket*_atm.pdb"):
        m = re.match(r"pocket(\d+)_atm", pf.stem)
        if not m:
            continue
        pid = int(m.group(1))
        d_score = drug.get(pid, 0.0)
        pocket_files.append((pf, d_score, pid))

    pocket_files.sort(key=lambda t: (-t[1], t[2]))  # druggability desc, pocket id asc tiebreak
    return [(pf, ds) for pf, ds, _ in pocket_files[:top_k]]


def run_vina(receptor: Path, ligand: Path, center: tuple, size: tuple,
             out_pdbqt: Path, log: Path,
             exhaustiveness: int = 8) -> tuple[bool, float | None, str]:
    cmd = [
        "vina",
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.2f}",
        "--size_y", f"{size[1]:.2f}",
        "--size_z", f"{size[2]:.2f}",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(VINA_NUM_MODES),
        "--cpu", "4",
        "--out", str(out_pdbqt),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0
    log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        return False, None, f"vina exit {proc.returncode}"
    # Parse first MODE 1 affinity line. Vina output has "REMARK VINA RESULT:    -X.X"
    aff = None
    if out_pdbqt.exists():
        with open(out_pdbqt) as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    try:
                        aff = float(parts[3])
                        break
                    except (IndexError, ValueError):
                        pass
    if aff is None:
        return False, None, "no REMARK VINA RESULT in output"
    return True, aff, f"affinity={aff:.2f} kcal/mol  elapsed={elapsed:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", help="Substring match on target gene name")
    ap.add_argument("--anesthetic", help="Substring match on anesthetic name")
    ap.add_argument("--top-k", type=int, default=3, help="How many fpocket pockets per target")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    args = ap.parse_args()
    exhaustiveness = args.exhaustiveness

    POSES_DIR.mkdir(parents=True, exist_ok=True)
    RECEPTORS_DIR.mkdir(parents=True, exist_ok=True)

    target_pdbs = sorted(STRUCTURES_DIR.glob("*.pdb"))
    if args.target:
        target_pdbs = [p for p in target_pdbs if args.target.lower() in p.stem.lower()]

    ligands = sorted(LIGANDS_DIR.glob("*.pdbqt"))
    if args.anesthetic:
        ligands = [p for p in ligands if args.anesthetic.lower() in p.stem.lower()]

    if not target_pdbs or not ligands:
        print(f"No targets ({len(target_pdbs)}) or ligands ({len(ligands)}) matched filters")
        return 1

    print(f"Targets: {len(target_pdbs)} | Anesthetics: {len(ligands)} | Top-K pockets: {args.top_k}")
    print(f"Total dockings: {len(target_pdbs) * len(ligands) * args.top_k}")
    print()

    results = []
    n_ok = 0
    for tgt in target_pdbs:
        gene_acc = tgt.stem
        gene = gene_acc.split("_")[0]
        receptor_pdbqt = RECEPTORS_DIR / f"{gene_acc}.pdbqt"
        ok, msg = prepare_receptor(tgt, receptor_pdbqt)
        if not ok:
            print(f"  RECEPTOR FAIL  {gene}: {msg}")
            continue

        # Identify fpocket output directory
        target_out = STRUCTURES_DIR / f"{gene_acc}_out"
        pocket_files = find_pocket_atms(target_out, args.top_k)
        if not pocket_files:
            print(f"  NO POCKETS  {gene} — run phase_a_pocket_detect.py first")
            continue

        for pf, drug_score in pocket_files:
            rank = int(re.match(r"pocket(\d+)_atm", pf.stem).group(1))
            coords = parse_pocket_atom_pdb(pf)
            if len(coords) == 0:
                continue
            center, size = pocket_box(coords)

            for lig in ligands:
                ane = lig.stem
                pose_out = POSES_DIR / f"{ane}_{gene_acc}_p{rank}_out.pdbqt"
                log_out = POSES_DIR / f"{ane}_{gene_acc}_p{rank}.log"
                print(f"  {ane:11s} -> {gene_acc:24s} pocket{rank:>2d} drug={drug_score:.3f} center=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})", flush=True)
                ok, aff, msg = run_vina(receptor_pdbqt, lig, center, size, pose_out, log_out, exhaustiveness=exhaustiveness)
                results.append({
                    "anesthetic": ane,
                    "gene": gene,
                    "uniprot_acc": gene_acc.split("_", 1)[1] if "_" in gene_acc else "",
                    "pocket_id": rank,
                    "druggability_score": drug_score,
                    "center_x": center[0], "center_y": center[1], "center_z": center[2],
                    "size_x": size[0], "size_y": size[1], "size_z": size[2],
                    "affinity_kcal_per_mol": aff if aff is not None else "",
                    "status": "OK" if ok else "FAIL",
                    "message": msg,
                })
                if ok:
                    n_ok += 1
                    print(f"      {msg}")
                else:
                    print(f"      FAIL: {msg}")

    if not results:
        print("No dockings ran.")
        return 1

    fieldnames = list(results[0].keys())
    VINA_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    write_header = not VINA_RESULTS.exists()
    with open(VINA_RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Successful dockings: {n_ok}/{len(results)}")
    print(f"Results appended to: {VINA_RESULTS}")
    if results and any(r["affinity_kcal_per_mol"] != "" for r in results):
        print("\nTop 10 best affinities:")
        ok = [r for r in results if r["affinity_kcal_per_mol"] != ""]
        ok.sort(key=lambda r: float(r["affinity_kcal_per_mol"]))
        for r in ok[:10]:
            print(f"  {r['anesthetic']:11s} -> {r['gene']:8s} p{r['pocket_id']:>2d} (drug={r['druggability_score']:.2f})  "
                  f"{float(r['affinity_kcal_per_mol']):>6.2f} kcal/mol")

    return 0


if __name__ == "__main__":
    sys.exit(main())
