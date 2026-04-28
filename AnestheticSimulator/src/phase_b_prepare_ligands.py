"""Phase B step 1 — prepare anesthetic ligands as PDBQT for Vina.

Reads `anesthetics/anesthetic_panel.csv`, builds 3D structures from SMILES
via RDKit, adds hydrogens, generates a single low-energy conformer per ligand,
and writes:
- anesthetics/anesthetic_smiles/<name>.sdf  (3D SDF with explicit Hs)
- anesthetics/anesthetic_smiles/<name>.pdbqt  (Vina-ready)

Usage:
    conda activate wave-p-docking
    python src/phase_b_prepare_ligands.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "anesthetics" / "anesthetic_panel.csv"
SDF_DIR = ROOT / "anesthetics" / "anesthetic_smiles"


def build_3d(smiles: str, name: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True) != 0:
        # retry without ETKDG
        if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True,
                                 useBasicKnowledge=False) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    mol.SetProp("_Name", name)
    return mol


def write_sdf(mol: Chem.Mol, path: Path) -> None:
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()


def sdf_to_pdbqt(sdf: Path, pdbqt: Path) -> tuple[bool, str]:
    """Use Meeko's mk_prepare_ligand.py to convert SDF -> PDBQT."""
    proc = subprocess.run(
        ["mk_prepare_ligand.py", "-i", str(sdf), "-o", str(pdbqt)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        return False, proc.stderr[:300]
    return True, "ok"


def main() -> int:
    SDF_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(PANEL) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    print(f"Anesthetics in panel: {len(rows)}")
    n_ok = 0
    summary = []
    for r in rows:
        name = r["name"].strip()
        smi = r["smiles"].strip()
        sdf_path = SDF_DIR / f"{name}.sdf"
        pdbqt_path = SDF_DIR / f"{name}.pdbqt"

        mol = build_3d(smi, name)
        if mol is None:
            print(f"  {name:12s}  FAIL — RDKit could not embed: {smi}")
            summary.append((name, "FAIL_RDKIT", smi, "", ""))
            continue
        write_sdf(mol, sdf_path)
        ok, msg = sdf_to_pdbqt(sdf_path, pdbqt_path)
        if not ok:
            print(f"  {name:12s}  FAIL Meeko: {msg.splitlines()[-1] if msg else ''}")
            summary.append((name, "FAIL_MEEKO", smi, str(sdf_path), msg))
            continue
        n_atoms = mol.GetNumHeavyAtoms()
        n_rotbond = AllChem.CalcNumRotatableBonds(mol)
        print(f"  {name:12s}  OK  heavy={n_atoms}  rotbonds={n_rotbond}  -> {pdbqt_path.name}")
        summary.append((name, "OK", smi, str(sdf_path), str(pdbqt_path)))
        n_ok += 1

    print()
    print(f"Successfully prepared: {n_ok}/{len(rows)}")
    print(f"Output dir: {SDF_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
