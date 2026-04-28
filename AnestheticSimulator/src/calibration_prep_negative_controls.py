"""Stage 1b — prepare negative-control + diagnostic-pair ligands.

Negative controls are small lipophilic molecules with little or no anesthetic
activity at sub-narcotic concentrations. The discriminative test compares
their target engagement profile against anesthetics at comparable concentrations.

The cis/trans-1,2-dichloroethylene pair is the load-bearing diagnostic per
Eger 2001 (PMID 11605945, presumed): nearly-identical lipid solubility but
only cis is anesthetic. If the pipeline ranks them similarly, it is responding
to bulk lipophilicity rather than target-specific fitting.

Output: anesthetics/negative_controls/{name}.{sdf,pdbqt}

Usage:
    conda activate wave-p-docking
    python src/calibration_prep_negative_controls.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "anesthetics" / "negative_controls"
PANEL_OUT = ROOT / "anesthetics" / "negative_control_panel.csv"

# (name, smiles, type, rationale)
CONTROLS = [
    # Inert / weakly-anesthetic small molecules at sub-narcotic doses
    ("npentane", "CCCCC", "alkane",
     "n-Pentane; lipophilic alkane, anesthetic only at lethal doses (>10% atm)"),
    ("methanol", "CO", "weak_anesthetic",
     "Methanol; very weak anesthetic, requires lethal concentrations for narcosis"),
    ("dimethyl_ether", "COC", "weak_anesthetic",
     "Dimethyl ether; small ether, anesthetic-weak"),
    ("benzene", "c1ccccc1", "weak_narcotic",
     "Benzene; weak narcotic at high concentrations; not used clinically"),
    # Eger 2001 cis/trans-DCE pair (load-bearing diagnostic)
    ("trans_12_dichloroethylene", "Cl/C=C/Cl", "non_anesthetic_conformer",
     "trans-1,2-dichloroethylene; non-anesthetic conformer (Eger 2001 anchor)"),
    ("cis_12_dichloroethylene", "Cl/C=C\\Cl", "anesthetic_conformer",
     "cis-1,2-dichloroethylene; anesthetic conformer of trans-DCE; tests pipeline shape sensitivity"),
    # Hydrocarbon negative controls
    ("cyclohexane", "C1CCCCC1", "alkane",
     "Cyclohexane; saturated cyclic alkane, weak narcotic at high concentrations"),
    ("hexafluoroethane", "FC(F)(F)C(F)(F)F", "perfluoroalkane",
     "Hexafluoroethane; halogenated alkane similar to volatile anesthetics in size/halogenation but non-anesthetic per Eger 1997"),
]


def build_3d(smiles: str, name: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=43, useRandomCoords=True,
                                 useBasicKnowledge=False) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    mol.SetProp("_Name", name)
    return mol


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_rows = []
    n_ok = 0
    for name, smi, type_, rationale in CONTROLS:
        sdf = OUT_DIR / f"{name}.sdf"
        pdbqt = OUT_DIR / f"{name}.pdbqt"

        mol = build_3d(smi, name)
        if mol is None:
            print(f"  {name:30s}  FAIL — RDKit could not embed: {smi}")
            panel_rows.append({"name": name, "smiles": smi, "type": type_,
                               "rationale": rationale, "sdf": "", "pdbqt": "",
                               "status": "FAIL_RDKIT"})
            continue
        Chem.SDWriter(str(sdf)).write(mol)

        proc = subprocess.run(
            ["mk_prepare_ligand.py", "-i", str(sdf), "-o", str(pdbqt)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            print(f"  {name:30s}  FAIL Meeko: {(proc.stderr or proc.stdout)[-200:]}")
            panel_rows.append({"name": name, "smiles": smi, "type": type_,
                               "rationale": rationale, "sdf": str(sdf),
                               "pdbqt": "", "status": "FAIL_MEEKO"})
            continue
        n_heavy = mol.GetNumHeavyAtoms()
        n_rotbond = AllChem.CalcNumRotatableBonds(mol)
        print(f"  {name:30s}  OK  type={type_:25s}  heavy={n_heavy:>2d}  rotbonds={n_rotbond}  -> {pdbqt.name}")
        panel_rows.append({"name": name, "smiles": smi, "type": type_,
                           "rationale": rationale, "sdf": str(sdf),
                           "pdbqt": str(pdbqt), "status": "OK"})
        n_ok += 1

    # Write panel CSV
    fieldnames = ["name", "smiles", "type", "rationale", "sdf", "pdbqt", "status"]
    with open(PANEL_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(panel_rows)

    print()
    print(f"Successfully prepared: {n_ok}/{len(CONTROLS)}")
    print(f"Panel CSV: {PANEL_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
