#!/usr/bin/env python3
"""
Wave P — anesthetic ligand preparation.

Reads each .smi file in anesthetics/anesthetic_smiles/, generates 3D coordinates
via RDKit, assigns AM1-BCC partial charges (via AmberTools antechamber if available;
falls back to Gasteiger), and writes both .pdbqt (for AutoDock Vina) and .sdf
(for DiffDock / GNINA / OpenMM) formats.

Status: SCAFFOLDED. Skeleton produces output via RDKit only; AM1-BCC charge
assignment requires AmberTools antechamber binary in PATH and is gated behind
--use-am1bcc flag.

Usage:
    python prepare_ligands.py --dry-run
    python prepare_ligands.py
    python prepare_ligands.py --use-am1bcc
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMI_DIR = ROOT / "anesthetic_smiles"
OUT_DIR_PDBQT = ROOT / "prepared_pdbqt"
OUT_DIR_SDF = ROOT / "prepared_sdf"
LOG_DIR = ROOT.parent / "artifacts" / "logs"


def setup_logger(verbose: bool = False) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"prepare_ligands_{date.today().isoformat()}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("prepare_ligands")


def list_smiles_files() -> list[Path]:
    return sorted(SMI_DIR.glob("*.smi"))


def read_smiles(smi_path: Path) -> tuple[str, str]:
    """Returns (smiles, name) from a .smi file."""
    with open(smi_path) as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 2:
        return parts[0], smi_path.stem
    return parts[0], parts[1]


def prepare_with_rdkit(smiles: str, name: str, out_sdf: Path, log: logging.Logger) -> bool:
    """Generate 3D conformer and write SDF using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        log.error("RDKit not installed — cannot prepare ligand %s", name)
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.error("RDKit failed to parse SMILES for %s: %s", name, smiles)
        return False

    mol_h = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    if res != 0:
        log.warning("Embed failed for %s; trying ETKDG", name)
        res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)

    writer = Chem.SDWriter(str(out_sdf))
    mol_h.SetProp("_Name", name)
    writer.write(mol_h)
    writer.close()
    log.info("Wrote SDF: %s", out_sdf)
    return True


def prepare_with_am1bcc(name: str, in_sdf: Path, out_sdf: Path, log: logging.Logger) -> bool:
    """Run AmberTools antechamber to assign AM1-BCC partial charges."""
    if shutil.which("antechamber") is None:
        log.warning("antechamber not in PATH; skipping AM1-BCC for %s", name)
        return False
    try:
        cmd = [
            "antechamber",
            "-i", str(in_sdf), "-fi", "sdf",
            "-o", str(out_sdf), "-fo", "sdf",
            "-c", "bcc",
            "-s", "2",
            "-pf", "y",
        ]
        log.info("Running antechamber: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.error("antechamber failed for %s: %s", name, result.stderr[-500:])
            return False
        log.info("AM1-BCC charges assigned: %s", out_sdf)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("antechamber exception for %s: %s", name, exc)
        return False


def sdf_to_pdbqt(sdf_path: Path, pdbqt_path: Path, log: logging.Logger) -> bool:
    """Convert SDF to PDBQT via OpenBabel."""
    if shutil.which("obabel") is None:
        log.warning("obabel not in PATH; cannot create PDBQT for %s", sdf_path.stem)
        return False
    try:
        cmd = [
            "obabel", str(sdf_path),
            "-O", str(pdbqt_path),
            "--partialcharge", "gasteiger",
            "-h",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.error("obabel failed: %s", result.stderr[-500:])
            return False
        log.info("Wrote PDBQT: %s", pdbqt_path)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("obabel exception: %s", exc)
        return False


def run(args: argparse.Namespace, log: logging.Logger) -> int:
    OUT_DIR_PDBQT.mkdir(parents=True, exist_ok=True)
    OUT_DIR_SDF.mkdir(parents=True, exist_ok=True)

    smiles_files = list_smiles_files()
    log.info("Found %d SMILES files in %s", len(smiles_files), SMI_DIR)

    if not smiles_files:
        log.error("No .smi files found at %s", SMI_DIR)
        return 1

    if args.dry_run:
        for smi in smiles_files:
            smiles, name = read_smiles(smi)
            log.info("[dry-run] would prepare %s (%s)", name, smiles)
        return 0

    n_ok = 0
    for smi in smiles_files:
        smiles, name = read_smiles(smi)
        log.info("=== Preparing %s ===", name)
        sdf_initial = OUT_DIR_SDF / f"{name}_rdkit.sdf"
        sdf_final = OUT_DIR_SDF / f"{name}.sdf"
        pdbqt_out = OUT_DIR_PDBQT / f"{name}.pdbqt"

        ok = prepare_with_rdkit(smiles, name, sdf_initial, log)
        if not ok:
            continue
        if args.use_am1bcc:
            ok = prepare_with_am1bcc(name, sdf_initial, sdf_final, log)
            if not ok:
                # Fall back to RDKit-only output as final
                shutil.copy(sdf_initial, sdf_final)
                log.warning("Using RDKit-only conformer for %s (no AM1-BCC)", name)
        else:
            shutil.copy(sdf_initial, sdf_final)

        sdf_to_pdbqt(sdf_final, pdbqt_out, log)
        n_ok += 1

    log.info("Prepared %d / %d ligands", n_ok, len(smiles_files))
    return 0 if n_ok == len(smiles_files) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Wave P anesthetic ligand prep")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files and SMILES; do not write outputs")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="DEBUG-level logging")
    parser.add_argument("--use-am1bcc", action="store_true",
                        help="Assign AM1-BCC partial charges via AmberTools antechamber")
    args = parser.parse_args()
    log = setup_logger(args.verbose)
    log.info("=== Wave P anesthetic ligand prep ===")
    log.info("SCAFFOLDED skeleton — RDKit-only path is functional; AM1-BCC requires antechamber")

    print("PHASE 0 SCAFFOLD — anesthetic ligand prep — see anesthetics/anesthetic_panel.csv")
    return run(args, log)


if __name__ == "__main__":
    sys.exit(main())
