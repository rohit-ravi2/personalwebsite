#!/usr/bin/env bash
# Wave P — local setup script
# Status: SCAFFOLDED. Not yet executed.
#
# This script creates three isolated venvs for Wave P phases and installs
# the necessary tools. It does NOT modify the production conda env at
# ~/miniconda3/envs/ml/.
#
# Usage:
#   bash setup_local.sh --phase A          # set up Phase A only
#   bash setup_local.sh --phase B          # Phase B only
#   bash setup_local.sh --phase D          # Phase D only (heaviest install)
#   bash setup_local.sh --all              # all phases
#
# Pre-requisites:
#   - Python 3.10+ available as `python3`
#   - CUDA 12.x for JAX / OpenMM GPU acceleration
#   - 30+ GB free disk for tools + databases

set -euo pipefail

PHASE="${1:-}"
WAVE_P_ROOT="/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator"
VENV_BASE="${HOME}/venvs"

mkdir -p "${VENV_BASE}"

setup_phase_a_b() {
    echo "[Wave P setup] Phase A + B docking environment"
    python3 -m venv "${VENV_BASE}/wave-p-dock"
    # shellcheck source=/dev/null
    source "${VENV_BASE}/wave-p-dock/bin/activate"
    pip install --upgrade pip
    pip install vina rdkit-pypi openbabel-wheel
    pip install biopython
    # ColabFold local install (lightweight; weights pulled separately)
    pip install colabfold
    # GNINA via conda recommended; install separately if not via conda
    echo "[Wave P setup] Phase A+B venv ready at ${VENV_BASE}/wave-p-dock"
    echo "[Wave P setup] NOTE: GNINA and DiffDock require separate conda or manual install"
    echo "[Wave P setup] See infrastructure/dependencies.md sections 1.1 and 1.2"
    deactivate
}

setup_phase_d() {
    echo "[Wave P setup] Phase D MD environment"
    python3 -m venv "${VENV_BASE}/wave-p-md"
    # shellcheck source=/dev/null
    source "${VENV_BASE}/wave-p-md/bin/activate"
    pip install --upgrade pip
    pip install openmm openmmtools mdanalysis parmed pdb-fixer
    pip install numpy scipy matplotlib
    echo "[Wave P setup] Phase D venv ready at ${VENV_BASE}/wave-p-md"
    echo "[Wave P setup] NOTE: AmberTools must be installed via conda or manual binary"
    echo "[Wave P setup]   conda create -n wave-p-amber -c conda-forge ambertools=23"
    deactivate
}

setup_phase_i() {
    echo "[Wave P setup] Phase I JAX environment"
    python3 -m venv "${VENV_BASE}/wave-p-jax"
    # shellcheck source=/dev/null
    source "${VENV_BASE}/wave-p-jax/bin/activate"
    pip install --upgrade pip
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install optax equinox brian2 numpy scipy matplotlib pandas
    echo "[Wave P setup] Phase I venv ready at ${VENV_BASE}/wave-p-jax"
    deactivate
}

download_target_structures() {
    echo "[Wave P setup] Downloading AlphaFold DB monomer structures"
    mkdir -p "${WAVE_P_ROOT}/artifacts/structures"
    # Read tier1_targets.csv, pull each AF DB monomer
    # Skip header row
    if [ ! -f "${WAVE_P_ROOT}/targets/tier1_targets.csv" ]; then
        echo "[Wave P setup] WARNING: tier1_targets.csv not found; skipping monomer pulls"
        return
    fi
    python3 - <<'PYEOF'
import csv
import os
import urllib.request
target_csv = os.environ.get("WAVE_P_ROOT", "/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator") + "/targets/tier1_targets.csv"
out_dir = os.environ.get("WAVE_P_ROOT", "/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator") + "/artifacts/structures"
os.makedirs(out_dir, exist_ok=True)
with open(target_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        gene = row.get("gene_name", "").strip()
        uniprot = row.get("uniprot_id", "").strip()
        if not uniprot or uniprot.lower() in ("", "n/a", "tbd"):
            print(f"  [skip] {gene}: no uniprot id")
            continue
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb"
        out_path = os.path.join(out_dir, f"{gene}_monomer_AFDB.pdb")
        if os.path.exists(out_path):
            print(f"  [exists] {gene}")
            continue
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"  [ok] {gene} <- {url}")
        except Exception as e:
            print(f"  [fail] {gene}: {e}")
PYEOF
}

case "${PHASE}" in
    --phase)
        shift
        case "${1:-}" in
            A|a) setup_phase_a_b ;;
            B|b) setup_phase_a_b ;;
            D|d) setup_phase_d ;;
            I|i) setup_phase_i ;;
            *) echo "Unknown phase. Use A, B, D, or I."; exit 1 ;;
        esac
        ;;
    --all)
        setup_phase_a_b
        setup_phase_d
        setup_phase_i
        download_target_structures
        ;;
    --download-structures)
        download_target_structures
        ;;
    *)
        cat <<'USAGE'
Wave P setup script.

Usage:
  bash setup_local.sh --phase A         # Phase A + B docking environment
  bash setup_local.sh --phase D         # Phase D MD environment
  bash setup_local.sh --phase I         # Phase I JAX environment
  bash setup_local.sh --all             # All environments + download AF DB monomers
  bash setup_local.sh --download-structures  # Just pull AF DB monomers

The script does NOT modify ~/miniconda3/envs/ml/ (production simulator env).

USAGE
        exit 1
        ;;
esac

echo "[Wave P setup] Done."
