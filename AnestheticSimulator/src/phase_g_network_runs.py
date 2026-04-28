#!/usr/bin/env python3
"""
Phase G — Network-level perturbation runs.

Status: SCAFFOLDED. Implementation pending.

Purpose
-------
Run 2,400 simulations spanning 3 anesthetics x 4 doses x 4 genotypes x 5 scenarios x 10 seeds.
Plus 40 lesion runs at WT halothane 1x EC50.

Per-target lesion analysis (G.2.0 vs G.2.1-G.2.7) is the LOAD-BEARING test of the
multi-target framing at the network level.

Inputs
------
- artifacts/occupancy/occupancy_matrix.npz (Phase C)
- artifacts/kinetics/anesthetic_kinetic_shifts.npz (Phase D)
- artifacts/markov/markov_synapse_module.py (Phase E; importable)
- artifacts/metabolic/metabolic_layer_module.py (Phase F; importable)
- /home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels/*.py (Wave 2)
- /home/rohit/Desktop/C-Elegans/New Notebooks/data_derived/connectome_adult.npz (notebook pipeline)

Outputs
-------
- artifacts/runs/<anesthetic>_<dose>_<genotype>_<scenario>_<seed>.npz (2,440 files)
- artifacts/runs/aggregated_ec50.csv
- artifacts/runs/lesion_comparison.csv
- artifacts/runs/lesion_test_result.md
- artifacts/runs/dose_response_curves.png
- artifacts/runs/phase_g_completion.md

Reference: preregistration/phase_g_network_perturbation.md
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "artifacts" / "runs"
LOG_DIR = ROOT / "artifacts" / "logs"

WAVE2_CHANNELS_DIR = Path("/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels")
NB_DATA_DIR = Path("/home/rohit/Desktop/C-Elegans/New Notebooks/data_derived")

GRID = {
    "anesthetics": ["halothane", "isoflurane", "propofol"],
    "doses": [0.5, 1.0, 2.0, 5.0],
    "genotypes": ["WT", "gas1", "unc79", "unc13"],
    "scenarios": ["spontaneous", "touch", "food", "osmotic", "NaCl"],
    "seeds": list(range(10)),
}

LESIONS = ["full", "GABA", "NCA", "K2P", "SNARE", "complexI", "GluCl", "nAChR"]


def setup_logger(verbose: bool = False) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"phase_g_{date.today().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("phase_g")


def grid_size() -> tuple[int, int]:
    main = (len(GRID["anesthetics"]) * len(GRID["doses"])
            * len(GRID["genotypes"]) * len(GRID["scenarios"]) * len(GRID["seeds"]))
    lesion = len(LESIONS) * 5  # 5 seeds per lesion
    return main, lesion


def integration_check(log: logging.Logger) -> dict:
    """Verify all upstream artifacts exist before starting the grid."""
    needed = {
        "Phase C occupancy": ROOT / "artifacts" / "occupancy" / "occupancy_matrix.npz",
        "Phase D kinetics": ROOT / "artifacts" / "kinetics" / "anesthetic_kinetic_shifts.npz",
        "Phase E markov": ROOT / "artifacts" / "markov" / "markov_synapse_module.py",
        "Phase F metabolic": ROOT / "artifacts" / "metabolic" / "metabolic_layer_module.py",
        "Wave 2 channels dir": WAVE2_CHANNELS_DIR,
        "Notebook connectome": NB_DATA_DIR / "connectome_adult.npz",
    }
    status = {}
    for name, path in needed.items():
        ok = path.exists()
        status[name] = {"path": str(path), "exists": ok}
        if not ok:
            log.warning("[%s] missing: %s", name, path)
    return status


def run_one(anesthetic: str, dose: float, genotype: str, scenario: str,
            seed: int, lesion: str = "full", out_dir: Path | None = None,
            log: logging.Logger | None = None) -> bool:
    """SCAFFOLD: would execute a single Brian2 run end-to-end."""
    if log:
        log.warning("run_one is SCAFFOLD; would simulate %s/%sx/%s/%s/seed%d/lesion=%s",
                    anesthetic, dose, genotype, scenario, seed, lesion)
    return False


def gate_g1_evaluation(log: logging.Logger) -> dict:
    log.warning("Gate G.1 evaluation is SCAFFOLD")
    return {
        "G.1.1_WT_EC50_within_2x": "PENDING",
        "G.1.2_gas1_hypersensitivity": "PENDING",
        "G.1.3_unc79_resistance": "PENDING",
        "G.1.4_unc13_hypersensitivity": "PENDING",
        "G.1.5_lesion_test_load_bearing": "PENDING",
        "overall": "PENDING",
    }


def run(args: argparse.Namespace, log: logging.Logger) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    main, lesion = grid_size()
    log.info("Grid: %d main runs + %d lesion runs = %d total", main, lesion, main + lesion)

    if args.integration_check:
        status = integration_check(log)
        with open(RUN_DIR / "integration_check.json", "w") as f:
            json.dump(status, f, indent=2)
        log.info("Integration check written")
        n_missing = sum(1 for v in status.values() if not v["exists"])
        log.info("Missing upstream artifacts: %d", n_missing)
        return 0 if n_missing == 0 else 2

    if args.dry_run:
        log.info("[dry-run] would execute %d runs (~%d hours at ~2 min/run)",
                 main + lesion, (main + lesion) * 2 / 60)
        return 0

    if args.run_grid:
        log.error("Full grid execution not implemented in scaffold. "
                  "See preregistration/phase_g_network_perturbation.md section 3.")
        return 2

    if args.run_lesion:
        log.error("Lesion sub-grid execution not implemented in scaffold.")
        return 2

    if args.gate_evaluation:
        verdict = gate_g1_evaluation(log)
        with open(RUN_DIR / "gate_g1_evaluation.json", "w") as f:
            json.dump(verdict, f, indent=2)
        log.info("Gate G.1 evaluation written (scaffold)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase G network perturbation runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--integration-check", action="store_true",
                        help="Check that all upstream Phase C/D/E/F + Wave 2 + nb artifacts exist")
    parser.add_argument("--run-grid", action="store_true",
                        help="Execute full 2,400-run grid (not implemented in scaffold)")
    parser.add_argument("--run-lesion", action="store_true",
                        help="Execute 40 lesion runs (not implemented in scaffold)")
    parser.add_argument("--gate-evaluation", action="store_true")
    args = parser.parse_args()
    log = setup_logger(args.verbose)

    print("PHASE G SCAFFOLD — implementation pending — see preregistration/phase_g_network_perturbation.md")

    if not any([args.dry_run, args.integration_check, args.run_grid,
                args.run_lesion, args.gate_evaluation]):
        parser.print_help()
        return 0

    return run(args, log)


if __name__ == "__main__":
    sys.exit(main())
