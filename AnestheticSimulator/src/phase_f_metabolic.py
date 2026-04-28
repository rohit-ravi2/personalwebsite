#!/usr/bin/env python3
"""
Phase F — Metabolic state layer (ATP[t] + K-ATP coupling).

Status: SCAFFOLDED. Core ODE skeleton present; full Brian2 integration pending.

Purpose
-------
Simulate per-cell ATP dynamics:
  d[ATP]/dt = production - consumption
  production = R_CI * eta + R_CII * eta + V_glycolysis * eta
  consumption = k_NaK * firing_rate + k_Ca * [Ca] + k_basal

Couple to K-ATP channel:
  P_open(ATP) = 1 / (1 + ([ATP] / K_ATP)^n)
  g_K_ATP = g_max * P_open

Apply Phase D's Complex I shift to model gas-1 hypersensitivity.

Validation
----------
- WT [ATP]_ss in 1.5-5 mM range
- gas-1 [ATP]_ss reduced 30-60% vs WT
- gas-1 EC50 leftward-shifted 1.5x-4x vs WT (Morgan & Sedensky 1995 PMID 7549290)
- mev-1 effect smaller than gas-1

Inputs
------
- artifacts/occupancy/occupancy_matrix.npz (Phase C; Complex I + II occupancies)
- artifacts/kinetics/anesthetic_kinetic_shifts.npz (Phase D; CI shift)

Outputs
-------
- artifacts/metabolic/metabolic_layer_module.py (importable Brian2 module)
- artifacts/metabolic/wt_baseline.npz
- artifacts/metabolic/gas1_baseline.npz
- artifacts/metabolic/mev1_baseline.npz
- artifacts/metabolic/atp2_baseline.npz
- artifacts/metabolic/wt_vs_gas1_halothane.npz
- artifacts/metabolic/calibration_report.md
- artifacts/metabolic/phase_f_completion.md

Reference: preregistration/phase_f_metabolic_layer.md
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MET_DIR = ROOT / "artifacts" / "metabolic"
LOG_DIR = ROOT / "artifacts" / "logs"


def setup_logger(verbose: bool = False) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"phase_f_{date.today().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("phase_f")


# Default parameters (calibration pending)
DEFAULTS = {
    "R_CI_baseline": 1.0,         # arbitrary; calibrated to give [ATP]_ss ~3 mM
    "R_CII_baseline": 0.3,
    "V_glycolysis": 0.1,
    "eta_CI": 1.0,
    "eta_CII": 0.5,
    "eta_gly": 2.0,
    "k_NaK": 0.05,                # per Hz
    "k_Ca": 0.02,                 # per uM Ca
    "k_basal": 0.1,               # per second
    "K_ATP_uM": 1000.0,           # K-ATP channel half-max ATP
    "n_ATP": 2,                   # Hill coefficient
    "g_KATP_max": 1.0,            # arbitrary nS scaled per cell
    "atp_synthase_factor_WT": 1.0,
}

GENOTYPE_MODIFIERS = {
    "WT":     {"gas1_factor": 1.0, "mev1_factor": 1.0, "atp_synthase_factor": 1.0},
    "gas1":   {"gas1_factor": 0.6, "mev1_factor": 1.0, "atp_synthase_factor": 1.0},
    "mev1":   {"gas1_factor": 1.0, "mev1_factor": 0.5, "atp_synthase_factor": 1.0},
    "atp2":   {"gas1_factor": 1.0, "mev1_factor": 1.0, "atp_synthase_factor": 0.4},
}


class MetabolicLayer:
    """SCAFFOLD class — full Brian2 ODE integration pending."""

    def __init__(self, genotype: str = "WT", occupancy_CI: float = 0.0,
                 occupancy_CII: float = 0.0, params: dict | None = None):
        self.params = {**DEFAULTS, **(params or {})}
        mod = GENOTYPE_MODIFIERS.get(genotype, GENOTYPE_MODIFIERS["WT"])
        self.gas1 = mod["gas1_factor"]
        self.mev1 = mod["mev1_factor"]
        self.atp_synth = mod["atp_synthase_factor"]
        self.occ_CI = occupancy_CI
        self.occ_CII = occupancy_CII
        self.genotype = genotype

    def production_rate(self) -> float:
        R_CI = self.params["R_CI_baseline"] * self.gas1 * (1 - self.occ_CI)
        R_CII = self.params["R_CII_baseline"] * self.mev1 * (1 - self.occ_CII)
        return ((R_CI * self.params["eta_CI"]
                 + R_CII * self.params["eta_CII"]
                 + self.params["V_glycolysis"] * self.params["eta_gly"])
                * self.atp_synth)

    def consumption_rate(self, firing_rate_Hz: float, Ca_uM: float = 0.1) -> float:
        return (self.params["k_NaK"] * firing_rate_Hz
                + self.params["k_Ca"] * Ca_uM
                + self.params["k_basal"])

    def steady_state_ATP_uM(self, firing_rate_Hz: float = 10.0,
                             Ca_uM: float = 0.1) -> float:
        """Crude steady-state ATP estimate (production / consumption)."""
        prod = self.production_rate()
        cons = self.consumption_rate(firing_rate_Hz, Ca_uM)
        # Calibrated such that WT prod/cons ~ 3000 (i.e., 3 mM)
        # Production unit calibration: prod/cons * 3000 -> uM
        if cons <= 0:
            return float("inf")
        return prod / cons * 3000.0

    def K_ATP_open_prob(self, ATP_uM: float) -> float:
        K = self.params["K_ATP_uM"]
        n = self.params["n_ATP"]
        return 1.0 / (1.0 + (ATP_uM / K) ** n)

    def predict_resting_V_shift_mV(self, ATP_uM: float,
                                    g_leak_nS: float = 0.5,
                                    E_K_mV: float = -85,
                                    V_rest_baseline_mV: float = -65) -> float:
        """Estimate hyperpolarization from K-ATP partial opening."""
        g_KATP = self.params["g_KATP_max"] * self.K_ATP_open_prob(ATP_uM)
        if g_KATP + g_leak_nS == 0:
            return 0.0
        # Weighted average of E_K (KATP) and baseline V (leak)
        V_new = (g_KATP * E_K_mV + g_leak_nS * V_rest_baseline_mV) / (g_KATP + g_leak_nS)
        return V_new - V_rest_baseline_mV


def smoke_test(log: logging.Logger) -> dict:
    """Quick sanity check: WT, gas-1, mev-1, atp-2 baseline ATP."""
    results = {}
    for geno in ["WT", "gas1", "mev1", "atp2"]:
        layer = MetabolicLayer(genotype=geno, occupancy_CI=0.0, occupancy_CII=0.0)
        atp = layer.steady_state_ATP_uM()
        v_shift = layer.predict_resting_V_shift_mV(atp)
        results[geno] = {"ATP_uM": atp, "K_ATP_P_open": layer.K_ATP_open_prob(atp),
                         "V_shift_mV": v_shift}
        log.info("[smoke] %s: ATP=%.1f uM, K-ATP P_open=%.3f, V_shift=%+.2f mV",
                 geno, atp, layer.K_ATP_open_prob(atp), v_shift)

    # Halothane overlay on WT vs gas-1 (Complex I occupancy = 0.3)
    for geno in ["WT", "gas1"]:
        layer = MetabolicLayer(genotype=geno, occupancy_CI=0.3, occupancy_CII=0.05)
        atp = layer.steady_state_ATP_uM()
        v_shift = layer.predict_resting_V_shift_mV(atp)
        key = f"{geno}_halothane_1x"
        results[key] = {"ATP_uM": atp, "K_ATP_P_open": layer.K_ATP_open_prob(atp),
                        "V_shift_mV": v_shift}
        log.info("[smoke] %s halothane 1x: ATP=%.1f uM, K-ATP=%.3f, V=%+.2f",
                 geno, atp, layer.K_ATP_open_prob(atp), v_shift)
    return results


def gate_f1_evaluation(log: logging.Logger) -> dict:
    log.warning("Gate F.1 evaluation is SCAFFOLD — uses smoke-test estimates only")
    smoke = smoke_test(log)
    wt_atp = smoke["WT"]["ATP_uM"]
    gas1_atp = smoke["gas1"]["ATP_uM"]

    f11_pass = 1500 <= wt_atp <= 5000
    f12_pass = 0.30 * wt_atp <= gas1_atp <= 0.7 * wt_atp
    return {
        "F.1.1_WT_baseline_ATP_uM": wt_atp,
        "F.1.1_pass": f11_pass,
        "F.1.2_gas1_ATP_reduction": (wt_atp - gas1_atp) / max(wt_atp, 1),
        "F.1.2_pass": f12_pass,
        "F.1.3_gas1_EC50_shift": "PENDING (requires Phase G)",
        "F.1.4_mev1_smaller_than_gas1": "PENDING",
        "overall": "PENDING (requires Phase G EC50 results)",
    }


def run(args: argparse.Namespace, log: logging.Logger) -> int:
    MET_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        log.info("[dry-run] would instantiate MetabolicLayer x 4 genotypes + halothane overlays")
        return 0

    if args.smoke_test:
        results = smoke_test(log)
        with open(MET_DIR / "smoke_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info("Smoke-test results written")

    if args.gate_evaluation:
        verdict = gate_f1_evaluation(log)
        with open(MET_DIR / "gate_f1_evaluation.json", "w") as f:
            json.dump(verdict, f, indent=2)
        log.info("Gate F.1 evaluation written (scaffold)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase F metabolic layer")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run baseline ATP estimates for WT, gas-1, mev-1, atp-2")
    parser.add_argument("--gate-evaluation", action="store_true")
    args = parser.parse_args()
    log = setup_logger(args.verbose)

    print("PHASE F SCAFFOLD — Core ODE skeleton functional; full Brian2 integration pending.")
    print("See preregistration/phase_f_metabolic_layer.md")

    if not any([args.dry_run, args.smoke_test, args.gate_evaluation]):
        parser.print_help()
        return 0

    return run(args, log)


if __name__ == "__main__":
    sys.exit(main())
