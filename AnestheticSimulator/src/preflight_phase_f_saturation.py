"""Pre-flight: Phase F sensitivity sweep to test saturation hypothesis.

Sweep block_factor (Complex I rate at 1× EC50) across [0.05..0.95] holding
GAS1_COMPLEX_I_FACTOR=0.4 fixed. Plot/print predicted gas-1 hypersensitivity
ratio. If ratio is constant ≈ 2.48 across wide range, model saturates and
predictions are not biologically informative.

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/preflight_phase_f_saturation.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import functions from phase_f
from phase_f_metabolic_layer import (
    atp_steady_state, k_atp_open_fraction, membrane_v_shift,
    predicted_anesthetic_dose_for_immobilization,
    GAS1_COMPLEX_I_FACTOR,
)

print(f"Sensitivity: block_factor (Complex I rate at 1× EC50) → predicted gas-1/WT ratio")
print(f"Holding gas-1 Complex I factor = {GAS1_COMPLEX_I_FACTOR} fixed")
print()
print(f"{'block_factor':>12s} {'block_pct':>10s} {'WT dose':>9s} {'gas1 dose':>11s} {'ratio':>9s}")
for bf in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 0.95]:
    wt_dose = predicted_anesthetic_dose_for_immobilization(1.0, bf)
    gas1_dose = predicted_anesthetic_dose_for_immobilization(GAS1_COMPLEX_I_FACTOR, bf)
    ratio = wt_dose / gas1_dose if gas1_dose > 0 and gas1_dose < float("inf") else float("nan")
    block_pct = (1 - bf) * 100
    print(f"  {bf:>10.2f} {block_pct:>10.0f}%  {wt_dose:>9.2f}  {gas1_dose:>11.2f}  {ratio:>9.3f}")

print()
print("Sensitivity: GAS1_COMPLEX_I_FACTOR (gas-1 Complex I residual rate) → predicted ratio")
print(f"Holding block_factor = 0.706 (halothane wave2 value) fixed")
print()
print(f"{'gas1_factor':>12s} {'WT dose':>9s} {'gas1 dose':>11s} {'ratio':>9s}")
for g1 in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    wt_dose = predicted_anesthetic_dose_for_immobilization(1.0, 0.706)
    gas1_dose = predicted_anesthetic_dose_for_immobilization(g1, 0.706)
    ratio = wt_dose / gas1_dose if gas1_dose > 0 and gas1_dose < float("inf") else float("nan")
    print(f"  {g1:>10.2f}  {wt_dose:>9.2f}  {gas1_dose:>11.2f}  {ratio:>9.3f}")

# Joint sweep — does clustering at ratio≈2.48 hold across the whole grid?
print()
print("Joint sweep — block_factor × gas1_factor:")
header_label = "gas1/block:"
print(f"  {header_label:>12s}", end="")
for bf in [0.10, 0.30, 0.50, 0.706, 0.85]:
    print(f" {bf:>9.2f}", end="")
print()
for g1 in [0.30, 0.40, 0.50, 0.60]:
    print(f"  {g1:>10.2f}: ", end="")
    for bf in [0.10, 0.30, 0.50, 0.706, 0.85]:
        wt_dose = predicted_anesthetic_dose_for_immobilization(1.0, bf)
        gas1_dose = predicted_anesthetic_dose_for_immobilization(g1, bf)
        if gas1_dose > 0 and gas1_dose < float("inf"):
            r = wt_dose / gas1_dose
            print(f" {r:>8.3f} ", end="")
        else:
            print(f"   inf   ", end="")
    print()
