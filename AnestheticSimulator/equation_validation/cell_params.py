"""Centralized parameter extraction for the 4 Wave 2 production cells.

Reads the canonical g_NS / surf / E_X constants from the Wave 2 cell-builder
modules without modifying them. Read-only access.

All cells use Nicoletti et al. parameterizations from the published AVAL /
AVAR / AIY / RIM whole-cell electrophysiology models.

Capacitance correction note (per Session 1's WB2 capacitance audit):
specific_cm × surf_cm² gives total cell capacitance in pF. The Path A
power-balance and bifurcation analyses use total capacitance in F (not
specific cm).
"""
from __future__ import annotations

from pathlib import Path

# Nicoletti AVAL parameters (read directly from option_alpha_ava_cell.py)
AVAL = {
    "name": "AVAL",
    "surf_cm2": 1123.84e-8,
    "cm_uFcm2": 0.859551,
    "e_leak_mV": -39.0,
    "e_K_mV": -80.0,
    "e_Ca_mV": 60.0,
    "e_Na_mV": 50.0,        # nominal; AVA has no Na channel but Nernst-bound check uses it
    "g_nS": {
        "egl19": 0.104385,   # L-type Ca channel
        "leak": 0.150164,
        "irk": 0.1,           # inward-rectifier K
        "nca": 0.0,           # Na leak — Nicoletti's AVAL has zero
    },
    "channel_reversal": {     # which E to use per channel
        "egl19": "e_Ca_mV",
        "leak": "e_leak_mV",
        "irk": "e_K_mV",
        "nca": "e_Na_mV",     # NCA is a Na leak channel
    },
}

AVAR = {
    "name": "AVAR",
    "surf_cm2": 1121.79e-8,
    "cm_uFcm2": 0.751761,
    "e_leak_mV": -37.0,
    "e_K_mV": -80.0,
    "e_Ca_mV": 60.0,
    "e_Na_mV": 50.0,
    "g_nS": {
        "egl19": 0.0643229,
        "leak": 0.225266,
        "irk": 0.0420709,
        "nca": 0.0493356,
        "unc103": 0.0481669,  # ERG-like K channel
    },
    "channel_reversal": {
        "egl19": "e_Ca_mV",
        "leak": "e_leak_mV",
        "irk": "e_K_mV",
        "nca": "e_Na_mV",
        "unc103": "e_K_mV",
    },
}

AIY = {
    "name": "AIY",
    "surf_cm2": 65.89e-8,
    "cm_uFcm2": 1.6,
    "e_leak_mV": -89.57,
    "e_K_mV": -80.0,
    "e_Ca_mV": 127.59,        # F18 finding — AIY's distinct E_Ca via dual-USEION
    "e_Na_mV": 50.0,
    "g_nS": {                 # raw nS values per Nicoletti
        "leak": 0.014,
        "slo1iso": 0.1,
        "kqt1": 0.02,
        "egl19": 0.01,
        "slo1egl19": 0.092,
        "nca": 0.006,
        "shl1": 0.05,
    },
    "channel_reversal": {
        "leak": "e_leak_mV",
        "slo1iso": "e_K_mV",
        "kqt1": "e_K_mV",
        "egl19": "e_Ca_mV",
        "slo1egl19": "e_K_mV",
        "nca": "e_Na_mV",
        "shl1": "e_K_mV",
    },
}

RIM = {
    "name": "RIM",
    "surf_cm2": 103.34e-8,
    "cm_uFcm2": 1.5,
    "e_leak_mV": -50.0,
    "e_K_mV": -80.0,
    "e_Ca_mV": 60.0,
    "e_Na_mV": 50.0,
    # RIM g already in S/cm² in Nicoletti; convert to nS by multiplying by surf
    "g_Scm2": {
        "shl1": 1.518e-3,
        "egl2": 1.518e-4,
        "irk": 3.035e-4,
        "cca1": 9.677e-6,
        "unc2": 9.677e-5,
        "egl19": 9.677e-6,
        "leak": 9.677e-5,
    },
    "channel_reversal": {
        "shl1": "e_K_mV",
        "egl2": "e_K_mV",
        "irk": "e_K_mV",
        "cca1": "e_Ca_mV",
        "unc2": "e_Ca_mV",
        "egl19": "e_Ca_mV",
        "leak": "e_leak_mV",
    },
}

ALL_CELLS = {"AVAL": AVAL, "AVAR": AVAR, "AIY": AIY, "RIM": RIM}


def total_capacitance_pF(cell: dict) -> float:
    """C_total (pF) = cm (μF/cm²) × surf (cm²) × 1e6 (μF→pF)."""
    return cell["cm_uFcm2"] * cell["surf_cm2"] * 1e6


def conductance_nS(cell: dict, channel: str) -> float:
    """Per-channel conductance in nS regardless of whether cell uses g_nS or g_Scm2."""
    if "g_nS" in cell and channel in cell["g_nS"]:
        return cell["g_nS"][channel]
    if "g_Scm2" in cell and channel in cell["g_Scm2"]:
        return cell["g_Scm2"][channel] * cell["surf_cm2"] * 1e9  # S → nS
    raise KeyError(f"channel {channel!r} not found in cell {cell['name']}")


def all_channels(cell: dict) -> list[str]:
    if "g_nS" in cell:
        return list(cell["g_nS"].keys())
    return list(cell["g_Scm2"].keys())


def channel_reversal_mV(cell: dict, channel: str) -> float:
    rev_key = cell["channel_reversal"][channel]
    return cell[rev_key]


def total_resting_g_nS(cell: dict, exclude_zero: bool = True) -> float:
    """Total conductance assuming all channels at fractional activation."""
    total = 0.0
    for ch in all_channels(cell):
        g = conductance_nS(cell, ch)
        if exclude_zero and g <= 0:
            continue
        total += g
    return total


# Sanity: print total capacitances when run directly
if __name__ == "__main__":
    for name, cell in ALL_CELLS.items():
        c_pF = total_capacitance_pF(cell)
        n_channels = sum(1 for ch in all_channels(cell)
                         if conductance_nS(cell, ch) > 0)
        print(f"{name}: C_total = {c_pF:.2f} pF, {n_channels} active channels")
