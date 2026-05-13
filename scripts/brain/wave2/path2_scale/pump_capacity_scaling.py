"""
Channel-load-proportional pump scaling.

Diagnosis (diagnose_ave_kdep.py): AVAL-anchored Na/K-ATPase I_max = 2.35e-4
mA/cm² is undersized for cells with rich channel inventories (e.g., AVE has
11 channels vs AVAL's 3). Cells with more channel leak need more pump.

Approach: for each cell, compute total channel gbar sum. Scale pump I_max
in proportion to (cell_channel_load / AVAL_channel_load), bounded.

This replaces the eat-6 TPM scaling (variant C) which broke balance, and
the no-scaling baseline (variant A) which left depolarized-rich cells
under-pumped.
"""
from __future__ import annotations


# AVAL channel inventory (Nicoletti calibrated values) — anchor reference
# Total channel gbar in S/cm² for AVAL:
#   egl19 = 9.288e-6 + irk = 8.898e-6 + nca = 0.0 = 1.819e-5
AVAL_CHANNEL_LOAD_Scm2 = 9.288e-6 + 8.898e-6 + 0.0  # = 1.819e-5

# Bounds: never reduce pump below AVAL anchor (channel-load < AVAL means
# cell is "under-channeled" — still needs full pump for ion homeostasis);
# cap upward scaling at 5× to avoid extreme pump rates.
MIN_PUMP_SCALE = 1.0
MAX_PUMP_SCALE = 5.0


def channel_load_scale(channel_gbar_dict: dict[str, float]) -> float:
    """Return pump scaling factor in [MIN_PUMP_SCALE, MAX_PUMP_SCALE]
    proportional to (total channel gbar / AVAL channel load)."""
    total_gbar = sum(channel_gbar_dict.values())
    raw_scale = total_gbar / AVAL_CHANNEL_LOAD_Scm2
    return max(MIN_PUMP_SCALE, min(MAX_PUMP_SCALE, raw_scale))
