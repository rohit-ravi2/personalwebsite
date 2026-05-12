"""
Path 2 derived channel parameters — Phase 5 deliverable.

Production code for gene-expression-derived channel gbar values per
§7.3.5 Path 2 methodology. Provides:

    get_derived_gbar(channel, cell) → S/cm² (intensive)

Inputs assembled from Phase 1-4 deliverables:
- C_global  (Phase 4 calibration): channels per (cm² · TPM unit)
- γ values  (Phase 2 inventory): single-channel conductance per channel
- TPMs      (Phase 3 inventory): mRNA abundance per (channel, cell)
- E_translation = 1.0 uniform (Decision 3 v1)

Formula (Path B intensive, per methodology §2.2):

    gbar_intensive[channel][cell] = γ[channel] × density[channel][cell]
    density[channel][cell]        = TPM[channel][cell] × E_translation × C_global

**SHIP STATUS:** Phase 5 validation triggered Tier 3 HARD STOP
(75% of (channel, cell) combinations beyond 5× Nicoletti agreement).
Module ships as Path 2 v1 infrastructure for future use under whichever
architectural direction Rohit selects. Production deployment in cell
builders awaits architectural decision per
`scripts/brain/wave2/artifacts/HARD_STOP_path2_phase5.md`.

See `docs/path2_channel_validation.md` for full validation results.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Calibrated constants (Phase 4 deliverable)
# ---------------------------------------------------------------------------

C_GLOBAL_CHANNELS_PER_CM2_PER_TPM: float = 1.7297e4
"""Calibrated 2026-05-12 from EGL-19 in AVAL reference.

  C_global = gbar_Nicoletti_AVAL_EGL19 / (γ_EGL19 × TPM_EGL19_AVA × E_translation)
           = 9.288e-6 S/cm² / (6e-12 S/channel × 89.5 × 1.0)
           = 1.7297e4 channels per (cm² · TPM unit)

Biophysical sanity checks pass (max density 3.5e6 channels/cm² < 1e7
saturation; max total channels per cell 39.6 > 1 minimum). Reference
verified by construction.
"""

E_TRANSLATION_UNIFORM_V1: float = 1.0
"""Pre-authorized Decision 3 (2026-05-12). Uniform across all channels in
v1. Per-channel-family E_translation is v2 refinement candidate."""


# ---------------------------------------------------------------------------
# γ values (Phase 2 inventory; physiological conditions where applicable)
# ---------------------------------------------------------------------------

GAMMA_PS: dict[str, float] = {
    # Ca channels (Ba²⁺ literature γ adjusted to physiological Ca²⁺ ~0.25-0.33×)
    "EGL-19":  6.0,    # Cav1.2 homolog
    "CCA-1":   3.0,    # Cav3.x T-type (smallest)
    "UNC-2":   5.0,    # Cav2.1 P/Q-type
    # K channels
    "IRK":    25.0,    # Kir2.1 chord conductance V=-100 mV
    "KQT-1":   3.0,    # KCNQ family
    "SHL-1":   6.0,    # Kv4.2 with DPP6-like auxiliary
    "EGL-2":   8.0,    # Kv10.1/Eag1
    "UNC-103": 2.0,    # hERG at physiological [K]_out=4 mM
    # Non-specific / sodium leak
    "NCA":     5.0,    # NALCN — estimated (literature gap; v1 placeholder)
}

GAMMA_S_PER_CHANNEL: dict[str, float] = {k: v * 1e-12 for k, v in GAMMA_PS.items()}
"""γ in Siemens per channel (= γ_pS × 1e-12)."""


# ---------------------------------------------------------------------------
# Per-(channel, CeNGEN_cell_class) TPMs (Phase 3 inventory)
# ---------------------------------------------------------------------------

TPM_BY_CHANNEL_CELL: dict[str, dict[str, float]] = {
    # CeNGEN cell classes: AVA (= AVAL + AVAR), AIY, RIM
    # Aggregation rules per methodology §2.4:
    #   single-gene: direct TPM
    #   IRK: sum(irk-1, irk-2, irk-3)
    #   NCA: nca-2 alone (nca-1 below T2 threshold; unc-77 auxiliary excluded)
    "EGL-19":  {"AVA":  89.5, "AIY": 30.3, "RIM": 132.9},
    "CCA-1":   {"AVA": 109.3, "AIY":  0.0, "RIM":  36.3},
    "UNC-2":   {"AVA": 203.9, "AIY": 93.9, "RIM":  57.2},
    "SHL-1":   {"AVA":   0.0, "AIY":  0.0, "RIM": 153.1},  # AIY=0 is T2 false neg per §3.5
    "KQT-1":   {"AVA":   0.0, "AIY": 63.4, "RIM":   0.0},  # heteromer rejected (kqt-3/kqt-1=0%)
    "EGL-2":   {"AVA":  64.5, "AIY":  0.0, "RIM":  65.8},
    "UNC-103": {"AVA":  46.1, "AIY":  0.0, "RIM": 112.2},
    "IRK":     {"AVA": 165.6, "AIY":  0.0, "RIM": 120.3},  # sum(irk-1,2,3)
    "NCA":     {"AVA": 153.2, "AIY": 29.2, "RIM":  88.0},  # nca-2 alone
}


# Cell name → CeNGEN class mapping (AVAL and AVAR both map to CeNGEN AVA class)
CELL_TO_CENGEN_CLASS: dict[str, str] = {
    "AVAL": "AVA",
    "AVAR": "AVA",
    "AIY":  "AIY",
    "RIM":  "RIM",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_derived_gbar(channel: str, cell: str) -> float:
    """Compute Path 2 derived intensive gbar (S/cm²) for a channel in a cell.

    Args:
        channel: channel name (must be in GAMMA_PS dict)
        cell: cell name (must be in CELL_TO_CENGEN_CLASS — AVAL/AVAR/AIY/RIM)

    Returns:
        gbar_intensive (S/cm²); 0.0 if TPM is 0 (channel not expressed at
        CeNGEN T2 threshold in this cell class)

    Raises:
        KeyError if channel or cell unknown.
    """
    if channel not in GAMMA_S_PER_CHANNEL:
        raise KeyError(
            f"Unknown channel {channel!r}; known: {sorted(GAMMA_S_PER_CHANNEL)}"
        )
    if cell not in CELL_TO_CENGEN_CLASS:
        raise KeyError(
            f"Unknown cell {cell!r}; known: {sorted(CELL_TO_CENGEN_CLASS)}"
        )
    gamma = GAMMA_S_PER_CHANNEL[channel]
    cengen_class = CELL_TO_CENGEN_CLASS[cell]
    tpm = TPM_BY_CHANNEL_CELL[channel][cengen_class]
    return gamma * tpm * E_TRANSLATION_UNIFORM_V1 * C_GLOBAL_CHANNELS_PER_CM2_PER_TPM


def get_derived_density(channel: str, cell: str) -> float:
    """Channel density (channels/cm²) under Path 2 derivation."""
    cengen_class = CELL_TO_CENGEN_CLASS[cell]
    tpm = TPM_BY_CHANNEL_CELL[channel][cengen_class]
    return tpm * E_TRANSLATION_UNIFORM_V1 * C_GLOBAL_CHANNELS_PER_CM2_PER_TPM


# ---------------------------------------------------------------------------
# Smoke test + Phase 5 validation table
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Path 2 derived channel parameters — Phase 5 deliverable")
    print("=" * 72)
    print(f"\nC_global = {C_GLOBAL_CHANNELS_PER_CM2_PER_TPM:.4e} channels per (cm² · TPM)")
    print(f"E_translation = {E_TRANSLATION_UNIFORM_V1} (uniform v1)")
    print(f"\nγ inventory (Phase 2):")
    for ch, g in GAMMA_PS.items():
        print(f"  {ch:<10} {g:>5.1f} pS")

    print(f"\nDerived gbars (S/cm²) for Wave 2 cell channel sets:")
    cell_channels = {
        "AVAL": ["EGL-19", "IRK", "NCA"],
        "AVAR": ["EGL-19", "IRK", "NCA", "UNC-103"],
        "AIY":  ["EGL-19", "KQT-1", "SHL-1", "NCA"],
        "RIM":  ["EGL-19", "SHL-1", "IRK", "CCA-1", "UNC-2", "EGL-2"],
    }
    for cell, channels in cell_channels.items():
        print(f"\n  {cell}:")
        for ch in channels:
            gbar = get_derived_gbar(ch, cell)
            density = get_derived_density(ch, cell)
            print(f"    {ch:<10} gbar = {gbar:.3e} S/cm²   density = {density:.3e} channels/cm²")

    print(f"\nReference verification (AVAL EGL-19 should match Nicoletti's 9.288e-6):")
    gbar_ref = get_derived_gbar("EGL-19", "AVAL")
    print(f"  derived: {gbar_ref:.6e} S/cm²")
    nicoletti = 9.288e-6
    ratio = gbar_ref / nicoletti
    print(f"  Nicoletti: {nicoletti:.6e} S/cm²")
    print(f"  ratio: {ratio:.6f}  {'PASS (by construction)' if abs(ratio - 1.0) < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
