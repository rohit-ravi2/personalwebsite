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
"""**v1 value — superseded by per-family C_GLOBAL in v2 (see C_GLOBAL_PER_FAMILY below).**
Calibrated 2026-05-12 from EGL-19 in AVAL reference (Nicoletti gbar anchor).
Failed Phase 5 (75% beyond 5×) and Phase 6 (0/4 cells pass) under v1
single-anchor approach. Retained for backward compatibility; new code
should use C_GLOBAL_PER_FAMILY."""

# v2 per-cell-family C_global (calibrated against V_rest measurements per
# §8.11 measurement-vs-fit audit; see docs/v_rest_targets.md and
# docs/c_global_per_family_calibration.md)
C_GLOBAL_PER_FAMILY: dict[str, float] = {
    "AVA": 1.0e4,      # v2 calibrated 2026-05-12: V_rest=-47.7 mV (target range [-50,-15], central -32)
    "AIY": 1.0e4,      # v2 calibrated 2026-05-12: V_rest=-85.4 mV (target range [-95,-55], central -75)
    "RIM": 1.0e4,      # v2 CALIBRATION FAILED: V_rest plateaus at -12 mV (target [-65,-40]); substrate-level pump+leak issue surfaces independent of C_global; documented substantive finding for v3
}
"""Per-cell-family C_global values. Calibrated against measured V_rest per
§3.0 v2 methodology. Updated 2026-05-12 by `calibrate_path2_v2.py`.

**AVA + AIY: calibrated successfully.** Single global value 1.0e4 happens
to satisfy both families' V_rest targets within range. This is convenient
emergence — the order-of-magnitude scan resolved both at the same
order, suggesting Layer 1 substrate's pump+leak system isn't strongly
cell-family-dependent for these two families.

**RIM: substantive finding documented.** Calibration sweep from 1e1 to
1e7 fails to produce V_rest in [-65, -40] range — RIM plateaus at
-12 mV across all C_global values from 10 to 10,000 (only depolarizes
further above 1e4). Cause: RIM's pump+leak balance from §7.2 v2
produces V_rest = -12 mV INDEPENDENT of channel parameterization
(channels at C_global = 10 contribute negligibly; cell stays at -12).
This is consistent with §7.2 v2 finding that RIM was an outlier in
pump-leak balance under linear-TPM-density assumption. RIM remains in
v2 deployment with documented Tier B failure; Phase 5/6 v2 validation
will surface its specific failure pattern. v3 candidate refinement:
RIM-specific leak split or pump capacity adjustment beyond TPM-linear
scaling."""

CELL_FAMILY_MAPPING: dict[str, str] = {
    "AVAL": "AVA",
    "AVAR": "AVA",
    "AIY":  "AIY",
    "RIM":  "RIM",
}
"""Cell name → CeNGEN class (used for both TPM lookup and C_global family
lookup)."""

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
    "IRK":    12.0,    # v2 refit: effective γ at physiological V (was 25 pS chord at V=-100 mV; Phase 2 uncertainty band 21-34 chord / 31-43 slope; refit captures rectification at substrate-relevant V_rest)
    "KQT-1":   3.0,    # KCNQ family
    "SHL-1":   6.0,    # Kv4.2 with DPP6-like auxiliary
    "EGL-2":   8.0,    # Kv10.1/Eag1
    "UNC-103": 2.0,    # hERG at physiological [K]_out=4 mM
    # Non-specific / sodium leak
    "NCA":     1.5,    # v2 refit: lower-end of Phase 2 1-20 pS uncertainty band (was 5 pS central placeholder; refit based on Phase 6 evidence)
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

def get_derived_gbar(channel: str, cell: str, use_v2_per_family: bool = True) -> float:
    """Compute Path 2 derived intensive gbar (S/cm²) for a channel in a cell.

    Args:
        channel: channel name (must be in GAMMA_PS dict)
        cell: cell name (must be in CELL_TO_CENGEN_CLASS — AVAL/AVAR/AIY/RIM)
        use_v2_per_family: if True (default), use C_GLOBAL_PER_FAMILY (v2);
                          if False, use C_GLOBAL_CHANNELS_PER_CM2_PER_TPM (v1)

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
    if use_v2_per_family:
        c_global = C_GLOBAL_PER_FAMILY[cengen_class]
    else:
        c_global = C_GLOBAL_CHANNELS_PER_CM2_PER_TPM
    return gamma * tpm * E_TRANSLATION_UNIFORM_V1 * c_global


def set_c_global_family(family: str, value: float) -> None:
    """Update C_GLOBAL_PER_FAMILY value for a cell family.
    Used by calibration sweep (Deliverable 4, Group C)."""
    if family not in C_GLOBAL_PER_FAMILY:
        raise KeyError(f"Unknown family {family!r}; known: {sorted(C_GLOBAL_PER_FAMILY)}")
    C_GLOBAL_PER_FAMILY[family] = value


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
