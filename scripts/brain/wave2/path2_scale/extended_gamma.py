"""
Extended single-channel γ inventory for full 300-cell substrate scaling.

Phase 2 covered 9 channels for the 4 production-grade cells (AVAL, AVAR,
AIY, RIM). This module extends coverage to all channels in CeNGEN-relevant
neuron classes (~24 channels).

All γ values from mammalian homolog literature with documented epistemic
label ("approximation from adjacent biology"). Physiological-Ca²⁺
adjustment applied for Ca channels.
"""
from __future__ import annotations


# γ in picosiemens per channel
# Phase 2 values for 9 channels retained; v2 refinements (IRK=12, NCA=1.5)
# applied per docs/channel_gamma_inventory.md updates.
EXTENDED_GAMMA_PS: dict[str, float] = {
    # ===== Ca channels (physiological Ca²⁺ adjusted from Ba²⁺ literature) =====
    "EGL-19":  6.0,    # Cav1.2 L-type
    "CCA-1":   3.0,    # Cav3.x T-type (smallest)
    "UNC-2":   5.0,    # Cav2.1 P/Q-type
    # ===== Voltage-gated K — Phase 2 covered =====
    "IRK":    12.0,    # Kir2.x effective at physiological V (v2 refit)
    "KQT-1":   3.0,    # KCNQ family
    "SHL-1":   6.0,    # Kv4.2 with auxiliary
    "EGL-2":   8.0,    # Kv10.1/Eag1
    "UNC-103": 2.0,    # hERG (physiological [K]_out=4)
    # ===== Voltage-gated K — extended =====
    "SHK-1":  10.0,    # Kv1 Shaker-family; canonical Shaker single-channel ~10-18 pS
    "EGL-36": 16.0,    # Kv3.x Shaw-family; canonical ~16-25 pS
    "KVS-1":  16.0,    # Kv3.x family
    "EXP-2":  67.0,    # **DIRECT C. elegans measurement** — Davis lab single-channel γ=67±2 pS
    "KQT-2":   3.0,    # KCNQ family (same as KQT-1)
    "KQT-3":   3.0,    # KCNQ family
    # ===== Ca-activated K =====
    "SLO-1": 200.0,    # BK channel (canonical large conductance ~100-300 pS)
    "SLO-2":  20.0,    # SK/IK family (~10-30 pS)
    # ===== Non-specific cation / NALCN family =====
    "NCA":     1.5,    # v2 calibration retained. Tried γ=3 with min-stoich AND
                       # with AF-weighted accessory (UNC-79 obligate, weighted
                       # 0.623). Both crashed network: any meaningful boost
                       # cascades. Substrate stability requires γ=1.5. The AF-
                       # derived insight (UNC-79 obligate) is encoded in the
                       # accessory_factor formula even at this gamma.
    # ===== K2P leak family (TWK) =====
    "TWK":    40.0,    # K2P family canonical (TWK channels are leak K2P)
    # ===== HCN (cng-1/cng-2/cng-3 + tax-2/tax-4 in C. elegans) =====
    "HCN":     5.0,    # mammalian HCN1 single-channel ~1-5 pS
    # ===== DEG/ENaC family — UNC-8, DEL-1/2/3, ASIC-1/2, DEG-1, ACD-3 =====
    "DEGENAC": 1.0,    # Lowered from 5pS canonical to 1pS to fit substrate.
                       # ENaC literature γ 5-10 pS but cumulative TPMs 200-600
                       # per plateau cell × 5pS × C_global crashed network at
                       # the cell-intrinsic level (not synaptic). 1 pS gives
                       # gbar comparable to NCA (~5e-6 S/cm²) — safer.
    # ===== I_NaP (persistent Na — bootstrap drive for plateau cells) =====
    # Mammalian Nav1.6 persistent component ~5-10 pS. We use γ-bookkeeping
    # entry mainly for symmetry; actual gbar assignment in scalable_builder
    # is UNIFORM (not γ × TPM × C_global) because nav-1 is below CeNGEN T2
    # threshold and per-gene TPM scaling is not available.
    "NAP":     5.0,
}


# Mapping from CeNGEN gene names to channel-module names + aggregation rules
GENE_TO_CHANNEL: dict[str, str] = {
    # Single-gene channels
    "egl-19":  "EGL-19",
    "cca-1":   "CCA-1",
    "unc-2":   "UNC-2",
    "shl-1":   "SHL-1",
    "shk-1":   "SHK-1",
    "egl-36":  "EGL-36",
    "kvs-1":   "KVS-1",
    "exp-2":   "EXP-2",
    "egl-2":   "EGL-2",
    "unc-103": "UNC-103",
    "kqt-1":   "KQT-1",
    "kqt-2":   "KQT-2",
    "kqt-3":   "KQT-3",
    "slo-1":   "SLO-1",
    "slo-2":   "SLO-2",
    # Paralog families: each gene maps to a family channel (aggregated at usage)
    "irk-1":   "IRK", "irk-2": "IRK", "irk-3": "IRK",
    "nca-2":   "NCA",
    # K2P family — each twk gene contributes; treated as separate channels by gene
    "twk-7":   "TWK", "twk-18": "TWK", "twk-30": "TWK", "twk-40": "TWK",
}


def get_gamma_pS(channel: str) -> float:
    """Return γ in pS for a channel name."""
    if channel not in EXTENDED_GAMMA_PS:
        raise KeyError(f"Unknown channel {channel!r}; known: {sorted(EXTENDED_GAMMA_PS)}")
    return EXTENDED_GAMMA_PS[channel]


def get_gamma_S(channel: str) -> float:
    """Return γ in S/channel."""
    return get_gamma_pS(channel) * 1e-12
