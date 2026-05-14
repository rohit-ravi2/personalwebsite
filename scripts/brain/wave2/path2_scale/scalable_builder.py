"""
Scalable Path 2 cell builder — full 128-neuron-class substrate.

Builds Brian2 cells from CeNGEN TPM data + extended γ inventory + Layer 1
substrate machinery (§7.1 ion dynamics + §7.2 v2 pumps). No per-cell
Nicoletti fits required.

**Coverage limitation:** Uses the 11 channels with existing NMODL/Brian2
modules (EGL-19, CCA-1, UNC-2, IRK, KQT-1, SHL-1, EGL-2, UNC-103, NCA +
SLO-1 via slo1_iso). Cells primarily expressing SLO-2, EGL-36, KVS-1,
EXP-2, or TWK family will have those channels SKIPPED — those need
Layer 2 kinetic-audit work to add NMODL stubs.

**Surface area:** Per-cell capacitance from Nicoletti for 4 production
cells (AVAL, AVAR, AIY, RIM); default 1.0 pF + 1.0 μF/cm² specific Cm
for the other 124 classes (= 100 μm² surface). Refine when NeuroMorpho/
WormAtlas cell-specific data integrated.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from path2_scale.cengen_tpm_data import CENGEN_T2_TPM, CENGEN_NEURONS
from path2_scale.extended_gamma import EXTENDED_GAMMA_PS, GENE_TO_CHANNEL

# Reuse existing Layer 1 cell infrastructure
from layer1_cells import CellSpec, build_layer1_cell

# Extend pump TPM dicts to all 128 CeNGEN classes (no-op after first import)
from path2_scale.pump_scaling import extend_pump_dicts
extend_pump_dicts()

from path2_scale.pump_capacity_scaling import channel_load_scale


# Channels with existing NMODL/Brian2 implementations in channels/ directory
SUPPORTED_CHANNELS = {"EGL-19", "CCA-1", "UNC-2", "IRK", "KQT-1", "SHL-1",
                     "EGL-2", "UNC-103", "NCA", "EXP-2", "SHK-1", "TWK",
                     "SLO-2", "EGL-36", "KVS-1"}
# SLO-1 has slo1_iso module but requires Ca pool integration — defer
UNSUPPORTED_CHANNELS = {"SLO-1", "KQT-2", "KQT-3"}


# Per-class capacitance + e_leak (Nicoletti where available; defaults otherwise)
NICOLETTI_CELLS = {
    "AVAL": {"cm_pF": 9.66, "cm_specific_uFcm2": 0.86, "e_leak_mV": -39.0, "v_init_mV": -39.0},
    "AVAR": {"cm_pF": 8.43, "cm_specific_uFcm2": 0.75, "e_leak_mV": -37.0, "v_init_mV": -37.0},
    "AIY":  {"cm_pF": 1.05, "cm_specific_uFcm2": 1.6,  "e_leak_mV": -89.57, "v_init_mV": -89.57},
    "RIM":  {"cm_pF": 1.55, "cm_specific_uFcm2": 1.5,  "e_leak_mV": -50.0, "v_init_mV": -50.0},
}

# Default substrate parameters for cells without Nicoletti data
DEFAULT_CM_PF = 1.0  # ~100 μm² surface area
DEFAULT_CM_SPECIFIC = 1.0  # standard biological membrane
DEFAULT_E_LEAK_MV = -60.0  # mid-range estimate
DEFAULT_V_INIT = -60.0
DEFAULT_G_LEAK_SCM2 = 1.0e-5  # within range of Nicoletti's 4 cells

# Path 2 v2 C_global per cell family (from §7.3.5 v2 calibration)
# For cells outside the 3 calibrated families, use AVA-class C_global as default
C_GLOBAL_DEFAULT = 1.0e4


@dataclass
class ScalableCellSpec:
    """Cell specification for scalable Path 2 builder.

    Stripped-down spec — replaces hardcoded layer1_cells.CellSpec with
    everything derivable from CeNGEN + biophysics + minimal cell metadata.
    """
    cengen_class: str  # AVA, AIY, ASEL, etc.
    name: str          # AVAL, AVAR (may differ from cengen_class for L/R splits)
    cm_pF: float
    cm_specific_uFcm2: float
    surf_cm2: float    # derived: cm_pF * 1e-12 / (cm_specific_uFcm2 * 1e-6)
    e_leak_mV: float
    v_init_mV: float
    g_leak_Scm2: float
    channels: dict[str, float]   # {channel_module_name: gbar_Scm2}
    rest_published_mV: tuple[float, float] = (-100.0, -10.0)  # wide default
    nicoletti_calibrated: bool = False


def build_scalable_spec(cengen_class: str, cell_name: Optional[str] = None,
                        c_global: float = C_GLOBAL_DEFAULT) -> ScalableCellSpec:
    """Build a Path 2 cell spec for any CeNGEN-covered neuron class.

    Args:
        cengen_class: CeNGEN neuron class name (e.g., 'AVA', 'AIY', 'ASEL')
        cell_name: optional specific cell name (e.g., 'AVAL' for L/R split);
                  defaults to cengen_class
        c_global: C_global value to use (default: AVA-class anchor 1.0e4)

    Returns:
        ScalableCellSpec with all parameters derived from CeNGEN + biophysics.
        Channels expressed above CeNGEN T2 threshold and supported by NMODL
        modules are included; unsupported channels are documented but skipped.
    """
    if cell_name is None:
        cell_name = cengen_class
    if cengen_class not in CENGEN_NEURONS:
        raise KeyError(f"Unknown CeNGEN class {cengen_class!r}; not in {len(CENGEN_NEURONS)} known classes")

    # Cell-specific metadata or defaults
    nicoletti_meta = NICOLETTI_CELLS.get(cell_name) or NICOLETTI_CELLS.get(cengen_class)
    if nicoletti_meta:
        cm_pF = nicoletti_meta["cm_pF"]
        cm_specific = nicoletti_meta["cm_specific_uFcm2"]
        e_leak = nicoletti_meta["e_leak_mV"]
        v_init = nicoletti_meta["v_init_mV"]
        nicoletti_calibrated = True
    else:
        cm_pF = DEFAULT_CM_PF
        cm_specific = DEFAULT_CM_SPECIFIC
        e_leak = DEFAULT_E_LEAK_MV
        v_init = DEFAULT_V_INIT
        nicoletti_calibrated = False

    surf_cm2 = cm_pF * 1e-12 / (cm_specific * 1e-6)

    # Pull channel inventory from CeNGEN; aggregate paralogs
    channel_gbar: dict[str, float] = {}
    skipped_channels: set[str] = set()

    # Aggregate IRK paralogs (sum)
    irk_tpm = sum(CENGEN_T2_TPM.get(g, {}).get(cengen_class, 0.0)
                  for g in ("irk-1", "irk-2", "irk-3"))
    if irk_tpm > 0:
        gamma_s = EXTENDED_GAMMA_PS["IRK"] * 1e-12
        gbar = gamma_s * irk_tpm * 1.0 * c_global
        channel_gbar["irk"] = gbar

    # Aggregate TWK paralogs (sum across twk-7/18/30/40 — K2P family)
    twk_tpm = sum(CENGEN_T2_TPM.get(g, {}).get(cengen_class, 0.0)
                  for g in ("twk-7", "twk-18", "twk-30", "twk-40"))
    if twk_tpm > 0:
        gamma_s = EXTENDED_GAMMA_PS["TWK"] * 1e-12
        gbar = gamma_s * twk_tpm * 1.0 * c_global
        channel_gbar["twk"] = gbar

    # Single-gene channels
    for gene, ch_mod_name in [
        ("egl-19", "egl19"), ("cca-1", "cca1"), ("unc-2", "unc2"),
        ("shl-1", "shl1"), ("egl-2", "egl2"), ("unc-103", "unc103"),
        ("kqt-1", "kqt1"), ("exp-2", "exp2"), ("shk-1", "shk1"),
        ("slo-2", "slo2"), ("egl-36", "egl36"), ("kvs-1", "kvs1"),
    ]:
        tpm = CENGEN_T2_TPM.get(gene, {}).get(cengen_class, 0.0)
        if tpm > 0:
            channel_name = GENE_TO_CHANNEL[gene]
            gamma_s = EXTENDED_GAMMA_PS[channel_name] * 1e-12
            gbar = gamma_s * tpm * 1.0 * c_global
            channel_gbar[ch_mod_name] = gbar

    # NCA (use nca-2 alone — nca-1 below T2 threshold)
    nca_tpm = CENGEN_T2_TPM.get("nca-2", {}).get(cengen_class, 0.0)
    if nca_tpm > 0:
        gamma_s = EXTENDED_GAMMA_PS["NCA"] * 1e-12
        gbar = gamma_s * nca_tpm * 1.0 * c_global
        channel_gbar["nca"] = gbar

    # Channels we'd want but don't have NMODL modules — document
    for gene_check in ("slo-1",):
        if CENGEN_T2_TPM.get(gene_check, {}).get(cengen_class, 0.0) > 0:
            skipped_channels.add(GENE_TO_CHANNEL[gene_check])

    return ScalableCellSpec(
        cengen_class=cengen_class,
        name=cell_name,
        cm_pF=cm_pF,
        cm_specific_uFcm2=cm_specific,
        surf_cm2=surf_cm2,
        e_leak_mV=e_leak,
        v_init_mV=v_init,
        g_leak_Scm2=DEFAULT_G_LEAK_SCM2,
        channels=channel_gbar,
        nicoletti_calibrated=nicoletti_calibrated,
    )


def to_layer1_cellspec(s: ScalableCellSpec) -> CellSpec:
    """Convert ScalableCellSpec → layer1_cells.CellSpec for use with
    existing build_layer1_cell().

    Pump scaling (variant A, 2026-05-13): Nicoletti cells use their own
    pump TPM entries; all other cells use the AVAL anchor (no per-cell
    scaling). The TPM-ratio variant (C) was found to actively break
    AWA/ASEL/AVE substrate by scaling Na/K-ATPase down without matched
    channel scaling. Per-cell scaling will require coordinated
    C_global recalibration; deferred.
    """
    if s.name in ("AVAL", "AVAR", "AIY", "RIM"):
        pump_key = s.name
        pump_scale = 1.0  # Nicoletti cells use their own calibrated pumps
    else:
        pump_key = "AVAL"
        pump_scale = channel_load_scale(s.channels)
    return CellSpec(
        name=s.name,
        e_leak_mV=s.e_leak_mV,
        g_leak_Scm2=s.g_leak_Scm2,
        cm_uFcm2=s.cm_specific_uFcm2,
        surf_cm2=s.surf_cm2,
        channels=s.channels,
        v_init_mV=s.v_init_mV,
        pump_cell_name=pump_key,
        rest_published_mV=s.rest_published_mV,
        pump_NaK_scale=pump_scale,
    )


def list_coverage() -> dict:
    """Return summary of coverage for all 128 CeNGEN neurons."""
    summary = {
        "total_neurons": len(CENGEN_NEURONS),
        "with_nicoletti_data": 0,
        "channels_per_neuron": {},
        "fully_supported_neurons": 0,    # all expressed channels have NMODL
        "partially_supported_neurons": 0,  # some channels skipped
    }
    for n in CENGEN_NEURONS:
        spec = build_scalable_spec(n)
        n_supported = len(spec.channels)
        # Check if cell expresses any unsupported channels
        n_skipped = 0
        for gene in ("slo-1",):
            if CENGEN_T2_TPM.get(gene, {}).get(n, 0.0) > 0:
                n_skipped += 1
        summary["channels_per_neuron"][n] = {"supported": n_supported, "skipped": n_skipped}
        if spec.nicoletti_calibrated:
            summary["with_nicoletti_data"] += 1
        if n_skipped == 0:
            summary["fully_supported_neurons"] += 1
        else:
            summary["partially_supported_neurons"] += 1
    return summary


if __name__ == "__main__":
    print("=" * 72)
    print("Scalable Path 2 cell builder — coverage assessment")
    print("=" * 72)
    summary = list_coverage()
    print(f"\nTotal CeNGEN neurons: {summary['total_neurons']}")
    print(f"With Nicoletti per-cell data: {summary['with_nicoletti_data']}")
    print(f"Fully supported (all expressed channels have NMODL): {summary['fully_supported_neurons']}")
    print(f"Partially supported (some channels skipped): {summary['partially_supported_neurons']}")

    # Sample a few diverse cell types
    print(f"\n=== Sample cells (first 5) ===")
    for n in ["AVA", "AIY", "RIM", "ASEL", "AWA"]:
        try:
            spec = build_scalable_spec(n)
            print(f"\n  {n}:")
            print(f"    surf = {spec.surf_cm2*1e8:.0f} μm² (Nicoletti: {spec.nicoletti_calibrated})")
            print(f"    e_leak = {spec.e_leak_mV} mV")
            print(f"    {len(spec.channels)} supported channels:")
            for ch, g in sorted(spec.channels.items()):
                print(f"      {ch:<10} gbar = {g:.3e} S/cm²")
        except KeyError as e:
            print(f"  {n}: {e}")
