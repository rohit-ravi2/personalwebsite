"""
Layer 1 §7.2 v2 — KCC-2 (Payne 1997 thermodynamic) + ABTS-1 (approximate)
as separate Cl-extrusion components.

Per Rohit's 2026-05-12 Path (b) Option (ii) authorization. Replaces v1's
single-Michaelis-Menten "lumped" abstraction (which had no thermodynamic
equilibrium and produced unphysical Cl drift) with two separate components:

- **KCC-2** (kcc-2 gene): K-Cl symport. **Payne 1997 thermodynamic form**
  with built-in equilibrium at [K]_in · [Cl]_in = [K]_out · [Cl]_out.
  Electroneutral: 1 K+ out + 1 Cl- out per cycle, net 0 charge moved.
  Bounded-driving-force functional form:
      v_KCC2 = I_KCC2_max · (P_in - P_out) / (P_in + P_out)
  where P_in = [K]_in · [Cl]_in, P_out = [K]_out · [Cl]_out. Bounded to
  [-I_max, +I_max]. At equilibrium, v_KCC2 = 0. Above equilibrium,
  v > 0 (extrudes K and Cl). Below equilibrium, v < 0 (loads K and Cl).
  Reverses sign cleanly.

- **ABTS-1** (abts-1 gene): Na-Cl/HCO₃ exchange. **Approximate
  first-order relaxation form** toward target [Cl]_in (default 5 mM):
      v_ABTS = I_ABTS_max · (Cl_in - Cl_target) / Cl_target
  Linear in Cl deviation. Explicitly approximate — full thermodynamics
  requires HCO₃ and pH state (Layer 1 v2 scope). v1 limitation documented
  per §2.8 epistemic-label framing.

Per-cell scaling NOW INDIVIDUAL (not combined): I_KCC2_max scales by kcc-2
TPM; I_ABTS_max scales by abts-1 TPM. AIY's biology becomes testable per
cell: 88.5% ABTS-1 dominance means AIY's Cl dynamics depend mostly on the
approximate form, not the thermodynamic KCC-2 form. Substantive prediction
of the substrate, not a parameter failure.

Net membrane current: 0 from both (electroneutral lumped abstraction;
real ABTS-1 charge balance is via Na/HCO₃ counter-transport that's
implicit in v1).

Per-ion contributions (outward-positive convention):

    KCC-2:
        iCl_contrib = -v_KCC2        (Cl OUT  → -current for anion)
        iK_contrib  = +v_KCC2        (K  OUT  → +current for cation)

    ABTS-1 (v1 approximation, ignores Na coupling):
        iCl_contrib = -v_ABTS        (Cl OUT  → -current for anion)
        No iK or iNa contribution.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Per-cell CeNGEN TPMs (threshold 2)
# ---------------------------------------------------------------------------

KCC2_TPM_CENGEN_T2: dict[str, float] = {
    "AVAL": 598.6,
    "AVAR": 598.6,
    "AIY":   26.9,
    "RIM":  234.2,
}

ABTS1_TPM_CENGEN_T2: dict[str, float] = {
    "AVAL": 569.5,
    "AVAR": 569.5,
    "AIY":  206.5,
    "RIM":  223.9,
}

ANCHOR_CELL = "AVAL"


# Default ABTS-1 target intracellular Cl (mM)
CL_TARGET_ABTS_mM_DEFAULT = 5.0


# ---------------------------------------------------------------------------
# Brian2 equation fragments — separate KCC-2 and ABTS-1
# ---------------------------------------------------------------------------

KCC2_EQS = """
# ---- KCC-2 (Payne 1997 thermodynamic K-Cl symport) ----
kcc2_I_max_mAcm2 : 1

# Thermodynamic driving force (bounded form, ∈ [-1, +1])
# At equilibrium [K]_in·[Cl]_in == [K]_out·[Cl]_out, driving = 0 → v_KCC2 = 0
kcc2_P_in  = K_in  * Cl_in  : 1
kcc2_P_out = K_out * Cl_out : 1
kcc2_drive = (kcc2_P_in - kcc2_P_out) / (kcc2_P_in + kcc2_P_out) : 1
kcc2_v_mAcm2 = kcc2_I_max_mAcm2 * kcc2_drive : 1

# Per-ion contributions (KCC-2 electroneutral: K out + Cl out per cycle)
kcc2_iCl_mAcm2 = -kcc2_v_mAcm2 : 1
kcc2_iK_mAcm2  = +kcc2_v_mAcm2 : 1
"""


ABTS1_EQS = """
# ---- ABTS-1 (v1 approximate first-order relaxation toward Cl target) ----
# Explicit v1 approximation: real ABTS-1 is Na/HCO3-coupled with thermodynamic
# equilibrium depending on Na gradient + HCO3/pH state (Layer 1 v2 scope).
abts1_I_max_mAcm2 : 1
abts1_Cl_target_mM : 1

# v_ABTS > 0 above target (extrudes); < 0 below target (loads — biologically
# wrong but v1 approximation)
abts1_v_mAcm2 = abts1_I_max_mAcm2 * (Cl_in - abts1_Cl_target_mM) / abts1_Cl_target_mM : 1

# Per-ion contribution (v1: Cl only; Na coupling deferred to v2)
abts1_iCl_mAcm2 = -abts1_v_mAcm2 : 1
"""


# Combined fragment for cell builders that want both components
LUMPED_CL_EXTRUDER_EQS = KCC2_EQS + ABTS1_EQS


# ---------------------------------------------------------------------------
# Parameter application + cross-cell scaling
# ---------------------------------------------------------------------------

def apply_kcc2_params(group, I_max_mAcm2: float) -> None:
    """Apply KCC-2 Payne-thermodynamic parameters."""
    group.kcc2_I_max_mAcm2 = I_max_mAcm2


def apply_abts1_params(
    group,
    I_max_mAcm2: float,
    Cl_target_mM: float = CL_TARGET_ABTS_mM_DEFAULT,
) -> None:
    """Apply ABTS-1 approximate parameters."""
    group.abts1_I_max_mAcm2 = I_max_mAcm2
    group.abts1_Cl_target_mM = Cl_target_mM


def scale_I_max_by_kcc2_tpm(I_max_anchor_mAcm2: float, cell_name: str) -> float:
    """Scale KCC-2 I_max from AVAL anchor by per-cell kcc-2 TPM ratio."""
    if cell_name not in KCC2_TPM_CENGEN_T2:
        raise KeyError(
            f"Unknown cell {cell_name!r}; known: {sorted(KCC2_TPM_CENGEN_T2)}"
        )
    return I_max_anchor_mAcm2 * (
        KCC2_TPM_CENGEN_T2[cell_name] / KCC2_TPM_CENGEN_T2[ANCHOR_CELL]
    )


def scale_I_max_by_abts1_tpm(I_max_anchor_mAcm2: float, cell_name: str) -> float:
    """Scale ABTS-1 I_max from AVAL anchor by per-cell abts-1 TPM ratio."""
    if cell_name not in ABTS1_TPM_CENGEN_T2:
        raise KeyError(
            f"Unknown cell {cell_name!r}; known: {sorted(ABTS1_TPM_CENGEN_T2)}"
        )
    return I_max_anchor_mAcm2 * (
        ABTS1_TPM_CENGEN_T2[cell_name] / ABTS1_TPM_CENGEN_T2[ANCHOR_CELL]
    )


# ---------------------------------------------------------------------------
# Helpers retained for backward compatibility (no longer load-bearing under v2)
# ---------------------------------------------------------------------------

def get_combined_tpm(cell_name: str) -> float:
    return KCC2_TPM_CENGEN_T2[cell_name] + ABTS1_TPM_CENGEN_T2[cell_name]


def get_kcc2_fraction(cell_name: str) -> float:
    """f_KCC2 = kcc-2 / (kcc-2 + abts-1); ∈ [0,1]."""
    total = get_combined_tpm(cell_name)
    return KCC2_TPM_CENGEN_T2[cell_name] / total
