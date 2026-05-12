"""
Layer 1 §7.2 — Na/K-ATPase pump (eat-6 in C. elegans).

Hill-kinetic three-substrate model: 3 Na out / 2 K in / 1 ATP per cycle.
Net +1 charge out per cycle → hyperpolarizing membrane current (electrogenic).

Per `docs/layer1_design_decisions.md` §4.2 + §6.3 authorizations.

Composition contract:
    `NA_K_ATPASE_EQS` declares:
        pump_NaK_I_max_mAcm2 : 1            parameter (calibrated on AVAL,
                                            scaled to other cells by eat-6 TPM)
        pump_NaK_K_Na_mM, K_K_mM, K_ATP_mM : Michaelis half-constants
        pump_NaK_n_Na, n_K, n_ATP          : Hill coefficients
        pump_NaK_ATP_mM                    : ATP (fixed in v1; dynamic from
                                            Phase F in v2 per §6.3)
        pump_NaK_f_Na, f_K, f_ATP          : computed saturation factors
        pump_NaK_I_mAcm2                   : net pump current density (mA/cm²,
                                            outward-positive)
        pump_NaK_iNa_mAcm2 = +3·I_mAcm2    : Na contribution to ion_iNa_total
        pump_NaK_iK_mAcm2  = -2·I_mAcm2    : K contribution to ion_iK_total

    The cell builder includes `pump_NaK_iNa_mAcm2` in `ion_iNa_total_mAcm2`,
    `pump_NaK_iK_mAcm2` in `ion_iK_total_mAcm2`, and `pump_NaK_I_mAcm2` as
    the net pump current in the dV/dt equation.

Reference:
    - eat-6 (α-subunit) Davis 1995 PMID 7905262
    - Mammalian Hill model: Lauger 1991 "Electrogenic Ion Pumps" Ch. 4
    - Cellular ATP usage: ~30-70% of neuronal ATP is Na/K-pump (Hodgkin 1975)
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Mammalian-default kinetic parameters (per §2.8: approximation from adjacent
# biology; awaiting C. elegans-specific empirical refinement)
# ---------------------------------------------------------------------------

K_NA_HALF_mM_DEFAULT = 10.0    # K_d for intracellular Na binding
K_K_HALF_mM_DEFAULT = 1.5      # K_d for extracellular K binding
K_ATP_HALF_mM_DEFAULT = 0.1    # K_d for ATP binding
N_HILL_NA_DEFAULT = 3.0        # cooperativity (matches stoichiometry)
N_HILL_K_DEFAULT = 2.0
N_HILL_ATP_DEFAULT = 1.0

# Baseline ATP (Layer 1 v1 fixed at saturating; Phase F restructure makes
# this dynamic in v2 per §6.3 preservation strategy)
ATP_BASELINE_mM_DEFAULT = 3.0


# ---------------------------------------------------------------------------
# CeNGEN-derived per-cell eat-6 TPM (threshold 2 from `021821_medium_threshold2.csv`)
# ---------------------------------------------------------------------------

EAT6_TPM_CENGEN_T2: dict[str, float] = {
    # CeNGEN classes (no L/R split). AVAL + AVAR Nicoletti cells inherit
    # CeNGEN "AVA" expression (CeNGEN doesn't distinguish L/R for AVA either).
    "AVAL": 1346.0,
    "AVAR": 1346.0,
    "AIY":   157.0,
    "RIM":   388.0,
}

EAT6_TPM_ANCHOR_CELL = "AVAL"


# ---------------------------------------------------------------------------
# Brian2 equation fragment
# ---------------------------------------------------------------------------

NA_K_ATPASE_EQS = """
# ---- Na/K-ATPase Hill-kinetic pump parameters ----
pump_NaK_I_max_mAcm2 : 1
pump_NaK_K_Na_mM     : 1
pump_NaK_K_K_mM      : 1
pump_NaK_K_ATP_mM    : 1
pump_NaK_ATP_mM      : 1
pump_NaK_n_Na        : 1
pump_NaK_n_K         : 1
pump_NaK_n_ATP       : 1

# ---- Hill saturation factors ∈ [0, 1] ----
pump_NaK_f_Na  = (Na_in / pump_NaK_K_Na_mM)**pump_NaK_n_Na / (1 + (Na_in / pump_NaK_K_Na_mM)**pump_NaK_n_Na) : 1
pump_NaK_f_K   = (K_out / pump_NaK_K_K_mM)**pump_NaK_n_K  / (1 + (K_out / pump_NaK_K_K_mM)**pump_NaK_n_K)   : 1
pump_NaK_f_ATP = (pump_NaK_ATP_mM / pump_NaK_K_ATP_mM)**pump_NaK_n_ATP / (1 + (pump_NaK_ATP_mM / pump_NaK_K_ATP_mM)**pump_NaK_n_ATP) : 1

# ---- Net pump current density (mA/cm², outward-positive) ----
# Net +1 charge out per cycle (3 Na out - 2 K in = +1 net out)
pump_NaK_I_mAcm2 = pump_NaK_I_max_mAcm2 * pump_NaK_f_Na * pump_NaK_f_K * pump_NaK_f_ATP : 1

# ---- Per-ion contributions to ion_iX_total accumulators ----
# 3 Na out per cycle → +3·I_mAcm2 contribution to outward-positive Na current
# 2 K  in  per cycle → -2·I_mAcm2 contribution to outward-positive K  current
pump_NaK_iNa_mAcm2 = 3 * pump_NaK_I_mAcm2 : 1
pump_NaK_iK_mAcm2  = -2 * pump_NaK_I_mAcm2 : 1
"""


# ---------------------------------------------------------------------------
# Parameter application + cross-cell TPM scaling
# ---------------------------------------------------------------------------

def apply_na_k_atpase_params(
    group,
    I_max_mAcm2: float,
    K_Na_mM: float = K_NA_HALF_mM_DEFAULT,
    K_K_mM: float = K_K_HALF_mM_DEFAULT,
    K_ATP_mM: float = K_ATP_HALF_mM_DEFAULT,
    ATP_mM: float = ATP_BASELINE_mM_DEFAULT,
    n_Na: float = N_HILL_NA_DEFAULT,
    n_K: float = N_HILL_K_DEFAULT,
    n_ATP: float = N_HILL_ATP_DEFAULT,
) -> None:
    """Apply Hill-kinetic pump parameters to a Brian2 NeuronGroup.

    `I_max_mAcm2` is the saturating net pump current density. For cross-cell
    scaling, use `scale_I_max_by_eat6_tpm`.
    """
    group.pump_NaK_I_max_mAcm2 = I_max_mAcm2
    group.pump_NaK_K_Na_mM = K_Na_mM
    group.pump_NaK_K_K_mM = K_K_mM
    group.pump_NaK_K_ATP_mM = K_ATP_mM
    group.pump_NaK_ATP_mM = ATP_mM
    group.pump_NaK_n_Na = n_Na
    group.pump_NaK_n_K = n_K
    group.pump_NaK_n_ATP = n_ATP


def scale_I_max_by_eat6_tpm(I_max_anchor_mAcm2: float, cell_name: str) -> float:
    """Cross-cell I_max scaling by eat-6 TPM relative to AVAL anchor.

    Per §2.8 epistemic labeling: this is approximation from adjacent biology
    under linear-density assumption. Deviations from prediction in validation
    are informative findings, not parameter failures.

    Args:
        I_max_anchor_mAcm2: I_max calibrated for AVAL (the anchor cell).
        cell_name: target cell (must be a key of EAT6_TPM_CENGEN_T2).

    Returns:
        I_max scaled for the target cell.
    """
    if cell_name not in EAT6_TPM_CENGEN_T2:
        raise KeyError(
            f"Unknown cell {cell_name!r}; known: {sorted(EAT6_TPM_CENGEN_T2)}"
        )
    return I_max_anchor_mAcm2 * (
        EAT6_TPM_CENGEN_T2[cell_name] / EAT6_TPM_CENGEN_T2[EAT6_TPM_ANCHOR_CELL]
    )
