"""
Layer 1 §7.2 v2 — Lumped Ca clearance with threshold-Michaelis-Menten form.

Per Rohit's 2026-05-12 Path (b) authorization: replaces v1's pure
Michaelis-Menten (which had no stopping mechanism and drove [Ca]_in toward
zero) with a threshold form that turns OFF below [Ca]_target ≈ 50 nM:

    v_Ca = I_Ca_max · max(0, [Ca]_in - [Ca]_target) / (K_half + max(0, [Ca]_in - [Ca]_target))

Stops below target (no extrusion when [Ca]_in below the PMCA-set baseline);
MM-saturating above target. Prevents pathological drift to zero or below.
Captures PMCA's effective irreversibility under physiological gradients.

Electrogenic: 1 Ca²⁺ out per cycle → net +2 charge out per cycle →
hyperpolarizing membrane current. Per §4.7 design doc.

Per-cell scaling by mca-3 CeNGEN TPM (unchanged from v1).

Lumped pathways (PMCA + NCX + SERCA proxy) — NCX's depolarizing
contribution (1 Ca out / 3 Na in → +1 charge in) is averaged in. AIY has no
NCX — see §6.2 v1 → v1.5 refactor trigger.
"""
from __future__ import annotations


# Mammalian PMCA Michaelis constant (per §2.8: approximation from adjacent biology)
K_CA_HALF_mM_DEFAULT = 5.0e-4    # 500 nM (typical PMCA Km)

# Resting [Ca]_in target — pump turns off below this value
CA_TARGET_mM_DEFAULT = 5.0e-5    # 50 nM (PMCA-set baseline)


MCA3_TPM_CENGEN_T2: dict[str, float] = {
    "AVAL": 478.0,
    "AVAR": 478.0,
    "AIY":   95.0,
    "RIM":  253.0,
}

ANCHOR_CELL = "AVAL"


# ---------------------------------------------------------------------------
# Brian2 equation fragment
# ---------------------------------------------------------------------------

LUMPED_CA_CLEARANCE_EQS = """
# ---- Lumped Ca clearance: threshold-MM form ----
ca_clear_I_max_mAcm2  : 1
ca_clear_K_half_mM    : 1
ca_clear_Ca_target_mM : 1

# delta > 0 above target; pump turns off below target via (delta > 0) gate
ca_clear_delta_mM     = Ca_in - ca_clear_Ca_target_mM : 1
ca_clear_delta_pos_mM = ca_clear_delta_mM * int(ca_clear_delta_mM > 0) : 1

# Michaelis-Menten on positive part (turns off at threshold)
ca_clear_I_mAcm2 = ca_clear_I_max_mAcm2 * ca_clear_delta_pos_mM / (ca_clear_K_half_mM + ca_clear_delta_pos_mM) : 1

# Per-ion contribution (positive = Ca out, contributes to outward-positive ion_iCa_total)
ca_clear_iCa_mAcm2 = ca_clear_I_mAcm2 : 1
"""


# ---------------------------------------------------------------------------
# Parameter application + cross-cell scaling
# ---------------------------------------------------------------------------

def apply_ca_clearance_params(
    group,
    I_max_mAcm2: float,
    K_half_mM: float = K_CA_HALF_mM_DEFAULT,
    Ca_target_mM: float = CA_TARGET_mM_DEFAULT,
) -> None:
    group.ca_clear_I_max_mAcm2 = I_max_mAcm2
    group.ca_clear_K_half_mM = K_half_mM
    group.ca_clear_Ca_target_mM = Ca_target_mM


def scale_I_max_by_mca3_tpm(I_max_anchor_mAcm2: float, cell_name: str) -> float:
    """Cross-cell I_max scaling by mca-3 TPM relative to AVAL anchor."""
    if cell_name not in MCA3_TPM_CENGEN_T2:
        raise KeyError(
            f"Unknown cell {cell_name!r}; known: {sorted(MCA3_TPM_CENGEN_T2)}"
        )
    return I_max_anchor_mAcm2 * (
        MCA3_TPM_CENGEN_T2[cell_name] / MCA3_TPM_CENGEN_T2[ANCHOR_CELL]
    )
