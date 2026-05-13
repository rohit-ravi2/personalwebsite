"""
Per-CeNGEN-class pump scaling.

Extends the existing pump TPM dicts (which only cover the 4 Nicoletti cells)
to all 128 CeNGEN classes by injecting entries from pump_tpm_data.

After calling `extend_pump_dicts()`, the existing
`scale_I_max_by_eat6_tpm/mca3_tpm/kcc2_tpm/abts1_tpm` functions work
transparently with any CeNGEN class name.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from path2_scale.pump_tpm_data import (
    EAT_6_TPM, MCA_3_TPM, KCC_2_TPM, ABTS_1_TPM,
)
from pumps.na_k_atpase import EAT6_TPM_CENGEN_T2
from pumps.ca_clearance import MCA3_TPM_CENGEN_T2
from pumps.kcc2_abts1_lumped import KCC2_TPM_CENGEN_T2, ABTS1_TPM_CENGEN_T2


_EXTENDED = False


def extend_pump_dicts(min_floor_tpm: float = 1.0) -> None:
    """Inject all 128 CeNGEN classes into the four pump TPM dicts.

    Existing Nicoletti-cell entries (AVAL, AVAR, AIY, RIM) are preserved;
    new CeNGEN class entries (AVA, ASEL, AWA, etc.) are added.

    A min_floor is applied to KCC-2 + ABTS-1 since some cells have CeNGEN
    T2 = 0 for those genes — using a strict 0× scaling would zero out Cl
    extrusion entirely, which is biologically implausible. The floor
    represents below-threshold-but-nonzero baseline expression.
    """
    global _EXTENDED
    if _EXTENDED:
        return

    # eat-6 and mca-3: no floor (always >0 in CeNGEN T2)
    for cls, tpm in EAT_6_TPM.items():
        if cls not in EAT6_TPM_CENGEN_T2:
            EAT6_TPM_CENGEN_T2[cls] = tpm
    for cls, tpm in MCA_3_TPM.items():
        if cls not in MCA3_TPM_CENGEN_T2:
            MCA3_TPM_CENGEN_T2[cls] = tpm

    # kcc-2 + abts-1: apply min_floor for cells below T2
    for cls, tpm in KCC_2_TPM.items():
        if cls not in KCC2_TPM_CENGEN_T2:
            KCC2_TPM_CENGEN_T2[cls] = max(tpm, min_floor_tpm)
    for cls, tpm in ABTS_1_TPM.items():
        if cls not in ABTS1_TPM_CENGEN_T2:
            ABTS1_TPM_CENGEN_T2[cls] = max(tpm, min_floor_tpm)

    _EXTENDED = True


def coverage_report() -> dict:
    """Report current dict coverage."""
    return {
        "eat-6":   len(EAT6_TPM_CENGEN_T2),
        "mca-3":   len(MCA3_TPM_CENGEN_T2),
        "kcc-2":   len(KCC2_TPM_CENGEN_T2),
        "abts-1":  len(ABTS1_TPM_CENGEN_T2),
    }


if __name__ == "__main__":
    print("Before:", coverage_report())
    extend_pump_dicts()
    print("After:",  coverage_report())
    print("\nSample non-Nicoletti entries:")
    for cls in ["AVA", "HSN", "RIB", "VD_DD", "ASEL"]:
        if cls in EAT6_TPM_CENGEN_T2:
            print(f"  {cls:8s}: eat-6={EAT6_TPM_CENGEN_T2[cls]:>8.1f}  "
                  f"mca-3={MCA3_TPM_CENGEN_T2[cls]:>8.1f}  "
                  f"kcc-2={KCC2_TPM_CENGEN_T2[cls]:>8.1f}  "
                  f"abts-1={ABTS1_TPM_CENGEN_T2[cls]:>8.1f}")
