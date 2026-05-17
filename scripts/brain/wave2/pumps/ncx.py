"""
NCX — Na/Ca exchanger.

3 Na+ in : 1 Ca++ out per cycle. Electrogenic: net charge moved per cycle =
+1 in (3+ in, 2+ out) → depolarizing membrane current.

In C. elegans, NCX is encoded by ncx-1, ncx-2, ncx-3 (and ncx-4, -9). Major
Ca extrusion pathway that does NOT saturate like PMCA — operates at high
Ca where PMCA has reached its V_max plateau. Hill-form Ca activation:
NCX rate ∝ Ca_in / (Km + Ca_in).

Forward-only model (Ca high, Na gradient favorable). Reverse-mode NCX
(Ca influx when Na_in saturates) deferred.

References:
  - Mullins 1977 (canonical thermodynamic model)
  - Yoshida 2018 (C. elegans NCX-1 in muscle Ca regulation)
  - Sharma 2013 (NCX-3 in neuronal Ca homeostasis)

Convention (outward-positive, matches our ca_clearance.py):
  ncx_iCa_mAcm2: positive when Ca leaves cell (carries +2 per Ca)
  ncx_iNa_mAcm2: negative when Na enters (3 Na per cycle, 1.5× Ca-charge ratio)
  ncx_I_mAcm2:   net membrane current = iCa + iNa = -0.5 × iCa (depolarizing)
"""
from __future__ import annotations


# Defaults — calibrated 2026-05-17 to match PMCA scale
KM_CA_NCX_mM_DEFAULT = 1.0e-3  # 1 μM half-activation (canonical)
N_HILL_NCX_DEFAULT = 1.0         # No cooperativity typical for NCX
# NCX I_max anchored at PMCA Ca-clearance rate. Initial 5× was too aggressive
# — 3 Na in per cycle dumped large Na load into cells, depleting K via pump
# saturation. Modest baseline 1× PMCA = balanced Ca clearance without
# overwhelming Na/K pump capacity.
I_MAX_NCX_mAcm2_DEFAULT = 2.0e-6  # = AVAL PMCA anchor (1× not 5×)


NCX_EQS = """
# NCX Na/Ca exchanger — Hill-form Ca activation, electrogenic (3 Na in : 1 Ca out per cycle).
ncx_I_max_mAcm2 : 1
ncx_Km_Ca_mM    : 1
ncx_n           : 1

# Hill-form Ca activation
ncx_f_Ca = (Ca_in / ncx_Km_Ca_mM)**ncx_n / (1.0 + (Ca_in / ncx_Km_Ca_mM)**ncx_n) : 1

# Ca extrusion current density (outward-positive, carries 2+ per Ca cycle)
ncx_iCa_mAcm2 = ncx_I_max_mAcm2 * ncx_f_Ca : 1

# Na influx current density (3 Na per 1 Ca = 1.5× Ca-charge ratio, opposite sign)
ncx_iNa_mAcm2 = -1.5 * ncx_iCa_mAcm2 : 1

# Net membrane current (depolarizing — net +1 charge moved in per cycle)
ncx_I_mAcm2 = ncx_iCa_mAcm2 + ncx_iNa_mAcm2 : 1
"""


def apply_ncx_params(group, I_max_mAcm2: float = I_MAX_NCX_mAcm2_DEFAULT,
                     Km_Ca_mM: float = KM_CA_NCX_mM_DEFAULT,
                     n_hill: float = N_HILL_NCX_DEFAULT) -> None:
    group.ncx_I_max_mAcm2 = I_max_mAcm2
    group.ncx_Km_Ca_mM = Km_Ca_mM
    group.ncx_n = n_hill


def scale_I_max_by_ncx_tpm(I_max_anchor_mAcm2: float, cell_name: str,
                            ncx_tpm_data: dict | None = None,
                            anchor_cell: str = "AVA") -> float:
    """Cross-cell I_max scaling by aggregate ncx-1/2/3 TPM relative to anchor."""
    if ncx_tpm_data is None:
        return I_max_anchor_mAcm2
    paralogs = ["NCX_1_TPM", "NCX_2_TPM", "NCX_3_TPM"]
    cell_tpm = sum(ncx_tpm_data.get(p, {}).get(cell_name, 0.0) for p in paralogs)
    anchor_tpm = sum(ncx_tpm_data.get(p, {}).get(anchor_cell, 0.0) for p in paralogs)
    if anchor_tpm == 0:
        return I_max_anchor_mAcm2
    return I_max_anchor_mAcm2 * (cell_tpm / anchor_tpm)
