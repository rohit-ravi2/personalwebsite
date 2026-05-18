"""
I_NaP — persistent sodium current.

Bootstrap depolarizing drive for plateau / command-interneuron cells (AVA,
AVB, AVD, AVE) that get trapped at the K-leak rest because Ca-NCA + EGL-19
need depolarization to engage, but they themselves cannot depolarize from
the K-dominated rest without an upstream voltage-activated Na current.

I_NaP biology
-------------
- Persistent (non-inactivating or very slowly inactivating) Na current.
- Voltage-activated, sub-threshold (activates below the AP threshold,
  starting around V = -60 to -50 mV).
- Sustained inward Na → tonic depolarizing drive.
- In mammals: NaV1.6/NaV1.2/NaV1.9 persistent component; sub-threshold
  pacemaker / plateau current; underlies "ramp" depolarization in
  motor neurons and pyramidal cells.

C. elegans homologs
-------------------
- nav-1 (only annotated voltage-gated Na channel) — restricted expression,
  NOT in CeNGEN T2 threshold csv (021821_medium_threshold2.csv: zero hits).
- UNC-8, DEL-1..9, ASIC-1..2 (ENaC/degenerin Na+ leak family) — broader.
- For Layer 2 substrate purposes we assign a UNIFORM default I_NaP gbar
  to all cells, recognising that this is a substrate-level "bootstrap"
  current rather than a per-gene-TPM-scaled current. Once a cell-specific
  expression signature emerges, the uniform gbar can be replaced with
  TPM-scaled per-cell values.

Kinetic model
-------------
Single fast activation gate (no h, fully persistent). Brian2 EQS:

  m_inf  = 1 / (1 + exp(-(V - V_half) / k))
  dm/dt  = (m_inf - m) / tau_m
  i_nap  = gbar · m · (V - e_NaP)

Defaults (sub-threshold persistent Na):
  V_half_m = -50 mV
  k        =   5 mV
  tau_m    =   1 ms (fast)
  e_NaP    = +30 mV (Na cation; NCA-style "own e", no Nernst bridge —
                     intentional simplification consistent with how NCA
                     and HCN are treated in this substrate)
  gbar     = 1.0e-5 S/cm² (uniform default; ~10× HCN's default γ-scaled
                            gbar in plateau cells → strong bootstrap drive
                            for cells stuck at K-rest, negligible effect
                            once cell is already depolarized).

Ion bookkeeping
---------------
Treated as Na-class — i_nap_mAcm2 flows into ion_iNa_total in the cell's
ion-balance equation (same handling as NCA, HCN).
"""
from __future__ import annotations


NAP_PARAMS = {
    # CALIBRATION HISTORY (2026-05-17):
    #   v1 — V_half=-50, gbar=1e-5     → no bootstrap (m_inf at -75 ≈ 0.002).
    #   v2 — V_half=-55, gbar=5e-5     → catastrophic Layer 2 NaN at t≈0.
    #   v3 — V_half=-65, gbar=1e-5     → catastrophic Layer 2 NaN at t≈0.
    # Conclusion: I_NaP at any meaningful uniform gbar with V_half in the
    # bootstrap range destabilizes the substrate because (a) every one of
    # the 300 cells gets the same Na influx and the Na/K-ATPase + NCX pumps
    # were calibrated without it, and (b) phasic cells with low g_leak get
    # depolarized into Ca-channel regime → CDI breaks → Ca runaway. A
    # functional bootstrap requires either per-cell gating (only command
    # plateau cells get nonzero I_NaP) or coordinated pump rescaling.
    #
    # Defaults below are set to v3 for the Layer 2 wiring tests above; in
    # practice scalable_builder.py applies this gbar uniformly to all 300
    # cells. Subsequent users: reduce gbar by ~100× OR restrict assignment
    # to a small set of plateau-command cells before re-running.
    "va_nap":         -65.0,    # half-activation (mV)
    "ka_nap":           5.0,    # slope (mV); positive → activates above va
    "mtau_nap":         1.0,    # activation time constant (ms) — fast
    "gbar_nap_Scm2":  1.0e-5,   # uniform default — DESTABILIZES SUBSTRATE
                                # when applied to all 300 cells; see above.
    "e_nap_mV":       30.0,     # Na cation reversal (own e; no Nernst bridge)
}


NAP_EQS = """
# I_NaP persistent Na current — single fast activation gate, no inactivation.
nap_minf = 1.0 / (1.0 + exp(-(v_mV - nap_va) / nap_ka)) : 1
dm_nap/dt = (nap_minf - m_nap) / (nap_mtau * ms) : 1
ik_nap_mAcm2 = nap_gbar * m_nap * (v_mV - nap_e) : 1
# Parameters:
nap_va : 1
nap_ka : 1
nap_mtau : 1
nap_gbar : 1
nap_e : 1
"""


def nap_apply_params(group, gbar_Scm2: float | None = None,
                      e_nap_mV: float | None = None,
                      params_override: dict | None = None) -> None:
    p = dict(NAP_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_nap_Scm2"] = gbar_Scm2
    if e_nap_mV is not None:
        p["e_nap_mV"] = e_nap_mV

    name_map = {
        "va_nap":         "nap_va",
        "ka_nap":         "nap_ka",
        "mtau_nap":       "nap_mtau",
        "gbar_nap_Scm2":  "nap_gbar",
        "e_nap_mV":       "nap_e",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def nap_init_states(group, v_mV: float = -60.0) -> None:
    import numpy as np
    p = NAP_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_nap"]) / p["ka_nap"]))
    group.m_nap = float(minf)


NAME = "nap"
EQS = NAP_EQS
apply_params = nap_apply_params
init_states = nap_init_states
