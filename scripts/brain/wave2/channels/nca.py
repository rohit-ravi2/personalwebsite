"""
NCA NALCN-homolog non-specific cation channel — Brian2 module.

Ca-ACTIVATED VERSION (2026-05-17 audit revision).

The original Nicoletti .mod modeled NCA as a passive leak: i = gbar*(v-e),
constant conductance, no state. This omits the Ca-activation that real
NCA/NALCN exhibits in C. elegans and mammals.

Biology (Yeh 2008, Humphrey 2007, Flourakis 2015, Cochet-Bissuel 2014):
  - NCA-1/NCA-2 form a heteromeric complex with UNC-79/UNC-80
  - The complex is voltage-INDEPENDENT (no gating by V)
  - But its open probability is potentiated by intracellular Ca²⁺
  - This is the molecular basis of "Ca-activated non-selective cation
    current" (CAN current) — sustained depolarizing inward current
    classically described in plateau-firing neurons
  - Couples Ca dynamics ↔ V dynamics via positive feedback (rising Ca
    opens NCA → depolarizes → more Ca → plateau)

Kinetic model:
  Hill-form Ca activation factor on top of constant-conductance leak.
  At baseline (Ca ≈ 0.05 μM, resting phasic cell), factor ≈ 1.0 — same
  as original leak. At elevated Ca (Ca > 1 μM, plateau cell), factor
  rises to a max_potentiation ceiling. Pairs with CDI on Ca channels:
      CDI (negative feedback): Ca↑ → Ca channels shut
      NCA-Ca-act (positive feedback): Ca↑ → NCA opens → V↑
  The balance produces stable plateau at intermediate Ca (~1-5 μM).

Default parameters:
  Kca_nca   = 1 μM = 1e-3 mM (half-activation)
  n_nca     = 2 (Hill coefficient)
  max_pot   = 5.0 (max potentiation at saturating Ca)
  gbar      = derived per-cell from γ × TPM × C_global (γ = 1.5 pS)
  e_nca     = +30 mV (non-specific cation reversal)

For phasic cells (Ca stays low), f_Ca ≈ 1 — no change from prior behavior.
For plateau cells (Ca elevated), f_Ca rises to 3-5×, providing the
sustained inward current that maintains depolarized rest emergently.
"""
from __future__ import annotations


NCA_PARAMS = {
    "gbar_nca_Scm2": 0.055,
    "e_nca_mV":      30.0,
    # Ca-activation (CAN current per Yeh 2008, Humphrey 2007)
    "Kca_nca_mM":    1.0e-3,   # 1 μM canonical (Yeh 2008). Tried 0.3 μM to
                                # bootstrap plateau from resting Ca but caused
                                # runaway in cells with moderate Ca (RIB went
                                # 0.06 → 9 μM) without helping AVA (whose Ca
                                # stays too low regardless: chicken-and-egg).
    "n_nca":         2.0,
    "max_pot_nca":   5.0,
}


NCA_EQS = """
# NCA Ca-activated non-specific cation channel.
# Ca-potentiation factor: 1 at Ca=0, rising to max_pot at saturating Ca.
nca_f_Ca = 1.0 + (nca_max_pot - 1.0) * (Ca_in / nca_Kca)**nca_n / (1.0 + (Ca_in / nca_Kca)**nca_n) : 1
ik_nca_mAcm2 = nca_gbar * nca_f_Ca * (v_mV - nca_e) : 1
nca_gbar : 1
nca_e : 1
nca_Kca : 1
nca_n : 1
nca_max_pot : 1
"""


def nca_apply_params(group, gbar_Scm2: float | None = None,
                     ek_mV: float | None = None,  # ignored; NCA uses its own e
                     params_override: dict | None = None) -> None:
    p = dict(NCA_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_nca_Scm2"] = gbar_Scm2

    setattr(group, "nca_gbar",    p["gbar_nca_Scm2"])
    setattr(group, "nca_e",       p["e_nca_mV"])
    setattr(group, "nca_Kca",     p["Kca_nca_mM"])
    setattr(group, "nca_n",       p["n_nca"])
    setattr(group, "nca_max_pot", p["max_pot_nca"])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def nca_init_states(group, v_mV: float = -60.0) -> None:
    pass


NAME = "nca"
EQS = NCA_EQS
apply_params = nca_apply_params
init_states = nca_init_states
