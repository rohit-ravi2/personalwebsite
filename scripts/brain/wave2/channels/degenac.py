"""
DEG/ENaC family — degenerin / epithelial Na channel.

Aggregates the depolarizing-leak members of the C. elegans DEG/ENaC family
(unc-8, del-1, del-2, del-3, asic-1, asic-2, deg-1, acd-1, acd-3, etc.).
Excludes the mechanosensory-specific members (mec-4, mec-10, deg-3, mec-6).

Biology:
  - Constitutive Na-selective leak channels (E_rev ≈ +50 mV per ENaC literature)
  - Voltage-INDEPENDENT (no V-gated opening) — major safety feature: avoids
    the positive-feedback cascade that universal I_NaP triggered
  - NOT Ca-activated — no Ca feedback amplification
  - SOME members are proton-gated (ASIC family especially)
  - DEL-4 specifically acts as ion-homeostasis regulator at the membrane,
    sensing pH/ion gradients to stabilize Na/K balance
  - "ionstasis" function critical for plateau cells maintaining their
    depolarized state without runaway

Per-cell selectivity emerges naturally from CeNGEN expression:
  Motor neurons (DA, VA, VB, DB): UNC-8, ASIC-1, DEL-1 (strong)
  Command interneurons (AVA, AVD, AVE): DEG-1, DEL-1 (moderate-strong)
  Dopaminergic (CEP, ADE, PDE): ASIC-1, DEL-2/3 (strong)
  Phasic sensory (ASE, AWC, AIY, AFD): NEAR ZERO — naturally excluded

This selectivity is what previous "universal I_NaP" attempts lacked.

Kinetic model (passive leak, no gating):
  i_DEG = gbar * (v - e_DEG)

Default parameters:
  e_DEG = +50 mV (Na-selective ENaC family)
  gbar = derived per-cell from γ × ΣTPM × C_global (γ = 10 pS)

For pH-dependent gating: not modeled here. When H+ dynamics added later,
DEL-4 / ASIC-1 should be modulated by intracellular/extracellular pH.

References:
  - Bianchi & Driscoll 2002 (DEG/ENaC family overview)
  - Schafer 2015 (UNC-8 in motor neuron Na current)
  - Voglis & Tavernarakis 2008 (DEG/ENaC roles)
  - Wang 2008 (DEL-4 as ionstasis regulator in dopaminergic neurons)
"""
from __future__ import annotations


DEGENAC_PARAMS = {
    "gbar_degenac_Scm2":   1.0e-5,    # overridden per-cell
    "e_degenac_mV":       50.0,        # Na-selective ENaC reversal
}


DEGENAC_EQS = """
# DEG/ENaC family — constitutive Na-selective leak. No gating.
ik_degenac_mAcm2 = degenac_gbar * (v_mV - degenac_e) : 1
degenac_gbar : 1
degenac_e : 1
"""


def degenac_apply_params(group, gbar_Scm2: float | None = None,
                          ek_mV: float | None = None,    # signature compat; not used
                          params_override: dict | None = None) -> None:
    p = dict(DEGENAC_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_degenac_Scm2"] = gbar_Scm2

    setattr(group, "degenac_gbar", p["gbar_degenac_Scm2"])
    setattr(group, "degenac_e", p["e_degenac_mV"])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def degenac_init_states(group, v_mV: float = -60.0) -> None:
    pass


NAME = "degenac"
EQS = DEGENAC_EQS
apply_params = degenac_apply_params
init_states = degenac_init_states
