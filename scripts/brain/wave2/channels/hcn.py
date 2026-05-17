"""
HCN — hyperpolarization-activated cation current (I_h).

Activated by hyperpolarization (not depolarization like classical channels).
Mixed Na+/K+ permeability with reversal ~ -30 mV. At hyperpolarized rest
(V < -60), HCN opens and conducts depolarizing inward current, providing
a "sag" pull toward depolarized voltages.

Critical for plateau cells: HCN activates regardless of Ca, providing
bootstrap depolarizing drive that escapes the K-equilibrium-dominated
hyperpolarized trap. Once V depolarizes enough, Ca channels engage →
Ca rises → Ca-NCA and NCX positive feedback → plateau reached.

C. elegans: HCN homologs include cng-1, cng-2, cng-3 (cyclic-nucleotide-
gated cation channels). Expression patterns concentrated in specific
neuron classes. Classical mammalian HCN kinetics are well-characterized.

Kinetic model (single activation gate, hyperpolarization-activated):
  i_h = gbar · m · (v - e_h)
  m_inf = 1 / (1 + exp((v - va) / ka))    [note: REVERSED slope sign vs
                                           depolarization-activated channels]
  tau_m: slow, ~50-200 ms (canonical HCN slow kinetics)

Default parameters (canonical mammalian HCN1):
  va = -75 mV (half-activation; HCN activates below -60)
  ka = 8 mV (slope; sign is reversed in eqs)
  tau_m = 100 ms (slow activation)
  e_h = -30 mV (mixed Na/K, reversal between E_Na=+60 and E_K=-90)
  gbar = derived per-cell from γ × TPM × C_global (γ = 1 pS — modest)

Cells with significant HCN expression (cng-1, tax-2/4): AFD, AWC, ASEL,
AVA (some studies), pharyngeal MC.
"""
from __future__ import annotations


HCN_PARAMS = {
    "va_hcn":         -75.0,    # half-activation mV (hyperpolarized)
    "ka_hcn":          8.0,     # slope (formula has reversed sign)
    "mtau_hcn":      100.0,     # slow time constant (ms)
    "gbar_hcn_Scm2":   1.0e-5,  # default — overridden per-cell
    "eh_mV":         -30.0,     # mixed-cation reversal
}


HCN_EQS = """
# HCN hyperpolarization-activated cation channel — single gate, slow.
# Note slope sign: minf = 1/(1+exp((v-va)/ka)) (POSITIVE in exp arg → activates
# at V BELOW va).
hcn_minf = 1.0 / (1.0 + exp((v_mV - hcn_va) / hcn_ka)) : 1
dm_hcn/dt = (hcn_minf - m_hcn) / (hcn_mtau * ms) : 1
ik_hcn_mAcm2 = hcn_gbar * m_hcn * (v_mV - hcn_eh) : 1
# Parameters:
hcn_va : 1
hcn_ka : 1
hcn_mtau : 1
hcn_gbar : 1
hcn_eh : 1
"""


def hcn_apply_params(group, gbar_Scm2: float | None = None,
                      eh_mV: float | None = None,
                      params_override: dict | None = None) -> None:
    p = dict(HCN_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_hcn_Scm2"] = gbar_Scm2
    if eh_mV is not None:
        p["eh_mV"] = eh_mV

    name_map = {
        "va_hcn":         "hcn_va",
        "ka_hcn":         "hcn_ka",
        "mtau_hcn":       "hcn_mtau",
        "gbar_hcn_Scm2":  "hcn_gbar",
        "eh_mV":          "hcn_eh",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def hcn_init_states(group, v_mV: float = -60.0) -> None:
    import numpy as np
    p = HCN_PARAMS
    minf = 1.0 / (1.0 + np.exp((v_mV - p["va_hcn"]) / p["ka_hcn"]))
    group.m_hcn = float(minf)


NAME = "hcn"
EQS = HCN_EQS
apply_params = hcn_apply_params
init_states = hcn_init_states
