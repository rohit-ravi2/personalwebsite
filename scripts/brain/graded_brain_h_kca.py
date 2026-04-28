#!/usr/bin/env python3
"""Wave 1 sandbox: GradedBrain + h-inactivation + I_KCa variants.

NOT a production patch. Sandbox for empirically testing whether β'
(GradedBrain extended with Ca-activated K+ current) closes the plateau
termination gap surfaced by Phase 0 analytic verification.

Phase 0 finding (verified empirically):
  h_ss = 0.3 / (0.3 + m_inf), where the hardcoded 0.3 is the
  inactivation-rate-ratio floor in the compartmental scaffold's
  h equation. As m_inf → 1, h_ss → 0.231, meaning ~23% of I_Ca
  remains permanently active at peak plateau voltage.

Three variants supported:
  - 'base'   : identical to graded_brain.py (no plateau termination)
  - 'h_only' : adds h-inactivation following compartmental_neurons.py
               pattern. Expected to FAIL Mellem cellular targets.
               This is the documented-failure reference for empirical
               confirmation of Phase 0 prediction at the GradedBrain
               layer.
  - 'h_kca'  : adds h-inactivation + intracellular [Ca] pool dynamics
               + Ca-activated K+ current (I_KCa). Primary patch
               hypothesis. K_Ca termination provides both plateau
               termination AND tonic baseline regulation (single
               mechanism for two problems).

Equations (h_kca variant):
  σ(V)         = 1 / (1 + exp(-(V - v_half)/k_half))
  m_Ca         = 1 / (1 + exp(-(V - v_Ca_half)/k_Ca))
  dh/dt        = (1 - h)/tau_h - (m_Ca * h)/(tau_h * 0.3)
  I_Ca         = g_Ca_local * m_Ca * h * (E_Ca - V)        # h-gated
  d[Ca]/dt     = +alpha_Ca * I_Ca - [Ca]/tau_Ca_decay      # Session 3
                                                           # sign fix
  f_Ca         = [Ca]^n / (K_d^n + [Ca]^n)
  I_KCa        = g_KCa * f_Ca * (E_K - V)

Default parameters (literature-grounded mid-range, fixed for cross-
session comparability with Session 3 compartmental work):
  alpha_Ca    = 0.05 µM/(pA·ms)        (range 0.01-0.1)
  tau_Ca_decay = 200 ms                (already in PARAMS, was dead code)
  n           = 4                       (BK channel Hill, Salkoff 2006)
  K_d         = 1.0 µM                  (range 0.5-2; Yuan 2000)
  g_KCa       = 2 nS                    (matched to plateau current scale)
  E_K         = -90 mV                  (standard K+ reversal)

NOTE on [Ca] units: implemented as dimensionless variable because
Brian2 unit-checking with umolar requires careful alpha_Ca declaration.
Numerically equivalent; the [Ca] state is interpreted as µM.
Phase 0.5 sanity check verifies [Ca] dynamics are in plausible range
under realistic I_Ca magnitudes.

Per-cell tau_h imported from compartmental_neurons.COMPARTMENTAL_ROSTER
for cells in overlap (AVA, AVE, AVB, PVC, RIS, DVA). Cells in
GradedBrain's PLATEAU_NEURONS but not compartmental's roster (AVD,
AIY) use AVA's tau_h = 350 ms as default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, StateMonitor, Network,
    ms, mV, nS, pF, Hz, pA,
    prefs, seed as brian2_seed,
)


prefs.codegen.target = "cython"

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "connectome.npz"


# ---------------------------------------------------------------------
# Parameters (extends graded_brain.PARAMS with h + Ca-K terms)
# ---------------------------------------------------------------------

PARAMS = dict(
    # Inherited from graded_brain.py
    tau=10 * ms,
    v_rest=-45 * mV,
    v_half=-30 * mV,
    k_half=6 * mV,
    C_mem=100 * pF,
    W_graded_I=5.0 * pA,
    g_gap=0.12 * nS,
    noise_sigma=4.0 * mV,
    # L-type Ca activation (unchanged from graded_brain)
    g_Ca_max=2.0 * nS,
    v_Ca_half=-25 * mV,         # Mellem 2008 grounded (KEEP, do NOT shift)
    k_Ca=5 * mV,
    E_Ca=50 * mV,
    # NEW: Ca pool dynamics — alpha_Ca and tau_Ca_decay activate the
    # previously-dead-code parameter.
    # alpha_Ca dimensions: dimensionless-Ca per (pA × ms).
    # I_Ca in pA, integrated over ms → [Ca] in dimensionless units
    # interpreted as µM.
    #
    # Phase 0.5 calibration: α_Ca = 0.05 (original spec) produced
    # [Ca]_ss ≈ 281 µM (1000× too high). Corrected to 0.0005 per
    # cross-session calibration, putting [Ca] into single-µM range.
    # Fall-back α_Ca = 0.0001 tested separately for graded-Hill
    # comparison (saturation vs graded f_Ca regime).
    alpha_Ca=0.0005 / (pA * ms),
    tau_Ca_decay=200 * ms,
    # NEW: I_KCa (Ca-activated K+) — Salkoff 2006 / Yuan 2000 BK params
    g_KCa=2.0 * nS,
    K_d_KCa=1.0,                 # dimensionless, interpreted as µM
    n_Hill=4.0,                  # standard for BK
    E_K=-90 * mV,
)


# Cells expressing plateau-generating L-type Ca (same list as graded_brain)
PLATEAU_NEURONS = [
    "AVAL", "AVAR",
    "AVEL", "AVER",
    "AVDL", "AVDR",
    "AVBL", "AVBR",
    "PVCL", "PVCR",
    "AIYL", "AIYR",
    "RIS",
    "DVA",
]


# Per-cell tau_h, sourced from compartmental_neurons.COMPARTMENTAL_ROSTER
# for cells in overlap. AVD, AIY default to AVA's tau_h = 350 ms (these
# cells are in GradedBrain's PLATEAU_NEURONS but not compartmental's
# roster; weakest literature evidence for plateau, so use the most-
# characterized cell as template).
_PER_CELL_TAU_H_MS = {
    # From compartmental_neurons.COMPARTMENTAL_ROSTER:
    "AVAL": 350.0, "AVAR": 350.0,
    "AVEL": 250.0, "AVER": 250.0,
    "AVBL": 300.0, "AVBR": 300.0,
    "PVCL": 200.0, "PVCR": 200.0,
    "RIS":  500.0,
    "DVA":  250.0,
    # Default for cells in GradedBrain plateau set but not compartmental:
    "AVDL": 350.0, "AVDR": 350.0,  # AVA template
    "AIYL": 350.0, "AIYR": 350.0,  # AVA template
}


# ---------------------------------------------------------------------
# Equation strings per variant
# ---------------------------------------------------------------------

# Variant 0 (base): identical to graded_brain.py — for completeness
EQS_BASE = """
dv/dt = (v_rest - v)/tau
        + (I_syn_exc + I_syn_inh + I_gap + I_ext + I_Ca)/C_mem
        + noise_sigma * xi / sqrt(tau) : volt
sigma = 1 / (1 + exp(-(v - v_half)/k_half)) : 1
m_Ca = 1 / (1 + exp(-(v - v_Ca_half)/k_Ca)) : 1
I_Ca = g_Ca_local * m_Ca * (E_Ca - v) : amp
g_Ca_local : siemens
I_syn_exc : amp
I_syn_inh : amp
I_gap : amp
I_ext : amp
"""


# Variant A (h_only): adds h inactivation. I_Ca gated by m_Ca * h.
EQS_H_ONLY = """
dv/dt = (v_rest - v)/tau
        + (I_syn_exc + I_syn_inh + I_gap + I_ext + I_Ca)/C_mem
        + noise_sigma * xi / sqrt(tau) : volt
sigma = 1 / (1 + exp(-(v - v_half)/k_half)) : 1
m_Ca = 1 / (1 + exp(-(v - v_Ca_half)/k_Ca)) : 1
I_Ca = g_Ca_local * m_Ca * h * (E_Ca - v) : amp
dh/dt = (1 - h)/tau_h - (m_Ca * h)/(tau_h * 0.3) : 1
g_Ca_local : siemens
tau_h : second
I_syn_exc : amp
I_syn_inh : amp
I_gap : amp
I_ext : amp
"""


# Variant B (h_kca): adds h-inactivation + Ca pool + I_KCa.
# d[Ca]/dt uses +alpha_Ca (Session 3 sign fix; I_Ca > 0 inward
# convention means inward Ca should INCREASE [Ca]).
EQS_H_KCA = """
dv/dt = (v_rest - v)/tau
        + (I_syn_exc + I_syn_inh + I_gap + I_ext + I_Ca + I_KCa)/C_mem
        + noise_sigma * xi / sqrt(tau) : volt
sigma = 1 / (1 + exp(-(v - v_half)/k_half)) : 1
m_Ca = 1 / (1 + exp(-(v - v_Ca_half)/k_Ca)) : 1
I_Ca = g_Ca_local * m_Ca * h * (E_Ca - v) : amp
dh/dt = (1 - h)/tau_h - (m_Ca * h)/(tau_h * 0.3) : 1
dCa_int/dt = alpha_Ca * I_Ca - Ca_int/tau_Ca_decay : 1
f_Ca = Ca_int**n_Hill / (K_d_KCa**n_Hill + Ca_int**n_Hill) : 1
I_KCa = g_KCa_local * f_Ca * (E_K - v) : amp
g_Ca_local : siemens
g_KCa_local : siemens
tau_h : second
I_syn_exc : amp
I_syn_inh : amp
I_gap : amp
I_ext : amp
"""


# ---------------------------------------------------------------------
# Variant build factory
# ---------------------------------------------------------------------


def build_neuron_group(variant: str, N: int, names: list[str]):
    """Construct a Brian2 NeuronGroup for one of the three variants.

    variant : 'base', 'h_only', or 'h_kca'

    Returns (neurons, ns) where ns is the namespace dict.
    """
    if variant not in ("base", "h_only", "h_kca"):
        raise ValueError(f"Unknown variant: {variant}")

    eqs = {"base": EQS_BASE, "h_only": EQS_H_ONLY, "h_kca": EQS_H_KCA}[variant]

    ns = {**PARAMS}
    neurons = NeuronGroup(N, eqs, method="euler", namespace=ns)
    neurons.v = PARAMS["v_rest"]

    # Per-neuron L-type Ca conductance (only PLATEAU_NEURONS get nonzero)
    has_ca = np.array(
        [1.0 if n in PLATEAU_NEURONS else 0.0 for n in names],
        dtype=np.float32,
    )
    g_Ca_values = has_ca * float(PARAMS["g_Ca_max"] / nS) * 1e-9  # in S
    neurons.g_Ca_local_ = g_Ca_values

    if variant in ("h_only", "h_kca"):
        # Initial h = 1 (no inactivation) at t=0
        neurons.h = 1.0
        # Per-cell tau_h
        tau_h_array = np.array(
            [_PER_CELL_TAU_H_MS.get(n, 350.0) for n in names],
            dtype=np.float32,
        )
        neurons.tau_h_ = tau_h_array * 1e-3  # ms → s

    if variant == "h_kca":
        # Initial intracellular [Ca] = 0
        neurons.Ca_int = 0.0
        # Per-neuron g_KCa (only on plateau cells; non-plateau cells
        # have g_KCa = 0 → I_KCa = 0)
        g_KCa_values = has_ca * float(PARAMS["g_KCa"] / nS) * 1e-9
        neurons.g_KCa_local_ = g_KCa_values

    return neurons, ns


# ---------------------------------------------------------------------
# Standalone smoke test — single-cell verification
# ---------------------------------------------------------------------


def _smoke_test():
    """Smoke test: AVAL with 50 pA / 100 ms injection on each variant.
    Verify equations parse and run without errors. Print key state
    values for visual inspection.
    """
    from brian2 import start_scope

    print("=" * 70)
    print("graded_brain_h_kca smoke test")
    print("=" * 70)

    # Single-cell test: just AVAL, set up Brian2 group with N=1
    test_names = ["AVAL"]

    for variant in ("base", "h_only", "h_kca"):
        print(f"\n--- Variant: {variant} ---")
        start_scope()
        neurons, ns = build_neuron_group(variant, 1, test_names)

        # Monitor key state variables
        record_vars = ["v", "sigma", "m_Ca", "I_Ca"]
        if variant in ("h_only", "h_kca"):
            record_vars.append("h")
        if variant == "h_kca":
            record_vars.extend(["Ca_int", "f_Ca", "I_KCa"])

        mon = StateMonitor(neurons, record_vars, record=True, dt=1*ms)
        net = Network(neurons, mon)

        # Settle
        net.run(200 * ms)
        v_settle = float(neurons.v[0] / mV)

        # Inject 50 pA for 100 ms
        neurons.I_ext_ = np.array([50e-12], dtype=np.float32)  # 50 pA in A
        net.run(100 * ms)
        v_peak = float(neurons.v[0] / mV)

        # Release
        neurons.I_ext_ = np.array([0.0], dtype=np.float32)
        net.run(900 * ms)
        v_post = float(neurons.v[0] / mV)

        print(f"  V settle  = {v_settle:+.1f} mV  (target ~-25)")
        print(f"  V peak    = {v_peak:+.1f} mV")
        print(f"  V post    = {v_post:+.1f} mV  (target settles to ~-25)")
        print(f"  ΔV peak   = {v_peak - v_settle:+.1f} mV")
        if variant in ("h_only", "h_kca"):
            print(f"  h final   = {float(neurons.h[0]):.3f}")
        if variant == "h_kca":
            print(f"  [Ca] final= {float(neurons.Ca_int[0]):.3f}  (µM-equivalent)")

    print("\n" + "=" * 70)
    print("All three variants instantiate and run without errors.")
    print("=" * 70)


if __name__ == "__main__":
    _smoke_test()
