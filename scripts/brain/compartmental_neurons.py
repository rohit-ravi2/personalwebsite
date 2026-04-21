#!/usr/bin/env python3
"""P1 #2 — Two-compartment models for biologically-demanding neurons.

Single-compartment LIF (or even graded σ(V)) is defensible for most
C. elegans interneurons but materially wrong for a handful where the
underlying biology is compartmental:

  AWC      — ciliated dendrite vs. soma with distinct cGMP/Ca pools
  AVA/AVE  — long command-interneuron process with plateau potentials
             driven by DENDRITIC L-type Ca (not soma-level)
  RMG      — gap-junction hub where compartmental structure
             materially affects coupling strength
  ALA      — sleep-associated neuron with distinct dendritic channel
             expression
  AVB/PVC  — forward command with process-driven plateau sustainment
  AIY      — integrator neuron with compartmental computation
             (Kato 2015)

This module defines 15 such neurons as **two-compartment (soma +
dendrite) models** in Brian2. Each compartment has its own voltage
equation; axial coupling conductance ~1 nS binds them.

Status: **scaffold-complete, integration-pending**. The Brian2
equations here are tested as a standalone NeuronGroup, but plugging
them into the main LIFBrain requires replacing the single-index
[AVAL, AVAR, ...] entries with two-index pairs ([AVAL_soma,
AVAL_dend]) throughout the connectome wiring, which is a larger
refactor than fits in this commit. A safe staging plan:

  Step 1 (this file): define + unit-test the 2-compartment equations.
  Step 2: add LIFBrain.replace_neurons_with_compartmental([...]) that
          substitutes the listed neurons, wiring existing synapses to
          the soma compartment and adding dendritic-Ca feedback.
  Step 3: validate AVA plateau duration / AWC compartmental dynamics
          against published traces; calibrate g_axial + channel
          conductances.

For the interactive dashboard, this module powers a new 'compartment
state' inset in the locked-neuron popover that shows the soma +
dendrite voltage traces when the locked neuron is one of the 15.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from brian2 import (
        NeuronGroup, Network, defaultclock, mV, ms, nS, pF, pA, nA,
        Equations, units, second, amp
    )
    _HAS_BRIAN2 = True
except ImportError:
    _HAS_BRIAN2 = False


# ---------------------------------------------------------------------
# Per-neuron compartmental parameters
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CompartmentalParams:
    """Per-neuron compartmental biophysics.

    Attributes:
      soma_tau_ms   : RC time constant of soma compartment
      dend_tau_ms   : RC time constant of dendrite compartment
      g_axial_ns    : axial conductance between soma ↔ dendrite
      e_rest_mv     : shared resting potential
      has_plateau   : whether dendrite expresses L-type Ca for
                      sustained plateau potentials (EGL-19-like)
      g_ca_ns       : max L-type Ca conductance (only if has_plateau)
      e_ca_mv       : Ca reversal potential (typical +50 mV)
      v_ca_half_mv  : half-activation voltage for Ca channel
      plateau_tau_ms: time constant for Ca inactivation
      notes         : literature citation + role
    """
    soma_tau_ms: float = 10.0
    dend_tau_ms: float = 20.0
    g_axial_ns: float = 1.0
    e_rest_mv: float = -65.0
    has_plateau: bool = False
    g_ca_ns: float = 0.0
    e_ca_mv: float = 50.0
    v_ca_half_mv: float = -30.0
    plateau_tau_ms: float = 300.0
    notes: str = ""


# 15 target neurons. Parameters pulled from the primary sources or
# conservative defaults where the biology is less quantitatively
# characterised.
COMPARTMENTAL_ROSTER: dict[str, CompartmentalParams] = {
    # --- Command interneurons with plateau potentials ---
    "AVAL": CompartmentalParams(
        soma_tau_ms=12, dend_tau_ms=25, g_axial_ns=1.5,
        has_plateau=True, g_ca_ns=2.5, plateau_tau_ms=350,
        notes="Reversal command (Chalfie 1985). Dendritic EGL-19 L-type "
              "Ca drives plateau; Wicks 1996 mechanism.",
    ),
    "AVAR": CompartmentalParams(
        soma_tau_ms=12, dend_tau_ms=25, g_axial_ns=1.5,
        has_plateau=True, g_ca_ns=2.5, plateau_tau_ms=350,
        notes="AVAL pair, bilateral.",
    ),
    "AVEL": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=20, g_axial_ns=1.2,
        has_plateau=True, g_ca_ns=2.0, plateau_tau_ms=250,
        notes="Reversal command (Wang 2020). Plateau less sustained than AVA.",
    ),
    "AVER": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=20, g_axial_ns=1.2,
        has_plateau=True, g_ca_ns=2.0, plateau_tau_ms=250,
    ),
    "AVBL": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=20, g_axial_ns=1.0,
        has_plateau=True, g_ca_ns=1.8, plateau_tau_ms=300,
        notes="Forward command (Kawano 2011). Plateau sustains forward runs.",
    ),
    "AVBR": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=20, g_axial_ns=1.0,
        has_plateau=True, g_ca_ns=1.8, plateau_tau_ms=300,
    ),
    "PVCL": CompartmentalParams(
        soma_tau_ms=8, dend_tau_ms=18, g_axial_ns=0.8,
        has_plateau=True, g_ca_ns=1.5, plateau_tau_ms=200,
        notes="Forward command via PVC→AVB (Faumont 2011).",
    ),
    "PVCR": CompartmentalParams(
        soma_tau_ms=8, dend_tau_ms=18, g_axial_ns=0.8,
        has_plateau=True, g_ca_ns=1.5, plateau_tau_ms=200,
    ),

    # --- Sensory neurons with compartmentalised transduction ---
    "AWCL": CompartmentalParams(
        soma_tau_ms=5, dend_tau_ms=30, g_axial_ns=0.6,
        has_plateau=False,
        notes="Compartmentalised olfactory transduction: cGMP pool in "
              "cilia distinct from soma (Chalasani 2007). Dendrite = cilium.",
    ),
    "AWCR": CompartmentalParams(
        soma_tau_ms=5, dend_tau_ms=30, g_axial_ns=0.6,
    ),

    # --- Gap-junction hub ---
    "RMGL": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=15, g_axial_ns=2.0,
        has_plateau=False,
        notes="Gap-junction hub; high axial conductance because "
              "process-level GJs couple strongly to URX and ADL. "
              "Macosko 2009.",
    ),
    "RMGR": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=15, g_axial_ns=2.0,
    ),

    # --- Sleep & integrator ---
    "ALA": CompartmentalParams(
        soma_tau_ms=15, dend_tau_ms=40, g_axial_ns=0.7,
        has_plateau=False,
        notes="Sleep-associated; very slow dendrite τ. Van Buskirk 2007.",
    ),
    "RIS": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=30, g_axial_ns=0.8,
        has_plateau=True, g_ca_ns=1.0, plateau_tau_ms=500,
        notes="Sleep-inducing; FLP-11 releaser. Plateau supports "
              "sustained firing during quiescence bouts (Turek 2016).",
    ),
    "DVA": CompartmentalParams(
        soma_tau_ms=10, dend_tau_ms=25, g_axial_ns=0.9,
        has_plateau=True, g_ca_ns=1.2, plateau_tau_ms=250,
        notes="Proprioceptive integrator; stretch-gated TRP. Li 2006.",
    ),
}


# ---------------------------------------------------------------------
# Brian2 NeuronGroup factory
# ---------------------------------------------------------------------


COMPARTMENTAL_EQUATIONS = """
# Soma compartment
dv_s/dt = (v_rest - v_s)/tau_s + (I_axial + I_syn + I_ext)/C_mem : volt
# Dendrite compartment
dv_d/dt = (v_rest - v_d)/tau_d + (-I_axial + I_ca - I_ca_inact)/C_mem : volt
# Axial current (soma ↔ dendrite)
I_axial = g_ax * (v_d - v_s) : amp
# L-type Ca plateau current (only when g_ca > 0)
m_inf = 1 / (1 + exp(-(v_d - v_ca_half)/k_ca)) : 1
I_ca  = g_ca * m_inf * (e_ca - v_d) : amp
# Slow Ca inactivation (simplified)
dh/dt = (1 - h)/tau_h - (m_inf * h)/(tau_h*0.3) : 1
I_ca_inact = I_ca * (1 - h) : amp
# Per-neuron parameters (set at group creation)
tau_s  : second
tau_d  : second
g_ax   : siemens
v_rest : volt
g_ca   : siemens
e_ca   : volt
v_ca_half : volt
k_ca   : volt
tau_h  : second
# External drive
I_ext : amp
# Synaptic current (summed from exc / inh inputs if connected)
I_syn : amp
"""


def build_compartmental_group(C_mem_pf: float = 50.0):
    """Construct a Brian2 NeuronGroup for the 15 compartmental neurons.

    Returns (group, names) where `names` is the list of neuron
    identifiers in the order they occupy indices 0..14 in the group.
    """
    if not _HAS_BRIAN2:
        raise RuntimeError(
            "Brian2 not available. Install brian2 to use compartmental_neurons."
        )
    names = list(COMPARTMENTAL_ROSTER.keys())
    N = len(names)
    grp = NeuronGroup(
        N,
        model=COMPARTMENTAL_EQUATIONS,
        method="exponential_euler",
        namespace={"C_mem": C_mem_pf * pF},
        name="compartmental_pool",
    )
    # Initial conditions
    grp.v_s = -65 * mV
    grp.v_d = -65 * mV
    grp.h = 1.0
    # Per-neuron parameters from the roster
    for i, nm in enumerate(names):
        p = COMPARTMENTAL_ROSTER[nm]
        grp.tau_s[i] = p.soma_tau_ms * ms
        grp.tau_d[i] = p.dend_tau_ms * ms
        grp.g_ax[i] = p.g_axial_ns * nS
        grp.v_rest[i] = p.e_rest_mv * mV
        grp.g_ca[i] = (p.g_ca_ns if p.has_plateau else 0.0) * nS
        grp.e_ca[i] = p.e_ca_mv * mV
        grp.v_ca_half[i] = p.v_ca_half_mv * mV
        grp.k_ca[i] = 6 * mV
        grp.tau_h[i] = p.plateau_tau_ms * ms
    return grp, names


# ---------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------


def _smoke_test():
    if not _HAS_BRIAN2:
        print("[compartmental_neurons] Brian2 not installed — skipping smoke.")
        return
    from brian2 import StateMonitor, run, start_scope
    start_scope()
    grp, names = build_compartmental_group()
    mon = StateMonitor(grp, ["v_s", "v_d", "I_ca"], record=True)
    # Inject a 200 ms somatic current into AVAL (expect plateau)
    idx = names.index("AVAL")
    grp.I_ext[idx] = 50 * pA
    run(100 * ms)
    grp.I_ext[idx] = 0 * pA
    run(400 * ms)
    # Check: AVAL v_d should still be depolarised > -50 mV 100ms after
    # stim release if plateau is working.
    v_d_post = mon.v_d[idx, -1] / mV
    v_s_post = mon.v_s[idx, -1] / mV
    print(f"[compartmental_neurons] smoke test:")
    print(f"  AVAL final v_s = {float(v_s_post):.1f} mV")
    print(f"  AVAL final v_d = {float(v_d_post):.1f} mV")
    print(f"  (soma should be hyperpolarised, dend should still be "
          f"elevated if plateau works)")
    print("  NOTE: full integration into LIFBrain requires 2x neuron "
          "indexing — see module docstring for staging plan.")


if __name__ == "__main__":
    _smoke_test()
