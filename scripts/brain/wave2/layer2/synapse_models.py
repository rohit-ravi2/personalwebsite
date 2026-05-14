"""
Layer 2 synapse models for graded transmission + gap junctions.

C. elegans neurons transmit primarily via graded release (continuous,
V-dependent), not action potential. Standard model:

  release_rate(V_pre) = sigmoid((V_pre - V_thr) / V_slope) / tau
  ds/dt = release_rate * (1 - s) - s / tau
  I_syn = gbar * s * (V_post - E_syn)

V_thr ≈ -30 mV, V_slope ≈ 5 mV (typical C. elegans graded transmission
threshold; see Lockery & Goodman 2009).

Gap junctions: Ohmic, symmetric.
  I_gap = g_gap * (V_pre - V_post)

Both currents are aggregated into per-cell I_syn, I_gap (summed via Brian2
Synapses) and added to the membrane current sum in network_builder.
"""
from __future__ import annotations


# Default kinetics for graded chemical synapse
DEFAULT_V_THR_mV = -30.0
DEFAULT_V_SLOPE_mV = 5.0
DEFAULT_TAU_ms = 5.0  # fast decay typical of small-molecule NTs

# Reversal potentials (Brian2 mV)
E_EXCITATORY_mV = 0.0    # ACh, Glu→iGluR
E_INHIBITORY_mV = -70.0  # GABA, Glu→GluCl


# Chemical synapse equations — graded release
# v_pre / mV converts pre-synaptic V (volts) to a dimensionless mV value
# inline, avoiding a subexpression _pre reference.
CHEMICAL_SYN_EQS = """
ds/dt = release_rate * (1 - s) - s / tau_syn : 1 (clock-driven)
release_rate = (1.0 / (1.0 + exp(-(v_pre / mV - V_thr_syn) / V_slope_syn))) / tau_syn : 1/second
I_syn_post = gbar_syn * s * (v_post - E_syn) : amp (summed)
gbar_syn : siemens
E_syn : volt
V_thr_syn : 1
V_slope_syn : 1
tau_syn : second
"""


# Gap junction equations — Ohmic, contributes to I_gap on both sides
GAP_JUNCTION_EQS = """
I_gap_post = g_gap * (v_pre - v_post) : amp (summed)
g_gap : siemens
"""


def chemical_syn_params(tau_ms: float = DEFAULT_TAU_ms,
                        V_thr_mV: float = DEFAULT_V_THR_mV,
                        V_slope_mV: float = DEFAULT_V_SLOPE_mV) -> dict:
    """Default kinetic parameters as a dict for vectorized assignment."""
    return {
        "tau_syn_ms":   tau_ms,
        "V_thr_syn_mV": V_thr_mV,
        "V_slope_syn_mV": V_slope_mV,
    }


def is_inhibitory(post_sign: int, primary_nt: str) -> bool:
    """Decide whether a chemical synapse from a given pre-cell is inhibitory.

    Args:
        post_sign: per the post_sign_glu field — for glutamate, 1=excitatory
            (iGluR), -1=inhibitory (GluCl); for non-glutamate, ignore.
        primary_nt: pre-synaptic neurotransmitter from CeNGEN.

    GABA → inhibitory.
    Glutamate → depends on post-synaptic receptor (iGluR vs GluCl).
    All others (ACh, Glu→iGluR, DA, 5HT, OA, TA, peptides) → excitatory baseline.
    """
    nt_norm = primary_nt.lower() if primary_nt else ""
    if "gaba" in nt_norm:
        return True
    if "glutamate" in nt_norm or "glu" in nt_norm.split():
        return post_sign < 0
    return False
