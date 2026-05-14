"""
Top-level Layer 2 network assembler.

Builds the 300-cell connectome network on the Path 2 substrate. Returns
a dict with the NeuronGroup, Synapse objects, monitor, network, and
metadata (cell_name → index mapping, neurotransmitter info, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
WAVE2_DIR = THIS_DIR.parent
sys.path.insert(0, str(WAVE2_DIR))

import numpy as np

from layer2.network_builder import (
    build_homogeneous_eqs, build_per_cell_params, apply_per_cell_params,
)
from layer2.synapse_models import (
    CHEMICAL_SYN_EQS, GAP_JUNCTION_EQS,
    DEFAULT_V_THR_mV, DEFAULT_V_SLOPE_mV, DEFAULT_TAU_ms,
    E_EXCITATORY_mV, E_INHIBITORY_mV, is_inhibitory,
)


CONNECTOME_PATH = Path("/home/rohit/Desktop/website/personalwebsite/scripts/brain/artifacts/connectome.npz")

# Synaptic weight scaling — connectome W_chem values are EM-count integers.
# Calibration: our default cells have surf=100 μm² with g_leak=1e-5 S/cm²
# → total leak ≈ 1e-11 S = 0.01 nS. Synaptic gbar must be in pS range to
# not dominate. Use ~1 pS per unit weight (factor 1000 smaller than c302
# reference for unit-membrane-area cells).
CHEM_WEIGHT_TO_GBAR_S = 1.0e-12  # 1 pS per unit weight
GAP_WEIGHT_TO_GBAR_S = 1.0e-14   # 0.01 pS per unit weight (gap junctions
                                  # are Ohmic and aggregate fast; need lower
                                  # scale to avoid voltage-clamp instability)


def assemble_layer2_network(connectome_path: Path = CONNECTOME_PATH,
                             chem_scale: float = CHEM_WEIGHT_TO_GBAR_S,
                             gap_scale: float = GAP_WEIGHT_TO_GBAR_S,
                             record_indices: list[int] | None = None,
                             dt_ms: float = 0.025):
    """Build the full Layer 2 network.

    Returns dict with:
      group: NeuronGroup (size N_cells)
      chem_syn_e, chem_syn_i: Synapses (excitatory + inhibitory chemical)
      gap_syn: Synapses (gap junctions, bidirectional via two unidirectional)
      monitor: StateMonitor
      network: Network
      meta: dict with cell_name_to_idx, neurotransmitters, etc.
    """
    from brian2 import (NeuronGroup, Synapses, StateMonitor, Network,
                        start_scope, defaultclock, prefs, ms, mV, pA, pF, nS, second)

    start_scope()
    prefs.codegen.target = "cython"
    defaultclock.dt = dt_ms * ms

    # Load connectome
    d = np.load(connectome_path, allow_pickle=True)
    names = list(d["names"])
    nt_primary = list(d["nt_primary"])
    W_chem = d["W_chem"]
    W_gap = d["W_gap"]
    post_sign_glu = d["post_sign_glu"]
    n_cells = len(names)

    # Build per-cell substrate params
    print(f"[layer2] Building substrate params for {n_cells} cells...")
    per_cell, unmapped = build_per_cell_params(names)
    if unmapped:
        print(f"[layer2] Unmapped cells (using fallback class): {unmapped}")

    # Build homogeneous eqs
    eqs = build_homogeneous_eqs()
    print(f"[layer2] Building NeuronGroup of {n_cells} cells (eqs ~{len(eqs)} chars)...")
    G = NeuronGroup(n_cells, eqs, method="rk4")

    # Apply per-cell parameters
    apply_per_cell_params(G, per_cell)
    cell_name_to_idx = {p["cell_name"]: i for i, p in enumerate(per_cell)}

    # ---------- Chemical synapses ----------
    # One Synapses object with per-edge E_syn. Avoids Brian2 conflict on
    # multiple summed-variable Synapses targeting the same I_syn variable.
    print(f"[layer2] Wiring chemical synapses ({(W_chem != 0).sum()} edges)...")
    pre_idx, post_idx = np.nonzero(W_chem)
    weights = W_chem[pre_idx, post_idx]

    inh_mask = np.array([is_inhibitory(int(post_sign_glu[i]), nt_primary[i])
                         for i in pre_idx])
    n_exc = (~inh_mask).sum()
    n_inh = inh_mask.sum()
    print(f"[layer2]   {n_exc} excitatory + {n_inh} inhibitory")

    E_syn_arr = np.where(inh_mask, E_INHIBITORY_mV, E_EXCITATORY_mV)

    chem_e = Synapses(G, G, model=CHEMICAL_SYN_EQS, method="rk4")
    chem_e.connect(i=pre_idx.tolist(), j=post_idx.tolist())
    chem_e.gbar_syn = weights * chem_scale * 1e9 * nS  # convert via nS unit
    chem_e.E_syn = E_syn_arr * mV
    chem_e.V_thr_syn = DEFAULT_V_THR_mV
    chem_e.V_slope_syn = DEFAULT_V_SLOPE_mV
    chem_e.tau_syn = DEFAULT_TAU_ms * ms
    chem_e.s = 0.0
    chem_i = None  # placeholder for backwards compat

    # ---------- Gap junctions ----------
    # Gap junctions are symmetric: create one entry per unordered pair as
    # two directed Synapses (i→j and j→i). The W_gap matrix may already be
    # symmetric; we iterate unique pairs.
    print(f"[layer2] Wiring gap junctions ({(W_gap != 0).sum()} edges)...")
    gap_pre, gap_post = np.nonzero(W_gap)
    gap_weights = W_gap[gap_pre, gap_post]

    gap_syn = Synapses(G, G, model=GAP_JUNCTION_EQS, method="rk4")
    if len(gap_pre) > 0:
        gap_syn.connect(i=gap_pre.tolist(), j=gap_post.tolist())
        gap_syn.g_gap = gap_weights * gap_scale * 1e9 * nS

    # ---------- Monitor ----------
    if record_indices is None:
        # Record all cells by default
        record_indices = list(range(n_cells))
    mon = StateMonitor(G, ["v", "K_in", "Na_in", "Cl_in", "Ca_in",
                            "I_syn", "I_gap", "I_inj"],
                       record=record_indices)

    net_objs = [G, chem_e, gap_syn, mon]
    if chem_i is not None:
        net_objs.append(chem_i)
    net = Network(*net_objs)

    return {
        "group": G,
        "chem": chem_e,
        "gap": gap_syn,
        "monitor": mon,
        "network": net,
        "meta": {
            "cell_names": names,
            "cell_name_to_idx": cell_name_to_idx,
            "nt_primary": nt_primary,
            "n_chem_excitatory": int(n_exc),
            "n_chem_inhibitory": int(n_inh),
            "n_gap": len(gap_pre),
            "unmapped_cells": unmapped,
            "per_cell_params": per_cell,
        },
    }
