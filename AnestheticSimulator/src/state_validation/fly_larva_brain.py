"""FlyLarvaBrain — Brian2 LIF brain on the Winding 2023 fly larva connectome.

Cross-species analog of LIFBrain (C. elegans simulator's worm brain on the Cook 2019
connectome). Same dynamics, same interface, different connectome substrate.

Architecture
------------
- 2,952 neurons (Winding 2023 brain connectome, complete L1/L3 larva)
- ~110,677 chemical synaptic edges (all-all matrix; compartmental refinement
  available in ad/da/aa/dd matrices but unused in V1)
- NT identity assigned by cell-type heuristic (fly_nt_identity_heuristic.csv)
  - KC, LHN, PN, MB-FBN/FFN, CN, RGN, ascending → ACh (excitatory)
  - LN → GABA (inhibitory)
  - sensory, MBON → Glu (excitatory in fly CNS, iGluR-dominant; opposite of worm)
  - MBIN → DA (treated as excitatory neuromodulator)
- Same LIF parameters as worm v3 LIFBrain (Mellem 2008 voltage scale): v_rest=-25 mV,
  v_thr=-10 mV, v_reset=-30 mV, tau=10 ms, t_ref=2 ms
- Same W_syn=0.8 mV scale; same noise σ=6 mV + bias=3 mV

Interface
---------
Mirrors LIFBrain so phase_g_state_validator.py works with either class:
  brain.neurons       — Brian2 NeuronGroup (with I_ext, v)
  brain.syn_exc       — excitatory Synapses (with .w writable)
  brain.syn_inh       — inhibitory Synapses (with .w writable)
  brain.spikes        — SpikeMonitor
  brain.run(ms)       — advance simulation
  brain.firing_rates(window_ms) → ndarray of per-neuron firing rates
  brain.names         — list of neuron string IDs (numeric IDs from Winding)
  brain.idx           — dict {name: index}
  brain.command_neurons — list of indices for the locomotor readout
                          (fly: DN-VNC + pre-DN-VNC; worm: AVA/AVB/etc.)

Key differences from worm LIFBrain
-----------------------------------
- No gap junctions (Winding all-all matrix is chemical only; gap junctions are
  rare in fly CNS and not in this dataset)
- Glu sign convention is +1 (iGluR-dominant in fly CNS), not -1 (GluCl in worm)
- No per-edge sign override (V1 uses cell-type heuristic for NT assignment;
  V2 can integrate FlyBase driver-line data when available)
- Larger (2952 vs 300 neurons) but Brian2 cython handles it at ~1.4× real-time
"""
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    defaultclock, ms, mV, nS, pF, Hz, second,
    prefs, seed as brian2_seed,
)

prefs.codegen.target = "cython"


# ===== Configurable paths =====

WINDING_DIR = Path('/mnt/ssd4tb/Desktop/C-Elegans/data/drosophila/winding2023/Supplementary-Data-S1')
NT_HEURISTIC_PATH = Path('/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/data/state_validation_fly/fly_nt_identity_heuristic.csv')


# ===== LIF parameters (mirror worm LIFBrain) =====

LIF_PARAMS = dict(
    tau=10 * ms,
    v_rest=-25 * mV,
    v_thr=-10 * mV,
    v_reset=-30 * mV,
    t_ref=2 * ms,
)

# W_syn calibrated to give ~2-3 Hz baseline firing on the Winding 2023 substrate
# (~5× lower than worm's 0.8 mV; reflects the fly's 18:1 E:I edge ratio + 3× higher
# average per-neuron synaptic input vs worm). Determined by sweep: 0.10 mV → 1.6 Hz,
# 0.15 mV → 2.3 Hz, 0.20 mV → 7.4 Hz, 0.80 mV → 119 Hz. 0.15 mV is the calibrated
# default for biologically-plausible baseline.
W_SYN_DEFAULT = 0.15 * mV
C_MEM_DEFAULT = 100 * pF
NOISE_SIGMA_DEFAULT = 6.0 * mV
V_REST_BIAS_DEFAULT = 3.0 * mV


# ===== Cell-type → NT identity (loaded from heuristic CSV) =====

def load_nt_heuristic(path: Path = NT_HEURISTIC_PATH) -> dict[str, tuple[str, int]]:
    """Returns {celltype: (nt_primary, sign)}."""
    out: dict[str, tuple[str, int]] = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            if line.startswith('celltype,'):
                continue
            parts = next(csv.reader([line]))
            celltype = parts[0]
            nt = parts[2]
            sign = int(parts[3])
            out[celltype] = (nt, sign)
    return out


# ===== Connectome loading =====

def load_winding_connectome(winding_dir: Path = WINDING_DIR) -> tuple[np.ndarray, list[str], list[str]]:
    """Load Winding all-all connectivity matrix + annotations.

    Returns:
        W_raw: (N, N) float32 — synaptic counts (unsigned)
        names: list of neuron string IDs (column headers from the matrix)
        celltypes: list of celltype labels per neuron (or 'unknown' if not in annotations)
    """
    import pandas as pd
    M = pd.read_csv(winding_dir / 'all-all_connectivity_matrix.csv', index_col=0)
    names = [str(n) for n in M.columns]
    W_raw = M.values.astype(np.float32)

    # Annotations: maps left_id and right_id to celltype
    ann = pd.read_csv(winding_dir / 'annotations.csv')
    celltype_lookup: dict[str, str] = {}
    for _, row in ann.iterrows():
        for col in ('left_id', 'right_id'):
            v = row.get(col)
            if v in (None, '', 'no pair') or (isinstance(v, float) and np.isnan(v)):
                continue
            celltype_lookup[str(v).strip()] = str(row['celltype']).strip()

    celltypes = [celltype_lookup.get(n, 'unknown') for n in names]
    return W_raw, names, celltypes


def assign_signs_by_nt_heuristic(celltypes: list[str], nt_heuristic: dict[str, tuple[str, int]]) -> tuple[np.ndarray, list[str]]:
    """For each neuron, assign sign based on cell-type heuristic.

    Returns:
        signs: (N,) int8 array of {-1, +1, 0 for unknown}
        nt_primary: (N,) list of NT names ('ACh', 'GABA', 'Glu', 'DA', 'unknown')
    """
    n = len(celltypes)
    signs = np.zeros(n, dtype=np.int8)
    nt_primary = []
    for i, ct in enumerate(celltypes):
        if ct in nt_heuristic:
            nt, sign = nt_heuristic[ct]
            signs[i] = sign
            nt_primary.append(nt)
        else:
            signs[i] = +1  # default excitatory for unknowns; conservative
            nt_primary.append('unknown')
    return signs, nt_primary


# ===== Identifying command neurons (locomotor readout substrate) =====

# DN-VNC: 91 descending neurons projecting to VNC (motor)
# pre-DN-VNC: 238 premotor neurons one synapse upstream
# These together are the fly larva analog of worm command interneurons.
COMMAND_CELLTYPES = ('DN-VNC', 'pre-DN-VNC')


def identify_command_neurons(celltypes: list[str]) -> list[int]:
    """Return indices of neurons in command-neuron set."""
    return [i for i, ct in enumerate(celltypes) if ct in COMMAND_CELLTYPES]


# ===== FlyLarvaBrain class =====

class FlyLarvaBrain:
    """Brian2 LIF brain on Winding 2023 fly larva connectome."""

    _brian2_seed = None

    def __init__(
        self,
        W_syn=W_SYN_DEFAULT,
        C_mem=C_MEM_DEFAULT,
        noise_sigma=NOISE_SIGMA_DEFAULT,
        v_rest_bias=V_REST_BIAS_DEFAULT,
        winding_dir: Path | None = None,
        nt_heuristic_path: Path | None = None,
    ):
        if self._brian2_seed is not None:
            brian2_seed(self._brian2_seed)

        # Load substrate
        winding_dir = winding_dir or WINDING_DIR
        nt_heuristic_path = nt_heuristic_path or NT_HEURISTIC_PATH
        W_raw, names, celltypes = load_winding_connectome(winding_dir)
        nt_heuristic = load_nt_heuristic(nt_heuristic_path)

        self.names: list[str] = names
        self.N = len(names)
        self.idx: dict[str, int] = {n: i for i, n in enumerate(names)}
        self.celltypes: list[str] = celltypes

        # Sign assignment
        signs, nt_primary = assign_signs_by_nt_heuristic(celltypes, nt_heuristic)
        self.sign = signs
        self.nt_primary = nt_primary

        # Build signed W_chem
        W_chem = signs[:, None].astype(np.float32) * W_raw

        # Save for diagnostic / mutation reference (not connected to Brian2 directly)
        self._W_chem_runtime = W_chem.copy()
        self._W_raw = W_raw

        # Build LIF NeuronGroup
        params = dict(LIF_PARAMS)
        namespace = {
            **params,
            'W_syn': W_syn,
            'C_mem': C_mem,
            'sigma': noise_sigma,
            'v_rest_bias': v_rest_bias,
        }
        eqs = """
        dv/dt = (v_rest - v + v_rest_bias)/tau + I_ext/C_mem + sigma*xi*tau**-0.5 : volt (unless refractory)
        I_ext : amp
        """
        self.neurons = NeuronGroup(
            self.N, eqs,
            threshold='v > v_thr',
            reset='v = v_reset',
            refractory='t_ref',
            method='euler',
            namespace=namespace,
        )
        self.neurons.v = LIF_PARAMS['v_rest']
        self.neurons.I_ext = 0 * 1e-12  # unitless, will be set as pA

        # Build chemical Synapses (exc + inh)
        exc_pre, exc_post = np.where(W_chem > 0)
        inh_pre, inh_post = np.where(W_chem < 0)
        exc_w = W_chem[exc_pre, exc_post].astype(np.float32)
        inh_w = (-W_chem[inh_pre, inh_post]).astype(np.float32)

        self.syn_exc = Synapses(
            self.neurons, self.neurons,
            model='w : 1',
            on_pre='v_post += W_syn * w',
            namespace={'W_syn': W_syn},
        )
        if len(exc_pre):
            self.syn_exc.connect(i=exc_pre.tolist(), j=exc_post.tolist())
            self.syn_exc.w = exc_w.tolist()

        self.syn_inh = Synapses(
            self.neurons, self.neurons,
            model='w : 1',
            on_pre='v_post -= W_syn * w',
            namespace={'W_syn': W_syn},
        )
        if len(inh_pre):
            self.syn_inh.connect(i=inh_pre.tolist(), j=inh_post.tolist())
            self.syn_inh.w = inh_w.tolist()

        # Spike monitor
        self.spikes = SpikeMonitor(self.neurons)

        # Network
        self.net = Network(self.neurons, self.syn_exc, self.syn_inh, self.spikes)

        # Command-neuron indices (locomotor readout)
        self.command_neurons_idx: list[int] = identify_command_neurons(celltypes)

        # Bookkeeping
        self.W_syn = W_syn  # kept for compatibility with worm interface; not mutable
        self._stim_cache: list = []

        # Diagnostic info
        self.n_exc_edges = int(len(exc_pre))
        self.n_inh_edges = int(len(inh_pre))
        self.n_command = len(self.command_neurons_idx)

    # ===== Public API mirroring LIFBrain =====

    def run(self, duration_ms: float) -> None:
        self.net.run(duration_ms * ms)

    def firing_rates(self, window_ms: float = 200.0) -> np.ndarray:
        """Per-neuron firing rate (Hz) over the last `window_ms`."""
        if len(self.spikes.t) == 0:
            return np.zeros(self.N)
        t_now = self.net.t
        t_cut = t_now - window_ms * ms
        ts = self.spikes.t[:]
        ids = self.spikes.i[:]
        recent = ts >= t_cut
        out = np.bincount(ids[recent], minlength=self.N).astype(np.float64)
        return out / (window_ms / 1000.0)


# ===== Diagnostic / smoke test =====

def smoke_test(duration_s: float = 15.0, seed: int = 42) -> dict:
    """Run a baseline simulation and report network state metrics.

    Expected behavior:
      - mean firing rate in 1-5 Hz range (matching Atanas-style baseline)
      - command-neuron mean rate similar (no special elevation or suppression)
      - LIF neurons firing without runaway / silence
    """
    import time

    print(f'=== FlyLarvaBrain smoke test (seed={seed}, duration={duration_s}s) ===')

    class SeededFlyBrain(FlyLarvaBrain):
        _brian2_seed = seed

    np.random.seed(seed)
    t0 = time.time()
    brain = SeededFlyBrain()
    print(f'  setup wall: {time.time()-t0:.1f}s')
    print(f'  N neurons: {brain.N}')
    print(f'  N excitatory edges: {brain.n_exc_edges}')
    print(f'  N inhibitory edges: {brain.n_inh_edges}')
    print(f'  N command neurons: {brain.n_command}  (DN-VNC + pre-DN-VNC)')
    print(f'  cell-type breakdown:')
    from collections import Counter
    c = Counter(brain.celltypes)
    for ct, n in sorted(c.items(), key=lambda kv: -kv[1])[:10]:
        nt, sign = '?', '?'
        if c.get(ct, 0) > 0:
            for i in range(brain.N):
                if brain.celltypes[i] == ct:
                    nt = brain.nt_primary[i]
                    sign = brain.sign[i]
                    break
        print(f'    {ct:>14s}  n={n:>4d}  nt={nt}  sign={sign:+d}')

    t1 = time.time()
    brain.run(duration_s * 1000.0)
    wall = time.time() - t1
    print(f'\n  {duration_s}s simulated:  {wall:.1f}s wall  →  {duration_s/wall:.2f}× real-time')

    # Network state metrics
    spike_t_array = np.asarray(brain.spikes.t / second)
    spike_i_array = np.asarray(brain.spikes.i, dtype=int)
    n_total_spikes = len(spike_t_array)
    mean_rate = n_total_spikes / brain.N / duration_s
    print(f'  total spikes: {n_total_spikes}')
    print(f'  network mean rate: {mean_rate:.2f} Hz')

    # Command-neuron mean rate
    cmd_mask = np.isin(spike_i_array, brain.command_neurons_idx)
    cmd_rate = cmd_mask.sum() / brain.n_command / duration_s if brain.n_command > 0 else 0.0
    print(f'  command-neuron mean rate: {cmd_rate:.2f} Hz')

    return {
        'n_neurons': brain.N,
        'n_exc_edges': brain.n_exc_edges,
        'n_inh_edges': brain.n_inh_edges,
        'n_command': brain.n_command,
        'wall_s': wall,
        'realtime_factor': duration_s / wall,
        'network_mean_rate_hz': mean_rate,
        'command_mean_rate_hz': cmd_rate,
        'total_spikes': n_total_spikes,
    }


if __name__ == '__main__':
    smoke_test()
