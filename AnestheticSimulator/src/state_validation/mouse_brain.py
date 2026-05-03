"""MouseBrain — V6 generic LIF random graph for mammalian cross-phylum test.

Per V5 M2, the architecture's predictions don't depend on specific connectome
topology beyond cell-type aggregates. So V6 uses a random graph of mammalian
cortex-like statistics (E:I ~80:20, mean degree ~30-60) at ~3000 neurons,
without a published connectome.

Same Brian2 LIF dynamics as worm V3 / fly V4. Same interface (neurons, syn_exc,
syn_inh, spikes, run, names, idx, command_neurons_idx) so phase_g_state_validator
transfers without modification.

Caveat (per V6 M0): tests only LRR / immobilization phenotype. Higher-order
mammalian anesthesia features (cortical EEG, NREM-like states, burst suppression,
consciousness disruption) require brain-region-specific architecture not modeled
here.
"""
from __future__ import annotations

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    defaultclock, ms, mV, nS, pF, Hz, second,
    prefs, seed as brian2_seed,
)

prefs.codegen.target = "cython"


# ===== LIF parameters (mirror worm/fly) =====

LIF_PARAMS = dict(
    tau=10 * ms,
    v_rest=-25 * mV,
    v_thr=-10 * mV,
    v_reset=-30 * mV,
    t_ref=2 * ms,
)

# W_syn calibrated to give ~2-3 Hz baseline. Mammalian random graph at 80:20 E:I
# with mean degree ~40 lands close to fly's W_syn=0.15 mV; calibrated to 0.18 mV.
W_SYN_DEFAULT = 0.18 * mV
C_MEM_DEFAULT = 100 * pF
NOISE_SIGMA_DEFAULT = 6.0 * mV
V_REST_BIAS_DEFAULT = 3.0 * mV


# ===== Mammalian cortex-like graph statistics =====

DEFAULT_N = 3000
DEFAULT_E_FRACTION = 0.80         # mammalian cortex ~80% excitatory pyramidal
DEFAULT_MEAN_DEGREE = 40          # local + lateral connectivity
DEFAULT_COMMAND_FRACTION = 0.10   # ~10% of neurons treated as locomotor / motor command
                                   # readout (analog of worm command interneurons / fly DN-VNC)

# Random graph generation seed — NOT the same as Brian2 sim seed; this fixes the
# substrate so multiple sims share identical wiring. Exposed for reproducibility.
DEFAULT_GRAPH_SEED = 20260503


def build_mouse_random_graph(
    N: int = DEFAULT_N,
    e_fraction: float = DEFAULT_E_FRACTION,
    mean_degree: float = DEFAULT_MEAN_DEGREE,
    graph_seed: int = DEFAULT_GRAPH_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a random graph with E:I sign assignment.

    Returns:
        W_chem: signed (N, N) float32 weights — small random magnitudes per edge
        signs: (N,) int8 — +1 for excitatory neurons, -1 for inhibitory
    """
    rng = np.random.default_rng(graph_seed)
    n_exc = int(round(N * e_fraction))
    signs = np.zeros(N, dtype=np.int8)
    exc_idxs = rng.choice(N, size=n_exc, replace=False)
    signs[:] = -1
    signs[exc_idxs] = +1

    # Erdős-Rényi: edge probability = mean_degree / N
    p = mean_degree / N
    edges_mask = rng.random((N, N)) < p
    np.fill_diagonal(edges_mask, False)
    W_chem = np.zeros((N, N), dtype=np.float32)
    edge_count = int(edges_mask.sum())
    # Random edge weights — mean ~1, modest spread (matches worm/fly conventions)
    W_chem[edges_mask] = rng.lognormal(mean=0.0, sigma=0.4, size=edge_count).astype(np.float32)
    # Apply sign by presynaptic identity
    W_chem = signs[:, None].astype(np.float32) * W_chem
    return W_chem, signs


# ===== MouseBrain class =====

class MouseBrain:
    """Brian2 LIF brain on a generic mammalian-cortex-like random graph.

    Mirrors LIFBrain / FlyLarvaBrain interface so phase_g_state_validator
    transfers without modification.
    """

    _brian2_seed = None

    def __init__(
        self,
        N: int = DEFAULT_N,
        e_fraction: float = DEFAULT_E_FRACTION,
        mean_degree: float = DEFAULT_MEAN_DEGREE,
        command_fraction: float = DEFAULT_COMMAND_FRACTION,
        graph_seed: int = DEFAULT_GRAPH_SEED,
        W_syn=W_SYN_DEFAULT,
        C_mem=C_MEM_DEFAULT,
        noise_sigma=NOISE_SIGMA_DEFAULT,
        v_rest_bias=V_REST_BIAS_DEFAULT,
    ):
        if self._brian2_seed is not None:
            brian2_seed(self._brian2_seed)

        # Build random graph
        W_chem, signs = build_mouse_random_graph(N, e_fraction, mean_degree, graph_seed)
        self.N = N
        self.sign = signs
        self.nt_primary = ['ACh' if s == 1 else 'GABA' for s in signs]
        self.names: list[str] = [f'M{i:04d}' for i in range(N)]
        self.idx: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self._W_chem_runtime = W_chem.copy()

        # Build LIF NeuronGroup
        params = dict(LIF_PARAMS)
        namespace = {**params, 'W_syn': W_syn, 'C_mem': C_mem,
                     'sigma': noise_sigma, 'v_rest_bias': v_rest_bias}
        eqs = """
        dv/dt = (v_rest - v + v_rest_bias)/tau + I_ext/C_mem + sigma*xi*tau**-0.5 : volt (unless refractory)
        I_ext : amp
        """
        self.neurons = NeuronGroup(
            N, eqs, threshold='v > v_thr', reset='v = v_reset',
            refractory='t_ref', method='euler', namespace=namespace,
        )
        self.neurons.v = LIF_PARAMS['v_rest']
        self.neurons.I_ext = 0

        # Build chemical Synapses
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

        # Spike monitor + Network
        self.spikes = SpikeMonitor(self.neurons)
        self.net = Network(self.neurons, self.syn_exc, self.syn_inh, self.spikes)

        # Command neurons: random subset of excitatory neurons (locomotor readout proxy)
        # ~10% of neurons treated as command interneurons (similar fraction to worm/fly)
        rng = np.random.default_rng(graph_seed + 999)
        exc_indices = np.where(signs == +1)[0]
        n_command = max(1, int(round(N * command_fraction)))
        self.command_neurons_idx = rng.choice(exc_indices, size=min(n_command, len(exc_indices)),
                                               replace=False).tolist()

        # Bookkeeping
        self.W_syn = W_syn
        self.n_exc_edges = int(len(exc_pre))
        self.n_inh_edges = int(len(inh_pre))
        self.n_command = len(self.command_neurons_idx)
        self.celltypes = ['exc' if s == 1 else 'inh' for s in signs]

    # ===== Public API mirroring LIFBrain =====

    def run(self, duration_ms: float) -> None:
        self.net.run(duration_ms * ms)

    def firing_rates(self, window_ms: float = 200.0) -> np.ndarray:
        if len(self.spikes.t) == 0:
            return np.zeros(self.N)
        t_now = self.net.t
        t_cut = t_now - window_ms * ms
        ts = self.spikes.t[:]
        ids = self.spikes.i[:]
        recent = ts >= t_cut
        out = np.bincount(ids[recent], minlength=self.N).astype(np.float64)
        return out / (window_ms / 1000.0)


# ===== smoke test =====

def smoke_test(duration_s: float = 15.0, seed: int = 42) -> dict:
    import time
    print(f'=== MouseBrain smoke test (seed={seed}, duration={duration_s}s) ===')

    class SeededMouseBrain(MouseBrain):
        _brian2_seed = seed

    np.random.seed(seed)
    t0 = time.time()
    brain = SeededMouseBrain()
    print(f'  setup wall: {time.time()-t0:.1f}s')
    print(f'  N neurons: {brain.N}  ({sum(s==1 for s in brain.sign)} exc, {sum(s==-1 for s in brain.sign)} inh)')
    print(f'  N excitatory edges: {brain.n_exc_edges}')
    print(f'  N inhibitory edges: {brain.n_inh_edges}')
    print(f'  N command neurons: {brain.n_command}  (~10% of excitatory)')
    print(f'  E:I ratio: {sum(s==1 for s in brain.sign)/brain.N*100:.0f}:{sum(s==-1 for s in brain.sign)/brain.N*100:.0f}')

    t1 = time.time()
    brain.run(duration_s * 1000.0)
    wall = time.time() - t1
    print(f'\n  {duration_s}s simulated:  {wall:.1f}s wall  →  {duration_s/wall:.2f}× real-time')

    spike_t = np.asarray(brain.spikes.t / second)
    spike_i = np.asarray(brain.spikes.i, dtype=int)
    n_total_spikes = len(spike_t)
    mean_rate = n_total_spikes / brain.N / duration_s
    print(f'  total spikes: {n_total_spikes}')
    print(f'  network mean rate: {mean_rate:.2f} Hz')

    cmd_mask = np.isin(spike_i, brain.command_neurons_idx)
    cmd_rate = cmd_mask.sum() / brain.n_command / duration_s if brain.n_command > 0 else 0.0
    print(f'  command-neuron mean rate: {cmd_rate:.2f} Hz')

    return {
        'n_neurons': brain.N, 'n_exc_edges': brain.n_exc_edges,
        'n_inh_edges': brain.n_inh_edges, 'n_command': brain.n_command,
        'wall_s': wall, 'realtime_factor': duration_s / wall,
        'network_mean_rate_hz': mean_rate, 'command_mean_rate_hz': cmd_rate,
        'total_spikes': n_total_spikes,
    }


if __name__ == '__main__':
    smoke_test()
