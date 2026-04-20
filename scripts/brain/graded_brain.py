#!/usr/bin/env python3
"""Phase T1a — Graded (non-spiking) dynamics for C. elegans neurons.

C. elegans neurons do not produce vertebrate-style sodium action
potentials; most operate in a graded / plateau-potential regime
(Goodman, Hall, Avery 1998). The LIF spiking model we used through
Phase 3 is biologically wrong at first principles for this organism.

This module implements the graded-dynamics alternative following
Kunert-Graf et al. 2014 (PLOS Comp Bio), the consensus worm-specific
formulation:

  τ dV/dt = (V_rest - V) + Σ_j w_ij · σ(V_j) + I_ext + noise

  σ(V) = 1 / (1 + exp(-(V - V_half) / k))    (Boltzmann output)

So presynaptic "output" is a graded function of membrane potential,
continuous in [0, 1], multiplied by signed weight to drive
postsynaptic current. No spikes, no refractory period.

Interface (for ClosedLoopEnv drop-in compatibility):
  - names, idx, N                 : identical to LIFBrain
  - run(duration_ms)              : advances simulation
  - outputs: σ(V) per neuron, retrievable via output_rates()
  - ablate, set_proprioception    : same as LIFBrain
  - _brian2_seed attribute respected

Stochasticity: we still use a noise term on V so baseline activity
is non-zero even without synaptic drive.

The "spikes" collected by SpikeMonitor become "firing events" when σ
crosses 0.5 (transitioning from low- to high-output state). This is
an interpretive mapping so downstream classifier bank (trained on
Atanas calcium ΔF/F) receives roughly the same readout pattern.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, StateMonitor, Network,
    ms, mV, nS, pF, Hz,
    prefs, seed as brian2_seed,
)


prefs.codegen.target = "numpy"

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "connectome.npz"

# Kunert-Graf 2014 worm graded parameters.
# Simplified current-based formulation (vs conductance-based): synaptic
# current is directly proportional to presynaptic σ output — scale set
# so a fully-active (σ=1) connection produces ~5 mV effect per unit of
# synaptic-count weight at 10 ms tau.
from brian2 import pA
# Worm interneurons sit at higher baseline V than vertebrate neurons
# (Lockery 2009 review: worm V_rest typically -35 to -50 mV, not -65).
# Picking v_rest = -40 mV and σ midpoint v_half = -30 mV puts baseline
# σ ≈ 0.27 — non-zero tonic output consistent with worm graded-release
# physiology where neurons continuously signal via transmitter release
# rate even without spikes.
PARAMS = dict(
    tau=10 * ms,
    v_rest=-45 * mV,
    v_half=-30 * mV,      # 15 mV above rest — baseline σ ≈ 0.08
    k_half=6 * mV,
    C_mem=100 * pF,
    W_graded_I=5.0 * pA,  # strong enough to drive but not saturate
    g_gap=0.12 * nS,
    noise_sigma=4.0 * mV,
)


class GradedBrain:
    """Graded-dynamics alternative to LIFBrain. Drop-in for
    ClosedLoopEnv via the same interface."""

    def __init__(self, use_per_edge_glu_signs: bool = False,
                 include_gap: bool = True):
        if hasattr(self, "_brian2_seed"):
            brian2_seed(self._brian2_seed)

        d = np.load(ARTIFACT, allow_pickle=True)
        self.names: list[str] = [str(n) for n in d["names"]]
        self.N = len(self.names)
        self.idx: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.nt_primary = [str(n) for n in d["nt_primary"]]
        sign_base = np.array(d["sign"], dtype=np.int8).copy()

        # Same per-edge vs per-neuron choice as LIFBrain
        self._using_per_edge_signs = (
            use_per_edge_glu_signs and "W_chem_per_edge" in d.files
        )
        if self._using_per_edge_signs:
            W_chem: np.ndarray = d["W_chem_per_edge"].astype(np.float32)
        else:
            # Apply Glu→iGluR overrides (from lif_brain.py)
            from lif_brain import DEFAULT_SIGN_OVERRIDES
            for name, new_sign in DEFAULT_SIGN_OVERRIDES.items():
                if name in self.idx:
                    sign_base[self.idx[name]] = new_sign
            W_chem_raw = d["W_chem_raw"].astype(np.float32)
            W_chem = (sign_base[:, None].astype(np.float32) * W_chem_raw)

        W_gap = d["W_gap"].astype(np.float32)

        # Build Brian2 graded network
        ns = {**PARAMS}
        # Equations: graded σ(V) output + integration of synaptic input.
        # Separate summed variables for excitatory and inhibitory drive
        # (Brian2 requires distinct target variables per Synapses group).
        eqs = """
        dv/dt = (v_rest - v)/tau + (I_syn_exc + I_syn_inh + I_gap + I_ext)/C_mem
                + noise_sigma * xi / sqrt(tau) : volt
        sigma = 1 / (1 + exp(-(v - v_half)/k_half)) : 1
        I_syn_exc : amp
        I_syn_inh : amp
        I_gap : amp
        I_ext : amp
        """
        self.neurons = NeuronGroup(
            self.N, eqs,
            method="euler",
            namespace=ns,
        )
        self.neurons.v = PARAMS["v_rest"]

        exc_pre, exc_post = np.where(W_chem > 0)
        inh_pre, inh_post = np.where(W_chem < 0)

        # Excitatory — current-based, proportional to σ_pre × weight
        self.syn_exc = Synapses(
            self.neurons, self.neurons,
            model="""
            w : 1
            I_syn_exc_post = W_graded_I * w * sigma_pre : amp (summed)
            """,
            namespace=ns,
        )
        if len(exc_pre):
            self.syn_exc.connect(i=exc_pre.tolist(), j=exc_post.tolist())
            self.syn_exc.w = W_chem[exc_pre, exc_post].tolist()

        # Inhibitory — negative current (magnitude stored in w, sign in eq)
        self.syn_inh = Synapses(
            self.neurons, self.neurons,
            model="""
            w : 1
            I_syn_inh_post = -W_graded_I * w * sigma_pre : amp (summed)
            """,
            namespace=ns,
        )
        if len(inh_pre):
            self.syn_inh.connect(i=inh_pre.tolist(), j=inh_post.tolist())
            self.syn_inh.w = (-W_chem[inh_pre, inh_post]).tolist()

        # Gap junctions (unchanged from LIFBrain pattern — still valid)
        self.syn_gap = None
        if include_gap:
            gap_i, gap_j = np.where(W_gap > 0)
            if len(gap_i):
                self.syn_gap = Synapses(
                    self.neurons, self.neurons,
                    model="""
                    w_gap : 1
                    I_gap_post = g_gap * w_gap * (v_pre - v_post) : amp (summed)
                    """,
                    namespace=ns,
                )
                self.syn_gap.connect(i=gap_i.tolist(), j=gap_j.tolist())
                self.syn_gap.w_gap = W_gap[gap_i, gap_j].tolist()

        # StateMonitor tracks sigma (graded output) for all neurons.
        # Used by ClosedLoopEnv as the "rate" readout analog.
        self.state = StateMonitor(self.neurons, "sigma", record=True, dt=5*ms)

        components = [self.neurons, self.syn_exc, self.syn_inh, self.state]
        if self.syn_gap is not None:
            components.append(self.syn_gap)
        self.net = Network(*components)

        # Ablation + proprioception — same as LIF
        self.ablation_current_pA = np.zeros(self.N, dtype=np.float32)
        self._setup_proprioception()

        self.sign_overrides_applied: list = []
        self._stim_cache: list = []

        self.summary = dict(
            N=self.N,
            n_exc_syn=int(len(exc_pre)),
            n_inh_syn=int(len(inh_pre)),
            n_gap=int(len(gap_i)) if include_gap and len(gap_i) else 0,
            dynamics="graded",
            per_edge_glu_signs=self._using_per_edge_signs,
        )

    def _setup_proprioception(self):
        proprio_names = ["PDEL", "PDER", "PDA", "DVA"]
        self.proprio_idx = [self.idx[n] for n in proprio_names if n in self.idx]
        if not self.proprio_idx:
            self.proprio_group = None
            return
        n_prop = len(self.proprio_idx)
        self.proprio_group = PoissonGroup(n_prop, rates=np.zeros(n_prop) * Hz)
        self.proprio_syn = Synapses(
            self.proprio_group, self.neurons,
            on_pre="v_post += 4*mV",
        )
        for i, j in enumerate(self.proprio_idx):
            self.proprio_syn.connect(i=i, j=j)
        self.net.add(self.proprio_group, self.proprio_syn)

    def set_proprioception(self, body_curvature_mag: float) -> None:
        if self.proprio_group is None:
            return
        rate_hz = float(min(150.0, max(0.0, 400.0 * body_curvature_mag)))
        n = len(self.proprio_idx)
        self.proprio_group.rates = np.full(n, rate_hz) * Hz

    def ablate(self, names: list[str], current_pA: float = -1000.0) -> list[str]:
        hit = []
        for n in names:
            if n in self.idx:
                self.ablation_current_pA[self.idx[n]] = current_pA
                hit.append(n)
        if not hasattr(self, "_ablation_op_attached"):
            from brian2 import network_operation
            @network_operation(dt=50*ms)
            def _push_ablation():
                if getattr(self, "_modulation_attached", False):
                    return
                self.neurons.I_ext_ = self.ablation_current_pA * 1e-12
            self._ablation_op = _push_ablation
            self.net.add(_push_ablation)
            self._ablation_op_attached = True
        return hit

    def inject_poisson(self, neuron: str, rate_hz: float,
                       weight_mv: float = 15.0) -> None:
        if neuron not in self.idx:
            raise KeyError(f"Unknown neuron: {neuron}")
        target_idx = self.idx[neuron]
        pg = PoissonGroup(1, rate_hz * Hz)
        syn = Synapses(
            pg, self.neurons,
            on_pre=f"v_post += {weight_mv}*mV",
        )
        syn.connect(i=0, j=target_idx)
        self._stim_cache.extend([pg, syn])
        self.net.add(pg, syn)

    def run(self, duration_ms: float) -> None:
        self.net.run(duration_ms * ms)

    def time_ms(self) -> float:
        return float(self.net.t / ms)

    def output_rates(self, window_ms: float = 200) -> np.ndarray:
        """Return (N,) per-neuron σ-averaged output over the last
        `window_ms` (proxy for firing rate in the spiking model).
        Scaled 0-100 so magnitudes feel rate-like for downstream
        code expecting Hz."""
        t_arr = self.state.t / ms
        if len(t_arr) == 0:
            return np.zeros(self.N, dtype=np.float32)
        t_now = float(t_arr[-1])
        mask = t_arr >= (t_now - window_ms)
        sigmas = self.state.sigma[:, mask]
        return (sigmas.mean(axis=1) * 100.0).astype(np.float32)

    # --- SpikeMonitor-compatible shim -------------------------
    # ClosedLoopEnv reads self.spikes.i / self.spikes.t to compute
    # spike counts per sync window. For graded dynamics we synthesize
    # "events" by detecting σ crossings above threshold.
    class _FakeSpikeMonitor:
        def __init__(self, graded_brain):
            self.gb = graded_brain
            self._last_sigma = np.zeros(graded_brain.N, dtype=np.float32)
            self._events_i: list[int] = []
            self._events_t: list[float] = []
        @property
        def i(self):
            return np.array(self._events_i, dtype=int)
        @property
        def t(self):
            from brian2 import ms as _ms
            return np.array(self._events_t, dtype=float) * _ms
        def poll(self):
            """Detect σ rising through 0.5 threshold → append events."""
            cur_sigma = self.gb.output_rates(window_ms=10) / 100.0  # back to [0,1]
            threshold = 0.5
            rising = (cur_sigma > threshold) & (self._last_sigma <= threshold)
            t_now = self.gb.time_ms()
            for idx in np.where(rising)[0]:
                self._events_i.append(int(idx))
                self._events_t.append(t_now)
            self._last_sigma = cur_sigma

    @property
    def spikes(self):
        if not hasattr(self, "_fake_spikes"):
            self._fake_spikes = GradedBrain._FakeSpikeMonitor(self)
        return self._fake_spikes


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from sensory_injection import stimulate

    print("Building graded brain…")
    g = GradedBrain()
    print(g.summary)

    print("\nRun 300 ms baseline…")
    g.run(300)
    r = g.output_rates(100)
    print(f"  σ output range: [{r.min():.2f}, {r.max():.2f}]  mean {r.mean():.2f}")
    print(f"  active (σ>30): {int(np.sum(r > 30))} / {g.N}")

    print("\nInject ASH stim + 500 ms…")
    stimulate(g, "osmotic_shock", intensity=1.0)
    g.run(500)
    r2 = g.output_rates(200)
    # Check key neurons
    for n in ["ASHL", "ASHR", "AIBL", "AIBR", "AVAL", "AVAR", "AVBL", "AVBR"]:
        if n in g.idx:
            print(f"  σ({n}) = {r2[g.idx[n]]:.1f}")
