#!/usr/bin/env python3
"""Phase 3a step 2 — Brian2 LIF brain from the C. elegans connectome.

This is the C. elegans analog of the Shiu et al. 2024 *Drosophila*
brain model (`philshiu/Drosophila_brain_model`), adapted for worm:

  - 300 neurons (Cook 2019 hermaphrodite connectome ∩ Loer & Rand
    2022 NT table; CANL/CANR have no characterised synaptic output
    so are excluded).
  - Chemical synapses: `v_post ± W_syn × cleft_count` on presynaptic
    spike. Sign from NT identity (Loer & Rand 2022). This is the
    Shiu pattern, with the NT-sign lookup known ahead of time instead
    of inferred from EM (Eckstein 2024) as in fly.
  - Gap junctions: continuous summed current `g_gap × w × (v_pre − v_post)`.
    Shiu's model did not include gap junctions; we add them because
    they are quantitatively important in worm (command-interneuron
    coupling) and Cook 2019 provides complete gap data.
  - One free scalar parameter `W_syn` (global chemical-weight scale),
    same convention as Shiu.

The LIF parameters (τ, V_thr, V_rest, V_reset, t_ref) are taken from
Kunert-Graf et al. 2014 (the canonical C. elegans LIF reference) — not
Shiu's fly values, which are fly-tuned.

Caveats documented on-site:
  - Glutamate is signed −1 under the Kunert / Varshney / Izquierdo
    consensus (GluCl is the dominant CNS receptor in worm). Some
    pathways that are functionally excitatory (e.g., ASH → AVA driving
    reversal via iGluR) will appear inhibitory in this v1 model. This
    is a *known* limitation of pure-NT-sign models; we flag it and
    plan a v2 with per-edge receptor assignments.
  - LIF has no adaptation, no plasticity, no neuromodulation. All
    upgrades are on the Phase 5+ roadmap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Brian2 imports at module scope so the namespace picks up units.
from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
    defaultclock, ms, mV, nS, pF, Hz, second,
    prefs, seed as brian2_seed,
)

# Silence Brian2's cython cache noise; we want deterministic output.
prefs.codegen.target = "numpy"


ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "connectome.npz"

# Kunert-Graf 2014 LIF parameters (worm, not fly).
LIF_PARAMS = dict(
    tau     = 10 * ms,     # membrane time constant
    v_rest  = -65 * mV,
    v_thr   = -50 * mV,
    v_reset = -70 * mV,
    t_ref   = 2 * ms,
)

# Shiu-style single free scalar for chemical synapses. Units: mV per
# serial-section unit.
# Tuning history (the ONE free parameter — documented, not arbitrary):
#   - 2.0 mV → saturation: single spike fires downstream, all command
#     neurons pin at refractory ceiling (~22 Hz).
#   - 0.5 mV → under-propagation: 1-hop only (ASH→AIB), no 2-hop.
#   - 0.8 mV → balanced: typical 7-section synapse = 5.6 mV (summation
#     required to cross ~15 mV gap), typical 15-section = 12 mV (near
#     threshold on single spike). Matches physiological regime.
W_SYN_DEFAULT = 0.8 * mV

# Gap-junction coupling (see module docstring).
G_GAP_DEFAULT = 0.1 * nS

# Neuron capacitance — only matters for gap-current → voltage conversion.
C_MEM_DEFAULT = 100 * pF

# Tonic noise σ = 5 mV; combined with a small depolarising bias below
# gives baseline rates in the ~1–3 Hz range — matches Atanas 2023
# whole-brain calcium where most identified neurons show sparse
# activation events rather than tonic high-rate firing.
#
# Tuning log (all at W_syn=0.8 mV, include_gap=True, 300 neurons):
#   bias=8, σ=7  → 30 Hz mean   (too hot; saturates downstream circuits)
#   bias=4, σ=7  → 13 Hz mean   (still hot — Atanas shows ~1–2 Hz typical)
#   bias=2, σ=5  → ~2 Hz mean   (this target; physiologically realistic)
NOISE_SIGMA_DEFAULT = 6.0 * mV
V_REST_BIAS_DEFAULT = 3.0 * mV

# ----- Per-neuron receptor-sign overrides ----------------------------
# The pure NT-sign convention (Glu = −1 because GluCl dominates the
# worm CNS) fails for pathways where iGluR is the dominant postsynaptic
# receptor instead. These overrides are documented cases in the
# literature where a glutamatergic neuron is FUNCTIONALLY excitatory:
#
#   ASH   : noxious mechano/osmotic sensor. Activates AVA/AIB/RIM for
#           reversal (Chalasani 2007, Piggott 2011). iGluR-dominant.
#   ASK   : avoidance sensor — excites command neurons (Hart 2006).
#   ASE   : salt-sensing. Glu → iGluR on AIA/AIB (Ortiz 2009).
#   AIY   : interneuron Glu → excitation of AIZ (Li 2014).
#
# Overrides are applied at SIGN level (presynaptic), so they flip ALL
# chemical synapses out of the named neuron. A more granular v2 would
# override per-edge, not per-neuron — but per-neuron captures most
# functional mismatches cheaply and is documentable.
DEFAULT_SIGN_OVERRIDES: dict[str, int] = {
    # --- Sensory neurons: glutamate → iGluR on postsynaptic targets ---
    "ASHL": +1, "ASHR": +1,   # Chalasani 2007, Piggott 2011 (noxious)
    "ASKL": +1, "ASKR": +1,   # Hart 2006 (avoidance)
    "ASEL": +1, "ASER": +1,   # Ortiz 2009 (salt)
    "AWCL": +1, "AWCR": +1,   # Chalasani 2007 (AWC → AIB/AIA via iGluR)
    "AWAL": +1, "AWAR": +1,   # Shinkai 2011 (AWA attractive odors)
    "ADLL": +1, "ADLR": +1,   # Guo 2015 (ADL → AIB avoidance)
    "AFDL": +1, "AFDR": +1,   # Beverly 2011 (AFD → AIY via iGluR)
    "ASGL": +1, "ASGR": +1,   # Bargmann 2006 (chemosensory excitation)
    "AUAL": +1, "AUAR": +1,   # glutamatergic sensory-interneuron
    "URYDL": +1, "URYDR": +1, # head sensor, excitatory to RMD
    "URYVL": +1, "URYVR": +1,

    # --- Interneurons: glutamatergic but functionally excitatory ---
    "AIYL": +1, "AIYR": +1,   # Li 2014 (AIY → AIZ via iGluR)
    "AIBL": +1, "AIBR": +1,   # AIB → AVA (iGluR), central reversal circuit
    "RIAL": +1, "RIAR": +1,   # RIA → RMD (Hendricks 2012)
}


class LIFBrain:
    """Brian2-backed LIF network loaded from the Cook 2019 / Loer&Rand
    connectome artifact produced by `build_connectome_matrix.py`."""

    def __init__(
        self,
        W_syn=W_SYN_DEFAULT,
        g_gap=G_GAP_DEFAULT,
        C_mem=C_MEM_DEFAULT,
        noise_sigma=NOISE_SIGMA_DEFAULT,
        v_rest_bias=V_REST_BIAS_DEFAULT,
        include_gap=True,
        sign_overrides: dict[str, int] | None = None,
        use_per_edge_glu_signs: bool = False,
    ):
        """LIF brain constructor.

        use_per_edge_glu_signs: if True and W_chem_per_edge is present
            in the connectome artifact, use CeNGEN-derived per-edge
            glutamate receptor signs (v3.2 infrastructure). If False
            (default), use legacy per-neuron NT signs + hand-picked
            overrides (v3.1 behaviour). Per-edge is more biologically
            accurate but requires re-tuning of modulation strengths
            and FSM thresholds — kept optional pending v3.3
            re-calibration.
        """
        if sign_overrides is None:
            sign_overrides = DEFAULT_SIGN_OVERRIDES

        # Deterministic Brian2 RNG. np.random.seed() doesn't lock
        # Brian2's internal noise generator — need brian2.seed()
        # explicitly. Without this, identical np.random seeds produce
        # different simulation traces due to membrane-noise drift.
        # (v3.3 reproducibility audit fix.)
        if hasattr(self, "_brian2_seed"):
            brian2_seed(self._brian2_seed)
        d = np.load(ARTIFACT, allow_pickle=True)
        self.names: list[str] = [str(n) for n in d["names"]]
        self.N = len(self.names)
        self.idx: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.nt_primary = [str(n) for n in d["nt_primary"]]
        self.sign = np.array(d["sign"], dtype=np.int8)

        W_chem_raw: np.ndarray = d["W_chem_raw"].astype(np.float32)
        sign_base = np.array(d["sign"], dtype=np.int8).copy()

        # v3.2: per-edge glutamate receptor signs (Phase 3d-4) are
        # available if computed by build_connectome_matrix.py. They're
        # more biologically accurate but require re-tuning of modulation
        # strengths + FSM thresholds — default off pending v3.3 recal.
        self.sign_overrides_applied: list[tuple[str, int, int]] = []
        has_per_edge = "W_chem_per_edge" in d.files
        if use_per_edge_glu_signs and has_per_edge:
            W_chem: np.ndarray = d["W_chem_per_edge"].astype(np.float32)
            self._using_per_edge_signs = True
        else:
            # Legacy path (default): apply per-neuron Glu→iGluR exceptions
            for name, new_sign in sign_overrides.items():
                if name in self.idx:
                    old = int(sign_base[self.idx[name]])
                    if old != new_sign:
                        sign_base[self.idx[name]] = new_sign
                        self.sign_overrides_applied.append((name, old, new_sign))
            W_chem: np.ndarray = (sign_base[:, None].astype(np.float32) * W_chem_raw)
            self._using_per_edge_signs = False

        W_gap: np.ndarray = d["W_gap"].astype(np.float32)

        # Build Brian2 network
        params = dict(LIF_PARAMS)
        params["v_rest"] = LIF_PARAMS["v_rest"] + v_rest_bias
        namespace = {**params, "W_syn": W_syn, "g_gap": g_gap,
                     "C_mem": C_mem, "noise_sigma": noise_sigma}

        # LIF with stochastic voltage noise (xi is Brian2's white-noise
        # primitive). Including noise forces stochastic integrator.
        eqs = """
        dv/dt = (v_rest - v)/tau + (I_gap + I_ext)/C_mem
                + noise_sigma * xi / sqrt(tau) : volt (unless refractory)
        I_gap : amp
        I_ext : amp
        """
        self.neurons = NeuronGroup(
            self.N,
            eqs,
            threshold="v > v_thr",
            reset="v = v_reset",
            refractory=LIF_PARAMS["t_ref"],
            method="euler",  # required for stochastic equations
            namespace=namespace,
        )
        self.neurons.v = params["v_rest"]

        # Chemical synapses: split by sign so each group has a clean
        # on_pre rule. Magnitude (unsigned cleft count) is the weight.
        exc_pre, exc_post = np.where(W_chem > 0)
        inh_pre, inh_post = np.where(W_chem < 0)
        exc_w = W_chem[exc_pre, exc_post].astype(np.float32)
        inh_w = (-W_chem[inh_pre, inh_post]).astype(np.float32)

        self.syn_exc = Synapses(
            self.neurons, self.neurons,
            model="w : 1",
            on_pre="v_post += W_syn * w",
            namespace=namespace,
        )
        if len(exc_pre):
            self.syn_exc.connect(i=exc_pre.tolist(), j=exc_post.tolist())
            self.syn_exc.w = exc_w.tolist()

        self.syn_inh = Synapses(
            self.neurons, self.neurons,
            model="w : 1",
            on_pre="v_post -= W_syn * w",
            namespace=namespace,
        )
        if len(inh_pre):
            self.syn_inh.connect(i=inh_pre.tolist(), j=inh_post.tolist())
            self.syn_inh.w = inh_w.tolist()

        # Gap junctions — summed current, symmetric. Iterate over ALL
        # non-zero entries (matrix is symmetric → each pair contributes
        # once for i→j and once for j→i, naturally giving bidirectional
        # coupling).
        self.syn_gap = None
        if include_gap:
            gap_i, gap_j = np.where(W_gap > 0)
            if len(gap_i):
                gap_w = W_gap[gap_i, gap_j].astype(np.float32)
                self.syn_gap = Synapses(
                    self.neurons, self.neurons,
                    model="""
                    w_gap : 1
                    I_gap_post = g_gap * w_gap * (v_pre - v_post) : amp (summed)
                    """,
                    namespace=namespace,
                )
                self.syn_gap.connect(i=gap_i.tolist(), j=gap_j.tolist())
                self.syn_gap.w_gap = gap_w.tolist()

        self.spikes = SpikeMonitor(self.neurons)

        components = [self.neurons, self.syn_exc, self.syn_inh, self.spikes]
        if self.syn_gap is not None:
            components.append(self.syn_gap)
        self.net = Network(*components)

        self._stim_cache: list = []  # keep Python refs alive

        # Per-neuron persistent current (pA). Used by ablate() to
        # hyperpolarise specific neurons out of the firing regime.
        # Read by ModulationLayer when assigning I_ext so the two
        # compose correctly.
        self.ablation_current_pA = np.zeros(self.N, dtype=np.float32)

        # Summary for eyeballing
        self.summary = dict(
            N=self.N,
            n_exc_syn=int(len(exc_pre)),
            n_inh_syn=int(len(inh_pre)),
            n_gap=int(len(gap_i)) if include_gap and len(gap_i) else 0,
            n_sign_overrides=len(self.sign_overrides_applied),
            per_edge_glu_signs=self._using_per_edge_signs,
        )

    # ------------------------------------------------------------

    def ablate(self, names: list[str], current_pA: float = -1000.0) -> list[str]:
        """Silence specific neurons by injecting strong persistent
        hyperpolarising current. Analog of laser or genetic ablation.

        Args:
            names: neuron names to ablate.
            current_pA: hyperpolarising current magnitude. -1000 pA
                        keeps V ≈ -165 mV (well below threshold -50 mV)
                        so ablated neurons never spike.

        Returns list of actually-ablated neuron names (intersection with
        our 300-neuron set).
        """
        hit = []
        for n in names:
            if n in self.idx:
                self.ablation_current_pA[self.idx[n]] = current_pA
                hit.append(n)
        # If no modulation layer is attached, we still need to push
        # ablation currents into I_ext. Use a network_operation here.
        if not hasattr(self, "_ablation_op_attached"):
            from brian2 import network_operation, ms
            @network_operation(dt=50*ms)
            def _push_ablation():
                # Only if modulation isn't already driving I_ext
                if getattr(self, "_modulation_attached", False):
                    return
                self.neurons.I_ext_ = self.ablation_current_pA * 1e-12
            self._ablation_op = _push_ablation
            self.net.add(_push_ablation)
            self._ablation_op_attached = True
        return hit

    def inject_poisson(self, neuron: str, rate_hz: float,
                       weight_mv: float = 15.0) -> None:
        """Add a Poisson stimulus onto a named neuron. Non-destructive —
        multiple calls accumulate."""
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

    def firing_rates(self, window_ms: float = 200.0) -> np.ndarray:
        """Return a (N,) array of firing rates (Hz) over the last
        `window_ms` of simulation."""
        if len(self.spikes.t) == 0:
            return np.zeros(self.N)
        t_now = self.net.t
        t_cut = t_now - window_ms * ms
        ts = self.spikes.t[:]
        ids = self.spikes.i[:]
        recent = ts >= t_cut
        out = np.bincount(ids[recent], minlength=self.N).astype(np.float64)
        return out / (window_ms / 1000.0)

    def time_ms(self) -> float:
        return float(self.net.t / ms)


# ---------- Phase 3a gate smoke test ----------

def smoke_test() -> None:
    """Run the canonical 'reversal circuit' test: baseline → ASH
    stimulation → check AVA rises (or doesn't — that's itself the
    finding if the Glu=−1 convention suppresses the pathway)."""

    print("Building LIF brain…")
    brain = LIFBrain()
    s = brain.summary
    print(f"  neurons:         {s['N']}")
    print(f"  chem syn (exc):  {s['n_exc_syn']}")
    print(f"  chem syn (inh):  {s['n_inh_syn']}")
    print(f"  gap junctions:   {s['n_gap']} (bidirectional pairs, double-counted)")
    print(f"  sign overrides:  {s['n_sign_overrides']} neurons "
          f"({[x[0] for x in brain.sign_overrides_applied]})")

    print("\nPhase 1: baseline (300 ms, no stimulus)…")
    brain.run(300)
    base = brain.firing_rates(200)
    n_active = int(np.sum(base > 0.5))
    print(f"  total spikes: {len(brain.spikes.t)}")
    print(f"  active neurons (>0.5 Hz): {n_active}")
    print(f"  mean rate: {base.mean():.2f} Hz  max: {base.max():.1f} Hz")

    print("\nPhase 2: ASH stim (500 ms at 200 Hz on ASHL + ASHR)…")
    brain.inject_poisson("ASHL", 200, weight_mv=5)
    brain.inject_poisson("ASHR", 200, weight_mv=5)
    brain.run(500)
    stim = brain.firing_rates(400)

    # Readout: key neurons in the reversal circuit.
    key = ["ASHL", "ASHR",
           "AIBL", "AIBR",
           "RIML", "RIMR",
           "AVAL", "AVAR",
           "AVBL", "AVBR",
           "PVCL", "PVCR",
           "AVDL", "AVDR",
           "AVEL", "AVER"]
    print(f"\n{'neuron':<8} {'baseline':>10} {'ash_stim':>10} {'Δ':>10}")
    for n in key:
        i = brain.idx[n]
        d = stim[i] - base[i]
        print(f"{n:<8} {base[i]:>9.1f}  {stim[i]:>9.1f}  {d:>+9.1f}")

    avg_ava = 0.5 * (stim[brain.idx['AVAL']] + stim[brain.idx['AVAR']])
    avg_avb = 0.5 * (stim[brain.idx['AVBL']] + stim[brain.idx['AVBR']])
    print()
    print(f"AVA pair mean: {avg_ava:.1f} Hz | AVB pair mean: {avg_avb:.1f} Hz")
    if avg_ava > avg_avb:
        print("→ Reversal command (AVA) dominates → would go BACKWARD.")
    else:
        print("→ Forward command (AVB) dominates → would go FORWARD.")
    print("(ASH noxious stim should produce REVERSAL if the pure-NT-sign")
    print(" convention is faithful. If not, the Glu=−1 caveat bites — see")
    print(" module docstring. That's a scientifically informative finding.)")


if __name__ == "__main__":
    smoke_test()
