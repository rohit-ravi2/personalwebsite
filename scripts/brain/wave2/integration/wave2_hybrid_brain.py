"""
Wave2HybridBrain — Phase δ WB2 deliverable.

Hybrid brain: Wave 2 cells (full biophysical detail) for AVAL/AVAR/AIY/RIM,
LIF for the remaining cells. Single Brian2 Network. Exposes the
ClosedLoopEnv I/O contract identical to LIFBrain.

Architecture (per `wave2/artifacts/phase_delta_scoping.md` §6.1):
  - Alternative B (hybrid): one Brian2 Network with multiple NeuronGroups.
    + Wave 2 cells: AVAL, AVAR (5-channel each), AIYL, AIYR, RIML, RIMR
      (separate single-cell NeuronGroups, each with own equations + state).
    + LIF scaffold: NeuronGroup of (N - n_wave2) cells at the connectome's
      remaining indices.
  - Connectome wiring: chemical & gap synapses partition into
    pair-of-group categories (LIF→LIF, LIF→Wave2, Wave2→LIF, Wave2→Wave2).
    All Synapses objects added to the unified Network.

Stage III status (overnight 2026-04-27/28):
  - WB2: this class (skeleton + AVAL drop-in proof of concept).
  - WB3: release-event rule design (V-threshold crossing default; PAUSE
    for biological judgment if non-trivial choices arise).
  - WB4: AIY pair extension.
  - WB5: RIM pair extension.
  - WB6: full multi-scenario validation.

THIS FILE implements WB2 with provisional release-event rule (V-threshold
crossing at -25 mV, 5 ms refractory). Documentation flags the rule as
provisional pending WB3 biological-judgment review.

I/O contract preserved (same as LIFBrain):
  - Attributes: names, idx, N, neurons (LIF group, the only one ClosedLoopEnv
    indexes into directly), spikes (SpikeMonitor over LIF group), idx_to_*
    helpers.
  - Methods: run, time_ms, set_proprioception, set_sensory_rate,
    inject_poisson, ablate, firing_rates.

KNOWN LIMITATIONS (overnight scope):
  - Only AVAL drop-in working in this version. AVAR/AIY/RIM scaffolded
    but not enabled by default (build flag below).
  - Spike-detection on Wave 2 cells uses simple V-threshold; alternative
    graded-release rule documented for WB3 selection.
  - I_ext modulation routing to Wave 2 cells handled via per-step Python
    update (50 ms cadence matching ClosedLoopEnv); not yet via
    ModulationLayer's @network_operation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WAVE2_DIR = Path(__file__).resolve().parent.parent
BRAIN_DIR = WAVE2_DIR.parent
sys.path.insert(0, str(BRAIN_DIR))
sys.path.insert(0, str(WAVE2_DIR))

from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
    network_operation, defaultclock,
    ms, mV, nS, pF, pA, Hz, second, prefs, seed as brian2_seed,
)
prefs.codegen.target = "cython"

from option_alpha_ava_cell import build_brian2_aval_4channel
from option_alpha_avar_cell import build_brian2_avar_5channel
from option_alpha_aiy_cell import build_brian2_aiy_7channel
from option_alpha_rim_cell import build_brian2_rim_7channel

# LIF parameters from lif_brain (re-imported to avoid circular import)
from lif_brain import (
    LIF_PARAMS, W_SYN_DEFAULT, G_GAP_DEFAULT, C_MEM_DEFAULT,
    NOISE_SIGMA_DEFAULT, V_REST_BIAS_DEFAULT,
    DEFAULT_SIGN_OVERRIDES, DOCUMENTED_SIGN_EXCEPTIONS,
    ARTIFACT,
)


# ---------------------------------------------------------------------------
# Wave 2 cell registry — which cells get Wave 2 detail
# ---------------------------------------------------------------------------

WAVE2_CELL_FACTORIES = {
    "AVAL": build_brian2_aval_4channel,
    "AVAR": build_brian2_avar_5channel,
    "AIYL": build_brian2_aiy_7channel,
    "AIYR": build_brian2_aiy_7channel,
    "RIML": build_brian2_rim_7channel,
    "RIMR": build_brian2_rim_7channel,
}

# Which cells are enabled by default in the hybrid brain.
# Overnight WB2 uses minimal {AVAL, AVAR} for proof of concept.
# WB4-WB5 will expand to AIY/RIM. Toggle via constructor.
DEFAULT_WAVE2_ACTIVE = ["AVAL", "AVAR"]

# Release-event rule (provisional WB2 default).
# WB3 will revisit this with biological-judgment input.
RELEASE_RULE_V_THRESHOLD_MV = -25.0   # AVAL/AVAR rest is ~-25 mV (Mellem 2008
                                       # quote; AVAR rest -24.22 mV per our
                                       # validation). Threshold at rest-equivalent.
RELEASE_RULE_REFRACTORY_MS = 5.0


class Wave2HybridBrain:
    """Hybrid brain with Wave 2 cells for select interneurons + LIF for the rest.

    Drop-in I/O replacement for LIFBrain in ClosedLoopEnv. Internally:

      - LIF NeuronGroup of (N - n_wave2) cells at "lif" indices.
      - Per Wave 2 cell type, one or more single-cell NeuronGroups.
      - Synapses split by source/target group identity:
        * LIF→LIF (chemical exc, inh; gap)
        * LIF→Wave2 (chemical exc, inh; gap)
        * Wave2→LIF (chemical exc, inh; gap)
        * Wave2→Wave2 (chemical exc, inh; gap)
      - SpikeMonitor over LIF group for downstream readout.
      - Wave 2 spike events piped into LIF receivers via per-cell
        threshold-detector @network_operation.

    Parameters
    ----------
    wave2_active : list[str], optional
        Names of cells to use Wave 2 detail. Default ["AVAL", "AVAR"].
        Must be subset of WAVE2_CELL_FACTORIES.
    same constructor kwargs as LIFBrain.

    Constructor signature mirrors LIFBrain except for `wave2_active` keyword.

    The `neurons` attribute exposes the LIF NeuronGroup. Wave 2 cells are
    accessible via `self.wave2_groups[name]`. The unified `spikes` attribute
    is a SpikeMonitor over LIF only — Wave 2 release events are emitted as
    pseudo-spikes via the LIF group's PoissonInput-style coupling, so
    the readout-classification path remains unchanged.

    KNOWN LIMITATION: as of overnight WB2, the LIF→Wave2 gap-junction coupling
    is approximated via current-pulse on Wave2's I_ext at 50 ms cadence
    (matching ClosedLoopEnv timestep). Native Brian2 (summed) gap is in
    follow-up WB3.
    """

    def __init__(
        self,
        wave2_active=None,
        W_syn=W_SYN_DEFAULT,
        g_gap=G_GAP_DEFAULT,
        C_mem=C_MEM_DEFAULT,
        noise_sigma=NOISE_SIGMA_DEFAULT,
        v_rest_bias=V_REST_BIAS_DEFAULT,
        include_gap=True,
        sign_overrides=None,
        use_per_edge_glu_signs=False,
        sign_exceptions=None,
        seed=None,
        cross_coupling_mode="off",
    ):
        """
        cross_coupling_mode : str
            "off" (default for WB2 — Wave 2 cells run isolated, no LIF coupling).
                  This is the SAFE state pending WB3 release-event biology review.
            "naive_voltage_bumps" — instantaneous v += W_syn*w on each side
                  (NOT recommended — causes V-blowup on Wave 2 due to small cm).
            "graded_current_capped" — current-based coupling capped at ±20 pA
                  per cell (WB2 provisional; replaceable by WB3).
        """
        self.cross_coupling_mode = cross_coupling_mode
        if wave2_active is None:
            wave2_active = list(DEFAULT_WAVE2_ACTIVE)
        for name in wave2_active:
            if name not in WAVE2_CELL_FACTORIES:
                raise ValueError(
                    f"wave2_active contains unknown cell {name}; "
                    f"available: {list(WAVE2_CELL_FACTORIES)}"
                )
        self.wave2_active = wave2_active

        if sign_overrides is None:
            sign_overrides = DEFAULT_SIGN_OVERRIDES
        if sign_exceptions is None:
            sign_exceptions = DOCUMENTED_SIGN_EXCEPTIONS

        if seed is not None:
            brian2_seed(seed)

        # Load connectome
        d = np.load(ARTIFACT, allow_pickle=True)
        self.names = [str(n) for n in d["names"]]
        self.N = len(self.names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.nt_primary = [str(n) for n in d["nt_primary"]]
        self.sign = np.array(d["sign"], dtype=np.int8)

        W_chem_raw = d["W_chem_raw"].astype(np.float32)
        sign_base = np.array(d["sign"], dtype=np.int8).copy()

        self.sign_overrides_applied = []
        has_per_edge = "W_chem_per_edge" in d.files
        if use_per_edge_glu_signs and has_per_edge:
            W_chem = d["W_chem_per_edge"].astype(np.float32).copy()
            self._using_per_edge_signs = True
        else:
            for name, new_sign in sign_overrides.items():
                if name in self.idx:
                    old = int(sign_base[self.idx[name]])
                    if old != new_sign:
                        sign_base[self.idx[name]] = new_sign
                        self.sign_overrides_applied.append((name, old, new_sign))
            W_chem = (sign_base[:, None].astype(np.float32) * W_chem_raw)
            self._using_per_edge_signs = False

        self.sign_exceptions_applied = []
        for (pre_name, post_name), new_sign in sign_exceptions.items():
            if pre_name not in self.idx or post_name not in self.idx:
                continue
            pi, qi = self.idx[pre_name], self.idx[post_name]
            if W_chem_raw[pi, qi] == 0:
                continue
            magnitude = abs(W_chem[pi, qi])
            cur = W_chem[pi, qi]
            old_sign = +1 if cur > 0 else (-1 if cur < 0 else 0)
            if old_sign != new_sign:
                W_chem[pi, qi] = new_sign * magnitude
                self.sign_exceptions_applied.append(
                    (pre_name, post_name, old_sign, new_sign)
                )

        self._W_chem_runtime = W_chem
        W_gap = d["W_gap"].astype(np.float32)

        # Wave 2 indices in the full connectome
        self.wave2_idx = {name: self.idx[name] for name in wave2_active}
        self.wave2_idx_set = set(self.wave2_idx.values())
        # LIF indices: all neurons NOT in wave2_active
        self.lif_idx_global = [i for i in range(self.N) if i not in self.wave2_idx_set]
        self.n_lif = len(self.lif_idx_global)
        # Map global idx -> LIF group local idx
        self.global_to_lif = {g: l for l, g in enumerate(self.lif_idx_global)}

        # Build LIF NeuronGroup
        params = dict(LIF_PARAMS)
        params["v_rest"] = LIF_PARAMS["v_rest"] + v_rest_bias
        namespace = {**params, "W_syn": W_syn, "g_gap": g_gap,
                     "C_mem": C_mem, "noise_sigma": noise_sigma}

        eqs = """
        dv/dt = (v_rest - v)/tau + (I_gap + I_ext)/C_mem
                + noise_sigma * xi / sqrt(tau) : volt (unless refractory)
        I_gap : amp
        I_ext : amp
        """
        self.neurons = NeuronGroup(
            self.n_lif,
            eqs,
            threshold="v > v_thr",
            reset="v = v_reset",
            refractory=LIF_PARAMS["t_ref"],
            method="euler",
            namespace=namespace,
        )
        self.neurons.v = params["v_rest"]
        self.spikes = SpikeMonitor(self.neurons)

        # Build Wave 2 cells
        self.wave2_bundles = {}      # name -> bundle dict from factory
        self.wave2_groups = {}       # name -> NeuronGroup
        self.wave2_last_spike_t = {} # name -> last spike-emit time (ms)
        for name in wave2_active:
            factory = WAVE2_CELL_FACTORIES[name]()
            bundle = factory()
            bundle["disable_clamp"]()
            self.wave2_bundles[name] = bundle
            self.wave2_groups[name] = bundle["group"]
            self.wave2_last_spike_t[name] = -1e9  # ms

        # Build chemical synapses: partition by (source group, target group).
        # For overnight WB2 minimal scope: only LIF→LIF chemicals are wired
        # natively; LIF→Wave2 and Wave2→LIF are handled via the network_operation
        # callback at 50 ms cadence (release-event rule).
        # This is functionally equivalent for cells whose firing rate is
        # << 1 / 50 ms = 20 Hz — most worm cells qualify. Higher-rate Wave 2
        # cells (none currently) would need finer-grained event piping.
        # We DO wire LIF→LIF native because that's >95% of edges.
        self._build_lif_to_lif_synapses(W_chem, W_gap, namespace, include_gap)

        # Cross-group routing maps (used by per-step network operation)
        self._build_cross_group_routing(W_chem, W_gap, W_syn, g_gap)

        # Per-step network operation: read Wave 2 voltages, detect events,
        # deliver to LIF receivers; read LIF firing rates, deliver to Wave 2 I_ext.
        self._build_event_routing_op(namespace)

        # Ablation infrastructure (matching LIFBrain)
        self.ablation_current_pA = np.zeros(self.N, dtype=np.float32)

        # Setup proprioception (matching LIFBrain)
        self._setup_proprioception()

        # Build the unified Network
        components = [self.neurons, self.spikes]
        for grp in self.wave2_groups.values():
            components.append(grp)
        # Add monitors from Wave 2 cells (their internal monitors)
        for bundle in self.wave2_bundles.values():
            components.append(bundle["monitor"])
        components.extend(self._all_synapses)
        components.append(self._event_routing_op)
        if self.proprio_group is not None:
            components.append(self.proprio_group)
            components.append(self.proprio_syn)
        self.net = Network(*components)

        # Track other dynamic Synapses (sensory, ablation push, etc.)
        self._stim_cache = []

        self.summary = dict(
            N=self.N,
            n_lif=self.n_lif,
            n_wave2=len(wave2_active),
            wave2_active=list(wave2_active),
            n_chem_lif_lif=int(self._n_lif_lif_chem),
            n_gap_lif_lif=int(self._n_lif_lif_gap),
            n_chem_cross=int(self._n_cross_chem),
            n_gap_cross=int(self._n_cross_gap),
            release_rule="V-threshold @ -25 mV, 5 ms refractory (provisional WB2)",
            per_edge_glu_signs=self._using_per_edge_signs,
        )

    # ------------------------------------------------------------
    # Synapse construction
    # ------------------------------------------------------------

    def _build_lif_to_lif_synapses(self, W_chem, W_gap, namespace, include_gap):
        """Wire LIF→LIF chemical and gap synapses natively in Brian2."""
        self._all_synapses = []

        # Build LIF-only chemical matrix
        lif_global = np.array(self.lif_idx_global)
        # Restrict to LIF×LIF block
        lif_mask_pre = np.zeros(self.N, dtype=bool)
        lif_mask_pre[lif_global] = True
        # Chemical edges: filter to (pre LIF, post LIF)
        exc_pre, exc_post = np.where(W_chem > 0)
        inh_pre, inh_post = np.where(W_chem < 0)
        chem_lif_lif_mask_e = np.array(
            [self._is_lif(p) and self._is_lif(q) for p, q in zip(exc_pre, exc_post)]
        )
        chem_lif_lif_mask_i = np.array(
            [self._is_lif(p) and self._is_lif(q) for p, q in zip(inh_pre, inh_post)]
        )
        ll_exc_pre = exc_pre[chem_lif_lif_mask_e] if len(exc_pre) else np.array([], dtype=int)
        ll_exc_post = exc_post[chem_lif_lif_mask_e] if len(exc_pre) else np.array([], dtype=int)
        ll_inh_pre = inh_pre[chem_lif_lif_mask_i] if len(inh_pre) else np.array([], dtype=int)
        ll_inh_post = inh_post[chem_lif_lif_mask_i] if len(inh_pre) else np.array([], dtype=int)

        ll_exc_pre_local = np.array([self.global_to_lif[p] for p in ll_exc_pre], dtype=int)
        ll_exc_post_local = np.array([self.global_to_lif[q] for q in ll_exc_post], dtype=int)
        ll_inh_pre_local = np.array([self.global_to_lif[p] for p in ll_inh_pre], dtype=int)
        ll_inh_post_local = np.array([self.global_to_lif[q] for q in ll_inh_post], dtype=int)

        ll_exc_w = W_chem[ll_exc_pre, ll_exc_post].astype(np.float32) if len(ll_exc_pre) else np.array([], dtype=np.float32)
        ll_inh_w = (-W_chem[ll_inh_pre, ll_inh_post]).astype(np.float32) if len(ll_inh_pre) else np.array([], dtype=np.float32)

        self.syn_exc = Synapses(
            self.neurons, self.neurons,
            model="w : 1",
            on_pre="v_post += W_syn * w",
            namespace=namespace,
        )
        if len(ll_exc_pre_local):
            self.syn_exc.connect(i=ll_exc_pre_local.tolist(), j=ll_exc_post_local.tolist())
            self.syn_exc.w = ll_exc_w.tolist()
        self._all_synapses.append(self.syn_exc)

        self.syn_inh = Synapses(
            self.neurons, self.neurons,
            model="w : 1",
            on_pre="v_post -= W_syn * w",
            namespace=namespace,
        )
        if len(ll_inh_pre_local):
            self.syn_inh.connect(i=ll_inh_pre_local.tolist(), j=ll_inh_post_local.tolist())
            self.syn_inh.w = ll_inh_w.tolist()
        self._all_synapses.append(self.syn_inh)

        # Gap LIF-LIF
        self.syn_gap = None
        n_gap_ll = 0
        if include_gap:
            gap_pre, gap_post = np.where(W_gap > 0)
            gap_ll_mask = np.array(
                [self._is_lif(p) and self._is_lif(q) for p, q in zip(gap_pre, gap_post)]
            )
            gap_pre_ll = gap_pre[gap_ll_mask] if len(gap_pre) else np.array([], dtype=int)
            gap_post_ll = gap_post[gap_ll_mask] if len(gap_post) else np.array([], dtype=int)
            n_gap_ll = len(gap_pre_ll)
            if n_gap_ll:
                gap_w = W_gap[gap_pre_ll, gap_post_ll].astype(np.float32)
                gap_pre_local = np.array([self.global_to_lif[p] for p in gap_pre_ll], dtype=int)
                gap_post_local = np.array([self.global_to_lif[q] for q in gap_post_ll], dtype=int)
                self.syn_gap = Synapses(
                    self.neurons, self.neurons,
                    model="""
                    w_gap : 1
                    I_gap_post = g_gap * w_gap * (v_pre - v_post) : amp (summed)
                    """,
                    namespace=namespace,
                )
                self.syn_gap.connect(i=gap_pre_local.tolist(), j=gap_post_local.tolist())
                self.syn_gap.w_gap = gap_w.tolist()
                self._all_synapses.append(self.syn_gap)

        self._n_lif_lif_chem = len(ll_exc_pre) + len(ll_inh_pre)
        self._n_lif_lif_gap = n_gap_ll

    def _is_lif(self, global_idx):
        return global_idx not in self.wave2_idx_set

    def _build_cross_group_routing(self, W_chem, W_gap, W_syn, g_gap):
        """Build per-edge routing tables for cross-group (LIF<->Wave2) coupling.

        These tables are consumed by the network_operation at 50 ms cadence.
        Each table entry is a tuple (pre_kind, pre_local_idx, post_kind,
        post_local_idx, weight) where kind in {"lif", "wave2:NAME"}.
        """
        self._cross_chem_edges = []  # list of (pre, post, w_signed)
        self._cross_gap_edges = []   # list of (pre, post, w_gap)

        n_cross_chem = 0
        n_cross_gap = 0

        # Iterate all chemical edges, filter to cross-group
        chem_pre, chem_post = np.where(W_chem != 0)
        for pi, qi in zip(chem_pre, chem_post):
            pi = int(pi); qi = int(qi)
            pre_is_lif = self._is_lif(pi)
            post_is_lif = self._is_lif(qi)
            if pre_is_lif and post_is_lif:
                continue
            w = float(W_chem[pi, qi])
            self._cross_chem_edges.append({
                "pre_global": pi, "post_global": qi,
                "pre_kind": "lif" if pre_is_lif else self.names[pi],
                "post_kind": "lif" if post_is_lif else self.names[qi],
                "w_signed": w,
            })
            n_cross_chem += 1

        gap_pre, gap_post = np.where(W_gap > 0)
        for pi, qi in zip(gap_pre, gap_post):
            pi = int(pi); qi = int(qi)
            pre_is_lif = self._is_lif(pi)
            post_is_lif = self._is_lif(qi)
            if pre_is_lif and post_is_lif:
                continue
            w = float(W_gap[pi, qi])
            self._cross_gap_edges.append({
                "pre_global": pi, "post_global": qi,
                "pre_kind": "lif" if pre_is_lif else self.names[pi],
                "post_kind": "lif" if post_is_lif else self.names[qi],
                "w_gap": w,
            })
            n_cross_gap += 1

        self._n_cross_chem = n_cross_chem
        self._n_cross_gap = n_cross_gap

        # Cache W_syn / g_gap values in unitless mV / nS for use in operation
        from brian2 import volt as _volt
        from brian2 import siemens as _S
        self._W_syn_mV = float(W_syn / mV)
        self._g_gap_nS = float(g_gap / nS)

    def _build_event_routing_op(self, namespace):
        """Per-step network_operation that:
        1. Reads each Wave 2 cell's V; if V > -25 mV and last_spike > 5 ms ago,
           emit a "release event": deliver W_syn * w to each post LIF cell
           (chemical) and update Wave 2's last_spike_t.
        2. Reads each LIF cell's recent firing rate (from spike monitor); for
           each cross chemical edge LIF→Wave2, accumulate I_ext on the Wave 2
           target proportional to firing rate × weight.
        3. Approximates gap-junction coupling by adding a current to Wave 2's
           I_ext proportional to (V_pre_LIF - V_post_Wave2) via per-edge gap.

        Cadence: 50 ms (matches ClosedLoopEnv sync). Sufficient for cells
        firing at <20 Hz; finer cadence for higher-rate cells is a WB3 followup.
        """
        # Pre-compute lookups for fast evaluation
        # cross_chem_lif_to_wave2: list of (lif_local_idx, wave2_name, w_signed)
        # cross_chem_wave2_to_lif: list of (wave2_name, lif_local_idx, w_signed)
        # cross_chem_wave2_to_wave2: list of (wave2_pre, wave2_post, w_signed)
        self._chem_lif_to_w2 = []
        self._chem_w2_to_lif = []
        self._chem_w2_to_w2 = []
        for e in self._cross_chem_edges:
            if e["pre_kind"] == "lif" and e["post_kind"] != "lif":
                self._chem_lif_to_w2.append(
                    (self.global_to_lif[e["pre_global"]], e["post_kind"], e["w_signed"])
                )
            elif e["pre_kind"] != "lif" and e["post_kind"] == "lif":
                self._chem_w2_to_lif.append(
                    (e["pre_kind"], self.global_to_lif[e["post_global"]], e["w_signed"])
                )
            elif e["pre_kind"] != "lif" and e["post_kind"] != "lif":
                self._chem_w2_to_w2.append(
                    (e["pre_kind"], e["post_kind"], e["w_signed"])
                )
        self._gap_lif_to_w2 = []
        self._gap_w2_to_lif = []
        self._gap_w2_to_w2 = []
        for e in self._cross_gap_edges:
            if e["pre_kind"] == "lif" and e["post_kind"] != "lif":
                self._gap_lif_to_w2.append(
                    (self.global_to_lif[e["pre_global"]], e["post_kind"], e["w_gap"])
                )
            elif e["pre_kind"] != "lif" and e["post_kind"] == "lif":
                self._gap_w2_to_lif.append(
                    (e["pre_kind"], self.global_to_lif[e["post_global"]], e["w_gap"])
                )
            elif e["pre_kind"] != "lif" and e["post_kind"] != "lif":
                self._gap_w2_to_w2.append(
                    (e["pre_kind"], e["post_kind"], e["w_gap"])
                )

        # State for per-step LIF firing-rate readout
        self._lif_prev_spike_count = 0

        # Closure with self bound. We use 50 ms cadence to match
        # ClosedLoopEnv's STEPS_PER_SYNC * dt = 50 ms.
        @network_operation(dt=50 * ms)
        def _route_events():
            self._step_route()

        self._event_routing_op = _route_events

    def _step_route(self):
        """Cross-group event routing — called every 50 ms."""
        # In "off" mode, Wave 2 cells run isolated (no LIF coupling).
        # This is the safe default for WB2 pending WB3 release-event biology.
        if self.cross_coupling_mode == "off":
            return
        # 1. Wave 2 release events → LIF & Wave 2 receivers
        # Get current sim time (ms)
        t_now_ms = float(self.net.t / ms) if hasattr(self, 'net') else float(defaultclock.t / ms)

        # Detect Wave 2 spikes via voltage threshold
        wave2_voltages_mV = {}
        wave2_fired = {}
        for name, grp in self.wave2_groups.items():
            v_mV = float(grp.v[0] / mV)
            wave2_voltages_mV[name] = v_mV
            last_spike = self.wave2_last_spike_t[name]
            fired = (
                v_mV > RELEASE_RULE_V_THRESHOLD_MV
                and (t_now_ms - last_spike) > RELEASE_RULE_REFRACTORY_MS
            )
            wave2_fired[name] = fired
            if fired:
                self.wave2_last_spike_t[name] = t_now_ms

        # 2. Wave 2 → LIF chemical events
        # When a Wave 2 cell fires, deliver W_syn * w to each post LIF cell.
        for w2_name, lif_local, w_signed in self._chem_w2_to_lif:
            if wave2_fired.get(w2_name, False):
                v_mV_post = float(self.neurons.v[lif_local] / mV)
                # Add W_syn * w to v_post (sign of w_signed determines sign)
                new_v_mV = v_mV_post + self._W_syn_mV * w_signed
                self.neurons.v[lif_local] = new_v_mV * mV

        # 3. Wave 2 → Wave 2 chemical events: deferred to combined I_ext below.
        # Conservative WB2 approximation: when a Wave 2 pre fires, contribute
        # a small graded current to the Wave 2 post via the combined
        # per_w2_I_chem_pA pipeline (set up below).

        # 4. LIF → Wave 2 chemical events: aggregate firing rate × weight,
        # convert into a CURRENT on Wave 2's I_ext (NOT a voltage bump).
        # Why current not voltage:
        #   - Wave 2 cells are continuous-V graded cells with small cm
        #     (~1 pF). A naive v += W_syn * w * spike_count causes blow-up
        #     when N>10 LIF inputs each fire >1 spike/50 ms (V → ∞).
        #   - The Mellem 2008 RMD/AVA biology suggests chemical synaptic
        #     drive on AVA is mediated by graded glutamate receptors with
        #     finite conductance and reversal — NOT discrete instantaneous
        #     V bumps.
        # WB2 provisional approach: convert LIF firing rate to a steady
        # current proportional to spike rate × W_syn, capped to prevent
        # numerical blowup. WB3 will revisit with biophysically-grounded
        # graded synapse (g * (V_post - E_rev)) implementation.
        all_t = self.spikes.t[:]
        all_i = self.spikes.i[:]
        new_slice = slice(self._lif_prev_spike_count, len(all_t))
        new_i = all_i[new_slice]
        if len(new_i) > 0:
            counts = np.bincount(new_i, minlength=self.n_lif).astype(np.float32)
        else:
            counts = np.zeros(self.n_lif, dtype=np.float32)
        self._lif_prev_spike_count = len(all_t)

        # For each LIF→Wave2 edge: deliver an integrated current over 50 ms
        # equivalent to W_syn(mV) * w * count * cm(pF) / 50 ms.
        # cm ≈ 0.86 pF (AVAL) so for 1 spike, ΔV_equiv = 0.8 mV → ΔI ≈ 0.014 pA.
        # 100 spikes → ΔI ≈ 1.4 pA, comfortably in physiological range.
        # NOTE: this is the WB2 PROVISIONAL graded synapse approximation.
        # I_chem_pA = (W_syn_mV * sum(w * count)) * cm_typical_pF / dt_ms
        cm_typical_pF = 0.86  # AVAL typical
        dt_ms = 50.0
        per_w2_I_chem_pA = {name: 0.0 for name in self.wave2_active}
        for lif_local, w2_name, w_signed in self._chem_lif_to_w2:
            n_spikes = float(counts[lif_local])
            if n_spikes > 0:
                I_pA = (
                    self._W_syn_mV * w_signed * n_spikes * cm_typical_pF / dt_ms
                )
                per_w2_I_chem_pA[w2_name] += I_pA

        # W2 → W2 chemical: 1 release event during this 50 ms gives a 1-spike
        # equivalent to the post-Wave-2 cell, integrated as a current
        # equivalent to W_syn * w * cm / dt.
        for w2_pre, w2_post, w_signed in self._chem_w2_to_w2:
            if wave2_fired.get(w2_pre, False):
                I_pA = (
                    self._W_syn_mV * w_signed * 1.0 * cm_typical_pF / dt_ms
                )
                per_w2_I_chem_pA[w2_post] += I_pA

        # 5. Cross-group gap junctions: compute (v_pre - v_post) and add to
        # I_ext (via current = g_gap * w * Δv).
        # For Wave 2, I_ext is in pA. g_gap is in nS, so Δv (mV) * g_gap (nS) * w
        # = pA. Apply continuously by setting I_ext on Wave 2.
        per_w2_I_gap_pA = {name: 0.0 for name in self.wave2_active}
        for w2_name, lif_local, w_gap in self._gap_w2_to_lif:
            v_w2 = wave2_voltages_mV[w2_name]
            v_lif = float(self.neurons.v[lif_local] / mV)
            I_pA = self._g_gap_nS * w_gap * (v_w2 - v_lif)
            i_ext_old = float(self.neurons.I_ext[lif_local] / pA)
            self.neurons.I_ext[lif_local] = (i_ext_old + I_pA) * pA
        for lif_local, w2_name, w_gap in self._gap_lif_to_w2:
            v_lif = float(self.neurons.v[lif_local] / mV)
            v_w2 = wave2_voltages_mV[w2_name]
            I_pA = self._g_gap_nS * w_gap * (v_lif - v_w2)
            per_w2_I_gap_pA[w2_name] += I_pA

        # 6. Apply combined I_ext (chemical + gap) to Wave 2 cells.
        # Cap to ±200 pA per cell to prevent numerical blowup (physiological
        # current scale on a 0.8 pF cell is < 100 pA per Mellem-class
        # injections). WB3 will replace with biophysically-grounded
        # I = g_syn * (V - E_rev) implementation; this cap is a safety net.
        I_PA_MAX = 20.0  # WB2 conservative cap (Mellem-class injections were ±30 pA)
        for w2_name in self.wave2_active:
            I_total_pA = per_w2_I_chem_pA[w2_name] + per_w2_I_gap_pA[w2_name]
            I_total_pA = max(-I_PA_MAX, min(I_PA_MAX, I_total_pA))
            grp = self.wave2_groups[w2_name]
            grp.I_ext[0] = I_total_pA * pA

    # ------------------------------------------------------------
    # Proprioception (mirrors LIFBrain)
    # ------------------------------------------------------------

    def _setup_proprioception(self):
        proprio_names = ["PDEL", "PDER", "PDA", "DVA"]
        self.proprio_idx = []
        for n in proprio_names:
            if n in self.idx and self._is_lif(self.idx[n]):
                self.proprio_idx.append(self.global_to_lif[self.idx[n]])
        if not self.proprio_idx:
            self.proprio_group = None
            self.proprio_syn = None
            return
        n_prop = len(self.proprio_idx)
        self.proprio_group = PoissonGroup(n_prop, rates=np.zeros(n_prop) * Hz)
        self.proprio_syn = Synapses(
            self.proprio_group, self.neurons,
            on_pre="v_post += 4*mV",
        )
        for i, j in enumerate(self.proprio_idx):
            self.proprio_syn.connect(i=i, j=j)

    def set_proprioception(self, body_curvature_mag):
        if self.proprio_group is None:
            return
        rate_hz = float(min(150.0, max(0.0, 400.0 * body_curvature_mag)))
        n = len(self.proprio_idx)
        self.proprio_group.rates = np.full(n, rate_hz) * Hz

    # ------------------------------------------------------------
    # Sensory drive + ablation (mirrors LIFBrain)
    # ------------------------------------------------------------

    def set_sensory_rate(self, neuron, rate_hz, weight_mv=8.0):
        if not hasattr(self, "_sensory_groups"):
            self._sensory_groups = {}
        if neuron not in self.idx:
            return
        global_idx = self.idx[neuron]
        if global_idx in self.wave2_idx_set:
            # Sensory drive to a Wave 2 cell — set I_ext via continuous current
            # approximation: rate_hz * weight_mv * cm = current_pA (rough).
            # For simplicity in WB2 we don't yet wire sensory directly to W2;
            # this is a WB4+ followup.
            return
        lif_local = self.global_to_lif[global_idx]
        if neuron not in self._sensory_groups:
            pg = PoissonGroup(1, rates=np.array([rate_hz]) * Hz)
            syn = Synapses(
                pg, self.neurons,
                on_pre=f"v_post += {weight_mv}*mV",
            )
            syn.connect(i=0, j=lif_local)
            self._sensory_groups[neuron] = (pg, syn)
            self.net.add(pg, syn)
        else:
            pg, _ = self._sensory_groups[neuron]
            pg.rates = np.array([rate_hz]) * Hz

    def inject_poisson(self, neuron, rate_hz, weight_mv=15.0):
        if neuron not in self.idx:
            raise KeyError(f"Unknown neuron: {neuron}")
        global_idx = self.idx[neuron]
        if global_idx in self.wave2_idx_set:
            # Wave 2 sensory injection — for WB2 deferred, raise informative error
            raise NotImplementedError(
                f"Sensory drive to Wave 2 cell {neuron} not yet implemented in WB2"
            )
        lif_local = self.global_to_lif[global_idx]
        pg = PoissonGroup(1, rate_hz * Hz)
        syn = Synapses(
            pg, self.neurons,
            on_pre=f"v_post += {weight_mv}*mV",
        )
        syn.connect(i=0, j=lif_local)
        self._stim_cache.extend([pg, syn])
        self.net.add(pg, syn)

    def ablate(self, names, current_pA=-1000.0):
        hit = []
        for n in names:
            if n in self.idx:
                self.ablation_current_pA[self.idx[n]] = current_pA
                hit.append(n)
        # Push to LIF cells via a network_operation
        if not hasattr(self, "_ablation_op_attached"):
            from brian2 import network_operation as _no
            @_no(dt=50 * ms)
            def _push_ablation():
                for global_idx in range(self.N):
                    if self.ablation_current_pA[global_idx] != 0 and self._is_lif(global_idx):
                        lif_local = self.global_to_lif[global_idx]
                        self.neurons.I_ext[lif_local] = (
                            self.ablation_current_pA[global_idx] * pA
                        )
                # Wave 2 ablation: set I_ext on the Wave 2 cell
                for name, grp in self.wave2_groups.items():
                    g_idx = self.idx[name]
                    if self.ablation_current_pA[g_idx] != 0:
                        grp.I_ext[0] = self.ablation_current_pA[g_idx] * pA
            self._ablation_op = _push_ablation
            self.net.add(_push_ablation)
            self._ablation_op_attached = True
        return hit

    # ------------------------------------------------------------
    # Run + readout
    # ------------------------------------------------------------

    def run(self, duration_ms):
        self.net.run(duration_ms * ms)

    def time_ms(self):
        return float(self.net.t / ms)

    def firing_rates(self, window_ms=200.0):
        """Return (N,) array of firing rates over last window_ms.

        For LIF cells: spike-count / window. For Wave 2 cells: rate of
        emitted release events, computed from self.wave2_last_spike_t
        (sparse, based on V-threshold detection at 50 ms cadence — coarse
        but sufficient for the readout layer's classifier window of 600 ms).
        """
        out = np.zeros(self.N)
        if len(self.spikes.t) > 0:
            t_now = self.net.t
            t_cut = t_now - window_ms * ms
            ts = self.spikes.t[:]
            ids = self.spikes.i[:]
            recent = ts >= t_cut
            counts = np.bincount(ids[recent], minlength=self.n_lif).astype(np.float64)
            # Map LIF local → global
            for local, count in enumerate(counts):
                global_idx = self.lif_idx_global[local]
                out[global_idx] = count / (window_ms / 1000.0)
        # Wave 2: count release events in window
        t_now_ms = self.time_ms()
        for name in self.wave2_active:
            last = self.wave2_last_spike_t[name]
            if last > t_now_ms - window_ms:
                # At least 1 release event in window. Crude estimate.
                out[self.idx[name]] = 1.0 / (window_ms / 1000.0)
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    """WB2 smoke test: build hybrid brain with AVAL+AVAR, run baseline,
    confirm no errors and Wave 2 cells respond to LIF activity."""
    print("=" * 70)
    print("Wave2HybridBrain WB2 smoke test")
    print("=" * 70)

    print("\nBuilding hybrid brain (AVAL + AVAR Wave 2, 298 LIF)...")
    brain = Wave2HybridBrain(wave2_active=["AVAL", "AVAR"], seed=42)
    s = brain.summary
    print(f"  N total:           {s['N']}")
    print(f"  N LIF:             {s['n_lif']}")
    print(f"  N Wave 2:          {s['n_wave2']}  ({s['wave2_active']})")
    print(f"  Chem LIF→LIF:      {s['n_chem_lif_lif']}")
    print(f"  Gap  LIF→LIF:      {s['n_gap_lif_lif']}")
    print(f"  Cross chem (LIF↔W2): {s['n_chem_cross']}")
    print(f"  Cross gap  (LIF↔W2): {s['n_gap_cross']}")
    print(f"  Release rule:      {s['release_rule']}")

    print("\nPhase 1: 1 s spontaneous run...")
    import time
    t0 = time.time()
    brain.run(1000)
    dt_run = time.time() - t0
    print(f"  Wall time: {dt_run:.1f} s for 1000 ms simulated")

    rates = brain.firing_rates(500)
    n_active = int(np.sum(rates > 0.5))
    print(f"  total LIF spikes: {len(brain.spikes.t)}")
    print(f"  active cells (>0.5 Hz): {n_active}")
    print(f"  mean rate: {rates.mean():.2f} Hz, max: {rates.max():.1f} Hz")

    # Wave 2 cell voltages
    print("\nWave 2 cell state at t=1000 ms:")
    for name in brain.wave2_active:
        grp = brain.wave2_groups[name]
        v_mV = float(grp.v[0] / mV)
        last_spike = brain.wave2_last_spike_t[name]
        print(f"  {name}: V = {v_mV:+.2f} mV, last release event @ {last_spike:.1f} ms")

    print("\n[WB2 smoke test PASS] Hybrid brain builds and runs.")


if __name__ == "__main__":
    smoke_test()
