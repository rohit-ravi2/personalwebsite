"""
Wave2HybridBrain — Phase δ WB2/WB3 hybrid Wave-2 + LIF brain.

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
  - WB2: skeleton + AVAL drop-in proof of concept (cross_coupling_mode="off").
  - WB3 (this revision, 2026-04-26): graded Boltzmann release rule
    (Wicks 1996 sigmoidal) with B2 sub-pattern: per-Synapses g_syn(t)
    decay, τ_syn = 10 ms. New cross_coupling="graded_b2" mode.
  - WB4: AIY pair extension.
  - WB5: RIM pair extension.
  - WB6: full multi-scenario validation.

WB3 cross-coupling release rule (graded_b2 mode):
  - Wave 2 → LIF: continuous σ(V_pre) Boltzmann sigmoid coupling via
    Brian2 (summed) cross-group Synapses, current `W_graded_I * w * σ`
    delivered to LIF v via dedicated `I_w2lif : amp (summed)` term.
    Excitatory (E_rev = 0 mV implicit through positive sign) and
    inhibitory (sign-flipped) edges share W_graded_I scale.
  - LIF → Wave 2: per-Synapses g_syn(t) state with τ_syn = 10 ms;
    LIF spike → g_syn += W_g * w (excitatory or inhibitory per edge sign);
    current = g_syn * (E_rev - v_post). Implemented as a numpy state
    array with run_regularly @ Wave 2 dt (0.025 ms) writing summed
    current to each Wave 2 cell's I_ext (since cell-builder NeuronGroups
    have I_ext as a free parameter, not as a summed-receiver). Native
    Brian2 (summed) on the Wave 2 NG would require modifying cell-builder
    factories — out of WB3 scope per spec; behavior is mathematically
    equivalent for τ_syn >> dt.
  - Wave 2 → Wave 2 (e.g., AVAL ↔ AVAR): per-edge graded current
    `W_graded_I * w * σ_pre` written to post W2 cell's I_ext via the
    same run_regularly path.
  - Soft-cap safety net: log warnings when |I_total per Wave 2| > 100 pA
    (no truncation); see SOFT_CAP_PA constant.

Per-cell-type readout API contract (canonical post-WB3 D7-followup):
  - LIF cells: `firing_rates(window_ms)` → spike-count / window in Hz.
  - Wave 2 cells (graded_b2 mode): `firing_rates(window_ms)` →
    σ-magnitude mean over window × 100 (Hz-equivalent rate proxy,
    matching `graded_brain.py output_rates()` line 378 precedent).
    For raw σ ∈ [0, 1] use `wave2_activities(window_ms)`.
  - Decision 7(a) σ>0.5 rising-threshold pseudo-spike emission is
    PRESERVED in `wave2_pseudo_spikes` for any consumer that explicitly
    inspects it, but NO LONGER drives `firing_rates()`. The legacy
    detector silently reported 0 Hz when σ saturated above threshold
    (the WB3 CP4 readout artifact); the σ-magnitude readout correctly
    reflects the saturated active state.

I/O contract preserved (same as LIFBrain):
  - Attributes: names, idx, N, neurons (LIF group, the only one ClosedLoopEnv
    indexes into directly), spikes (SpikeMonitor over LIF group), idx_to_*
    helpers.
  - Methods: run, time_ms, set_proprioception, set_sensory_rate,
    inject_poisson, ablate, firing_rates.
  - Wave2HybridBrain-specific extensions: wave2_activities(window_ms)
    for raw σ ∈ [0, 1] readout per Wave 2 cell; soft_cap_warning_count().
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
    ms, mV, nS, pF, pA, Hz, second, amp, siemens, prefs,
    seed as brian2_seed,
)
prefs.codegen.target = "cython"

import logging
import warnings

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
RELEASE_RULE_V_THRESHOLD_MV = -25.0   # AVAL/AVAR rest is ~-25 mV (Mellem 2008
                                       # quote; AVAR rest -24.22 mV per our
                                       # validation). Threshold at rest-equivalent.
RELEASE_RULE_REFRACTORY_MS = 5.0


# ===========================================================================
# WB3 graded Boltzmann release rule parameters (graded_b2 mode)
# ===========================================================================
# Decisions adjudicated 2026-04-26 in
# `wave2/artifacts/phase_delta_wb3_release_rule_options.md`.
# All seven defaults accepted. Two caveats apply:
#   - Caveat 1 (Decision 3): AIY/RIM V_half is anchored extrapolation from
#     Wave 2 cell-builder validation (option_b_aiy_results.json:303
#     baseline_pre_mV ≈ -55.2; option_b_rim_results.json:303 baseline_pre_mV
#     ≈ -43.3); NOT direct synaptic release literature. Sensitivity check
#     in CP3.
#   - Caveat 2 (Decision 4): W_graded_I starting at 0.3 pA per Mellem 2008
#     -30/+30 pA injection range / typical Σ|w|·σ ~100 = 0.3 pA per unit
#     weight. CP4 may retune empirically if AVA Δ peri-touch <+5 Hz.

# Per-cell-class V_half (mV): midpoint of Wicks-style σ(V) Boltzmann.
# AVAL/AVAR: anchored to Mellem 2008 -20 to -30 mV rest range; midpoint -25.
# AIY/RIM: anchored to cell-builder validation V_rest (Caveat 1 — extrapolation).
WB3_V_HALF_MV = {
    "AVAL": -25.0,    # Mellem 2008 quote (verified)
    "AVAR": -25.0,    # Mellem 2008 quote (verified)
    "AIYL": -55.0,    # cell-builder validation V_rest -55.2 mV (extrapolation)
    "AIYR": -55.0,    # same as AIYL
    "RIML": -43.0,    # cell-builder validation V_rest -43.3 mV (extrapolation)
    "RIMR": -43.0,    # same as RIML
}

# Per-cell-class slope k (mV) — Wicks 1996 sigmoidal slope.
# k = 6 mV uniformly; matches graded_brain.py:71 PARAMS["k_half"].
WB3_K_MV = {name: 6.0 for name in WB3_V_HALF_MV}

# V_half for LIF cells (used when LIF→W2 σ-coupling is needed; not used in B2).
# B2 uses g_syn(t) state, no σ for LIF.
WB3_LIF_V_HALF_MV = -25.0
WB3_LIF_K_MV = 6.0

# W_graded_I — Mellem-calibrated starting point (Decision 4 default).
# 0.3 pA per unit weight: Mellem ±30 pA injection / Σw·σ ~100 at saturation.
# Caveat 2: retune in CP4 if AVA Δ peri-touch <+5 Hz; document trajectory.
WB3_W_GRADED_I_PA_DEFAULT = 0.3

# E_rev (mV) for LIF→W2 conductance-based current g_syn * (E_rev - v_post).
# vertebrate convention; not directly measured for C. elegans interneurons
# (Decision 5; Wicks 1996 V_RANGE = -35 mV is incompatible with this
# interpretation, which is why graded_brain.py also uses 0/-70).
WB3_E_REV_EXC_MV = 0.0
WB3_E_REV_INH_MV = -70.0

# τ_syn (ms) for LIF→W2 g_syn(t) decay (B2 sub-pattern).
WB3_TAU_SYN_MS = 10.0

# W_g — peak conductance kick per LIF spike per unit weight.
# Calibrate so a single W_g unit kick produces order-of-magnitude
# response comparable to LIF→LIF voltage bump. With τ_syn=10 ms,
# E_exc=0 mV, V_w2=-25 mV at rest, peak current per spike ≈ W_g · 25 mV.
# To match W_graded_I scale (~0.3-1 pA at saturation), W_g ≈ 0.04-0.16 nS
# delivers ~1-4 pA peak per single spike per unit weight.
WB3_W_G_NS_DEFAULT = 0.05    # 50 pS per unit weight per spike

# Soft-cap safety net (Decision 6 default ii): log warning when |I_total per
# Wave 2| > SOFT_CAP_PA (do NOT truncate).
WB3_SOFT_CAP_PA = 100.0

# Pseudo-spike emission threshold (Decision 7 default a): σ > 0.5 rising
# crossing emits a pseudo-spike for the cell, preserving firing_rates() API.
WB3_SIGMA_SPIKE_THRESHOLD = 0.5
WB3_SIGMA_POLL_DT_MS = 10.0   # σ poll cadence (matches graded_brain.py:268)


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

    Per-cell-type readout API (canonical post-WB3 D7-followup):
      - `firing_rates(window_ms)` returns (N,) array; LIF cells report
        spike-count/window (Hz), Wave 2 cells report σ-magnitude mean
        × 100 (Hz-equivalent rate proxy matching graded_brain.py
        output_rates() line 378). Resolves WB3 CP4 readout artifact
        (rising-threshold pseudo-spike rate → 0 in σ-saturated regime).
      - `wave2_activities(window_ms)` returns dict[str, float] of raw
        σ ∈ [0, 1] mean for each active Wave 2 cell.

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
        cross_coupling=None,
        W_graded_I_pA=WB3_W_GRADED_I_PA_DEFAULT,
        W_g_nS=WB3_W_G_NS_DEFAULT,
        v_half_overrides=None,
    ):
        """
        cross_coupling_mode : str  (legacy WB2)
            "off" (default for WB2 — Wave 2 cells run isolated, no LIF coupling).
                  This is the SAFE state pending WB3 release-event biology review.
            "naive_voltage_bumps" — instantaneous v += W_syn*w on each side
                  (NOT recommended — causes V-blowup on Wave 2 due to small cm).
            "graded_current_capped" — current-based coupling capped at ±20 pA
                  per cell (WB2 provisional; replaceable by WB3).

        cross_coupling : str  (WB3+; preferred; takes precedence over
                              cross_coupling_mode if non-None)
            "off" — equivalent to cross_coupling_mode="off".
            "graded_b2" (WB3 default release rule, adjudicated 2026-04-26;
            readout updated WB3 D7-followup):
                graded Boltzmann release (Wicks 1996 sigmoidal) on Wave 2 →
                LIF via Brian2 (summed) Synapses; per-Synapses g_syn(t)
                decay (τ_syn = 10 ms) on LIF → Wave 2 written to Wave 2's
                I_ext at fast cadence; soft cap ±100 pA + log warnings.
                Wave 2 cell readout: σ-magnitude mean × 100 in
                firing_rates() (Hz-equivalent proxy matching
                graded_brain.py output_rates() precedent); raw σ via
                wave2_activities(). Legacy σ > 0.5 rising-threshold
                pseudo-spike events preserved in self.wave2_pseudo_spikes
                but no longer drive firing_rates() (Decision 7 artifact:
                detector blind in saturated regime).

        W_graded_I_pA : float
            Per-unit-weight current scale (pA) for σ-modulated coupling.
            Default 0.3 pA per Mellem 2008 calibration (Decision 4). May
            be retuned in CP4 per Caveat 2.

        W_g_nS : float
            Per-unit-weight conductance kick (nS) for LIF→Wave 2 g_syn(t)
            on each LIF spike. Default 0.05 nS (50 pS).

        v_half_overrides : dict[str, float] | None
            Optional override map {cell_name: V_half_mV} for sensitivity
            analysis (Caveat 1 + CP3.2). Default None uses WB3_V_HALF_MV.
        """
        # WB3 cross_coupling takes precedence; legacy cross_coupling_mode
        # remains supported via "off"/"naive_voltage_bumps"/"graded_current_capped".
        if cross_coupling is not None:
            self.cross_coupling = cross_coupling
            self.cross_coupling_mode = (
                "off" if cross_coupling in ("off", "graded_b2") else cross_coupling
            )
        else:
            self.cross_coupling = cross_coupling_mode
            self.cross_coupling_mode = cross_coupling_mode
        self.W_graded_I_pA = float(W_graded_I_pA)
        self.W_g_nS = float(W_g_nS)
        self._v_half_overrides = dict(v_half_overrides or {})
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

        # WB3: in graded_b2 mode, add summed-receiver variables for
        # cross-group W2→LIF coupling. Brian2 (summed) idiom allows ONE
        # Synapses object to write to a given summed variable. To allow
        # both excitatory and inhibitory contributions from each W2 source
        # cell, we declare per-source per-sign summed variables and sum
        # them in the dv/dt equation. For N W2 cells × 2 signs = 2N extra
        # variables (e.g. 4 for AVAL+AVAR; 12 for the full AVAL/AVAR/AIY/
        # RIM set). Each is `: amp` free on the post-NG; the (summed)
        # declaration lives on the Synapses model side.
        if self.cross_coupling == "graded_b2":
            sum_terms = []
            decl_lines = []
            for name in wave2_active:
                for sign_tag in ("e", "i"):
                    var = f"I_w2lif_{name.lower()}_{sign_tag}"
                    sum_terms.append(var)
                    decl_lines.append(f"            {var} : amp")
            sum_expr = " + ".join(sum_terms) if sum_terms else "0*amp"
            decl_block = "\n".join(decl_lines)
            eqs = f"""
            dv/dt = (v_rest - v)/tau + (I_gap + I_ext + {sum_expr})/C_mem
                    + noise_sigma * xi / sqrt(tau) : volt (unless refractory)
            I_gap : amp
            I_ext : amp
{decl_block}
            """
        else:
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

        if self.cross_coupling == "graded_b2":
            # WB3 graded Boltzmann release: native (summed) Wave 2 → LIF
            # Synapses; per-Synapses g_syn(t) state for LIF → Wave 2 written
            # to W2 I_ext via run_regularly. See class docstring + WB3
            # findings for biological grounding (Wicks 1996, Mellem 2008,
            # Lockery & Goodman 2009).
            self._build_graded_b2_cross_synapses(namespace, include_gap)
            # Pseudo-spike σ-poll machinery for Wave 2 (Decision 7 (a)).
            self._build_wave2_sigma_poll()
            # Soft-cap warning logger (Decision 6 (ii)).
            self._soft_cap_warnings = []
            # Build the per-step W2 I_ext writer (combines per-edge g_syn(t)
            # decay/kicks + W2→W2 σ-coupling + cross gap; soft-cap warnings).
            self._build_graded_b2_w2_current_writer()
        else:
            # Per-step network operation: read Wave 2 voltages, detect events,
            # deliver to LIF receivers; read LIF firing rates, deliver to
            # Wave 2 I_ext. Legacy WB2 path.
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
        if self.cross_coupling == "graded_b2":
            components.extend(self._cross_synapses_graded_b2)
            components.append(self._sigma_poll_op)
            components.append(self._w2_current_writer_op)
        else:
            components.append(self._event_routing_op)
        if self.proprio_group is not None:
            components.append(self.proprio_group)
            components.append(self.proprio_syn)
        self.net = Network(*components)

        # Track other dynamic Synapses (sensory, ablation push, etc.)
        self._stim_cache = []

        if self.cross_coupling == "graded_b2":
            release_rule_summary = (
                "graded_b2: Wicks 1996 sigmoidal release (Wave 2→LIF native "
                "(summed); LIF→Wave 2 per-Synapses g_syn(t), τ_syn=10 ms); "
                f"W_graded_I={self.W_graded_I_pA} pA; W_g={self.W_g_nS} nS; "
                "soft cap ±100 pA + log warnings; W2 readout = σ-magnitude "
                "× 100 (D7-followup; matches graded_brain.py output_rates())."
            )
        else:
            release_rule_summary = (
                f"{self.cross_coupling_mode} (legacy WB2 path)"
            )
        self.summary = dict(
            N=self.N,
            n_lif=self.n_lif,
            n_wave2=len(wave2_active),
            wave2_active=list(wave2_active),
            n_chem_lif_lif=int(self._n_lif_lif_chem),
            n_gap_lif_lif=int(self._n_lif_lif_gap),
            n_chem_cross=int(self._n_cross_chem),
            n_gap_cross=int(self._n_cross_gap),
            release_rule=release_rule_summary,
            per_edge_glu_signs=self._using_per_edge_signs,
            cross_coupling=self.cross_coupling,
            W_graded_I_pA=self.W_graded_I_pA,
            W_g_nS=self.W_g_nS,
            v_half_overrides=dict(self._v_half_overrides),
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

    # ------------------------------------------------------------
    # WB3 graded_b2 cross-coupling (Phase δ WB3, 2026-04-26)
    # ------------------------------------------------------------

    def _resolve_v_half_mV(self, name):
        """Resolve V_half (mV) for a Wave 2 cell, applying CP3.2 sensitivity
        overrides if present. Caveat 1 (AIY/RIM extrapolation) — overrides
        feed sensitivity sweep at ±5 mV around cellular-anchored default."""
        if name in self._v_half_overrides:
            return float(self._v_half_overrides[name])
        return float(WB3_V_HALF_MV[name])

    def _build_graded_b2_cross_synapses(self, namespace, include_gap):
        """WB3 cross-group Synapses for graded_b2 mode.

        Builds three groups of native Brian2 Synapses:

        1. Wave 2 → LIF chemical (excitatory + inhibitory, per W2 source NG):
           One Synapses per (W2 source NG, sign-class). Model:

               w : 1
               sigma_pre = 1.0/(1.0 + exp(-(v_pre - v_half_mV*mV)/(k_mV*mV))) : 1
               I_w2lif_<name>_<sign>_post = sign_factor * W_graded_I * w * sigma_pre
                                            : amp (summed)

           v_half_mV/k_mV are namespace constants per W2 cell type.

        2. LIF → Wave 2 chemical: per-Synapses g_syn(t) state with τ_syn,
           kicked by on_pre on LIF spikes. Wave 2 NG cell-builder eqs do
           NOT have a summed-receiver variable; we maintain g_syn arrays
           in numpy and write the summed current to W2 I_ext via
           _build_graded_b2_w2_current_writer. Each LIF→W2 chemical edge
           gets a per-edge g_syn entry plus a sign tag (+1 exc / -1 inh).
           These are stored in self._lifw2_state, populated here.

        3. Wave 2 → Wave 2 chemical: same σ-pre coupling as W2→LIF, but
           target W2 cells. Recorded in self._w2w2_edges and applied via
           the same writer (since target W2 cells lack summed receivers).

        Cross-group gap junctions (LIF↔W2): same g_gap*(v_pre - v_post)
        formula as LIF↔LIF. Built natively where LIF is post (pure summed),
        and via the writer where W2 is post.
        """
        # Per-cell V_half / k namespace constants — computed per-call so
        # that v_half_overrides apply.
        # We construct a dict with "_w2_<name>_v_half_mV" / "_w2_<name>_k_mV"
        # keys to be resolved within the Synapses model strings.
        ns_extra = {}
        for w2_name in self.wave2_active:
            ns_extra[f"v_half_{w2_name.lower()}_mV"] = self._resolve_v_half_mV(w2_name)
            ns_extra[f"k_{w2_name.lower()}_mV"] = float(WB3_K_MV[w2_name])
        ns_extra["W_graded_I_pA"] = self.W_graded_I_pA
        ns_extra["W_g_nS"] = self.W_g_nS
        ns_extra["tau_syn"] = WB3_TAU_SYN_MS * ms

        # Bake the WB3 namespace onto each Synapses' namespace explicitly.
        # We do not mutate the LIF NG namespace because Brian2 binds a
        # NeuronGroup's namespace at construction.

        self._cross_synapses_graded_b2 = []

        # ----- (1) Wave 2 → LIF chemical, per W2 cell, per sign -----
        # We pre-categorize cross_chem_edges into per-(w2_source, sign).
        for w2_name in self.wave2_active:
            edges_e = []  # (w2_local_idx_always_0, lif_local_idx, |w|)
            edges_i = []
            for e in self._cross_chem_edges:
                if e["pre_kind"] == w2_name and e["post_kind"] == "lif":
                    lif_local = self.global_to_lif[e["post_global"]]
                    w_signed = e["w_signed"]
                    if w_signed > 0:
                        edges_e.append((0, lif_local, abs(w_signed)))
                    elif w_signed < 0:
                        edges_i.append((0, lif_local, abs(w_signed)))
            for sign_tag, edges, sign_factor in (
                ("e", edges_e, +1.0),
                ("i", edges_i, -1.0),
            ):
                summed_var = f"I_w2lif_{w2_name.lower()}_{sign_tag}"
                # Build Synapses; use namespace baked with this cell's
                # v_half / k. We write the (summed) variable directly.
                v_half_key = f"v_half_{w2_name.lower()}_mV"
                k_key = f"k_{w2_name.lower()}_mV"
                local_ns = {
                    **ns_extra,
                }
                model_str = (
                    "w : 1\n"
                    f"{summed_var}_post = "
                    f"{sign_factor:+.1f} * W_graded_I_pA * pA * w * "
                    f"(1.0 / (1.0 + exp(-(v_pre - {v_half_key}*mV) / ({k_key}*mV))))"
                    " : amp (summed)\n"
                )
                syn = Synapses(
                    self.wave2_groups[w2_name],
                    self.neurons,
                    model=model_str,
                    namespace=local_ns,
                )
                if len(edges):
                    pre_idx = [pe[0] for pe in edges]
                    post_idx = [pe[1] for pe in edges]
                    weights = [pe[2] for pe in edges]
                    syn.connect(i=pre_idx, j=post_idx)
                    syn.w = weights
                self._cross_synapses_graded_b2.append(syn)

        # ----- (2) LIF → Wave 2 chemical: per-Synapses g_syn(t) state
        # in numpy. Track per-edge g_syn (nS) decaying with τ_syn; on each
        # LIF spike, kick g_syn += W_g * w (with sign mapped to E_rev).
        # The current per W2 cell = Σ g_syn * (E_rev - v_post) is computed
        # in _build_graded_b2_w2_current_writer.
        self._lifw2_state = {}   # name -> dict(pre_lif_local, weight, sign_factor, g_syn)
        for w2_name in self.wave2_active:
            pre_arr = []
            w_arr = []
            sign_arr = []
            for e in self._cross_chem_edges:
                if e["pre_kind"] == "lif" and e["post_kind"] == w2_name:
                    pre_arr.append(self.global_to_lif[e["pre_global"]])
                    w_arr.append(abs(e["w_signed"]))
                    sign_arr.append(+1 if e["w_signed"] > 0 else -1)
            self._lifw2_state[w2_name] = {
                "pre_lif_local": np.array(pre_arr, dtype=np.int32),
                "weight": np.array(w_arr, dtype=np.float64),
                "sign": np.array(sign_arr, dtype=np.int8),
                "g_syn_nS": np.zeros(len(pre_arr), dtype=np.float64),
            }

        # Track the LIF spike count consumed so far (to identify new spikes
        # each writer step).
        self._lifw2_prev_spike_count = 0

        # ----- (3) Wave 2 → Wave 2 chemical: σ-pre coupling, target is W2
        # cell; cell-builder NG lacks summed-receiver variables, so we
        # record edges and apply via the writer using direct V reads.
        self._w2w2_edges = []
        for e in self._cross_chem_edges:
            if (
                e["pre_kind"] != "lif"
                and e["post_kind"] != "lif"
                and e["pre_kind"] != e["post_kind"]
            ):
                self._w2w2_edges.append({
                    "pre": e["pre_kind"],
                    "post": e["post_kind"],
                    "w_signed": e["w_signed"],
                })

        # ----- (4) Cross-group gap junctions: LIF↔W2.
        # LIF post (W2 → LIF gap): native (summed) into LIF I_gap.
        # W2 post (LIF → W2 gap; W2 → W2 gap): record for writer.
        self._gap_lif_post_per_w2 = {}    # name -> list of (pre_local_idx_in_w2_NG=0, lif_local, w)
        self._gap_w2_post_records = []    # for writer: dict(post W2 name, src kind, src loc, w)
        if include_gap:
            for e in self._cross_gap_edges:
                if e["pre_kind"] != "lif" and e["post_kind"] == "lif":
                    name = e["pre_kind"]
                    lif_local = self.global_to_lif[e["post_global"]]
                    self._gap_lif_post_per_w2.setdefault(name, []).append(
                        (0, lif_local, e["w_gap"])
                    )
                elif e["pre_kind"] == "lif" and e["post_kind"] != "lif":
                    self._gap_w2_post_records.append({
                        "post": e["post_kind"],
                        "src_kind": "lif",
                        "src_lif_local": self.global_to_lif[e["pre_global"]],
                        "w_gap": e["w_gap"],
                    })
                elif (e["pre_kind"] != "lif" and e["post_kind"] != "lif"
                      and e["pre_kind"] != e["post_kind"]):
                    self._gap_w2_post_records.append({
                        "post": e["post_kind"],
                        "src_kind": e["pre_kind"],
                        "w_gap": e["w_gap"],
                    })

        # Build the W2→LIF gap Synapses (one per W2 source cell) using
        # native (summed) into LIF's existing I_gap. Note: LIF's I_gap
        # is currently written by syn_gap (LIF→LIF gap, summed). Brian2
        # only allows ONE Synapses to write a given (summed) variable on
        # a target NG. Therefore we use a NEW summed variable I_gap_w2
        # per W2 source on LIF; if no W2→LIF gaps exist for a cell, skip.
        # However, since I_gap was declared as `: amp` free in the LIF
        # eqs (which doesn't reference I_gap_w2), we need to either rebuild
        # the LIF eqs to include I_gap_w2 in dv/dt, OR fold W2→LIF gaps
        # into the writer too. To keep the LIF eqs change minimal and
        # avoid a second eqs reconstruction loop, we put W2→LIF gap into
        # the writer as well (writer reads LIF V, writes back to LIF I_ext).
        # This loses some Brian2 nativity for gap junctions but is
        # mathematically equivalent because I_ext is already an additive
        # current term in the LIF dv/dt.
        # (W2→LIF gap remains a low-volume edge stream — 186 cross gaps
        # for the AVAL/AVAR pair vs 2002 LIF↔LIF gaps that are already
        # native.)

    def _build_graded_b2_w2_current_writer(self):
        """run_regularly @ Wave 2 dt that:

        1. Reads W2 voltages V_pre, V_post.
        2. For each W2 cell, decays per-edge g_syn(t) by exp(-dt/τ_syn).
        3. Reads new LIF spikes since last call; for each spike on a LIF
           cell that pre-syns to a W2 cell, kicks the corresponding
           per-edge g_syn += W_g_nS * w (sign-aware with E_rev_exc/inh).
        4. Computes per-W2 summed current:
              I_lifw2 = Σ_e g_syn_e * (E_rev_e - V_w2_post)
              I_w2w2  = Σ_e (sign * W_graded_I) * w * σ(V_pre)
              I_gap_w2_post = Σ (g_gap * w_gap * (V_pre - V_w2))
        5. Optionally writes back to LIF I_ext for W2→LIF gap junctions
           (folded here for consistency since LIF I_gap is already a
           native (summed) target for LIF→LIF).
        6. Soft-cap warning: if |I_total per W2| > SOFT_CAP_PA, log warning
           (NO truncation per Decision 6 (ii)).
        7. Writes I_total to W2's I_ext.

        Cadence: Wave 2 cell dt (default Brian2 defaultclock 0.1 ms; cell
        builder uses 0.025 ms internally for clamping). To keep dt mismatch
        per spec: we run this writer at 0.1 ms (LIF dt) — matches Brian2
        defaultclock, fine-grained enough for τ_syn = 10 ms decay
        (|exp(-0.1/10) - (1 - 0.01)| ≈ 5e-5 step error per dt). The W2 cell's
        own internal 0.025 ms clamp clock is preserved.
        """
        # Pre-compute per-W2 cached current values for efficiency.
        # E_rev arrays per LIF→W2 edge: + → E_rev_exc, − → E_rev_inh.
        for name, st in self._lifw2_state.items():
            st["E_rev_mV"] = np.where(
                st["sign"] > 0, WB3_E_REV_EXC_MV, WB3_E_REV_INH_MV
            ).astype(np.float64)

        # Decay multiplier per writer step
        # (computed lazily so we use the correct defaultclock dt at run-time).
        self._writer_dt_ms_cached = None
        self._writer_decay_factor = None

        @network_operation(dt=0.1 * ms)
        def _write_w2_currents():
            self._write_w2_currents_step()
        self._w2_current_writer_op = _write_w2_currents

    def _write_w2_currents_step(self):
        """Inner step routine for graded_b2 W2 I_ext writer.

        Called every 0.1 ms (network_operation cadence).
        """
        dt_ms = 0.1
        if self._writer_dt_ms_cached != dt_ms:
            self._writer_dt_ms_cached = dt_ms
            self._writer_decay_factor = float(np.exp(-dt_ms / WB3_TAU_SYN_MS))

        # 1. Read W2 voltages (mV)
        v_w2_mV = {}
        for name, grp in self.wave2_groups.items():
            v_w2_mV[name] = float(grp.v[0] / mV)

        # 2. Decay all per-edge g_syn(t)
        for st in self._lifw2_state.values():
            st["g_syn_nS"] *= self._writer_decay_factor

        # 3. Apply LIF spike kicks
        all_t = self.spikes.t[:]
        all_i = self.spikes.i[:]
        new_slice = slice(self._lifw2_prev_spike_count, len(all_t))
        new_spike_lif_local = all_i[new_slice]
        if len(new_spike_lif_local):
            # Build a mask of fired LIF cells (any cell that fired at least
            # 1 spike in this slice). Each spike contributes one kick.
            # For simplicity (and rate-faithfulness), we count spikes per
            # LIF cell and apply that many kicks.
            counts = np.bincount(new_spike_lif_local, minlength=self.n_lif).astype(np.float64)
            for name, st in self._lifw2_state.items():
                if len(st["pre_lif_local"]) == 0:
                    continue
                # For each edge, kick g_syn += W_g * w * count (per pre)
                kicks = self.W_g_nS * st["weight"] * counts[st["pre_lif_local"]]
                # Sign-aware: excitatory vs inhibitory share W_g; sign is
                # encoded via E_rev (st["E_rev_mV"]). All edges kick
                # additively. Conductance is positive; sign of effect
                # comes from (E_rev - V) driving force.
                st["g_syn_nS"] += kicks
        self._lifw2_prev_spike_count = len(all_t)

        # 4. Compute per-W2 summed currents
        i_lifw2_pA = {name: 0.0 for name in self.wave2_active}
        i_w2w2_pA = {name: 0.0 for name in self.wave2_active}
        i_gap_w2_pA = {name: 0.0 for name in self.wave2_active}
        i_gap_lif_pA = np.zeros(self.n_lif, dtype=np.float64)

        # 4a. LIF → W2 chemical (g_syn(t) * (E_rev - V_post))
        for name, st in self._lifw2_state.items():
            if len(st["g_syn_nS"]) == 0:
                continue
            v_post_mV = v_w2_mV[name]
            # Current per edge: g_syn (nS) * (E_rev - V) (mV) = pA
            per_edge_pA = st["g_syn_nS"] * (st["E_rev_mV"] - v_post_mV)
            i_lifw2_pA[name] = float(per_edge_pA.sum())

        # 4b. W2 → W2 chemical (σ-modulated; instantaneous, no g_syn state)
        for edge in self._w2w2_edges:
            v_pre = v_w2_mV[edge["pre"]]
            v_half = self._resolve_v_half_mV(edge["pre"])
            k = float(WB3_K_MV[edge["pre"]])
            sigma = 1.0 / (1.0 + np.exp(-(v_pre - v_half) / k))
            sign_factor = +1.0 if edge["w_signed"] > 0 else -1.0
            mag = abs(edge["w_signed"])
            i_w2w2_pA[edge["post"]] += (
                sign_factor * self.W_graded_I_pA * mag * sigma
            )

        # 4c. Cross-group gap junctions
        # LIF → W2 (W2 post, LIF source)
        # W2 → W2 (different W2 NGs)
        for rec in self._gap_w2_post_records:
            v_post = v_w2_mV[rec["post"]]
            if rec["src_kind"] == "lif":
                v_src = float(self.neurons.v[rec["src_lif_local"]] / mV)
            else:
                v_src = v_w2_mV[rec["src_kind"]]
            i_gap_w2_pA[rec["post"]] += (
                self._g_gap_nS * rec["w_gap"] * (v_src - v_post)
            )

        # W2 → LIF (LIF post, W2 source); fold into LIF I_ext additive
        for w2_name, edges in self._gap_lif_post_per_w2.items():
            v_w2 = v_w2_mV[w2_name]
            for _, lif_local, w_gap in edges:
                v_lif = float(self.neurons.v[lif_local] / mV)
                i_gap_lif_pA[lif_local] += self._g_gap_nS * w_gap * (v_w2 - v_lif)

        # 5. Soft-cap warning + I_ext write to W2
        for name in self.wave2_active:
            i_total_pA = (
                i_lifw2_pA[name] + i_w2w2_pA[name] + i_gap_w2_pA[name]
            )
            if abs(i_total_pA) > WB3_SOFT_CAP_PA:
                # Soft cap (Decision 6 (ii)): log + DO NOT TRUNCATE.
                self._soft_cap_warnings.append({
                    "t_ms": float(self.net.t / ms) if hasattr(self, "net") else 0.0,
                    "cell": name,
                    "I_total_pA": i_total_pA,
                    "I_lifw2_pA": i_lifw2_pA[name],
                    "I_w2w2_pA": i_w2w2_pA[name],
                    "I_gap_w2_pA": i_gap_w2_pA[name],
                })
            grp = self.wave2_groups[name]
            grp.I_ext[0] = i_total_pA * pA

        # 6. Apply W2→LIF gap currents to LIF cells (additive on top of
        # whatever I_ext was already set by ablation/sensory).
        # We do this LAST and we ADD to current I_ext, but I_ext on LIF
        # is ALSO used for ablation forcing. To preserve ablation while
        # adding gap junction currents, we maintain a separate baseline
        # via `self._lif_i_ext_baseline_pA` (set by ablation) and write
        # baseline + gap_contribution each step.
        if hasattr(self, "ablation_current_pA"):
            # Compute LIF baseline I_ext from ablation_current_pA mapped
            # into LIF-local indices. Wave 2 cells already have their
            # ablation handled at I_ext set above (overwritten if cell is
            # ablated; ablation supersedes).
            baseline = np.zeros(self.n_lif, dtype=np.float64)
            for global_idx in range(self.N):
                if (
                    self.ablation_current_pA[global_idx] != 0
                    and self._is_lif(global_idx)
                ):
                    baseline[self.global_to_lif[global_idx]] = (
                        self.ablation_current_pA[global_idx]
                    )
            # Sensory injections via _sensory_groups and inject_poisson
            # use Brian2 PoissonGroup → on_pre v_post += w*mV; they don't
            # touch I_ext, so no conflict.
            self.neurons.I_ext = (baseline + i_gap_lif_pA) * pA

    def _build_wave2_sigma_poll(self):
        """Wave 2 σ-magnitude history recorder (canonical WB3 D7-followup
        readout) + legacy σ>0.5 rising-threshold pseudo-spike emission.

        Polls each Wave 2 cell's V every WB3_SIGMA_POLL_DT_MS, computes σ
        from V_half/k of that cell type, and:

        1. Records (t_ms, σ) into per-cell time-series buffers
           `self._wave2_sigma_history[name] = (times_ms, sigmas)` —
           consumed by `wave2_activities(window_ms)` and the Wave 2
           branch of `firing_rates(window_ms)` for σ-magnitude readout
           (matches `graded_brain.py output_rates()` line 378 precedent:
           windowed σ mean × 100 for Hz-equivalent rate proxy).

        2. (Legacy, Decision 7(a)) Detects σ > 0.5 rising-threshold
           crossings → appends event time to `self.wave2_pseudo_spikes
           [name]`. Preserved for backward compatibility with consumers
           that explicitly inspect `wave2_pseudo_spikes`. NOT consumed
           by `firing_rates()` anymore — that uses the σ-magnitude
           readout. The rising-threshold detector returns 0 events when
           σ saturates above 0.5 (the WB3 CP4 artifact); the σ-magnitude
           readout correctly reflects the saturated active state.

        Storage choice: per-cell python lists (times, sigmas). Cheap;
        appended at WB3_SIGMA_POLL_DT_MS = 10 ms cadence, so a 30 s run
        accumulates 3000 entries per cell — negligible memory.
        """
        self.wave2_pseudo_spikes = {name: [] for name in self.wave2_active}
        self._wave2_last_sigma = {name: 0.0 for name in self.wave2_active}
        # σ-magnitude history per cell: list of (t_ms, sigma) sampled at
        # WB3_SIGMA_POLL_DT_MS cadence. Lists for cheap append; converted
        # to numpy by the readout consumers when needed.
        self._wave2_sigma_history = {
            name: {"t_ms": [], "sigma": []} for name in self.wave2_active
        }

        @network_operation(dt=WB3_SIGMA_POLL_DT_MS * ms)
        def _poll_sigma():
            t_ms = float(self.net.t / ms) if hasattr(self, "net") else 0.0
            for name, grp in self.wave2_groups.items():
                v_mV = float(grp.v[0] / mV)
                v_half = self._resolve_v_half_mV(name)
                k = float(WB3_K_MV[name])
                sigma = 1.0 / (1.0 + np.exp(-(v_mV - v_half) / k))
                # Record σ history (canonical readout substrate).
                hist = self._wave2_sigma_history[name]
                hist["t_ms"].append(t_ms)
                hist["sigma"].append(sigma)
                # Legacy rising-threshold pseudo-spike detector — preserved
                # but no longer drives firing_rates(). Kept for any
                # consumer that explicitly inspects wave2_pseudo_spikes.
                last = self._wave2_last_sigma[name]
                if sigma > WB3_SIGMA_SPIKE_THRESHOLD and last <= WB3_SIGMA_SPIKE_THRESHOLD:
                    self.wave2_pseudo_spikes[name].append(t_ms)
                    self.wave2_last_spike_t[name] = t_ms
                self._wave2_last_sigma[name] = sigma

        self._sigma_poll_op = _poll_sigma

    # ------------------------------------------------------------
    # Legacy WB2 event routing
    # ------------------------------------------------------------

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
        # In graded_b2 mode the W2 current writer already reads
        # self.ablation_current_pA every 0.1 ms and writes it to LIF I_ext
        # (along with the gap junction contribution), and W2 ablation is
        # written via the W2 I_ext slot. So no extra ablation op is needed
        # in graded_b2 mode — populating ablation_current_pA suffices.
        if self.cross_coupling == "graded_b2":
            return hit
        # Legacy WB2 path: push to LIF/W2 cells via a 50 ms network_operation.
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
        """Return (N,) per-cell-type activity readout over last window_ms.

        Per-cell-type readout API contract (canonical post-WB3 D7-followup):

          * **LIF cells** — spike-count / window in Hz. Standard rate-coded
            interpretation (matches LIFBrain.firing_rates()).

          * **Wave 2 cells (graded_b2 mode)** — σ-magnitude continuous
            readout, scaled `× 100` to "feel rate-like" for downstream
            spike-rate consumers. Matches `graded_brain.py output_rates()`
            (line 378) precedent. Replaces the legacy WB3 CP4
            σ>0.5 rising-threshold pseudo-spike rate readout, which
            silently reports 0 Hz when σ saturates above threshold
            (the WB3 CP4 readout artifact). σ-magnitude correctly
            reflects the saturated active state. Returned value is the
            per-cell mean σ over the window × 100 (so σ ∈ [0, 1] →
            output ∈ [0, 100]).

          * **Wave 2 cells (legacy modes)** — V-threshold release-event
            rate (1.0 / window_s if any release in window). Crude
            estimate; preserved for backward compat with the WB2 path.

        For consumers that need pure σ ∈ [0, 1] without the ×100 rate
        scaling, use `wave2_activities(window_ms)` directly.
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
        # Wave 2 cells
        if self.cross_coupling == "graded_b2":
            # Canonical WB3 D7-followup readout: σ-magnitude × 100 over
            # window. Matches graded_brain.py output_rates() line 378
            # precedent. Resolves the saturation artifact: even when σ
            # is pinned near 1.0, the readout reports the active state.
            sigma_means = self._wave2_sigma_window_mean(window_ms)
            for name, sigma_mean in sigma_means.items():
                out[self.idx[name]] = float(sigma_mean) * 100.0
        else:
            t_now_ms = self.time_ms()
            for name in self.wave2_active:
                last = self.wave2_last_spike_t[name]
                if last > t_now_ms - window_ms:
                    # At least 1 release event in window. Crude estimate.
                    out[self.idx[name]] = 1.0 / (window_ms / 1000.0)
        return out

    def wave2_activities(self, window_ms=200.0):
        """Return per-Wave-2-cell σ-magnitude mean over `window_ms`.

        Canonical Wave 2 cell activity readout (graded_b2 mode). Returns
        a dict `{cell_name: sigma_mean ∈ [0, 1]}` — the raw σ scale,
        without the ×100 rate-scaling that `firing_rates()` applies for
        Hz-flavored consumers.

        Matches `graded_brain.py output_rates()` (line 378) precedent in
        substrate (windowed σ mean) but returns σ ∈ [0, 1] instead of
        the ×100 rate-equivalent. Use this when downstream consumers
        want σ-magnitude directly (e.g., FSM classifiers thresholded on
        σ ≥ 0.7 for "actively releasing"); use `firing_rates()` when
        consumers want a Hz-comparable rate proxy (Phase G, ablation
        harness, dashboard).

        In legacy modes (cross_coupling != "graded_b2"), returns an
        empty dict — σ history is only recorded under graded_b2.

        Parameters
        ----------
        window_ms : float
            Trailing window (ms). Default 200 ms.

        Returns
        -------
        dict[str, float]
            {cell_name: σ_mean ∈ [0, 1]} for each active Wave 2 cell.
            Cells with no σ history in the window return 0.0.
        """
        if self.cross_coupling != "graded_b2":
            return {}
        return {
            name: float(sigma_mean)
            for name, sigma_mean in self._wave2_sigma_window_mean(window_ms).items()
        }

    def _wave2_sigma_window_mean(self, window_ms):
        """Internal: compute per-cell σ mean over the trailing window.

        Uses σ history populated by `_build_wave2_sigma_poll`. Returns a
        dict {cell_name: σ_mean}. Empty/no-history cells return 0.0.
        """
        out = {}
        if not hasattr(self, "_wave2_sigma_history"):
            return {name: 0.0 for name in self.wave2_active}
        t_now_ms = self.time_ms()
        t_cut_ms = t_now_ms - float(window_ms)
        for name in self.wave2_active:
            hist = self._wave2_sigma_history.get(name)
            if hist is None or len(hist["t_ms"]) == 0:
                out[name] = 0.0
                continue
            t_arr = np.asarray(hist["t_ms"], dtype=np.float64)
            s_arr = np.asarray(hist["sigma"], dtype=np.float64)
            mask = t_arr >= t_cut_ms
            if not np.any(mask):
                # Window earlier than recorded history (shouldn't happen
                # in normal use); fall back to most recent sample.
                out[name] = float(s_arr[-1]) if len(s_arr) else 0.0
            else:
                out[name] = float(s_arr[mask].mean())
        return out

    def soft_cap_warning_count(self):
        """Return total count of soft-cap warnings logged in graded_b2 mode.

        Returns 0 in legacy modes. Each warning is a dict in
        self._soft_cap_warnings with keys: t_ms, cell, I_total_pA,
        I_lifw2_pA, I_w2w2_pA, I_gap_w2_pA.
        """
        if not hasattr(self, "_soft_cap_warnings"):
            return 0
        return len(self._soft_cap_warnings)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(cross_coupling="off"):
    """Smoke test: build hybrid brain with AVAL+AVAR, run baseline,
    confirm no errors and Wave 2 cells respond to LIF activity.

    cross_coupling : str
        "off"        — legacy WB2 isolated mode
        "graded_b2"  — WB3 graded Boltzmann release rule (CP2 deliverable)
    """
    print("=" * 70)
    print(f"Wave2HybridBrain smoke test  (cross_coupling={cross_coupling})")
    print("=" * 70)

    print("\nBuilding hybrid brain (AVAL + AVAR Wave 2, 298 LIF)...")
    brain = Wave2HybridBrain(
        wave2_active=["AVAL", "AVAR"],
        cross_coupling=cross_coupling,
        seed=42,
    )
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

    if cross_coupling == "graded_b2":
        warns = brain.soft_cap_warning_count()
        print(f"\nSoft-cap warnings (|I_total per W2| > {WB3_SOFT_CAP_PA} pA): {warns}")

    print("\n[smoke test PASS] Hybrid brain builds and runs.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "off"
    smoke_test(cross_coupling=mode)
