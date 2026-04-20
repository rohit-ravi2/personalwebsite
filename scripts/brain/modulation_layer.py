#!/usr/bin/env python3
"""Phase 3d-2 — Brian2-integrated slow neuromodulation layer.

Overlays 9 neuromodulators (FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5HT,
DA, TA, OA) on top of the existing fast-LIF brain. Each modulator is
a scalar concentration that:
  - decays exponentially with τ ∈ [4, 30] s
  - is incremented by spike events from its releaser neurons
  - produces per-neuron modulation current I_mod proportional to
    target receptor expression × concentration

The I_mod current feeds into the existing LIF membrane equation via
`I_ext`, producing slow modulatory effects that operate on seconds-
to-minutes timescales alongside the millisecond fast-synapse dynamics.

Architecture:
  Brian2 runs fast LIF at dt = 0.1 ms internally.
  A @network_operation(dt = 50 ms) updates modulator concentrations
  and rewrites `neurons.I_ext` from the modulation current vector.
  Between updates, I_ext is held constant (step function, OK given
  modulator timescales are ≥ 4 s).

Parameters (tuned for biological plausibility):
  release_gain   — contribution per spike per releaser weight-unit
  mod_strength   — pA per (concentration × target_weight) unit

Defaults produce ~5 mV slow modulation on targets of well-expressed
modulators firing at 5 Hz, matching typical neuromodulator effect
sizes in whole-brain imaging.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from brian2 import amp, pA, ms, network_operation


ART = Path(__file__).resolve().parent / "artifacts"
TABLES = ART / "modulator_tables.npz"

# Default update cadence — 50 ms, same as closed-loop sync. All
# modulators have τ ≥ 4 s so this is well below their dynamics.
DEFAULT_UPDATE_DT_MS = 50.0

# Release gain: impulse contribution to concentration per spike from a
# releaser with weight=1. Calibrated so a releaser firing at ~50 Hz
# (our LIF's characteristic high-firing rate) saturates the
# concentration to ~C_max in a few seconds.
DEFAULT_RELEASE_GAIN = 0.02

# Concentration saturation cap (dimensionless units, per modulator).
# Reflects biological binding-site saturation: extracellular peptide
# beyond a threshold doesn't produce additional receptor activation.
# Set so max I_mod (at cap × top target weight × strength) is ~50 pA,
# producing a ~5 mV slow modulation — biologically meaningful but not
# overwhelming.
DEFAULT_CONCENTRATION_CAP = 10.0

# Modulation strength: pA of slow current per (concentration ×
# target_weight) unit. With C=10 (cap) and target_weight=-0.8,
# I_mod = -8 × strength → -40 pA → ~-4 mV slow inhibition.
DEFAULT_MOD_STRENGTH_PA = 5.0


class ModulationLayer:
    """Slow neuromodulation overlay on a Brian2 LIF network."""

    def __init__(
        self,
        brain_neuron_names: list[str],
        tables_path: Path = TABLES,
        update_dt_ms: float = DEFAULT_UPDATE_DT_MS,
        release_gain: float = DEFAULT_RELEASE_GAIN,
        mod_strength_pa: float = DEFAULT_MOD_STRENGTH_PA,
        concentration_cap: float = DEFAULT_CONCENTRATION_CAP,
    ):
        d = np.load(tables_path, allow_pickle=True)
        self.modulators = [str(m) for m in d["modulators"]]
        table_order = [str(n) for n in d["neuron_order"]]
        self.M = len(self.modulators)
        self.N = len(brain_neuron_names)

        # Reorder table neurons → brain neuron order.
        # Typically identical (both from connectome.npz) but check.
        if table_order != brain_neuron_names:
            reorder = [table_order.index(n) if n in table_order else -1
                       for n in brain_neuron_names]
        else:
            reorder = list(range(self.N))
        reorder_arr = np.array(reorder)

        # Stack per-modulator releaser & target weights into matrices
        self.releaser_weights = np.zeros((self.M, self.N), dtype=np.float32)
        self.target_weights = np.zeros((self.M, self.N), dtype=np.float32)
        self.taus_s = np.zeros(self.M, dtype=np.float32)

        for mi, mod in enumerate(self.modulators):
            rw_raw = d[f"releaser_weights_{mod}"]
            tw_raw = d[f"target_weights_{mod}"]
            # Reorder to brain neuron order (fill zeros for missing)
            rw = np.zeros(self.N, dtype=np.float32)
            tw = np.zeros(self.N, dtype=np.float32)
            for bi, ti in enumerate(reorder):
                if 0 <= ti < len(rw_raw):
                    rw[bi] = rw_raw[ti]
                    tw[bi] = tw_raw[ti]
            # Normalise releaser weights per modulator so max=1
            rw_max = float(rw.max())
            if rw_max > 0:
                rw = rw / rw_max
            self.releaser_weights[mi] = rw
            self.target_weights[mi] = tw
            self.taus_s[mi] = float(d[f"tau_{mod}"])

        # Concentration state
        self.concentrations = np.zeros(self.M, dtype=np.float32)

        # Params
        self.update_dt_ms = update_dt_ms
        self.update_dt_s = update_dt_ms / 1000.0
        self.release_gain = release_gain
        self.mod_strength_pa = mod_strength_pa
        self.concentration_cap = concentration_cap

        # Cached decay factors per modulator
        self._decay = np.exp(-self.update_dt_s / self.taus_s).astype(np.float32)

        # Tracking: we read spikes from a SpikeMonitor. Track last-seen
        # spike index so each update only processes new events.
        self._prev_spike_idx = 0

        # History for diagnostics (optional, small memory footprint)
        self.history_concentrations: list[np.ndarray] = []
        self.history_times_s: list[float] = []

    # --------------------------------------------------------------

    def step(self, spike_counts: np.ndarray) -> np.ndarray:
        """Apply one update step: decay concentrations + apply release +
        compute per-neuron modulation current.

        Args:
            spike_counts: (N,) integer array of spikes fired per neuron
                in the last `update_dt_ms` window.

        Returns:
            (N,) float32 array of modulation currents in pA to be added
            to each neuron's I_ext.
        """
        # Release contribution per modulator: dot product of releaser
        # weights with spike counts. Non-releasers have zero weight so
        # they don't contribute.
        release = self.releaser_weights @ spike_counts.astype(np.float32)

        # Exponential-decay + impulse update, clamped to the saturation
        # cap to model biological binding-site saturation:
        #   C_new = min(C_old × exp(-dt/τ) + release_gain × release, C_max)
        self.concentrations = np.minimum(
            self.concentrations * self._decay + self.release_gain * release,
            self.concentration_cap,
        )

        # Per-neuron modulation current:
        #   I_mod[i] = Σ_m (target_weights[m, i] × concentration[m]) × strength
        I_mod = (self.target_weights.T @ self.concentrations) * self.mod_strength_pa
        return I_mod.astype(np.float32)

    def record_history(self, t_s: float) -> None:
        self.history_concentrations.append(self.concentrations.copy())
        self.history_times_s.append(t_s)

    # --------------------------------------------------------------

    def attach_to_brain(self, brain) -> None:
        """Attach this modulation layer to a LIFBrain instance.

        Adds a @network_operation that runs every `update_dt_ms` sim-
        time, reads recent spike counts from `brain.spikes`, advances
        the modulation dynamics, and rewrites `brain.neurons.I_ext`
        with the resulting modulation current.
        """
        self.brain = brain
        self._prev_spike_idx = 0
        update_dt = self.update_dt_ms * ms

        @network_operation(dt=update_dt)
        def _update_modulation():
            # Get counts of new spikes per neuron since last call
            all_i = brain.spikes.i[:]
            n_total = len(all_i)
            new_i = all_i[self._prev_spike_idx:n_total]
            self._prev_spike_idx = n_total

            counts = np.zeros(brain.N, dtype=np.float32)
            if len(new_i) > 0:
                np.add.at(counts, new_i, 1.0)

            I_mod_pA = self.step(counts)
            # Compose with any persistent ablation current, then assign
            # to Brian2 neurons.I_ext (amps).
            I_total = I_mod_pA + brain.ablation_current_pA
            brain.neurons.I_ext_ = I_total * 1e-12  # pA → A

            # Occasional diagnostic snapshot
            t_now_s = float(brain.net.t / ms / 1000)
            self.record_history(t_now_s)

        # Save a reference so Python doesn't garbage-collect the closure
        self._op = _update_modulation
        brain.net.add(_update_modulation)
        # Flag on the brain so LIFBrain.ablate() knows modulation is
        # already assigning I_ext (prevents double-assignment).
        brain._modulation_attached = True

    # --------------------------------------------------------------

    def diagnostics(self) -> str:
        lines = []
        lines.append(f"Modulation layer ({self.M} modulators, "
                      f"update dt={self.update_dt_ms:.0f} ms):")
        lines.append(f"  {'modulator':<10} {'τ (s)':>6} {'#releasers':>11} "
                     f"{'|mean tgt|':>11} {'current C':>10}")
        for mi, mod in enumerate(self.modulators):
            n_rel = int(np.sum(self.releaser_weights[mi] > 0.01))
            mean_tgt = float(np.mean(np.abs(self.target_weights[mi])))
            C = float(self.concentrations[mi])
            lines.append(f"  {mod:<10} {self.taus_s[mi]:>6.1f} {n_rel:>11} "
                         f"{mean_tgt:>11.3f} {C:>10.3f}")
        return "\n".join(lines)


# ---------- Smoke test ----------

def smoke_test():
    """Build a LIFBrain, attach modulation, run 30 s of baseline + ASH
    stimulation, check that:
      - FLP-11 concentration is nonzero (RIS is firing at least noise-level)
      - 5HT concentration rises during stimulation
      - Modulation current vectors make biological sense
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lif_brain import LIFBrain
    from sensory_injection import stimulate

    print("Building brain + modulation layer…")
    brain = LIFBrain()
    mod = ModulationLayer(brain.names)
    mod.attach_to_brain(brain)
    print(mod.diagnostics())

    print("\nRun 1: 15 s baseline (no stim)…")
    brain.run(15000)  # 15 s in ms
    print(f"Concentrations after baseline:")
    for m, c in zip(mod.modulators, mod.concentrations):
        print(f"  {m:<8}: {c:.3f}")

    print("\nRun 2: ASH stim + 15 s observation…")
    stimulate(brain, "osmotic_shock", intensity=1.0)
    brain.run(15000)
    print(f"Concentrations after ASH stim:")
    for m, c in zip(mod.modulators, mod.concentrations):
        print(f"  {m:<8}: {c:.3f}")

    # Show modulation-current effects on key command neurons
    import numpy as np
    I_mod = mod.step(np.zeros(brain.N, dtype=np.float32))  # current I_mod
    for n in ["AVAL", "AVAR", "AVBL", "AVBR", "RIS", "NSML", "NSMR"]:
        if n in brain.idx:
            print(f"  I_mod[{n}] = {I_mod[brain.idx[n]]:+.1f} pA")


if __name__ == "__main__":
    smoke_test()
