#!/usr/bin/env python3
"""P1 #8 — Sensory transduction cascade ODEs.

Replaces the `sensory_injection` stub path (direct current injection
into sensory neurons) with biologically-grounded transduction
cascades that run their own internal ODE state in parallel with the
Brian2 neuron network. Each cascade takes a stimulus input (salt
concentration, O2 fraction, force, temperature, etc.) and produces
a time-varying current that is delivered to the corresponding
sensory neuron as an injected Poisson spike rate.

Implemented cascades (ordered by effect-size magnitude for worm
phenotypes):

  1. ASE salt-sensing (GCY-22/14 → cGMP → TAX-2/TAX-4)
     — Bargmann 1993, Suzuki 2008, Thiele 2009
  2. AWC olfactory OFF-cell (ODR-10 → Gα_i → GCY → cGMP drop)
     — Chalasani 2007, Zaslaver 2015
  3. ASH polymodal (OSM-9 + OCR-2 TRPV → Ca influx, +TRPA-1 slow)
     — Colbert 1997, Hilliard 2005, Kahn-Kirby 2004
  4. AFD thermal (GCY-8/18/23 → cGMP → TAX-2/TAX-4)
     — Komatsu 1996, Garrity 2010
  5. ALM/AVM touch (MEC-4/MEC-10 DEG/ENaC direct mechano-current)
     — Chalfie & Sulston 1981, O'Hagan 2005

Each cascade exposes:
  .sense(stimulus, dt) -> outputs state-advance + a "conductance"
  .to_poisson_rate()   -> scaled Hz rate for downstream Brian2 injection
  .telemetry()         -> dict snapshot for UI / audit

Integration: `TransductionSensory` aggregates cascades and offers the
same `inject_into_brain(brain)` interface as the simple injection
stub, so ClosedLoopEnv can swap one for the other by a flag.

These cascades are first-order ODEs with literature-derived time
constants. Conversion to current uses a linear receptor-to-rate map
that preserves the max-rate ceiling of the old direct-injection
preset (e.g., 220 Hz for ASH osmotic), so phenotype reproduction
degrades gracefully rather than re-calibrating downstream from
scratch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------


@dataclass
class _Cascade:
    """Base class: tracks state + last output rate."""
    name: str = ""
    neurons: tuple[str, ...] = ()
    state: dict = field(default_factory=dict)
    last_rate_hz: float = 0.0

    def sense(self, stimulus: float, dt_s: float) -> float:
        """Advance one dt, return Poisson rate (Hz)."""
        raise NotImplementedError

    def telemetry(self) -> dict:
        return {"name": self.name, "rate_hz": round(self.last_rate_hz, 1),
                **{k: round(v, 4) for k, v in self.state.items()}}


# ---------------------------------------------------------------------
# 1. ASE — salt sensing
# ---------------------------------------------------------------------


class ASESaltCascade(_Cascade):
    """ASE salt transduction.

    Cascade: [NaCl] → GCY-22/14 guanylyl cyclase activation → cGMP
    rise → TAX-2/TAX-4 CNG channel opening → depolarising current.

    ASEL is the ON-cell (responds to salt increase), ASER is the OFF-
    cell (responds to salt decrease). We track a single slow cGMP
    state + short-timescale receptor adaptation and route to the
    appropriate ASE L/R cell based on dC/dt sign.

    ODEs (Thiele 2009 constants, simplified):
      dcGMP/dt = k_prod · sensor_activation(stim) − k_deg · cGMP
      sensor_activation = max(0, |dC/dt|) × receptor_fraction
      receptor_fraction decays by 1/τ_adapt on sustained stimulus
    """

    TAU_CGMP = 0.3        # 300 ms cGMP time constant
    TAU_ADAPT = 4.0       # 4 s sensory adaptation (fits Suzuki 2008)
    TAU_WINDOW = 0.5      # window for dC/dt computation
    MAX_RATE_HZ = 150.0

    def __init__(self):
        super().__init__(
            name="ASE_salt",
            neurons=("ASEL", "ASER"),
            state={"cGMP": 0.0, "receptor_frac": 1.0, "prev_C": 0.0},
        )

    def sense(self, concentration: float, dt_s: float) -> float:
        prev_C = self.state["prev_C"]
        dCdt = (concentration - prev_C) / max(1e-4, dt_s)
        self.state["prev_C"] = concentration
        # Receptor adaptation — magnitude-sensitive
        stim_mag = min(1.0, abs(dCdt) * 2.0)  # 0.5 mM/s saturates
        alpha_adapt = 1 - math.exp(-dt_s / self.TAU_ADAPT)
        target_recept = 1.0 - 0.6 * stim_mag  # drops to 0.4 at full stim
        self.state["receptor_frac"] = (
            (1 - alpha_adapt) * self.state["receptor_frac"]
            + alpha_adapt * target_recept
        )
        # cGMP — accumulates on sustained high stim, decays
        activation = stim_mag * self.state["receptor_frac"]
        alpha_cgmp = 1 - math.exp(-dt_s / self.TAU_CGMP)
        self.state["cGMP"] = (
            (1 - alpha_cgmp) * self.state["cGMP"]
            + alpha_cgmp * activation
        )
        rate = self.MAX_RATE_HZ * self.state["cGMP"]
        self.last_rate_hz = rate
        self._asel_dCdt = dCdt  # L/R routing in inject()
        return rate

    def inject(self, brain) -> None:
        # ASEL = ON-cell (fires on +dC/dt), ASER = OFF-cell (-dC/dt)
        dCdt = getattr(self, "_asel_dCdt", 0.0)
        rate = self.last_rate_hz
        if dCdt >= 0:
            brain.set_sensory_rate("ASEL", rate, weight_mv=12)
            brain.set_sensory_rate("ASER", 0.3 * rate, weight_mv=12)
        else:
            brain.set_sensory_rate("ASEL", 0.3 * rate, weight_mv=12)
            brain.set_sensory_rate("ASER", rate, weight_mv=12)


# ---------------------------------------------------------------------
# 2. AWC — olfactory OFF-cell
# ---------------------------------------------------------------------


class AWCOlfactoryCascade(_Cascade):
    """AWC off-cell olfactory transduction.

    Cascade: odorant → ODR-10/STR-family GPCR → Gα_i inhibition of
    ODR-1 guanylyl cyclase → cGMP drop → TAX-2/TAX-4 closure →
    hyperpolarisation. When odorant is REMOVED, cGMP rebounds and
    AWC fires (hence "OFF-cell").

    Behavioural signature: AWC firing increases on odour offset
    (Chalasani 2007). We model this as baseline cGMP that drops in
    proportion to odorant, with firing rate ∝ time derivative of
    (baseline - cGMP) when negative (i.e., on odour removal).
    """

    TAU_CGMP = 0.5
    BASELINE_CGMP = 0.8
    TAU_HPOL = 0.3
    MAX_RATE_HZ = 140.0

    def __init__(self):
        super().__init__(
            name="AWC_olfactory",
            neurons=("AWCL", "AWCR"),
            state={"cGMP": 0.8, "V_diff": 0.0, "prev_cGMP": 0.8},
        )

    def sense(self, odorant: float, dt_s: float) -> float:
        # Odorant drives cGMP toward 0 (inhibits cyclase)
        target = self.BASELINE_CGMP * max(0.0, 1 - 3 * odorant)
        alpha = 1 - math.exp(-dt_s / self.TAU_CGMP)
        self.state["cGMP"] = (1 - alpha) * self.state["cGMP"] + alpha * target
        dCGMP = (self.state["cGMP"] - self.state["prev_cGMP"]) / max(1e-4, dt_s)
        self.state["prev_cGMP"] = self.state["cGMP"]
        # AWC fires on cGMP RISE (odour removal). Map to V_diff which
        # then drives firing via a first-order filter.
        drive = max(0.0, dCGMP) * 5.0
        alpha_v = 1 - math.exp(-dt_s / self.TAU_HPOL)
        self.state["V_diff"] = (1 - alpha_v) * self.state["V_diff"] + alpha_v * drive
        rate = self.MAX_RATE_HZ * min(1.0, self.state["V_diff"])
        self.last_rate_hz = rate
        return rate

    def inject(self, brain) -> None:
        for n in self.neurons:
            brain.set_sensory_rate(n, self.last_rate_hz, weight_mv=12)


# ---------------------------------------------------------------------
# 3. ASH — polymodal avoidance
# ---------------------------------------------------------------------


class ASHPolymodalCascade(_Cascade):
    """ASH polymodal transduction.

    Two time-scales (Kahn-Kirby 2004, Hilliard 2005):
      - FAST: OSM-9 + OCR-2 TRPV channels → ~100 ms onset
      - SLOW: TRPA-1 → ~500 ms onset, mediates nociceptive summation

    Stimuli: osmotic shock (dC/dt), harsh touch (force spike), bitter
    (chemical). For v1 we model a single "aversive-strength" input
    that drives both channels in parallel.
    """

    TAU_FAST = 0.1
    TAU_SLOW = 0.5
    TAU_ADAPT = 10.0
    MAX_RATE_HZ = 220.0

    def __init__(self):
        super().__init__(
            name="ASH_polymodal",
            neurons=("ASHL", "ASHR"),
            state={"I_fast": 0.0, "I_slow": 0.0, "recept": 1.0},
        )

    def sense(self, aversive_strength: float, dt_s: float) -> float:
        s = min(1.0, max(0.0, aversive_strength))
        recept = self.state["recept"]
        alpha_adapt = 1 - math.exp(-dt_s / self.TAU_ADAPT)
        target_recept = 1.0 - 0.5 * s
        self.state["recept"] = (1 - alpha_adapt) * recept + alpha_adapt * target_recept

        alpha_fast = 1 - math.exp(-dt_s / self.TAU_FAST)
        alpha_slow = 1 - math.exp(-dt_s / self.TAU_SLOW)
        drive = s * self.state["recept"]
        self.state["I_fast"] = (1 - alpha_fast) * self.state["I_fast"] + alpha_fast * drive
        self.state["I_slow"] = (1 - alpha_slow) * self.state["I_slow"] + alpha_slow * drive
        # Combined current: TRPV dominates early, TRPA-1 sustains
        I = 0.75 * self.state["I_fast"] + 0.25 * self.state["I_slow"]
        rate = self.MAX_RATE_HZ * I
        self.last_rate_hz = rate
        return rate

    def inject(self, brain) -> None:
        for n in self.neurons:
            brain.set_sensory_rate(n, self.last_rate_hz, weight_mv=15)


# ---------------------------------------------------------------------
# 4. AFD — thermal
# ---------------------------------------------------------------------


class AFDThermalCascade(_Cascade):
    """AFD thermosensory transduction.

    Cascade: T° → GCY-8/18/23 guanylyl cyclases → cGMP → TAX-2/4.
    AFD encodes deviation from cultivation temperature T_c. Firing
    increases on warming ABOVE T_c (Kimata 2012, Clark 2006).

    We track a slow T_c memory (~5 min would be ideal but 20 s is
    enough for a scenario) and fire when current T > T_c.
    """

    TAU_CGMP = 0.2
    TAU_TC = 20.0
    MAX_RATE_HZ = 130.0

    def __init__(self, initial_tc_c: float = 20.0):
        super().__init__(
            name="AFD_thermal",
            neurons=("AFDL", "AFDR"),
            state={"T_c": initial_tc_c, "cGMP": 0.0, "T_current": initial_tc_c},
        )

    def sense(self, temp_c: float, dt_s: float) -> float:
        self.state["T_current"] = temp_c
        # T_c memory integrates toward current T slowly
        alpha_tc = 1 - math.exp(-dt_s / self.TAU_TC)
        self.state["T_c"] = (1 - alpha_tc) * self.state["T_c"] + alpha_tc * temp_c
        dev = max(0.0, temp_c - self.state["T_c"])  # warming above T_c
        drive = min(1.0, dev / 3.0)  # 3°C above T_c saturates
        alpha_c = 1 - math.exp(-dt_s / self.TAU_CGMP)
        self.state["cGMP"] = (1 - alpha_c) * self.state["cGMP"] + alpha_c * drive
        rate = self.MAX_RATE_HZ * self.state["cGMP"]
        self.last_rate_hz = rate
        return rate

    def inject(self, brain) -> None:
        for n in self.neurons:
            brain.set_sensory_rate(n, self.last_rate_hz, weight_mv=12)


# ---------------------------------------------------------------------
# 5. ALM/AVM — gentle touch (MEC-4/MEC-10)
# ---------------------------------------------------------------------


class ALMTouchCascade(_Cascade):
    """ALM/AVM gentle-touch transduction.

    MEC-4/MEC-10 DEG/ENaC channels open on mechanical force, producing
    a fast-adapting Na+ current (O'Hagan 2005). ~10 ms rise, ~50 ms
    decay. Stimulus is a boolean pulse or a time-varying force.
    """

    TAU_RISE = 0.01
    TAU_DECAY = 0.05
    MAX_RATE_HZ = 180.0

    def __init__(self, posterior: bool = False):
        super().__init__(
            name="PLM_touch" if posterior else "ALM_touch",
            neurons=("PLML", "PLMR") if posterior else ("ALML", "ALMR", "AVM"),
            state={"I": 0.0, "I_rise": 0.0},
        )

    def sense(self, force: float, dt_s: float) -> float:
        f = min(1.0, max(0.0, force))
        alpha_r = 1 - math.exp(-dt_s / self.TAU_RISE)
        alpha_d = 1 - math.exp(-dt_s / self.TAU_DECAY)
        # Two-compartment — rise then decay
        self.state["I_rise"] = (1 - alpha_r) * self.state["I_rise"] + alpha_r * f
        drive = self.state["I_rise"]
        self.state["I"] = (1 - alpha_d) * self.state["I"] + alpha_d * drive
        rate = self.MAX_RATE_HZ * self.state["I"]
        self.last_rate_hz = rate
        return rate

    def inject(self, brain) -> None:
        for n in self.neurons:
            brain.set_sensory_rate(n, self.last_rate_hz, weight_mv=15)


# ---------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------


class TransductionSensory:
    """Hub for all cascades. Update once per sync with the current
    per-modality stimulus values; inject_into_brain() delivers the
    summed effect."""

    def __init__(self):
        self.ase = ASESaltCascade()
        self.awc = AWCOlfactoryCascade()
        self.ash = ASHPolymodalCascade()
        self.afd = AFDThermalCascade()
        self.alm = ALMTouchCascade(posterior=False)
        self.plm = ALMTouchCascade(posterior=True)
        self.cascades = [self.ase, self.awc, self.ash, self.afd,
                         self.alm, self.plm]
        # Pending stimulus map (applied next update)
        self._pending: dict[str, float] = {}

    def set_stimulus(self, kind: str, value: float) -> None:
        """Queue a stimulus for next update. `kind` one of:
          'salt' (ASE concentration, ~0–1)
          'odor' (AWC odorant, ~0–1)
          'aversive' (ASH strength, ~0–1)
          'temp' (AFD temperature °C, ~15–25)
          'touch_anterior', 'touch_posterior' (0 or 1)
        """
        self._pending[kind] = value

    def update(self, dt_s: float) -> None:
        self.ase.sense(self._pending.get("salt", 0.0), dt_s)
        self.awc.sense(self._pending.get("odor", 0.0), dt_s)
        self.ash.sense(self._pending.get("aversive", 0.0), dt_s)
        self.afd.sense(self._pending.get("temp", 20.0), dt_s)
        self.alm.sense(self._pending.get("touch_anterior", 0.0), dt_s)
        self.plm.sense(self._pending.get("touch_posterior", 0.0), dt_s)

    def inject_into_brain(self, brain) -> None:
        for c in self.cascades:
            c.inject(brain)

    def telemetry(self) -> list[dict]:
        return [c.telemetry() for c in self.cascades]


if __name__ == "__main__":
    # Smoke test: apply a salt ramp + odor pulse + touch, print rates
    sys = TransductionSensory()
    for i in range(200):
        t = i * 0.05
        salt = min(1.0, t / 5.0)  # ramp up over 5 s
        odor = 1.0 if 3.0 <= t < 6.0 else 0.0  # pulse on 3-6 s
        aversive = 1.0 if 8.0 <= t < 8.5 else 0.0  # quick hit at 8 s
        touch = 1.0 if 4.0 <= t < 4.2 else 0.0
        sys.set_stimulus("salt", salt)
        sys.set_stimulus("odor", odor)
        sys.set_stimulus("aversive", aversive)
        sys.set_stimulus("touch_anterior", touch)
        sys.update(0.05)
        if i % 20 == 0:
            print(f"t={t:5.2f}s  "
                  f"ASE {sys.ase.last_rate_hz:>5.1f} Hz  "
                  f"AWC {sys.awc.last_rate_hz:>5.1f} Hz  "
                  f"ASH {sys.ash.last_rate_hz:>5.1f} Hz  "
                  f"ALM {sys.alm.last_rate_hz:>5.1f} Hz")
