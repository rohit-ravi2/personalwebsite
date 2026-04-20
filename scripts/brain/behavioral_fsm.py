#!/usr/bin/env python3
"""Phase 3c-2 — Behavioral finite-state machine.

Takes per-event probabilities from the ClassifierBank and updates a
5-state behavioral mode (forward/reverse/omega/pirouette/quiescent).
Each state maps to CPG parameters for the MuJoCo body.

Transition logic:
  - Each transition uses its own probability threshold (tuned per-event
    based on worm_01 AUC; onsets gate higher than offsets to avoid
    jitter).
  - State-hold minimums (refractory windows) prevent rapid oscillation:
      omega: 1.0 s minimum once entered
      reverse: 0.5 s minimum
      pirouette: 3.0 s minimum
  - Pirouette absorbs nested reversals/omegas (doesn't trigger new
    state entries while active).

CPG parameters per state derived from Boyle-Berri-Cohen 2012 and the
existing Phase 2a physics sim tuning:
  forward:    freq=1.0, λ=0.65, amp=0.35, phase_sign=+1 (head → tail)
  reverse:    freq=1.0, λ=0.65, amp=0.35, phase_sign=-1 (tail → head)
  omega:      freq=0.3, λ=0.3, amp=0.55 (sharp sustained curvature)
  pirouette:  alternates reverse/omega packets
  quiescent:  freq=0, amp=0 (muscles at rest)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class State(Enum):
    FORWARD = auto()
    REVERSE = auto()
    OMEGA = auto()
    PIROUETTE = auto()
    QUIESCENT = auto()


# State-specific CPG parameters
STATE_CPG = {
    State.FORWARD:   {"freq": 1.0, "wavelength": 0.65,
                      "amplitude": 0.35, "phase_sign": +1.0, "turn_bias": 0.0},
    State.REVERSE:   {"freq": 1.0, "wavelength": 0.65,
                      "amplitude": 0.35, "phase_sign": -1.0, "turn_bias": 0.0},
    State.OMEGA:     {"freq": 0.3, "wavelength": 0.30,
                      "amplitude": 0.55, "phase_sign": +1.0, "turn_bias": 0.6},
    State.PIROUETTE: {"freq": 0.6, "wavelength": 0.4,
                      "amplitude": 0.45, "phase_sign": -1.0, "turn_bias": 0.3},
    State.QUIESCENT: {"freq": 0.0, "wavelength": 0.65,
                      "amplitude": 0.0, "phase_sign": +1.0, "turn_bias": 0.0},
}

# Transition thresholds per event (independent). Tuned from worm_01
# validation: events with higher AUC get lower thresholds (more
# sensitive trigger). Calibrated to produce transition rates consistent
# with Atanas statistics (~3 reversals/min, ~1 omega/min).
# Thresholds calibrated empirically against synthetic-calcium classifier
# outputs. The classifier bank was trained on Atanas ΔF/F which has
# different distribution than Brian2-derived synthetic calcium (flagged
# in Phase 3c planning as the architectural risk). Without retraining,
# we compensate by raising thresholds so events fire at biologically
# plausible rates (~2-3 reversals/min, ~1 omega/min).
TRANSITION_THRESHOLDS = {
    "reversal_onset":     0.80,
    "reversal_offset":    0.70,   # asymmetric — offset needs less
    "forward_run_onset":  0.85,
    "forward_run_offset": 0.85,
    "omega_onset":        0.92,
    "pirouette_entry":    0.97,   # extremely rare under this classifier
    "quiescence_onset":   0.95,
    "speed_burst_onset":  0.90,
}

# Minimum-hold time per state (seconds). Once entered, can't exit
# before this window elapses.
STATE_HOLD_S = {
    State.FORWARD:   0.3,
    State.REVERSE:   0.5,
    State.OMEGA:     1.0,
    State.PIROUETTE: 3.0,
    State.QUIESCENT: 2.0,
}


@dataclass
class FSMTrace:
    """Record of state transitions, for plotting + analysis."""
    times: list[float] = field(default_factory=list)
    states: list[State] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)


class BehavioralFSM:
    def __init__(self, initial_state: State = State.FORWARD):
        self.state = initial_state
        self.entered_t = 0.0
        self.trace = FSMTrace()
        self._log(0.0, "init")

    def _log(self, t: float, trigger: str):
        self.trace.times.append(t)
        self.trace.states.append(self.state)
        self.trace.triggers.append(trigger)

    def _transition(self, new_state: State, t: float, trigger: str):
        if new_state == self.state:
            return
        self.state = new_state
        self.entered_t = t
        self._log(t, trigger)

    def _can_exit(self, t: float) -> bool:
        return (t - self.entered_t) >= STATE_HOLD_S[self.state]

    def update(self, t: float, event_probs: dict[str, float]) -> State:
        """Advance FSM given per-event probabilities at time t."""
        # Priority order: pirouette > omega > reverse > quiescent > forward
        # Pirouette is absorbing while active
        if self.state == State.PIROUETTE and not self._can_exit(t):
            return self.state

        if self.state == State.OMEGA and not self._can_exit(t):
            return self.state

        # Pirouette entry (overrides most other states)
        if (event_probs.get("pirouette_entry", 0)
                >= TRANSITION_THRESHOLDS["pirouette_entry"]
                and self.state != State.PIROUETTE):
            self._transition(State.PIROUETTE, t, "pirouette_entry")
            return self.state

        # Omega (overrides forward/reverse if active)
        if (event_probs.get("omega_onset", 0)
                >= TRANSITION_THRESHOLDS["omega_onset"]
                and self.state not in (State.OMEGA, State.PIROUETTE)):
            self._transition(State.OMEGA, t, "omega_onset")
            return self.state

        # Quiescence (very rare, requires high probability)
        if (event_probs.get("quiescence_onset", 0)
                >= TRANSITION_THRESHOLDS["quiescence_onset"]
                and self.state == State.FORWARD):
            self._transition(State.QUIESCENT, t, "quiescence_onset")
            return self.state

        # State-specific transitions
        if self.state == State.FORWARD:
            if self._can_exit(t) and (event_probs.get("reversal_onset", 0)
                    >= TRANSITION_THRESHOLDS["reversal_onset"]):
                self._transition(State.REVERSE, t, "reversal_onset")
            elif (event_probs.get("forward_run_offset", 0)
                  >= TRANSITION_THRESHOLDS["forward_run_offset"]
                  and self._can_exit(t)):
                # fwd ended but no reversal → go quiescent briefly
                self._transition(State.QUIESCENT, t, "forward_run_offset")

        elif self.state == State.REVERSE:
            if self._can_exit(t) and (event_probs.get("reversal_offset", 0)
                    >= TRANSITION_THRESHOLDS["reversal_offset"]):
                self._transition(State.FORWARD, t, "reversal_offset")

        elif self.state == State.OMEGA:
            if self._can_exit(t):
                self._transition(State.FORWARD, t, "omega_timeout")

        elif self.state == State.PIROUETTE:
            if self._can_exit(t):
                self._transition(State.FORWARD, t, "pirouette_timeout")

        elif self.state == State.QUIESCENT:
            if (event_probs.get("forward_run_onset", 0)
                    >= TRANSITION_THRESHOLDS["forward_run_onset"]
                    and self._can_exit(t)):
                self._transition(State.FORWARD, t, "forward_run_onset")
            elif (self._can_exit(t) and event_probs.get("speed_burst_onset", 0)
                  >= TRANSITION_THRESHOLDS["speed_burst_onset"]):
                self._transition(State.FORWARD, t, "speed_burst_onset")

        return self.state

    def cpg_params(self) -> dict:
        return STATE_CPG[self.state]


if __name__ == "__main__":
    # Smoke test: synthetic event probability stream
    import numpy as np
    fsm = BehavioralFSM()
    print(f"{'t':>5} {'state':<12} {'trigger':<20}")
    print("-" * 42)

    # Simulate: forward, then reversal event, then omega, etc.
    schedule = [
        (0.5, {"reversal_onset": 0.9}),
        (1.5, {"reversal_offset": 0.9}),
        (3.0, {"omega_onset": 0.9}),
        (4.5, {}),        # let omega time out
        (5.5, {"reversal_onset": 0.7, "pirouette_entry": 0.8}),
        (8.0, {}),        # let pirouette time out
        (12.0, {"reversal_onset": 0.9}),
        (14.0, {"reversal_offset": 0.9}),
    ]
    prev_state = None
    for t, probs in schedule:
        s = fsm.update(t, probs)
        if s != prev_state:
            print(f"{t:>5.1f} {s.name:<12} "
                  f"{fsm.trace.triggers[-1]:<20}")
            prev_state = s
