#!/usr/bin/env python3
"""P1 #4 — Activity-driven FSM: drive behavioural state directly from
neuron firing rates instead of via the Atanas-trained classifier bank.

Rationale (see v3.3 audit): upgrading brain dynamics (graded, Ca
plateau, volume transmission) collapses phenotype reproduction
because the classifier bank was trained on LIF-derived synthetic
calcium. That coupling is an artefact of the readout-as-inference
layer. The fix is to let the FSM read neural activity directly from
biologically-privileged command neurons and trigger on their rates
relative to a sliding baseline.

Transitions (all thresholds as z-scores of short-window rate vs.
long-window baseline, literature-grounded targets):

  FORWARD → REVERSE   : AVA firing z > z_th        (Chalfie 1985)
  REVERSE → FORWARD   : AVB firing z > z_th        (Gray 2005)
                         OR AVA drops back to baseline
  FORWARD → OMEGA     : SMDV and RIV co-active     (Gray 2005)
  any → QUIESCENT     : RIS sustained high         (Turek 2016)
  any → FORWARD       : (default)
  REVERSE → PIROUETTE : omega fired within 2 s of reversal onset
                         (Gray 2005 definition of a pirouette)

When a trigger neuron is missing from the network (e.g. a subset
simulation), that transition is simply disabled and a warning
printed once.

API mirrors BehavioralFSM so this is a drop-in replacement:
    fsm = ActivityFSM(brain)
    fsm.update(t_s, spike_rates_hz_per_neuron)
    fsm.cpg_params()

`spike_rates_hz_per_neuron` is a dict[name → rate Hz] over the
200 ms window ending at t.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from behavioral_fsm import STATE_CPG, STATE_HOLD_S, FSMTrace, State


# Command/command-adjacent neurons per role. Literature sources:
#   AVA/AVD/AVE   : reversal command           (Chalfie 1985, Gray 2005, Wang 2020)
#   AVB/PVC/RIB   : forward command            (Chalfie 1985, Faumont 2011)
#   SMDV/RIV/RMED : omega turn                 (Gray 2005, Donnelly 2013)
#   RIS           : sleep-like quiescence      (Turek 2016)
#   NSM           : feeding dwell (QUI on food)(Flavell 2013)
#   RIM/RIC       : reversal gating via TA/OA  (Alkema 2005, Pirri 2009)
#   AIB           : reversal initiation        (Piggott 2011)
ROLE_NEURONS: dict[str, list[str]] = {
    "reverse_cmd":   ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER", "AIBL", "AIBR"],
    "forward_cmd":   ["AVBL", "AVBR", "PVCL", "PVCR", "RIBL", "RIBR"],
    "omega_cmd":     ["SMDVL", "SMDVR", "RIVL", "RIVR", "RMEL"],
    "quiescent_cmd": ["RIS"],
    "feeding_dwell": ["NSML", "NSMR"],
}

# Per-role z-score thresholds (rate vs baseline). Higher = stricter.
# Re-calibrated after T0 smoke-run: v3 LIF brain fires command neurons
# at ~30 Hz tonic baseline (not the 2 Hz literature value), so the
# stim-driven *delta* relative to that hot baseline is what matters.
ROLE_Z_THRESHOLD: dict[str, float] = {
    "reverse_cmd":   3.0,   # AVA needs to clearly exceed baseline
    "forward_cmd":   2.5,
    "omega_cmd":     3.0,
    "quiescent_cmd": 3.0,
    "feeding_dwell": 3.0,   # T0 fix: was 1.5, fired spuriously from noise
}

# T0 CAVEAT — v3 LIF brain empirical finding:
# Profiling the current brain with FSM_MODE=classifier and a touch
# stim shows that AVA/AVE command neurons DECREASE firing on touch
# (AVER drops 36 Hz → 28 Hz) rather than producing the literature-
# canonical reversal burst. The classifier's ΔREV=-0.57±0.37
# phenotype reproduction runs via a multi-neuron correlation pattern
# among the 18-readout set, NOT via biologically-correct AVA drive.
#
# ActivityFSM reading AVA directly therefore does NOT reproduce the
# reversal phenotype on the current v3 LIF network. This is a v3
# brain-calibration issue, not an FSM-architecture issue. Fix paths:
#   a) retune LIF synaptic weights so ALM → AIB → AVA cascade actually
#      depolarises AVA (requires brain work, v3.5+)
#   b) switch to graded dynamics (GradedBrain) where continuous σ(V)
#      output may propagate more reliably
#   c) include AIB *and* AVA in the reverse_cmd role pool and require
#      either to exceed threshold (partial workaround)
# For now, ActivityFSM ships as opt-in and the default remains the
# classifier path. See artifacts/activity_fsm_v3lif_audit.md for data.

# Baseline-tracking half-lives (seconds). T0 fix: was 4 s, pushed to
# 20 s so stim-driven excursions of 2-4 s don't get absorbed into the
# baseline. Warmup window below prevents early-transient false
# transitions.
BASELINE_TAU_S = 20.0
WARMUP_S = 2.0      # no transitions allowed during initial baseline estimation
WINDOW_S = 0.4      # rate integration window

# Pirouette detection: reversal followed by omega within this window
PIROUETTE_LINK_S = 2.0


@dataclass
class _RoleStats:
    """Rolling baseline + recent-rate estimates for a role's neurons."""
    present: list[str] = field(default_factory=list)
    baseline_rate: float = 2.0   # Hz, conservative default
    baseline_std: float = 1.5

    def z(self, recent_rate: float) -> float:
        # Clamp baseline std to avoid tiny-denominator blowups.
        sd = max(0.5, self.baseline_std)
        return (recent_rate - self.baseline_rate) / sd


class ActivityFSM:
    """Direct-from-neural-activity behavioural FSM.

    Takes a brain instance (duck-typed: expects brain.names as a list
    of neuron names). update() is called each sync with the rate
    vector over the window_s window ending at t.
    """

    def __init__(self, brain, initial_state: State = State.FORWARD):
        self.state = initial_state
        self.entered_t = 0.0
        self.trace = FSMTrace()

        # Resolve which role neurons are present in this brain.
        name_set = set(brain.names)
        self.role_stats: dict[str, _RoleStats] = {}
        warned = set()
        for role, neurons in ROLE_NEURONS.items():
            present = [n for n in neurons if n in name_set]
            missing = [n for n in neurons if n not in name_set]
            self.role_stats[role] = _RoleStats(present=present)
            if missing and role not in warned:
                # One-time warn so you know which neurons dropped out
                print(f"[ActivityFSM] role={role}: missing {missing}; "
                      f"present {present}")
                warned.add(role)

        # Rolling EMA baselines + a short recent-rate buffer per role
        self._recent_rates: dict[str, deque[tuple[float, float]]] = {
            r: deque(maxlen=64) for r in ROLE_NEURONS
        }

        # Event log for pirouette detection
        self._last_reverse_onset_t: float | None = None

        self._log(0.0, "init")

    def _log(self, t: float, trigger: str):
        self.trace.times.append(t)
        self.trace.states.append(self.state)
        self.trace.triggers.append(trigger)

    def _transition(self, new_state: State, t: float, trigger: str):
        if new_state == self.state:
            return
        if new_state == State.REVERSE:
            self._last_reverse_onset_t = t
        self.state = new_state
        self.entered_t = t
        self._log(t, trigger)

    def _can_exit(self, t: float) -> bool:
        return (t - self.entered_t) >= STATE_HOLD_S[self.state]

    @staticmethod
    def _mean_rate(rates: dict[str, float], neurons: list[str]) -> float:
        if not neurons:
            return 0.0
        total = 0.0
        count = 0
        for n in neurons:
            if n in rates:
                total += rates[n]
                count += 1
        return total / max(1, count)

    def _update_baselines(self, t: float, rates: dict[str, float]) -> None:
        """Advance per-role EMA baseline + std estimates.

        Sample cadence is assumed ≈ 50 ms; alpha is the per-sample
        mix-in fraction for a baseline time constant of BASELINE_TAU_S
        seconds. During the warmup window use a much faster alpha so
        the baseline quickly catches up with the actual tonic rate
        (v3 LIF's ~30 Hz) from the 2 Hz prior — otherwise τ=20 s
        convergence would not kick in until ~40 s.
        """
        DT_S = 0.05
        import math
        if t < WARMUP_S:
            alpha = 1 - math.exp(-DT_S / 0.5)   # τ=0.5 s for fast warmup
        else:
            alpha = 1 - math.exp(-DT_S / BASELINE_TAU_S)
        for role, stats in self.role_stats.items():
            rate = self._mean_rate(rates, stats.present)
            # EMA mean
            stats.baseline_rate = (1 - alpha) * stats.baseline_rate + alpha * rate
            # EMA variance approximation
            dev = rate - stats.baseline_rate
            var_est = stats.baseline_std * stats.baseline_std
            var_est = (1 - alpha) * var_est + alpha * dev * dev
            stats.baseline_std = max(0.5, var_est ** 0.5)
            self._recent_rates[role].append((t, rate))

    def _recent_role_rate(self, role: str) -> float:
        buf = self._recent_rates[role]
        if not buf:
            return 0.0
        # Mean over last ~400 ms (approx 8 samples at 50 ms sync)
        tail = list(buf)[-8:]
        return sum(r for _, r in tail) / len(tail)

    def update(self, t: float, rates: dict[str, float]) -> State:
        """Advance FSM. `rates` = {neuron_name: Hz}."""
        self._update_baselines(t, rates)

        # T0 fix: warmup window — let the baseline EMA converge before
        # allowing any transitions, otherwise the first few samples
        # (while EMA still sits at the 2 Hz prior against an actual
        # 30 Hz tonic) produce huge artefactual z-scores.
        if t < WARMUP_S:
            return self.state

        def z_of(role: str) -> float:
            return self.role_stats[role].z(self._recent_role_rate(role))

        # Precompute role z-scores once
        z_rev = z_of("reverse_cmd")
        z_fwd = z_of("forward_cmd")
        z_omg = z_of("omega_cmd")
        z_qui = z_of("quiescent_cmd")
        z_dwell = z_of("feeding_dwell")

        # PIROUETTE absorbs (state-hold honoured)
        if self.state == State.PIROUETTE and not self._can_exit(t):
            return self.state
        if self.state == State.OMEGA and not self._can_exit(t):
            return self.state

        # Quiescence trigger — RIS sustained, overrides everything
        # except pirouette-in-progress
        if (z_qui >= ROLE_Z_THRESHOLD["quiescent_cmd"]
                and self.state != State.QUIESCENT):
            self._transition(State.QUIESCENT, t, f"RIS z={z_qui:.1f}")
            return self.state

        # Omega — head-curl circuit co-active
        if (z_omg >= ROLE_Z_THRESHOLD["omega_cmd"]
                and self.state not in (State.OMEGA, State.PIROUETTE)):
            # If reverse just happened, this is a pirouette
            if (self._last_reverse_onset_t is not None
                    and t - self._last_reverse_onset_t <= PIROUETTE_LINK_S):
                self._transition(State.PIROUETTE, t,
                                 f"omega-after-reverse z={z_omg:.1f}")
            else:
                self._transition(State.OMEGA, t, f"SMDV/RIV z={z_omg:.1f}")
            return self.state

        # State-specific transitions
        if self.state == State.FORWARD:
            if (self._can_exit(t)
                    and z_rev >= ROLE_Z_THRESHOLD["reverse_cmd"]):
                self._transition(State.REVERSE, t, f"AVA z={z_rev:.1f}")
            elif (self._can_exit(t)
                  and z_dwell >= ROLE_Z_THRESHOLD["feeding_dwell"]):
                self._transition(State.QUIESCENT, t, f"NSM z={z_dwell:.1f}")

        elif self.state == State.REVERSE:
            # Exit when AVA drops back to baseline AND (AVB rises or
            # hold timer expires) — biologically accurate: reversal
            # bout ends when command drive falls, forward picks up.
            if self._can_exit(t):
                if z_rev < 0.5:
                    # AVA activity subsided
                    if z_fwd >= ROLE_Z_THRESHOLD["forward_cmd"]:
                        self._transition(State.FORWARD, t,
                                         f"AVB z={z_fwd:.1f}")
                    else:
                        self._transition(State.FORWARD, t,
                                         "AVA-returned-to-baseline")

        elif self.state == State.OMEGA:
            if self._can_exit(t):
                self._transition(State.FORWARD, t, "omega_timeout")

        elif self.state == State.PIROUETTE:
            if self._can_exit(t):
                self._transition(State.FORWARD, t, "pirouette_timeout")

        elif self.state == State.QUIESCENT:
            # Exit quiescence when RIS drops and forward command
            # picks up, or when the hold timer expires.
            if (self._can_exit(t) and z_qui < 0.8
                    and z_fwd >= ROLE_Z_THRESHOLD["forward_cmd"]):
                self._transition(State.FORWARD, t,
                                 f"exit-QUI AVB z={z_fwd:.1f}")

        return self.state

    def cpg_params(self) -> dict:
        return STATE_CPG[self.state]

    # ---- Diagnostic introspection --------------------------------

    def debug_snapshot(self) -> dict:
        """Current z-scores + role rates for dashboard telemetry."""
        out = {}
        for role, stats in self.role_stats.items():
            recent = self._recent_role_rate(role)
            out[role] = {
                "recent_rate_hz": round(recent, 2),
                "baseline_hz": round(stats.baseline_rate, 2),
                "baseline_sd": round(stats.baseline_std, 2),
                "z": round(stats.z(recent), 2),
                "neurons_present": stats.present,
                "threshold_z": ROLE_Z_THRESHOLD[role],
            }
        return out


if __name__ == "__main__":
    class _StubBrain:
        names = [
            # Reverse command
            "AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER", "AIBL", "AIBR",
            # Forward command
            "AVBL", "AVBR", "PVCL", "PVCR", "RIBL", "RIBR",
            # Omega
            "SMDVL", "SMDVR", "RIVL", "RIVR", "RMEL",
            # Quiescent
            "RIS",
            # Feeding
            "NSML", "NSMR",
        ]

    fsm = ActivityFSM(_StubBrain())
    import random
    random.seed(0)
    prev = None
    for i in range(200):
        t = i * 0.05
        # synthetic drive: AVA fires during 3-5 s, SMDV during 8-9 s
        rates = {n: 2.0 + random.gauss(0, 0.5) for n in _StubBrain.names}
        if 3.0 <= t <= 5.0:
            for n in ("AVAL", "AVAR", "AVDL", "AVDR"):
                rates[n] = 40 + random.gauss(0, 5)
        if 8.0 <= t <= 9.0:
            for n in ("SMDVL", "SMDVR", "RIVL", "RIVR"):
                rates[n] = 40 + random.gauss(0, 5)
        s = fsm.update(t, rates)
        if s != prev:
            print(f"t={t:5.2f}s  state={s.name:<10} trigger={fsm.trace.triggers[-1]}")
            prev = s
    print("\nFinal debug snapshot:")
    for role, info in fsm.debug_snapshot().items():
        print(f"  {role}: z={info['z']} recent={info['recent_rate_hz']}Hz "
              f"baseline={info['baseline_hz']}Hz threshold={info['threshold_z']}")
