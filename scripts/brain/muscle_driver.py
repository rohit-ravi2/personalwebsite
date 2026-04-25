#!/usr/bin/env python3
"""Phase 2 — T4-1 — muscle driver.

Sits between LIFBrain's motor-neuron spike rates and MuJoCo's muscle
actuator ctrls. Replaces `cpg_ctrl()` from `closed_loop_env.py` when
`body_driver="muscle"` is selected.

Pipeline per BRAIN_SYNC_MS (50 ms):
  1. Read firing rates for the ~76 motor neurons from the brain's
     full spike buffer (DA/DB/VA/VB/AS/DD/VD/VC classes).
  2. Apply the 540-weight sparse innervation matrix (White 1986 + Cook
     2019) from motor_innervation.json to get per-muscle activation:
         M[m] = sum_n (rate[n] / R_max) * weight(n,m) * sign(n,m)
     (rates normalised against R_max = 100 Hz saturation ceiling;
     GABAergic class D-D/V-D have sign=-1 acting as disinhibition on
     the opposing quadrant — see class table in
     motor_innervation.json).
  3. Aggregate 95 muscles → 80 (segment × quadrant) composites via
     segment_aggregation.
  4. Output dict {actuator_name: activation_in_[0,1]} for MuJoCo's
     muscle ctrl. The 4 quadrant muscles per transition are driven
     directly by the segment-quadrant composites. DL/DR drive the
     dorsal pair (top of the worm), VL/VR the ventral — during a
     forward bout, DL+DR alternate with VL+VR at the locomotion
     wave frequency.

Validation hooks:
  - `drive_test_forward()`: simulate a constant forward-drive pattern
    (DB*, VB* at fixed rate) and return the 80 activations. Used for
    the T4-1 smoke test before brain integration.
  - `drive_test_reverse()`: same for DA*, VA*.

NOTE: this module does NOT step MuJoCo. It just transforms brain
output → actuator ctrl. Integration into ClosedLoopEnv happens in
Phase 2 when body_driver="muscle" path is added.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
INNERV_JSON = REPO / "public" / "data" / "motor_innervation.json"

# Max saturating rate for motor neurons (worm motor neurons rarely
# exceed ~100 Hz per Faumont 2011 / Chalfie 1985 calcium).
R_MAX_HZ = 100.0


@dataclass
class MuscleDriver:
    """Translates motor-neuron Hz → per-muscle and per-segment-quadrant
    activation."""
    innervation_json_path: Path = INNERV_JSON
    r_max_hz: float = R_MAX_HZ

    # populated in __post_init__
    muscles: list[str] = field(default_factory=list, repr=False)
    neurons: list[str] = field(default_factory=list, repr=False)
    classes: dict = field(default_factory=dict, repr=False)
    W: np.ndarray = field(default=None, repr=False)  # (n_muscles, n_neurons)
    S: np.ndarray = field(default=None, repr=False)  # sign (n_muscles, n_neurons)
    seg_agg: dict = field(default_factory=dict, repr=False)
    agg_matrix: np.ndarray = field(default=None, repr=False)  # (80, 95)
    # Map composite name → index into flattened (80,) output
    composite_names: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        data = json.loads(Path(self.innervation_json_path).read_text())
        self.muscles = list(data["muscles"])
        self.neurons = list(data["neurons"])
        self.classes = data["classes"]

        M = len(self.muscles)
        N = len(self.neurons)
        m_idx = {m: i for i, m in enumerate(self.muscles)}
        n_idx = {n: i for i, n in enumerate(self.neurons)}
        W = np.zeros((M, N), dtype=np.float32)
        S = np.zeros((M, N), dtype=np.int8)
        for neuron, muscle, weight, sign in data["weights"]:
            if muscle not in m_idx or neuron not in n_idx:
                continue
            W[m_idx[muscle], n_idx[neuron]] = float(weight)
            S[m_idx[muscle], n_idx[neuron]] = int(sign)
        self.W = W
        self.S = S

        # Segment-quadrant aggregation
        self.seg_agg = data["segment_aggregation"]["aggregates"]
        # Build a (80, 95) matrix: each row = 1/|members| for the member
        # muscle indices, 0 otherwise.
        self.composite_names = sorted(
            self.seg_agg.keys(),
            key=lambda k: (self.seg_agg[k]["segment"],
                           ["DL", "DR", "VL", "VR"].index(
                               self.seg_agg[k]["quadrant"]))
        )
        A = np.zeros((len(self.composite_names), M), dtype=np.float32)
        for i, comp_name in enumerate(self.composite_names):
            members = self.seg_agg[comp_name]["members"]
            if not members:
                continue
            w = 1.0 / len(members)
            for m in members:
                if m in m_idx:
                    A[i, m_idx[m]] = w
        self.agg_matrix = A

    # ---- Main pipeline ---------------------------------------------

    def motor_rates_from_brain_rates(
        self, brain_rates: dict[str, float]
    ) -> np.ndarray:
        """Extract the motor-neuron subset from a {neuron_name: Hz} dict.

        Returns (n_neurons,) array normalised to [0, 1] (post-R_MAX clip).
        Neurons absent from brain_rates default to 0.
        """
        r = np.zeros(len(self.neurons), dtype=np.float32)
        for i, n in enumerate(self.neurons):
            hz = brain_rates.get(n, 0.0)
            r[i] = min(1.0, max(0.0, hz / self.r_max_hz))
        return r

    def muscle_activations(self, motor_norm: np.ndarray) -> np.ndarray:
        """Per-muscle raw activation (can be negative from GABAergic input).

        Returns (n_muscles,) clipped to [0, 1].
        """
        raw = (self.W * self.S).astype(np.float32) @ motor_norm
        return np.clip(raw, 0.0, 1.0)

    def composite_activations(self, muscle_act: np.ndarray) -> dict[str, float]:
        """Aggregate muscles → (segment × quadrant) composites.

        Returns {actuator_name: activation_[0,1]} for the 80 composites.
        Actuator names match the v3 MJCF convention: `muscle_{seg}_{quad}`.
        """
        comp = self.agg_matrix @ muscle_act  # (80,)
        out = {}
        for i, comp_name in enumerate(self.composite_names):
            seg = self.seg_agg[comp_name]["segment"]
            quad = self.seg_agg[comp_name]["quadrant"]
            # v3 MJCF only has 76 muscles (transitions 0..18), not 80.
            # Drop segment_19 composites — tail has no further joint.
            # Also: v3 emits muscle_i_Q for i in 0..18 inclusive.
            if seg >= 19:
                continue
            out[f"muscle_{seg}_{quad}"] = float(comp[i])
        return out

    def step(self, brain_rates: dict[str, float]) -> dict[str, float]:
        """End-to-end: brain rates → 76 muscle ctrls."""
        motor_norm = self.motor_rates_from_brain_rates(brain_rates)
        muscle_act = self.muscle_activations(motor_norm)
        return self.composite_activations(muscle_act)

    # ---- Test drives -----------------------------------------------

    def drive_test_forward(self, rate_hz: float = 50.0) -> dict[str, float]:
        """Constant forward-drive: all DB*, VB* at rate_hz."""
        rates = {}
        for n in self.neurons:
            if n.startswith("DB") or n.startswith("VB"):
                rates[n] = rate_hz
        return self.step(rates)

    def drive_test_reverse(self, rate_hz: float = 50.0) -> dict[str, float]:
        """Constant reverse-drive: all DA*, VA* at rate_hz."""
        rates = {}
        for n in self.neurons:
            if n.startswith("DA") or n.startswith("VA"):
                rates[n] = rate_hz
        return self.step(rates)


def _smoke_test():
    """Sanity-check the driver: produce forward- and reverse-drive
    activations and print dorsal vs ventral totals."""
    d = MuscleDriver()
    print(f"Loaded {len(d.muscles)} muscles, {len(d.neurons)} neurons, "
          f"{len(d.composite_names)} composites")
    print(f"W shape {d.W.shape}, nonzeros {int((d.W != 0).sum())}")
    print()

    fwd = d.drive_test_forward(rate_hz=50.0)
    rev = d.drive_test_reverse(rate_hz=50.0)

    # Dorsal vs ventral totals per scenario
    def dv_totals(acts):
        d_total = sum(v for k, v in acts.items()
                      if k.endswith("_DL") or k.endswith("_DR"))
        v_total = sum(v for k, v in acts.items()
                      if k.endswith("_VL") or k.endswith("_VR"))
        return d_total, v_total

    fwd_d, fwd_v = dv_totals(fwd)
    rev_d, rev_v = dv_totals(rev)
    print(f"FORWARD drive (DB*, VB* @ 50Hz):")
    print(f"  dorsal total activation:  {fwd_d:.2f}")
    print(f"  ventral total activation: {fwd_v:.2f}")
    print(f"REVERSE drive (DA*, VA* @ 50Hz):")
    print(f"  dorsal total activation:  {rev_d:.2f}")
    print(f"  ventral total activation: {rev_v:.2f}")
    print()

    # Sanity expectation: forward/reverse should produce similar magnitude
    # (both classes innervate dorsal + ventral sides), but driving DB or VB
    # only excites one side. Actual biology: forward runs have DB->dorsal
    # and VB->ventral firing out-of-phase via the locomotion CPG. Our
    # constant drive is unphysiological but the driver should still respond.
    # What matters: activations are in [0, 1] and not all-zero.
    assert all(0.0 <= v <= 1.0 for v in fwd.values()), "activation out of range"
    assert any(v > 0 for v in fwd.values()), "forward drive produced nothing"
    assert any(v > 0 for v in rev.values()), "reverse drive produced nothing"

    # Print a per-segment summary for the forward drive
    print("Forward drive per-segment (dorsal L/R vs ventral L/R):")
    for i in range(19):
        dl = fwd.get(f"muscle_{i}_DL", 0.0)
        dr = fwd.get(f"muscle_{i}_DR", 0.0)
        vl = fwd.get(f"muscle_{i}_VL", 0.0)
        vr = fwd.get(f"muscle_{i}_VR", 0.0)
        print(f"  seg{i:2d}: DL={dl:.2f} DR={dr:.2f} VL={vl:.2f} VR={vr:.2f}")


if __name__ == "__main__":
    _smoke_test()
