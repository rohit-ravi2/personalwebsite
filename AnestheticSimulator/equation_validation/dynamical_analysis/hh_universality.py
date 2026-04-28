"""CP B.3 — Hodgkin-Huxley universality test.

H-H universality: a cell with sufficient inward depolarizing channels +
outward repolarizing K channels SHOULD spike under sufficient depolarization,
and spike shape should follow H-H predictions:
  - Spike threshold ≈ V at which inward current exceeds outward
  - Spike peak ≈ E_Na (or E_Ca for Ca-spikes)
  - Refractory period ≈ K-channel deactivation time

Wave 2 cells are graded — they're biologically NOT expected to spike at
all under physiological stimuli (this is a feature, not a bug; C. elegans
is largely non-spiking). This validator tests:

1. Under STRONG depolarization (large I_inj), does the cell exhibit
   regenerative (positive feedback) dynamics that look like a spike?
2. If yes, does the spike peak approach E_Ca (Ca-spike) or stay below?
3. If no spike under any tested I_inj (up to +200 pA), the cell is
   confirmed graded — biologically correct for most C. elegans neurons.

Output: artifacts/hh_universality.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cell_params import ALL_CELLS, total_capacitance_pF
from dynamical_analysis.phase_planes import (
    SLOW_GATE, total_current_pA, gate_steady_state, boltzmann
)

OUT = ROOT / "artifacts"
CHK = ROOT / "checkpoints" / "path_b_cp3_hh.json"


def simulate_with_dynamic_gate(cell: dict, slow_gate_spec: dict, I_inj_pA: float,
                                 duration_ms: float = 200, dt_ms: float = 0.05) -> dict:
    """Simulate cell with single dynamic slow gate variable.

    State: (V, gate). Equations:
      dV/dt = I_total / C
      dGate/dt = (gate_inf(V) - gate) / tau

    Returns trace + extracted features (spike count, peak V, threshold).
    """
    C_pF = total_capacitance_pF(cell)
    V = -55.0
    gate = gate_steady_state(V, slow_gate_spec)
    tau = slow_gate_spec["tau_ms"]
    n_steps = int(duration_ms / dt_ms)
    V_trace = []
    gate_trace = []
    for step in range(n_steps):
        I = total_current_pA(cell, V, gate, slow_gate_spec["channel"], I_inj_pA)
        dV = (I / C_pF) * dt_ms
        gate_inf = gate_steady_state(V, slow_gate_spec)
        dGate = (gate_inf - gate) / tau * dt_ms
        V += dV
        gate += dGate
        # Numerical safety
        if math.isnan(V) or math.isnan(gate):
            return {"V_max": float("nan"), "V_min": float("nan"), "n_spikes": 0,
                    "regenerative": False, "trace_length": step}
        V = max(-150.0, min(100.0, V))
        gate = max(0.0, min(1.0, gate))
        V_trace.append(V)
        gate_trace.append(gate)

    V_max = max(V_trace)
    V_min = min(V_trace)
    V_amplitude = V_max - V_min

    # Detect spikes: count zero-crossings of (V - V_threshold)
    # Use V_threshold = mean V + 50% of amplitude as adaptive threshold
    V_mean = sum(V_trace) / len(V_trace)
    threshold = V_mean + 0.5 * (V_max - V_mean)
    n_spikes = 0
    above = False
    for v in V_trace:
        if v > threshold and not above:
            n_spikes += 1
            above = True
        elif v < threshold and above:
            above = False

    # Regenerative dynamics: if V transiently exceeds steady-state by > 5 mV
    V_final = V_trace[-1]
    overshoot = V_max - V_final
    regenerative = overshoot > 5.0

    return {
        "V_max": round(V_max, 2),
        "V_min": round(V_min, 2),
        "V_amplitude": round(V_amplitude, 2),
        "V_final": round(V_final, 2),
        "n_spikes": n_spikes,
        "regenerative": regenerative,
        "overshoot_above_steady_mV": round(overshoot, 2),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CHK.parent.mkdir(parents=True, exist_ok=True)

    test_currents = [0, 10, 30, 50, 100, 200]

    md_path = OUT / "hh_universality.md"
    cell_data = {}

    with open(md_path, "w") as f:
        f.write("# CP B.3 — Hodgkin-Huxley universality test\n\n")
        f.write("**Date:** 2026-04-28\n\n")
        f.write("Test whether Wave 2 cells exhibit regenerative spike-like dynamics under "
                "strong depolarization, and if so whether spike properties match H-H "
                "predictions. C. elegans cells are biologically expected to be GRADED "
                "(non-spiking), so this validator confirms or rejects that biological "
                "expectation at the equation level.\n\n")
        f.write("**Method:** simulate single-compartment cell with dynamic slow gate, "
                "scan I_inj from 0 to +200 pA, extract V_max, spike count, regenerative "
                "overshoot. A cell that doesn't spike under +200 pA is confirmed graded.\n\n")

        for cell_name, cell in ALL_CELLS.items():
            slow = SLOW_GATE[cell_name]
            f.write(f"## {cell_name}\n\n")
            f.write("| I_inj (pA) | V_max (mV) | V_min (mV) | amplitude | spikes | regenerative |\n")
            f.write("|---|---|---|---|---|---|\n")
            results_per_I = {}
            for I in test_currents:
                r = simulate_with_dynamic_gate(cell, slow, I)
                results_per_I[I] = r
                f.write(f"| {I} | {r['V_max']} | {r['V_min']} | {r['V_amplitude']} | "
                        f"{r['n_spikes']} | {'YES' if r['regenerative'] else 'no'} |\n")

            # Verdict
            max_spikes = max(r["n_spikes"] for r in results_per_I.values())
            any_regenerative = any(r["regenerative"] for r in results_per_I.values())
            if max_spikes > 0:
                verdict = f"SPIKING under strong drive ({max_spikes} spikes at strongest tested current)"
            elif any_regenerative:
                verdict = "REGENERATIVE but non-spiking — Ca-driven plateau without full repolarization"
            else:
                verdict = "GRADED — no spikes or regenerative overshoot under +200 pA. Confirms biological expectation for C. elegans graded neurons."
            f.write(f"\n### Verdict: {verdict}\n\n")

            # H-H prediction comparison
            E_Ca = cell["e_Ca_mV"]
            E_K = cell["e_K_mV"]
            f.write(f"**H-H prediction comparison:**\n\n")
            f.write(f"- E_Ca = {E_Ca} mV (would be Ca-spike peak if Ca-spiking)\n")
            f.write(f"- E_K = {E_K} mV (would be after-hyperpolarization floor)\n")
            v_max_observed = max(r["V_max"] for r in results_per_I.values())
            f.write(f"- Observed V_max (across tested currents) = {v_max_observed} mV\n")
            if v_max_observed > E_Ca - 10:
                f.write(f"- ✓ V_max approaches E_Ca → Ca-spike-consistent if spiking detected\n")
            elif v_max_observed > -20:
                f.write(f"- Observed V_max in Mellem-AVA range; Ca-channel partial activation but no Ca-spike overshoot\n")
            else:
                f.write(f"- Observed V_max below -20 mV; cell stays subthreshold for Ca activation under tested currents\n")
            f.write("\n")

            cell_data[cell_name] = {
                "results_per_I": results_per_I,
                "verdict": verdict,
                "max_spikes": max_spikes,
                "any_regenerative": any_regenerative,
            }

        f.write("## Cross-cell synthesis\n\n")
        n_graded = sum(1 for d in cell_data.values() if d["max_spikes"] == 0 and not d["any_regenerative"])
        n_regenerative = sum(1 for d in cell_data.values() if d["max_spikes"] == 0 and d["any_regenerative"])
        n_spiking = sum(1 for d in cell_data.values() if d["max_spikes"] > 0)
        f.write(f"- Graded (no spike, no regenerative): **{n_graded}/{len(cell_data)}** cells\n")
        f.write(f"- Regenerative but non-spiking: **{n_regenerative}/{len(cell_data)}** cells\n")
        f.write(f"- Spiking under strong drive: **{n_spiking}/{len(cell_data)}** cells\n\n")
        f.write("**H-H universality verdict:** the canonical H-H formalism is implemented in "
                "the cell-builder code; whether or not a cell spikes depends on the channel "
                "suite balance (inward vs outward currents). Wave 2 cells use Nicoletti's "
                "channel suites which are optimized for graded validated phenotypes. "
                "Regenerative behavior under non-physiological strong drive (+200 pA) is "
                "informative about the cell's potential dynamical regimes, not its biological "
                "operating mode.\n")

    state = {
        "checkpoint": "path_b_cp3_hh",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell_data": cell_data,
    }
    json.dump(state, open(CHK, "w"), indent=2, default=str)

    print(f"H-H universality test:\n")
    for cell_name, data in cell_data.items():
        print(f"  {cell_name:6s}: max {data['max_spikes']} spikes, "
              f"regenerative={data['any_regenerative']} → {data['verdict']}")
    print(f"\nMD: {md_path}")
    print(f"Checkpoint: {CHK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
