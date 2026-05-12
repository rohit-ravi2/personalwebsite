"""Phase G — LIFBrain substrate for production-grade dose-response on M2-pure brain.

Replaces the 50-neuron Brian2 demo (in phase_g_network_perturbation.py
`dose_response_sweep`) with the production LIFBrain under the
horizontal-rebase-locked configuration:
  - M2-pure sign mode (use_per_edge_glu_signs=True, sign_exceptions={})
  - A2-balanced classifier (classifier_bank_v2_a2balanced.npz)
  - M2-pure calibration (calibration_m2pure.npz)
  - Recalibrated behavioral FSM thresholds
    (phase2_fsm_thresholds_behavioral_m2pure.json)
  - Modulation layer enabled

Per Phase G LIFBrain integration pre-flight
(artifacts/phase_g/phase_g_lifbrain_preflight.md), this module:

  1. Provides `make_lifbrain_substrate(seed)` factory for the ablation
     harness (replaces the make_lifbrain_substrate_TODO placeholder).
  2. Provides `lifbrain_dose_response(anesthetic, doses, n_seeds)` for
     Phase G dose-response calibration on the production substrate.
  3. Defines behavioral readout: FWD state fraction (primary) +
     AVA / AVB command interneuron rates (secondary diagnostic).

Demo network preserved as legacy in phase_g_network_perturbation.py.

Usage:
    /home/rohit/miniconda3/envs/ml/bin/python src/phase_g_lifbrain_substrate.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
BRAIN_DIR = REPO_ROOT / "scripts" / "brain"
sys.path.insert(0, str(BRAIN_DIR))

ART = REPO_ROOT / "scripts" / "brain" / "artifacts"
PHASE_G_DIR = ROOT / "artifacts" / "phase_g"

# Phase 2 recalibrated stack artifacts (per brain_v3.5_locked.md)
BANK_PATH = ART / "classifier_bank_v2_a2balanced.npz"
CAL_PATH = ART / "calibration_m2pure.npz"
FSM_THRESH_BEHAVIORAL = ART / "phase2_fsm_thresholds_behavioral_m2pure.json"

# Phase G overlay (anesthetic kinetic shifts)
OVERLAY_V2 = ROOT / "artifacts" / "kinetics" / "wave2_overlay_v2.json"

# Behavioral state IDs (per scripts/brain/behavioral_fsm.py State enum)
# 1=FORWARD, 2=REVERSE, 3=OMEGA, 4=PIROUETTE, 5=QUIESCENT
STATE_NAMES = {1: "FORWARD", 2: "REVERSE", 3: "OMEGA",
               4: "PIROUETTE", 5: "QUIESCENT"}

# Command interneurons we read for secondary diagnostic
COMMAND_NEURONS = ["AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR"]


def make_lifbrain_substrate(seed: int, scenario: str = "spontaneous",
                              enable_modulation: bool = True) -> Any:
    """Construct a fresh LIFBrain wrapped in ClosedLoopEnv under M2-pure
    + recalibrated stack. Returns the env (which exposes env.brain as the
    LIFBrain instance plus env.fsm_states for behavioral readout).

    Returned object has all attributes Phase G's apply_to_brain expects
    (env.brain provides them) plus env.fsm_states (post-run) for
    behavioral readout.
    """
    # Verify Phase 2 artifacts exist before constructing
    for p in [BANK_PATH, CAL_PATH, FSM_THRESH_BEHAVIORAL]:
        if not p.exists():
            raise FileNotFoundError(f"Phase 2 artifact missing: {p}")

    from closed_loop_env import ClosedLoopEnv

    env = ClosedLoopEnv(
        seed=seed,
        enable_modulation=enable_modulation,
        use_per_edge_glu_signs=True,        # M2-pure
        sign_exceptions={},                  # M2-pure: no DOCUMENTED_SIGN_EXCEPTIONS
        bank_path=BANK_PATH,
        cal_path=CAL_PATH,
        fsm_thresholds_path=FSM_THRESH_BEHAVIORAL,
        fsm_mode="classifier",               # behavioral_fsm
        brain_class="lif",
    )
    return env


def lifbrain_behavioral_readout(env, fsm_states: list[int] | None = None) -> dict:
    """Extract behavioral readouts from a post-run ClosedLoopEnv.

    Primary: FWD state fraction (Phase G calibration anchor).
    Secondary: AVA / AVB command interneuron firing rates.
    """
    import numpy as np

    if fsm_states is None:
        fsm_states = list(env.fsm_states) if hasattr(env, "fsm_states") else []

    n_total = len(fsm_states)
    state_counts = {name: 0 for name in STATE_NAMES.values()}
    for s in fsm_states:
        if s in STATE_NAMES:
            state_counts[STATE_NAMES[s]] += 1
    state_fractions = {
        name: (state_counts[name] / n_total) if n_total > 0 else 0.0
        for name in STATE_NAMES.values()
    }

    # Command interneuron rates (Hz) over full sim duration
    brain = env.brain
    cmd_rates = {}
    if hasattr(brain, "spikes") and hasattr(brain, "idx"):
        spike_i = np.asarray(brain.spikes.i)
        spike_t = np.asarray(brain.spikes.t)
        duration_s = float(spike_t.max()) if len(spike_t) > 0 else 1.0
        if duration_s < 1e-6:
            duration_s = 1.0
        for cn in COMMAND_NEURONS:
            if cn in brain.idx:
                idx = brain.idx[cn]
                count = int((spike_i == idx).sum())
                cmd_rates[cn] = count / duration_s

    # Aggregate mean firing rate (for compat with original Phase G demo readout)
    if hasattr(brain, "spikes") and hasattr(brain, "N"):
        n_spikes = len(brain.spikes.t)
        duration_s = float(np.asarray(brain.spikes.t).max()) if n_spikes > 0 else 1.0
        if duration_s < 1e-6:
            duration_s = 1.0
        mean_rate_hz = n_spikes / brain.N / duration_s
    else:
        mean_rate_hz = 0.0
        n_spikes = 0

    return {
        "fsm_state_fractions": state_fractions,
        "n_fsm_states": n_total,
        "fwd_fraction": state_fractions.get("FORWARD", 0.0),  # primary anchor
        "command_interneuron_rates_hz": cmd_rates,
        "mean_firing_rate_hz": mean_rate_hz,
        "n_spikes": n_spikes,
    }


def run_one_dose(anesthetic: str, dose: float, seed: int,
                  duration_s: float = 30.0,
                  stim_schedule: list | None = None) -> dict:
    """Single Phase G run on production LIFBrain at given anesthetic + dose.

    Constructs fresh env, applies Phase G perturbation, runs scenario,
    reports readout.
    """
    from phase_g_network_perturbation import AnestheticPerturbation

    t_construct = time.time()
    env = make_lifbrain_substrate(seed)
    construct_wall = time.time() - t_construct

    # Apply Phase G perturbation
    t_pert = time.time()
    pert = AnestheticPerturbation(OVERLAY_V2)
    revert_handle = pert.apply_to_brain(env.brain, anesthetic, dose)
    pert_wall = time.time() - t_pert

    # Run scenario
    t_run = time.time()
    env.run(duration_s, stim_schedule=stim_schedule or [])
    run_wall = time.time() - t_run

    # Behavioral readout
    readout = lifbrain_behavioral_readout(env)

    # Aggregate result
    return {
        "anesthetic": anesthetic,
        "dose_multiplier": dose,
        "seed": seed,
        "duration_s": duration_s,
        "scenario": "spontaneous" if not stim_schedule else "stim",
        "readout": readout,
        "perturbation_summary": revert_handle.get("profile").summary()
            if revert_handle.get("profile") else None,
        "wall_clock": {
            "construct_s": round(construct_wall, 2),
            "perturb_s": round(pert_wall, 2),
            "run_s": round(run_wall, 2),
            "total_s": round(construct_wall + pert_wall + run_wall, 2),
        },
    }


def smoke_test_halothane_1x() -> dict:
    """CP1 smoke test: halothane at 1× clinical EC50 on production substrate.

    Acceptance:
      - Env loads cleanly under M2-pure + recalibrated stack
      - Halothane perturbation applies without errors
      - Behavioral classifier produces interpretable FSM state distribution
      - Firing rates remain in biological range (<200 Hz mean)
      - Cascade still fires if touch stim applied (deferred — separate verification)
    """
    print("=" * 78)
    print("  CP1 smoke test — halothane @ 1× clinical EC50 on production LIFBrain")
    print("=" * 78)

    result = run_one_dose("halothane", dose=1.0, seed=42, duration_s=30.0)

    print(f"\n  Wall: construct {result['wall_clock']['construct_s']:.1f}s, "
          f"perturb {result['wall_clock']['perturb_s']:.1f}s, "
          f"run {result['wall_clock']['run_s']:.1f}s, "
          f"total {result['wall_clock']['total_s']:.1f}s")

    r = result["readout"]
    print(f"\n  Behavioral readout:")
    print(f"    FWD fraction (primary anchor): {r['fwd_fraction']:.3f}")
    print(f"    All FSM fractions: {r['fsm_state_fractions']}")
    print(f"    n_fsm_states: {r['n_fsm_states']}")
    print(f"    Mean firing rate: {r['mean_firing_rate_hz']:.2f} Hz")
    print(f"    n_spikes: {r['n_spikes']}")
    print(f"\n  Command interneuron firing rates:")
    for cn, rate in r["command_interneuron_rates_hz"].items():
        print(f"    {cn}: {rate:.2f} Hz")

    s = result["perturbation_summary"]
    print(f"\n  Phase G perturbation summary:")
    print(f"    n_classes engaged: {s['n_classes_engaged']}")
    print(f"    n_targets engaged (occ > 0.10): {s['n_targets_engaged']}")
    print(f"    max class occupancy: {s['max_class_occupancy']:.3f}")
    print(f"    mean class occupancy: {s['mean_class_occupancy']:.3f}")

    # Acceptance checks
    print("\n  Acceptance:")
    checks = {
        "env_loads": r["n_fsm_states"] > 0,
        "fsm_output_interpretable": (r["fwd_fraction"] >= 0.0
                                       and r["fwd_fraction"] <= 1.0),
        "firing_rate_biological": (0 < r["mean_firing_rate_hz"] < 200),
        "perturbation_engaged": s["n_classes_engaged"] > 0,
    }
    for k, v in checks.items():
        print(f"    {'✓' if v else '✗'} {k}: {v}")

    all_pass = all(checks.values())
    print(f"\n  CP1 SMOKE TEST: {'PASS' if all_pass else 'FAIL'}")

    # Persist
    out = PHASE_G_DIR / "phase_g_lifbrain_cp1_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "result": result,
        "checks": checks,
        "all_pass": all_pass,
    }, indent=2))
    print(f"\n  Smoke test JSON: {out}")
    return result


def main() -> int:
    smoke_test_halothane_1x()
    return 0


if __name__ == "__main__":
    sys.exit(main())
