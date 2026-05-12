#!/usr/bin/env python3
"""Phase 2 sub-tasks 2.4a + 2.4b — FSM threshold recalibration under M2-pure.

Reproduces the v1.5 methodology used to set
`behavioral_fsm.TRANSITION_THRESHOLDS` and
`activity_fsm.ROLE_Z_THRESHOLD`, but under M2-pure brain mode + new
classifier (`classifier_bank_v2_a2balanced.npz`) + new calibration
(`calibration_m2pure.npz`).

Methodology (per behavioral_fsm.py:60-71):
  Run closed_loop_env under varied stimuli; for each event classifier,
  collect output probability distribution; set transition threshold near
  the 95th percentile of observed values. For activity_fsm: collect
  per-role z-scored rate distributions during baseline + stim windows;
  set z-score thresholds where stim-driven activation is separable from
  baseline noise.

Outputs:
  artifacts/phase2_fsm_thresholds_behavioral_m2pure.json
  artifacts/phase2_fsm_thresholds_activity_m2pure.json
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from closed_loop_env import ClosedLoopEnv  # noqa: E402
from activity_fsm import ROLE_NEURONS, BASELINE_TAU_S, WINDOW_S  # noqa: E402

ART = THIS_DIR / "artifacts"
BANK_PATH = ART / "classifier_bank_v2_a2balanced.npz"
CAL_PATH = ART / "calibration_m2pure.npz"

# Events we recalibrate (matches classifier bank events)
EVENTS = [
    "reversal_onset", "reversal_offset",
    "forward_run_onset", "forward_run_offset",
    "omega_onset", "pirouette_entry",
    "quiescence_onset", "speed_burst_onset",
]

# Scenarios for varied-stim distribution sampling (~30 s each)
SCENARIOS = [
    ("spontaneous", []),
    ("touch", [(5.0, "touch_anterior", 1.0)]),
    ("osmotic_shock", [(5.0, "osmotic_shock", 1.0)]),
    ("food", [(2.0, "food_signal", 1.0)]),
]

DURATION_S = 30.0
N_SEEDS = 3  # keep modest — distribution sampling, not phenotype


def m2pure_kwargs():
    """Cfg kwargs for ClosedLoopEnv under M2-pure + new classifier + new calibration."""
    return dict(
        use_per_edge_glu_signs=True,
        sign_exceptions={},
        bank_path=BANK_PATH,
        cal_path=CAL_PATH,
        enable_modulation=True,
    )


def collect_classifier_outputs(scenarios, n_seeds, duration_s):
    """For each scenario × seed, run closed loop, collect per-event classifier
    output probability time-series, return dict event → list of arrays."""
    per_event = {e: [] for e in EVENTS}
    n_runs = len(scenarios) * n_seeds
    print(f"  collecting classifier outputs over {n_runs} runs...")
    run_idx = 0
    for scen_name, stim in scenarios:
        for seed in range(42, 42 + n_seeds):
            run_idx += 1
            print(f"    run {run_idx}/{n_runs}: {scen_name} seed={seed}")
            env = ClosedLoopEnv(seed=seed, **m2pure_kwargs())
            env.run(duration_s, stim_schedule=stim)
            # env.event_probs is dict[event → list[prob per classifier step]]
            for e in EVENTS:
                if e in env.event_probs:
                    per_event[e].extend(float(p) for p in env.event_probs[e])
    return per_event


def collect_role_rate_distributions(scenarios, n_seeds, duration_s):
    """Per-role per-cell firing-rate distributions over time. For each role,
    compute (rate_mean, rate_std, baseline EMA) per scenario. Returns dict."""
    role_traces = defaultdict(list)  # role -> list of (T, n_role_cells) arrays
    role_baselines = defaultdict(list)
    n_runs = len(scenarios) * n_seeds
    run_idx = 0
    print(f"  collecting role rate distributions over {n_runs} runs...")
    for scen_name, stim in scenarios:
        for seed in range(42, 42 + n_seeds):
            run_idx += 1
            print(f"    run {run_idx}/{n_runs}: {scen_name} seed={seed}")
            env = ClosedLoopEnv(seed=seed, **m2pure_kwargs())
            env.run(duration_s, stim_schedule=stim)

            # Reconstruct per-cell firing rates from full_spike_buffer
            from closed_loop_env import BRAIN_SYNC_MS
            if not env.full_spike_buffer:
                continue
            fsb = np.stack(env.full_spike_buffer)  # (T, N) uint8
            dt_s = BRAIN_SYNC_MS / 1000.0
            T = fsb.shape[0]
            # Per-role: for each role's cells in env.brain.idx, compute rate
            for role, cells in ROLE_NEURONS.items():
                cell_idx = [env.brain.idx[c] for c in cells if c in env.brain.idx]
                if not cell_idx:
                    continue
                # Rolling-window firing rate at WINDOW_S resolution
                w_steps = max(1, int(WINDOW_S / dt_s))
                # Use simple moving-mean for rate
                cum = np.cumsum(fsb[:, cell_idx], axis=0).astype(np.float32)
                rate_per_step = (cum[w_steps:] - cum[:-w_steps]) / (w_steps * dt_s)
                # Mean across role cells (FSM uses max — capture both)
                role_max_rate = rate_per_step.max(axis=1)
                role_traces[role].append(role_max_rate)

    return role_traces


def main():
    assert BANK_PATH.exists(), f"missing {BANK_PATH}"
    assert CAL_PATH.exists(), f"missing {CAL_PATH}"
    print("=" * 78)
    print("  Phase 2 sub-tasks 2.4 — FSM threshold recalibration under M2-pure")
    print("=" * 78)
    print(f"  bank: {BANK_PATH.name}")
    print(f"  calibration: {CAL_PATH.name}")
    print(f"  scenarios: {[s[0] for s in SCENARIOS]}")
    print(f"  n_seeds × duration: {N_SEEDS} × {DURATION_S} s")
    print()

    # =================================================================
    # Sub-task 2.4a — behavioral_fsm thresholds (classifier-based)
    # =================================================================
    print(">>> Sub-task 2.4a: behavioral_fsm threshold recalibration")
    print(">>> Approach: run closed loop under varied stim; collect classifier")
    print(">>> output distribution per event; set thresholds at chosen percentile.")
    print()
    per_event = collect_classifier_outputs(SCENARIOS, N_SEEDS, DURATION_S)

    behavioral_thresholds = {}
    print(f"\n  {'event':<22} {'n_obs':>8} {'p50':>8} {'p90':>8} "
          f"{'p95':>8} {'p99':>8} {'recommended':>12}")
    for e in EVENTS:
        obs = np.array(per_event[e]) if per_event[e] else np.array([])
        if len(obs) == 0:
            print(f"  {e:<22} (no observations — env may not log per-step probs)")
            behavioral_thresholds[e] = None
            continue
        p50 = float(np.percentile(obs, 50))
        p90 = float(np.percentile(obs, 90))
        p95 = float(np.percentile(obs, 95))
        p99 = float(np.percentile(obs, 99))
        # Pick threshold: 95th percentile (matches v1.5 methodology)
        threshold = p95
        behavioral_thresholds[e] = {
            "threshold_p95": threshold,
            "n_obs": int(len(obs)),
            "p50": p50, "p90": p90, "p95": p95, "p99": p99,
            "min": float(obs.min()), "max": float(obs.max()),
            "mean": float(obs.mean()), "std": float(obs.std()),
        }
        print(f"  {e:<22} {len(obs):>8} {p50:>8.3f} {p90:>8.3f} "
              f"{p95:>8.3f} {p99:>8.3f} {threshold:>12.3f}")

    out_a = ART / "phase2_fsm_thresholds_behavioral_m2pure.json"
    out_a.write_text(json.dumps({
        "method": "95th-percentile of classifier output under varied stim, M2-pure brain",
        "bank": str(BANK_PATH.name),
        "cal": str(CAL_PATH.name),
        "scenarios": [s[0] for s in SCENARIOS],
        "n_seeds": N_SEEDS,
        "duration_s": DURATION_S,
        "thresholds": behavioral_thresholds,
    }, indent=2))
    print(f"  wrote {out_a}")
    print()

    # =================================================================
    # Sub-task 2.4b — activity_fsm thresholds (rate-based z-score)
    # =================================================================
    print(">>> Sub-task 2.4b: activity_fsm threshold recalibration")
    print(">>> Approach: collect per-role max-rate-across-cells over scenarios;")
    print(">>> set z-score threshold separating stim activation from baseline noise.")
    print()
    role_traces = collect_role_rate_distributions(SCENARIOS, N_SEEDS, DURATION_S)

    activity_thresholds = {}
    print(f"\n  {'role':<18} {'baseline μ':>11} {'baseline σ':>11} "
          f"{'stim peak':>10} {'z stim':>8} {'recommended':>12}")
    for role in ROLE_NEURONS.keys():
        traces = role_traces.get(role, [])
        if not traces:
            print(f"  {role:<18} (no traces)")
            continue
        all_rates = np.concatenate(traces)
        # Estimate baseline μ/σ from earliest 30% of each trace (pre-stim window)
        baseline_samples = []
        for tr in traces:
            n_baseline = int(0.3 * len(tr))
            baseline_samples.append(tr[:n_baseline])
        baseline = np.concatenate(baseline_samples)
        b_mu, b_sd = float(baseline.mean()), float(baseline.std() + 1e-6)
        # Stim-period peak: take 95th percentile of the stim window (50-70% of run)
        stim_samples = []
        for tr in traces:
            stim_start = int(0.5 * len(tr))
            stim_end = int(0.7 * len(tr))
            stim_samples.append(tr[stim_start:stim_end])
        stim = np.concatenate(stim_samples) if stim_samples else np.array([])
        if len(stim) > 0:
            stim_peak = float(np.percentile(stim, 95))
            z_stim = (stim_peak - b_mu) / b_sd
        else:
            stim_peak = float("nan")
            z_stim = float("nan")
        # Recommended: z-threshold = max(2.5, z_stim * 0.7) — capture stim
        # activation at 70% of observed peak, but never below 2.5σ
        recommended = max(2.5, z_stim * 0.7) if not np.isnan(z_stim) else 2.5
        activity_thresholds[role] = {
            "baseline_mean_hz": b_mu,
            "baseline_std_hz": b_sd,
            "stim_peak_hz_p95": stim_peak,
            "z_stim_observed": z_stim,
            "recommended_z_threshold": recommended,
            "n_traces": len(traces),
        }
        print(f"  {role:<18} {b_mu:>11.3f} {b_sd:>11.3f} {stim_peak:>10.3f} "
              f"{z_stim:>8.2f} {recommended:>12.2f}")

    out_b = ART / "phase2_fsm_thresholds_activity_m2pure.json"
    out_b.write_text(json.dumps({
        "method": ("baseline μ/σ from first 30% of run; stim peak from 50-70%; "
                   "z-threshold = max(2.5, z_stim * 0.7)"),
        "bank": str(BANK_PATH.name),
        "cal": str(CAL_PATH.name),
        "scenarios": [s[0] for s in SCENARIOS],
        "n_seeds": N_SEEDS,
        "duration_s": DURATION_S,
        "baseline_tau_s": BASELINE_TAU_S,
        "window_s": WINDOW_S,
        "thresholds": activity_thresholds,
    }, indent=2))
    print(f"  wrote {out_b}")
    print()
    print("=" * 78)
    print("  Phase 2 sub-tasks 2.4 complete")
    print("=" * 78)


if __name__ == "__main__":
    main()
