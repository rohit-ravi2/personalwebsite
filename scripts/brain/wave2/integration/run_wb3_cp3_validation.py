"""
Phase δ WB3 CP3 — Numerical stability validation + AIY/RIM V_half sensitivity.

Three smoke tests under cross_coupling="graded_b2":
  CP3.1.a: 1 s spontaneous (no stim)
  CP3.1.b: 10 s spontaneous (no stim)
  CP3.1.c: 30 s spontaneous + touch_anterior at t=5s

For each: assert no NaN/Inf voltages, biological V range (-100 to +20 mV),
mean firing rate <100 Hz, count soft-cap warnings.

CP3.2: V_half ± 5 mV sensitivity analysis for AIY + RIM (Caveat 1).
30 s touch_anterior under three V_half values per cell:
  - cellular-anchored default (D)
  - D - 5 mV
  - D + 5 mV
Measure AIY/RIM firing rate change peri-touch + downstream effect on
LIF cells receiving from AIY/RIM.

CP3.3: soft-cap warning analysis. Tally + report per scenario.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
WAVE2_DIR = THIS_DIR.parent
BRAIN_DIR = WAVE2_DIR.parent
sys.path.insert(0, str(BRAIN_DIR))
sys.path.insert(0, str(WAVE2_DIR))
sys.path.insert(0, str(THIS_DIR))

from brian2 import ms, mV, pA, defaultclock

from wave2_hybrid_brain import Wave2HybridBrain, WB3_V_HALF_MV, WB3_SOFT_CAP_PA


def _measure_voltages(brain):
    """Return dict of W2 cell name -> V_mV at current time."""
    out = {}
    for name, grp in brain.wave2_groups.items():
        out[name] = float(grp.v[0] / mV)
    return out


def _summarize_run(brain, label):
    """Compute summary metrics for the current state of `brain`."""
    rates = brain.firing_rates(window_ms=500)
    v_w2 = _measure_voltages(brain)
    lif_v = np.asarray(brain.neurons.v[:] / mV, dtype=np.float64)
    return {
        "label": label,
        "t_ms": brain.time_ms(),
        "lif_spike_count": int(len(brain.spikes.t)),
        "n_active_gt0p5Hz": int(np.sum(rates > 0.5)),
        "mean_rate_Hz": float(rates.mean()),
        "max_rate_Hz": float(rates.max()),
        "lif_v_min_mV": float(lif_v.min()),
        "lif_v_max_mV": float(lif_v.max()),
        "lif_v_mean_mV": float(lif_v.mean()),
        "wave2_voltages_mV": v_w2,
        "soft_cap_warnings_total": int(brain.soft_cap_warning_count()),
    }


def _check_stability(metrics):
    """Assert stability invariants. Return list of failures (empty = pass)."""
    fails = []
    # Voltage finiteness — no NaN/Inf
    if not np.isfinite(metrics["lif_v_min_mV"]):
        fails.append(f"LIF V_min not finite: {metrics['lif_v_min_mV']}")
    if not np.isfinite(metrics["lif_v_max_mV"]):
        fails.append(f"LIF V_max not finite: {metrics['lif_v_max_mV']}")
    for name, v in metrics["wave2_voltages_mV"].items():
        if not np.isfinite(v):
            fails.append(f"{name} V not finite: {v}")
        elif not (-100.0 <= v <= 20.0):
            fails.append(f"{name} V outside biological range [-100, +20] mV: {v:.2f}")
    # LIF voltage range — biological window for LIF (parameters: v_rest=-22 mV
    # with bias, v_thr=-10, v_reset=-30, noise sigma 6 mV → expect [-50, -8])
    if metrics["lif_v_min_mV"] < -100:
        fails.append(f"LIF V_min < -100 mV: {metrics['lif_v_min_mV']}")
    if metrics["lif_v_max_mV"] > 20:
        fails.append(f"LIF V_max > +20 mV: {metrics['lif_v_max_mV']}")
    # Firing rate sanity — no runaway
    if metrics["mean_rate_Hz"] > 100.0:
        fails.append(f"Runaway: mean rate {metrics['mean_rate_Hz']:.1f} Hz > 100")
    return fails


# ---------------------------------------------------------------------------
# CP3.1 — Numerical stability smoke tests
# ---------------------------------------------------------------------------

def cp3_1_a_smoke_1s():
    print("\n" + "=" * 70)
    print("CP3.1.a: 1 s spontaneous smoke (graded_b2)")
    print("=" * 70)
    defaultclock.dt = 0.1 * ms
    brain = Wave2HybridBrain(
        wave2_active=["AVAL", "AVAR"],
        cross_coupling="graded_b2",
        seed=42,
    )
    t0 = time.time()
    brain.run(1000)
    wall = time.time() - t0
    m = _summarize_run(brain, "1s_spontaneous")
    m["wall_time_s"] = wall
    m["fails"] = _check_stability(m)
    print(json.dumps(m, indent=2))
    return m


def cp3_1_b_smoke_10s():
    print("\n" + "=" * 70)
    print("CP3.1.b: 10 s spontaneous smoke (graded_b2)")
    print("=" * 70)
    defaultclock.dt = 0.1 * ms
    brain = Wave2HybridBrain(
        wave2_active=["AVAL", "AVAR"],
        cross_coupling="graded_b2",
        seed=42,
    )
    t0 = time.time()
    brain.run(10000)
    wall = time.time() - t0
    m = _summarize_run(brain, "10s_spontaneous")
    m["wall_time_s"] = wall
    m["fails"] = _check_stability(m)
    print(json.dumps(m, indent=2))
    return m


def cp3_1_c_smoke_30s():
    print("\n" + "=" * 70)
    print("CP3.1.c: 30 s spontaneous + touch_anterior @ t=5s (graded_b2)")
    print("=" * 70)
    defaultclock.dt = 0.1 * ms
    brain = Wave2HybridBrain(
        wave2_active=["AVAL", "AVAR"],
        cross_coupling="graded_b2",
        seed=42,
    )
    t0 = time.time()
    # 5 s spontaneous → 2 s touch → 23 s recovery (= 30 s total)
    brain.run(5000)
    pre_metrics = _summarize_run(brain, "pre_touch_5s")
    for n in ["ALML", "ALMR", "AVM"]:
        if n in brain.idx:
            brain.inject_poisson(n, 200, weight_mv=8)
    brain.run(2000)
    touch_metrics = _summarize_run(brain, "touch_2s")
    brain.run(23000)
    wall = time.time() - t0
    m = _summarize_run(brain, "30s_with_touch")
    m["wall_time_s"] = wall
    m["pre_touch"] = pre_metrics
    m["touch"] = touch_metrics
    m["fails"] = _check_stability(m)
    print(json.dumps(m, indent=2))
    return m


# ---------------------------------------------------------------------------
# CP3.2 — V_half ± 5 mV sensitivity for AIY + RIM (Caveat 1)
# ---------------------------------------------------------------------------

def _run_v_half_sweep_for_cell(cell_name, deltas_mV, duration_ms=30000):
    """Run a 30 s touch_anterior scenario for each V_half offset.

    Note: AIYL/AIYR/RIML/RIMR are stored in WB3_V_HALF_MV. We sweep them
    as a coordinated pair (left+right share the same V_half override).
    """
    print(f"\n--- V_half sweep for {cell_name} ---")
    pair_left, pair_right = (
        (f"{cell_name}L", f"{cell_name}R")
        if cell_name in ("AIY", "RIM") else (cell_name, cell_name)
    )
    default_V_half = WB3_V_HALF_MV[pair_left]
    results = []
    for delta in deltas_mV:
        new_v_half = default_V_half + delta
        v_half_overrides = {pair_left: new_v_half, pair_right: new_v_half}
        print(f"  {cell_name} V_half = {new_v_half:+.1f} mV (default {default_V_half:+.1f}, delta {delta:+.1f})")
        defaultclock.dt = 0.1 * ms
        # AIY/RIM may not be in the WAVE2_CELL_FACTORIES default set
        # active by Wave2HybridBrain. Include them when sweeping.
        wave2_active = ["AVAL", "AVAR"]
        if cell_name == "AIY":
            wave2_active = ["AVAL", "AVAR", "AIYL", "AIYR"]
        elif cell_name == "RIM":
            wave2_active = ["AVAL", "AVAR", "RIML", "RIMR"]
        brain = Wave2HybridBrain(
            wave2_active=wave2_active,
            cross_coupling="graded_b2",
            v_half_overrides=v_half_overrides,
            seed=42,
        )
        # 5 s spontaneous → 2 s touch → 23 s recovery
        brain.run(5000)
        baseline_rates = brain.firing_rates(2000)
        baseline_v = _measure_voltages(brain)
        for n in ["ALML", "ALMR", "AVM"]:
            if n in brain.idx:
                brain.inject_poisson(n, 200, weight_mv=8)
        brain.run(2000)
        touch_rates = brain.firing_rates(2000)
        touch_v = _measure_voltages(brain)
        brain.run(23000)
        recovery_rates = brain.firing_rates(2000)

        # Downstream LIF cells: cells that receive from AIY/RIM
        # (LIF post + AIY/RIM source). Find via _cross_chem_edges.
        downstream_cells = set()
        for e in brain._cross_chem_edges:
            if e["pre_kind"] in (pair_left, pair_right) and e["post_kind"] == "lif":
                downstream_cells.add(brain.names[e["post_global"]])

        cell_metrics = {}
        for cn in [pair_left, pair_right]:
            if cn in brain.idx:
                gi = brain.idx[cn]
                cell_metrics[cn] = {
                    "baseline_Hz": float(baseline_rates[gi]),
                    "touch_Hz": float(touch_rates[gi]),
                    "recovery_Hz": float(recovery_rates[gi]),
                    "delta_touch_Hz": float(touch_rates[gi] - baseline_rates[gi]),
                    "baseline_V_mV": baseline_v.get(cn, float("nan")),
                    "touch_V_mV": touch_v.get(cn, float("nan")),
                }

        downstream_metrics = {}
        for cn in sorted(downstream_cells):
            if cn in brain.idx:
                gi = brain.idx[cn]
                downstream_metrics[cn] = {
                    "baseline_Hz": float(baseline_rates[gi]),
                    "touch_Hz": float(touch_rates[gi]),
                    "delta_touch_Hz": float(touch_rates[gi] - baseline_rates[gi]),
                }

        results.append({
            "cell": cell_name,
            "delta_V_half_mV": float(delta),
            "V_half_mV": float(new_v_half),
            "cell_metrics": cell_metrics,
            "n_downstream_cells": len(downstream_metrics),
            "downstream_metrics": downstream_metrics,
            "soft_cap_warnings_total": int(brain.soft_cap_warning_count()),
        })

        for cn, m in cell_metrics.items():
            print(
                f"    {cn}: baseline {m['baseline_Hz']:.2f} Hz, "
                f"touch {m['touch_Hz']:.2f} Hz "
                f"(Δ {m['delta_touch_Hz']:+.2f} Hz), "
                f"V {m['baseline_V_mV']:+.2f} → {m['touch_V_mV']:+.2f} mV"
            )
    return results


def cp3_2_v_half_sensitivity():
    print("\n" + "=" * 70)
    print("CP3.2: V_half ± 5 mV sensitivity for AIY + RIM (Caveat 1)")
    print("=" * 70)
    deltas_mV = [-5.0, 0.0, +5.0]
    aiy_results = _run_v_half_sweep_for_cell("AIY", deltas_mV)
    rim_results = _run_v_half_sweep_for_cell("RIM", deltas_mV)
    return {"AIY": aiy_results, "RIM": rim_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out = {}

    # CP3.1
    out["cp3_1_a_1s"] = cp3_1_a_smoke_1s()
    out["cp3_1_b_10s"] = cp3_1_b_smoke_10s()
    out["cp3_1_c_30s_touch"] = cp3_1_c_smoke_30s()

    # CP3.2
    out["cp3_2_v_half_sensitivity"] = cp3_2_v_half_sensitivity()

    # CP3.3 soft-cap warning analysis (using existing data)
    cp3_3_summary = {}
    for label in ["cp3_1_a_1s", "cp3_1_b_10s", "cp3_1_c_30s_touch"]:
        cp3_3_summary[label] = out[label]["soft_cap_warnings_total"]
    out["cp3_3_soft_cap_warning_counts"] = cp3_3_summary

    # Persist
    out_path = WAVE2_DIR / "artifacts" / "phase_delta_wb3_cp3_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[CP3 results written] {out_path}")

    # Print acceptance summary
    print("\n" + "=" * 70)
    print("CP3 Acceptance Summary")
    print("=" * 70)
    for label in ["cp3_1_a_1s", "cp3_1_b_10s", "cp3_1_c_30s_touch"]:
        m = out[label]
        status = "PASS" if not m["fails"] else "FAIL"
        print(f"  [{status}] {label}: V_w2 in {min(m['wave2_voltages_mV'].values()):+.1f} to "
              f"{max(m['wave2_voltages_mV'].values()):+.1f} mV; "
              f"mean LIF rate {m['mean_rate_Hz']:.1f} Hz; "
              f"soft-cap warnings {m['soft_cap_warnings_total']}; "
              f"wall time {m['wall_time_s']:.1f} s; "
              f"fails: {m['fails']}")
    return out


if __name__ == "__main__":
    main()
