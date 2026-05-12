"""
Phase δ WB3 D7-followup CP2 — touch-cascade re-validation with σ-magnitude readout.

Replays the WB3 CP4 touch_anterior protocol under the canonical post-WB3-D7-followup
σ-magnitude readout. The original CP4 (`run_wb3_cp4_validation.py`) measured Wave 2
activity via `firing_rates()` which, in graded_b2 mode, used the σ > 0.5 rising-
threshold pseudo-spike detector (Decision 7 (a)). That detector reports 0 events when
σ saturates above threshold — the WB3 CP4 readout artifact. The σ-magnitude readout
(Wave2HybridBrain.firing_rates() returning σ_mean × 100; Wave2HybridBrain.
wave2_activities() returning raw σ ∈ [0, 1]) replaces the rising-threshold readout
in firing_rates() while preserving wave2_pseudo_spikes for explicit consumers.

CP2 acceptance: AVA Δ peri-touch via σ-magnitude readout is substantively non-zero
at the W_graded_I values where the original CP4 reported Δ < ±1 Hz.

Test points:
  - W_graded_I = 0.3 pA (Mellem-calibrated default; original CP4 AVAL Δ -0.5 Hz)
  - W_graded_I = 10  pA (CP4 ceiling; original CP4 AVAL Δ 0.0 Hz, V saturated -16 mV)

Per-cell-type readout contract documented in Wave2HybridBrain.firing_rates() docstring:
  - LIF cells:  firing_rates() → spike-count / window in Hz (genuine rate)
  - Wave 2:     firing_rates() → σ_mean × 100 (Hz-flavored proxy, [0, 100])
                wave2_activities() → σ_mean ∈ [0, 1] (raw magnitude)
  - Legacy:     wave2_pseudo_spikes still populated (rising-threshold events) but
                NOT consumed by firing_rates() — preserved for explicit consumers.
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

from brian2 import ms, mV, defaultclock

from wave2_hybrid_brain import Wave2HybridBrain


# Stage IV per-edge LIF baseline (from stage_IV_touch_cascade_findings.md)
STAGE_IV_LIF_AVAL_DELTA_HZ = 7.50

# Original CP4 results (pseudo-spike rate readout — the artifact we're correcting)
# Source: phase_delta_wb3_cp4_results.json
CP4_ORIGINAL = {
    "0.3": {
        "AVAL_baseline_Hz_artifact": 1.5,
        "AVAL_touch_Hz_artifact": 1.0,
        "AVAL_delta_Hz_artifact": -0.5,
        "AVAR_delta_Hz_artifact": 1.0,
        "AVAL_pseudo_spike_count_total": 21,
        "AVAR_pseudo_spike_count_total": 21,
        "AVAL_v_end_mV": -23.71,
        "AVAR_v_end_mV": -24.26,
    },
    "10.0": {
        "AVAL_baseline_Hz_artifact": 0.0,
        "AVAL_touch_Hz_artifact": 0.0,
        "AVAL_delta_Hz_artifact": 0.0,
        "AVAR_delta_Hz_artifact": 0.0,
        "AVAL_pseudo_spike_count_total": 1,
        "AVAR_pseudo_spike_count_total": 1,
        "AVAL_v_end_mV": -16.09,
        "AVAR_v_end_mV": -15.61,
    },
}

# Match WB3 CP4 cell list for direct comparison (LIF cascade unchanged)
KEY_LIF_CELLS = [
    "ALML", "ALMR", "AVM", "PVCL", "PVCR", "AVDL", "AVDR",
    "AVEL", "AVER", "AVBL", "AVBR",
    "AIBL", "AIBR", "RIML", "RIMR", "AIYL", "AIYR",
]
WAVE2_CELLS = ["AVAL", "AVAR"]


def run_cp2_touch_cascade(W_graded_I_pA, seed=42, use_per_edge_glu_signs=False):
    """30 s touch_anterior protocol; record both new σ-magnitude readout and
    legacy pseudo-spike counts for direct artifact-resolution comparison.

    Protocol matches WB3 CP4 (`run_wb3_cp4_validation.py:run_touch_cascade`):
      0-2 s    settle
      2-5 s    spontaneous baseline
      5-7 s    touch_anterior (200 Hz Poisson on ALML/ALMR/AVM @ 8 mV)
      7-9 s   recovery

    Sign mode: pass `use_per_edge_glu_signs=True` to test under per-edge
    CeNGEN-derived sign convention. Default (False) uses per-presynaptic-neuron
    NT signs with ~26 hand-picked overrides (`DEFAULT_SIGN_OVERRIDES` in
    `lif_brain.py`). Per-edge mode is documented in §5 / Stage IV as the
    sign mode under which the touch cascade fires through to AVA in pure LIF.
    """
    defaultclock.dt = 0.1 * ms
    brain = Wave2HybridBrain(
        wave2_active=WAVE2_CELLS,
        cross_coupling="graded_b2",
        W_graded_I_pA=W_graded_I_pA,
        seed=seed,
        use_per_edge_glu_signs=use_per_edge_glu_signs,
    )

    # 2 s settle
    brain.run(2000)

    # 3 s baseline
    brain.run(3000)
    rates_baseline = brain.firing_rates(window_ms=2000)
    sigma_baseline = brain.wave2_activities(window_ms=2000)

    # 2 s touch
    for n in ["ALML", "ALMR", "AVM"]:
        if n in brain.idx:
            brain.inject_poisson(n, 200, weight_mv=8)
    brain.run(2000)
    rates_touch = brain.firing_rates(window_ms=2000)
    sigma_touch = brain.wave2_activities(window_ms=2000)

    # 2 s recovery
    brain.run(2000)
    rates_recovery = brain.firing_rates(window_ms=2000)
    sigma_recovery = brain.wave2_activities(window_ms=2000)

    v_w2 = {name: float(grp.v[0] / mV) for name, grp in brain.wave2_groups.items()}

    out = {
        "W_graded_I_pA": float(W_graded_I_pA),
        "wave2_active": list(WAVE2_CELLS),
        "soft_cap_warnings_total": int(brain.soft_cap_warning_count()),
        "wave2_voltages_end_mV": v_w2,
        "lif_cells": {},
        "wave2_cells": {},
        "wave2_legacy_pseudo_spike_counts": {
            name: len(brain.wave2_pseudo_spikes.get(name, []))
            for name in WAVE2_CELLS
        },
    }
    for c in KEY_LIF_CELLS:
        if c in brain.idx:
            i = brain.idx[c]
            out["lif_cells"][c] = {
                "baseline_Hz": float(rates_baseline[i]),
                "touch_Hz": float(rates_touch[i]),
                "recovery_Hz": float(rates_recovery[i]),
                "delta_touch_Hz": float(rates_touch[i] - rates_baseline[i]),
            }
    out["sign_mode"] = "per_edge" if use_per_edge_glu_signs else "default"
    for c in WAVE2_CELLS:
        if c in brain.idx:
            i = brain.idx[c]
            sb = float(sigma_baseline.get(c, 0.0))
            st = float(sigma_touch.get(c, 0.0))
            sr = float(sigma_recovery.get(c, 0.0))
            out["wave2_cells"][c] = {
                # σ-magnitude readout (canonical D7-followup)
                "sigma_baseline": sb,
                "sigma_touch": st,
                "sigma_recovery": sr,
                "sigma_delta_touch": st - sb,
                # Hz-equivalent proxy via firing_rates (×100 of σ)
                "firing_rate_baseline_proxy": float(rates_baseline[i]),
                "firing_rate_touch_proxy": float(rates_touch[i]),
                "firing_rate_delta_proxy": float(rates_touch[i] - rates_baseline[i]),
            }
    return out


def format_report(out):
    lines = []
    W = out["W_graded_I_pA"]
    lines.append(
        f"\n  W_graded_I = {W} pA, soft-cap warnings {out['soft_cap_warnings_total']}, "
        f"AVAL V_end {out['wave2_voltages_end_mV'].get('AVAL', float('nan')):+.1f} mV, "
        f"AVAR V_end {out['wave2_voltages_end_mV'].get('AVAR', float('nan')):+.1f} mV"
    )
    lines.append(f"\n  LIF cascade (firing_rates() → Hz, genuine rate):")
    lines.append(f"  {'cell':<8}{'baseline':>10}{'touch':>10}{'Δ_touch':>10}")
    for c in KEY_LIF_CELLS:
        if c in out["lif_cells"]:
            m = out["lif_cells"][c]
            lines.append(
                f"  {c:<6}{m['baseline_Hz']:>10.2f}{m['touch_Hz']:>10.2f}"
                f"{m['delta_touch_Hz']:>+10.2f}"
            )
    lines.append(
        f"\n  Wave 2 cells (σ-magnitude readout — canonical post-D7-followup):"
    )
    lines.append(
        f"  {'cell':<8}{'σ_base':>10}{'σ_touch':>10}{'Δσ':>10}"
        f"{'rate_base':>12}{'rate_touch':>12}{'Δrate':>10}"
    )
    for c in WAVE2_CELLS:
        if c in out["wave2_cells"]:
            m = out["wave2_cells"][c]
            lines.append(
                f"  {c:<6}{m['sigma_baseline']:>10.4f}{m['sigma_touch']:>10.4f}"
                f"{m['sigma_delta_touch']:>+10.4f}"
                f"{m['firing_rate_baseline_proxy']:>12.2f}"
                f"{m['firing_rate_touch_proxy']:>12.2f}"
                f"{m['firing_rate_delta_proxy']:>+10.2f}"
            )
    lines.append(
        f"\n  Legacy pseudo-spike counts (Decision 7(a) detector, NOT consumed "
        f"by firing_rates()):"
    )
    for c in WAVE2_CELLS:
        n = out["wave2_legacy_pseudo_spike_counts"].get(c, 0)
        lines.append(f"    {c}: {n} events over 9 s")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-edge",
        action="store_true",
        help="Run with use_per_edge_glu_signs=True (CeNGEN per-edge sign mode). "
             "Default off (per-presynaptic-neuron NT signs).",
    )
    args = parser.parse_args()
    use_per_edge = args.per_edge
    sign_mode_label = "per_edge" if use_per_edge else "default"

    out = {
        "test_protocol": "30 s touch_anterior (2 s settle + 3 s baseline + 2 s touch + 2 s recovery)",
        "stim": "200 Hz Poisson on ALML/ALMR/AVM, 8 mV/spike",
        "sign_mode": sign_mode_label,
        "use_per_edge_glu_signs": use_per_edge,
        "wave2_active": list(WAVE2_CELLS),
        "readout_contract": {
            "LIF_cells": "firing_rates() → spike-count/window in Hz (genuine rate)",
            "Wave2_cells_sigma_magnitude": "wave2_activities() → σ_mean ∈ [0, 1] (raw)",
            "Wave2_cells_rate_proxy": "firing_rates() → σ_mean × 100 (Hz-flavored proxy)",
            "Wave2_cells_legacy_pseudo_spikes": (
                "wave2_pseudo_spikes still populated by σ>0.5 rising-threshold "
                "detector but NOT driving firing_rates() (CP4 saturation artifact)"
            ),
        },
        "stage_iv_per_edge_LIF_baseline_AVAL_delta_Hz": STAGE_IV_LIF_AVAL_DELTA_HZ,
        "cp4_original_pseudo_spike_artifact": CP4_ORIGINAL,
    }

    test_points = [0.3, 10.0]
    for W in test_points:
        print("\n" + "=" * 78)
        print(f"  CP2 test point: W_graded_I = {W} pA, sign_mode={sign_mode_label}")
        print("=" * 78)
        t0 = time.time()
        result = run_cp2_touch_cascade(
            W_graded_I_pA=W, seed=42, use_per_edge_glu_signs=use_per_edge,
        )
        result["wall_time_s"] = time.time() - t0
        print(format_report(result))
        out[f"W_{W}_pA"] = result

    suffix = "_peredge" if use_per_edge else "_default"
    out_path = WAVE2_DIR / "artifacts" / f"phase_delta_wb3_d7followup_cp2{suffix}_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[CP2 results written] {out_path}")

    # Acceptance summary
    print("\n" + "=" * 78)
    print("  CP2 ACCEPTANCE SUMMARY — σ-magnitude readout vs original CP4 artifact")
    print("=" * 78)
    print(f"  {'W_pA':>6}{'cell':>6}"
          f"{'orig pseudo Δ_Hz':>22}"
          f"{'σ Δ (raw)':>14}"
          f"{'σ×100 Δ (proxy)':>20}")
    for W in test_points:
        r = out[f"W_{W}_pA"]
        for cell in WAVE2_CELLS:
            orig = CP4_ORIGINAL[str(W)][f"{cell}_delta_Hz_artifact"]
            new_raw = r["wave2_cells"][cell]["sigma_delta_touch"]
            new_proxy = r["wave2_cells"][cell]["firing_rate_delta_proxy"]
            print(f"  {W:>6}{cell:>6}"
                  f"{orig:>+22.4f}"
                  f"{new_raw:>+14.4f}"
                  f"{new_proxy:>+20.4f}")
    print(
        "\n  ARTIFACT RESOLVED if σ Δ shows substantive cascade activation where "
        "original\n  pseudo-spike Δ was ≈ 0 (especially at W=10 pA where σ saturated)."
    )

    return out


if __name__ == "__main__":
    main()
