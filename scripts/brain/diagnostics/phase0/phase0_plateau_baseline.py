#!/usr/bin/env python3
"""Phase 0 — W0.4a — T4-2 plateau baseline.

For each of the 15 neurons in COMPARTMENTAL_ROSTER, run a current-
injection protocol and measure plateau characteristics vs Gao & Hobert
2020 target values. Baseline documents the current-state gap per neuron
before T4-2 calibration work begins.

Protocol (Gao & Hobert 2020 Fig 3 analog):
  t=0..200ms    settle (I_ext = 0)
  t=200..300ms  inject 50 pA into soma (v_s)
  t=300..1200ms record post-injection v_s and v_d (plateau detection)

Measured per neuron:
  v_s_peak_mv     peak soma voltage during injection
  v_d_peak_mv     peak dendrite voltage during injection
  v_d_sustained   v_d 500ms post-injection (should stay elevated if plateau works)
  plateau_duration_ms  time from injection-release to v_d within 5mV of v_rest
  plateau_amplitude_mv v_d_peak - v_rest

Gao & Hobert 2020 targets (AVA specifically):
  plateau duration:   400-800 ms
  plateau amplitude:  ~20 mV above rest (v_rest = -65mV, so plateau ~-45mV)
  non-plateau neurons should show NO sustained depolarisation

Output: artifacts/phase0_plateau_baseline.csv + .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brian2 import (
    StateMonitor, Network, start_scope, ms, mV, pA, second, defaultclock
)

from compartmental_neurons import (
    build_compartmental_group, COMPARTMENTAL_ROSTER
)

ART = Path(__file__).resolve().parent.parent.parent / "artifacts"
OUT_CSV = ART / "phase0_plateau_baseline.csv"
OUT_MD = ART / "phase0_plateau_baseline.md"

V_REST_MV = -65.0  # from CompartmentalParams default
PLATEAU_SETTLE_THRESHOLD_MV = 5.0  # within 5mV of rest = "returned"

# Gao & Hobert 2020 targets (AVA primary; others scaled)
TARGETS = {
    "AVAL": {"duration_ms": 600, "amplitude_mv": 20.0,
             "ref": "Gao & Hobert 2020 Fig 3"},
    "AVAR": {"duration_ms": 600, "amplitude_mv": 20.0,
             "ref": "Gao & Hobert 2020 Fig 3"},
    "AVEL": {"duration_ms": 400, "amplitude_mv": 15.0,
             "ref": "Wang 2020 (less sustained than AVA)"},
    "AVER": {"duration_ms": 400, "amplitude_mv": 15.0,
             "ref": "Wang 2020"},
    "AVBL": {"duration_ms": 500, "amplitude_mv": 18.0,
             "ref": "Kawano 2011 forward command"},
    "AVBR": {"duration_ms": 500, "amplitude_mv": 18.0,
             "ref": "Kawano 2011"},
    "PVCL": {"duration_ms": 350, "amplitude_mv": 14.0,
             "ref": "Faumont 2011"},
    "PVCR": {"duration_ms": 350, "amplitude_mv": 14.0,
             "ref": "Faumont 2011"},
    "RIS": {"duration_ms": 700, "amplitude_mv": 18.0,
            "ref": "Turek 2016 (long plateau supports quiescence)"},
    "DVA": {"duration_ms": 400, "amplitude_mv": 16.0,
            "ref": "Li 2006 stretch-gated TRP"},
    "AWCL": {"duration_ms": 0, "amplitude_mv": 0.0,
             "ref": "No plateau (non-plateau neuron)"},
    "AWCR": {"duration_ms": 0, "amplitude_mv": 0.0,
             "ref": "No plateau"},
    "RMGL": {"duration_ms": 0, "amplitude_mv": 0.0,
             "ref": "No plateau"},
    "RMGR": {"duration_ms": 0, "amplitude_mv": 0.0,
             "ref": "No plateau"},
    "ALA": {"duration_ms": 0, "amplitude_mv": 0.0,
            "ref": "No plateau (Van Buskirk 2007)"},
}


def run_injection(neuron_name: str, inject_pa: float = 50.0,
                  inject_ms: float = 100.0, post_ms: float = 900.0):
    """Run the current-injection protocol on one neuron.

    Protocol:
      0..settle_ms: I_ext=0, settle to rest
      settle_ms..settle_ms+inject_ms: I_ext=inject_pa on target neuron's soma
      settle_ms+inject_ms..end: I_ext=0, watch plateau relax

    Returns dict with measured metrics.
    """
    start_scope()
    defaultclock.dt = 0.1 * ms
    grp, names = build_compartmental_group()
    idx = names.index(neuron_name)
    mon = StateMonitor(grp, ["v_s", "v_d", "I_ca", "h"], record=[idx])
    net = Network(grp, mon)

    settle_ms = 200.0

    # Settle
    net.run(settle_ms * ms)
    v_s_baseline = float(mon.v_s[0, -1] / mV)
    v_d_baseline = float(mon.v_d[0, -1] / mV)

    # Inject
    grp.I_ext[idx] = inject_pa * pA
    net.run(inject_ms * ms)
    v_s_peak = float(np.max(mon.v_s[0] / mV))
    v_d_peak = float(np.max(mon.v_d[0] / mV))

    # Release
    grp.I_ext[idx] = 0 * pA
    net.run(post_ms * ms)

    # Analyse post-injection: find time (from release) when v_d returns
    # to within threshold of rest.
    v_d_trace = np.array(mon.v_d[0] / mV)
    t_trace_ms = np.array(mon.t / ms)
    release_ms = settle_ms + inject_ms
    # Indices after release
    post_mask = t_trace_ms >= release_ms
    t_post = t_trace_ms[post_mask] - release_ms
    v_d_post = v_d_trace[post_mask]
    # Plateau duration: first time post-release where v_d < v_rest + threshold
    target_v = V_REST_MV + PLATEAU_SETTLE_THRESHOLD_MV
    settled = v_d_post <= target_v
    if settled.all():
        plateau_duration_ms = 0.0
    elif settled.any():
        first_settle = int(np.argmax(settled))
        plateau_duration_ms = float(t_post[first_settle])
    else:
        # Never settles within the recording — saturated
        plateau_duration_ms = float(t_post[-1])  # lower bound

    v_d_at_release = float(v_d_post[0])
    v_d_500ms_post = (
        float(v_d_post[np.searchsorted(t_post, 500.0)])
        if t_post[-1] >= 500.0 else float('nan')
    )
    plateau_amplitude_mv = v_d_peak - v_d_baseline

    return {
        "neuron": neuron_name,
        "v_s_baseline_mv": round(v_s_baseline, 2),
        "v_d_baseline_mv": round(v_d_baseline, 2),
        "v_s_peak_mv": round(v_s_peak, 2),
        "v_d_peak_mv": round(v_d_peak, 2),
        "v_d_at_release_mv": round(v_d_at_release, 2),
        "v_d_500ms_post_mv": round(v_d_500ms_post, 2)
        if not np.isnan(v_d_500ms_post) else None,
        "plateau_amplitude_mv": round(plateau_amplitude_mv, 2),
        "plateau_duration_ms": round(plateau_duration_ms, 1),
        "has_plateau_cfg": bool(COMPARTMENTAL_ROSTER[neuron_name].has_plateau),
        "g_ca_ns": COMPARTMENTAL_ROSTER[neuron_name].g_ca_ns,
        "tau_h_ms": COMPARTMENTAL_ROSTER[neuron_name].plateau_tau_ms,
    }


def main():
    t0 = time.time()
    rows = []
    for name in COMPARTMENTAL_ROSTER:
        t_r = time.time()
        result = run_injection(name)
        dt = time.time() - t_r
        tgt = TARGETS.get(name, {"duration_ms": 0, "amplitude_mv": 0.0,
                                  "ref": "no reference"})
        result["target_duration_ms"] = tgt["duration_ms"]
        result["target_amplitude_mv"] = tgt["amplitude_mv"]
        result["reference"] = tgt["ref"]
        # Gap vs target (negative = shortfall)
        result["duration_gap_ms"] = round(
            result["plateau_duration_ms"] - tgt["duration_ms"], 1)
        result["amplitude_gap_mv"] = round(
            result["plateau_amplitude_mv"] - tgt["amplitude_mv"], 2)
        # Pass/fail at current state (within 20% of target = pass;
        # target=0 = non-plateau, pass if duration < 50ms)
        if tgt["duration_ms"] > 0:
            dur_pass = (
                0.8 * tgt["duration_ms"]
                <= result["plateau_duration_ms"]
                <= 1.2 * tgt["duration_ms"]
            )
            amp_pass = (
                0.8 * tgt["amplitude_mv"]
                <= result["plateau_amplitude_mv"]
                <= 1.2 * tgt["amplitude_mv"]
            )
        else:
            dur_pass = result["plateau_duration_ms"] < 50.0
            amp_pass = result["plateau_amplitude_mv"] < 3.0
        result["status"] = ("PASS" if (dur_pass and amp_pass)
                            else "FAIL")
        rows.append(result)
        print(f"  {name:6s} | v_d_peak {result['v_d_peak_mv']:+6.1f}mV | "
              f"amp {result['plateau_amplitude_mv']:+5.1f}mV | "
              f"dur {result['plateau_duration_ms']:7.1f}ms | "
              f"target {tgt['duration_ms']}/{tgt['amplitude_mv']} | "
              f"{result['status']} | {dt:.1f}s wall")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s ({len(rows)} neurons)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Write markdown summary
    lines = [
        "# Phase 0 — W0.4a — T4-2 plateau baseline",
        "",
        "Current-state measurement of the 15-neuron compartmental scaffold ",
        "(`compartmental_neurons.py`) plateau response to a 50 pA / 100 ms ",
        "somatic current injection. Compares to Gao & Hobert 2020 / Wang 2020 ",
        "targets for the plateau-expressing neurons; non-plateau neurons ",
        "(AWC, RMG, ALA) should show no sustained depolarisation.",
        "",
        "## Protocol",
        "",
        "`t=0..200ms`: settle; `t=200..300ms`: inject 50 pA on soma; ",
        "`t=300..1200ms`: record dendritic plateau dynamics.",
        "",
        "Plateau duration = time from injection-release until ",
        f"v_d within {PLATEAU_SETTLE_THRESHOLD_MV} mV of v_rest "
        f"({V_REST_MV} mV).",
        "",
        "## Per-neuron baseline",
        "",
        "| neuron | v_d peak (mV) | amp (mV) | dur (ms) | target dur | target amp | gap (dur) | gap (amp) | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['neuron']} | {r['v_d_peak_mv']:+.1f} | "
            f"{r['plateau_amplitude_mv']:+.1f} | "
            f"{r['plateau_duration_ms']:.1f} | "
            f"{r['target_duration_ms']} | {r['target_amplitude_mv']:.1f} | "
            f"{r['duration_gap_ms']:+.1f} | {r['amplitude_gap_mv']:+.1f} | "
            f"**{r['status']}** |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    fail_neurons = [r['neuron'] for r in rows if r['status'] == 'FAIL']
    pass_neurons = [r['neuron'] for r in rows if r['status'] == 'PASS']
    lines.append(f"- **{len(pass_neurons)}/{len(rows)} neurons** currently "
                 f"pass within ±20% of target values.")
    if fail_neurons:
        lines.append(
            f"- **{len(fail_neurons)} fail**: {', '.join(fail_neurons)}. "
            f"Expected — `compartmental_neurons.py` docstring marks plateau "
            f"dynamics as calibration-pending. Current parameters are "
            f"conservative defaults, not fits to voltage-clamp data."
        )
    lines.append("")
    lines.append("## T4-2 exit threshold (ratified against this baseline)")
    lines.append("")
    lines.append("- AVA plateau duration within 20% of 600 ms (Gao & Hobert 2020)")
    lines.append("- AVA plateau amplitude within 10% of 20 mV (target: 18-22 mV)")
    lines.append("- Non-plateau neurons (AWC, RMG, ALA) show no sustained "
                 "depolarisation (amp < 3 mV, duration < 50 ms).")
    lines.append("- All 15 neurons report status=PASS post-calibration.")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
