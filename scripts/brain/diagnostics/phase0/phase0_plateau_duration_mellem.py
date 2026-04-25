#!/usr/bin/env python3
"""Phase 0 — follow-up to plateau_diagnostic: measure plateau DURATION
under Mellem 2008 v_rest.

Probe 5 of phase0_plateau_diagnostic showed that switching v_rest from
−65 mV to −25 mV is sufficient to activate the plateau under 50 pA
somatic injection. That run only tracked 100 ms post-release. This
script extends the post-release monitoring to 1500 ms and records the
full v_d relaxation curve, so we can measure the actual plateau
duration and compare to Mellem 2008's 400–800 ms target.

Also runs the same protocol on all 8 plateau-expressing compartmental
neurons at Mellem rest (−25 mV), so we can see which pass/fail at 15/15.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brian2 import (
    start_scope, StateMonitor, Network, defaultclock, ms, mV, pA, nS,
)
from compartmental_neurons import (
    build_compartmental_group, COMPARTMENTAL_ROSTER, CompartmentalParams,
)

ART = Path(__file__).resolve().parent.parent.parent / "artifacts"
OUT_JSON = ART / "phase0_plateau_duration_mellem.json"
OUT_MD = ART / "phase0_plateau_duration_mellem.md"

V_REST_MELLEM_MV = -25.0
PLATEAU_SETTLE_THRESHOLD_MV = 5.0


def run_extended_injection(neuron_name: str, v_rest_mv: float,
                           inject_pa: float = 50.0, inject_ms: float = 100.0,
                           post_ms: float = 1500.0) -> dict:
    start_scope()
    defaultclock.dt = 0.1 * ms

    # Override the roster entry for this neuron's v_rest
    original = COMPARTMENTAL_ROSTER[neuron_name]
    override = CompartmentalParams(
        soma_tau_ms=original.soma_tau_ms,
        dend_tau_ms=original.dend_tau_ms,
        g_axial_ns=original.g_axial_ns,
        e_rest_mv=v_rest_mv,
        has_plateau=original.has_plateau,
        g_ca_ns=original.g_ca_ns,
        e_ca_mv=original.e_ca_mv,
        v_ca_half_mv=original.v_ca_half_mv,
        plateau_tau_ms=original.plateau_tau_ms,
        notes=original.notes,
    )
    COMPARTMENTAL_ROSTER[neuron_name] = override
    try:
        grp, names = build_compartmental_group()
        idx = names.index(neuron_name)
        grp.v_s = v_rest_mv * mV
        grp.v_d = v_rest_mv * mV
        grp.h = 1.0

        mon = StateMonitor(grp, ["v_d", "I_ca", "h"], record=[idx])
        net = Network(grp, mon)
        net.run(100 * ms)  # settle
        grp.I_ext[idx] = inject_pa * pA
        net.run(inject_ms * ms)
        grp.I_ext[idx] = 0 * pA
        net.run(post_ms * ms)
    finally:
        COMPARTMENTAL_ROSTER[neuron_name] = original

    t = np.array(mon.t / ms)
    v_d = np.array(mon.v_d[0] / mV)
    i_ca = np.array(mon.I_ca[0] / pA)
    h = np.array(mon.h[0])

    inject_end_t = 100.0 + inject_ms
    during_mask = (t >= 100.0) & (t < inject_end_t)
    post_mask = t >= inject_end_t
    v_d_post = v_d[post_mask]
    t_post = t[post_mask] - inject_end_t

    # Plateau duration: time post-release where v_d returns to within
    # threshold of rest.
    target_v = v_rest_mv + PLATEAU_SETTLE_THRESHOLD_MV
    settled = v_d_post <= target_v
    if settled.all():
        plateau_duration_ms = 0.0
    elif settled.any():
        plateau_duration_ms = float(t_post[int(np.argmax(settled))])
    else:
        plateau_duration_ms = float(t_post[-1])  # lower bound

    v_d_peak_during = float(np.max(v_d[during_mask]))
    amplitude = v_d_peak_during - v_rest_mv

    return {
        "neuron": neuron_name,
        "v_rest_mv": v_rest_mv,
        "v_d_peak_mv": round(v_d_peak_during, 2),
        "plateau_amplitude_mv": round(amplitude, 2),
        "plateau_duration_ms": round(plateau_duration_ms, 1),
        "v_d_at_200ms_post": (
            round(float(v_d_post[int(200.0 / 0.1)]), 2)
            if len(v_d_post) > int(200.0 / 0.1) else None
        ),
        "v_d_at_500ms_post": (
            round(float(v_d_post[int(500.0 / 0.1)]), 2)
            if len(v_d_post) > int(500.0 / 0.1) else None
        ),
        "v_d_at_1000ms_post": (
            round(float(v_d_post[int(1000.0 / 0.1)]), 2)
            if len(v_d_post) > int(1000.0 / 0.1) else None
        ),
        "h_at_inject_end": round(float(h[during_mask][-1]), 3),
        "h_at_1000ms_post": (
            round(float(h[post_mask][int(1000.0 / 0.1)]), 3)
            if len(h[post_mask]) > int(1000.0 / 0.1) else None
        ),
    }


# Plateau-expressing neurons + Mellem-style targets
TARGETS = {
    "AVAL": (600, 20.0),
    "AVAR": (600, 20.0),
    "AVEL": (400, 15.0),
    "AVER": (400, 15.0),
    "AVBL": (500, 18.0),
    "AVBR": (500, 18.0),
    "PVCL": (350, 14.0),
    "PVCR": (350, 14.0),
    "RIS":  (700, 18.0),
    "DVA":  (400, 16.0),
}


def main():
    print(f"Plateau duration at Mellem 2008 v_rest = {V_REST_MELLEM_MV} mV")
    print()
    results = []
    for name in TARGETS.keys():
        r = run_extended_injection(name, V_REST_MELLEM_MV)
        tgt_ms, tgt_mv = TARGETS[name]
        dur_pass = 0.8 * tgt_ms <= r["plateau_duration_ms"] <= 1.2 * tgt_ms
        amp_pass = 0.8 * tgt_mv <= r["plateau_amplitude_mv"] <= 1.2 * tgt_mv
        r["target_duration_ms"] = tgt_ms
        r["target_amplitude_mv"] = tgt_mv
        r["status"] = "PASS" if (dur_pass and amp_pass) else "FAIL"
        results.append(r)
        print(f"  {name:6s} | v_d_peak {r['v_d_peak_mv']:+6.1f} mV | "
              f"amp {r['plateau_amplitude_mv']:+5.1f} mV "
              f"(target {tgt_mv}) | "
              f"dur {r['plateau_duration_ms']:6.1f} ms (target {tgt_ms}) "
              f"| v_d@500ms {r['v_d_at_500ms_post']} | {r['status']}")

    # Also the 5 non-plateau neurons: stay not-plateau-ing at Mellem rest
    non_plateau_targets = ["AWCL", "AWCR", "RMGL", "RMGR", "ALA"]
    print("\nNon-plateau sanity (should NOT show sustained depolarisation):")
    for name in non_plateau_targets:
        r = run_extended_injection(name, V_REST_MELLEM_MV)
        print(f"  {name:6s} | v_d_peak {r['v_d_peak_mv']:+6.1f} mV | "
              f"amp {r['plateau_amplitude_mv']:+5.1f} mV | "
              f"dur {r['plateau_duration_ms']:6.1f} ms | "
              f"v_d@500ms {r['v_d_at_500ms_post']}")
        results.append({**r, "target_duration_ms": 0, "target_amplitude_mv": 0,
                        "status": "non_plateau"})

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_total_plateau = sum(1 for r in results
                          if r["status"] in ("PASS", "FAIL"))
    print(f"\nPlateau neurons: {n_pass}/{n_total_plateau} pass at "
          f"v_rest = {V_REST_MELLEM_MV} mV "
          f"(was 2/15 at v_rest = -65 mV)")

    OUT_JSON.write_text(json.dumps({
        "v_rest_mv": V_REST_MELLEM_MV,
        "results": results,
        "n_plateau_pass": n_pass,
        "n_plateau_total": n_total_plateau,
    }, indent=2))
    print(f"Wrote {OUT_JSON}")

    # Markdown
    lines = [
        "# Plateau duration at Mellem 2008 v_rest",
        "",
        "Follow-up to `phase0_plateau_diagnostic.py` Probe 5.",
        "",
        f"**v_rest set to {V_REST_MELLEM_MV} mV** (Mellem 2008 AVA range: "
        "−20 to −30 mV). All other compartmental roster parameters unchanged.",
        "",
        "## Plateau neurons",
        "",
        "| neuron | v_d peak | amplitude | duration | target amp | target dur | @500 ms | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("status") == "non_plateau":
            continue
        lines.append(f"| {r['neuron']} | {r['v_d_peak_mv']:+.1f} | "
                     f"{r['plateau_amplitude_mv']:+.1f} | "
                     f"{r['plateau_duration_ms']:.0f} | "
                     f"{r['target_amplitude_mv']} | "
                     f"{r['target_duration_ms']} | "
                     f"{r['v_d_at_500ms_post']} | **{r['status']}** |")
    lines += [
        "",
        "## Non-plateau neurons (sanity check)",
        "",
        "| neuron | v_d peak | amplitude | duration | @500 ms |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("status") != "non_plateau":
            continue
        lines.append(f"| {r['neuron']} | {r['v_d_peak_mv']:+.1f} | "
                     f"{r['plateau_amplitude_mv']:+.1f} | "
                     f"{r['plateau_duration_ms']:.0f} | "
                     f"{r['v_d_at_500ms_post']} |")
    lines += [
        "",
        f"## Summary: **{n_pass}/{n_total_plateau} plateau neurons pass** "
        f"at Mellem v_rest (was 2/15 at scaffold default −65 mV).",
        "",
        "Implication: T4-2's primary calibration knob is v_rest, not "
        "g_ca / tau_h. A single-parameter change (v_rest: −65 → −25 mV) "
        "may resolve most of the plateau gap. Secondary fine-tuning on "
        "tau_h and plateau-duration targets per neuron is still needed "
        "(most won't hit exact Mellem durations at default tau_h = 350 ms).",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
