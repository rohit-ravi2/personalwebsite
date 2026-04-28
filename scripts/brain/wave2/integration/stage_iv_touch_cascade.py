"""
Stage IV touch cascade biological validation (overnight run).

Given Stage III WB3 PAUSED for biology review (LIF→Wave2 graded coupling
needs human judgment), this Stage IV runs in two reduced-scope modes:

1. **LIF baseline characterization.** Run touch_anterior scenario under
   pure LIFBrain with per-edge sign mode (current production baseline,
   per `claude-chat-context.md` §5 resolution). Record AVAL/AVAR firing
   rates pre-touch vs peri-touch. This is the reference Stage IV claim
   should be measured against.

2. **Wave 2 AVAL plateau characterization under direct injection.**
   Run isolated Wave 2 AVAL with current injection at touch-equivalent
   levels (~30 pA, matching ASH→AVA pathway-equivalent drive per
   Nicoletti 2024 protocol). Characterize plateau amplitude, duration,
   and whether the response would activate an FSM activity-mode
   classifier.

The goal: produce diagnostic data on whether Wave 2 cellular detail
adds value over LIF for the §5 cascade test, without requiring the
WB3 biological-judgment-paused integration.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

WAVE2_DIR = Path(__file__).resolve().parent.parent
BRAIN_DIR = WAVE2_DIR.parent
sys.path.insert(0, str(BRAIN_DIR))
sys.path.insert(0, str(WAVE2_DIR))

from brian2 import ms, mV, pA, defaultclock

from option_alpha_ava_cell import build_brian2_aval_4channel
from option_alpha_avar_cell import build_brian2_avar_5channel


# ---------------------------------------------------------------------------
# Component 1 — LIF baseline characterization
# ---------------------------------------------------------------------------

def run_lif_baseline_touch(use_per_edge=True, seed=42):
    """Run LIFBrain under touch_anterior — record AVAL/AVAR firing rates.

    Pre-touch baseline: 0-3 s spontaneous.
    Peri-touch: 3-5 s with 200 Hz Poisson on ALML/ALMR/AVM.
    Post-touch: 5-7 s spontaneous recovery.
    """
    from lif_brain import LIFBrain

    print(f"\n[LIF baseline, per_edge_signs={use_per_edge}]")
    brain = LIFBrain(use_per_edge_glu_signs=use_per_edge)
    brain._brian2_seed = seed
    print(f"  Loaded LIFBrain with {brain.summary['N']} neurons")

    print("  Phase 1: 3 s spontaneous baseline")
    brain.run(3000)
    base_rates = brain.firing_rates(2000)

    print("  Phase 2: 2 s touch_anterior stim (200 Hz Poisson on ALML, ALMR, AVM)")
    for n in ["ALML", "ALMR", "AVM"]:
        if n in brain.idx:
            brain.inject_poisson(n, 200, weight_mv=8)
    brain.run(2000)
    stim_rates = brain.firing_rates(2000)

    print("  Phase 3: 2 s recovery")
    brain.run(2000)
    recovery_rates = brain.firing_rates(2000)

    key_cells = ["ALML", "ALMR", "AVM", "PVCL", "PVCR",
                 "AVDL", "AVDR", "AVEL", "AVER", "AVAL", "AVAR",
                 "AVBL", "AVBR", "AIBL", "AIBR", "RIML", "RIMR",
                 "AIYL", "AIYR"]
    out = {}
    print(f"\n{'cell':<8}{'baseline':>10}{'touch':>10}{'recovery':>10}{'Δ_touch':>10}")
    for c in key_cells:
        if c in brain.idx:
            i = brain.idx[c]
            b = float(base_rates[i])
            s = float(stim_rates[i])
            r = float(recovery_rates[i])
            out[c] = {"baseline_Hz": b, "touch_Hz": s, "recovery_Hz": r}
            print(f"  {c:<6}{b:>10.2f}{s:>10.2f}{r:>10.2f}{s-b:>+10.2f}")
    return out


# ---------------------------------------------------------------------------
# Component 2 — Wave 2 AVAL plateau response under cascade-equivalent input
# ---------------------------------------------------------------------------

def characterize_wave2_aval_plateau():
    """Run Wave 2 AVAL under -30 to +30 pA injection (Nicoletti CC range).

    Touch-cascade-equivalent inputs in vivo: ASH/AVM glutamate via iGluR,
    ~30-50 pA peak under sustained noxious stim per Mellem 2008
    + Piggott 2011 pathway analysis.

    This characterizes:
      - Plateau onset/offset latency
      - Plateau amplitude across injection levels
      - Whether the response would cross any reasonable FSM-classifier
        threshold
    """
    print("\n[Wave 2 AVAL plateau characterization]")
    factory = build_brian2_aval_4channel(record_components=False)
    bundle = factory()
    bundle["disable_clamp"]()

    # 7 injection levels matching Nicoletti's CC sweep
    injections_pa = list(np.linspace(-30, 30, 7))
    print(f"  Injection levels: {[f'{x:+.0f}' for x in injections_pa]} pA")

    results = {}
    defaultclock.dt = 0.025 * ms
    for inj in injections_pa:
        # Re-build factory each sweep (clean state)
        factory = build_brian2_aval_4channel(record_components=False)
        bundle = factory()
        bundle["disable_clamp"]()

        # 500 ms baseline → 2000 ms injection → 500 ms recovery
        bundle["inject_pA"](0.0)
        bundle["network"].run(500 * ms)
        bundle["inject_pA"](inj)
        bundle["network"].run(2000 * ms)
        bundle["inject_pA"](0.0)
        bundle["network"].run(500 * ms)

        mon = bundle["monitor"]
        v = np.asarray(mon.v[0]) * 1e3  # mV
        t = np.asarray(mon.t) * 1e3      # ms

        # Features
        baseline_v = float(np.mean(v[(t >= 100) & (t < 500)]))
        peak_v = float(np.max(v[(t >= 500) & (t < 2500)]))
        plateau_v = float(np.mean(v[(t >= 2300) & (t < 2500)]))
        recovery_v = float(np.mean(v[(t >= 2700) & (t <= 3000)]))

        results[f"{inj:+.0f}"] = {
            "injection_pA": float(inj),
            "baseline_V_mV": baseline_v,
            "peak_V_mV": peak_v,
            "plateau_V_mV": plateau_v,
            "recovery_V_mV": recovery_v,
            "delta_peak_mV": peak_v - baseline_v,
            "delta_plateau_mV": plateau_v - baseline_v,
        }
        print(f"  inj={inj:+5.0f} pA: base={baseline_v:+5.1f}  peak={peak_v:+5.1f}  "
              f"plateau={plateau_v:+5.1f}  Δplat={plateau_v-baseline_v:+5.1f} mV")

    return results


# ---------------------------------------------------------------------------
# Component 3 — Wave 2 AVAR plateau (5-channel; published-grade)
# ---------------------------------------------------------------------------

def characterize_wave2_avar_plateau():
    """Same as Wave 2 AVAL but for AVAR (5-channel, includes UNC-103)."""
    print("\n[Wave 2 AVAR plateau characterization]")

    injections_pa = list(np.linspace(-30, 30, 7))
    results = {}
    defaultclock.dt = 0.025 * ms
    for inj in injections_pa:
        factory = build_brian2_avar_5channel(record_components=False)
        bundle = factory()
        bundle["disable_clamp"]()

        bundle["inject_pA"](0.0)
        bundle["network"].run(500 * ms)
        bundle["inject_pA"](inj)
        bundle["network"].run(2000 * ms)
        bundle["inject_pA"](0.0)
        bundle["network"].run(500 * ms)

        mon = bundle["monitor"]
        v = np.asarray(mon.v[0]) * 1e3
        t = np.asarray(mon.t) * 1e3

        baseline_v = float(np.mean(v[(t >= 100) & (t < 500)]))
        peak_v = float(np.max(v[(t >= 500) & (t < 2500)]))
        plateau_v = float(np.mean(v[(t >= 2300) & (t < 2500)]))
        recovery_v = float(np.mean(v[(t >= 2700) & (t <= 3000)]))

        results[f"{inj:+.0f}"] = {
            "injection_pA": float(inj),
            "baseline_V_mV": baseline_v,
            "peak_V_mV": peak_v,
            "plateau_V_mV": plateau_v,
            "recovery_V_mV": recovery_v,
            "delta_peak_mV": peak_v - baseline_v,
            "delta_plateau_mV": plateau_v - baseline_v,
        }
        print(f"  inj={inj:+5.0f} pA: base={baseline_v:+5.1f}  peak={peak_v:+5.1f}  "
              f"plateau={plateau_v:+5.1f}  Δplat={plateau_v-baseline_v:+5.1f} mV")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    findings = {
        "stage": "IV",
        "mode": "reduced-scope (cross-coupling paused at WB3)",
        "components": {},
    }

    print("=" * 70)
    print("Stage IV — Touch cascade biological validation")
    print("=" * 70)

    # Component 1 — LIF baseline (per-edge mode, current production default)
    try:
        lif_per_edge = run_lif_baseline_touch(use_per_edge=True)
        findings["components"]["lif_per_edge_baseline"] = lif_per_edge
    except Exception as e:
        print(f"\n[Component 1 ERROR] {type(e).__name__}: {e}")
        findings["components"]["lif_per_edge_baseline"] = {"error": str(e)}

    # Component 2 — Wave 2 AVAL plateau
    try:
        aval_plateau = characterize_wave2_aval_plateau()
        findings["components"]["wave2_aval_plateau"] = aval_plateau
    except Exception as e:
        print(f"\n[Component 2 ERROR] {type(e).__name__}: {e}")
        findings["components"]["wave2_aval_plateau"] = {"error": str(e)}

    # Component 3 — Wave 2 AVAR plateau
    try:
        avar_plateau = characterize_wave2_avar_plateau()
        findings["components"]["wave2_avar_plateau"] = avar_plateau
    except Exception as e:
        print(f"\n[Component 3 ERROR] {type(e).__name__}: {e}")
        findings["components"]["wave2_avar_plateau"] = {"error": str(e)}

    findings["elapsed_s"] = time.time() - t_start

    out_path = WAVE2_DIR / "artifacts" / "stage_IV_findings.json"
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {findings['elapsed_s']:.1f} s")

    return findings


if __name__ == "__main__":
    main()
