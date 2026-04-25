#!/usr/bin/env python3
"""Phase 0 — W0.4b — T2-#4 sensory cascade baseline.

For each of the 5 transduction cascades in `sensory_transduction.py`,
run a canonical stimulus protocol and record the rate trace. This
establishes the **current, uncalibrated** response shape per cascade.
Later T2-#4 calibration fits parameters against digitized ΔF/F data
from the reference papers; Phase 0 produces the "before" traces and
measures initial Frechet distance to targets once references are in.

Canonical protocols (matched to the reference figures):

  ASE  — 200 mM NaCl step: baseline 0 → 1 at t=2s, hold 10s
         (Thiele 2009 Fig 2)
  AWC  — odorant pulse: 0 at t=0..3s, 1 at 3..6s, 0 at 6..15s;
         we expect firing on the offset (t=6s)
         (Chalasani 2007 Fig 1)
  ASH  — osmotic shock: 0 at t=0..2s, 1 at 2..4s, 0 after
         (Hilliard 2005 Fig 3)
  AFD  — warming: T=20°C for 0..5s, ramp to 25°C over 5..10s, hold
         (Clark 2006 Fig 2 analog)
  ALM  — mechanical touch: impulse at t=2s, 100ms duration
         (O'Hagan 2005 Fig 2)

Traces sampled at 20 Hz (50 ms dt_s), matching the BRAIN_SYNC_MS
used by ClosedLoopEnv. Output: rate_hz(t) per cascade.

Output: artifacts/phase0_cascade_baseline.npz + .md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sensory_transduction import (
    ASESaltCascade, AWCOlfactoryCascade, ASHPolymodalCascade,
    AFDThermalCascade, ALMTouchCascade,
)

ART = Path(__file__).resolve().parent.parent.parent / "artifacts"
OUT_NPZ = ART / "phase0_cascade_baseline.npz"
OUT_MD = ART / "phase0_cascade_baseline.md"

DT_S = 0.05
DURATION_S = 15.0
N_STEPS = int(DURATION_S / DT_S)


def run_ase():
    """NaCl step: 0 → 1 at t=2s, hold until end."""
    cascade = ASESaltCascade()
    rates = np.zeros(N_STEPS, dtype=np.float32)
    stim_trace = np.zeros(N_STEPS, dtype=np.float32)
    for i in range(N_STEPS):
        t = i * DT_S
        stim = 1.0 if t >= 2.0 else 0.0
        stim_trace[i] = stim
        rates[i] = cascade.sense(stim, DT_S)
    return {"name": "ASE_salt", "rates_hz": rates, "stim": stim_trace}


def run_awc():
    """Odorant pulse: 1 during 3-6s, OFF-cell fires on offset."""
    cascade = AWCOlfactoryCascade()
    rates = np.zeros(N_STEPS, dtype=np.float32)
    stim_trace = np.zeros(N_STEPS, dtype=np.float32)
    for i in range(N_STEPS):
        t = i * DT_S
        stim = 1.0 if (3.0 <= t < 6.0) else 0.0
        stim_trace[i] = stim
        rates[i] = cascade.sense(stim, DT_S)
    return {"name": "AWC_olfactory", "rates_hz": rates, "stim": stim_trace}


def run_ash():
    """Osmotic shock: 1 during 2-4s."""
    cascade = ASHPolymodalCascade()
    rates = np.zeros(N_STEPS, dtype=np.float32)
    stim_trace = np.zeros(N_STEPS, dtype=np.float32)
    for i in range(N_STEPS):
        t = i * DT_S
        stim = 1.0 if (2.0 <= t < 4.0) else 0.0
        stim_trace[i] = stim
        rates[i] = cascade.sense(stim, DT_S)
    return {"name": "ASH_polymodal", "rates_hz": rates, "stim": stim_trace}


def run_afd():
    """Warming: 20°C baseline, ramp to 25°C during 5-10s."""
    cascade = AFDThermalCascade(initial_tc_c=20.0)
    rates = np.zeros(N_STEPS, dtype=np.float32)
    stim_trace = np.zeros(N_STEPS, dtype=np.float32)
    for i in range(N_STEPS):
        t = i * DT_S
        if t < 5.0:
            temp = 20.0
        elif t < 10.0:
            temp = 20.0 + 5.0 * (t - 5.0) / 5.0
        else:
            temp = 25.0
        stim_trace[i] = temp
        rates[i] = cascade.sense(temp, DT_S)
    return {"name": "AFD_thermal", "rates_hz": rates, "stim": stim_trace}


def run_alm():
    """Touch impulse at t=2s for 100ms."""
    cascade = ALMTouchCascade(posterior=False)
    rates = np.zeros(N_STEPS, dtype=np.float32)
    stim_trace = np.zeros(N_STEPS, dtype=np.float32)
    for i in range(N_STEPS):
        t = i * DT_S
        stim = 1.0 if (2.0 <= t < 2.1) else 0.0
        stim_trace[i] = stim
        rates[i] = cascade.sense(stim, DT_S)
    return {"name": "ALM_touch", "rates_hz": rates, "stim": stim_trace}


def characterize(trace: dict) -> dict:
    """Measure basic shape characteristics of a rate trace."""
    rates = trace["rates_hz"]
    stim = trace["stim"]
    t = np.arange(len(rates)) * DT_S
    peak = float(np.max(rates))
    peak_t = float(t[np.argmax(rates)])
    # Time to reach 50% of peak (rise time from first non-zero stim)
    stim_on = np.argmax(stim > 0) if (stim > 0).any() else 0
    stim_on_t = t[stim_on]
    if peak > 0:
        half_peak = peak * 0.5
        rising = rates >= half_peak
        if rising.any():
            rise_idx = np.argmax(rising)
            rise_t = t[rise_idx] - stim_on_t
        else:
            rise_t = np.nan
        # Decay: time from peak to 10% of peak
        decay_threshold = peak * 0.1
        post_peak = np.argmax(rates)
        if post_peak < len(rates) - 1:
            post_rates = rates[post_peak:]
            post_t = t[post_peak:]
            below = post_rates <= decay_threshold
            if below.any():
                decay_t = post_t[np.argmax(below)] - peak_t
            else:
                decay_t = np.nan
        else:
            decay_t = np.nan
    else:
        rise_t = np.nan
        decay_t = np.nan
    # Final steady state
    final = float(np.mean(rates[-20:]))  # last 1s
    return {
        "peak_hz": round(peak, 2),
        "peak_t_s": round(peak_t, 3),
        "rise_t_s": round(float(rise_t), 3) if not np.isnan(rise_t) else None,
        "decay_t_s": round(float(decay_t), 3) if not np.isnan(decay_t) else None,
        "final_hz": round(final, 2),
    }


def main():
    t0 = time.time()
    traces = {
        "ASE": run_ase(),
        "AWC": run_awc(),
        "ASH": run_ash(),
        "AFD": run_afd(),
        "ALM": run_alm(),
    }
    # Package for npz output
    save_dict = {}
    for key, tr in traces.items():
        save_dict[f"{key}_rates"] = tr["rates_hz"]
        save_dict[f"{key}_stim"] = tr["stim"]
    save_dict["t_s"] = np.arange(N_STEPS, dtype=np.float32) * DT_S
    save_dict["dt_s"] = np.float32(DT_S)
    np.savez_compressed(OUT_NPZ, **save_dict)

    # Characterise
    chars = {k: characterize(v) for k, v in traces.items()}

    total = time.time() - t0
    print(f"Total: {total:.2f}s ({N_STEPS} timesteps × 5 cascades)")
    print(f"Wrote {OUT_NPZ}")

    lines = [
        "# Phase 0 — W0.4b — T2-#4 sensory cascade baseline",
        "",
        "Current-state shape characterisation of the 5 transduction cascades ",
        "in `sensory_transduction.py`. Runs each cascade standalone with its ",
        "canonical stimulus protocol (Thiele 2009 / Chalasani 2007 / Hilliard ",
        "2005 / Clark 2006 / O'Hagan 2005 analogs) and records the rate trace.",
        "",
        "Used as the **pre-calibration reference**: T2-#4 will refit parameters ",
        "against digitised ΔF/F from the reference figures (pending `docs/",
        "references/` data). Frechet distance between these baseline traces and ",
        "the calibrated versions is the exit metric.",
        "",
        "## Trace characteristics",
        "",
        "| cascade | peak (Hz) | peak t (s) | rise τ (s) | decay τ (s) | final (Hz) |",
        "|---|---|---|---|---|---|",
    ]
    for key in ["ASE", "AWC", "ASH", "AFD", "ALM"]:
        c = chars[key]
        lines.append(
            f"| {key} | {c['peak_hz']} | {c['peak_t_s']} | "
            f"{c['rise_t_s'] if c['rise_t_s'] is not None else '—'} | "
            f"{c['decay_t_s'] if c['decay_t_s'] is not None else '—'} | "
            f"{c['final_hz']} |"
        )
    lines.append("")
    lines.append("## Canonical stimulus protocols")
    lines.append("")
    lines.append("- **ASE**: 0 → 1 salt step at t=2s, sustained.")
    lines.append("- **AWC**: odorant pulse 3-6s (expect firing on offset at t=6s).")
    lines.append("- **ASH**: aversive pulse 2-4s.")
    lines.append("- **AFD**: 20°C baseline, ramp to 25°C during 5-10s.")
    lines.append("- **ALM**: 100ms touch impulse at t=2s.")
    lines.append("")
    lines.append("## T2-#4 exit threshold (ratified)")
    lines.append("")
    lines.append("- Each cascade's simulated rate trace within ≤10% Frechet ")
    lines.append("  distance of the digitised published ΔF/F (after z-score ")
    lines.append("  normalisation to account for amplitude-unit mismatch).")
    lines.append("- No regression on touch / osmotic / salt / food scenarios ")
    lines.append("  with `sensory_mode=transduction` in the ensemble audit.")
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append("- Thiele T, Faumont S, Lockery SR (2009) J Neurosci — ASE salt.")
    lines.append("- Chalasani SH et al. (2007) Nature — AWC OFF-cell.")
    lines.append("- Hilliard MA et al. (2005) EMBO — ASH polymodal.")
    lines.append("- Clark DA et al. (2006) J Neurosci — AFD thermal.")
    lines.append("- O'Hagan R et al. (2005) Nat Neurosci — ALM MEC-4/10.")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")

    # Console summary
    print("\nTrace characteristics:")
    for key, c in chars.items():
        print(f"  {key}: peak {c['peak_hz']:5.1f} Hz at t={c['peak_t_s']:.2f}s, "
              f"rise τ={c['rise_t_s']}s, decay τ={c['decay_t_s']}s, "
              f"final {c['final_hz']:.1f} Hz")


if __name__ == "__main__":
    main()
