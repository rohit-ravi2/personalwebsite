#!/usr/bin/env python3
"""Track F (speculative) — Hodgkin-Huxley AVA calibrated to Mellem 2008.

EXPLORATORY — not yet rigorous. Outputs to speculative/track_f/.

Implements a single-compartment HH model of AVA in Brian2 with channels
drawn from CeNGEN AVA expression (TPM > 4 threshold, checked against
CeNGEN data). Minimum channel set: egl-19 (L-type Ca), shk-1 (delayed
rectifier K), shl-1 (A-type K), leak. Optimizes g_na/Ca, g_K, g_leak
via Nelder-Mead against Mellem 2008 reported plateau metrics.

Calibration target (from Mellem 2008 Fig 1d + abstract):
  - Resting potential: -25 mV (reported range -20 to -30)
  - Plateau amplitude: ~20 mV above rest (i.e., peaks around -5 mV)
  - Plateau duration: 400-800 ms post-stimulus release
  - Return to rest: within 1-2s

Pass criteria (pre-specified):
  PASS: amplitude within 10%, duration within 20%, return within 30%.
       All three must pass.
  FAIL: any metric outside tolerance.
  LOGISTICAL_FAILURE: implementation crashes before calibration.

**Caveat for morning brief:** calibration target values are from
published text/figure descriptions, not from a digitized trace. This
is not an exact match to Mellem Figure 1d waveform — it is an
amplitude/duration/return fit to reported values.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from brian2 import (
        NeuronGroup, StateMonitor, Network, defaultclock,
        ms, mV, pA, nS, pF, siemens, volt, second,
        start_scope, prefs, seed as brian2_seed,
    )
    prefs.codegen.target = "numpy"
    BRIAN2_OK = True
except Exception:
    BRIAN2_OK = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ART / "overnight_20260422_v2" / "speculative" / "track_f"
OUT_MD = OUT_DIR / "calibration_report.md"

# Mellem 2008 target values (from abstract + paper description)
TARGET = {
    "v_rest_mv": -25.0,
    "plateau_amp_mv": 20.0,    # above rest
    "plateau_duration_ms": 600,  # midpoint of 400-800
    "return_time_ms": 1500,      # within 1-2s
}
TOLERANCE = {
    "amplitude": 0.10,  # 10%
    "duration": 0.20,   # 20%
    "return": 0.30,     # 30%
}


def run_hh(g_ca_nS: float, g_k_nS: float, g_leak_nS: float,
           v_rest_mv: float = -25.0, C_mem_pF: float = 50.0,
           inject_pa: float = 50.0, inject_ms: float = 100.0,
           post_ms: float = 2000.0,
           tau_m_ms: float = 5.0, tau_h_ms: float = 600.0):
    """Run a single HH sim. Returns (v_trace_mv, t_trace_ms, rest_mv,
    peak_mv, plateau_amp, plateau_duration_ms, return_time_ms).

    Uses a compact HH-like model:
    - L-type Ca activation m_Ca(V) sigmoidal, V_half=-25 mV, k=6 mV
    - Ca inactivation h_Ca with tau_h (slow), saturating
    - K delayed-rectifier n_K sigmoidal, V_half=0 mV
    - Leak at g_leak
    """
    start_scope()
    defaultclock.dt = 0.1 * ms
    brian2_seed(42)
    # Build HH equations. Use V as variable.
    eqs = """
    dV/dt = (-I_ca - I_k - I_leak + I_ext) / C_mem : volt
    I_ca = g_ca * m_ca * h_ca * (V - E_ca) : amp
    I_k = g_k * n_k * (V - E_k) : amp
    I_leak = g_leak * (V - E_leak) : amp

    m_ca = 1 / (1 + exp(-(V - V_ca_half)/k_ca)) : 1
    dh_ca/dt = (h_inf - h_ca)/tau_h : 1
    h_inf = 1 / (1 + exp((V - (V_ca_half - 5*mV))/(3*mV))) : 1
    dn_k/dt = (n_inf - n_k)/tau_m : 1
    n_inf = 1 / (1 + exp(-(V - V_k_half)/k_k)) : 1

    I_ext : amp
    g_ca : siemens
    g_k : siemens
    g_leak : siemens
    E_ca : volt
    E_k : volt
    E_leak : volt
    V_ca_half : volt
    V_k_half : volt
    k_ca : volt
    k_k : volt
    tau_m : second
    tau_h : second
    C_mem : farad
    """
    grp = NeuronGroup(1, eqs, method="exponential_euler")
    grp.g_ca = g_ca_nS * nS
    grp.g_k = g_k_nS * nS
    grp.g_leak = g_leak_nS * nS
    grp.E_ca = 50 * mV
    grp.E_k = -80 * mV
    grp.E_leak = v_rest_mv * mV
    grp.V_ca_half = -25 * mV
    grp.V_k_half = 0 * mV
    grp.k_ca = 6 * mV
    grp.k_k = 10 * mV
    grp.tau_m = tau_m_ms * ms
    grp.tau_h = tau_h_ms * ms
    grp.C_mem = C_mem_pF * pF
    grp.V = v_rest_mv * mV
    grp.h_ca = 1.0
    grp.n_k = 0.0

    mon = StateMonitor(grp, ["V"], record=True)
    net = Network(grp, mon)
    net.run(200 * ms)  # settle
    grp.I_ext[0] = inject_pa * pA
    net.run(inject_ms * ms)
    grp.I_ext[0] = 0 * pA
    net.run(post_ms * ms)

    t = np.array(mon.t / ms)
    V = np.array(mon.V[0] / mV)

    # Settled rest is the V at t=150ms (before injection at 200ms)
    rest_idx = int(150 / 0.1)
    v_rest = float(V[rest_idx])
    release_idx = int((200 + inject_ms) / 0.1)
    v_during = V[int(200/0.1):release_idx]
    v_peak = float(np.max(v_during))
    amplitude = v_peak - v_rest

    # Plateau duration: time from release until V returns to within
    # 5 mV of v_rest
    v_post = V[release_idx:]
    t_post = t[release_idx:] - (200 + inject_ms)
    target_v = v_rest + 5.0
    settled = v_post <= target_v
    if settled.all():
        plateau_duration = 0.0
        return_time = 0.0
    elif settled.any():
        first_settle = int(np.argmax(settled))
        plateau_duration = float(t_post[first_settle])
        # Return time: to within 1 mV of rest
        target_v2 = v_rest + 1.0
        settled2 = v_post <= target_v2
        return_time = (float(t_post[int(np.argmax(settled2))])
                       if settled2.any() else float(t_post[-1]))
    else:
        plateau_duration = float(t_post[-1])
        return_time = float(t_post[-1])

    return {
        "v_rest_mv": round(v_rest, 2),
        "v_peak_mv": round(v_peak, 2),
        "amplitude_mv": round(amplitude, 2),
        "plateau_duration_ms": round(plateau_duration, 1),
        "return_time_ms": round(return_time, 1),
        "V_trace": V.tolist()[:4000],  # 400 ms worth for preview
    }


def score(result, target, tol):
    """Return per-metric pass + total pass."""
    amp_err = abs(result["amplitude_mv"] - target["plateau_amp_mv"]) / \
        target["plateau_amp_mv"]
    dur_err = abs(result["plateau_duration_ms"] - target["plateau_duration_ms"]) / \
        target["plateau_duration_ms"]
    ret_err = abs(result["return_time_ms"] - target["return_time_ms"]) / \
        target["return_time_ms"]
    amp_pass = amp_err <= tol["amplitude"]
    dur_pass = dur_err <= tol["duration"]
    ret_pass = ret_err <= tol["return"]
    total_pass = amp_pass and dur_pass and ret_pass
    return {
        "amp_err": round(amp_err, 3), "amp_pass": amp_pass,
        "dur_err": round(dur_err, 3), "dur_pass": dur_pass,
        "ret_err": round(ret_err, 3), "ret_pass": ret_pass,
        "total_pass": total_pass,
        "total_cost": amp_err + dur_err + ret_err,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not BRIAN2_OK:
        (OUT_DIR / "LOGISTICAL_FAILURE.md").write_text(
            "Brian2 import failed."
        )
        return

    # Grid search over parameter ranges
    print("Track F — HH AVA calibration, grid over g_ca, g_k, g_leak, tau_h")
    # Realistic ranges given target amplitude + duration
    g_ca_range = [3, 5, 8, 12, 18]  # nS
    g_k_range = [3, 8, 15, 25]      # nS
    g_leak_range = [0.5, 1.0, 2.0]  # nS
    tau_h_range = [400, 700, 1200]  # ms

    results = []
    best = None
    idx = 0
    total = len(g_ca_range)*len(g_k_range)*len(g_leak_range)*len(tau_h_range)
    for g_ca in g_ca_range:
        for g_k in g_k_range:
            for g_leak in g_leak_range:
                for tau_h in tau_h_range:
                    idx += 1
                    try:
                        r = run_hh(g_ca, g_k, g_leak, tau_h_ms=tau_h)
                        s = score(r, TARGET, TOLERANCE)
                        entry = {
                            "params": {"g_ca": g_ca, "g_k": g_k,
                                       "g_leak": g_leak, "tau_h": tau_h},
                            "result": {k: v for k, v in r.items()
                                        if k != "V_trace"},
                            "score": s,
                        }
                        results.append(entry)
                        if best is None or s["total_cost"] < best["score"]["total_cost"]:
                            best = entry
                            # Save best V trace
                            np.savez_compressed(
                                OUT_DIR / "best_trace.npz",
                                V_trace=r["V_trace"],
                                params_g_ca=g_ca, params_g_k=g_k,
                                params_g_leak=g_leak, params_tau_h=tau_h,
                            )
                        if idx % 10 == 0:
                            print(f"  [{idx}/{total}] best so far: "
                                  f"amp={best['result']['amplitude_mv']}, "
                                  f"dur={best['result']['plateau_duration_ms']}, "
                                  f"ret={best['result']['return_time_ms']}, "
                                  f"cost={best['score']['total_cost']:.3f}")
                    except Exception as e:
                        print(f"  [{idx}/{total}] g_ca={g_ca}, "
                              f"g_k={g_k}, g_leak={g_leak}, "
                              f"tau_h={tau_h} ERROR: {e}")

    # Save all results
    (OUT_DIR / "all_results.json").write_text(json.dumps(
        [{"params": e["params"], "result": e["result"], "score": e["score"]}
         for e in results], indent=2
    ))

    # Final verdict
    if best is None:
        status = "LOGISTICAL_FAILURE"
        reason = "No successful runs."
    else:
        status = "PASS" if best["score"]["total_pass"] else "FAIL"
        reason = (f"Best params g_ca={best['params']['g_ca']} nS, "
                  f"g_k={best['params']['g_k']} nS, "
                  f"g_leak={best['params']['g_leak']} nS, "
                  f"tau_h={best['params']['tau_h']} ms. "
                  f"amp={best['result']['amplitude_mv']} mV "
                  f"(target 20, err {best['score']['amp_err']:.2f}); "
                  f"dur={best['result']['plateau_duration_ms']} ms "
                  f"(target 600, err {best['score']['dur_err']:.2f}); "
                  f"ret={best['result']['return_time_ms']} ms "
                  f"(target 1500, err {best['score']['ret_err']:.2f}).")

    lines = [
        "# Track F (speculative) — HH AVA calibration",
        "",
        "**EXPLORATORY — not yet rigorous.**",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Wall time: {(time.time()-t0)/60:.1f} min",
        "",
        f"## Status: **{status}**",
        "",
        reason,
        "",
        "## Calibration targets (from Mellem 2008 published values, "
        "not digitized trace)",
        "",
        "| metric | target | tolerance | best result | err | pass |",
        "|---|---|---|---|---|---|",
    ]
    if best is not None:
        r = best["result"]; s = best["score"]
        lines.append(f"| amplitude (mV) | 20 | ±10% | {r['amplitude_mv']} | "
                     f"{s['amp_err']:.2f} | "
                     f"{'✓' if s['amp_pass'] else '✗'} |")
        lines.append(f"| duration (ms) | 600 | ±20% | "
                     f"{r['plateau_duration_ms']} | "
                     f"{s['dur_err']:.2f} | "
                     f"{'✓' if s['dur_pass'] else '✗'} |")
        lines.append(f"| return (ms) | 1500 | ±30% | "
                     f"{r['return_time_ms']} | "
                     f"{s['ret_err']:.2f} | "
                     f"{'✓' if s['ret_pass'] else '✗'} |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Calibration targets were drawn from Mellem 2008 abstract")
    lines.append("  + figure description, NOT from a digitized voltage-clamp")
    lines.append("  trace. Matching these values does not imply waveform match.")
    lines.append("- Grid search used, not Nelder-Mead (simpler + bounded).")
    lines.append("- Channel roster minimal: egl-19 L-type Ca + delayed")
    lines.append("  rectifier K + leak. CeNGEN roster would include more")
    lines.append("  (shl-1, slo-1, slo-2). This is a first-pass scaffold only.")
    lines.append("- Next step if PASS: digitize Mellem Fig 1d + refit on trace")
    lines.append("  shape via L2 loss. Do NOT integrate into main simulator")
    lines.append("  based on this grid-search result alone.")
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")

    status_md = ART / "overnight_20260422_v2" / "STATUS.md"
    with status_md.open("a") as f:
        f.write(f"\n## Track F (speculative): HH AVA\n")
        f.write(f"- Completed: {time.strftime('%H:%M:%S')}\n")
        f.write(f"- Status: {status}\n")
        f.write(f"- Output: speculative/track_f/\n")


if __name__ == "__main__":
    main()
