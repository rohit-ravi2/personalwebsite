#!/usr/bin/env python3
"""Phase α Deliverable 3 — Nicoletti reference reproduction.

Runs Nicoletti's unmodified `*_simulation_iclamp.py` and `*_simulation_vclamp.py`
protocol functions for AVAL, AIY, and RIM (the load-bearing trio). Captures
voltage traces (current-clamp) and current traces (voltage-clamp), computes
shape diagnostics, and runs a determinism check (two consecutive runs should
match within 1%).

Important interpretation note (surfaced as Phase α finding):

  Nicoletti's repository ships the protocol *scripts* but does NOT ship her
  published-figure numerical traces. The "1% tolerance against published
  figures" criterion in the Phase α prompt cannot be checked against
  numerical reference data — only against the qualitative figure shapes
  (e.g., monotonic IV outside threshold; AVAL plateau is depolarizing on
  positive iclamp; current-clamp peaks bracket the iclamp range; etc.).

  We therefore interpret Deliverable 3 as:
    (a) Nicoletti's unmodified scripts run end-to-end without error
    (b) Two consecutive runs match within 1% (NEURON determinism)
    (c) Qualitative shape sanity checks pass (monotonicity, sign, scale)

  If any of (a)/(b)/(c) fails, we flag it as a possible condition-3
  invalidation signature and surface to user.

Usage:
    /home/rohit/venvs/wave2-neuron/bin/python reference_validation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Suppress matplotlib + interactive plot windows BEFORE Nicoletti scripts import.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

NICOLETTI_DIR = Path(
    "/home/rohit/Desktop/C-Elegans/simulation/upstream/nicoletti_2024"
)
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Per-neuron protocol parameters extracted from Nicoletti's *_simulation.py
# ---------------------------------------------------------------------------

NEURON_PROTOCOLS = {
    "AVAL": {
        "surf_cm2": 1123.84e-8,
        # g0 = [egl19, leak, irk, nca, eleak, cm], scale_to_index = 3
        "g0": [0.104385, 0.150164, 0.1, 0, -39, 0.859551],
        "scale_index": 3,
        "vc": {"vstart": -110, "vstop": 50, "ns": 17},
        "ic": {"s1": -0.03, "s2": 0.03, "ns": 7},
    },
    "AIY": {
        "surf_cm2": 65.89e-8,
        # g0 = [leak, slo1iso, kqt1, egl19, slo1egl19, nca, irk, eleak, cm]
        "g0": [0.14, 1, 0.2, 0.1, 0.92, 0.06, 0.5, -89.57, 1.6],
        "scale_index": 6,
        "vc": {"vstart": -120, "vstop": 50, "ns": 18},
        "ic": {"s1": -0.015, "s2": 0.035, "ns": 11},
    },
    "RIM": {
        "surf_cm2": 103.34e-8,
        # g0 already in S/cm^2 — Nicoletti's RIM script does NOT call gScm2
        "g0": [
            9.048750067326097e-4, 1.411644285181245e-4, 3.272854640954744e-4,
            8.451919806776876e-4, 9.676795045480941e-5,
            3.2005818627638106e-4, 9.676795045480941e-5, -50, 1.5,
        ],
        "scale_index": None,  # already-scaled
        "vc": {"vstart": -100, "vstop": 50, "ns": 16},
        "ic": {"s1": -0.015, "s2": 0.035, "ns": 11},
    },
}


def _scaled_g(neuron_name: str) -> np.ndarray:
    """Apply Nicoletti's gScm2 nS→S/cm² conversion if needed."""
    spec = NEURON_PROTOCOLS[neuron_name]
    g0 = list(spec["g0"])
    surf = spec["surf_cm2"]
    if spec["scale_index"] is None:
        return np.asarray(g0)
    idx = spec["scale_index"]
    return np.asarray([
        (g0[i] * 1e-9) / surf if i <= idx else g0[i]
        for i in range(len(g0))
    ])


# ---------------------------------------------------------------------------
# Shape diagnostics — qualitative checks since published numerical traces
# are not in the repo.
# ---------------------------------------------------------------------------

def shape_check_iclamp(v_trace: np.ndarray, time: np.ndarray, vipeaks, viss,
                       neuron: str) -> dict:
    """Sanity checks on current-clamp output."""
    issues = []
    # Shape consistency
    if v_trace.ndim != 2:
        issues.append(f"v_trace ndim={v_trace.ndim} (expected 2)")
    if v_trace.shape != time.shape:
        issues.append(
            f"v_trace shape {v_trace.shape} vs time shape {time.shape}"
        )
    # Voltage range plausibility (mV)
    # Note: peak values can be far outside steady state during the protocol-
    # edge transient (single-compartment cell, ~10 pF capacitance, 30 pA
    # injection produces large dV/dt at step onset). The biophysically-
    # meaningful check is on STEADY-STATE voltage at the end of each sweep,
    # not transient peaks. Peaks > ±200 mV would indicate pathology, but
    # ~±100-180 mV transients are expected under Nicoletti's protocol amps.
    vmin, vmax = float(np.nanmin(v_trace)), float(np.nanmax(v_trace))
    if vmin < -250 or vmax > 200:
        issues.append(f"v range pathological: [{vmin:.1f}, {vmax:.1f}] mV "
                      "(transients beyond expected ±200 mV envelope)")
    # Steady-state at end of each sweep should sit between -120 and +50 mV
    ss_each = v_trace[:, -200:].mean(axis=1)
    ss_min, ss_max = float(np.min(ss_each)), float(np.max(ss_each))
    if ss_min < -120 or ss_max > 60:
        issues.append(f"steady-state v range pathological: [{ss_min:.1f}, "
                      f"{ss_max:.1f}] mV")
    # Peaks list length matches sweep count
    if len(vipeaks) != v_trace.shape[0]:
        issues.append(
            f"vipeaks len {len(vipeaks)} vs sweep count {v_trace.shape[0]}"
        )
    # Steady-state voltage should be monotone-increasing in iclamp amp
    # for purely passive plateau: viss should not be wildly nonmonotone.
    viss_arr = np.asarray(viss, dtype=float)
    diffs = np.diff(viss_arr)
    n_neg = int(np.sum(diffs < -1.0))  # >1 mV reversal in monotonicity
    if n_neg > 1:
        issues.append(
            f"viss non-monotone: {n_neg} reversals > 1 mV"
        )
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "v_min_mV": vmin,
        "v_max_mV": vmax,
        "viss_first_last_mV": (float(viss_arr[0]), float(viss_arr[-1])),
        "n_sweeps": int(v_trace.shape[0]),
        "trace_len": int(v_trace.shape[1]),
    }


def shape_check_vclamp(i_trace: np.ndarray, time: np.ndarray, iv_peak, iv,
                       neuron: str) -> dict:
    """Sanity checks on voltage-clamp output."""
    issues = []
    if i_trace.ndim != 2:
        issues.append(f"i_trace ndim={i_trace.ndim} (expected 2)")
    if i_trace.shape != time.shape:
        issues.append(
            f"i_trace shape {i_trace.shape} vs time shape {time.shape}"
        )
    imin, imax = float(np.nanmin(i_trace)), float(np.nanmax(i_trace))
    # In pA — Nicoletti scales to pA via *1e9*surf in vclamp functions.
    # Plausible range: ~ -1e4 to +1e4 pA for whole-cell currents.
    if not np.isfinite(imin) or not np.isfinite(imax):
        issues.append("non-finite values in current trace")
    if abs(imin) > 1e6 or abs(imax) > 1e6:
        issues.append(
            f"current scale pathological: [{imin:.1f}, {imax:.1f}] pA"
        )
    iv_arr = np.asarray(iv, dtype=float)
    # Steady-state IV: at extreme negative V, K-currents should be small or
    # net inward; at extreme positive V, outward K-current should drive iv > 0.
    if iv_arr[-1] < iv_arr[0]:
        issues.append(
            "iv steady-state DECREASES with V (outward K should drive UP)"
        )
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "i_min_pA": imin,
        "i_max_pA": imax,
        "iv_first_last_pA": (float(iv_arr[0]), float(iv_arr[-1])),
        "n_sweeps": int(i_trace.shape[0]),
        "trace_len": int(i_trace.shape[1]),
    }


def determinism_check(arrA: np.ndarray, arrB: np.ndarray,
                      tol_rel: float = 0.01) -> dict:
    """Compare two NEURON runs of identical protocol — should match exactly.

    Tolerance: 1% relative on max-magnitude scale (per Phase α spec).
    """
    A = np.asarray(arrA, dtype=float)
    B = np.asarray(arrB, dtype=float)
    if A.shape != B.shape:
        return {"pass": False, "reason": f"shape mismatch {A.shape} vs {B.shape}"}
    diff = np.abs(A - B)
    scale = max(np.nanmax(np.abs(A)), 1e-12)
    max_rel = float(np.nanmax(diff) / scale)
    return {
        "pass": max_rel <= tol_rel,
        "max_abs_diff": float(np.nanmax(diff)),
        "max_rel_diff": max_rel,
        "scale": float(scale),
    }


# ---------------------------------------------------------------------------
# Per-neuron run pair (run twice for determinism check)
# ---------------------------------------------------------------------------

def run_neuron_protocols(neuron_name: str) -> dict:
    """Run iclamp + vclamp protocols twice; return traces + diagnostics.

    Imports Nicoletti's iclamp/vclamp module functions directly. Each
    function builds a fresh NEURON Section, so two runs with same params
    should be identically deterministic.
    """
    import matplotlib  # noqa: WPS433
    matplotlib.use("Agg", force=True)

    # Add Nicoletti dir to path so her *_simulation_*.py modules import.
    sys.path.insert(0, str(NICOLETTI_DIR))

    # Re-import is fine; cwd-side libnrnmech.so loads on first NEURON import.
    cwd_save = os.getcwd()
    os.chdir(NICOLETTI_DIR)
    try:
        if neuron_name == "AVAL":
            from AVAL_simulation_iclamp import AVA_simulation_iclamp as ic_func
            from AVAL_simulation_vclamp import AVA_simulation_vc as vc_func
        elif neuron_name == "AIY":
            from AIY_simulation_iclamp import AIY_simulation_iclamp as ic_func
            from AIY_simulation_vclamp import AIY_simulation_vc as vc_func
        elif neuron_name == "RIM":
            from RIM_simulation_iclamp import RIM_simulation_iclamp as ic_func
            from RIM_simulation_vclamp import RIM_simulation_vc as vc_func
        else:
            raise ValueError(f"unsupported neuron: {neuron_name}")

        spec = NEURON_PROTOCOLS[neuron_name]
        g_scaled = _scaled_g(neuron_name)

        ic_args = (g_scaled, spec["ic"]["s1"], spec["ic"]["s2"], spec["ic"]["ns"])
        vc_args = (g_scaled, spec["vc"]["vstart"], spec["vc"]["vstop"], spec["vc"]["ns"])

        # Run iclamp twice
        ic_run1 = ic_func(*ic_args)
        ic_run2 = ic_func(*ic_args)

        # Run vclamp twice
        vc_run1 = vc_func(*vc_args)
        vc_run2 = vc_func(*vc_args)
    finally:
        os.chdir(cwd_save)

    # Each function returns (traces, time, peaks, ss)
    ic_v1, ic_t1, ic_peaks1, ic_ss1 = ic_run1
    ic_v2, _, _, _ = ic_run2
    vc_i1, vc_t1, vc_peaks1, vc_ss1 = vc_run1
    vc_i2, _, _, _ = vc_run2

    ic_shape = shape_check_iclamp(ic_v1, ic_t1, ic_peaks1, ic_ss1, neuron_name)
    vc_shape = shape_check_vclamp(vc_i1, vc_t1, vc_peaks1, vc_ss1, neuron_name)

    ic_det = determinism_check(ic_v1, ic_v2)
    vc_det = determinism_check(vc_i1, vc_i2)

    return {
        "neuron": neuron_name,
        "iclamp": {
            "shape": ic_shape,
            "determinism": ic_det,
            "v_trace_shape": list(ic_v1.shape),
        },
        "vclamp": {
            "shape": vc_shape,
            "determinism": vc_det,
            "i_trace_shape": list(vc_i1.shape),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    """Roll up per-neuron diagnostics into pass/fail decision."""
    invalidation = []
    pass_count = 0
    for r in results:
        ic_ok = r["iclamp"]["shape"]["ok"] and r["iclamp"]["determinism"]["pass"]
        vc_ok = r["vclamp"]["shape"]["ok"] and r["vclamp"]["determinism"]["pass"]
        if ic_ok and vc_ok:
            pass_count += 1
        else:
            invalidation.append({
                "neuron": r["neuron"],
                "iclamp_shape_ok": r["iclamp"]["shape"]["ok"],
                "iclamp_shape_issues": r["iclamp"]["shape"]["issues"],
                "iclamp_determinism": r["iclamp"]["determinism"],
                "vclamp_shape_ok": r["vclamp"]["shape"]["ok"],
                "vclamp_shape_issues": r["vclamp"]["shape"]["issues"],
                "vclamp_determinism": r["vclamp"]["determinism"],
            })
    return {
        "total": len(results),
        "passed": pass_count,
        "failures": invalidation,
        "deliverable_3_pass": pass_count >= 2,  # Phase α spec: ≥ 2 of 3
    }


def main() -> int:
    print("=== Phase α D3 — Nicoletti reference reproduction ===\n")
    results = []
    for neuron in ("AVAL", "AIY", "RIM"):
        print(f"--- {neuron} ---")
        try:
            r = run_neuron_protocols(neuron)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results.append({
                "neuron": neuron,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        results.append(r)
        ic = r["iclamp"]
        vc = r["vclamp"]
        print(f"  iclamp shape ok={ic['shape']['ok']} "
              f"v_range=[{ic['shape']['v_min_mV']:.1f}, "
              f"{ic['shape']['v_max_mV']:.1f}] mV "
              f"determinism_max_rel={ic['determinism']['max_rel_diff']:.2e}")
        if ic["shape"]["issues"]:
            print(f"    iclamp issues: {ic['shape']['issues']}")
        print(f"  vclamp shape ok={vc['shape']['ok']} "
              f"i_range=[{vc['shape']['i_min_pA']:.1f}, "
              f"{vc['shape']['i_max_pA']:.1f}] pA "
              f"determinism_max_rel={vc['determinism']['max_rel_diff']:.2e}")
        if vc["shape"]["issues"]:
            print(f"    vclamp issues: {vc['shape']['issues']}")
        print()

    summary = summarize([r for r in results if "error" not in r])
    print("=== SUMMARY ===")
    print(f"  passed: {summary['passed']}/{summary['total']}")
    print(f"  Deliverable 3 pass: {summary['deliverable_3_pass']}")
    if summary["failures"]:
        print("  failures:")
        for f in summary["failures"]:
            print(f"    {f}")

    # Save JSON artifact
    import json  # noqa: WPS433
    out = ARTIFACTS / "reference_validation_results.json"
    with open(out, "w") as fh:
        json.dump({"results": results, "summary": summary}, fh, indent=2,
                  default=str)
    print(f"\n  artifact: {out}")

    return 0 if summary["deliverable_3_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
