#!/usr/bin/env python3
"""Phase α Deliverable 6 — smoke tests for both harnesses.

Voltage-clamp harness smoke test:
  - Leak-only Brian2 cell vs leak analytic reference
  - IV should be linear; harness reports |rel_diff| ≤ 5% across 7 holds

Plateau harness smoke tests:
  - passing_scaffold_factory: should classify as overall pass, with
    architectural_signature='active_termination'
  - failing_scaffold_factory: should classify as overall fail (amp or dur),
    with architectural_signature='leak_dominated' or 'no_termination'

If either smoke test fails to behave as expected, treat as harness bug.

Usage:
    /home/rohit/venvs/wave2-neuron/bin/python smoke_tests.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voltage_clamp_harness import (  # noqa: E402
    voltage_clamp_compare,
    leak_brian2_factory,
    leak_analytic_reference,
)
from plateau_harness import (  # noqa: E402
    current_clamp_plateau,
    passing_scaffold_factory,
    failing_scaffold_factory,
)


def smoke_voltage_clamp() -> dict:
    """Leak-only cell, 7 holding potentials. Expect linear IV, near-zero rel_diff."""
    g_leak_nS = 1.0
    e_leak_mV = -70.0
    holds = [-100, -80, -60, -40, -20, 0, 20]
    factory = leak_brian2_factory(g_leak_nS=g_leak_nS, e_leak_mV=e_leak_mV)
    ref = leak_analytic_reference(g_leak_nS=g_leak_nS, e_leak_mV=e_leak_mV)
    result = voltage_clamp_compare(
        factory, ref, holds,
        duration_ms=200.0, dt_ms=0.025, settle_window_ms=20.0,
        tolerance=0.05,
    )
    print("[VC smoke] leak-only cell vs analytic")
    for s in result["per_step"]:
        print(f"  hold {s['hold_mV']:+6.1f} mV  brian2={s['brian2_I_pA']:+8.3f} pA  "
              f"ref={s['ref_I_pA']:+8.3f} pA  rel_diff={s['rel_diff']:.2e}")
    print(f"  PASS={result['pass']}  max_div={result['max_divergence']:.2e}")
    return result


def smoke_plateau_passing() -> dict:
    """Passing scaffold: should classify pass + active_termination."""
    factory = passing_scaffold_factory()
    result = current_clamp_plateau(
        factory,
        stim_amp_pA=30.0,
        stim_duration_ms=600.0,
        total_duration_ms=2000.0,
        dt_ms=0.025,
        targets={
            "amplitude_mV": (15.0, 25.0),
            "duration_ms": (400.0, 800.0),
            "baseline_settle_mV": 5.0,
        },
        release_test_at_ms=300.0,
    )
    m = result["measured"]
    rd = result["release_dynamics"]
    print("[CC smoke] passing scaffold")
    print(f"  amplitude={m['amplitude_mV']:+.2f} mV  duration={m['duration_ms']:.1f} ms  "
          f"settle_offset={m['settle_offset_mV']:.2f} mV")
    print(f"  pass_amp={m['pass_amp']}  pass_dur={m['pass_dur']}  "
          f"pass_settle={m['pass_settle']}")
    print(f"  release_dyn: tau_release={rd.get('tau_release_ms', 'nan')!r:>8} ms  "
          f"tau_m={rd.get('tau_m_ms', 'nan')!r} ms  "
          f"ratio={rd.get('ratio', 'nan')!r:>8}  "
          f"signature={rd.get('architectural_signature')}")
    print(f"  overall PASS={result['pass']}")
    return result


def smoke_plateau_failing() -> dict:
    """Leak-only scaffold: should classify fail + leak_dominated."""
    factory = failing_scaffold_factory()
    result = current_clamp_plateau(
        factory,
        stim_amp_pA=30.0,
        stim_duration_ms=600.0,
        total_duration_ms=2000.0,
        dt_ms=0.025,
        targets={
            "amplitude_mV": (15.0, 25.0),
            "duration_ms": (400.0, 800.0),
            "baseline_settle_mV": 5.0,
        },
        release_test_at_ms=300.0,
    )
    m = result["measured"]
    rd = result["release_dynamics"]
    print("[CC smoke] failing scaffold (leak-only)")
    print(f"  amplitude={m['amplitude_mV']:+.2f} mV  duration={m['duration_ms']:.1f} ms  "
          f"settle_offset={m['settle_offset_mV']:.2f} mV")
    print(f"  pass_amp={m['pass_amp']}  pass_dur={m['pass_dur']}  "
          f"pass_settle={m['pass_settle']}")
    print(f"  release_dyn: tau_release={rd.get('tau_release_ms', 'nan')!r:>8} ms  "
          f"tau_m={rd.get('tau_m_ms', 'nan')!r} ms  "
          f"ratio={rd.get('ratio', 'nan')!r:>8}  "
          f"signature={rd.get('architectural_signature')}")
    print(f"  overall PASS={result['pass']}  (expected fail)")
    return result


def main() -> int:
    print("=== Phase α D6 — Harness smoke tests ===\n")
    print("--- Voltage-clamp harness ---")
    vc = smoke_voltage_clamp()
    print()
    print("--- Plateau harness — passing scaffold ---")
    cc_pass = smoke_plateau_passing()
    print()
    print("--- Plateau harness — failing scaffold ---")
    cc_fail = smoke_plateau_failing()
    print()

    # Decision: smoke suite passes if
    # 1. VC harness pass=True
    # 2. Passing-scaffold result.pass = True
    # 3. Failing-scaffold result.pass = False (correctly classified as fail)
    vc_ok = vc["pass"]
    pass_ok = cc_pass["pass"]
    fail_ok = (not cc_fail["pass"])
    # Also check architectural-signature label distinguishes them
    pass_sig = cc_pass["release_dynamics"].get("architectural_signature")
    fail_sig = cc_fail["release_dynamics"].get("architectural_signature")
    sig_ok = (pass_sig == "active_termination") and (
        fail_sig in ("leak_dominated", "no_termination")
    )

    print("=== SMOKE TEST SUMMARY ===")
    print(f"  voltage-clamp leak match: {vc_ok}")
    print(f"  plateau pass-scaffold pass-as-pass: {pass_ok}")
    print(f"  plateau fail-scaffold pass-as-fail: {fail_ok}")
    print(f"  release-signature distinguishes: {sig_ok}  "
          f"(pass={pass_sig!r}, fail={fail_sig!r})")
    overall = bool(vc_ok and pass_ok and fail_ok and sig_ok)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
