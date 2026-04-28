"""CP2 — codegen switch + smoke test using actual wave2 cell factory.

Verifies that:
1. Setting prefs.codegen.target = 'cython' AFTER the wave2 factory's internal
   `prefs.codegen.target = "cython"` line still routes to cython at run time.
   (Brian2 reads the pref at code-object creation, which happens at network.run.)
2. Numerical results match between numpy and cython codegen.
3. Speedup is observable on the actual production AVAL cell (4 channels).

Strategy: import option_alpha_ava_cell, build the factory, then override
prefs.codegen.target before calling factory() — that override should win
because the prefs object is module-level singleton.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
import time
from pathlib import Path

WAVE2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAVE2_DIR))


def run_aval_smoke(target: str, run_ms: float = 500.0) -> dict:
    """Build AVAL cell, run for run_ms with no clamp/inject. Time it."""
    from brian2 import ms, defaultclock, prefs
    from option_alpha_ava_cell import build_brian2_aval_4channel

    factory = build_brian2_aval_4channel(record_components=False)

    # IMPORTANT: factory() internally sets prefs.codegen.target = "cython".
    # Override AFTER the factory returns and BEFORE network.run().
    bundle = factory()
    prefs.codegen.target = target
    print(f"  prefs.codegen.target after override = {prefs.codegen.target!r}")

    defaultclock.dt = 0.025 * ms
    bundle["disable_clamp"]()
    bundle["inject_pA"](0.0)

    t0 = time.time()
    bundle["network"].run(run_ms * ms)
    elapsed = time.time() - t0

    mon = bundle["monitor"]
    final_v = float(mon.v[0][-1] / (1e-3))  # mV
    return {
        "target": target,
        "run_ms": run_ms,
        "elapsed_s": elapsed,
        "final_v_mV": final_v,
    }


def main():
    print("=" * 70)
    print("CP2 — codegen switch + AVAL passive-cell smoke test")
    print("=" * 70)

    results = []

    run_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 500.0

    print(f"\n[numpy run, {run_ms:.0f} ms]")
    r_numpy = run_aval_smoke("numpy", run_ms=run_ms)
    print(f"  elapsed: {r_numpy['elapsed_s']:.3f} s   final v: {r_numpy['final_v_mV']:.4f} mV")
    results.append(r_numpy)

    print(f"\n[cython run, {run_ms:.0f} ms]")
    r_cython = run_aval_smoke("cython", run_ms=run_ms)
    print(f"  elapsed: {r_cython['elapsed_s']:.3f} s   final v: {r_cython['final_v_mV']:.4f} mV")
    results.append(r_cython)

    # Speedup
    speedup = r_numpy["elapsed_s"] / max(r_cython["elapsed_s"], 1e-6)
    v_diff = abs(r_numpy["final_v_mV"] - r_cython["final_v_mV"])
    print()
    print(f"Speedup (numpy / cython): {speedup:.2f}x")
    print(f"final_v difference: {v_diff:.6f} mV")

    out = Path(__file__).parent / "cp2_smoke_result.json"
    out.write_text(json.dumps({
        "results": results,
        "speedup_x": speedup,
        "final_v_diff_mV": v_diff,
    }, indent=2))
    print(f"\nSaved to {out}")

    return speedup, v_diff


if __name__ == "__main__":
    main()
