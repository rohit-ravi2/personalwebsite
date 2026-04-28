"""CP5 — RIM re-validation under cython codegen.

Strategy: monkey-patch `option_alpha_rim_cell.build_brian2_rim_7channel` so
the factory it returns wraps prefs.codegen.target = 'cython' on top of the
hardcoded numpy. Then invoke `run_option_b_rim.main()`.

Pre-migration baseline:
  - voltage-clamp: 11/11 holds, max div ≤ 0.0043
  - current-clamp: 11/11 sweeps with 0.000 mV residuals across 55,000 timepoints

Specific F18 concerns:
  - RIM has 3 USEION ca (cca1+unc2+egl19) — symmetric, so eca preserved at
    60 mV (NOT 127.59). Verify this still holds under cython.
  - UNC-2 has GLOBAL declarations (minf, hinf, mtau, htau, munc2, hunc2)
    that auto-resolve under Brian2 per-cell-by-default. Verify cython
    namespace handling preserves auto-resolution.

RIM is the cleanest baseline — meaningfully different residuals would
indicate a cython-specific issue.
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import time
from pathlib import Path

WAVE2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAVE2_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cython_wrapper import wrap_factory_with_codegen


def main(target: str = "cython"):
    print("=" * 70)
    print(f"CP5 — RIM re-validation under codegen='{target}'")
    print("=" * 70)
    print()

    import option_alpha_rim_cell as rim_mod
    orig_build = rim_mod.build_brian2_rim_7channel

    def patched_build_brian2_rim_7channel(*args, **kwargs):
        inner_factory = orig_build(*args, **kwargs)
        return wrap_factory_with_codegen(inner_factory, target)

    rim_mod.build_brian2_rim_7channel = patched_build_brian2_rim_7channel

    import importlib
    import run_option_b_rim
    importlib.reload(run_option_b_rim)
    run_option_b_rim.build_brian2_rim_7channel = patched_build_brian2_rim_7channel

    # run_option_b_rim.main reads sys.argv to gate cp5/cp6. Inject cp5 cp6
    # explicitly so both are run regardless of how this driver is invoked.
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "cp5", "cp6"]
    try:
        t0 = time.time()
        verdict, summary = run_option_b_rim.main()
        elapsed = time.time() - t0
    finally:
        sys.argv = saved_argv

    out = Path(__file__).parent / f"cp5_rim_{target}_result.json"
    cp5_summary = {
        "checkpoint": "cython_CP5",
        "target": target,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "summary": {k: v for k, v in summary.items() if k != "trajectories"},
    }
    print()
    print(f"Wall-clock: {elapsed:.1f} s")
    print(f"Saved to {out}")
    out.write_text(json.dumps(cp5_summary, indent=2, default=str))
    return verdict, cp5_summary


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "cython"
    main(target)
