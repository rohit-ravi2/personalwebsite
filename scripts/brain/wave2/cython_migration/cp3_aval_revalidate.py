"""CP3 — AVAL re-validation under cython codegen.

Strategy: monkey-patch `option_alpha_ava_cell.build_brian2_aval_4channel` so
the factory it returns wraps prefs.codegen.target = 'cython' on top of the
hardcoded numpy. Then invoke `run_option_alpha_cp4.main()` which runs the
exact Layer A vclamp + cclamp validation used to establish the AVAL
pre-migration baseline.

Pre-migration baseline (from `option_alpha_summary.md` and
`option_alpha_phase_f_results.json`):
  - voltage-clamp: 11/11 holds, max div ≤ 0.0035
  - current-clamp: 7/7 sweeps, V agreement ~5 decimal places

Post-cython acceptance: residuals match within ~10% of baseline values.
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

# Import target codegen wrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cython_wrapper import wrap_factory_with_codegen


def main(target: str = "cython"):
    print("=" * 70)
    print(f"CP3 — AVAL re-validation under codegen='{target}'")
    print("=" * 70)
    print()

    # Patch the cell module so all consumers (run_option_alpha_cp4) get
    # a cython-wrapped factory transparently.
    import option_alpha_ava_cell as ava_mod
    orig_build = ava_mod.build_brian2_aval_4channel

    def patched_build_brian2_aval_4channel(*args, **kwargs):
        inner_factory = orig_build(*args, **kwargs)
        return wrap_factory_with_codegen(inner_factory, target)

    ava_mod.build_brian2_aval_4channel = patched_build_brian2_aval_4channel

    # Reload run_option_alpha_cp4 so its top-level import binds the patched
    # symbol. Brian2-side: clear any cached state via start_scope on each
    # factory call (already done inside the cell builder).
    import importlib
    import run_option_alpha_cp4
    importlib.reload(run_option_alpha_cp4)

    # Re-patch after reload (reload re-binds `build_brian2_aval_4channel`
    # in run_option_alpha_cp4's namespace from the reloaded ava_mod).
    run_option_alpha_cp4.build_brian2_aval_4channel = patched_build_brian2_aval_4channel

    t0 = time.time()
    verdict, summary = run_option_alpha_cp4.main()
    elapsed = time.time() - t0

    # Extract residuals + speedup data
    out = Path(__file__).parent / f"cp3_aval_{target}_result.json"
    cp3_summary = {
        "checkpoint": "cython_CP3",
        "target": target,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "component_2a": {
            "panel_pass": summary["component_2a"]["panel_pass"],
            "n_holds": summary["component_2a"]["n_holds"],
            "n_holds_passing": summary["component_2a"]["n_holds_passing"],
            "fraction_passing": summary["component_2a"]["fraction_passing"],
        },
        "component_2b": {
            "panel_pass": summary["component_2b"]["panel_pass"],
            "n_sweeps": summary["component_2b"]["n_sweeps"],
            "n_sweeps_passing": summary["component_2b"]["n_sweeps_passing"],
            "fraction_sweeps_passing": summary["component_2b"]["fraction_sweeps_passing"],
            "aggregate_timepoint_pass_fraction": summary["component_2b"]["aggregate_timepoint_pass_fraction"],
        },
        "max_peak_div_2a": max(
            (e["feature_results"]["peak_I_pA"]["divergence"]
             for e in summary["component_2a"]["per_step_evaluations"]),
            default=None,
        ),
        "max_ss_div_2a": max(
            (e["feature_results"]["ss_I_pA"]["divergence"]
             for e in summary["component_2a"]["per_step_evaluations"]),
            default=None,
        ),
    }
    print()
    print(f"Wall-clock: {elapsed:.1f} s")
    print(f"max_peak_div: {cp3_summary['max_peak_div_2a']}")
    print(f"max_ss_div:   {cp3_summary['max_ss_div_2a']}")
    print(f"Saved to {out}")
    out.write_text(json.dumps(cp3_summary, indent=2))
    return verdict, cp3_summary


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "cython"
    main(target)
