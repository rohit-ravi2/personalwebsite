"""CP4 — AIY re-validation under cython codegen.

Strategy: monkey-patch `option_alpha_aiy_cell.build_brian2_aiy_7channel` so
the factory it returns wraps prefs.codegen.target = 'cython' on top of the
hardcoded numpy. Then invoke `run_option_b_aiy.main()`.

Pre-migration baseline:
  - voltage-clamp: 11/11 holds, max div ≤ 0.0113
  - current-clamp: 10/11 sweeps (-15 pA fails due to KQT-1 slow-gate drift)

Post-cython acceptance: residuals match within ~10% of baseline; -15 pA
expected to still fail with similar plateau drift.

Specific F18 concern: AIY uses AIY_ECA_MV = 127.59 (asymmetric USEION ca:
slo1egl19 reads but doesn't write ica). Verify this is preserved under
cython.
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
    print(f"CP4 — AIY re-validation under codegen='{target}'")
    print("=" * 70)
    print()

    import option_alpha_aiy_cell as aiy_mod
    orig_build = aiy_mod.build_brian2_aiy_7channel

    def patched_build_brian2_aiy_7channel(*args, **kwargs):
        inner_factory = orig_build(*args, **kwargs)
        return wrap_factory_with_codegen(inner_factory, target)

    aiy_mod.build_brian2_aiy_7channel = patched_build_brian2_aiy_7channel

    import importlib
    import run_option_b_aiy
    importlib.reload(run_option_b_aiy)
    run_option_b_aiy.build_brian2_aiy_7channel = patched_build_brian2_aiy_7channel

    t0 = time.time()
    verdict, summary = run_option_b_aiy.main()
    elapsed = time.time() - t0

    out = Path(__file__).parent / f"cp4_aiy_{target}_result.json"
    cp4_summary = {
        "checkpoint": "cython_CP4",
        "target": target,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "component_3_vc": summary.get("component_3", summary.get("component_3_vc", {})),
        "component_4_cc": summary.get("component_4", summary.get("component_4_cc", {})),
    }
    print()
    print(f"Wall-clock: {elapsed:.1f} s")
    print(f"Saved to {out}")
    out.write_text(json.dumps(cp4_summary, indent=2, default=str))
    return verdict, cp4_summary


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "cython"
    main(target)
