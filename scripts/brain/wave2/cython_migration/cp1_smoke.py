"""CP1 minimal smoke test — pre-migration baseline timing.

Tiny Brian2 model, no external dependencies, run under whatever codegen
target is currently active. Records compile + run wall clock to establish
pre-migration baseline.

Usage:
    python cp1_smoke.py              # uses current prefs (auto/numpy/cython)
    python cp1_smoke.py cython       # force cython
    python cp1_smoke.py numpy        # force numpy
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
import time
from pathlib import Path


def main(target_arg: str | None = None) -> dict:
    from brian2 import (
        NeuronGroup, StateMonitor, Network, defaultclock,
        ms, mV, second, prefs,
    )

    if target_arg is not None:
        prefs.codegen.target = target_arg

    target_resolved = prefs.codegen.target
    print(f"prefs.codegen.target = {target_resolved!r}")

    # Tiny single-cell model: leaky integrator with 5-channel-like complexity
    eqs = """
    dv/dt = (-(v - v_rest) - i_a + i_inj) / tau : volt
    di_a/dt = (i_inf - i_a) / tau_a : volt
    i_inf = i_amp * 1.0 / (1.0 + exp(-(v - v_half) / k_slope)) : volt
    i_inj : volt
    v_rest : volt
    v_half : volt
    k_slope : volt
    i_amp : volt
    tau : second
    tau_a : second
    """

    G = NeuronGroup(1, eqs, method="rk4")
    G.v = -60 * mV
    G.i_a = 0 * mV
    G.i_inj = 5 * mV
    G.v_rest = -60 * mV
    G.v_half = -30 * mV
    G.k_slope = 8 * mV
    G.i_amp = 10 * mV
    G.tau = 20 * ms
    G.tau_a = 50 * ms

    M = StateMonitor(G, ("v",), record=True)

    net = Network(G, M)
    defaultclock.dt = 0.025 * ms

    # Two-phase timing: first run includes any compile time; second is pure run.
    t_first_start = time.time()
    net.run(500 * ms)
    t_first = time.time() - t_first_start

    # Reset and rerun (no fresh compile expected for cython since cache is warm)
    G.v = -60 * mV
    G.i_a = 0 * mV
    t_second_start = time.time()
    net.run(500 * ms)
    t_second = time.time() - t_second_start

    final_v = float(M.v[0][-1] / mV)
    print(f"  first run (incl. any compile): {t_first:.3f} s")
    print(f"  second run (warm):              {t_second:.3f} s")
    print(f"  final v: {final_v:.3f} mV")

    return {
        "codegen_target_arg": target_arg,
        "codegen_target_resolved": target_resolved,
        "first_run_s": t_first,
        "second_run_s": t_second,
        "final_v_mV": final_v,
    }


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    result = main(arg)
    out = Path(__file__).parent / "cp1_smoke_result.json"
    # Append to a list if already exists
    if out.exists():
        existing = json.loads(out.read_text())
        if isinstance(existing, dict):
            existing = [existing]
    else:
        existing = []
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2))
    print(f"  saved to {out}")
