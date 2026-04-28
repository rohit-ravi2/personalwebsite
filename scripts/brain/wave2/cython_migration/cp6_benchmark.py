"""CP6 — Phase-δ-like representative benchmark.

Workload: each of {AVAL, AIY, RIM} runs 10 seconds simulated time with a
current-injection schedule (1 s baseline, 5 s step at +10 pA, 4 s recovery).
Single-cell (NeuronGroup with N=1), no clamp, free running. Approximates the
per-cell compute cost when integrating all 3 into Phase δ network coupling.

Two-pass: numpy then cython. Side-by-side wall-clock with identical model
graph. Numerical output sampled at end (final v, plateau v) for sanity check
of equivalence.

Outputs:
  cp6_benchmark_result.json  — timing measurements per cell per codegen
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


def run_cell(cell_name: str, target: str, run_total_ms: float = 10000.0,
             step_amp_pA: float = 10.0,
             baseline_ms: float = 1000.0,
             step_dur_ms: float = 5000.0) -> dict:
    """Run a single cell for run_total_ms with a step current injection.

    Returns timing + final-state info.
    """
    from brian2 import ms, defaultclock, prefs, pA
    import numpy as np

    if cell_name == "AVAL":
        from option_alpha_ava_cell import build_brian2_aval_4channel
        inner_factory = build_brian2_aval_4channel(record_components=False)
    elif cell_name == "AIY":
        from option_alpha_aiy_cell import build_brian2_aiy_7channel
        inner_factory = build_brian2_aiy_7channel(record_components=False)
    elif cell_name == "RIM":
        from option_alpha_rim_cell import build_brian2_rim_7channel
        inner_factory = build_brian2_rim_7channel(record_components=False)
    else:
        raise ValueError(f"unknown cell: {cell_name}")

    factory = wrap_factory_with_codegen(inner_factory, target)
    bundle = factory()
    print(f"  prefs.codegen.target = {prefs.codegen.target!r}")

    defaultclock.dt = 0.025 * ms
    bundle["disable_clamp"]()

    recovery_ms = run_total_ms - baseline_ms - step_dur_ms

    t0 = time.time()
    # Baseline
    bundle["inject_pA"](0.0)
    bundle["network"].run(baseline_ms * ms)
    # Step
    bundle["inject_pA"](step_amp_pA)
    bundle["network"].run(step_dur_ms * ms)
    # Recovery
    bundle["inject_pA"](0.0)
    bundle["network"].run(recovery_ms * ms)
    elapsed = time.time() - t0

    mon = bundle["monitor"]
    v_arr = np.asarray(mon.v[0]) * 1e3  # mV
    final_v = float(v_arr[-1])
    # Plateau: median over final 500 ms of step
    t_arr = np.asarray(mon.t) * 1e3  # ms
    plateau_window = (t_arr >= baseline_ms + step_dur_ms - 500) & (t_arr < baseline_ms + step_dur_ms)
    plateau_v = float(np.median(v_arr[plateau_window])) if plateau_window.any() else final_v

    return {
        "cell": cell_name,
        "target": target,
        "run_total_ms": run_total_ms,
        "step_amp_pA": step_amp_pA,
        "elapsed_s": elapsed,
        "final_v_mV": final_v,
        "plateau_v_mV": plateau_v,
    }


def main():
    print("=" * 70)
    print("CP6 — Phase-δ-like representative benchmark")
    print("=" * 70)
    print()
    print("Workload: each cell × 10s simulated time, +10 pA step injection")
    print("(1s baseline, 5s step, 4s recovery). dt=0.025 ms, rk4 integrator.")
    print()

    cells = ["AVAL", "AIY", "RIM"]
    results = {}

    # Numpy pass first (slower; warm CPU but cython codegen cache cold across
    # cells — that's fine, cython compile happens at cython pass)
    print("-" * 70)
    print("Pass 1: numpy")
    print("-" * 70)
    for cell in cells:
        print(f"\n[{cell}, numpy]")
        r = run_cell(cell, "numpy")
        print(f"  elapsed: {r['elapsed_s']:.2f} s   final v: {r['final_v_mV']:.4f} mV   plateau v: {r['plateau_v_mV']:.4f} mV")
        results.setdefault(cell, {})["numpy"] = r

    print()
    print("-" * 70)
    print("Pass 2: cython")
    print("-" * 70)
    for cell in cells:
        print(f"\n[{cell}, cython]")
        r = run_cell(cell, "cython")
        print(f"  elapsed: {r['elapsed_s']:.2f} s   final v: {r['final_v_mV']:.4f} mV   plateau v: {r['plateau_v_mV']:.4f} mV")
        results.setdefault(cell, {})["cython"] = r

    # Speedup table
    print()
    print("=" * 70)
    print("CP6 results")
    print("=" * 70)
    print(f"{'Cell':6s}  {'numpy (s)':>12s}  {'cython (s)':>12s}  {'speedup':>8s}  {'Δ final v':>12s}  {'Δ plateau':>12s}")
    print("-" * 70)
    speedups = []
    for cell in cells:
        n = results[cell]["numpy"]
        c = results[cell]["cython"]
        sp = n["elapsed_s"] / max(c["elapsed_s"], 1e-6)
        d_final = abs(n["final_v_mV"] - c["final_v_mV"])
        d_plat = abs(n["plateau_v_mV"] - c["plateau_v_mV"])
        speedups.append(sp)
        print(f"{cell:6s}  {n['elapsed_s']:>12.2f}  {c['elapsed_s']:>12.2f}  {sp:>7.2f}x  {d_final:>10.6f} mV  {d_plat:>10.6f} mV")

    n_total = sum(results[c]["numpy"]["elapsed_s"] for c in cells)
    c_total = sum(results[c]["cython"]["elapsed_s"] for c in cells)
    aggregate_speedup = n_total / max(c_total, 1e-6)
    print("-" * 70)
    print(f"{'TOTAL':6s}  {n_total:>12.2f}  {c_total:>12.2f}  {aggregate_speedup:>7.2f}x")
    print()
    print(f"Mean per-cell speedup: {sum(speedups)/len(speedups):.2f}x")

    out = Path(__file__).parent / "cp6_benchmark_result.json"
    out.write_text(json.dumps({
        "results": results,
        "n_total_s": n_total,
        "c_total_s": c_total,
        "aggregate_speedup_x": aggregate_speedup,
        "per_cell_speedups": dict(zip(cells, speedups)),
    }, indent=2))
    print(f"Saved to {out}")
    return results, aggregate_speedup


if __name__ == "__main__":
    main()
