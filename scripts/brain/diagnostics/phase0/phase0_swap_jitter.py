#!/usr/bin/env python3
"""Phase 0 — W0.4c — Swap-jitter measurement.

Runs the T4-2 plateau injection protocol on a single AVA 100 times and
measures wall-time variance. If σ(wall) > 10% of the expected
plateau-duration discrimination window (20% of 400-800ms → ~80-160ms
tolerance, jitter tolerance set at 15ms), flag that swap pressure will
degrade T4-2 plateau calibration discrimination.

Uses a mini-version of the plateau protocol so each iteration is
~300ms simulated — keeps the total runtime bounded while providing
real jitter statistics.

Output: artifacts/phase0_swap_jitter.json + .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brian2 import (
    StateMonitor, Network, start_scope, ms, mV, pA, defaultclock
)
from compartmental_neurons import build_compartmental_group

ART = Path(__file__).resolve().parent.parent.parent / "artifacts"
OUT_JSON = ART / "phase0_swap_jitter.json"
OUT_MD = ART / "phase0_swap_jitter.md"

N_ITER = 100
SETTLE_MS = 50.0
INJECT_MS = 100.0
POST_MS = 150.0
JITTER_TOLERANCE_MS = 15.0
TARGET_NEURON = "AVAL"


def single_run():
    """One mini-protocol. Returns wall seconds."""
    t0 = time.time()
    start_scope()
    defaultclock.dt = 0.1 * ms
    grp, names = build_compartmental_group()
    idx = names.index(TARGET_NEURON)
    net = Network(grp)
    net.run(SETTLE_MS * ms)
    grp.I_ext[idx] = 50 * pA
    net.run(INJECT_MS * ms)
    grp.I_ext[idx] = 0 * pA
    net.run(POST_MS * ms)
    return time.time() - t0


def main():
    print(f"Swap-jitter test: {N_ITER} repetitions of "
          f"{SETTLE_MS+INJECT_MS+POST_MS}ms plateau protocol on {TARGET_NEURON}")
    wall_times_s = []
    t0 = time.time()
    # Warm-up run (first Brian2 compile is always anomalously slow)
    warmup_s = single_run()
    print(f"  warmup: {warmup_s*1000:.1f}ms (excluded)")

    for i in range(N_ITER):
        w = single_run()
        wall_times_s.append(w)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_ITER}] last={w*1000:.1f}ms "
                  f"running_mean={np.mean(wall_times_s)*1000:.1f}ms")

    total_wall = time.time() - t0
    wall_ms = np.array(wall_times_s) * 1000
    stats = {
        "n_iterations": N_ITER,
        "warmup_ms": round(warmup_s * 1000, 1),
        "mean_ms": round(float(np.mean(wall_ms)), 2),
        "median_ms": round(float(np.median(wall_ms)), 2),
        "std_ms": round(float(np.std(wall_ms)), 2),
        "min_ms": round(float(np.min(wall_ms)), 2),
        "max_ms": round(float(np.max(wall_ms)), 2),
        "p5_ms": round(float(np.percentile(wall_ms, 5)), 2),
        "p95_ms": round(float(np.percentile(wall_ms, 95)), 2),
        "cv_pct": round(float(np.std(wall_ms) / np.mean(wall_ms) * 100), 2),
        "total_wall_s": round(total_wall, 2),
        "jitter_tolerance_ms": JITTER_TOLERANCE_MS,
        "exceeds_tolerance": bool(np.std(wall_ms) > JITTER_TOLERANCE_MS),
    }

    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Mean wall: {stats['mean_ms']}ms | σ: {stats['std_ms']}ms | "
          f"CV: {stats['cv_pct']}%")
    print(f"Tolerance (15ms for T4-2): "
          f"{'EXCEEDED' if stats['exceeds_tolerance'] else 'OK'}")

    lines = [
        "# Phase 0 — W0.4c — Swap-jitter measurement",
        "",
        f"Executed {N_ITER} repetitions of a "
        f"{int(SETTLE_MS + INJECT_MS + POST_MS)} ms plateau protocol on "
        f"{TARGET_NEURON} (50 pA somatic injection, 100 ms pulse). Warm-up ",
        "run excluded. Brian2 uses `codegen.target='numpy'` (CPU, no cython ",
        "JIT), so jitter comes from Python GC, OS scheduling, and — if ",
        "relevant — swap pressure.",
        "",
        "## Statistics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| warmup | {stats['warmup_ms']} ms (excluded from stats) |",
        f"| mean | {stats['mean_ms']} ms |",
        f"| median | {stats['median_ms']} ms |",
        f"| std dev (σ) | {stats['std_ms']} ms |",
        f"| coefficient of variation | {stats['cv_pct']} % |",
        f"| p5 / p95 | {stats['p5_ms']} / {stats['p95_ms']} ms |",
        f"| min / max | {stats['min_ms']} / {stats['max_ms']} ms |",
        "",
        "## Decision",
        "",
        f"Tolerance threshold for T4-2 plateau calibration discrimination: "
        f"**{JITTER_TOLERANCE_MS} ms** (15 ms = ~2% of the 400-800 ms plateau "
        f"duration window). Measured σ = **{stats['std_ms']} ms**.",
        "",
    ]
    if stats["exceeds_tolerance"]:
        lines.append(
            "⚠️ **σ exceeds tolerance.** T4-2 plateau duration fits will "
            "be noise-limited under current execution conditions. Recommend "
            "Phase 1.5 intervention: either switch Brian2 to cython target "
            "(`prefs.codegen.target='cython'`) or batch plateau protocols "
            "across neurons to amortise per-run overhead."
        )
    else:
        lines.append(
            "✅ **σ within tolerance.** T4-2 plateau calibration can proceed "
            "on current hardware/execution settings without Cython target "
            "migration. Revisit if σ grows during longer ensemble runs."
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
