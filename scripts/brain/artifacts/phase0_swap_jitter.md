# Phase 0 — W0.4c — Swap-jitter measurement

Executed 100 repetitions of a 300 ms plateau protocol on AVAL (50 pA somatic injection, 100 ms pulse). Warm-up 
run excluded. Brian2 uses `codegen.target='numpy'` (CPU, no cython 
JIT), so jitter comes from Python GC, OS scheduling, and — if 
relevant — swap pressure.

## Statistics

| metric | value |
|---|---|
| warmup | 264.7 ms (excluded from stats) |
| mean | 102.63 ms |
| median | 102.13 ms |
| std dev (σ) | 4.75 ms |
| coefficient of variation | 4.63 % |
| p5 / p95 | 96.39 / 110.26 ms |
| min / max | 92.94 / 119.19 ms |

## Decision

Tolerance threshold for T4-2 plateau calibration discrimination: **15.0 ms** (15 ms = ~2% of the 400-800 ms plateau duration window). Measured σ = **4.75 ms**.

✅ **σ within tolerance.** T4-2 plateau calibration can proceed on current hardware/execution settings without Cython target migration. Revisit if σ grows during longer ensemble runs.
