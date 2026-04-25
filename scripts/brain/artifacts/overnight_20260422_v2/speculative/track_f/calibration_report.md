# Track F (speculative) — HH AVA calibration

**EXPLORATORY — not yet rigorous.**

Generated: 2026-04-22 15:30:22
Wall time: 3.4 min

## Status: **FAIL**

Best params g_ca=3 nS, g_k=25 nS, g_leak=0.5 nS, tau_h=400 ms. amp=10.55 mV (target 20, err 0.47); dur=1.4 ms (target 600, err 1.00); ret=5.5 ms (target 1500, err 1.00).

## Calibration targets (from Mellem 2008 published values, not digitized trace)

| metric | target | tolerance | best result | err | pass |
|---|---|---|---|---|---|
| amplitude (mV) | 20 | ±10% | 10.55 | 0.47 | ✗ |
| duration (ms) | 600 | ±20% | 1.4 | 1.00 | ✗ |
| return (ms) | 1500 | ±30% | 5.5 | 1.00 | ✗ |

## Caveats

- Calibration targets were drawn from Mellem 2008 abstract
  + figure description, NOT from a digitized voltage-clamp
  trace. Matching these values does not imply waveform match.
- Grid search used, not Nelder-Mead (simpler + bounded).
- Channel roster minimal: egl-19 L-type Ca + delayed
  rectifier K + leak. CeNGEN roster would include more
  (shl-1, slo-1, slo-2). This is a first-pass scaffold only.
- Next step if PASS: digitize Mellem Fig 1d + refit on trace
  shape via L2 loss. Do NOT integrate into main simulator
  based on this grid-search result alone.