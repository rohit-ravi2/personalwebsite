# Phase 1 — T4-1 curvature comparison

**Mode:** CPG baseline (Phase 0)
**Reference:** 5 Tierpsy streaks at 25.0 Hz
**Scenarios scanned:** 1
**Forward bouts compared:** 10
**Total cross-correlation samples:** 1000

## Global statistics

- Median ρ across all (segment × bout × reference) triples: **-0.001**
- Standard deviation: 0.371
- T4-1 pass threshold (post-calibration): **≥ 0.60**

## Per-segment ρ distribution

| segment | median | std | n |
|---|---|---|---|
| 0 | +0.192 | 0.377 | 50 |
| 1 | +0.192 | 0.377 | 50 |
| 2 | +0.059 | 0.423 | 50 |
| 3 | +0.036 | 0.379 | 50 |
| 4 | +0.018 | 0.302 | 50 |
| 5 | -0.133 | 0.336 | 50 |
| 6 | -0.295 | 0.304 | 50 |
| 7 | -0.010 | 0.402 | 50 |
| 8 | +0.067 | 0.438 | 50 |
| 9 | -0.212 | 0.439 | 50 |
| 10 | +0.018 | 0.285 | 50 |
| 11 | -0.210 | 0.368 | 50 |
| 12 | +0.098 | 0.273 | 50 |
| 13 | -0.210 | 0.448 | 50 |
| 14 | +0.184 | 0.378 | 50 |
| 15 | +0.181 | 0.291 | 50 |
| 16 | -0.151 | 0.352 | 50 |
| 17 | -0.181 | 0.314 | 50 |
| 18 | -0.139 | 0.264 | 50 |
| 19 | -0.139 | 0.264 | 50 |

## Interpretation

- **Mid-body segments (5-15)** should carry the locomotion 
  wave — highest ρ expected here.
- **Head (0-2) and tail (18-19)** often lower ρ due to 
  endpoint boundary effects in the turning-angle curvature 
  definition. This is expected, not a bug.
- If mid-body ρ < 0.4 on CPG baseline, CPG parameters may 
  differ significantly from Tierpsy worm wavelength/frequency.
