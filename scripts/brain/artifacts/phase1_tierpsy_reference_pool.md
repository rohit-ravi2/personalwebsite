# Phase 1 — Tierpsy reference pool (T4-1 curvature baseline)

Source: `data/external/wormpose/tierpsy_test_data/data/MANUAL_FEATS/Results/MANUAL_FEATS_skeletons.hdf5`
Sampling rate: 25.00 Hz
Streaks extracted: 5 (top by length, ≥ 30 contiguous good frames).

## Per-streak statistics

| idx | worm_id | length (frames) | length (s) |
|---|---|---|---|
| 0 | 2 | 1207 | 48.28 |
| 1 | 3 | 899 | 35.96 |
| 2 | 8 | 884 | 35.36 |
| 3 | 6 | 716 | 28.64 |
| 4 | 4 | 591 | 23.64 |

## T4-1 validation plan

- Simulator body output at 20 Hz (scenario audit `body_xy`)
- For each forward-bout window (FSM state FORWARD):
  1. Resample to TARGET_HZ
  2. Compute per-segment curvature (this file's schema)
  3. For each reference streak, compute per-segment ρ across time lags ± body-length-equivalent
  4. Pool into a (n_sim × n_ref × n_segments) cross-correlation distribution
- Exit threshold: median ρ ≥ max(0.6, CPG_baseline + 0.15)

## Caveats

- Tierpsy body length per worm normalised via arclength — 
  scale invariant, shape-comparison only.
- Tierpsy sampling is ~30 Hz; simulator is 20 Hz. Downsample 
  reference via linear interp to 20 Hz during comparison.
- Pool is from a single Tierpsy file with multiple worms. 
  Additional files (Zenodo 3837679 full release, WormPose 
  repo) can be added by re-running this with a SRC list.