# Phase 3b multi-event harness — results summary

**Source:** 10 Atanas 2023 worms, ~16 min/worm, 1.67 Hz calcium + scalar behavior.  

**Targets:** 22 across 5 tiers.  

**Horizons:** +1, +3, +8, +16 samples (~0.6, 1.8, 4.8, 9.6 s).  

**Feature sets:** values, lags (t,t-1,t-2), derivs (t, d/dt).  

**Split:** 70% train / 60 s embargo / 30% test.  

**Baselines:** AR(3) on same target, best Ridge α ∈ {.1, 1, 10, 100, 1000} picked on inner val split.


## Overall: 15/22 targets pass tier-stratified thresholds


### Tier 1 — 6/7 pass

| target | kind | horizon | features | AR | neural | lift | worms≥thr / 10 | pass |
|---|---|---:|---|---:|---:|---:|---:|:---:|
| `head_bias` | continuous | 8 | lags | -0.02 | -0.00 | **+0.021** ± 0.02 | 3 | ✗ |
| `headswing_flip` | event | 1 | derivs | 0.53 | 0.62 | **+0.087** ± 0.09 | 9 | ✓ |
| `omega_onset` | event | 1 | derivs | 0.51 | 0.93 | **+0.416** ± 0.08 | 6 | ✓ |
| `pirouette_entry` | event | 16 | lags | 0.51 | 0.70 | **+0.189** ± 0.12 | 6 | ✓ |
| `reversal_offset` | event | 1 | derivs | 0.54 | 0.84 | **+0.297** ± 0.07 | 10 | ✓ |
| `reversal_onset` | event | 1 | derivs | 0.54 | 0.75 | **+0.212** ± 0.10 | 10 | ✓ |
| `reversal_state` | state | 8 | derivs | 0.63 | 0.72 | **+0.084** ± 0.09 | 10 | ✓ |

### Tier 2 — 6/7 pass

| target | kind | horizon | features | AR | neural | lift | worms≥thr / 10 | pass |
|---|---|---:|---|---:|---:|---:|---:|:---:|
| `forward_run_offset` | event | 3 | derivs | 0.52 | 0.81 | **+0.294** ± 0.08 | 10 | ✓ |
| `forward_run_onset` | event | 1 | derivs | 0.55 | 0.82 | **+0.271** ± 0.09 | 10 | ✓ |
| `forward_run_state` | state | 8 | values | 0.54 | 0.73 | **+0.185** ± 0.15 | 10 | ✓ |
| `quiescence_onset` | event | 1 | lags | 0.52 | 0.77 | **+0.250** ± 0.14 | 4 | ✓ |
| `quiescence_state` | state | 3 | lags | 0.49 | 0.83 | **+0.332** ± 0.30 | 4 | ✓ |
| `roaming_onset` | event | 16 | values | 0.43 | 0.58 | **+0.155** ± 0.20 | 6 | ✓ |
| `roaming_state` | state | 16 | derivs | 0.77 | 0.72 | **-0.052** ± 0.27 | 3 | ✗ |

### Tier 3 — 1/5 pass

| target | kind | horizon | features | AR | neural | lift | worms≥thr / 10 | pass |
|---|---|---:|---|---:|---:|---:|---:|:---:|
| `ang_acc` | continuous | 8 | derivs | 0.03 | 0.05 | **+0.028** ± 0.36 | 6 | ✗ |
| `body_curv_rate` | continuous | 3 | lags | 0.02 | 0.06 | **+0.040** ± 0.05 | 3 | ✗ |
| `head_ang_vel` | continuous | 16 | lags | -0.01 | -0.00 | **+0.003** ± 0.01 | 1 | ✗ |
| `speed_burst_onset` | event | 1 | derivs | 0.49 | 0.77 | **+0.285** ± 0.19 | 8 | ✓ |
| `velocity_acc` | continuous | 3 | derivs | 0.00 | 0.06 | **+0.056** ± 0.06 | 2 | ✗ |

### Tier 4 — 2/3 pass

| target | kind | horizon | features | AR | neural | lift | worms≥thr / 10 | pass |
|---|---|---:|---|---:|---:|---:|---:|:---:|
| `headswing_dir` | multiclass | 1 | derivs | 0.85 | 0.88 | **+0.030** ± 0.13 | 4 | ✓ |
| `rev_mode_3class` | multiclass | 8 | derivs | 0.61 | 0.74 | **+0.125** ± 0.16 | 4 | ✓ |
| `state_5class` | multiclass | 16 | lags | 0.37 | 0.39 | **+0.020** ± 0.10 | 4 | ✗ |

## Cross-worm generalization (train 1-8, test 9-10)

Strict all-10 intersection neuron set used (22 targets aligned).

| target | tier | horizon | features | AR | neural | combined | lift |
|---|---|---:|---|---:|---:|---:|---:|
| `head_bias` | 1 | 16 | lags | -0.02 | -0.01 | -0.01 | **+0.009** |
| `headswing_flip` | 1 | 8 | derivs | 0.52 | 0.58 | 0.56 | **+0.061** |
| `omega_onset` | 1 | 1 | derivs | 0.49 | 0.85 | 0.85 | **+0.361** |
| `pirouette_entry` | 1 | 3 | lags | 0.52 | 0.80 | 0.80 | **+0.278** |
| `reversal_offset` | 1 | 1 | lags | 0.54 | 0.81 | 0.83 | **+0.270** |
| `reversal_onset` | 1 | 1 | derivs | 0.54 | 0.79 | 0.81 | **+0.250** |
| `reversal_state` | 1 | 3 | derivs | 0.75 | 0.85 | 0.85 | **+0.098** |
| `forward_run_offset` | 2 | 3 | derivs | 0.48 | 0.75 | 0.75 | **+0.266** |
| `forward_run_onset` | 2 | 1 | derivs | 0.56 | 0.75 | 0.80 | **+0.192** |
| `forward_run_state` | 2 | 3 | derivs | 0.65 | 0.79 | 0.81 | **+0.144** |
| `quiescence_onset` | 2 | 3 | lags | 0.51 | 0.79 | 0.79 | **+0.273** |
| `quiescence_state` | 2 | 8 | values | 0.59 | 0.84 | 0.85 | **+0.250** |
| `roaming_onset` | 2 | 8 | values | 0.46 | 0.62 | 0.61 | **+0.163** |
| `roaming_state` | 2 | 16 | derivs | 0.79 | 0.72 | 0.83 | **-0.071** |
| `ang_acc` | 3 | 8 | lags | 0.05 | 0.14 | 0.17 | **+0.089** |
| `body_curv_rate` | 3 | 1 | derivs | -0.03 | 0.07 | 0.06 | **+0.095** |
| `head_ang_vel` | 3 | 16 | derivs | 0.00 | -0.00 | -0.00 | **-0.000** |
| `speed_burst_onset` | 3 | 1 | derivs | 0.55 | 0.76 | 0.76 | **+0.211** |
| `velocity_acc` | 3 | 1 | lags | 0.02 | 0.10 | 0.11 | **+0.082** |
| `headswing_dir` | 4 | 1 | values | 0.80 | 0.80 | 0.80 | **+0.000** |
| `rev_mode_3class` | 4 | 8 | values | 0.44 | 0.54 | 0.60 | **+0.095** |
| `state_5class` | 4 | 16 | lags | 0.39 | 0.40 | 0.40 | **+0.012** |

## Executive summary

Neural activity provides meaningful signal above AR baseline for **15/22 targets**. Strongest lifts: `omega_onset` (+0.42), `quiescence_state` (+0.33), `reversal_offset` (+0.30), `forward_run_offset` (+0.29), `speed_burst_onset` (+0.28).

The continuous-behavior failure from fit_interface_v2 (neural ≤ AR on velocity, curvatures) is confirmed — but refocusing on **transition events** and **state onsets/offsets** exposes robust neural signal that survives the AR and embargo controls. This validates the event-based pivot.