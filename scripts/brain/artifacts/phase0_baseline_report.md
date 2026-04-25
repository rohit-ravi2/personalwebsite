# Phase 0 — Baseline measurement report

Generated: 2026-04-21 09:08:10

Current-state measurements across the Tier 2 / Tier 4 validation surface. Every downstream phase's pass threshold is ratified against these numbers — see the *Ratified thresholds* section.

## 1. Wall/simulated compute ratio

- **Measured ratio:** 3.06× wall seconds per simulated second (368 s wall for 120 s sim).
- **vs v3.3-audit extrapolation (2.69×):** 1.14× (within tolerance).
- **W0.2 full phenotype audit predicted:** 6.1 hours wall (120 runs × 60s).
- **W0.3 full scenario audit predicted:** 3.1 hours wall (60 runs × 60s).

## 2. Swap-jitter (Cython-migration decision)

- Mean wall: **102.63 ms** (σ = 4.75 ms, CV = 4.63%)
- Tolerance for T4-2 plateau discrimination: 15.0 ms.
- ✅ Within tolerance. T4-2 calibration can proceed on current hardware.

## 3. Phenotype ablation baseline (n=10 × 60s)

Current-state ablation deltas, v3 LIF brain, classifier-mode FSM.

| ablation | ΔREV | ΔQUI | ΔFWD | ΔOMG | ΔPIR |
|---|---|---|---|---|---|
| RIS / osmotic_shock | +0.00 ± 0.04 (4/10↓) | -0.00 ± 0.02 (4/10↓) | -0.00 ± 0.01 (5/10↓) | +0.00 ± 0.00 (0/10↓) | +0.00 ± 0.03 (2/10↓) |
| NSM / food | -0.47 ± 0.17 (7/7↓) | +0.40 ± 0.14 (0/7↓) | +0.02 ± 0.01 (0/7↓) | +0.11 ± 0.05 (0/7↓) | -0.06 ± 0.04 (5/7↓) |

## 4. Touch-scenario command-neuron rates (T4-3 baseline)

n=10 seeds, touch_anterior at t=5s. Pre window 1-5s, peri window 5-7s.

| neuron | pre (Hz) | peri (Hz) | Δ (Hz) | cascade role |
|---|---|---|---|---|
| ALML | 2.4 ± 0.6 | 88.2 ± 5.0 | +85.8 ± 4.9 | sensory |
| ALMR | 0.8 ± 0.5 | 85.8 ± 2.8 | +85.1 ± 3.1 | sensory |
| AVM | 0.7 ± 0.5 | 88.5 ± 1.8 | +87.8 ± 1.8 | sensory |
| AIBL | 8.3 ± 1.2 | 10.2 ± 1.9 | +1.8 ± 1.9 | 1st-order interneuron |
| AIBR | 14.8 ± 1.9 | 14.3 ± 2.5 | -0.4 ± 2.8 | 1st-order interneuron |
| AVAL | 45.2 ± 3.1 | 41.1 ± 3.6 | -4.0 ± 4.0 | reversal command |
| AVAR | 46.0 ± 3.3 | 42.9 ± 3.7 | -3.1 ± 4.6 | reversal command |
| AVEL | 29.3 ± 2.6 | 24.4 ± 2.5 | -5.0 ± 4.1 | secondary reversal |
| AVER | 33.9 ± 2.7 | 29.4 ± 2.6 | -4.4 ± 3.6 | secondary reversal |
| AVDL | 42.2 ± 2.8 | 35.3 ± 3.5 | -6.9 ± 3.9 | tertiary reversal |
| AVDR | 41.6 ± 2.8 | 34.4 ± 3.6 | -7.2 ± 4.4 | tertiary reversal |
| RIML | 32.1 ± 1.9 | 31.6 ± 3.6 | -0.6 ± 3.9 | tyraminergic gate |
| RIMR | 32.2 ± 2.4 | 34.4 ± 2.7 | +2.1 ± 3.6 | tyraminergic gate |

## 5. Scenario state distributions (T4-6 baseline)

| scenario | FWD | REV | OMG | PIR | QUI |
|---|---|---|---|---|---|
| spontaneous | 0.04 ± 0.01 | 0.79 ± 0.09 | 0.02 ± 0.00 | 0.09 ± 0.05 | 0.06 ± 0.05 |
| touch | 0.03 ± 0.01 | 0.82 ± 0.08 | 0.02 ± 0.00 | 0.07 ± 0.07 | 0.07 ± 0.04 |
| osmotic_shock | 0.02 ± 0.01 | 0.89 ± 0.04 | 0.02 ± 0.00 | 0.02 ± 0.03 | 0.05 ± 0.03 |
| food | 0.03 ± 0.01 | 0.85 ± 0.06 | 0.02 ± 0.00 | 0.06 ± 0.05 | 0.05 ± 0.03 |
| chemotaxis | 0.03 ± 0.01 | 0.85 ± 0.09 | 0.01 ± 0.01 | 0.09 ± 0.08 | 0.03 ± 0.03 |
| aerotaxis | 0.03 ± 0.01 | 0.84 ± 0.08 | 0.01 ± 0.01 | 0.09 ± 0.07 | 0.03 ± 0.03 |

## 6. T4-2 plateau baseline (15 neurons × 50 pA / 100 ms)

- **2/15 neurons** within ±20% of Gao & Hobert / Wang / Kawano targets.
- Failing neurons: AVAL, AVAR, AVEL, AVER, AVBL, AVBR, PVCL, PVCR, RMGL, RMGR, ALA, RIS, DVA

See `phase0_plateau_baseline.md` for per-neuron voltages and gaps.

## 7. T2-#4 sensory cascade baseline (5 cascades)

Uncalibrated cascade peak rates under canonical stimuli:

- **ASE**: peak 22.9 Hz
- **AWC**: peak 140.0 Hz
- **ASH**: peak 204.2 Hz
- **AFD**: peak 130.0 Hz
- **ALM**: peak 155.4 Hz

References in `references/` — digitisation pending for Frechet-distance evaluation.

## 8. Ratified pass thresholds per phase

| phase | current baseline | pass threshold | reference |
|---|---|---|---|
| **T4-3 synaptic calibration** | AVAL peri=41.1±3.6Hz, Δ=-4.0±4.0Hz | AVAL peri ≥ 20 Hz AND Δ ≥ +15 Hz on ≥ 8/10 seeds | Chalfie 1985 command cascade biology |
| **T4-5 RIS/Turek phenotype** | ΔQUI=-0.00±0.02 (n=10 seeds) | ΔQUI ≤ −0.30 with 95% CI excluding zero | Turek 2016 |
| **T4-2 AVA plateau (Gao & Hobert 2020)** | see phase0_plateau_baseline.csv — AVA g_ca_ns, tau_h_ms | plateau duration ∈ [480, 720] ms (20% of 600 ms target); amplitude ∈ [18, 22] mV above rest | Gao & Hobert 2020 Fig 3 |
| **T2-#4 sensory cascade calibration** | see phase0_cascade_baseline.npz — uncalibrated shape per cascade | Each cascade's rate trace ≤ 10% Frechet distance to digitised published ΔF/F (Thiele 2009, Chalasani 2007, Hilliard 2005, Clark 2006, O'Hagan 2005) | per-cascade primary refs |
| **T4-1 motor coupling (curvature ρ)** | CPG-driven forward bout body trace in scenario_traces/ | median ρ vs Tierpsy pool ≥ max(0.6, CPG_baseline + 0.15) | Tierpsy centerlines in data/external/wormpose |
| **T4-6 trajectory correlation** | per-neuron × per-event ρ distribution vs Atanas (current v3 LIF) | median ρ increases ≥ +0.10 relative to baseline; tail named in paper | Atanas 2023 (10 worms) |

## 9. Audit infrastructure status

`phase0_audit.py` implements the 3-tier config:

| tier | seeds | duration | use |
|---|---|---|---|
| `--quick` | 5 | 30 s | dev iteration (signal-visible) |
| `--default` | 10 | 60 s | phase gate / Phase 0 baseline |
| `--audit-long` | 10 | 120 s | final phenotype claims |
| `--v33-compat` | 3 | 20 s × 3 configs | historical reproducibility |
