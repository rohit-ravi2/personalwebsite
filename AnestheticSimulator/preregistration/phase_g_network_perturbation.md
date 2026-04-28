# Phase G — Network-level perturbation runs

**Phase letter:** G
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** Phases D, E, F all complete. Phase G is the integration phase.
**Successor:** Phase H (empirical validation) consumes Phase G's run outputs.
**Compute:** local Brian2, ~80 GPU-hours equivalent.

---

## 1. Goal

Run **2,400 perturbation simulations** spanning 3 anesthetics × 4 doses × 4 genotypes × 5 scenarios × 10 seeds. Each simulation applies Phase D's per-target kinetic shifts (anesthetic-perturbed channels), Phase E's Markov synapses, and Phase F's metabolic layer to the production simulator's network architecture. Per-target lesion runs (anesthetic effect applied **only** at one mechanism class at a time) test whether the multi-target framing is empirically supported.

The phase delivers the network-level prediction set that Phase H validates against published wet-lab data.

---

## 2. Background

### 2.1 Run grid

| Dimension | Values | Count |
|---|---|---|
| Anesthetic | halothane, isoflurane, propofol | 3 |
| Dose (× clinical EC50) | 0.5×, 1×, 2×, 5× | 4 |
| Genotype | WT, gas-1(fc21), unc-79(e1068), unc-13(s69) | 4 |
| Scenario | spontaneous, touch_response, food_navigation, osmotic_avoidance, NaCl_chemotaxis | 5 |
| Seed | 0-9 | 10 |
| **Total** | | **2,400** |

### 2.2 Per-target lesion sub-grid

Within the WT-halothane-1× condition (chosen as the load-bearing test point), Wave P additionally runs:

| Lesion class | Kinetic shifts applied |
|---|---|
| G.2.0 — full multi-target | All Phase D shifts applied |
| G.2.1 — GABA only | Only UNC-49 / EXP-1 potentiation |
| G.2.2 — NCA only | Only NCA-1 / NCA-2 / UNC-79 / UNC-80 / NLF-1 block |
| G.2.3 — K2P only | Only TWK-18 / TWK-7 / TWK-29 potentiation |
| G.2.4 — SNARE only | Only Markov synapse Ca cooperativity shift |
| G.2.5 — Complex I only | Only metabolic-layer Complex I block |
| G.2.6 — GluCl only | Only AVR-14 / AVR-15 / GLC-1 / GLC-2 |
| G.2.7 — nAChR only | Only ACR-16 / ACR-2 / UNC-29 / UNC-38 / UNC-63 / LEV-1 |

5 seeds × 8 lesion conditions = 40 additional runs. Comparison of G.2.0 vs G.2.1-G.2.7 is the load-bearing **multi-target falsifiability test at the network level**.

### 2.3 Behavioral readout

The simulator's FSM produces per-time-step state (FORWARD, REVERSE, QUIESCENT). Wave P adds an **IMMOBILIZED** state defined as:

```
IMMOBILIZED if (mean firing rate of {AVA, AVB, AVD, AVE, PVC} over last 10 s) < threshold
            AND (variance of motor-pool firing < motor_threshold) over last 10 s
```

Threshold is calibrated from G.1.0 (the WT no-anesthetic control run): set such that the WT control spends < 5% of time in IMMOBILIZED state. Calibration freezes after the WT control runs.

### 2.4 Locomotion rate

For chemotaxis / navigation scenarios, the simulator reports body-axis displacement per second. Loss-of-locomotion at clinical anesthetic concentration should be < 50% of baseline (from Crowder 1996's data).

### 2.5 EC50 fitting

For each (anesthetic, genotype, scenario, lesion-class) cell, fit a Hill curve to the 4-dose data:

```
fraction_immobilized(dose) = dose^h / (EC50^h + dose^h)
```

with `h` (Hill coefficient) and `EC50` as fitting parameters. EC50 is the primary readout.

---

## 3. Method

### 3.1 Run driver

```python
# src/phase_g_network_runs.py
def run_grid(out_dir):
    for anesthetic in ["halothane", "isoflurane", "propofol"]:
        for dose_mult in [0.5, 1.0, 2.0, 5.0]:
            for genotype in ["WT", "gas1", "unc79", "unc13"]:
                for scenario in ["spontaneous", "touch", "food", "osmotic", "NaCl"]:
                    for seed in range(10):
                        run_one(anesthetic, dose_mult, genotype, scenario, seed, out_dir)

    # Lesion sub-grid (only for halothane, WT, 1x, single scenario for tractability)
    for lesion in ["full", "GABA", "NCA", "K2P", "SNARE", "complexI", "GluCl", "nAChR"]:
        for seed in range(5):
            run_one_lesion("halothane", 1.0, "WT", "spontaneous", lesion, seed, out_dir)
```

### 3.2 Per-run config

```python
def run_one(anesthetic, dose_mult, genotype, scenario, seed, out_dir):
    occupancy = load_occupancy(anesthetic, dose_mult)         # Phase C
    kinetic_shifts = load_kinetic_shifts(occupancy)           # Phase D
    metabolic = MetabolicLayer(genotype=genotype, occupancy_CI=occupancy["CI"])  # Phase F
    synapses = MarkovSynapseModule(occupancy_SNARE=occupancy["SNARE"])  # Phase E
    network = ProductionSimulator(
        wave2_channels=load_wave2_channels(),
        kinetic_shifts=kinetic_shifts,
        synapse_module=synapses,
        metabolic_module=metabolic,
        scenario=scenario,
        seed=seed,
    )
    network.run(duration=120 * second)
    save_trace(network, out_dir / f"{anesthetic}_{dose_mult}_{genotype}_{scenario}_{seed}.npz")
```

### 3.3 Per-run timing

Each Brian2 LIF network run with metabolic layer + Markov synapses + 300 cells × 120 s simulation takes ~2 minutes wall clock on RTX 4060 Ti (Brian2 + numpy backend).

2,400 runs × 2 min = 80 hours. Plus 40 lesion runs ≈ 1.5 hours. Total compute ~82 hours.

### 3.4 Output structure

Each run produces:

```
artifacts/runs/<anesthetic>_<dose_mult>_<genotype>_<scenario>_<seed>.npz:
  - V_traces:        (300 cells, T)    membrane voltage
  - firing_rate:     (300, T_bin)      binned firing rate
  - fsm_state:       (T,)              FSM state per time step
  - immobilized:     (T,)              IMMOBILIZED indicator
  - locomotion:      (T,)              body-axis displacement rate
  - ATP:             (300, T)          per-cell ATP
  - g_K_ATP:         (300, T)          per-cell K-ATP open conductance
  - meta:            dict              run config snapshot
```

Aggregated results:

```
artifacts/runs/aggregated_ec50.csv:
  anesthetic, dose, genotype, scenario, lesion_class,
  fraction_immobilized_mean, fraction_immobilized_std,
  locomotion_rate_mean, locomotion_rate_std,
  fitted_EC50, fitted_hill_n
```

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| 2,400 main grid runs | local Brian2 (Brian2 + numpy on CPU; optional Brian2GeNN if available) | 80 | $0 |
| 40 lesion runs | local | 1.5 | $0 |
| Aggregation + EC50 fitting | local CPU | 2 | $0 |
| Visualization | local | 1 | $0 |
| **Total Phase G** | | **~85 hours** | **$0** |

If 80 hours becomes infeasible on the local machine, the lesion subset (40 runs) is non-negotiable but the main grid can be reduced to 5 seeds (1,200 runs) as a fallback. Document the fallback in the completion report.

---

## 5. Preregistered success criteria (Gate G.1)

1. **G.1.1 — WT EC50 sanity:** simulated WT halothane immobilization EC50 within 2× of Crowder 1996's 3% atm; simulated WT isoflurane EC50 within 2× of Morgan 1995's 5%.
2. **G.1.2 — gas-1 hypersensitivity:** simulated gas-1 EC50 leftward-shifted by 1.5×-4× vs WT (matching Morgan & Sedensky 1995's 2-3× within 50%).
3. **G.1.3 — unc-79 resistance:** simulated unc-79 EC50 rightward-shifted by 1.5×-4× vs WT (matching Sedensky 1992's 2-3× within 50%).
4. **G.1.4 — unc-13 hypersensitivity:** simulated unc-13(s69) EC50 leftward-shifted by 1.5×-4× vs WT (matching van Swinderen 1999 within 50%).
5. **G.1.5 — Multi-target lesion test (load-bearing):** **No single lesion class (G.2.1-G.2.7) reproduces the full effect (G.2.0) within 50%.** All single-class lesions must produce a smaller fractional immobilization than the full effect at WT halothane 1× EC50. This is the network-level falsification test of the multi-target framing.

If G.1.5 fails — i.e., a single lesion class reproduces > 80% of the full effect — the multi-target framing is empirically falsified at the network level. The simulator's behavior is dominated by one mechanism, and Wave P's elaborate multi-target machinery is overengineered relative to a single-target model.

---

## 6. Halting rules

**Pause and surface:**

- WT EC50 simulation off by > 5× from Crowder/Morgan published values → binding-affinity calibration is wrong (Phase B/C/D rebuild needed).
- All four genotypes show identical EC50 → genotype-specific perturbations are not propagating; integration bug.
- Single-class lesion at G.2.x reproduces > 80% of the full effect → multi-target framing falsified at network level; halt and surface.
- Brian2 numerical instability surfaces (e.g., NaN voltages, infinite firing) on > 5% of runs → numerical issue; halt to debug.

**Document and continue:**

- A single (anesthetic, genotype, scenario) cell shows unexpected behavior → flag, run extra seeds to determine if statistical or structural.
- Locomotion readout is noisy in osmotic / chemotaxis scenarios → use spontaneous + touch as primary readouts.
- One scenario produces unstable FSM → use only the stable scenarios for Phase H.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `artifacts/runs/<config>.npz` | Per-run traces (2,440 files) |
| `artifacts/runs/aggregated_ec50.csv` | Aggregated EC50 table |
| `artifacts/runs/lesion_comparison.csv` | Per-lesion-class comparison vs full effect |
| `artifacts/runs/lesion_test_result.md` | **Load-bearing**: G.1.5 evaluation |
| `artifacts/runs/dose_response_curves.png` | Hill-fit visualization |
| `artifacts/runs/phase_g_completion.md` | end-of-block report |

---

## 8. Falsifiability checks

The phase's premise: **"The multi-target binding profile from Phase C, translated into kinetic shifts via Phase D and applied to the production simulator network with Phase E synapses and Phase F metabolic layer, reproduces published *C. elegans* anesthetic phenotypes — and per-target lesion analysis confirms the multi-target framing."**

Falsified if:

1. **Per-target lesion reproduces full effect (G.1.5 fails)** — single-target framing is sufficient; multi-target is overengineered.
2. **WT EC50 wrong by > 5× (G.1.1 fails)** — binding/kinetic calibration is structurally wrong.
3. **Genotype-specific shifts do not appear (G.1.2-G.1.4 all fail)** — mutant simulation does not propagate; the simulator cannot distinguish genotypes.

A network-level falsification would be a publishable negative result.

---

## 9. Integration points

**Inputs from earlier phases:**

- Phase C `artifacts/occupancy/occupancy_matrix.npz`
- Phase D `artifacts/kinetics/anesthetic_kinetic_shifts.npz`
- Phase E `artifacts/markov/markov_synapse_module.py`
- Phase F `artifacts/metabolic/metabolic_layer_module.py`
- Wave 2 `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels/*.py`
- Notebook pipeline `/home/rohit/Desktop/C-Elegans/New Notebooks/data_derived/connectome_adult.npz` etc.

**Outputs consumed by:**

- **Phase H** (`src/phase_h_validation.py`) — reads `aggregated_ec50.csv` and compares to anchor predictions.
- **Phase I** (stretch) — reads per-run traces for inverse-design comparison.
- **Phase J** (stretch) — reads per-run V_traces for Phi / Lyapunov computation.

---

## 10. Citation hygiene declaration

- Crowder 1996 — PMID 8855256. [VERIFIED]
- Morgan 1995 — PMID lookup needed.
- Morgan & Sedensky 1995 — PMID 7549290. [VERIFIED]
- Sedensky 1992 — PMID 1346264. [VERIFIED]
- van Swinderen 1999, unc-13 hypersensitivity — (PMID lookup needed). [BLOCKING]
- van Swinderen 2004 — (PMID lookup needed). [BLOCKING]
- Sedensky 2001, twk-18 — PMID 11756669. [VERIFIED]

**Pre-flight verification status:** 4 of 7 verified.

---

## 11. Risk register (Phase G)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Brian2 LIF + Markov synapses + metabolic layer numerically unstable | Medium | Use semi-implicit solver; bound state variables; smoke-test on small networks first |
| 80 hours of Brian2 runs exceeds available local compute window | Medium | Reduce seeds to 5 (40 hours); document fallback |
| FSM IMMOBILIZED threshold calibration is sensitive to WT control variability | Medium | Use 90th percentile of WT as threshold; sensitivity analysis on ±10% threshold |
| EC50 fits are noisy with only 4 doses | Medium | Add 0.25× and 10× doses if needed; use bootstrap confidence intervals |
| Lesion test fails because per-class effects don't sum linearly | Low | Document non-linearity; non-linearity is consistent with multi-target framing actually |

---

## 12. Phase G execution plan

1. Pre-flight citation verification (3 PMIDs).
2. Integration test: run 1 WT halothane 1× spontaneous seed=0 end-to-end. Confirm trace plausibility.
3. Calibrate FSM IMMOBILIZED threshold against WT control (no-anesthetic) run.
4. Execute main grid (2,400 runs).
5. Execute lesion sub-grid (40 runs).
6. Aggregate; fit Hill curves; produce `aggregated_ec50.csv`.
7. Compile `lesion_test_result.md` (Gate G.1.5 the load-bearing artifact).
8. Gate G.1 evaluation; end-of-block report.
