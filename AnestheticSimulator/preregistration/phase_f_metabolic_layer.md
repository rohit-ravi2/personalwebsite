# Phase F — Metabolic state layer (ATP[t] dynamics + K-ATP coupling)

**Phase letter:** F
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** Phase C (occupancy matrix). Phase F runs in parallel with Phases D, E after Gate C.1.
**Successor:** Phase G consumes Phase F's metabolic-layer module.
**Compute:** local CPU minutes; no GPU.

---

## 1. Goal

Add an **ATP[t] dynamics layer** to the simulator: per-cell ATP concentration evolves under production (Complex I + Complex II + glycolytic input V) and consumption (Na/K-ATPase × firing rate + Ca-ATPase × Ca²⁺ load). Couple ATP[t] to K-ATP channel open probability (g_K-ATP_open ∝ 1 / (1 + [ATP] / K_ATP)). Apply Phase D's per-target Complex I shift to model gas-1-style mitochondrial perturbation under anesthetic.

The phase delivers the metabolic-anesthesia mechanism that explains gas-1 hypersensitivity (Morgan & Sedensky 1995): reduced Complex I capacity → ATP drift → K-ATP partial open → resting hyperpolarization → cell becomes more sensitive to anesthetic-induced excitability shifts.

---

## 2. Background

### 2.1 The gas-1 hypothesis

Morgan & Sedensky 1995 (PMID 7549290) reported that *C. elegans* gas-1(fc21) mutants are **2-3× more sensitive** to volatile anesthetic immobilization than WT (i.e., EC50 shifted leftward by 2-3×). GAS-1 is the *C. elegans* homolog of mammalian NDUFS2, a core 49 kDa subunit of mitochondrial Complex I.

The mechanistic explanation has been refined over several papers:

- Kayser 2001 (PMID lookup needed): GAS-1 mutant has ~30-50% reduced Complex I activity.
- Kayser 2004, 2008 follow-ups: hypersensitivity is reproducible across volatile anesthetics; isolated to Complex I (mev-1 Complex II mutants show different / smaller effects); ATP-related.
- The link to anesthetic immobilization: Complex I anesthetic binding produces an additional ~15-30% reduction in WT activity at clinical concentration; combined with gas-1's 30-50% basal reduction, the worm crosses an excitability threshold.

The mechanism Wave P models:

1. Complex I baseline activity is `R_CI`.
2. Anesthetic occupancy at Complex I subunits reduces `R_CI` by `× (1 - occupancy)`.
3. gas-1 mutation reduces `R_CI` by 30-50% baseline (independent of anesthetic).
4. ATP production = `R_CI × η + R_CII × η + V_glycolysis × η`, where η is approximate ATP yield per electron-pair flux.
5. ATP consumption = `k_NaK × firing_rate + k_Ca × [Ca²⁺]_internal + k_basal`.
6. ATP[t] reaches steady state at `[ATP]_ss = production / consumption`.
7. K-ATP channel: `P_open(ATP) = 1 / (1 + ([ATP] / K_ATP)^n_ATP)` where K_ATP ≈ 100 µM.
8. K-ATP partial open hyperpolarizes the cell, which compounds anesthetic-induced shifts (e.g., GABA-A potentiation produces a larger fractional change at a hyperpolarized resting state).

### 2.2 Quantitative anchors

| Parameter | Value | Source |
|---|---|---|
| Mammalian neuron resting [ATP] | ~2-5 mM | textbook |
| K_ATP for K-ATP channel | ~100 µM (cardiac); ~1 mM (neuronal) | varies; per-tissue |
| n_ATP (Hill coeff for K-ATP closure by ATP) | ~2 | textbook |
| Complex I anesthetic occupancy at clinical halothane | TBD (Phase C output) | Phase C |
| gas-1 fc21 Complex I activity reduction | ~30-50% | Kayser 2001 |

### 2.3 Cross-checks to validate the metabolic layer

The metabolic layer must reproduce:

- WT baseline: [ATP] ≈ 2-3 mM, K-ATP P_open < 5% under spontaneous firing.
- gas-1 baseline: [ATP] reduced (~50% of WT), K-ATP P_open elevated (~20-30%), resting V slightly hyperpolarized (~5-10 mV).
- WT + halothane 1× EC50: small additional [ATP] reduction (Complex I is partially blocked by halothane), small K-ATP P_open elevation.
- gas-1 + halothane 1× EC50: large [ATP] reduction (compounded), large K-ATP P_open, large hyperpolarization. Quantitatively, anesthetic immobilization EC50 should be 2-3× lower in gas-1 vs WT.
- mev-1 Complex II mutant: smaller effect than gas-1 (Complex II contributes less anesthetic-relevant flux).

---

## 3. Method

### 3.1 ATP[t] dynamics ODE

```
d[ATP]/dt = production - consumption

production = R_CI(occupancy_CI, gas1_factor) × η_CI
            + R_CII(occupancy_CII, mev1_factor) × η_CII
            + V_glycolysis × η_gly

consumption = k_NaK × firing_rate
              + k_Ca × ([Ca²⁺]_internal / K_Ca_ATPase)
              + k_basal
```

with parameters (per-cell, default neuronal):

```
R_CI_baseline = 1.0   # arbitrary units; calibrated to give [ATP]_ss ≈ 3 mM
R_CII_baseline = 0.3  # ~30% of CI flux in neurons
V_glycolysis = 0.1    # baseline glycolytic ATP, small in neurons
η_CI = 1.0            # arbitrary unit
η_CII = 0.5           # CII produces less ATP per flux
η_gly = 2.0           # glycolysis produces 2 ATP per glucose, but unit is calibrated
k_NaK = 0.05          # per Hz of firing
k_Ca = 0.02           # per µM Ca²⁺
k_basal = 0.1         # per second housekeeping
gas1_factor = 1.0 (WT) or 0.6 (fc21 mutant)
mev1_factor = 1.0 (WT) or 0.5 (mev-1 mutant)
occupancy_CI from Phase C (anesthetic effect)
occupancy_CII from Phase C (much smaller; mev-1 is a control)
```

### 3.2 K-ATP coupling

Per-cell K-ATP channel:

```
P_open_K_ATP(ATP) = 1 / (1 + ([ATP] / K_ATP)^n_ATP)
g_K_ATP(ATP) = g_K_ATP_max × P_open_K_ATP(ATP)
```

K_ATP = 1 mM (neuronal); n_ATP = 2 default; g_K_ATP_max calibrated per cell (Wave 2 has not yet translated K-ATP channels — Wave P uses a phenomenological default at the single-compartment level).

### 3.3 Mutant simulation

```python
# WT
metabolic = MetabolicLayer(gas1_factor=1.0, mev1_factor=1.0)
# gas-1(fc21)
metabolic_gas1 = MetabolicLayer(gas1_factor=0.6, mev1_factor=1.0)
# mev-1 (Complex II)
metabolic_mev1 = MetabolicLayer(gas1_factor=1.0, mev1_factor=0.5)
# atp-2 (severe ATP synthase defect)
metabolic_atp2 = MetabolicLayer(gas1_factor=1.0, mev1_factor=1.0, atp_synthase_factor=0.4)
```

### 3.4 Coupling to network

The metabolic layer publishes per-cell `[ATP](t)` and `g_K_ATP(t)` via a Brian2 NeuronGroup state variable. Phase G's network module reads these and applies them to the cell's Brian2 equations as an additive K conductance.

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| Module implementation + smoke tests | local CPU | 3 | $0 |
| WT, gas-1, mev-1, atp-2 baseline validation | local CPU | 1 | $0 |
| Anesthetic perturbation at WT vs gas-1 | local CPU | 1 | $0 |
| **Total Phase F** | | **~5 hours** | **$0** |

Phase F is the lightest substantive phase computationally. The implementation is a small ODE module + K-ATP channel.

---

## 5. Preregistered success criteria (Gate F.1)

1. **F.1.1 — WT baseline:** simulated [ATP]_ss between 1.5 and 5 mM; K-ATP P_open between 0.1% and 5%. Resting V shift due to K-ATP < 2 mV.
2. **F.1.2 — gas-1 baseline:** simulated [ATP]_ss reduced by 30-60% vs WT; K-ATP P_open elevated to 10-40%; resting V hyperpolarized by 3-15 mV vs WT.
3. **F.1.3 — gas-1 anesthetic hypersensitivity (load-bearing):** simulated immobilization EC50 in gas-1 + halothane is **leftward-shifted by 1.5×-4×** relative to WT + halothane (matching Morgan & Sedensky 1995's 2-3× within 50% bracket).
4. **F.1.4 — mev-1 differential:** simulated mev-1 effect is qualitatively smaller than gas-1 (e.g., mev-1 EC50 shift < 1.5× and gas-1 shift > 1.5×), consistent with Complex I being the dominant anesthetic-sensitive node.

F.1.3 is the load-bearing test of the metabolic layer. Failure here invalidates the metabolic-anesthesia mechanism Phase F is built on.

---

## 6. Halting rules

**Pause and surface:**

- WT [ATP]_ss outside 0.5-10 mM under any reasonable parameter set → metabolic ODE structurally wrong.
- gas-1 hypersensitivity does not appear in the simulator (i.e., gas-1 EC50 ≈ WT EC50) → either the mechanism is wrong or the parameter coupling is too weak.
- gas-1 EC50 leftward shift is much greater than 4× — over-sensitive — suggests K_ATP coupling parameters are too aggressive.

**Document and continue:**

- mev-1 effect direction or magnitude unexpected → flag, document, allow Phase G to test.
- K-ATP channel parameters are placeholders → calibrate per Wave 2's eventual K-ATP translation; for now, document defaults.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `src/phase_f_metabolic.py` | ATP dynamics module (skeleton at kickoff) |
| `artifacts/metabolic/metabolic_layer_module.py` | Brian2 module ready for Phase G import |
| `artifacts/metabolic/wt_baseline.npz` | WT ATP, K-ATP traces |
| `artifacts/metabolic/gas1_baseline.npz` | gas-1 baseline traces |
| `artifacts/metabolic/mev1_baseline.npz` | mev-1 baseline traces |
| `artifacts/metabolic/atp2_baseline.npz` | atp-2 baseline traces |
| `artifacts/metabolic/wt_vs_gas1_halothane.npz` | EC50 comparison |
| `artifacts/metabolic/calibration_report.md` | Gate F.1 evaluation |
| `artifacts/metabolic/phase_f_completion.md` | end-of-block report |

---

## 8. Falsifiability checks

The phase's premise: **"A coupled ATP-dynamics + K-ATP-channel metabolic layer reproduces gas-1 hypersensitivity to anesthetics."**

Falsified if:

1. F.1.3 fails: gas-1 EC50 shift not in [1.5×, 4×] at any reasonable parameter set.
2. The metabolic layer requires extreme parameter values (5+ orders of magnitude from textbook) to produce the gas-1 effect.
3. mev-1 hypersensitivity equals or exceeds gas-1 — Complex II is not the dominant target, this contradicts the literature.

Failure of F.1.3 is a major finding: it would suggest the gas-1 hypersensitivity mechanism is not metabolic / K-ATP-mediated and Wave P should explore alternatives (direct Complex I redox sensing, ROS-based, etc.).

---

## 9. Integration points

**Inputs from earlier phases:**

- Phase C `artifacts/occupancy/occupancy_matrix.npz` — Complex I and Complex II occupancies.
- Phase D `artifacts/kinetics/anesthetic_kinetic_shifts.npz` — Complex I shift form and parameters.

**Outputs consumed by:**

- **Phase G** (`src/phase_g_network_runs.py`) — imports metabolic layer for per-cell ATP / K-ATP state.

---

## 10. Citation hygiene declaration

- Morgan & Sedensky 1995 — PMID 7549290. [VERIFIED]
- Kayser 2001 — (PMID lookup needed). [BLOCKING]
- Kayser 2004 follow-up — (PMID lookup needed).
- Kayser 2008 — (PMID lookup needed).
- Falk 2006 — gas-1 effects (review). (PMID lookup needed).
- K-ATP channel mechanism — Nichols 2006, *Nature*; (PMID lookup needed).
- Munro 1990 — neural ATP consumption rates. (PMID lookup needed).

**Pre-flight verification status:** 1 of 7 verified. 6 PMIDs blocking.

---

## 11. Risk register (Phase F)

| Risk | Likelihood | Mitigation |
|---|---|---|
| ATP dynamics parameters poorly constrained | High | Use textbook + per-cell sensitivity analysis; document all defaults |
| gas-1 hypersensitivity does not reproduce | Medium | Pause and re-evaluate mechanism; alternative: direct anesthetic effect on K-ATP, etc. |
| K-ATP channel parameters are placeholders without Wave 2 validation | High | Document; Wave 2 may translate K-ATP later; until then, phenomenological coupling |
| Steady-state ATP convergence numerically unstable | Low | Use semi-implicit ODE solver; clip [ATP] to physiological range |
| Mev-1 / atp-2 mutant data are sparse | Medium | Use them as qualitative consistency checks, not quantitative gates |

---

## 12. Phase F execution plan

1. Pre-flight citation verification (6 PMIDs).
2. Implement metabolic layer module.
3. Smoke-test WT baseline; tune to physiological [ATP] range.
4. Test gas-1 / mev-1 / atp-2 baselines.
5. Apply halothane occupancy at Complex I; measure EC50 shift WT vs gas-1.
6. Compare to Morgan & Sedensky 1995 2-3× shift target.
7. Gate F.1 evaluation; end-of-block report.
