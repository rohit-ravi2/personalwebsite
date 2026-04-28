# Phase E — Markov synaptic transmission with SNARE dynamics

**Phase letter:** E
**Status:** SCAFFOLDED. Not yet executed.
**Predecessor:** Phase C (occupancy matrix). Phase E does not require Gate D.1 — it can run in parallel after Gate C.1 passes.
**Successor:** Phase G (network runs) consumes Phase E's Markov synaptic module.
**Compute:** local CPU ~10 hours.

---

## 1. Goal

Replace the production simulator's deterministic LIF synapse model (`W_chem × spike → conductance pulse`) with a **stochastic Gillespie SSA Markov model** of vesicle release that explicitly models the Ca → SNARE-assembly → fusion-probability → quantal-release → recycle cycle. The Markov model exposes the SNARE-machinery cooperativity coefficient `n` as a tunable parameter, which is the load-bearing kinetic shift for SNARE-targeting anesthetics (van Swinderen 2004).

The phase delivers a Brian2 module that can be plugged into the production simulator's synapse layer at Phase G, with anesthetic effects applied as Phase D-derived shifts to `n`.

---

## 2. Background

### 2.1 SNARE machinery and Ca cooperativity

Synaptic vesicle fusion at the *C. elegans* NMJ (and in mammals) is a stochastic process driven by:

1. Ca influx through presynaptic Ca channels (primarily UNC-2 in worms; CaV2 in mammals).
2. Ca binding to synaptotagmin-1 (SNT-1 in worms) — the Ca sensor.
3. Synaptotagmin triggers SNARE complex completion (UNC-64 syntaxin / RIC-4 SNAP-25 / SNB-1 synaptobrevin).
4. SNARE bundle drives membrane fusion.
5. Vesicle recycles via endocytosis.

The release probability has cooperative dependence on Ca:

```
p_release = (Ca / K_Ca)^n_Ca / (1 + (Ca / K_Ca)^n_Ca)
```

with measured `n_Ca ≈ 3-5` in well-characterized systems (Dodge & Rahamimoff 1967 in vertebrates; van Swinderen 2004 in *C. elegans* implies similar). Halothane (and other volatile anesthetics) reduce `n_Ca` by approximately 1.5 at clinical concentration:

> "Halothane reduces the apparent Ca²⁺-cooperativity of release at the *C. elegans* NMJ from ~3.5 to ~2.0 at 1 vol% atm."

— van Swinderen 2004 (PMID lookup needed)

This is the load-bearing finding for the SNARE-machinery anesthetic mechanism. The deterministic LIF synapse cannot represent this — it has no `n` parameter, only a peak conductance amplitude.

### 2.2 Why Markov / Gillespie

A deterministic Hill-equation synapse can represent average release probability but not:

- Quantal release statistics (mEPSC frequency, amplitude variance).
- Failure modes at low Ca.
- Vesicle pool depletion under high firing.
- Stochastic timing of release events.

A Markov / Gillespie model represents these explicitly. The state space is:

- `[Ca]_pre` — presynaptic Ca concentration (continuous; ODE on top of Gillespie).
- `n_RR` — number of release-ready vesicles (discrete; pool size ~10-50 per synapse).
- `n_DR` — number of docked-but-not-Ca-loaded vesicles.
- `n_RC` — number of recycling vesicles.

Transitions:

- `RR → released` at rate `r_release(Ca, n)` per vesicle (Hill cooperativity).
- `released → RC` at rate `r_endocytose` (~50 ms time constant).
- `RC → DR` at rate `r_redock` (~500 ms).
- `DR → RR` at rate `r_prime` (Ca-dependent; modulated by UNC-13 occupancy, which is anesthetic-relevant).

### 2.3 Anesthetic effects on the Markov model

| Target class | Effect on Markov model |
|---|---|
| SNARE bundle (UNC-64, RIC-4, SNB-1) | Reduce `r_release` peak amplitude (g_max equivalent) |
| UNC-13 (priming) | Reduce `r_prime` |
| UNC-18 | Reduce SNARE-complex formation efficiency (slows transition) |
| SNT-1 (Ca sensor) | Reduce `n_Ca` cooperativity (the van Swinderen 2004 effect) |

The aggregate effect at the synapse level is:

- Lower release probability per vesicle event.
- Lower Ca cooperativity (reduced ability to amplify high-Ca firing into high release).
- Slower recovery from depletion.

### 2.4 Validation anchors

The Phase E module is validated against:

1. **mEPSC frequency at *C. elegans* NMJ in WT**: ~20-50 Hz at rest (Liu, Hu, Wang published; specific paper lookup).
2. **Ca cooperativity n_Ca = 3-5** measured in *C. elegans* and vertebrate NMJ.
3. **unc-13(s69) hypomorph release deficit**: ~80-90% reduction in release (Richmond 1999, *Nature Neuroscience*; PMID 10570485 — verify).
4. **Halothane release-p reduction**: published frog NMJ data scaled to worm (Krasowski & Harrison 1999 review).

---

## 3. Method

### 3.1 Brian2 module structure

```python
# src/phase_e_markov_synapses.py
class MarkovSynapse(brian2.Synapses):
    """
    Stochastic Markov model of vesicle release.
    State variables (per synapse):
      - n_RR (release-ready vesicle count, integer)
      - n_DR (docked, not Ca-loaded, integer)
      - n_RC (recycling, integer)
      - Ca_pre (continuous, ODE)
    Anesthetic-modulated parameters:
      - n_Ca: Hill cooperativity coefficient (default 3.5)
      - g_max: peak release amplitude (default from Wave 2)
      - r_prime: priming rate (UNC-13 modulated)
    """
```

Implementation uses Brian2's built-in stochastic transitions (`'(rate * dt)**0.5*xi'` style) for the Ca ODE, and explicit Gillespie SSA via Brian2 events for the discrete vesicle-state transitions.

### 3.2 Per-pair anesthetic effect application

```python
n_Ca_baseline = 3.5
n_Ca_anesthetic = n_Ca_baseline - 1.5 * occupancy_SNT1
# (or, more accurately, occupancy of the relevant Ca-sensor target)

g_max_anesthetic = g_max_baseline * (1 - 0.3 * occupancy_SNARE_bundle)
r_prime_anesthetic = r_prime_baseline * (1 - 0.5 * occupancy_UNC13)
```

The per-target occupancies come from Phase C; the proportionality constants come from Phase D's literature-direct shifts.

### 3.3 Validation harness

```bash
# Generate WT NMJ spontaneous mEPSC trace
python src/phase_e_markov_synapses.py --validate \
    --scenario spontaneous_mEPSC_WT \
    --duration 60 \
    --output artifacts/markov/mEPSC_WT.npz

# Generate Ca cooperativity calibration
python src/phase_e_markov_synapses.py --validate \
    --scenario Ca_cooperativity_curve \
    --output artifacts/markov/cooperativity_curve.npz

# unc-13 hypomorph
python src/phase_e_markov_synapses.py --validate \
    --scenario unc13_hypomorph \
    --output artifacts/markov/unc13_hypomorph.npz

# Halothane on WT
python src/phase_e_markov_synapses.py --validate \
    --scenario halothane_WT_1xEC50 \
    --occupancy-source artifacts/occupancy/occupancy_matrix.npz \
    --output artifacts/markov/halothane_WT.npz
```

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| Module implementation + smoke tests | local CPU | 4 | $0 |
| Validation runs (4 scenarios × 60 s simulation × ~10 seeds) | local CPU | 4 | $0 |
| Parameter sweep (n_Ca, g_max, r_prime calibration) | local CPU | 2 | $0 |
| **Total Phase E** | | **~10 hours** | **$0** |

Phase E is the most compute-light substantive phase. Most of the work is implementation + validation, not simulation.

---

## 5. Preregistered success criteria (Gate E.1)

1. **E.1.1 — mEPSC frequency at WT:** simulated mEPSC frequency at WT NMJ within 20% of published 20-50 Hz range. Specific target: 30 ± 10 Hz.
2. **E.1.2 — Ca cooperativity:** fitted n_Ca from simulated dose-response curve in [3, 5].
3. **E.1.3 — unc-13 hypomorph reproduction:** simulated unc-13(s69) release reduction within 50% of published 80-90% (so the simulated reduction should be between 40% and 95%).
4. **E.1.4 — Halothane release-p shift:** with halothane occupancy at WT 1× EC50, simulated release probability reduces by ≥ 30% from baseline. Direction-of-effect required; magnitude within 50% of frog NMJ scaled value.

---

## 6. Halting rules

**Pause and surface:**

- mEPSC frequency at WT off by > 5× → fundamental error in Markov model parameters; halt.
- Ca cooperativity exits [2, 7] under any baseline parameter set → cooperativity calibration is structurally broken.
- unc-13 hypomorph reproduces with < 10% release reduction or > 99% reduction → release model is qualitatively wrong.

**Document and continue:**

- mEPSC amplitude variance differs from published → document, flag as a refinement target.
- Halothane shift magnitude off by 2× from frog NMJ scaled prediction → document, use simulated value with explicit uncertainty in Phase G.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `src/phase_e_markov_synapses.py` | Brian2 Markov synapse implementation (skeleton at kickoff; full implementation at Phase E execution) |
| `artifacts/markov/markov_synapse_module.py` | Brian2 module ready for production-simulator import |
| `artifacts/markov/mEPSC_WT.npz` | WT spontaneous mEPSC validation |
| `artifacts/markov/cooperativity_curve.npz` | Ca dose-response curve |
| `artifacts/markov/unc13_hypomorph.npz` | unc-13 mutant validation |
| `artifacts/markov/halothane_WT.npz` | Halothane perturbation validation |
| `artifacts/markov/calibration_report.md` | Gate E.1 evaluation |
| `artifacts/markov/phase_e_completion.md` | end-of-block report |

---

## 8. Falsifiability checks

The phase's premise: **"A Markov SNARE/Gillespie synapse model can reproduce *C. elegans* NMJ baseline release statistics, the Ca cooperativity, and the unc-13 hypomorph deficit, with anesthetic effects applied as parameter shifts derived from Phase C/D."**

Falsified if:

1. Baseline mEPSC frequency cannot be reached for any reasonable parameter set.
2. Ca cooperativity cannot be made consistent with published n_Ca = 3-5.
3. Anesthetic effects on the model produce no effect on release (i.e., the parameter shifts from Phase D produce < 5% effect on release p).

---

## 9. Integration points

**Inputs from earlier phases:**

- Phase C `artifacts/occupancy/occupancy_matrix.npz` — SNARE-target occupancies at relevant concentrations.
- Phase D `artifacts/kinetics/anesthetic_kinetic_shifts.npz` — per-target shift forms and parameters for SNARE machinery.

**Outputs consumed by:**

- **Phase G** (`src/phase_g_network_runs.py`) — imports the Markov synapse module to replace LIF synapses for the network simulation.

---

## 10. Citation hygiene declaration

- van Swinderen 2004, halothane Ca cooperativity in *C. elegans* — (PMID lookup needed). [BLOCKING]
- Dodge & Rahamimoff 1967, vertebrate NMJ Ca cooperativity, *J Physiol* — (PMID lookup needed; classical paper).
- Richmond 1999, unc-13 release deficit, *Nature Neurosci* 2:959-964 — PMID 10570485. [VERIFIED]
- Krasowski & Harrison 1999, anesthetic mechanism review, *Cell Mol Life Sci* — (PMID lookup needed).
- Liu, Hu, Wang, *C. elegans* NMJ mEPSC — (specific paper + PMID needed).
- Stimberg 2019, Brian2 — DOI `10.7554/eLife.47314`. [VERIFIED]
- Gillespie 1977, SSA — DOI `10.1021/j100540a008`. [VERIFIED]

**Pre-flight verification status:** 3 of 7 verified. 4 PMIDs blocking.

---

## 11. Risk register (Phase E)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Brian2 doesn't support efficient Gillespie SSA at network scale | Medium | Use Brian2's hybrid ODE+event approach; or implement custom CUDA Gillespie (Wave 3 work) |
| Vesicle pool size hard to calibrate without direct *C. elegans* data | Medium | Use mammalian default (10-50 RR vesicles); document |
| n_Ca baseline value disputed in literature | Low | Use mid-range (3.5) as default; sensitivity analysis at n=3, 4, 5 |
| Halothane release shift magnitude uncertain | Medium | Use frog NMJ scaled with explicit factor-of-2 uncertainty band; document |
| Markov model couples poorly to LIF spiking layer | Medium | Test integration in Phase G with explicit cross-checking; fall back to deterministic Hill if integration fails |

---

## 12. Phase E execution plan

1. Pre-flight citation verification (4 PMIDs).
2. Implement Brian2 Markov synapse module (skeleton already in `src/`).
3. Validate mEPSC frequency at WT.
4. Calibrate Ca cooperativity to n = 3.5.
5. Validate unc-13 hypomorph.
6. Validate halothane shift.
7. Gate E.1 evaluation; end-of-block report.
