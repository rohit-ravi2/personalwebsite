# Stage D — Wave P × Wave 2 integration scoping

**Date:** 2026-04-28 overnight Stage D
**Cross-thread context:** Session 1 is running an overnight Phase δ work block expanding the Wave 2 cellular layer (literature scoping → cellular validation → network integration → touch cascade verdict). This document scopes how Wave P's Phase G perturbation predictions land on the eventual Phase δ-expanded substrate.

---

## Substrate landscape (current + projected)

### Wave 2 brain (current state)

- `LIFBrain` — Brian2 LIF over Cook 2019 connectome (300 neurons), 1 free scalar (W_syn)
- `GradedBrain` / `graded_brain_h_kca.py` — graded-potential brain with H + KCa channels for Wave 2 production-grade cells
- Existing production-grade cells (Wave 2 alpha): AVAL, AIY, RIM (3 cells with Brian2-validated biophysics, calibrated against intracellular recordings)
- Connectome: Cook 2019 hermaphrodite × Loer & Rand 2022 NT signs

### Phase δ-expanded substrate (projected outcome of Session 1's overnight)

Probable additions (depending on Phase δ overnight outcome):
- 3-5 additional production-grade cells (e.g., AVB, AIB, RIME, ALM, AVM)
- Touch cascade closure: ALM/AVM → AIB → AVA reversal pathway with biophysical verification
- Possible: PVD, AVD, AVE for fuller reversal-circuit coverage

### Brain options for Phase G integration

| Substrate | When to use | Phase G hook approach |
|---|---|---|
| Wave 2 LIFBrain (300 neurons) | Default Phase G test bed; full connectome coverage | apply_to_brain() modifies W_chem + I_ext per AnestheticPerturbation class |
| Wave 2 GradedBrain | Higher-fidelity dynamics for cells that require continuous voltage | Same hook approach; graded vs spiking is transparent to perturbation manager |
| Phase δ-expanded production cells | Cell-specific predictions (e.g., gas-1 effect on AVA pacemaker) | Per-cell channel inventory required; perturbation hooks per channel type |
| Minimal Brian2 demo (this overnight Stage B) | Architecture validation only | Already implemented; use for unit-level testing of perturbation logic |

---

## Phase G × Phase δ integration matrix

For each anesthetic mechanism class in `wave2_overlay_v2.json`, identify which production-grade Phase δ cells (existing + projected) receive the perturbation:

| mechanism_class | Wave P targets (binding) | Phase δ cells receiving perturbation | Phase G hook |
|---|---|---|---|
| `gaba_potentiation` | UNC-49 | AVAL (UNC-49 expressing per CeNGEN), AIY, RIM, AVB | enhance inhibitory weights from GABAergic presynaptic neurons onto these cells |
| `glucl_potentiation` | AVR-14, AVR-15, GLC-1/2/3/4 | RIM (AVR-15), AIB (AVR-14) — motor neurons heavily expressed but not in command set | enhance Glu→post-cell weights where postsynaptic expresses GluCl |
| `nachr_antagonism` | ACR-16, UNC-29/38/63 | AVAL, AVDL/R (ACR-16), body wall muscle (UNC-29/38/63 — outside command set) | reduce ACh→post-cell weights where postsynaptic expresses nAChR |
| `k2p_potentiation` | TWK-18, TWK-29 | AVAL/R, AVDL/R, AVBL/R (TWK-18 expressing per Singaram 2011) | add hyperpolarizing K-leak current to expressing cells |
| `complex_i_block` | NUO-1, NDUFS2, NDUFV2 | ALL cells (mitochondrial, universal) | add small uniform K-ATP-coupled hyperpolarizing current |
| `complex_ii_block` | MEV-1 | ALL cells | additional mitochondrial perturbation; v1 lumps with complex_i_block |
| `nca_block` | NCA-1, UNC-79, UNC-80 | NCA-expressing cells (broadly: pacemaker neurons, esp. command interneurons) | reduce baseline depolarizing leak (NCA = Na leak channel) |
| `snare_cooperativity` | UNC-64, RIC-4, SNB-1 | ALL chemical synapses (presynaptic) | scale W_syn globally by Phase E fold-change |

### Touch cascade × anesthesia (highest-value cross-thread integration)

If Phase δ closes the touch cascade biophysically (ALM/AVM → AIB → AVA reversal), Wave P predicts the following anesthetic effects on touch reversal:

- **Halothane suppression of touch reversal:** all 8 mechanism classes engage at clinical EC50. Combined hyperpolarization + reduced excitatory drive + reduced release probability → AVA reversal command suppressed → touch elicits no reversal. **Predicted: complete touch reversal suppression at 1× clinical EC50, partial suppression at 0.5×.**

- **Etomidate suppression:** GABA-A potentiation dominant; minimal Complex I, K2P engagement. AVA receives enhanced inhibition from GABAergic synapses but retains intrinsic excitability. **Predicted: partial touch reversal suppression at 1× clinical EC50, with retained reflex amplitude vs halothane's complete suppression.**

- **Differential anesthetic phenotype prediction:** halothane and etomidate produce qualitatively different patterns. Halothane: complete loss of touch reversal. Etomidate: reduced amplitude but preserved cascade onset. This is testable against published electrophysiology in mammalian touch cascade homologs (Crosson et al. on volatiles vs Belelli on etomidate-specific GABA potentiation patterns).

### Mutant phenotype predictions (testable via Phase G + Phase δ substrate variants)

| Variant | Setup | Predicted phenotype | Anchor |
|---|---|---|---|
| gas-1(fc21) mutant | Set GAS1_COMPLEX_I_FACTOR=0.4 across all cells in Phase δ substrate | gas-1 mutant immobilizes at 0.4× clinical halothane (lower EC50 than WT) | Morgan & Sedensky 1994 PMID 7943840 |
| twk-18(cn110) GoF | Increase K2P leak conductance × 2 on expressing cells | K2P-gf hypersensitive: immobilizes at 0.7× WT halothane EC50 | Singaram 2011 PMID 22137475 (corrected per CP6) |
| sup-9(n180) LoF | Decrease K2P leak conductance × 0.3 on expressing cells | Modestly resistant: 1.1× WT halothane EC50 | Singaram 2011 |
| unc-13(s69) hypomorph | Reduce W_syn ~80% globally | Already low release-p baseline; halothane has less margin to suppress; should appear hypersensitive to halothane | Nguyen 1995 PMID 7647836 |

---

## Phase G test plan against expanded substrate

Bounded execution sequence post-overnight, in priority order:

### Test 1 — Halothane vs etomidate touch cascade discrimination (HIGHEST VALUE)

**Substrate:** Phase δ-expanded brain with touch cascade closure (ALM/AVM → AIB → AVA)
**Setup:** baseline scenario with simulated touch input (transient AVM/ALM activation)
**Procedure:**
1. Run baseline (no anesthetic): observe AVA reversal command activation
2. Run halothane @ 1× clinical EC50: observe predicted complete suppression
3. Run etomidate @ 1× clinical EC50: observe predicted partial suppression with preserved cascade onset
4. Quantify: AVA peak rate, reversal command persistence, cascade onset latency

**Expected outcome (testable vs falsifiable):**
- Halothane: AVA peak rate < 20% of baseline; cascade onset latency unchanged or slowed
- Etomidate: AVA peak rate 40-60% of baseline; cascade onset latency unchanged
- If observed instead: halothane retains > 40% AVA rate, the binding-pipeline + Phase G chain has missed something at the network level — falsification.

### Test 2 — gas-1 hypersensitivity at network level

**Substrate:** Phase δ substrate with gas-1 variant (Complex I scaling × 0.4)
**Setup:** spontaneous scenario, no touch input
**Procedure:**
1. WT halothane dose-response: 0.1×, 0.3×, 1×, 3× → command interneuron suppression
2. gas-1 halothane dose-response: same doses → suppression
3. Compute behavioral EC50 from each (define EC50 as dose at which command activity drops to 50% of pre-anesthetic baseline)
4. Ratio = WT_EC50 / gas1_EC50

**Expected outcome:** ratio in 2-3× band (Morgan anchor), confirming Phase G captures network-level gas-1 hypersensitivity. CP1 noted Phase F is parameter-locked at the threshold layer; Phase G's network-level test is the independent test that bypasses Phase F.

### Test 3 — twk-18 GoF hypersensitivity (CP6-corrected direction)

**Substrate:** Phase δ substrate with K2P leak × 2 on TWK-18 expressing cells
**Setup:** spontaneous + touch scenarios
**Procedure:** dose-response same as Test 2, comparing WT vs twk-18(cn110) substrate
**Expected outcome:** twk-18(cn110) immobilizes at 0.7× WT halothane EC50

### Test 4 — Hexafluoroethane null perturbation (Eger non-immobilizer prediction)

**Substrate:** Phase δ substrate
**Setup:** apply Phase G perturbation with hexafluoroethane "occupancy profile" derived from CP7 negative-control test (occupancies are similar to halothane!)
**Procedure:** dose-response and check whether network-level integration produces null phenotype despite high binding occupancy
**Expected outcome:**
- If network produces full immobilization at 1× clinical: Phase G + binding pipeline reproduces the Eger non-immobilizer puzzle's failure mode (binding alone is insufficient)
- If network produces minimal phenotype despite occupancy: Phase G captures something that distinguishes integrated multi-target engagement profiles in a way that binding alone misses
- Either outcome is informative: documents whether Phase G adds discrimination beyond CP3/CP7 boundary failures

This is the single most important Phase G validation. The Eger puzzle is a known frontier; whether Phase G makes progress is genuinely open.

### Test 5 — Per-anesthetic dose-response curves (extending Stage B halothane)

Extend the dose-response from Stage B's minimal demo to LIFBrain substrate, all 6 anesthetics. Generates a 6 × 8 × 300 = 14,400 (anesthetic × dose × neuron) matrix for downstream analysis.

---

## Cross-thread coordination

### Information needed from Session 1's Phase δ overnight

For Stage D test plan execution, Wave P needs from Phase δ:

1. **Cell roster:** which production-grade cells are available post-Phase-δ (existing + new)
2. **Channel expression matrix:** per-cell channel expression (CeNGEN-derived if possible) so Phase G can localize K2P, nAChR, GABA-A perturbations
3. **Touch cascade closure status:** is ALM/AVM → AIB → AVA biophysically closed? If yes → Test 1 unblocked.
4. **Per-cell biophysics validation status:** which cells have been validated against intracellular recordings → Phase G perturbation effects on these cells are interpretable
5. **API contract for substrate construction:** how to instantiate the Phase δ-expanded brain object that Wave P's `apply_to_brain()` consumes

### Information Wave P provides to Session 1

For Phase δ execution, Wave P provides:

1. `wave2_overlay_v2.json` — kinetic shifts per (anesthetic, target) for downstream consumption
2. `AnestheticPerturbation` class API — drop-in perturbation hook for any Brian2 brain
3. CP6 anchor classification — provides validation phenotypes for differential anesthetic sensitivity
4. CP7 corrected occupancies — feed into per-cell parameter perturbations
5. Phase G smoke test + dose-response infrastructure — reusable test scaffolding

---

## Cross-thread acceptance criteria

For the morning after both overnight runs, the cross-thread integration is ready if:

- ✅ Phase δ produces validated production-grade cell roster (Session 1 deliverable)
- ✅ Touch cascade closure status documented (PASS/FAIL/PARTIAL)
- ✅ Wave P Phase G architecture + smoke test landed (this Stage B deliverable)
- ✅ Wave P case studies documented (this Stage C deliverable)
- ✅ This integration scoping document landed (this Stage D deliverable)
- ✅ Wave P methodology paper case study suite ready for Phase 2 paper integration

If all green: morning brief can declare cross-thread integration is unblocked, and the next bounded work block can execute Tests 1-4 above on the actual Phase δ substrate.

If Phase δ overnight pauses (architectural ambiguity, etc.): Wave P Phase G can still proceed against existing 3-cell production substrate, with the test plan above scaled down.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase δ overnight produces incompatible substrate API | LOW | HIGH | Phase G `apply_to_brain()` is duck-typed; needs only `.names`, `.neurons.I_ext`, `._W_chem_runtime`, `.W_syn` |
| Phase δ touch cascade fails to close | MEDIUM | MEDIUM | Tests 2-5 don't require touch cascade; Test 1 deferred |
| LIFBrain perturbation produces NaN/inf at saturation | MEDIUM | LOW | Existing mitigation in Phase G v1 (clip magnitudes; reject NaN runs) |
| Phase G dose-response doesn't reach 50% suppression at clinical 1× EC50 | HIGH | LOW | Already documented in Stage B as honest finding; behavioral threshold calibration is separate work block |
| Perturbation magnitudes (50 pA per Complex I unit, etc.) are wrong scale for LIFBrain | HIGH | MEDIUM | Calibrate during Test 5 sweep; scale magnitudes uniformly to land halothane @ 1× at 50% AVA suppression |

---

## Standing follow-ups for next work block

1. **CeNGEN expression matrix integration** — replace simplified hand-curated CHANNEL_EXPRESSION dict with full CeNGEN per-cell expression matrix. Sharper localization of K2P, nAChR, GABA-A perturbations.

2. **Phase G v2 calibration against LIFBrain command interneuron baseline** — rather than the minimal demo network, calibrate Phase G perturbation magnitudes so halothane @ 1× clinical EC50 produces ~50% AVA suppression on LIFBrain. This converts Stage B's binding-saturation gap into a behavioral-threshold-calibrated dose-response.

3. **Mutant variant infrastructure** — Phase G needs a clean way to produce gas-1, twk-18(cn110), unc-13(s69) substrate variants. v1 implementation: per-variant scaling factors applied at brain construction time.

4. **Locomotor parameter sweep readout** — beyond aggregate firing rate, compute predicted forward locomotion frequency, reversal frequency, ω-turn frequency from command interneuron activity. Maps Phase G output onto behavioral phenotypes that can be compared against published *C. elegans* anesthesia data.

5. **Methodology paper integration** — the 5 case studies + Phase G + integration scoping forms a coherent methods paper draft. Outline:
   - Section 1: Pre-flight pushback as systematic methodology (case study 5)
   - Section 2: Citation hygiene as load-bearing infrastructure (subset of case study 5)
   - Section 3: Parameter-lock detection (case study 1)
   - Section 4: Calibration ground-truth audit (case study 2)
   - Section 5: Boundary tests as scientific commitments (case study 3)
   - Section 6: Direction-inversion as recurring failure mode (case study 4)
   - Section 7: Worked example — Wave P pharmacology pipeline rigor pass
   - Section 8: Phase G as the next-step network-level test

---

## Summary

Wave P × Wave 2 integration scoping documents the test plan, expected outcomes, and risk register for the next bounded work block. Critical dependencies are clear (touch cascade closure for Test 1; CeNGEN expression matrix for Tests 2-3; per-anesthetic substrate variants for Test 4-5). The scoping prepares the ground for both threads to execute in coordinated fashion the morning after both overnight runs.
