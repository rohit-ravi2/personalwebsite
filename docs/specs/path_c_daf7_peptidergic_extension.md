# Path C — DAF-7 / DAF-1 / RIS peptidergic extension: engineering spec

**Status:** Draft, session 3 work block, 2026-04-25.
**Scope:** Extend the modulator infrastructure with a DAF-7 peptide family to provide a satiety-quiescence pathway that elevates RIS during food sensing under per-edge mode.
**Gating:** Implementation depends on session 2's T0 closure commit landing.

---

## 0. Executive summary

The food → RIS quiescence pathway is structurally absent from the current simulator. The connectome topology investigation in this thread established that no direct ASI/ASJ/ADF → RIS edges exist; nine 2-hop pathways exist via cholinergic and gap intermediates, all with sign-preserving topology under per-edge mode. The food-quiescence breakdown therefore is **not a sign-convention bug** — it is **missing peptidergic infrastructure**, specifically the DAF-7 (TGF-β-class) signal that ASI is known to release in response to nutrient sensing.

This spec proposes adding DAF-7 as a 10th modulator in the existing modulator_tables.npz infrastructure. The implementation reuses the rate-coupled peptide release machinery already proven for the 9-modulator v3 set; the new content is (a) DAF-7 release function from ASI, (b) DAF-1/DAF-4 receptor weights from CeNGEN expression, (c) a verification protocol that checks whether the resulting downstream propagation produces RIS elevation during food.

**Critical caveat surfaced during inventory work:** CeNGEN L4 single-cell expression data does NOT support the literature claim that RIM/RIC express DAF-1 receptors. Both RIM (daf-1 = 0.019, daf-4 = 0.012) and RIC (daf-1 = 0.007, daf-4 = 0.012) are below the CeNGEN expression floor for these receptors. The top daf-1 expressers per CeNGEN are I3, ASG, AIM, ADF, I5, HSN, PVT, PDB. **The Davis 2021 / You 2008 narrative places DAF-1 in RIM/RIC; CeNGEN places it elsewhere.** This is the same class of data-vs-literature mismatch already documented for NLP-22/RIA in the project's citation-correction memory. The spec proposes a two-tier modeling approach that handles both possibilities.

---

## 1. DAF-7 as a peptide family in the modulator infrastructure

### 1.1 Schema fit

The existing modulator_tables.npz uses 4 keys per modulator:

```
releasers_<MOD>           : (300,) bool   — neurons above synthesis-gene threshold
releaser_weights_<MOD>    : (300,) float  — synthesis expression for continuous release
target_weights_<MOD>      : (300,) float  — signed sum (receptor expression × per-receptor sign)
tau_<MOD>                 : scalar float  — concentration decay timescale (s)
```

Adding DAF-7 means extending `MODULATORS` dict in `scripts/brain/build_modulator_tables.py` with one entry, then re-running the build script to emit a new `modulator_tables.npz`. The existing `modulation_layer.ModulationLayer` consumes the new modulator transparently (loop iterates over the modulators list, no per-modulator special-casing).

### 1.2 MODULATORS entry — proposed

```python
"DAF-7": {
    "synthesis_gene": "daf-7",
    "receptors": {"daf-1": +1, "daf-4": +1},   # type-I + type-II TGF-β receptors;
                                                # heterodimer required for signaling.
                                                # Sign +1 = excitatory slow current
                                                # in target cell (modeling choice; see §3).
    "tau_s": 60.0,
    "note": "Released by ASI on nutrient sensing → activates DAF-1/DAF-4 receptors → "
            "downstream excitation (You 2008 PMID 18334214 [VERIFY]; Greer 2008; "
            "Davis 2021 review [VERIFY]). Mediates satiety quiescence.",
},
```

### 1.3 Release function — uses existing rate-coupled mechanism

The current release function (modulation_layer.py:246-250) is purely rate-coupled:

```
release = releaser_weights @ spike_counts
concentration_t+1 = min(decay × concentration_t + release_gain × release, cap)
```

For DAF-7: release_gain = 0.02 (default), releaser_weight derived from CeNGEN daf-7 expression. ASI's daf-7 expression in CeNGEN is **38.9** (overwhelmingly dominant; ASG = 8.9 distant second, ASJ = 3.7 third). ASI fires at ~20 Hz during food per the session 3 RIS scan. Predicted DAF-7 steady-state concentration: `0.02 × normalized_releaser_weight × 20 Hz × 60s = 24` units. With saturation cap at 10, DAF-7 will saturate at the cap during food — appropriate for a tonic hormonal signal.

**Pros of reusing the rate-coupled mechanism:** zero new infrastructure; per-modulator parameters carry the biology. **Cons:** TGF-β release in real biology is regulated by transcription, not spike-coupled vesicle release. The simplification treats DAF-7 as if it were a fast neuropeptide, which is an abstraction. Documented as such; behavioral phenomenology is still produced as long as ASI firing during food correlates with the desired downstream effect.

### 1.4 Diffusion length scale

DIFFUSION_LENGTH_UM in modulation_layer.py currently has:
- Peptides (FLP family, NLP, PDF): 400-700 µm
- Monoamines: 150-250 µm

DAF-7 is hormone-like (TGF-β-class; secreted, long-range, often endocrine). Should be the longest in the table. **Proposed: 1000 µm.** Rationale: DAF-7 has been demonstrated to act on neurons beyond ASI's immediate neighborhood (Greer 2008 showed DAF-7 affects feeding behavior via remote targets). Actual diffusion in the C. elegans pseudocoelom is largely unbounded for hormonal peptides; 1000 µm captures "essentially whole-body" without making the modeling pathological.

### 1.5 Decay constant tau_s

Proposed: **tau_s = 60.0 seconds.** Matches the existing INS family choice in v4 (which is also TGF-β-class signaling via DAF-2). FLP/monoamines use 4-30 s tau because they're synaptic; DAF-7 is hormonal. Actual TGF-β half-life in vivo is hours-to-days because of transcriptional persistence; the 60s tau models the membrane-level effective timescale, not the molecular half-life. Same compromise the v4 INS family makes.

### 1.6 Storage — no schema change

DAF-7 fits cleanly into the existing 4-key-per-modulator format. No new TGF-β-class abstraction required. The only tension is that DAF-7's "receptor effect" is mechanistically transcriptional, not ionotropic — but the membrane-level abstraction (slow current proportional to concentration × receptor weight) is exactly what the existing infrastructure provides for all peptides.

---

## 2. DAF-1 / DAF-4 receptor representation

### 2.1 The CeNGEN-vs-literature conflict

**Literature claim (Davis 2021, You 2008, Greer 2008):** DAF-1 is expressed in RIM and RIC, mediating DAF-7's effect on RIS activation.

**CeNGEN L4 single-cell expression (verified during inventory):**

| neuron | daf-1 | daf-4 | top-10 ranking |
|---|---:|---:|---|
| RIM | 0.019 | 0.012 | NO (not in top 10 for either) |
| RIC | 0.007 | 0.012 | NO |
| RIS | 0.000 | 0.018 | NO |
| **I3** | **0.100** | 0.072 | daf-1 #1 |
| **ASG** | **0.077** | 0.050 | daf-1 #2 |
| **AIM** | **0.077** | 0.027 | daf-1 #3 |
| **ADF** | **0.075** | 0.101 | daf-1 #4 |
| **I5** | **0.072** | 0.117 | daf-1 #5 |
| **HSN** | **0.072** | 0.066 | daf-1 #6 |
| **PVT** | **0.068** | 0.114 | daf-1 #7 |

The expression magnitudes are uniformly low (max 0.10), an order of magnitude below the typical CeNGEN-resolved-expression floor of ~1-5 for confident expression. This is consistent with TGF-β receptor expression being low-abundance / regulatory in nature. **It does not, however, support a strong "RIM/RIC are the DAF-1 targets" narrative.**

### 2.2 Modeling choices — proposed two-tier approach

**Tier 1 (CeNGEN-grounded — default, build first):**
- Use actual CeNGEN daf-1 + daf-4 expression to compute target_weights via the existing `target_weights = Σ_r (receptor_expression × sign_r)` formula.
- Top targets will be I3, ASG, AIM, ADF, I5, HSN, PVT (all low-magnitude).
- Tier 1 tests whether the **data-driven** propagation produces RIS elevation through the existing chemical/gap connectivity.
- If yes: project remains CeNGEN-consistent and RIS-elevation-via-DAF-7 is established.

**Tier 2 (literature-grounded receptor override — fallback if Tier 1 fails):**
- If Tier 1 sweep produces no detectable RIS elevation, add a documented receptor-expression override: artificially set RIM and RIC daf-1 weights to the literature-supported levels (e.g., match the CeNGEN top-10 magnitudes ~0.08-0.10).
- Rationale: project pattern already uses DOCUMENTED_SIGN_EXCEPTIONS for cases where literature evidence trumps CeNGEN data (the AIY → AIZ override session 2 is implementing). Same precedent applies for receptor expression where CeNGEN may be below detection floor.
- Override would be a small static dict in build_modulator_tables.py: `DAF1_LITERATURE_OVERRIDES = {"RIML": 0.08, "RIMR": 0.08, "RICL": 0.08, "RICR": 0.08}` — applied as a post-CeNGEN augmentation.
- **Cite Davis 2021, You 2008 as the override basis. PMID verification required before commit.**

### 2.3 DAF-1 sign — modeling choice

Sign +1 (excitatory) is proposed because:
- Davis 2021 narrative is "DAF-1 binding → RIM/RIC sustained activation → RIS activation." Sustained activation = depolarizing effect.
- If sign were -1, DAF-1 would inhibit RIM/RIC, which doesn't match the satiety-quiescence framing.
- TGF-β at the membrane level has heterogeneous effects across cell types; for the satiety-quiescence application, +1 captures the documented behavioral consequence.

Documented as a modeling choice; if Tier 2 override is needed and the +1 sign produces the wrong behavioral direction, sign can be flipped to -1 with the rationale that DAF-1 might inhibit RIM (which is itself inhibitory to RIS), thus disinhibiting RIS. Both signs are mechanistically coherent narratives; only the verification protocol can distinguish.

---

## 3. DAF-1 → RIS activation mechanism

The molecular pathway from DAF-1 binding to RIS firing is **not fully characterized in the literature**. Davis 2021 explicitly states "presumably involves RIS activation." The spec must make a modeling choice and document the rationale.

### 3.1 Two plausible mechanisms

**(a) Indirect via existing synaptic connectivity (proposed default):**
- DAF-7 → DAF-1 binding produces sustained depolarizing current in DAF-1-expressing cells (RIM/RIC under Tier 2 override, or I3/ASG/AIM/ADF/I5/HSN/PVT under Tier 1).
- These cells then propagate to RIS through the existing chemical and gap junction synapses.
- For RIM specifically: RIM has chem edges to RIS (W=2 each from RIML/RIMR, currently signed -1 due to TA-primary). Sustained RIM activation would *suppress* RIS via TA inhibition — the OPPOSITE of the intended quiescence pathway. **This is a problem with the literature-grounded RIM-as-DAF-1-target framing**: if RIM is tyraminergic-inhibitory to RIS, activating RIM via DAF-7 would inhibit RIS, not activate it.
- For RIC: RIC's outgoing chem edges to RIS — checking connectome — none direct (RIC has only OA outputs at sign 0). RIC's effect on RIS would be entirely through gap junctions or indirect routes.
- **This raises the possibility that Maluck 2020's RIM-glutamate-co-release mechanism is load-bearing: RIM may activate RIS via Glu (excitatory) co-transmission, not via the canonical TA inhibition.** See §6 for the RIM secondary-NT correction.

**(b) Direct peptidergic activation of RIS via a separate downstream peptide:**
- ALA expresses FLP-13; FLP-13 has receptors on RIS (DMSR family).
- Hypothesis: DAF-7 → ALA (which would need to be added as a DAF-1-expressing target) → FLP-13 release → RIS activation.
- ALA's daf-1 expression in CeNGEN is **0.011** — also below floor. Would require Tier 2 override.

### 3.2 Recommended modeling choice

**Default mechanism (a) with the RIM-Glu co-release nuance as a critical sub-component.** Here's the chain:

1. ASI fires during food sensing (already operational in current simulator at ~20 Hz).
2. DAF-7 release scales with ASI firing.
3. DAF-1/DAF-4 expressing targets receive sustained depolarizing modulation current.
4. **Under Tier 2 override (RIM/RIC included):** RIM's sustained activation drives Glu co-release to RIS — this requires the RIM Glu-co-release pathway to be active in the simulator. See §6.
5. **Under Tier 1 (CeNGEN only):** ADF/ASG/AIM excitation propagates through 2-hop chemical pathways to RIS via RIBL/RIBR (W=5 ACh edges to RIS, mode-independent excitatory).

The Tier 1 ADF/RIBL pathway is mechanistically the most defensible CeNGEN-grounded route to RIS, because:
- ADF expresses daf-1 (0.075, in top-10)
- ADF has direct chem edges to RIBL (W=4)
- RIBL has direct chem edge to RIS (W=5, +1 cholinergic)
- ADF also fires at ~23 Hz during food (sensor activity already validated)

The combined chain ASI fires → DAF-7 → ADF gets DAF-1 boost → ADF fires more → RIBL gets ADF cholinergic input → RIBL excites RIS — uses entirely already-present chemical connectivity, with only the new DAF-1 modulation step as the novel infrastructure.

### 3.3 Verification of mechanism — built into Tier 1 sweep

The verification protocol (§5) measures:
- DAF-7 concentration (should rise during food)
- ADF firing rate (should elevate above baseline given sustained DAF-1 modulation)
- RIBL firing rate (should elevate from ADF input)
- RIS firing rate (the question)

If RIS elevates: mechanism (a) works through the CeNGEN-grounded ADF/RIBL pathway.
If RIBL elevates but RIS doesn't: mechanism (a) fails at the last step (RIBL→RIS edge insufficient).
If ADF elevates but RIBL doesn't: mechanism (a) fails at the propagation step.
If ADF doesn't elevate: DAF-1 modulation strength too weak.

Each failure mode points to a specific tunable parameter or escalation to Tier 2.

---

## 4. Integration with FLP-11 / RIS / quiescence pathway

### 4.1 The existing FLP-11 pathway (verified during inventory)

- RIS has the highest flp-11 expression in CeNGEN (985.196 — orders of magnitude above any other neuron). Confirmed RIS is the FLP-11 releaser.
- RIS expresses unc-25 (GAD, 15.86) and unc-47 (vGAT, 1.66) — confirms GABAergic identity.
- RIS does NOT express flp-13 (0.000); ALA does (verified in earlier C. elegans investigations).
- FLP-11 receptors per existing MODULATORS dict: NPR-1, NPR-22, DMSR-1, DMSR-7, NPR-11 (all sign -1 = inhibitory).
- The release function for FLP-11 is identical to all other modulators: rate-coupled to RIS firing.

### 4.2 DAF-7 → RIS → FLP-11 chain — already plumbed

If DAF-7 elevates RIS firing rate, FLP-11 release will scale linearly per the existing infrastructure (verified in earlier session 1 / 2 work: peptide release is purely rate-coupled, no thresholds). Therefore:
- DAF-7 saturates → RIS fires at elevated rate → FLP-11 concentration rises proportionally → systemic inhibition via NPR-1/NPR-22/DMSR-1 receptors.
- This produces behavioral quiescence as the existing modulator infrastructure already does for stimulated FLP-11 release.

**No new code is required for the RIS → FLP-11 → systemic quiescence step.** The DAF-7 extension only needs to provide the upstream DAF-7 → RIS activation; downstream is already operational.

### 4.3 Predicted full chain

| step | mechanism | infrastructure |
|---|---|---|
| 1. ASI fires during food | already operational (sensor cascade) | existing |
| 2. DAF-7 release scales with ASI rate | new — DAF-7 added to modulator_tables | NEW |
| 3. DAF-1/DAF-4-expressing cells receive slow excitation | new — target_weights from CeNGEN | NEW |
| 4. Targets (ADF/ASG/AIM/etc) propagate to RIS via chem+gap | already operational | existing |
| 5. RIS firing rate elevates | emergent from steps 2-4 | emergent |
| 6. FLP-11 release scales with RIS rate | already operational | existing |
| 7. FLP-11 receptors mediate systemic inhibition | already operational | existing |
| 8. Behavioral quiescence emerges | already operational (FSM reads classifier) | existing |

---

## 5. Verification protocol

### 5.1 Unit-level verification (run after build, before sweep)

**U1. DAF-7 release function:** Construct ModulationLayer with new tables; inject 10 spikes/step into ASI; confirm DAF-7 concentration rises by `0.02 × releaser_weight[ASI] × 10` per step. Match expected formula.

**U2. DAF-1 receptor weights:** Load tables; for each DAF-1-expressing cell (per CeNGEN top-10), confirm `target_weights_DAF-7[cell]` is non-zero and proportional to (daf-1_expr + daf-4_expr).

**U3. Modulation current to target cell:** With DAF-7 concentration set to cap (10), measure I_mod assigned to ADF (top CeNGEN target). Should equal `concentration × target_weight × mod_strength_pa` ≈ 10 × 0.01 × 5 = 0.5 pA. Modest but detectable.

**U4. Brain integration:** Build LIFBrain + ModulationLayer + run 30s with ASI stimulated; confirm DAF-7 concentration rises over time and approaches saturation. Confirm no Brian2 errors.

### 5.2 Network-level verification sweep

**Setup:** per-edge mode + DOCUMENTED_SIGN_EXCEPTIONS (session 2's T0 closure) + new DAF-7 modulator_tables. Food scenario, n=10 × 60s.

**Measurement targets:**
1. RIS firing rate during settled-food window (t=20-45s)
2. FLP-11 concentration during settled-food
3. Behavioral state proportions (specifically QUIESCENT fraction)
4. ADF, RIBL firing rates (mechanism check — does DAF-7 elevate the proximal chain?)
5. DAF-7 concentration trajectory (sanity check)

### 5.3 Pre-specified outcomes

**Outcome ε (target — ship):** RIS food firing ≥ 5 Hz mean across seeds. FLP-11 concentration > 1 unit (vs. ~0.01 currently). Behavioral state shows ≥ 5% QUIESCENT fraction. **This is the success case.** Mechanism is established via CeNGEN-grounded Tier 1 pathway; commit and report.

**Outcome ζ (partial — calibrate):** RIS food firing elevates but to sub-biological level (1-3 Hz). DAF-7 mechanism propagates through some chain; magnitudes need parameter tuning (release_gain, mod_strength_pa, or target_weights scaling). Document the gap; commit Tier 1 implementation; flag calibration as follow-on work block.

**Outcome η (no elevation — diagnose):** RIS food firing unchanged (≤ 1.5 Hz, statistically indistinguishable from current 0.6 Hz baseline). Tier 1 pathway insufficient. Diagnose at unit level: which step fails (DAF-7 release? Target activation? Propagation to RIS?). If diagnosis points to insufficient receptor expression, escalate to Tier 2 (literature-grounded RIM/RIC override). If diagnosis points to weak propagation, calibration in Tier 1.

**Outcome θ (RIS elevates but other phenotypes break — investigate):** RIS food firing elevates, but other audit-validated phenotypes regress (touch dREV, AVA-Chalfie, Mode-1/2/3 modulator phenotypes). DAF-7 extension is destabilizing the broader network. Don't commit; bisect to identify which target neurons are over-driven; restrict target weights or escalate to dedicated calibration block.

### 5.4 Verification ordering

1. U1-U4 unit tests (~10 min wall) — must pass before sweep.
2. Network sweep (n=10 × 60s × 1 scenario × 2 conditions [DAF-7 on/off as ablation control] = ~60 min wall sequential, less with parallelism).
3. Compare phenotype regressions on touch + AVA-ablation scenarios using existing per-edge baseline CSVs as reference.
4. Classify outcome and decide commit vs. defer.

---

## 6. RIM secondary NT correction

### 6.1 Current state (verified during inventory)

- `connectome.npz` already has `nt_secondary='Glu'` for both RIML (idx 197) and RIMR (idx 198). The data IS already corrected to match Loer & Rand 2022. The user's note about "Tyramine-primary with no secondary" reflects an older state of the data.
- However: the brain construction in `lif_brain.py` does NOT consult `nt_secondary`. It only uses `nt_primary` → `sign` (per-presynaptic-neuron in legacy mode) or `post_sign_glu` (in per-edge mode).

### 6.2 Implication for the spec

For RIM to actually release Glu in the simulator (per Maluck 2020 RIM→RIS Glu activation), the brain construction would need to handle dual-NT releasers. **This is a separate engineering question from the DAF-7 extension proper.**

Two options:
**(a) Defer RIM-Glu co-release to a follow-on work block.** The DAF-7 extension proceeds with RIM-as-TA-only (the current behavior). Tier 1 mechanism via ADF/RIBL is the primary path; RIM is not load-bearing in Tier 1.

**(b) Add RIM-Glu co-release as part of the DAF-7 extension.** Required if Tier 2 (literature override on RIM DAF-1) is needed AND the mechanism depends on RIM activating RIS via Glu rather than inhibiting via TA. Engineering: extend `build_connectome_matrix.py` to emit additional Glu-classified outgoing edges from RIM, signed per RIS's `post_sign_glu` (which is -1, i.e., inhibitory!) — this would mean RIM Glu output ALSO inhibits RIS. The Maluck 2020 claim of RIM-Glu activating RIS would only be reproduced if RIS expressed an iGluR (it doesn't, per CeNGEN: iGluR/GluCl = 0.21/2.25).

### 6.3 Recommendation

**Choose option (a) for this work block.** The DAF-7 extension proceeds with current RIM/RIS pharmacology. RIM Glu co-release is a separate scope that would require either (i) extending the per-edge sign convention to handle non-canonical iGluR splice variants on RIS, or (ii) accepting the data-vs-literature conflict and adding RIM→RIS as a custom non-CeNGEN-grounded edge.

Either route is its own ~1-day investigation. Out of overnight scope.

---

## 7. Specific file changes

### 7.1 New / modified files

| file | change | risk |
|---|---|---|
| `scripts/brain/build_modulator_tables.py` | Add DAF-7 entry to MODULATORS dict (§1.2). If Tier 2 needed: add DAF1_LITERATURE_OVERRIDES dict + apply post-CeNGEN. | low — additive only, doesn't touch existing 9 modulators |
| `scripts/brain/modulation_layer.py` | Add `"DAF-7": 1000.0` to `DIFFUSION_LENGTH_UM`. | trivial |
| `scripts/brain/artifacts/modulator_tables.npz` | Regenerated by build script. Backup the existing file before regen. | low — backed up |
| `scripts/brain/phase0_daf7_unit_tests.py` | NEW — unit tests U1-U4. | low — test-only file |
| `scripts/brain/phase0_daf7_food_sweep.py` | NEW — network-level food sweep harness. | low — copies existing scenario-scan pattern |
| `scripts/brain/phase0_daf7_food_sweep_compare.py` | NEW — compare DAF-7-on vs DAF-7-off (ablation control) phenotypes. | low |

**No changes required to:**
- `lif_brain.py` (modulator infrastructure handles new modulator transparently)
- `graded_brain.py` (same — if it uses ModulationLayer)
- `closed_loop_env.py` (same — receives ModulationLayer attached to brain)
- `behavioral_fsm.py` (no FSM changes; quiescence_onset classifier already exists and would trigger if RIS-mediated FLP-11 broadcast produces the right rate pattern)
- `neural_classifier_bank.py` (no retraining needed; the existing classifier was trained on Atanas data which presumably already contained satiety-quiescence behavior in the food/satiety conditions)

### 7.2 Build sequence

1. Backup current `modulator_tables.npz` → `modulator_tables.npz.pre-daf7.bak`.
2. Edit `build_modulator_tables.py` MODULATORS dict to add DAF-7.
3. Edit `modulation_layer.py` DIFFUSION_LENGTH_UM dict.
4. Run build script; emit new `modulator_tables.npz` with 10 modulators.
5. Run `phase0_daf7_unit_tests.py`. Must all pass before proceeding.
6. Run `phase0_daf7_food_sweep.py` (per-edge mode + DOCUMENTED_SIGN_EXCEPTIONS).
7. Run `phase0_daf7_food_sweep_compare.py` to assess DAF-7 ablation effect.
8. Classify outcome ε/ζ/η/θ.
9. If ε or ζ: commit. If η: diagnose, escalate to Tier 2 if appropriate. If θ: don't commit, investigate.

### 7.3 Rollback plan

If unit tests pass but sweep produces θ (other phenotypes break), rollback is single-command:
```
mv scripts/brain/artifacts/modulator_tables.npz.pre-daf7.bak \
   scripts/brain/artifacts/modulator_tables.npz
```
Plus revert the ~5-line build_modulator_tables.py and modulation_layer.py changes via git.

---

## 8. Effort estimate

### 8.1 Phase 1 (this spec) — completed in this session

~3 hours actual: codebase inventory + CeNGEN expression checks + spec drafting.

### 8.2 Phase 2 (implementation, gated on T0)

| sub-task | estimate | risk |
|---|---:|---|
| Edit MODULATORS + DIFFUSION_LENGTH + run build script | 1 h | low |
| Write unit tests U1-U4 | 1.5 h | low |
| Write network sweep harness (mostly copy from phase0_ris_scenario_scan.py) | 1 h | low |
| Run unit tests | 0.5 h wall | low |
| Run network sweep (n=10 × 60s × food × per-edge × DAF-7 on/off) | 1.5-2 h wall | low |
| Analyze, classify outcome | 1 h | medium (requires interpretation) |
| If outcome ζ: tune one or two parameters, re-sweep | 1-2 h conditional | medium |
| If outcome η: escalate to Tier 2 (add literature override, re-build, re-sweep) | 2-3 h conditional | medium |
| If outcome θ: bisect target weights, identify destabilizing target, re-build | 2-4 h conditional | high |
| Commit + write report | 0.5 h | low |

**Best-case total (outcome ε):** 6-7 hours.
**Likely (outcome ζ or escalation to Tier 2):** 8-12 hours.
**Worst case (outcome θ):** beyond overnight scope; stop and report at clean checkpoint.

### 8.3 Honest assessment

**This will probably not all complete tonight.** The spec is the load-bearing deliverable. Implementation should aim to:
- Land the build_modulator_tables.py change + emit new tables (~1 h).
- Run unit tests (~30 min wall).
- Kick off the network sweep in background (~1.5-2 h wall).
- Analyze the sweep result and classify outcome.
- Commit if outcome ε or ζ (the "implementation got the right shape, magnitudes need calibration" case is committable as Tier 1 baseline).
- If outcome η or θ: do not push to commit overnight; document for follow-on work block.

If session 2's T0 closure has not landed by the time spec is done, **stop after spec phase**. Don't implement on an unstable base.

### 8.4 What success looks like by morning

| scenario | deliverable |
|---|---|
| Best | Spec + Tier 1 implementation + sweep + outcome ε committed. Path C closes. |
| Likely | Spec + Tier 1 implementation + sweep + outcome ζ committed with tuning gap documented. |
| Honest fallback | Spec + Tier 1 implementation + sweep + outcome η/θ documented; no commit; clear next-steps. |
| Gated | Spec only, T0 closure not landed; defer implementation to next session. |

---

## 9. Open questions for follow-on work blocks

These are flagged but **not in scope** for this work block:

1. **RIM-Glu co-release infrastructure.** Maluck 2020 supports RIM activating RIS via Glu, but RIS is GluCl-dominant (would mean inhibition under per-edge), AND the brain doesn't consume `nt_secondary` for edge sign assignment. Resolution requires either iGluR splice-variant override on RIS or non-CeNGEN-grounded RIM→RIS edge. ~1 day.
2. **ALA / FLP-13 quiescence pathway.** ALA is the second canonical quiescence-driving neuron (FLP-13 release, Nath 2016). Adding FLP-13 as a modulator would parallel the DAF-7 extension. ~2-3 hours of additional work.
3. **DEFAULT_SIGN_OVERRIDES cleanup.** Session 2 is reducing this from 24 entries to 7 documented exceptions. The remaining 17 deprecated overrides (including URYVL→RIS) would need re-evaluation if they have downstream consequences for the per-edge brain dynamics.
4. **Quantitative calibration of DAF-7 effect magnitudes.** Literature gives qualitative direction; matching simulated RIS firing rate during food to a specific biological number (e.g., 10-20 Hz per literature consensus) is iterative parameter-fitting work.

---

## Appendix A — Citations to verify before commit

All references in this spec marked [VERIFY] require PMID-level confirmation per project hygiene rule (Gao-Hobert 2020 → Mellem 2008 correction precedent). Specifically:
- You 2008: "daf-7 mutants spend 22 ± 4% time quiescent during refeeding vs. 59 ± 3% wild-type." Search: You YJ, Avery L, "Insulin, cGMP, and TGF-β signals regulate food intake and quiescence in C. elegans" Cell Metabolism 2008. PMID claim by user: 18334214 (verify).
- Davis 2021 review on RIS activation pathway: full citation pending.
- Greer 2008 on DAF-7 / feeding behavior: full citation pending.
- Maluck 2020 on RIM-Glu activation of RIS: full citation pending.

The implementation phase should add `[VERIFY: PMID]` placeholders in code comments and YAML docstrings, mirroring the citation-discipline established in the AVA channel-roster scout YAML.

---

## Appendix B — CeNGEN expression evidence summary

All values from `data/expression/cengen/derived/expression_neuron_mean.csv` (CeNGEN L4 single-cell, 91 neuron classes × 22468 genes), gene-symbol mapping via `data/wormbase_release_WS297/orthologs/c_elegans.PRJNA13758.WS297.alaska_ids.tsv`.

| gene | top expressers (top 5) | RIM | RIC | RIS | ALA |
|---|---|---:|---:|---:|---:|
| daf-7 (synthesis) | ASI=38.9, ASG=8.9, ASJ=3.7, AWA=2.8, ASK=2.1 | — | — | — | — |
| daf-1 (receptor) | I3=0.10, ASG=0.077, AIM=0.077, ADF=0.075, I5=0.072 | 0.019 | 0.007 | 0.000 | 0.011 |
| daf-4 (receptor) | PLM=0.168, URB=0.143, AVM=0.141, AWA=0.128, M1=0.118 | 0.012 | 0.012 | 0.018 | n/a |
| flp-11 (synthesis) | (RIS dominant per project work) | 0.004 | n/a | 985.196 | n/a |
| unc-25 (GAD) | (GABAergic neurons) | n/a | n/a | 15.857 | n/a |
| tdc-1 (TYR synthesis) | (TA neurons) | 15.236 | 1.188 | n/a | n/a |
| tbh-1 (OA synthesis) | (OA neurons) | 0.006 | 2.788 | n/a | n/a |
| eat-4 (vGluT) | (Glu-co-release) | 0.630 | 1.243 | n/a | n/a |

Two flagged data-vs-literature conflicts:
1. RIM/RIC daf-1 below CeNGEN floor despite literature attribution.
2. RIM eat-4 = 0.63 (low) — Maluck 2020 RIM-Glu co-release is supported but borderline in CeNGEN.

---

**End of spec.** Implementation gate: session 2's T0 closure commit must be present in repo before any source-modifying work begins.
